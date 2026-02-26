"""
TFT-ASRO Inference Pipeline.

Produces live multi-quantile predictions by:
    1. Assembling the latest feature vector from all data sources
    2. Running through the trained TFT model
    3. Formatting the output as a structured prediction dict

Designed to run in parallel with the existing XGBoost inference pipeline
for A/B comparison.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from deep_learning.config import TFTASROConfig, get_tft_config

logger = logging.getLogger(__name__)


class TFTPredictor:
    """
    Stateful predictor that holds the loaded model and PCA transformer.
    Thread-safe for read-only inference (no internal mutation after init).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        cfg: Optional[TFTASROConfig] = None,
    ):
        self.cfg = cfg or get_tft_config()
        self._checkpoint_path = checkpoint_path or self.cfg.training.best_model_path
        self._model = None
        self._pca = None
        self._hub_checked = False

    def _ensure_local_artifacts(self) -> None:
        """Download checkpoint from HF Hub if not present locally."""
        if self._hub_checked:
            return
        self._hub_checked = True

        if Path(self._checkpoint_path).exists():
            return

        try:
            from deep_learning.models.hub import download_tft_artifacts

            tft_dir = Path(self._checkpoint_path).parent
            downloaded = download_tft_artifacts(
                local_dir=tft_dir,
                repo_id=self.cfg.training.hf_model_repo,
            )
            if downloaded:
                logger.info("TFT checkpoint downloaded from HF Hub")
            else:
                logger.warning("TFT checkpoint not available on HF Hub")
        except Exception as exc:
            logger.warning("HF Hub download attempt failed: %s", exc)

    @property
    def model(self):
        if self._model is None:
            self._ensure_local_artifacts()
            if not Path(self._checkpoint_path).exists():
                raise FileNotFoundError(
                    f"TFT checkpoint not found: {self._checkpoint_path}"
                )
            from deep_learning.models.tft_copper import load_tft_model
            self._model = load_tft_model(self._checkpoint_path)
        return self._model

    @property
    def pca(self):
        if self._pca is None:
            self._ensure_local_artifacts()
            pca_path = self.cfg.embedding.pca_model_path
            if Path(pca_path).exists():
                from deep_learning.data.embeddings import load_pca
                self._pca = load_pca(pca_path)
        return self._pca

    def predict(self, session, symbol: str = "HG=F") -> Dict[str, Any]:
        """
        Generate a TFT-ASRO prediction for the given symbol.

        Returns a dict with:
            - predicted_return_median, q10, q90
            - predicted_price_median, q10, q90
            - confidence_band_96
            - volatility_estimate
            - quantiles (all 7)
            - model_info
        """
        from deep_learning.data.feature_store import build_tft_dataframe
        from deep_learning.data.dataset import build_datasets, create_dataloaders
        from deep_learning.models.tft_copper import format_prediction
        from pytorch_forecasting import TimeSeriesDataSet

        master_df, tv_unknown, tv_known, target_cols = build_tft_dataframe(session, self.cfg)

        last_known_price = self._get_last_price(session, symbol)

        encoder_length = self.cfg.model.max_encoder_length
        prediction_length = self.cfg.model.max_prediction_length

        recent = master_df.tail(encoder_length + prediction_length).copy()
        if len(recent) < encoder_length + 1:
            return {"error": f"Insufficient data: {len(recent)} rows, need {encoder_length + 1}"}

        recent["time_idx"] = np.arange(len(recent))

        target = target_cols[0] if target_cols else "target"

        try:
            ds = TimeSeriesDataSet(
                recent,
                time_idx="time_idx",
                target=target,
                group_ids=["group_id"],
                max_encoder_length=encoder_length,
                max_prediction_length=prediction_length,
                time_varying_unknown_reals=tv_unknown,
                time_varying_known_reals=tv_known,
                static_categoricals=["group_id"],
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
                allow_missing_timesteps=True,
            )
        except Exception as exc:
            logger.error("Failed to create inference dataset: %s", exc)
            return {"error": str(exc)}

        dl = ds.to_dataloader(train=False, batch_size=1, num_workers=0)

        try:
            raw = self.model.predict(dl, mode="raw")

            if isinstance(raw, dict):
                pred_tensor = raw.get("prediction", list(raw.values())[0])
            else:
                pred_tensor = raw

            import torch
            if isinstance(pred_tensor, torch.Tensor):
                pred_np = pred_tensor.cpu().numpy()
            else:
                pred_np = np.array(pred_tensor)

            if pred_np.ndim == 3:
                pred_for_format = pred_np[0]
            elif pred_np.ndim == 2:
                pred_for_format = pred_np
            else:
                pred_for_format = pred_np.reshape(1, -1)

        except Exception as exc:
            logger.error("TFT prediction failed: %s", exc)
            return {"error": str(exc)}

        result = format_prediction(
            pred_for_format,
            quantiles=self.cfg.model.quantiles,
            baseline_price=last_known_price,
        )

        result["model_info"] = {
            "type": "TFT-ASRO",
            "checkpoint": self._checkpoint_path,
            "encoder_length": encoder_length,
            "prediction_length": prediction_length,
            "n_features_unknown": len(tv_unknown),
            "n_features_known": len(tv_known),
        }

        result["generated_at"] = datetime.now(timezone.utc).isoformat()
        result["symbol"] = symbol

        return result

    def _get_last_price(self, session, symbol: str) -> float:
        """Fetch the latest close price from the database."""
        from app.models import PriceBar

        row = (
            session.query(PriceBar.close)
            .filter(PriceBar.symbol == symbol)
            .order_by(PriceBar.bar_time.desc())
            .first()
        )
        return float(row.close) if row else 1.0

    def get_model_metadata(self, session) -> Optional[Dict]:
        """Load persisted TFT model metadata from DB."""
        from app.models import TFTModelMetadata

        meta = (
            session.query(TFTModelMetadata)
            .filter(TFTModelMetadata.symbol == self.cfg.feature_store.target_symbol)
            .first()
        )
        if meta is None:
            return None

        return {
            "symbol": meta.symbol,
            "trained_at": meta.trained_at.isoformat() if meta.trained_at else None,
            "checkpoint_path": meta.checkpoint_path,
            "config": json.loads(meta.config_json) if meta.config_json else {},
            "metrics": json.loads(meta.metrics_json) if meta.metrics_json else {},
        }


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_predictor: Optional[TFTPredictor] = None


def get_tft_predictor(cfg: Optional[TFTASROConfig] = None) -> TFTPredictor:
    """Singleton-style access to the TFT predictor."""
    global _predictor
    if _predictor is None:
        _predictor = TFTPredictor(cfg=cfg)
    return _predictor


def generate_tft_analysis(session, symbol: str = "HG=F") -> Dict[str, Any]:
    """
    High-level API for generating a TFT-ASRO analysis report.

    Designed to mirror the interface of the existing
    ``app.inference.generate_analysis_report``.
    """
    predictor = get_tft_predictor()

    prediction = predictor.predict(session, symbol)
    if "error" in prediction:
        return prediction

    metadata = predictor.get_model_metadata(session)

    median_ret = prediction.get("predicted_return_median", 0)
    if median_ret > 0.005:
        direction = "YUKARI"
    elif median_ret < -0.005:
        direction = "ASAGI"
    else:
        direction = "YATAY"

    vol = prediction.get("volatility_estimate", 0)
    if vol > 0.02:
        risk_level = "YUKSEK"
    elif vol > 0.01:
        risk_level = "ORTA"
    else:
        risk_level = "DUSUK"

    return {
        "symbol": symbol,
        "model_type": "TFT-ASRO",
        "direction": direction,
        "risk_level": risk_level,
        "prediction": prediction,
        "model_metadata": metadata,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

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

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
    module="sklearn",
)
warnings.filterwarnings(
    "ignore",
    message=".*is an instance of `nn.Module` and is already saved during checkpointing.*",
    category=UserWarning,
    module="lightning.pytorch.utilities",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module="lightning.pytorch.utilities",
)

from deep_learning.config import TFTASROConfig, get_tft_config
from app.instruments import TARGET_DISPLAY_NAME, TARGET_SYMBOL

logger = logging.getLogger(__name__)

# Suppress PyTorch Lightning promotional tips ("litlogger", "litmodels")
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)


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

    def predict(self, session, symbol: str = TARGET_SYMBOL) -> Dict[str, Any]:
        """
        Generate a TFT-ASRO prediction for the given symbol.

        Lazily triggers price ingestion when the most recent PriceBar for
        the target is older than the configured freshness threshold. This
        prevents the frontend from showing a forecast anchored to a stale
        close when the weekend scheduler hasn't run or failed to catch up.

        Returns a dict with:
            - predicted_return_median, q10, q90
            - predicted_price_median, q10, q90
            - confidence_band_96
            - volatility_estimate
            - quantiles (all 7)
            - model_info
            - instrument:          {"symbol": "HG=F", "kind": "futures", "name": ...}
            - reference_price_date
            - baseline_staleness_days (int, 0 = fresh)
        """
        from deep_learning.data.feature_store import build_tft_dataframe
        from deep_learning.data.dataset import build_datasets, create_dataloaders
        from deep_learning.models.tft_copper import format_prediction
        from pytorch_forecasting import TimeSeriesDataSet

        # 1. Pre-flight: check PriceBar freshness and lazy-ingest if stale
        staleness_days, ingest_triggered = self._check_price_freshness(session, symbol)

        master_df, tv_unknown, tv_known, target_cols, last_known_price = build_tft_dataframe(session, self.cfg)

        # Extract the date of the baseline price so the frontend can show
        # "% change relative to <date> close" unambiguously.
        reference_price_date: Optional[str] = None
        try:
            if hasattr(master_df.index, "max"):
                last_date = master_df.index.max()
                reference_price_date = str(last_date)[:10] if last_date is not None else None
        except Exception:
            reference_price_date = None

        logger.info(
            "TFT predict: baseline_price=%.4f (date=%s) for %s",
            last_known_price, reference_price_date, symbol,
        )

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

        _nw = 0 if os.name == "nt" else 2
        dl = ds.to_dataloader(train=False, batch_size=1, num_workers=_nw)

        try:
            import torch

            # mode="quantiles" returns a plain Tensor (n_samples, pred_len, n_quantiles)
            # Avoids the inhomogeneous-shape error from mode="raw" which returns a
            # NamedTuple; np.array() cannot convert that to a uniform array.
            pred_tensor = self.model.predict(dl, mode="quantiles")

            if isinstance(pred_tensor, torch.Tensor):
                pred_np = pred_tensor.cpu().numpy()
            else:
                pred_np = np.array(pred_tensor)

            # Take first sample: (pred_len, n_quantiles)
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
            reference_price_date=reference_price_date,
        )

        result["model_info"] = {
            "type": "TFT-ASRO",
            "checkpoint": self._checkpoint_path,
            "encoder_length": encoder_length,
            "prediction_length": prediction_length,
            "n_features_unknown": len(tv_unknown),
            "n_features_known": len(tv_known),
        }

        # Surface freshness + instrument identity so the UI can label the
        # baseline unambiguously ("Futures HG=F, close of 2026-04-16").
        result["instrument"] = self._describe_instrument(symbol)
        result["baseline_staleness_days"] = staleness_days
        result["lazy_ingest_triggered"] = bool(ingest_triggered)

        result["generated_at"] = datetime.now(timezone.utc).isoformat()
        result["symbol"] = symbol

        return result

    # ------------------------------------------------------------------
    # Freshness helpers
    # ------------------------------------------------------------------

    # How stale (in calendar days) the latest PriceBar can be before we
    # attempt a lazy backfill. Futures markets close Fri→Sun globally, so
    # 3 days covers a standard weekend without spamming yfinance.
    FRESHNESS_THRESHOLD_DAYS = 3

    @staticmethod
    def _describe_instrument(symbol: str) -> Dict[str, str]:
        """Return a structured label for the traded instrument."""
        mapping = {
            TARGET_SYMBOL: {
                "symbol": TARGET_SYMBOL,
                "kind": "futures",
                "name": TARGET_DISPLAY_NAME,
                "note": (
                    "Continuous front-month contract (CME Group). Prices "
                    "are used consistently across training, inference and UI."
                ),
            },
        }
        return mapping.get(
            symbol,
            {"symbol": symbol, "kind": "unknown", "name": symbol, "note": ""},
        )

    def _check_price_freshness(self, session, symbol: str) -> tuple[int, bool]:
        """
        Compute staleness in calendar days for the target symbol and, when
        the latest bar is older than `FRESHNESS_THRESHOLD_DAYS`, trigger a
        best-effort incremental price ingest.

        Returns:
            (staleness_days, ingest_triggered)
        """
        try:
            from app.models import PriceBar

            latest = (
                session.query(PriceBar.date)
                .filter(PriceBar.symbol == symbol)
                .order_by(PriceBar.date.desc())
                .first()
            )
        except Exception as exc:
            logger.warning("Freshness check skipped: %s", exc)
            return (0, False)

        if latest is None or latest[0] is None:
            logger.warning("No PriceBar rows for %s — triggering ingest", symbol)
            return self._trigger_lazy_ingest(session, reason="no-bars")

        last_date = latest[0]
        if last_date.tzinfo is None:
            last_date = last_date.replace(tzinfo=timezone.utc)
        staleness = (datetime.now(timezone.utc) - last_date).days
        staleness = max(int(staleness), 0)

        if staleness >= self.FRESHNESS_THRESHOLD_DAYS:
            logger.warning(
                "PriceBar for %s is %sd stale (last=%s) — lazy ingest",
                symbol, staleness, last_date,
            )
            fresh_staleness, triggered = self._trigger_lazy_ingest(
                session, reason=f"stale-{staleness}d"
            )
            # After ingest, prefer the newly-computed staleness if it
            # improved; otherwise fall back to the pre-ingest value.
            if triggered and fresh_staleness < staleness:
                return (fresh_staleness, True)
            return (staleness, triggered)

        return (staleness, False)

    def _trigger_lazy_ingest(self, session, *, reason: str) -> tuple[int, bool]:
        """Run a short incremental price fetch and return updated staleness."""
        try:
            from app.data_manager import ingest_prices
            from app.models import PriceBar

            logger.info("Lazy price ingest triggered (reason=%s)", reason)
            ingest_prices(session)
            session.commit()

            latest = (
                session.query(PriceBar.date)
                .filter(PriceBar.symbol == self.cfg.feature_store.target_symbol)
                .order_by(PriceBar.date.desc())
                .first()
            )
            if latest is None or latest[0] is None:
                return (0, True)
            last_date = latest[0]
            if last_date.tzinfo is None:
                last_date = last_date.replace(tzinfo=timezone.utc)
            updated = max(int((datetime.now(timezone.utc) - last_date).days), 0)
            logger.info("Lazy ingest complete — staleness now %sd", updated)
            return (updated, True)
        except Exception as exc:
            logger.error("Lazy ingest failed: %s", exc)
            try:
                session.rollback()
            except Exception:
                pass
            return (0, False)

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
# Ensemble: XGBoost + TFT directional voting
# ---------------------------------------------------------------------------

def ensemble_directional_vote(
    xgb_return: float,
    tft_return: float,
    xgb_bias_correction: float = 0.0,
) -> Dict[str, Any]:
    """
    Combine XGBoost and TFT directional signals into a consensus.

    Rules:
        - Both agree on direction → strong signal, full size
        - Disagree → low confidence, reduced position or wait
        - One near-zero (< 0.2% threshold) → defer to the other

    NOTE: XGBoost is known to have extreme negative bias (rarely predicts
    negative returns, and when it does they're too small).  The
    xgb_bias_correction parameter is subtracted from xgb_return to
    compensate (set via historical calibration).

    Args:
        xgb_return:          XGBoost's predicted next-day return.
        tft_return:          TFT-ASRO's predicted next-day median return.
        xgb_bias_correction: Subtracted from xgb_return to debias.

    Returns:
        Dict with consensus direction, confidence, and component signals.
    """
    NEUTRAL_THRESHOLD = 0.002

    xgb_adj = xgb_return - xgb_bias_correction
    xgb_dir = 1 if xgb_adj > NEUTRAL_THRESHOLD else (-1 if xgb_adj < -NEUTRAL_THRESHOLD else 0)
    tft_dir = 1 if tft_return > NEUTRAL_THRESHOLD else (-1 if tft_return < -NEUTRAL_THRESHOLD else 0)

    if xgb_dir == tft_dir and xgb_dir != 0:
        consensus = "BULLISH" if xgb_dir > 0 else "BEARISH"
        confidence = "HIGH"
        position_scale = 1.0
    elif xgb_dir == 0 and tft_dir != 0:
        consensus = "BULLISH" if tft_dir > 0 else "BEARISH"
        confidence = "MEDIUM"
        position_scale = 0.6
    elif tft_dir == 0 and xgb_dir != 0:
        consensus = "BULLISH" if xgb_dir > 0 else "BEARISH"
        confidence = "MEDIUM"
        position_scale = 0.5
    elif xgb_dir != tft_dir:
        consensus = "NEUTRAL"
        confidence = "LOW"
        position_scale = 0.0
    else:
        consensus = "NEUTRAL"
        confidence = "LOW"
        position_scale = 0.0

    blended_return = 0.4 * xgb_adj + 0.6 * tft_return

    return {
        "consensus_direction": consensus,
        "confidence": confidence,
        "position_scale": position_scale,
        "blended_return": blended_return,
        "xgb_return_raw": xgb_return,
        "xgb_return_adjusted": xgb_adj,
        "tft_return": tft_return,
        "xgb_direction": xgb_dir,
        "tft_direction": tft_dir,
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


def generate_tft_analysis(session, symbol: str = TARGET_SYMBOL) -> Dict[str, Any]:
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

    # Direction based on T+1 (most reliable signal)
    median_ret = prediction.get("predicted_return_median", 0)
    if median_ret > 0.005:
        direction = "BULLISH"
    elif median_ret < -0.005:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    # Weekly trend based on T+5 (end-of-horizon)
    weekly_ret = prediction.get("weekly_return", median_ret)
    if weekly_ret > 0.005:
        weekly_trend = "BULLISH"
    elif weekly_ret < -0.005:
        weekly_trend = "BEARISH"
    else:
        weekly_trend = "NEUTRAL"

    vol = prediction.get("volatility_estimate", 0)
    if vol > 0.02:
        risk_level = "HIGH"
    elif vol > 0.01:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    import math
    def _sanitize_floats(obj: Any) -> Any:
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        elif isinstance(obj, dict):
            return {k: _sanitize_floats(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(_sanitize_floats(v) for v in obj)
        return obj

    raw_result = {
        "symbol": symbol,
        "model_type": "TFT-ASRO",
        "direction": direction,
        "weekly_trend": weekly_trend,
        "risk_level": risk_level,
        "prediction": prediction,
        "model_metadata": metadata,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    
    return _sanitize_floats(raw_result)

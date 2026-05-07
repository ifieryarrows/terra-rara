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
from deep_learning.contract import (
    FORECAST_CONTRACT_VERSION,
    RETURN_SPACE,
    TARGET_RETURN_TYPE,
)

logger = logging.getLogger(__name__)

# Suppress PyTorch Lightning promotional tips ("litlogger", "litmodels")
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)


class IncompatibleTFTCheckpointError(RuntimeError):
    """Raised when a checkpoint predates the weekly log-return contract."""


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
        self._metadata_checked = False

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
            self._validate_checkpoint_contract()
            from deep_learning.models.tft_copper import load_tft_model
            self._model = load_tft_model(self._checkpoint_path)
        return self._model

    def _validate_checkpoint_contract(self) -> None:
        if self._metadata_checked:
            return

        metadata_path = Path(self._checkpoint_path).parent / "tft_metadata.json"
        if not metadata_path.exists():
            raise IncompatibleTFTCheckpointError(
                "Incompatible TFT checkpoint: missing weekly_log_v1 metadata. Retraining required."
            )

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise IncompatibleTFTCheckpointError(
                f"Incompatible TFT checkpoint: unreadable metadata ({exc}). Retraining required."
            ) from exc

        config = metadata.get("config") or {}
        version = metadata.get("forecast_contract_version") or config.get("forecast_contract_version")
        target_return_type = metadata.get("target_return_type") or config.get("target_return_type")
        primary_horizon = metadata.get("primary_horizon_days") or config.get("primary_horizon_days")
        return_space = metadata.get("return_space") or config.get("return_space")

        if (
            version != FORECAST_CONTRACT_VERSION
            or target_return_type != TARGET_RETURN_TYPE
            or int(primary_horizon or 0) != self.cfg.forecast.primary_horizon_days
            or return_space != RETURN_SPACE
        ):
            raise IncompatibleTFTCheckpointError(
                "Incompatible TFT checkpoint: expected weekly_log_v1 contract. Retraining required."
            )
        self._metadata_checked = True

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
        from deep_learning.data.future_frame import build_future_decoder_rows
        from deep_learning.data.dataset import _identity_target_normalizer
        from deep_learning.models.tft_copper import format_prediction
        from pytorch_forecasting import TimeSeriesDataSet

        # 1. Pre-flight: check PriceBar freshness and lazy-ingest if stale
        staleness_days, ingest_triggered = self._check_price_freshness(session, symbol)

        master_df, tv_unknown, tv_known, target_cols, last_known_price = build_tft_dataframe(
            session,
            self.cfg,
            drop_missing_target=False,
        )

        # Track feature freshness separately from the baseline close. In live
        # inference the feature frame keeps the final bar with a dummy target,
        # while training still drops it.
        feature_last_date: Optional[str] = None
        try:
            if hasattr(master_df.index, "max"):
                last_date = master_df.index.max()
                feature_last_date = self._date_label(last_date)
        except Exception:
            feature_last_date = None

        baseline_price, reference_price_date = self._latest_price_baseline(
            session,
            symbol,
            fallback_price=last_known_price,
            fallback_date=feature_last_date,
        )

        logger.info(
            "TFT predict: baseline_price=%.4f reference_price_date=%s "
            "feature_last_date=%s baseline_staleness_days=%s for %s",
            baseline_price,
            reference_price_date,
            feature_last_date,
            staleness_days,
            symbol,
        )

        encoder_length = self.cfg.model.max_encoder_length
        prediction_length = self.cfg.model.max_prediction_length

        history = master_df.tail(encoder_length).copy()
        future = build_future_decoder_rows(history, prediction_length, self.cfg)
        recent = pd.concat([history, future], axis=0)
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
                target_normalizer=_identity_target_normalizer(),
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

        except IncompatibleTFTCheckpointError as exc:
            logger.warning("TFT checkpoint incompatible: %s", exc)
            return self._degraded_retrain_required(str(exc))
        except Exception as exc:
            logger.error("TFT prediction failed: %s", exc)
            return {"error": str(exc)}

        conformal_adjustment = self._conformal_adjustment_for_latest_regime(master_df)
        result = format_prediction(
            pred_for_format,
            quantiles=self.cfg.model.quantiles,
            baseline_price=baseline_price,
            reference_price_date=reference_price_date,
            conformal_adjustment=conformal_adjustment,
        )

        result["model_info"] = {
            "type": "TFT-ASRO",
            "checkpoint": self._checkpoint_path,
            "encoder_length": encoder_length,
            "prediction_length": prediction_length,
            "n_features_unknown": len(tv_unknown),
            "n_features_known": len(tv_known),
            "forecast_contract_version": FORECAST_CONTRACT_VERSION,
            "target_return_type": TARGET_RETURN_TYPE,
            "return_space": RETURN_SPACE,
        }

        # Surface freshness + instrument identity so the UI can label the
        # baseline unambiguously ("Futures HG=F, close of 2026-04-16").
        result["instrument"] = self._describe_instrument(symbol)
        result["baseline_staleness_days"] = staleness_days
        result["lazy_ingest_triggered"] = bool(ingest_triggered)

        result["generated_at"] = datetime.now(timezone.utc).isoformat()
        result["symbol"] = symbol

        return result

    @staticmethod
    def _degraded_retrain_required(message: str) -> Dict[str, Any]:
        return {
            "model_state": "retrain_required",
            "quality_state": "degraded",
            "message": message,
            "return_space": RETURN_SPACE,
        }

    def _conformal_adjustment_for_latest_regime(self, master_df: pd.DataFrame) -> float:
        try:
            from deep_learning.calibration.conformal import bucketize_regime, select_bucket_adjustment

            path = Path(self._checkpoint_path).parent / "conformal_calibration.json"
            if not path.exists() or master_df.empty:
                return 0.0
            calibration = json.loads(path.read_text(encoding="utf-8"))
            bucket = bucketize_regime(master_df.iloc[-1])
            return select_bucket_adjustment(calibration, bucket)
        except Exception as exc:
            logger.debug("Conformal adjustment unavailable: %s", exc)
            return 0.0

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

    @staticmethod
    def _date_label(value: Any) -> Optional[str]:
        """Format a date-like value as YYYY-MM-DD for API/log fields."""
        if value is None:
            return None
        try:
            return value.date().isoformat()
        except AttributeError:
            return str(value)[:10]
        except Exception:
            return None

    def _latest_price_baseline(
        self,
        session,
        symbol: str,
        *,
        fallback_price: float,
        fallback_date: Optional[str],
    ) -> tuple[float, Optional[str]]:
        """
        Return the latest close and date from PriceBar.

        The feature frame may drop or dummy-fill the final target depending on
        training vs inference mode. Baseline reporting must therefore come from
        the canonical price table, not from the post-target-filter frame.
        """
        baseline_price = fallback_price
        reference_price_date = fallback_date

        try:
            from app.models import PriceBar

            latest_bar = (
                session.query(PriceBar)
                .filter(PriceBar.symbol == symbol)
                .order_by(PriceBar.date.desc())
                .first()
            )
            if latest_bar is not None and latest_bar.close is not None:
                baseline_price = float(latest_bar.close)
                reference_price_date = self._date_label(latest_bar.date)
        except Exception as exc:
            logger.warning(
                "Latest PriceBar baseline lookup failed for %s; using feature fallback: %s",
                symbol,
                exc,
            )

        return baseline_price, reference_price_date

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
    if prediction.get("model_state") == "retrain_required":
        return prediction
    if "error" in prediction:
        return prediction

    metadata = predictor.get_model_metadata(session)

    weekly_ret = prediction.get("weekly_return", 0.0)
    t1_ret = prediction.get("predicted_return_median", 0.0)

    if weekly_ret > predictor.cfg.forecast.weekly_direction_threshold:
        direction = "BULLISH"
    elif weekly_ret < -predictor.cfg.forecast.weekly_direction_threshold:
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    if t1_ret > predictor.cfg.forecast.t1_direction_threshold:
        t1_impulse = "BULLISH"
    elif t1_ret < -predictor.cfg.forecast.t1_direction_threshold:
        t1_impulse = "BEARISH"
    else:
        t1_impulse = "NEUTRAL"

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
        "weekly_trend": direction,
        "primary_horizon": "5D",
        "primary_forecast_return": weekly_ret,
        "primary_forecast_q10": prediction.get("weekly_return_q10_calibrated"),
        "primary_forecast_q90": prediction.get("weekly_return_q90_calibrated"),
        "t1_impulse": t1_impulse,
        "t1_return": t1_ret,
        "weekly_forecast": {
            "horizon": "5D",
            "expected_return": weekly_ret,
            "q10_return": prediction.get("weekly_return_q10_calibrated"),
            "q90_return": prediction.get("weekly_return_q90_calibrated"),
            "calibrated": bool(prediction.get("weekly_interval_calibrated", False)),
            "calibration_adjustment": prediction.get("weekly_interval_calibration_adjustment"),
            "t1_impulse": t1_impulse,
            "t1_return": t1_ret,
            "regime": None,
        },
        "risk_level": risk_level,
        "prediction": prediction,
        "model_metadata": metadata,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    
    return _sanitize_floats(raw_result)

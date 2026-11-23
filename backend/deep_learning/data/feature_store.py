"""
Centralised Feature Store for TFT-ASRO.

Fuses all heterogeneous data sources (price, sentiment, embeddings, LME,
calendar) into a single long-format DataFrame suitable for
pytorch_forecasting.TimeSeriesDataSet.

TFT data categories:
    1. time_varying_unknown_reals  - observed only in the past
    2. time_varying_known_reals    - known into the future (calendar, etc.)
    3. static_reals / static_categoricals - time-invariant per group
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session as SASession

from deep_learning.config import TFTASROConfig, get_tft_config

logger = logging.getLogger(__name__)

_FEATURE_SNAPSHOT_FORMAT_VERSION = 1
TFT_KNOWN_CALENDAR_FEATURES: tuple[str, ...] = ("day_of_week",)


def data_snapshot_sha256(master_df: pd.DataFrame) -> str:
    """Return the digest used to identify an exact TFT feature frame."""
    import hashlib

    digest = hashlib.sha256()
    digest.update("\x1f".join(str(column) for column in master_df.columns).encode())
    digest.update("\x1f".join(str(dtype) for dtype in master_df.dtypes).encode())
    digest.update(pd.util.hash_pandas_object(master_df, index=True).to_numpy().tobytes())
    return digest.hexdigest()


_UNSIGNED_VOL_RE = re.compile(r"_vol_\d+$")


def _is_unsigned_vol_feature(col: str) -> bool:
    """Exclude strictly-positive unsigned volatility series from TFT encoder inputs."""
    return (
        not col.endswith("_signed_vol_ratio")
        and (col.endswith("_vol_5_20_ratio") or bool(_UNSIGNED_VOL_RE.search(col)))
    )


def _feature_snapshot_metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _load_feature_snapshot(
    path: Path,
    *,
    drop_missing_target: bool,
) -> Optional[tuple[pd.DataFrame, list[str], list[str], list[str], float]]:
    """Load a previously captured post-selection TFT feature frame."""
    if not path.exists():
        return None

    metadata_path = _feature_snapshot_metadata_path(path)
    if not metadata_path.exists():
        raise RuntimeError(
            f"TFT feature snapshot metadata is missing: {metadata_path}"
        )

    import json

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != _FEATURE_SNAPSHOT_FORMAT_VERSION:
        raise RuntimeError(
            "Unsupported TFT feature snapshot format: "
            f"{metadata.get('format_version')!r}"
        )
    if bool(metadata.get("drop_missing_target", True)) != drop_missing_target:
        raise RuntimeError(
            "TFT feature snapshot target mode does not match the requested run: "
            f"snapshot={metadata.get('drop_missing_target')!r} "
            f"requested={drop_missing_target!r}"
        )

    requested_as_of = os.environ.get("TFT_DATA_AS_OF", "").strip() or None
    captured_as_of = metadata.get("tft_data_as_of") or None
    if requested_as_of and captured_as_of and requested_as_of != captured_as_of:
        raise RuntimeError(
            "TFT feature snapshot cutoff mismatch: "
            f"snapshot={captured_as_of!r} requested={requested_as_of!r}"
        )

    master_df = pd.read_pickle(path)
    actual_sha = data_snapshot_sha256(master_df)
    expected_sha = str(metadata.get("sha256", "")).lower()
    if actual_sha != expected_sha:
        raise RuntimeError(
            "TFT feature snapshot integrity check failed: "
            f"expected={expected_sha} actual={actual_sha}"
        )

    configured_sha = os.environ.get("TFT_EXPECTED_DATA_SNAPSHOT_SHA", "").strip().lower()
    if configured_sha and actual_sha != configured_sha:
        raise RuntimeError(
            "TFT feature snapshot does not match TFT_EXPECTED_DATA_SNAPSHOT_SHA: "
            f"expected={configured_sha} actual={actual_sha}"
        )

    logger.info(
        "Loaded immutable TFT feature snapshot: sha256=%s rows=%d first=%s last=%s",
        actual_sha,
        len(master_df),
        metadata.get("first_index"),
        metadata.get("last_index"),
    )
    return (
        master_df,
        list(metadata["time_varying_unknown_reals"]),
        list(metadata["time_varying_known_reals"]),
        list(metadata["target_cols"]),
        float(metadata["last_close"]),
    )


def _write_feature_snapshot(
    path: Path,
    master_df: pd.DataFrame,
    *,
    time_varying_unknown: list[str],
    time_varying_known: list[str],
    target_cols: list[str],
    last_close: float,
    drop_missing_target: bool,
) -> None:
    """Persist the exact post-selection frame used by a replayable run."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    digest = data_snapshot_sha256(master_df)
    master_df.to_pickle(path)
    metadata = {
        "format_version": _FEATURE_SNAPSHOT_FORMAT_VERSION,
        "sha256": digest,
        "rows": int(len(master_df)),
        "columns": int(master_df.shape[1]),
        "first_index": str(master_df.index.min()) if len(master_df) else None,
        "last_index": str(master_df.index.max()) if len(master_df) else None,
        "tft_data_as_of": os.environ.get("TFT_DATA_AS_OF") or None,
        "drop_missing_target": drop_missing_target,
        "time_varying_unknown_reals": time_varying_unknown,
        "time_varying_known_reals": time_varying_known,
        "target_cols": target_cols,
        "last_close": last_close,
    }
    _feature_snapshot_metadata_path(path).write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(
        "Wrote immutable TFT feature snapshot: path=%s sha256=%s rows=%d",
        path,
        digest,
        len(master_df),
    )


def _resolve_data_end_date() -> datetime:
    """Resolve the feature-store cutoff, allowing reproducible replay runs."""
    raw_value = os.environ.get("TFT_DATA_AS_OF", "").strip()
    if not raw_value:
        return datetime.now(timezone.utc)

    try:
        parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            "TFT_DATA_AS_OF must be an ISO-8601 timestamp, for example "
            "2026-08-23T23:59:59Z"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Screener bridge: load correlated symbols from active.json / screener output
# ---------------------------------------------------------------------------

def load_training_symbols() -> list[str]:
    """
    Load the active symbol set from ``config/symbol_sets/active.json``.

    Falls back to settings.training_symbols if the file cannot be read.
    This bridges the screener's challenger/champion pipeline with the
    TFT feature store so that the same statistically validated symbols
    feed both XGBoost and TFT.
    """
    import json
    from pathlib import Path

    backend_root = Path(__file__).resolve().parent.parent.parent
    active_path = backend_root / "config" / "symbol_sets" / "active.json"

    if active_path.exists():
        try:
            data = json.loads(active_path.read_text(encoding="utf-8"))
            symbols = data.get("symbols", [])
            if symbols:
                logger.info(
                    "Loaded %d training symbols from %s (v%s)",
                    len(symbols), active_path.name, data.get("version", "?"),
                )
                return symbols
        except Exception as exc:
            logger.warning("Failed to read %s: %s", active_path, exc)

    try:
        from app.settings import get_settings
        return get_settings().training_symbols
    except Exception:
        return ["HG=F", "DX-Y.NYB", "CL=F", "FXI"]


def load_screener_selected_symbols(
    artifacts_dir: str = "artifacts/runs/latest",
) -> list[dict]:
    """
    Read the screener's ``selected_symbols.json`` to get the full audit-trail
    entries including IS/OOS Pearson, category, and lead-lag information.

    Returns a list of dicts (one per selected symbol).
    """
    import json
    from pathlib import Path

    backend_root = Path(__file__).resolve().parent.parent.parent
    selected_path = backend_root / artifacts_dir / "selected_symbols.json"

    if not selected_path.exists():
        logger.info("No screener selected_symbols.json found at %s", selected_path)
        return []

    try:
        data = json.loads(selected_path.read_text(encoding="utf-8"))
        selected = data.get("selected", [])
        logger.info(
            "Loaded %d screener-selected symbols (rules v%s, run %s)",
            len(selected),
            data.get("selection_rules_version", "?"),
            data.get("screener_run_id", "?"),
        )
        return selected
    except Exception as exc:
        logger.warning("Failed to read screener output: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Sentiment quality control
# ---------------------------------------------------------------------------

def _sentiment_qc(sentiment: pd.Series, close: pd.Series) -> None:
    """Log lagged correlation between sentiment and next-day returns."""
    try:
        returns = close.pct_change(fill_method=None).shift(-1)
        raw_aligned = pd.DataFrame({"sentiment": sentiment, "return": returns})
        raw_values = raw_aligned[["sentiment", "return"]].to_numpy(dtype=float)
        if np.isinf(raw_values).any():
            logger.warning("Sentiment QC skipped: non-finite values detected")
            return

        aligned = raw_aligned
        aligned = aligned.replace([np.inf, -np.inf], np.nan).dropna()
        if len(aligned) < 30:
            return

        values = aligned[["sentiment", "return"]].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            logger.warning("Sentiment QC skipped: non-finite values detected")
            return

        sentiment_std = float(aligned["sentiment"].std(ddof=0) or 0.0)
        if sentiment_std < 1e-9:
            logger.warning("Sentiment QC skipped: sentiment has insufficient variance")
            return

        return_std = float(aligned["return"].std(ddof=0) or 0.0)
        if return_std < 1e-9:
            logger.warning("Sentiment QC skipped: return has insufficient variance")
            return

        corr_0 = aligned["sentiment"].corr(aligned["return"])
        corr_1 = aligned["sentiment"].shift(1).corr(aligned["return"])
        if not np.isfinite([corr_0, corr_1]).all():
            logger.warning("Sentiment QC skipped: correlation produced non-finite values")
            return

        logger.info(
            "Sentiment QC: corr(sent, ret_t+1)=%.4f  corr(sent_t-1, ret_t+1)=%.4f  n=%d",
            corr_0, corr_1, len(aligned),
        )

        if abs(corr_0) < 0.02 and abs(corr_1) < 0.02:
            logger.warning(
                "Sentiment QC: very low correlation with returns — "
                "sentiment features may not add predictive value"
            )
    except Exception as exc:
        logger.debug("Sentiment QC check failed: %s", exc)


# ---------------------------------------------------------------------------
# Calendar / known-future features
# ---------------------------------------------------------------------------

def _build_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic features known at any future date."""
    cal = pd.DataFrame(index=index)

    cal["day_of_week"] = np.clip(index.dayofweek, 0, 4).astype(np.float32) / 4.0
    cal["day_of_month"] = index.day.astype(np.float32) / 31.0
    cal["month"] = index.month.astype(np.float32) / 12.0

    day_frac = 2 * np.pi * index.dayofyear / 365.25
    cal["cal_sin_day"] = np.sin(day_frac).astype(np.float32)
    cal["cal_cos_day"] = np.cos(day_frac).astype(np.float32)

    month_frac = 2 * np.pi * index.month / 12.0
    cal["cal_sin_month"] = np.sin(month_frac).astype(np.float32)
    cal["cal_cos_month"] = np.cos(month_frac).astype(np.float32)

    cal["is_monday"] = (index.dayofweek == 0).astype(np.float32)
    cal["is_friday"] = (index.dayofweek == 4).astype(np.float32)

    cal["is_month_start"] = index.is_month_start.astype(np.float32)
    cal["is_month_end"] = index.is_month_end.astype(np.float32)
    cal["is_quarter_end"] = index.is_quarter_end.astype(np.float32)

    return cal


# ---------------------------------------------------------------------------
# Price / technical features (reuses existing helpers)
# ---------------------------------------------------------------------------

from app.price_utils import sanitize_close_spikes as _sanitize_close_spikes


def _build_price_features(
    session,
    symbol: str,
    start_date,
    end_date,
    price_df: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load price data and compute stationary technical features for *symbol*.

    Returns (raw_price_df, features_df).
    """
    from app.features import load_price_data, generate_symbol_features

    if price_df is None:
        price_df = load_price_data(session, symbol, start_date, end_date)
    if price_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    sanitized_close = _sanitize_close_spikes(price_df["close"], symbol=symbol)
    copied = False
    if not sanitized_close.equals(price_df["close"]):
        price_df = price_df.copy()
        price_df["close"] = sanitized_close
        copied = True
    if "adj_close" in price_df.columns:
        sanitized_adj = _sanitize_close_spikes(price_df["adj_close"], symbol=f"{symbol}[adj]", ffill_invalid=False)
        if not sanitized_adj.equals(price_df["adj_close"]):
            if not copied:
                price_df = price_df.copy()
            price_df["adj_close"] = sanitized_adj

    features = generate_symbol_features(price_df, symbol, include_raw_ma=False)
    return price_df, features


# ---------------------------------------------------------------------------
# Embedding features (daily aggregated PCA vectors)
# ---------------------------------------------------------------------------

def _build_daily_embedding_features(
    session,
    index: pd.DatetimeIndex,
    pca_dim: int = 32,
) -> pd.DataFrame:
    """
    Load PCA-reduced FinBERT embeddings, aggregate to daily level,
    and reindex onto the trading calendar.
    """
    from app.models import NewsEmbedding, NewsProcessed, NewsRaw
    from deep_learning.data.embeddings import bytes_to_embedding, aggregate_daily_embeddings
    from pipelines.market_calendar import assign_market_date
    from sqlalchemy import func
    from sqlalchemy.orm import Session as SASession

    start_date = index.min().to_pydatetime()
    end_date = index.max().to_pydatetime()
    available_expr = func.coalesce(NewsRaw.fetched_at, NewsRaw.published_at)

    query = (
        session.query(
            NewsRaw.published_at,
            NewsRaw.fetched_at,
            NewsEmbedding.embedding_pca,
        )
        .join(NewsProcessed, NewsEmbedding.news_processed_id == NewsProcessed.id)
        .join(NewsRaw, NewsProcessed.raw_id == NewsRaw.id)
        .filter(NewsRaw.published_at <= end_date, available_expr <= end_date)
    )
    if isinstance(session, SASession):
        lower_bound = start_date - timedelta(days=7)
        query = query.filter(available_expr >= lower_bound)

    rows = query.order_by(NewsRaw.published_at.asc()).all()

    if not rows:
        cols = [f"emb_pca_{i}" for i in range(pca_dim)]
        return pd.DataFrame(0.0, index=index, columns=cols)

    start_day = pd.Timestamp(start_date).date()
    end_day = pd.Timestamp(end_date).date()
    date_groups: dict[str, list[np.ndarray]] = {}
    for r in rows:
        published_market_date = assign_market_date(r.published_at)
        available_at = getattr(r, "fetched_at", None) or r.published_at
        available_market_date = assign_market_date(available_at)
        market_date = max(published_market_date, available_market_date)
        if not (start_day <= market_date <= end_day):
            continue
        d = market_date.isoformat()
        vec = bytes_to_embedding(r.embedding_pca, dim=pca_dim)
        # bytes_to_embedding now always returns dim-length arrays, but
        # guard against any future shape surprises to keep stack safe.
        if vec.shape == (pca_dim,):
            date_groups.setdefault(d, []).append(vec)
        else:
            logger.debug("Skipping embedding with unexpected shape %s on %s", vec.shape, d)

    records = []
    for d, vecs in date_groups.items():
        if not vecs:
            continue
        agg = aggregate_daily_embeddings(np.stack(vecs))
        record = {"date": pd.Timestamp(d)}
        for i, v in enumerate(agg):
            record[f"emb_pca_{i}"] = float(v)
        records.append(record)

    if not records:
        cols = [f"emb_pca_{i}" for i in range(pca_dim)]
        return pd.DataFrame(0.0, index=index, columns=cols)

    emb_df = pd.DataFrame(records).set_index("date").sort_index()
    emb_df.index = pd.to_datetime(emb_df.index)

    emb_aligned = emb_df.reindex(index).ffill(limit=3).fillna(0.0)
    return emb_aligned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_tft_dataframe(
    session,
    cfg: Optional[TFTASROConfig] = None,
    *,
    drop_missing_target: bool = True,
    apply_mrmr: bool = True,
    use_snapshot: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str], list[str], float]:
    """
    Build the master DataFrame for TFT training / inference.

    Returns:
        (df, time_varying_unknown_reals, time_varying_known_reals, target_cols, last_close)

    The returned df has:
        - "time_idx"  : integer time index (required by pytorch_forecasting)
        - "group_id"  : constant "copper" (single series)
        - "target"    : realized 1-day log return ``log(close[t]/close[t-1])``
        - columns for all three TFT feature categories

    Training and validation only require valid ``target`` rows (retaining the
    final 5 bars so ``TimeSeriesDataSet`` can decode the last 5 OOS weekly
    windows ending on the newest trading day). Live inference sets
    ``drop_missing_target=False`` to keep the latest anchor bar. Passing
    ``apply_mrmr=False, use_snapshot=False`` yields the full pre-selection
    candidate frame for per-fold leak-free CV feature selection.
    """
    if cfg is None:
        cfg = get_tft_config()

    snapshot_path_raw = os.environ.get("TFT_FEATURE_SNAPSHOT_PATH", "").strip()
    snapshot_path = Path(snapshot_path_raw) if snapshot_path_raw else None
    if snapshot_path is not None and use_snapshot and apply_mrmr:
        captured = _load_feature_snapshot(
            snapshot_path,
            drop_missing_target=drop_missing_target,
        )
        if captured is not None:
            return captured

    target_symbol = cfg.feature_store.target_symbol
    end_date = _resolve_data_end_date()
    start_date = end_date - timedelta(days=cfg.training.lookback_days)
    logger.info("Feature store cutoff: %s", end_date.isoformat())

    # ---- 1. Price & technical indicators ----
    # Use screener-validated symbols from active.json
    training_symbols = load_training_symbols()
    logger.info("Building features with %d symbols: %s", len(training_symbols), training_symbols[:5])

    from app.features import (
        load_multi_price_data,
        generate_aligned_symbol_features,
    )

    correl_symbols = [sym for sym in training_symbols if sym != target_symbol]
    if isinstance(session, SASession):
        price_map = load_multi_price_data(
            session, [target_symbol] + correl_symbols, start_date, end_date
        )
    else:
        price_map = {}

    if target_symbol in price_map:
        price_df, price_features = _build_price_features(
            session,
            target_symbol,
            start_date,
            end_date,
            price_df=price_map[target_symbol],
        )
    else:
        price_df, price_features = _build_price_features(
            session,
            target_symbol,
            start_date,
            end_date,
        )
    if price_df.empty:
        raise ValueError(f"No price data for {target_symbol}")
    weekday_mask = price_df.index.dayofweek < 5
    if not weekday_mask.all():
        dropped_wknd = int((~weekday_mask).sum())
        logger.warning("Dropped %d weekend rows from %s price_df", dropped_wknd, target_symbol)
        price_df = price_df.loc[weekday_mask]
        price_features = price_features.loc[price_df.index]
        if price_df.empty:
            raise ValueError(f"No weekday price data for {target_symbol}")

    if correl_symbols and not isinstance(session, SASession):
        price_map.update(
            load_multi_price_data(session, correl_symbols, start_date, end_date)
        )

    other_dfs = {
        sym: sym_df
        for sym in correl_symbols
        if not (sym_df := price_map.get(sym, pd.DataFrame())).empty
    }

    if other_dfs:
        feature_frames = [price_features]
        for sym, df in other_dfs.items():
            if not df.empty:
                aligned_feats = generate_aligned_symbol_features(
                    df,
                    sym,
                    price_df.index,
                    max_ffill=cfg.feature_store.max_ffill,
                    include_raw_ma=False,
                    sanitize_spikes=True,
                )
                if not aligned_feats.empty:
                    feature_frames.append(aligned_feats)
        if len(feature_frames) > 1:
            price_features = pd.concat(feature_frames, axis=1)
        logger.info("Added native-bar features from %d correlated symbols", len(other_dfs))

    price_features = price_features.copy()

    # Keep HG=F_ret1 as clean realized 1d log return matching `target`
    # for autoregressive encoder context before macro residual calculation.
    target_close_clean = _sanitize_close_spikes(price_df["close"], symbol=target_symbol)
    price_features[f"{target_symbol}_ret1"] = np.log(
        target_close_clean.where(target_close_clean > 0.0, np.nan)
    ).diff()

    from deep_learning.calibration.hybrid import (
        compute_causal_macro_fair_value_zscore,
        compute_causal_theta_5d_series,
    )

    price_features["macro_fair_value_zscore"] = compute_causal_macro_fair_value_zscore(
        price_features,
        target_prefix=target_symbol,
    )

    target_index = price_df.index
    logger.info("Price data: %d bars for %s", len(target_index), target_symbol)

    # ---- 2. Sentiment & 3. Embedding features (decoupled from TFT when False) ----
    if getattr(cfg.feature_store, "include_news_features", False):
        from deep_learning.data.sentiment_features import (
            build_all_sentiment_features,
        )
        from deep_learning.data.sentiment_market_date import (
            build_market_date_event_counts_from_db,
            build_market_date_sentiment_frame,
        )

        sent_df = build_market_date_sentiment_frame(session, start_date, end_date)
        if not sent_df.empty:
            sent_aligned = sent_df.reindex(target_index).ffill(limit=cfg.feature_store.max_ffill)
            sent_aligned["sentiment_index"] = sent_aligned["sentiment_index"].fillna(0.0)
            sent_aligned["news_count"] = sent_aligned["news_count"].fillna(0)
            for col, default in (
                ("material_news_count", 0.0),
                ("after_close_news_count", 0.0),
                ("days_since_last_material_news", 30.0),
                ("stale_sentiment_flag", 1.0),
            ):
                sent_aligned[col] = sent_aligned[col].fillna(default)
            sent_aligned["days_since_last_material_news"] = (
                sent_aligned["days_since_last_material_news"].astype(float).clip(lower=0.0, upper=30.0)
            )

            event_counts = build_market_date_event_counts_from_db(session, start_date, end_date)
            advanced_sent = build_all_sentiment_features(sent_aligned, event_counts=event_counts, cfg=cfg.sentiment)
        else:
            sent_aligned = pd.DataFrame(
                {
                    "sentiment_index": 0.0,
                    "news_count": 0.0,
                    "material_news_count": 0.0,
                    "after_close_news_count": 0.0,
                    "days_since_last_material_news": 30.0,
                    "stale_sentiment_flag": 1.0,
                },
                index=target_index,
            )
            advanced_sent = pd.DataFrame(index=target_index)

        if not sent_aligned.empty and "sentiment_index" in sent_aligned.columns:
            _sentiment_qc(sent_aligned["sentiment_index"], price_df["close"])

        emb_features = _build_daily_embedding_features(session, target_index, pca_dim=cfg.embedding.pca_dim)
    else:
        sent_aligned = pd.DataFrame(
            {
                "sentiment_index": 0.0,
                "news_count": 0.0,
                "material_news_count": 0.0,
                "after_close_news_count": 0.0,
                "days_since_last_material_news": 30.0,
                "stale_sentiment_flag": 1.0,
            },
            index=target_index,
        )
        advanced_sent = pd.DataFrame(index=target_index)
        emb_features = pd.DataFrame(index=target_index)

    # ---- 4. LME / physical market features ----
    from deep_learning.data.lme_warehouse import (
        NON_STATIONARY_LME_FEATURES,
        PROXY_LME_FEATURES,
        compute_lme_features,
        compute_proxy_lme_features,
        fetch_lme_data,
    )
    from deep_learning.data.futures_curve import (
        SYNTHETIC_PROXY_CURVE_FEATURES,
        build_futures_features_from_yfinance,
    )

    lme_raw = fetch_lme_data(
        cfg.lme,
        start_date=start_date.date().isoformat(),
        end_date=end_date.date().isoformat(),
        session=session,
    )
    has_real_lme = not lme_raw.empty
    if has_real_lme:
        lme_features = compute_lme_features(lme_raw, windows=cfg.lme.stock_change_windows)
        lme_features = lme_features.reindex(target_index).ffill(limit=cfg.lme.max_ffill_days)
    else:
        lme_features = compute_proxy_lme_features(price_df)

    futures_features = build_futures_features_from_yfinance(
        session,
        target_symbol,
        cfg.training.lookback_days,
        end_date=end_date,
        hg_df=price_df,
    )
    futures_is_synthetic = bool(getattr(futures_features, "attrs", {}).get("is_synthetic_proxy", True))
    if not futures_features.empty:
        futures_features = futures_features.reindex(target_index).ffill(limit=3)
    else:
        futures_features = pd.DataFrame(index=target_index)

    # Track non-stationary or fake physical/curve proxy columns to suppress from TFT inputs
    suppressed_proxy_cols: set[str] = set()
    if has_real_lme:
        suppressed_proxy_cols.update(NON_STATIONARY_LME_FEATURES)
    else:
        suppressed_proxy_cols.update(PROXY_LME_FEATURES)
        suppressed_proxy_cols.update(lme_features.columns)
        suppressed_proxy_cols.add("regime_inventory_tightness")
    if futures_is_synthetic:
        suppressed_proxy_cols.update(SYNTHETIC_PROXY_CURVE_FEATURES)
        suppressed_proxy_cols.update(futures_features.columns)
        if not has_real_lme:
            suppressed_proxy_cols.add("regime_inventory_tightness")

    # ---- 5. Calendar (known future) ----
    calendar_features = _build_calendar_features(target_index)

    # ---- 6. Target: daily log-return path plus explicit weekly helpers ----
    close = _sanitize_close_spikes(price_df["close"], symbol=target_symbol)
    theta_prior_raw_col = f"{target_symbol}_theta_5d_prior"
    theta_prior_series_pre = compute_causal_theta_5d_series(
        close,
        horizon=cfg.forecast.primary_horizon_days,
        max_eval_tail=(
            None
            if drop_missing_target
            else max(320, cfg.model.max_encoder_length + 300)
        ),
    )
    theta_prior_series_pre.name = theta_prior_raw_col
    log_close = np.log(close)

    target_1d = log_close.shift(-cfg.forecast.auxiliary_horizon_days) - log_close
    target_1d.name = cfg.forecast.auxiliary_target_col

    target_5d = log_close.shift(-cfg.forecast.primary_horizon_days) - log_close
    target_5d.name = cfg.forecast.primary_target_col

    # ``target`` (TimeSeriesDataSet target) MUST be the realized 1-day log return
    # at bar t: log(close[t] / close[t-1]).  TimeSeriesDataSet slices decoder
    # horizons [t+1 .. t+5] relative to encoder end t, so summing decoder steps
    # equals log(close[t+5] / close[t]) == target_5d_log_return[t] without any
    # 1-day forward double-shift.
    target_model = log_close.diff()
    target_model.name = cfg.forecast.model_daily_target_col

    observed_return_1d = target_model.fillna(0.0)
    observed_return_1d.name = "observed_return_1d"

    realized_vol_20d = (
        observed_return_1d.rolling(20, min_periods=5)
        .std()
        .replace(0, np.nan)
        .ffill()
        .fillna(0.01)
    )
    realized_vol_20d.name = "realized_vol_20d"

    material_move_5d = (
        target_5d.abs()
        > cfg.forecast.material_move_vol_multiple
        * realized_vol_20d
        * np.sqrt(cfg.forecast.primary_horizon_days)
    ).astype(float)
    material_move_5d.name = "material_move_5d"

    from deep_learning.data.regime_features import build_regime_event_features

    sentiment_cols = [
        "sentiment_index",
        "news_count",
        "material_news_count",
        "after_close_news_count",
        "days_since_last_material_news",
        "stale_sentiment_flag",
    ]
    pre_regime = pd.concat(
        [
            price_features,
            sent_aligned[sentiment_cols],
            advanced_sent,
            lme_features,
            futures_features,
            target_model.to_frame(),
            observed_return_1d.to_frame(),
        ],
        axis=1,
    ).reindex(target_index).fillna(0.0)
    regime_features = build_regime_event_features(pre_regime)

    # ---- Assemble master DataFrame ----
    parts = [
        price_features,
        theta_prior_series_pre.to_frame(),
        sent_aligned[sentiment_cols],
        advanced_sent,
        emb_features,
        lme_features,
        futures_features,
        regime_features,
        calendar_features,
        target_model.to_frame(),
        target_1d.to_frame(),
        target_5d.to_frame(),
        realized_vol_20d.to_frame(),
        material_move_5d.to_frame(),
    ]

    master = pd.concat(parts, axis=1)
    master = master.loc[target_index]

    if drop_missing_target:
        # Filter only on causal realized 1d target so the final 5 valid OOS
        # weekly decoder windows [T-4 .. T] are preserved for TimeSeriesDataSet.
        valid_mask = master[cfg.forecast.model_daily_target_col].notna()
        master = master[valid_mask].copy()
    else:
        master = master.copy()

    # Trim leading indicator warmup rows instead of zero-filling incomplete
    # 20d/60d rolling windows (theta_5d_prior, macro_fair_value_zscore, ret20)
    # into StandardScaler when enough history is available.
    if len(master) >= 150:
        master = master.iloc[59:].copy()
    elif len(master) >= 80:
        master = master.iloc[20:].copy()

    master = master.fillna(0.0)

    # Sanitize column names: pytorch_forecasting forbids '.' and '-' in names
    master.columns = [
        col.replace(".", "_").replace("-", "_")
        for col in master.columns
    ]

    master["time_idx"] = np.arange(len(master))
    master["group_id"] = "copper"

    # Categorise columns – use sanitized calendar col names.
    # Restrict known future calendar covariates to stationary high-frequency
    # indicators (day_of_week) and forbid excluded annual calendar columns from
    # leaking into unknown features or MRMR.
    calendar_cols = [
        c.replace(".", "_").replace("-", "_")
        for c in calendar_features.columns
    ]
    calendar_col_set = set(calendar_cols)
    excluded_calendar_cols = calendar_col_set - set(TFT_KNOWN_CALENDAR_FEATURES)
    target_cols = [cfg.forecast.model_daily_target_col]

    non_feature_cols = {
        "time_idx",
        "group_id",
        cfg.forecast.model_daily_target_col,
        cfg.forecast.auxiliary_target_col,
        cfg.forecast.primary_target_col,
        "realized_vol_20d",
        "material_move_5d",
    }
    unsigned_vol_cols = {c for c in master.columns if _is_unsigned_vol_feature(c)}
    sanitized_proxy_cols = {
        c.replace(".", "_").replace("-", "_") for c in suppressed_proxy_cols
    }
    duplicate_ratio_cols = {
        c for c in master.columns
        if c.endswith("_price_sma_ratio")
        and f"{c[:-len('_price_sma_ratio')]}_close_sma_20_ratio" in master.columns
    }
    forbidden_cols = excluded_calendar_cols | unsigned_vol_cols | sanitized_proxy_cols | duplicate_ratio_cols
    all_feature_cols = [
        c for c in master.columns
        if c not in non_feature_cols and c not in forbidden_cols
    ]
    time_varying_known = [c for c in TFT_KNOWN_CALENDAR_FEATURES if c in master.columns]
    time_varying_unknown = [c for c in all_feature_cols if c not in calendar_col_set]

    from deep_learning.data.regime_features import REGIME_FEATURES

    target_prefix = target_symbol.replace(".", "_").replace("-", "_")
    theta_prior_col = f"{target_prefix}_theta_5d_prior"
    core_target_candidates = [
        *(
            f"{target_prefix}_{suffix}"
            for suffix in (
                "ret1",
                "ret5",
                "close_sma_20_ratio",
                "signed_vol_ratio",
                "theta_5d_prior",
            )
        ),
        "macro_fair_value_zscore",
    ]
    forced_unknown = [c for c in core_target_candidates if c in master.columns]
    time_varying_unknown = sorted(set(time_varying_unknown) | set(forced_unknown))

    missing_regime = [c for c in REGIME_FEATURES if c not in master.columns]
    if missing_regime:
        raise ValueError(f"Missing regime features: {missing_regime}")

    group_prefixes = list(
        dict.fromkeys(
            [target_prefix]
            + [sym.replace(".", "_").replace("-", "_") for sym in training_symbols]
        )
    )
    master.attrs["feature_selection_contract"] = {
        "known_features": list(time_varying_known),
        "forced_unknown_features": list(forced_unknown),
        "forbidden_features": sorted(forbidden_cols),
        "preserve_features": sorted(non_feature_cols),
        "group_prefixes": group_prefixes,
        "theta_prior_col": theta_prior_col,
    }

    logger.info(
        "Feature store raw: %d rows, %d unknown features, %d known features, %d embedding dims",
        len(master),
        len(time_varying_unknown),
        len(time_varying_known),
        len([c for c in master.columns if c.startswith("emb_pca_")]),
    )

    # ---- 7. MRMR & training-health feature selection ----
    # Fit on chronological training partition only to avoid lookahead leakage, and
    # prune zero-variance / sparse-backfill columns before model capacity is wasted.
    theta_prior_backup = (
        master[theta_prior_col].copy() if theta_prior_col in master.columns else None
    )
    if apply_mrmr and cfg.feature_store.mrmr_top_k > 0 and len(time_varying_unknown) > 0:
        from deep_learning.data.dataset import compute_dataset_split_bounds
        from deep_learning.data.feature_selection import select_features

        try:
            bounds = compute_dataset_split_bounds(master, cfg)
            train_end_pos = int(bounds["train_end_pos"])
        except ValueError:
            test_size = int(len(master) * cfg.training.test_ratio)
            val_size = int(len(master) * cfg.training.val_ratio)
            train_end_pos = len(master) - val_size - test_size
        if train_end_pos <= 0:
            raise ValueError("Feature selection requires a non-empty chronological training partition")

        horizon = int(cfg.forecast.primary_horizon_days)
        selection_end = (
            max(1, train_end_pos - horizon)
            if train_end_pos > horizon + 30
            else train_end_pos
        )

        master, time_varying_unknown, time_varying_known = select_features(
            master,
            target_col=cfg.forecast.primary_target_col,
            mrmr_top_k=cfg.feature_store.mrmr_top_k,
            known_features=time_varying_known,
            forced_unknown_features=forced_unknown,
            forbidden_features=sorted(forbidden_cols),
            preserve_features=sorted(non_feature_cols),
            selection_df=master.iloc[:selection_end],
            max_per_group=2,
            group_prefixes=group_prefixes,
        )
    if theta_prior_backup is not None and theta_prior_col not in master.columns:
        master[theta_prior_col] = theta_prior_backup.reindex(master.index).fillna(0.0)

    from deep_learning.data.validation import validate_weekly_target_contract

    validate_weekly_target_contract(
        master,
        mode="train" if drop_missing_target else "inference",
    )

    # Also return the last valid close price for baseline_price calculation
    valid_close = close.dropna()
    last_close = float(valid_close.iloc[-1]) if len(valid_close) > 0 else float('nan')

    configured_sha = os.environ.get("TFT_EXPECTED_DATA_SNAPSHOT_SHA", "").strip().lower()
    if configured_sha and apply_mrmr:
        actual_sha = data_snapshot_sha256(master)
        if actual_sha != configured_sha:
            raise RuntimeError(
                "Built TFT feature frame does not match TFT_EXPECTED_DATA_SNAPSHOT_SHA: "
                f"expected={configured_sha} actual={actual_sha}"
            )

    if snapshot_path is not None and use_snapshot and apply_mrmr:
        _write_feature_snapshot(
            snapshot_path,
            master,
            time_varying_unknown=time_varying_unknown,
            time_varying_known=time_varying_known,
            target_cols=target_cols,
            last_close=last_close,
            drop_missing_target=drop_missing_target,
        )

    return master, time_varying_unknown, time_varying_known, target_cols, last_close

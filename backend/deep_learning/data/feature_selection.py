"""
Feature Selection Pipeline for TFT-ASRO.

Two-stage dimensionality reduction to combat the curse of dimensionality
(~200+ features for ~500 training samples):

    Stage 1 — MRMR Pre-Filter (before training):
        Statistical filter using Mutual Information for relevance and
        Pearson correlation for redundancy.  Reduces 200+ → top-K features.

    Stage 2 — VSN Importance Pruning (after initial training):
        Uses TFT's Variable Selection Network weights to identify which
        features the model actually attends to, then prunes the bottom tier.

References:
    - Ding & Peng (2005) "Minimum Redundancy Feature Selection"
    - Lim et al. (2021) "Temporal Fusion Transformers" — VSN interpretability
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1: MRMR Pre-Filter
# ---------------------------------------------------------------------------

def _mutual_info_relevance(
    X: pd.DataFrame,
    y: pd.Series,
    n_neighbors: int = 5,
) -> pd.Series:
    """Compute MI(feature, target) for each column in X."""
    from sklearn.feature_selection import mutual_info_regression

    mi = mutual_info_regression(
        X.values,
        y.values,
        n_neighbors=n_neighbors,
        random_state=42,
    )
    return pd.Series(mi, index=X.columns)


def _hybrid_relevance(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    spearman_weight: float = 0.85,
    n_neighbors: int = 5,
) -> pd.Series:
    """Combine |Spearman| rank correlation with MI so directional signals outrank pure volatility."""
    from scipy.stats import spearmanr

    mi = _mutual_info_relevance(X, y, n_neighbors=n_neighbors)
    mi_arr = np.nan_to_num(mi.reindex(X.columns).to_numpy(dtype=np.float64), nan=0.0)

    y_vals = y.to_numpy(dtype=np.float64)
    y_constant = np.nanstd(y_vals) < 1e-12
    sp_vals: list[float] = []
    for col in X.columns:
        try:
            col_vals = X[col].to_numpy(dtype=np.float64)
            if y_constant or np.nanstd(col_vals) < 1e-12:
                sp_vals.append(0.0)
                continue
            stat = spearmanr(col_vals, y_vals).statistic
            sp_vals.append(abs(float(stat)) if stat is not None and np.isfinite(stat) else 0.0)
        except Exception:
            sp_vals.append(0.0)
    sp_arr = np.asarray(sp_vals, dtype=np.float64)

    mi_norm = mi_arr / max(float(mi_arr.max()) if mi_arr.size else 0.0, 1e-12)
    sp_norm = sp_arr / max(float(sp_arr.max()) if sp_arr.size else 0.0, 1e-12)
    combined = spearman_weight * sp_norm + (1.0 - spearman_weight) * mi_norm
    return pd.Series(combined, index=X.columns)


def _pairwise_correlation(X: pd.DataFrame) -> pd.DataFrame:
    """Absolute Pearson correlation matrix."""
    return X.corr().abs()


def _filter_healthy_training_features(
    df: pd.DataFrame,
    candidate_cols: list[str],
    *,
    min_std: float = 1e-8,
    min_non_mode_rate: float = 0.08,
) -> tuple[list[str], dict[str, str]]:
    """Drop zero-variance or sparse-backfill features using training split only."""
    if not candidate_cols or len(df) == 0:
        return list(candidate_cols), {}

    enforce_distribution_checks = len(df) >= 100
    kept: list[str] = []
    dropped: dict[str, str] = {}
    for col in candidate_cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        valid = series.dropna()
        if valid.empty:
            dropped[col] = "all_nan"
            continue
        if not enforce_distribution_checks:
            kept.append(col)
            continue

        arr = valid.to_numpy(dtype=np.float64)
        if not np.isfinite(arr).all():
            dropped[col] = "non_finite"
            continue
        std = float(np.std(arr))
        if std < min_std:
            dropped[col] = "zero_variance"
            continue

        mode_val = float(valid.mode().iloc[0])
        non_mode_rate = float((np.abs(arr - mode_val) > 1e-12).mean())
        if non_mode_rate < min_non_mode_rate:
            dropped[col] = f"sparse_coverage({non_mode_rate:.3f}<{min_non_mode_rate:.2f})"
            continue

        kept.append(col)

    if dropped:
        logger.info(
            "Filtered %d unhealthy training features (%s): %s",
            len(dropped),
            f"n={len(df)}",
            dict(list(dropped.items())[:10]),
        )
    return kept, dropped


def _infer_feature_group(col: str, known_prefixes: Sequence[str] = ()) -> str:
    """Infer feature group/family so one correlated symbol cannot crowd out all slots."""
    for prefix in sorted(set(known_prefixes), key=len, reverse=True):
        if prefix and col.startswith(f"{prefix}_"):
            return prefix
    if col.startswith("emb_pca_"):
        return "emb_pca"
    if (
        col.startswith("sentiment_")
        or col.startswith("event_")
        or col
        in {
            "news_count",
            "material_news_count",
            "after_close_news_count",
            "days_since_last_material_news",
            "stale_sentiment_flag",
        }
    ):
        return "sentiment"
    if col.startswith("regime_"):
        return "regime"
    if col.startswith("lme_") or col.startswith("proxy_") or col.startswith("futures_"):
        return "physical"
    return col


def mrmr_select(
    df: pd.DataFrame,
    target_col: str = "target",
    top_k: int = 80,
    exclude_cols: Optional[list[str]] = None,
    max_pairwise_corr: float = 0.85,
    forced_cols: Optional[list[str]] = None,
    max_per_group: Optional[int] = None,
    group_prefixes: Optional[Sequence[str]] = None,
) -> list[str]:
    """
    Scale-consistent Minimum Redundancy Maximum Relevance (mRMR-Q) selection.

    Combines normalized |Spearman| directional relevance with mutual information
    to prevent volatility-only features from crowding out directional signals,
    respects pre-seeded ``forced_cols`` within the ``top_k`` budget, and caps
    pairwise collinearity and per-group concentration.
    """
    if exclude_cols is None:
        exclude_cols = []

    meta_cols = {target_col, "time_idx", "group_id"} | set(exclude_cols)
    feature_cols = [c for c in df.columns if c not in meta_cols]

    if top_k <= 0:
        return []

    healthy_cols, _ = _filter_healthy_training_features(df, feature_cols)
    if healthy_cols:
        feature_cols = healthy_cols

    if len(feature_cols) <= top_k:
        logger.info("MRMR: %d features <= top_k=%d, skipping selection", len(feature_cols), top_k)
        return feature_cols

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    mask = y.notna() & X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]

    if len(X) < 30:
        logger.warning("MRMR: only %d valid samples, skipping", len(X))
        return feature_cols

    relevance = _hybrid_relevance(X, y)
    corr_matrix = _pairwise_correlation(X)

    n_feats = len(feature_cols)
    prefixes = tuple(group_prefixes or ())
    col_groups = [_infer_feature_group(c, prefixes) for c in feature_cols]
    group_counts: dict[str, int] = {}

    rel_arr = np.nan_to_num(
        relevance.reindex(feature_cols).to_numpy(dtype=np.float64),
        nan=0.0,
    )
    rel_max = float(rel_arr.max()) if len(rel_arr) else 0.0
    rel_norm = rel_arr / max(rel_max, 1e-12)
    corr_arr = np.nan_to_num(
        corr_matrix.reindex(index=feature_cols, columns=feature_cols).to_numpy(dtype=np.float64),
        nan=0.0,
    )

    col_to_idx = {c: i for i, c in enumerate(feature_cols)}
    forced_indices: list[int] = []
    if forced_cols:
        seen_forced: set[int] = set()
        for col in forced_cols:
            idx = col_to_idx.get(col)
            if idx is not None and idx not in seen_forced:
                forced_indices.append(idx)
                seen_forced.add(idx)

    selected_indices: list[int] = []
    selected_mask = np.zeros(n_feats, dtype=bool)
    running_corr_sum = np.zeros(n_feats, dtype=np.float64)
    max_corr_to_selected = np.zeros(n_feats, dtype=np.float64)

    if forced_indices:
        for idx in forced_indices:
            selected_indices.append(idx)
            selected_mask[idx] = True
            group_counts[col_groups[idx]] = group_counts.get(col_groups[idx], 0) + 1
            running_corr_sum += corr_arr[:, idx]
            max_corr_to_selected = np.maximum(max_corr_to_selected, corr_arr[:, idx])
    else:
        first_idx = int(np.argmax(rel_norm))
        selected_indices.append(first_idx)
        selected_mask[first_idx] = True
        group_counts[col_groups[first_idx]] = group_counts.get(col_groups[first_idx], 0) + 1
        running_corr_sum += corr_arr[:, first_idx]
        max_corr_to_selected = np.maximum(max_corr_to_selected, corr_arr[:, first_idx])

    while len(selected_indices) < top_k:
        redundancy = running_corr_sum / max(1, len(selected_indices))
        scores = rel_norm / (redundancy + 0.50)
        scores[selected_mask] = -np.inf

        corr_eligible = scores.copy()
        corr_eligible[max_corr_to_selected > max_pairwise_corr] = -np.inf

        group_eligible = corr_eligible.copy()
        if max_per_group is not None and max_per_group > 0:
            for idx, group_name in enumerate(col_groups):
                if group_counts.get(group_name, 0) >= max_per_group:
                    group_eligible[idx] = -np.inf

        best_idx = int(np.argmax(group_eligible))
        if group_eligible[best_idx] == -np.inf:
            # First relax per-group concentration cap while keeping collinearity guard
            best_idx = int(np.argmax(corr_eligible))
            if corr_eligible[best_idx] == -np.inf:
                # Final fallback if hard collinearity cap exhausted all candidates before top_k
                best_idx = int(np.argmax(scores))
                if scores[best_idx] == -np.inf:
                    break

        selected_indices.append(best_idx)
        selected_mask[best_idx] = True
        group_counts[col_groups[best_idx]] = group_counts.get(col_groups[best_idx], 0) + 1
        running_corr_sum += corr_arr[:, best_idx]
        max_corr_to_selected = np.maximum(max_corr_to_selected, corr_arr[:, best_idx])

    selected = [feature_cols[i] for i in selected_indices]

    logger.info(
        "MRMR selected %d/%d features (top relevance=%.4f, bottom=%.4f)",
        len(selected), len(feature_cols),
        relevance[selected[0]] if selected else 0,
        relevance[selected[-1]] if selected else 0,
    )
    return selected


# ---------------------------------------------------------------------------
# Stage 2: VSN Importance Pruning
# ---------------------------------------------------------------------------

def vsn_prune(
    importance: dict[str, float],
    feature_list: list[str],
    min_features: int = 40,
    cumulative_threshold: float = 0.92,
) -> list[str]:
    """
    Prune features using TFT Variable Selection Network importance scores.

    Keeps features until their cumulative importance exceeds the threshold,
    with a minimum floor to avoid over-pruning.

    Args:
        importance:           {feature_name: normalised_importance} from
                              ``get_variable_importance()``.
        feature_list:         Full list of features currently in use.
        min_features:         Never prune below this count.
        cumulative_threshold: Keep features until cumulative importance hits this.

    Returns:
        Pruned feature list (subset of feature_list).
    """
    if not importance:
        logger.info("VSN prune: no importance scores, returning all %d features", len(feature_list))
        return feature_list

    scored = {f: importance.get(f, 0.0) for f in feature_list}
    ranked = sorted(scored.items(), key=lambda x: -x[1])

    total = sum(v for _, v in ranked)
    if total < 1e-12:
        return feature_list

    kept: list[str] = []
    cumulative = 0.0

    for feat, score in ranked:
        kept.append(feat)
        cumulative += score / total
        if cumulative >= cumulative_threshold and len(kept) >= min_features:
            break

    if len(kept) < min_features:
        kept = [f for f, _ in ranked[:min_features]]

    logger.info(
        "VSN pruned %d → %d features (cumulative importance=%.2f%%)",
        len(feature_list), len(kept), cumulative * 100,
    )
    return kept


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

def select_features(
    df: pd.DataFrame,
    target_col: str = "target",
    mrmr_top_k: int = 80,
    known_features: Optional[list[str]] = None,
    forced_unknown_features: Optional[list[str]] = None,
    forbidden_features: Optional[list[str]] = None,
    selection_df: Optional[pd.DataFrame] = None,
    max_per_group: Optional[int] = None,
    group_prefixes: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Run MRMR selection on unknown features while preserving known features.

    Args:
        df:              Master DataFrame from feature_store.
        target_col:      Target column name.
        mrmr_top_k:      How many unknown features to keep.
        known_features:  List of time_varying_known_reals (calendar etc.) — always kept.
        selection_df: Training-only frame used to fit MRMR. ``df`` remains the
            full chronological frame returned to downstream split builders.

    Returns:
        (filtered_df, new_unknown_features, known_features)
    """
    if known_features is None:
        known_features = []
    if forced_unknown_features is None:
        forced_unknown_features = []
    if forbidden_features is None:
        forbidden_features = []

    if selection_df is None:
        selection_df = df
    missing_selection_cols = [col for col in df.columns if col not in selection_df.columns]
    if missing_selection_cols:
        raise ValueError(
            "selection_df must contain every source column required for feature selection: "
            f"{missing_selection_cols}"
        )

    forbidden = set(forbidden_features)
    meta_cols = ["time_idx", "group_id", target_col]
    preserve_cols = set(meta_cols) | set(known_features) | forbidden

    unknown_candidates = [
        c for c in df.columns
        if c not in preserve_cols and c not in forbidden
    ]
    raw_forced_unknown = [
        c for c in forced_unknown_features
        if c in df.columns and c not in preserve_cols
    ]

    healthy_unknown, _ = _filter_healthy_training_features(selection_df, unknown_candidates)
    healthy_set = set(healthy_unknown)
    forced_unknown = [c for c in raw_forced_unknown if c in healthy_set]
    if len(raw_forced_unknown) != len(forced_unknown):
        deferred = [c for c in raw_forced_unknown if c not in healthy_set]
        logger.info("Deferred sparse/constant forced features: %s", deferred)

    # Preserve healthy candidates while keeping non-empty fallback when synthetic frames fail filter
    effective_candidates = healthy_unknown if healthy_unknown else unknown_candidates

    if len(effective_candidates) <= mrmr_top_k:
        logger.info(
            "Feature selection: %d unknown features <= top_k=%d, pruning unhealthy only",
            len(effective_candidates), mrmr_top_k,
        )
        selected = sorted(set(effective_candidates) | set(forced_unknown))
        keep_cols = [c for c in list(preserve_cols) + selected if c in df.columns]
        return df[keep_cols].copy(), selected, known_features

    mrmr_kwargs: dict = {
        "target_col": target_col,
        "top_k": mrmr_top_k,
        "exclude_cols": list(
            preserve_cols | forbidden | (set(unknown_candidates) - set(effective_candidates))
        ),
    }
    if forced_unknown:
        mrmr_kwargs["forced_cols"] = forced_unknown
    if max_per_group is not None:
        mrmr_kwargs["max_per_group"] = max_per_group
    if group_prefixes is not None:
        mrmr_kwargs["group_prefixes"] = group_prefixes

    selected_unknown = mrmr_select(selection_df, **mrmr_kwargs)
    selected_unknown = sorted(set(selected_unknown) | set(forced_unknown))

    keep_cols = list(preserve_cols) + selected_unknown
    keep_cols = [c for c in keep_cols if c in df.columns]

    filtered = df[keep_cols].copy()

    logger.info(
        "Feature selection complete: %d cols → %d cols "
        "(%d unknown, %d known, %d meta)",
        len(df.columns), len(filtered.columns),
        len(selected_unknown), len(known_features), len(meta_cols),
    )

    return filtered, selected_unknown, known_features

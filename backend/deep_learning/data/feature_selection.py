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
from typing import Optional

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


def _pairwise_correlation(X: pd.DataFrame) -> pd.DataFrame:
    """Absolute Pearson correlation matrix."""
    return X.corr().abs()


def mrmr_select(
    df: pd.DataFrame,
    target_col: str = "target",
    top_k: int = 80,
    exclude_cols: Optional[list[str]] = None,
) -> list[str]:
    """
    Minimum Redundancy Maximum Relevance feature selection.

    Greedily selects features that are highly relevant to the target while
    being minimally redundant with already-selected features.

    Args:
        df:          DataFrame containing features and the target column.
        target_col:  Name of the target column.
        top_k:       Number of features to select.
        exclude_cols: Columns to exclude from selection (e.g. time_idx, group_id).

    Returns:
        List of selected feature names, ordered by selection round.
    """
    if exclude_cols is None:
        exclude_cols = []

    meta_cols = {target_col, "time_idx", "group_id"} | set(exclude_cols)
    feature_cols = [c for c in df.columns if c not in meta_cols]

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

    relevance = _mutual_info_relevance(X, y)
    corr_matrix = _pairwise_correlation(X)

    selected: list[str] = []
    remaining = set(feature_cols)

    first = relevance.idxmax()
    selected.append(first)
    remaining.discard(first)

    for _ in range(top_k - 1):
        if not remaining:
            break

        best_score = -np.inf
        best_feat = None

        for feat in remaining:
            rel = relevance[feat]
            if not selected:
                score = rel
            else:
                redundancy = corr_matrix.loc[feat, selected].mean()
                score = rel - redundancy

            if score > best_score:
                best_score = score
                best_feat = feat

        if best_feat is None:
            break

        selected.append(best_feat)
        remaining.discard(best_feat)

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
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Run MRMR selection on unknown features while preserving known features.

    Args:
        df:              Master DataFrame from feature_store.
        target_col:      Target column name.
        mrmr_top_k:      How many unknown features to keep.
        known_features:  List of time_varying_known_reals (calendar etc.) — always kept.

    Returns:
        (filtered_df, new_unknown_features, known_features)
    """
    if known_features is None:
        known_features = []

    meta_cols = ["time_idx", "group_id", target_col]
    preserve_cols = set(meta_cols) | set(known_features)

    unknown_candidates = [c for c in df.columns if c not in preserve_cols]

    if len(unknown_candidates) <= mrmr_top_k:
        logger.info(
            "Feature selection: %d unknown features <= top_k=%d, no pruning needed",
            len(unknown_candidates), mrmr_top_k,
        )
        return df, unknown_candidates, known_features

    selected_unknown = mrmr_select(
        df,
        target_col=target_col,
        top_k=mrmr_top_k,
        exclude_cols=list(preserve_cols),
    )

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

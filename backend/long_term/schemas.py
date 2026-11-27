"""Pydantic schemas for the Long-Term Copper Outlook Engine (20D / 60D)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

StanceType = Literal["BULLISH", "NEUTRAL", "BEARISH"]
InstitutionalRatingType = Literal[
    "STRONG_OVERWEIGHT",
    "OVERWEIGHT",
    "NEUTRAL",
    "UNDERWEIGHT",
    "STRONG_UNDERWEIGHT",
]
PillarKeyType = Literal[
    "trend_structure",
    "usd_macro",
    "china_industrial_demand",
    "cross_asset_confirmation",
    "physical_positioning",
]
HorizonKeyType = Literal["20D", "60D", "120D"]


class LongTermPillarMetric(BaseModel):
    """Single human-readable quantitative metric inside a macro scorecard pillar."""

    name: str
    value: Optional[float] = None
    formatted_value: str
    signal: StanceType


class LongTermMacroPillar(BaseModel):
    """Normalized directional score and key drivers for one macro pillar."""

    key: PillarKeyType
    label: str
    score: float = Field(..., ge=-1.0, le=1.0)
    score_pct: int = Field(..., ge=0, le=100)
    stance: StanceType
    summary: str
    metrics: list[LongTermPillarMetric] = Field(default_factory=list)


class LongTermMacroScorecard(BaseModel):
    """4-pillar macro factor scorecard summarizing current regime conditions."""

    composite_score: float = Field(..., ge=-1.0, le=1.0)
    composite_stance: StanceType
    pillars: dict[str, LongTermMacroPillar] = Field(default_factory=dict)


class LongTermDriverItem(BaseModel):
    """Top standardized feature contribution from the Ridge macro-trend model."""

    feature: str
    label: str
    pillar: str
    z_score: float
    contribution_return: float
    direction: StanceType


class LongTermEnsembleComponents(BaseModel):
    """Decomposition of the long-term forecast committee."""

    macro_ridge_return: float
    theta_damped_return: float
    macro_ridge_weight: float = Field(..., ge=0.0, le=1.0)
    theta_weight: float = Field(..., ge=0.0, le=1.0)
    realized_vol_annualized: float = Field(..., ge=0.0)
    horizon_vol: float = Field(..., ge=0.0)
    cqr_q10_width: float = Field(..., ge=0.0)
    cqr_q90_width: float = Field(..., ge=0.0)
    catboost_return: Optional[float] = None
    xgboost_return: Optional[float] = None
    mlp_plr_return: Optional[float] = None
    tree_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    mlp_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    shrinkage_alpha: Optional[float] = Field(None, ge=0.0, le=1.0)


class LongTermHorizonForecast(BaseModel):
    """Calibrated point, quantile, and regime forecast for one long-term horizon."""

    horizon_key: HorizonKeyType
    horizon_label: str
    horizon_days: int
    horizon_weeks: int
    target_date: Optional[str] = None
    reference_price: float
    target_price_q50: float
    expected_return: float = Field(..., description="Expected simple return as decimal (e.g. 0.035 = +3.5%).")
    expected_return_pct: float = Field(..., description="Expected simple return in percentage points (e.g. 3.5).")
    expected_log_return: float
    price_q10: float
    price_q25: float
    price_q75: float
    price_q90: float
    return_q10: float
    return_q25: float
    return_q75: float
    return_q90: float
    stance: StanceType
    institutional_rating: Optional[InstitutionalRatingType] = None
    asymmetry_ratio: Optional[float] = Field(
        None,
        description="Upside vs downside PI80 asymmetry ratio (price_q90 - spot) / max(spot - price_q10, 1e-4).",
    )
    prob_bullish: float = Field(..., ge=0.0, le=1.0)
    prob_neutral: float = Field(..., ge=0.0, le=1.0)
    prob_bearish: float = Field(..., ge=0.0, le=1.0)
    conviction: float = Field(..., ge=0.0, le=1.0)
    conviction_label: Literal["HIGH", "MEDIUM", "LOW"]
    regime_threshold_pct: float = Field(..., ge=0.0)
    components: LongTermEnsembleComponents
    key_drivers: list[LongTermDriverItem] = Field(default_factory=list)


class LongTermFoldMetrics(BaseModel):
    """Expanding-origin purged walk-forward fold telemetry."""

    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    n_non_overlapping: int
    directional_accuracy: float
    non_overlapping_da: float
    mae_vs_naive_zero: float
    mae_vs_theta: float
    sharpe_ratio: float
    pi80_coverage: float
    brier_score: float


class LongTermValidationSummary(BaseModel):
    """Purged horizon-matched OOS walk-forward summary for one horizon."""

    horizon_key: HorizonKeyType
    horizon_weeks: int
    purge_gap_weeks: int
    annualisation: float
    n_folds: int
    n_purged_oos_samples: int
    n_non_overlapping_samples: int
    directional_accuracy: float
    non_overlapping_da: float
    mae_vs_naive_zero: float
    mae_vs_theta: float
    theta_directional_accuracy: float
    sharpe_ratio: float
    non_overlapping_sharpe: float
    pi80_coverage: float
    pi50_coverage: float
    pinball_loss_mean: float
    pinball_skill_vs_empirical: float
    brier_score: float
    brier_score_prior_baseline: Optional[float] = Field(
        None, description="Brier score of the training class-frequency prior on the same OOS rows."
    )
    regime_accuracy: float
    quantile_crossing_rate: float
    folds: list[LongTermFoldMetrics] = Field(default_factory=list)


class LongTermDataWindow(BaseModel):
    """Metadata describing the historical weekly dataset used to build the outlook."""

    start_date: Optional[str] = None
    end_date: Optional[str] = None
    weekly_bars: int = 0
    years_covered: float = 0.0
    fred_macro_enabled: bool = False
    physical_data_enabled: bool = False
    cot_positioning_enabled: bool = False
    symbols_used: list[str] = Field(default_factory=list)


class LongTermOutlookResponse(BaseModel):
    """Snapshot-only API response contract for GET /api/outlook/long-term/{symbol}."""

    status: Literal["ok", "unavailable"]
    symbol: str
    as_of_date: Optional[str] = None
    generated_at: Optional[str] = None
    model_version: Optional[str] = None
    source: Literal["snapshot", "unavailable"]
    current_price: Optional[float] = None
    available_horizons: list[str] = Field(default_factory=lambda: ["20D", "60D", "120D"])
    default_horizon: str = "60D"
    data_window: Optional[LongTermDataWindow] = None
    horizons: dict[str, LongTermHorizonForecast] = Field(default_factory=dict)
    macro_scorecard: Optional[LongTermMacroScorecard] = None
    validation: dict[str, LongTermValidationSummary] = Field(default_factory=dict)
    message: Optional[str] = None

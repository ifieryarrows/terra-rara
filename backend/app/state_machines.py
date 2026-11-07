"""
Explicit finite state machines for CopperMind backend lifecycles and protocols.

Centralizes state definitions, transition tables, and pure transition reducers
that previously lived as implicit combinations of string literals, nullable
timestamps, boolean flags, and module-level globals across the backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import StrEnum
from typing import Any, Optional


# =============================================================================
# 1. Pipeline Run Lifecycle & Health Projection State Machine
# =============================================================================


class PipelineRunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SKIPPED_LOCKED = "skipped_locked"
    SUCCESS = "success"
    DEGRADED = "degraded"
    FAILED = "failed"


class PipelineQualityState(StrEnum):
    QUEUED = "queued"
    OK = "ok"
    STALE = "stale"
    DEGRADED = "degraded"
    SKIPPED = "skipped"
    FAILED = "failed"


class PipelineRunEvent(StrEnum):
    ENQUEUE = "enqueue"
    ENQUEUE_FAILED = "enqueue_failed"
    WORKER_START = "worker_start"
    SKIP_LOCKED = "skip_locked"
    COMPLETE_SUCCESS = "complete_success"
    COMPLETE_DEGRADED = "complete_degraded"
    COMPLETE_FAILED = "complete_failed"
    EXPIRE_ORPHAN = "expire_orphan"
    LOST_LOCK = "lost_lock"


VALID_PIPELINE_TRANSITIONS: dict[
    tuple[Optional[PipelineRunStatus], PipelineRunEvent],
    tuple[PipelineRunStatus, PipelineQualityState],
] = {
    (None, PipelineRunEvent.ENQUEUE): (
        PipelineRunStatus.QUEUED,
        PipelineQualityState.QUEUED,
    ),
    (PipelineRunStatus.QUEUED, PipelineRunEvent.ENQUEUE_FAILED): (
        PipelineRunStatus.FAILED,
        PipelineQualityState.FAILED,
    ),
    (None, PipelineRunEvent.WORKER_START): (
        PipelineRunStatus.RUNNING,
        PipelineQualityState.QUEUED,
    ),
    (PipelineRunStatus.QUEUED, PipelineRunEvent.WORKER_START): (
        PipelineRunStatus.RUNNING,
        PipelineQualityState.QUEUED,
    ),
    (PipelineRunStatus.RUNNING, PipelineRunEvent.WORKER_START): (
        PipelineRunStatus.RUNNING,
        PipelineQualityState.QUEUED,
    ),
    (PipelineRunStatus.QUEUED, PipelineRunEvent.SKIP_LOCKED): (
        PipelineRunStatus.SKIPPED_LOCKED,
        PipelineQualityState.SKIPPED,
    ),
    (PipelineRunStatus.RUNNING, PipelineRunEvent.SKIP_LOCKED): (
        PipelineRunStatus.SKIPPED_LOCKED,
        PipelineQualityState.SKIPPED,
    ),
    (PipelineRunStatus.RUNNING, PipelineRunEvent.COMPLETE_SUCCESS): (
        PipelineRunStatus.SUCCESS,
        PipelineQualityState.OK,
    ),
    (PipelineRunStatus.RUNNING, PipelineRunEvent.COMPLETE_DEGRADED): (
        PipelineRunStatus.DEGRADED,
        PipelineQualityState.DEGRADED,
    ),
    (PipelineRunStatus.QUEUED, PipelineRunEvent.COMPLETE_FAILED): (
        PipelineRunStatus.FAILED,
        PipelineQualityState.FAILED,
    ),
    (PipelineRunStatus.RUNNING, PipelineRunEvent.COMPLETE_FAILED): (
        PipelineRunStatus.FAILED,
        PipelineQualityState.FAILED,
    ),
    (PipelineRunStatus.QUEUED, PipelineRunEvent.EXPIRE_ORPHAN): (
        PipelineRunStatus.FAILED,
        PipelineQualityState.FAILED,
    ),
    (PipelineRunStatus.RUNNING, PipelineRunEvent.EXPIRE_ORPHAN): (
        PipelineRunStatus.FAILED,
        PipelineQualityState.FAILED,
    ),
    (PipelineRunStatus.RUNNING, PipelineRunEvent.LOST_LOCK): (
        PipelineRunStatus.FAILED,
        PipelineQualityState.FAILED,
    ),
}


TERMINAL_PIPELINE_STATUSES: frozenset[PipelineRunStatus] = frozenset(
    {
        PipelineRunStatus.SKIPPED_LOCKED,
        PipelineRunStatus.SUCCESS,
        PipelineRunStatus.DEGRADED,
        PipelineRunStatus.FAILED,
    }
)


def transition_pipeline_run(
    metrics: Any,
    event: PipelineRunEvent,
    *,
    now: Optional[datetime] = None,
    error_message: Optional[str] = None,
    quality_state_override: Optional[str] = None,
    job_id: Optional[str] = None,
) -> tuple[PipelineRunStatus, PipelineQualityState]:
    """
    Validate and apply an explicit state transition onto a PipelineRunMetrics row.
    """
    ts = now or datetime.now(timezone.utc)
    raw_current = getattr(metrics, "status", None)
    current_status: Optional[PipelineRunStatus] = None
    if raw_current is not None:
        try:
            current_status = PipelineRunStatus(str(raw_current))
        except ValueError:
            current_status = None

    transition_key = (current_status, event)
    if transition_key not in VALID_PIPELINE_TRANSITIONS:
        fallback_map: dict[PipelineRunEvent, tuple[PipelineRunStatus, PipelineQualityState]] = {
            PipelineRunEvent.ENQUEUE: (PipelineRunStatus.QUEUED, PipelineQualityState.QUEUED),
            PipelineRunEvent.ENQUEUE_FAILED: (PipelineRunStatus.FAILED, PipelineQualityState.FAILED),
            PipelineRunEvent.WORKER_START: (PipelineRunStatus.RUNNING, PipelineQualityState.QUEUED),
            PipelineRunEvent.SKIP_LOCKED: (PipelineRunStatus.SKIPPED_LOCKED, PipelineQualityState.SKIPPED),
            PipelineRunEvent.COMPLETE_SUCCESS: (PipelineRunStatus.SUCCESS, PipelineQualityState.OK),
            PipelineRunEvent.COMPLETE_DEGRADED: (PipelineRunStatus.DEGRADED, PipelineQualityState.DEGRADED),
            PipelineRunEvent.COMPLETE_FAILED: (PipelineRunStatus.FAILED, PipelineQualityState.FAILED),
            PipelineRunEvent.EXPIRE_ORPHAN: (PipelineRunStatus.FAILED, PipelineQualityState.FAILED),
            PipelineRunEvent.LOST_LOCK: (PipelineRunStatus.FAILED, PipelineQualityState.FAILED),
        }
        next_status, next_quality = fallback_map[event]
    else:
        next_status, next_quality = VALID_PIPELINE_TRANSITIONS[transition_key]

    if quality_state_override is not None:
        try:
            next_quality = PipelineQualityState(str(quality_state_override))
        except ValueError:
            pass

    metrics.status = next_status.value
    metrics.quality_state = next_quality.value

    if job_id is not None:
        metrics.job_id = job_id

    if event == PipelineRunEvent.ENQUEUE:
        if getattr(metrics, "enqueued_at", None) is None:
            metrics.enqueued_at = ts
        if getattr(metrics, "run_started_at", None) is None:
            metrics.run_started_at = ts
    elif event == PipelineRunEvent.WORKER_START:
        metrics.run_started_at = ts
        metrics.worker_started_at = ts
    elif next_status in TERMINAL_PIPELINE_STATUSES:
        metrics.run_completed_at = ts
        started_at = getattr(metrics, "run_started_at", None)
        if started_at is not None:
            norm_start = started_at if started_at.tzinfo is not None else started_at.replace(tzinfo=timezone.utc)
            metrics.duration_seconds = (ts - norm_start).total_seconds()

    if error_message is not None:
        metrics.error_message = error_message

    return next_status, next_quality


def map_pipeline_status_to_external(raw_status: Optional[str]) -> Optional[str]:
    """
    Map internal PipelineRunMetrics.status to external health `pipeline_status`.
    """
    normalized = (raw_status or "").strip().lower()
    if not normalized:
        return None
    if normalized == PipelineRunStatus.SUCCESS.value:
        return PipelineQualityState.OK.value
    return normalized


# =============================================================================
# 2. Market Heatmap Cache SWR & Metadata Circuit Breaker State Machines
# =============================================================================


class HeatmapExternalState(StrEnum):
    EMPTY = "empty"
    REFRESHING = "refreshing"
    FRESH = "fresh"
    STALE = "stale"


class HeatmapCachePhase(StrEnum):
    EMPTY = "empty"
    REFRESHING_EMPTY = "refreshing_empty"
    FRESH = "fresh"
    STALE_REVALIDATING = "stale_revalidating"
    STALE_REFRESHING = "stale_refreshing"
    RETRY_BACKOFF = "retry_backoff"


@dataclass(frozen=True)
class HeatmapCacheDecision:
    external_state: HeatmapExternalState
    phase: HeatmapCachePhase
    is_stale: bool
    stuck_lock_expired: bool
    refresh_in_progress: bool
    retry_backoff: bool
    should_trigger_refresh: bool


def resolve_heatmap_external_state(
    *,
    payload_count: int,
    is_stale: bool,
    refresh_in_progress: bool,
) -> HeatmapExternalState:
    """
    Resolve the external `cache_state` / `X-Heatmap-Cache` value from SWR flags.
    """
    if payload_count == 0:
        return HeatmapExternalState.EMPTY
    if refresh_in_progress:
        return HeatmapExternalState.REFRESHING
    if is_stale:
        return HeatmapExternalState.STALE
    return HeatmapExternalState.FRESH


def evaluate_heatmap_cache_state(
    *,
    payload_count: int,
    needs_enrichment: bool = False,
    expires_at: Optional[datetime],
    refresh_started_at: Optional[datetime],
    refresh_error: Optional[str],
    now: datetime,
    stuck_timeout_seconds: float = 180.0,
) -> HeatmapCacheDecision:
    """
    Pure state machine reducer for HeatmapCache Stale-While-Revalidate (SWR) lifecycle.
    """
    norm_expires = expires_at
    if norm_expires is not None and norm_expires.tzinfo is None:
        norm_expires = norm_expires.replace(tzinfo=timezone.utc)

    norm_started = refresh_started_at
    if norm_started is not None and norm_started.tzinfo is None:
        norm_started = norm_started.replace(tzinfo=timezone.utc)

    stuck_lock_expired = bool(
        norm_started is not None
        and (now - norm_started).total_seconds() > stuck_timeout_seconds
    )
    refresh_in_progress = bool(norm_started is not None and not stuck_lock_expired)
    retry_backoff = bool(
        refresh_error and norm_expires is not None and now < norm_expires
    )
    is_stale = bool(
        not norm_expires
        or now >= norm_expires
        or needs_enrichment
        or bool(refresh_error)
    )
    should_trigger_refresh = bool(
        (payload_count == 0 or is_stale)
        and not refresh_in_progress
        and not retry_backoff
    )
    effective_refresh_in_progress = refresh_in_progress or should_trigger_refresh

    external_state = resolve_heatmap_external_state(
        payload_count=payload_count,
        is_stale=is_stale,
        refresh_in_progress=effective_refresh_in_progress,
    )

    if payload_count == 0:
        phase = (
            HeatmapCachePhase.REFRESHING_EMPTY
            if effective_refresh_in_progress
            else (
                HeatmapCachePhase.RETRY_BACKOFF
                if retry_backoff
                else HeatmapCachePhase.EMPTY
            )
        )
    elif retry_backoff:
        phase = HeatmapCachePhase.RETRY_BACKOFF
    elif effective_refresh_in_progress:
        phase = HeatmapCachePhase.STALE_REFRESHING
    elif is_stale:
        phase = HeatmapCachePhase.STALE_REVALIDATING
    else:
        phase = HeatmapCachePhase.FRESH

    return HeatmapCacheDecision(
        external_state=external_state,
        phase=phase,
        is_stale=is_stale,
        stuck_lock_expired=stuck_lock_expired,
        refresh_in_progress=effective_refresh_in_progress,
        retry_backoff=retry_backoff,
        should_trigger_refresh=should_trigger_refresh,
    )


class HeatmapMetadataCircuitState(StrEnum):
    CLOSED = "closed"
    OPEN_RATE_LIMITED = "open_rate_limited"
    OPEN_PROVIDER_ERROR = "open_provider_error"


@dataclass
class HeatmapMetadataCircuitBreaker:
    """
    Explicit circuit breaker for per-ticker yfinance metadata enrichment inside heatmap refreshes.
    """

    state: HeatmapMetadataCircuitState = HeatmapMetadataCircuitState.CLOSED
    error_message: Optional[str] = None

    @property
    def is_open(self) -> bool:
        return self.state == HeatmapMetadataCircuitState.OPEN_RATE_LIMITED

    @property
    def is_rate_limited(self) -> bool:
        return self.state == HeatmapMetadataCircuitState.OPEN_RATE_LIMITED

    def trip_rate_limit(self, message: str) -> None:
        self.state = HeatmapMetadataCircuitState.OPEN_RATE_LIMITED
        if not self.error_message:
            self.error_message = message

    def record_provider_error(self, message: str) -> None:
        if self.state == HeatmapMetadataCircuitState.CLOSED:
            self.state = HeatmapMetadataCircuitState.OPEN_PROVIDER_ERROR
        if not self.error_message:
            self.error_message = message

    def select_backoff(
        self,
        *,
        rate_limit_backoff: timedelta,
        provider_backoff: timedelta,
    ) -> timedelta:
        if self.state == HeatmapMetadataCircuitState.OPEN_RATE_LIMITED:
            return rate_limit_backoff
        return provider_backoff


# =============================================================================
# 3. Dual-Model Artifact Promotion & TFT Predictor Loader State Machine
# =============================================================================


class XGBArtifactStatus(StrEnum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    ARCHIVED = "archived"


class TFTCalibrationStatus(StrEnum):
    FIT = "fit"
    SKIPPED_OVERCOVERED = "skipped_interval_already_overcovered"
    FIT_WIDTH_FLOOR = "fit_width_floor"


class TFTPredictorLifecycleState(StrEnum):
    UNINITIALIZED = "uninitialized"
    ARTIFACTS_READY = "artifacts_ready"
    MODEL_LOADED = "model_loaded"
    DEGRADED = "degraded"


@dataclass
class TFTPredictorLifecycleMachine:
    """
    Explicit lifecycle state machine for TFTPredictor lazy artifact discovery and loading.
    """

    state: TFTPredictorLifecycleState = TFTPredictorLifecycleState.UNINITIALIZED
    hub_checked: bool = False
    metadata_checked: bool = False
    artifact_issue: Optional[str] = None

    def mark_artifacts_ready(self) -> None:
        self.artifact_issue = None
        if self.state != TFTPredictorLifecycleState.MODEL_LOADED:
            self.state = TFTPredictorLifecycleState.ARTIFACTS_READY

    def mark_model_loaded(self) -> None:
        self.artifact_issue = None
        self.state = TFTPredictorLifecycleState.MODEL_LOADED

    def mark_degraded(self, issue: str) -> None:
        self.artifact_issue = issue
        self.state = TFTPredictorLifecycleState.DEGRADED

    def reset_for_reload(self) -> None:
        self.state = TFTPredictorLifecycleState.UNINITIALIZED
        self.artifact_issue = None


# =============================================================================
# 4. Snapshot Freshness, Cross-Table Validity & Forecast Health State Machine
# =============================================================================


class SnapshotQualityState(StrEnum):
    OK = "ok"
    STALE = "stale"
    DEGRADED = "degraded"
    MISSING = "missing"


class ModelServingState(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    RETRAIN_REQUIRED = "retrain_required"


@dataclass(frozen=True)
class XGBSnapshotEvaluation:
    is_stale: bool
    age_hours: float
    quality_state: str
    model_state: str


def evaluate_xgb_snapshot_state(
    *,
    generated_at: datetime,
    now: datetime,
    stale_threshold_hours: float = 36.0,
) -> XGBSnapshotEvaluation:
    """
    Pure state machine reducer for XGBoost AnalysisSnapshot freshness and serving state.
    """
    gen_ts = generated_at
    if gen_ts.tzinfo is None:
        gen_ts = gen_ts.replace(tzinfo=timezone.utc)
    age_hours = (now - gen_ts).total_seconds() / 3600.0
    is_stale = age_hours > stale_threshold_hours

    if is_stale:
        quality_state = SnapshotQualityState.STALE.value
        model_state = ModelServingState.DEGRADED.value
    else:
        quality_state = SnapshotQualityState.OK.value
        model_state = ModelServingState.OK.value

    return XGBSnapshotEvaluation(
        is_stale=is_stale,
        age_hours=age_hours,
        quality_state=quality_state,
        model_state=model_state,
    )


def evaluate_tft_snapshot_freshness(
    *,
    snapshot_generated_at: Optional[datetime],
    promoted_model_trained_at: Optional[datetime],
    reference_price_date_raw: Optional[str],
    parsed_ref_date: Optional[date],
    latest_price_date: Optional[date],
    today_utc: date,
    max_unanchored_staleness_days: int = 3,
) -> list[str]:
    """
    Pure cross-table freshness evaluator for `TFTPredictionSnapshot`.
    """
    reasons: list[str] = []
    if (
        promoted_model_trained_at is not None
        and snapshot_generated_at is not None
        and promoted_model_trained_at > snapshot_generated_at
    ):
        reasons.append(
            "model trained after snapshot "
            f"(trained_at={promoted_model_trained_at.isoformat()} > snapshot_at={snapshot_generated_at.isoformat()})"
        )

    if parsed_ref_date is None:
        reasons.append("snapshot reference_price_date missing")
    elif latest_price_date is not None and parsed_ref_date != latest_price_date:
        reasons.append(
            "snapshot is not anchored to latest PriceBar "
            f"(snapshot={parsed_ref_date.isoformat()}, "
            f"latest_price_bar={latest_price_date.isoformat()})"
        )
    elif latest_price_date is None:
        staleness_days = (today_utc - parsed_ref_date).days
        if staleness_days >= max_unanchored_staleness_days:
            reasons.append(
                f"reference_price_date stale ({reference_price_date_raw}, {staleness_days}d)"
            )
    return reasons


# =============================================================================
# 5. OpenRouter Transport, V2 Escalation & Commentary Fallback State Machines
# =============================================================================


class TransportAction(StrEnum):
    RETRY_SAME_KEY = "retry_same_key"
    ROTATE_KEY = "rotate_key"
    ADVANCE_MODEL = "advance_model"
    FAIL_CHAIN = "fail_chain"


def decide_openrouter_transport_action(
    *,
    category: str,
    rotate_key: bool,
    no_upstreams: bool,
    skip_same_model_timeout_retry: bool,
    has_more_attempts: bool,
    has_more_keys: bool,
) -> TransportAction:
    """
    Pure transition reducer for the OpenRouter multi-model, multi-key retry loop.
    """
    if category == "bad_request":
        return TransportAction.ADVANCE_MODEL
    if no_upstreams:
        return TransportAction.ADVANCE_MODEL
    if rotate_key:
        return (
            TransportAction.ROTATE_KEY
            if has_more_keys
            else TransportAction.ADVANCE_MODEL
        )
    if skip_same_model_timeout_retry:
        return TransportAction.ADVANCE_MODEL
    if has_more_attempts:
        return TransportAction.RETRY_SAME_KEY
    if has_more_keys and category == "auth":
        return TransportAction.ROTATE_KEY
    return TransportAction.ADVANCE_MODEL


class V2DecodeStage(StrEnum):
    PRIMARY = "primary"
    LENGTH_RETRY = "length_retry"
    PARSE_RETRY = "parse_retry"
    JSON_REPAIR = "json_repair"
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


class V2ArticleRoute(StrEnum):
    FAST_ACCEPTED = "fast_accepted"
    ESCALATE_MISSING = "escalate_missing"
    ESCALATE_CONFLICT = "escalate_conflict"
    RELIABLE_ACCEPTED = "reliable_accepted"
    FINBERT_FALLBACK = "finbert_fallback"


class DailyModelQuotaTracker(dict):
    """
    Explicit per-model UTC daily quota state machine (AVAILABLE vs EXHAUSTED_TODAY).
    Inherits from `dict` (`{model_name: utc_date_iso}`) so legacy callers/tests that
    inspect `_rate_limited_by_model` as a mapping continue to work unchanged.
    """

    def is_exhausted_today(self, model_name: str, today_utc: str) -> bool:
        return self.get(model_name) == today_utc

    def are_both_exhausted(
        self, fast_model: str, reliable_model: str, today_utc: str
    ) -> bool:
        return self.is_exhausted_today(
            fast_model, today_utc
        ) and self.is_exhausted_today(reliable_model, today_utc)

    def mark_exhausted(self, model_name: str, today_utc: str) -> None:
        if model_name:
            self[model_name] = today_utc


class CommentaryGenerationMode(StrEnum):
    IFM_K2 = "ifm_k2"
    LLM = "llm"
    LLM_REPAIRED = "llm_repaired"
    DETERMINISTIC_FALLBACK = "deterministic_fallback"
    UNAVAILABLE = "unavailable"


class SentimentSummarySource(StrEnum):
    DAILY_V2 = "daily_v2"
    ROLLING_V2 = "rolling_v2"
    LEGACY_V1 = "legacy_v1"
    NONE = "none"


# =============================================================================
# 6. Live Price Provider Circuit Breaker & WebSocket Lifecycle State Machines
# =============================================================================


class CrumbCircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"


@dataclass
class YFinanceCrumbCircuitBreaker:
    """
    Explicit circuit breaker guarding `ticker.info` when Yahoo returns 401 / Invalid Crumb.
    """

    blocked_until: float = 0.0
    cooldown_seconds: float = 900.0

    def state(self, now_monotonic: float) -> CrumbCircuitState:
        return (
            CrumbCircuitState.OPEN
            if now_monotonic < self.blocked_until
            else CrumbCircuitState.CLOSED
        )

    def is_open(self, now_monotonic: float) -> bool:
        return self.state(now_monotonic) == CrumbCircuitState.OPEN

    def trip(self, now_monotonic: float) -> float:
        self.blocked_until = now_monotonic + self.cooldown_seconds
        return self.blocked_until

    def reset(self) -> None:
        self.blocked_until = 0.0


class LivePriceWsPhase(StrEnum):
    CONNECTED = "connected"
    INITIAL_SNAPSHOT = "initial_snapshot"
    SUBSCRIBED = "subscribed"
    STREAMING = "streaming"
    RECONNECTING = "reconnecting"
    DISCONNECTED = "disconnected"


@dataclass
class LivePriceWsStateMachine:
    """
    Explicit state machine for `/ws/live-price` connection lifecycle.
    """

    phase: LivePriceWsPhase = LivePriceWsPhase.CONNECTED
    reconnect_attempts: int = 0

    def transition(self, next_phase: LivePriceWsPhase) -> LivePriceWsPhase:
        if next_phase == LivePriceWsPhase.RECONNECTING:
            self.reconnect_attempts += 1
        elif next_phase in (
            LivePriceWsPhase.SUBSCRIBED,
            LivePriceWsPhase.STREAMING,
        ):
            self.reconnect_attempts = 0
        self.phase = next_phase
        return self.phase

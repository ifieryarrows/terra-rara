"""
Unit tests for explicit backend finite state machines in `app.state_machines`.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

from app.state_machines import (
    CommentaryGenerationMode,
    CrumbCircuitState,
    DailyModelQuotaTracker,
    HeatmapCachePhase,
    HeatmapExternalState,
    HeatmapMetadataCircuitBreaker,
    HeatmapMetadataCircuitState,
    LivePriceWsPhase,
    LivePriceWsStateMachine,
    ModelServingState,
    PipelineQualityState,
    PipelineRunEvent,
    PipelineRunStatus,
    SentimentSummarySource,
    SnapshotQualityState,
    TFTCalibrationStatus,
    TFTPredictorLifecycleMachine,
    TFTPredictorLifecycleState,
    TransportAction,
    XGBArtifactStatus,
    YFinanceCrumbCircuitBreaker,
    decide_openrouter_transport_action,
    evaluate_heatmap_cache_state,
    evaluate_tft_snapshot_freshness,
    evaluate_xgb_snapshot_state,
    map_pipeline_status_to_external,
    transition_pipeline_run,
)


def test_pipeline_run_lifecycle_transitions():
    now = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)
    metrics = SimpleNamespace(
        status=None,
        quality_state=None,
        enqueued_at=None,
        run_started_at=None,
        worker_started_at=None,
        run_completed_at=None,
        duration_seconds=None,
        error_message=None,
        job_id=None,
    )

    status, quality = transition_pipeline_run(
        metrics, PipelineRunEvent.ENQUEUE, now=now, job_id="job-1"
    )
    assert (status, quality) == (PipelineRunStatus.QUEUED, PipelineQualityState.QUEUED)
    assert metrics.status == "queued"
    assert metrics.quality_state == "queued"
    assert metrics.job_id == "job-1"

    worker_ts = now + timedelta(seconds=5)
    status, quality = transition_pipeline_run(
        metrics, PipelineRunEvent.WORKER_START, now=worker_ts
    )
    assert (status, quality) == (PipelineRunStatus.RUNNING, PipelineQualityState.QUEUED)
    assert metrics.worker_started_at == worker_ts

    done_ts = worker_ts + timedelta(seconds=30)
    status, quality = transition_pipeline_run(
        metrics,
        PipelineRunEvent.COMPLETE_SUCCESS,
        now=done_ts,
        quality_state_override="stale",
    )
    assert (status, quality) == (PipelineRunStatus.SUCCESS, PipelineQualityState.STALE)
    assert metrics.run_completed_at == done_ts
    assert metrics.duration_seconds == 30.0
    assert map_pipeline_status_to_external(metrics.status) == "ok"


def test_pipeline_orphan_and_lost_lock_transitions():
    now = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)
    orphan = SimpleNamespace(
        status="queued",
        quality_state="queued",
        run_started_at=now - timedelta(hours=1),
        run_completed_at=None,
        duration_seconds=None,
        error_message=None,
    )
    transition_pipeline_run(
        orphan,
        PipelineRunEvent.EXPIRE_ORPHAN,
        now=now,
        error_message="worker_interrupted",
    )
    assert orphan.status == "failed"
    assert orphan.quality_state == "failed"
    assert orphan.error_message == "worker_interrupted"

    running = SimpleNamespace(
        status="running",
        quality_state="queued",
        run_started_at=now - timedelta(minutes=5),
        run_completed_at=None,
        duration_seconds=None,
        error_message=None,
    )
    transition_pipeline_run(
        running,
        PipelineRunEvent.LOST_LOCK,
        now=now,
        error_message="worker_lost_lock_before_finalizing",
    )
    assert running.status == "failed"
    assert running.error_message == "worker_lost_lock_before_finalizing"


def test_heatmap_swr_cache_state_machine():
    now = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)

    # 1. Empty cache triggers refresh and reports "empty"
    empty_dec = evaluate_heatmap_cache_state(
        payload_count=0,
        expires_at=now,
        refresh_started_at=None,
        refresh_error=None,
        now=now,
    )
    assert empty_dec.external_state == HeatmapExternalState.EMPTY
    assert empty_dec.phase == HeatmapCachePhase.REFRESHING_EMPTY
    assert empty_dec.should_trigger_refresh is True
    assert empty_dec.refresh_in_progress is True

    # 2. Populated stale cache triggers background refresh and reports "refreshing"
    stale_dec = evaluate_heatmap_cache_state(
        payload_count=80,
        expires_at=now - timedelta(minutes=1),
        refresh_started_at=None,
        refresh_error=None,
        now=now,
    )
    assert stale_dec.external_state == HeatmapExternalState.REFRESHING
    assert stale_dec.phase == HeatmapCachePhase.STALE_REFRESHING
    assert stale_dec.should_trigger_refresh is True

    # 3. Stuck refresh marker (>180s) expires and re-triggers refresh
    stuck_dec = evaluate_heatmap_cache_state(
        payload_count=80,
        expires_at=now - timedelta(minutes=5),
        refresh_started_at=now - timedelta(seconds=200),
        refresh_error=None,
        now=now,
    )
    assert stuck_dec.stuck_lock_expired is True
    assert stuck_dec.should_trigger_refresh is True

    # 4. Retry backoff suppresses immediate re-trigger when refresh_error is active before expires_at
    backoff_dec = evaluate_heatmap_cache_state(
        payload_count=80,
        expires_at=now + timedelta(minutes=5),
        refresh_started_at=None,
        refresh_error="provider 429",
        now=now,
    )
    assert backoff_dec.retry_backoff is True
    assert backoff_dec.should_trigger_refresh is False
    assert backoff_dec.external_state == HeatmapExternalState.STALE
    assert backoff_dec.phase == HeatmapCachePhase.RETRY_BACKOFF


def test_heatmap_metadata_circuit_breaker():
    cb = HeatmapMetadataCircuitBreaker()
    assert cb.state == HeatmapMetadataCircuitState.CLOSED
    assert not cb.is_open

    cb.record_provider_error("timeout")
    assert cb.state == HeatmapMetadataCircuitState.OPEN_PROVIDER_ERROR
    assert not cb.is_open

    cb.trip_rate_limit("429 Too Many Requests")
    assert cb.state == HeatmapMetadataCircuitState.OPEN_RATE_LIMITED
    assert cb.is_open
    assert cb.select_backoff(
        rate_limit_backoff=timedelta(minutes=30),
        provider_backoff=timedelta(minutes=5),
    ) == timedelta(minutes=30)


def test_tft_predictor_lifecycle_machine():
    m = TFTPredictorLifecycleMachine()
    assert m.state == TFTPredictorLifecycleState.UNINITIALIZED
    m.mark_artifacts_ready()
    assert m.state == TFTPredictorLifecycleState.ARTIFACTS_READY
    m.mark_model_loaded()
    assert m.state == TFTPredictorLifecycleState.MODEL_LOADED
    m.mark_degraded("missing_checkpoint")
    assert m.state == TFTPredictorLifecycleState.DEGRADED
    assert m.artifact_issue == "missing_checkpoint"


def test_snapshot_freshness_evaluators():
    now = datetime(2026, 11, 7, 12, 0, tzinfo=timezone.utc)
    fresh = evaluate_xgb_snapshot_state(
        generated_at=now - timedelta(hours=4),
        now=now,
    )
    assert fresh.quality_state == SnapshotQualityState.OK.value
    assert fresh.model_state == ModelServingState.OK.value

    stale = evaluate_xgb_snapshot_state(
        generated_at=now - timedelta(hours=40),
        now=now,
    )
    assert stale.is_stale is True
    assert stale.quality_state == SnapshotQualityState.STALE.value
    assert stale.model_state == ModelServingState.DEGRADED.value

    reasons = evaluate_tft_snapshot_freshness(
        snapshot_generated_at=now - timedelta(hours=2),
        promoted_model_trained_at=now - timedelta(hours=1),
        reference_price_date_raw="2026-11-05",
        parsed_ref_date=date(2026, 11, 5),
        latest_price_date=date(2026, 11, 6),
        today_utc=date(2026, 11, 7),
    )
    assert len(reasons) == 2
    assert "model trained after snapshot" in reasons[0]
    assert "not anchored to latest PriceBar" in reasons[1]


def test_openrouter_transport_and_quota_machines():
    assert (
        decide_openrouter_transport_action(
            category="rate_limit",
            rotate_key=True,
            no_upstreams=False,
            skip_same_model_timeout_retry=False,
            has_more_attempts=False,
            has_more_keys=True,
        )
        == TransportAction.ROTATE_KEY
    )
    assert (
        decide_openrouter_transport_action(
            category="timeout",
            rotate_key=False,
            no_upstreams=False,
            skip_same_model_timeout_retry=True,
            has_more_attempts=True,
            has_more_keys=False,
        )
        == TransportAction.ADVANCE_MODEL
    )

    tracker = DailyModelQuotaTracker()
    tracker.mark_exhausted("fast-m", "2026-11-07")
    assert tracker.is_exhausted_today("fast-m", "2026-11-07")
    assert not tracker.are_both_exhausted("fast-m", "rel-m", "2026-11-07")
    tracker.mark_exhausted("rel-m", "2026-11-07")
    assert tracker.are_both_exhausted("fast-m", "rel-m", "2026-11-07")


def test_crumb_circuit_and_ws_machine():
    cb = YFinanceCrumbCircuitBreaker(cooldown_seconds=900.0)
    assert cb.state(100.0) == CrumbCircuitState.CLOSED
    until = cb.trip(100.0)
    assert until == 1000.0
    assert cb.is_open(500.0) is True
    assert cb.is_open(1001.0) is False

    ws = LivePriceWsStateMachine()
    assert ws.phase == LivePriceWsPhase.CONNECTED
    ws.transition(LivePriceWsPhase.RECONNECTING)
    assert ws.reconnect_attempts == 1
    ws.transition(LivePriceWsPhase.SUBSCRIBED)
    assert ws.reconnect_attempts == 0
    assert XGBArtifactStatus.ACTIVE.value == "active"
    assert TFTCalibrationStatus.FIT.value == "fit"
    assert CommentaryGenerationMode.IFM_K2.value == "ifm_k2"
    assert SentimentSummarySource.DAILY_V2.value == "daily_v2"

"""Tests for the Cuprum companion contract and `GET /api/commentary` enrichment.

Contract under test (ORDER #2 — GLM 5.3 Flash):
- `app/companion.py`: deterministic state engine (`build_companion_state`,
  `extract_commentary_sections`) with NO external I/O or LLM calls.
- `app/main.py::GET /api/commentary`: backward-compatible payload + new
  `sections` and `companion_state` keys, with `xai_highlights` gated on TFT
  snapshot freshness (`_tft_snapshot_invalid_reasons`) and never triggering
  live TFT inference.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app import main
from app.companion import build_companion_state, extract_commentary_sections
from app.models import AICommentary, PriceBar, TFTModelMetadata, TFTPredictionSnapshot


# =============================================================================
# Deterministic fixtures
# =============================================================================


LLM_COMMENTARY = """RISKS (3 factors):
• Copper inventories near multi-year lows leave the market vulnerable to a supply shock.
• Sentiment regime is positive with score 0.412, which can reverse quickly.
• Model forecast carries a wide daily band; positioning data is one day delayed.

OPPORTUNITIES (3 factors):
• Persistent backwardation signals sustained physical demand into the quarter.
• Chinese grid investment tailwinds support a medium-term demand floor.
• Momentum tilt is BULLISH with rising volume confirmation.

SYNTHESIS:
Copper looks constructive near 5.10. Momentum is aligned with a bullish 5-day outlook. The weight of evidence points higher.

BIAS ACKNOWLEDGMENT:
This analysis inherits model and data limitations and could be invalidated by a broad risk-off move.

DISCLAIMER:
This is NOT financial advice.
"""

FALLBACK_COMMENTARY = """Risks:
1. Model indicates upside uncertainty around the next-day move (0.85%).
2. Sentiment regime is Positive with score 0.300, which can reverse quickly.
3. News sample size (12) may be insufficient for stable short-horizon inference.
Opportunities:
1. The model-derived price is $5.1400 from the finite baseline $5.1000; the live display price is unavailable.
2. Feature signal concentration around `HG=F_close_score` can support tactical monitoring.
3. Secondary drivers `HG=F_ma7` and `HG=F_rsi14` provide confirmation checkpoints.
Summary: Current model inputs suggest a cautious upside bias with elevated uncertainty.
Bias warning: This view is model-driven and sensitive to news mix, data latency, and feature drift.
This is NOT financial advice.
"""


def _explain_payload(reference_date: str = "2026-09-01") -> dict:
    """Mirror the explain_json schema persisted by the TFT pipeline."""
    return {
        "schema_version": 1,
        "symbol": "HG=F",
        "reference_price_date": reference_date,
        "generated_at": f"{reference_date}T20:05:00+00:00",
        "run_id": f"run-{reference_date}",
        "model": {"checkpoint": "best.ckpt", "trained_at": None},
        "horizon": {"label": "5-day cumulative log return (primary)", "days": 5},
        "variable_selection": {
            "method": "vsn",
            "encoder": [
                {"feature": "HG=F_close", "label": "Close price", "category": "price", "weight": 0.40, "recent": {"kind": "pct", "change_pct": 2.1}},
                {"feature": "HG=F_sentiment_mean", "label": "News sentiment", "category": "sentiment", "weight": 0.22},
                {"feature": "cot_index", "label": "COT positioning", "category": "positioning", "weight": 0.15},
                {"feature": "dow_sin", "weight": 0.10},
                {"feature": "usd_index", "label": "USD index", "category": "macro", "weight": 0.08},
                {"feature": "excluded_feature", "weight": 0.05},
            ],
            "decoder": [{"feature": "dow_sin", "weight": 1.0}],
            "static": [],
        },
        "attention": {
            "method": "tft_attention",
            "step": {"kind": "first_forecast_day", "horizon_index": 0},
            "window": {"start": "2026-08-12", "end": reference_date, "days": 16},
            "top_days": [
                {"date": reference_date, "weight": 0.5},
                {"date": "2026-08-29", "weight": 0.25},
                {"date": "2026-08-25", "weight": 0.125},
                {"date": "2026-08-20", "weight": 0.125},
            ],
            "series": [],
        },
        "notes": "Feature weights reflect model focus, not causal effects.",
    }


def _tft_payload(direction: str = "BULLISH", risk_level: str = "MEDIUM") -> dict:
    """Mirror the top-level keys of the TFT snapshot payload."""
    return {
        "symbol": "HG=F",
        "model_type": "TFT-ASRO",
        "is_forecast_healthy": True,
        "direction": direction,
        "risk_level": risk_level,
        "primary_horizon": "5D",
        "prediction": {"reference_price_date": "2026-09-01", "weekly_return": 0.01},
    }


def _commentary_record(**overrides) -> dict:
    record = {
        "commentary": LLM_COMMENTARY,
        "generated_at": "2026-09-01T20:10:00+00:00",
        "ai_stance": "BULLISH",
        "generation_mode": "llm",
        "model_name": "test/model",
        "fallback_reason": None,
    }
    record.update(overrides)
    return record


# =============================================================================
# Task A: build_companion_state
# =============================================================================


class TestBuildCompanionStateMood:
    def test_bullish_stance_maps_to_molten_bull(self):
        state = build_companion_state(symbol="HG=F", commentary_record=_commentary_record(ai_stance="BULLISH"))
        assert state["character"] == "Cuprum"
        assert state["stance"] == "BULLISH"
        assert state["mood"] == "molten_bull"

    def test_bearish_stance_maps_to_patina_bear(self):
        state = build_companion_state(symbol="HG=F", commentary_record=_commentary_record(ai_stance="BEARISH"))
        assert state["mood"] == "patina_bear"
        assert state["stance"] == "BEARISH"

    def test_neutral_stance_maps_to_tempered_neutral(self):
        state = build_companion_state(symbol="HG=F", commentary_record=_commentary_record(ai_stance="NEUTRAL"))
        assert state["mood"] == "tempered_neutral"
        assert state["stance"] == "NEUTRAL"

    def test_invalid_ai_stance_falls_back_to_tft_direction(self):
        state = build_companion_state(
            symbol="HG=F",
            commentary_record=_commentary_record(ai_stance="SASSY"),
            tft_payload=_tft_payload(direction="BULLISH"),
        )
        assert state["stance"] == "BULLISH"
        assert state["mood"] == "molten_bull"

    def test_no_record_prefers_tft_payload_direction(self):
        state = build_companion_state(symbol="HG=F", commentary_record=None, tft_payload=_tft_payload(direction="BEARISH"))
        assert state["stance"] == "BEARISH"
        assert state["mood"] == "patina_bear"

    def test_no_information_at_all_defaults_to_neutral(self):
        state = build_companion_state(symbol="HG=F", commentary_record=None)
        assert state["stance"] == "NEUTRAL"
        assert state["mood"] == "tempered_neutral"


class TestBuildCompanionStateEnergy:
    @pytest.mark.parametrize("risk_level,expected", [
        ("HIGH", "volatile"),
        ("MEDIUM", "charged"),
        ("LOW", "calm"),
        ("EXTREME", "calm"),
        (None, "calm"),
    ])
    def test_risk_level_maps_to_energy(self, risk_level, expected):
        tft_payload = _tft_payload(risk_level=risk_level) if risk_level is not None else _tft_payload()
        if risk_level is None:
            tft_payload.pop("risk_level")
        state = build_companion_state(symbol="HG=F", commentary_record=None, tft_payload=tft_payload)
        assert state["energy"] == expected

    def test_missing_tft_payload_defaults_to_calm(self):
        state = build_companion_state(symbol="HG=F", commentary_record=None)
        assert state["energy"] == "calm"


class TestBuildCompanionStateConfidenceBand:
    def test_llm_mode_with_explain_is_high(self):
        state = build_companion_state(
            symbol="HG=F",
            commentary_record=_commentary_record(generation_mode="llm"),
            explain_payload=_explain_payload(),
        )
        assert state["confidence_band"] == "high"

    def test_llm_repaired_mode_with_explain_is_high(self):
        state = build_companion_state(
            symbol="HG=F",
            commentary_record=_commentary_record(generation_mode="llm_repaired"),
            explain_payload=_explain_payload(),
        )
        assert state["confidence_band"] == "high"

    def test_llm_mode_without_explain_is_medium(self):
        state = build_companion_state(
            symbol="HG=F",
            commentary_record=_commentary_record(generation_mode="llm"),
        )
        assert state["confidence_band"] == "medium"

    def test_fallback_mode_with_explain_is_medium(self):
        state = build_companion_state(
            symbol="HG=F",
            commentary_record=_commentary_record(generation_mode="deterministic_fallback"),
            explain_payload=_explain_payload(),
        )
        assert state["confidence_band"] == "medium"

    def test_explain_without_commentary_is_medium(self):
        state = build_companion_state(symbol="HG=F", commentary_record=None, explain_payload=_explain_payload())
        assert state["confidence_band"] == "medium"

    def test_nothing_at_all_is_low(self):
        state = build_companion_state(symbol="HG=F", commentary_record=None)
        assert state["confidence_band"] == "low"


class TestBuildCompanionStateXaiHighlights:
    def test_full_extraction_from_explain_payload(self):
        explain = _explain_payload()
        state = build_companion_state(symbol="HG=F", commentary_record=None, explain_payload=explain)

        highlights = state["xai_highlights"]
        assert highlights is not None
        assert highlights["horizon_label"] == "5-day cumulative log return (primary)"
        assert highlights["reference_price_date"] == "2026-09-01"
        assert highlights["window"] == {"start": "2026-08-12", "end": "2026-09-01", "days": 16}
        assert highlights["notes"] == "Feature weights reflect model focus, not causal effects."

        top_signals = highlights["top_signals"]
        assert len(top_signals) == 5  # 6 available, top 5 kept
        assert top_signals[0] == {
            "feature": "HG=F_close",
            "label": "Close price",
            "category": "price",
            "weight": 0.40,
            "recent": {"kind": "pct", "change_pct": 2.1},
        }
        assert top_signals[3] == {
            "feature": "dow_sin",
            "label": None,
            "category": None,
            "weight": 0.10,
            "recent": None,
        }
        assert top_signals[4]["feature"] == "usd_index"
        assert "excluded_feature" not in [s["feature"] for s in top_signals]

        anchors = highlights["memory_anchors"]
        assert len(anchors) == 3  # 4 available, top 3 kept
        assert anchors[0] == {"date": "2026-09-01", "weight": 0.5}
        assert anchors[2] == {"date": "2026-08-25", "weight": 0.125}

    def test_missing_explain_payload_drops_highlights(self):
        state = build_companion_state(symbol="HG=F", commentary_record=_commentary_record())
        assert state["xai_highlights"] is None

    def test_non_dict_explain_payload_drops_highlights(self):
        state = build_companion_state(symbol="HG=F", commentary_record=None, explain_payload="not-a-dict")
        assert state["xai_highlights"] is None


class TestBuildCompanionStateHeadline:
    def test_headline_quote_is_first_two_sentences_of_synthesis(self):
        state = build_companion_state(
            symbol="HG=F",
            commentary_record=_commentary_record(),
        )
        assert state["headline_quote"] == "Copper looks constructive near 5.10. Momentum is aligned with a bullish 5-day outlook."

    def test_headline_quote_defaults_when_commentary_unavailable(self):
        state = build_companion_state(symbol="HG=F", commentary_record=None)
        assert state["headline_quote"] == "Cuprum is awaiting the next pipeline refresh."


# =============================================================================
# Task A: extract_commentary_sections
# =============================================================================


class TestExtractCommentarySections:
    def test_parses_llm_commentary(self):
        sections = extract_commentary_sections(LLM_COMMENTARY)
        assert sections["synthesis"] == (
            "Copper looks constructive near 5.10. "
            "Momentum is aligned with a bullish 5-day outlook. "
            "The weight of evidence points higher."
        )
        assert sections["risks"] == [
            "Copper inventories near multi-year lows leave the market vulnerable to a supply shock.",
            "Sentiment regime is positive with score 0.412, which can reverse quickly.",
            "Model forecast carries a wide daily band; positioning data is one day delayed.",
        ]
        assert sections["opportunities"] == [
            "Persistent backwardation signals sustained physical demand into the quarter.",
            "Chinese grid investment tailwinds support a medium-term demand floor.",
            "Momentum tilt is BULLISH with rising volume confirmation.",
        ]
        assert sections["bias_warning"] == (
            "This analysis inherits model and data limitations and could be "
            "invalidated by a broad risk-off move."
        )
        assert sections["xai_brief"] is None

    def test_parses_fallback_commentary(self):
        sections = extract_commentary_sections(FALLBACK_COMMENTARY)
        assert sections["synthesis"] == (
            "Current model inputs suggest a cautious upside bias with elevated uncertainty."
        )
        assert sections["risks"] == [
            "Model indicates upside uncertainty around the next-day move (0.85%).",
            "Sentiment regime is Positive with score 0.300, which can reverse quickly.",
            "News sample size (12) may be insufficient for stable short-horizon inference.",
        ]
        assert sections["opportunities"][1] == (
            "Feature signal concentration around HG=F_close_score can support tactical monitoring."
        )
        assert len(sections["opportunities"]) == 3
        assert sections["bias_warning"] == (
            "This view is model-driven and sensitive to news mix, data latency, and feature drift."
        )

    def test_none_commentary_returns_empty_sections(self):
        sections = extract_commentary_sections(None)
        assert sections["synthesis"] == ""
        assert sections["risks"] == []
        assert sections["opportunities"] == []
        assert sections["xai_brief"] is None
        assert sections["bias_warning"] is None

    def test_xai_brief_mentions_top_signals_and_anchors(self):
        sections = extract_commentary_sections(LLM_COMMENTARY, _explain_payload())
        brief = sections["xai_brief"]
        assert isinstance(brief, str)
        assert "Close price" in brief
        assert "2026-09-01" in brief
        # Non-causal framing is mandatory in the contract.
        assert "non-causal" in brief.lower() or "model focus" in brief.lower()

    def test_limits_briefs_to_available_data(self):
        sections = extract_commentary_sections(None, {"variable_selection": {}, "attention": {}})
        assert sections["xai_brief"] is None

    def test_delegates_to_parse_commentary_sections_when_available(self, monkeypatch):
        """The Order #1 seam: delegate when app.commentary provides the parser."""
        import app.commentary as commentary_module

        sentinel = {
            "synthesis": "DELEGATED",
            "risks": ["r"],
            "opportunities": [],
            "xai_brief": "x",
            "bias_warning": None,
        }
        # The attribute does not exist on main today (Order #1 adds it in a
        # parallel branch) — the stub pretends it already landed.
        monkeypatch.setattr(
            commentary_module,
            "parse_commentary_sections",
            lambda commentary, tft_explain=None: sentinel,
            raising=False,
        )
        assert extract_commentary_sections("anything") == sentinel


# =============================================================================
# Task B: GET /api/commentary enrichment
# =============================================================================


@pytest.fixture
def commentary_session_factory(monkeypatch):
    """In-memory SQLite session factory with the tables the endpoint touches."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    for table in (
        AICommentary.__table__,
        PriceBar.__table__,
        TFTModelMetadata.__table__,
        TFTPredictionSnapshot.__table__,
    ):
        table.create(bind=engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr(main, "SessionLocal", Session)
    try:
        yield Session
    finally:
        engine.dispose()


def _forbid_live_inference(monkeypatch) -> None:
    """Make any live-inference attempt fail the test loudly."""
    import deep_learning.inference.predictor as predictor_module

    monkeypatch.setattr(
        predictor_module,
        "generate_tft_analysis",
        lambda *_args, **_kwargs: pytest.fail(
            "live TFT inference must never run for /api/commentary"
        ),
    )


def _seed_commentary(Session, *, stance: str = "BULLISH", mode: str = "llm") -> None:
    with Session() as session:
        session.add(
            AICommentary(
                symbol="HG=F",
                commentary=LLM_COMMENTARY,
                current_price=5.10,
                predicted_price=5.14,
                predicted_return=0.0085,
                sentiment_label="Positive",
                ai_stance=stance,
                model_name="test/model",
                generation_mode=mode,
            )
        )
        session.commit()


def _seed_snapshot_and_price(
    Session,
    *,
    snapshot_date: str = "2026-09-01",
    price_date: str = "2026-09-01",
    explain: dict | None = None,
    direction: str = "BULLISH",
    risk_level: str = "MEDIUM",
) -> None:
    payload = _tft_payload(direction=direction, risk_level=risk_level)
    with Session() as session:
        session.add(
            PriceBar(
                symbol="HG=F",
                date=datetime.fromisoformat(f"{price_date}T04:00:00+00:00"),
                close=6.0,
            )
        )
        session.add(
            TFTPredictionSnapshot(
                symbol="HG=F",
                payload_json=payload,
                generated_at=datetime(2026, 9, 1, 20, 5, tzinfo=timezone.utc),
                reference_price_date=snapshot_date,
                explain_json=explain,
            )
        )
        session.commit()


def _get_commentary(monkeypatch) -> dict:
    response = TestClient(main.app).get("/api/commentary")
    assert response.status_code == 200, response.text
    return response.json()


class TestCommentaryEndpointContract:
    def test_returns_sections_and_companion_state_with_fresh_snapshot(
        self, monkeypatch, commentary_session_factory
    ):
        Session = commentary_session_factory
        _seed_commentary(Session)
        _seed_snapshot_and_price(Session, explain=_explain_payload())
        _forbid_live_inference(monkeypatch)

        body = _get_commentary(monkeypatch)

        # --- backward-compatible top-level keys remain intact ----------------
        assert body["symbol"] == "HG=F"
        assert body["commentary"] == LLM_COMMENTARY
        assert body["error"] is None
        assert body["generated_at"] is not None
        assert body["ai_stance"] == "BULLISH"
        assert body["generation_mode"] == "llm"
        assert body["model_name"] == "test/model"
        assert body["fallback_reason"] is None

        # --- new keys --------------------------------------------------------
        assert body["sections"]["synthesis"].startswith("Copper looks constructive")
        assert len(body["sections"]["risks"]) == 3

        companion = body["companion_state"]
        assert companion["character"] == "Cuprum"
        assert companion["mood"] == "molten_bull"
        assert companion["energy"] == "charged"
        assert companion["confidence_band"] == "high"
        assert companion["stance"] == "BULLISH"

        highlights = companion["xai_highlights"]
        assert highlights is not None
        assert highlights["horizon_label"] == "5-day cumulative log return (primary)"
        assert highlights["top_signals"][0]["feature"] == "HG=F_close"
        assert highlights["memory_anchors"][0]["date"] == "2026-09-01"

        assert body["sections"]["xai_brief"] is not None

    def test_new_price_bar_drops_xai_highlights_without_live_inference(
        self, monkeypatch, commentary_session_factory
    ):
        Session = commentary_session_factory
        _seed_commentary(Session)
        _seed_snapshot_and_price(Session, snapshot_date="2026-09-01", price_date="2026-09-02", explain=_explain_payload())
        _forbid_live_inference(monkeypatch)

        body = _get_commentary(monkeypatch)

        assert body["companion_state"]["xai_highlights"] is None
        assert body["sections"]["xai_brief"] is None
        # Companion core still renders from the stored commentary.
        assert body["companion_state"]["mood"] == "molten_bull"
        assert body["sections"]["synthesis"].startswith("Copper looks constructive")

    def test_newer_gate_passed_model_drops_xai_highlights(
        self, monkeypatch, commentary_session_factory
    ):
        Session = commentary_session_factory
        _seed_commentary(Session)
        _seed_snapshot_and_price(Session, explain=_explain_payload())
        with Session() as session:
            session.add(
                TFTModelMetadata(
                    symbol="HG=F",
                    trained_at=datetime(2026, 9, 2, 6, 0, tzinfo=timezone.utc),
                    quality_gate_passed=True,
                    checkpoint_path="newer.ckpt",
                )
            )
            session.commit()
        _forbid_live_inference(monkeypatch)

        body = _get_commentary(monkeypatch)

        assert body["companion_state"]["xai_highlights"] is None

    def test_unavailable_commentary_still_returns_companion_state(
        self, monkeypatch, commentary_session_factory
    ):
        """No AICommentary row, but a fresh snapshot: the unavailable branch of
        the endpoint must still carry `sections` and `companion_state`."""
        Session = commentary_session_factory
        _seed_snapshot_and_price(Session, explain=_explain_payload())
        _forbid_live_inference(monkeypatch)

        body = _get_commentary(monkeypatch)

        assert body["commentary"] is None
        assert body["error"] is not None
        assert body["ai_stance"] == "NEUTRAL"
        assert body["generation_mode"] == "unavailable"
        assert body["fallback_reason"] == "not_generated"

        assert body["sections"]["synthesis"] == ""
        assert body["sections"]["risks"] == []

        companion = body["companion_state"]
        assert companion["character"] == "Cuprum"
        # TFT payload direction/risk_level drive stance/energy even without commentary.
        assert companion["stance"] == "BULLISH"
        assert companion["mood"] == "molten_bull"
        assert companion["energy"] == "charged"
        assert companion["headline_quote"] == "Cuprum is awaiting the next pipeline refresh."
        assert companion["confidence_band"] == "medium"
        assert companion["xai_highlights"] is not None
        assert companion["xai_highlights"]["top_signals"][0]["feature"] == "HG=F_close"

    def test_no_snapshot_drops_xai_highlights(self, monkeypatch, commentary_session_factory):
        Session = commentary_session_factory
        _seed_commentary(Session)
        _forbid_live_inference(monkeypatch)

        body = _get_commentary(monkeypatch)

        assert body["companion_state"]["xai_highlights"] is None
        assert body["companion_state"]["mood"] == "molten_bull"
        assert body["companion_state"]["confidence_band"] == "medium"
        assert body["sections"]["xai_brief"] is None

    def test_snapshot_explain_not_a_dict_is_ignored(
        self, monkeypatch, commentary_session_factory
    ):
        Session = commentary_session_factory
        _seed_commentary(Session)
        _seed_snapshot_and_price(Session, explain="corrupt")
        _forbid_live_inference(monkeypatch)

        body = _get_commentary(monkeypatch)

        assert body["companion_state"]["xai_highlights"] is None
        assert body["sections"]["xai_brief"] is None

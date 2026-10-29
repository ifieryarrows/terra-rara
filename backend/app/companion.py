"""Cuprum companion state engine ("Cuprum / Külçe Ruh").

Deterministic, I/O-free state derivation for the UI companion character
"Cuprum". Pure functions only: no database, no network, no LLM calls — the
endpoint layer (`app.main`) supplies every input it needs read-only.

Two responsibilities:

1. ``extract_commentary_sections`` — parse any stored commentary text (LLM or
   deterministic fallback) into the structured sections the companion renders.
   When Order #1's ``parse_commentary_sections`` (from ``app.commentary``)
   exists, we delegate to it; while the branches run in parallel that import
   fails and the built-in parser below keeps this module self-sufficient.
2. ``build_companion_state`` — map commentary stance, TFT snapshot direction
   and risk level into the companion's mood / energy / confidence contract,
   plus ``xai_highlights`` extracted from the TFT ``explain_json``.

TFT explain data is pre-calibration model focus — never causal evidence. It
degrades to ``None`` and must never fail the companion contract.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = ["build_companion_state", "extract_commentary_sections"]

_CHARACTER = "Cuprum"

_VALID_STANCES = {"BULLISH", "BEARISH", "NEUTRAL"}
_STANCE_MOODS = {
    "BULLISH": "molten_bull",
    "BEARISH": "patina_bear",
    "NEUTRAL": "tempered_neutral",
}
_RISK_LEVEL_ENERGY = {"HIGH": "volatile", "MEDIUM": "charged"}
_LLM_GENERATION_MODES = ("ifm_k2", "llm", "llm_repaired")

_DEFAULT_HORIZON_LABEL = "5-day cumulative log return (primary)"
_DEFAULT_HEADLINE_QUOTE = "Cuprum is awaiting the next pipeline refresh."

_MAX_BULLETS = 3
_MAX_SIGNALS = 5
_MAX_ANCHORS = 3

# Bullet prefixes accepted in commentary text: LLM output uses "•", the
# deterministic fallback uses numbered "1." lines.
_BULLET_PREFIX_RE = re.compile(r"^(?:[•\-*–]|\d+[.)])\s*")

# Section headers seen in both commentary shapes:
#   LLM:  "RISKS (3 factors):", "OPPORTUNITIES (3 factors):", "SYNTHESIS:",
#         "BIAS ACKNOWLEDGMENT:" (own-line headers, prose on the next lines)
#   Fallback: "Risks:", "Opportunities:", "Summary: <inline>",
#         "Bias warning: <inline>" (inline headers)
_HEADER_RES = (
    ("risks", re.compile(r"^RISKS?\s*(?:\([^)]*\))?\s*:\s*(.*)$", re.IGNORECASE)),
    (
        "opportunities",
        re.compile(r"^OPPORTUNITIE?S?\s*(?:\([^)]*\))?\s*:\s*(.*)$", re.IGNORECASE),
    ),
    ("synthesis", re.compile(r"^SYNTHESIS\s*:\s*(.*)$", re.IGNORECASE)),
    ("summary", re.compile(r"^SUMMARY\s*:\s*(.*)$", re.IGNORECASE)),
    ("bias", re.compile(r"^BIAS\s+ACKNOWLEDGMENT\s*:\s*(.*)$", re.IGNORECASE)),
    ("bias", re.compile(r"^BIAS\s+WARNING\s*:\s*(.*)$", re.IGNORECASE)),
    ("disclaimer", re.compile(r"^DISCLAIMER\s*:?\s*(.*)$", re.IGNORECASE)),
    ("cuprum", re.compile(r"^CUPRUM\s*(?:DIALOGUE)?\s*:\s*(.*)$", re.IGNORECASE)),
)


def _clean_bullet(line: str) -> str:
    """Strip bullet/number markers, markdown backticks, and whitespace."""
    cleaned = _BULLET_PREFIX_RE.sub("", line).replace("`", "").strip()
    return cleaned


def _join_lines(lines: list[str]) -> str:
    text = " ".join(part.strip() for part in lines if part and part.strip())
    return text.strip()


def _ends_sentence(text: str) -> bool:
    """True when a text line ends a sentence (., !, ?, or quotes/follow-ups)."""
    return bool(text) and text.rstrip('"\'”’)').endswith((".", "!", "?"))


def _first_nonempty_paragraph(text: str) -> str:
    """Fallback `synthesis` source: first meaningful paragraph of raw text."""
    for block in re.split(r"\n\s*\n", text or ""):
        para = _join_lines(block.splitlines())
        if not para:
            continue
        # Skip lines that are a section header or a bullet: picking those as
        # the synthesis paragraph would render scaffolding instead of prose.
        if any(pattern.match(para) for _, pattern in _HEADER_RES):
            continue
        if _BULLET_PREFIX_RE.match(para):
            continue
        return para
    return ""


def _parse_commentary_sections_fallback(
    commentary: str, tft_explain: Optional[dict]
) -> dict:
    """Self-sufficient section parser (used until Order #1 lands).

    Handles both commentary shapes currently produced by `app.commentary`:
    the LLM template (own-line headers, "•" bullets) and the deterministic
    fallback template ("1." bullets, inline "Summary:"/"Bias warning:").
    """
    bullets: dict[str, list] = {"risks": [], "opportunities": []}
    prose: dict[str, list] = {"synthesis": [], "summary": [], "bias": []}
    current: Optional[str] = None

    for raw_line in (commentary or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        for section, pattern in _HEADER_RES:
            match = pattern.match(line)
            if not match:
                continue
            inline = _clean_bullet(match.group(1) or "")
            if section in prose:
                current = section
                if inline:
                    prose[section].append(inline)
                    if section == "bias" and _ends_sentence(inline):
                        # The bias acknowledgment is a single sentence by
                        # contract; stop before trailing boilerplate.
                        current = "disclaimer"
            elif section in bullets:
                current = section
                if inline:
                    bullets[section].append(inline)
            else:  # disclaimer / unknown terminal header: stop accumulating
                current = section
            break
        else:
            if _BULLET_PREFIX_RE.match(line):
                if current in bullets and len(bullets[current]) < _MAX_BULLETS:
                    bullets[current].append(_clean_bullet(line))
                continue
            if current in prose:
                prose[current].append(line)
                if current == "bias" and _ends_sentence(line):
                    current = "disclaimer"
            # Non-prose continuation lines (e.g. under bullets or the
            # disclaimer) are scaffolding, not renderable content.

    synthesis = _join_lines(prose["synthesis"])
    if not synthesis:
        synthesis = _join_lines(prose["summary"])
    if not synthesis:
        synthesis = _first_nonempty_paragraph(commentary or "")

    cuprum_daily = None
    if "cuprum" in prose and prose["cuprum"]:
        combined = " ".join(prose["cuprum"])
        start = combined.find("{")
        end = combined.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                import json
                parsed_c = json.loads(combined[start : end + 1])
                if isinstance(parsed_c, dict):
                    cuprum_daily = parsed_c
            except Exception:
                pass

    return {
        "synthesis": synthesis,
        "risks": bullets["risks"][:_MAX_BULLETS],
        "opportunities": bullets["opportunities"][:_MAX_BULLETS],
        "xai_brief": _extract_xai_brief(tft_explain),
        "bias_warning": _join_lines(prose["bias"]) or None,
        "cuprum_daily": cuprum_daily,
    }


def _encoder_entries(explain_payload: dict) -> list[dict]:
    selection = explain_payload.get("variable_selection")
    if not isinstance(selection, dict):
        return []
    encoder = selection.get("encoder")
    if not isinstance(encoder, list):
        return []
    return [entry for entry in encoder if isinstance(entry, dict)]


def _attention_days(explain_payload: dict) -> list[dict]:
    attention = explain_payload.get("attention")
    if not isinstance(attention, dict):
        return []
    top_days = attention.get("top_days")
    if not isinstance(top_days, list):
        return []
    return [entry for entry in top_days if isinstance(entry, dict)]


def _signal_name(entry: dict) -> str:
    for key in ("label", "feature"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown signal"


def _signal_weight_pct(entry: dict) -> str:
    weight = entry.get("weight")
    if isinstance(weight, (int, float)):
        return f"{round(float(weight) * 100, 1)}%"
    return "n/a"


def _anchor_date(entry: dict) -> Optional[str]:
    value = entry.get("date")
    return value if isinstance(value, str) and value.strip() else None


def _extract_xai_brief(explain_payload: Optional[dict]) -> Optional[str]:
    """1-2 sentence non-causal summary of the TFT model focus, or None."""
    if not isinstance(explain_payload, dict):
        return None

    signals = _encoder_entries(explain_payload)[:_MAX_SIGNALS]
    anchors = _attention_days(explain_payload)[:_MAX_ANCHORS]
    if not signals and not anchors:
        return None

    sentences = []
    if signals:
        signal_text = ", ".join(
            f"{_signal_name(entry)} ({_signal_weight_pct(entry)})" for entry in signals
        )
        sentences.append(
            f"Model focus (non-causal): the network leaned on {signal_text} "
            "for the 5-day horizon."
        )
    anchor_dates = [date for date in (_anchor_date(entry) for entry in anchors) if date]
    if anchor_dates:
        sentences.append(
            "Historical attention concentrated on "
            + ", ".join(anchor_dates)
            + "."
        )

    return " ".join(sentences)


def _clean_signal(entry: dict) -> dict:
    return {
        "feature": entry.get("feature"),
        "label": entry.get("label"),
        "category": entry.get("category"),
        "weight": entry.get("weight"),
        "recent": entry.get("recent"),
    }


def _clean_anchor(entry: dict) -> dict:
    return {"date": entry.get("date"), "weight": entry.get("weight")}


def _extract_xai_highlights(explain_payload: Optional[dict]) -> Optional[dict]:
    """Pass-through-safe xai_highlights block; None when explain is absent.

    Mirrors the snapshot-first contract: this only reshapes data the caller
    already vetted for freshness — it must not read, infer, or guess.
    """
    if not isinstance(explain_payload, dict):
        return None

    horizon = explain_payload.get("horizon")
    horizon_label = _DEFAULT_HORIZON_LABEL
    if isinstance(horizon, dict):
        label = horizon.get("label")
        if isinstance(label, str) and label.strip():
            horizon_label = label

    attention = explain_payload.get("attention")
    window = attention.get("window") if isinstance(attention, dict) else None

    return {
        "horizon_label": horizon_label,
        "reference_price_date": explain_payload.get("reference_price_date"),
        "top_signals": [
            _clean_signal(entry) for entry in _encoder_entries(explain_payload)[:_MAX_SIGNALS]
        ],
        "memory_anchors": [
            _clean_anchor(entry) for entry in _attention_days(explain_payload)[:_MAX_ANCHORS]
        ],
        "window": window,
        "notes": explain_payload.get("notes"),
    }


def _first_sentences(text: str, count: int) -> str:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return ""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned)]
    picked = [part for part in sentences if part][:count]
    return " ".join(picked)


def _resolve_stance(
    commentary_record: Optional[dict], tft_payload: Optional[dict]
) -> str:
    """Prefer the commentary stance; fall back to TFT direction; else NEUTRAL."""
    record_stance = (commentary_record or {}).get("ai_stance")
    if isinstance(record_stance, str) and record_stance.strip().upper() in _VALID_STANCES:
        return record_stance.strip().upper()

    direction = (tft_payload or {}).get("direction")
    if isinstance(direction, str) and direction.strip().upper() in _VALID_STANCES:
        return direction.strip().upper()

    return "NEUTRAL"


def _resolve_energy(tft_payload: Optional[dict]) -> str:
    risk_level = (tft_payload or {}).get("risk_level")
    if isinstance(risk_level, str):
        return _RISK_LEVEL_ENERGY.get(risk_level.strip().upper(), "calm")
    return "calm"


def _resolve_confidence_band(
    commentary_record: Optional[dict], explain_payload: Optional[dict]
) -> str:
    mode = (commentary_record or {}).get("generation_mode")
    has_explain = isinstance(explain_payload, dict)
    if mode in _LLM_GENERATION_MODES and has_explain:
        return "high"
    if commentary_record is not None or has_explain:
        return "medium"
    return "low"


def extract_commentary_sections(
    commentary: Optional[str] = None, explain_payload: Optional[dict] = None
) -> dict:
    """Parse stored commentary text into the UI companion's section contract.

    Delegates to Order #1's ``parse_commentary_sections`` when available; the
    built-in parser below keeps this module working standalone so the two
    branches can land independently without either breaking the contract.

    The companion contract is fail-open: a delegate that is absent (normal
    while the branches run in parallel) degrades silently, and a delegate
    that misbehaves at parse time degrades to the built-in parser with a
    warning — it must never raise into ``build_companion_state`` or the
    endpoint.
    """
    try:
        from app.commentary import parse_commentary_sections  # type: ignore
    except ImportError:
        # Expected while Order #1 is unmerged; stay quiet and self-sufficient.
        return _parse_commentary_sections_fallback(commentary, explain_payload)

    try:
        return parse_commentary_sections(
            commentary, tft_explain=explain_payload
        )
    except Exception as exc:
        logger.warning(
            "parse_commentary_sections delegate failed (%s); "
            "degrading to built-in companion parser",
            exc,
        )
        return _parse_commentary_sections_fallback(commentary, explain_payload)


def build_companion_state(
    *,
    symbol: str,
    commentary_record: Optional[dict] = None,
    explain_payload: Optional[dict] = None,
    tft_payload: Optional[dict] = None,
) -> dict:
    """Deterministic companion state for one symbol snapshot.

    Every argument is caller-supplied, already-persisted data (commentary
    record, TFT snapshot payloads). This function performs no I/O and never
    triggers inference; missing inputs degrade each field independently.
    """
    sections = extract_commentary_sections(
        (commentary_record or {}).get("commentary"), explain_payload
    )

    stance = _resolve_stance(commentary_record, tft_payload)
    mood = _STANCE_MOODS[stance]
    energy = _resolve_energy(tft_payload)
    confidence_band = _resolve_confidence_band(commentary_record, explain_payload)

    synthesis = sections.get("synthesis") if isinstance(sections, dict) else ""
    headline_quote = (
        _first_sentences(synthesis or "", 2)
        or _DEFAULT_HEADLINE_QUOTE
    )

    cuprum_daily = sections.get("cuprum_daily") if isinstance(sections, dict) else None
    return {
        "character": _CHARACTER,
        "mood": mood,
        "stance": stance,
        "energy": energy,
        "confidence_band": confidence_band,
        "headline_quote": headline_quote,
        "xai_highlights": _extract_xai_highlights(explain_payload),
        "cuprum_daily": cuprum_daily,
    }

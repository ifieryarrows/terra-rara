"""Offline unit tests for free physical & positioning feeds (Westmetall LME, CFTC COT, FRED JSON)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from deep_learning.data.physical_feeds import (
    parse_cftc_cot_json,
    parse_westmetall_lme_html,
)
from long_term.features import fetch_fred_macro_frames


def test_parse_westmetall_lme_html_extracts_spread_and_stocks():
    html = """
    <table>
      <tr><th>date</th><th>LME Copper Cash-Settlement</th><th>LME Copper 3-month</th><th>LME Copper stock</th></tr>
      <tr><td>26. November 2026</td><td>11,965.00</td><td>11,960.00</td><td>337,925</td></tr>
      <tr><td>25. November 2026</td><td>11,703.00</td><td>11,758.00</td><td>336,750</td></tr>
      <tr><td>22. November 2026</td><td>11,800.00</td><td>11,810.00</td><td>335,000</td></tr>
      <tr><td>20. November 2026</td><td>-</td><td>-</td><td>-</td></tr>
    </table>
    """
    df = parse_westmetall_lme_html(html)
    # Sunday 2026-11-22 and dash row must be excluded
    assert list(df.index.strftime("%Y-%m-%d")) == ["2026-11-25", "2026-11-26"]
    assert df.loc["2026-11-26", "lme_stock_tonnes"] == pytest.approx(337925.0)
    # Cash > 3M -> positive backwardation
    assert df.loc["2026-11-26", "lme_cash_3m_spread_pct"] > 0.0
    # Cash < 3M -> negative contango
    assert df.loc["2026-11-25", "lme_cash_3m_spread_pct"] < 0.0


def test_parse_cftc_cot_json_computes_net_pct_oi():
    records = [
        {
            "report_date_as_yyyy_mm_dd": "2026-11-10T00:00:00.000",
            "open_interest_all": "200000",
            "m_money_positions_long_all": "60000",
            "m_money_positions_short_all": "40000",
            "prod_merc_positions_long_all": "50000",
            "prod_merc_positions_short_all": "90000",
        },
        {
            "report_date_as_yyyy_mm_dd": "2026-11-17T00:00:00.000",
            "open_interest_all": "250000",
            "m_money_positions_long_all": "50000",
            "m_money_positions_short_all": "75000",
            "prod_merc_positions_long_all": "80000",
            "prod_merc_positions_short_all": "60000",
        },
    ]
    df = parse_cftc_cot_json(records)
    assert list(df.index.strftime("%Y-%m-%d")) == ["2026-11-10", "2026-11-17"]
    assert df.loc["2026-11-10", "cot_mm_net_pct_oi"] == pytest.approx(0.10)
    assert df.loc["2026-11-10", "cot_prod_net_pct_oi"] == pytest.approx(-0.20)
    assert df.loc["2026-11-17", "cot_mm_net_pct_oi"] == pytest.approx(-0.10)
    assert np.isfinite(df.loc["2026-11-17", "cot_mm_net"])


def test_fetch_fred_macro_frames_prefers_json_api_when_key_set(monkeypatch):
    import requests

    calls: list[tuple[str, dict]] = []

    def fake_get(url, params=None, **_kwargs):
        calls.append((url, dict(params or {})))
        if "api.stlouisfed.org" in url:
            return SimpleNamespace(
                status_code=200,
                json=lambda: {
                    "observations": [
                        {"date": "2026-11-20", "value": "4.20"},
                        {"date": "2026-11-21", "value": "."},
                        {"date": "2026-11-22", "value": "4.25"},
                    ]
                },
            )
        return SimpleNamespace(status_code=200, text="DATE,DGS10\n2026-11-20,4.10\n")

    monkeypatch.setattr(requests, "get", fake_get)

    monkeypatch.setenv("FRED_API_KEY", "test-fred-key")
    frames_json = fetch_fred_macro_frames(("DGS10",))
    assert "DGS10" in frames_json
    assert len(frames_json["DGS10"]) == 2
    assert "api.stlouisfed.org" in calls[0][0]
    assert calls[0][1]["api_key"] == "test-fred-key"

    calls.clear()
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    frames_csv = fetch_fred_macro_frames(("DGS10",))
    assert "DGS10" in frames_csv
    assert "fredgraph.csv" in calls[0][0]

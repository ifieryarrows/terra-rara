"""Tests for the hybrid Rust PyO3 / C-SIMD accelerator bridge (`app.accel`)."""

import numpy as np
from app.accel import (
    ACCEL_BACKEND,
    HAS_RUST_EXT,
    fast_canonical_title,
    fast_sharpe_ratio_hac,
    fast_softplus_np,
    fast_strip_html,
)


def test_accel_backend_active():
    assert ACCEL_BACKEND in ("rust_pyo3", "native_c_simd")
    assert isinstance(HAS_RUST_EXT, bool)


def test_fast_strip_html_and_canonical_title():
    assert fast_strip_html("<div><p>Hello <b>World</b></p></div>") == "Hello World"
    assert fast_strip_html("Plain   text\nwith\ttabs") == "Plain text with tabs"
    assert fast_strip_html("<script>alert(1)</script><p>Safe</p>") == "Safe"
    assert fast_canonical_title("Copper Futures: +3.5% in Q1!") == "copper futures 3 5 in q1"


def test_fast_softplus_matches_exact_log1p_exp():
    x = np.linspace(-25.0, 25.0, 1001, dtype=np.float64)
    expected = np.logaddexp(0.0, x)
    actual = fast_softplus_np(x)
    assert np.allclose(actual, expected, atol=1e-14, rtol=1e-14)


def test_fast_sharpe_ratio_hac_parity():
    rng = np.random.default_rng(7)
    returns = rng.normal(0.001, 0.02, size=120)
    sr = fast_sharpe_ratio_hac(returns, risk_free_rate=0.0, annualisation=52.0, max_lag=4)
    assert np.isfinite(sr)
    assert fast_sharpe_ratio_hac(np.zeros(20)) == 0.0
    assert fast_sharpe_ratio_hac(np.array([0.01])) == 0.0

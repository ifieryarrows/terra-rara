use pyo3::prelude::*;

/// Newey-West (Bartlett kernel) autocorrelation-adjusted Annualised Sharpe Ratio
/// for overlapping multi-day cumulative returns (MA(h-1) serial correlation).
#[pyfunction]
#[pyo3(signature = (returns, risk_free_rate=0.0, annualisation=52.0, max_lag=4))]
fn sharpe_ratio_hac_rs(
    returns: Vec<f64>,
    risk_free_rate: f64,
    annualisation: f64,
    max_lag: usize,
) -> f64 {
    let n = returns.len();
    if n < 2 {
        return 0.0;
    }
    let rf_step = risk_free_rate / annualisation;
    let mean_excess: f64 = returns.iter().map(|r| *r - rf_step).sum::<f64>() / (n as f64);
    let centered: Vec<f64> = returns.iter().map(|r| (*r - rf_step) - mean_excess).collect();

    let gamma_0: f64 = centered.iter().map(|c| c * c).sum::<f64>() / (n as f64);
    if gamma_0 < 1e-18 {
        return 0.0;
    }

    let mut hac_var = gamma_0;
    let effective_lag = max_lag.min(n - 1);
    for lag in 1..=effective_lag {
        let weight = 1.0 - (lag as f64) / ((effective_lag + 1) as f64);
        let mut cov_sum = 0.0;
        for i in lag..n {
            cov_sum += centered[i] * centered[i - lag];
        }
        let gamma_lag = cov_sum / (n as f64);
        hac_var += 2.0 * weight * gamma_lag;
    }

    let hac_std = hac_var.max(gamma_0 * 1e-6).sqrt();
    if hac_std < 1e-9 {
        return 0.0;
    }
    annualisation.sqrt() * mean_excess / hac_std
}

/// Fast single-pass HTML tag stripping and whitespace normalization.
#[pyfunction]
fn clean_text_rs(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let mut out = String::with_capacity(text.len());
    let mut in_tag = false;
    let mut last_was_space = true;

    for ch in text.chars() {
        match ch {
            '<' => {
                in_tag = true;
                if !last_was_space {
                    out.push(' ');
                    last_was_space = true;
                }
            }
            '>' => {
                in_tag = false;
            }
            c if !in_tag => {
                if c.is_whitespace() {
                    if !last_was_space {
                        out.push(' ');
                        last_was_space = true;
                    }
                } else {
                    out.push(c);
                    last_was_space = false;
                }
            }
            _ => {}
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

/// Canonicalize a news title (lowercase, alphanumeric + whitespace collapse).
#[pyfunction]
fn canonical_title_rs(title: &str) -> String {
    if title.is_empty() {
        return String::new();
    }
    let mut out = String::with_capacity(title.len());
    let mut last_was_space = true;

    for ch in title.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            for lc in ch.to_lowercase() {
                out.push(lc);
            }
            last_was_space = false;
        } else if !last_was_space {
            out.push(' ');
            last_was_space = true;
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

#[pymodule]
fn copper_rust_accel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sharpe_ratio_hac_rs, m)?)?;
    m.add_function(wrap_pyfunction!(clean_text_rs, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_title_rs, m)?)?;
    Ok(())
}

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

#[inline(always)]
fn bytes_as_f64_slice(bytes: &[u8]) -> &[f64] {
    assert_eq!(bytes.len() % 8, 0, "Byte buffer length must be a multiple of 8");
    assert_eq!(
        (bytes.as_ptr() as usize) % std::mem::align_of::<f64>(),
        0,
        "Unaligned f64 buffer"
    );
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f64, bytes.len() / 8) }
}

#[inline(always)]
fn softplus_f64(x: f64) -> f64 {
    (-x.abs()).exp().ln_1p() + x.max(0.0)
}

/// Core Newey-West HAC Sharpe computation on a slice of f64.
#[inline]
fn compute_hac_sharpe_slice(
    arr: &[f64],
    risk_free_rate: f64,
    annualisation: f64,
    max_lag: usize,
) -> f64 {
    let n = arr.len();
    if n < 2 {
        return 0.0;
    }
    let rf_step = risk_free_rate / annualisation;
    let inv_n = 1.0 / (n as f64);
    let mean_excess: f64 = arr.iter().map(|r| *r - rf_step).sum::<f64>() * inv_n;

    let mut gamma_0 = 0.0;
    let mut centered = Vec::with_capacity(n);
    for &r in arr {
        let c = (r - rf_step) - mean_excess;
        gamma_0 += c * c;
        centered.push(c);
    }
    gamma_0 *= inv_n;
    if gamma_0 < 1e-18 {
        return 0.0;
    }

    let mut hac_var = gamma_0;
    let effective_lag = max_lag.min(n - 1);
    let denom = (effective_lag + 1) as f64;
    for lag in 1..=effective_lag {
        let weight = 1.0 - (lag as f64) / denom;
        let mut cov_sum = 0.0;
        let c_hi = &centered[lag..n];
        let c_lo = &centered[..n - lag];
        for (&a, &b) in c_hi.iter().zip(c_lo.iter()) {
            cov_sum += a * b;
        }
        hac_var += 2.0 * weight * (cov_sum * inv_n);
    }

    let hac_std = hac_var.max(gamma_0 * 1e-6).sqrt();
    if hac_std < 1e-9 {
        return 0.0;
    }
    annualisation.sqrt() * mean_excess / hac_std
}

/// Newey-West HAC Sharpe from a Python list or sequence.
#[pyfunction]
#[pyo3(signature = (returns, risk_free_rate=0.0, annualisation=52.0, max_lag=4))]
fn sharpe_ratio_hac_rs(
    returns: Vec<f64>,
    risk_free_rate: f64,
    annualisation: f64,
    max_lag: usize,
) -> f64 {
    compute_hac_sharpe_slice(&returns, risk_free_rate, annualisation, max_lag)
}

/// Zero-copy Newey-West HAC Sharpe from a contiguous float64 buffer.
#[pyfunction]
#[pyo3(signature = (buf, risk_free_rate=0.0, annualisation=52.0, max_lag=4))]
fn sharpe_ratio_hac_bytes_rs(
    buf: &[u8],
    risk_free_rate: f64,
    annualisation: f64,
    max_lag: usize,
) -> f64 {
    let slice = bytes_as_f64_slice(buf);
    compute_hac_sharpe_slice(slice, risk_free_rate, annualisation, max_lag)
}

/// Single-pass register-level quantile crossing rate over a flattened [N_rows, Q] f64 buffer.
#[pyfunction]
#[pyo3(signature = (buf, q_dim, eps=1e-6))]
fn quantile_crossing_rate_bytes_rs(buf: &[u8], q_dim: usize, eps: f64) -> f64 {
    if q_dim < 2 || buf.is_empty() {
        return 0.0;
    }
    let slice = bytes_as_f64_slice(buf);
    let n_rows = slice.len() / q_dim;
    if n_rows == 0 {
        return 0.0;
    }
    let mut violations: usize = 0;
    let neg_eps = -eps;
    for row in slice.chunks_exact(q_dim) {
        for j in 0..(q_dim - 1) {
            if (row[j + 1] - row[j]) < neg_eps {
                violations += 1;
            }
        }
    }
    (violations as f64) / ((n_rows * (q_dim - 1)) as f64)
}

/// Stack-sorted (zero heap allocation for Q <= 16) median sort gap (mean, max).
#[pyfunction]
fn quantile_median_sort_gap_bytes_rs(
    buf: &[u8],
    q_dim: usize,
    median_idx: usize,
) -> (f64, f64) {
    if q_dim == 0 || buf.is_empty() || median_idx >= q_dim {
        return (0.0, 0.0);
    }
    let slice = bytes_as_f64_slice(buf);
    let n_rows = slice.len() / q_dim;
    if n_rows == 0 {
        return (0.0, 0.0);
    }

    let mut sum_gap = 0.0;
    let mut max_gap = 0.0_f64;

    if q_dim <= 16 {
        let mut stack_buf = [0.0_f64; 16];
        for row in slice.chunks_exact(q_dim) {
            let orig_med = row[median_idx];
            stack_buf[..q_dim].copy_from_slice(row);
            stack_buf[..q_dim].sort_by(|a, b| a.total_cmp(b));
            let gap = (orig_med - stack_buf[median_idx]).abs();
            sum_gap += gap;
            if gap > max_gap {
                max_gap = gap;
            }
        }
    } else {
        let mut heap_buf = vec![0.0_f64; q_dim];
        for row in slice.chunks_exact(q_dim) {
            let orig_med = row[median_idx];
            heap_buf.copy_from_slice(row);
            heap_buf.sort_by(|a, b| a.total_cmp(b));
            let gap = (orig_med - heap_buf[median_idx]).abs();
            sum_gap += gap;
            if gap > max_gap {
                max_gap = gap;
            }
        }
    }

    (sum_gap / (n_rows as f64), max_gap)
}

/// Monotonic quantile transformation in Rust (zero PyTorch overhead, stack-allocated steps).
#[pyfunction]
#[pyo3(signature = (buf, q_dim, median_idx, min_gap=1e-5, gap_scale=0.03, init_bias=-3.0))]
fn monotonic_quantiles_bytes_rs<'py>(
    py: Python<'py>,
    buf: &[u8],
    q_dim: usize,
    median_idx: usize,
    min_gap: f64,
    gap_scale: f64,
    init_bias: f64,
) -> Bound<'py, PyBytes> {
    if q_dim == 0 || buf.is_empty() {
        return PyBytes::new_bound(py, buf);
    }
    let slice = bytes_as_f64_slice(buf);
    let mut out = vec![0.0_f64; slice.len()];

    for (row_in, row_out) in slice.chunks_exact(q_dim).zip(out.chunks_exact_mut(q_dim)) {
        let base = row_in[median_idx];
        row_out[median_idx] = base;

        let mut cum_low = 0.0;
        for k in 0..median_idx {
            let src_idx = median_idx - 1 - k;
            let step = min_gap + gap_scale * softplus_f64(row_in[src_idx] + init_bias);
            cum_low += step;
            row_out[src_idx] = base - cum_low;
        }

        let mut cum_high = 0.0;
        for src_idx in (median_idx + 1)..q_dim {
            let step = min_gap + gap_scale * softplus_f64(row_in[src_idx] + init_bias);
            cum_high += step;
            row_out[src_idx] = base + cum_high;
        }
    }

    let out_bytes =
        unsafe { std::slice::from_raw_parts(out.as_ptr() as *const u8, out.len() * 8) };
    PyBytes::new_bound(py, out_bytes)
}

/// Rayon-parallelized weekly interval calibration grid search over (power_candidates x scale_candidates).
/// Eliminates the 4D intermediate tensor allocation and evaluates all candidates across all CPU cores.
#[pyfunction]
#[pyo3(signature = (actual_weekly_buf, weekly_base_buf, offset_active_buf, centered_active_buf, candidates_buf, clipped_ratio_buf, power, n_samples, horizon, n_low, n_high, min_scale, max_scale, min_gap, gap_scale, alpha))]
#[allow(clippy::too_many_arguments)]
fn fit_weekly_interval_grid_rs(
    actual_weekly_buf: &[u8],
    weekly_base_buf: &[u8],
    offset_active_buf: &[u8],
    centered_active_buf: &[u8],
    candidates_buf: &[u8],
    clipped_ratio_buf: Option<&[u8]>,
    power: f64,
    n_samples: usize,
    horizon: usize,
    n_low: usize,
    n_high: usize,
    min_scale: f64,
    max_scale: f64,
    min_gap: f64,
    gap_scale: f64,
    alpha: f64,
) -> (Vec<f64>, Vec<f64>) {
    let actual_weekly = bytes_as_f64_slice(actual_weekly_buf);
    let weekly_base = bytes_as_f64_slice(weekly_base_buf);
    let offset_active = bytes_as_f64_slice(offset_active_buf);
    let centered_active = bytes_as_f64_slice(centered_active_buf);
    let candidates = bytes_as_f64_slice(candidates_buf);
    let clipped_ratio = clipped_ratio_buf.map(bytes_as_f64_slice);

    let n_active = n_low + n_high;
    let base_gap_low = (horizon * n_low) as f64 * min_gap;
    let base_gap_high = (horizon * n_high) as f64 * min_gap;
    let inv_alpha_2 = 2.0 / alpha;
    let inv_n = 1.0 / (n_samples as f64);

    let factors: Option<Vec<f64>> =
        clipped_ratio.map(|cr| cr.iter().map(|&r| r.powf(power)).collect());

    let results: Vec<(f64, f64)> = candidates
        .par_iter()
        .map(|&cand| {
            let mut covered_count = 0usize;
            let mut score_sum = 0.0_f64;

            for i in 0..n_samples {
                let mult = match &factors {
                    Some(f) => (cand * f[i]).clamp(min_scale, max_scale),
                    None => cand,
                };
                let mut low_sp_sum = 0.0_f64;
                let mut high_sp_sum = 0.0_f64;

                let sample_offset_base = i * horizon;
                let sample_active_base = i * horizon * n_active;

                for h in 0..horizon {
                    let off = offset_active[sample_offset_base + h];
                    let row_base = sample_active_base + h * n_active;
                    for c in 0..n_low {
                        low_sp_sum += softplus_f64(off + mult * centered_active[row_base + c]);
                    }
                    for c in n_low..n_active {
                        high_sp_sum += softplus_f64(off + mult * centered_active[row_base + c]);
                    }
                }

                let lo = weekly_base[i] - (base_gap_low + gap_scale * low_sp_sum);
                let hi = weekly_base[i] + (base_gap_high + gap_scale * high_sp_sum);
                let y = actual_weekly[i];

                if y >= lo && y <= hi {
                    covered_count += 1;
                }
                let width = hi - lo;
                let below = (lo - y).max(0.0);
                let above = (y - hi).max(0.0);
                score_sum += width + inv_alpha_2 * (below + above);
            }

            ((covered_count as f64) * inv_n, score_sum * inv_n)
        })
        .collect();

    let mut coverages = Vec::with_capacity(results.len());
    let mut scores = Vec::with_capacity(results.len());
    for (c, s) in results {
        coverages.push(c);
        scores.push(s);
    }
    (coverages, scores)
}

#[inline]
fn decode_entity_str(entity: &str) -> Option<char> {
    match entity {
        "amp" => Some('&'),
        "quot" => Some('"'),
        "apos" => Some('\''),
        "lt" => Some('<'),
        "gt" => Some('>'),
        "nbsp" => Some(' '),
        "copy" => Some('©'),
        "reg" => Some('®'),
        "trade" => Some('™'),
        "mdash" => Some('—'),
        "ndash" => Some('–'),
        "hellip" => Some('…'),
        "lsquo" => Some('‘'),
        "rsquo" => Some('’'),
        "ldquo" => Some('“'),
        "rdquo" => Some('”'),
        "bull" => Some('•'),
        "cent" => Some('¢'),
        "pound" => Some('£'),
        "yen" => Some('¥'),
        "euro" => Some('€'),
        "deg" => Some('°'),
        "plusmn" => Some('±'),
        "times" => Some('×'),
        "divide" => Some('÷'),
        "micro" => Some('µ'),
        "para" => Some('¶'),
        "middot" => Some('·'),
        "frac14" => Some('¼'),
        "frac12" => Some('½'),
        "frac34" => Some('¾'),
        "sup1" => Some('¹'),
        "sup2" => Some('²'),
        "sup3" => Some('³'),
        "sect" => Some('§'),
        "laquo" => Some('«'),
        "raquo" => Some('»'),
        "iquest" => Some('¿'),
        "iexcl" => Some('¡'),
        "Agrave" => Some('À'),
        "Aacute" => Some('Á'),
        "Acirc" => Some('Â'),
        "Atilde" => Some('Ã'),
        "Auml" => Some('Ä'),
        "Aring" => Some('Å'),
        "AElig" => Some('Æ'),
        "Ccedil" => Some('Ç'),
        "Egrave" => Some('È'),
        "Eacute" => Some('É'),
        "Ecirc" => Some('Ê'),
        "Euml" => Some('Ë'),
        "Igrave" => Some('Ì'),
        "Iacute" => Some('Í'),
        "Icirc" => Some('Î'),
        "Iuml" => Some('Ï'),
        "ETH" => Some('Ð'),
        "Ntilde" => Some('Ñ'),
        "Ograve" => Some('Ò'),
        "Oacute" => Some('Ó'),
        "Ocirc" => Some('Ô'),
        "Otilde" => Some('Õ'),
        "Ouml" => Some('Ö'),
        "Oslash" => Some('Ø'),
        "Ugrave" => Some('Ù'),
        "Uacute" => Some('Ú'),
        "Ucirc" => Some('Û'),
        "Uuml" => Some('Ü'),
        "Yacute" => Some('Ý'),
        "THORN" => Some('Þ'),
        "szlig" => Some('ß'),
        "agrave" => Some('à'),
        "aacute" => Some('á'),
        "acirc" => Some('â'),
        "atilde" => Some('ã'),
        "auml" => Some('ä'),
        "aring" => Some('å'),
        "aelig" => Some('æ'),
        "ccedil" => Some('ç'),
        "egrave" => Some('è'),
        "eacute" => Some('é'),
        "ecirc" => Some('ê'),
        "euml" => Some('ë'),
        "igrave" => Some('ì'),
        "iacute" => Some('í'),
        "icirc" => Some('î'),
        "iuml" => Some('ï'),
        "eth" => Some('ð'),
        "ntilde" => Some('ñ'),
        "ograve" => Some('ò'),
        "oacute" => Some('ó'),
        "ocirc" => Some('ô'),
        "otilde" => Some('õ'),
        "ouml" => Some('ö'),
        "oslash" => Some('ø'),
        "ugrave" => Some('ù'),
        "uacute" => Some('ú'),
        "ucirc" => Some('û'),
        "uuml" => Some('ü'),
        "yacute" => Some('ý'),
        "thorn" => Some('þ'),
        "yuml" => Some('ÿ'),
        s if s.starts_with("#x") || s.starts_with("#X") => {
            u32::from_str_radix(&s[2..], 16).ok().and_then(char::from_u32)
        }
        s if s.starts_with('#') => {
            s[1..].parse::<u32>().ok().and_then(char::from_u32)
        }
        _ => None,
    }
}

/// Fast single-pass HTML tag stripping, entity decoding, and whitespace normalization.
#[pyfunction]
fn clean_text_rs(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let mut out = String::with_capacity(text.len());
    let mut in_tag = false;
    let mut last_was_space = true;
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        let b = bytes[i];
        match b {
            b'<' => {
                in_tag = true;
                if !last_was_space {
                    out.push(' ');
                    last_was_space = true;
                }
                i += 1;
            }
            b'>' => {
                in_tag = false;
                i += 1;
            }
            _ if in_tag => {
                i += 1;
            }
            b'&' => {
                let mut found_semi = false;
                let end = (i + 12).min(len);
                for j in (i + 1)..end {
                    let c = bytes[j];
                    if c == b';' {
                        if let Ok(cand) = std::str::from_utf8(&bytes[i + 1..j]) {
                            if let Some(decoded_char) = decode_entity_str(cand) {
                                if decoded_char.is_whitespace() {
                                    if !last_was_space {
                                        out.push(' ');
                                        last_was_space = true;
                                    }
                                } else {
                                    out.push(decoded_char);
                                    last_was_space = false;
                                }
                                i = j + 1;
                                found_semi = true;
                                break;
                            }
                        }
                        break;
                    } else if c.is_ascii_whitespace() || c == b'<' || c == b'&' {
                        break;
                    }
                }
                if !found_semi {
                    out.push('&');
                    last_was_space = false;
                    i += 1;
                }
            }
            _ => {
                let remaining = &text[i..];
                let ch = remaining.chars().next().unwrap_or(' ');
                if ch.is_whitespace() {
                    if !last_was_space {
                        out.push(' ');
                        last_was_space = true;
                    }
                } else {
                    out.push(ch);
                    last_was_space = false;
                }
                i += ch.len_utf8();
            }
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

/// Batch parallel canonicalization of news titles using Rayon.
#[pyfunction]
fn canonical_title_batch_rs(titles: Vec<String>) -> Vec<String> {
    titles.par_iter().map(|t| canonical_title_rs(t)).collect()
}

/// Fast row-wise OLS slope over a 2D [n_rows, n_cols] float64 buffer with NaN handling.
#[pyfunction]
fn ols_row_slope_bytes_rs<'py>(
    py: Python<'py>,
    buf: &[u8],
    n_rows: usize,
    n_cols: usize,
) -> Bound<'py, PyBytes> {
    let slice = bytes_as_f64_slice(buf);
    let mut out = vec![f64::NAN; n_rows];

    if n_cols >= 2 && slice.len() == n_rows * n_cols {
        for (row_idx, row) in slice.chunks_exact(n_cols).enumerate() {
            let mut count = 0usize;
            let mut sum_r = 0.0_f64;
            let mut sum_v = 0.0_f64;
            for (c, &v) in row.iter().enumerate() {
                if !v.is_nan() {
                    count += 1;
                    sum_r += c as f64;
                    sum_v += v;
                }
            }
            if count < 2 {
                continue;
            }
            let mean_r = sum_r / (count as f64);
            let mean_v = sum_v / (count as f64);
            let mut num = 0.0_f64;
            let mut den = 0.0_f64;
            for (c, &v) in row.iter().enumerate() {
                if !v.is_nan() {
                    let dr = (c as f64) - mean_r;
                    let dv = v - mean_v;
                    num += dr * dv;
                    den += dr * dr;
                }
            }
            if den > 0.0 {
                out[row_idx] = num / den;
            }
        }
    }

    let out_bytes =
        unsafe { std::slice::from_raw_parts(out.as_ptr() as *const u8, out.len() * 8) };
    PyBytes::new_bound(py, out_bytes)
}

#[pymodule]
fn copper_rust_accel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sharpe_ratio_hac_rs, m)?)?;
    m.add_function(wrap_pyfunction!(sharpe_ratio_hac_bytes_rs, m)?)?;
    m.add_function(wrap_pyfunction!(quantile_crossing_rate_bytes_rs, m)?)?;
    m.add_function(wrap_pyfunction!(quantile_median_sort_gap_bytes_rs, m)?)?;
    m.add_function(wrap_pyfunction!(monotonic_quantiles_bytes_rs, m)?)?;
    m.add_function(wrap_pyfunction!(fit_weekly_interval_grid_rs, m)?)?;
    m.add_function(wrap_pyfunction!(clean_text_rs, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_title_rs, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_title_batch_rs, m)?)?;
    m.add_function(wrap_pyfunction!(ols_row_slope_bytes_rs, m)?)?;
    Ok(())
}

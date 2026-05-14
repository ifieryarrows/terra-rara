from __future__ import annotations

import numpy as np


def compute_structural_invalidity_report(fold_diagnostics: list) -> dict:
    completed = [d for d in fold_diagnostics if d["state"] == "COMPLETE"]
    n = len(completed)
    if n == 0:
        return {"completed_trials": 0, "verdict": "NO_COMPLETE_TRIALS"}

    checks = {
        "public_crossing_le_0_001": sum(
            1
            for d in completed
            if d.get("avg_quantile_crossing_rate", 1.0) <= 0.001
        ),
        "weekly_magnitude_le_3_0": sum(
            1 for d in completed if d.get("avg_weekly_magnitude_ratio", 999) <= 3.0
        ),
        "weekly_pi80_width_ratio_le_4_0": sum(
            1 for d in completed if d.get("avg_weekly_pi80_width_ratio", 999) <= 4.0
        ),
        "variance_ratio_le_3_0": sum(
            1 for d in completed if d.get("avg_variance_ratio", 999) <= 3.0
        ),
        "directional_accuracy_ge_0_50": sum(
            1 for d in completed if d.get("avg_directional_accuracy", 0) >= 0.50
        ),
    }

    all_pass_count = sum(
        1
        for d in completed
        if (
            d.get("avg_quantile_crossing_rate", 1.0) <= 0.001
            and d.get("avg_weekly_magnitude_ratio", 999) <= 3.0
            and d.get("avg_weekly_pi80_width_ratio", 999) <= 4.0
            and d.get("avg_variance_ratio", 999) <= 3.0
        )
    )

    verdict = (
        "STRUCTURAL_FAILURE"
        if all_pass_count == 0
        else "PARTIAL_STRUCTURAL_FAILURE"
        if all_pass_count < n // 2
        else "ACCEPTABLE"
    )

    next_action = {
        "STRUCTURAL_FAILURE": (
            "Do not run additional hyperopt. Fix quantile head architecture "
            "and loss function before any further search."
        ),
        "PARTIAL_STRUCTURAL_FAILURE": (
            "Some trials pass structural checks. Investigate what differentiates "
            "passing from failing configurations before expanding search."
        ),
        "ACCEPTABLE": "Proceed with expanded hyperopt search.",
    }[verdict]

    return {
        "completed_trials": n,
        "structural_check_results": checks,
        "trials_passing_all_checks": all_pass_count,
        "verdict": verdict,
        "next_action": next_action,
    }


def compute_trial_distribution_summary(fold_diagnostics: list) -> dict:
    completed = [d for d in fold_diagnostics if d["state"] == "COMPLETE"]
    if not completed:
        return {}

    metrics_of_interest = [
        "avg_quantile_crossing_rate",
        "avg_raw_quantile_crossing_rate",
        "avg_weekly_raw_crossing_rate",
        "avg_weekly_magnitude_ratio",
        "avg_weekly_pi80_width_ratio",
        "avg_weekly_pi96_width_ratio",
        "fold_score_std",
        "avg_variance_ratio",
        "avg_directional_accuracy",
    ]

    summary = {}
    for metric in metrics_of_interest:
        values = [d.get(metric) for d in completed if d.get(metric) is not None]
        if values:
            summary[metric] = {
                "min": float(np.min(values)),
                "p25": float(np.percentile(values, 25)),
                "median": float(np.median(values)),
                "p75": float(np.percentile(values, 75)),
                "max": float(np.max(values)),
            }

    return summary


def best_trial_preflight_check(best_trial_diagnostics: dict) -> dict:
    checks = {
        "public_crossing_le_0_001": best_trial_diagnostics.get(
            "avg_quantile_crossing_rate", 1.0
        )
        <= 0.001,
        "weekly_magnitude_le_3_0": best_trial_diagnostics.get(
            "avg_weekly_magnitude_ratio", 999
        )
        <= 3.0,
        "weekly_pi80_width_le_4_0": best_trial_diagnostics.get(
            "avg_weekly_pi80_width_ratio", 999
        )
        <= 4.0,
        "variance_ratio_le_3_0": best_trial_diagnostics.get(
            "avg_variance_ratio", 999
        )
        <= 3.0,
        "directional_accuracy_ge_0_50": best_trial_diagnostics.get(
            "avg_directional_accuracy", 0
        )
        >= 0.50,
    }

    passed = sum(checks.values())
    total = len(checks)
    all_pass = passed == total

    return {
        "checks": checks,
        "passed": passed,
        "total": total,
        "preflight_passed": all_pass,
        "recommendation": (
            "Proceed to final training."
            if all_pass
            else (
                f"Best trial failed {total - passed}/{total} structural checks. "
                "Final training will likely produce a quality gate failure. "
                "Fix architecture before proceeding."
            )
        ),
    }

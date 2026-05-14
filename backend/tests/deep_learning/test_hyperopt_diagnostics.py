from scripts.hyperopt_diagnostics import (
    best_trial_preflight_check,
    compute_structural_invalidity_report,
    compute_trial_distribution_summary,
)


def test_structural_invalidity_report_blocks_when_no_completed_trial_passes():
    fold_diagnostics = [
        {
            "state": "COMPLETE",
            "avg_quantile_crossing_rate": 0.0,
            "avg_weekly_magnitude_ratio": 17.1,
            "avg_weekly_pi80_width_ratio": 9.7,
            "avg_variance_ratio": 5.9,
            "avg_directional_accuracy": 0.49,
        },
        {
            "state": "PRUNED",
            "avg_weekly_magnitude_ratio": 99.0,
        },
    ]

    report = compute_structural_invalidity_report(fold_diagnostics)

    assert report["completed_trials"] == 1
    assert report["trials_passing_all_checks"] == 0
    assert report["verdict"] == "STRUCTURAL_FAILURE"
    assert "Fix quantile head architecture" in report["next_action"]


def test_trial_distribution_summary_reports_percentiles():
    summary = compute_trial_distribution_summary(
        [
            {"state": "COMPLETE", "avg_variance_ratio": 1.0, "fold_score_std": 2.0},
            {"state": "COMPLETE", "avg_variance_ratio": 3.0, "fold_score_std": 4.0},
        ]
    )

    assert summary["avg_variance_ratio"]["min"] == 1.0
    assert summary["avg_variance_ratio"]["median"] == 2.0
    assert summary["fold_score_std"]["max"] == 4.0


def test_best_trial_preflight_check_recommends_architecture_fix_when_failed():
    result = best_trial_preflight_check(
        {
            "avg_quantile_crossing_rate": 0.0,
            "avg_weekly_magnitude_ratio": 17.0,
            "avg_weekly_pi80_width_ratio": 9.0,
            "avg_variance_ratio": 5.0,
            "avg_directional_accuracy": 0.51,
        }
    )

    assert result["preflight_passed"] is False
    assert result["passed"] == 2
    assert "Fix architecture" in result["recommendation"]

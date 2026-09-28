import numpy as np
import pandas as pd

from ads_rank_lab.evaluation.pctr import evaluate_by_advertiser, evaluate_pctr, expected_calibration_error


def test_expected_calibration_error_is_zero_for_perfect_bins() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, p, n_bins=2) == 0.0


def test_evaluate_pctr_returns_core_metrics() -> None:
    metrics = evaluate_pctr(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    for key in (
        "roc_auc",
        "pr_auc",
        "pr_auc_baseline",
        "pr_auc_lift",
        "log_loss",
        "brier",
        "ece",
        "ctr",
        "predicted_ctr_mean",
        "predicted_to_observed_ctr_ratio",
    ):
        assert key in metrics


def test_evaluate_by_advertiser_reports_calibration_gap() -> None:
    frame = pd.DataFrame({"advertiser_id": ["a", "a", "b"], "clicked": [0, 1, 0]})
    result = evaluate_by_advertiser(frame, [0.2, 0.8, 0.1], min_rows=2)
    assert result["advertiser_id"].tolist() == ["a"]
    assert result.loc[0, "rows"] == 2
    assert result.loc[0, "ctr"] == 0.5
    assert result.loc[0, "predicted_ctr_mean"] == 0.5


def test_pr_auc_lift_uses_ctr_as_random_ranking_baseline() -> None:
    metrics = evaluate_pctr(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert metrics["pr_auc_baseline"] == 0.5
    assert metrics["pr_auc_lift"] >= 1.0


def test_advertiser_metrics_include_relative_calibration() -> None:
    frame = pd.DataFrame({"advertiser_id": ["a", "a"], "clicked": [0, 1]})
    result = evaluate_by_advertiser(frame, [0.2, 0.8], min_rows=2)
    assert result.loc[0, "absolute_calibration_gap"] == 0.0
    assert result.loc[0, "predicted_to_observed_ctr_ratio"] == 1.0

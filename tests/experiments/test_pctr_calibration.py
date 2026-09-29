import numpy as np

from ads_rank_lab.evaluation.pctr import evaluate_pctr
from ads_rank_lab.models.calibration import fit_intercept_calibrator


def test_intercept_calibration_keeps_ranking_metrics_unchanged() -> None:
    y_validation = np.array([0, 0, 0, 1])
    p_validation = np.array([0.10, 0.20, 0.30, 0.60])
    y_test = np.array([0, 0, 1, 1])
    p_test = np.array([0.15, 0.25, 0.50, 0.70])

    calibrator = fit_intercept_calibrator(y_validation, p_validation)
    raw = evaluate_pctr(y_test, p_test)
    calibrated = evaluate_pctr(y_test, calibrator.predict(p_test))

    assert calibrated["roc_auc"] == raw["roc_auc"]
    assert calibrated["pr_auc"] == raw["pr_auc"]

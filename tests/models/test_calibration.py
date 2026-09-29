import numpy as np

from ads_rank_lab.models.calibration import (
    fit_intercept_calibrator,
    fit_platt_calibrator,
)


def test_intercept_calibrator_matches_validation_prevalence_and_preserves_order() -> None:
    y = np.array([0, 0, 0, 1, 1])
    raw = np.array([0.05, 0.10, 0.20, 0.30, 0.40])

    calibrator = fit_intercept_calibrator(y, raw)
    calibrated = calibrator.predict(raw)

    assert abs(calibrated.mean() - y.mean()) < 1e-10
    assert np.array_equal(np.argsort(raw), np.argsort(calibrated))


def test_platt_calibrator_returns_probabilities_and_preserves_order_on_monotone_fit() -> None:
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    raw = np.array([0.02, 0.05, 0.10, 0.20, 0.55, 0.70, 0.80, 0.90])

    calibrator = fit_platt_calibrator(y, raw)
    calibrated = calibrator.predict(raw)

    assert calibrator.slope > 0
    assert ((calibrated >= 0.0) & (calibrated <= 1.0)).all()
    assert np.array_equal(np.argsort(raw), np.argsort(calibrated))

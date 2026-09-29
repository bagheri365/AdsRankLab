"""Probability calibration for pCTR models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression


def _as_probability_array(probabilities) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1:
        raise ValueError("probabilities must be one-dimensional")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    return values


def probability_to_logit(probabilities, *, eps: float = 1e-12) -> np.ndarray:
    """Convert probabilities to finite logits."""
    values = _as_probability_array(probabilities)
    clipped = np.clip(values, eps, 1.0 - eps)
    return np.log(clipped) - np.log1p(-clipped)


def sigmoid(values) -> np.ndarray:
    """Numerically stable logistic transform."""
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values, dtype=float)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


@dataclass(frozen=True)
class InterceptCalibrator:
    """Log-odds intercept shift that preserves ranking exactly."""

    delta: float

    def predict(self, probabilities) -> np.ndarray:
        return sigmoid(probability_to_logit(probabilities) + self.delta)


@dataclass(frozen=True)
class PlattCalibrator:
    """Affine logistic calibration fitted on raw prediction logits."""

    slope: float
    intercept: float

    def predict(self, probabilities) -> np.ndarray:
        logits = probability_to_logit(probabilities)
        return sigmoid(self.slope * logits + self.intercept)


def fit_intercept_calibrator(y_true, probabilities) -> InterceptCalibrator:
    """Fit one log-odds offset to match validation prevalence.

    The transformation is strictly monotonic and therefore preserves ranking,
    ROC-AUC, and PR-AUC up to floating-point ties.
    """
    y_true = np.asarray(y_true, dtype=float)
    logits = probability_to_logit(probabilities)
    if len(y_true) != len(logits):
        raise ValueError("y_true and probabilities must have the same length")
    if len(y_true) == 0:
        raise ValueError("cannot fit calibration on an empty sample")
    target = float(y_true.mean())
    if not 0.0 < target < 1.0:
        raise ValueError("calibration requires both positive and negative labels")

    def objective(delta: float) -> float:
        return float(sigmoid(logits + delta).mean() - target)

    delta = float(brentq(objective, -50.0, 50.0))
    return InterceptCalibrator(delta=delta)


def fit_platt_calibrator(y_true, probabilities) -> PlattCalibrator:
    """Fit Platt scaling on validation-set raw logits only."""
    y_true = np.asarray(y_true, dtype=int)
    logits = probability_to_logit(probabilities)
    if len(y_true) != len(logits):
        raise ValueError("y_true and probabilities must have the same length")
    if len(np.unique(y_true)) < 2:
        raise ValueError("calibration requires both positive and negative labels")

    model = LogisticRegression(
        solver="lbfgs",
        C=1e6,
        max_iter=1000,
        class_weight=None,
    )
    model.fit(logits.reshape(-1, 1), y_true)
    return PlattCalibrator(
        slope=float(model.coef_[0, 0]),
        intercept=float(model.intercept_[0]),
    )

"""Evaluation metrics for pCTR experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def expected_calibration_error(y_true, y_prob, *, n_bins: int = 10) -> float:
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        raise ValueError("Cannot compute calibration error on an empty sample.")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for idx in range(n_bins):
        mask = bins == idx
        if not np.any(mask):
            continue
        ece += (mask.sum() / len(y_true)) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return float(ece)


def evaluate_pctr(y_true, y_prob, *, n_bins: int = 10) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        raise ValueError("pCTR evaluation requires both positive and negative labels.")

    ctr = float(y_true.mean())
    pr_auc = float(average_precision_score(y_true, y_prob))
    predicted_ctr_mean = float(y_prob.mean())
    return {
        "rows": float(len(y_true)),
        "ctr": ctr,
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": pr_auc,
        "pr_auc_baseline": ctr,
        "pr_auc_lift": float(pr_auc / ctr) if ctr > 0 else float("nan"),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
        "predicted_ctr_mean": predicted_ctr_mean,
        "predicted_to_observed_ctr_ratio": (
            float(predicted_ctr_mean / ctr) if ctr > 0 else float("nan")
        ),
    }


def evaluate_by_advertiser(frame: pd.DataFrame, probabilities, *, min_rows: int = 1000) -> pd.DataFrame:
    if "advertiser_id" not in frame.columns or "clicked" not in frame.columns:
        raise ValueError("frame must contain advertiser_id and clicked")
    working = frame[["advertiser_id", "clicked"]].copy()
    working["pctr"] = np.asarray(probabilities, dtype=float)
    rows = []
    for advertiser_id, group in working.groupby("advertiser_id", dropna=False):
        if len(group) < min_rows:
            continue
        ctr = float(group["clicked"].mean())
        predicted = float(group["pctr"].mean())
        rows.append({
            "advertiser_id": advertiser_id,
            "rows": int(len(group)),
            "ctr": ctr,
            "predicted_ctr_mean": predicted,
            "calibration_gap": predicted - ctr,
            "absolute_calibration_gap": abs(predicted - ctr),
            "predicted_to_observed_ctr_ratio": (
                predicted / ctr if ctr > 0 else float("nan")
            ),
        })
    columns = [
        "advertiser_id",
        "rows",
        "ctr",
        "predicted_ctr_mean",
        "calibration_gap",
        "absolute_calibration_gap",
        "predicted_to_observed_ctr_ratio",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values("rows", ascending=False, ignore_index=True)

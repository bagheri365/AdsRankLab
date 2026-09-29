"""Validation-fitted calibration study for the Season 2 pCTR baseline."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from ads_rank_lab.evaluation.pctr import evaluate_by_advertiser, evaluate_pctr
from ads_rank_lab.experiments.pctr_baseline import SEASON2_SPLIT, load_dates
from ads_rank_lab.models.calibration import (
    fit_intercept_calibrator,
    fit_platt_calibrator,
)
from ads_rank_lab.models.pctr import (
    PctrBaselineConfig,
    build_pctr_pipeline,
    feature_columns,
    validate_feature_frame,
)


def _method_result(y_validation, validation_prob, y_test, test_prob) -> dict:
    return {
        "validation": evaluate_pctr(y_validation, validation_prob),
        "test": evaluate_pctr(y_test, test_prob),
    }


def run_season2_calibration(
    dataset_root: str | Path,
    *,
    config: PctrBaselineConfig | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Fit the ranking model on train, calibrators on validation, evaluate on test."""
    config = config or PctrBaselineConfig()
    train = load_dates(dataset_root, season=2, dates=SEASON2_SPLIT["train"])
    validation = load_dates(dataset_root, season=2, dates=SEASON2_SPLIT["validation"])
    test = load_dates(dataset_root, season=2, dates=SEASON2_SPLIT["test"])

    for frame in (train, validation, test):
        validate_feature_frame(frame, config)

    features = list(feature_columns(config))
    pipeline = build_pctr_pipeline(config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(train[features], train["clicked"])

    convergence_warnings = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]

    validation_raw = pipeline.predict_proba(validation[features])[:, 1]
    test_raw = pipeline.predict_proba(test[features])[:, 1]

    intercept = fit_intercept_calibrator(validation["clicked"], validation_raw)
    platt = fit_platt_calibrator(validation["clicked"], validation_raw)

    validation_intercept = intercept.predict(validation_raw)
    test_intercept = intercept.predict(test_raw)
    validation_platt = platt.predict(validation_raw)
    test_platt = platt.predict(test_raw)

    fitted_model = pipeline.named_steps["model"]
    iterations = int(fitted_model.n_iter_.max())

    raw_metrics = _method_result(
        validation["clicked"], validation_raw, test["clicked"], test_raw
    )
    intercept_metrics = _method_result(
        validation["clicked"], validation_intercept, test["clicked"], test_intercept
    )
    platt_metrics = _method_result(
        validation["clicked"], validation_platt, test["clicked"], test_platt
    )

    results = {
        "population": "won_impressions_only",
        "split": {key: list(value) for key, value in SEASON2_SPLIT.items()},
        "calibration_fit_split": "validation",
        "test_split_used_for_calibration": False,
        "features": features,
        "base_model": {
            "type": "logistic_regression",
            "solver": "saga",
            "max_iter": config.max_iter,
            "tol": config.tol,
            "iterations": iterations,
            "stopped_before_max_iter": iterations < config.max_iter,
            "warning_free_fit": len(convergence_warnings) == 0,
            "convergence_warnings": convergence_warnings,
        },
        "methods": {
            "raw": raw_metrics,
            "intercept_only": {
                "parameters": {"delta_log_odds": intercept.delta},
                **intercept_metrics,
            },
            "platt": {
                "parameters": {
                    "slope": platt.slope,
                    "intercept": platt.intercept,
                },
                **platt_metrics,
            },
        },
        "ranking_invariance": {
            "intercept_only_test_roc_auc_delta": (
                intercept_metrics["test"]["roc_auc"] - raw_metrics["test"]["roc_auc"]
            ),
            "intercept_only_test_pr_auc_delta": (
                intercept_metrics["test"]["pr_auc"] - raw_metrics["test"]["pr_auc"]
            ),
            "platt_test_roc_auc_delta": (
                platt_metrics["test"]["roc_auc"] - raw_metrics["test"]["roc_auc"]
            ),
            "platt_test_pr_auc_delta": (
                platt_metrics["test"]["pr_auc"] - raw_metrics["test"]["pr_auc"]
            ),
        },
    }

    advertiser_frames = []
    for method, probabilities in (
        ("raw", test_raw),
        ("intercept_only", test_intercept),
        ("platt", test_platt),
    ):
        metrics = evaluate_by_advertiser(test, probabilities, min_rows=1000)
        metrics.insert(0, "calibration_method", method)
        advertiser_frames.append(metrics)

    advertiser_results = pd.concat(advertiser_frames, ignore_index=True)
    return results, advertiser_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare raw, intercept-only, and Platt-calibrated Season 2 pCTR."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/ipinyou.contest.dataset"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("artifacts/results/pctr_calibration_season2.json"),
    )
    parser.add_argument(
        "--advertiser-results",
        type=Path,
        default=Path("artifacts/results/pctr_calibration_season2_advertisers.csv"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results, advertiser_results = run_season2_calibration(args.dataset_root)

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    args.advertiser_results.parent.mkdir(parents=True, exist_ok=True)
    advertiser_results.to_csv(args.advertiser_results, index=False)

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

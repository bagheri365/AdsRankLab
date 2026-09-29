"""Tie-aware robustness analysis for value-aware ranking policies."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning

from ads_rank_lab.evaluation.tie_aware import (
    normalized_rank_shift_summary,
    tie_aware_cutoff,
    tie_aware_top_k_comparison,
)
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
from ads_rank_lab.policies.value import build_policy_scores


POLICY_ORDER = (
    "bid_only",
    "pctr_raw",
    "bid_x_raw_pctr",
    "bid_x_intercept_pctr",
    "bid_x_platt_pctr",
)

TOP_FRACTIONS = (0.001, 0.01, 0.05)


def run_season2_ranking_robustness(
    dataset_root: str | Path,
    *,
    config: PctrBaselineConfig | None = None,
) -> dict:
    """Measure tie-aware ranking robustness on the held-out June 12 population."""
    config = config or PctrBaselineConfig()

    train = load_dates(dataset_root, season=2, dates=SEASON2_SPLIT["train"])
    validation = load_dates(
        dataset_root,
        season=2,
        dates=SEASON2_SPLIT["validation"],
    )
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

    scores = build_policy_scores(
        test,
        raw_pctr=test_raw,
        intercept_pctr=intercept.predict(test_raw),
        platt_pctr=platt.predict(test_raw),
    )

    cutoffs = {
        policy: [
            tie_aware_cutoff(scores[policy], fraction=fraction)
            for fraction in TOP_FRACTIONS
        ]
        for policy in POLICY_ORDER
    }

    reference = "bid_x_raw_pctr"
    comparisons = {}
    for candidate in (
        "bid_only",
        "pctr_raw",
        "bid_x_intercept_pctr",
        "bid_x_platt_pctr",
    ):
        comparisons[candidate] = {
            "rank_shift_vs_raw_value": normalized_rank_shift_summary(
                scores[reference],
                scores[candidate],
            ),
            "tie_aware_top_k_vs_raw_value": [
                tie_aware_top_k_comparison(
                    scores[reference],
                    scores[candidate],
                    fraction=fraction,
                )
                for fraction in TOP_FRACTIONS
            ],
        }

    fitted_model = pipeline.named_steps["model"]
    iterations = int(fitted_model.n_iter_.max())

    return {
        "population": "held_out_won_impressions_only",
        "evaluation_scope": (
            "tie-aware ranking robustness on historically observed won "
            "impressions; not counterfactual auction outcomes"
        ),
        "split": {key: list(value) for key, value in SEASON2_SPLIT.items()},
        "calibration_fit_split": "validation",
        "test_split_used_for_calibration": False,
        "test_rows": int(len(test)),
        "base_model": {
            "iterations": iterations,
            "max_iter": config.max_iter,
            "tol": config.tol,
            "stopped_before_max_iter": iterations < config.max_iter,
            "warning_free_fit": len(convergence_warnings) == 0,
            "convergence_warnings": convergence_warnings,
        },
        "tie_aware_cutoffs": cutoffs,
        "comparisons": comparisons,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run tie-aware robustness analysis for Season 2 rankings."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/ipinyou.contest.dataset"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("artifacts/results/ranking_robustness_season2.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = run_season2_ranking_robustness(args.dataset_root)

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

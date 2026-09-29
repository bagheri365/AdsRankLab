"""Value-aware ranking sensitivity study on the held-out Season 2 test day."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from ads_rank_lab.evaluation.ranking import (
    advertiser_top_share,
    rank_shift_summary,
    score_margin_summary,
    top_k_overlap,
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


def run_season2_value_ranking(
    dataset_root: str | Path,
    *,
    config: PctrBaselineConfig | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Compare value-aware rankings on the untouched June 12 won impressions.

    This is a ranking-sensitivity experiment, not a counterfactual auction
    simulator. All policies are evaluated on the same historically observed
    won-impression population.
    """
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

    test_intercept = intercept.predict(test_raw)
    test_platt = platt.predict(test_raw)

    scores = build_policy_scores(
        test,
        raw_pctr=test_raw,
        intercept_pctr=test_intercept,
        platt_pctr=test_platt,
    )

    margins = {
        policy: score_margin_summary(scores[policy], top_fraction=0.01)
        for policy in POLICY_ORDER
    }

    comparisons = {}
    reference = "bid_x_raw_pctr"
    for candidate in ("bid_x_intercept_pctr", "bid_x_platt_pctr"):
        comparisons[candidate] = {
            "rank_shift_vs_raw_value": rank_shift_summary(
                scores[reference],
                scores[candidate],
            ),
            "top_k_overlap_vs_raw_value": [
                top_k_overlap(
                    scores[reference],
                    scores[candidate],
                    fraction=fraction,
                )
                for fraction in TOP_FRACTIONS
            ],
        }

    # Broader comparison shows how much the economic value proxy differs from
    # pure bid and pure response ranking.
    broad_comparisons = {}
    for candidate in ("bid_only", "pctr_raw"):
        broad_comparisons[candidate] = {
            "rank_shift_vs_raw_value": rank_shift_summary(
                scores[reference],
                scores[candidate],
            ),
            "top_k_overlap_vs_raw_value": [
                top_k_overlap(
                    scores[reference],
                    scores[candidate],
                    fraction=fraction,
                )
                for fraction in TOP_FRACTIONS
            ],
        }

    fitted_model = pipeline.named_steps["model"]
    iterations = int(fitted_model.n_iter_.max())

    results = {
        "population": "held_out_won_impressions_only",
        "evaluation_scope": (
            "ranking sensitivity on historically observed won impressions; "
            "not counterfactual auction outcomes"
        ),
        "split": {key: list(value) for key, value in SEASON2_SPLIT.items()},
        "calibration_fit_split": "validation",
        "test_split_used_for_calibration": False,
        "test_rows": int(len(test)),
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
        "calibration_parameters": {
            "intercept_only_delta_log_odds": intercept.delta,
            "platt_slope": platt.slope,
            "platt_intercept": platt.intercept,
        },
        "policies": {
            "bid_only": "logged bid price",
            "pctr_raw": "raw predicted CTR",
            "bid_x_raw_pctr": "bid price × raw pCTR",
            "bid_x_intercept_pctr": "bid price × intercept-calibrated pCTR",
            "bid_x_platt_pctr": "bid price × Platt-calibrated pCTR",
        },
        "score_margins_at_top_1pct": margins,
        "calibration_value_comparisons": comparisons,
        "bid_and_response_comparisons": broad_comparisons,
    }

    advertiser_frames = [
        advertiser_top_share(
            test.reset_index(drop=True),
            scores[policy].to_numpy(),
            policy=policy,
            top_fraction=0.01,
        )
        for policy in POLICY_ORDER
    ]
    advertiser_results = pd.concat(advertiser_frames, ignore_index=True)

    return results, advertiser_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare bid-only, pCTR-only, and calibrated value-aware rankings "
            "on held-out Season 2 won impressions."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/ipinyou.contest.dataset"),
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("artifacts/results/value_ranking_season2.json"),
    )
    parser.add_argument(
        "--advertiser-results",
        type=Path,
        default=Path("artifacts/results/value_ranking_season2_advertisers.csv"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results, advertiser_results = run_season2_value_ranking(args.dataset_root)

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

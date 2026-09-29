"""Chronological Season 2 logistic-regression pCTR baseline."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from ads_rank_lab.data.pctr import load_won_impression_day
from ads_rank_lab.evaluation.pctr import evaluate_by_advertiser, evaluate_pctr
from ads_rank_lab.models.pctr import PctrBaselineConfig, build_pctr_pipeline, feature_columns, validate_feature_frame


SEASON2_SPLIT = {
    "train": ("20130606", "20130607", "20130608", "20130609", "20130610"),
    "validation": ("20130611",),
    "test": ("20130612",),
}


def load_dates(dataset_root: str | Path, *, season: int, dates: tuple[str, ...]) -> pd.DataFrame:
    return pd.concat(
        [load_won_impression_day(dataset_root, season=season, date=d) for d in dates],
        ignore_index=True,
    )


def run_season2_baseline(dataset_root: str | Path, *, config: PctrBaselineConfig | None = None) -> tuple[dict, pd.DataFrame]:
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

    train_prob = pipeline.predict_proba(train[features])[:, 1]
    validation_prob = pipeline.predict_proba(validation[features])[:, 1]
    test_prob = pipeline.predict_proba(test[features])[:, 1]

    fitted_model = pipeline.named_steps["model"]
    iterations = int(fitted_model.n_iter_.max())
    stopped_before_max_iter = iterations < config.max_iter
    warning_free_fit = len(convergence_warnings) == 0

    results = {
        "model": {
            "type": "logistic_regression",
            "solver": "saga",
            "max_iter": config.max_iter,
            "tol": config.tol,
            "iterations": iterations,
            "stopped_before_max_iter": stopped_before_max_iter,
            "warning_free_fit": warning_free_fit,
            "convergence_warnings": convergence_warnings,
            "numeric_scaling": "StandardScaler(with_mean=False)",
        },
        "population": "won_impressions_only",
        "split": {k: list(v) for k, v in SEASON2_SPLIT.items()},
        "features": features,
        "train": evaluate_pctr(train["clicked"], train_prob),
        "validation": evaluate_pctr(validation["clicked"], validation_prob),
        "test": evaluate_pctr(test["clicked"], test_prob),
    }
    return results, evaluate_by_advertiser(test, test_prob, min_rows=1000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the chronological Season 2 pCTR logistic baseline.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/ipinyou.contest.dataset"))
    parser.add_argument("--results", type=Path, default=Path("artifacts/results/pctr_baseline_season2.json"))
    parser.add_argument("--advertiser-results", type=Path, default=Path("artifacts/results/pctr_baseline_season2_advertisers.csv"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results, advertiser_metrics = run_season2_baseline(args.dataset_root)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.advertiser_results.parent.mkdir(parents=True, exist_ok=True)
    advertiser_metrics.to_csv(args.advertiser_results, index=False)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

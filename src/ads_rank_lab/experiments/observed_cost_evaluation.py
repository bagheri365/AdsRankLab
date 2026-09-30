"""Observed-cost evaluation for held-out Season 2 won impressions."""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from ads_rank_lab.data.analytical import build_training_day_table
from ads_rank_lab.evaluation.observed_cost import advertiser_observed_cost, summarize_observed_cost, valid_observed_cost_population
from ads_rank_lab.experiments.pctr_baseline import SEASON2_SPLIT, load_dates
from ads_rank_lab.models.calibration import fit_intercept_calibrator, fit_platt_calibrator
from ads_rank_lab.models.pctr import PctrBaselineConfig, build_pctr_pipeline, feature_columns, validate_feature_frame
from ads_rank_lab.policies.value import build_policy_scores

POLICY_ORDER = ("bid_only","pctr_raw","bid_x_raw_pctr","bid_x_intercept_pctr","bid_x_platt_pctr")
TOP_FRACTIONS = (0.001,0.01,0.05)

def run_season2_observed_cost_evaluation(dataset_root: str | Path, *, config: PctrBaselineConfig | None = None):
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
    convergence_warnings = [str(x.message) for x in caught if issubclass(x.category, ConvergenceWarning)]
    validation_raw = pipeline.predict_proba(validation[features])[:,1]
    intercept = fit_intercept_calibrator(validation["clicked"], validation_raw)
    platt = fit_platt_calibrator(validation["clicked"], validation_raw)
    test_raw_all = pipeline.predict_proba(test[features])[:,1]
    test_intercept_all = intercept.predict(test_raw_all)
    test_platt_all = platt.predict(test_raw_all)

    # The pCTR population intentionally excludes post-auction fields such as
    # paying price. Reattach observed test-day cost by bid_id from the frozen
    # analytical table instead of weakening the leakage-safe modeling loader.
    cost_frames = []
    for date in SEASON2_SPLIT["test"]:
        analytical, _ = build_training_day_table(
            dataset_root,
            season=2,
            date=date,
        )
        cost_frames.append(
            analytical.loc[
                analytical["won"].astype(bool),
                ["bid_id", "observed_paying_price", "paying_price_conflict"],
            ]
        )
    observed_costs = pd.concat(cost_frames, ignore_index=True)
    observed_costs["bid_id"] = observed_costs["bid_id"].astype("string")
    observed_costs = observed_costs.drop_duplicates("bid_id", keep="first")

    scored_test = test.copy().reset_index(drop=True)
    scored_test["bid_id"] = scored_test["bid_id"].astype("string")
    scored_test["_raw_pctr"] = test_raw_all
    scored_test["_intercept_pctr"] = test_intercept_all
    scored_test["_platt_pctr"] = test_platt_all
    scored_test = scored_test.merge(
        observed_costs,
        on="bid_id",
        how="left",
        validate="one_to_one",
    )

    test_cost = valid_observed_cost_population(scored_test)
    raw = test_cost.pop("_raw_pctr").to_numpy(dtype=float)
    intercept_prob = test_cost.pop("_intercept_pctr").to_numpy(dtype=float)
    platt_prob = test_cost.pop("_platt_pctr").to_numpy(dtype=float)

    scores = build_policy_scores(test_cost, raw_pctr=raw, intercept_pctr=intercept_prob, platt_pctr=platt_prob)
    policy_results = {}
    advertiser_frames = []
    for policy in POLICY_ORDER:
        policy_results[policy] = {
            f"top_{fraction:g}": summarize_observed_cost(
                test_cost, scores=scores[policy].to_numpy(), fraction=fraction
            )
            for fraction in TOP_FRACTIONS
        }
        advertiser_frames.append(
            advertiser_observed_cost(
                test_cost, scores=scores[policy].to_numpy(), policy=policy, fraction=0.01
            )
        )

    fitted = pipeline.named_steps["model"]
    iterations = int(fitted.n_iter_.max())
    results = {
        "population": "held_out_won_impressions_with_observed_conflict_free_cost",
        "evaluation_scope": "descriptive observed-cost comparison on historical wins only; not counterfactual auction evaluation",
        "split": {k:list(v) for k,v in SEASON2_SPLIT.items()},
        "calibration_fit_split": "validation",
        "test_split_used_for_calibration": False,
        "test_rows_before_cost_filter": int(len(test)),
        "test_rows_with_observed_cost": int(len(test_cost)),
        "test_rows_without_joined_observed_cost": int(
            scored_test["observed_paying_price"].isna().sum()
        ),
        "excluded_test_rows_missing_or_conflicting_cost": int(len(test)-len(test_cost)),
        "base_model": {
            "iterations": iterations,
            "max_iter": config.max_iter,
            "tol": config.tol,
            "stopped_before_max_iter": iterations < config.max_iter,
            "warning_free_fit": len(convergence_warnings) == 0,
            "convergence_warnings": convergence_warnings,
        },
        "observed_cost_metrics": policy_results,
        "limitations": [
            "Paying price is observed only for historically won impressions.",
            "Lost bids are not assigned synthetic clearing prices or response outcomes.",
            "Policy-selected slices are descriptive subsets of historical wins, not simulated auction results.",
            "Observed cost-per-click is not a causal estimate of policy performance.",
        ],
    }
    return results, pd.concat(advertiser_frames, ignore_index=True)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/ipinyou.contest.dataset"))
    parser.add_argument("--results", type=Path, default=Path("artifacts/results/observed_cost_season2.json"))
    parser.add_argument("--advertiser-results", type=Path, default=Path("artifacts/results/observed_cost_season2_advertisers.csv"))
    args = parser.parse_args()
    results, advertiser_results = run_season2_observed_cost_evaluation(args.dataset_root)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    args.advertiser_results.parent.mkdir(parents=True, exist_ok=True)
    advertiser_results.to_csv(args.advertiser_results, index=False)
    print(json.dumps(results, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()

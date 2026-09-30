# Observed-cost evaluation

This experiment extends the value-ranking study with historical cost diagnostics while staying only where auction cost is observed.

## Population

The held-out Season 2 test day (`20130612`) is filtered to historically won impressions with an observed, conflict-free paying price, valid logged bid price, and observed click label. No clearing price is imputed for lost bids.

## Metrics

For bid-only, raw pCTR, raw bid × pCTR, intercept-calibrated bid × pCTR, and Platt-calibrated bid × pCTR, the experiment inspects the top 0.1%, 1%, and 5% and reports observed total/mean/median/p95 paying price, logged bid totals, bid-to-pay ratios, observed CTR, observed cost per click, and clicks per 1,000 cost units.

A separate CSV reports advertiser-level cost composition for the top 1%.

## Interpretation limits

These are descriptive properties of policy-selected subsets of historical wins, not counterfactual auction outcomes. Paying price and response are not fabricated for lost bids, and observed cost per click is not a causal policy-performance estimate.

## Run

```bash
python -m ads_rank_lab.experiments.observed_cost_evaluation
```

Outputs:

```text
artifacts/results/observed_cost_season2.json
artifacts/results/observed_cost_season2_advertisers.csv
```

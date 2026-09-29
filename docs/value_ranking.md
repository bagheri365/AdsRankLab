# Value-aware ranking sensitivity

This experiment studies whether probability calibration changes economic
ordering even when pCTR ranking metrics do not change.

## Evaluation population

All comparisons use the held-out Season 2 test day (`20130612`) and only the
historically observed won-impression population used by the pCTR experiment.

This is **not** a counterfactual auction simulator. The experiment does not
claim to know clearing prices, clicks, or outcomes for bids that did not win.

## Compared scores

- `bid_only`: logged bid price;
- `pctr_raw`: raw predicted CTR;
- `bid_x_raw_pctr`: bid × raw pCTR;
- `bid_x_intercept_pctr`: bid × validation-fitted intercept-calibrated pCTR;
- `bid_x_platt_pctr`: bid × validation-fitted Platt-calibrated pCTR.

The bid × pCTR scores are value proxies used for ranking sensitivity analysis.
They are not realized profit or advertiser utility.

## Questions

The experiment asks:

1. How different is value-aware ordering from bid-only and pCTR-only ordering?
2. Can a monotonic calibration transform leave ROC-AUC/PR-AUC unchanged while
   changing the ordering of `bid × pCTR`?
3. How many items move rank after calibration?
4. How stable are the top 0.1%, 1%, and 5% selections?
5. How narrow is the score margin around the top-1% decision boundary?
6. Does advertiser representation in the top 1% change across policies?

## Metrics

For calibrated value rankings relative to raw `bid × pCTR`:

- fraction of rows whose rank changes;
- median, mean, p95, and maximum absolute rank shift;
- adjacent-order flip rate among non-tied raw-value neighbors;
- top-k overlap and replacement rates at 0.1%, 1%, and 5%.

Each policy also reports its top-1% boundary score and margin. A separate CSV
shows advertiser population share, selected share, and representation ratio in
the top 1%.

## Run

```bash
python -m ads_rank_lab.experiments.value_ranking
```

Outputs:

```text
artifacts/results/value_ranking_season2.json
artifacts/results/value_ranking_season2_advertisers.csv
```

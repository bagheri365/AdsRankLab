# pCTR baseline protocol

Milestone 2 begins with a transparent logistic-regression baseline.

## Modeling population

The initial supervised population contains **won impressions only**. Click
outcomes are observed for served impressions under the logged policy; lost bids
are not automatically treated as known negative click outcomes.

## Chronological Season 2 split

- train: 2013-06-06 through 2013-06-10;
- validation: 2013-06-11;
- test: 2013-06-12.

No random row split is used.

## Baseline features

Numeric: hour, slot width, slot height, slot price.

Categorical: weekday, region, city, ad exchange, slot visibility, slot format,
advertiser ID.

Higher-cardinality features such as domain, creative, user ID, URL, and user
tags are deferred until the baseline is established.

## Explicitly excluded post-auction information

The model does not use paying price, win status, impression count, click count,
conversion outcome, log type, key page, or features created from future event
streams.

## Metrics

ROC-AUC, PR-AUC, log loss, Brier score, expected calibration error, observed
CTR, average predicted CTR, and advertiser-level prevalence/calibration.

## Run

```bash
python -m ads_rank_lab.experiments.pctr_baseline
```


## Convergence

The logistic baseline records the fitted iteration count and whether the
optimizer stopped before `max_iter`. Numeric variables are standardized with
`StandardScaler(with_mean=False)` and the default iteration cap is 1000.

A run that reaches `max_iter` is not considered the frozen baseline result even
if its predictive metrics are otherwise usable.

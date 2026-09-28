# Frozen analytical-table protocol

AdsRankLab constructs a bid-level analytical table from the original iPinYou
bid, impression, click, and conversion streams.

## Row universe

Every row represents an observed **bid request**.

## Cleaning rules

1. A bid is marked `won=True` only when an impression event exists.
2. Duplicate impression rows are aggregated by `bid_id`.
3. Duplicate impressions that agree on paying price preserve that value.
4. Duplicate impressions that disagree on paying price are flagged with
   `paying_price_conflict=True`, and `observed_paying_price` is left missing.
5. A click is used as a positive label only when the same `bid_id` also has an
   observed impression.
6. A conversion is used as a positive label only when the same `bid_id` also
   has an observed impression.
7. Orphan click/conversion events remain visible in diagnostics and are not
   silently reinterpreted as wins.

## Why orphan clicks are excluded from positive labels

The Season 2 raw logs contain a very small number of click events whose `bid_id`
does not appear in the same day's impression stream. These records have no
supporting impression evidence, so AdsRankLab does not use them to overturn the
definition of a win.

The project's bundled known-bugs file documents a separate Season 1 conversion
log-type issue, not these Season 2 orphan clicks. Therefore the conservative
treatment is to report them as anomalous event evidence rather than infer a
missing impression.

## Observational semantics

`response_observed`, `click_label_observed`, and
`conversion_label_observed` are true only for won impressions.

A lost bid is therefore not treated as a fully observed negative response for
counterfactual policy evaluation. Later pCTR work must explicitly define the
population on which supervised labels are valid.

## Frozen daily build

```bash
python -m ads_rank_lab.experiments.build_analytical_table   --season 2   --date 20130606
```

Optional parquet/diagnostic artifacts can be requested with `--output` and
`--diagnostics`.

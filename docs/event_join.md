# Bid-level observed-outcome table

The original iPinYou training release stores bidding, impression, click, and
conversion evidence in separate event streams. AdsRankLab therefore constructs
a bid-level table explicitly rather than treating any one event file as a fully
labeled training dataset.

## Join key

Events are related using `bid_id`.

The bid stream is the anchor because it represents the decision opportunities
available to the logged bidding policy.

## Outcome semantics

For each observed bid request:

- `won = True` when at least one matching impression event exists;
- `impression_count` records how many matching impression rows exist;
- `click_count` is the number of matching click events;
- `clicked = True` when at least one click event exists;
- `converted = True` when at least one conversion event exists;
- `observed_paying_price` is retained when duplicate impression rows agree;
- `paying_price_conflict = True` when duplicate rows disagree on paying price.

Multiple clicks are counted rather than collapsed during ingestion because the
dataset documentation permits more than one click to be related to an
impression.

## Counterfactual boundary

A lost bid has **no observed paying price** in this table. AdsRankLab leaves
`observed_paying_price` missing for lost bids rather than setting it to zero or
treating another auction's market price as ground truth.

Likewise, `clicked = False` for a lost bid means **no click event was observed
under the logged policy**. It does not mean the user would not have clicked had
the bid won.

This distinction must be preserved in later pCTR and policy-evaluation work.

## Diagnostics

The join reports:

- impressions with no matching bid;
- clicked bid IDs with no impression;
- converted bid IDs with no impression.

The default mode reports these inconsistencies without deleting data. Strict
mode can be used when a frozen experiment requires rejecting inconsistent input.

## Duplicate impression rows

The released logs contain some repeated `bid_id` values in impression streams.
AdsRankLab therefore aggregates impression evidence at bid level instead of
assuming one raw impression row per bid ID.

When duplicate rows agree on paying price, the shared value is retained. When
they disagree, the analytical table leaves `observed_paying_price` missing and
sets `paying_price_conflict = True`. Strict mode rejects such conflicts.

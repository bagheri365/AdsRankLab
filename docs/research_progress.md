# Research progress

## Milestone 1 — Data pipeline and auction analysis

Status: **in progress**

### Implemented

- [x] Python package scaffold
- [x] Raw/processed data conventions
- [x] Tabular loading helper
- [x] Canonical iPinYou schema contract
- [x] Raw-to-canonical column aliases
- [x] Schema validation and safe type coercion
- [x] Explicit observed-versus-unobserved data documentation
- [x] Deterministic split utility
- [x] Basic missing/categorical preprocessing helpers
- [x] Descriptive dataset-audit library
- [x] JSON dataset-audit CLI
- [x] Original Season 2/3 `.bz2` event-log ingestion
- [x] Exact bid vs. outcome-event schema contracts
- [x] Deterministic event-log discovery by date
- [x] Leakage-aware bid/impression/click/conversion join
- [x] Event-consistency diagnostics and multi-click aggregation
- [x] Frozen daily analytical-table construction
- [x] Orphan-click/conversion cleaning protocol
- [x] Unit tests for core Milestone 1 utilities

### Milestone 2 — pCTR baseline

- [x] Define won-impression supervised population
- [x] Freeze chronological Season 2 split
- [x] Implement leakage-safe logistic-regression baseline
- [x] Add discrimination, loss, Brier, and calibration metrics
- [x] Add advertiser-level evaluation summaries
- [ ] Run the full Season 2 baseline and preserve result artifacts

### Next

- [ ] Add exact iPinYou season/file setup instructions after local dataset inspection
- [x] Verify Season 2/3 raw schemas against the downloaded release
- [x] Inspect known-data-bug notes before freezing analytical tables
- [x] Define and test the `bid_id` event-join protocol
- [ ] Run and preserve the first real-data audit artifact
- [ ] Produce advertiser/traffic-volume summaries from the real release
- [ ] Freeze the Milestone 1 train/validation/test split protocol

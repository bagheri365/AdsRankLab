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
- [x] Unit tests for core Milestone 1 utilities

### Next

- [ ] Add exact iPinYou season/file setup instructions after local dataset inspection
- [ ] Verify raw schemas against the downloaded release
- [ ] Expand canonical mapping only where supported by the actual files
- [ ] Run and preserve the first real-data audit artifact
- [ ] Produce advertiser/traffic-volume summaries from the real release
- [ ] Freeze the Milestone 1 train/validation/test split protocol

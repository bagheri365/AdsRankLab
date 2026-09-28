# AdsRankLab

AdsRankLab is an experimental advertising-ranking project for studying how predicted user response, advertiser bids, and economic value interact in real-time bidding (RTB) decisions.

The project is intentionally narrower than a production ad platform. It focuses on response prediction, probability calibration, value-aware scoring, auction outcomes, and robustness of decision policies using real RTB logs.

## Research question

**How should an advertising system combine predicted user response and economic value when making auction-time decisions, and when do prediction or calibration errors lead to worse allocation outcomes?**

## Current status

Milestone 1 foundation:

- reproducible Python package;
- raw/processed data conventions;
- schema-oriented loading helpers;
- canonical iPinYou schema validation;
- deterministic train/validation/test splitting;
- descriptive dataset-audit CLI;
- basic preprocessing utilities;
- initial tests.

The repository does **not** include the iPinYou dataset. See `data/README.md` for expected local layout.

Raw Season 2/3 contest logs can be read directly from their original `.txt.bz2` files; see `docs/raw_ingestion.md`.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```

## Dataset audit

After placing a real RTB log under `data/raw/`, create a descriptive JSON audit with:

```bash
adsrank-audit data/raw/<file> --output artifacts/results/dataset_audit.json
```

The audit reports schema coverage, missingness, click/conversion prevalence, bid/pay/floor summaries, and selected segment cardinalities. It is descriptive only and does not infer unobserved counterfactual outcomes.

## Project structure

```text
AdsRankLab/
├── README.md
├── pyproject.toml
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
├── docs/
│   ├── dataset_notes.md
│   └── research_progress.md
├── artifacts/
│   ├── figures/
│   └── results/
├── src/
│   └── ads_rank_lab/
│       ├── data/
│       ├── features/
│       ├── models/
│       ├── policies/
│       ├── evaluation/
│       └── experiments/
└── tests/
```

## Scope

AdsRankLab studies a defensible subset of RTB decision-making. It does not claim to reconstruct a complete modern commercial ad auction, because public RTB logs do not expose every competing candidate, competitor bid, proprietary quality signal, landing-page quality signal, or true advertiser conversion value.

# Data

AdsRankLab is designed around the **iPinYou Real-Time Bidding dataset**, preferably Seasons 2/3 where richer advertiser and user-profile fields are available.

The dataset is not redistributed in this repository.

## Local layout

Place original downloaded files under:

```text
data/raw/
```

Reproducible derived files should be written under:

```text
data/processed/
```

Both directories are ignored by Git except for `.gitkeep` placeholders.

## Milestone 1 data goals

The first milestone should establish:

- dataset setup/download instructions;
- schema documentation;
- deterministic train/validation/test splits;
- missing-value and categorical-feature handling;
- CTR and conversion prevalence;
- bid, floor-price, and paying-price distributions;
- advertiser and traffic-volume summaries;
- explicit documentation of observed versus unobserved information.

Do not treat lost-auction outcomes or other missing counterfactual outcomes as observed data.

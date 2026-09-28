"""Construction of the observed pCTR modeling population."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ads_rank_lab.data.ipinyou import read_training_day
from ads_rank_lab.models.pctr import add_time_features


def build_won_impression_population(impressions: pd.DataFrame, clicks: pd.DataFrame) -> pd.DataFrame:
    """Build one-row-per-won-impression data with an observed click label."""
    if "bid_id" not in impressions.columns:
        raise ValueError("impressions must contain bid_id")
    if "bid_id" not in clicks.columns:
        raise ValueError("clicks must contain bid_id")

    base = impressions.loc[impressions["bid_id"].notna()].copy()
    base["bid_id"] = base["bid_id"].astype("string")
    base = base.drop_duplicates(subset=["bid_id"], keep="first").reset_index(drop=True)

    clicked_ids = set(clicks["bid_id"].dropna().astype("string"))
    base["clicked"] = base["bid_id"].isin(clicked_ids).astype("int8")

    drop_post_auction = [
        c for c in ("log_type", "paying_price", "key_page") if c in base.columns
    ]
    base = base.drop(columns=drop_post_auction)
    return add_time_features(base)


def load_won_impression_day(dataset_root: str | Path, *, season: int, date: str) -> pd.DataFrame:
    impressions = read_training_day(dataset_root, season=season, event_type="imp", date=date)
    clicks = read_training_day(dataset_root, season=season, event_type="clk", date=date)
    return build_won_impression_population(impressions, clicks)

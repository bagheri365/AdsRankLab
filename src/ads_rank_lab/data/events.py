"""Leakage-aware joins for iPinYou bid and outcome event streams."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EventJoinDiagnostics:
    """Diagnostics describing evidence consistency across event streams."""

    bid_rows: int
    unique_bids: int
    impression_rows: int
    unique_impressions: int
    duplicate_impression_bid_ids: int
    conflicting_paying_price_bid_ids: int
    click_rows: int
    unique_clicked_bids: int
    conversion_rows: int
    unique_converted_bids: int
    impressions_without_bid: int
    clicks_without_impression: int
    conversions_without_impression: int


def _require_columns(frame: pd.DataFrame, required: set[str], *, name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _require_unique_bid_ids(frame: pd.DataFrame, *, name: str) -> None:
    duplicated = frame["bid_id"].dropna().duplicated(keep=False)
    if duplicated.any():
        examples = (
            frame.loc[frame["bid_id"].notna() & frame["bid_id"].duplicated(keep=False), "bid_id"]
            .astype("string")
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        raise ValueError(
            f"{name} must contain at most one row per bid_id; "
            f"duplicate examples: {examples}"
        )


def _count_by_bid_id(frame: pd.DataFrame, *, name: str) -> pd.Series:
    _require_columns(frame, {"bid_id"}, name=name)
    return frame["bid_id"].dropna().astype("string").value_counts()



def _aggregate_impressions(
    impressions: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int]:
    """Reduce possibly duplicated impression rows to bid-level evidence.

    Duplicate impression rows occur in the released logs. They are preserved as
    an ``impression_count`` diagnostic. Paying price is retained only when all
    non-missing observations for a bid agree; conflicting prices remain missing
    in the joined analytical table rather than being arbitrarily selected.
    """
    _require_columns(impressions, {"bid_id", "paying_price"}, name="impressions")

    working = impressions[["bid_id", "paying_price"]].copy()
    working = working.loc[working["bid_id"].notna()].copy()
    working["bid_id"] = working["bid_id"].astype("string")

    grouped = working.groupby("bid_id", sort=False, dropna=False)
    counts = grouped.size().rename("impression_count")
    price_nunique = grouped["paying_price"].nunique(dropna=True)

    duplicate_ids = int((counts > 1).sum())
    conflicting_ids = int((price_nunique > 1).sum())

    def agreed_price(series: pd.Series):
        observed = series.dropna()
        if observed.empty:
            return pd.NA
        unique = observed.astype("string").drop_duplicates()
        if len(unique) == 1:
            return observed.iloc[0]
        return pd.NA

    prices = grouped["paying_price"].apply(agreed_price).rename(
        "observed_paying_price"
    )
    evidence = pd.concat([counts, prices], axis=1).reset_index()
    conflict_ids = set(price_nunique[price_nunique > 1].index.astype("string"))
    evidence["paying_price_conflict"] = evidence["bid_id"].astype("string").isin(
        conflict_ids
    )
    return evidence, duplicate_ids, conflicting_ids

def build_observed_outcomes(
    bids: pd.DataFrame,
    impressions: pd.DataFrame,
    clicks: pd.DataFrame,
    conversions: pd.DataFrame,
    *,
    strict: bool = False,
) -> tuple[pd.DataFrame, EventJoinDiagnostics]:
    """Join separate iPinYou event streams into one bid-level observed table.

    The returned table is anchored on **bid requests**. Outcome columns mean:

    - ``won``: at least one matching impression is observed;
    - ``impression_count``: number of matching impression rows;
    - ``click_count``: number of click events observed for the bid;
    - ``clicked``: at least one click event is observed;
    - ``converted``: at least one conversion event is observed;
    - ``observed_paying_price``: retained only when duplicate impression rows
      agree on the non-missing paying price;
    - ``paying_price_conflict``: duplicate impression evidence disagrees.

    No market price is imputed for lost bids, and no counterfactual click or
    conversion outcome is inferred for bids that did not win.

    Click logs may legitimately contain multiple rows for one impression, so
    clicks are aggregated rather than forced to be unique. Conversion is
    reduced to a binary observed indicator.
    """
    _require_columns(bids, {"bid_id"}, name="bids")
    _require_columns(impressions, {"bid_id", "paying_price"}, name="impressions")
    _require_columns(clicks, {"bid_id"}, name="clicks")
    _require_columns(conversions, {"bid_id"}, name="conversions")

    _require_unique_bid_ids(bids, name="bids")
    bid_ids = set(bids["bid_id"].dropna().astype("string"))

    impression_evidence, duplicate_impression_ids, conflicting_price_ids = (
        _aggregate_impressions(impressions)
    )
    impression_ids = set(impression_evidence["bid_id"].astype("string"))

    click_counts = _count_by_bid_id(clicks, name="clicks")
    conversion_counts = _count_by_bid_id(conversions, name="conversions")
    clicked_ids = set(click_counts.index)
    converted_ids = set(conversion_counts.index)

    diagnostics = EventJoinDiagnostics(
        bid_rows=int(len(bids)),
        unique_bids=int(len(bid_ids)),
        impression_rows=int(len(impressions)),
        unique_impressions=int(len(impression_ids)),
        duplicate_impression_bid_ids=duplicate_impression_ids,
        conflicting_paying_price_bid_ids=conflicting_price_ids,
        click_rows=int(len(clicks)),
        unique_clicked_bids=int(len(clicked_ids)),
        conversion_rows=int(len(conversions)),
        unique_converted_bids=int(len(converted_ids)),
        impressions_without_bid=int(len(impression_ids - bid_ids)),
        clicks_without_impression=int(len(clicked_ids - impression_ids)),
        conversions_without_impression=int(len(converted_ids - impression_ids)),
    )

    if strict:
        problems = []
        if diagnostics.impressions_without_bid:
            problems.append(
                f"{diagnostics.impressions_without_bid} impressions have no matching bid"
            )
        if diagnostics.clicks_without_impression:
            problems.append(
                f"{diagnostics.clicks_without_impression} clicked bids have no impression"
            )
        if diagnostics.conversions_without_impression:
            problems.append(
                f"{diagnostics.conversions_without_impression} converted bids have no impression"
            )
        if diagnostics.conflicting_paying_price_bid_ids:
            problems.append(
                f"{diagnostics.conflicting_paying_price_bid_ids} impression groups "
                "have conflicting paying prices"
            )
        if problems:
            raise ValueError("Event evidence is inconsistent: " + "; ".join(problems))

    result = bids.copy()
    result["bid_id"] = result["bid_id"].astype("string")

    result = result.merge(
        impression_evidence,
        on="bid_id",
        how="left",
        validate="one_to_one",
    )

    result["won"] = result["bid_id"].isin(impression_ids)
    result["impression_count"] = result["impression_count"].fillna(0).astype("int64")
    result["paying_price_conflict"] = result["paying_price_conflict"].eq(True)
    result["click_count"] = result["bid_id"].map(click_counts).fillna(0).astype("int64")
    result["clicked"] = result["click_count"] > 0
    result["converted"] = result["bid_id"].isin(converted_ids)

    # Paying price is evidence from a won impression only. Keep lost-bid values
    # missing rather than filling them with zero or a modeled market price.
    result.loc[~result["won"], "observed_paying_price"] = pd.NA

    return result, diagnostics

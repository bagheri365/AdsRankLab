"""Construction of frozen bid-level analytical tables."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from ads_rank_lab.data.events import EventJoinDiagnostics, build_observed_outcomes
from ads_rank_lab.data.ipinyou import read_training_day


@dataclass(frozen=True)
class AnalyticalTableDiagnostics:
    """Diagnostics for one frozen analytical-table build."""

    season: int
    date: str
    rows: int
    observed_wins: int
    observed_clicks_on_wins: int
    observed_conversions_on_wins: int
    orphan_click_rows: int
    orphan_click_bid_ids: int
    orphan_conversion_rows: int
    orphan_conversion_bid_ids: int
    conflicting_paying_price_bid_ids: int
    duplicate_impression_bid_ids: int


def _observed_click_ids(
    clicks: pd.DataFrame,
    impressions: pd.DataFrame,
) -> set[str]:
    """Return click bid IDs that also have impression evidence."""
    impression_ids = set(impressions["bid_id"].dropna().astype("string"))
    click_ids = set(clicks["bid_id"].dropna().astype("string"))
    return click_ids & impression_ids


def _observed_conversion_ids(
    conversions: pd.DataFrame,
    impressions: pd.DataFrame,
) -> set[str]:
    """Return conversion bid IDs that also have impression evidence."""
    impression_ids = set(impressions["bid_id"].dropna().astype("string"))
    conversion_ids = set(conversions["bid_id"].dropna().astype("string"))
    return conversion_ids & impression_ids


def build_analytical_table(
    bids: pd.DataFrame,
    impressions: pd.DataFrame,
    clicks: pd.DataFrame,
    conversions: pd.DataFrame,
    *,
    season: int,
    date: str,
) -> tuple[pd.DataFrame, AnalyticalTableDiagnostics]:
    """Build the frozen bid-level table used by downstream experiments.

    Cleaning rules:

    1. Bid requests are the row universe.
    2. A win requires impression evidence.
    3. Click/conversion labels are positive only when the same ``bid_id`` also
       has impression evidence.
    4. Orphan click/conversion events are retained in diagnostics but are not
       used as positive response labels.
    5. Duplicate impression rows are aggregated by the event-join layer.
    6. Conflicting duplicate paying prices remain missing and explicitly
       flagged rather than being arbitrarily resolved.

    These rules distinguish *observed outcomes under the logged policy* from
    counterfactual outcomes. In particular, a lost bid is not treated as a
    known negative click opportunity for causal/policy evaluation.
    """
    joined, join_diagnostics = build_observed_outcomes(
        bids,
        impressions,
        clicks,
        conversions,
        strict=False,
    )

    impression_ids = set(impressions["bid_id"].dropna().astype("string"))
    all_click_ids = set(clicks["bid_id"].dropna().astype("string"))
    all_conversion_ids = set(conversions["bid_id"].dropna().astype("string"))

    valid_click_ids = _observed_click_ids(clicks, impressions)
    valid_conversion_ids = _observed_conversion_ids(conversions, impressions)

    click_counts = (
        clicks.loc[clicks["bid_id"].astype("string").isin(valid_click_ids), "bid_id"]
        .astype("string")
        .value_counts()
    )

    result = joined.copy()
    result["click_count"] = (
        result["bid_id"].map(click_counts).fillna(0).astype("int64")
    )
    result["clicked"] = result["bid_id"].isin(valid_click_ids)
    result["converted"] = result["bid_id"].isin(valid_conversion_ids)

    # Explicitly name the observational semantics for modeling code.
    result["response_observed"] = result["won"]
    result["click_label_observed"] = result["won"]
    result["conversion_label_observed"] = result["won"]

    orphan_click_rows = int(
        clicks["bid_id"].dropna().astype("string").isin(all_click_ids - impression_ids).sum()
    )
    orphan_conversion_rows = int(
        conversions["bid_id"]
        .dropna()
        .astype("string")
        .isin(all_conversion_ids - impression_ids)
        .sum()
    )

    diagnostics = AnalyticalTableDiagnostics(
        season=int(season),
        date=date,
        rows=int(len(result)),
        observed_wins=int(result["won"].sum()),
        observed_clicks_on_wins=int(result["clicked"].sum()),
        observed_conversions_on_wins=int(result["converted"].sum()),
        orphan_click_rows=orphan_click_rows,
        orphan_click_bid_ids=int(len(all_click_ids - impression_ids)),
        orphan_conversion_rows=orphan_conversion_rows,
        orphan_conversion_bid_ids=int(len(all_conversion_ids - impression_ids)),
        conflicting_paying_price_bid_ids=(
            join_diagnostics.conflicting_paying_price_bid_ids
        ),
        duplicate_impression_bid_ids=join_diagnostics.duplicate_impression_bid_ids,
    )

    return result, diagnostics


def build_training_day_table(
    dataset_root: str | Path,
    *,
    season: int,
    date: str,
) -> tuple[pd.DataFrame, AnalyticalTableDiagnostics]:
    """Load all four event streams for one day and build the analytical table."""
    bids = read_training_day(
        dataset_root, season=season, event_type="bid", date=date
    )
    impressions = read_training_day(
        dataset_root, season=season, event_type="imp", date=date
    )
    clicks = read_training_day(
        dataset_root, season=season, event_type="clk", date=date
    )
    conversions = read_training_day(
        dataset_root, season=season, event_type="conv", date=date
    )

    return build_analytical_table(
        bids,
        impressions,
        clicks,
        conversions,
        season=season,
        date=date,
    )


def diagnostics_to_dict(
    diagnostics: AnalyticalTableDiagnostics,
) -> dict[str, int | str]:
    """Return JSON-serializable analytical-table diagnostics."""
    return asdict(diagnostics)

"""Readers for the original iPinYou contest event logs.

This module preserves event-level evidence. It reads the original compressed
tab-separated logs for Seasons 2/3 without inventing click labels,
counterfactual outcomes, or auction candidates that are not present in a given
event file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd


EventType = Literal["bid", "imp", "clk", "conv"]
Season = Literal[2, 3]


BID_COLUMNS_S23: tuple[str, ...] = (
    "bid_id",
    "timestamp",
    "user_id",
    "user_agent",
    "ip",
    "region",
    "city",
    "ad_exchange",
    "domain",
    "url",
    "url_id",
    "slot_id",
    "slot_width",
    "slot_height",
    "slot_visibility",
    "slot_format",
    "slot_price",
    "creative_id",
    "bid_price",
    "advertiser_id",
    "user_tags",
)

EVENT_COLUMNS_S23: tuple[str, ...] = (
    "bid_id",
    "timestamp",
    "log_type",
    "user_id",
    "user_agent",
    "ip",
    "region",
    "city",
    "ad_exchange",
    "domain",
    "url",
    "url_id",
    "slot_id",
    "slot_width",
    "slot_height",
    "slot_visibility",
    "slot_format",
    "slot_price",
    "creative_id",
    "bid_price",
    "paying_price",
    "key_page",
    "advertiser_id",
    "user_tags",
)

EXPECTED_LOG_TYPE = {
    "imp": 1,
    "clk": 2,
    "conv": 3,
}


@dataclass(frozen=True)
class IpinYouLogFile:
    """Metadata parsed from one original iPinYou event-log path."""

    path: Path
    season: int
    event_type: EventType
    date: str


def _season_directory(season: int) -> str:
    mapping = {2: "training2nd", 3: "training3rd"}
    try:
        return mapping[season]
    except KeyError as exc:
        raise ValueError(
            "Initial AdsRankLab ingestion supports iPinYou Seasons 2 and 3 only."
        ) from exc


def discover_training_logs(
    dataset_root: str | Path,
    *,
    season: Season,
    event_type: EventType,
) -> list[IpinYouLogFile]:
    """Discover original compressed training logs in deterministic date order."""
    root = Path(dataset_root)
    directory = root / _season_directory(season)
    if not directory.exists():
        raise FileNotFoundError(f"iPinYou season directory not found: {directory}")

    logs: list[IpinYouLogFile] = []
    for path in sorted(directory.glob(f"{event_type}.*.txt.bz2")):
        parts = path.name.split(".")
        if len(parts) != 4 or parts[0] != event_type or parts[2:] != ["txt", "bz2"]:
            continue
        date = parts[1]
        if len(date) != 8 or not date.isdigit():
            continue
        logs.append(
            IpinYouLogFile(
                path=path,
                season=season,
                event_type=event_type,
                date=date,
            )
        )
    return logs


def columns_for_event(*, season: Season, event_type: EventType) -> tuple[str, ...]:
    """Return the exact raw column contract for one supported event type."""
    _season_directory(season)
    if event_type == "bid":
        return BID_COLUMNS_S23
    if event_type in EXPECTED_LOG_TYPE:
        return EVENT_COLUMNS_S23
    raise ValueError(f"Unsupported iPinYou event type: {event_type}")


def read_event_log(
    path: str | Path,
    *,
    season: Season,
    event_type: EventType,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read one original ``.txt.bz2`` iPinYou event log.

    The source files have no header, so exact Season 2/3 column names are
    supplied explicitly. ``null`` and empty fields are treated as missing.

    Timestamp is intentionally kept as a string because it encodes event
    ordering at sub-second precision; datetime conversion belongs in a later
    transformation layer.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"iPinYou event log not found: {source}")

    columns = columns_for_event(season=season, event_type=event_type)
    frame = pd.read_csv(
        source,
        sep="\t",
        header=None,
        names=list(columns),
        dtype="string",
        na_values=["null", ""],
        keep_default_na=True,
        compression="bz2",
        nrows=nrows,
    )

    if frame.empty:
        raise ValueError(f"iPinYou event log is empty: {source}")

    if event_type != "bid":
        expected = EXPECTED_LOG_TYPE[event_type]
        parsed = pd.to_numeric(frame["log_type"], errors="coerce")
        invalid = parsed.dropna()[parsed.dropna() != expected]
        if not invalid.empty:
            values = sorted({int(value) for value in invalid.tolist()})
            raise ValueError(
                f"{event_type} log contains unexpected log_type values: {values}; "
                f"expected {expected}"
            )

    return frame


def coerce_event_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce stable numeric fields while preserving IDs/timestamps as strings."""
    result = frame.copy()
    numeric_columns = (
        "region",
        "city",
        "ad_exchange",
        "slot_width",
        "slot_height",
        "slot_visibility",
        "slot_format",
        "slot_price",
        "bid_price",
        "paying_price",
        "log_type",
    )
    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def read_training_day(
    dataset_root: str | Path,
    *,
    season: Season,
    event_type: EventType,
    date: str,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read one event type for one training date from the original release."""
    if len(date) != 8 or not date.isdigit():
        raise ValueError("date must use YYYYMMDD format")

    directory = Path(dataset_root) / _season_directory(season)
    path = directory / f"{event_type}.{date}.txt.bz2"
    return read_event_log(
        path,
        season=season,
        event_type=event_type,
        nrows=nrows,
    )

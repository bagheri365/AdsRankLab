"""Canonical schema helpers for iPinYou-style RTB logs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class FieldSpec:
    """Definition of a canonical AdsRankLab field."""

    name: str
    required: bool
    role: str
    observed: bool = True


CANONICAL_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("click", True, "binary response label"),
    FieldSpec("conversion", False, "binary conversion label"),
    FieldSpec("bid_price", True, "advertiser/DSP bid"),
    FieldSpec("paying_price", True, "observed clearing/winning cost"),
    FieldSpec("floor_price", False, "auction floor price"),
    FieldSpec("advertiser_id", False, "advertiser identifier"),
    FieldSpec("creative_id", False, "creative/ad identifier"),
    FieldSpec("user_id", False, "user identifier"),
    FieldSpec("timestamp", False, "event time"),
    FieldSpec("slot_width", False, "ad-slot width"),
    FieldSpec("slot_height", False, "ad-slot height"),
    FieldSpec("slot_visibility", False, "slot visibility/category"),
    FieldSpec("slot_format", False, "slot format/category"),
    FieldSpec("region", False, "traffic region"),
    FieldSpec("city", False, "traffic city"),
    FieldSpec("user_agent", False, "browser/device user-agent context"),
)

REQUIRED_COLUMNS = frozenset(field.name for field in CANONICAL_FIELDS if field.required)
OPTIONAL_COLUMNS = frozenset(field.name for field in CANONICAL_FIELDS if not field.required)
KNOWN_COLUMNS = REQUIRED_COLUMNS | OPTIONAL_COLUMNS

BINARY_COLUMNS = ("click", "conversion")
NUMERIC_COLUMNS = ("bid_price", "paying_price", "floor_price", "slot_width", "slot_height")
CATEGORICAL_COLUMNS = (
    "advertiser_id",
    "creative_id",
    "user_id",
    "slot_visibility",
    "slot_format",
    "region",
    "city",
    "user_agent",
)


# Raw iPinYou releases can vary by season/file. This map is intentionally a
# normalization layer rather than a claim that every raw file contains every
# field below.
IPINYOU_ALIASES: dict[str, str] = {
    "click": "click",
    "conversion": "conversion",
    "bidprice": "bid_price",
    "bid_price": "bid_price",
    "payprice": "paying_price",
    "payingprice": "paying_price",
    "paying_price": "paying_price",
    "slotprice": "floor_price",
    "floorprice": "floor_price",
    "floor_price": "floor_price",
    "advertiser": "advertiser_id",
    "advertiserid": "advertiser_id",
    "advertiser_id": "advertiser_id",
    "creative": "creative_id",
    "creativeid": "creative_id",
    "creative_id": "creative_id",
    "userid": "user_id",
    "user_id": "user_id",
    "timestamp": "timestamp",
    "slotwidth": "slot_width",
    "slot_width": "slot_width",
    "slotheight": "slot_height",
    "slot_height": "slot_height",
    "slotvisibility": "slot_visibility",
    "slot_visibility": "slot_visibility",
    "slotformat": "slot_format",
    "slot_format": "slot_format",
    "region": "region",
    "city": "city",
    "useragent": "user_agent",
    "user_agent": "user_agent",
}


@dataclass(frozen=True)
class SchemaValidationResult:
    """Summary returned by :func:`validate_schema`."""

    missing_required: tuple[str, ...]
    present_optional: tuple[str, ...]
    unknown_columns: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.missing_required


def canonicalize_column_name(name: str) -> str:
    """Map a raw column name to the AdsRankLab canonical naming convention."""
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    compact = normalized.replace("_", "")
    return IPINYOU_ALIASES.get(normalized, IPINYOU_ALIASES.get(compact, normalized))


def canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with recognized raw iPinYou columns renamed canonically.

    Raises when normalization would create duplicate column names because that
    ambiguity should be resolved explicitly rather than silently overwritten.
    """
    renamed = [canonicalize_column_name(column) for column in frame.columns]
    if len(renamed) != len(set(renamed)):
        duplicates = sorted({name for name in renamed if renamed.count(name) > 1})
        raise ValueError(f"Canonicalization creates duplicate columns: {duplicates}")

    result = frame.copy()
    result.columns = renamed
    return result


def validate_schema(
    frame: pd.DataFrame,
    *,
    allowed_extra_columns: Iterable[str] = (),
) -> SchemaValidationResult:
    """Validate required canonical fields while preserving useful extras."""
    columns = set(frame.columns)
    allowed_extra = set(allowed_extra_columns)

    return SchemaValidationResult(
        missing_required=tuple(sorted(REQUIRED_COLUMNS - columns)),
        present_optional=tuple(sorted(OPTIONAL_COLUMNS & columns)),
        unknown_columns=tuple(sorted(columns - KNOWN_COLUMNS - allowed_extra)),
    )


def coerce_schema_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce recognized canonical fields to stable analysis-friendly types."""
    result = frame.copy()

    for column in NUMERIC_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    for column in BINARY_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
            invalid = result[column].dropna()[~result[column].dropna().isin([0, 1])]
            if not invalid.empty:
                values = sorted(invalid.unique().tolist())
                raise ValueError(f"{column} must be binary (0/1); found {values}")

    for column in CATEGORICAL_COLUMNS:
        if column in result.columns:
            result[column] = result[column].astype("string")

    if "timestamp" in result.columns:
        # Keep a nullable datetime representation where parsing is possible.
        result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")

    return result

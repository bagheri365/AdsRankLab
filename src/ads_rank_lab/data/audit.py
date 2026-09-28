"""Dataset audit utilities for canonical RTB tables."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from ads_rank_lab.data.schema import validate_schema


DEFAULT_NUMERIC_AUDIT_COLUMNS = (
    "bid_price",
    "paying_price",
    "floor_price",
)

DEFAULT_CATEGORICAL_AUDIT_COLUMNS = (
    "advertiser_id",
    "creative_id",
    "region",
    "city",
    "slot_visibility",
    "slot_format",
)


def _json_scalar(value):
    """Convert pandas/numpy scalar values into JSON-friendly Python values."""
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _binary_prevalence(series: pd.Series) -> dict[str, float | int | None]:
    observed = series.dropna()
    positives = int((observed == 1).sum())
    negatives = int((observed == 0).sum())
    count = int(len(observed))
    prevalence = positives / count if count else None
    return {
        "observed": count,
        "positive": positives,
        "negative": negatives,
        "prevalence": prevalence,
    }


def _numeric_summary(series: pd.Series) -> dict[str, float | int | None]:
    observed = pd.to_numeric(series, errors="coerce").dropna()
    if observed.empty:
        return {
            "observed": 0,
            "min": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "max": None,
        }

    return {
        "observed": int(len(observed)),
        "min": float(observed.min()),
        "p25": float(observed.quantile(0.25)),
        "median": float(observed.median()),
        "mean": float(observed.mean()),
        "p75": float(observed.quantile(0.75)),
        "max": float(observed.max()),
    }


def _categorical_summary(series: pd.Series, *, top_n: int) -> dict:
    observed = series.dropna()
    counts = observed.astype("string").value_counts(dropna=False).head(top_n)
    return {
        "observed": int(len(observed)),
        "cardinality": int(observed.astype("string").nunique(dropna=True)),
        "top_values": [
            {"value": str(value), "count": int(count)}
            for value, count in counts.items()
        ],
    }


def build_dataset_audit(
    frame: pd.DataFrame,
    *,
    numeric_columns: Iterable[str] = DEFAULT_NUMERIC_AUDIT_COLUMNS,
    categorical_columns: Iterable[str] = DEFAULT_CATEGORICAL_AUDIT_COLUMNS,
    top_n: int = 10,
) -> dict:
    """Build a JSON-serializable descriptive audit for a canonical RTB table.

    The audit is descriptive only. It does not infer counterfactual outcomes or
    claim that a changed policy would have produced the logged responses.
    """
    if frame.empty:
        raise ValueError("Cannot audit an empty DataFrame.")
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    schema = validate_schema(frame)
    missingness = {
        column: {
            "missing": int(frame[column].isna().sum()),
            "fraction": float(frame[column].isna().mean()),
        }
        for column in frame.columns
    }

    labels = {}
    for column in ("click", "conversion"):
        if column in frame.columns:
            labels[column] = _binary_prevalence(frame[column])

    numeric = {
        column: _numeric_summary(frame[column])
        for column in numeric_columns
        if column in frame.columns
    }

    categorical = {
        column: _categorical_summary(frame[column], top_n=top_n)
        for column in categorical_columns
        if column in frame.columns
    }

    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "column_count": int(len(frame.columns)),
        "schema": {
            "is_valid": schema.is_valid,
            "missing_required": list(schema.missing_required),
            "present_optional": list(schema.present_optional),
            "unknown_columns": list(schema.unknown_columns),
        },
        "missingness": missingness,
        "labels": labels,
        "numeric": numeric,
        "categorical": categorical,
    }

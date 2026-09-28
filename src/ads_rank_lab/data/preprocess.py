"""Small, explicit preprocessing helpers."""

from __future__ import annotations

import pandas as pd


def fill_missing_categories(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    missing_token: str = "__MISSING__",
) -> pd.DataFrame:
    """Return a copy with missing categorical values replaced consistently."""
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            raise KeyError(f"Missing categorical column: {column}")
        result[column] = result[column].astype("string").fillna(missing_token)
    return result


def coerce_numeric(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Return a copy with selected columns coerced to numeric values."""
    result = frame.copy()
    for column in columns:
        if column not in result.columns:
            raise KeyError(f"Missing numeric column: {column}")
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result

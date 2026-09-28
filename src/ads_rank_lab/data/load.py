"""Tabular loading helpers for RTB logs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_table(
    path: str | Path,
    *,
    sep: str = "\t",
    columns: list[str] | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Load a delimited RTB log into a DataFrame.

    Parameters
    ----------
    path:
        Local path to the source file.
    sep:
        Field delimiter. iPinYou files are commonly tab-delimited.
    columns:
        Optional explicit column names for headerless sources.
    read_csv_kwargs:
        Additional keyword arguments forwarded to ``pandas.read_csv``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the loaded table is empty.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"RTB source file not found: {source}")

    kwargs = dict(read_csv_kwargs)
    if columns is not None:
        kwargs.setdefault("names", columns)
        kwargs.setdefault("header", None)

    frame = pd.read_csv(source, sep=sep, **kwargs)
    if frame.empty:
        raise ValueError(f"RTB source file is empty: {source}")
    return frame

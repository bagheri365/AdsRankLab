"""Deterministic dataset splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSplits:
    """Container for train, validation, and test partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def deterministic_split(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    random_state: int = 365,
) -> DatasetSplits:
    """Split rows reproducibly without mutating the input frame.

    This is the initial generic split utility. The project should replace or
    wrap it with a time-aware protocol if inspection of the actual iPinYou
    files shows that chronological splitting is the more defensible design.
    """
    if frame.empty:
        raise ValueError("Cannot split an empty DataFrame.")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be < 1.")

    rng = np.random.default_rng(random_state)
    indices = np.arange(len(frame))
    rng.shuffle(indices)

    n_train = int(len(frame) * train_fraction)
    n_validation = int(len(frame) * validation_fraction)

    train_idx = indices[:n_train]
    validation_idx = indices[n_train : n_train + n_validation]
    test_idx = indices[n_train + n_validation :]

    return DatasetSplits(
        train=frame.iloc[train_idx].copy().reset_index(drop=True),
        validation=frame.iloc[validation_idx].copy().reset_index(drop=True),
        test=frame.iloc[test_idx].copy().reset_index(drop=True),
    )

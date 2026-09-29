"""Scalable logistic-regression baseline for click-through-rate prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_NUMERIC_FEATURES: tuple[str, ...] = (
    "hour",
    "slot_width",
    "slot_height",
    "slot_price",
)

DEFAULT_CATEGORICAL_FEATURES: tuple[str, ...] = (
    "weekday",
    "region",
    "city",
    "ad_exchange",
    "slot_visibility",
    "slot_format",
    "advertiser_id",
)


@dataclass(frozen=True)
class PctrBaselineConfig:
    """Configuration for the first transparent pCTR baseline."""

    numeric_features: tuple[str, ...] = DEFAULT_NUMERIC_FEATURES
    categorical_features: tuple[str, ...] = DEFAULT_CATEGORICAL_FEATURES
    c: float = 1.0
    max_iter: int = 3000
    class_weight: str | None = None
    tol: float = 1e-6


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive bid-time calendar features from the raw iPinYou timestamp."""
    if "timestamp" not in frame.columns:
        raise ValueError("timestamp is required to derive time features")

    result = frame.copy()
    timestamp = result["timestamp"].astype("string")
    if timestamp.str.len().lt(10).any():
        raise ValueError("timestamp values are too short for YYYYMMDDHH parsing")

    parsed_date = pd.to_datetime(timestamp.str.slice(0, 8), format="%Y%m%d", errors="coerce")
    hour = pd.to_numeric(timestamp.str.slice(8, 10), errors="coerce")
    result["hour"] = hour
    result["weekday"] = parsed_date.dt.dayofweek.astype("Int64").astype("string")
    return result


def build_pctr_pipeline(config: PctrBaselineConfig | None = None) -> Pipeline:
    """Build a leakage-safe logistic regression pipeline with sparse one-hot features."""
    config = config or PctrBaselineConfig()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), list(config.numeric_features)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32),
                list(config.categorical_features),
            ),
        ],
        sparse_threshold=1.0,
    )

    model = LogisticRegression(
        C=config.c,
        max_iter=config.max_iter,
        solver="saga",
        class_weight=config.class_weight,
        tol=config.tol,
    )

    return Pipeline([("features", preprocessor), ("model", model)])


def feature_columns(config: PctrBaselineConfig | None = None) -> tuple[str, ...]:
    config = config or PctrBaselineConfig()
    return config.numeric_features + config.categorical_features


def validate_feature_frame(frame: pd.DataFrame, config: PctrBaselineConfig | None = None) -> None:
    required = set(feature_columns(config))
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing pCTR baseline features: {missing}")

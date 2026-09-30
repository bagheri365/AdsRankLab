"""Observed-cost diagnostics on historically won impressions."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

def valid_observed_cost_population(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"observed_paying_price", "bid_price", "clicked", "advertiser_id"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"frame is missing required columns: {missing}")
    result = frame.copy()
    result["observed_paying_price"] = pd.to_numeric(result["observed_paying_price"], errors="coerce")
    result["bid_price"] = pd.to_numeric(result["bid_price"], errors="coerce")
    result["clicked"] = pd.to_numeric(result["clicked"], errors="coerce")
    valid = (
        result["observed_paying_price"].notna()
        & result["bid_price"].notna()
        & result["clicked"].notna()
        & (result["observed_paying_price"] >= 0)
        & (result["bid_price"] >= 0)
    )
    if "paying_price_conflict" in result.columns:
        valid &= ~result["paying_price_conflict"].fillna(False).astype(bool)
    return result.loc[valid].reset_index(drop=True)

def top_fraction_indices(values, *, fraction: float) -> np.ndarray:
    scores = np.asarray(values, dtype=float)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not np.isfinite(scores).all():
        raise ValueError("values must contain only finite values")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    k = max(1, int(math.ceil(len(scores) * fraction)))
    return np.lexsort((np.arange(len(scores)), -scores))[:k]

def summarize_observed_cost(frame: pd.DataFrame, *, scores, fraction: float) -> dict:
    if len(frame) != len(scores):
        raise ValueError("frame and scores must have equal length")
    selected = frame.iloc[top_fraction_indices(scores, fraction=fraction)]
    pay = selected["observed_paying_price"].to_numpy(dtype=float)
    bid = selected["bid_price"].to_numpy(dtype=float)
    clicks = selected["clicked"].to_numpy(dtype=float)
    total_cost = float(pay.sum())
    total_bid = float(bid.sum())
    click_count = int(clicks.sum())
    rows = int(len(selected))
    positive = pay > 0
    return {
        "fraction": float(fraction),
        "rows": rows,
        "observed_clicks": click_count,
        "observed_ctr": float(clicks.mean()) if rows else 0.0,
        "observed_total_paying_price": total_cost,
        "observed_mean_paying_price": float(pay.mean()) if rows else 0.0,
        "observed_median_paying_price": float(np.median(pay)) if rows else 0.0,
        "observed_p95_paying_price": float(np.quantile(pay, 0.95)) if rows else 0.0,
        "logged_total_bid_price": total_bid,
        "mean_bid_to_paying_price_ratio": (
            float(np.mean(bid[positive] / pay[positive])) if np.any(positive) else float("nan")
        ),
        "aggregate_bid_to_paying_price_ratio": (
            float(total_bid / total_cost) if total_cost > 0 else float("nan")
        ),
        "observed_cost_per_click": (
            float(total_cost / click_count) if click_count > 0 else float("nan")
        ),
        "observed_clicks_per_1000_cost_units": (
            float(click_count * 1000.0 / total_cost) if total_cost > 0 else float("nan")
        ),
    }

def advertiser_observed_cost(frame: pd.DataFrame, *, scores, policy: str, fraction: float) -> pd.DataFrame:
    if len(frame) != len(scores):
        raise ValueError("frame and scores must have equal length")
    selected = frame.iloc[top_fraction_indices(scores, fraction=fraction)].copy()
    grouped = (
        selected.groupby("advertiser_id", dropna=False)
        .agg(
            selected_rows=("advertiser_id", "size"),
            observed_clicks=("clicked", "sum"),
            observed_total_paying_price=("observed_paying_price", "sum"),
            observed_mean_paying_price=("observed_paying_price", "mean"),
        )
        .reset_index()
    )
    total_rows = len(selected)
    total_cost = float(selected["observed_paying_price"].sum())
    grouped["policy"] = policy
    grouped["top_fraction"] = float(fraction)
    grouped["selected_share"] = grouped["selected_rows"] / total_rows
    grouped["observed_cost_share"] = grouped["observed_total_paying_price"] / total_cost if total_cost > 0 else np.nan
    grouped["observed_ctr"] = grouped["observed_clicks"] / grouped["selected_rows"]
    grouped["observed_cost_per_click"] = np.where(
        grouped["observed_clicks"] > 0,
        grouped["observed_total_paying_price"] / grouped["observed_clicks"],
        np.nan,
    )
    cols = [
        "policy","top_fraction","advertiser_id","selected_rows","selected_share",
        "observed_clicks","observed_ctr","observed_total_paying_price",
        "observed_cost_share","observed_mean_paying_price","observed_cost_per_click",
    ]
    return grouped[cols].sort_values(["policy","observed_cost_share"], ascending=[True,False], ignore_index=True)

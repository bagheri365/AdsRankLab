"""Value-aware ranking scores for held-out won-impression analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric_bid_price(frame: pd.DataFrame) -> np.ndarray:
    if "bid_price" not in frame.columns:
        raise ValueError("frame must contain bid_price")
    bids = pd.to_numeric(frame["bid_price"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(bids).any():
        raise ValueError("bid_price contains missing or non-numeric values")
    if np.any(bids < 0):
        raise ValueError("bid_price must be non-negative")
    return bids


def build_policy_scores(
    frame: pd.DataFrame,
    *,
    raw_pctr,
    intercept_pctr,
    platt_pctr,
) -> pd.DataFrame:
    """Build comparable ranking scores on the same observed population.

    ``bid_x_*`` is a value proxy, not realized economic value. The score uses
    the logged bid as an advertiser-value signal and predicted CTR as a response
    signal. No claim is made that this reproduces a production auction.
    """
    bids = _numeric_bid_price(frame)
    raw = np.asarray(raw_pctr, dtype=float)
    intercept = np.asarray(intercept_pctr, dtype=float)
    platt = np.asarray(platt_pctr, dtype=float)

    n = len(frame)
    for name, values in (
        ("raw_pctr", raw),
        ("intercept_pctr", intercept),
        ("platt_pctr", platt),
    ):
        if values.shape != (n,):
            raise ValueError(f"{name} must have shape ({n},)")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{name} must lie in [0, 1]")

    return pd.DataFrame(
        {
            "bid_only": bids,
            "pctr_raw": raw,
            "bid_x_raw_pctr": bids * raw,
            "bid_x_intercept_pctr": bids * intercept,
            "bid_x_platt_pctr": bids * platt,
        },
        index=frame.index,
    )

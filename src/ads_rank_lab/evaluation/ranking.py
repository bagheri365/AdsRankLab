"""Ranking-comparison metrics for value-aware policy experiments."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _as_scores(values, *, name: str) -> np.ndarray:
    scores = np.asarray(values, dtype=float)
    if scores.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(scores) == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(scores).all():
        raise ValueError(f"{name} must contain only finite values")
    return scores


def top_k_indices(values, *, fraction: float) -> np.ndarray:
    """Return deterministic indices for the highest-scoring fraction."""
    scores = _as_scores(values, name="values")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")

    k = max(1, int(math.ceil(len(scores) * fraction)))
    order = np.lexsort((np.arange(len(scores)), -scores))
    return order[:k]


def top_k_overlap(
    reference,
    candidate,
    *,
    fraction: float,
) -> dict[str, float | int]:
    """Compare the selected top fraction under two scores."""
    ref_idx = top_k_indices(reference, fraction=fraction)
    cand_idx = top_k_indices(candidate, fraction=fraction)
    overlap = len(np.intersect1d(ref_idx, cand_idx, assume_unique=True))
    k = len(ref_idx)
    return {
        "fraction": float(fraction),
        "k": int(k),
        "overlap_count": int(overlap),
        "overlap_rate": float(overlap / k),
        "replacement_rate": float(1.0 - overlap / k),
    }


def rank_shift_summary(reference, candidate) -> dict[str, float | int]:
    """Summarize exact item-rank movement between two score vectors."""
    reference = _as_scores(reference, name="reference")
    candidate = _as_scores(candidate, name="candidate")
    if len(reference) != len(candidate):
        raise ValueError("reference and candidate must have equal length")

    n = len(reference)
    tie_break = np.arange(n)

    ref_order = np.lexsort((tie_break, -reference))
    cand_order = np.lexsort((tie_break, -candidate))

    ref_rank = np.empty(n, dtype=np.int64)
    cand_rank = np.empty(n, dtype=np.int64)
    ref_rank[ref_order] = np.arange(n, dtype=np.int64)
    cand_rank[cand_order] = np.arange(n, dtype=np.int64)

    shift = np.abs(cand_rank - ref_rank)

    # Local flip diagnostic: after sorting by the reference score, how often
    # does the candidate reverse the ordering of adjacent non-tied items?
    ref_sorted = reference[ref_order]
    cand_sorted = candidate[ref_order]
    comparable = ref_sorted[:-1] > ref_sorted[1:]
    local_flips = comparable & (cand_sorted[:-1] < cand_sorted[1:])
    comparable_count = int(comparable.sum())

    return {
        "rows": int(n),
        "rank_changed_fraction": float(np.mean(shift > 0)),
        "median_absolute_rank_shift": float(np.median(shift)),
        "mean_absolute_rank_shift": float(np.mean(shift)),
        "p95_absolute_rank_shift": float(np.quantile(shift, 0.95)),
        "max_absolute_rank_shift": int(shift.max()),
        "adjacent_reference_pairs": comparable_count,
        "adjacent_order_flip_rate": (
            float(local_flips.sum() / comparable_count)
            if comparable_count
            else 0.0
        ),
    }


def score_margin_summary(values, *, top_fraction: float = 0.01) -> dict[str, float | int]:
    """Describe score separation around the top-fraction decision boundary."""
    scores = _as_scores(values, name="values")
    if not 0.0 < top_fraction < 1.0:
        raise ValueError("top_fraction must lie in (0, 1)")

    n = len(scores)
    k = max(1, int(math.ceil(n * top_fraction)))
    if k >= n:
        raise ValueError("top_fraction selects the entire population")

    order = np.lexsort((np.arange(n), -scores))
    ranked = scores[order]
    kth = float(ranked[k - 1])
    next_score = float(ranked[k])
    absolute_gap = kth - next_score

    positive = ranked[ranked > 0]
    median_positive = float(np.median(positive)) if len(positive) else 0.0

    return {
        "fraction": float(top_fraction),
        "k": int(k),
        "boundary_score": kth,
        "next_score": next_score,
        "absolute_boundary_gap": float(absolute_gap),
        "gap_over_median_positive_score": (
            float(absolute_gap / median_positive)
            if median_positive > 0
            else 0.0
        ),
    }


def advertiser_top_share(
    frame: pd.DataFrame,
    scores,
    *,
    policy: str,
    top_fraction: float = 0.01,
) -> pd.DataFrame:
    """Report how a policy changes advertiser representation in its top set."""
    if "advertiser_id" not in frame.columns:
        raise ValueError("frame must contain advertiser_id")

    idx = top_k_indices(scores, fraction=top_fraction)
    advertisers = frame["advertiser_id"].astype("string").reset_index(drop=True)
    if len(advertisers) != len(scores):
        raise ValueError("frame and scores must have equal length")

    population_share = advertisers.value_counts(normalize=True, dropna=False)
    selected = advertisers.iloc[idx]
    selected_share = selected.value_counts(normalize=True, dropna=False)
    selected_count = selected.value_counts(dropna=False)

    rows = []
    all_ids = population_share.index.union(selected_share.index)
    for advertiser_id in all_ids:
        pop = float(population_share.get(advertiser_id, 0.0))
        sel = float(selected_share.get(advertiser_id, 0.0))
        rows.append(
            {
                "policy": policy,
                "top_fraction": float(top_fraction),
                "advertiser_id": advertiser_id,
                "selected_rows": int(selected_count.get(advertiser_id, 0)),
                "population_share": pop,
                "selected_share": sel,
                "representation_ratio": float(sel / pop) if pop > 0 else float("nan"),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["policy", "selected_share"],
        ascending=[True, False],
        ignore_index=True,
    )

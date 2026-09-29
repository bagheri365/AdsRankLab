"""Tie-aware robustness metrics for ranking-policy comparisons."""

from __future__ import annotations

import math

import numpy as np


def _as_scores(values, *, name: str) -> np.ndarray:
    scores = np.asarray(values, dtype=float)
    if scores.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(scores) == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(scores).all():
        raise ValueError(f"{name} must contain only finite values")
    return scores


def tie_aware_cutoff(values, *, fraction: float) -> dict[str, float | int]:
    """Describe guaranteed and ambiguous membership at a score cutoff."""
    scores = _as_scores(values, name="values")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")

    k = max(1, int(math.ceil(len(scores) * fraction)))
    order = np.lexsort((np.arange(len(scores)), -scores))
    cutoff = float(scores[order[k - 1]])

    strictly_above = int(np.sum(scores > cutoff))
    tied_at_cutoff = int(np.sum(scores == cutoff))
    selected_from_tie = int(max(0, k - strictly_above))

    return {
        "fraction": float(fraction),
        "k": int(k),
        "cutoff_score": cutoff,
        "strictly_above_cutoff": strictly_above,
        "tied_at_cutoff": tied_at_cutoff,
        "selected_from_cutoff_tie": selected_from_tie,
        "guaranteed_selected": strictly_above,
        "ambiguous_boundary_rows": tied_at_cutoff,
        "tie_dependent_fraction_of_selected": float(selected_from_tie / k),
    }


def normalized_rank_shift_summary(reference, candidate) -> dict[str, float | int]:
    """Summarize exact rank movement and normalize it by population size."""
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
    denominator = float(max(1, n - 1))

    return {
        "rows": int(n),
        "rank_changed_fraction": float(np.mean(shift > 0)),
        "median_absolute_rank_shift": float(np.median(shift)),
        "mean_absolute_rank_shift": float(np.mean(shift)),
        "p95_absolute_rank_shift": float(np.quantile(shift, 0.95)),
        "max_absolute_rank_shift": int(shift.max()),
        "median_absolute_rank_shift_fraction": float(np.median(shift) / denominator),
        "mean_absolute_rank_shift_fraction": float(np.mean(shift) / denominator),
        "p95_absolute_rank_shift_fraction": float(
            np.quantile(shift, 0.95) / denominator
        ),
        "max_absolute_rank_shift_fraction": float(shift.max() / denominator),
    }


def tie_aware_top_k_comparison(
    reference,
    candidate,
    *,
    fraction: float,
) -> dict[str, float | int]:
    """Compare top-k regions without imposing an ordering inside cutoff ties."""
    ref = _as_scores(reference, name="reference")
    cand = _as_scores(candidate, name="candidate")
    if len(ref) != len(cand):
        raise ValueError("reference and candidate must have equal length")

    ref_info = tie_aware_cutoff(ref, fraction=fraction)
    cand_info = tie_aware_cutoff(cand, fraction=fraction)

    ref_cut = float(ref_info["cutoff_score"])
    cand_cut = float(cand_info["cutoff_score"])

    ref_guaranteed = set(np.flatnonzero(ref > ref_cut).tolist())
    cand_guaranteed = set(np.flatnonzero(cand > cand_cut).tolist())
    ref_possible = set(np.flatnonzero(ref >= ref_cut).tolist())
    cand_possible = set(np.flatnonzero(cand >= cand_cut).tolist())

    guaranteed_union = ref_guaranteed | cand_guaranteed
    guaranteed_intersection = ref_guaranteed & cand_guaranteed
    possible_union = ref_possible | cand_possible
    possible_intersection = ref_possible & cand_possible

    return {
        "fraction": float(fraction),
        "reference_guaranteed_rows": int(len(ref_guaranteed)),
        "candidate_guaranteed_rows": int(len(cand_guaranteed)),
        "guaranteed_intersection_rows": int(len(guaranteed_intersection)),
        "guaranteed_jaccard": (
            float(len(guaranteed_intersection) / len(guaranteed_union))
            if guaranteed_union
            else 1.0
        ),
        "reference_possible_rows": int(len(ref_possible)),
        "candidate_possible_rows": int(len(cand_possible)),
        "possible_intersection_rows": int(len(possible_intersection)),
        "possible_jaccard": (
            float(len(possible_intersection) / len(possible_union))
            if possible_union
            else 1.0
        ),
        "reference_ambiguous_boundary_rows": int(
            ref_info["ambiguous_boundary_rows"]
        ),
        "candidate_ambiguous_boundary_rows": int(
            cand_info["ambiguous_boundary_rows"]
        ),
    }

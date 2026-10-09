"""Statistics: AUROC and AUPRC with grouped bootstrap intervals, and agreement between tiers.

Adapted from evo2-cloud-operations src/study/analysis.py (AUPRC and a metric argument added).
"""
from __future__ import annotations

import numpy as np


def average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks with ties given their average rank."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Probability that a positive scores above a negative (Mann-Whitney), ties counted as half."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=float)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve as average precision (step-wise, ties grouped)."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=float)
    n_pos = labels.sum()
    if n_pos == 0 or n_pos == len(labels):
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores, sorted_labels = scores[order], labels[order]
    # Evaluate precision and recall only at the last index of each run of tied scores.
    last = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    true_positives = np.cumsum(sorted_labels)[last]
    precision = true_positives / (last + 1)
    recall = true_positives / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


METRICS = {"auroc": auroc, "auprc": average_precision}


def grouped_bootstrap(labels, score_sets: dict[str, np.ndarray], groups, n: int = 2000, seed: int = 7,
                      metric: str = "auroc") -> dict:
    """A metric for each score set, with percentile intervals from resampling whole groups.

    The same resamples are used for every score set, so paired differences are also returned
    relative to the first score set.
    """
    metric_fn = METRICS[metric]
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(seed)
    names = list(score_sets)
    point = {k: metric_fn(labels, v) for k, v in score_sets.items()}
    draws = {k: np.empty(n) for k in names}
    for b in range(n):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(unique), len(unique))])
        for k in names:
            draws[k][b] = metric_fn(labels[idx], score_sets[k][idx])
    out = {}
    first = names[0]
    for k in names:
        entry = {metric: point[k], "ci95": [float(np.nanpercentile(draws[k], 2.5)), float(np.nanpercentile(draws[k], 97.5))]}
        if k != first:
            diff = draws[first] - draws[k]
            entry[f"difference_vs_{first}"] = point[first] - point[k]
            entry[f"difference_vs_{first}_ci95"] = [float(np.nanpercentile(diff, 2.5)), float(np.nanpercentile(diff, 97.5))]
        out[k] = entry
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(average_ranks(np.asarray(a, float)), average_ranks(np.asarray(b, float)))[0, 1])


def agreement(reference: np.ndarray, other: np.ndarray, rank_fraction: float = 0.01) -> dict:
    """How closely one platform's scores match the reference platform's, variant by variant."""
    reference, other = np.asarray(reference, float), np.asarray(other, float)
    diff = np.abs(reference - other)
    rank_ref, rank_other = average_ranks(reference), average_ranks(other)
    moved = np.abs(rank_ref - rank_other) > rank_fraction * len(reference)
    return {
        "n": int(len(reference)),
        "max_abs_difference": float(diff.max()),
        "median_abs_difference": float(np.median(diff)),
        "spearman": spearman(reference, other),
        "rank_moves_over_1pct": int(moved.sum()),
    }

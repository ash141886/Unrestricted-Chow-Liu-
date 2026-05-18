"""
Evaluation metrics: edge-level precision, recall, F_1, structural
Hamming distance, and Benjamini--Hochberg correction for paired
significance tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import reproduce_all as _ra  # noqa: E402


def f1_shd(true_edges, predicted_edges):
    """Edge-level precision, recall, F_1 and structural Hamming
    distance between a ground-truth edge set and a recovered edge set.

    Edges are passed as iterables of ``(i, j)`` tuples with ``i < j``.

    Returns
    -------
    precision, recall, f1, shd : float
    """
    return _ra.metrics(list(true_edges), list(predicted_edges))


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05):
    """Benjamini--Hochberg correction for a list of p-values.

    Parameters
    ----------
    p_values : sequence of float
        Raw p-values (unordered).
    alpha : float, optional
        Target false discovery rate (default 0.05).

    Returns
    -------
    adjusted : numpy.ndarray
        Step-up Benjamini--Hochberg adjusted p-values, in the same
        order as the input ``p_values``.
    rejected : numpy.ndarray of bool
        Indicator vector marking which hypotheses are rejected at the
        target FDR.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    sorted_p = p[order]
    adj_sorted = np.empty(m)
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        candidate = sorted_p[i] * m / rank
        running_min = min(running_min, candidate)
        adj_sorted[i] = running_min
    adjusted = np.empty(m)
    adjusted[order] = adj_sorted
    return adjusted, adjusted < alpha

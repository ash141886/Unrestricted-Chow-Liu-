"""
Top-level Chow--Liu structure-selection algorithms.

Three variants are exposed:

- :func:`unrestricted_chowliu` --- Algorithm 1 in the paper. Kruskal's
  algorithm on the pairwise mutual-information matrix with no
  type-pair restriction.
- :func:`edwards_restricted_chowliu` --- Algorithm 1a, the
  Edwards-restricted baseline used for comparison.
- :func:`gaussian_chowliu` --- Algorithm 1b, a Gaussian-shortcut
  variant for large graphs (continuous--continuous pairs use the
  closed-form Gaussian mutual information).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import reproduce_all as _ra  # noqa: E402


def _validate(variables, types):
    if len(variables) != len(types):
        raise ValueError(
            f"variables and types must have equal length "
            f"(got {len(variables)} and {len(types)})"
        )
    for t in types:
        if t not in ("c", "d"):
            raise ValueError(f"types entries must be 'c' or 'd' (got {t!r})")


def unrestricted_chowliu(variables, types, tau: float = 0.0):
    """Algorithm 1: unrestricted Chow--Liu structure selection.

    Parameters
    ----------
    variables : sequence of numpy.ndarray
        Length-``p`` sequence of one-dimensional arrays of equal length
        (the sample size ``n``).
    types : sequence of str
        Length-``p`` sequence of type labels: ``'c'`` for continuous,
        ``'d'`` for discrete.
    tau : float, optional
        Edge-weight threshold (default 0). When ``tau > 0`` the returned
        structure is the maximum spanning forest after edges with
        empirical mutual information ``<= tau`` are excluded.

    Returns
    -------
    edges : list of tuple of int
        List of ``(i, j)`` edge indices (with ``i < j``) of the
        selected tree or forest. Each edge weight is the pairwise
        mutual information estimated by the type-aware estimator
        (KSG on cc pairs, mixed-pair k-NN on cd pairs, plug-in on dd
        pairs).
    """
    _validate(variables, types)
    mi = _ra.compute_mi_matrix(variables, types, cc="knn")
    return _ra.kruskal_unrestricted(mi, types, tau=tau)


def edwards_restricted_chowliu(variables, types, tau: float = 0.0):
    """Algorithm 1a: Edwards-restricted Chow--Liu structure selection.

    The Kruskal loop rejects any candidate edge whose addition would
    place a continuous node on the unique path between two existing
    discrete components.

    Parameters and returns are the same as
    :func:`unrestricted_chowliu`.
    """
    _validate(variables, types)
    mi = _ra.compute_mi_matrix(variables, types, cc="knn")
    return _ra.kruskal_edwards(mi, types, tau=tau)


def gaussian_chowliu(variables, types, tau: float = 0.0):
    """Algorithm 1b: Gaussian-shortcut variant of Algorithm 1.

    Continuous--continuous pairs use the closed-form Gaussian mutual
    information ``-0.5 log(1 - rho^2)`` instead of the KSG estimator.
    Faster than Algorithm 1 on large graphs but consistent only under
    joint Gaussianity for the continuous variables.

    Parameters and returns are the same as
    :func:`unrestricted_chowliu`.
    """
    _validate(variables, types)
    mi = _ra.compute_mi_matrix(variables, types, cc="gauss")
    return _ra.kruskal_unrestricted(mi, types, tau=tau)

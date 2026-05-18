"""
Mutual-information estimators used by the unrestricted Chow--Liu
pipeline.

These functions are thin wrappers around the implementations in
``reproduce_all.py`` so that downstream users can import them with a
stable, documented signature without depending on the entire
reproduction script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make ``reproduce_all`` importable when the package is installed in
# editable mode from the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import reproduce_all as _ra  # noqa: E402


def mi_ksg(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """Kraskov--Stoegbauer--Grassberger mutual information for two
    continuous variables.

    Parameters
    ----------
    x, y : numpy.ndarray
        One-dimensional arrays of equal length holding the paired
        continuous observations.
    k : int, optional
        Number of nearest neighbours (default 5).

    Returns
    -------
    float
        Estimated mutual information in nats.
    """
    return _ra.mi_knn_cc(np.asarray(x).reshape(-1), np.asarray(y).reshape(-1), k=k)


def mi_mixed_knn(g: np.ndarray, d: np.ndarray, k: int = 5) -> float:
    """Mixed-pair k-NN mutual information of Gao et al. (2017) for one
    continuous variable and one discrete variable.

    Parameters
    ----------
    g : numpy.ndarray
        One-dimensional array of continuous observations.
    d : numpy.ndarray
        One-dimensional array of discrete observations (integer labels).
    k : int, optional
        Number of nearest neighbours (default 5).

    Returns
    -------
    float
        Estimated mutual information in nats.
    """
    return _ra.mi_knn_gd(np.asarray(g).reshape(-1), np.asarray(d).reshape(-1), k=k)


def mi_plugin_dd(x: np.ndarray, y: np.ndarray) -> float:
    """Empirical plug-in mutual information for two discrete variables.

    Parameters
    ----------
    x, y : numpy.ndarray
        One-dimensional arrays of discrete observations (integer labels).

    Returns
    -------
    float
        Estimated mutual information in nats.
    """
    return _ra.mi_discrete(np.asarray(x).reshape(-1), np.asarray(y).reshape(-1))


def mi_mdl_discretisation(
    g: np.ndarray, d: np.ndarray, bins_list=(2, 4, 8, 16, 32)
) -> float:
    """MDL-discretisation mutual information (Suzuki, 2017) for one
    continuous variable and one discrete variable.

    Parameters
    ----------
    g : numpy.ndarray
        One-dimensional array of continuous observations.
    d : numpy.ndarray
        One-dimensional array of discrete observations (integer labels).
    bins_list : tuple of int, optional
        Grid of candidate bin counts. The MDL-penalised mutual
        information is computed for each bin count and the maximum is
        returned.

    Returns
    -------
    float
        MDL-penalised mutual information in nats.
    """
    return _ra.mi_suzuki_cd(
        np.asarray(g).reshape(-1), np.asarray(d).reshape(-1), bins_list=bins_list
    )

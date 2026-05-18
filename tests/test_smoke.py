"""
Smoke tests for the unrestricted Chow--Liu package.

These tests check that the public API imports cleanly and that the
core algorithms recover the obvious tree on a tiny synthetic problem.
They are not exhaustive correctness tests; the full empirical
validation lives in ``reproduce_all.py``.
"""

import numpy as np
import pytest

from unrestricted_chowliu import algorithm, mi_estimators, metrics


@pytest.fixture
def small_dgd_chain():
    """A three-node D_1 - G - D_2 chain with strong signal.

    Edwards' restriction would forbid the chain through G; Algorithm 1
    must recover both edges; Algorithm 1a must miss at least one.
    """
    rng = np.random.default_rng(0)
    n = 1500
    d1 = rng.integers(0, 3, size=n)
    g = 2.0 * d1 + rng.standard_normal(size=n)
    thirds = np.quantile(g, [1 / 3, 2 / 3])
    d2 = (g > thirds[0]).astype(int) + (g > thirds[1]).astype(int)
    return [d1, g, d2], ["d", "c", "d"]


def test_imports():
    assert hasattr(algorithm, "unrestricted_chowliu")
    assert hasattr(algorithm, "edwards_restricted_chowliu")
    assert hasattr(algorithm, "gaussian_chowliu")
    assert hasattr(mi_estimators, "mi_ksg")
    assert hasattr(mi_estimators, "mi_mixed_knn")
    assert hasattr(mi_estimators, "mi_plugin_dd")
    assert hasattr(mi_estimators, "mi_mdl_discretisation")
    assert hasattr(metrics, "f1_shd")
    assert hasattr(metrics, "benjamini_hochberg")


def test_unrestricted_recovers_dgd_chain(small_dgd_chain):
    variables, types = small_dgd_chain
    edges = algorithm.unrestricted_chowliu(variables, types)
    assert (0, 1) in edges
    assert (1, 2) in edges
    assert len(edges) == 2


def test_edwards_misses_dgd_chain(small_dgd_chain):
    variables, types = small_dgd_chain
    edges = algorithm.edwards_restricted_chowliu(variables, types)
    # The restricted procedure cannot place the continuous G between
    # the two discrete variables D_1, D_2. It therefore returns at
    # least one edge that is not in the true chain.
    assert (0, 1) not in edges or (1, 2) not in edges


def test_benjamini_hochberg_arithmetic():
    p = [0.0001, 0.04, 0.20, 0.50, 0.80]
    adjusted, rejected = metrics.benjamini_hochberg(p, alpha=0.05)
    assert adjusted.shape == (5,)
    assert rejected.dtype == bool
    # The smallest p-value should be the smallest adjusted p-value.
    assert adjusted[0] == adjusted.min()


def test_mi_ksg_nonnegative():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(size=500)
    y = 0.7 * x + np.sqrt(1 - 0.7**2) * rng.standard_normal(size=500)
    mi = mi_estimators.mi_ksg(x, y, k=5)
    assert mi >= 0.0

"""
unrestricted_chowliu
====================

A clean Python interface to the unrestricted Chow--Liu tree-selection
procedure for mixed discrete-continuous data.

The full reproduction code for the accompanying paper lives in
``reproduce_all.py`` at the repository root. This package re-exports
the most reusable pieces of that script so users can build their own
mixed-type Chow-Liu pipelines.

Key entries
-----------
- :func:`algorithm.unrestricted_chowliu` --- Algorithm 1 (default).
- :func:`algorithm.edwards_restricted_chowliu` --- Algorithm 1a baseline.
- :func:`algorithm.gaussian_chowliu` --- Algorithm 1b Gaussian variant.
- :func:`mi_estimators.mi_ksg` --- continuous--continuous KSG estimator.
- :func:`mi_estimators.mi_mixed_knn` --- continuous--discrete k-NN
  estimator (Gao et al., 2017).
- :func:`mi_estimators.mi_plugin_dd` --- discrete--discrete plug-in.
- :func:`mi_estimators.mi_mdl_discretisation` --- MDL-discretisation
  estimator (Suzuki, 2017).
- :func:`metrics.f1_shd` --- precision / recall / F_1 / structural
  Hamming distance.
- :func:`metrics.benjamini_hochberg` --- Benjamini--Hochberg correction
  for paired t-tests.
"""

from . import algorithm
from . import mi_estimators
from . import metrics

__all__ = ["algorithm", "mi_estimators", "metrics"]
__version__ = "1.0.0"

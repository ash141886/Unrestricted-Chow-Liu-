# Unrestricted Chow–Liu

Reproduction code for the paper

> **A Note on Unrestricted Chow–Liu Tree Selection for Mixed
> Discrete–Continuous Data**

The manuscript is currently under review. This repository contains
the code, data, and figures needed to reproduce every table and
figure in the paper. A link to the published version will be added
once it is available.

---

## Overview

The classical Chow–Liu algorithm returns the maximum-weight
spanning tree of pairwise mutual informations. Earlier mixed-type
variants imposed a *type-pair* restriction (a continuous variable
cannot lie between two discrete variables along a tree path)
because the subsequent step of fitting a parametric conditional
Gaussian density on the chosen tree cannot represent such a path.

We show that this restriction is unnecessary for the
**structure-selection step**: tree selection depends only on
pairwise mutual-information values, which are well-defined for
every combination of variable types. The restriction applies only
to the later density-fitting step.

The repository implements:

- **Algorithm 1** — the unrestricted procedure (default). Kruskal
  on the pairwise mutual-information matrix with no type
  constraint, using the KSG estimator for continuous–continuous
  pairs, the mixed-pair k-NN estimator of Gao et al. for
  continuous–discrete pairs, and the empirical plug-in for
  discrete–discrete pairs.
- **Algorithm 1a** — the Edwards-restricted baseline used for
  comparison in every experiment.
- **Algorithm 1b** — a Gaussian-shortcut variant for large graphs.
- **MDL discretisation estimator** (Suzuki 2017), used for the
  sample-size comparison in Section 4.7.
- All experiments: the random-mixed-tree generator, the synthetic
  eQTL network, the Asia Bayesian sanity check, and the
  breast-cancer illustration.

---

## Installation

Python 3.10 or later is recommended.

```bash
git clone https://github.com/ash141886/Unrestricted-Chow-Liu-.git
cd Unrestricted-Chow-Liu-
pip install -r requirements.txt
```

The main dependencies are NumPy, SciPy, scikit-learn, pandas,
matplotlib, and networkx.

---

## Quick start

### Reproduce every table and figure in the paper

```bash
cd Unrestricted-Chow-Liu-
python reproduce_all.py
```

This runs all experiments in Section 4 and prints the numbers used
in Tables 1–14. Figures are regenerated in `figures/`. The full
script completes in approximately four minutes on a consumer
laptop CPU.

### Run a single experiment

The Python snippets below must be run **inside a Python
interpreter, not in the shell**. Start the interpreter from the
repo root:

```bash
cd Unrestricted-Chow-Liu-
python                # or: ipython
```

then paste, inside the interpreter:

```python
from reproduce_all import exp_4_1_mi_accuracy, exp_4_4_scaling
exp_4_1_mi_accuracy(n=2000, n_trials=15)
exp_4_4_scaling()
```

### Use the importable package

The clean importable layer lives in `src/unrestricted_chowliu/`.
To make it available system-wide, install it once in editable
mode from the repo root:

```bash
cd Unrestricted-Chow-Liu-
pip install -e .
```

After this, start Python from any directory and run:

```python
import numpy as np
from unrestricted_chowliu import algorithm

rng = np.random.default_rng(0)
d1 = rng.integers(0, 3, size=1500)
g  = 2.0 * d1 + rng.standard_normal(1500)
thirds = np.quantile(g, [1/3, 2/3])
d2 = (g > thirds[0]).astype(int) + (g > thirds[1]).astype(int)

edges = algorithm.unrestricted_chowliu([d1, g, d2], ["d", "c", "d"])
print(edges)
# Expected output: {(0, 1), (1, 2)}
```

If you would rather skip the `pip install -e .` step, set
`PYTHONPATH` to the `src/` folder at the command line instead:

```bash
cd Unrestricted-Chow-Liu-
PYTHONPATH=src python      # then paste the same Python block as above
```

---

## Repository layout

```
unrestricted-chow-liu/
├── README.md                  # This file
├── LICENSE                    # MIT
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project metadata
├── .gitignore
├── reproduce_all.py           # Main reproduction script
├── data/
│   ├── README.md              # Dataset documentation
│   ├── asia.csv               # Asia Bayesian network, 5000 samples
│   └── breastcancer.csv       # West et al. gene-expression dataset
├── figures/                   # Six figures used in the manuscript
├── src/unrestricted_chowliu/  # Importable Python package
│   ├── __init__.py
│   ├── algorithm.py           # Algorithm 1, 1a, 1b
│   ├── mi_estimators.py       # KSG, mixed-pair k-NN, plug-in, MDL
│   └── metrics.py             # F1, SHD, Benjamini–Hochberg helpers
└── tests/
    └── test_smoke.py          # Smoke tests for the public API
```

---

## Reproducibility map

Every numerical result in the paper is reproduced by a single
function in `reproduce_all.py`.

| Section | Experiment | Reproduced by |
|---|---|---|
| 4.1 | Mutual-information estimator accuracy on four conditional families | `exp_4_1_mi_accuracy` |
| 4.2 | D–G–D recovery (sanity check) | `exp_4_2_dgd_sanity` |
| 4.3 | Synthetic eQTL network | `exp_4_3_eqtl` |
| 4.4 | Random mixed graphs: scaling, effect-size, heterogeneous | `exp_4_4_scaling`, `exp_4_4_effect_sweep`, `exp_4_4_heterogeneous` |
| 4.5 | Sample-size sweep on the D–G–D chain | `exp_4_5_near_tie` |
| 4.6 | Shrinking-threshold variant of Algorithm 1 | `exp_4_6_shrinking` |
| 4.7 | Sample-size sweep against the MDL-discretisation estimator | `exp_4_7_suzuki_sweep` |
| 4.8 | Mutual-information estimator ablation on the eQTL benchmark | `exp_4_8_ablation` |
| 4.9 | Asia network (fully discrete sanity check) | `exp_4_9_asia` |
| 4.10 | Breast-cancer Chow–Liu forest (illustration only) | `exp_4_10_breast_cancer` |

The Benjamini–Hochberg correction reported in Sections 5 and 6 for
the 20 paired t-tests is recomputed by `exp_bh_correction`.

---

## Data sources

- **Asia network** (`data/asia.csv`): 5000 samples from the
  canonical Asia Bayesian network, distributed with the
  [gRbase](https://cran.r-project.org/package=gRbase) R package.
- **Breast cancer** (`data/breastcancer.csv`): the West et al.
  (2001) breast-cancer gene-expression dataset; publicly available.
- **Synthetic data**: generated on the fly by the script. The
  random-mixed-tree generator is specified in Appendix A.2 of the
  paper.

---

## Tests

A small set of smoke tests is provided. After installing the
dependencies and adding `src/` to the path, run:

```bash
PYTHONPATH=src pytest tests/
```

The tests verify that the package imports cleanly, that the
unrestricted procedure recovers a D–G–D chain that the
Edwards-restricted baseline cannot represent, and that the
Benjamini–Hochberg helper returns the expected adjusted p-values.

---

## License

Released under the MIT License. See `LICENSE`.

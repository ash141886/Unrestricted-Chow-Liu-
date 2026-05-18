# Unrestricted Chow–Liu

Reproduction code accompanying the paper

> **A Note on Unrestricted Chow–Liu Tree Selection for Mixed
> Discrete–Continuous Data**

The compiled paper (`docs/manuscript.pdf`) explains the method, the
theoretical results, and the empirical findings. This repository
provides the code, data, and figures needed to reproduce every
table and figure in the paper.

---

## What this code does

The classical Chow–Liu algorithm picks the maximum-weight spanning
tree of pairwise mutual informations. Earlier mixed-type variants
imposed a *type-pair* restriction (forbidding a continuous variable
between two discrete variables along the tree) because the
subsequent step of fitting a parametric conditional Gaussian density
on the chosen tree could not represent such a path. We show that
this restriction is unnecessary for the **structure-selection
step**: tree selection depends only on pairwise mutual-information
values, which are well-defined for every combination of variable
types. The restriction applies only to the later density-fitting
step.

The repository implements:

- **Algorithm 1** (unrestricted, default) — Kruskal on pairwise
  mutual-information weights with no type constraint, using the KSG
  estimator for continuous–continuous pairs, the mixed-pair $k$-NN
  estimator of Gao et al. for continuous–discrete pairs, and the
  empirical plug-in for discrete–discrete pairs.
- **Algorithm 1a** — the Edwards-restricted baseline, used for
  comparison in every experiment.
- **Algorithm 1b** — a Gaussian-shortcut variant for large graphs.
- **MDL discretisation estimator** (Suzuki 2017) for the sample-size
  comparison in §4.7.
- All experiments, including the random-mixed-tree generator, the
  synthetic eQTL benchmark, the Asia network sanity check, and the
  breast-cancer illustration.

---

## Installation

Python 3.10 or later is recommended. Install dependencies with:

```bash
pip install -r requirements.txt
```

The main dependencies are NumPy, SciPy, scikit-learn, pandas,
matplotlib, and networkx.

---

## Quick start

To reproduce every table and figure in the paper:

```bash
python reproduce_all.py
```

This runs all experiments in §4 and prints the numbers used in
Tables 1–14 of the paper. Figures are regenerated in the `figures/`
directory. The full script completes in approximately four minutes
on a consumer laptop CPU (Apple Silicon / mid-range Intel).

To run a single experiment, the script's `__main__` block can be
edited, or individual functions can be imported and called:

```python
from reproduce_all import exp_4_1_mi_accuracy, exp_4_4_scaling
exp_4_1_mi_accuracy(n=2000, n_trials=15)
exp_4_4_scaling()
```

---

## Repository layout

```
unrestricted-chow-liu/
├── README.md                  # This file
├── LICENSE                    # MIT
├── CITATION.cff               # Citation metadata
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project metadata
├── .gitignore
├── reproduce_all.py           # Main reproduction script (all experiments)
├── data/
│   ├── asia.csv               # Asia Bayesian network, 5000 samples
│   └── breastcancer.csv       # West et al. gene-expression dataset
├── figures/                   # Six figures used in the manuscript
├── docs/
│   ├── manuscript.pdf         # Compiled paper
│   ├── manuscript.tex         # LaTeX source
│   └── references.bib         # Bibliography
├── src/unrestricted_chowliu/  # Importable Python package
│   ├── __init__.py
│   ├── algorithm.py           # Algorithm 1, 1a, 1b
│   ├── mi_estimators.py       # KSG, mixed-pair k-NN, plug-in, MDL, ANOVA
│   └── metrics.py             # F1, SHD, paired-t helpers
└── tests/
    └── test_smoke.py          # Smoke tests
```

---

## What each experiment shows

| Section | Experiment | Reproduced by |
|---|---|---|
| §4.1 | MI estimator accuracy on four conditional families | `exp_4_1_mi_accuracy` |
| §4.2 | $D$–$G$–$D$ recovery (sanity check) | `exp_4_2_dgd_sanity` |
| §4.3 | Synthetic eQTL network | `exp_4_3_eqtl` |
| §4.4 | Random mixed graphs (scaling, effect-size, heterogeneous) | `exp_4_4_scaling`, `exp_4_4_effect_sweep`, `exp_4_4_heterogeneous` |
| §4.5 | Sample-size sweep on $D$–$G$–$D$ chain | `exp_4_5_near_tie` |
| §4.6 | Shrinking-threshold variant | `exp_4_6_shrinking` |
| §4.7 | Sample-size sweep against MDL discretisation | `exp_4_7_suzuki_sweep` |
| §4.8 | MI estimator ablation on eQTL benchmark | `exp_4_8_ablation` |
| §4.9 | Asia network sanity check | `exp_4_9_asia` |
| §4.10 | Breast-cancer Chow–Liu forest (illustration only) | `exp_4_10_breast_cancer` |

The Benjamini–Hochberg correction on the 20 paired $t$-tests
reported in the paper is recomputed by `exp_bh_correction`.

---

## Data sources

- **Asia network** (`data/asia.csv`): sampled from the canonical
  Asia Bayesian network, available with the [gRbase](https://cran.r-project.org/package=gRbase)
  R package. We use 5000 samples drawn once.
- **Breast cancer** (`data/breastcancer.csv`): the West et al.
  (2001) breast-cancer gene-expression dataset. Publicly available.
- **Synthetic data**: generated on the fly by the script; the
  random-mixed-tree generator is described in Appendix A.2 of the
  paper.

---

## Citation

If you use this code or build on the results, please cite the paper.
A machine-readable citation entry is provided in `CITATION.cff`.

---

## License

MIT. See `LICENSE`.

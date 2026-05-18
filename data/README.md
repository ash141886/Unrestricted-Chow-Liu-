# Data files

## `asia.csv`

5000-row sample from the canonical Asia Bayesian network of Lauritzen
and Spiegelhalter (1988). Eight binary variables. The same sample is
distributed with the [gRbase](https://cran.r-project.org/package=gRbase)
R package.

| Column | Variable |
|---|---|
| `A` | Visit to Asia |
| `T` | Tuberculosis |
| `S` | Smoker |
| `L` | Lung cancer |
| `B` | Bronchitis |
| `E` | Either tuberculosis or lung cancer |
| `X` | Positive X-ray |
| `D` | Dyspnoea |

## `breastcancer.csv`

Breast-cancer gene-expression dataset from West et al. (2001),
*PNAS* **98**(20). 250 samples × 1000 most-variable genes, plus a
binary class label (ER+ vs ER-). The dataset is publicly available
and is included here for reproducibility.

The first column is the binary class label; the remaining columns
are continuous gene-expression values (probe IDs as column headers).

## Synthetic data

All synthetic experiments generate their data inside
`reproduce_all.py` and do not require an external dataset. The
random-mixed-tree generator is described in Appendix A.2 of the
manuscript.

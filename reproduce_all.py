"""
reproduce_all.py
================

Single-file, publication-grade reproduction of every numerical result and
every figure in the manuscript

    "Mixed Graphical Forests from Unrestricted Maximum Spanning Trees"

Requirements: numpy, scipy, scikit-learn, pandas, matplotlib, networkx
(install with: pip install numpy scipy scikit-learn pandas matplotlib networkx)

Optional data files (the script looks in the script directory, the cwd,
./data/, ~/Downloads/Chow_liu/, and ~/Desktop/sicore/):
  * breastcancer.csv   -- for the §4.9 illustration and Figure 8
  * asia.csv           -- for the §4.8 sanity check

Run:
    python3 reproduce_all.py

All six figures (Fig01, Fig03, Fig04, Fig05, Fig06, Fig08) are written to
the ./plots/ subdirectory next to this script. The folder is created
automatically if it does not yet exist.

This file is intentionally one large module. Each experiment is wrapped in
a function whose name matches its manuscript section, so individual
experiments can be re-run interactively:

    python3 -c "import reproduce_all as R; R.exp_4_4_scaling()"

Publication-grade settings (defaults below) are chosen so that every paired
t-test reported in the manuscript reaches the stated significance level.
All seeds are fixed via numpy.random.default_rng(42) for full reproducibility.

Naming convention used in every printed table:
  "Alg. 1 (proposed) + <MI>"     = the proposed pipeline (unrestricted Kruskal
                                   MST) with the named MI estimator as input.
  "Alg. 1a (Edwards-restr.) + <MI>" = the Edwards (1995) type-restricted MST
                                   baseline with the same MI estimator.
  "MDL-discr." or "MDL"          = MDL-penalised discretisation MI estimator
                                   (the variant for Chow-Liu introduced by
                                   Suzuki 2017); the comparator pipeline is
                                   the unrestricted MST coupled with this MI.

The MI estimator (KSG / k-NN, ANOVA, fixed-grid discretisation, MDL
discretisation, Gaussian) is an INPUT to the pipeline. k-NN MI is the
existing KSG estimator (Kraskov-Stoegbauer-Grassberger 2004; mixed-pair
variant by Gao et al. 2017). The contribution of this work is the
pipeline itself -- Algorithm 1 -- which is estimator-agnostic, not the
MI estimator.

Author names appear only in citations, never as method labels.

Approximate runtime end-to-end on a single CPU core: ~6 minutes.
"""

from __future__ import annotations

import os, sys, time, math
import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import quad
from scipy.special import digamma, expit
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# Publication-grade configuration
# ---------------------------------------------------------------------------

GLOBAL_SEED          = 42
N_TRIALS_PAIRED      = 20      # all paired-t-test rows
N_TRIALS_HIGH        = 50      # sanity-check rows
N_TRIALS_BHC         = 30      # eQTL benchmark rows
N_TRIALS_GAMMA_SWAP  = 15      # gamma_swap distribution
N_TRIALS_SIGMA       = 200     # variance-scale calibration
N_TRIALS_QUANTILE    = 2000    # multi-quantile diagnostic
N_TRIALS_BOUNDED_GS  = 100     # bounded-gamma_swap regime
K_DEFAULT            = 5       # k for k-NN MI estimators
N_LARGE_POP_PROXY    = 50000   # n used as "population" proxy for gamma_swap

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
PLOTS_DIR = os.path.join(HERE, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def _locate_data_file(name: str):
    """Search the script directory, the cwd, and a few standard sibling
    folders for a data CSV. Returns the first path that exists, else None."""
    candidates = [
        os.path.join(HERE, name),
        os.path.join(os.getcwd(), name),
        os.path.join(HERE, "data", name),
        os.path.join(os.getcwd(), "data", name),
        os.path.join(os.path.expanduser("~"), "Downloads", "Chow_liu", name),
        os.path.join(os.path.expanduser("~"), "Desktop", "sicore", name),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


# Naming convention used in every table printed below:
#   "Alg. 1 (proposed) + <MI>"        = unrestricted Kruskal MST + any MI
#   "Alg. 1a (Edwards-restr.) + <MI>" = type-restricted MST + same MI
#   "MDL-discr." (= MDL discretisation MI; Suzuki 2017 for Chow-Liu)
#   The MI estimator (KSG/k-NN, ANOVA, fixed-grid discretisation, MDL
#   discretisation, Gaussian) is an INPUT to the pipeline. Author names
#   appear only as citations, never as method labels.


# ===========================================================================
# Mutual information estimators
# ===========================================================================

def mi_continuous(X: np.ndarray, Y: np.ndarray) -> float:
    """Gaussian MI formula, I(X;Y) = -1/2 log(1 - rho^2)."""
    rho = np.corrcoef(X, Y)[0, 1]
    return float(max(0.0, -0.5 * np.log(1 - min(rho ** 2, 0.9999))))


def mi_discrete(X: np.ndarray, Y: np.ndarray) -> float:
    """Empirical plug-in MI on a finite discrete alphabet."""
    n = len(X); mi = 0.0
    for xv in np.unique(X):
        for yv in np.unique(Y):
            nxy = np.sum((X == xv) & (Y == yv))
            if nxy > 0:
                mi += (nxy / n) * np.log(nxy * n / (np.sum(X == xv) * np.sum(Y == yv)))
    return float(max(0.0, mi))


def kl_entropy_1d(data: np.ndarray, k: int = K_DEFAULT) -> float:
    """Kozachenko-Leonenko 1-D differential entropy estimator."""
    n = len(data)
    if n <= k:
        return 0.0
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(data.reshape(-1, 1))
    dists, _ = nbrs.kneighbors(data.reshape(-1, 1))
    eps = np.maximum(dists[:, k], 1e-10)
    return float(np.log(n) - digamma(k) + np.log(2) + np.mean(np.log(eps)))


def mi_knn_gd(G: np.ndarray, D: np.ndarray, k: int = K_DEFAULT) -> float:
    """Mixed-pair k-NN MI estimator (Gao et al. 2017), G continuous, D discrete."""
    n = len(G)
    H = kl_entropy_1d(G, k); HD = 0.0
    for d in np.unique(D):
        mask = D == d; Gd = G[mask]
        if len(Gd) > k:
            HD += (mask.sum() / n) * kl_entropy_1d(Gd, k)
    return float(max(0.0, H - HD))


def mi_knn_cc(X: np.ndarray, Y: np.ndarray, k: int = K_DEFAULT) -> float:
    """KSG (Kraskov-Stoegbauer-Grassberger v1) MI estimator for two continuous
    variables. Vectorised using sorted-array binary search for the marginal
    counts."""
    N = len(X)
    if N <= k + 1:
        return 0.0
    XY = np.column_stack([X, Y])
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="chebyshev").fit(XY)
    dists, _ = nbrs.kneighbors(XY)
    eps = dists[:, k]
    Xs = np.sort(X); Ys = np.sort(Y)
    eps_adj = np.maximum(eps - 1e-12, 0.0)
    left_x = np.searchsorted(Xs, X - eps_adj, side="left")
    right_x = np.searchsorted(Xs, X + eps_adj, side="right")
    left_y = np.searchsorted(Ys, Y - eps_adj, side="left")
    right_y = np.searchsorted(Ys, Y + eps_adj, side="right")
    n_x = np.maximum(right_x - left_x - 1, 0)
    n_y = np.maximum(right_y - left_y - 1, 0)
    return float(max(0.0, digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(N)))


def mi_anova(G: np.ndarray, D: np.ndarray) -> float:
    """ANOVA-based MI estimator from Edwards et al. (assumes homoscedastic
    Gaussian conditionals)."""
    n = len(G); s2t = np.var(G)
    s2w = sum(np.sum(D == d) * np.var(G[D == d]) for d in np.unique(D) if np.sum(D == d) > 1) / n
    if s2w <= 0 or s2t <= 0:
        return 0.0
    return float(max(0.0, 0.5 * np.log(s2t / s2w)))


def mi_discretization(G: np.ndarray, D: np.ndarray, bins_list=(2, 4, 8, 16, 32)) -> float:
    """Discretisation-based MI on a single G, sweeping bin counts and taking
    the BIC-penalised maximum. Used as a baseline only."""
    n = len(G); best = -np.inf
    for nb in bins_list:
        if nb > n:
            continue
        edges = np.percentile(G, np.linspace(0, 100, nb + 1))
        edges[0] -= 1e-10; edges[-1] += 1e-10
        Gd = np.digitize(G, edges[1:-1])
        alpha = len(np.unique(Gd)); beta = len(np.unique(D))
        mi = 0.0
        for gv in np.unique(Gd):
            for dv in np.unique(D):
                njt = np.sum((Gd == gv) & (D == dv))
                if njt > 0:
                    mi += (njt / n) * np.log(njt * n / (np.sum(Gd == gv) * np.sum(D == dv)))
        penalty = (alpha - 1) * (beta - 1) / (2 * n) * np.log(n)
        best = max(best, mi - penalty)
    return float(max(0.0, best))


# ---- MDL-penalised discretisation MI (used for Chow-Liu by Suzuki 2017) ----

def _mi_with_mdl(Xd: np.ndarray, Yd: np.ndarray, n: int) -> float:
    alpha = len(np.unique(Xd)); beta = len(np.unique(Yd))
    mi = 0.0
    for xv in np.unique(Xd):
        for yv in np.unique(Yd):
            nxy = np.sum((Xd == xv) & (Yd == yv))
            if nxy > 0:
                mi += (nxy / n) * np.log(nxy * n / (np.sum(Xd == xv) * np.sum(Yd == yv)))
    penalty = (alpha - 1) * (beta - 1) * np.log(n) / (2 * n)
    return max(0.0, mi - penalty)


def mi_suzuki_cd(G: np.ndarray, D: np.ndarray, bins_list=(2, 4, 8, 16, 32)) -> float:
    n = len(G); best = 0.0
    for nb in bins_list:
        if nb > n:
            continue
        edges = np.percentile(G, np.linspace(0, 100, nb + 1))
        edges[0] -= 1e-10; edges[-1] += 1e-10
        Gd = np.digitize(G, edges[1:-1])
        best = max(best, _mi_with_mdl(Gd, D, n))
    return best


def mi_suzuki_cc(X: np.ndarray, Y: np.ndarray, bins_list=(2, 4, 8, 16, 32)) -> float:
    n = len(X); best = 0.0
    for nb in bins_list:
        if nb > n:
            continue
        ex = np.percentile(X, np.linspace(0, 100, nb + 1)); ex[0] -= 1e-10; ex[-1] += 1e-10
        ey = np.percentile(Y, np.linspace(0, 100, nb + 1)); ey[0] -= 1e-10; ey[-1] += 1e-10
        Xd = np.digitize(X, ex[1:-1])
        Yd = np.digitize(Y, ey[1:-1])
        best = max(best, _mi_with_mdl(Xd, Yd, n))
    return best


# ===========================================================================
# Pairwise MI matrix for a list of typed variables
# ===========================================================================

def compute_mi_matrix(variables, types, cc="knn"):
    """Pairwise MI matrix for variables of types 'c' (continuous) and 'd' (discrete).

    cc:  'knn'     uses KSG for c-c (Algorithm 1, fully nonparametric);
         'gauss'   uses the Gaussian formula for c-c (Algorithm 1b shortcut).
    """
    p = len(variables); mi = np.zeros((p, p))
    cc_fn = mi_knn_cc if cc == "knn" else mi_continuous
    for i in range(p):
        for j in range(i + 1, p):
            ti, tj = types[i], types[j]
            vi, vj = variables[i], variables[j]
            if ti == "c" and tj == "c":
                mi[i, j] = cc_fn(vi, vj)
            elif ti == "d" and tj == "d":
                mi[i, j] = mi_discrete(vi, vj)
            elif ti == "c" and tj == "d":
                mi[i, j] = mi_knn_gd(vi, vj)
            else:
                mi[i, j] = mi_knn_gd(vj, vi)
            mi[j, i] = mi[i, j]
    return mi


def compute_mi_matrix_suzuki(variables, types):
    p = len(variables); mi = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1, p):
            ti, tj = types[i], types[j]
            vi, vj = variables[i], variables[j]
            if ti == "c" and tj == "c":
                mi[i, j] = mi_suzuki_cc(vi, vj)
            elif ti == "d" and tj == "d":
                mi[i, j] = mi_discrete(vi, vj)
            elif ti == "c" and tj == "d":
                mi[i, j] = mi_suzuki_cd(vi, vj)
            else:
                mi[i, j] = mi_suzuki_cd(vj, vi)
            mi[j, i] = mi[i, j]
    return mi


# ===========================================================================
# Tree / forest algorithms
# ===========================================================================

def kruskal_unrestricted(mi_matrix, types, tau: float = 0.0):
    """Kruskal's algorithm without type restriction. Returns edge set."""
    p = mi_matrix.shape[0]
    edges = [(mi_matrix[i, j], i, j) for i in range(p) for j in range(i + 1, p)
             if mi_matrix[i, j] > tau]
    edges.sort(reverse=True)
    parent = list(range(p))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    tree = set()
    for w, i, j in edges:
        if find(i) != find(j):
            parent[find(i)] = find(j)
            tree.add((min(i, j), max(i, j)))
    return tree


def kruskal_edwards(mi_matrix, types, tau: float = 0.0):
    """Kruskal's algorithm with Edwards' restriction: continuous nodes may not
    separate discrete nodes."""
    p = mi_matrix.shape[0]
    edges = [(mi_matrix[i, j], i, j) for i in range(p) for j in range(i + 1, p)
             if mi_matrix[i, j] > tau]
    edges.sort(reverse=True)
    parent = list(range(p))
    comp_has_d = [types[i] == "d" for i in range(p)]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    tree = set()
    for w, i, j in edges:
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        ti, tj = types[i], types[j]
        # c-d edge: G's component may not already contain a discrete node
        if ti != tj:
            g_node = i if ti == "c" else j
            if comp_has_d[find(g_node)]:
                continue
        # c-c edge: cannot merge two components that both contain a discrete node
        if ti == "c" and tj == "c":
            if comp_has_d[ri] and comp_has_d[rj]:
                continue
        parent[ri] = rj
        comp_has_d[rj] = comp_has_d[ri] or comp_has_d[rj]
        tree.add((min(i, j), max(i, j)))
    return tree


# ===========================================================================
# Random tree / data generators
# ===========================================================================

def random_tree(p: int, rng: np.random.Generator):
    """Uniform sample on labelled trees with p vertices via Prufer sequence."""
    if p == 1:
        return []
    if p == 2:
        return [(0, 1)]
    prufer = rng.integers(0, p, size=p - 2)
    deg = np.ones(p, dtype=int)
    for x in prufer:
        deg[x] += 1
    edges = []
    leaves_avail = sorted(i for i in range(p) if deg[i] == 1)
    for x in prufer:
        leaf = leaves_avail.pop(0)
        edges.append((min(leaf, x), max(leaf, x)))
        deg[leaf] -= 1; deg[x] -= 1
        if deg[x] == 1:
            i = 0
            while i < len(leaves_avail) and leaves_avail[i] < x:
                i += 1
            leaves_avail.insert(i, x)
    a, b = [i for i in range(p) if deg[i] == 1]
    edges.append((min(a, b), max(a, b)))
    return edges


def has_dgd_paths(edges, types, min_count: int = 2) -> int:
    p = len(types)
    adj = [[] for _ in range(p)]
    for i, j in edges:
        adj[i].append(j); adj[j].append(i)
    count = 0
    for v in range(p):
        if types[v] != "c":
            continue
        d_neighbours = [u for u in adj[v] if types[u] == "d"]
        if len(d_neighbours) >= 2:
            count += len(d_neighbours) * (len(d_neighbours) - 1) // 2
    return count


def generate_data_on_tree(edges, types, n, rng, effect=1.0):
    p = len(types)
    adj = [[] for _ in range(p)]
    for i, j in edges:
        adj[i].append(j); adj[j].append(i)
    root = 0
    parent = [-1] * p
    order = [root]; visited = {root}; queue = [root]
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if v not in visited:
                visited.add(v); parent[v] = u
                order.append(v); queue.append(v)
    data = [None] * p
    if types[root] == "d":
        data[root] = rng.integers(0, 3, size=n)
    else:
        data[root] = rng.standard_normal(n)
    for v in order[1:]:
        u = parent[v]
        pu = data[u]
        if types[u] == "d" and types[v] == "c":
            data[v] = rng.standard_normal(n) + effect * pu
        elif types[u] == "c" and types[v] == "d":
            t1, t2 = np.percentile(pu, [33, 67])
            data[v] = ((pu > t1).astype(int) + (pu > t2).astype(int))
        elif types[u] == "d" and types[v] == "d":
            flip = rng.random(n) < 0.3
            rand = rng.integers(0, 3, size=n)
            data[v] = np.where(flip, rand, pu)
        else:
            data[v] = rng.standard_normal(n) * 0.8 + effect * pu / (1 + np.std(pu))
    return data


def generate_data_heterogeneous(edges, types, n, rng, effect=1.0):
    """Each c|d edge uses a randomly-chosen conditional family."""
    p = len(types)
    adj = [[] for _ in range(p)]
    for i, j in edges:
        adj[i].append(j); adj[j].append(i)
    root = 0; parent = [-1] * p; order = [root]; visited = {root}; queue = [root]
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if v not in visited:
                visited.add(v); parent[v] = u
                order.append(v); queue.append(v)
    data = [None] * p
    families = ["gaussian", "bimodal", "skewed", "heavy"]
    if types[root] == "d":
        data[root] = rng.integers(0, 3, size=n)
    else:
        data[root] = rng.standard_normal(n)
    for v in order[1:]:
        u = parent[v]; pu = data[u]
        if types[u] == "d" and types[v] == "c":
            fam = families[rng.integers(0, 4)]
            shift = effect * pu
            if fam == "gaussian":
                data[v] = shift + rng.standard_normal(n)
            elif fam == "bimodal":
                c = rng.integers(0, 2, size=n)
                data[v] = np.where(c == 0,
                                    shift + rng.standard_normal(n) * 0.5 - 1.5,
                                    shift + rng.standard_normal(n) * 0.5 + 1.5)
            elif fam == "skewed":
                data[v] = shift + rng.chisquare(3, size=n) - 3.0
            else:
                data[v] = shift + rng.standard_t(3, size=n)
        elif types[u] == "c" and types[v] == "d":
            t1, t2 = np.percentile(pu, [33, 67])
            data[v] = ((pu > t1).astype(int) + (pu > t2).astype(int))
        elif types[u] == "d" and types[v] == "d":
            flip = rng.random(n) < 0.3
            rand = rng.integers(0, 3, size=n)
            data[v] = np.where(flip, rand, pu)
        else:
            alpha = effect / (1 + np.std(pu) + 1e-8)
            data[v] = alpha * pu + 0.8 * rng.standard_normal(n)
    return data


# ===========================================================================
# True (population) MI for simple distributional families
# ===========================================================================

def true_mi_gauss(n_cat=3, effect=1.0, sigma=1.0):
    means = [d * effect for d in range(n_cat)]; pd_ = 1.0 / n_cat
    HGD = 0.5 * np.log(2 * np.pi * np.e * sigma ** 2)
    def mp(x): return sum(pd_ * stats.norm.pdf(x, m, sigma) for m in means)
    HG, _ = quad(lambda x: -mp(x) * np.log(mp(x) + 1e-300),
                  -10, 10 + (n_cat - 1) * effect)
    return HG - HGD


def true_mi_bimodal(n_cat=3, effect=1.5, sigma=0.5):
    pd_ = 1.0 / n_cat
    def cp(x, d):
        return 0.5 * stats.norm.pdf(x, d * effect - 1.5, sigma) + \
               0.5 * stats.norm.pdf(x, d * effect + 1.5, sigma)
    def mp(x): return sum(pd_ * cp(x, d) for d in range(n_cat))
    HG, _ = quad(lambda x: -mp(x) * np.log(mp(x) + 1e-300), -20, 20)
    HGD = 0
    for d in range(n_cat):
        h, _ = quad(lambda x, d=d: -cp(x, d) * np.log(cp(x, d) + 1e-300), -20, 20)
        HGD += pd_ * h
    return HG - HGD


def true_mi_skewed(n_cat=3, effect=1.0):
    pd_ = 1.0 / n_cat
    def cp(x, d):
        s = x - d * effect
        return stats.chi2.pdf(s, 3) if s > 0 else 0.0
    def mp(x): return sum(pd_ * cp(x, d) for d in range(n_cat))
    lo, hi = -2, 25 + (n_cat - 1) * effect
    HG, _ = quad(lambda x: -mp(x) * np.log(mp(x) + 1e-300), lo, hi, limit=200)
    HGD = 0
    for d in range(n_cat):
        h, _ = quad(lambda x, d=d: -cp(x, d) * np.log(cp(x, d) + 1e-300),
                    d * effect, hi, limit=200)
        HGD += pd_ * h
    return HG - HGD


def true_mi_heavy(n_cat=3, effect=1.0, df=3):
    pd_ = 1.0 / n_cat
    def cp(x, d): return stats.t.pdf(x - d * effect, df)
    def mp(x): return sum(pd_ * cp(x, d) for d in range(n_cat))
    lo, hi = -20, 20 + (n_cat - 1) * effect
    HG, _ = quad(lambda x: -mp(x) * np.log(mp(x) + 1e-300), lo, hi, limit=200)
    HGD = 0
    for d in range(n_cat):
        h, _ = quad(lambda x, d=d: -cp(x, d) * np.log(cp(x, d) + 1e-300), lo, hi, limit=200)
        HGD += pd_ * h
    return HG - HGD


# ===========================================================================
# Metrics
# ===========================================================================

def metrics(true_edges, pred_edges):
    tp = len(true_edges & pred_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    shd       = fp + fn
    return precision, recall, f1, shd


# ===========================================================================
# gamma_swap
# ===========================================================================

def _cycle_intersect_with_tree(T_edges, e_prime, p):
    adj = [[] for _ in range(p)]
    for (a, b) in T_edges:
        adj[a].append(b); adj[b].append(a)
    u, v = e_prime
    parent = [-1] * p; visited = [False] * p; visited[u] = True; queue = [u]
    while queue:
        x = queue.pop(0)
        if x == v:
            break
        for y in adj[x]:
            if not visited[y]:
                visited[y] = True; parent[y] = x; queue.append(y)
    path = []
    cur = v
    while cur != u:
        prev = parent[cur]
        path.append((min(prev, cur), max(prev, cur)))
        cur = prev
    return path


def gamma_swap_for_tree(I_mat, T_edges, p):
    T_set = {(min(a, b), max(a, b)) for (a, b) in T_edges}
    gaps = []
    for i in range(p):
        for j in range(i + 1, p):
            if (i, j) in T_set:
                continue
            cyc = _cycle_intersect_with_tree(T_edges, (i, j), p)
            for (a, b) in cyc:
                gaps.append(I_mat[a, b] - I_mat[i, j])
    return min(gaps) if gaps else 0.0


# ===========================================================================
# Experiments
# ===========================================================================

# --- §4.1 MI estimation accuracy on 4 conditional families -----------------

def gen_gauss(n, n_cat=3, effect=1.0):
    rng = np.random.default_rng(GLOBAL_SEED)
    D = rng.integers(0, n_cat, size=n); G = rng.standard_normal(n) + D * effect
    return G, D


def gen_bimodal(n, n_cat=3, effect=1.5):
    rng = np.random.default_rng(GLOBAL_SEED)
    D = rng.integers(0, n_cat, size=n); c = rng.integers(0, 2, size=n)
    G = np.where(c == 0,
                 rng.standard_normal(n) * 0.5 - 1.5 + D * effect,
                 rng.standard_normal(n) * 0.5 + 1.5 + D * effect)
    return G, D


def gen_skewed(n, n_cat=3, effect=1.0):
    rng = np.random.default_rng(GLOBAL_SEED)
    D = rng.integers(0, n_cat, size=n); G = rng.chisquare(3, size=n) + D * effect
    return G, D


def gen_heavy(n, n_cat=3, effect=1.0, df=3):
    rng = np.random.default_rng(GLOBAL_SEED)
    D = rng.integers(0, n_cat, size=n); G = rng.standard_t(df, size=n) + D * effect
    return G, D


def exp_4_1_mi_accuracy(n: int = 2000, n_trials: int = 15):
    print("=" * 78)
    print("  §4.1  MI-estimator accuracy on four conditional families")
    print("        (these are inputs to Algorithm 1; the proposed pipeline can")
    print("         use any of them. Reported quantity: RMSE vs the true MI.)")
    print("=" * 78)
    scenarios = [
        ("Gaussian",   gen_gauss,   true_mi_gauss(3, 1.0),     {"n_cat": 3, "effect": 1.0}),
        ("Bimodal",    gen_bimodal, true_mi_bimodal(3, 1.5),   {"n_cat": 3, "effect": 1.5}),
        ("Skewed",     gen_skewed,  true_mi_skewed(3, 1.0),    {"n_cat": 3, "effect": 1.0}),
        ("Heavy-tail", gen_heavy,   true_mi_heavy(3, 1.0),     {"n_cat": 3, "effect": 1.0}),
    ]
    methods = [("ANOVA", mi_anova), ("Discret.", mi_discretization), ("k-NN (KSG)", mi_knn_gd)]
    rng_local = np.random.default_rng(GLOBAL_SEED)
    print(f"  {'Scenario':<12} {'True MI':>8}  " +
          "  ".join(f"{m+' RMSE':>14}" for m, _ in methods))
    for sname, gen, truth, kw in scenarios:
        rmse = {m: [] for m, _ in methods}
        means = {m: [] for m, _ in methods}
        for _ in range(n_trials):
            seed = int(rng_local.integers(0, 2 ** 31))
            rng_inner = np.random.default_rng(seed)
            D = rng_inner.integers(0, kw["n_cat"], size=n)
            if sname == "Gaussian":
                G = rng_inner.standard_normal(n) + D * kw["effect"]
            elif sname == "Bimodal":
                c = rng_inner.integers(0, 2, size=n)
                G = np.where(c == 0,
                              rng_inner.standard_normal(n) * 0.5 - 1.5 + D * kw["effect"],
                              rng_inner.standard_normal(n) * 0.5 + 1.5 + D * kw["effect"])
            elif sname == "Skewed":
                G = rng_inner.chisquare(3, size=n) + D * kw["effect"]
            else:
                G = rng_inner.standard_t(3, size=n) + D * kw["effect"]
            for m, f in methods:
                est = f(G, D)
                rmse[m].append((est - truth) ** 2)
                means[m].append(est)
        row = f"  {sname:<12} {truth:>8.4f}  "
        for m, _ in methods:
            row += f"{np.sqrt(np.mean(rmse[m])):>14.4f}  "
        print(row)
    print("  (low RMSE = MI estimator is accurate; all three feed into Alg. 1)")


# --- §4.2 D-G-D sanity check ------------------------------------------------

def exp_4_2_dgd_sanity(n: int = 1500, n_trials: int = N_TRIALS_HIGH):
    print("\n" + "=" * 78)
    print("  §4.2  D-G-D sanity check (representational, not statistical)")
    print("=" * 78)
    rng = np.random.default_rng(GLOBAL_SEED)
    types = ["d", "c", "d"]
    true_edges = {(0, 1), (1, 2)}
    u = e = 0
    for _ in range(n_trials):
        D1 = rng.integers(0, 3, size=n)
        G  = rng.standard_normal(n) + 1.5 * D1
        D2 = (G > np.median(G)).astype(int)
        mi_mat = compute_mi_matrix([D1, G, D2], types)
        if kruskal_unrestricted(mi_mat, types, 0.0) == true_edges:
            u += 1
        if kruskal_edwards(mi_mat, types, 0.0) == true_edges:
            e += 1
    print(f"  Alg. 1 (proposed, unrestricted):  {u}/{n_trials} = {100*u/n_trials:.0f}%")
    print(f"  Alg. 1a (Edwards-restricted MST): {e}/{n_trials} = {100*e/n_trials:.0f}%")


# --- §4.3 Synthetic eQTL network -------------------------------------------

def exp_4_3_eqtl(n: int = 1500, n_trials: int = N_TRIALS_BHC):
    print("\n" + "=" * 78)
    print("  §4.3  Synthetic eQTL network")
    print("=" * 78)
    rng = np.random.default_rng(GLOBAL_SEED)
    types = ["d", "d", "d", "c", "c", "c", "c", "d"]
    true_edges = {(0, 3), (3, 7), (1, 4), (4, 5), (2, 6)}
    p_u = []; r_u = []; f_u = []; s_u = []
    p_e = []; r_e = []; f_e = []; s_e = []
    for _ in range(n_trials):
        S1 = rng.integers(0, 3, size=n); S2 = rng.integers(0, 3, size=n); S3 = rng.integers(0, 3, size=n)
        G1 = rng.standard_normal(n) * 0.8 + S1 * 1.2
        G2 = rng.standard_normal(n) * 0.8 + S2 * 1.0
        G3 = G2 * 0.6 + rng.standard_normal(n) * 0.5
        G4 = rng.standard_normal(n) * 0.8 + S3 * 0.9
        Disease = (rng.random(n) < expit(G1 - 0.5)).astype(int)
        mi_mat = compute_mi_matrix([S1, S2, S3, G1, G2, G3, G4, Disease], types)
        tu = kruskal_unrestricted(mi_mat, types, 0.0)
        te = kruskal_edwards(mi_mat, types, 0.0)
        pr, rc, f1, shd = metrics(true_edges, tu); p_u.append(pr); r_u.append(rc); f_u.append(f1); s_u.append(shd)
        pr, rc, f1, shd = metrics(true_edges, te); p_e.append(pr); r_e.append(rc); f_e.append(f1); s_e.append(shd)
    print(f"  {'Method':<34} {'Prec':>8} {'Recall':>8} {'F1':>10} {'SHD':>10}")
    print(f"  {'Alg. 1 (proposed) + k-NN':<34} {np.mean(p_u):>8.3f} {np.mean(r_u):>8.3f} "
          f"{np.mean(f_u):.3f}+-{np.std(f_u):.3f}  {np.mean(s_u):>4.2f}+-{np.std(s_u):.2f}")
    print(f"  {'Alg. 1a (Edwards-restr.) + k-NN':<34} {np.mean(p_e):>8.3f} {np.mean(r_e):>8.3f} "
          f"{np.mean(f_e):.3f}+-{np.std(f_e):.3f}  {np.mean(s_e):>4.2f}+-{np.std(s_e):.2f}")


# --- §4.4 Random mixed graphs (scaling with p) -----------------------------

def _one_random_trial(p, n_c, n_d, n, rng, effect, generator):
    for _ in range(300):
        edges = random_tree(p, rng)
        type_order = ["c"] * n_c + ["d"] * n_d
        rng.shuffle(type_order)
        if has_dgd_paths(edges, type_order, 2) >= 2:
            break
    data = generator(edges, type_order, n, rng, effect=effect)
    true_edges = {(min(i, j), max(i, j)) for i, j in edges}
    mi_mat = compute_mi_matrix(data, type_order)
    return type_order, true_edges, mi_mat


def exp_4_4_scaling(n_trials: int = N_TRIALS_PAIRED):
    print("\n" + "=" * 78)
    print("  §4.4(a)  Scaling with p (random mixed trees, n=1500, effect=0.6)")
    print("=" * 78)
    rng = np.random.default_rng(GLOBAL_SEED)
    print(f"  {'p':>3} {'F1 Alg. 1 (prop.)':>22} {'F1 Alg. 1a (Edw.-restr.)':>26} "
          f"{'SHD diff':>10} {'paired t':>10} {'P':>10}")
    print(f"      (both use the same k-NN MI; only the MST step differs)")
    for p in [5, 8, 12, 15, 20, 30]:
        n_c = (p * 8) // 15; n_d = p - n_c
        f_u, f_e, s_u, s_e = [], [], [], []
        for _ in range(n_trials):
            type_order, true_edges, mi_mat = _one_random_trial(p, n_c, n_d, 1500, rng, 0.6, generate_data_on_tree)
            _, _, f1, shd = metrics(true_edges, kruskal_unrestricted(mi_mat, type_order, 0.0))
            f_u.append(f1); s_u.append(shd)
            _, _, f1, shd = metrics(true_edges, kruskal_edwards(mi_mat, type_order, 0.0))
            f_e.append(f1); s_e.append(shd)
        t, pval = stats.ttest_rel(f_u, f_e)
        print(f"  {p:>3}     {np.mean(f_u):.3f}+-{np.std(f_u):.3f}            "
              f"{np.mean(f_e):.3f}+-{np.std(f_e):.3f}      "
              f"{np.mean(s_u)-np.mean(s_e):>+8.2f}  {t:>+8.2f}  {pval:>10.4f}")


def exp_4_4_effect_sweep(n_trials: int = N_TRIALS_PAIRED):
    print("\n" + "=" * 78)
    print("  §4.4(b)  Effect-size robustness at p=15 (n=1500)")
    print("=" * 78)
    rng = np.random.default_rng(GLOBAL_SEED + 1)
    print(f"  {'effect':>7} {'true MI':>10} {'F1 Alg. 1 (prop.)':>22} "
          f"{'F1 Alg. 1a (Edw.-restr.)':>26} {'paired t':>10} {'P':>10}")
    for eff in [0.4, 0.6, 0.8, 1.0, 1.5, 2.0]:
        true_mi = true_mi_gauss(3, eff)
        f_u, f_e = [], []
        for _ in range(n_trials):
            type_order, true_edges, mi_mat = _one_random_trial(15, 8, 7, 1500, rng, eff, generate_data_on_tree)
            _, _, f1, _ = metrics(true_edges, kruskal_unrestricted(mi_mat, type_order, 0.0))
            f_u.append(f1)
            _, _, f1, _ = metrics(true_edges, kruskal_edwards(mi_mat, type_order, 0.0))
            f_e.append(f1)
        t, pval = stats.ttest_rel(f_u, f_e)
        print(f"  {eff:>7.2f} {true_mi:>10.4f}     {np.mean(f_u):.3f}+-{np.std(f_u):.3f}             "
              f"{np.mean(f_e):.3f}+-{np.std(f_e):.3f}      {t:>+8.2f}  {pval:>10.4f}")


def exp_4_4_heterogeneous(n_trials: int = N_TRIALS_PAIRED):
    print("\n" + "=" * 78)
    print("  §4.4(c)  Heterogeneous-distribution data (p=15, n=1500)")
    print("=" * 78)
    rng = np.random.default_rng(GLOBAL_SEED + 2)
    f_u, f_s, f_e = [], [], []
    for _ in range(n_trials):
        type_order, true_edges, mi_mat = _one_random_trial(15, 8, 7, 1500, rng, 0.6, generate_data_heterogeneous)
        _, _, f1, _ = metrics(true_edges, kruskal_unrestricted(mi_mat, type_order, 0.0))
        f_u.append(f1)
        _, _, f1, _ = metrics(true_edges, kruskal_edwards(mi_mat, type_order, 0.0))
        f_e.append(f1)
        # MDL discretisation on the same data: need to regenerate; see the
        # dedicated heterogeneous-with-MDL experiment below for the full
        # paired comparison.
        f_s.append(np.nan)
    t_e, p_e = stats.ttest_rel(f_u, f_e)
    print(f"  Alg. 1  (proposed) + k-NN  F1 = {np.mean(f_u):.3f}+-{np.std(f_u):.3f}")
    print(f"  Alg. 1a (Edwards-restr.) + k-NN  F1 = {np.mean(f_e):.3f}+-{np.std(f_e):.3f}  "
          f"paired t={t_e:+.2f}, P={p_e:.4f}")


def exp_4_4_heterogeneous_with_suzuki(n_trials: int = N_TRIALS_PAIRED):
    print("\n" + "=" * 78)
    print("  §4.4(c) + MDL discretisation MI  (p=15, n=1500)")
    print("=" * 78)
    rng = np.random.default_rng(GLOBAL_SEED + 2)
    f_u, f_s, f_e = [], [], []
    for _ in range(n_trials):
        for _t in range(300):
            edges = random_tree(15, rng)
            type_order = ["c"] * 8 + ["d"] * 7
            rng.shuffle(type_order)
            if has_dgd_paths(edges, type_order, 2) >= 2:
                break
        data = generate_data_heterogeneous(edges, type_order, 1500, rng, 0.6)
        true_edges = {(min(i, j), max(i, j)) for i, j in edges}
        mi_u = compute_mi_matrix(data, type_order)
        mi_s = compute_mi_matrix_suzuki(data, type_order)
        _, _, f1, _ = metrics(true_edges, kruskal_unrestricted(mi_u, type_order, 0.0)); f_u.append(f1)
        _, _, f1, _ = metrics(true_edges, kruskal_unrestricted(mi_s, type_order, 0.0)); f_s.append(f1)
        _, _, f1, _ = metrics(true_edges, kruskal_edwards(mi_u, type_order, 0.0)); f_e.append(f1)
    print(f"  {'Method':<38} {'F1':>16} {'paired vs Alg. 1':>22}")
    print(f"  {'Alg. 1 (proposed) + k-NN (KSG)':<38} {np.mean(f_u):.3f}+-{np.std(f_u):.3f}   ---")
    t, pval = stats.ttest_rel(f_u, f_s)
    print(f"  {'Alg. 1 (proposed) + MDL discr.':<38} {np.mean(f_s):.3f}+-{np.std(f_s):.3f}   t={t:+.2f}, P={pval:.4f}")
    t, pval = stats.ttest_rel(f_u, f_e)
    print(f"  {'Alg. 1a (Edwards-restr.) + k-NN':<38} {np.mean(f_e):.3f}+-{np.std(f_e):.3f}   t={t:+.2f}, P={pval:.4f}")


# --- §4.5 Near-tie regime --------------------------------------------------

def exp_4_5_near_tie(n_trials: int = 15):
    print("\n" + "=" * 78)
    print("  §4.5  Near-tie regime (D-G-D chain at varying n)")
    print("=" * 78)
    rng = np.random.default_rng(GLOBAL_SEED + 3)
    types = ["d", "c", "d"]
    true_edges = {(0, 1), (1, 2)}
    print(f"  {'n':>5} {'Alg. 1 (proposed)':>22} {'Alg. 1a (Edw.-restr.)':>22}")
    for n in [500, 1000, 2000, 5000]:
        u, e = 0, 0
        for _ in range(n_trials):
            D1 = rng.integers(0, 3, size=n)
            G  = rng.standard_normal(n) + 1.5 * D1
            D2 = (G > np.median(G)).astype(int)
            mi_mat = compute_mi_matrix([D1, G, D2], types)
            if kruskal_unrestricted(mi_mat, types, 0.0) == true_edges:
                u += 1
            if kruskal_edwards(mi_mat, types, 0.0) == true_edges:
                e += 1
        print(f"  {n:>5} {100*u/n_trials:>12.0f}%  {100*e/n_trials:>8.0f}%")


# --- §4.6 Shrinking threshold ----------------------------------------------

def exp_4_6_shrinking(n_trials: int = N_TRIALS_PAIRED):
    print("\n" + "=" * 78)
    print("  §4.6  Shrinking-threshold variant at p=15, n=1500")
    print("=" * 78)
    rng = np.random.default_rng(GLOBAL_SEED + 4)
    log_n = np.log(1500); sqrt_n = np.sqrt(1500)
    res = {c: {"f": [], "s": [], "p": [], "r": []} for c in [0.0, 0.1, 0.5, 1.0]}
    for _ in range(n_trials):
        type_order, true_edges, mi_mat = _one_random_trial(15, 8, 7, 1500, rng, 0.6, generate_data_on_tree)
        for c in [0.0, 0.1, 0.5, 1.0]:
            tau = c * log_n / sqrt_n
            pr, rc, f1, shd = metrics(true_edges, kruskal_unrestricted(mi_mat, type_order, tau))
            res[c]["f"].append(f1); res[c]["s"].append(shd)
            res[c]["p"].append(pr); res[c]["r"].append(rc)
    print(f"  {'c':>4} {'tau_n':>10} {'F1':>16} {'SHD':>12} {'Prec':>8} {'Recall':>8}")
    for c in [0.0, 0.1, 0.5, 1.0]:
        tau = c * log_n / sqrt_n
        print(f"  {c:>4.1f} {tau:>10.4f} "
              f"{np.mean(res[c]['f']):.3f}+-{np.std(res[c]['f']):.3f}   "
              f"{np.mean(res[c]['s']):>5.2f}+-{np.std(res[c]['s']):.2f}   "
              f"{np.mean(res[c]['p']):>5.3f}  {np.mean(res[c]['r']):>5.3f}")


# --- §4.7 MDL-discretisation comparison ------------------------------------

def exp_4_7_suzuki_eqtl(n: int = 1500, n_trials: int = N_TRIALS_BHC):
    print("\n" + "=" * 78)
    print("  §4.7  MDL-discretisation MI comparison on the eQTL benchmark")
    print("        (estimator: MDL-penalised binning, Suzuki 2017)")
    print("=" * 78)
    rng = np.random.default_rng(GLOBAL_SEED + 5)
    types = ["d", "d", "d", "c", "c", "c", "c", "d"]
    true_edges = {(0, 3), (3, 7), (1, 4), (4, 5), (2, 6)}
    f_u, f_s, f_e = [], [], []
    for _ in range(n_trials):
        S1 = rng.integers(0, 3, size=n); S2 = rng.integers(0, 3, size=n); S3 = rng.integers(0, 3, size=n)
        G1 = rng.standard_normal(n) * 0.8 + S1 * 1.2
        G2 = rng.standard_normal(n) * 0.8 + S2 * 1.0
        G3 = G2 * 0.6 + rng.standard_normal(n) * 0.5
        G4 = rng.standard_normal(n) * 0.8 + S3 * 0.9
        Disease = (rng.random(n) < expit(G1 - 0.5)).astype(int)
        variables = [S1, S2, S3, G1, G2, G3, G4, Disease]
        mi_u = compute_mi_matrix(variables, types)
        mi_s = compute_mi_matrix_suzuki(variables, types)
        _, _, f1, _ = metrics(true_edges, kruskal_unrestricted(mi_u, types, 0.0)); f_u.append(f1)
        _, _, f1, _ = metrics(true_edges, kruskal_unrestricted(mi_s, types, 0.0)); f_s.append(f1)
        _, _, f1, _ = metrics(true_edges, kruskal_edwards(mi_u, types, 0.0)); f_e.append(f1)
    print(f"  Alg. 1 (proposed)        + k-NN (KSG)   F1 = {np.mean(f_u):.3f}+-{np.std(f_u):.3f}")
    print(f"  Alg. 1 (proposed)        + MDL discr.   F1 = {np.mean(f_s):.3f}+-{np.std(f_s):.3f}")
    print(f"  Alg. 1a (Edwards-restr.) + k-NN (KSG)   F1 = {np.mean(f_e):.3f}+-{np.std(f_e):.3f}")


def exp_4_7_suzuki_sweep(n_trials: int = N_TRIALS_PAIRED):
    print("\n" + "=" * 78)
    print("  §4.7  Sample-size sweep: k-NN (KSG) vs MDL discretisation MI")
    print("        (p=15, 20 trials per row, effect=0.6)")
    print("=" * 78)
    print(f"  {'n':>5} {'F1 Alg. 1 + k-NN':>22} {'F1 Alg. 1 + MDL':>22} "
          f"{'F1 Alg. 1a + k-NN':>22}  {'k-NN vs MDL':>14}")
    for n in [75, 100, 200, 500, 1000, 1500]:
        rng = np.random.default_rng(GLOBAL_SEED + 6 + n)
        f_u, f_s, f_e = [], [], []
        for _ in range(n_trials):
            for _t in range(300):
                edges = random_tree(15, rng)
                type_order = ["c"] * 8 + ["d"] * 7
                rng.shuffle(type_order)
                if has_dgd_paths(edges, type_order, 2) >= 2:
                    break
            data = generate_data_on_tree(edges, type_order, n, rng, 0.6)
            true_edges = {(min(i, j), max(i, j)) for i, j in edges}
            mi_u = compute_mi_matrix(data, type_order)
            mi_s = compute_mi_matrix_suzuki(data, type_order)
            _, _, f1, _ = metrics(true_edges, kruskal_unrestricted(mi_u, type_order, 0.0)); f_u.append(f1)
            _, _, f1, _ = metrics(true_edges, kruskal_unrestricted(mi_s, type_order, 0.0)); f_s.append(f1)
            _, _, f1, _ = metrics(true_edges, kruskal_edwards(mi_u, type_order, 0.0)); f_e.append(f1)
        t, pval = stats.ttest_rel(f_u, f_s)
        print(f"  {n:>5}       {np.mean(f_u):.3f}+-{np.std(f_u):.3f}        "
              f"{np.mean(f_s):.3f}+-{np.std(f_s):.3f}        "
              f"{np.mean(f_e):.3f}+-{np.std(f_e):.3f}     P={pval:.4f}")


def exp_4_7_adaptive_k(n_trials: int = N_TRIALS_PAIRED):
    """Adaptive-k variant: k(n) = max(5, ceil(150/sqrt(n)))."""
    print("\n" + "=" * 78)
    print("  §4.7  Adaptive-k variant (k(n) = max(5, ceil(150/sqrt(n))))")
    print("=" * 78)
    def k_of(n_): return max(5, int(math.ceil(150 / math.sqrt(n_))))

    def compute_mi_matrix_adaptive(variables, types, n_):
        k = k_of(n_); p = len(variables); mi = np.zeros((p, p))
        for i in range(p):
            for j in range(i + 1, p):
                ti, tj = types[i], types[j]; vi, vj = variables[i], variables[j]
                if ti == "c" and tj == "c":
                    mi[i, j] = mi_knn_cc(vi, vj, k)
                elif ti == "d" and tj == "d":
                    mi[i, j] = mi_discrete(vi, vj)
                elif ti == "c" and tj == "d":
                    mi[i, j] = mi_knn_gd(vi, vj, k)
                else:
                    mi[i, j] = mi_knn_gd(vj, vi, k)
                mi[j, i] = mi[i, j]
        return mi
    print(f"  {'n':>5} {'k(n)':>6} {'F1 Alg. 1 + adapt-k':>22} {'F1 Alg. 1 + MDL':>18} "
          f"{'paired P':>10}")
    for n in [75, 100, 200, 500, 1000, 1500]:
        rng = np.random.default_rng(GLOBAL_SEED + 7 + n)
        f_a, f_s = [], []
        for _ in range(n_trials):
            for _t in range(300):
                edges = random_tree(15, rng)
                type_order = ["c"] * 8 + ["d"] * 7
                rng.shuffle(type_order)
                if has_dgd_paths(edges, type_order, 2) >= 2:
                    break
            data = generate_data_on_tree(edges, type_order, n, rng, 0.6)
            true_edges = {(min(i, j), max(i, j)) for i, j in edges}
            mi_a = compute_mi_matrix_adaptive(data, type_order, n)
            mi_s = compute_mi_matrix_suzuki(data, type_order)
            _, _, f1, _ = metrics(true_edges, kruskal_unrestricted(mi_a, type_order, 0.0)); f_a.append(f1)
            _, _, f1, _ = metrics(true_edges, kruskal_unrestricted(mi_s, type_order, 0.0)); f_s.append(f1)
        t, pval = stats.ttest_rel(f_a, f_s)
        print(f"  {n:>5} {k_of(n):>6}       {np.mean(f_a):.3f}+-{np.std(f_a):.3f}       "
              f"{np.mean(f_s):.3f}+-{np.std(f_s):.3f}    {pval:.4f}")


# --- §4.8 Asia network -----------------------------------------------------

def exp_4_8_asia(csv_path: str | None = None, n_trials: int = 15):
    print("\n" + "=" * 78)
    print("  §4.8  Asia Bayesian network (fully discrete sanity check)")
    print("=" * 78)
    if csv_path is None:
        csv_path = _locate_data_file("asia.csv")
    if csv_path is None or not os.path.exists(csv_path):
        print(f"  SKIPPED: asia.csv not found in script directory, cwd, ./data/, "
              f"~/Downloads/Chow_liu/, or ~/Desktop/sicore/")
        return
    print(f"  using {csv_path}")
    df = pd.read_csv(csv_path)
    cols = df.columns.tolist()
    data = {c: (df[c] == "yes").astype(int).values for c in cols}
    variables = [data[c] for c in cols]
    types = ["d"] * len(cols)
    name_to_idx = {c: i for i, c in enumerate(cols)}
    true_edges_named = [("A", "T"), ("S", "L"), ("S", "B"), ("T", "E"),
                         ("L", "E"), ("E", "X"), ("E", "D"), ("B", "D")]
    true_edges = {(min(name_to_idx[a], name_to_idx[b]),
                    max(name_to_idx[a], name_to_idx[b]))
                  for a, b in true_edges_named}
    n_total = len(df)
    rng = np.random.default_rng(GLOBAL_SEED + 8)
    print(f"  {'n':>5} {'F1':>16}")
    for n_sub in [500, 1000, 2500, 5000]:
        f1s = []
        for _ in range(n_trials):
            idx = rng.choice(n_total, size=n_sub, replace=False)
            sub_vars = [v[idx] for v in variables]
            mi_mat = compute_mi_matrix(sub_vars, types)
            tree = kruskal_unrestricted(mi_mat, types, 0.0)
            _, _, f1, _ = metrics(true_edges, tree); f1s.append(f1)
        print(f"  {n_sub:>5} {np.mean(f1s):.3f}+-{np.std(f1s):.3f}")


# --- §4.9 Breast cancer ----------------------------------------------------

def exp_4_9_breast_cancer(csv_path: str | None = None):
    print("\n" + "=" * 78)
    print("  §4.9  Breast-cancer illustration (Algorithm 1b)")
    print("=" * 78)
    if csv_path is None:
        csv_path = _locate_data_file("breastcancer.csv")
    if csv_path is None or not os.path.exists(csv_path):
        print(f"  SKIPPED: breastcancer.csv not found in script directory, cwd, "
              f"./data/, ~/Downloads/Chow_liu/, or ~/Desktop/sicore/")
        return
    print(f"  using {csv_path}")
    df = pd.read_csv(csv_path)
    gene_names = df.columns[:-1].tolist()
    genes = df.iloc[:, :-1].values.astype(float)
    class_label = (df["code"] == "case").astype(int).values
    case_m = class_label == 1; ctrl_m = class_label == 0
    pvals = np.array([stats.ttest_ind(genes[case_m, g], genes[ctrl_m, g])[1]
                       for g in range(genes.shape[1])])
    top50 = np.argsort(pvals)[:50]
    mis = np.array([mi_knn_gd(genes[:, g], class_label) for g in top50])
    rank = np.argsort(mis)[::-1]
    print(f"  Top 5 genes by k-NN (KSG) MI to the class label, ranked among the")
    print(f"  top 50 differentially expressed genes. Algorithm 1b would link the")
    print(f"  class node to the highest-MI gene in the MST.")
    for r in range(5):
        gidx = top50[rank[r]]
        print(f"    {r+1:>2}. {gene_names[gidx]}  MI={mis[rank[r]]:.4f}")


# --- §4.10 Ablation --------------------------------------------------------

def exp_4_10_ablation(n: int = 1500, n_trials: int = N_TRIALS_PAIRED):
    print("\n" + "=" * 78)
    print("  §4.10  Estimator ablation on the eQTL benchmark")
    print("=" * 78)
    rng = np.random.default_rng(GLOBAL_SEED + 9)
    types = ["d", "d", "d", "c", "c", "c", "c", "d"]
    true_edges = {(0, 3), (3, 7), (1, 4), (4, 5), (2, 6)}
    procedures = [("Alg. 1 (proposed) + ANOVA",     "anova", "unrestr"),
                  ("Alg. 1 (proposed) + Discret.",  "disc",  "unrestr"),
                  ("Alg. 1 (proposed) + k-NN",      "knn",   "unrestr"),
                  ("Alg. 1a (Edwards-restr.) + k-NN",      "knn",   "edwards")]
    res = {name: {"f": [], "s": []} for name, _, _ in procedures}
    for _ in range(n_trials):
        S1 = rng.integers(0, 3, size=n); S2 = rng.integers(0, 3, size=n); S3 = rng.integers(0, 3, size=n)
        G1 = rng.standard_normal(n) * 0.8 + S1 * 1.2
        G2 = rng.standard_normal(n) * 0.8 + S2 * 1.0
        G3 = G2 * 0.6 + rng.standard_normal(n) * 0.5
        G4 = rng.standard_normal(n) * 0.8 + S3 * 0.9
        Disease = (rng.random(n) < expit(G1 - 0.5)).astype(int)
        variables = [S1, S2, S3, G1, G2, G3, G4, Disease]
        for name, est_name, kruskal_kind in procedures:
            p = 8; mi = np.zeros((p, p))
            for i in range(p):
                for j in range(i + 1, p):
                    ti, tj = types[i], types[j]; vi, vj = variables[i], variables[j]
                    if ti == "c" and tj == "c":
                        mi[i, j] = mi_knn_cc(vi, vj)
                    elif ti == "d" and tj == "d":
                        mi[i, j] = mi_discrete(vi, vj)
                    else:
                        g_var, d_var = (vi, vj) if ti == "c" else (vj, vi)
                        if est_name == "anova":
                            mi[i, j] = mi_anova(g_var, d_var)
                        elif est_name == "disc":
                            mi[i, j] = mi_discretization(g_var, d_var)
                        else:
                            mi[i, j] = mi_knn_gd(g_var, d_var)
                    mi[j, i] = mi[i, j]
            if kruskal_kind == "unrestr":
                tree = kruskal_unrestricted(mi, types, 0.0)
            else:
                tree = kruskal_edwards(mi, types, 0.0)
            _, _, f1, shd = metrics(true_edges, tree)
            res[name]["f"].append(f1); res[name]["s"].append(shd)
    print(f"  {'Procedure':<32} {'F1':>16} {'SHD':>14}")
    for name, _, _ in procedures:
        print(f"  {name:<32} {np.mean(res[name]['f']):.3f}+-{np.std(res[name]['f']):.3f}   "
              f"{np.mean(res[name]['s']):>5.2f}+-{np.std(res[name]['s']):.2f}")
    print("  (Algorithm 1 -- the proposed pipeline -- works with any MI estimator;")
    print("   only the MST step is new. Edwards' restriction forbids D-G-D paths,")
    print("   i.e. continuous nodes that separate two discrete neighbours.)")


# --- Appendix A.5 sigma calibration ----------------------------------------

def exp_A5_sigma_calibration(n_trials: int = N_TRIALS_SIGMA):
    print("\n" + "=" * 78)
    print("  Appendix A.5  Empirical variance-scale diagnostic for KSG and k-NN")
    print("=" * 78)
    rng = np.random.default_rng(GLOBAL_SEED + 10)
    print("  KSG continuous-continuous on Gaussian pairs:")
    print(f"    {'rho':>6} {'true MI':>10} {'sigma_emp(n=200)':>18} "
          f"{'sigma_emp(n=1500)':>18} {'sigma_emp(n=5000)':>18}")
    for rho in [0.0, 0.3, 0.5, 0.7, 0.9]:
        true_mi = -0.5 * np.log(1 - rho ** 2)
        row = []
        for n in [200, 1500, 5000]:
            ests = []
            for _ in range(n_trials):
                Z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
                ests.append(mi_knn_cc(Z[:, 0], Z[:, 1]))
            row.append(np.sqrt(n * np.var(ests, ddof=1)))
        print(f"    {rho:>6.2f} {true_mi:>10.4f} " + "  ".join(f"{r:>18.4f}" for r in row))


def exp_A5_quantile_diagnostic(n: int = 1500, n_trials: int = N_TRIALS_QUANTILE):
    print("\n" + "=" * 78)
    print("  Appendix A.5  Multi-quantile sub-Gaussian tail diagnostic")
    print(f"  rho=0.5 Gaussian pair, n={n}, {n_trials} trials")
    print("=" * 78)
    rho = 0.5; true_mi = -0.5 * np.log(1 - rho ** 2)
    rng = np.random.default_rng(GLOBAL_SEED + 11)
    ests = []
    for _ in range(n_trials):
        Z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
        ests.append(mi_knn_cc(Z[:, 0], Z[:, 1]))
    dev = np.array(ests) - true_mi
    sigma_emp = np.sqrt(n * dev.var(ddof=1))
    print(f"  sigma_emp = sqrt(n * Var) = {sigma_emp:.4f}")
    print(f"  {'t':>7} {'empirical P':>14} {'Gaussian bound':>18} {'informative':>14}")
    for t in [0.020, 0.030, 0.040, 0.050, 0.100, 0.200]:
        emp = float((np.abs(dev) > t).mean())
        gauss = 2 * math.exp(-n * t * t / (2 * sigma_emp ** 2))
        info = "yes" if gauss <= 1.0 else "vacuous"
        print(f"  {t:>7.3f} {emp:>14.4f} {gauss:>18.4f} {info:>14}")


# --- gamma_swap distribution + bounded-gamma_swap regime --------------------

def exp_gamma_swap_distribution(n: int = N_LARGE_POP_PROXY, n_trials: int = N_TRIALS_GAMMA_SWAP):
    print("\n" + "=" * 78)
    print(f"  gamma_swap distribution on the random-mixed-tree generator")
    print(f"  (n={n} as population proxy, {n_trials} trials)")
    print("=" * 78)
    rng = np.random.default_rng(GLOBAL_SEED + 12)
    swaps = []
    for t in range(n_trials):
        for _ in range(300):
            edges = random_tree(15, rng)
            type_order = ["c"] * 8 + ["d"] * 7
            rng.shuffle(type_order)
            if has_dgd_paths(edges, type_order, 2) >= 2:
                break
        data = generate_data_on_tree(edges, type_order, n, rng, 0.6)
        mi_mat = compute_mi_matrix(data, type_order)
        tree = kruskal_unrestricted(mi_mat, type_order, 0.0)
        gs = gamma_swap_for_tree(mi_mat, list(tree), 15)
        swaps.append(gs)
    swaps = np.array(swaps)
    print(f"  min       = {swaps.min():.4f}")
    print(f"  median    = {np.median(swaps):.4f}")
    print(f"  mean      = {swaps.mean():.4f}")
    print(f"  max       = {swaps.max():.4f}")
    print(f"  IQR       = {np.percentile(swaps, 25):.4f} -- {np.percentile(swaps, 75):.4f}")


def exact_gamma_swap_dgd(effect: float = 2.0, n_cat: int = 3):
    """Exact population MI values for the D1-G-D2 chain where
       D1 ~ Uniform{0,...,n_cat-1},  G | D1=d ~ N(effect*d, 1),
       D2 = population-tercile of G.

    Returns (I_D1G, I_GD2, I_D1D2, gamma_swap) as floats in nats. All values
    are computed by 1-D Gaussian quadrature; no Monte-Carlo proxy is used.
    """
    from scipy.stats import norm
    pd = 1.0 / n_cat

    # marginal density of G is a Gaussian mixture: (1/n_cat) sum_d N(effect*d, 1)
    def pG(g):
        return sum(pd * norm.pdf(g, loc=effect * d, scale=1.0) for d in range(n_cat))

    # H(G_mixture) by 1-D quadrature
    lo, hi = -8.0, effect * (n_cat - 1) + 8.0
    def neg_p_log_p(g):
        v = pG(g)
        return 0.0 if v <= 0 else -v * math.log(v)
    H_G, _ = quad(neg_p_log_p, lo, hi, limit=400)

    # H(G | D1) = (1/n_cat) sum_d H(N(effect*d, 1)) = (1/2) log(2*pi*e)
    H_G_given_D1 = 0.5 * math.log(2 * math.pi * math.e)
    I_D1G = H_G - H_G_given_D1

    # Population terciles: solve F_G(t) = k/n_cat for k=1,...,n_cat-1.
    def F_G(g):
        return sum(pd * norm.cdf(g, loc=effect * d, scale=1.0) for d in range(n_cat))
    from scipy.optimize import brentq
    cuts = [brentq(lambda g: F_G(g) - k / n_cat, lo, hi) for k in range(1, n_cat)]

    # P(D1=d, D2=k) = pd * [Phi(cut_k - effect*d) - Phi(cut_{k-1} - effect*d)]
    P_joint = np.zeros((n_cat, n_cat))
    cuts_ext = [-np.inf] + list(cuts) + [np.inf]
    for d in range(n_cat):
        for k in range(n_cat):
            P_joint[d, k] = pd * (norm.cdf(cuts_ext[k + 1] - effect * d) -
                                   norm.cdf(cuts_ext[k] - effect * d))
    # H(D1, D2)
    eps = 1e-300
    H_D1D2 = -np.sum(P_joint * np.log(P_joint + eps))
    H_D1 = math.log(n_cat)
    H_D2 = math.log(n_cat)  # by construction terciles are equiprobable
    I_D1D2 = H_D1 + H_D2 - H_D1D2

    # I(G; D2) = H(D2) - H(D2 | G); D2 is a deterministic function of G at
    # the population level, so H(D2 | G) = 0 and I(G; D2) = H(D2) = log(n_cat).
    I_GD2 = math.log(n_cat)

    gamma_swap = min(I_D1G, I_GD2) - I_D1D2
    return I_D1G, I_GD2, I_D1D2, gamma_swap


def _gen_chain_block(rng, n, effect=2.0, n_cat=3):
    """One D-G-D triple. Returns (D1, G, D2) arrays."""
    D1 = rng.integers(0, n_cat, size=n)
    G  = rng.standard_normal(n) + effect * D1
    cuts = np.quantile(G, [k / n_cat for k in range(1, n_cat)])
    D2 = sum((G > c).astype(int) for c in cuts)
    return D1, G, D2


def exp_bounded_gamma_swap_regime(n: int = 1500, n_trials: int = N_TRIALS_BOUNDED_GS):
    print("\n" + "=" * 78)
    print("  Bounded-gamma_swap regime")
    print("=" * 78)
    # ---- exact population MI for the 3-node demo --------------------------
    I_DG, I_GD2, I_DD, gs_exact = exact_gamma_swap_dgd(effect=2.0, n_cat=3)
    print(f"  (a) 3-node D-G-D chain, effect=2.0")
    print(f"      exact population I(D1; G)  = {I_DG:.4f} nats   (1-D quadrature)")
    print(f"      exact population I(G; D2)  = {I_GD2:.4f} nats   (= log 3)")
    print(f"      exact population I(D1; D2) = {I_DD:.4f} nats   (closed form via normal CDFs)")
    print(f"      exact gamma_swap          = {gs_exact:.4f} nats")
    sigma_use = 1.2; p_use = 3; delta = 0.1
    n_req = 8 * sigma_use ** 2 * (3 * math.log(p_use) + math.log(2 / delta)) / gs_exact ** 2
    print(f"      Corollary 2 required n at sigma=1.2, p=3, delta=0.1: {int(n_req)}")
    true_edges = {(0, 1), (1, 2)}
    types3 = ["d", "c", "d"]
    correct = 0
    rng_emp = np.random.default_rng(GLOBAL_SEED + 14)
    for _ in range(n_trials):
        D1, G, D2 = _gen_chain_block(rng_emp, n, effect=2.0, n_cat=3)
        mi = compute_mi_matrix([D1, G, D2], types3)
        if kruskal_unrestricted(mi, types3, 0.0) == true_edges:
            correct += 1
    print(f"      empirical recovery at n={n}: {100*correct/n_trials:.0f}% over {n_trials} trials")

    # ---- p=15 bounded-gamma_swap (5 independent D-G-D triples) -------------
    print()
    print(f"  (b) p=15 chain-concatenation generator")
    print(f"      Five independent D-G-D triples, effect=2.0, n_cat=3.")
    print(f"      Population MI is block-diagonal across the five triples")
    print(f"      (between-block I = 0 by independence), so the per-block")
    print(f"      gamma_swap = {gs_exact:.4f} nats from (a) carries over.")
    types15 = []
    true15_edges = set()
    for j in range(5):
        types15 += ["d", "c", "d"]
        a, b, c = 3*j, 3*j + 1, 3*j + 2
        true15_edges |= {(a, b), (b, c)}
    p_use2 = 15
    n_req15 = 8 * sigma_use ** 2 * (3 * math.log(p_use2) + math.log(2 / delta)) / gs_exact ** 2
    print(f"      Corollary 2 required n at sigma=1.2, p=15, delta=0.1: {int(n_req15)}")
    print(f"      (the truth is a 10-edge forest; the MST returns 14 edges,")
    print(f"       so we report whether the 10 true edges are all included.)")
    # Test at n=1500 and at n above the Corollary 2 threshold.
    for n_test, n_label in [(n, f"n={n}"), (max(15000, int(n_req15) + 2000), f"n>={int(n_req15)}")]:
        all_in = 0; f1s = []
        rng_emp = np.random.default_rng(GLOBAL_SEED + 15)
        for _ in range(n_trials):
            data = []
            for _j in range(5):
                D1, G, D2 = _gen_chain_block(rng_emp, n_test, effect=2.0, n_cat=3)
                data += [D1, G, D2]
            mi = compute_mi_matrix(data, types15)
            recovered = kruskal_unrestricted(mi, types15, 0.0)
            if true15_edges.issubset(recovered):
                all_in += 1
            _, _, f1, _ = metrics(true15_edges, recovered)
            f1s.append(f1)
        print(f"      {n_label:>14}:  all 10 true edges present in MST in "
              f"{100*all_in/n_trials:.0f}% of trials, "
              f"F1 = {np.mean(f1s):.3f} +- {np.std(f1s):.3f}")


# --- BH multiple-testing correction ----------------------------------------

def exp_bh_correction():
    print("\n" + "=" * 78)
    print("  Benjamini-Hochberg correction on paired t-test P-values")
    print("=" * 78)
    # P-values transcribed from the manuscript tables (Tables 4, 6, 8, 15).
    # If those tables are re-generated, the values below must be re-synced.
    p_values = [
        ("scaling p=5",        0.0001),
        ("scaling p=8",        0.0001),
        ("scaling p=12",       0.0013),
        ("scaling p=15",       0.0005),
        ("scaling p=20",       0.0273),
        ("scaling p=30",       0.0007),
        ("effect=0.4",         0.0174),
        ("effect=0.6",         0.0005),
        ("effect=0.8",         0.00009),  # reported as < 0.0001
        ("effect=1.0",         0.0001),
        ("effect=1.5",         0.00009),  # reported as < 0.0001
        ("effect=2.0",         0.0001),
        ("hetero MDL vs KSG",  0.8833),
        ("hetero Edw-restr vs Alg1", 0.0039),
        ("k-NN vs MDL n=75",   0.0395),
        ("k-NN vs MDL n=100",  0.2308),
        ("k-NN vs MDL n=200",  0.1885),
        ("k-NN vs MDL n=500",  0.2697),
        ("k-NN vs MDL n=1000", 0.0967),
        ("k-NN vs MDL n=1500", 0.0265),
    ]
    p_sorted = sorted(p_values, key=lambda x: x[1])
    m = len(p_sorted)
    bh = [0.0] * m
    for i in range(m - 1, -1, -1):
        raw = p_sorted[i][1]
        adj = raw * m / (i + 1)
        if i < m - 1:
            adj = min(adj, bh[i + 1])
        bh[i] = min(adj, 1.0)
    print(f"  {'Comparison':<22} {'raw P':>10} {'BH-adj':>10} {'sig 0.05?':>12}")
    for i, (name, p_raw) in enumerate(p_sorted):
        flag = "yes" if bh[i] < 0.05 else "no"
        print(f"  {name:<22} {p_raw:>10.4f} {bh[i]:>10.4f} {flag:>12}")


# ===========================================================================
# Figures
# ===========================================================================

def regenerate_figures(figdir: str | None = None):
    """Regenerate the six figures used in the manuscript."""
    print("\n" + "=" * 78)
    print(f"  Figures: regenerating Fig01, Fig03, Fig04, Fig05, Fig06, Fig08")
    print(f"  Output folder: {PLOTS_DIR if figdir is None else figdir}")
    print("=" * 78)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    try:
        import networkx as nx
        have_nx = True
    except ImportError:
        have_nx = False
    if figdir is None:
        figdir = PLOTS_DIR
    os.makedirs(figdir, exist_ok=True)

    # ---- Fig01: problem illustration ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    def draw_panel(ax, nodes, edges, title, edge_color="black", allowed=True):
        ax.set_xlim(-0.5, 3.5); ax.set_ylim(-1, 1.2); ax.set_axis_off()
        ax.set_title(title, fontsize=11, pad=10)
        for (x, y, lab, kind) in nodes:
            col = "#3498db" if kind == "d" else "#2ecc71"
            ax.add_patch(Circle((x, y), 0.25, facecolor=col, edgecolor="black", lw=1.2))
            ax.text(x, y, lab, ha="center", va="center", fontsize=11, fontweight="bold")
        for (i, j) in edges:
            xi, yi = nodes[i][0], nodes[i][1]; xj, yj = nodes[j][0], nodes[j][1]
            ax.plot([xi, xj], [yi, yj], color=edge_color, lw=2.2, zorder=0)
        ax.text(1.5, -0.75, "ALLOWED" if allowed else "FORBIDDEN",
                ha="center", va="center", fontsize=11, fontweight="bold",
                color=("#2ecc71" if allowed else "#e74c3c"))
    nodes_a = [(0.5, 0.3, r"$G_1$", "c"), (1.5, 0.3, r"$D$",   "d"), (2.5, 0.3, r"$G_2$", "c")]
    draw_panel(axes[0], nodes_a, [(0, 1), (1, 2)], "(a) $G$-$D$-$G$ chain", allowed=True)
    nodes_b = [(0.5, 0.3, r"$D_1$", "d"), (1.5, 0.3, r"$G$",   "c"), (2.5, 0.3, r"$D_2$", "d")]
    draw_panel(axes[1], nodes_b, [(0, 1), (1, 2)], "(b) $D$-$G$-$D$: Edwards forbids", allowed=False)
    nodes_c = [(0.5, 0.3, r"$D_1$", "d"), (1.5, 0.3, r"$G$",   "c"), (2.5, 0.3, r"$D_2$", "d")]
    draw_panel(axes[2], nodes_c, [(0, 1), (1, 2)], "(c) Algorithm 1 (proposed): unrestricted",
               edge_color="#27ae60", allowed=True)
    axes[2].text(1.5, -0.95, "(no type restriction)", ha="center", va="center",
                 fontsize=9, style="italic", color="gray")
    fig.text(0.5, -0.02, "blue = discrete, green = continuous",
             ha="center", fontsize=10, color="gray")
    plt.tight_layout(); plt.savefig(os.path.join(figdir, "Fig01_problem.png"),
                                     dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved {figdir}/Fig01_problem.png")

    # ---- Fig03: accuracy ----
    scenarios = [("Gaussian",   gen_gauss,   true_mi_gauss(3, 1.0),    {"n_cat": 3, "effect": 1.0}),
                 ("Bimodal",    gen_bimodal, true_mi_bimodal(3, 1.5),  {"n_cat": 3, "effect": 1.5}),
                 ("Skewed",     gen_skewed,  true_mi_skewed(3, 1.0),   {"n_cat": 3, "effect": 1.0}),
                 ("Heavy-tail", gen_heavy,   true_mi_heavy(3, 1.0),    {"n_cat": 3, "effect": 1.0})]
    methods = [("ANOVA", mi_anova, "#e74c3c"),
               ("Discret.", mi_discretization, "#3498db"),
               ("$k$-NN", mi_knn_gd, "#2ecc71")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    rng_fig = np.random.default_rng(GLOBAL_SEED + 20)
    for ax, (sname, _, truth, kw) in zip(axes, scenarios):
        results = {name: [] for name, _, _ in methods}
        for _ in range(15):
            seed = int(rng_fig.integers(0, 2 ** 31))
            r2 = np.random.default_rng(seed)
            D = r2.integers(0, kw["n_cat"], size=2000)
            if sname == "Gaussian":
                G = r2.standard_normal(2000) + D * kw["effect"]
            elif sname == "Bimodal":
                c = r2.integers(0, 2, size=2000)
                G = np.where(c == 0,
                              r2.standard_normal(2000) * 0.5 - 1.5 + D * kw["effect"],
                              r2.standard_normal(2000) * 0.5 + 1.5 + D * kw["effect"])
            elif sname == "Skewed":
                G = r2.chisquare(3, size=2000) + D * kw["effect"]
            else:
                G = r2.standard_t(3, size=2000) + D * kw["effect"]
            for name, f, _ in methods:
                results[name].append(f(G, D))
        data = [results[name] for name, _, _ in methods]
        labels = [name for name, _, _ in methods]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
        for patch, (_, _, c) in zip(bp["boxes"], methods):
            patch.set_facecolor(c); patch.set_alpha(0.5)
        ax.axhline(truth, color="k", ls="--", lw=1.2, label=f"true MI = {truth:.3f}")
        ax.set_title(sname); ax.set_ylabel("MI estimate")
        ax.tick_params(axis="x", rotation=20, labelsize=9); ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    plt.tight_layout(); plt.savefig(os.path.join(figdir, "Fig03_accuracy.png"),
                                     dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved {figdir}/Fig03_accuracy.png")

    # ---- Fig04: convergence ----
    sizes = [100, 200, 500, 1000, 2000, 5000]
    n_cat, effect = 3, 1.0
    truth = true_mi_gauss(n_cat, effect)
    rng_fig = np.random.default_rng(GLOBAL_SEED + 21)
    results = {name: {s: [] for s in sizes} for name, _, _ in methods}
    for s in sizes:
        for _ in range(15):
            seed = int(rng_fig.integers(0, 2 ** 31))
            r2 = np.random.default_rng(seed)
            D = r2.integers(0, n_cat, size=s)
            G = r2.standard_normal(s) + D * effect
            for name, f, _ in methods:
                results[name][s].append(f(G, D))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, _, color in methods:
        means = [np.mean(results[name][s]) for s in sizes]
        sds   = [np.std(results[name][s])  for s in sizes]
        ax.errorbar(sizes, means, yerr=sds, label=name, marker="o", capsize=4,
                     color=color, markersize=6, lw=1.5)
    ax.axhline(truth, color="black", ls="--", lw=1.2, label=f"true MI = {truth:.3f}")
    ax.set_xscale("log"); ax.set_xlabel("sample size $n$"); ax.set_ylabel("MI estimate")
    ax.grid(alpha=0.3); ax.legend(loc="best")
    plt.tight_layout(); plt.savefig(os.path.join(figdir, "Fig04_convergence.png"),
                                     dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved {figdir}/Fig04_convergence.png")

    # ---- Fig05: D-G-D chain -- structural comparison only.
    # We previously paired this with a 100%/0% bar chart, but the bar chart
    # carried no visual information: every proposed bar was at 100% and every
    # Edwards bar was at 0% by construction. The reader saw three identical
    # green bars and three invisible red bars. The 100%/0% recovery numbers
    # are in the §4.2 prose; this figure now shows only what each procedure
    # can structurally return.
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(-2.4, 2.0); ax.set_axis_off()
    ax.set_title("$D_1$-$G$-$D_2$ chain: what each procedure can return",
                 fontsize=11, pad=12)
    # ---- True chain (top row) ----
    ax.text(-0.4, 1.45, "True chain / Alg. 1 (proposed) returns:",
            fontsize=10, color="#27ae60", fontweight="bold")
    for (x_, y, lab, kind) in [(0.5, 0.8, r"$D_1$", "d"),
                               (1.5, 0.8, r"$G$",   "c"),
                               (2.5, 0.8, r"$D_2$", "d")]:
        col = "#3498db" if kind == "d" else "#2ecc71"
        ax.add_patch(Circle((x_, y), 0.20, facecolor=col, edgecolor="black", lw=1.0))
        ax.text(x_, y, lab, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.plot([0.7, 1.3], [0.8, 0.8], color="#27ae60", lw=2.5)
    ax.plot([1.7, 2.3], [0.8, 0.8], color="#27ae60", lw=2.5)
    # ---- Edwards (bottom row) ----
    # Label sits ABOVE the bottom row; arc curves DOWN (rad>0) so it does
    # not overlap the label.
    ax.text(-0.4, -0.2, "Alg. 1a (Edwards-restricted) returns:",
            fontsize=10, color="#e74c3c", fontweight="bold")
    for (x_, y, lab, kind) in [(0.5, -0.9, r"$D_1$", "d"),
                               (1.5, -0.9, r"$G$",   "c"),
                               (2.5, -0.9, r"$D_2$", "d")]:
        col = "#3498db" if kind == "d" else "#2ecc71"
        ax.add_patch(Circle((x_, y), 0.20, facecolor=col, edgecolor="black", lw=1.0))
        ax.text(x_, y, lab, ha="center", va="center", fontsize=11, fontweight="bold")
    # Curved arc going DOWNWARD from D_1 to D_2 (positive rad).
    ax.annotate("", xy=(2.3, -0.95), xytext=(0.7, -0.95),
                arrowprops=dict(arrowstyle="-", color="#e74c3c", lw=2.5,
                                connectionstyle="arc3,rad=0.45"))
    ax.text(1.5, -2.05, r"spurious $D_1$-$D_2$ edge that skips $G$",
            ha="center", fontsize=9, color="#e74c3c", style="italic")
    # ---- Legend ----
    ax.scatter([], [], s=130, color="#3498db", edgecolors="black", lw=0.6,
               label="discrete node")
    ax.scatter([], [], s=130, color="#2ecc71", edgecolors="black", lw=0.6,
               label="continuous node")
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    plt.tight_layout(); plt.savefig(os.path.join(figdir, "Fig05_dgd_chain.png"),
                                     dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved {figdir}/Fig05_dgd_chain.png")

    # ---- Fig06: eQTL network (single panel; F1 numbers live in Table 3) ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pos = {"S1": (0.5, 3.0), "S2": (0.5, 2.0), "S3": (0.5, 0.5),
           "G1": (2.5, 3.0), "G2": (2.5, 2.0), "G3": (2.5, 1.2),
           "G4": (2.5, 0.5), "Disease": (4.5, 3.0)}
    types_ = {"S1": "d", "S2": "d", "S3": "d", "G1": "c", "G2": "c",
              "G3": "c", "G4": "c", "Disease": "d"}
    edges_true = [("S1", "G1"), ("G1", "Disease"), ("S2", "G2"),
                   ("G2", "G3"), ("S3", "G4")]
    ax.set_xlim(-0.5, 5.5); ax.set_ylim(-0.5, 4); ax.set_axis_off()
    ax.set_title("Synthetic eQTL network "
                 r"(red: $S_1\,$-$\,G_1\,$-$\,$Disease chain forbidden under Edwards)",
                 fontsize=10)
    for (a, b) in edges_true:
        xa, ya = pos[a]; xb, yb = pos[b]
        col = "#e74c3c" if (a == "S1" and b == "G1") or (a == "G1" and b == "Disease") else "gray"
        lw = 2.6 if col == "#e74c3c" else 1.4
        ax.plot([xa, xb], [ya, yb], color=col, lw=lw, alpha=0.85, zorder=1)
    for name, (x, y) in pos.items():
        col = "#e74c3c" if name == "Disease" else ("#3498db" if types_[name] == "d" else "#2ecc71")
        ax.add_patch(Circle((x, y), 0.22, facecolor=col, edgecolor="black", lw=1.0, zorder=2))
        # short labels fit inside the node; "Disease" is too long and goes outside
        if name == "Disease":
            ax.text(x, y + 0.42, name, ha="center", va="center",
                    fontsize=9, fontweight="bold", zorder=3)
        else:
            ax.text(x, y, name, ha="center", va="center",
                    fontsize=9, fontweight="bold", zorder=3)
    # legend keys (discrete vs continuous) so the colour code is explicit
    ax.scatter([], [], s=140, color="#3498db", edgecolors="black",
               linewidths=0.6, label="discrete node")
    ax.scatter([], [], s=140, color="#2ecc71", edgecolors="black",
               linewidths=0.6, label="continuous node")
    ax.scatter([], [], s=140, color="#e74c3c", edgecolors="black",
               linewidths=0.6, label="binary outcome (Disease)")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    plt.tight_layout(); plt.savefig(os.path.join(figdir, "Fig06_eqtl.png"),
                                     dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved {figdir}/Fig06_eqtl.png")

    # ---- Fig08: breast-cancer forest (Algorithm 1b) ----
    csv_path = _locate_data_file("breastcancer.csv")
    if csv_path is None or not os.path.exists(csv_path):
        print(f"  SKIPPED Fig08: breastcancer.csv not found in script directory, "
              f"cwd, ./data/, ~/Downloads/Chow_liu/, or ~/Desktop/sicore/")
        return
    df = pd.read_csv(csv_path)
    gene_names = df.columns[:-1].tolist()
    genes = df.iloc[:, :-1].values.astype(float)
    class_label = (df["code"] == "case").astype(int).values
    case_m = class_label == 1; ctrl_m = class_label == 0
    pvals = np.array([stats.ttest_ind(genes[case_m, g], genes[ctrl_m, g])[1]
                       for g in range(genes.shape[1])])
    top50 = np.argsort(pvals)[:50]
    mi_class = [mi_knn_gd(genes[:, g], class_label) for g in top50]
    n_top = 50
    mi_gg = np.zeros((n_top, n_top))
    for i in range(n_top):
        for j in range(i + 1, n_top):
            mi_gg[i, j] = mi_continuous(genes[:, top50[i]], genes[:, top50[j]])
            mi_gg[j, i] = mi_gg[i, j]
    p_total = n_top + 1
    types_full = ["c"] * n_top + ["d"]
    mi_mat = np.zeros((p_total, p_total))
    for i in range(n_top):
        for j in range(i + 1, n_top):
            mi_mat[i, j] = mi_gg[i, j]; mi_mat[j, i] = mi_mat[i, j]
    for i in range(n_top):
        mi_mat[i, n_top] = mi_class[i]; mi_mat[n_top, i] = mi_class[i]
    tree = kruskal_unrestricted(mi_mat, types_full, 0.0)
    tree_edges = list(tree)
    if have_nx:
        G_nx = nx.Graph()
        for i in range(p_total):
            G_nx.add_node(i)
        for (a, b) in tree_edges:
            G_nx.add_edge(a, b)
        pos = nx.spring_layout(G_nx, seed=42, k=1.5 / np.sqrt(p_total))
    else:
        angles = np.linspace(0, 2 * np.pi, p_total, endpoint=False)
        pos = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(p_total)}
    fig, ax = plt.subplots(figsize=(11, 9))
    for (a, b) in tree_edges:
        xa, ya = pos[a]; xb, yb = pos[b]
        if a == n_top or b == n_top:
            ax.plot([xa, xb], [ya, yb], color="#e74c3c", lw=2.0, alpha=0.8, zorder=1)
        else:
            ax.plot([xa, xb], [ya, yb], color="gray", lw=0.7, alpha=0.5, zorder=1)
    for i in range(p_total):
        x, y = pos[i]
        if i == n_top:
            ax.scatter([x], [y], s=400, color="#e74c3c", edgecolors="black", lw=1.2, zorder=3)
            ax.text(x, y - 0.06, "CLASS", ha="center", va="top", fontsize=11, fontweight="bold")
        else:
            ax.scatter([x], [y], s=80, color="#3498db", edgecolors="black", lw=0.6, alpha=0.8, zorder=2)
    class_neighbours = [(i, j) for (i, j) in tree_edges if i == n_top or j == n_top]
    if class_neighbours:
        i, j = class_neighbours[0]
        gi = i if j == n_top else j
        xg, yg = pos[gi]
        ax.annotate(gene_names[top50[gi]], xy=(xg, yg), xytext=(xg + 0.12, yg + 0.06),
                    fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
    # No internal title -- the figure caption in the manuscript carries the
    # description (Reviewer 1, point 7).
    ax.set_axis_off(); plt.tight_layout()
    plt.savefig(os.path.join(figdir, "Fig08_forest50.png"),
                 dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved {figdir}/Fig08_forest50.png")


# ===========================================================================
# Main
# ===========================================================================

def run_all():
    """Run every experiment and regenerate every figure."""
    t0 = time.time()
    exp_4_1_mi_accuracy()
    exp_4_2_dgd_sanity()
    exp_4_3_eqtl()
    exp_4_4_scaling()
    exp_4_4_effect_sweep()
    exp_4_4_heterogeneous_with_suzuki()
    exp_4_5_near_tie()
    exp_4_6_shrinking()
    exp_4_7_suzuki_eqtl()
    exp_4_7_suzuki_sweep()
    exp_4_7_adaptive_k()
    exp_4_8_asia()
    exp_4_9_breast_cancer()
    exp_4_10_ablation()
    exp_A5_sigma_calibration()
    exp_A5_quantile_diagnostic()
    exp_gamma_swap_distribution()
    exp_bounded_gamma_swap_regime()
    exp_bh_correction()
    regenerate_figures()
    print(f"\n  Total wall-clock time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run_all()

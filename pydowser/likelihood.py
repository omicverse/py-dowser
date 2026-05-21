"""Pure-Python maximum-likelihood phylogenetics (tractable subset).

dowser's maximum-likelihood route delegates to ``phangorn::optim.pml`` /
IgPhyML.  Reproducing the full GTR+gamma optimiser bit-for-bit in pure
Python is out of scope; this module implements a *tractable* ML engine:

* Felsenstein's pruning algorithm for the tree log-likelihood under the
  Jukes-Cantor (``"JC"``) model, working on compressed site patterns.
* per-branch length optimisation by 1-D golden-section search.
* NNI topology search hill-climbing on log-likelihood.

This gives sensible ML branch lengths and topologies for small B-cell
lineages; it is **not** numerically identical to phangorn (see the README
phylogenetics caveats).  The parsimony route (:func:`buildPratchet`) is the
R-parity reference.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from .parsimony import _nni_neighbours, _topology_order, build_nj_tree
from .phylo import Phylo

__all__ = ["optimise_ml_tree", "tree_loglik"]

_BASES = "ACGT"
_IDX = {b: i for i, b in enumerate(_BASES)}


def _encode_likelihood(seq: str) -> np.ndarray:
    """Encode a sequence into a (n_sites, 4) tip likelihood matrix."""
    amb = {"A": [1, 0, 0, 0], "C": [0, 1, 0, 0], "G": [0, 0, 1, 0],
           "T": [0, 0, 0, 1], "U": [0, 0, 0, 1]}
    out = np.ones((len(seq), 4))
    for i, c in enumerate(seq.upper()):
        out[i] = amb.get(c, [1, 1, 1, 1])
    return out


def _jc_p(t: float) -> np.ndarray:
    """Jukes-Cantor transition probability matrix for branch length ``t``."""
    e = math.exp(-4.0 * t / 3.0)
    same = 0.25 + 0.75 * e
    diff = 0.25 - 0.25 * e
    P = np.full((4, 4), diff)
    np.fill_diagonal(P, same)
    return P


def _compress(tip_lik: Dict[str, np.ndarray], labels: List[str]):
    """Compress identical alignment columns into weighted unique patterns."""
    # stack along (n_tip, n_sites, 4) and find unique site columns
    stacked = np.stack([tip_lik[lab] for lab in labels], axis=0)
    n_tip, n_sites, _ = stacked.shape
    flat = stacked.transpose(1, 0, 2).reshape(n_sites, -1)
    _, inv, counts = np.unique(flat, axis=0, return_inverse=True,
                               return_counts=True)
    n_pat = counts.shape[0]
    comp = {lab: np.zeros((n_pat, 4)) for lab in labels}
    seen = np.zeros(n_pat, dtype=bool)
    for s in range(n_sites):
        p = inv[s]
        if not seen[p]:
            for li, lab in enumerate(labels):
                comp[lab][p] = stacked[li, s]
            seen[p] = True
    return comp, counts.astype(float)


def tree_loglik(tree: Phylo, tip_lik: Dict[str, np.ndarray],
                weights: np.ndarray = None) -> float:
    """Felsenstein pruning log-likelihood of ``tree`` under JC.

    If ``weights`` is given ``tip_lik`` is assumed already compressed to
    unique site patterns with those per-pattern multiplicities.
    """
    n_sites = next(iter(tip_lik.values())).shape[0]
    if weights is None:
        weights = np.ones(n_sites)
    lik: Dict[int, np.ndarray] = {}
    for i, lab in enumerate(tree.tip_label):
        lik[i + 1] = tip_lik[lab]
    elen = {(int(p), int(c)): max(tree.edge_length[k], 1e-9)
            for k, (p, c) in enumerate(tree.edge)}
    order, children, root = _topology_order(tree)
    for node in order:
        kids = children.get(node, ())
        if not kids:
            continue
        acc = np.ones((n_sites, 4))
        for c in kids:
            P = _jc_p(elen[(node, c)])
            acc = acc * (lik[c] @ P.T)
        lik[node] = acc
    site_lik = lik[root] @ np.full(4, 0.25)
    site_lik = np.clip(site_lik, 1e-300, None)
    return float((np.log(site_lik) * weights).sum())


def _optimise_branch(tree: Phylo, tip_lik, weights, edge_idx: int) -> None:
    """Golden-section optimise one branch length in place."""
    def f(x):
        old = tree.edge_length[edge_idx]
        tree.edge_length[edge_idx] = x
        ll = tree_loglik(tree, tip_lik, weights)
        tree.edge_length[edge_idx] = old
        return -ll

    lo, hi = 1e-6, 1.5
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(18):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    tree.edge_length[edge_idx] = (a + b) / 2


def optimise_ml_tree(seqs: Dict[str, str], labels: List[str],
                     sub_model: str = "JC", max_iter: int = 8,
                     seed: int = 1, verbose: bool = False) -> Phylo:
    """Build and optimise an ML tree (NJ start + NNI + branch optimisation).

    Returns an unrooted binary :class:`Phylo` with ML branch lengths.  Only
    the JC model is implemented; ``sub_model`` is accepted for API parity.
    Site patterns are compressed for speed.
    """
    raw_lik = {lab: _encode_likelihood(seqs[lab]) for lab in labels}
    tip_lik, weights = _compress(raw_lik, labels)
    n = len(labels)
    # NJ starting tree on JC distances
    D = np.zeros((n, n))
    for i in range(n):
        si = seqs[labels[i]]
        for j in range(i + 1, n):
            sj = seqs[labels[j]]
            diff = sum(a != b for a, b in zip(si, sj)
                       if a in _IDX and b in _IDX)
            tot = sum(1 for a, b in zip(si, sj) if a in _IDX and b in _IDX)
            p = diff / tot if tot else 0.0
            p = min(p, 0.74)
            d = -0.75 * math.log(1 - 4 * p / 3) if p > 0 else 0.0
            D[i, j] = D[j, i] = d
    tree = build_nj_tree(labels, D)
    tree.edge_length = np.clip(tree.edge_length, 1e-6, None)

    def optimise_branches(t, rounds=2):
        for _ in range(rounds):
            for k in range(t.n_edge):
                _optimise_branch(t, tip_lik, weights, k)
        return t

    tree = optimise_branches(tree)
    best_ll = tree_loglik(tree, tip_lik, weights)
    if verbose:
        print(f"ML start logL = {best_ll:.3f}")

    for it in range(max_iter):
        improved = False
        for nb in _nni_neighbours(tree):
            nb.edge_length = np.clip(nb.edge_length, 1e-6, None)
            # quick score without re-optimising (cheap filter)
            ll0 = tree_loglik(nb, tip_lik, weights)
            if ll0 < best_ll - 5.0:
                continue
            optimise_branches(nb, rounds=1)
            ll = tree_loglik(nb, tip_lik, weights)
            if ll > best_ll + 1e-4:
                best_ll, tree = ll, nb
                improved = True
        if not improved:
            break
        optimise_branches(tree, rounds=1)
        best_ll = tree_loglik(tree, tip_lik, weights)
        if verbose:
            print(f"  ML it {it}: logL = {best_ll:.3f}")
    return tree

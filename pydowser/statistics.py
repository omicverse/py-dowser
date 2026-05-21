"""Measurable-evolution statistics -- the root-to-tip ``correlationTest``.

Port of dowser's ``correlationTest`` / ``runCorrelationTest`` (``R/
Statistics.R``).  These implement the date-randomisation permutation test
for temporal signal of Murray et al. (2016) as applied to B-cell lineages
by Hoehn et al. (2021): the correlation between each tip's root-to-tip
divergence and its sampling time is compared with a permutation null in
which time labels are shuffled among (clustered) tips.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from .clones import AirrClone
from .phylo import Phylo
from .trees import di2multi, getDivergence, resolvePolytomies

__all__ = ["correlationTest", "runCorrelationTest"]


def _lm(x: np.ndarray, y: np.ndarray):
    """Simple OLS regression; return (slope, intercept)."""
    n = len(x)
    if n < 2 or np.allclose(x, x[0]):
        return float("nan"), float("nan")
    xm, ym = x.mean(), y.mean()
    sxx = np.sum((x - xm) ** 2)
    sxy = np.sum((x - xm) * (y - ym))
    slope = sxy / sxx
    intercept = ym - slope * xm
    return float(slope), float(intercept)


def _subtrees_tipsets(tree: Phylo):
    """Return tip-label sets of every monophyletic clade with >1 tip."""
    out = []
    for node in tree.internal_indices():
        tips = tree.descendant_tips(node)
        if len(tips) > 1:
            out.append(tips)
    return out


def runCorrelationTest(phy: Phylo, clone: AirrClone, permutations: int,
                       minlength: float = 0.001, polyresolve: bool = True,
                       permutation: str = "clustered", time: str = "time",
                       sequence: str = "sequence_id",
                       germline: str = "Germline", verbose: bool = False,
                       alternative: str = "greater", seed: int = 1) -> dict:
    """Root-to-tip regression permutation test for a single clone.

    Port of dowser's ``runCorrelationTest``.  See :func:`correlationTest`
    for the meaning of the returned statistics.
    """
    data = clone.data.copy()
    rng = np.random.default_rng(seed)
    if verbose:
        print(f"Analyzing clone: {clone.clone}")

    if polyresolve:
        phy = resolvePolytomies(phy, clone, minlength, time=time,
                                sequence=sequence, germline=germline,
                                verbose=verbose)

    # drop germline tip
    if germline in phy.tip_label:
        phy = _drop_tip(phy, germline)
    tips = phy.tip_label

    dates = dict(zip(data[sequence].astype(str), data[time]))
    date_vec = np.asarray([dates[t] for t in tips], dtype=float)

    nan = float("nan")
    if len(np.unique(date_vec)) == 1:
        warnings.warn(f"Only one timepoint cluster in clone {clone.clone}")
        return {"correlation": nan, "clone": clone, "tree": phy,
                "random": nan, "random_correlation": nan, "p_gt": nan,
                "p_lt": nan, "nposs": nan, "nclust": nan, "p": nan,
                "min_p": nan, "slope": nan, "intercept": nan}

    divergence = getDivergence(phy, minlength)
    data = data.copy()
    data["divergence"] = data[sequence].astype(str).map(divergence)

    # cluster assignment
    cl = np.arange(1, len(tips) + 1)
    tip_index = {t: i for i, t in enumerate(tips)}
    if permutation == "clustered":
        for tipset in _subtrees_tipsets(phy):
            ds = {dates[t] for t in tipset}
            if len(ds) == 1:
                idxs = [tip_index[t] for t in tipset]
                cl[idxs] = cl[idxs[0]]
        # renumber clusters 1..k
        _, cl = np.unique(cl, return_inverse=True)
        cl = cl + 1
    cluster = dict(zip(tips, cl))
    data["cluster"] = data[sequence].astype(str).map(cluster)

    # ordered timepoint per cluster
    ctimes = []
    for c in sorted(pd.unique(data["cluster"])):
        ts = pd.unique(data.loc[data["cluster"] == c, time])
        ctimes.append(ts[0])
    ctimes = np.asarray(ctimes, dtype=float)

    true = float(np.corrcoef(data[time].astype(float),
                             data["divergence"].astype(float))[0, 1])
    cluster_arr = data["cluster"].to_numpy()
    div_arr = data["divergence"].to_numpy(dtype=float)
    random = np.zeros(permutations)
    for i in range(permutations):
        perm = ctimes[rng.permutation(len(ctimes))]
        rt = perm[cluster_arr - 1]
        random[i] = np.corrcoef(div_arr, rt)[0, 1]

    nclust = len(np.unique(cl))
    _, tcounts = np.unique(ctimes, return_counts=True)
    nposs = math.exp(math.lgamma(nclust + 1)
                     - sum(math.lgamma(c + 1) for c in tcounts))

    minp = 0.5 if alternative == "two.sided" else 1.0
    slope, intercept = _lm(data[time].to_numpy(dtype=float), div_arr)

    full = np.append(random, true)
    p_gt = (np.sum(true < full) + np.sum(true == full) * 0.5) / len(full)
    p_lt = (np.sum(true > full) + np.sum(true == full) * 0.5) / len(full)
    clone = clone.copy()
    clone.data = data
    return {
        "correlation": true, "clone": clone, "tree": phy,
        "random": random, "random_correlation": float(random.mean()),
        "p_gt": p_gt, "p_lt": p_lt, "nposs": nposs, "nclust": nclust,
        "p": float(np.mean(true <= full)),
        "min_p": max(minp / nposs, minp / (permutations + 1)),
        "slope": slope, "intercept": intercept,
    }


def _drop_tip(tree: Phylo, label: str) -> Phylo:
    """Remove a single tip from a tree, suppressing degree-2 nodes (``ape::drop.tip``)."""
    if label not in tree.tip_label:
        return tree
    tip = tree.get_tip_index(label)
    # working edge list (directed parent->child) and lengths
    edges = [[int(p), int(c)] for p, c in tree.edge]
    lens = [float(x) for x in tree.edge_length]
    # remove the edge leading to the dropped tip
    keep = [(e, w) for e, w in zip(edges, lens) if e[1] != tip]
    edges = [e for e, w in keep]
    lens = [w for e, w in keep]

    # iteratively suppress internal nodes with exactly one child (or that
    # have become tips), and the root if it has a single child
    changed = True
    while changed:
        changed = False
        # degree of each node as a parent / child
        n_children: dict = {}
        is_child_of: dict = {}
        for (p, c) in edges:
            n_children[p] = n_children.get(p, 0) + 1
            is_child_of[c] = p
        all_internal = {p for p, c in edges} - {c for p, c in edges
                                                if c <= tree.n_tip}
        for node in list(n_children.keys()):
            if node <= tree.n_tip:
                continue
            if n_children.get(node, 0) == 1:
                # suppress this node: connect its parent to its child
                child = next(c for (p, c) in edges if p == node)
                cw = next(w for (p, c), w in zip(edges, lens) if p == node)
                if node in is_child_of:
                    par = is_child_of[node]
                    pw = next(w for (p, c), w in zip(edges, lens)
                              if p == par and c == node)
                    new_edges, new_lens = [], []
                    for (p, c), w in zip(edges, lens):
                        if p == node:
                            continue
                        if p == par and c == node:
                            new_edges.append([par, child])
                            new_lens.append(pw + cw)
                        else:
                            new_edges.append([p, c])
                            new_lens.append(w)
                    edges, lens = new_edges, new_lens
                else:
                    # node is the root with one child -> child becomes root
                    new_edges, new_lens = [], []
                    for (p, c), w in zip(edges, lens):
                        if p == node:
                            continue
                        new_edges.append([p, c])
                        new_lens.append(w)
                    edges, lens = new_edges, new_lens
                changed = True
                break

    # renumber nodes: tips 1..n_tip, internals contiguous after
    new_tips = [t for t in tree.tip_label if t != label]
    old_tip_ids = [tree.get_tip_index(t) for t in new_tips]
    nodemap = {old: i + 1 for i, old in enumerate(old_tip_ids)}
    internal = sorted({p for p, c in edges} | {c for p, c in edges
                      if c > tree.n_tip})
    nxt = len(new_tips) + 1
    for n in internal:
        if n not in nodemap:
            nodemap[n] = nxt
            nxt += 1
    out_edges = [[nodemap[p], nodemap[c]] for p, c in edges]
    new = Phylo(edge=out_edges, edge_length=lens, tip_label=new_tips,
                Nnode=len({p for p, c in out_edges}), name=tree.name)
    new.edge_type = tree.edge_type
    new.tree_method = tree.tree_method
    return new


def correlationTest(clones: pd.DataFrame, permutations: int = 1000,
                    minlength: float = 0.001, perm_type: str = "clustered",
                    time: str = "time", sequence: str = "sequence_id",
                    germline: str = "Germline", verbose: bool = False,
                    polyresolve: bool = True, alternative: str = "greater",
                    storeTree: bool = False, seed: int = 1) -> pd.DataFrame:
    """Date-randomisation test for temporal signal across a set of trees.

    Faithful port of dowser's ``correlationTest``.  For every clone the
    correlation between root-to-tip divergence and sampling time is
    computed and compared with a permutation null (time labels shuffled
    among single-timepoint clades).

    Returns the input data frame with added per-clone columns: ``slope``,
    ``p``, ``correlation``, ``random_correlation``, ``min_p``, ``nposs``,
    ``nclust``, ``intercept`` (and ``p_gt`` / ``p_lt`` when
    ``alternative='two.sided'``).
    """
    if "trees" not in clones.columns:
        raise ValueError("clones must have trees column!")
    for _, row in clones.iterrows():
        if time not in row["data"].data.columns:
            raise ValueError(
                f"Time column {time} not found in clone object "
                "(must be a trait value in formatClones)")
        if not pd.api.types.is_numeric_dtype(row["data"].data[time]):
            raise ValueError(
                f"Time column {time} contains non-numeric values")

    results = []
    for i, (_, row) in enumerate(clones.iterrows()):
        results.append(runCorrelationTest(
            row["trees"], row["data"], permutations, minlength=minlength,
            polyresolve=polyresolve, permutation=perm_type, time=time,
            sequence=sequence, germline=germline, verbose=verbose,
            alternative=alternative, seed=seed + i))

    out = clones.copy().reset_index(drop=True)
    out["data"] = [r["clone"] for r in results]
    if storeTree:
        out["test_trees"] = [r["tree"] for r in results]
    cols = ["slope", "p", "correlation", "random_correlation", "min_p",
            "nposs", "nclust", "intercept"]
    if alternative == "two.sided":
        cols += ["p_gt", "p_lt"]
    for col in cols:
        out[col] = [r[col] for r in results]
    return out

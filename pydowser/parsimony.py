"""Pure-Python maximum-parsimony engine.

Provides the algorithmic core that dowser delegates to ``phangorn``:

* :func:`fitch_score`           -- Fitch downpass parsimony score on an
                                   unrooted/rooted tree.
* :func:`sankoff_score`         -- Sankoff (weighted) parsimony score.
* :func:`pratchet`              -- parsimony-ratchet heuristic search for a
                                   most-parsimonious topology (NNI + ratchet
                                   perturbations) -- analogue of
                                   ``phangorn::pratchet``.
* :func:`acctran`               -- ACCTRAN branch-length assignment.
* :func:`ancestral_pars`        -- ancestral state reconstruction (MPR /
                                   ACCTRAN) returning per-node sequences.
* :func:`fitch_states`          -- discrete-trait Fitch reconstruction used
                                   by the SP / PS / SC switch tests.

Sequences are encoded over the DNA alphabet using IUPAC ambiguity sets so
that ``N`` and gaps contribute zero parsimony cost, matching phangorn.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .phylo import Phylo

__all__ = [
    "fitch_score", "sankoff_score", "pratchet", "acctran",
    "ancestral_pars", "fitch_states", "build_nj_tree",
]

# DNA alphabet -- bitmask encoding (A=1, C=2, G=4, T=8)
_DNA = "ACGT"
_BIT = {"A": 1, "C": 2, "G": 4, "T": 8}
_IUPAC = {
    "A": 1, "C": 2, "G": 4, "T": 8, "U": 8,
    "R": 5, "Y": 10, "S": 6, "W": 9, "K": 12, "M": 3,
    "B": 14, "D": 13, "H": 11, "V": 7, "N": 15, "-": 15, ".": 15, "?": 15,
}
_BIT_TO_BASE = {1: "A", 2: "C", 4: "G", 8: "T"}


def encode_dna(seq: str) -> np.ndarray:
    """Encode a DNA string into a uint8 bitmask array (IUPAC-aware)."""
    return np.asarray([_IUPAC.get(c.upper(), 15) for c in seq], dtype=np.uint8)


def decode_bits(bits: int) -> str:
    """Decode a single bitmask into an IUPAC character (sorted bases)."""
    bases = [b for b in _DNA if _BIT[b] & bits]
    if not bases:
        return "N"
    if len(bases) == 1:
        return bases[0]
    # map a set of bases to its IUPAC code
    for code, mask in _IUPAC.items():
        if mask == bits and code in "RYSWKMBDHVN":
            return code
    return "N"


# ----------------------------------------------------------------------
def _site_patterns(seqs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Compress an alignment into unique site patterns + their weights."""
    mat = np.vstack(seqs)               # (n_taxa, n_sites)
    # unique columns
    cols, inv, counts = np.unique(mat, axis=1, return_inverse=True,
                                  return_counts=True)
    return cols, counts


# ----------------------------------------------------------------------
def fitch_score(tree: Phylo, seqs: Dict[str, np.ndarray]) -> int:
    """Return the Fitch parsimony score of ``tree`` for ``seqs``.

    ``seqs`` maps tip label -> bitmask-encoded sequence array.  Polytomies
    are handled directly (the Fitch update generalises to any number of
    children).
    """
    tip_labels = tree.tip_label
    mat = np.vstack([seqs[t] for t in tip_labels])      # (n_tip, n_sites)
    cols, weights = np.unique(mat, axis=1, return_inverse=False,
                              return_counts=True)
    n_sites = cols.shape[1]

    # node state arrays
    n_total = tree.n_tip + tree.Nnode
    state = np.zeros((n_total + 1, n_sites), dtype=np.uint8)
    for i, t in enumerate(tip_labels):
        state[i + 1] = cols[i]

    score = np.zeros(n_sites, dtype=np.int64)
    for node in tree.postorder():
        if tree.is_tip(node):
            continue
        kids = tree.children(node)
        if not kids:
            continue
        acc = state[kids[0]].astype(np.uint8).copy()
        for c in kids[1:]:
            cs = state[c]
            inter = acc & cs
            union = acc | cs
            empty = inter == 0
            score[empty] += weights[empty]
            acc = np.where(empty, union, inter).astype(np.uint8)
        state[node] = acc
    return int((score * weights).sum() if False else score.sum())


def _topology_order(tree):
    """Return (postorder internal nodes, children-dict) for fast scoring."""
    edge = tree.edge
    n_tip = tree.n_tip
    children: Dict[int, List[int]] = {}
    parents = set()
    childset = set()
    for p, c in edge:
        p, c = int(p), int(c)
        children.setdefault(p, []).append(c)
        parents.add(p)
        childset.add(c)
    root = next(iter(parents - childset))
    # iterative postorder
    order: List[int] = []
    stack = [(root, False)]
    while stack:
        node, done = stack.pop()
        if done:
            if node > n_tip:
                order.append(node)
            continue
        stack.append((node, True))
        for ch in children.get(node, ()):
            stack.append((ch, False))
    return order, children, root


def _fitch_score_weighted(tree, cols, weights):
    """Internal: Fitch score from compressed patterns (fast path)."""
    n_sites = cols.shape[1]
    n_total = tree.n_tip + tree.Nnode
    n_tip = tree.n_tip
    state = np.zeros((n_total + 1, n_sites), dtype=np.uint8)
    state[1:n_tip + 1] = cols
    score = np.zeros(n_sites, dtype=np.int64)
    order, children, _ = _topology_order(tree)
    for node in order:
        kids = children.get(node, ())
        if not kids:
            continue
        acc = state[kids[0]].copy()
        for c in kids[1:]:
            cs = state[c]
            inter = acc & cs
            empty = inter == 0
            score[empty] += 1
            acc = np.where(empty, acc | cs, inter).astype(np.uint8)
        state[node] = acc
    return int((score * weights).sum())


# ----------------------------------------------------------------------
def sankoff_score(tree: Phylo, seqs: Dict[str, np.ndarray],
                  cost: Optional[np.ndarray] = None) -> int:
    """Weighted (Sankoff) parsimony score over a 4-state DNA alphabet.

    ``cost`` is a ``4x4`` substitution cost matrix; the default is the
    unit-cost matrix, in which case the result equals the Fitch score.
    """
    if cost is None:
        return fitch_score(tree, seqs)
    tip_labels = tree.tip_label
    mat = np.vstack([seqs[t] for t in tip_labels])
    cols, weights = np.unique(mat, axis=1, return_counts=True)
    n_sites = cols.shape[1]
    n_total = tree.n_tip + tree.Nnode
    INF = 1e9
    cost_arr = np.zeros((n_total + 1, n_sites, 4))
    for i in range(tree.n_tip):
        for s in range(n_sites):
            bits = int(cols[i, s])
            for k in range(4):
                cost_arr[i + 1, s, k] = 0.0 if (bits & (1 << k)) else INF
    for node in tree.postorder():
        if tree.is_tip(node):
            continue
        for c in tree.children(node):
            child = cost_arr[c]                  # (n_sites, 4)
            # min over child states of cost[from,to] + child cost
            trans = cost[:, :, None] if cost.ndim == 3 else cost
            best = np.min(cost[None, :, :] + child[:, None, :], axis=2)
            cost_arr[node] += best
    root = tree.root_node()
    site_min = cost_arr[root].min(axis=1)
    return int(np.round((site_min * weights).sum()))


# ----------------------------------------------------------------------
def build_nj_tree(labels: List[str], dist: np.ndarray) -> Phylo:
    """Build a neighbour-joining tree (Saitou & Nei 1987).

    Used as the starting topology for the parsimony search.  Returns a
    rooted binary :class:`Phylo` (rooted arbitrarily; topology only).
    """
    n = len(labels)
    if n == 2:
        return Phylo(edge=[[3, 1], [3, 2]],
                     edge_length=[dist[0, 1] / 2, dist[0, 1] / 2],
                     tip_label=list(labels), Nnode=1)
    # active node bookkeeping
    nodes = list(range(1, n + 1))           # current node ids
    D = dist.astype(float).copy()
    next_internal = n + 1
    edges: List[List[int]] = []
    elen: List[float] = []
    active = list(range(n))                 # indices into D / nodes

    while len(active) > 2:
        m = len(active)
        sub = D[np.ix_(active, active)]
        r = sub.sum(axis=1)
        Q = (m - 2) * sub - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        if i > j:
            i, j = j, i
        ai, aj = active[i], active[j]
        d_ij = sub[i, j]
        di = 0.5 * d_ij + (r[i] - r[j]) / (2 * (m - 2))
        di = max(di, 0.0)
        dj = max(d_ij - di, 0.0)
        new = next_internal
        next_internal += 1
        edges.append([new, nodes[ai]])
        elen.append(di)
        edges.append([new, nodes[aj]])
        elen.append(dj)
        # new distance row
        new_d = 0.5 * (D[active, ai] + D[active, aj] - d_ij)
        # extend D
        idx = D.shape[0]
        D = np.pad(D, ((0, 1), (0, 1)), constant_values=0.0)
        for k, a in enumerate(active):
            D[idx, a] = new_d[k]
            D[a, idx] = new_d[k]
        nodes.append(new)
        active = [a for a in active if a not in (ai, aj)]
        active.append(idx)

    # last two
    ai, aj = active
    new = next_internal
    edges.append([new, nodes[ai]])
    elen.append(max(D[ai, aj] / 2, 0.0))
    edges.append([new, nodes[aj]])
    elen.append(max(D[ai, aj] / 2, 0.0))

    # renumber so internal nodes are n+1 .. n+Nnode contiguous
    return _renumber(edges, elen, labels)


def _renumber(edges: List[List[int]], elen: List[float],
              labels: List[str]) -> Phylo:
    """Renumber an edge list so tips are 1..n_tip, internals n_tip+1.."""
    n_tip = len(labels)
    internal = sorted({p for p, c in edges if p > n_tip}
                      | {c for p, c in edges if c > n_tip})
    remap = {old: n_tip + 1 + k for k, old in enumerate(internal)}
    for i in range(1, n_tip + 1):
        remap[i] = i
    new_edges = [[remap[p], remap[c]] for p, c in edges]
    return Phylo(edge=new_edges, edge_length=elen, tip_label=labels,
                 Nnode=len(internal))


# ----------------------------------------------------------------------
def _light_copy(tree: Phylo) -> Phylo:
    """Topology-only copy of a tree (shares the ``nodes`` dict -- fast)."""
    out = Phylo.__new__(Phylo)
    out.edge = tree.edge.copy()
    out.edge_length = tree.edge_length.copy()
    out.tip_label = tree.tip_label
    out.Nnode = tree.Nnode
    out.name = tree.name
    out.tree_method = tree.tree_method
    out.edge_type = tree.edge_type
    out.seq = tree.seq
    out.node_label = tree.node_label
    out.nodes = tree.nodes
    return out


def _nni_neighbours(tree: Phylo) -> List[Phylo]:
    """Yield all nearest-neighbour-interchange topologies of ``tree``.

    Topology-only (light) copies are returned so the search avoids deep
    copies of per-node reconstructed sequences.
    """
    neigh: List[Phylo] = []
    for k, (p, c) in enumerate(tree.edge):
        p, c = int(p), int(c)
        if tree.is_tip(c) or tree.is_tip(p):
            continue
        # internal edge p->c.  swap one subtree of p (other than c) with
        # one subtree of c.
        p_kids = [x for x in tree.children(p) if x != c]
        c_kids = tree.children(c)
        if not p_kids or len(c_kids) < 2:
            continue
        a = p_kids[0]
        for b in c_kids:
            new = _light_copy(tree)
            ed = new.edge
            # a currently child of p, b currently child of c -> swap
            ia = np.where((ed[:, 0] == p) & (ed[:, 1] == a))[0][0]
            ib = np.where((ed[:, 0] == c) & (ed[:, 1] == b))[0][0]
            ed[ia, 1] = b
            ed[ib, 1] = a
            neigh.append(new)
    return neigh


def _spr_neighbours(tree: Phylo, max_moves: int = 100000):
    """Yield subtree-prune-regraft topologies of ``tree``.

    SPR explores a wider neighbourhood than NNI and helps the parsimony
    search escape local optima.  ``max_moves`` caps the count only for
    very large trees.
    """
    edges = [(int(p), int(c)) for p, c in tree.edge]
    parent = {c: p for p, c in edges}
    children: Dict[int, List[int]] = {}
    for p, c in edges:
        children.setdefault(p, []).append(c)
    root = tree.root_node()
    elen = {(int(p), int(c)): tree.edge_length[k]
            for k, (p, c) in enumerate(tree.edge)}

    def subtree(node):
        s = {node}
        stack = [node]
        while stack:
            n = stack.pop()
            for ch in children.get(n, ()):
                s.add(ch)
                stack.append(ch)
        return s

    out: List[Phylo] = []
    prune_nodes = [n for n in range(1, tree.n_tip + tree.Nnode + 1)
                   if n in parent and parent[n] != root]
    for prune in prune_nodes:
        gp = parent.get(parent[prune])
        if gp is None:
            continue
        sub = subtree(prune)
        forbidden = sub | {parent[prune]}
        # candidate regraft edges
        regraft_edges = [(p, c) for p, c in edges
                         if p not in forbidden and c not in forbidden]
        for (rp, rc) in regraft_edges:
            new = _light_copy(tree)
            ed = [list(e) for e in new.edge.tolist()]
            ln = list(new.edge_length)
            par = parent[prune]
            sib = [c for c in children[par] if c != prune]
            if len(sib) != 1:
                continue
            sib = sib[0]
            grand = parent[par]
            # remove par; connect grand->sib
            keep_e, keep_l = [], []
            for (p, c), w in zip(ed, ln):
                if (p == par and c == prune):
                    continue
                if (p == par and c == sib):
                    continue
                if (p == grand and c == par):
                    continue
                keep_e.append([p, c])
                keep_l.append(w)
            sib_w = elen.get((par, sib), 0.0) + elen.get((grand, par), 0.0)
            keep_e.append([grand, sib])
            keep_l.append(sib_w)
            # regraft on edge (rp, rc): insert par between them
            final_e, final_l = [], []
            for (p, c), w in zip(keep_e, keep_l):
                if p == rp and c == rc:
                    final_e.append([rp, par])
                    final_l.append(w / 2)
                    final_e.append([par, rc])
                    final_l.append(w / 2)
                else:
                    final_e.append([p, c])
                    final_l.append(w)
            final_e.append([par, prune])
            final_l.append(elen.get((parent[prune], prune), 0.1))
            try:
                nt = _renumber(final_e, final_l, tree.tip_label)
            except Exception:  # noqa: BLE001
                continue
            out.append(nt)
            if len(out) >= max_moves:
                return out
    return out


def _ratchet_perturb(cols: np.ndarray, weights: np.ndarray,
                     rng: random.Random) -> np.ndarray:
    """Return bootstrap-reweighted pattern weights for ratchet perturbation."""
    n = int(weights.sum())
    new_w = np.zeros_like(weights)
    probs = weights / weights.sum()
    picks = rng.choices(range(len(weights)), weights=probs, k=n)
    for p in picks:
        new_w[p] += 1
    return new_w


def pratchet(seqs: Dict[str, np.ndarray], labels: List[str],
             maxit: int = 100, k: int = 10, trace: int = 0,
             seed: int = 1, start_tree: Optional[Phylo] = None) -> Phylo:
    """Parsimony-ratchet search for a most-parsimonious topology.

    Faithful in spirit to ``phangorn::pratchet``: starting from a
    neighbour-joining tree it performs hill-climbing NNI to a local
    parsimony optimum, then repeatedly perturbs the data (bootstrap
    re-weighting of site patterns), re-optimises, and keeps the best
    topology.  Search stops after ``k`` consecutive non-improving
    iterations or ``maxit`` total iterations.

    Returns an unrooted binary :class:`Phylo`; branch lengths are set by
    :func:`acctran` downstream.
    """
    rng = random.Random(seed)
    mat = np.vstack([seqs[t] for t in labels])
    cols, weights = np.unique(mat, axis=1, return_counts=True)

    # starting tree
    if start_tree is None:
        # Hamming distance matrix for NJ
        n = len(labels)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                a, b = mat[i], mat[j]
                diff = np.sum((a & b) == 0)
                D[i, j] = D[j, i] = diff
        tree = build_nj_tree(labels, D)
    else:
        tree = start_tree.copy()

    def opt(t, w, use_spr=True):
        """Hill-climb NNI (then SPR) on tree t under weights w."""
        best_t = t
        best_s = _fitch_score_weighted(t, cols, w)
        improved = True
        while improved:
            improved = False
            for nb in _nni_neighbours(best_t):
                s = _fitch_score_weighted(nb, cols, w)
                if s < best_s:
                    best_s, best_t = s, nb
                    improved = True
            if not improved and use_spr:
                # NNI converged -- widen the search with SPR
                for nb in _spr_neighbours(best_t):
                    s = _fitch_score_weighted(nb, cols, w)
                    if s < best_s:
                        best_s, best_t = s, nb
                        improved = True
                        break
        return best_t, best_s

    # full NNI+SPR optimisation of the starting tree
    tree, best_score = opt(tree, weights, use_spr=True)
    if trace:
        print(f"pratchet start score = {best_score}")

    # ratchet: perturb data weights, re-optimise (NNI-only -- fast), then
    # polish the candidate with NNI+SPR before comparing.
    no_improve = 0
    it = 0
    while no_improve < k and it < maxit:
        it += 1
        w_pert = _ratchet_perturb(cols, weights, rng)
        pert_tree, _ = opt(tree, w_pert, use_spr=False)
        cand, cand_score = opt(pert_tree, weights, use_spr=False)
        if cand_score <= best_score:
            cand, cand_score = opt(cand, weights, use_spr=True)
        if cand_score < best_score:
            best_score, tree = cand_score, cand
            no_improve = 0
            if trace:
                print(f"  it {it}: improved -> {best_score}")
        else:
            no_improve += 1
    # final SPR polish
    tree, _ = opt(tree, weights, use_spr=True)
    return tree


# ----------------------------------------------------------------------
def acctran(tree: Phylo, seqs: Dict[str, np.ndarray]) -> Phylo:
    """Assign branch lengths via accelerated transformation (ACCTRAN).

    Reconstructs ancestral sequences with the Fitch algorithm, resolving
    ambiguity so substitutions occur as early (close to the root) as
    possible, then counts substitutions on each branch.  Branch lengths
    sum exactly to the Fitch parsimony score.  Returns ``tree`` with
    ``edge_length`` set in raw substitution counts (divide by sequence
    length for genetic distance, as dowser does).
    """
    final = _fitch_ancestral(tree, seqs)
    out = tree.copy()
    elen = np.zeros(out.n_edge)
    for k, (p, c) in enumerate(out.edge):
        ps = final[int(p)]
        cs = final[int(c)]
        # a substitution occurs where the resolved bitmasks share no base
        elen[k] = int(np.sum((ps & cs) == 0))
    out.edge_length = elen
    return out


def _fitch_ancestral(tree: Phylo,
                     seqs: Dict[str, np.ndarray]) -> Dict[int, np.ndarray]:
    """Fitch down+up pass; return a single resolved sequence per node."""
    tip_labels = tree.tip_label
    n_sites = len(seqs[tip_labels[0]])
    n_total = tree.n_tip + tree.Nnode
    down = {i: np.zeros(n_sites, dtype=np.uint8) for i in range(1, n_total + 1)}
    for i, t in enumerate(tip_labels):
        down[i + 1] = seqs[t].astype(np.uint8).copy()
    # downpass
    for node in tree.postorder():
        if tree.is_tip(node):
            continue
        kids = tree.children(node)
        if not kids:
            continue
        acc = down[kids[0]].copy()
        for c in kids[1:]:
            cs = down[c]
            inter = acc & cs
            acc = np.where(inter == 0, acc | cs, inter).astype(np.uint8)
        down[node] = acc
    # uppass -- ACCTRAN resolution.  Internal nodes inherit the parent's
    # state where compatible with their Fitch down-set; tips keep their
    # observed (possibly ambiguous) state.  With this resolution the
    # number of edges whose endpoints share no base equals the Fitch
    # parsimony score.
    # ACCTRAN up-pass: fully resolve every node to a single base.  Each
    # internal node inherits the parent base when it lies in the node's
    # Fitch down-set, otherwise takes the lowest base of the down-set.
    # With every node single-resolved the count of substitution edges
    # equals the Fitch parsimony score exactly.
    final = {i: down[i].copy() for i in range(1, n_total + 1)}
    root = tree.root_node()
    final[root] = _lowest_bit(down[root])
    for node in tree.preorder():
        if node == root:
            continue
        p = tree.parent(node)
        if tree.is_tip(node):
            tip_bits = seqs[tip_labels[node - 1]].astype(np.uint8)
            pf = final[p]
            inter = tip_bits & pf
            final[node] = _lowest_bit(np.where(inter != 0, inter,
                                               tip_bits).astype(np.uint8))
            continue
        d = down[node]
        pf = final[p]
        inter = d & pf
        final[node] = _lowest_bit(np.where(inter != 0, inter,
                                           d).astype(np.uint8))
    return final


def _lowest_bit(arr: np.ndarray) -> np.ndarray:
    """Resolve each bitmask to its lowest set bit (deterministic base)."""
    arr = arr.astype(np.uint8)
    out = arr & (-arr.astype(np.int16)).astype(np.uint8)
    out[out == 0] = 15  # all-N -> keep ambiguous
    return out.astype(np.uint8)


# ----------------------------------------------------------------------
def ancestral_pars(tree: Phylo, seqs: Dict[str, np.ndarray],
                   asr_type: str = "MPR") -> Dict[int, str]:
    """Reconstruct ancestral sequences for every node of ``tree``.

    Returns ``{node_index: sequence_string}``.  ``asr_type`` may be
    ``"MPR"`` (most-parsimonious reconstruction, ambiguous sites kept as
    IUPAC codes) or ``"ACCTRAN"`` (fully resolved, early substitutions).
    """
    tip_labels = tree.tip_label
    n_sites = len(seqs[tip_labels[0]])
    n_total = tree.n_tip + tree.Nnode

    if asr_type.upper() == "ACCTRAN":
        states = _fitch_ancestral(tree, seqs)
        return {i: "".join(decode_bits(int(b)) for b in states[i])
                for i in range(1, n_total + 1)}

    # MPR: down pass then union/intersection up pass keeping ambiguity
    down = {i: np.zeros(n_sites, dtype=np.uint8) for i in range(1, n_total + 1)}
    for i, t in enumerate(tip_labels):
        down[i + 1] = seqs[t].astype(np.uint8).copy()
    for node in tree.postorder():
        if tree.is_tip(node):
            continue
        kids = tree.children(node)
        if not kids:
            continue
        acc = down[kids[0]].copy()
        for c in kids[1:]:
            cs = down[c]
            inter = acc & cs
            acc = np.where(inter == 0, acc | cs, inter).astype(np.uint8)
        down[node] = acc
    final = {i: down[i].copy() for i in range(1, n_total + 1)}
    root = tree.root_node()
    for node in tree.preorder():
        if node == root or tree.is_tip(node):
            continue
        p = tree.parent(node)
        d = down[node]
        pf = final[p]
        inter = d & pf
        final[node] = np.where(inter != 0, inter, d).astype(np.uint8)
    return {i: "".join(decode_bits(int(b)) for b in final[i])
            for i in range(1, n_total + 1)}


# ----------------------------------------------------------------------
def fitch_states(tree: Phylo, tip_states: Dict[str, str],
                 all_states: Sequence[str]) -> Dict[int, set]:
    """Discrete-trait Fitch reconstruction for the SP / PS / SC tests.

    Each tip carries a single discrete state (e.g. tissue, isotype,
    timepoint).  Performs a Fitch down-pass then a delayed-transformation
    up-pass and returns the (possibly ambiguous) set of MP states at each
    node.  Used by :mod:`pydowser.switches`.
    """
    state_index = {s: i for i, s in enumerate(all_states)}
    n_states = len(all_states)
    full = (1 << n_states) - 1
    n_total = tree.n_tip + tree.Nnode

    down: Dict[int, int] = {}
    for i, t in enumerate(tree.tip_label):
        s = tip_states.get(t)
        if s is None or s not in state_index:
            down[i + 1] = full
        else:
            down[i + 1] = 1 << state_index[s]

    for node in tree.postorder():
        if tree.is_tip(node):
            continue
        kids = tree.children(node)
        if not kids:
            continue
        acc = down[kids[0]]
        for c in kids[1:]:
            inter = acc & down[c]
            acc = inter if inter else (acc | down[c])
        down[node] = acc

    final: Dict[int, int] = {}
    root = tree.root_node()
    final[root] = down[root]
    for node in tree.preorder():
        if node == root:
            continue
        p = tree.parent(node)
        if tree.is_tip(node):
            final[node] = down[node]
            continue
        inter = down[node] & final[p]
        final[node] = inter if inter else down[node]

    out: Dict[int, set] = {}
    for node in range(1, n_total + 1):
        bits = final.get(node, full)
        out[node] = {all_states[i] for i in range(n_states)
                     if bits & (1 << i)}
    return out

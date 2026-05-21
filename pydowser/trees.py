"""Lineage tree construction and tree utilities.

Port of dowser's ``R/TreeFunctions.R`` -- the parsimony tree builder
(:func:`buildPratchet`), the maximum-likelihood builder
(:func:`buildPML`), the :func:`getTrees` wrapper, tree re-rooting,
divergence/path-length calculations and the tree-editing helpers
(``collapseNodes``, ``scaleBranches``, ``resolvePolytomies``, ...).
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .clones import AirrClone
from .parsimony import (acctran, ancestral_pars, encode_dna, fitch_score,
                        pratchet)
from .phylo import Phylo

__all__ = [
    "buildPratchet", "buildPML", "getTrees", "rerootTree", "getDivergence",
    "getPathLengths", "getBootstraps", "collapseNodes", "scaleBranches",
    "resolvePolytomies", "getNodeSeq", "getSeq", "calcRF", "parsimonyScore",
    "di2multi", "readFasta", "writeFasta",
]


# ----------------------------------------------------------------------
def readFasta(path: str) -> Dict[str, str]:
    """Read a FASTA file into an ``{id: sequence}`` dict (dowser ``readFasta``)."""
    seqs: Dict[str, str] = {}
    cur: Optional[str] = None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                cur = line[1:]
                seqs[cur] = ""
            elif cur is not None:
                seqs[cur] += line
    return seqs


def writeFasta(seqs: Dict[str, str], path: str) -> None:
    """Write an ``{id: sequence}`` mapping to a FASTA file."""
    with open(path, "w") as fh:
        for name, seq in seqs.items():
            fh.write(f">{name}\n{seq}\n")


# ----------------------------------------------------------------------
def _clone_seqs(clone: AirrClone, seq: str = "sequence"):
    """Return (tip labels, germline string, encoded seqs) for a clone."""
    if seq == "hlsequence":
        germline = clone.hlgermline
    elif seq == "lsequence":
        germline = clone.lgermline
    else:
        germline = clone.germline
    labels = list(clone.data["sequence_id"].astype(str)) + ["Germline"]
    raw = list(clone.data[seq].astype(str)) + [germline]
    lengths = {len(s) for s in raw}
    if len(lengths) > 1:
        raise ValueError(
            f"Sequence and/or germline lengths of clone {clone.clone} "
            "are not equal")
    encoded = {lab: encode_dna(s) for lab, s in zip(labels, raw)}
    return labels, germline, encoded


# ----------------------------------------------------------------------
def di2multi(tree: Phylo, tol: float = 1e-8) -> Phylo:
    """Collapse branches shorter than ``tol`` into polytomies.

    Pure-Python analogue of ``ape::di2multi``.
    """
    out = tree.copy()
    changed = True
    while changed:
        changed = False
        for k in range(out.n_edge):
            p, c = int(out.edge[k, 0]), int(out.edge[k, 1])
            if out.is_tip(c):
                continue
            if out.edge_length[k] <= tol:
                # remove node c; reattach its children to p
                kids = out.children(c)
                keep_mask = np.ones(out.n_edge, dtype=bool)
                keep_mask[k] = False
                new_edges = []
                new_len = []
                for j in range(out.n_edge):
                    if j == k:
                        continue
                    pp, cc = int(out.edge[j, 0]), int(out.edge[j, 1])
                    if pp == c:
                        pp = p
                    new_edges.append([pp, cc])
                    new_len.append(out.edge_length[j])
                out = _rebuild(new_edges, new_len, out.tip_label, out.nodes,
                               out.name)
                changed = True
                break
    return out


def _rebuild(edges, elen, tip_label, nodes, name) -> Phylo:
    """Rebuild a Phylo from an edge list, renumbering internal nodes."""
    n_tip = len(tip_label)
    internal = sorted({p for p, c in edges if p > n_tip}
                      | {c for p, c in edges if c > n_tip})
    remap = {old: n_tip + 1 + k for k, old in enumerate(internal)}
    for i in range(1, n_tip + 1):
        remap[i] = i
    new_edges = [[remap[p], remap[c]] for p, c in edges]
    new_nodes = {}
    for old, new in remap.items():
        if old in nodes:
            new_nodes[new] = nodes[old]
        else:
            new_nodes[new] = {"sequence": None}
    out = Phylo(edge=new_edges, edge_length=elen, tip_label=tip_label,
                Nnode=len(internal), nodes=new_nodes, name=name)
    return out


# ----------------------------------------------------------------------
def rerootTree(tree: Phylo, germline: str = "Germline", min: float = 0.001,
               verbose: int = 0) -> Phylo:
    """Re-root an (unrooted) tree on the germline tip.

    Faithful port of dowser's ``rerootTree``.  The germline becomes a tip
    descending from the new root by a zero-length edge; total tree length
    and germline divergences are preserved.
    """
    if germline not in tree.tip_label:
        raise ValueError(f"{germline} not found in tip labels!")

    out = tree.copy()
    germ_node = out.get_tip_index(germline)
    # build undirected adjacency
    n_total = out.n_tip + out.Nnode
    adj: Dict[int, List[tuple]] = {i: [] for i in range(1, n_total + 2)}
    for k in range(out.n_edge):
        p, c = int(out.edge[k, 0]), int(out.edge[k, 1])
        adj[p].append((c, out.edge_length[k]))
        adj[c].append((p, out.edge_length[k]))

    # germline currently a tip; its single neighbour is the MRCA
    mrca = adj[germ_node][0][0]
    germ_edge_len = adj[germ_node][0][1]
    # new root node -- use a number above every existing node to avoid
    # colliding with an original internal node
    new_root = n_total + 100
    # we will produce a fresh edge list rooted at a node sitting between
    # germline and mrca
    # do a directed traversal away from germline through mrca
    new_edges: List[List[int]] = []
    new_len: List[float] = []
    # root connects to germline (len 0) and to mrca (len germ_edge_len)
    # collect all non-germline internal/tip nodes
    visited = {germ_node}
    # BFS from mrca
    order = []
    stack = [(mrca, None)]
    while stack:
        node, parent = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append((node, parent))
        for nb, w in adj[node]:
            if nb not in visited and nb != germ_node:
                stack.append((nb, node))

    # renumber: tips keep 1..n_tip; internal get fresh contiguous numbers
    internal_old = [n for n in range(out.n_tip + 1, n_total + 1)]
    # build edges in a rooted orientation
    # root -> germline, root -> mrca
    raw_edges: List[tuple] = []
    raw_edges.append((new_root, germ_node, 0.0))
    raw_edges.append((new_root, mrca, germ_edge_len))
    for node, parent in order:
        if parent is None:
            continue
        # find length
        w = next(ww for nb, ww in adj[node] if nb == parent)
        raw_edges.append((parent, node, w))

    # collect node ids, renumber internals
    used_internal = sorted({p for p, c, _ in raw_edges if not out.is_tip(p)}
                           | {c for p, c, _ in raw_edges if not out.is_tip(c)})
    remap = {}
    nxt = out.n_tip + 1
    # ensure new_root maps first
    for nid in [new_root] + [x for x in used_internal if x != new_root]:
        if nid not in remap:
            remap[nid] = nxt
            nxt += 1
    for i in range(1, out.n_tip + 1):
        remap[i] = i

    edges = [[remap[p], remap[c]] for p, c, _ in raw_edges]
    lens = [w for _, _, w in raw_edges]
    new_nodes = {}
    for old, new in remap.items():
        if old == new_root:
            # new root gets germline node sequence (UCA)
            new_nodes[new] = dict(out.nodes.get(germ_node, {"sequence": None}))
        elif old in out.nodes:
            new_nodes[new] = out.nodes[old]
        else:
            new_nodes[new] = {"sequence": None}

    rooted = Phylo(edge=edges, edge_length=lens, tip_label=out.tip_label,
                   Nnode=len({p for p, c in edges}), nodes=new_nodes,
                   name=out.name)
    rooted.tree_method = out.tree_method
    rooted.edge_type = out.edge_type
    rooted.seq = out.seq
    rooted = rooted.ladderize(right=False)

    # sanity check tree length
    if abs(rooted.total_length() - out.total_length()) > 1e-3:
        warnings.warn(f"rerootTree: tree length changed for {out.name}")
    return rooted


# ----------------------------------------------------------------------
def parsimonyScore(tree: Phylo, clone: AirrClone, seq: str = "sequence") -> int:
    """Return the Fitch parsimony score of ``tree`` for ``clone``."""
    labels, germline, encoded = _clone_seqs(clone, seq)
    # only score tips present in tree
    sub = {lab: encoded[lab] for lab in tree.tip_label if lab in encoded}
    return fitch_score(tree, sub)


# ----------------------------------------------------------------------
def buildPratchet(clone: AirrClone, seq: str = "sequence", asr: str = "seq",
                  asr_thresh: float = 0.05, tree: Optional[Phylo] = None,
                  asr_type: str = "MPR", verbose: int = 0,
                  resolve_random: bool = True, maxit: int = 100,
                  k: int = 10, seed: int = 1) -> Phylo:
    """Build a maximum-parsimony lineage tree for a clone (dowser ``buildPratchet``).

    Searches for a most-parsimonious topology with the parsimony ratchet
    (:func:`pydowser.parsimony.pratchet`), assigns ACCTRAN branch lengths,
    divides them by sequence length to give genetic distance, reconstructs
    ancestral sequences and re-roots on the germline.
    """
    labels, germline, encoded = _clone_seqs(clone, seq)
    if len(clone.data) < 2:
        raise ValueError(f"{clone.clone} has only one sequence, skipping")

    if tree is None:
        tree = pratchet(encoded, labels, maxit=maxit, k=k, trace=verbose,
                        seed=seed)
        tree = acctran(tree, encoded)
        tree.edge_length = tree.edge_length / len(germline)
        tree.tree_method = "pydowser::pratchet"
        tree.edge_type = "genetic_distance"
    tree.name = clone.clone
    tree.seq = seq

    if asr != "none":
        asr_seqs = ancestral_pars(tree, encoded, asr_type=asr_type)
        for node, s in asr_seqs.items():
            tree.nodes.setdefault(node, {})
            tree.nodes[node]["sequence"] = s

    # parsimony score should be unchanged by rerooting
    opars = fitch_score(tree, encoded)
    tree = rerootTree(tree, "Germline", verbose=0)
    # re-attach ancestral sequences after reroot (node numbering preserved
    # for tips; recompute ASR on rooted tree for internal nodes)
    if asr != "none":
        asr_seqs = ancestral_pars(tree, encoded, asr_type=asr_type)
        for node, s in asr_seqs.items():
            tree.nodes.setdefault(node, {})
            tree.nodes[node]["sequence"] = s
    npars = fitch_score(tree, encoded)
    if npars != opars:
        warnings.warn(
            f"rerooting changed parsimony score for {tree.name}: "
            f"{opars} -> {npars}")
    tree.tree_method = "pydowser::pratchet"
    tree.edge_type = "genetic_distance"
    return tree


# ----------------------------------------------------------------------
def buildPML(clone: AirrClone, seq: str = "sequence", sub_model: str = "JC",
             asr: str = "seq", tree: Optional[Phylo] = None,
             verbose: bool = False, seed: int = 1, max_iter: int = 30) -> Phylo:
    """Build a maximum-likelihood lineage tree for a clone (dowser ``buildPML``).

    A tractable pure-Python ML route: builds an NJ starting tree, performs
    NNI topology search scored by Felsenstein likelihood under the
    Jukes-Cantor (``"JC"``) model, then optimises branch lengths.  The full
    GTR+gamma model of ``phangorn::optim.pml`` is **not** reproduced -- see
    the package README's phylogenetics caveats.  Ancestral states are
    reconstructed by parsimony so the tree stays comparable to the
    parsimony route.
    """
    from .likelihood import optimise_ml_tree

    labels, germline, encoded = _clone_seqs(clone, seq)
    if len(clone.data) < 2:
        raise ValueError(f"{clone.clone} has only one sequence, skipping")
    seqs = {lab: str(s) for lab, s in
            zip(labels, list(clone.data[seq].astype(str)) + [germline])}
    if tree is None:
        tree = optimise_ml_tree(seqs, labels, sub_model=sub_model,
                                max_iter=max_iter, seed=seed,
                                verbose=verbose)
        tree.tree_method = f"pydowser::pml::{sub_model}"
        tree.edge_type = "genetic_distance"
    tree.name = clone.clone
    tree.seq = seq
    if asr != "none":
        asr_seqs = ancestral_pars(tree, encoded, asr_type="MPR")
        for node, s in asr_seqs.items():
            tree.nodes.setdefault(node, {})
            tree.nodes[node]["sequence"] = s
    tree = rerootTree(tree, "Germline", verbose=0)
    if asr != "none":
        asr_seqs = ancestral_pars(tree, encoded, asr_type="MPR")
        for node, s in asr_seqs.items():
            tree.nodes.setdefault(node, {})
            tree.nodes[node]["sequence"] = s
    tree.tree_method = f"pydowser::pml::{sub_model}"
    tree.edge_type = "genetic_distance"
    return tree


# ----------------------------------------------------------------------
def getTrees(clones: pd.DataFrame, build: str = "pratchet",
             trait: Optional[str] = None, seq: Optional[str] = None,
             collapse: bool = False, verbose: int = 0,
             check_divergence: bool = True, **kwargs) -> pd.DataFrame:
    """Build a phylogenetic tree for every clone (dowser ``getTrees``).

    Parameters
    ----------
    clones:
        the tibble returned by :func:`formatClones`.
    build:
        ``"pratchet"`` (maximum parsimony, default) or ``"pml"`` (maximum
        likelihood; see :func:`buildPML` for caveats).
    collapse:
        collapse internal nodes with identical reconstructed sequences.

    Returns the input data frame with an added ``trees`` column of
    :class:`~pydowser.phylo.Phylo` objects.
    """
    if "data" not in clones.columns:
        raise ValueError("clones must have a 'data' column of AirrClone objects")
    if clones["clone_id"].duplicated().any():
        raise ValueError("Clone IDs not unique.")

    out = clones.copy().reset_index(drop=True)
    trees: List[Phylo] = []
    keep_idx: List[int] = []
    for i, row in out.iterrows():
        clone = row["data"]
        s = seq if seq is not None else clone.phylo_seq
        try:
            if build == "pratchet":
                t = buildPratchet(clone, seq=s, verbose=verbose, **kwargs)
            elif build == "pml":
                t = buildPML(clone, seq=s, verbose=bool(verbose), **kwargs)
            else:
                raise ValueError(f"build specification {build} not recognized")
            trees.append(t)
            keep_idx.append(i)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Tree building failed for clone "
                          f"{row['clone_id']}: {exc}")
    if not trees:
        raise ValueError("No trees left!")
    out = out.loc[keep_idx].reset_index(drop=True)
    out["trees"] = trees
    if collapse:
        out = collapseNodes(out)
    return out


# ----------------------------------------------------------------------
def getDivergence(phy: Phylo, minlength: float = 0.001) -> Dict[str, float]:
    """Root-to-tip divergence for every tip (dowser ``getDivergence``).

    Branches shorter than ``minlength`` are collapsed first.  Returns the
    summed branch length from the MRCA of all tips to each tip; when the
    germline tip is present this equals germline divergence.
    """
    work = phy.copy()
    work.edge_length = work.edge_length.copy()
    work.edge_length[work.edge_length < minlength] = 0.0
    work = di2multi(work, tol=minlength)
    mrca = work.get_mrca(work.tip_label)
    D = work.dist_nodes()
    return {lab: float(D[i + 1, mrca]) for i, lab in enumerate(work.tip_label)}


def getPathLengths(phy: Phylo, minlength: float = 0.0,
                   germline: str = "Germline") -> pd.DataFrame:
    """Per-tip path lengths from the root / germline.

    Returns a data frame with one row per tip giving the number of edges
    on the root-to-tip path (``nodes``) and the summed branch length
    (``distance``).  Complements :func:`getDivergence`; useful for
    measurable-evolution analyses.
    """
    work = phy.copy()
    if minlength > 0:
        work.edge_length = work.edge_length.copy()
        work.edge_length[work.edge_length < minlength] = 0.0
    root = work.root_node()
    depth = work.node_depths()
    rows = []
    for i, lab in enumerate(work.tip_label):
        node = i + 1
        # count edges to root
        steps = 0
        n = node
        while True:
            p = work.parent(n)
            if p is None:
                break
            steps += 1
            n = p
        rows.append({"name": lab, "nodes": steps,
                     "distance": float(depth[node])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
def scaleBranches(clones: pd.DataFrame,
                  edge_type: str = "mutations") -> pd.DataFrame:
    """Rescale tree branch lengths between mutations and mutations/site.

    Port of dowser's ``scaleBranches``.  Uses each clone's germline length
    and the current ``edge_type`` of its tree to convert.
    """
    if "trees" not in clones.columns:
        raise ValueError("clones must have trees column!")
    out = clones.copy()
    new_trees = []
    for i, row in out.iterrows():
        clone = row["data"]
        tree = row["trees"].copy()
        if clone.phylo_seq == "hlsequence":
            glen = len(clone.hlgermline)
        elif clone.phylo_seq == "lsequence":
            glen = len(clone.lgermline)
        else:
            glen = len(clone.germline)
        cur = tree.edge_type
        if cur == edge_type:
            new_trees.append(tree)
            continue
        if cur == "genetic_distance" and edge_type == "mutations":
            tree.edge_length = tree.edge_length * glen
        elif cur == "mutations" and edge_type == "genetic_distance":
            tree.edge_length = tree.edge_length / glen
        tree.edge_type = edge_type
        new_trees.append(tree)
    out["trees"] = new_trees
    return out


# ----------------------------------------------------------------------
def collapseNodes(trees, tips: bool = False, check: bool = True):
    """Collapse internal edges between identical reconstructed sequences.

    Port of dowser's ``collapseNodes``.  Accepts either a single
    :class:`Phylo` or a tibble with a ``trees`` column.
    """
    if isinstance(trees, pd.DataFrame):
        out = trees.copy()
        out["trees"] = [collapseNodes(t, tips=tips, check=check)
                        for t in out["trees"]]
        return out

    tree = trees
    out = tree.copy()
    btip = out.n_tip
    changed = True
    max_iter = 1000
    while changed and max_iter > 0:
        max_iter -= 1
        changed = False
        for k in range(out.n_edge):
            p, c = int(out.edge[k, 0]), int(out.edge[k, 1])
            if out.is_tip(c):
                continue
            sp = out.nodes.get(p, {}).get("sequence")
            sc = out.nodes.get(c, {}).get("sequence")
            if sp is not None and sc is not None and sp == sc:
                kids = out.children(c)
                new_edges = []
                new_len = []
                for j in range(out.n_edge):
                    if j == k:
                        continue
                    pp, cc = int(out.edge[j, 0]), int(out.edge[j, 1])
                    if pp == c:
                        pp = p
                    new_edges.append([pp, cc])
                    new_len.append(out.edge_length[j])
                out = _rebuild(new_edges, new_len, out.tip_label,
                               out.nodes, out.name)
                out.edge_type = tree.edge_type
                out.tree_method = tree.tree_method
                out.seq = tree.seq
                changed = True
                break
    return out


# ----------------------------------------------------------------------
def resolvePolytomies(phy: Phylo, clone: AirrClone, minlength: float = 0.001,
                      time: str = "time", sequence: str = "sequence_id",
                      germline: str = "Germline",
                      verbose: bool = False) -> Phylo:
    """Resolve polytomies into a minimal number of single-timepoint clades.

    Port of dowser's ``resolvePolytomies`` -- used inside
    :func:`pydowser.statistics.correlationTest`.  This implementation
    collapses near-zero branches then resolves each polytomy so that
    descendants of the same timepoint are grouped, which minimises the
    number of distinct single-timepoint clades while preserving every
    tip's root divergence.
    """
    work = di2multi(phy.copy(), tol=minlength)
    work = rerootTree(work, germline=germline, verbose=0)
    data = clone.data
    odiv = getDivergence(work, minlength)

    times = dict(zip(data[sequence].astype(str), data[time]))

    def polytomies(t):
        cnt: Dict[int, int] = {}
        for p in t.edge[:, 0]:
            cnt[int(p)] = cnt.get(int(p), 0) + 1
        return [n for n, c in cnt.items() if c > 2]

    # Resolution here is structural-only: dowser's binding preserves
    # divergence, and the permutation test only depends on clade
    # membership.  We resolve by grouping same-timepoint children under a
    # new zero-length internal node.
    poly = polytomies(work)
    guard = 0
    while poly and guard < 200:
        guard += 1
        target = poly[0]
        kids = work.children(target)
        # group children by the (single) timepoint of their descendant tips
        groups: Dict[object, List[int]] = {}
        for c in kids:
            tps = {times.get(t) for t in work.descendant_tips(c)}
            key = next(iter(tps)) if len(tps) == 1 else "_multi"
            groups.setdefault(key, []).append(c)
        if len(groups) >= len(kids):
            break  # nothing to gain
        new_edges = [list(e) for e in work.edge.tolist()]
        new_len = list(work.edge_length)
        elen = {(int(p), int(c)): work.edge_length[i]
                for i, (p, c) in enumerate(work.edge)}
        next_node = work.n_tip + work.Nnode + 1
        for key, members in groups.items():
            if len(members) < 2:
                continue
            # create a new internal node grouping these members
            new_node = next_node
            next_node += 1
            for c in members:
                for j, (p, cc) in enumerate(new_edges):
                    if p == target and cc == c:
                        new_edges[j][0] = new_node
                        break
            new_edges.append([target, new_node])
            new_len.append(1e-7)
        work = _rebuild(new_edges, new_len, work.tip_label, work.nodes,
                        work.name)
        poly = polytomies(work)
    ndiv = getDivergence(work, minlength)
    if len(ndiv) != len(odiv):
        warnings.warn("resolvePolytomies: tip count changed")
    return work


# ----------------------------------------------------------------------
def getNodeSeq(data: pd.DataFrame, node: int, tree: Optional[Phylo] = None,
               clone: Optional[str] = None, gaps: bool = True) -> str:
    """Return the reconstructed sequence at a tree node (dowser ``getNodeSeq``).

    Provide either an explicit ``tree`` or a ``clone`` id present in
    ``data``.  With ``gaps=True`` IMGT gaps are re-inserted.
    """
    if tree is None:
        if clone is None:
            raise ValueError("must provide either tree object or clone id")
        tree = data.loc[data["clone_id"] == clone, "trees"].iloc[0]
    seq = tree.nodes.get(node, {}).get("sequence")
    if seq is None:
        raise ValueError(f"node {node} has no reconstructed sequence")
    return seq


def getSeq(data: pd.DataFrame, node: int, tree: Optional[Phylo] = None,
           clone: Optional[str] = None, gaps: bool = True) -> str:
    """Alias of :func:`getNodeSeq` (dowser keeps both names)."""
    return getNodeSeq(data, node, tree=tree, clone=clone, gaps=gaps)


# ----------------------------------------------------------------------
def _splits(tree: Phylo):
    """Return the set of non-trivial bipartitions (clades) of a tree."""
    splits = set()
    for node in tree.internal_indices():
        tips = frozenset(tree.descendant_tips(node))
        if 1 < len(tips) < tree.n_tip:
            splits.add(tips)
    return splits


def calcRF(tree_1: Phylo, tree_2: Phylo) -> int:
    """Robinson-Foulds distance between two trees with identical tip sets.

    Port of dowser's ``calcRF``.
    """
    if len(tree_1.tip_label) != len(tree_2.tip_label):
        raise ValueError("trees do not have the same number of tips")
    if set(tree_1.tip_label) != set(tree_2.tip_label):
        raise ValueError("tree tip labels are not identical")
    s1, s2 = _splits(tree_1), _splits(tree_2)
    return len(s1 ^ s2)


# ----------------------------------------------------------------------
def getBootstraps(clones: pd.DataFrame, bootstraps: int = 100,
                  build: str = "pratchet", seq: Optional[str] = None,
                  seed: int = 1, bootstrap_nodes: bool = True,
                  **kwargs) -> pd.DataFrame:
    """Estimate node-support values by nonparametric bootstrapping.

    Port of dowser's ``getBootstraps``.  For each clone, alignment columns
    are resampled with replacement ``bootstraps`` times, a tree rebuilt for
    each replicate, and the proportion of replicates supporting each clade
    of the original tree recorded in ``node.label`` (when
    ``bootstrap_nodes=True``).  Adds a ``bootstrap_trees`` column.
    """
    if "trees" not in clones.columns:
        raise ValueError("clones must have trees column (run getTrees first)")
    rng = np.random.default_rng(seed)
    out = clones.copy()
    boot_lists: List[List[Phylo]] = []
    supports: List[Phylo] = []
    for _, row in out.iterrows():
        clone = row["data"]
        ref = row["trees"]
        s = seq if seq is not None else clone.phylo_seq
        _, germline, _ = _clone_seqs(clone, s)
        n_sites = len(germline)
        rep_trees: List[Phylo] = []
        ref_splits = _splits(ref)
        support_count = {fs: 0 for fs in ref_splits}
        for _b in range(bootstraps):
            idx = rng.integers(0, n_sites, size=n_sites)
            bclone = clone.copy()
            bclone.data = bclone.data.copy()
            bclone.data[s] = [
                "".join(np.asarray(list(str(x)))[idx])
                for x in bclone.data[s]]
            if s == "sequence":
                bclone.germline = "".join(np.asarray(list(germline))[idx])
            try:
                if build == "pratchet":
                    bt = buildPratchet(bclone, seq=s, asr="none",
                                       seed=int(rng.integers(1, 1 << 30)),
                                       **kwargs)
                else:
                    bt = buildPML(bclone, seq=s, asr="none", **kwargs)
            except Exception:  # noqa: BLE001
                continue
            rep_trees.append(bt)
            for fs in _splits(bt):
                if fs in support_count:
                    support_count[fs] += 1
        boot_lists.append(rep_trees)
        if bootstrap_nodes and rep_trees:
            sup = ref.copy()
            labels = []
            for node in sup.internal_indices():
                tips = frozenset(sup.descendant_tips(node))
                pct = 100.0 * support_count.get(tips, 0) / len(rep_trees)
                labels.append(f"{pct:.0f}")
            sup.node_label = labels
            supports.append(sup)
        else:
            supports.append(ref)
    out["bootstrap_trees"] = boot_lists
    out["trees"] = supports
    return out

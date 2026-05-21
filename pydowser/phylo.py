"""Phylogenetic tree data structure for pydowser.

This module defines :class:`Phylo`, a faithful pure-Python analogue of the
``ape`` ``phylo`` object used throughout the R package ``dowser``.  Trees are
stored exactly as in ``ape``:

* ``edge``        -- ``(n_edges, 2)`` integer array of ``(parent, child)``
                     node indices.  Node indexing is **1-based** to match R.
* ``edge_length`` -- ``(n_edges,)`` float array of branch lengths.
* ``tip_label``   -- list of ``n_tip`` leaf names; tip ``i`` is node ``i + 1``.
* ``Nnode``       -- number of internal nodes.  Internal nodes are numbered
                     ``n_tip + 1 .. n_tip + Nnode``.
* ``nodes``       -- per-node reconstructed sequence / trait dictionary
                     (1-based, indexed ``nodes[i]`` for node ``i``).

The germline / root sequence is carried as an ordinary tip named
``Germline`` (dowser convention), and the tree is rooted so that the
germline tip hangs off the root node by a zero-length edge.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ["Phylo"]


class Phylo:
    """An ``ape``-style phylogenetic tree.

    Parameters
    ----------
    edge:
        ``(n_edges, 2)`` array-like of 1-based ``(parent, child)`` pairs.
    edge_length:
        branch length for each edge.
    tip_label:
        leaf names; tip ``i`` (1-based) corresponds to ``tip_label[i-1]``.
    Nnode:
        number of internal nodes.
    nodes:
        optional ``{node_index: {"sequence": str, ...}}`` mapping of
        reconstructed states.
    """

    def __init__(
        self,
        edge: Sequence[Sequence[int]],
        edge_length: Sequence[float],
        tip_label: Sequence[str],
        Nnode: int,
        nodes: Optional[Dict[int, dict]] = None,
        name: Optional[str] = None,
        node_label: Optional[Sequence[str]] = None,
    ) -> None:
        self.edge = np.asarray(edge, dtype=int).reshape(-1, 2)
        self.edge_length = np.asarray(edge_length, dtype=float)
        self.tip_label = list(tip_label)
        self.Nnode = int(Nnode)
        self.name: Optional[str] = name
        self.tree_method: Optional[str] = None
        self.edge_type: Optional[str] = None
        self.seq: Optional[str] = None
        self.node_label = list(node_label) if node_label is not None else None
        n_total = self.n_tip + self.Nnode
        if nodes is None:
            self.nodes: Dict[int, dict] = {i: {"sequence": None}
                                           for i in range(1, n_total + 1)}
        else:
            self.nodes = dict(nodes)

    # ------------------------------------------------------------------
    @property
    def n_tip(self) -> int:
        """Number of leaves (tips)."""
        return len(self.tip_label)

    @property
    def n_edge(self) -> int:
        """Number of branches."""
        return self.edge.shape[0]

    def copy(self) -> "Phylo":
        """Return a deep-ish copy of this tree."""
        new = Phylo(
            self.edge.copy(),
            self.edge_length.copy(),
            list(self.tip_label),
            self.Nnode,
            nodes={k: dict(v) for k, v in self.nodes.items()},
            name=self.name,
            node_label=None if self.node_label is None else list(self.node_label),
        )
        new.tree_method = self.tree_method
        new.edge_type = self.edge_type
        new.seq = self.seq
        return new

    # ------------------------------------------------------------------
    def root_node(self) -> int:
        """Return the root node index (``n_tip + 1`` by ape convention)."""
        children = set(self.edge[:, 1].tolist())
        parents = set(self.edge[:, 0].tolist())
        roots = parents - children
        if len(roots) == 1:
            return next(iter(roots))
        # fall back to ape convention
        return self.n_tip + 1

    def children(self, node: int) -> List[int]:
        """Return child node indices of ``node``."""
        return self.edge[self.edge[:, 0] == node, 1].tolist()

    def parent(self, node: int) -> Optional[int]:
        """Return the parent node index of ``node`` (``None`` for root)."""
        idx = np.where(self.edge[:, 1] == node)[0]
        if len(idx) == 0:
            return None
        return int(self.edge[idx[0], 0])

    def is_tip(self, node: int) -> bool:
        """True if ``node`` is a leaf."""
        return 1 <= node <= self.n_tip

    def postorder(self) -> List[int]:
        """Return node indices in post-order (children before parents)."""
        root = self.root_node()
        order: List[int] = []
        # iterative DFS
        stack = [(root, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                order.append(node)
                continue
            stack.append((node, True))
            for c in self.children(node):
                stack.append((c, False))
        return order

    def preorder(self) -> List[int]:
        """Return node indices in pre-order (parents before children)."""
        root = self.root_node()
        order: List[int] = []
        stack = [root]
        while stack:
            node = stack.pop()
            order.append(node)
            for c in reversed(self.children(node)):
                stack.append(c)
        return order

    def tip_indices(self) -> List[int]:
        """Return 1-based indices of all tips."""
        return list(range(1, self.n_tip + 1))

    def internal_indices(self) -> List[int]:
        """Return 1-based indices of all internal nodes."""
        return list(range(self.n_tip + 1, self.n_tip + self.Nnode + 1))

    def get_tip_index(self, label: str) -> int:
        """Return 1-based node index of tip named ``label``."""
        return self.tip_label.index(label) + 1

    # ------------------------------------------------------------------
    def node_depths(self) -> np.ndarray:
        """Return summed branch length from the root to every node.

        Index 0 is unused; ``depths[i]`` is the root-to-node distance for
        node ``i`` (1-based).
        """
        n_total = self.n_tip + self.Nnode
        depth = np.zeros(n_total + 1)
        elen = {(int(p), int(c)): self.edge_length[k]
                for k, (p, c) in enumerate(self.edge)}
        for node in self.preorder():
            p = self.parent(node)
            if p is None:
                depth[node] = 0.0
            else:
                depth[node] = depth[p] + elen[(p, node)]
        return depth

    def dist_nodes(self) -> np.ndarray:
        """Pairwise patristic distance among all nodes (``ape::dist.nodes``).

        Returns an ``(N+1, N+1)`` matrix with row/column 0 unused so node
        indices line up with the 1-based scheme.
        """
        n_total = self.n_tip + self.Nnode
        adj: Dict[int, List[tuple]] = {i: [] for i in range(1, n_total + 1)}
        for k, (p, c) in enumerate(self.edge):
            p, c = int(p), int(c)
            adj[p].append((c, self.edge_length[k]))
            adj[c].append((p, self.edge_length[k]))
        D = np.full((n_total + 1, n_total + 1), np.inf)
        for src in range(1, n_total + 1):
            D[src, src] = 0.0
            # BFS/DFS over the tree
            stack = [(src, 0.0, -1)]
            while stack:
                node, dist, prev = stack.pop()
                D[src, node] = dist
                for nb, w in adj[node]:
                    if nb != prev:
                        stack.append((nb, dist + w, node))
        D[0, :] = 0.0
        D[:, 0] = 0.0
        return D

    def total_length(self) -> float:
        """Sum of all branch lengths."""
        return float(self.edge_length.sum())

    # ------------------------------------------------------------------
    def get_mrca(self, tips: Optional[Sequence[str]] = None) -> int:
        """Return the most-recent common ancestor node of ``tips``.

        With no argument the MRCA of all tips (i.e. the root) is returned.
        """
        if tips is None:
            tip_nodes = self.tip_indices()
        else:
            tip_nodes = [self.get_tip_index(t) for t in tips]
        # path from each tip to root
        def path_to_root(n):
            p = [n]
            while True:
                par = self.parent(n)
                if par is None:
                    break
                p.append(par)
                n = par
            return p
        paths = [path_to_root(t) for t in tip_nodes]
        common = set(paths[0])
        for pth in paths[1:]:
            common &= set(pth)
        # deepest common node = last in any path that is in common
        for node in paths[0]:
            if node in common:
                # check it is ancestor of all -> deepest one
                pass
        # MRCA = the common node furthest from root = max depth
        depth = self.node_depths()
        return int(max(common, key=lambda n: depth[n]))

    def descendant_tips(self, node: int) -> List[str]:
        """Return tip labels descending from ``node``."""
        out: List[str] = []
        stack = [node]
        while stack:
            n = stack.pop()
            if self.is_tip(n):
                out.append(self.tip_label[n - 1])
            else:
                stack.extend(self.children(n))
        return out

    # ------------------------------------------------------------------
    def ladderize(self, right: bool = False) -> "Phylo":
        """Reorder edges so clades are sorted by descendant count.

        Mirrors ``ape::ladderize``; does not change topology, only edge
        order (and therefore plotting order).
        """
        ndesc: Dict[int, int] = {}
        for node in self.postorder():
            if self.is_tip(node):
                ndesc[node] = 1
            else:
                ndesc[node] = sum(ndesc[c] for c in self.children(node))
        new_edges: List[List[int]] = []
        new_len: List[float] = []
        elen = {(int(p), int(c)): self.edge_length[k]
                for k, (p, c) in enumerate(self.edge)}
        stack = [self.root_node()]
        while stack:
            node = stack.pop()
            kids = self.children(node)
            kids = sorted(kids, key=lambda c: ndesc[c], reverse=not right)
            for c in kids:
                new_edges.append([node, c])
                new_len.append(elen[(node, c)])
            for c in reversed(kids):
                stack.append(c)
        out = self.copy()
        out.edge = np.asarray(new_edges, dtype=int).reshape(-1, 2)
        out.edge_length = np.asarray(new_len, dtype=float)
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"Phylo(name={self.name!r}, tips={self.n_tip}, "
                f"Nnode={self.Nnode}, edges={self.n_edge})")

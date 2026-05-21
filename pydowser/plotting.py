"""Lineage-tree plotting -- a matplotlib port of dowser's ``plotTrees``.

dowser draws trees with ``ggtree``; this module renders the same
information (rectangular phylogram, tip labels, optional trait colouring of
tips, optional node-support labels) using matplotlib so pydowser has no
``ggplot``/``ggtree`` dependency.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .phylo import Phylo

__all__ = ["getPalette", "plotTrees", "plotTree"]

# a qualitative palette (Brewer "Set1"-style) used for trait colouring
_DEFAULT_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62", "#8da0cb",
]


def getPalette(states: List[str], palette: Optional[str] = None
               ) -> Dict[str, str]:
    """Return a ``{state: hex-colour}`` mapping for a set of trait values.

    Port of dowser's ``getPalette``.  ``Germline`` always maps to black.
    """
    nongerm = [s for s in dict.fromkeys(states) if s != "Germline"]
    pal = {s: _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
           for i, s in enumerate(nongerm)}
    if "Germline" in states:
        pal["Germline"] = "#000000"
    return pal


# ----------------------------------------------------------------------
def _layout(tree: Phylo):
    """Compute (x, y) coordinates for every node of a rectangular phylogram."""
    depth = tree.node_depths()
    y: Dict[int, float] = {}
    # tips get evenly-spaced y in ladderized order
    order = []
    for node in tree.postorder():
        if tree.is_tip(node):
            order.append(node)
    for i, node in enumerate(order):
        y[node] = float(i)
    for node in tree.postorder():
        if not tree.is_tip(node):
            kids = tree.children(node)
            if kids:
                y[node] = float(np.mean([y[c] for c in kids]))
    x = {n: float(depth[n]) for n in range(1, tree.n_tip + tree.Nnode + 1)}
    return x, y


def plotTree(tree: Phylo, ax=None, tips: Optional[Dict[str, str]] = None,
             palette: Optional[Dict[str, str]] = None, tipsize: float = 40.0,
             tip_labels: bool = True, node_support: bool = False,
             title: Optional[str] = None, scale: bool = True):
    """Draw a single lineage tree onto a matplotlib axis.

    Parameters
    ----------
    tree:
        a :class:`~pydowser.phylo.Phylo` object.
    tips:
        optional ``{tip_label: trait_value}`` mapping used to colour tips.
    palette:
        optional ``{trait_value: colour}`` mapping (see :func:`getPalette`).
    node_support:
        annotate internal nodes with their ``node_label`` (e.g. bootstrap
        support set by :func:`pydowser.getBootstraps`).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, max(3, 0.22 * tree.n_tip)))

    x, y = _layout(tree)
    # draw branches
    for k in range(tree.n_edge):
        p, c = int(tree.edge[k, 0]), int(tree.edge[k, 1])
        # horizontal segment
        ax.plot([x[p], x[c]], [y[c], y[c]], color="black", lw=1.0,
                zorder=1)
    # vertical connectors
    for node in tree.internal_indices():
        kids = tree.children(node)
        if len(kids) >= 2:
            ys = [y[c] for c in kids]
            ax.plot([x[node], x[node]], [min(ys), max(ys)],
                    color="black", lw=1.0, zorder=1)

    # tip points / labels
    if tips is not None and palette is None:
        palette = getPalette(sorted(set(tips.values()) | {"Germline"}))
    for i, lab in enumerate(tree.tip_label):
        node = i + 1
        if tips is not None:
            col = palette.get(tips.get(lab, "Germline"), "#999999")
            if lab == "Germline":
                col = "#000000"
            ax.scatter([x[node]], [y[node]], s=tipsize, color=col,
                       zorder=3, edgecolors="black", linewidths=0.4)
        if tip_labels:
            ax.text(x[node] + 0.01 * max(x.values() or [1]), y[node], lab,
                    va="center", ha="left", fontsize=7)

    if node_support and tree.node_label is not None:
        for j, node in enumerate(tree.internal_indices()):
            if j < len(tree.node_label) and tree.node_label[j]:
                ax.text(x[node], y[node], str(tree.node_label[j]),
                        fontsize=6, color="#3333aa", ha="right",
                        va="bottom")

    ax.set_title(title or (tree.name or ""))
    ax.set_yticks([])
    if scale:
        ax.set_xlabel(tree.edge_type or "divergence")
    else:
        ax.set_xticks([])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    # legend
    if tips is not None and palette is not None:
        import matplotlib.patches as mpatches
        handles = [mpatches.Patch(color=c, label=s)
                   for s, c in palette.items()]
        ax.legend(handles=handles, fontsize=7, loc="lower right",
                  frameon=False)
    return ax


def plotTrees(clones, trait: Optional[str] = None,
              palette: Optional[Dict[str, str]] = None,
              tips: bool = True, tip_labels: bool = True,
              node_support: bool = False, scale: bool = True,
              ncol: int = 1):
    """Plot one or more lineage trees (matplotlib port of dowser ``plotTrees``).

    Parameters
    ----------
    clones:
        the tibble returned by :func:`pydowser.getTrees` (with ``data`` and
        ``trees`` columns), or a single :class:`~pydowser.phylo.Phylo`.
    trait:
        optional clone-data column used to colour tips by trait value.

    Returns a list of matplotlib axes (one per tree).
    """
    import matplotlib.pyplot as plt

    if isinstance(clones, Phylo):
        ax = plotTree(clones, tip_labels=tip_labels,
                      node_support=node_support, scale=scale)
        return [ax]

    n = len(clones)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(7 * ncol, max(3, 3 * nrow)),
                             squeeze=False)
    axes_flat = axes.ravel()
    out = []
    for i, (_, row) in enumerate(clones.iterrows()):
        tree = row["trees"]
        tip_map = None
        if trait is not None and tips:
            data = row["data"].data
            tip_map = {str(s): str(v) for s, v
                       in zip(data["sequence_id"], data[trait])}
            tip_map["Germline"] = "Germline"
        plotTree(tree, ax=axes_flat[i], tips=tip_map, palette=palette,
                 tip_labels=tip_labels, node_support=node_support,
                 title=str(row["clone_id"]), scale=scale)
        out.append(axes_flat[i])
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.tight_layout()
    return out

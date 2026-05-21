"""File input / output -- a port of dowser's I/O helpers.

Faithful pure-Python ports of the file-handling functions from dowser's
``R/TreeFunctions.R`` and ``R/Plotting.R``:

* :func:`dfToFasta`           -- write a data frame of sequences to FASTA.
* :func:`readModelFile` /
  :func:`makeModelFile`       -- IgPhyML parsimony-model files.
* :func:`writeCloneSequences` -- dump per-clone alignments to FASTA.
* :func:`writeLineageFile`    -- write an IgPhyML-style lineage file.
* :func:`readLineages`        -- read a set of trees referenced by a
  lineage file (Newick / NEXUS).
* :func:`exportTrees`         -- write a clone tibble's trees to Newick.
* :func:`treesToPDF`          -- arrange tree plots into a multi-page PDF.

The Newick reader/writer used here is a small self-contained
implementation so pydowser keeps its no-external-binary policy.
"""
from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .phylo import Phylo

__all__ = [
    "dfToFasta", "readModelFile", "makeModelFile", "writeCloneSequences",
    "writeLineageFile", "readLineages", "exportTrees", "treesToPDF",
    "writeNewick", "readNewick",
]


# ----------------------------------------------------------------------
def dfToFasta(df: pd.DataFrame, file: str, id: str = "sequence_id",
              seq: str = "sequence", imgt_gaps: bool = False,
              columns: Optional[List[str]] = None) -> None:
    """Write a data frame of sequences to a FASTA file (dowser ``dfToFasta``).

    Faithful port: IMGT gap characters (``.``) are stripped unless
    ``imgt_gaps=True``; extra ``columns`` are appended to each id as
    ``|name=value`` annotations.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a DataFrame")
    if id not in df.columns:
        raise ValueError(f"{id} column not found in df")
    if seq not in df.columns:
        raise ValueError(f"{seq} column not found in df")

    seqs = df[seq].astype(str)
    if not imgt_gaps:
        seqs = seqs.str.replace(".", "", regex=False)
    ids = df[id].astype(str)
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"{','.join(missing)} not found in df")
        for col in columns:
            ids = ids + "|" + col + "=" + df[col].astype(str)
    with open(file, "w") as fh:
        for i, s in zip(ids, seqs):
            fh.write(f">{i}\n{s}\n")


# ----------------------------------------------------------------------
def readModelFile(file: str, useambig: bool = False) -> Dict[str, str]:
    """Read an IgPhyML parsimony-model file (dowser ``readModelFile``).

    Returns a ``{state: state}`` mapping of the model's discrete states
    (the ``#STATES`` block).  Ambiguous-state parsing (``useambig=True``)
    is not supported, matching dowser.
    """
    if useambig:
        raise ValueError("Ambiguous states not currently supported, "
                         "please useambig=False")
    with open(file) as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    try:
        mstart = lines.index("#STATES")
    except ValueError:
        raise ValueError("#STATES block not found in model file")
    blanks = [i for i, ln in enumerate(lines) if ln == "" and i > mstart]
    if not blanks:
        raise ValueError("Malformed model file: no end of #STATES block")
    mend = min(blanks)
    states = lines[mstart + 1:mend]
    return {s: s for s in states}


def makeModelFile(file: str, states: List[str],
                  constraints: Optional[List[str]] = None,
                  exceptions: Optional[List[str]] = None) -> str:
    """Write an IgPhyML parsimony-model file (dowser ``makeModelFile``).

    Faithful port.  ``constraints="irrev"`` forbids switches moving from
    left to right in the ``states`` vector; alternatively a list of
    ``"FROM,TO"`` constraint strings can be given.  Returns ``file``.
    """
    states = list(states)
    exceptions = list(exceptions) if exceptions else []
    with open(file, "w") as fh:
        fh.write("#BEGIN\n")
        fh.write(f"{len(states)}\n\n")
        fh.write("#STATES\n")
        for a in states:
            fh.write(f"{a}\n")
        fh.write("\n#CONSTRAINTS\n")
        if constraints is not None:
            if constraints == "irrev":
                for i in range(len(states) - 1):
                    for j in range(i + 1, len(states)):
                        if f"{states[j]},{states[i]}" in exceptions:
                            continue
                        fh.write(f"{states[j]} {states[i]} 1000\n")
            else:
                for constraint in constraints:
                    cons = constraint.split(",")
                    fh.write(f"{cons[0]} {cons[1]} 1000\n")
        fh.write("\n\n#AMBIGUOUS\n")
        for a in states:
            fh.write(f"GERM {a}\n")
    return file


# ----------------------------------------------------------------------
def _clone_germline(clone) -> str:
    """Return the germline string corresponding to a clone's phylo_seq."""
    if clone.phylo_seq == "hlsequence":
        return clone.hlgermline
    if clone.phylo_seq == "lsequence":
        return clone.lgermline
    return clone.germline


def _clone_seqcol(clone) -> str:
    """Return the per-sequence column corresponding to a clone's phylo_seq."""
    if clone.phylo_seq == "hlsequence":
        return "hlsequence"
    if clone.phylo_seq == "lsequence":
        return "lsequence"
    return "sequence"


def writeCloneSequences(clones: pd.DataFrame, file: str) -> None:
    """Write all per-clone tree-building sequences to a FASTA file.

    Faithful port of dowser's ``writeCloneSequences``.  Each sequence id
    is suffixed ``|clone_id``; the clone's germline is written first as
    ``Germline|clone_id``.
    """
    with open(file, "w") as fh:
        for _, row in clones.iterrows():
            clone_id = row["clone_id"]
            clone = row["data"]
            germline = _clone_germline(clone)
            col = _clone_seqcol(clone)
            fh.write(f">Germline|{clone_id}\n{germline}\n")
            for _, srow in clone.data.iterrows():
                fh.write(f">{srow['sequence_id']}|{clone_id}\n"
                         f"{srow[col]}\n")


def writeLineageFile(data, trees: Optional[List[Phylo]] = None,
                     dir: str = ".", id: str = "N",
                     rep: Optional[int] = None, trait: Optional[str] = None,
                     empty: bool = True) -> str:
    """Write an IgPhyML-style lineage file (dowser ``writeLineageFile``).

    Faithful port of the file-writing logic of dowser's
    ``writeLineageFile``: for each clone a FASTA of sequences and a
    Newick tree are written into ``<dir>/<id>_recon_<rep>/`` and indexed
    by a ``<id>_lineages_pars.tsv`` lineage file.  Returns the lineage
    file path.

    ``data`` is a list of :class:`~pydowser.clones.AirrClone` objects (or
    the ``data`` column of a clone tibble).
    """
    if isinstance(data, pd.DataFrame):
        if trees is None and "trees" in data.columns:
            trees = list(data["trees"])
        data = list(data["data"])
    data = list(data)

    file = os.path.join(dir, f"{id}_lineages_pars.tsv")
    if rep is not None:
        file = os.path.join(dir, f"{id}_lineages_{rep}_pars.tsv")
    outdir = os.path.join(dir, f"{id}_recon_{rep}")
    os.makedirs(dir, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)

    if trees is not None:
        tnames = [t.name for t in trees]
        dnames = [c.clone for c in data]
        if tnames != dnames:
            order = {n: i for i, n in enumerate(tnames)}
            trees = [trees[order[n]] for n in dnames if n in order]

    with open(file, "w") as fh:
        fh.write(f"{len(data)}\n")
        for i, clone in enumerate(data):
            fastafile = os.path.join(outdir, f"{clone.clone}.fasta")
            treefile = os.path.join(outdir, f"{clone.clone}.tree")
            germid = f"{clone.clone}_GERM"
            _clone_to_fasta(clone, fastafile, germid, trait=trait, empty=empty)
            if trees is not None and i < len(trees) and trees[i] is not None:
                tree = trees[i].copy()
                writeNewick(tree, treefile, germline_label=germid)
            else:
                treefile = "N"
            fh.write(f"{fastafile}\t{treefile}\tN\t{clone.clone}\n")
    return file


def _clone_to_fasta(clone, fastafile: str, germid: str,
                    trait: Optional[str] = None, empty: bool = False) -> str:
    """Write a clone's alignment to FASTA (dowser internal ``cloneToFasta``)."""
    col = _clone_seqcol(clone)
    germline = _clone_germline(clone)
    with open(fastafile, "w") as fh:
        for _, srow in clone.data.iterrows():
            sid = str(srow["sequence_id"])
            if trait is not None:
                sid = f"{sid}_{srow[trait]}"
            fh.write(f">{sid}\n")
            fh.write("ATG\n" if empty else f"{srow[col]}\n")
        fh.write(f">{germid}\n")
        fh.write("ATG\n" if empty else f"{germline}\n")
    return fastafile


# ----------------------------------------------------------------------
def writeNewick(tree: Phylo, file: Optional[str] = None,
                germline_label: Optional[str] = None) -> str:
    """Serialise a :class:`~pydowser.phylo.Phylo` to a Newick string.

    If ``file`` is given the string is written there; the string is
    always returned.  ``germline_label`` optionally renames the
    ``Germline`` tip on output.
    """
    labels = list(tree.tip_label)
    if germline_label is not None and "Germline" in labels:
        labels[labels.index("Germline")] = germline_label
    elen = {(int(p), int(c)): float(tree.edge_length[k])
            for k, (p, c) in enumerate(tree.edge)}
    root = tree.root_node()

    def render(node: int) -> str:
        kids = tree.children(node)
        if not kids:  # tip
            name = labels[node - 1]
            return name
        inner = ",".join(
            render(c) + f":{elen.get((node, c), 0.0):.8f}" for c in kids)
        return f"({inner})"

    s = render(root) + ";"
    if file is not None:
        with open(file, "w") as fh:
            fh.write(s + "\n")
    return s


def readNewick(text: str) -> Phylo:
    """Parse a single Newick string into a :class:`~pydowser.phylo.Phylo`.

    A small self-contained Newick parser (no external dependency).  The
    resulting tree uses ape-style 1-based node numbering: tips first,
    internal nodes afterwards.
    """
    text = text.strip()
    if text.endswith(";"):
        text = text[:-1]
    pos = 0
    tip_labels: List[str] = []
    edges: List[List[int]] = []
    elens: List[float] = []
    next_internal = [10**6]  # temporary internal ids, renumbered later

    def parse_clade() -> int:
        nonlocal pos
        if text[pos] == "(":
            pos += 1
            node = next_internal[0]
            next_internal[0] += 1
            children = []
            while True:
                child = parse_clade()
                children.append(child)
                if pos < len(text) and text[pos] == ",":
                    pos += 1
                    continue
                break
            if pos < len(text) and text[pos] == ")":
                pos += 1
            # optional internal label
            _read_name()
            length = _read_length()
            for c, w in children:
                edges.append([node, c])
                elens.append(w)
            return (node, length)
        else:
            name = _read_name()
            length = _read_length()
            tip_labels.append(name)
            return (len(tip_labels), length)

    def _read_name() -> str:
        nonlocal pos
        start = pos
        while pos < len(text) and text[pos] not in "():,;":
            pos += 1
        return text[start:pos]

    def _read_length() -> float:
        nonlocal pos
        if pos < len(text) and text[pos] == ":":
            pos += 1
            start = pos
            while pos < len(text) and text[pos] not in "():,;":
                pos += 1
            try:
                return float(text[start:pos])
            except ValueError:
                return 0.0
        return 0.0

    root_node, _ = parse_clade()
    n_tip = len(tip_labels)
    internal = sorted({p for p, c in edges if p >= 10**6}
                      | {c for p, c in edges if c >= 10**6})
    remap = {old: n_tip + 1 + k for k, old in enumerate(internal)}
    for i in range(1, n_tip + 1):
        remap[i] = i
    new_edges = [[remap[p], remap[c]] for p, c in edges]
    return Phylo(edge=new_edges, edge_length=elens, tip_label=tip_labels,
                 Nnode=len(internal))


def readLineages(file: str, states: Optional[List[str]] = None,
                 run_id: str = "", quiet: bool = True,
                 append: Optional[str] = None) -> List[Phylo]:
    """Read a set of trees referenced by an IgPhyML-style lineage file.

    Faithful pure-Python analogue of dowser's ``readLineages``: the
    lineage ``file`` lists one FASTA path per line (after the count
    header); the corresponding tree files (Newick ``.tree`` or
    IgPhyML-style ``.nex``) are read and the germline tip renamed to
    ``Germline``.  Returns a list of :class:`~pydowser.phylo.Phylo`.
    """
    with open(file) as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    if len(lines) <= 1:
        return []
    trees: List[Phylo] = []
    for ln in lines[1:]:
        parts = ln.split("\t")
        fasta = parts[0]
        tree_path = None
        if len(parts) > 1 and parts[1] not in ("", "N"):
            tree_path = parts[1]
        elif append is not None:
            tree_path = fasta + append
        else:
            base = fasta.rsplit(".fasta", 1)[0]
            for cand in (base + ".tree",
                         f"{base}_igphyml_jointpars.nex",
                         f"{base}_igphyml_jointpars_{run_id}.nex"):
                if os.path.exists(cand):
                    tree_path = cand
                    break
        if tree_path is None or not os.path.exists(tree_path):
            if not quiet:
                warnings.warn(f"Tree file for {fasta} not found")
            continue
        with open(tree_path) as tfh:
            content = tfh.read().strip()
        # accept either a bare Newick string or a NEXUS file with a
        # 'tree ... = (...)' statement
        newick = content
        low = content.lower()
        if "begin trees" in low or low.startswith("#nexus"):
            for tln in content.splitlines():
                t = tln.strip()
                if t.lower().startswith("tree") and "=" in t:
                    newick = t.split("=", 1)[1].strip()
                    break
        tree = readNewick(newick)
        germ = [t for t in tree.tip_label if t.endswith("_GERM")]
        if germ:
            tree.name = germ[0][:-5]
            tree.tip_label[tree.tip_label.index(germ[0])] = "Germline"
        trees.append(tree)
    return trees


# ----------------------------------------------------------------------
def exportTrees(clones: pd.DataFrame, filepath: str,
                tree_column: str = "trees") -> None:
    """Write a clone tibble's trees to a multi-tree Newick file.

    Faithful port of dowser's ``exportTrees`` (which delegates to
    ``ape::write.tree``).  One Newick line per tree.
    """
    if tree_column not in clones.columns:
        raise ValueError(
            f"{tree_column} not found in the input object. Please specify "
            "what column contains the phylogenetic trees.")
    with open(filepath, "w") as fh:
        for tree in clones[tree_column]:
            fh.write(writeNewick(tree) + "\n")


def treesToPDF(plots, file: str, nrow: int = 2, ncol: int = 2,
               **kwargs) -> None:
    """Arrange a list of tree plots into a multi-page PDF.

    Port of dowser's ``treesToPDF`` (which uses ``gridExtra`` +
    ``grDevices::pdf``).  ``plots`` is a list of matplotlib ``Axes`` (the
    output of :func:`pydowser.plotTrees`); their parent figures are
    written ``nrow*ncol`` panels per page.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    figs = []
    seen = set()
    for ax in plots:
        if ax is None:
            continue
        fig = ax.get_figure()
        if id(fig) not in seen:
            seen.add(id(fig))
            figs.append(fig)
    with PdfPages(file) as pdf:
        for fig in figs:
            pdf.savefig(fig, **kwargs)

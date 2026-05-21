"""pydowser: pure-Python port of the R/CRAN package ``dowser``.

A standalone, binary-free Python implementation of the B-cell receptor
phylogenetics toolkit ``dowser`` (Hoehn, Pybus & Kleinstein, *PLoS Comput.
Biol.* 2022; part of the Immcantation framework, Kleinstein lab).  pydowser
builds and analyses B-cell lineage trees from clonal AIRR data:

Core workflow
-------------
* :func:`formatClones`  -- collapse an AIRR data frame into per-clone
  :class:`AirrClone` objects (gap masking, ragged-end masking, duplicate
  collapsing, trait handling, IMGT region/locus assignment).
* :func:`getTrees`      -- build a phylogenetic tree per clone.  The
  maximum-parsimony route (``build="pratchet"``) is a faithful port of
  dowser's parsimony-ratchet pipeline; ``build="pml"`` provides a tractable
  pure-Python maximum-likelihood route.
* :func:`plotTrees`     -- draw the lineage-tree figures (matplotlib).

Discrete-trait phylogenetic tests
---------------------------------
* :func:`findSwitches`  -- reconstruct trait-state switches (migration /
  differentiation / class switching) on observed and permuted trees.
* :func:`testPS`, :func:`testSP`, :func:`testSC` -- the parsimony-score,
  switch-proportion and switch-count tests.

Measurable evolution
--------------------
* :func:`correlationTest` -- root-to-tip divergence-vs-time date
  randomisation test for temporal signal.

Tree utilities
--------------
* :func:`getBootstraps`, :func:`collapseNodes`, :func:`scaleBranches`,
  :func:`resolvePolytomies`, :func:`getDivergence`, :func:`getPathLengths`,
  :func:`getNodeSeq`, :func:`getSeq`, :func:`calcRF`, :func:`rerootTree`,
  :func:`readFasta`, :func:`makeAirrClone`.

Phylogenetics caveats
---------------------
dowser delegates tree building to external binaries (IgPhyML, RAxML,
PHYLIP).  pydowser re-implements the algorithms in pure Python: the
maximum-parsimony score is exact, but for clones with ambiguous /
equally-parsimonious topologies the recovered tree may differ from R's;
the maximum-likelihood route is a tractable JC-model approximation, not a
GTR+gamma reproduction; ``findSwitches`` reconstructs trait states by
maximum parsimony rather than IgPhyML's likelihood model.  See ``README``.

Notes
-----
Reuses ``pyalakazam`` (already-ported alakazam) for sequence masking,
duplicate collapsing and gene-call parsing, and ``pyshazam`` for IMGT
region boundaries.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .clones import AirrClone, formatClones, makeAirrClone
from .datasets import load_example_airr, load_example_changeo
from .likelihood import tree_loglik
from .parsimony import (acctran, ancestral_pars, fitch_score, fitch_states,
                        pratchet, sankoff_score)
from .phylo import Phylo
from .plotting import getPalette, plotTree, plotTrees
from .statistics import correlationTest, runCorrelationTest
from .switches import (countSwitches, findSwitches, testPS, testSC, testSP)
from .trees import (buildPML, buildPratchet, calcRF, collapseNodes, di2multi,
                    getBootstraps, getDivergence, getNodeSeq, getPathLengths,
                    getSeq, getTrees, parsimonyScore, readFasta, rerootTree,
                    resolvePolytomies, scaleBranches, writeFasta)

__all__ = [
    # data structures
    "AirrClone", "Phylo",
    # clone formatting
    "formatClones", "makeAirrClone",
    # tree building
    "getTrees", "buildPratchet", "buildPML",
    # tree utilities
    "rerootTree", "collapseNodes", "scaleBranches", "resolvePolytomies",
    "getBootstraps", "getDivergence", "getPathLengths", "getNodeSeq",
    "getSeq", "calcRF", "parsimonyScore", "di2multi", "readFasta",
    "writeFasta",
    # parsimony engine
    "fitch_score", "sankoff_score", "pratchet", "acctran", "ancestral_pars",
    "fitch_states", "tree_loglik",
    # discrete-trait tests
    "findSwitches", "countSwitches", "testPS", "testSP", "testSC",
    # measurable evolution
    "correlationTest", "runCorrelationTest",
    # plotting
    "plotTrees", "plotTree", "getPalette",
    # datasets
    "load_example_airr", "load_example_changeo",
    # version
    "__version__",
]

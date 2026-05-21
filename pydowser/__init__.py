"""pydowser: pure-Python port of the R/CRAN package ``dowser``.

A standalone, binary-free Python implementation of the B-cell receptor
phylogenetics toolkit ``dowser`` (Hoehn, Pybus & Kleinstein, *PLoS Comput.
Biol.* 2022; part of the Immcantation framework, Kleinstein lab).  pydowser
builds and analyses B-cell lineage trees from clonal AIRR data:

Core workflow
-------------
* :func:`createGermlines` -- reconstruct clonal germline (UCA) sequences
  from an IMGT reference (:func:`readIMGT`).
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
* :func:`makeSkyline`, :func:`getSkylines`, :func:`plotSkylines` -- Bayesian
  skyline (effective-population-size-through-time) reconstruction from a
  completed BEAST run.

Tree / clone / sequence utilities
---------------------------------
* :func:`getBootstraps`, :func:`collapseNodes`, :func:`scaleBranches`,
  :func:`resolvePolytomies`, :func:`getDivergence`, :func:`checkDivergence`,
  :func:`getPathLengths`, :func:`getNodeSeq`, :func:`getSeq`,
  :func:`getAllSeqs`, :func:`getSubTaxa`, :func:`condenseTrees`,
  :func:`calcRF`, :func:`rerootTree`, :func:`downsampleClone`,
  :func:`sampleClones`, :func:`maskCodons`, :func:`maskSequences`,
  :func:`resolveLightChains`, :func:`makeAirrClone`.
* I/O: :func:`readFasta`, :func:`writeFasta`, :func:`dfToFasta`,
  :func:`readIMGT`, :func:`readLineages`, :func:`readModelFile`,
  :func:`makeModelFile`, :func:`exportTrees`, :func:`treesToPDF`,
  :func:`writeCloneSequences`, :func:`writeLineageFile`.

Phylogenetics caveats
---------------------
dowser delegates tree building to external binaries (IgPhyML, RAxML,
PHYLIP, BEAST).  pydowser re-implements the algorithms in pure Python: the
maximum-parsimony score is exact, but for clones with ambiguous /
equally-parsimonious topologies the recovered tree may differ from R's;
the maximum-likelihood route is a tractable JC-model approximation, not a
GTR+gamma reproduction; ``findSwitches`` reconstructs trait states by
maximum parsimony rather than IgPhyML's likelihood model.

External-binary wrappers intentionally **not** ported (pure-Python policy)
-------------------------------------------------------------------------
``buildBeast``, ``buildIgphyml``, ``buildRAxML``, ``readBEAST`` and
``reconIgPhyML`` drive third-party phylogenetics binaries (BEAST, IgPhyML,
RAxML) and have no pure-Python equivalent, so they are deliberately omitted.
For tree building use the native :func:`buildPratchet` (maximum parsimony)
and :func:`buildPML` (maximum likelihood) builders instead.  The BEAST
*orchestrators* :func:`getTimeTrees` / :func:`getTimeTreesIterate` are
ported with faithful validation logic but require an external BEAST
install; the *pure-algorithm* skyline functions (:func:`makeSkyline`,
:func:`getSkylines`, :func:`plotSkylines`) are fully ported and only parse
BEAST text output.

Notes
-----
Reuses ``pyalakazam`` (already-ported alakazam) for sequence masking,
duplicate collapsing and gene-call parsing, and ``pyshazam`` for IMGT
region boundaries.
"""
from __future__ import annotations

__version__ = "0.1.1"

from .clones import (AirrClone, formatClones, getSubclones, makeAirrClone,
                     maskCodons, maskSequences, resolveLightChains,
                     sampleClones)
from .datasets import (load_example_airr, load_example_changeo,
                       load_imgt_human_vdj)
from .germlines import (buildClonalGermline, buildGermline, createGermlines,
                        getGermline, readIMGT, stitchRegions, stitchVDJ)
from .io import (dfToFasta, exportTrees, makeModelFile, readLineages,
                 readModelFile, readNewick, treesToPDF, writeCloneSequences,
                 writeLineageFile, writeNewick)
from .likelihood import tree_loglik
from .parsimony import (acctran, ancestral_pars, fitch_score, fitch_states,
                        pratchet, sankoff_score)
from .phylo import Phylo
from .plotting import condenseTrees, getPalette, plotTree, plotTrees
from .skyline import (getSkylines, getTimeTrees, getTimeTreesIterate,
                      makeSkyline, plotSkylines)
from .statistics import correlationTest, runCorrelationTest
from .switches import countSwitches, findSwitches, testPS, testSC, testSP
from .trees import (bootstrapTrees, buildPML, buildPratchet, calcRF,
                    checkDivergence, collapseNodes, di2multi, downsampleClone,
                    getAllSeqs, getBootstraps, getDivergence, getNodeSeq,
                    getPathLengths, getSeq, getSubTaxa, getTrees,
                    parsimonyScore, readFasta, rerootTree, resolvePolytomies,
                    scaleBranches, writeFasta)

__all__ = [
    # data structures
    "AirrClone", "Phylo",
    # germline reconstruction
    "createGermlines", "readIMGT", "buildGermline", "buildClonalGermline",
    "getGermline", "stitchVDJ", "stitchRegions",
    # clone formatting
    "formatClones", "makeAirrClone", "maskCodons", "maskSequences",
    "resolveLightChains", "getSubclones", "sampleClones", "downsampleClone",
    # tree building
    "getTrees", "buildPratchet", "buildPML",
    # tree utilities
    "rerootTree", "collapseNodes", "scaleBranches", "resolvePolytomies",
    "getBootstraps", "getDivergence", "checkDivergence", "getPathLengths",
    "getNodeSeq", "getSeq", "getAllSeqs", "getSubTaxa", "condenseTrees",
    "calcRF", "parsimonyScore", "di2multi", "bootstrapTrees",
    # parsimony engine
    "fitch_score", "sankoff_score", "pratchet", "acctran", "ancestral_pars",
    "fitch_states", "tree_loglik",
    # discrete-trait tests
    "findSwitches", "countSwitches", "testPS", "testSP", "testSC",
    # measurable evolution
    "correlationTest", "runCorrelationTest",
    # time-resolved phylodynamics
    "makeSkyline", "getSkylines", "plotSkylines", "getTimeTrees",
    "getTimeTreesIterate",
    # plotting
    "plotTrees", "plotTree", "getPalette",
    # I/O
    "readFasta", "writeFasta", "dfToFasta", "readLineages", "readModelFile",
    "makeModelFile", "exportTrees", "treesToPDF", "writeCloneSequences",
    "writeLineageFile", "readNewick", "writeNewick",
    # datasets
    "load_example_airr", "load_example_changeo", "load_imgt_human_vdj",
    # version
    "__version__",
]

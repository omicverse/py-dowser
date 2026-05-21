# py-dowser

**Pure-Python port of the R/CRAN package [`dowser`](https://cran.r-project.org/package=dowser)** — the B-cell receptor phylogenetics toolkit of the [Immcantation](https://immcantation.readthedocs.io/) framework (Hoehn, Pybus & Kleinstein, *PLoS Comput. Biol.* 2022).

`pydowser` builds and analyses B-cell lineage trees from clonal AIRR data: lineage-tree inference, measurable-evolution tests, and migration / differentiation / class-switching tests — all in pure Python (numpy / scipy / pandas / matplotlib + biopython). **No rpy2, no external phylogenetics binaries** (IgPhyML / RAxML / PHYLIP): the tree building and ancestral reconstruction are re-implemented from scratch.

```python
import pydowser as dw

airr   = dw.load_example_airr()                       # dowser's ExampleAirr
clones = dw.formatClones(airr, traits=["sample_id"])   # -> per-clone AirrClone objects
trees  = dw.getTrees(clones, build="pratchet")         # maximum-parsimony lineage trees
dw.plotTrees(trees, trait="sample_id")                 # matplotlib lineage figures
```

## Installation

```bash
pip install pydowser
```

Depends on the already-ported [`pyalakazam`](https://github.com/omicverse/py-alakazam) (sequence masking, duplicate collapsing, gene-call parsing) and [`pyshazam`](https://github.com/omicverse/py-shazam) (IMGT region boundaries).

## What is ported

| dowser function | pydowser | notes |
|---|---|---|
| `createGermlines`, `readIMGT` | `createGermlines`, `readIMGT` | clonal germline (UCA) reconstruction from an IMGT reference — **bit-exact** |
| `formatClones`, `makeAirrClone` | `formatClones`, `makeAirrClone` | gap/end masking, duplicate collapse, trait handling, IMGT regions |
| `maskCodons`, `maskSequences` | same names | mask V-gene codons split by insertions — deterministic, R-parity |
| `resolveLightChains`, `getSubclones` | same names | light-chain V/J subgroup resolution (`getSubclones` deprecated → error) |
| `getTrees` (`build="pratchet"`) | `getTrees`, `buildPratchet` | **maximum parsimony** — faithful, parsimony score exact |
| `getTrees` (`build="pml"`) | `getTrees`, `buildPML` | maximum likelihood — tractable JC approximation (see caveats) |
| `findSwitches` | `findSwitches`, `countSwitches` | parsimony trait reconstruction (see caveats) |
| `testPS`, `testSP`, `testSC` | `testPS`, `testSP`, `testSC` | PS / SP / SC migration & differentiation tests |
| `correlationTest` | `correlationTest`, `runCorrelationTest` | root-to-tip date-randomisation test |
| `makeSkyline`, `getSkylines`, `plotSkylines` | same names | Bayesian skyline from completed BEAST output — `makeSkyline` **bit-exact** |
| `getTimeTrees`, `getTimeTreesIterate` | same names | BEAST time-tree orchestrators (require external BEAST — see caveats) |
| `getBootstraps`, `bootstrapTrees` | same names | bootstrap node support (`bootstrapTrees` deprecated → `findSwitches`) |
| `downsampleClone`, `sampleClones` | same names | clone down-sampling (seeded for reproducibility) |
| `collapseNodes`, `scaleBranches`, `resolvePolytomies`, `condenseTrees` | same names | tree editing |
| `getDivergence`, `checkDivergence`, `getPathLengths` | same names | tree divergence metrics |
| `getNodeSeq`, `getSeq`, `getAllSeqs`, `getSubTaxa` | same names | sequence / clade retrieval |
| `calcRF`, `rerootTree` | same names | Robinson-Foulds, re-rooting |
| `readFasta`, `dfToFasta`, `readModelFile`, `makeModelFile` | same names | FASTA & parsimony-model file I/O |
| `exportTrees`, `readLineages`, `writeCloneSequences`, `writeLineageFile`, `treesToPDF` | same names | tree / lineage / PDF export |
| `plotTrees`, `getPalette` | `plotTrees`, `plotTree`, `getPalette` | matplotlib lineage figures |

The parsimony engine (`fitch_score`, `sankoff_score`, `pratchet`, `acctran`, `ancestral_pars`, `fitch_states`) is exposed directly.

### External-binary wrappers intentionally not ported

`buildBeast`, `buildIgphyml`, `buildRAxML`, `readBEAST` and `reconIgPhyML` drive third-party phylogenetics binaries (BEAST, IgPhyML, RAxML) and have **no pure-Python equivalent**, so they are deliberately omitted in line with pydowser's no-external-binary policy. For tree building use the native pure-Python `buildPratchet` (maximum parsimony) and `buildPML` (maximum likelihood) builders instead. The BEAST *orchestrators* `getTimeTrees` / `getTimeTreesIterate` are ported with faithful input-validation logic but raise an informative `NotImplementedError` because they require an external BEAST install; the *pure-algorithm* skyline functions (`makeSkyline`, `getSkylines`, `plotSkylines`) are fully ported and only parse BEAST's text output.

## R-parity

Validated against **dowser 2.4.1** on its bundled `ExampleAirr` dataset (`tests/test_r_parity.py`, skipped automatically when R is unavailable):

* **`formatClones`** — per-clone sequence counts and germline lengths match R **exactly**.
* **`getTrees` / parsimony** — the maximum-parsimony **score matches R exactly** for the great majority of clones; for a small fraction of large, hard clones the heuristic ratchet search ends at most one step above the global optimum (topology may differ for ambiguous clones — Robinson-Foulds distance asserted small).
* **`getDivergence`** — root-to-tip divergences match R closely.
* **`correlationTest`** — the observed correlation and slope match R; the permutation p-value matches within Monte-Carlo tolerance.

## Phylogenetics caveats (read this)

dowser delegates tree building to external binaries (IgPhyML, RAxML, PHYLIP `dnapars`/`dnaml`). pydowser re-implements the algorithms in pure Python, with these honest limitations:

1. **Maximum parsimony (`build="pratchet"`)** is the R-parity reference. The Fitch parsimony **score is exact**; the parsimony-ratchet topology search (NNI + SPR + bootstrap-reweighting perturbations) is a heuristic — for clones with many equally-parsimonious topologies the recovered tree may differ from R's, and very large clones may occasionally land one step above the global optimum.
2. **Maximum likelihood (`build="pml"`)** is a *tractable approximation*: Felsenstein pruning under the **Jukes-Cantor** model with NNI search and golden-section branch optimisation. It is **not** a bit-for-bit reproduction of `phangorn::optim.pml`'s GTR+gamma optimiser.
3. **`findSwitches`** reconstructs discrete trait states by **maximum parsimony** (Fitch down/up pass) rather than IgPhyML's likelihood model. The PS / SP / SC statistics and their permutation nulls are reproduced faithfully; only the likelihood-weighted tie-breaking of ambiguous states differs.
4. Trees are stored exactly as in `ape` (1-based `edge` matrix, `tip.label`, `Nnode`, `nodes`), so results interoperate cleanly with the rest of the workflow.

## Reference

Hoehn KB, Pybus OG, Kleinstein SH (2022). *Phylogenetic analysis of migration, differentiation, and class switching in B cells.* PLoS Computational Biology. <https://doi.org/10.1371/journal.pcbi.1009885>

## License

GNU Affero General Public License v3 — the same license as the upstream `dowser` package. Original `dowser` © the dowser authors (Kleinstein lab, Immcantation framework).

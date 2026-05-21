"""Smoke tests for pydowser -- exercise the whole API without R.

These run fast (a few small clones) and verify that every public function
produces structurally sane output: clone formatting, parsimony /
maximum-likelihood tree building, the discrete-trait switch tests, the
measurable-evolution correlation test, plotting and the tree utilities.
"""
from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

import pydowser as dw  # noqa: E402

SMALL = [3170, 3184, 3141]


# ----------------------------------------------------------------------
def test_import_and_all():
    assert dw.__version__ == "0.1.1"
    for name in dw.__all__:
        assert hasattr(dw, name), f"missing export {name}"


def test_load_example_data():
    airr = dw.load_example_airr()
    assert airr.shape == (389, 30)
    assert "clone_id" in airr.columns
    ch = dw.load_example_changeo()
    assert ch.shape[0] > 0


# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def clones():
    airr = dw.load_example_airr()
    sub = airr[airr["clone_id"].isin(SMALL)].copy()
    return dw.formatClones(sub, traits=["sample_id"])


@pytest.fixture(scope="module")
def trees(clones):
    return dw.getTrees(clones)


def test_format_clones(clones):
    assert isinstance(clones, pd.DataFrame)
    assert set(["clone_id", "data", "locus", "seqs"]).issubset(clones.columns)
    assert len(clones) == 3
    # sorted by decreasing size
    assert list(clones["seqs"]) == sorted(clones["seqs"], reverse=True)
    for ac in clones["data"]:
        assert isinstance(ac, dw.AirrClone)
        assert len(ac.germline) == len(ac.region) == len(ac.numbers)
        assert "sequence_id" in ac.data.columns
        assert "sequence" in ac.data.columns


def test_make_airr_clone():
    airr = dw.load_example_airr()
    one = airr[airr["clone_id"] == 3184]
    ac = dw.makeAirrClone(one)
    assert isinstance(ac, dw.AirrClone)
    assert ac.clone == "3184"
    assert len(ac.data) >= 2


def test_get_trees_parsimony(trees):
    assert "trees" in trees.columns
    for _, row in trees.iterrows():
        t = row["trees"]
        assert isinstance(t, dw.Phylo)
        assert "Germline" in t.tip_label
        assert t.edge_type == "genetic_distance"
        # branch lengths sum to (parsimony score / germline length)
        ps = dw.parsimonyScore(t, row["data"])
        glen = len(row["data"].germline)
        assert abs(t.total_length() * glen - ps) < glen * 0.05


def test_phylo_structure(trees):
    t = trees["trees"].iloc[0]
    assert t.n_tip == len(t.tip_label)
    assert t.n_edge == t.n_tip + t.Nnode - 1 or t.n_edge >= t.n_tip
    # postorder visits every node
    assert len(set(t.postorder())) == t.n_tip + t.Nnode
    D = t.dist_nodes()
    assert D.shape[0] == t.n_tip + t.Nnode + 1


def test_reroot_preserves_length(trees):
    t = trees["trees"].iloc[0]
    rt = dw.rerootTree(t, "Germline")
    assert abs(rt.total_length() - t.total_length()) < 1e-3


def test_get_divergence(trees):
    t = trees["trees"].iloc[0]
    div = dw.getDivergence(t)
    assert set(div.keys()) == set(t.tip_label)
    assert all(v >= 0 for v in div.values())


def test_get_path_lengths(trees):
    pl = dw.getPathLengths(trees["trees"].iloc[0])
    assert set(["name", "nodes", "distance"]).issubset(pl.columns)
    assert (pl["nodes"] >= 1).all()


def test_scale_branches(trees):
    sb = dw.scaleBranches(trees, edge_type="mutations")
    t = sb["trees"].iloc[0]
    assert t.edge_type == "mutations"
    # round-trip back to genetic distance
    gd = dw.scaleBranches(sb, edge_type="genetic_distance")
    assert abs(gd["trees"].iloc[0].total_length()
               - trees["trees"].iloc[0].total_length()) < 1e-6


def test_collapse_nodes(trees):
    cn = dw.collapseNodes(trees)
    assert cn["trees"].iloc[0].n_tip == trees["trees"].iloc[0].n_tip


def test_get_node_seq(trees):
    t = trees["trees"].iloc[0]
    s = dw.getNodeSeq(trees, node=t.root_node(), tree=t)
    assert isinstance(s, str) and len(s) > 0


def test_calc_rf(trees):
    t = trees["trees"].iloc[0]
    assert dw.calcRF(t, t) == 0


def test_di2multi(trees):
    t = trees["trees"].iloc[0]
    m = dw.di2multi(t, tol=1e-9)
    assert m.n_tip == t.n_tip


# ----------------------------------------------------------------------
def test_parsimony_engine():
    from pydowser.parsimony import (acctran, ancestral_pars, encode_dna,
                                    fitch_score, pratchet)
    labels = ["a", "b", "c", "d", "Germline"]
    seqs = {"a": encode_dna("AAAA"), "b": encode_dna("AAAT"),
            "c": encode_dna("TTAA"), "d": encode_dna("TTAT"),
            "Germline": encode_dna("AAAA")}
    tree = pratchet(seqs, labels, maxit=20, seed=1)
    score = fitch_score(tree, seqs)
    assert score >= 1
    tree2 = acctran(tree, seqs)
    # ACCTRAN branch lengths sum to the parsimony score
    assert abs(tree2.edge_length.sum() - score) < 1e-6
    asr = ancestral_pars(tree, seqs)
    assert all(len(s) == 4 for s in asr.values())


def test_fitch_states():
    from pydowser.parsimony import build_nj_tree, fitch_states
    labels = ["a", "b", "c", "d"]
    D = np.array([[0, 1, 2, 2], [1, 0, 2, 2],
                  [2, 2, 0, 1], [2, 2, 1, 0]], dtype=float)
    tree = build_nj_tree(labels, D)
    states = fitch_states(tree, {"a": "X", "b": "X", "c": "Y", "d": "Y"},
                          ["X", "Y"])
    assert all(isinstance(v, set) for v in states.values())


# ----------------------------------------------------------------------
def test_find_switches_and_tests(trees):
    sw = dw.findSwitches(trees, permutations=20, trait="sample_id", seed=1)
    assert "switches" in sw
    s = sw["switches"]
    assert set(["CLONE", "REP", "TYPE", "FROM", "TO", "SWITCHES"]) \
        .issubset(s.columns)
    assert set(s["TYPE"].unique()) >= {"RECON", "PERMUTE"}

    ps = dw.testPS(s)
    assert "means" in ps and "reps" in ps
    assert "RECON" in ps["means"].columns
    assert (ps["means"]["STAT"] == "PS").all()

    sp = dw.testSP(s)
    assert (sp["means"]["STAT"] == "SP").all()
    assert "DELTA" in sp["means"].columns

    sc = dw.testSC(s)
    assert (sc["means"]["STAT"] == "SC").all()


def test_count_switches(trees):
    row = trees.iloc[0]
    data = row["data"].data
    tip_traits = {str(s): str(v) for s, v
                  in zip(data["sequence_id"], data["sample_id"])}
    states = sorted(set(tip_traits.values()))
    cs = dw.countSwitches(row["trees"], tip_traits, states)
    assert {"FROM", "TO", "SWITCHES"}.issubset(cs.columns)
    assert (cs["SWITCHES"] >= 0).all()


# ----------------------------------------------------------------------
def test_correlation_test():
    airr = dw.load_example_airr()
    g = airr.groupby("clone_id")["timepoint"].nunique()
    multi = g[g > 1].index.tolist()[:4]
    sub = airr[airr["clone_id"].isin(multi)].copy()
    clones = dw.formatClones(sub, traits=["timepoint"])
    trees = dw.getTrees(clones)
    ct = dw.correlationTest(trees, permutations=50, time="timepoint", seed=1)
    assert "correlation" in ct.columns
    assert "slope" in ct.columns
    assert "p" in ct.columns
    valid = ct["correlation"].dropna()
    assert ((valid >= -1.0) & (valid <= 1.0)).all()


# ----------------------------------------------------------------------
def test_get_bootstraps(trees):
    bt = dw.getBootstraps(trees.iloc[[2]], bootstraps=8, seed=1)
    assert "bootstrap_trees" in bt.columns
    assert len(bt["bootstrap_trees"].iloc[0]) <= 8
    nl = bt["trees"].iloc[0].node_label
    assert nl is not None


# ----------------------------------------------------------------------
def test_plot_trees(trees):
    axes = dw.plotTrees(trees, trait="sample_id")
    assert len(axes) == len(trees)
    single = dw.plotTrees(trees["trees"].iloc[0])
    assert len(single) == 1


def test_get_palette():
    pal = dw.getPalette(["A", "B", "Germline"])
    assert pal["Germline"] == "#000000"
    assert pal["A"] != pal["B"]


def test_fasta_io(tmp_path):
    p = tmp_path / "x.fa"
    dw.writeFasta({"s1": "ACGT", "s2": "ACGA"}, str(p))
    seqs = dw.readFasta(str(p))
    assert seqs == {"s1": "ACGT", "s2": "ACGA"}


def test_build_pml():
    airr = dw.load_example_airr()
    sub = airr[airr["clone_id"] == 3184].copy()
    clones = dw.formatClones(sub, traits=["sample_id"])
    t = dw.buildPML(clones["data"].iloc[0])
    assert isinstance(t, dw.Phylo)
    assert "Germline" in t.tip_label
    assert t.total_length() > 0


# ----------------------------------------------------------------------
# germline reconstruction
# ----------------------------------------------------------------------
def test_read_imgt():
    imgt = dw.load_imgt_human_vdj()
    assert "IGH" in imgt
    assert set(imgt["IGH"].keys()) == {"V", "D", "J"}
    assert "IGHV3-11*05" in imgt["IGH"]["V"]


def _synthetic_germline_row():
    imgt = dw.load_imgt_human_vdj()
    sv = imgt["IGH"]["V"]["IGHV3-11*05"]
    sd = imgt["IGH"]["D"]["IGHD3-10*01"]
    sj = imgt["IGH"]["J"]["IGHJ5*02"]
    np1, np2 = 5, 7
    full = sv + "A" * np1 + sd + "C" * np2 + sj
    df = pd.DataFrame([{
        "sequence_id": "s1", "sequence_alignment": full, "clone_id": "c1",
        "v_call": "IGHV3-11*05", "d_call": "IGHD3-10*01",
        "j_call": "IGHJ5*02", "v_germline_start": 1,
        "v_germline_end": len(sv), "d_germline_start": 1,
        "d_germline_end": len(sd), "j_germline_start": 1,
        "j_germline_end": len(sj), "np1_length": np1, "np2_length": np2,
        "locus": "IGH"}])
    return df, imgt, sv, sd, sj


def test_create_germlines():
    df, imgt, sv, sd, sj = _synthetic_germline_row()
    db = dw.createGermlines(df, imgt)
    assert "germline_alignment" in db.columns
    assert "germline_alignment_d_mask" in db.columns
    g = db["germline_alignment_d_mask"].iloc[0]
    assert len(g) == len(sv) + 5 + len(sd) + 7 + len(sj)
    # D region masked
    assert g[len(sv):len(sv) + 5 + len(sd) + 7] == "N" * (5 + len(sd) + 7)
    # full germline keeps the D segment
    full = db["germline_alignment"].iloc[0]
    assert sd in full


def test_build_germline_helpers():
    df, imgt, sv, sd, sj = _synthetic_germline_row()
    rec = df.iloc[0].to_dict()
    v = dw.getGermline(rec, imgt["IGH"]["V"], "V", "v_call",
                       "v_germline_start", "v_germline_end" if False
                       else "v_germline_length")
    # length column not present -> falls back to 0; use createGermlines path
    g = dw.buildGermline(
        {**rec, "v_germline_length": len(sv), "d_germline_length": len(sd),
         "j_germline_length": len(sj)}, imgt["IGH"])
    assert g["full"] is not None
    assert g["vonly"] == sv
    assert dw.stitchVDJ(rec, "AAA", "CC", "GGG", np_char="N").count("N") >= 0


# ----------------------------------------------------------------------
# clone / sequence utilities
# ----------------------------------------------------------------------
def test_mask_codons():
    # identical sequences -> unchanged
    r = dw.maskCodons("s1", "ACGACG", "ACGACG")
    assert r["sequence_masked_v"] == "ACGACG"
    # clean 3-nt insertion -> original sequence kept (no partial codon)
    r = dw.maskCodons("s2", "ACGTTTACG", "ACGACG", keep_alignment=True)
    assert r["sequence_masked_v"] == "ACGACG"
    # partial-codon gap -> frameshift note, masked None
    r = dw.maskCodons("s4", "ACGTTACGTAC", "ACGACGTAC")
    assert r["sequence_masked_v"] is None
    assert "Frameshift" in r["masking_note"]


def test_mask_sequences():
    df = pd.DataFrame([{
        "sequence_id": "s1", "sequence": "ACGACG",
        "sequence_alignment": "ACGACG", "v_sequence_start": 1,
        "v_sequence_end": 6, "v_germline_start": 1, "v_germline_end": 6,
        "junction_length": 3}])
    out = dw.maskSequences(df, mask_cdr3=False)
    assert "sequence_masked" in out.columns
    assert out["sequence_masked"].iloc[0] == "ACGACG"


def test_sample_clones(clones):
    sc = dw.sampleClones(clones, size=3, seed=1)
    assert (sc["seqs"] <= 3).all()
    assert all(len(c.data) <= 3 for c in sc["data"])


def test_downsample_clone(clones):
    ds = dw.downsampleClone(clones["data"].iloc[0], trait="sample_id",
                            tip_switch=3, seed=1)
    assert isinstance(ds["clone"], dw.AirrClone)
    assert len(ds["clone"].data) <= len(clones["data"].iloc[0].data)


def test_resolve_light_chains():
    # heavy + light chains in two cells
    df = pd.DataFrame([
        {"sequence_id": "h1", "cell_id": "c1", "clone_id": "1",
         "locus": "IGH", "v_call": "IGHV1*01", "j_call": "IGHJ1*01",
         "junction_length": 30, "sequence_alignment": "ACGT" * 5},
        {"sequence_id": "h2", "cell_id": "c2", "clone_id": "1",
         "locus": "IGH", "v_call": "IGHV1*01", "j_call": "IGHJ1*01",
         "junction_length": 30, "sequence_alignment": "ACGA" * 5},
        {"sequence_id": "l1", "cell_id": "c1", "clone_id": "1",
         "locus": "IGK", "v_call": "IGKV1*01", "j_call": "IGKJ1*01",
         "junction_length": 27, "sequence_alignment": "TTTT" * 5},
        {"sequence_id": "l2", "cell_id": "c2", "clone_id": "1",
         "locus": "IGK", "v_call": "IGKV1*01", "j_call": "IGKJ1*01",
         "junction_length": 27, "sequence_alignment": "TTTA" * 5}])
    out = dw.resolveLightChains(df)
    assert "clone_subgroup" in out.columns
    assert "clone_subgroup_id" in out.columns


def test_get_subclones_deprecated():
    with pytest.raises(NotImplementedError):
        dw.getSubclones()


# ----------------------------------------------------------------------
# tree utilities
# ----------------------------------------------------------------------
def test_check_divergence(trees):
    cd = dw.checkDivergence(trees, verbose=False)
    assert {"clone", "hamming", "divergence", "diff", "node"}.issubset(
        cd.columns)
    assert (cd["divergence"] >= 0).all()


def test_get_sub_taxa(trees):
    t = trees["trees"].iloc[0]
    tips = dw.getSubTaxa(t.root_node(), t)
    assert set(tips) == set(t.tip_label)
    # a tip node returns itself
    assert dw.getSubTaxa(1, t) == [t.tip_label[0]]


def test_get_all_seqs(trees):
    seqs = dw.getAllSeqs(trees)
    assert {"clone_id", "node_id", "node", "locus", "sequence"}.issubset(
        seqs.columns)
    assert len(seqs) > 0


def test_condense_trees(trees):
    t = trees["trees"].iloc[0]
    states = sorted({str(v) for v
                     in trees["data"].iloc[0].data["sample_id"]})
    ct = dw.condenseTrees(t, states)
    assert len(ct.node_label) == ct.Nnode
    # also accepts a list
    ct2 = dw.condenseTrees([t, t], states)
    assert ct2.n_tip == t.n_tip


def test_bootstrap_trees_deprecated(trees):
    with pytest.warns(UserWarning):
        out = dw.bootstrapTrees(trees, bootstraps=5, trait="sample_id",
                                seed=1)
    assert "switches" in out


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------
def test_df_to_fasta(tmp_path):
    df = pd.DataFrame({"sequence_id": ["x", "y"],
                       "sequence": ["ACG.", "ACGT"]})
    p = tmp_path / "x.fa"
    dw.dfToFasta(df, str(p))
    seqs = dw.readFasta(str(p))
    assert seqs == {"x": "ACG", "y": "ACGT"}
    dw.dfToFasta(df, str(p), imgt_gaps=True)
    assert dw.readFasta(str(p))["x"] == "ACG."


def test_model_file_roundtrip(tmp_path):
    p = tmp_path / "model.txt"
    dw.makeModelFile(str(p), ["A", "B", "C"])
    states = dw.readModelFile(str(p))
    assert list(states.keys()) == ["A", "B", "C"]
    # irreversible constraint variant
    p2 = tmp_path / "model2.txt"
    dw.makeModelFile(str(p2), ["A", "B"], constraints="irrev")
    assert dw.readModelFile(str(p2))


def test_newick_roundtrip(trees):
    t = trees["trees"].iloc[0]
    nwk = dw.writeNewick(t)
    t2 = dw.readNewick(nwk)
    assert set(t2.tip_label) == set(t.tip_label)
    assert t2.Nnode == t.Nnode


def test_export_trees(trees, tmp_path):
    p = tmp_path / "trees.nwk"
    dw.exportTrees(trees, str(p))
    with open(p) as fh:
        lines = [ln for ln in fh if ln.strip()]
    assert len(lines) == len(trees)


def test_write_clone_sequences(trees, tmp_path):
    p = tmp_path / "clones.fa"
    dw.writeCloneSequences(trees, str(p))
    seqs = dw.readFasta(str(p))
    assert len(seqs) > 0
    assert any(k.startswith("Germline|") for k in seqs)


def test_write_lineage_file(trees, tmp_path):
    lf = dw.writeLineageFile(trees, dir=str(tmp_path), id="test", rep=1)
    rl = dw.readLineages(lf)
    assert len(rl) == len(trees)
    for t in rl:
        assert "Germline" in t.tip_label


def test_trees_to_pdf(trees, tmp_path):
    axes = dw.plotTrees(trees)
    p = tmp_path / "trees.pdf"
    dw.treesToPDF(axes, str(p))
    assert p.exists() and p.stat().st_size > 0


# ----------------------------------------------------------------------
# skyline
# ----------------------------------------------------------------------
def _write_skyline_files(d):
    log = d / "x.log"
    log.write_text(
        "Sample\tTreeHeight\tbPopSizes.1\tbPopSizes.2\t"
        "bGroupSizes.1\tbGroupSizes.2\n"
        "0\t1.0\t10.0\t20.0\t2\t2\n"
        "1\t1.2\t12.0\t22.0\t2\t2\n"
        "2\t0.9\t9.0\t19.0\t2\t2\n"
        "3\t1.1\t11.0\t21.0\t2\t2\n")
    trees = d / "x.trees"
    trees.write_text(
        "#NEXUS\nBEGIN TREES;\nTRANSLATE\n1 t1,\n2 t2,\n3 t3,\n4 t4,\n"
        "5 t5;\n"
        "tree STATE_0 = ((1:0.5,2:0.5):0.5,(3:0.3,(4:0.1,5:0.1):0.2):0.7):0.0;\n"
        "tree STATE_1 = ((1:0.6,2:0.6):0.6,(3:0.4,(4:0.2,5:0.2):0.2):0.8):0.0;\n"
        "tree STATE_2 = ((1:0.45,2:0.45):0.45,(3:0.25,(4:0.1,5:0.1):0.15):0.65):0.0;\n"
        "tree STATE_3 = ((1:0.55,2:0.55):0.55,(3:0.35,(4:0.15,5:0.15):0.2):0.75):0.0;\n"
        "END;\n")
    return log, trees


def test_make_skyline(tmp_path):
    log, trees = _write_skyline_files(tmp_path)
    sky = dw.makeSkyline(str(log), str(trees), burnin=0, bins=10,
                         clone_id="c1", max_height="min")
    assert {"bin", "mean", "median", "lci", "uci"}.issubset(sky.columns)
    assert (sky["uci"] >= sky["lci"]).all()
    assert (sky["mean"] > 0).all()


def test_get_skylines_and_plot(tmp_path):
    log, trees = _write_skyline_files(tmp_path)
    # build a clone whose id matches the skyline file naming
    airr = dw.load_example_airr()
    sub = airr[airr["clone_id"] == 3170].copy()
    clones = dw.formatClones(sub, traits=["sample_id"])
    # rename files to match getSkylines naming: <id>_<clone>.log/.trees
    cid = clones["data"].iloc[0].clone
    (tmp_path / f"sky_{cid}.log").write_text(log.read_text())
    (tmp_path / f"sky_{cid}.trees").write_text(trees.read_text())
    out = dw.getSkylines(clones, dir=str(tmp_path), id="sky",
                         time="sample_id", burnin=0, bins=10, forward=False)
    assert "skyline" in out.columns
    assert isinstance(out["skyline"].iloc[0], pd.DataFrame)
    axes = dw.plotSkylines(out)
    assert len(axes) == 1


def test_get_time_trees_requires_beast():
    # BEAST is an external binary; pydowser raises an informative error
    airr = dw.load_example_airr()
    sub = airr[airr["clone_id"].isin(SMALL)].copy()
    clones = dw.formatClones(sub, traits=["timepoint"])
    with pytest.raises(NotImplementedError):
        dw.getTimeTrees(clones, time="timepoint")
    with pytest.raises(NotImplementedError):
        dw.getTimeTreesIterate(clones, time="timepoint")

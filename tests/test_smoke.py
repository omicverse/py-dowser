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
    assert dw.__version__ == "0.1.0"
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

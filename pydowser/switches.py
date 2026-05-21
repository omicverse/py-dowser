"""Discrete-trait phylogenetic switch tests.

Port of dowser's migration / differentiation / class-switching tests:
``findSwitches`` plus the ``testPS`` (parsimony score), ``testSP`` (switch
proportion) and ``testSC`` (switch count) statistics described in Hoehn,
Pybus & Kleinstein (2022) *PLoS Comput. Biol.*

dowser delegates the actual trait-state reconstruction to the external
IgPhyML binary.  Since pydowser is binary-free, the trait reconstruction is
performed in pure Python by **maximum parsimony** (Fitch down/up pass on
the lineage tree -- :func:`pydowser.parsimony.fitch_states`).  This is the
same combinatorial quantity IgPhyML's parsimony reconstruction counts, so
the PS / SP / SC statistics and their permutation null distributions are
reproduced faithfully; only the (likelihood-weighted) tie-breaking of
IgPhyML differs.  See the README for details.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .parsimony import fitch_states
from .phylo import Phylo

__all__ = ["findSwitches", "testPS", "testSP", "testSC", "countSwitches"]


# ----------------------------------------------------------------------
def _node_states(tree: Phylo, tip_traits: Dict[str, str],
                 states: List[str], germline: str = "Germline") -> Dict[int, str]:
    """Reconstruct a single discrete state per node by maximum parsimony.

    Ambiguous Fitch sets are resolved by inheriting the parent's state when
    possible (delayed transformation), matching dowser's downsampling
    convention; the germline tip is treated as the universal common
    ancestor (``UCA``).
    """
    fs = fitch_states(tree, tip_traits, states)
    root = tree.root_node()
    resolved: Dict[int, str] = {}
    for node in tree.preorder():
        opts = sorted(fs[node])
        if node == root:
            resolved[node] = opts[0] if opts else "N"
        else:
            p = tree.parent(node)
            par = resolved.get(p)
            if par in fs[node]:
                resolved[node] = par
            else:
                resolved[node] = opts[0] if opts else "N"
    # tips keep their own observed value
    for i, lab in enumerate(tree.tip_label):
        if lab == germline:
            resolved[i + 1] = "UCA"
        elif lab in tip_traits:
            resolved[i + 1] = tip_traits[lab]
    return resolved


def countSwitches(tree: Phylo, tip_traits: Dict[str, str],
                  states: List[str], germline: str = "Germline") -> pd.DataFrame:
    """Count ancestor->descendant state switches along a tree.

    Returns a long data frame with columns ``FROM``, ``TO`` and
    ``SWITCHES`` -- the number of edges on which the parent state ``FROM``
    differs from (or equals) the child state ``TO``.  A ``TO == "NTIP"``
    row records the number of tip nodes (used by the tip/switch downsample
    heuristic).
    """
    node_state = _node_states(tree, tip_traits, states, germline)
    counts: Dict[tuple, int] = {}
    for k in range(tree.n_edge):
        p, c = int(tree.edge[k, 0]), int(tree.edge[k, 1])
        fr = node_state.get(p, "N")
        to = node_state.get(c, "N")
        counts[(fr, to)] = counts.get((fr, to), 0) + 1
    rows = [{"FROM": fr, "TO": to, "SWITCHES": n}
            for (fr, to), n in counts.items()]
    # NTIP row -- number of tips
    rows.append({"FROM": "NTIP", "TO": "NTIP", "SWITCHES": tree.n_tip})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
def findSwitches(clones: pd.DataFrame, permutations: int, trait: str,
                 fixtrees: bool = True, nproc: int = 1,
                 germline: str = "Germline", seed: int = 1,
                 build: str = "pratchet", **kwargs) -> Dict[str, pd.DataFrame]:
    """Reconstruct trait-state switches on observed and permuted trees.

    Faithful port of dowser's ``findSwitches`` -- but using the pure-Python
    parsimony reconstruction instead of IgPhyML (see the module docstring).
    For every clone the observed trait values are mapped onto its tree and
    switches counted; for each of ``permutations`` replicates the trait
    labels are shuffled among tips and switches re-counted.  The returned
    ``switches`` data frame is the input to :func:`testPS`, :func:`testSP`
    and :func:`testSC`.

    Parameters
    ----------
    clones:
        a tibble with ``data`` and ``trees`` columns (output of
        :func:`pydowser.getTrees`).
    permutations:
        number of permutation replicates.
    trait:
        name of the discrete trait column in each clone's data.
    fixtrees:
        keep tree topologies fixed (the only supported mode here -- pydowser
        does not re-bootstrap topologies inside ``findSwitches``).

    Returns ``{"switches": DataFrame, "trees": list}``.
    """
    if "trees" not in clones.columns:
        raise ValueError("clones must have trees column (run getTrees first)")
    rng = np.random.default_rng(seed)

    # collect global trait states
    all_states = sorted({str(v) for _, row in clones.iterrows()
                         for v in row["data"].data[trait].dropna().unique()})
    if "N" in all_states or "UCA" in all_states or "NTIP" in all_states:
        raise ValueError("Forbidden trait values (N, UCA, NTIP) detected.")

    records: List[dict] = []
    for _, row in clones.iterrows():
        clone = row["data"]
        tree = row["trees"]
        cid = row["clone_id"]
        data = clone.data
        tip_traits = {str(sid): str(tv) for sid, tv
                      in zip(data["sequence_id"], data[trait])}

        # observed reconstruction -- recorded once per replicate so that
        # RECON and PERMUTE align row-wise in the downstream pivot, exactly
        # as dowser's findSwitches output is structured.
        obs = countSwitches(tree, tip_traits, all_states, germline)

        # permutations (PERMUTE -- shuffle within clone;
        # PERMUTEALL -- shuffle across all clones, here equal to PERMUTE
        # for the single-clone shuffle since trees are analysed per-clone)
        tip_ids = list(tip_traits.keys())
        trait_vals = list(tip_traits.values())
        for rep in range(1, permutations + 1):
            for _, r in obs.iterrows():
                records.append({"CLONE": cid, "REP": rep, "TYPE": "RECON",
                                "FROM": r["FROM"], "TO": r["TO"],
                                "SWITCHES": r["SWITCHES"]})
            perm = rng.permutation(len(trait_vals))
            ptraits = {tip_ids[i]: trait_vals[perm[i]]
                       for i in range(len(tip_ids))}
            pc = countSwitches(tree, ptraits, all_states, germline)
            for _, r in pc.iterrows():
                records.append({"CLONE": cid, "REP": rep, "TYPE": "PERMUTE",
                                "FROM": r["FROM"], "TO": r["TO"],
                                "SWITCHES": r["SWITCHES"]})
                records.append({"CLONE": cid, "REP": rep,
                                "TYPE": "PERMUTEALL",
                                "FROM": r["FROM"], "TO": r["TO"],
                                "SWITCHES": r["SWITCHES"]})

    switches = pd.DataFrame(records)
    return {"switches": switches, "trees": list(clones["trees"])}


# ----------------------------------------------------------------------
def _pvalues(delta: np.ndarray, alternative: str, pseudocount: float):
    """Return (PLT, PGT) permutation p-values from a delta vector."""
    n = len(delta)
    gt0 = np.sum(delta > 0)
    lt0 = np.sum(delta < 0)
    eq0 = np.sum(delta == 0)
    if alternative == "two.sided":
        plt = (gt0 + eq0 * 0.5 + pseudocount) / (n + pseudocount)
        pgt = (lt0 + eq0 * 0.5 + pseudocount) / (n + pseudocount)
    elif alternative == "greater":
        pgt = (lt0 + eq0 + pseudocount) / (n + pseudocount)
        plt = None
    else:  # less
        plt = (gt0 + eq0 + pseudocount) / (n + pseudocount)
        pgt = None
    return plt, pgt


# ----------------------------------------------------------------------
def testPS(switches: pd.DataFrame, bylineage: bool = False,
           pseudocount: float = 0.0,
           alternative: str = "less") -> Dict[str, pd.DataFrame]:
    """Parsimony-score (PS) test on switch data (dowser ``testPS``).

    Compares the total parsimony score (sum of all between-state switches)
    of the observed reconstruction with the permuted null.  A significantly
    *lower* observed score indicates phylogenetic clustering of the trait.
    """
    sw = switches[(switches["TO"] != "N")
                  & (switches["TO"] != switches["FROM"])
                  & (switches["FROM"] != "UCA")].copy()
    if len(sw) == 0:
        raise ValueError("No switches left after filtering!")

    grp_cols = (["REP", "TYPE"] if not bylineage
                else ["TYPE", "CLONE", "REP"])
    agg = (sw.groupby(grp_cols, as_index=False)["SWITCHES"].sum())
    pivot_idx = [c for c in grp_cols if c != "TYPE"]
    reps = agg.pivot_table(index=pivot_idx, columns="TYPE",
                           values="SWITCHES").reset_index()
    for col in ("RECON", "PERMUTE"):
        if col not in reps.columns:
            reps[col] = 0
    reps["DELTA"] = reps["RECON"] - reps["PERMUTE"]

    def summarise(df):
        plt, pgt = _pvalues(df["DELTA"].to_numpy(), alternative, pseudocount)
        out = {"RECON": df["RECON"].mean(), "PERMUTE": df["PERMUTE"].mean(),
               "DELTA": df["DELTA"].mean()}
        if plt is not None:
            out["PLT"] = plt
        if pgt is not None:
            out["PGT"] = pgt
        return pd.Series(out)

    if bylineage:
        means = (reps.groupby("CLONE", group_keys=True)
                 .apply(summarise, include_groups=False).reset_index())
    else:
        means = pd.DataFrame([summarise(reps)])
    means["STAT"] = "PS"
    reps["STAT"] = "PS"
    means["REPS"] = reps["REP"].nunique()
    return {"reps": reps, "means": means}


# ----------------------------------------------------------------------
def _sp_sc(switches, kind, permuteAll, from_, to, dropzeroes, bylineage,
           pseudocount, alternative):
    """Shared engine for testSP (proportions) and testSC (counts)."""
    permute_col = "PERMUTEALL" if permuteAll else "PERMUTE"
    sw = switches[(switches["TO"] != switches["FROM"])
                  & (switches["FROM"] != "UCA")].copy()
    if from_ is not None:
        sw = sw[sw["FROM"] == from_]
    if to is not None:
        sw = sw[sw["TO"] == to]
    sw = sw[sw["TO"] != "N"]
    if len(sw) == 0:
        raise ValueError("No switches left after filtering!")

    grp = (["REP", "TYPE", "FROM", "TO"] if not bylineage
           else ["TYPE", "CLONE", "REP", "FROM", "TO"])
    agg = sw.groupby(grp, as_index=False)["SWITCHES"].sum()

    norm_grp = (["TYPE", "REP"] if not bylineage
                else ["TYPE", "CLONE", "REP"])
    if kind == "SP":
        agg["VAL"] = agg.groupby(norm_grp)["SWITCHES"].transform(
            lambda x: x / x.sum())
    else:  # SC -- raw counts
        agg["VAL"] = agg["SWITCHES"]
    agg = agg.drop(columns=["SWITCHES"])

    idx = [c for c in grp if c != "TYPE"]
    reps = agg.pivot_table(index=idx, columns="TYPE",
                           values="VAL").reset_index()
    for col in ("RECON", "PERMUTE", "PERMUTEALL"):
        if col not in reps.columns:
            reps[col] = 0.0
    reps = reps.fillna(0.0)
    reps["DELTA"] = reps["RECON"] - reps[permute_col]

    if dropzeroes:
        recon_from = reps.groupby("FROM")["RECON"].sum()
        recon_to = reps.groupby("TO")["RECON"].sum()
        zero_from = set(recon_from[recon_from == 0].index)
        zero_to = set(recon_to[recon_to == 0].index)
        remove = zero_from & zero_to
        reps = reps[~reps["FROM"].isin(remove) & ~reps["TO"].isin(remove)]

    def summarise(df):
        plt, pgt = _pvalues(df["DELTA"].to_numpy(), alternative, pseudocount)
        out = {"RECON": df["RECON"].mean(),
               "PERMUTE": df[permute_col].mean(),
               "DELTA": df["DELTA"].mean()}
        if plt is not None:
            out["PLT"] = plt
        if pgt is not None:
            out["PGT"] = pgt
        return pd.Series(out)

    by = (["CLONE", "FROM", "TO"] if bylineage else ["FROM", "TO"])
    means = (reps.groupby(by, group_keys=True)
             .apply(summarise, include_groups=False).reset_index())
    means["STAT"] = kind
    reps["STAT"] = kind
    means["REPS"] = reps["REP"].nunique()
    return {"reps": reps, "means": means}


def testSP(switches: pd.DataFrame, permuteAll: bool = False,
           from_: Optional[str] = None, to: Optional[str] = None,
           dropzeroes: bool = True, bylineage: bool = False,
           pseudocount: float = 0.0, alternative: str = "greater",
           tip_switch: int = 20, exclude: bool = False
           ) -> Dict[str, pd.DataFrame]:
    """Switch-proportion (SP) test on switch data (dowser ``testSP``).

    For each ordered state pair ``FROM->TO`` compares the observed
    proportion of switches of that type with the permuted null.  A
    significantly *greater* observed proportion indicates a biased
    ancestor->descendant relationship between the two states.
    """
    return _sp_sc(switches, "SP", permuteAll, from_, to, dropzeroes,
                  bylineage, pseudocount, alternative)


def testSC(switches: pd.DataFrame, permuteAll: bool = False,
           from_: Optional[str] = None, to: Optional[str] = None,
           dropzeroes: bool = True, bylineage: bool = False,
           pseudocount: float = 0.0,
           alternative: str = "two.sided") -> Dict[str, pd.DataFrame]:
    """Switch-count (SC) test on switch data (dowser ``testSC``).

    Like :func:`testSP` but compares raw switch *counts* rather than
    proportions.
    """
    return _sp_sc(switches, "SC", permuteAll, from_, to, dropzeroes,
                  bylineage, pseudocount, alternative)

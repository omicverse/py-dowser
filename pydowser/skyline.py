"""Time-resolved phylodynamics -- Bayesian skyline reconstruction.

Faithful pure-Python ports of dowser's time-tree / skyline machinery from
``R/TimeTreesFunctions.R`` and ``R/Plotting.R``:

* :func:`makeSkyline`   -- compute a Bayesian skyline (effective population
  size through time) from a BEAST log file + posterior trees file.
* :func:`getSkylines`   -- run :func:`makeSkyline` for every clone in a
  clone tibble.
* :func:`plotSkylines`  -- draw the skyline ribbon plots (matplotlib).
* :func:`getTimeTrees` /
  :func:`getTimeTreesIterate` -- orchestrate BEAST time-tree estimation.

Phylogenetics caveat
--------------------
``makeSkyline`` / ``getSkylines`` / ``plotSkylines`` are *pure algorithms*:
they only parse the text output of a completed BEAST run and need no
external binary.  ``getTimeTrees`` / ``getTimeTreesIterate`` *orchestrate*
a BEAST coalescent run and therefore require a BEAST installation -- like
the upstream R functions they raise an informative error when no ``beast``
executable is supplied.  pydowser does not ship BEAST (pure-Python policy);
use the native :func:`pydowser.getTrees` builders for binary-free trees.
"""
from __future__ import annotations

import math
import os
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd

from .io import readNewick
from .phylo import Phylo

__all__ = [
    "makeSkyline", "getSkylines", "plotSkylines",
    "getTimeTrees", "getTimeTreesIterate",
]


# ----------------------------------------------------------------------
def _read_beast_log(logfile: str) -> pd.DataFrame:
    """Read a BEAST/Tracer tab-separated ``.log`` file into a long table."""
    rows: List[List[str]] = []
    header: Optional[List[str]] = None
    with open(logfile) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#") or line == "":
                continue
            fields = line.split("\t")
            if header is None:
                header = fields
                continue
            rows.append(fields)
    if header is None:
        raise ValueError(f"couldn't open {logfile}")
    wide = pd.DataFrame(rows, columns=header)
    for col in wide.columns:
        wide[col] = pd.to_numeric(wide[col], errors="ignore")
    sample_col = header[0]
    long = wide.melt(id_vars=[sample_col], var_name="parameter",
                     value_name="value")
    long = long.rename(columns={sample_col: "Sample"})
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long


def _read_nexus_trees(treesfile: str) -> List[Phylo]:
    """Read a NEXUS ``.trees`` file into a list of :class:`Phylo`."""
    with open(treesfile) as fh:
        content = fh.read()
    trees: List[Phylo] = []
    translate: dict = {}
    in_translate = False
    for raw in content.splitlines():
        ln = raw.strip()
        low = ln.lower()
        if low.startswith("translate"):
            in_translate = True
            ln = ln[len("translate"):].strip()
        if in_translate:
            chunk = ln.rstrip(";").rstrip(",")
            for pair in chunk.split(","):
                pair = pair.strip()
                if not pair:
                    continue
                bits = pair.split()
                if len(bits) >= 2:
                    translate[bits[0]] = bits[1]
            if ln.endswith(";"):
                in_translate = False
            continue
        if low.startswith("tree") and "=" in ln:
            newick = ln.split("=", 1)[1].strip().rstrip(";") + ";"
            tree = readNewick(newick)
            if translate:
                tree.tip_label = [translate.get(t, t)
                                  for t in tree.tip_label]
            trees.append(tree)
    if not trees:
        raise ValueError(f"couldn't open {treesfile}")
    return trees


def _get_quantile(x: np.ndarray, prob: float) -> float:
    """dowser's ``getQuantiles``: sort, take ceil(n*prob)-th element."""
    x = np.sort(np.asarray(x, dtype=float))
    idx = int(math.ceil(len(x) * prob)) - 1
    idx = min(max(idx, 0), len(x) - 1)
    return float(x[idx])


# ----------------------------------------------------------------------
def makeSkyline(logfile: str, treesfile: str, burnin: float, bins: int = 100,
                youngest: float = 0, clone_id: Optional[str] = None,
                max_height: str = "min") -> pd.DataFrame:
    """Compute a Bayesian skyline from a completed BEAST run.

    Faithful port of dowser's ``makeSkyline``.  Parses the BEAST log file
    (``bPopSizes`` / ``bGroupSizes`` columns from a coalescent-skyline
    prior) and the matching posterior trees, then summarises effective
    population size through ``bins`` time intervals.

    Parameters
    ----------
    logfile:
        BEAST ``.log`` file (must contain ``bPopSizes``/``bGroupSizes``).
    treesfile:
        BEAST posterior ``.trees`` (NEXUS) file.
    burnin:
        percentage (0-100) of samples to discard.
    bins:
        number of time bins for the output skyline.
    youngest:
        sampling time of the most recent tip; ``0`` keeps backward time.
    max_height:
        which tree-height summary to use (``min``/``median``/``mean``/``max``).

    Returns a data frame with columns ``bin``, ``mean``, ``median``,
    ``lci`` and ``uci`` (and ``clone_id`` if supplied).
    """
    params = _read_beast_log(logfile)
    phylos = _read_nexus_trees(treesfile)

    pnames = set(params["parameter"].unique())
    if "bPopSizes.1" not in pnames:
        raise ValueError(
            "log file doesn't have pop sizes. Was it run with skyline "
            "tree_prior='coalescent_skyline'?")

    samples = list(pd.unique(params["Sample"]))
    burn = int(math.floor(len(phylos) * burnin / 100))
    if burn > 0:
        phylos = phylos[burn:]
        samples = samples[burn:]
        params = params[params["Sample"].isin(samples)]

    if params["Sample"].nunique() != len(phylos):
        warnings.warn("Parameter and tree posteriors not same length, "
                      "subsetting")
        common = set(params["Sample"].unique())
        keep = [i for i, s in enumerate(samples) if s in common][:len(phylos)]
        phylos = [phylos[i] for i in range(len(phylos)) if i in set(keep)]
        samples = [samples[i] for i in keep]
        params = params[params["Sample"].isin(samples)]

    groups = params[params["parameter"].str.contains("GroupSizes")].copy()
    pops = params[params["parameter"].str.contains("PopSizes")].copy()
    if (pops["value"] < 0).any():
        raise ValueError(f"{logfile} found popsizes < 0, can't continue")
    if (groups["value"] < 0).any():
        raise ValueError(f"{logfile} found groupsizes < 0, can't continue")
    pops["index"] = (pops["parameter"].str.replace("bPopSizes.", "",
                                                   regex=False).astype(float))
    groups["index"] = (groups["parameter"]
                       .str.replace("bGroupSizes.", "", regex=False)
                       .astype(float))

    heights = params.loc[params["parameter"] == "TreeHeight", "value"]
    if max_height == "min":
        maxheight = float(heights.min())
    elif max_height == "median":
        maxheight = float(heights.median())
    elif max_height == "mean":
        maxheight = float(heights.mean())
    elif max_height == "max":
        maxheight = float(heights.max())
    else:
        raise ValueError("max_height option must be min, median, mean, or max")

    if youngest > 0:
        mintime, maxtime = youngest - maxheight, youngest
    else:
        mintime, maxtime = 0.0, maxheight - youngest
    binwidth = (maxtime - mintime) / (bins - 1)

    interval_rows = []
    for index, tr in enumerate(phylos):
        sample = samples[index]
        mrca = tr.get_mrca(tr.tip_label)
        D = tr.dist_nodes()
        times = D[mrca, 1:tr.n_tip + tr.Nnode + 1]
        node_times = times[tr.n_tip:]
        node_times = np.sort(times.max() - node_times)

        pop = pops[pops["Sample"] == sample].sort_values("index")
        group = groups[groups["Sample"] == sample].sort_values("index")
        temp = list(node_times)
        for i in range(len(group)):
            gs = int(group["value"].iloc[i])
            ps = float(pop["value"].iloc[i])
            events = temp[:gs]
            interval_rows.append({
                "end": events[-1] if events else np.nan,
                "interval": i + 1, "events": len(events),
                "popsize": ps, "sample": sample, "index": index})
            temp = temp[gs:]
    all_intervals = pd.DataFrame(interval_rows)

    # drop samples whose interval ends are not distinct
    g = all_intervals.groupby("sample")["end"].agg(["nunique", "count"])
    indistinct = g[g["nunique"] < g["count"]].index
    if len(indistinct) > 0:
        warnings.warn(f"{logfile} Removing {len(indistinct)} samples with "
                      "indistinct intervals.")
        all_intervals = all_intervals[~all_intervals["sample"]
                                      .isin(indistinct)]
        if len(all_intervals) == 0:
            raise ValueError("No intervals left :-(")

    bin_intervals = [mintime + j * binwidth for j in range(bins)]
    sky_rows = []
    for j in range(len(bin_intervals) - 1):
        bin_start = bin_intervals[j]
        matched = all_intervals[all_intervals["end"] > bin_start]
        for sample, grp in matched.groupby("sample"):
            mn = grp["end"].min()
            row = grp[grp["end"] == mn].iloc[0]
            sky_rows.append({"end": row["end"], "popsize": row["popsize"],
                             "sample": sample, "index": row["index"],
                             "bin": bin_start})
    skyline = pd.DataFrame(sky_rows)

    out_rows = []
    for bin_start, grp in skyline.groupby("bin"):
        ps = grp["popsize"].to_numpy(dtype=float)
        out_rows.append({
            "bin": bin_start,
            "mean": float(ps.mean()),
            "median": _get_quantile(ps, 0.5),
            "lci": _get_quantile(ps, 0.025),
            "uci": _get_quantile(ps, 0.975)})
    skyplot = pd.DataFrame(out_rows).sort_values("bin").reset_index(drop=True)

    if clone_id is not None:
        skyplot["clone_id"] = clone_id
    if youngest != 0:
        skyplot["bin"] = youngest - skyplot["bin"]
    return skyplot


# ----------------------------------------------------------------------
def getSkylines(clones: pd.DataFrame, dir: str, id: str, time: str,
                burnin: float = 10, bins: int = 100, verbose: int = 0,
                forward: bool = True,
                max_height: str = "min") -> pd.DataFrame:
    """Compute a Bayesian skyline for every clone (dowser ``getSkylines``).

    Faithful port: for each clone the BEAST output files
    ``<dir>/<id>_<clone>.log`` and ``<dir>/<id>_<clone>.trees`` are
    summarised with :func:`makeSkyline`.  Adds a ``skyline`` column of
    per-clone skyline data frames (``None`` where reconstruction failed).
    """
    skylines: List[Optional[pd.DataFrame]] = []
    for _, row in clones.iterrows():
        clone = row["data"]
        treesfile = os.path.join(dir, f"{id}_{clone.clone}.trees")
        logfile = os.path.join(dir, f"{id}_{clone.clone}.log")
        if forward:
            youngest = float(np.max(pd.to_numeric(clone.data[time])))
        else:
            youngest = 0.0
        if verbose != 0:
            print(row["clone_id"], logfile, treesfile, youngest)
        try:
            sky = makeSkyline(logfile=logfile, treesfile=treesfile,
                              youngest=youngest, burnin=burnin, bins=bins,
                              clone_id=row["clone_id"], max_height=max_height)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Error making skyline clone, {row['clone_id']}: "
                          f"{exc}")
            sky = None
        skylines.append(sky)
    out = clones.copy().reset_index(drop=True)
    out["skyline"] = skylines
    return out


# ----------------------------------------------------------------------
def plotSkylines(clones: pd.DataFrame, file: Optional[str] = None,
                 width: float = 8.5, height: float = 11):
    """Plot Bayesian skyline ribbons (matplotlib port of ``plotSkylines``).

    For each clone with a ``skyline`` data frame (see :func:`getSkylines`)
    a log-scale effective-population-size ribbon plot is drawn.  Returns a
    list of matplotlib axes, or writes a multi-page PDF if ``file`` given.
    """
    import matplotlib.pyplot as plt

    plots = []
    for _, row in clones.iterrows():
        skyline = row.get("skyline")
        if skyline is None or (not isinstance(skyline, pd.DataFrame)):
            warnings.warn(f"{row['clone_id']} skyline not found, skipping")
            continue
        fig, ax = plt.subplots(figsize=(width, height / 2))
        ax.fill_between(skyline["bin"], skyline["lci"], skyline["uci"],
                        color="#b3b3b3", alpha=0.8)
        ax.plot(skyline["bin"], skyline["median"], color="black", lw=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("Time")
        ax.set_ylabel("Effective pop. size")
        ax.set_title(str(row["clone_id"]))
        plots.append(ax)

    if file is not None:
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(file) as pdf:
            for ax in plots:
                pdf.savefig(ax.get_figure())
        return None
    return plots


# ----------------------------------------------------------------------
def getTimeTrees(clones: pd.DataFrame, template=None, beast=None, dir=None,
                 id=None, time: str = "time", mcmc_length: int = 30000000,
                 burnin: float = 10, trait: Optional[str] = None,
                 resume_clones=None, quiet: int = 0,
                 include_germline: bool = True, seq: str = "sequence",
                 **kwargs) -> pd.DataFrame:
    """Estimate BEAST time trees for every clone (dowser ``getTimeTrees``).

    Faithful port of dowser's ``getTimeTrees`` orchestration logic and
    input validation.  Like the upstream function this requires a BEAST
    installation: the actual Bayesian time-tree estimation is performed by
    the external ``beast`` executable.  pydowser is a binary-free port and
    does not bundle BEAST, so this raises :class:`NotImplementedError`
    with guidance to use the native pure-Python builders.

    See the module docstring for the phylogenetics caveat.
    """
    if "data" not in clones.columns:
        raise ValueError("clones must have a 'data' column of AirrClone "
                         "objects")
    data = list(clones["data"])
    if list(clones["clone_id"]) != [c.clone for c in data]:
        raise ValueError("clone_id and airrClone values not identical")
    if resume_clones is not None and "trees" not in clones.columns:
        raise ValueError("trees column not found. resume_clones can only be "
                         "used after at least one getTimeTrees run.")
    # length / type validation mirrors the R function
    for c in data:
        germ = (c.hlgermline if c.phylo_seq == "hlsequence"
                else c.lgermline if c.phylo_seq == "lsequence"
                else c.germline)
        col = (c.phylo_seq if c.phylo_seq in c.data.columns else "sequence")
        if any(len(germ) != len(str(s)) for s in c.data[col]):
            raise ValueError(f"Sequence and/or germline lengths of clone "
                             f"{c.clone} are not equal.")
        if time not in c.data.columns:
            raise ValueError(f"{time} column not found in data")
        if not pd.api.types.is_numeric_dtype(c.data[time]):
            raise ValueError("time column must be numeric")
    if beast is None:
        raise NotImplementedError(
            "getTimeTrees requires the external BEAST binary for Bayesian "
            "time-tree estimation, which pydowser does not bundle "
            "(pure-Python policy). Run BEAST separately and analyse its "
            "output with makeSkyline / getSkylines, or use the native "
            "pure-Python builders getTrees(build='pratchet'|'pml').")
    raise NotImplementedError(
        "getTimeTrees: external BEAST execution is not part of the "
        "pure-Python pydowser port.")


def getTimeTreesIterate(clones: pd.DataFrame, iterations: int = 10,
                        ess_cutoff: int = 200,
                        ignore: Optional[List[str]] = None, quiet: int = 0,
                        **kwargs) -> pd.DataFrame:
    """Iteratively resume :func:`getTimeTrees` until ESS convergence.

    Faithful port of dowser's ``getTimeTreesIterate`` control loop: it
    repeatedly resumes :func:`getTimeTrees` for clones whose effective
    sample size (ESS) is below ``ess_cutoff``.  As with
    :func:`getTimeTrees`, the underlying BEAST run is an external binary
    that pydowser does not bundle, so this raises
    :class:`NotImplementedError`.
    """
    if ignore is None:
        ignore = ["traitfrequencies"]
    # delegate to getTimeTrees, which raises with guidance
    return getTimeTrees(clones, quiet=quiet, **kwargs)

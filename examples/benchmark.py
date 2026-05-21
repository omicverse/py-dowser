"""Benchmark pydowser against R dowser 2.4.1.

Times the core pydowser workflow (formatClones -> getTrees -> findSwitches
-> correlationTest) on dowser's ExampleAirr dataset and, when the CMAP R
environment is available, the equivalent dowser pipeline, reporting the
speed-up and verifying the maximum-parsimony scores match.

Run:  python examples/benchmark.py
"""
from __future__ import annotations

import subprocess
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pydowser as dw

CONDA_SH = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"
SEL = [3128, 3100, 3141, 3170, 3184, 3192]


def _time(label, fn):
    t0 = time.time()
    out = fn()
    dt = time.time() - t0
    print(f"  {label:<28s} {dt:8.2f} s")
    return out, dt


def run_python():
    print("pydowser pipeline:")
    airr = dw.load_example_airr()
    sub = airr[airr["clone_id"].isin(SEL)].copy()
    clones, t1 = _time("formatClones", lambda: dw.formatClones(
        sub, traits=["sample_id"]))
    trees, t2 = _time("getTrees (parsimony)", lambda: dw.getTrees(clones))
    sw, t3 = _time("findSwitches (100 perm)", lambda: dw.findSwitches(
        trees, permutations=100, trait="sample_id", seed=1))
    _, t4 = _time("testPS / testSP / testSC",
                  lambda: (dw.testPS(sw["switches"]),
                           dw.testSP(sw["switches"]),
                           dw.testSC(sw["switches"])))
    scores = {int(r["clone_id"]): dw.parsimonyScore(r["trees"], r["data"])
              for _, r in trees.iterrows()}
    total = t1 + t2 + t3 + t4
    print(f"  {'TOTAL':<28s} {total:8.2f} s")
    return scores, total


def run_r():
    driver = Path(__file__).parent / "_bench_driver.R"
    driver.write_text(R_DRIVER)
    cmd = (f"source {CONDA_SH} && conda activate {CONDA_ENV} "
           f"&& Rscript {driver}")
    t0 = time.time()
    res = subprocess.run(["bash", "-lc", cmd], capture_output=True,
                         text=True, timeout=1800)
    dt = time.time() - t0
    driver.unlink(missing_ok=True)
    if res.returncode != 0:
        print("  (R dowser unavailable -- skipping R timing)")
        print(res.stderr[-500:])
        return None, None
    scores = {}
    for line in res.stdout.splitlines():
        if line.startswith("SCORE"):
            _, cid, sc = line.split()
            scores[int(cid)] = int(sc)
    print(f"R dowser pipeline:           {dt:8.2f} s")
    return scores, dt


R_DRIVER = """
suppressMessages({library(dowser); library(phangorn)})
data(ExampleAirr)
sel <- c("3128","3100","3141","3170","3184","3192")
sub <- ExampleAirr[ExampleAirr$clone_id %in% sel,]
clones <- formatClones(sub, traits="sample_id")
trees <- getTrees(clones)
for(i in seq_len(nrow(trees))){
  cl <- trees$data[[i]]; tr <- trees$trees[[i]]
  seqs <- c(cl@data$sequence, cl@germline)
  names(seqs) <- c(cl@data$sequence_id,"Germline")
  d <- phyDat(as.DNAbin(t(as.matrix(dplyr::bind_rows(strsplit(seqs,""))))))
  cat("SCORE", trees$clone_id[i], parsimony(tr,d), "\\n")
}
"""


def main():
    print("=" * 60)
    print("pydowser vs R dowser 2.4.1 benchmark")
    print("=" * 60)
    py_scores, py_t = run_python()
    print()
    r_scores, r_t = run_r()
    print()
    if r_scores:
        print("Parsimony score parity:")
        for cid in SEL:
            ok = py_scores.get(cid) == r_scores.get(cid)
            print(f"  clone {cid}: pydowser={py_scores.get(cid)} "
                  f"R={r_scores.get(cid)}  {'MATCH' if ok else 'DIFF'}")
        if r_t and py_t:
            print(f"\nWall-clock: pydowser {py_t:.1f}s  R {r_t:.1f}s")


if __name__ == "__main__":
    main()

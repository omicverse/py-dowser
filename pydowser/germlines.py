"""Clonal germline reconstruction -- a port of dowser's ``R/Germlines.R``.

Reconstructs the unmutated germline (UCA) sequence of a clone by stitching
together IMGT reference V, D and J gene segments with N/P junction
nucleotides.  This is the standard Immcantation ``CreateGermlines``
algorithm, used to populate the ``germline_alignment`` /
``germline_alignment_d_mask`` columns that :func:`pydowser.formatClones`
requires.

Ported functions
----------------
* :func:`readIMGT`        -- load an Immcantation-format IMGT reference
  directory (``list[locus][segment] = {allele: sequence}``).
* :func:`createGermlines` -- reconstruct per-clone germline sequences for an
  AIRR data frame.

The single-segment helpers (:func:`getGermline`, :func:`stitchVDJ`,
:func:`stitchRegions`, :func:`buildGermline`, :func:`buildClonalGermline`)
are exposed for completeness and faithfully mirror the R internals.
"""
from __future__ import annotations

import glob
import os
import warnings
from typing import Dict, List, Optional

import pandas as pd

try:  # alakazam helper
    from pyalakazam import getAllele
    _HAVE_ALAKAZAM = True
except Exception:  # pragma: no cover
    _HAVE_ALAKAZAM = False

    def getAllele(call, strip_d=False):  # type: ignore
        if call is None or (isinstance(call, float)):
            return call
        return str(call).split(",")[0]

from .trees import readFasta

__all__ = [
    "readIMGT", "createGermlines", "getGermline", "stitchVDJ",
    "stitchRegions", "buildGermline", "buildClonalGermline",
]


# ----------------------------------------------------------------------
def readIMGT(directory: str, quiet: bool = False) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Load all reference germlines from an Immcantation-format IMGT database.

    Faithful port of dowser's ``readIMGT``.  ``directory`` must contain
    Immcantation-standard FASTA files named ``*_<locus><segment>.fasta``
    (e.g. ``imgt_human_IGHV.fasta``).

    Returns a nested mapping ``database[locus][segment] = {allele: seq}``
    where ``locus`` is e.g. ``IGH`` and ``segment`` is ``V``, ``D`` or
    ``J``.  Sequences are upper-cased.
    """
    files = sorted(glob.glob(os.path.join(directory, "*.fasta"))
                   + glob.glob(os.path.join(directory, "*.fa")))
    if not files:
        raise ValueError("No fasta files found in directory")
    database: Dict[str, Dict[str, Dict[str, str]]] = {}
    sequences = 0
    for file in files:
        fasta_list = {k: str(v).upper() for k, v in readFasta(file).items()}
        info = os.path.basename(file).replace(".fasta", "").replace(".fa", "")
        info = info.split("_")
        if len(info) < 3:
            raise ValueError(f"Improperly formatted input file name: {file}")
        tag = info[-1]
        segment = tag[3:4]
        locus = tag[0:3]
        fasta: Dict[str, str] = {}
        duplicates: List[str] = []
        for n, seq in fasta_list.items():
            name = getAllele(n, strip_d=False)
            if name in fasta:
                duplicates.append(name)
            else:
                fasta[name] = seq
        if duplicates and not quiet:
            warnings.warn(f"Segment IDs not unique in {file}: "
                          + ",".join(duplicates))
        database.setdefault(locus, {})
        database[locus][segment] = fasta
        sequences += len(fasta)
    if not quiet:
        print(f"Read in {sequences} from {len(files)} fasta files")
    return database


# ----------------------------------------------------------------------
def _na(x) -> bool:
    return x is None or (isinstance(x, float) and pd.isna(x)) or x != x


def getGermline(receptor: dict, references: Dict[str, str], segment: str,
                field: str, germ_start: str, germ_length: str,
                pad_char: str = "N") -> Optional[str]:
    """Return one germline gene segment aligned to ``receptor``'s sequence.

    Port of dowser's internal ``getGermline``.  ``segment`` is ``V``,
    ``D`` or ``J``; ``references`` is the allele->sequence dict for that
    locus + segment.  Returns ``None`` if the allele is missing from the
    reference database.
    """
    gene = getAllele(receptor.get(field), strip_d=False)
    start = receptor.get(germ_start)
    length = receptor.get(germ_length)
    if _na(start):
        start = 1
    if _na(length):
        length = 0
    start = int(start)
    length = int(length)

    if segment in ("V", "J"):
        if _na(gene):
            return pad_char * length
        if gene in references:
            seq = references[gene]
            seq_sub = seq[start - 1:]
            pad = max(0, length - len(seq_sub))
            return seq[start - 1:start - 1 + length] + pad_char * pad
        return None
    elif segment == "D":
        if _na(gene):
            return ""
        if gene in references:
            seq = references[gene]
            return seq[start - 1:start - 1 + length]
        return None
    raise ValueError(f"Segment {segment} not found")


def stitchVDJ(receptor: dict, v_seq: str, d_seq: str, j_seq: str,
              np1_length: str = "np1_length", np2_length: str = "np2_length",
              np_char: str = "N") -> str:
    """Concatenate V, N1, D, N2, J germline pieces (dowser ``stitchVDJ``)."""
    np1 = receptor.get(np1_length)
    np2 = receptor.get(np2_length)
    np1 = 0 if _na(np1) else int(np1)
    np2 = 0 if _na(np2) else int(np2)
    return (v_seq + np_char * np1 + d_seq + np_char * np2 + j_seq)


def stitchRegions(receptor: dict, v_seq: str, d_seq: str, j_seq: str,
                  np1_length: str = "np1_length",
                  np2_length: str = "np2_length") -> str:
    """Return a per-position VDJ region string (dowser ``stitchRegions``)."""
    np1 = receptor.get(np1_length)
    np2 = receptor.get(np2_length)
    np1 = 0 if _na(np1) else int(np1)
    np2 = 0 if _na(np2) else int(np2)
    return ("V" * len(v_seq) + "N" * np1 + "D" * len(d_seq)
            + "N" * np2 + "J" * len(j_seq))


def buildGermline(receptor: dict, references: Dict[str, Dict[str, str]],
                  seq: str = "sequence_alignment", clone: str = "clone_id",
                  v_call: str = "v_call", d_call: str = "d_call",
                  j_call: str = "j_call",
                  v_germ_start: str = "v_germline_start",
                  v_germ_length: str = "v_germline_length",
                  d_germ_start: str = "d_germline_start",
                  d_germ_length: str = "d_germline_length",
                  j_germ_start: str = "j_germline_start",
                  j_germ_length: str = "j_germline_length",
                  np1_length: str = "np1_length",
                  np2_length: str = "np2_length") -> Dict[str, Optional[str]]:
    """Reconstruct full / D-masked / V-only germlines for one receptor.

    Faithful port of dowser's ``buildGermline``.  ``references`` is the
    per-locus mapping ``{"V": {...}, "D": {...}, "J": {...}}``.  Returns a
    dict with keys ``full``, ``dmask``, ``vonly`` and ``regions``; values
    are ``None`` when reconstruction fails.
    """
    germ_v = getGermline(receptor, references.get("V", {}), "V",
                         v_call, v_germ_start, v_germ_length)
    germ_d = getGermline(receptor, references.get("D", {}), "D",
                         d_call, d_germ_start, d_germ_length)
    germ_j = getGermline(receptor, references.get("J", {}), "J",
                         j_call, j_germ_start, j_germ_length)

    out: Dict[str, Optional[str]] = {"full": None, "dmask": None,
                                     "vonly": None, "regions": None}
    if germ_v is None or germ_d is None or germ_j is None:
        return out

    germ_seq = stitchVDJ(receptor, germ_v, germ_d, germ_j,
                         np1_length=np1_length, np2_length=np2_length)
    regions = stitchRegions(receptor, germ_v, germ_d, germ_j,
                            np1_length=np1_length, np2_length=np2_length)
    obs = str(receptor.get(seq, ""))
    if len(obs) == 0:
        raise ValueError(f"Sequence is missing from the sequence field "
                         f"{receptor.get(clone)}")
    if len(germ_seq) - len(obs) != 0:
        raise ValueError(
            f"Germline sequence differs from input sequence by "
            f"{len(germ_seq) - len(obs)} in clone {receptor.get(clone)}")

    pad_char = "N"
    mask_len = len(germ_seq) - len(germ_v) - len(germ_j)
    germ_dmask = germ_seq[:len(germ_v)] + pad_char * mask_len
    germ_dmask = germ_dmask + germ_seq[len(germ_dmask):]
    if len(germ_dmask) - len(obs) != 0:
        raise ValueError(
            f"Germline dmask sequence differs from input sequence by "
            f"{len(germ_dmask) - len(obs)} in clone {receptor.get(clone)}")

    out["full"] = germ_seq
    out["dmask"] = germ_dmask
    out["vonly"] = germ_v
    out["regions"] = regions
    return out


def buildClonalGermline(receptors: pd.DataFrame,
                         references: Dict[str, Dict[str, Dict[str, str]]],
                         chain: str = "IGH", use_regions: bool = False,
                         vonly: bool = False, seq: str = "sequence_alignment",
                         id: str = "sequence_id", clone: str = "clone_id",
                         v_call: str = "v_call", j_call: str = "j_call",
                         j_germ_length: str = "j_germline_length",
                         **kwargs) -> pd.DataFrame:
    """Reconstruct one consensus germline for an entire clone.

    Faithful port of dowser's ``buildClonalGermline``.  The consensus V
    and J calls (most frequent in the clone) and the longest sequence are
    used to pick a representative receptor whose germline is reconstructed
    and broadcast to every sequence in the clone.
    """
    receptors = receptors.copy().reset_index(drop=True)
    pad_char = "N"
    v_dict = [getAllele(x, strip_d=False) for x in receptors[v_call]]
    j_dict = [getAllele(x, strip_d=False) for x in receptors[j_call]]
    seq_len = [len(str(x)) for x in receptors[seq]]

    vcounts = pd.Series(v_dict).value_counts()
    jcounts = pd.Series(j_dict).value_counts()
    v_cons = set(vcounts[vcounts == vcounts.max()].index)
    j_cons = set(jcounts[jcounts == jcounts.max()].index)
    max_len = max(seq_len)

    cons_index = [v in v_cons and j in j_cons and sl == max_len
                  for v, j, sl in zip(v_dict, j_dict, seq_len)]
    if not any(cons_index):
        cons_index = [v in v_cons and j in j_cons
                      for v, j in zip(v_dict, j_dict)]

    if not any(cons_index):
        warnings.warn(f"Clone {receptors[clone].iloc[0]} no sequence found "
                      "with both consensus V and J calls.")
        germlines: Dict[str, Optional[str]] = {"full": None, "dmask": None,
                                               "regions": None, "vonly": None}
    else:
        cons_rows = receptors[pd.Series(cons_index).values]
        cons_id = sorted(cons_rows[id].astype(str))[0]
        cons = receptors[receptors[id].astype(str) == cons_id].iloc[0].to_dict()
        gap_length = max_len - len(str(cons[seq]))
        if gap_length > 0:
            cons[j_germ_length] = (0 if _na(cons.get(j_germ_length))
                                   else int(cons[j_germ_length])) + gap_length
            cons[seq] = str(cons[seq]) + pad_char * gap_length

        # pad all clone sequences to the max length
        new_seqs = []
        for i in range(len(receptors)):
            l = max_len - len(str(receptors[seq].iloc[i]))
            jl = receptors[j_germ_length].iloc[i]
            receptors.at[i, j_germ_length] = (0 if _na(jl) else int(jl)) + l
            new_seqs.append(str(receptors[seq].iloc[i]) + pad_char * l)
        receptors[seq] = new_seqs

        sub_db = references.get(chain, {})
        if len(sub_db) == 0:
            raise ValueError(f"Reference database for {chain} is empty")
        try:
            germlines = buildGermline(cons, references=sub_db, seq=seq,
                                      clone=clone, v_call=v_call,
                                      j_call=j_call,
                                      j_germ_length=j_germ_length, **kwargs)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Clone {receptors[clone].iloc[0]} germline "
                          f"reconstruction error.\n {exc}")
            germlines = {"full": None, "dmask": None, "regions": None,
                         "vonly": None}

    receptors["germline_alignment"] = germlines["full"]
    receptors["germline_alignment_d_mask"] = germlines["dmask"]
    if use_regions:
        receptors["regions"] = germlines["regions"]
    if vonly:
        receptors["germline_alignment_vonly"] = germlines["vonly"]
    return receptors


# ----------------------------------------------------------------------
def createGermlines(data: pd.DataFrame,
                    references: Dict[str, Dict[str, Dict[str, str]]],
                    locus: str = "locus", trim_lengths: bool = False,
                    seq: str = "sequence_alignment", v_call: str = "v_call",
                    d_call: str = "d_call", j_call: str = "j_call",
                    id: str = "sequence_id", clone: str = "clone_id",
                    v_germ_start: str = "v_germline_start",
                    v_germ_end: str = "v_germline_end",
                    v_germ_length: str = "v_germline_length",
                    d_germ_start: str = "d_germline_start",
                    d_germ_end: str = "d_germline_end",
                    d_germ_length: str = "d_germline_length",
                    j_germ_start: str = "j_germline_start",
                    j_germ_end: str = "j_germline_end",
                    j_germ_length: str = "j_germline_length",
                    np1_length: str = "np1_length",
                    np2_length: str = "np2_length", na_rm: bool = True,
                    fields: Optional[List[str]] = None,
                    verbose: int = 0) -> pd.DataFrame:
    """Reconstruct clonal germline sequences for an AIRR data frame.

    Faithful port of dowser's ``createGermlines``.  For every clone the
    consensus germline is reconstructed from the IMGT ``references``
    (see :func:`readIMGT`) and the ``germline_alignment`` /
    ``germline_alignment_d_mask`` columns are added.  This is the
    preprocessing step that supplies the germline columns required by
    :func:`pydowser.formatClones`.

    Parameters
    ----------
    data:
        AIRR-format data frame with V/D/J gene calls and germline indices.
    references:
        nested IMGT reference returned by :func:`readIMGT`.
    locus:
        name of the locus *column* (not a value).
    na_rm:
        drop clones whose germline reconstruction failed.

    Returns the input data frame with ``germline_alignment`` and
    ``germline_alignment_d_mask`` columns added/updated.
    """
    if len(data) == 0:
        warnings.warn("No data provided!")
        return data
    if locus in ("IGH", "IGK", "IGL"):
        raise ValueError(
            "locus option now indicates locus column name, not value.")
    data = data.copy().reset_index(drop=True)
    if locus not in data.columns:
        warnings.warn(f"{locus} column not found, extracting locus from V call")
        data[locus] = data[v_call].astype(str).str[:3]

    fields = list(fields) if fields else []
    required = [seq, id, clone, np1_length, v_call, d_call, j_call,
                v_germ_start, v_germ_end, d_germ_start, d_germ_end,
                j_germ_start, j_germ_end, locus] + fields
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("Required columns not found in data: "
                         + ", ".join(missing))
    if data[clone].isna().any():
        raise ValueError("NA values in clone id column found, please remove.")

    data[d_call] = data[d_call].replace("", pd.NA)

    # derive segment lengths from start/end if not present
    for length, start, end in (
            (v_germ_length, v_germ_start, v_germ_end),
            (d_germ_length, d_germ_start, d_germ_end),
            (j_germ_length, j_germ_start, j_germ_end)):
        if length not in data.columns:
            data[length] = data[end] - data[start] + 1
        else:
            na = data[length].isna()
            if na.any():
                data.loc[na, length] = (data.loc[na, end]
                                        - data.loc[na, start] + 1)
    if data[v_germ_length].isna().any() or data[j_germ_length].isna().any():
        raise ValueError("Missing values in v_germ_length or j_germ_length")

    # warn about / optionally trim sequences longer than predicted germline
    g_lengths = (data[[v_germ_length, np1_length, d_germ_length,
                       np2_length, j_germ_length]]
                 .fillna(0).astype(float).sum(axis=1))
    g_diffs = data[seq].astype(str).str.len() - g_lengths
    if (g_diffs > 0).any() and not trim_lengths:
        warnings.warn(
            f"{int((g_diffs > 0).sum())} sequence lengths longer than "
            "predicted germlines, consider setting trim_lengths=True")
    elif (g_diffs > 0).any() and trim_lengths:
        new = data[seq].astype(str).tolist()
        for i, d in enumerate(g_diffs):
            if d > 0:
                tail = new[i][int(len(new[i]) - d):]
                if not any(c in "ATCG" for c in tail):
                    new[i] = new[i][:int(len(new[i]) - d)]
        data[seq] = new

    group_cols = list(dict.fromkeys([clone] + fields))
    parts = []
    for _, grp in data.groupby(group_cols, sort=False, dropna=False):
        if verbose > 0:
            print(grp[clone].iloc[0])
        glines = []
        for loc, lgrp in grp.groupby(locus, sort=False, dropna=False):
            glines.append(buildClonalGermline(
                lgrp, references=references, chain=str(loc), seq=seq,
                v_call=v_call, d_call=d_call, j_call=j_call, id=id,
                clone=clone, v_germ_start=v_germ_start,
                v_germ_length=v_germ_length, d_germ_start=d_germ_start,
                d_germ_length=d_germ_length, j_germ_start=j_germ_start,
                j_germ_length=j_germ_length, np1_length=np1_length,
                np2_length=np2_length))
        parts.append(pd.concat(glines, ignore_index=True))
    results = pd.concat(parts, ignore_index=True)

    if na_rm:
        bad = results.loc[results["germline_alignment_d_mask"].isna(), clone]
        bad = bad.dropna().unique()
        if len(bad) > 0:
            warnings.warn(f"Removing {len(bad)} failed clonal germlines. "
                          f"Clones: {','.join(map(str, bad))}")
            results = results[~results["germline_alignment_d_mask"].isna()]
    return results.reset_index(drop=True)

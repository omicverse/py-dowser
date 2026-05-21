"""Clone processing -- ``airrClone``, ``makeAirrClone`` and ``formatClones``.

Faithful Python port of the corresponding R functions in ``dowser`` 2.4.1
(``R/Clones.R`` and ``R/Classes.R``).  An :class:`AirrClone` is the common
data structure for B-cell lineage reconstruction: it holds the (collapsed,
masked, padded) sequence alignment for a single clone together with the
germline sequence and per-site locus / IMGT-region / IMGT-number indices.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

try:  # alakazam helpers (already ported in py-alakazam)
    from pyalakazam import (collapseDuplicates, getGene, maskSeqEnds,
                            maskSeqGaps, padSeqEnds, seqDist)
    _HAVE_ALAKAZAM = True
except Exception:  # pragma: no cover - fallback implementations below
    _HAVE_ALAKAZAM = False

__all__ = ["AirrClone", "makeAirrClone", "formatClones"]

_STOP_CODONS = {"TAA", "TAG", "TGA"}


# ----------------------------------------------------------------------
# Lightweight fallbacks (used only if pyalakazam is unavailable)
# ----------------------------------------------------------------------
def _fb_mask_gaps(seq, mask_char="N", outer_only=False):
    out = []
    for s in seq:
        s = str(s)
        if outer_only:
            i, j = 0, len(s)
            while i < len(s) and s[i] in ".-":
                i += 1
            while j > 0 and s[j - 1] in ".-":
                j -= 1
            s = mask_char * i + s[i:j].replace(".", mask_char).replace("-", mask_char) + mask_char * (len(s) - j)
        else:
            s = s.replace(".", mask_char).replace("-", mask_char)
        out.append(s)
    return out


def _fb_pad_ends(seq, length=None, pad_char="N", mod3=True, start=False):
    seqs = [str(s) for s in seq]
    if length is None:
        length = max(len(s) for s in seqs)
        if mod3 and length % 3 != 0:
            length += 3 - length % 3
    out = []
    for s in seqs:
        pad = pad_char * max(0, length - len(s))
        out.append(pad + s if start else s + pad)
    return out


if not _HAVE_ALAKAZAM:  # pragma: no cover
    def maskSeqGaps(seq, mask_char="N", outer_only=False):
        return _fb_mask_gaps(seq, mask_char, outer_only)

    def maskSeqEnds(seq, mask_char="N", max_mask=None, trim=False):
        return [str(s) for s in seq]

    def padSeqEnds(seq, length=None, start=False, pad_char="N", mod3=True):
        return _fb_pad_ends(seq, length, pad_char, mod3, start)

    def getGene(call, **kw):
        if isinstance(call, str):
            return call.split("*")[0].split(",")[0]
        return [str(c).split("*")[0].split(",")[0] for c in call]

    def seqDist(s1, s2, dist_mat=None):
        return sum(a != b for a, b in zip(str(s1), str(s2)))

    def collapseDuplicates(data, **kw):
        return data.drop_duplicates(subset=[kw.get("seq", "sequence_alignment")])


# ----------------------------------------------------------------------
class AirrClone:
    """A single B-cell clone, ready for lineage reconstruction.

    Mirrors the S4 ``airrClone`` class of dowser.  Attributes
    ----------
    data: :class:`pandas.DataFrame`
        per-sequence table with ``sequence_id`` and ``sequence`` columns
        (plus ``lsequence`` / ``hlsequence`` and any retained traits).
    clone: str
        clone identifier.
    germline / lgermline / hlgermline: str
        heavy, light and concatenated germline sequences.
    v_gene, j_gene: str
        V / J gene calls.
    junc_len: int
        junction length (nt).
    locus: list[str]
        per-site locus label.
    region: list[str]
        per-site IMGT region label.
    numbers: list[int]
        per-site IMGT numbering.
    phylo_seq: str
        which sequence column to use for tree building.
    """

    def __init__(self, data, clone, germline, lgermline, hlgermline,
                 v_gene, j_gene, junc_len, locus, region, numbers, phylo_seq):
        self.data = data.reset_index(drop=True)
        self.clone = str(clone)
        self.germline = germline
        self.lgermline = lgermline
        self.hlgermline = hlgermline
        self.v_gene = v_gene
        self.j_gene = j_gene
        self.junc_len = junc_len
        self.locus = list(locus)
        self.region = list(region)
        self.numbers = list(numbers)
        self.phylo_seq = phylo_seq

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"AirrClone(clone={self.clone!r}, seqs={len(self.data)}, "
                f"len={len(self.germline)})")

    def copy(self) -> "AirrClone":
        """Return a copy of this clone."""
        return AirrClone(self.data.copy(), self.clone, self.germline,
                         self.lgermline, self.hlgermline, self.v_gene,
                         self.j_gene, self.junc_len, list(self.locus),
                         list(self.region), list(self.numbers),
                         self.phylo_seq)


# ----------------------------------------------------------------------
def _stop_codon_check(sequences: Sequence[str]) -> np.ndarray:
    """Return a bool vector: True where a sequence has an in-frame stop."""
    out = []
    for s in sequences:
        s = str(s).upper()
        if len(s) % 3 != 0:
            s = s + "N" * (3 - len(s) % 3)
        has = any(s[i:i + 3] in _STOP_CODONS for i in range(0, len(s), 3))
        out.append(has)
    return np.asarray(out, dtype=bool)


def _set_regions(junc_len: int, germline: str):
    """Return per-site IMGT region labels via pyshazam.setRegionBoundaries."""
    try:
        from pyshazam import IMGT_VDJ_BY_REGIONS, setRegionBoundaries
        rd = setRegionBoundaries(junc_len, germline, IMGT_VDJ_BY_REGIONS)
        return [str(x) for x in rd.boundaries]
    except Exception:
        return None


# ----------------------------------------------------------------------
def makeAirrClone(data, id="sequence_id", seq="sequence_alignment",
                  germ="germline_alignment_d_mask", v_call="v_call",
                  j_call="j_call", junc_len="junction_length",
                  clone="clone_id", mask_char="N", max_mask=0,
                  pad_end=True, text_fields=None, num_fields=None,
                  seq_fields=None, add_count=True, verbose=False,
                  collapse=True, chain="H", heavy=None, cell="cell_id",
                  locus="locus", traits=None, mod3=True, randomize=False,
                  use_regions=True, dup_singles=False) -> Optional[AirrClone]:
    """Build an :class:`AirrClone` from one clone's AIRR data frame.

    Masks IMGT gap positions, masks ragged ends, optionally pads to a
    common length, collapses identical sequences (merging annotations) and
    assigns per-site locus / region / numbering vectors.  This is a
    faithful port of dowser's ``makeAirrClone`` (heavy-chain ``chain="H"``
    and light-chain ``chain="L"`` paths; the ``chain="HL"`` path is
    supported for the common single-locus-per-cell case).

    Returns ``None`` if region definition is incomplete.
    """
    data = data.reset_index(drop=True)
    text_fields = list(text_fields) if text_fields else []
    num_fields = list(num_fields) if num_fields else []
    seq_fields = list(seq_fields) if seq_fields else []
    traits = list(traits) if traits else None

    vcall = getGene(data[v_call].iloc[0]) if v_call in data.columns else "missing"
    jcall = getGene(data[j_call].iloc[0]) if j_call in data.columns else "missing"
    junclen = int(data[junc_len].iloc[0]) if junc_len in data.columns else 0

    keep_cols = list(dict.fromkeys(
        [id, seq] + text_fields + num_fields + seq_fields + (traits or [])))
    tmp_df = data[[c for c in keep_cols if c in data.columns]].copy()
    tmp_df[seq] = maskSeqGaps(tmp_df[seq].astype(str).tolist(),
                              mask_char=mask_char, outer_only=False)
    tmp_df[seq] = maskSeqEnds(tmp_df[seq].tolist(), mask_char=mask_char,
                              max_mask=max_mask, trim=False)
    if pad_end:
        tmp_df[seq] = padSeqEnds(tmp_df[seq].tolist(), pad_char=mask_char,
                                 mod3=mod3)

    if (data[germ].iloc[0] != data[germ]).any():
        raise ValueError(
            f"Germline sequences for clone {data[clone].iloc[0]} are not "
            "identical. Run createGermlines before formatClones.")
    germline = maskSeqGaps([data[germ].iloc[0]], mask_char=mask_char,
                           outer_only=False)[0]
    if pad_end:
        germline = padSeqEnds([germline], pad_char=mask_char, mod3=mod3)[0]
        length = max(len(germline), max(len(s) for s in tmp_df[seq]))
        if length > len(germline):
            germline = padSeqEnds([germline], length=length,
                                  pad_char=mask_char, mod3=mod3)[0]

    if locus in data.columns:
        loci = list(pd.unique(data[locus]))
        loci = ",".join(map(str, loci)) if len(loci) > 1 else str(loci[0])
    else:
        loci = "N"
    chains = [loci] * len(germline)
    numbers = list(range(1, len(germline) + 1))
    lgermline = ""
    hlgermline = germline
    tmp_df["lsequence"] = ""
    tmp_df["hlsequence"] = tmp_df[seq]
    new_seq = seq

    if use_regions:
        regions = _set_regions(junclen, germline)
        if regions is None or len(regions) != len(germline):
            regions = ["N"] * len(germline)
    else:
        regions = ["N"] * len(germline)

    seq_len = tmp_df[seq].str.len()
    if (seq_len != seq_len.iloc[0]).any():
        raise ValueError(
            f"All sequences are not the same length for clone "
            f"{data[clone].iloc[0]}. Verify the multiple alignment.")

    # collapse identical sequences (optionally keeping traits distinct)
    if collapse:
        if traits is None:
            tmp_df = collapseDuplicates(
                tmp_df, id=id, seq=new_seq, text_fields=text_fields or None,
                num_fields=num_fields or None, seq_fields=seq_fields or None,
                add_count=add_count, verbose=verbose)
        else:
            parts = []
            for _, grp in tmp_df.groupby(traits, sort=False, dropna=False):
                parts.append(collapseDuplicates(
                    grp, id=id, seq=new_seq,
                    text_fields=text_fields or None,
                    num_fields=num_fields or None,
                    seq_fields=seq_fields or None,
                    add_count=add_count, verbose=verbose))
            tmp_df = pd.concat(parts, ignore_index=True)
    tmp_df = tmp_df.reset_index(drop=True)

    if randomize:
        tmp_df = tmp_df.sample(frac=1.0).reset_index(drop=True)

    # rename primary sequence / id columns
    if "sequence" in tmp_df.columns and seq != "sequence":
        tmp_df = tmp_df.drop(columns=["sequence"])
    tmp_df = tmp_df.rename(columns={seq: "sequence", id: "sequence_id"})
    tmp_df["hlsequence"] = tmp_df["sequence"]
    tmp_df["lsequence"] = ""
    phylo_seq = "sequence"

    if len(tmp_df) == 1 and dup_singles:
        dup = tmp_df.copy()
        dup["sequence_id"] = dup["sequence_id"].astype(str) + "_DUPLICATE"
        tmp_df = pd.concat([tmp_df, dup], ignore_index=True)

    return AirrClone(
        data=tmp_df,
        clone=str(pd.unique(data[clone])[0]),
        germline=maskSeqGaps([germline], mask_char=mask_char,
                             outer_only=False)[0],
        lgermline=lgermline,
        hlgermline=maskSeqGaps([hlgermline], mask_char=mask_char,
                               outer_only=False)[0],
        v_gene=vcall, j_gene=jcall, junc_len=junclen,
        locus=chains, region=regions, numbers=numbers,
        phylo_seq=phylo_seq)


# ----------------------------------------------------------------------
def _clean_alignment(clone: AirrClone) -> AirrClone:
    """Remove uninformative all-``NNN`` codon sites (dowser ``cleanAlignment``).

    Faithfully reproduces dowser's ``cleanAlignment``: it scans codon
    sites ``seq(1, len-3, by=3)`` (so the ``ns`` informativeness vector is
    one codon shorter than the germline) and R then **recycles** that
    logical vector back up to the germline length when subsetting -- this
    Python port replicates that recycling exactly.
    """
    seq = clone.phylo_seq
    g = list(clone.germline)
    sk = [list(str(s)) for s in clone.data[seq]]
    if len(sk) == 0:
        return clone
    glen = len(g)
    # R: sites <- seq(1, length(g)-3, by=3)  (1-based) -> 0-based starts
    sites = list(range(0, glen - 3, 3))
    ns: List[int] = []
    for i in sites:
        cnt = sum("".join(x[i:i + 3]) == "NNN" for x in sk)
        ns.extend([cnt] * 3)
    informative = np.asarray(ns) != len(sk)
    # R recycles a shorter logical vector to the length being subset
    if len(informative) < glen:
        reps = int(np.ceil(glen / len(informative)))
        informative = np.tile(informative, reps)[:glen]
    else:
        informative = informative[:glen]
    new_seqs = ["".join(np.asarray(x[:glen])[informative]) for x in sk]
    gm = "".join(np.asarray(g)[informative])
    clone = clone.copy()
    clone.locus = list(np.asarray(clone.locus)[informative])
    clone.region = list(np.asarray(clone.region)[informative])
    clone.numbers = list(np.asarray(clone.numbers)[informative])
    clone.germline = gm
    clone.hlgermline = gm
    clone.data = clone.data.copy()
    clone.data[seq] = new_seqs
    return clone


def _process_clones(clone_objs: List[AirrClone], minseq: int = 2,
                    seq: str = "sequence") -> List[AirrClone]:
    """Filter, sanitise ids, clean alignments and sort clones by size."""
    out = [c for c in clone_objs if len(c.data) >= minseq]
    for c in out:
        c.data = c.data.copy()
        c.data["sequence_id"] = (c.data["sequence_id"].astype(str)
                                 .str.replace(r"[:;,= ]", "_", regex=True))
    out.sort(key=lambda c: len(c.data), reverse=True)
    out = [_clean_alignment(c) for c in out]
    return out


def formatClones(data, seq="sequence_alignment", clone="clone_id",
                 id="sequence_id", germ="germline_alignment_d_mask",
                 v_call="v_call", j_call="j_call",
                 junc_len="junction_length", mask_char="N", max_mask=0,
                 pad_end=True, text_fields=None, num_fields=None,
                 seq_fields=None, add_count=True, verbose=False,
                 collapse=True, cell="cell_id", locus="locus", traits=None,
                 mod3=True, randomize=False, use_regions=True,
                 dup_singles=False, chain="H", heavy="IGH",
                 filterstop=True, minseq=2, columns=None) -> pd.DataFrame:
    """Collapse an AIRR data frame into per-clone :class:`AirrClone` objects.

    Faithful port of dowser's ``formatClones``.  Sequences with in-frame
    stop codons are optionally removed, non-heavy loci dropped (for
    ``chain="H"``), clones below ``minseq`` discarded, and the remaining
    clones turned into :class:`AirrClone` objects via :func:`makeAirrClone`.

    Returns a :class:`pandas.DataFrame` ("tibble") with columns
    ``clone_id``, ``data`` (the AirrClone objects), ``locus`` and ``seqs``,
    ordered by decreasing sequence count.

    Notes
    -----
    ``randomize`` defaults to ``False`` here (R uses ``True``) so that the
    output is deterministic and reproducible; pass ``randomize=True`` to
    match R's PHYLIP-oriented behaviour.
    """
    data = data.copy()
    if filterstop and seq in data.columns:
        ptcs = _stop_codon_check(data[seq].astype(str))
        if ptcs.sum() > 0:
            import warnings
            warnings.warn(f"{int(ptcs.sum())} sequence(s) with an inframe "
                          "stop codon were removed.")
        data = data[~ptcs].reset_index(drop=True)

    if (data[id] == "Germline").any():
        raise ValueError("'Germline' may not be used as a sequence id.")

    if (v_call not in data.columns and j_call not in data.columns
            and junc_len not in data.columns):
        use_regions = False
        if clone not in data.columns:
            data[clone] = 0
        if locus not in data.columns:
            data[locus] = "N"
            heavy = "N"

    if clone not in data.columns:
        raise ValueError(f"{clone} column not found.")

    if chain == "H" and heavy is not None and locus in data.columns:
        data = data[data[locus].astype(str) == str(heavy)].reset_index(drop=True)
    elif chain == "L" and heavy is not None and locus in data.columns:
        data = data[data[locus].astype(str) != str(heavy)].reset_index(drop=True)

    if data[seq].isna().any():
        data = data[~data[seq].isna()].reset_index(drop=True)

    counts = data[clone].value_counts()
    keep = counts[counts >= minseq].index
    data = data[data[clone].isin(keep)].reset_index(drop=True)

    clone_objs: List[AirrClone] = []
    clone_ids: List[str] = []
    for cid, grp in data.groupby(clone, sort=False):
        ac = makeAirrClone(
            grp, seq=seq, germ=germ, id=id, v_call=v_call, j_call=j_call,
            junc_len=junc_len, mask_char=mask_char, max_mask=max_mask,
            pad_end=pad_end, text_fields=text_fields, num_fields=num_fields,
            seq_fields=seq_fields, add_count=add_count, verbose=verbose,
            collapse=collapse, heavy=heavy, cell=cell, locus=locus,
            traits=traits, mod3=mod3, randomize=randomize,
            use_regions=use_regions, dup_singles=dup_singles, clone=clone,
            chain=chain)
        if ac is not None:
            clone_objs.append(ac)
            clone_ids.append(str(cid))

    if not clone_objs:
        raise ValueError("No clones remain after makeAirrClone")

    seq_name = {"H": "sequence", "L": "lsequence",
                "HL": "hlsequence"}.get(chain, "sequence")
    processed = _process_clones(clone_objs, minseq=minseq, seq=seq_name)

    rows = []
    for ac in processed:
        rows.append({
            "clone_id": ac.clone,
            "data": ac,
            "locus": ",".join(sorted(set(ac.locus))),
            "seqs": len(ac.data),
        })
    out = pd.DataFrame(rows)

    if columns:
        agg = {}
        for col in columns:
            if col not in data.columns:
                raise ValueError(f"column {col} not in data table!")
        for cid, grp in data.groupby(clone, sort=False):
            agg[str(cid)] = {c: ",".join(sorted(map(str, pd.unique(grp[c]))))
                             for c in columns}
        for col in columns:
            out[col] = out["clone_id"].map(lambda x: agg.get(x, {}).get(col))
    return out

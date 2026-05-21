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

__all__ = ["AirrClone", "makeAirrClone", "formatClones", "maskCodons",
           "maskSequences", "resolveLightChains", "getSubclones",
           "sampleClones"]

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


# ----------------------------------------------------------------------
def _str_locate_all(s: str, sub: str):
    """Return all (start, end) 1-based positions of ``sub`` in ``s``."""
    out = []
    i = s.find(sub)
    while i != -1:
        out.append((i + 1, i + len(sub)))
        i = s.find(sub, i + 1)
    return out


def _global_align(query: str, subject: str, gap_opening: float = 5,
                   gap_extension: float = 1):
    """Global (Needleman-Wunsch) alignment; return (aligned_q, aligned_s)."""
    from Bio import Align
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 1
    aligner.mismatch_score = 0
    aligner.open_gap_score = -float(gap_opening)
    aligner.extend_gap_score = -float(gap_extension)
    aln = aligner.align(query, subject)[0]
    return str(aln[0]), str(aln[1])


def maskCodons(id, q: str, s: str, keep_alignment: bool = False,
               gap_opening: float = 5, gap_extension: float = 1,
               keep_insertions: bool = False, mask: bool = True) -> dict:
    """Mask codons in ``s`` that are split by insertions relative to ``q``.

    Faithful port of dowser's ``maskCodons``.  ``q`` is the query (raw)
    V-segment sequence and ``s`` is the IMGT-gapped subject; codons of the
    subject that are split by an insertion in the query are masked with
    ``N``.  Returns a dict with ``sequence_masked_v``, ``masking_note``
    and (optionally) alignment / insertion fields.
    """
    results = {"sequence_id": id, "sequence_masked": "", "masking_note": "",
               "insertions": "", "subject_alignment": "",
               "query_alignment": ""}

    sg = s.replace("...", "")
    if sg == q or not mask:
        results["subject_alignment"] = sg
        results["query_alignment"] = q
        results["sequence_masked_v"] = s
        return results

    gaps = _str_locate_all(s, "...")
    sgf = sg.replace("---", "")
    if "-" in sgf:  # frameshift gaps present
        results["sequence_masked_v"] = None
        results["masking_note"] = "Frameshift in sequence"
        return results
    sg_marked = sg.replace("---", "XXX")

    qa, sa = _global_align(q, sg_marked, gap_opening, gap_extension)
    if keep_alignment:
        results["subject_alignment"] = sa
        results["query_alignment"] = qa

    if keep_insertions:
        ins = _str_locate_all(sa.replace("-", "\0"), "")  # placeholder
        indels = ""
        runs = []
        i = 0
        while i < len(sa):
            if sa[i] == "-":
                j = i
                while j < len(sa) and sa[j] == "-":
                    j += 1
                runs.append((i + 1, j))
                i = j
            else:
                i += 1
        for k, (st, en) in enumerate(runs):
            piece = qa[st - 1:en]
            indels = (f"{st}-{piece}" if k == 0
                      else indels + f",{st}-{piece}")
        results["insertions"] = indels

    if sg_marked.startswith(".."):
        sa = ".." + sa
        if keep_alignment:
            results["subject_alignment"] = sa
    elif sg_marked.startswith("."):
        sa = "." + sa
        if keep_alignment:
            results["subject_alignment"] = sa

    if len(sa) < len(sg_marked) or len(qa) < len(q):
        results["sequence_masked_v"] = None
        results["masking_note"] = "Alignment error"
        return results
    if "-" in sa.replace("---", ""):
        results["sequence_masked_v"] = None
        results["masking_note"] = "Frameshift after alignment"
        return results

    if qa != sa and "-" in sa:
        sas = list(sa)
        mask_flags = []
        nseq = []
        for j in range(int(np.ceil(len(sas) / 3))):
            index = j * 3
            triple = "".join(sas[index:index + 3])
            m = 0
            if "-" in triple:
                triple = "".join("N" if c.isalpha() else c for c in triple)
                triple = triple.replace("-", "")
                if len(triple) != 0:
                    m = 1
            mask_flags.append(m)
            nseq.append(triple)
        maskseq = "".join(nseq)
        if len(maskseq) != len(sg):
            raise ValueError("Sequence masking failed")
        sequence_alignment = maskseq
        for (gs, ge) in gaps:
            sequence_alignment = (sequence_alignment[:gs - 1] + "..."
                                  + sequence_alignment[gs - 1:])
        sequence_alignment = sequence_alignment.replace("X", "-")
        results["sequence_masked_v"] = sequence_alignment
        masked_positions = [i + 1 for i, m in enumerate(mask_flags) if m == 1]
        if masked_positions:
            results["masking_note"] = ",".join(map(str, masked_positions))
        return results
    else:
        results["sequence_masked_v"] = s
        return results


def maskSequences(data: pd.DataFrame, sequence_id: str = "sequence_id",
                  sequence: str = "sequence",
                  sequence_alignment: str = "sequence_alignment",
                  v_sequence_start: str = "v_sequence_start",
                  v_sequence_end: str = "v_sequence_end",
                  v_germline_start: str = "v_germline_start",
                  v_germline_end: str = "v_germline_end",
                  junction_length: str = "junction_length",
                  keep_alignment: bool = False, keep_insertions: bool = False,
                  mask_codons: bool = True,
                  mask_cdr3: bool = True) -> pd.DataFrame:
    """Mask codons split by insertions in the V gene (dowser ``maskSequences``).

    Faithful port of dowser's ``maskSequences``: per-sequence
    :func:`maskCodons` is run on the V segment, the masked V is rejoined
    with the rest of ``sequence_alignment``, and (with ``mask_cdr3=True``)
    the CDR3 region is masked with ``N``.  Adds a ``sequence_masked``
    column.
    """
    data = data.copy().reset_index(drop=True)
    ids = data[sequence_id].astype(str).tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("Sequence IDs are not unique")

    qi = [str(seq)[int(a) - 1:int(b)] for seq, a, b in
          zip(data[sequence], data[v_sequence_start], data[v_sequence_end])]
    si = [str(seq)[int(a) - 1:int(b)] for seq, a, b in
          zip(data[sequence_alignment], data[v_germline_start],
              data[v_germline_end])]
    ei = [str(seq)[int(b):] for seq, b in
          zip(data[sequence_alignment], data[v_germline_end])]

    masked = []
    notes = []
    for x in range(len(qi)):
        m = maskCodons(ids[x], qi[x], si[x],
                       keep_alignment=keep_alignment,
                       keep_insertions=keep_insertions, mask=mask_codons)
        if m["sequence_masked_v"] is None:
            masked.append(None)
        else:
            masked.append(m["sequence_masked_v"] + ei[x])
        notes.append(m["masking_note"])
    data["sequence_masked"] = masked
    data["masking_note"] = notes

    if mask_cdr3:
        out = []
        for x in range(len(data)):
            sm = data["sequence_masked"].iloc[x]
            if sm is None:
                out.append(None)
                continue
            regions = _set_regions(int(data[junction_length].iloc[x]), sm)
            if regions is not None and "cdr3" in regions:
                chars = list(sm)
                for i, r in enumerate(regions):
                    if i < len(chars) and r == "cdr3":
                        chars[i] = "N"
                out.append("".join(chars))
            else:
                out.append(sm)
        data["sequence_masked"] = out
    return data


# ----------------------------------------------------------------------
def resolveLightChains(data: pd.DataFrame, minseq: int = 1,
                       locus: str = "locus", heavy: str = "IGH",
                       id: str = "sequence_id",
                       seq: str = "sequence_alignment",
                       clone: str = "clone_id", cell: str = "cell_id",
                       v_call: str = "v_call", j_call: str = "j_call",
                       junc_len: str = "junction_length",
                       nolight: str = "missing") -> pd.DataFrame:
    """Resolve light-chain V/J subgroups within clones (dowser ``resolveLightChains``).

    Faithful port of dowser's ``resolveLightChains``.  Within each clone,
    paired light chains are partitioned into subgroups by their (gene-level)
    V/J/junction-length rearrangement; a ``clone_subgroup`` column (and the
    derived ``clone_subgroup_id`` / ``vj_gene`` / ``vj_cell``) is added so
    that each subgroup has at most one light chain per cell.

    Cells missing a light chain are grouped with the heavy chain of the
    closest-matching subgroup (Hamming distance, breaking ties by largest
    then lowest-indexed subgroup).
    """
    subgroup = "clone_subgroup"
    data = data.copy()
    light = data[data[locus].astype(str) != str(heavy)].copy()
    heavy_df = data[data[locus].astype(str) == str(heavy)].copy()
    if len(heavy_df) == 0:
        raise ValueError("No heavy chains found in data")
    if len(light) == 0:
        warnings.warn("No light chains found in data! Assigning all "
                      "sequences to subgroup 1.")
        data[subgroup] = 1
        return data

    scount = heavy_df[clone].value_counts()
    big = scount[scount >= minseq].index
    heavy_df = heavy_df[heavy_df[clone].isin(big)]

    if heavy_df[id].duplicated().any() or light[id].duplicated().any():
        raise ValueError("Sequence IDs must be unique!")
    if heavy_df[cell].value_counts().max() > 1:
        raise ValueError("Cells with multiple heavy chains found.")
    if light[cell].isna().any():
        warnings.warn(f"removing {int(light[cell].isna().sum())} light "
                      "chains with missing cell IDs.")
        light = light[~light[cell].isna()]

    paired = []
    for cloneid in pd.unique(heavy_df[clone]):
        hd = heavy_df[heavy_df[clone] == cloneid].copy()
        ld = light[light[cell].isin(hd[cell])].copy()
        ld = ld[~ld[cell].isna()]
        hd[subgroup] = 1
        hd["vj_gene"] = nolight
        if len(ld) == 0:
            hd["clone_subgroup_id"] = (hd[clone].astype(str) + "_"
                                       + hd[subgroup].astype(str))
            hd["vj_cell"] = hd["vj_gene"]
            paired.append(hd)
            continue
        # gene-level V/J for partitioning
        ld = ld.copy()
        try:
            ld["_v"] = getGene(ld[v_call].tolist())
            ld["_j"] = getGene(ld[j_call].tolist())
        except Exception:  # pragma: no cover
            ld["_v"] = ld[v_call].astype(str).str.split("*").str[0]
            ld["_j"] = ld[j_call].astype(str).str.split("*").str[0]
        ld["_combo"] = (ld["_v"].astype(str) + ":" + ld["_j"].astype(str)
                        + ";" + ld[junc_len].astype(str))
        ld[subgroup] = 0
        ld["vj_gene"] = nolight

        remaining = ld.copy()
        lclone = 1
        assigned = []
        while len(remaining) > 0:
            # most common combo across cells
            cell_combos = remaining.groupby(cell)["_combo"].apply(
                lambda x: set(x))
            counts: dict = {}
            for combos in cell_combos:
                for c in combos:
                    counts[c] = counts.get(c, 0) + 1
            top = max(counts, key=counts.get)
            sel = remaining["_combo"] == top
            chunk = remaining[sel].copy()
            chunk[subgroup] = lclone
            chunk["vj_gene"] = top
            # one rearrangement per cell -- keep most ACGT-rich
            keep_ids = []
            for cname, crows in chunk.groupby(cell):
                if len(crows) == 1:
                    keep_ids.append(crows[id].iloc[0])
                else:
                    counts_acgt = crows[seq].astype(str).apply(
                        lambda s: sum(c in "ACGT" for c in s))
                    keep_ids.append(crows[id].iloc[int(counts_acgt.values
                                                       .argmax())])
            chunk = chunk[chunk[id].isin(keep_ids)]
            assigned.append(chunk)
            sel_cells = set(chunk[cell])
            remaining = remaining[~remaining[cell].isin(sel_cells)]
            lclone += 1
        ld_assigned = pd.concat(assigned, ignore_index=True)
        ld_assigned[clone] = cloneid

        # map subgroup onto paired heavy chains
        hd = hd.copy()
        sub_map = dict(zip(ld_assigned[cell], ld_assigned[subgroup]))
        vj_map = dict(zip(ld_assigned[cell], ld_assigned["vj_gene"]))
        hd[subgroup] = hd[cell].map(sub_map).fillna(1).astype(int)
        hd["vj_gene"] = hd[cell].map(vj_map).fillna(nolight)

        comb = pd.concat([hd, ld_assigned], ignore_index=True, sort=False)
        comb[subgroup] = comb[subgroup].astype(int)
        comb["clone_subgroup_id"] = (comb[clone].astype(str) + "_"
                                     + comb[subgroup].astype(str))
        comb["vj_cell"] = comb["vj_gene"]
        # renumber subgroups by descending size
        size = comb[subgroup].value_counts()
        if not (np.diff(size.sort_index().values) <= 0).all():
            order = size.sort_values(ascending=False)
            remap = {old: i + 1 for i, old in enumerate(order.index)}
            comb[subgroup] = comb[subgroup].map(remap)
        paired.append(comb)
    out = pd.concat(paired, ignore_index=True, sort=False)
    out["vj_gene"] = out["vj_gene"].astype(str).str.replace(
        r"\;.*", "", regex=True)
    drop = [c for c in ("_v", "_j", "_combo") if c in out.columns]
    return out.drop(columns=drop)


def getSubclones(heavy=None, light=None, **kwargs):
    """Deprecated -- use :func:`resolveLightChains` (dowser ``getSubclones``).

    dowser deprecated ``getSubclones``; this port raises the same error.
    """
    raise NotImplementedError(
        "getSubclones has been deprecated. Please use resolveLightChains.")


# ----------------------------------------------------------------------
def _sample_clone(clone: AirrClone, size: int, weight: Optional[str],
                  group, rng) -> AirrClone:
    """Down-sample a single clone (dowser ``sampleCloneMultiGroup``)."""
    cdata = clone.data
    if size >= len(cdata):
        return clone
    if group is None:
        ids = cdata["sequence_id"].astype(str).to_numpy()
        if weight is None:
            sd = rng.choice(ids, size=size, replace=False)
        else:
            p = cdata[weight].to_numpy(dtype=float)
            p = p / p.sum()
            sd = rng.choice(ids, size=size, replace=False, p=p)
    else:
        groups = group if isinstance(group, (list, tuple)) else [group]
        missing = [g for g in groups if g not in cdata.columns]
        if missing:
            raise ValueError("One or more grouping columns not found in "
                             "clone data")
        # group sequences evenly
        group_lists = []
        for _, grp in cdata.groupby(groups, sort=False, dropna=False):
            ids = grp["sequence_id"].astype(str).to_numpy()
            if weight is None:
                group_lists.append(list(rng.permutation(ids)))
            else:
                p = grp[weight].to_numpy(dtype=float)
                p = p / p.sum()
                group_lists.append(list(rng.choice(ids, size=len(ids),
                                                   replace=False, p=p)))
        rng.shuffle(group_lists)
        sd = []
        counters = [0] * len(group_lists)
        gi = 0
        while len(sd) < size:
            if (not group_lists[gi]
                    or counters[gi] >= len(group_lists[gi])):
                gi = (gi + 1) % len(group_lists)
                if all(counters[k] >= len(group_lists[k])
                       for k in range(len(group_lists))):
                    break
                continue
            sd.append(group_lists[gi][counters[gi]])
            counters[gi] += 1
            gi = (gi + 1) % len(group_lists)
    new = clone.copy()
    new.data = cdata[cdata["sequence_id"].astype(str).isin(set(sd))].copy()
    new.data = new.data.reset_index(drop=True)
    return new


def sampleClones(clones: pd.DataFrame, size: int, weight: Optional[str] = None,
                 group=None, seed: int = 1) -> pd.DataFrame:
    """Down-sample every clone to a target size (dowser ``sampleClones``).

    Faithful port of dowser's ``sampleClones`` / ``sampleCloneMultiGroup``.
    Each clone is reduced to ``size`` sequences; ``weight`` gives a
    per-sequence sampling-probability column and ``group`` one or more
    columns to sample evenly across.  ``seed`` makes it reproducible.
    """
    rng = np.random.default_rng(seed)
    out = clones.copy().reset_index(drop=True)
    new_data = [_sample_clone(c, size, weight, group, rng)
                for c in out["data"]]
    out["data"] = new_data
    out["seqs"] = [len(c.data) for c in new_data]
    return out

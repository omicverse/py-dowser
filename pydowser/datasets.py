"""Bundled example datasets (mirrors dowser's ``data()`` objects).

``ExampleAirr`` and ``ExampleDbChangeo`` are the AIRR- and Change-O-format
B-cell repertoire subsets shipped with dowser (Laserson & Vigneault et al.,
*PNAS* 2014), exported verbatim from R.
"""
from __future__ import annotations

import os

import pandas as pd

__all__ = ["load_example_airr", "load_example_changeo", "load_imgt_human_vdj"]

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_example_airr() -> pd.DataFrame:
    """Return dowser's ``ExampleAirr`` data frame (389 sequences, 30 columns)."""
    return pd.read_csv(os.path.join(_DATA_DIR, "ExampleAirr.csv"))


def load_example_changeo() -> pd.DataFrame:
    """Return dowser's ``ExampleDbChangeo`` data frame (Change-O format)."""
    return pd.read_csv(os.path.join(_DATA_DIR, "ExampleDbChangeo.csv"))


def load_imgt_human_vdj():
    """Return the bundled minimal human IMGT VDJ reference database.

    A three-gene reference (``IGHV3-11*05``, ``IGHD3-10*01``,
    ``IGHJ5*02``) shipped with dowser as ``extdata/germlines/imgt`` --
    enough to reconstruct germlines for ``ExampleDbChangeo[0]``.  The
    return value is the nested ``readIMGT`` mapping
    ``database[locus][segment] = {allele: sequence}``.
    """
    from .germlines import readIMGT
    return readIMGT(os.path.join(_DATA_DIR, "imgt_human_vdj"), quiet=True)

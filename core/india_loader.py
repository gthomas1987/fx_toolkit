"""Loader for the India macro CSV used by apps/2_india.py.

Expected file format: a CSV where:
  - First column is a date column (named 'date', 'Date', 'DATES', etc.)
  - One optional row near the top can hold per-column metadata
    (category, description, sign, etc.) in a leading comment-prefixed
    rows that are ignored by pd.read_csv unless you handle them
  - Remaining columns are Bloomberg tickers or arbitrary column names
    matched against `core.india_signals.SIGNALS` to determine which
    is a tracked indicator and which is reference data.

Returns (df, meta_list) where:
  - df: pd.DataFrame with date index, all other columns as numeric
  - meta_list: list of column metadata dicts (may be empty if the
    file has no header metadata)

The companion module `core.india_signals` interprets the columns into
SIGNAL objects with category, description, and sign attached.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_india_data(file_path: str) -> tuple[pd.DataFrame, list[dict]]:
    """Load the India CSV. Returns (df, meta_list).

    For now `meta_list` is always an empty list — column metadata is
    inferred from the column names themselves via `core.india_signals.
    SIGNALS`. If you have a header row with metadata, extend this
    loader to parse it.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"India data file not found: {file_path}")
    df = pd.read_csv(p)
    # Detect date column
    date_col = next(
        (c for c in df.columns
         if c.lower() in ("date", "dates", "datetime", "timestamp")),
        df.columns[0],
    )
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    # Coerce non-string columns to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    meta_list: list[dict] = []
    return df, meta_list

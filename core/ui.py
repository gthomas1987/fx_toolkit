"""Streamlit UI helpers shared across the market-data tabs.

Public API:
    data_dir_input() → str | None
        Sidebar input for the market-data folder path. Returns None
        if the path is empty or doesn't exist (caller should st.stop()).

    lookback_selector(default="3Y", options=None) → (label, days)
        Sidebar dropdown for trailing-window lookback. Returns the
        chosen label and its day count.

    app_header(title, subtitle) → None
        Renders a styled title + subtitle pair at the top of the app.

    format_value(value, decimals=2) → str
        Number formatter with thousands separators and graceful NaN/None
        handling. Returns '—' for missing values.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st


# Standard lookback options shared across tabs. Order matters — first
# entries appear at the top of the dropdown. Days values are calendar
# days (not trading days) for compatibility with pd.Timedelta(days=...).
# "Full" → None signals "no lookback limit" to consumers. The previous
# value (10^9) overflows pd.Timedelta (which is nanosecond-based with
# a ~292-year range), so any downstream code calling
# `pd.Timedelta(days=lookback_days)` blew up on "Full". Consumers must
# check for None and skip the cutoff arithmetic.
_LOOKBACK_OPTIONS: dict[str, int | None] = {
    "1Y":   365,
    "2Y":   730,
    "3Y":   1095,
    "5Y":   1825,
    "Full": None,
}


def data_dir_input(default: str = "") -> str | None:
    """Sidebar text input for the data folder. Returns folder path or
    None if empty / nonexistent.

    Default value resolution order:
      1. st.session_state["data_dir"] (sticky across reruns)
      2. MARKET_DATA_DIR env var
      3. `default` argument passed by the caller
      4. Empty string
    """
    if "data_dir" not in st.session_state:
        st.session_state["data_dir"] = (
            os.environ.get("MARKET_DATA_DIR", "") or default
        )
    folder = st.sidebar.text_input(
        "Market data folder",
        value=st.session_state["data_dir"],
        help=("Path to a folder with `_index.csv` (preferred) or raw "
               "CSV files. Set the MARKET_DATA_DIR env var to skip "
               "typing it each time."),
        key="data_dir_input",
    )
    st.session_state["data_dir"] = folder
    if not folder:
        st.sidebar.info("Enter a data folder to continue.")
        return None
    if not Path(folder).exists():
        st.sidebar.error(f"Folder doesn't exist: {folder}")
        return None
    return folder


def lookback_selector(default: str = "3Y",
                         options: list[str] | None = None,
                         label: str = "Lookback",
                         key: str = "lookback") -> tuple[str, int | None]:
    """Sidebar dropdown for trailing-window lookback. Returns (label, days).

    `options` lets you restrict / reorder the standard options. If
    `default` isn't in `options`, falls back to the first option.

    Returns days as `int` for finite windows, or `None` for "Full"
    (full history; consumers must skip cutoff arithmetic and use
    the entire series).
    """
    opts = options if options is not None else list(_LOOKBACK_OPTIONS.keys())
    if default not in opts:
        default = opts[0]
    choice = st.sidebar.selectbox(
        label, opts, index=opts.index(default), key=key,
    )
    days = _LOOKBACK_OPTIONS.get(choice)
    return choice, days


def app_header(title: str, subtitle: str = "") -> None:
    """Top-of-page title + subtitle. Simple and consistent across tabs."""
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def format_value(value, decimals: int = 2) -> str:
    """Format a numeric value with thousands separators. Returns '—'
    for None / NaN. Strings pass through unchanged."""
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)

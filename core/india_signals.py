"""Signal definitions and z-score helpers for apps/2_india.py.

# IMPORTANT: SIGNAL LIST IS A SKELETON
# ====================================
# The app docstring references "43 macro indicators across 5 categories".
# The SIGNALS list below is a STARTER — populate it with your actual
# Bloomberg tickers + descriptions + signs. The framework around it
# (z-score math, composite aggregation, signed sums) is fully working
# regardless of how many signals you add.
#
# To populate SIGNALS:
#   1. Identify each indicator's Bloomberg ticker as it appears as a
#      column header in india_data.csv (e.g. "INRREP01 Index").
#   2. Decide its CATEGORY (one of CATEGORY_ORDER below).
#   3. Decide its SIGN: +1 means "higher value = more bullish USDINR
#      (bearish INR)"; -1 means the opposite.
#   4. Add a Signal(ticker=..., description=..., category=..., sign=...,
#      note="optional explanation").

Public API:
    SIGNALS, MANUAL_SIGNALS, CATEGORY_ORDER, EXCLUDED_TICKERS_PREFIXES
    is_excluded(col), resolve_signals(meta_list, columns)
    get_series_for_signal(df, columns, sig)
    native_z_score(series, lookback_days) → series
    latest_z_score(series, lookback_days) → float
    build_signed_z_table(df, columns, signals, lookback_days) → DataFrame
    composite_history(df, columns, signals, lookback_days) → DataFrame
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# The five categories the app docstring references. Order is the
# display order in the dashboard (gauges and table).
CATEGORY_ORDER: list[str] = [
    "External Drivers",
    "Domestic Rates",
    "Equity & Flows",
    "Macro Fundamentals",
    "Liquidity",
]


# Bloomberg ticker prefixes to ignore (utility / reference / scratch
# columns that shouldn't appear in the indicator list).
EXCLUDED_TICKERS_PREFIXES: tuple[str, ...] = (
    "_",         # leading underscore (scratch columns)
    "__",        # placeholder columns
    "INDEX_",    # reference index columns
)


@dataclass
class Signal:
    """One macro indicator with all the metadata needed to aggregate
    it into the directional composite.

    Attributes:
        ticker: column name in india_data.csv (e.g. "INRREP01 Index")
        description: human-readable name shown in the UI
        category: one of CATEGORY_ORDER
        sign: +1 (higher → bullish USDINR / bearish INR) or -1 (opposite)
        note: optional explanation tooltip
    """
    ticker: str
    description: str
    category: str
    sign: int          # +1 or -1
    note: str = ""


# ============================================================================
# SIGNAL LIST
# ============================================================================
# Populate with your 43 indicators. Below is a *starter* set of common
# India macro indicators with sensible defaults — fill in / replace
# with your actual tickers.
SIGNALS: list[Signal] = [
    # ---- External Drivers ----
    Signal(ticker="DXY Index", description="DXY (USD index)",
            category="External Drivers", sign=+1,
            note="Stronger USD → bullish USDINR"),
    Signal(ticker="CL1 Comdty", description="Brent crude (front)",
            category="External Drivers", sign=+1,
            note="Higher oil → India current account worse → bullish USDINR"),
    Signal(ticker="VIX Index", description="VIX",
            category="External Drivers", sign=+1,
            note="Risk-off → EM weakness → bullish USDINR"),
    Signal(ticker="USDCNH Curncy", description="USDCNH",
            category="External Drivers", sign=+1,
            note="CNH weakness drags INR weaker"),

    # ---- Domestic Rates ----
    Signal(ticker="INRREPO Index", description="RBI repo rate",
            category="Domestic Rates", sign=-1,
            note="Higher repo → INR support → bearish USDINR"),
    Signal(ticker="GIND10YR Index", description="India 10Y govt yield",
            category="Domestic Rates", sign=-1,
            note="Higher yield → INR demand → bearish USDINR"),

    # ---- Equity & Flows ----
    Signal(ticker="NIFTY Index", description="Nifty 50",
            category="Equity & Flows", sign=-1,
            note="Equity rally → FII inflows → bearish USDINR"),

    # ---- Macro Fundamentals ----
    Signal(ticker="INMSCPI YOY Index", description="India CPI YoY",
            category="Macro Fundamentals", sign=+1,
            note="Higher inflation → INR weakness → bullish USDINR"),

    # ---- Liquidity ----
    Signal(ticker="INRMIBOR Index", description="MIBOR overnight",
            category="Liquidity", sign=-1,
            note="Tighter liquidity → INR support → bearish USDINR"),
]


# Manually-keyed signals — for cases where a column in the CSV doesn't
# have a clean Bloomberg ticker (e.g. a derived series the user
# computed offline). Same Signal schema.
MANUAL_SIGNALS: list[Signal] = []


def is_excluded(col: str) -> bool:
    """Should this column be ignored from the indicator universe?"""
    return any(col.startswith(p) for p in EXCLUDED_TICKERS_PREFIXES)


def resolve_signals(meta_list: list[dict],
                       columns: list[str]) -> list[Signal]:
    """Pick the signals from SIGNALS+MANUAL_SIGNALS whose ticker is
    actually present in `columns`. Drops signals with missing data so
    the dashboard never references a column that doesn't exist."""
    available = set(columns)
    out: list[Signal] = []
    for sig in SIGNALS + MANUAL_SIGNALS:
        if sig.ticker in available:
            out.append(sig)
    return out


def get_series_for_signal(df: pd.DataFrame,
                              columns: list[str],
                              sig: Signal) -> pd.Series:
    """Extract the time series for a signal. Returns an empty Series
    if the column isn't in the frame."""
    if sig.ticker not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[sig.ticker], errors="coerce")


def _trailing_window(s: pd.Series, lookback_days: int) -> pd.Series:
    if s.empty or lookback_days >= 10 ** 8:
        return s
    if isinstance(s.index, pd.DatetimeIndex):
        cutoff = s.index[-1] - pd.Timedelta(days=int(lookback_days))
        return s[s.index >= cutoff]
    return s.iloc[-int(lookback_days):]


def native_z_score(series: pd.Series,
                       lookback_days: int = 730) -> pd.Series:
    """Z-score series at its native frequency, using a trailing-N-day
    window for mean/std. Each output point uses only data available at
    that point — no look-ahead."""
    s = pd.Series(series).dropna()
    if s.empty:
        return s
    if not isinstance(s.index, pd.DatetimeIndex):
        # Fallback: expanding stats
        mean = s.expanding(min_periods=20).mean()
        std = s.expanding(min_periods=20).std()
        return (s - mean) / std

    # Time-based rolling window. Note: rolling with a freq window
    # requires a sorted DatetimeIndex which we have.
    window_str = f"{int(lookback_days)}D"
    mean = s.rolling(window_str, min_periods=20).mean()
    std = s.rolling(window_str, min_periods=20).std()
    z = (s - mean) / std
    return z


def latest_z_score(series: pd.Series, lookback_days: int = 730) -> float:
    """Latest z-score of the series — scalar."""
    z = native_z_score(series, lookback_days)
    z_clean = z.dropna()
    if z_clean.empty:
        return float("nan")
    return float(z_clean.iloc[-1])


def build_signed_z_table(df: pd.DataFrame,
                            columns: list[str],
                            signals: list[Signal],
                            lookback_days: int = 730) -> pd.DataFrame:
    """For each signal, compute latest raw-z, signed-z, last value/date.

    Returned columns:
        Category, Indicator, Sign, Last value, Last update, Raw z, Signed z
    """
    rows = []
    for sig in signals:
        s = get_series_for_signal(df, columns, sig).dropna()
        if s.empty:
            rows.append({
                "Category": sig.category,
                "Indicator": sig.description,
                "Sign": int(sig.sign),
                "Last value": float("nan"),
                "Last update": pd.NaT,
                "Raw z": float("nan"),
                "Signed z": float("nan"),
            })
            continue
        raw_z = latest_z_score(s, lookback_days)
        signed = raw_z * sig.sign if pd.notna(raw_z) else float("nan")
        rows.append({
            "Category": sig.category,
            "Indicator": sig.description,
            "Sign": int(sig.sign),
            "Last value": float(s.iloc[-1]),
            "Last update": s.index[-1],
            "Raw z": raw_z,
            "Signed z": signed,
        })
    return pd.DataFrame(rows)


def composite_history(df: pd.DataFrame,
                         columns: list[str],
                         signals: list[Signal],
                         lookback_days: int = 730) -> pd.DataFrame:
    """Daily history of the composite + per-category sub-composites.

    For each date d:
      - For each signal s, compute signed z-score at date d (using s's
        trailing window ending at d).
      - Group by category, take equal-weighted mean across signals
        within that category → category sub-composite at d.
      - Composite at d = equal-weighted mean of the category sub-composites.

    Returned DataFrame: date index, columns = ['Composite'] + categories.
    """
    if not signals:
        return pd.DataFrame(columns=["Composite"] + CATEGORY_ORDER)

    # Compute a signed-z time series per signal
    per_signal_signed: dict[str, pd.Series] = {}
    for sig in signals:
        s = get_series_for_signal(df, columns, sig).dropna()
        if s.empty:
            continue
        z = native_z_score(s, lookback_days)
        per_signal_signed[id(sig)] = z * sig.sign

    if not per_signal_signed:
        return pd.DataFrame(columns=["Composite"] + CATEGORY_ORDER)

    # Build per-category DataFrame, then aggregate
    union_index = pd.DatetimeIndex([])
    for s in per_signal_signed.values():
        union_index = union_index.union(s.index)

    cat_frames: dict[str, list[pd.Series]] = {c: [] for c in CATEGORY_ORDER}
    for sig in signals:
        if id(sig) in per_signal_signed:
            cat_frames[sig.category].append(
                per_signal_signed[id(sig)].reindex(union_index)
            )

    out = pd.DataFrame(index=union_index, dtype=float)
    for cat in CATEGORY_ORDER:
        if not cat_frames[cat]:
            continue
        cat_df = pd.concat(cat_frames[cat], axis=1)
        out[cat] = cat_df.mean(axis=1, skipna=True)

    # Composite = mean across category sub-composites (equal-weighted)
    available_cats = [c for c in CATEGORY_ORDER if c in out.columns]
    if available_cats:
        out["Composite"] = out[available_cats].mean(axis=1, skipna=True)
    else:
        out["Composite"] = float("nan")

    return out[["Composite"] + available_cats]

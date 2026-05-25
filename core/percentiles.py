"""Percentile-rank helpers for the market-data tabs (apps 1a / 1b / ...).

Public API — all functions accept a `lookback_days` arg controlling
how far back to look when computing percentiles:

  - current_percentile(series, lookback_days)  → scalar 0-100
  - expanding_percentile(series, lookback_days) → Series
        For each point, percentile within trailing-N window. No look-ahead.
  - reference_quantiles(series, lookback_days, levels=(10,25,50,75,90))
        → DataFrame of trailing-window quantile CUTOFF values.
  - historical_percentiles(series, percentiles) → dict
  - percentile_band(series, lower, upper) → tuple
  - quantile_at_levels(series, levels) → Series
  - rolling_percentile(series, window) → Series  (fixed-OBS window)

Conventions:
  - Output is 0-100 (not 0-1).
  - "≤" semantic: the Nth percentile means N% of obs were at-or-below.
  - NaN preserved — empty / all-NaN inputs return NaN/empty so the UI
    renders '—' rather than misleading 0%.
  - `lookback_days=None` or `<= 0` means "use full history".
  - No look-ahead bias in time-series outputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _trailing_window(s: pd.Series, lookback_days: int | None) -> pd.Series:
    """Trailing-`lookback_days` window ending at the last index of s.
    None → full series. Non-datetime index → row-based fallback."""
    if lookback_days is None or lookback_days <= 0 or s.empty:
        return s
    if isinstance(s.index, pd.DatetimeIndex):
        cutoff = s.index[-1] - pd.Timedelta(days=int(lookback_days))
        return s[s.index >= cutoff]
    return s.iloc[-int(lookback_days):]


def current_percentile(series: pd.Series,
                          lookback_days: int | None = None,
                          method: str = "rank") -> float:
    """Percentile rank (0-100) of the LAST valid value of `series`
    within its trailing-`lookback_days` window.

    NaN if window empty / all-NaN. `method='midrank'` averages strict
    and at-or-below ranks for tie robustness."""
    s = pd.Series(series)
    window = _trailing_window(s, lookback_days).dropna()
    full_clean = s.dropna()
    if window.empty or full_clean.empty:
        return float("nan")
    last = float(full_clean.iloc[-1])
    n = len(window)
    if method == "midrank":
        below = (window < last).sum()
        at_or_below = (window <= last).sum()
        return float(100.0 * (below + at_or_below) / (2 * n))
    return float(100.0 * (window <= last).sum() / n)


def expanding_percentile(series: pd.Series,
                            lookback_days: int | None = None,
                            min_periods: int = 20,
                            method: str = "rank") -> pd.Series:
    """For each point t, percentile rank of value-at-t within the
    trailing-`lookback_days` window ending at t. No look-ahead.

    If `lookback_days` is None, behaves as a true expanding rank
    (all data ≤ t). Early rows with < `min_periods` obs return NaN.
    """
    s = pd.Series(series)
    n = len(s)
    out = pd.Series(index=s.index, dtype=float)
    if n == 0 or not isinstance(s.index, pd.DatetimeIndex):
        return out

    values = s.values
    dates = s.index

    for i in range(n):
        v = values[i]
        if pd.isna(v):
            continue
        if lookback_days is None or lookback_days <= 0:
            window_data = pd.Series(values[: i + 1]).dropna().values
        else:
            cutoff = dates[i] - pd.Timedelta(days=int(lookback_days))
            mask = (dates[: i + 1] >= cutoff)
            window_data = pd.Series(values[: i + 1][mask]).dropna().values
        if len(window_data) < min_periods:
            continue
        if method == "midrank":
            below = (window_data < v).sum()
            at_or_below = (window_data <= v).sum()
            out.iloc[i] = 100.0 * (below + at_or_below) / (2 * len(window_data))
        else:
            out.iloc[i] = 100.0 * (window_data <= v).sum() / len(window_data)
    return out


def rolling_percentile(series: pd.Series,
                          window: int = 252,
                          min_periods: int | None = None,
                          method: str = "rank") -> pd.Series:
    """Rolling-window percentile rank. Uses last-N OBSERVATIONS not
    last-N DAYS. Useful for irregular date spacing."""
    s = pd.Series(series)
    if min_periods is None:
        min_periods = max(2, window // 2)

    def _rank_last(x: np.ndarray) -> float:
        x_valid = x[~np.isnan(x)]
        if len(x_valid) < min_periods:
            return float("nan")
        last = x_valid[-1]
        if method == "midrank":
            below = (x_valid < last).sum()
            at_or_below = (x_valid <= last).sum()
            return float(100.0 * (below + at_or_below) / (2 * len(x_valid)))
        return float(100.0 * (x_valid <= last).sum() / len(x_valid))

    return s.rolling(window=window, min_periods=min_periods).apply(
        _rank_last, raw=True
    )


def historical_percentiles(series: pd.Series,
                              percentiles: tuple[float, ...] = (
                                  10, 25, 50, 75, 90
                              )) -> dict[float, float]:
    """Static cutoff values. {25: 0.062, 50: 0.075, ...} etc."""
    s = pd.Series(series).dropna()
    if s.empty:
        return {}
    return {float(p): float(np.percentile(s.values, p)) for p in percentiles}


def percentile_band(series: pd.Series,
                       lower: float = 25,
                       upper: float = 75) -> "tuple[float, float]":
    """(lower_pct_value, upper_pct_value) tuple."""
    s = pd.Series(series).dropna()
    if s.empty:
        return (float("nan"), float("nan"))
    return (float(np.percentile(s.values, lower)),
            float(np.percentile(s.values, upper)))


def reference_quantiles(series: pd.Series,
                           lookback_days: int | None = None,
                           levels: "tuple[float, ...]" = (
                               10, 25, 50, 75, 90
                           ),
                           min_periods: int = 20) -> pd.DataFrame:
    """Time-series of quantile cutoff values for drawing reference
    bands. DataFrame with columns 'p10', 'p25', ..., each value at
    row t is the percentile cutoff of the trailing-`lookback_days`
    window ending at t. No look-ahead."""
    s = pd.Series(series)
    out = pd.DataFrame(index=s.index,
                          columns=[f"p{int(L)}" for L in levels],
                          dtype=float)
    n = len(s)
    if n == 0 or not isinstance(s.index, pd.DatetimeIndex):
        return out

    values = s.values
    dates = s.index

    for i in range(n):
        if pd.isna(values[i]):
            continue
        if lookback_days is None or lookback_days <= 0:
            window_data = pd.Series(values[: i + 1]).dropna().values
        else:
            cutoff = dates[i] - pd.Timedelta(days=int(lookback_days))
            mask = (dates[: i + 1] >= cutoff)
            window_data = pd.Series(values[: i + 1][mask]).dropna().values
        if len(window_data) < min_periods:
            continue
        for L in levels:
            out.iloc[i, out.columns.get_loc(f"p{int(L)}")] = float(
                np.percentile(window_data, L)
            )
    return out


def quantile_at_levels(series: pd.Series,
                          levels: "tuple[float, ...]" = (
                              10, 25, 50, 75, 90
                          )) -> "pd.Series":
    """Static cutoffs as a Series indexed by 'pNN'."""
    s = pd.Series(series).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    return pd.Series(
        {f"p{int(L)}": float(np.percentile(s.values, L)) for L in levels}
    )


def extremity_distance(p: "float | None") -> float:
    """How extreme is a percentile? Distance from the median, in pct.

    Returns NaN for NaN input; otherwise `|p - 50|`. Values close to
    50 → near-median (boring); values close to 50 (the maximum) →
    sitting at p=0 or p=100. Sort descending to bubble the most
    extreme metrics to the top of an alerts table.
    """
    if p is None or (isinstance(p, float) and pd.isna(p)):
        return float("nan")
    return abs(float(p) - 50.0)

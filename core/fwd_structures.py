"""Forward-points calendar structures: spreads and butterflies.

Used by apps/1c_fwd_spreads.py and apps/1d_fwd_butterflies.py.

A *spread* is (long_tenor_pts − short_tenor_pts), the term-structure
slope between two points. Positive = forward curve is upward-sloping
between those tenors.

A *butterfly* is (2 × body − wing_short − wing_long), measuring the
curvature of the forward curve at the body tenor. Positive = body is
"rich" (humped curve); negative = body is "cheap" (V-shape).

Both structures are built directly from the fwd-points panel dict
(tenor → Series), date-aligned via outer join + fillna(None).
"""
from __future__ import annotations

import pandas as pd


# (label, long_tenor, short_tenor) — long minus short
# in pip units.
SPREAD_SPECS: list[tuple[str, str, str]] = [
    ("1W-1M", "1M", "1W"),
    ("1M-3M", "3M", "1M"),
    ("3M-6M", "6M", "3M"),
    ("3M-1Y", "1Y", "3M"),
]


# (label, short_tenor, body_tenor, long_tenor)
# Butterfly = 2 × body − short − long.
BUTTERFLY_SPECS: list[tuple[str, str, str, str]] = [
    ("1W-1M-3M", "1W", "1M", "3M"),
    ("1M-3M-6M", "1M", "3M", "6M"),
    ("3M-6M-1Y", "3M", "6M", "1Y"),
]


def _aligned(*series: pd.Series) -> tuple[pd.Series, ...]:
    """Align multiple series on the union of their indices. Returns
    tuples of forward-filled series so a missing day in one input
    doesn't blank out the entire output row."""
    if not series:
        return ()
    idx = series[0].index
    for s in series[1:]:
        idx = idx.union(s.index)
    return tuple(s.reindex(idx).ffill() for s in series)


def compute_spread(fwd_panels: dict[str, pd.Series],
                       long_tenor: str,
                       short_tenor: str) -> pd.Series:
    """Single spread series. Returns empty Series if either tenor
    missing from `fwd_panels`."""
    if long_tenor not in fwd_panels or short_tenor not in fwd_panels:
        return pd.Series(dtype=float)
    long_s, short_s = _aligned(fwd_panels[long_tenor],
                                  fwd_panels[short_tenor])
    spread = (long_s - short_s).dropna()
    return spread


def all_spreads(fwd_panels: dict[str, pd.Series]
                  ) -> dict[str, pd.Series]:
    """Build all spreads from SPREAD_SPECS for which both legs are
    present. Returns `{label: Series}`."""
    out: dict[str, pd.Series] = {}
    for label, long_t, short_t in SPREAD_SPECS:
        s = compute_spread(fwd_panels, long_t, short_t)
        if not s.empty:
            out[label] = s
    return out


def compute_butterfly(fwd_panels: dict[str, pd.Series],
                          short_tenor: str,
                          body_tenor: str,
                          long_tenor: str) -> pd.Series:
    """Single butterfly series: 2·body − short − long. Returns empty
    Series if any leg missing."""
    if (short_tenor not in fwd_panels or body_tenor not in fwd_panels
            or long_tenor not in fwd_panels):
        return pd.Series(dtype=float)
    short_s, body_s, long_s = _aligned(
        fwd_panels[short_tenor],
        fwd_panels[body_tenor],
        fwd_panels[long_tenor],
    )
    bf = (2 * body_s - short_s - long_s).dropna()
    return bf


def all_butterflies(fwd_panels: dict[str, pd.Series]
                       ) -> dict[str, pd.Series]:
    """Build all butterflies from BUTTERFLY_SPECS for which all three
    legs are present."""
    out: dict[str, pd.Series] = {}
    for label, short_t, body_t, long_t in BUTTERFLY_SPECS:
        s = compute_butterfly(fwd_panels, short_t, body_t, long_t)
        if not s.empty:
            out[label] = s
    return out

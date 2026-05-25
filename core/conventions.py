"""FX market conventions — pip scales, pair structure.

Shared between the ko_pricer/joint-distribution tools (strategies 7-10)
and app_11 (the portfolio risk monitor). Both code paths use
`get_pip_scale(pair)`; app_11 additionally uses the helpers below
(is_xxx_usd, foreign_ccy, quote_decimals).
"""
from __future__ import annotations


# Pip scale per pair. Used by:
#   - The KO/worst-of engines (forward-points → spot units conversion)
#   - app_11 (display formatting, forward-points handling)
# Wider coverage than strictly needed; safe defaults for omitted pairs.
PAIR_PIP_SCALE: dict[str, float] = {
    "EURUSD": 1e-4, "GBPUSD": 1e-4, "AUDUSD": 1e-4, "NZDUSD": 1e-4,
    "USDJPY": 1e-2, "USDCAD": 1e-4, "USDCHF": 1e-4,
    "USDNOK": 1e-4, "USDSEK": 1e-4, "USDMXN": 1e-4,
    "AUDCAD": 1e-4, "AUDNZD": 1e-4, "AUDJPY": 1e-2,
    "EURGBP": 1e-4, "EURJPY": 1e-2,
    "GBPJPY": 1e-2, "EURCHF": 1e-4, "USDZAR": 1e-4, "NZDCAD": 1e-4,
    "USDCNH": 1e-4, "USDSGD": 1e-4, "USDINR": 1e-2, "USDKRW": 1.0,
    "USDIDR": 1.0, "USDPHP": 1e-2, "USDTHB": 1e-2, "USDTWD": 1e-3,
    "USDMYR": 1e-4, "USDCNY": 1e-4, "USDHKD": 1e-4,
}

# Alternate spelling for callers that prefer the Bloomberg-style
# forward-points DIVISOR view of the same information. pip_scale = 1 / div.
# Auto-derived from PAIR_PIP_SCALE so the two views can't drift.
FWD_POINTS_DIVISOR: dict[str, float] = {
    p: (1.0 / s if s > 0 else 1.0) for p, s in PAIR_PIP_SCALE.items()
}


def get_pip_scale(pair: str) -> float:
    """Pip scale for a pair. Default 1e-4 for unknown pairs."""
    return PAIR_PIP_SCALE.get(pair, 1e-4)


def is_xxx_usd(pair: str) -> bool:
    """True for pairs quoted XXX/USD (USD is the second leg)."""
    return len(pair) == 6 and pair[3:] == "USD"


def foreign_ccy(pair: str) -> str:
    """The non-USD leg of the pair. For XXXUSD that's the first 3
    chars; for USDXXX that's the last 3."""
    if is_xxx_usd(pair):
        return pair[:3]
    return pair[3:]


def quote_decimals(pair: str) -> int:
    """Number of decimal places to display for the pair's spot."""
    if pair in ("USDJPY", "EURJPY", "GBPJPY", "AUDJPY"):
        return 3
    if pair in ("USDKRW", "USDIDR", "USDINR", "USDTHB", "USDPHP"):
        return 2
    return 4


# =============================================================================
# Tenor / vol-category conventions
# =============================================================================
# Canonical tenor lists for the data tabs (apps 1a / 1b / etc.). Kept here
# rather than scattered across the app files so they stay consistent
# between the vol-skew, forward-points, and any future market-data views.
#
# Note: the strategy backtest engines (apps 9 / 10) use their own
# TENOR_LIST in apps/9_ko_pricer.py (a smaller subset suited to the
# 1M-3M strategy window). These lists are the FULL set commonly seen
# on Bloomberg vol/forward panels.

# Vol-surface tenors — what FXO desks quote ATM/RR/BF at. Order is
# chronological so it can be used directly for x-axis ticks. Matches
# the apps/1_vol_skew.py landing-page grid (6-up: ON, 1W, 1M, 3M, 6M, 1Y).
VOL_TENORS: list[str] = [
    "ON", "1W", "1M", "3M", "6M", "1Y",
]

# Forward-points tenors — typically extend further both shorter
# (overnight, tom-next, spot-next) and longer than vol tenors.
FWD_TENORS: list[str] = [
    "ON", "TN", "SN", "1W", "2W", "1M", "2M", "3M",
    "6M", "9M", "1Y", "2Y",
]

# Map of canonical category code → display label. Used by the snapshot
# / heatmap tables in 1_vol_skew.py. Keys are the category strings as
# they appear in the data index (core/data_loader.py canonicalises raw
# CSV filenames to these codes via the _CATEGORY_ALIASES map).
#
# Both 25Δ and 10Δ entries exist; if a pair has no 10Δ data, the smile
# panels just leave those columns as NaN and the chart shows 3 points.
VOL_CATEGORIES: dict[str, str] = {
    "VOL_ATM":     "ATM vol",
    "VOL_RR_25D":  "25Δ risk reversal",
    "VOL_BF_25D":  "25Δ butterfly",
    "VOL_RR_10D":  "10Δ risk reversal",
    "VOL_BF_10D":  "10Δ butterfly",
}

# Market-convention display order: ATM first (the curve), then 25Δ
# wings, then 10Δ wings. Same content as VOL_CATEGORIES.keys() but
# explicitly ordered.
VOL_CATEGORY_ORDER: list[str] = [
    "VOL_ATM", "VOL_RR_25D", "VOL_BF_25D", "VOL_RR_10D", "VOL_BF_10D",
]

# Aliases that map ALTERNATE canonical names back to the codes above.
# Lets older code that uses 'VOL_25R' / 'VOL_RR_25' / 'VOL_BF_25' still
# work — canon_category() translates these on lookup. Every value
# must be a key of VOL_CATEGORIES.
_VOL_CATEGORY_ALIASES: dict[str, str] = {
    "VOL_25R":   "VOL_RR_25D",
    "VOL_25B":   "VOL_BF_25D",
    "VOL_10R":   "VOL_RR_10D",
    "VOL_10B":   "VOL_BF_10D",
    "VOL_RR_25": "VOL_RR_25D",
    "VOL_BF_25": "VOL_BF_25D",
    "VOL_RR_10": "VOL_RR_10D",
    "VOL_BF_10": "VOL_BF_10D",
    "RR_25D":    "VOL_RR_25D",
    "BF_25D":    "VOL_BF_25D",
    "RR_10D":    "VOL_RR_10D",
    "BF_10D":    "VOL_BF_10D",
}


def canon_category(category: str) -> str:
    """Normalise a category code so old-style and new-style names map
    to the same canonical value. Returns the input unchanged if it's
    not in the alias table — letting unknown categories pass through
    unchanged so callers can use this safely on arbitrary strings."""
    if category in VOL_CATEGORIES:
        return category
    return _VOL_CATEGORY_ALIASES.get(category, category)

# Tenor → approximate days, used by tenor_sort_key. Covers everything
# in VOL_TENORS and FWD_TENORS. Days-not-years lets us avoid float
# precision issues when sorting near-equal tenors (e.g. 1M vs 6W).
_TENOR_DAYS: dict[str, float] = {
    "ON": 1.0,
    "TN": 2.0,
    "SN": 3.0,
    "1W": 7.0,
    "2W": 14.0,
    "3W": 21.0,
    "1M": 30.0,
    "6W": 42.0,
    "2M": 60.0,
    "10W": 70.0,
    "3M": 91.0,
    "4M": 122.0,
    "6M": 182.0,
    "9M": 273.0,
    "1Y": 365.0,
    "18M": 547.0,
    "2Y": 730.0,
    "3Y": 1095.0,
    "5Y": 1825.0,
}


def tenor_sort_key(tenor: str) -> float:
    """Sortable key for a tenor string. Handles common FX tenors plus
    unknown 'NW' / 'NM' / 'NY' patterns via a simple parser fallback.

    Used by the market-data tabs (1a, 1b, ...) to put a heterogeneous
    list of tenor labels into chronological order on chart axes.
    Returns +inf for completely unrecognised strings so they sort last
    rather than crash.
    """
    if tenor in _TENOR_DAYS:
        return _TENOR_DAYS[tenor]
    # Fallback parser: 'NW' = N weeks, 'NM' = N months, 'NY' = N years.
    # Case-insensitive, ignores whitespace.
    s = tenor.strip().upper()
    if len(s) >= 2 and s[-1] in ("W", "M", "Y") and s[:-1].isdigit():
        n = float(s[:-1])
        unit = s[-1]
        if unit == "W":
            return n * 7.0
        if unit == "M":
            return n * 30.0
        if unit == "Y":
            return n * 365.0
    return float("inf")

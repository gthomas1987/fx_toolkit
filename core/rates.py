"""Rate loading and tenor interpolation for FX option pricing.

USD rates: SOFR OIS series (SOFRRATE Index for ON, USOSFR<X> Curncy for term).
JPY rates: TONA OIS series (MUTKCALM Index for ON, JYSO<X> Curncy for term).

Linear interpolation in T (years) between bracketing standard tenors.

# Rate magnitude convention
All rates are stored in DECIMAL (e.g. 0.05 for 5%) after loading. Bloomberg
quote data typically arrives in PERCENT (5.0 for 5%), so we divide by 100 on
load. If your CSV is already in decimal, comment out the division in
`load_rates_panel`.
"""
from __future__ import annotations
from typing import Optional, Callable

import numpy as np
import pandas as pd


# Standard tenor → years (calendar-day approximation, used for the
# interpolation grid only — actual T_v for an option is computed from the
# option-date calendar)
TENOR_YEARS = {
    "ON": 1.0 / 365.0,
    "1W": 7.0 / 365.0,
    "1M": 30.0 / 365.0,
    "2M": 60.0 / 365.0,
    "3M": 91.0 / 365.0,
    "6M": 182.0 / 365.0,
    "9M": 273.0 / 365.0,
    "1Y": 1.0,
}


# Bloomberg ticker → standard tenor mapping per currency
RATE_TICKERS = {
    "USD": {
        "ON":  "SOFRRATE Index",
        "1W":  "USOSFR1Z Curncy",
        "1M":  "USOSFRA Curncy",
        "2M":  "USOSFRB Curncy",
        "3M":  "USOSFRC Curncy",
        "6M":  "USOSFRF Curncy",
        "9M":  "USOSFRI Curncy",
        "1Y":  "USOSFR1 Curncy",
    },
    "JPY": {
        "ON":  "MUTKCALM Index",
        "1W":  "JYSO1Z Curncy",
        "1M":  "JYSOA Curncy",
        "2M":  "JYSOB Curncy",
        "3M":  "JYSOC Curncy",
        "6M":  "JYSOF Curncy",
        "9M":  "JYSOI Curncy",
        "1Y":  "JYSO1 Curncy",
    },
    "EUR": {
        "ON":  "ESTRON Index",
        "1W":  "EESWE1Z Curncy",
        "1M":  "EESWEA Curncy",
        "2M":  "EESWEB Curncy",
        "3M":  "EESWEC Curncy",
        "6M":  "EESWEF Curncy",
        "9M":  "EESWEI Curncy",
        "1Y":  "EESWE1 Curncy",
    },
    "KRW": {
        # KWCDC (3M CD rate) is used flat from ON to 3M per user spec —
        # no liquid short OIS market for KRW, so the 3M CD is the
        # benchmark for the front of the curve.
        "ON":  "KWCDC Curncy",
        "1W":  "KWCDC Curncy",
        "1M":  "KWCDC Curncy",
        "2M":  "KWCDC Curncy",
        "3M":  "KWCDC Curncy",
        "6M":  "KWSWOF Curncy",
        "9M":  "KWSWOI Curncy",
        "1Y":  "KWSWO1 Curncy",
    },
}


def load_rates_panel(folder: str, currency: str,
                       load_by_ticker_fn: Callable[[str, str], pd.Series]
                       ) -> pd.DataFrame:
    """Load all rate tenors for a currency into a wide-format DataFrame.

    Parameters
    ----------
    folder : str
        Market data folder (must contain `_index.csv` with `bbg_ticker` column)
    currency : str
        ISO currency code, e.g. 'USD', 'JPY', 'EUR', 'KRW'
    load_by_ticker_fn : callable
        Function (folder, ticker) -> pd.Series. Provided as a parameter to
        avoid circular imports between rates and data_loader.

    Returns
    -------
    pd.DataFrame with columns = tenor labels (subset of TENOR_YEARS keys
    that were actually found in the folder), index = dates, values = rates
    in DECIMAL.

    Notes
    -----
    Multiple tenors mapping to the same ticker (e.g. KRW where KWCDC is
    used for ON-3M) are loaded once and broadcast across all relevant
    tenor columns. The interpolator handles flat regions correctly.
    """
    if currency not in RATE_TICKERS:
        return pd.DataFrame()

    out = {}
    ticker_cache: dict[str, pd.Series] = {}
    for tenor, ticker in RATE_TICKERS[currency].items():
        if ticker not in ticker_cache:
            ticker_cache[ticker] = load_by_ticker_fn(folder, ticker)
        ser = ticker_cache[ticker]
        if ser is not None and not ser.empty:
            out[tenor] = ser / 100.0       # % -> decimal

    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_index()


def interpolate_rate_at_T(rates_at_date: dict, T_target: float
                            ) -> Optional[float]:
    """Linear interp in T given {tenor_label: rate} for one date.

    Returns None if no rates available. Extrapolates flat at endpoints.
    """
    if not rates_at_date:
        return None
    points = sorted(
        [(TENOR_YEARS[t], r) for t, r in rates_at_date.items()
         if t in TENOR_YEARS and pd.notna(r)],
        key=lambda x: x[0],
    )
    if not points:
        return None
    Ts = np.array([p[0] for p in points])
    rs = np.array([p[1] for p in points])

    if T_target <= Ts[0]:
        return float(rs[0])
    if T_target >= Ts[-1]:
        return float(rs[-1])
    return float(np.interp(T_target, Ts, rs))


def get_rate_at(rates_panel: pd.DataFrame, T_target: float,
                  valuation_date) -> Optional[float]:
    """Look up an interpolated rate at T_target for a specific date.

    Forward-fills missing dates (uses the most recent available row at or
    before the valuation_date).
    """
    if rates_panel.empty:
        return None
    valuation_ts = pd.Timestamp(valuation_date)
    valid = rates_panel.loc[:valuation_ts]
    if valid.empty:
        return None
    row = valid.iloc[-1]
    rates_dict = {t: row[t] for t in row.index if pd.notna(row[t])}
    return interpolate_rate_at_T(rates_dict, T_target)

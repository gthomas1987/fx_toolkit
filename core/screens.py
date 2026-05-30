"""Multi-pair screen helpers for Vol Dashboard.

The existing Vol Dashboard tabs (Skew, Implied Vol, Heatmap, Alerts)
deep-dive a SINGLE pair selected in the sidebar. The 'screen' tabs
added later have a different shape — they SCAN all available USD pairs
at once, matching the Goldman 'Best-Of Screens' layout, and return a
sortable table + scatter + drill-down.

This module collects math primitives shared by those screen tabs.
Phase 1 ships only what the 3m Vol Screen needs; subsequent phases
will add DNT probability, ATMF/ATMS carry, binary strike solver, and
realized-correlation utilities.

Public API
----------
    ASIA_EM_PAIRS         — set of pair codes with on/off-shore split
    realized_vol_annualized(spot, window_days)
    load_pair_panel(folder, pair, category, tenor, prefer_em)
    scan_3m_vol(folder, pairs, lookback_years, prefer_em)
    scan_3m_vol_history(folder, pair, prefer_em)

Conventions
-----------
- Vols in this module are returned in PERCENT (e.g. 8.5 = 8.5%) to
  match the Goldman screens directly. The existing dashboard code
  sometimes uses decimals (0.085) — be mindful of the boundary.
- Realized vol uses √252 annualization on log-returns; no calendar-day
  scaling.
- 'EM preference' defaults to 'offshore' for Asia EM (matches existing
  dashboard default, see `vol_dashboard.py` sidebar block).
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from core.ts_loader import load_panel


# Asia EM pairs that exist in both onshore and offshore variants in
# our `_index.csv`. Anything else is treated as a single-variant pair
# and the `prefer_em` argument is ignored for it.
ASIA_EM_PAIRS = {
    "USDCNH", "USDCNY", "USDINR", "USDIDR", "USDKRW",
    "USDMYR", "USDPHP", "USDTHB", "USDTWD",
}


# Calendar tenor → trading days. Used wherever we need a realized-vol
# window that matches an implied-vol tenor. 21 BDays/month is the
# standard FX-options convention.
TENOR_TRADING_DAYS = {
    "1W": 5, "2W": 10, "1M": 21, "2M": 42, "3M": 63,
    "6M": 126, "9M": 189, "12M": 252, "1Y": 252,
}


# ---------------------------------------------------------------------------
# Realized vol
# ---------------------------------------------------------------------------
def realized_vol_annualized(spot: pd.Series,
                            window_days: int = 63) -> pd.Series:
    """Rolling annualized realized vol from log returns, in PERCENT.

    Parameters
    ----------
    spot : pd.Series
        Spot price series, business-daily, indexed by date.
    window_days : int
        Trading-day window. Default 63 ≈ 3m, matching the Goldman
        '3m Vol Screen'. Use TENOR_TRADING_DAYS for other tenors.

    Returns
    -------
    pd.Series
        Annualized realized vol in percent (8.5 means 8.5%). Aligned to
        the input index; first `window_days` values are NaN.

    Notes
    -----
    - ddof=1 (pandas default — sample stdev). Matches what most FX
      desks publish; if you want to match a specific desk's convention
      you can override later.
    - No clipping or winsorization. If you have known dirty prints in
      the spot series, clean them upstream.
    """
    s = spot.dropna()
    log_ret = np.log(s / s.shift(1))
    rv = log_ret.rolling(window=window_days,
                          min_periods=window_days).std()
    return rv * np.sqrt(252.0) * 100.0


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------
def _prefer_for(pair: str, em_default: str = "offshore") -> str:
    """Return 'offshore'/'onshore' for Asia EM; else just 'offshore'.

    Non-EM pairs ignore the preference inside `load_panel` (there's
    only one variant), so the value is harmless.
    """
    return em_default if pair in ASIA_EM_PAIRS else "offshore"


def load_pair_panel(folder: str,
                    pair: str,
                    category: str,
                    tenor: Optional[str] = None,
                    prefer_em: str = "offshore") -> pd.Series:
    """Load a single pair's series for one (category, tenor).

    Thin wrapper over `load_panel` to:
      1. apply on/offshore preference per pair,
      2. return a NaN-dropped Series rather than a 1-column DataFrame,
      3. return an empty Series (not raise) on missing data.

    Use this inside multi-pair scan loops — keeps the call-sites tidy.
    """
    prefer = _prefer_for(pair, prefer_em)
    df = load_panel(folder, category, tenor=tenor,
                    prefer=prefer, pairs=(pair,))
    if df is None or df.empty or pair not in df.columns:
        return pd.Series(dtype=float)
    return df[pair].dropna()


# ---------------------------------------------------------------------------
# 3m Vol Screen (Goldman image 6)
# ---------------------------------------------------------------------------
def scan_3m_vol(folder: str,
                pairs: Iterable[str],
                lookback_years: int = 2,
                realized_window_days: int = 63,
                prefer_em: str = "offshore") -> pd.DataFrame:
    """Build the Goldman 'Best 3m Vol Screen' table.

    One row per pair with columns:
        Cross, 3m Implied, 3m Realized, Diff,
        2y Low, 2y High, Percentile

    Vols in PERCENT (matches the screen). Sorted ascending by Diff so
    the most-negative (= implied cheap vs realized, candidate to BUY
    vol) is on top.

    Parameters
    ----------
    folder : str
        Data folder (passed through to `load_panel`).
    pairs : Iterable[str]
        Pair codes to scan. Missing data → pair is silently skipped.
    lookback_years : int
        History window for percentile / low / high. Default 2y to
        match the Goldman screen.
    realized_window_days : int
        Trading-day window for the realized-vol estimate. Default 63
        ≈ 3m, to compare like-for-like with 3m implied.
    prefer_em : {'offshore','onshore'}
        Variant preference for Asia EM pairs.
    """
    rows: list[dict] = []
    for pair in pairs:
        atm = load_pair_panel(folder, pair, "VOL_ATM",
                              tenor="3M", prefer_em=prefer_em)
        spot = load_pair_panel(folder, pair, "SPOT",
                               tenor=None, prefer_em=prefer_em)
        if atm.empty or spot.empty:
            continue

        rv = realized_vol_annualized(spot, realized_window_days).dropna()
        if rv.empty:
            continue

        # Align realized onto the implied-vol timeline (forward-fill
        # the last available estimate as of each implied-vol date).
        rv_aligned = rv.reindex(atm.index, method="ffill")

        cur_imp = float(atm.iloc[-1])
        cur_rv_raw = rv_aligned.iloc[-1]
        cur_rv = float(cur_rv_raw) if pd.notna(cur_rv_raw) else np.nan
        diff = cur_imp - cur_rv if pd.notna(cur_rv) else np.nan

        # 2y trailing window for percentile + low/high
        cutoff = atm.index[-1] - pd.Timedelta(days=lookback_years * 365)
        win = atm[atm.index >= cutoff]
        if win.empty:
            continue

        pct = float((win <= cur_imp).sum()) / float(len(win)) * 100.0
        rows.append({
            "Cross": pair,
            "3m Implied": cur_imp,
            "3m Realized": cur_rv,
            "Diff": diff,
            "2y Low": float(win.min()),
            "2y High": float(win.max()),
            "Percentile": pct,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "Cross", "3m Implied", "3m Realized", "Diff",
            "2y Low", "2y High", "Percentile",
        ])
    return (pd.DataFrame(rows)
              .sort_values("Diff", ascending=True, na_position="last")
              .reset_index(drop=True))


def scan_3m_vol_history(folder: str,
                        pair: str,
                        realized_window_days: int = 63,
                        prefer_em: str = "offshore") -> pd.DataFrame:
    """Time-series of 3m Implied and 3m Realized for one pair.

    Returns a DataFrame with two columns ('3m Implied', '3m Realized'),
    indexed by date. Used by the Vol Screen tab's drill-down chart.
    """
    atm = load_pair_panel(folder, pair, "VOL_ATM",
                          tenor="3M", prefer_em=prefer_em)
    spot = load_pair_panel(folder, pair, "SPOT",
                           tenor=None, prefer_em=prefer_em)
    if atm.empty:
        return pd.DataFrame(columns=["3m Implied", "3m Realized"])
    rv = (realized_vol_annualized(spot, realized_window_days)
          if not spot.empty
          else pd.Series(dtype=float))
    df = pd.DataFrame({"3m Implied": atm, "3m Realized": rv})
    # Sort & drop rows where both are NaN
    return df.sort_index().dropna(how="all")


# ===========================================================================
# Phase 2 — Static Carry (ATMF/ATMS put spread)
# ===========================================================================
# Math reference (matches Goldman "Best Carry Screen" image 5):
#
#   The structure: buy 1 ATMF put, sell 1 ATMS put, same tenor.
#   Under "spot rolls to spot" at expiry (i.e. S_T = S₀), the payoff
#   equals (F - S) when F > S, or (S - F) handled as the symmetric
#   call-spread when F < S. Either way the static payoff magnitude
#   is |F - S| and the premium spread magnitude is |Put_ATMF - Put_ATMS|
#   (in forward value; the exp(-r_d·T) discount factor cancels in the
#   ratio so we don't need to load each currency's risk-free rate).
#
#   Black-76 forward values:
#     Put_ATMF (K=F):    F · (2·N(σ√T/2) − 1)
#     Put_ATMS (K=S):    S·N(−d2) − F·N(−d1)
#       where d1 = (ln(F/S) + σ²T/2) / (σ√T),  d2 = d1 − σ√T
#
#   Static-carry ratio = |F − S| / |Put_ATMF − Put_ATMS|
#
#   Direction (which currency you're long):
#     For USDXXX pairs:  Long XXX  if F > S  (XXX high-yielder)
#                        Short XXX otherwise
#     For XXXUSD pairs:  Long XXX  if F < S  (XXX high-yielder)
#                        Short XXX otherwise
#
# Caveat: Goldman's published numbers use the FULL VOL SMILE (not just
# ATM), so expect our ratios to differ by ~0.1–0.4x for the same date.
# The DIRECTION and the RANKING across pairs should still match closely.

import math


# Standard Bloomberg forward-points conventions. Maps each pair to the
# multiplier that converts raw fwd_points → spot units, i.e.
#   F = S + fwd_points * PIP_FACTOR[pair]
# Pairs not in the table fall back to a magnitude heuristic.
# Users with non-BBG data should override this dict (or just edit it).
PIP_FACTOR = {
    "EURUSD": 1e-4, "GBPUSD": 1e-4, "AUDUSD": 1e-4, "NZDUSD": 1e-4,
    "USDCAD": 1e-4, "USDCHF": 1e-4, "USDNOK": 1e-4, "USDSEK": 1e-4,
    "USDJPY": 1e-2,
    "USDCNH": 1e-4, "USDCNY": 1e-4, "USDHKD": 1e-4, "USDSGD": 1e-4,
    "USDINR": 1e-4, "USDMYR": 1e-4, "USDPHP": 1e-4, "USDTHB": 1e-4,
    "USDIDR": 1.0,  "USDKRW": 1e-2, "USDTWD": 1e-4,
}


# Tenor → year-fraction (ACT/365 convention is fine for screening; if you
# need ACT/360 for USD-linked pairs, override at call site).
TENOR_YEAR_FRACTION = {
    "1W":  7/365, "1M":  30/365, "2M":  60/365, "3M":  90/365,
    "6M": 180/365, "9M": 270/365, "12M": 365/365, "1Y": 365/365,
}


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf — no scipy dependency."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def forward_from_points(pair: str, spot: float, fwd_points: float) -> float:
    """Compute F = S + fwd_points * pip_factor.

    Looks up the per-pair pip factor in `PIP_FACTOR`. Pairs not in the
    table use a heuristic based on spot magnitude — works for most
    spot levels but verify against your data conventions.
    """
    pip = PIP_FACTOR.get(pair)
    if pip is None:
        # Heuristic fallback
        if spot < 10:
            pip = 1e-4
        elif spot < 500:
            pip = 1e-2
        else:
            pip = 1.0
    return spot + fwd_points * pip


def black76_atmf_put_fv(F: float, T: float, sigma: float) -> float:
    """Forward (undiscounted) value of an ATMF put.

    Parameters
    ----------
    F : forward
    T : year-fraction
    sigma : implied vol (decimal, e.g. 0.085 for 8.5%)
    """
    return F * (2.0 * _norm_cdf(sigma * math.sqrt(T) / 2.0) - 1.0)


def black76_atms_put_fv(S: float, F: float, T: float, sigma: float) -> float:
    """Forward (undiscounted) value of an ATMS put (K=S)."""
    sig_sqrt_T = sigma * math.sqrt(T)
    d1 = (math.log(F / S) + sigma * sigma * T / 2.0) / sig_sqrt_T
    d2 = d1 - sig_sqrt_T
    return S * _norm_cdf(-d2) - F * _norm_cdf(-d1)


def static_carry(S: float, F: float, T: float, sigma: float,
                 pair: str) -> tuple[float, str]:
    """Compute the ATMF/ATMS static-carry ratio and direction for one pair.

    Returns
    -------
    (ratio, direction) : (float, str)
        ratio    — |F-S| / |Put_ATMF - Put_ATMS|, both in forward value.
                   NaN if inputs are degenerate.
        direction — "Long" if you'd be long the non-USD currency under
                    the implied carry direction; "Short" otherwise.
    """
    if not (S > 0 and F > 0 and T > 0 and sigma > 0):
        return float("nan"), ""
    atmf_fv = black76_atmf_put_fv(F, T, sigma)
    atms_fv = black76_atms_put_fv(S, F, T, sigma)
    premium = atmf_fv - atms_fv
    if abs(premium) < 1e-12:
        return float("nan"), ""
    ratio = abs(F - S) / abs(premium)
    if pair.startswith("USD"):
        direction = "Long" if F > S else "Short"
    elif pair.endswith("USD"):
        direction = "Long" if F < S else "Short"
    else:
        direction = ""
    return ratio, direction


def smoothed_1m_return(spot: pd.Series,
                       pair: str,
                       window_days: int = 21) -> float:
    """1m smoothed spot return via linear regression slope, expressed as
    NON-USD CURRENCY APPRECIATION (positive = the non-USD ccy strengthened
    over the last month — matches the Goldman screen convention).

    Uses OLS on log(spot) vs day index; slope × window = log return ≈
    percent return for small magnitudes. Less noisy than a simple
    point-to-point return.
    """
    s = spot.dropna().iloc[-window_days:]
    if len(s) < max(5, window_days // 2):
        return float("nan")
    x = np.arange(len(s), dtype=float)
    y = np.log(s.values)
    slope, _ = np.polyfit(x, y, 1)
    spot_ret_pct = slope * window_days * 100.0
    # USDXXX: spot up = USD up = XXX down → flip sign for "XXX appreciation"
    if pair.startswith("USD"):
        return -spot_ret_pct
    elif pair.endswith("USD"):
        return spot_ret_pct
    return spot_ret_pct


def scan_static_carry(folder: str,
                      pairs: Iterable[str],
                      tenor: str = "3M",
                      prefer_em: str = "offshore") -> pd.DataFrame:
    """Build the Goldman 'Best Carry Screen' table (image 5).

    Returns columns:
        Pair, Spot, Forward, Implied Vol (%), Static Carry (x),
        Direction, 1m Return (%)

    Sorted descending by Static Carry (highest ratio on top).
    """
    if tenor not in TENOR_YEAR_FRACTION:
        raise ValueError(f"Unknown tenor: {tenor!r}")
    T = TENOR_YEAR_FRACTION[tenor]

    rows: list[dict] = []
    for pair in pairs:
        spot = load_pair_panel(folder, pair, "SPOT",
                                tenor=None, prefer_em=prefer_em)
        fp = load_pair_panel(folder, pair, "FWD_POINTS",
                              tenor=tenor, prefer_em=prefer_em)
        atm = load_pair_panel(folder, pair, "VOL_ATM",
                               tenor=tenor, prefer_em=prefer_em)
        if spot.empty or fp.empty or atm.empty:
            continue

        S = float(spot.iloc[-1])
        FP = float(fp.iloc[-1])
        F = forward_from_points(pair, S, FP)
        sigma_pct = float(atm.iloc[-1])     # in %
        sigma = sigma_pct / 100.0

        ratio, direction = static_carry(S, F, T, sigma, pair)
        ret_1m = smoothed_1m_return(spot, pair, window_days=21)

        rows.append({
            "Pair": pair,
            "Spot": S,
            "Forward": F,
            "Implied Vol": sigma_pct,
            "Static Carry": ratio,
            "Direction": direction,
            "1m Return": ret_1m,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "Pair", "Spot", "Forward", "Implied Vol",
            "Static Carry", "Direction", "1m Return",
        ])
    return (pd.DataFrame(rows)
              .sort_values("Static Carry", ascending=False,
                              na_position="last")
              .reset_index(drop=True))


def scan_static_carry_history(folder: str,
                              pair: str,
                              tenor: str = "3M",
                              prefer_em: str = "offshore") -> pd.DataFrame:
    """Static-carry-ratio time series for one pair.

    Returns DataFrame indexed by date with columns:
        Spot, Forward, Implied Vol, Static Carry

    Useful for the drill-down chart in the Static Carry tab.
    """
    if tenor not in TENOR_YEAR_FRACTION:
        raise ValueError(f"Unknown tenor: {tenor!r}")
    T = TENOR_YEAR_FRACTION[tenor]

    spot = load_pair_panel(folder, pair, "SPOT",
                            tenor=None, prefer_em=prefer_em)
    fp = load_pair_panel(folder, pair, "FWD_POINTS",
                          tenor=tenor, prefer_em=prefer_em)
    atm = load_pair_panel(folder, pair, "VOL_ATM",
                           tenor=tenor, prefer_em=prefer_em)
    if spot.empty or fp.empty or atm.empty:
        return pd.DataFrame()

    idx = spot.index.intersection(fp.index).intersection(atm.index)
    if len(idx) == 0:
        return pd.DataFrame()
    s_a = spot.reindex(idx)
    fp_a = fp.reindex(idx)
    atm_a = atm.reindex(idx)

    # Vectorized would be cleaner, but per-row keeps the math
    # transparent and we're talking O(2000) rows max per pair.
    fwd_vals = np.full(len(idx), np.nan)
    ratio_vals = np.full(len(idx), np.nan)
    for i in range(len(idx)):
        S = s_a.iloc[i]; FP_i = fp_a.iloc[i]; sig = atm_a.iloc[i]
        if pd.isna(S) or pd.isna(FP_i) or pd.isna(sig):
            continue
        F = forward_from_points(pair, S, FP_i)
        r, _ = static_carry(S, F, T, sig / 100.0, pair)
        fwd_vals[i] = F
        ratio_vals[i] = r

    return pd.DataFrame({
        "Spot": s_a,
        "Forward": pd.Series(fwd_vals, index=idx),
        "Implied Vol": atm_a,
        "Static Carry": pd.Series(ratio_vals, index=idx),
    })


# ===========================================================================
# Phase 3a — Double-No-Touch (DNT) screen
# ===========================================================================
# Goldman "Best DNT Screen" (image 3) — most attractive vols to SELL in
# DNT format. Range constructed as symmetric ± half-width around current
# spot, where half-width = 1.20 × max(|S_t - S_now|) over a lookback
# (default 3m, matching the DNT tenor).
#
# We report THEORETICAL Black-Scholes survival probability under
# continuous monitoring. Goldman backs out IMPLIED probability from
# market DNT quotes, which trade at a large premium — expect their
# numbers to be 3-5× higher than ours. The RANKING across pairs should
# still be informative (which DNTs have the largest theoretical-vs-
# implied gap = the richest to sell), but the absolute level isn't
# directly comparable to the GS screen.


def dnt_survival_probability(S: float, L: float, U: float,
                              T: float, sigma: float,
                              mu: float = 0.0,
                              max_terms: int = 50) -> float:
    """Black-Scholes survival probability for a double-no-touch.

    Probability that log-spot stays inside (ln L, ln U) over [0, T]
    under GBM with vol σ and log-drift μ. Uses the standard
    reflection-principle series:

        P = Σ_n [N((b + 2nW)/σ√T - μ√T/σ)
                  − N((a + 2nW)/σ√T - μ√T/σ)]
            − e^{2μa/σ²} · Σ_n [N((b + 2nW − 2a)/σ√T - μ√T/σ)
                                  − N((a + 2nW − 2a)/σ√T - μ√T/σ)]

    where a = ln(L/S), b = ln(U/S), W = b − a. Series converges very
    quickly — 20 terms suffices for any reasonable corridor.

    Parameters
    ----------
    S : current spot
    L, U : lower/upper barriers (L < S < U)
    T : year-fraction to expiry
    sigma : implied vol (decimal, e.g. 0.067 for 6.7%)
    mu : log-spot drift; default 0 (no-drift, fine for FX vol screening)

    Returns
    -------
    float : probability in [0, 1]. Returns 0 if spot is already
            outside the corridor, NaN on degenerate inputs.
    """
    if not (S > 0 and L > 0 and U > L and T > 0 and sigma > 0):
        return float("nan")
    if not (L < S < U):
        return 0.0

    a = math.log(L / S)
    b = math.log(U / S)
    w = b - a
    sst = sigma * math.sqrt(T)
    ms = mu * math.sqrt(T) / sigma if mu != 0 else 0.0

    s1 = s2 = 0.0
    for n in range(-max_terms, max_terms + 1):
        s1 += (_norm_cdf((b + 2 * n * w) / sst - ms)
               - _norm_cdf((a + 2 * n * w) / sst - ms))
        s2 += (_norm_cdf((b + 2 * n * w - 2 * a) / sst - ms)
               - _norm_cdf((a + 2 * n * w - 2 * a) / sst - ms))
    correction = math.exp(2 * mu * a / (sigma * sigma)) if mu != 0 else 1.0
    return max(0.0, min(1.0, s1 - correction * s2))


def construct_dnt_range(spot: pd.Series,
                        lookback_days: int = 63,
                        widen_pct: float = 20.0) -> float:
    """Build the symmetric DNT half-width as a fraction of current spot.

    Goldman convention: half-width = (1 + widen_pct/100) ×
    max(|S_t − S_now|/S_now) over the lookback window. We then place
    barriers symmetrically at S × (1 ± half_width).

    Default widen_pct = 20 (so range is 20% wider than the realized
    historical excursion); default lookback = 63 BDays ≈ 3m to match
    the DNT tenor.
    """
    s = spot.dropna().iloc[-lookback_days:]
    if len(s) < 5:
        return float("nan")
    S_now = float(s.iloc[-1])
    max_dist = float((s - S_now).abs().max()) / S_now
    return max_dist * (1.0 + widen_pct / 100.0)


def scan_dnt(folder: str,
             pairs: Iterable[str],
             tenor: str = "3M",
             lookback_days: int = 63,
             widen_pct: float = 20.0,
             realized_window_days: int = 63,
             prefer_em: str = "offshore") -> pd.DataFrame:
    """Build the 'Best DNT Screen' table.

    Returns one row per pair:
        Cross, Spot, Range%, Lower KO, Upper KO,
        Implied Vol, Realized Vol, Survival Prob (%)

    Sorted ascending by survival probability — lowest prob = most
    attractive DNT to sell (highest premium relative to expected payoff
    under continuous Black-Scholes).
    """
    if tenor not in TENOR_YEAR_FRACTION:
        raise ValueError(f"Unknown tenor: {tenor!r}")
    T = TENOR_YEAR_FRACTION[tenor]

    rows: list[dict] = []
    for pair in pairs:
        spot = load_pair_panel(folder, pair, "SPOT",
                                tenor=None, prefer_em=prefer_em)
        atm = load_pair_panel(folder, pair, "VOL_ATM",
                               tenor=tenor, prefer_em=prefer_em)
        if spot.empty or atm.empty:
            continue

        S = float(spot.iloc[-1])
        sigma_pct = float(atm.iloc[-1])
        sigma = sigma_pct / 100.0
        half_width = construct_dnt_range(spot, lookback_days, widen_pct)
        if not (half_width > 0):
            continue
        L = S * (1.0 - half_width)
        U = S * (1.0 + half_width)

        p = dnt_survival_probability(S, L, U, T, sigma) * 100.0

        rv = realized_vol_annualized(spot, realized_window_days).dropna()
        rv_now = float(rv.iloc[-1]) if not rv.empty else float("nan")

        rows.append({
            "Cross": pair,
            "Spot": S,
            "Range": half_width * 100.0,
            "Lower KO": L,
            "Upper KO": U,
            "Implied Vol": sigma_pct,
            "Realized Vol": rv_now,
            "Survival Prob": p,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "Cross", "Spot", "Range", "Lower KO", "Upper KO",
            "Implied Vol", "Realized Vol", "Survival Prob",
        ])
    return (pd.DataFrame(rows)
              .sort_values("Survival Prob", ascending=True,
                              na_position="last")
              .reset_index(drop=True))


def scan_dnt_history(folder: str,
                     pair: str,
                     tenor: str = "3M",
                     lookback_days: int = 63,
                     widen_pct: float = 20.0,
                     prefer_em: str = "offshore") -> pd.DataFrame:
    """Time series of spot + KO levels for one pair (for drill-down chart).

    Returns DataFrame indexed by date with columns:
        Spot, Lower KO, Upper KO

    KO levels are computed point-in-time using the trailing `lookback_days`
    window — so the bands evolve as the recent spot range changes.
    """
    if tenor not in TENOR_YEAR_FRACTION:
        raise ValueError(f"Unknown tenor: {tenor!r}")

    spot = load_pair_panel(folder, pair, "SPOT",
                            tenor=None, prefer_em=prefer_em)
    if spot.empty:
        return pd.DataFrame()

    s_vals = spot.values
    n = len(spot)
    L_vals = np.full(n, np.nan)
    U_vals = np.full(n, np.nan)
    for i in range(lookback_days, n):
        window = s_vals[i - lookback_days:i + 1]
        S_now = window[-1]
        max_dist = np.abs(window - S_now).max() / S_now
        hw = max_dist * (1.0 + widen_pct / 100.0)
        L_vals[i] = S_now * (1.0 - hw)
        U_vals[i] = S_now * (1.0 + hw)

    return pd.DataFrame({
        "Spot": spot,
        "Lower KO": pd.Series(L_vals, index=spot.index),
        "Upper KO": pd.Series(U_vals, index=spot.index),
    })


# ===========================================================================
# Phase 3b — Binary 10:1 OTMS screen
# ===========================================================================
# Goldman "Binary OTMS" (image 4) — how far OTMS a 3m binary needs to be
# struck for a payout ratio of approximately 10:1. The strike is found
# by inverting the binary-option pricing equation:
#
#   Binary call:  P(S_T > K) = 1/payout_ratio  ⇒
#                 N(d2) = 1/R where d2 = (ln(F/K) − σ²T/2) / (σ√T)
#                 ⇒ K = F · exp(z · σ√T + σ²T/2)   with z = N⁻¹(1 − 1/R)
#
#   Binary put:   P(S_T < K) = 1/payout_ratio  ⇒
#                 K = F · exp(−z · σ√T − σ²T/2)
#
# We use spot (not forward) for simplicity — matches GS's column layout
# of % OTMS vs spot. Forward drift correction is small at 3m for major
# pairs; if you want it, swap S for F using `forward_from_points`.
#
# Normalized column from GS: % OTMS / realized vol. Useful to compare
# binaries across pairs of different vol regimes.
#
# Caveat: GS uses the FULL VOL SMILE; we use ATM vol. For skewed pairs
# (USDCAD, AUDUSD), expect ~1pp discrepancy. Ranking is preserved.


def _norm_inv(p: float) -> float:
    """Inverse standard normal CDF — Beasley-Springer-Moro algorithm.

    Accurate to ~1e-9 over the central region. Sufficient for screening.
    No scipy dependency.
    """
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /                (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /             ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def binary_otms_strike(S: float, T: float, sigma: float,
                       payout_ratio: float = 10.0,
                       is_call: bool = True) -> tuple[float, float]:
    """Solve for the strike of a binary with given payout ratio.

    payout_ratio = 10 means $1 premium → $10 payout if struck, i.e.
    breakeven probability = 1/10 = 10%.

    Returns
    -------
    (K, pct_otms) : (float, float)
        K       — strike level
        pct_otms — (K/S − 1) × 100  (positive for calls, negative for puts)
    """
    if not (S > 0 and T > 0 and sigma > 0 and payout_ratio > 1):
        return float("nan"), float("nan")
    p_breakeven = 1.0 / payout_ratio
    z = _norm_inv(1.0 - p_breakeven)
    sst = sigma * math.sqrt(T)
    drift = 0.5 * sigma * sigma * T
    if is_call:
        ln_K_over_S = z * sst + drift
    else:
        ln_K_over_S = -z * sst - drift
    K = S * math.exp(ln_K_over_S)
    pct_otms = (K / S - 1.0) * 100.0
    return K, pct_otms


def _pair_side_for_usd_call(pair: str) -> str:
    """For a USD-call structure, which DIRECTION moves the pair?

    Returns "call" if USDXXX (pair up when USD strengthens) or
    "put" if XXXUSD (pair down when USD strengthens).
    """
    if pair.startswith("USD"):
        return "call"
    if pair.endswith("USD"):
        return "put"
    return ""


def scan_binary_otms(folder: str,
                      pairs: Iterable[str],
                      tenor: str = "3M",
                      direction: str = "USD_CALL",
                      payout_ratio: float = 10.0,
                      realized_window_days: int = 63,
                      prefer_em: str = "offshore") -> pd.DataFrame:
    """Build the Goldman 'Binary 10:1' screen for one direction.

    Parameters
    ----------
    direction : {'USD_CALL', 'USD_PUT'}
        USD_CALL = USD strengthens vs the other currency.
        USD_PUT  = USD weakens.

    Returns
    -------
    DataFrame with columns:
        Currency, Pair, Spot, Strike, % OTMS, Normalized

    where Normalized = |% OTMS| / realized vol. Sorted ascending by
    |% OTMS| — tightest strike (cheapest binary for the same payout)
    on top.
    """
    if tenor not in TENOR_YEAR_FRACTION:
        raise ValueError(f"Unknown tenor: {tenor!r}")
    T = TENOR_YEAR_FRACTION[tenor]
    if direction not in ("USD_CALL", "USD_PUT"):
        raise ValueError(f"direction must be USD_CALL or USD_PUT, got {direction!r}")

    rows: list[dict] = []
    for pair in pairs:
        side = _pair_side_for_usd_call(pair)
        if not side:
            continue  # skip non-USD pairs (we don't have any in our data)
        if direction == "USD_PUT":
            side = "put" if side == "call" else "call"

        spot = load_pair_panel(folder, pair, "SPOT",
                                tenor=None, prefer_em=prefer_em)
        atm = load_pair_panel(folder, pair, "VOL_ATM",
                               tenor=tenor, prefer_em=prefer_em)
        if spot.empty or atm.empty:
            continue

        S = float(spot.iloc[-1])
        sigma = float(atm.iloc[-1]) / 100.0
        K, pct_otms = binary_otms_strike(S, T, sigma, payout_ratio,
                                            is_call=(side == "call"))

        rv = realized_vol_annualized(spot, realized_window_days).dropna()
        rv_now = float(rv.iloc[-1]) if not rv.empty else float("nan")
        norm = (abs(pct_otms) / rv_now) if (rv_now and rv_now > 0) else float("nan")

        # Display ccy = the non-USD ccy
        if pair.startswith("USD"):
            ccy = pair[3:]
        else:
            ccy = pair[:3]

        rows.append({
            "Currency": ccy,
            "Pair": pair,
            "Spot": S,
            "Strike": K,
            "% OTMS": pct_otms,
            "Normalized": norm,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "Currency", "Pair", "Spot", "Strike", "% OTMS", "Normalized",
        ])
    return (pd.DataFrame(rows)
              .assign(_abs_otms=lambda d: d["% OTMS"].abs())
              .sort_values("_abs_otms", ascending=True, na_position="last")
              .drop(columns=["_abs_otms"])
              .reset_index(drop=True))

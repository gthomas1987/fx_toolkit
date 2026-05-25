"""Correlation estimators for FX pairs.

Two estimators for the joint dynamics of two FX pairs:

  1. `rolling_realized_correlation(spot_a, spot_b, window)`
        Rolling N-day log-return Pearson correlation between two spot
        series. Backward-looking, no vol quotes needed; useful for
        backtests over periods when implied-cross vols aren't available.

  2. `triangulated_correlation(sigma_a, sigma_b, sigma_cross, eta)`
        FORWARD-LOOKING implied correlation via the no-arbitrage
        cross-vol identity. Free in FX since major cross vols are
        already quoted by the market.

# FX triangulation — derivation

For three pairs A, B, X where X is the natural cross of A and B,
no-arbitrage gives a deterministic relationship between log(S_X) and
log(S_A), log(S_B):

    log S_X = ε_A · log S_A  +  ε_B · log S_B

with ε_A, ε_B ∈ {±1} depending on which side of each pair the
shared currency sits.

Taking variances:
    σ²_X = σ²_A + σ²_B + 2 · ε_A · ε_B · ρ_AB · σ_A · σ_B

Solving for ρ:
    ρ_AB = η · (σ²_A + σ²_B − σ²_X) / (2 σ_A σ_B)
where η := −ε_A·ε_B.

# Sign convention (η)

| Shared currency position | ε_A·ε_B | η  |
|--------------------------|---------|----|
| Same side  (both DOM or both FOR) | −1  | +1 |
| Opposite sides                    | +1  | −1 |

# Cross-pair name resolution

Two FX market quoting conventions exist for crosses — e.g., "EURJPY"
vs "JPYEUR". The triangulation formula is symmetric in the cross-vol
(σ_X² is the same either way), so it doesn't matter which orientation
the data provides. `triangulation_eta_and_cross` returns a canonical
name (FOR first), and the loader tries both orderings against the
data folder before giving up.

# Pair format

This module assumes 6-character pair codes with FOR (base) first and
DOM (quote) last — the conventional FX form: `"USDJPY"` means JPY per
USD, FOR=USD, DOM=JPY.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np
import pandas as pd


# =============================================================================
# Rolling realized correlation
# =============================================================================
def rolling_realized_correlation(
        spot_a: pd.Series, spot_b: pd.Series, window: int = 60,
) -> pd.Series:
    """Rolling N-business-day log-return Pearson correlation.

    Returns a pd.Series aligned to the COMMON-dates intersection of the
    two inputs. Values for dates before the rolling window is full are
    NaN — callers must either ignore those or fall back to a default ρ.

    Note: this is BACKWARD-looking and HISTORICAL. For a forward-looking
    estimate (the relevant quantity for option pricing), use
    `triangulated_correlation`.
    """
    df = pd.concat([spot_a, spot_b], axis=1, join="inner").dropna()
    if df.shape[1] != 2:
        return pd.Series(dtype=float)
    df.columns = ["a", "b"]
    log_a = np.log(df["a"]).diff()
    log_b = np.log(df["b"]).diff()
    return log_a.rolling(window).corr(log_b)


def realized_correlation_at(
        spot_a: pd.Series, spot_b: pd.Series, val_date, window: int = 60,
) -> "tuple[Optional[float], int]":
    """Single-value lookup of the rolling correlation at `val_date`.

    Returns (rho, n_obs) where n_obs is the number of overlapping
    dates available up to and including val_date. rho is None if
    fewer than `window`+1 overlapping dates are available.
    """
    df = pd.concat([spot_a, spot_b], axis=1, join="inner").dropna()
    df = df.loc[:pd.Timestamp(val_date)]
    n_obs = len(df)
    if n_obs < window + 1:
        return None, n_obs
    log_a = np.log(df.iloc[:, 0]).diff()
    log_b = np.log(df.iloc[:, 1]).diff()
    corr = log_a.rolling(window).corr(log_b).iloc[-1]
    if pd.isna(corr):
        return None, n_obs
    return float(corr), n_obs


# =============================================================================
# Triangulation: determining the cross pair and η sign
# =============================================================================
def triangulation_eta_and_cross(pair_a: str, pair_b: str
                                  ) -> "Optional[tuple[int, str]]":
    """Determine the FX triangulation cross and sign for two pairs.

    Returns (eta, cross_pair_name) or None when the pairs share no
    currency (triangulation undefined).

    eta convention:
        +1  shared currency on the same side (both DOM or both FOR)
        −1  shared currency on opposite sides (DOM of one, FOR of other)

    cross_pair_name is the natural FOR+DOM name. The same cross may
    also be quoted in the market under the reversed name — callers
    should try both when looking up the cross-vol.

    Examples
    --------
    >>> triangulation_eta_and_cross("AUDUSD", "EURUSD")
    (1, 'AUDEUR')                          # same DOM (USD)
    >>> triangulation_eta_and_cross("USDJPY", "USDMXN")
    (1, 'JPYMXN')                          # same FOR (USD)
    >>> triangulation_eta_and_cross("USDJPY", "EURUSD")
    (-1, 'EURJPY')                         # USD opposite sides
    >>> triangulation_eta_and_cross("EURUSD", "USDJPY")
    (-1, 'EURJPY')                         # symmetric — same cross
    """
    if len(pair_a) != 6 or len(pair_b) != 6:
        return None
    A_for, A_dom = pair_a[:3].upper(), pair_a[3:].upper()
    B_for, B_dom = pair_b[:3].upper(), pair_b[3:].upper()

    # Same DOM, e.g., AUDUSD + EURUSD (both DOM=USD). Cross is FOR_A/FOR_B.
    if A_dom == B_dom:
        return +1, A_for + B_for
    # Same FOR, e.g., USDJPY + USDMXN (both FOR=USD). Cross is DOM_A/DOM_B.
    if A_for == B_for:
        return +1, A_dom + B_dom
    # FOR_A = DOM_B, e.g., USDJPY + EURUSD → USD = FOR_A = DOM_B
    # Non-shared currencies: A's DOM (JPY) and B's FOR (EUR). Cross = EURJPY.
    # S_A · S_B = (JPY/USD) · (USD/EUR) = JPY/EUR  →  EURJPY pair (DOM=JPY, FOR=EUR).
    if A_for == B_dom:
        return -1, B_for + A_dom
    # DOM_A = FOR_B, e.g., EURUSD + USDJPY → USD = DOM_A = FOR_B
    # Non-shared currencies: A's FOR (EUR) and B's DOM (JPY). Cross = EURJPY.
    # S_A · S_B = (USD/EUR) · (JPY/USD) = JPY/EUR  →  EURJPY pair.
    if A_dom == B_for:
        return -1, A_for + B_dom
    return None


def triangulated_correlation(
        sigma_a: float, sigma_b: float, sigma_cross: float, eta: int,
        clip: bool = True,
) -> float:
    """Implied A-B correlation from the cross-vol identity.

        ρ_AB = η · (σ²_A + σ²_B − σ²_X) / (2 σ_A σ_B)

    Parameters
    ----------
    sigma_a, sigma_b : float (decimal, e.g. 0.08 for 8% vol)
    sigma_cross      : float, vol of the implied cross
    eta              : int, ±1 per `triangulation_eta_and_cross`
    clip             : if True, clip the result to [-1, 1] when the
                       quoted vols imply an out-of-range correlation
                       (slightly stale quotes or no-arbitrage-violating
                       data can give |ρ| > 1; in practice the right
                       response is to clip and flag).

    Returns
    -------
    rho_implied : float in [-1, 1] (or unclipped if clip=False).
    """
    if eta not in (-1, +1):
        raise ValueError(f"eta must be ±1, got {eta}")
    if sigma_a <= 0.0 or sigma_b <= 0.0:
        raise ValueError(
            f"non-positive vol: sigma_a={sigma_a}, sigma_b={sigma_b}"
        )
    if sigma_cross < 0.0:
        raise ValueError(f"negative cross vol: sigma_cross={sigma_cross}")
    rho = eta * (sigma_a**2 + sigma_b**2 - sigma_cross**2) / (2.0 * sigma_a * sigma_b)
    if clip:
        rho = max(-1.0, min(1.0, rho))
    return float(rho)


# =============================================================================
# End-to-end implied-correlation lookup against a data folder
# =============================================================================
@dataclass
class TriangulationResult:
    """Diagnostic result of an end-to-end triangulation lookup."""
    pair_a: str
    pair_b: str
    cross_pair: str            # the cross-pair NAME as found in the data
    eta: int
    sigma_a: float             # decimal
    sigma_b: float
    sigma_cross: float
    rho_implied: float
    clipped: bool = False
    notes: str = ""


def implied_correlation_at_T(
        folder: str,
        pair_a: str, pair_b: str,
        T: float, valuation_date,
        prefer_a: str = "offshore",
        prefer_b: str = "offshore",
        prefer_cross: str = "offshore",
) -> "Optional[TriangulationResult]":
    """End-to-end triangulation: pulls σ_A, σ_B, σ_X from the data folder
    at the given (T, val_date), inverts the identity to get ρ.

    Returns None when:
      - the two pairs share no currency (triangulation undefined),
      - any of σ_A, σ_B, σ_X is missing for (T, val_date) in the folder.

    Uses `core.data_loader.get_pair_value_at_T` for the VOL_ATM panel
    lookups; tries both orientations of the cross-pair name (since the
    market may quote either, e.g. "EURJPY" or "JPYEUR"). The result's
    `cross_pair` field records whichever orientation was actually used.
    """
    # Lazy import to avoid circular dependency at module-load time.
    from core.data_loader import get_pair_value_at_T

    res = triangulation_eta_and_cross(pair_a, pair_b)
    if res is None:
        return None
    eta, cross_canonical = res

    sigma_a_pct = get_pair_value_at_T(folder, pair_a, prefer_a,
                                        "VOL_ATM", T, valuation_date)
    sigma_b_pct = get_pair_value_at_T(folder, pair_b, prefer_b,
                                        "VOL_ATM", T, valuation_date)
    if sigma_a_pct is None or sigma_b_pct is None:
        return None

    # Try the canonical cross orientation, then the reversed one.
    sigma_x_pct = None
    cross_found = ""
    for candidate in (cross_canonical,
                       cross_canonical[3:] + cross_canonical[:3]):
        v = get_pair_value_at_T(folder, candidate, prefer_cross,
                                  "VOL_ATM", T, valuation_date)
        if v is not None:
            sigma_x_pct = v
            cross_found = candidate
            break
    if sigma_x_pct is None:
        return None

    sigma_a = sigma_a_pct / 100.0
    sigma_b = sigma_b_pct / 100.0
    sigma_x = sigma_x_pct / 100.0
    rho_unclipped = triangulated_correlation(
        sigma_a, sigma_b, sigma_x, eta, clip=False,
    )
    rho = max(-1.0, min(1.0, rho_unclipped))
    clipped = (rho != rho_unclipped)

    notes = ""
    if clipped:
        notes = (f"Triangulation gave |ρ| = {abs(rho_unclipped):.3f} > 1 "
                  "(quote inconsistency); clipped to ±1.")

    return TriangulationResult(
        pair_a=pair_a, pair_b=pair_b, cross_pair=cross_found,
        eta=eta,
        sigma_a=sigma_a, sigma_b=sigma_b, sigma_cross=sigma_x,
        rho_implied=rho,
        clipped=clipped, notes=notes,
    )


def implied_correlation_time_series(
        folder: str,
        pair_a: str, pair_b: str,
        tenor_label: str = "1M",
        prefer_a: str = "offshore",
        prefer_b: str = "offshore",
        prefer_cross: str = "offshore",
) -> pd.Series:
    """Time series of triangulation-implied correlation.

    Pulls VOL_ATM panels for pair_a, pair_b, and their implied cross
    at the SPECIFIED standard tenor (one of '1M', '2M', '3M', '6M',
    '9M', '1Y' — not interpolated), aligns on common dates, and
    inverts the identity per date.

    For per-trade backtest usage you want this once (pre-compute), then
    look up the value at each trade date — much faster than calling
    `implied_correlation_at_T` per trade.

    Returns an empty Series if any of the three VOL_ATM panels is
    missing at the specified tenor.
    """
    from core.data_loader import load_panel

    res = triangulation_eta_and_cross(pair_a, pair_b)
    if res is None:
        return pd.Series(dtype=float)
    eta, cross_canonical = res

    df_a = load_panel(folder, "VOL_ATM", tenor_label,
                       prefer=prefer_a, pairs=(pair_a,))
    df_b = load_panel(folder, "VOL_ATM", tenor_label,
                       prefer=prefer_b, pairs=(pair_b,))
    if df_a.empty or df_b.empty or pair_a not in df_a.columns or pair_b not in df_b.columns:
        return pd.Series(dtype=float)

    # Try both cross orientations
    cross_ser = pd.Series(dtype=float)
    for candidate in (cross_canonical,
                       cross_canonical[3:] + cross_canonical[:3]):
        df_x = load_panel(folder, "VOL_ATM", tenor_label,
                           prefer=prefer_cross, pairs=(candidate,))
        if not df_x.empty and candidate in df_x.columns:
            cross_ser = df_x[candidate].dropna() / 100.0
            break
    if cross_ser.empty:
        return pd.Series(dtype=float)

    sigma_a_ser = df_a[pair_a].dropna() / 100.0
    sigma_b_ser = df_b[pair_b].dropna() / 100.0
    aligned = pd.concat(
        [sigma_a_ser, sigma_b_ser, cross_ser],
        axis=1, join="inner",
    ).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    aligned.columns = ["sa", "sb", "sx"]
    rho = eta * (aligned["sa"]**2 + aligned["sb"]**2 - aligned["sx"]**2) \
            / (2.0 * aligned["sa"] * aligned["sb"])
    return rho.clip(lower=-1.0, upper=1.0)

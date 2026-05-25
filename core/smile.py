"""FX smile interpolation: ATM + 25Δ RR + 25Δ BF → vol at any strike.

# Standard FX market convention

For each tenor the market quotes three numbers (per pair):
- ATM vol            (strike at the ATM-DNS or ATM-forward delta)
- 25Δ Risk Reversal  RR_25 = σ_25c − σ_25p          (typically positive for USD-EM)
- 25Δ Butterfly      BF_25 = (σ_25c + σ_25p) / 2 − σ_atm

From these, the wing vols are recovered:
    σ_25c = σ_atm + 0.5·RR + BF
    σ_25p = σ_atm − 0.5·RR + BF

# Vol at an arbitrary strike

Linear interpolation in CALL-DELTA space across the three anchors:
    Δ_c = 0.25 → σ_25c   (deep OTM call)
    Δ_c = 0.50 → σ_atm
    Δ_c = 0.75 → σ_25p   (deep OTM put; 25-put has |Δ_put| = 0.25 ⇔ Δ_call = 0.75)

Outside [0.25, 0.75], the curve is held flat at the wing vol (no extrapolation).
This matches what most flow-trading desks do as a first approximation.

# Convention used to evaluate Δ_K

Δ_call(K) is computed at σ_atm (the standard FX "fixed-vol" delta for
smile interpolation). This avoids the chicken-and-egg of "delta depends on
vol depends on K's smile vol depends on delta..." that would otherwise need
iteration. Most BBG-style desks accept this convention.

# Note vs Vanna-Volga

This is a vol-at-strike approximation (single σ per option, smile-aware),
not a structure-level Vanna-Volga correction. For pure vanilla and for
barrier options where the barrier is comfortably outside the smile's
sensitive region, this is usually adequate. For barrier options where the
KO sits in the deep-wing area, V-V's vega/vanna/volga replication would
give a different (often lower) price than this method — that's a known
limitation. For the case where you've already accepted flat-vol Black-
Scholes as the model, this is the cleanest smile-aware extension.
"""
from __future__ import annotations

from core.vanilla import vanilla_spot_delta


def wing_vols_25d(sigma_atm: float, rr_25: float, bf_25: float
                   ) -> tuple[float, float]:
    """Return (σ_25_call, σ_25_put) given (ATM, RR, BF) — all in decimal."""
    sigma_25c = sigma_atm + 0.5 * rr_25 + bf_25
    sigma_25p = sigma_atm - 0.5 * rr_25 + bf_25
    return float(sigma_25c), float(sigma_25p)


def smile_vol_at_call_delta(call_delta: float, sigma_atm: float,
                              rr_25: float, bf_25: float) -> float:
    """Linear interp of smile in call-delta space.

    Anchors: 0.25 (25C wing), 0.50 (ATM), 0.75 (25P wing).
    Flat outside [0.25, 0.75].
    """
    sigma_25c, sigma_25p = wing_vols_25d(sigma_atm, rr_25, bf_25)
    if call_delta <= 0.25:
        return sigma_25c
    if call_delta >= 0.75:
        return sigma_25p
    if call_delta < 0.5:
        # interp between 25C (Δ=0.25) and ATM (Δ=0.50)
        f = (0.5 - call_delta) / 0.25
        return sigma_atm + f * (sigma_25c - sigma_atm)
    if call_delta > 0.5:
        # interp between ATM (Δ=0.50) and 25P (Δ=0.75)
        f = (call_delta - 0.5) / 0.25
        return sigma_atm + f * (sigma_25p - sigma_atm)
    return sigma_atm


def smile_vol_at_strike(S: float, K: float, T: float, sigma_atm: float,
                          rr_25: float, bf_25: float,
                          r_d: float, r_f: float) -> float:
    """Compute smile-adjusted vol at strike K via spot-delta interpolation.

    Returns σ_atm if rr_25 and bf_25 are both effectively zero (no smile
    information available)."""
    if abs(rr_25) < 1e-12 and abs(bf_25) < 1e-12:
        return sigma_atm
    delta_c = vanilla_spot_delta('call', S, K, T, sigma_atm, r_d, r_f)
    # Numerical guard for very ITM/OTM strikes
    delta_c = max(0.0, min(1.0, float(delta_c)))
    return smile_vol_at_call_delta(delta_c, sigma_atm, rr_25, bf_25)


# =============================================================================
# Smile-panel construction for apps/1_vol_skew.py
# =============================================================================
# DELTA_STRIKES is the canonical 5-point smile grid used by the vol-skew
# dashboard. Order matters: lowest call-delta (most-OTM put) → ATM →
# highest call-delta (most-OTM call). Plotted left-to-right on the chart.
DELTA_STRIKES: list[str] = [
    "10Δ Put", "25Δ Put", "ATM", "25Δ Call", "10Δ Call",
]


def _maybe_pct_to_decimal(s: "pd.Series | None") -> "pd.Series | None":
    """Single-series version: divide by 100 iff |max| > 1.

    Used only for the ATM series (whose typical magnitude — 5..30 in
    percent, or 0.05..0.30 in decimal — gives a reliable signal).

    DO NOT use this for RR or BF directly: their typical values are
    small enough (often well under 1, even in percent) that the
    heuristic gives the wrong answer. Use `_convert_smile_inputs`
    instead, which decides based on the ATM scale and applies the
    same rule consistently to all five inputs.
    """
    import pandas as pd
    if s is None or s.empty:
        return s
    if float(s.abs().max()) > 1.0:
        return s / 100.0
    return s


def _convert_smile_inputs(atm: "pd.Series",
                                  *others: "pd.Series | None"
                                  ) -> "tuple[pd.Series, ...]":
    """Unit-convert ATM + smile inputs consistently.

    Bloomberg quotes vol-surface data in PERCENT (e.g. ATM=7.5, RR=-0.4,
    BF=0.3). The engine and `compute_smile_panel` need DECIMAL
    (0.075, -0.004, 0.003). All five series — ATM, RR_25, BF_25,
    RR_10, BF_10 — share units in the source data: if any one is in
    percent, all are.

    Why we don't decide per-series: ATM is reliably > 1 in percent,
    but RR/BF are often in the 0.1..0.9 range even in percent. The
    naive `_maybe_pct_to_decimal` would correctly divide ATM by 100
    but leave RR/BF unchanged, mixing scales and inflating the smile
    wings by ~100x. This bug was visible on USDJPY 3M/6M smiles where
    25Δ wings spiked to ~4.5× ATM (correct value ~1.05-1.15).

    The fix: use ATM's magnitude as the single arbiter. If ATM
    arrives in percent (|max| > 1), divide every series by 100;
    otherwise leave them all as-is.
    """
    if atm is None or atm.empty:
        return (atm, *others)
    divide = float(atm.abs().max()) > 1.0
    if not divide:
        return (atm, *others)
    out_atm = atm / 100.0
    out_others = tuple((s / 100.0) if (s is not None and not s.empty) else s
                              for s in others)
    return (out_atm, *out_others)


def compute_smile_panel(atm: "pd.Series",
                            rr_25: "pd.Series | None",
                            bf_25: "pd.Series | None",
                            rr_10: "pd.Series | None" = None,
                            bf_10: "pd.Series | None" = None,
                            ) -> "pd.DataFrame":
    """Compute the NORMALISED smile panel: vol-at-strike / vol-ATM.

    Returns a DataFrame with the same date index as `atm` and 5
    columns (DELTA_STRIKES). The ATM column is 1.0 by construction;
    the wing columns are σ_strike / σ_atm.

    Strike-specific vols (market convention):
        σ_25C = σ_atm + 0.5 × RR_25 + BF_25
        σ_25P = σ_atm − 0.5 × RR_25 + BF_25
        σ_10C = σ_atm + 0.5 × RR_10 + BF_10
        σ_10P = σ_atm − 0.5 × RR_10 + BF_10

    Missing 10Δ data → those two columns are filled with NaN.
    Missing 25Δ data → returns an empty DataFrame.
    """
    import pandas as pd
    if atm is None or atm.empty:
        return pd.DataFrame()
    if rr_25 is None or bf_25 is None:
        return pd.DataFrame()

    atm, rr_25, bf_25, rr_10, bf_10 = _convert_smile_inputs(
        atm.copy(),
        rr_25.copy() if rr_25 is not None else None,
        bf_25.copy() if bf_25 is not None else None,
        rr_10.copy() if rr_10 is not None else None,
        bf_10.copy() if bf_10 is not None else None,
    )

    # Align all smile inputs on the ATM index, forward-filling so a
    # missing RR/BF day doesn't blank out the whole row.
    rr25 = rr_25.reindex(atm.index).ffill() if rr_25 is not None else None
    bf25 = bf_25.reindex(atm.index).ffill() if bf_25 is not None else None
    rr10 = rr_10.reindex(atm.index).ffill() if rr_10 is not None else None
    bf10 = bf_10.reindex(atm.index).ffill() if bf_10 is not None else None

    out = pd.DataFrame(index=atm.index, dtype=float)
    safe_atm = atm.where(atm > 1e-9)  # avoid divide-by-near-zero
    out["ATM"] = 1.0

    sigma_25c = atm + 0.5 * rr25 + bf25
    sigma_25p = atm - 0.5 * rr25 + bf25
    out["25Δ Call"] = sigma_25c / safe_atm
    out["25Δ Put"] = sigma_25p / safe_atm

    if rr10 is not None and bf10 is not None:
        sigma_10c = atm + 0.5 * rr10 + bf10
        sigma_10p = atm - 0.5 * rr10 + bf10
        out["10Δ Call"] = sigma_10c / safe_atm
        out["10Δ Put"] = sigma_10p / safe_atm
    else:
        out["10Δ Call"] = float("nan")
        out["10Δ Put"] = float("nan")

    return out[DELTA_STRIKES]


def compute_absolute_smile_panel(atm: "pd.Series",
                                       rr_25: "pd.Series | None",
                                       bf_25: "pd.Series | None",
                                       rr_10: "pd.Series | None" = None,
                                       bf_10: "pd.Series | None" = None,
                                       ) -> "pd.DataFrame":
    """Like compute_smile_panel but returns ABSOLUTE vol levels (not
    normalised to ATM). Same DELTA_STRIKES columns. ATM column is the
    raw σ_atm series. Wings via market-convention formulas (no /ATM
    normalisation). Returned in decimal units (e.g. 0.075 = 7.5%)."""
    import pandas as pd
    if atm is None or atm.empty:
        return pd.DataFrame()
    if rr_25 is None or bf_25 is None:
        return pd.DataFrame()

    atm, rr_25, bf_25, rr_10, bf_10 = _convert_smile_inputs(
        atm.copy(),
        rr_25.copy() if rr_25 is not None else None,
        bf_25.copy() if bf_25 is not None else None,
        rr_10.copy() if rr_10 is not None else None,
        bf_10.copy() if bf_10 is not None else None,
    )

    rr25 = rr_25.reindex(atm.index).ffill() if rr_25 is not None else None
    bf25 = bf_25.reindex(atm.index).ffill() if bf_25 is not None else None
    rr10 = rr_10.reindex(atm.index).ffill() if rr_10 is not None else None
    bf10 = bf_10.reindex(atm.index).ffill() if bf_10 is not None else None

    out = pd.DataFrame(index=atm.index, dtype=float)
    out["ATM"] = atm
    out["25Δ Call"] = atm + 0.5 * rr25 + bf25
    out["25Δ Put"] = atm - 0.5 * rr25 + bf25
    if rr10 is not None and bf10 is not None:
        out["10Δ Call"] = atm + 0.5 * rr10 + bf10
        out["10Δ Put"] = atm - 0.5 * rr10 + bf10
    else:
        out["10Δ Call"] = float("nan")
        out["10Δ Put"] = float("nan")

    return out[DELTA_STRIKES]

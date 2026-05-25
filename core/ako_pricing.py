"""Shared single-leg RKO (American-barrier KO) pricing dispatcher.

Three pricing models for a single-leg American-barrier knock-out,
all sharing the same Reiner-Rubinstein closed form for the BS engine
but differing in WHICH σ enters:

    flat_atm        ako_closed_form at σ_atm   (smile-ignorant baseline)
    vol_at_strike   ako_closed_form at σ_smile(K)
    vanna_volga     vv_price_ko on top of ako_closed_form  (smile-corrected)

This is the American-barrier analog of `core.eko_pricing`. Same API,
same three model labels — the only differences are:
  - the underlying BS engine (ako_closed_form instead of ko_price)
  - the meaning of the barrier (continuous monitoring rather than
    terminal-only)

Used by:
  - pages/rko_pricer.py    (single-leg live pricer)
  - core/backtest_american.py  (single-leg backtester)

The two callers MUST go through this dispatcher so a sidebar/UI choice
of model takes effect uniformly across live valuation and backtest —
otherwise PnL ledgers and live mid-marks would silently disagree.

Mirrors the dispatcher in core.eko_pricing one-for-one. Adding a new
model is a single edit in BOTH modules (or, longer term, refactor to a
single generic dispatcher parameterized by the BS engine).
"""
from __future__ import annotations
from core.american_barrier import ako_closed_form


PRICING_MODELS = ("flat_atm", "vol_at_strike", "vanna_volga")
PRICING_MODEL_LABELS = {
    "flat_atm": "Flat BS (σ_atm)",
    "vol_at_strike": "Vol-at-strike σ_smile(K)",
    "vanna_volga": "Vanna-Volga",
}


def price_ako_dispatch(
        option_type: str, barrier_type: str,
        S: float, K: float, H: float, T: float,
        sigma_atm: float, sigma_smile: float,
        rr_25: float, bf_25: float,
        r_d: float, r_f: float,
        model: str,
) -> "tuple[float, dict]":
    """Price an RKO under the selected single-leg pricing model.

    Parameters
    ----------
    option_type   : 'call' | 'put'
    barrier_type  : 'up_and_out' | 'down_and_out'
    S, K, H, T    : spot, strike, barrier, years-to-expiry
    sigma_atm     : ATM vol (decimal). Used by flat_atm and vanna_volga.
    sigma_smile   : vol-at-strike σ_smile(K) (decimal). Used by
                    vol_at_strike and reported as `vol_used` for that
                    branch.
    rr_25, bf_25  : 25Δ risk-reversal and butterfly (decimals). Required
                    by vanna_volga; ignored by the other branches.
    r_d, r_f      : domestic / foreign continuously-compounded rates.
    model         : one of PRICING_MODELS.

    Returns
    -------
    (price_per_unit, detail_dict)
        detail_dict always has 'model' and 'vol_used'. For 'vanna_volga'
        it also has 'correction', 'price_bs', and 'vv_detail' (the raw
        output from core.vanna_volga.vv_correction — hedge weights,
        smile costs, condition number, etc.).
    """
    if model == "flat_atm":
        p = ako_closed_form(option_type, barrier_type, S, K, H, T,
                             sigma_atm, r_d, r_f)
        return float(p), {"model": "flat_atm", "vol_used": float(sigma_atm)}

    if model == "vol_at_strike":
        p = ako_closed_form(option_type, barrier_type, S, K, H, T,
                             sigma_smile, r_d, r_f)
        return float(p), {"model": "vol_at_strike", "vol_used": float(sigma_smile)}

    if model == "vanna_volga":
        # Lazy import — avoids loading the VV machinery for the simple
        # flat / vol-at-strike paths.
        from core.vanna_volga import vv_price_ko
        vv_out = vv_price_ko(
            option_type, barrier_type, S, K, H, T,
            sigma_atm, rr_25, bf_25, r_d, r_f,
            flat_vol_pricer=ako_closed_form,
        )
        return float(vv_out["price_vv"]), {
            "model": "vanna_volga",
            "vol_used": float(sigma_atm),
            "correction": float(vv_out["correction"]),
            "price_bs": float(vv_out["price_bs"]),
            "vv_detail": vv_out["detail"],
        }

    raise ValueError(
        f"Unknown pricing model: {model!r}. "
        f"Choices: {PRICING_MODELS}."
    )

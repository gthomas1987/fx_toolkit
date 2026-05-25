"""Shared single-leg EKO pricing dispatcher.

Three pricing models for a European knock-out leg, all sharing the same
ko_price closed form for the BS engine but differing in WHICH σ enters:

    flat_atm        ko_price at σ_atm   (smile-ignorant baseline)
    vol_at_strike   ko_price at σ_smile(K)   (historical default)
    vanna_volga     vv_price_ko on top of ko_price   (smile-corrected)

Used by:
  - pages/eko_pricer.py TAB 1 (single-leg live pricer)
  - core/backtest.py    run_single_strategy (single-leg backtester)

The two callers MUST go through this dispatcher so a sidebar/UI choice
of model takes effect uniformly across live valuation and backtest —
otherwise PnL ledgers and live mid-marks would silently disagree.

The dispatcher takes BOTH σ_atm and σ_smile (rather than choosing one
upstream) so the same call handles all three branches. The lazy
import of vv_price_ko keeps the flat-vol path import-free.
"""
from __future__ import annotations
from core.ko import ko_price


PRICING_MODELS = ("flat_atm", "vol_at_strike", "vanna_volga")
PRICING_MODEL_LABELS = {
    "flat_atm": "Flat BS (σ_atm)",
    "vol_at_strike": "Vol-at-strike σ_smile(K)",
    "vanna_volga": "Vanna-Volga",
}


def price_eko_dispatch(
        option_type: str, barrier_type: str,
        S: float, K: float, H: float, T: float,
        sigma_atm: float, sigma_smile: float,
        rr_25: float, bf_25: float,
        r_d: float, r_f: float,
        model: str,
) -> "tuple[float, dict]":
    """Price an EKO under the selected single-leg pricing model.

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
        p = ko_price(option_type, barrier_type, S, K, H, T,
                     sigma_atm, r_d, r_f)
        return float(p), {"model": "flat_atm", "vol_used": float(sigma_atm)}

    if model == "vol_at_strike":
        p = ko_price(option_type, barrier_type, S, K, H, T,
                     sigma_smile, r_d, r_f)
        return float(p), {"model": "vol_at_strike", "vol_used": float(sigma_smile)}

    if model == "vanna_volga":
        # Lazy import — avoids loading the VV machinery for the simple
        # flat / vol-at-strike paths.
        from core.vanna_volga import vv_price_ko
        vv_out = vv_price_ko(
            option_type, barrier_type, S, K, H, T,
            sigma_atm, rr_25, bf_25, r_d, r_f,
            flat_vol_pricer=ko_price,
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

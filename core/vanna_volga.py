"""Vanna-Volga smile-adjusted pricing for FX exotics — Castagna-Mercurio
(2007) first-order formulation.

# Why
The Reiner-Rubinstein closed form (and any other "single-vol" pricer)
uses one constant σ across all strikes and times. Real FX vol surfaces
have a smile: σ depends on strike (and tenor). For exotic options
whose value depends on prices ACROSS the smile (barriers, digitals,
one-touches), a single-vol price systematically misses the smile
premium.

Vanna-Volga is the standard industry approach to incorporate the smile
for FX exotics without resorting to a full local-vol or stochastic-vol
PDE solve. It uses three liquid market reference vanillas (ATM, 25Δ
call, 25Δ put) and computes the difference between their smile-vol
prices and their flat-ATM prices, weighted by hedge ratios that match
the exotic's vega, vanna, and volga.

# The formula (Castagna-Mercurio 2007)
    P_VV = P_BS(σ_atm)
         + x_ATM × [C(K_ATM, σ_ATM) − C(K_ATM, σ_ATM)]    (= 0 for ATM)
         + x_25C × [C(K_25C, σ_25C) − C(K_25C, σ_ATM)]
         + x_25P × [C(K_25P, σ_25P) − C(K_25P, σ_ATM)]

where the hedging weights solve the 3×3 system:
    [vega_ATM  vega_25C  vega_25P ]   [x_ATM]   [vega_O ]
    [vanna_ATM vanna_25C vanna_25P] × [x_25C] = [vanna_O]
    [volga_ATM volga_25C volga_25P]   [x_25P]   [volga_O]

(all Greeks evaluated at σ_ATM for the reference vanillas; the exotic
Greeks come from finite-difference on the supplied flat-vol pricer.)

# What this matches
Empirically validated against Bloomberg OVML: for a USDJPY 1M ATM
up-and-out call with H = +2.7% (smile: σ_atm = 7.468%, RR_25 = −1.435%,
BF_25 = +0.208%), unscaled VV here matches BBG-VV within 0.5% — well
inside the noise of vol-input rounding. Bloomberg's "Vanna-Volga"
model in OVML appears to be this same unscaled first-order formulation.

# What this does NOT match
- The "survival-probability-weighted" VV variant (Bossens et al. 2010)
  scales the correction by P(no-touch). Some banks use this for KOs.
  Available via the `weight_by_survival` flag — defaults to False
  because BBG matches the unscaled version.
- Local-volatility or stochastic-vol prices. For very long-dated or
  deep-OTM trades, all VV-style approaches lose accuracy vs full PDE
  solves with a calibrated surface.
- Second-order VV variants (Castagna 2010 ch. 5). The first-order is
  more than adequate for short-dated FX KOs (< 6M).

# Performance
~5 vanilla evaluations + 9 Greek calls + 1 linear solve = O(0.5 ms)
per VV price. The dominant cost is the FOUR finite-difference exotic
evaluations (for vanna), each of which is one closed-form call.
"""
from __future__ import annotations

import numpy as np

from core.vanilla import (
    vanilla_price, vanilla_vega, vanilla_vanna, vanilla_volga,
    strike_from_delta, atm_forward_strike,
)


def _exotic_greeks_fd(pricer, S: float, T: float, sigma: float,
                          r_d: float, r_f: float,
                          eps_sigma: float = 1e-4,
                          eps_S_rel: float = 1e-4
                          ) -> "tuple[float, float, float]":
    """Compute (vega, vanna, volga) of an exotic via central FD.

    `pricer(S, sigma)` must be a closure that prices the exotic at the
    given (S, sigma) holding ALL other parameters (K, H, T, r_d, r_f,
    barrier type, etc.) fixed.

    vega  = ∂P/∂σ                    central FD,    2 calls
    volga = ∂²P/∂σ²                  central FD,    2 calls (reuses center)
    vanna = ∂²P/(∂σ ∂S)              4-point cross, 4 calls

    Returns the Greeks evaluated at (S, sigma).
    """
    eps_S = max(S * eps_S_rel, 1e-9)

    # Center
    p0 = pricer(S, sigma)
    # Pure-σ FD
    p_sig_up = pricer(S, sigma + eps_sigma)
    p_sig_dn = pricer(S, sigma - eps_sigma)
    vega = (p_sig_up - p_sig_dn) / (2.0 * eps_sigma)
    volga = (p_sig_up - 2.0 * p0 + p_sig_dn) / (eps_sigma * eps_sigma)
    # Cross-FD for vanna
    p_S_up_sig_up = pricer(S + eps_S, sigma + eps_sigma)
    p_S_up_sig_dn = pricer(S + eps_S, sigma - eps_sigma)
    p_S_dn_sig_up = pricer(S - eps_S, sigma + eps_sigma)
    p_S_dn_sig_dn = pricer(S - eps_S, sigma - eps_sigma)
    vanna = (p_S_up_sig_up - p_S_up_sig_dn
              - p_S_dn_sig_up + p_S_dn_sig_dn) / (4.0 * eps_S * eps_sigma)
    return vega, vanna, volga


def vv_correction(pricer,
                       S: float, T: float, sigma_atm: float,
                       rr_25: float, bf_25: float,
                       r_d: float, r_f: float,
                       weight_by_survival: bool = False,
                       survival_prob: float = 1.0,
                       ) -> "dict":
    """Compute the Vanna-Volga smile correction (Castagna-Mercurio 2007).

    Returns a dict with:
        'correction'        — the VV adjustment to ADD to P_BS(σ_atm)
        'weights'           — (x_atm, x_25C, x_25P)
        'smile_costs'       — per-vanilla [C(σ_smile) − C(σ_atm)]
        'exotic_greeks'     — (vega, vanna, volga) of the exotic
        'reference_strikes' — (K_atm, K_25C, K_25P)
        'reference_vols'    — (σ_atm, σ_25C, σ_25P)
        'condition_number'  — of the 3×3 hedge system (sanity check)

    Args:
        pricer: callable pricer(S, sigma) → price, holding all other
            trade params fixed. The exotic's vega/vanna/volga come from
            FD on this function.
        S, T, sigma_atm, r_d, r_f: standard FX option params.
        rr_25: 25Δ risk reversal in decimal (e.g. -0.01435 for -1.435%).
            Convention: rr_25 = σ_25C - σ_25P. Negative = puts more
            expensive than calls.
        bf_25: 25Δ butterfly in decimal (e.g. 0.00208 for +0.208%).
            σ_25C = σ_atm + 0.5*rr_25 + bf_25
            σ_25P = σ_atm - 0.5*rr_25 + bf_25
        weight_by_survival: if True, scale the correction by
            `survival_prob`. Defaults to False — Bloomberg's VV does
            NOT use this scaling, empirically.
        survival_prob: P(barrier not hit). Only used when
            weight_by_survival is True.

    If rr_25 == 0 and bf_25 == 0 (flat smile), the correction is zero
    and this returns a dict with correction=0 and Nones for the rest.
    """
    if rr_25 == 0.0 and bf_25 == 0.0:
        return {
            "correction": 0.0,
            "weights": None,
            "smile_costs": None,
            "exotic_greeks": None,
            "reference_strikes": None,
            "reference_vols": None,
            "condition_number": None,
            "note": "Flat smile (RR = BF = 0) → no VV correction.",
        }

    # 1. Reference strikes (ATM-forward + 25Δ call/put wings)
    K_atm = atm_forward_strike(S, T, r_d, r_f)
    K_25c = strike_from_delta('call', 0.25, S, T, sigma_atm, r_d, r_f)
    K_25p = strike_from_delta('put',  0.25, S, T, sigma_atm, r_d, r_f)

    # 2. Smile vols at each wing
    sigma_25c = sigma_atm + 0.5 * rr_25 + bf_25
    sigma_25p = sigma_atm - 0.5 * rr_25 + bf_25

    ref_K = [K_atm, K_25c, K_25p]
    ref_sigma_smile = [sigma_atm, sigma_25c, sigma_25p]

    # 3. Exotic Greeks via FD on supplied pricer
    vega_O, vanna_O, volga_O = _exotic_greeks_fd(
        pricer, S, T, sigma_atm, r_d, r_f
    )

    # 4. Vanilla Greeks at each reference strike, evaluated at σ_atm
    vegas = [vanilla_vega(S, K_, T, sigma_atm, r_d, r_f) for K_ in ref_K]
    vannas = [vanilla_vanna(S, K_, T, sigma_atm, r_d, r_f) for K_ in ref_K]
    volgas = [vanilla_volga(S, K_, T, sigma_atm, r_d, r_f) for K_ in ref_K]

    # 5. Solve 3×3 system M @ x = rhs
    M = np.array([vegas, vannas, volgas])
    rhs = np.array([vega_O, vanna_O, volga_O])
    cond = float(np.linalg.cond(M))
    try:
        x = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return {
            "correction": 0.0,
            "weights": None,
            "smile_costs": None,
            "exotic_greeks": (vega_O, vanna_O, volga_O),
            "reference_strikes": tuple(ref_K),
            "reference_vols": tuple(ref_sigma_smile),
            "condition_number": cond,
            "note": "VV hedge system singular — falling back to BS.",
        }

    # 6. Per-vanilla smile cost: C(σ_smile) − C(σ_atm)
    smile_costs = []
    for K_, sig_smile in zip(ref_K, ref_sigma_smile):
        c_smile = vanilla_price('call', S, K_, T, sig_smile, r_d, r_f)
        c_flat = vanilla_price('call', S, K_, T, sigma_atm, r_d, r_f)
        smile_costs.append(c_smile - c_flat)

    # 7. Weighted correction
    correction = float(sum(xi * ci for xi, ci in zip(x, smile_costs)))
    if weight_by_survival:
        correction *= float(survival_prob)

    return {
        "correction": correction,
        "weights": tuple(float(xi) for xi in x),
        "smile_costs": tuple(smile_costs),
        "exotic_greeks": (float(vega_O), float(vanna_O), float(volga_O)),
        "reference_strikes": tuple(ref_K),
        "reference_vols": tuple(ref_sigma_smile),
        "condition_number": cond,
    }


def vv_price_ko(option_type: str, barrier_type: str,
                  S: float, K: float, H: float, T: float,
                  sigma_atm: float, rr_25: float, bf_25: float,
                  r_d: float, r_f: float,
                  flat_vol_pricer=None,
                  weight_by_survival: bool = False,
                  ) -> "dict":
    """Vanna-Volga price for a KO option.

    Wraps `vv_correction` for the specific case of an American- or
    European-barrier KO, by constructing the appropriate (S, σ)→price
    closure around the supplied flat-vol pricer.

    Args:
        flat_vol_pricer: a callable with the same signature as
            `core.american_barrier.ako_closed_form` — i.e.
            (option_type, barrier_type, S, K, H, T, sigma, r_d, r_f) → price.
            Defaults to importing ako_closed_form. Pass core.ko.ko_price
            for European-barrier VV (App 9 use case).
        weight_by_survival: see vv_correction. Default False
            (matches Bloomberg OVML's VV).

    Returns a dict:
        'price_vv'     — final VV price (per unit FOR notional)
        'price_bs'     — flat-vol BS price (for reference)
        'correction'   — VV adjustment added to price_bs
        'detail'       — the full output from vv_correction()
    """
    if flat_vol_pricer is None:
        # Late import to avoid circular dependency
        from core.american_barrier import ako_closed_form as flat_vol_pricer

    # BS reference price
    p_bs = flat_vol_pricer(option_type, barrier_type, S, K, H, T,
                              sigma_atm, r_d, r_f)

    # Survival probability (for the optional weighting variant)
    survival = 1.0
    if weight_by_survival:
        from core.american_barrier import ako_probability_continuous
        # Only meaningful for American barriers; for European barriers
        # the "no-touch" notion is different (just terminal not-breached).
        # Use continuous-monitoring prob as the universal proxy.
        ko_prob = ako_probability_continuous(barrier_type, S, H, T,
                                                  sigma_atm, r_d, r_f)
        survival = 1.0 - ko_prob

    # Build the closure for FD Greeks: holds K, H, option_type,
    # barrier_type fixed; varies (S, sigma).
    def closure(S_, sigma_):
        return flat_vol_pricer(option_type, barrier_type, S_, K, H, T,
                                  sigma_, r_d, r_f)

    detail = vv_correction(
        closure, S, T, sigma_atm, rr_25, bf_25, r_d, r_f,
        weight_by_survival=weight_by_survival,
        survival_prob=survival,
    )
    correction = detail["correction"]
    price_vv = max(0.0, float(p_bs + correction))

    return {
        "price_vv": price_vv,
        "price_bs": float(p_bs),
        "correction": float(correction),
        "detail": detail,
    }

"""Step 1a — validate Vanna-Volga on EKO closed form.

Goal: confirm that `vv_price_ko(flat_vol_pricer=ko_price, ...)` produces
a sensible VV-corrected EKO price on a representative trade, and quantify
the difference vs the current "vol-at-strike" method.

Reference benchmarks reported:
    A. Flat BS at σ_atm                 — baseline, no smile
    B. Vol-at-strike σ_smile(K)         — current toolkit method
    C. Vanna-Volga (structure-level)    — proposed upgrade
    M1. MC at σ_atm   (cross-checks A)
    M2. MC at σ_smile (cross-checks B)

The Bloomberg/OVML reference in vanna_volga.py's docstring (USDJPY 1M UO
call, H = +2.7%) is the headline test case.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
from core.ko import ko_price
from core.ko_mc import mc_ko_price
from core.smile import smile_vol_at_strike
from core.vanna_volga import vv_price_ko
from core.vanilla import atm_forward_strike, strike_from_delta


def _row(label, price, baseline=None, extra=""):
    pct = "" if baseline is None else f"  ({(price/baseline - 1)*100:+6.2f}% vs A)"
    return f"  {label:<40s}  {price:10.6f}{pct}  {extra}"


def run_one_case(name, option_type, barrier_type, S, T,
                 sigma_atm, rr_25, bf_25, r_d, r_f,
                 K_spec, H_spec, n_mc_paths=400_000, mc_seed=42):
    """Run all five pricers for one trade and pretty-print."""
    # Resolve strike: K_spec may be a number, "ATMF", or "25dC"/"25dP"
    if isinstance(K_spec, str):
        if K_spec == "ATMF":
            K = atm_forward_strike(S, T, r_d, r_f)
        elif K_spec.endswith("dC"):
            K = strike_from_delta("call", float(K_spec[:-2])/100, S, T,
                                  sigma_atm, r_d, r_f)
        elif K_spec.endswith("dP"):
            K = strike_from_delta("put", float(K_spec[:-2])/100, S, T,
                                  sigma_atm, r_d, r_f)
        else:
            raise ValueError(f"Bad K_spec: {K_spec}")
    else:
        K = float(K_spec)

    # Resolve barrier: H_spec may be a number or "+X%"/"-X%" relative to spot
    if isinstance(H_spec, str) and H_spec.endswith("%"):
        H = S * (1.0 + float(H_spec.rstrip("%"))/100)
    else:
        H = float(H_spec)

    # Smile vol at K
    sigma_smile = smile_vol_at_strike(S, K, T, sigma_atm, rr_25, bf_25, r_d, r_f)

    # A. Flat BS at σ_atm
    p_A = ko_price(option_type, barrier_type, S, K, H, T, sigma_atm, r_d, r_f)

    # B. Vol-at-strike at σ_smile(K)
    p_B = ko_price(option_type, barrier_type, S, K, H, T, sigma_smile, r_d, r_f)

    # C. Vanna-Volga on EKO closed form
    vv_out = vv_price_ko(option_type, barrier_type, S, K, H, T,
                        sigma_atm, rr_25, bf_25, r_d, r_f,
                        flat_vol_pricer=ko_price)
    p_C = vv_out["price_vv"]

    # MC sanity checks
    mc_atm = mc_ko_price(option_type, barrier_type, S, K, H, T,
                         sigma_atm, r_d, r_f,
                         n_paths=n_mc_paths, seed=mc_seed)
    mc_smile = mc_ko_price(option_type, barrier_type, S, K, H, T,
                           sigma_smile, r_d, r_f,
                           n_paths=n_mc_paths, seed=mc_seed)

    print(f"\n=== {name} ===")
    print(f"  S={S:.4f}, K={K:.4f}, H={H:.4f}, T={T:.4f}y")
    print(f"  σ_atm={sigma_atm*100:.3f}%, RR_25={rr_25*100:+.3f}%, BF_25={bf_25*100:+.3f}%")
    print(f"  σ_smile(K)={sigma_smile*100:.3f}%, r_d={r_d*100:.2f}%, r_f={r_f*100:.2f}%")
    print(f"  {option_type} / {barrier_type}\n")
    print(_row("A. Flat BS @ σ_atm", p_A))
    print(_row("B. Vol-at-strike @ σ_smile(K)", p_B, p_A))
    print(_row("C. Vanna-Volga", p_C, p_A))
    print(_row("M1. MC @ σ_atm", mc_atm["price"], p_A,
               f"±{1.96*mc_atm['std_err']:.6f}"))
    print(_row("M2. MC @ σ_smile(K)", mc_smile["price"], p_A,
               f"±{1.96*mc_smile['std_err']:.6f}"))
    # Sanity: |B - M2| should be MC-noise size
    print(f"\n  Sanity: |A - M1| = {abs(p_A - mc_atm['price'])*1e6:.1f}e-6  "
          f"(MC 95% half-width: {1.96*mc_atm['std_err']*1e6:.1f}e-6)")
    print(f"  Sanity: |B - M2| = {abs(p_B - mc_smile['price'])*1e6:.1f}e-6  "
          f"(MC 95% half-width: {1.96*mc_smile['std_err']*1e6:.1f}e-6)")
    print(f"\n  VV detail: weights={tuple(round(x,3) for x in vv_out['detail']['weights'])}")
    print(f"             smile costs={tuple(round(x,6) for x in vv_out['detail']['smile_costs'])}")
    print(f"             correction = {vv_out['correction']:.6f}  "
          f"({vv_out['correction']/p_A*100:+.2f}% of A)")


# ----------------------------------------------------------------------------
# Test cases
# ----------------------------------------------------------------------------

# Headline: USDJPY 1M UO call — matches vanna_volga.py docstring example
# (where it says BBG-VV matches within 0.5%).
print("="*78)
print("Vanna-Volga on EKO — validation")
print("="*78)

run_one_case(
    "USDJPY 1M ATMF UO call, H = spot+2.7% (docstring case)",
    option_type="call", barrier_type="up_and_out",
    S=155.00, T=1/12,
    sigma_atm=0.07468, rr_25=-0.01435, bf_25=0.00208,
    r_d=0.005, r_f=0.045,  # JPY = DOM, USD = FOR
    K_spec="ATMF", H_spec="+2.7%",
)

run_one_case(
    "USDJPY 3M 25dC UO call, H = spot+5%",
    option_type="call", barrier_type="up_and_out",
    S=155.00, T=3/12,
    sigma_atm=0.08, rr_25=-0.012, bf_25=0.003,
    r_d=0.005, r_f=0.045,
    K_spec="25dC", H_spec="+5%",
)

run_one_case(
    "EURUSD 1M ATMF DO put, H = spot-3%",
    option_type="put", barrier_type="down_and_out",
    S=1.0850, T=1/12,
    sigma_atm=0.075, rr_25=-0.003, bf_25=0.0015,
    r_d=0.045, r_f=0.025,  # USD = DOM, EUR = FOR
    K_spec="ATMF", H_spec="-3%",
)

run_one_case(
    "AUDUSD 2M 25dP DO put, H = spot-4% (left-skewed pair)",
    option_type="put", barrier_type="down_and_out",
    S=0.6600, T=2/12,
    sigma_atm=0.105, rr_25=-0.018, bf_25=0.0042,  # left-skew (RR < 0)
    r_d=0.045, r_f=0.04,
    K_spec="25dP", H_spec="-4%",
)

print("\n" + "="*78)
print("Interpretation:")
print("  - M1 ≈ A and M2 ≈ B confirms closed-form plumbing is correct.")
print("  - C - B = the smile-correction step VV adds beyond vol-at-strike.")
print("  - VV correction > 0 on right-skewed pairs for UO calls (USDJPY).")
print("  - VV correction > 0 on left-skewed pairs for DO puts (AUDUSD).")
print("="*78)

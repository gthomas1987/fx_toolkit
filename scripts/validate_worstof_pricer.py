"""Step 2a — validate the worst-of EKO pricer.

Tests four properties:

  1. Sanity vs Monte Carlo
     For a representative worst-of UO call (USDJPY × USDMXN style),
     closed-form price should fall inside the MC 95% CI.

  2. Single-leg reduction
     With leg-2 alive-ITM region covering ~all probability mass AND
     I_2 always >= leg-1 max payoff, the worst-of should reduce to
     leg-1's single-leg EKO price (the leg-1 intrinsic is never the
     'min'). Cross-check against core.ko.ko_price.

  3. Rho monotonicity
     For two UO calls with identical-direction smiles, worst-of price
     should be MONOTONIC INCREASING in rho. (Higher correlation =
     more joint-survival probability; with positive correlated legs,
     the legs are more likely to both pay or both not pay rather
     than offsetting.)
     Edge: rho = 1 and identical legs => price = single-leg EKO.

  4. Barrier-direction coverage
     Run a handful of (opt, bar_dir) combinations and confirm CF
     matches MC within noise. Covers UO-call, DO-put, mixed UO-call
     + DO-put.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np

from core.worstof_pricer import (
    WorstOfLeg, worstof_eko_price_cf, worstof_eko_price_mc,
)
from core.ko import ko_price


def _hdr(s):
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


def _show(label, cf, mc):
    in_ci = (mc["ci_95_lower"] <= cf["price"] <= mc["ci_95_upper"])
    flag = "OK" if in_ci else "OUTSIDE CI"
    print(f"  {label:<45s}  CF={cf['price']:11.8f}  "
          f"MC={mc['price']:11.8f} ± {1.96*mc['std_err']:.8f}   {flag}")
    return in_ci


# =============================================================================
# Test 1: Headline case, UO call x UO call (USDJPY x USDMXN style)
# =============================================================================
_hdr("Test 1 — headline case: USDJPY × USDMXN, UO call × UO call")

# Assume both pairs use USD as DOM for this test (after inversion if needed).
# So we're pretending these are USDJPY and USDMXN inverted to JPYUSD and MXNUSD.
# Hypothetical setup:
leg_a = WorstOfLeg(S=0.0065, K=0.0064, H=0.0068,
                   sigma=0.08, r_d=0.045, r_f=0.005,  # USD DOM, JPY FOR
                   opt="call", bar_dir="up_and_out")
leg_b = WorstOfLeg(S=0.058, K=0.057, H=0.061,
                   sigma=0.12, r_d=0.045, r_f=0.080,  # USD DOM, MXN FOR
                   opt="call", bar_dir="up_and_out")
T = 3 / 12
r_d = 0.045    # USD DOM
rho = 0.40     # USDJPY and USDMXN are usually moderately positively correlated

cf = worstof_eko_price_cf(leg_a, leg_b, T, rho, r_d, n_quad=80)
mc = worstof_eko_price_mc(leg_a, leg_b, T, rho, r_d, n_paths=2_000_000, seed=42)
ok1 = _show("UO call × UO call, rho=0.40", cf, mc)
print(f"     p_alive_joint   CF={cf['p_alive_joint']:.4f}   MC={mc['p_alive_joint']:.4f}")
print(f"     p_alive_leg1    CF={cf['p_alive_leg1']:.4f}   MC={mc['p_alive_leg1']:.4f}")
print(f"     p_alive_leg2    CF={cf['p_alive_leg2']:.4f}   MC={mc['p_alive_leg2']:.4f}")
print(f"     p_both_itm_alive CF={cf['p_both_itm_and_alive']:.4f}   "
       f"MC={mc['p_both_itm_and_alive']:.4f}")


# =============================================================================
# Test 2: Single-leg reduction
# Trick: make leg_b's intrinsic huge by setting K_b very small and barrier
# far above spot. Then I_2 = S_b^T - K_b >> leg_a max payoff (H_a - K_a),
# so the min always picks leg-a's intrinsic.
# =============================================================================
_hdr("Test 2 — degenerate to single-leg EKO when leg 2's intrinsic dominates")

leg_a = WorstOfLeg(S=1.0, K=1.0, H=1.05,
                   sigma=0.10, r_d=0.045, r_f=0.045,
                   opt="call", bar_dir="up_and_out")
# leg_b: K = 0.001 (very deep ITM), H = 1e6 (effectively no barrier).
# I_b ≈ S_b - K_b ≈ S_b is around leg_b.S = 1.0, while leg_a max payoff
# = H_a - K_a = 0.05.  So leg_b's intrinsic always dominates.
leg_b = WorstOfLeg(S=1.0, K=0.001, H=1e6,
                   sigma=0.10, r_d=0.045, r_f=0.045,
                   opt="call", bar_dir="up_and_out")
T = 1 / 12
r_d = 0.045
rho = 0.30

# Single-leg leg_a price (UO call)
single_leg_a = ko_price("call", "up_and_out",
                         leg_a.S, leg_a.K, leg_a.H, T,
                         leg_a.sigma, r_d, leg_a.r_f)
cf = worstof_eko_price_cf(leg_a, leg_b, T, rho, r_d, n_quad=120)
mc = worstof_eko_price_mc(leg_a, leg_b, T, rho, r_d, n_paths=2_000_000, seed=99)

print(f"  single-leg-A EKO price                   = {single_leg_a:.8f}")
print(f"  worst-of CF (leg-B dominates intrinsic)  = {cf['price']:.8f}")
print(f"  worst-of MC (validation)                 = {mc['price']:.8f} ± {1.96*mc['std_err']:.8f}")
diff_cf = abs(cf["price"] - single_leg_a)
ok2_cf = diff_cf < 1e-5
ok2_mc = mc["ci_95_lower"] <= single_leg_a <= mc["ci_95_upper"]
print(f"  |CF - single_leg_A| = {diff_cf:.2e}   ({'OK' if ok2_cf else 'FAIL'})")
print(f"  single_leg_A in MC CI: {ok2_mc}")


# =============================================================================
# Test 3: Rho monotonicity (for two UO calls with similar setups)
# =============================================================================
_hdr("Test 3 — rho monotonicity for two UO calls")

leg_a = WorstOfLeg(S=1.0, K=1.0, H=1.04,
                   sigma=0.10, r_d=0.045, r_f=0.045,
                   opt="call", bar_dir="up_and_out")
leg_b = WorstOfLeg(S=1.0, K=1.0, H=1.04,
                   sigma=0.10, r_d=0.045, r_f=0.045,
                   opt="call", bar_dir="up_and_out")
T = 1 / 12
r_d = 0.045
rhos = [-0.9, -0.5, -0.2, 0.0, 0.2, 0.5, 0.9, 1.0]

cf_prices = []
mc_results = []
for rho in rhos:
    cf = worstof_eko_price_cf(leg_a, leg_b, T, rho, r_d, n_quad=120)
    mc = worstof_eko_price_mc(leg_a, leg_b, T, rho, r_d,
                              n_paths=1_000_000, seed=11)
    cf_prices.append(cf["price"])
    mc_results.append(mc)
    in_ci = mc["ci_95_lower"] <= cf["price"] <= mc["ci_95_upper"]
    flag = "OK" if in_ci else "OUTSIDE CI"
    print(f"  rho={rho:+.2f}   CF={cf['price']:.8f}   "
          f"MC={mc['price']:.8f} ± {1.96*mc['std_err']:.8f}   "
          f"p_alive_joint(CF)={cf['p_alive_joint']:.4f}   {flag}")

# All CF prices should be in MC 95% CI
ok3_mc = all(mc["ci_95_lower"] <= p <= mc["ci_95_upper"]
              for p, mc in zip(cf_prices, mc_results))

# Single-leg reference at rho = 1
single_leg_ref = ko_price("call", "up_and_out", 1.0, 1.0, 1.04, T,
                           0.10, r_d, 0.045)
print(f"\n  rho=1.0 should match single-leg EKO ({single_leg_ref:.8f}):")
print(f"     CF[rho=1] = {cf_prices[-1]:.8f}    "
       f"|diff| = {abs(cf_prices[-1] - single_leg_ref):.2e}")
ok3a = abs(cf_prices[-1] - single_leg_ref) < 1e-6

# Monotonicity check
diffs = np.diff(cf_prices)
ok3b = bool(np.all(diffs >= -1e-10))
print(f"  Monotonic non-decreasing in rho: {ok3b}    "
       f"min(diff)={float(diffs.min()):.2e}")


# =============================================================================
# Test 4: Coverage — barrier direction combinations
# =============================================================================
_hdr("Test 4 — coverage of barrier/option combinations vs MC")

T = 2 / 12
r_d = 0.045
rho = 0.30
N_MC = 2_000_000

# (leg_a opt, bar_dir, K, H), (leg_b opt, bar_dir, K, H)
cases = [
    ("UO call × UO call",
     ("call", "up_and_out", 1.00, 1.05),
     ("call", "up_and_out", 1.00, 1.05)),
    ("DO put × DO put",
     ("put", "down_and_out", 1.00, 0.95),
     ("put", "down_and_out", 1.00, 0.95)),
    ("UO call × DO put (cross)",
     ("call", "up_and_out", 1.00, 1.05),
     ("put",  "down_and_out", 1.00, 0.95)),
    ("UO put × UO put (barrier non-binding for OTM put)",
     ("put", "up_and_out", 0.98, 1.06),
     ("put", "up_and_out", 0.98, 1.06)),
    ("DO call × DO call (barrier non-binding for OTM call)",
     ("call", "down_and_out", 1.02, 0.95),
     ("call", "down_and_out", 1.02, 0.95)),
    ("UO call × no-barrier vanilla call",
     ("call", "up_and_out", 1.00, 1.05),
     ("call", "none", 1.00, 1e6)),
]

all_ok_test4 = True
for name, (oa, ba, Ka, Ha), (ob, bb, Kb, Hb) in cases:
    leg_a = WorstOfLeg(S=1.0, K=Ka, H=Ha, sigma=0.10, r_d=0.045, r_f=0.045, opt=oa, bar_dir=ba)
    leg_b = WorstOfLeg(S=1.0, K=Kb, H=Hb, sigma=0.10, r_d=0.045, r_f=0.045, opt=ob, bar_dir=bb)
    cf = worstof_eko_price_cf(leg_a, leg_b, T, rho, r_d, n_quad=120)
    mc = worstof_eko_price_mc(leg_a, leg_b, T, rho, r_d, n_paths=N_MC, seed=7)
    ok = _show(name, cf, mc)
    if not ok:
        all_ok_test4 = False


# =============================================================================
# Summary
# =============================================================================
_hdr("Summary")
print(f"  Test 1 (headline UO×UO):              {'PASS' if ok1 else 'FAIL'}")
print(f"  Test 2 (single-leg reduction):        "
      f"{'PASS' if (ok2_cf and ok2_mc) else 'FAIL'}")
print(f"  Test 3a (rho=1 matches single-leg):   {'PASS' if ok3a else 'FAIL'}")
print(f"  Test 3b (monotonic in rho):           {'PASS' if ok3b else 'FAIL'}")
print(f"  Test 3c (CF in MC CI across rho):     {'PASS' if ok3_mc else 'FAIL'}")
print(f"  Test 4 (barrier coverage):            "
      f"{'PASS' if all_ok_test4 else 'FAIL'}")

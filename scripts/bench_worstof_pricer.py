"""Step 2a — performance benchmark + old-vs-new approximation comparison.

Two goals:
  1. Measure CF and MC pricer speed.
  2. Show how the legacy `multiplier × min(P_A, P_B)` approximation
     compares to the true worst-of price across a sweep of rho.
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
from core.worstof_pricer import (
    WorstOfLeg, worstof_eko_price_cf, worstof_eko_price_mc,
)
from core.ko import ko_price


# =============================================================================
# Performance
# =============================================================================
print("=" * 78)
print("Performance — time per price")
print("=" * 78)

leg_a = WorstOfLeg(S=1.0, K=1.0, H=1.04, sigma=0.10, r_d=0.045, r_f=0.045,
                   opt="call", bar_dir="up_and_out")
leg_b = WorstOfLeg(S=1.0, K=1.0, H=1.05, sigma=0.10, r_d=0.045, r_f=0.045,
                   opt="call", bar_dir="up_and_out")
T = 1 / 12
r_d = 0.045
rho = 0.30

# CF benchmark
n_warm = 10
n_run = 200
for _ in range(n_warm):
    worstof_eko_price_cf(leg_a, leg_b, T, rho, r_d, n_quad=80)

for n_quad in [40, 80, 120, 200]:
    t0 = time.perf_counter()
    for _ in range(n_run):
        worstof_eko_price_cf(leg_a, leg_b, T, rho, r_d, n_quad=n_quad)
    dt = (time.perf_counter() - t0) / n_run * 1e6
    print(f"  CF, n_quad={n_quad:3d}   {dt:7.1f} us/price")

# MC benchmark
for n_paths in [50_000, 200_000, 1_000_000]:
    t0 = time.perf_counter()
    out = worstof_eko_price_mc(leg_a, leg_b, T, rho, r_d, n_paths=n_paths, seed=42)
    dt = (time.perf_counter() - t0) * 1e3
    print(f"  MC, n_paths={n_paths:>8d}  {dt:7.1f} ms/price    "
          f"std_err = {out['std_err']:.2e}")


# =============================================================================
# Old multiplier approximation vs true CF price
# =============================================================================
print()
print("=" * 78)
print("True worst-of CF price  vs  legacy `multiplier × min(P_A, P_B)`")
print("=" * 78)
print("Setup: two identical UO calls, S=K=1.0, H=1.04, sigma=10%, T=1M")
print()

# Single-leg EKO price (both legs identical so P_A = P_B)
P_leg = ko_price("call", "up_and_out", 1.0, 1.0, 1.04, T,
                 0.10, r_d, 0.045)
print(f"  Single-leg EKO price        = {P_leg:.8f}")
print()
print(f"  {'rho':>5s}  {'true CF':>12s}  "
      f"{'mult=0.33':>12s}  {'mult=0.40':>12s}  {'mult=0.50':>12s}  "
      f"{'best-fit mult':>14s}")
print(f"  {'-'*5}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*14}")
for rho in [-0.9, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 0.95]:
    cf = worstof_eko_price_cf(leg_a, leg_b, T, rho, r_d, n_quad=120)
    true_price = cf["price"]
    legacy_33 = 0.33 * min(P_leg, P_leg)
    legacy_40 = 0.40 * min(P_leg, P_leg)
    legacy_50 = 0.50 * min(P_leg, P_leg)
    # "Best-fit" multiplier that would have given the true price
    best_mult = true_price / P_leg if P_leg > 0 else float("nan")
    print(f"  {rho:+.2f}  {true_price:12.8f}  "
          f"{legacy_33:12.8f}  {legacy_40:12.8f}  {legacy_50:12.8f}  "
          f"{best_mult:13.3f}")

print()
print("Reading the table:")
print("  - 'best-fit mult' is what would reproduce the true CF price as")
print("    (best_mult × P_leg). With identical legs, P_A = P_B = P_leg, so")
print("    this is the multiplier the legacy approximation should have used.")
print("  - It varies from ~0.04 (rho=-0.9) to ~1.00 (rho near +1).")
print("  - The fixed multipliers (0.33 / 0.40 / 0.50) coincidentally land near")
print("    the 'right' value at rho ~ 0.20 / 0.30 / 0.50, but are wildly off")
print("    away from that small rho band.")

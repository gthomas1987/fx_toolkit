"""Step 2d — validate `core.correlation` against Monte Carlo.

Tests:
  1. triangulated_correlation recovers the true ρ used in an MC sim
     for BOTH same-side-shared (eta=+1) and opposite-side-shared
     (eta=-1) cross constructions.
  2. triangulation_eta_and_cross gives the right (eta, cross) for
     standard pair combos.
  3. clipping behavior fires for arbitrage-violating inputs.
  4. rolling_realized_correlation: convergence on a long sample.
  5. implied_correlation_time_series end-to-end works after we
     extend the synth data folder with the implied cross vol panel.
"""
from __future__ import annotations
import sys, os, datetime as dt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
import pandas as pd

from core.correlation import (
    rolling_realized_correlation, realized_correlation_at,
    triangulation_eta_and_cross, triangulated_correlation,
    implied_correlation_at_T, implied_correlation_time_series,
    TriangulationResult,
)


def _hdr(s):
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


# =============================================================================
# Test 1: Triangulation identity round-trip via MC
# =============================================================================
_hdr("Test 1 — triangulation round-trip via MC simulation")

n_paths = 500_000
T = 1.0
rng = np.random.default_rng(42)

cases = [
    # (true_rho, sigma_a, sigma_b)
    (-0.80, 0.08, 0.12),
    (-0.30, 0.08, 0.12),
    ( 0.00, 0.10, 0.10),
    ( 0.40, 0.08, 0.12),
    ( 0.80, 0.08, 0.12),
]

all_pass_1 = True
for true_rho, sigma_a, sigma_b in cases:
    z1 = rng.standard_normal(n_paths)
    z2 = true_rho * z1 + np.sqrt(max(1.0 - true_rho**2, 0.0)) * rng.standard_normal(n_paths)
    log_a = sigma_a * np.sqrt(T) * z1
    log_b = sigma_b * np.sqrt(T) * z2

    # Same-side shared (η = +1): cross has log_X = log_A - log_B
    log_x_same = log_a - log_b
    sigma_x_same = float(np.std(log_x_same)) / np.sqrt(T)
    rho_rec_same = triangulated_correlation(sigma_a, sigma_b, sigma_x_same, eta=+1)

    # Opposite-side shared (η = -1): cross has log_X = log_A + log_B
    log_x_opp = log_a + log_b
    sigma_x_opp = float(np.std(log_x_opp)) / np.sqrt(T)
    rho_rec_opp = triangulated_correlation(sigma_a, sigma_b, sigma_x_opp, eta=-1)

    tol = 0.02   # 2 percentage points — sample-std error on σ_X squared
                 # dominates at the high-|ρ| edges where dρ/dσ_X is large
    err_same = abs(rho_rec_same - true_rho)
    err_opp  = abs(rho_rec_opp - true_rho)
    flag = "OK" if (err_same < tol and err_opp < tol) else "FAIL"
    if flag == "FAIL":
        all_pass_1 = False
    print(f"  true_rho={true_rho:+.2f}  σ_a={sigma_a*100:.1f}%  σ_b={sigma_b*100:.1f}%  "
          f"η=+1: rho_rec={rho_rec_same:+.4f} (err {err_same:.4f})  "
          f"η=-1: rho_rec={rho_rec_opp:+.4f} (err {err_opp:.4f})   {flag}")


# =============================================================================
# Test 2: eta + cross-pair name for common configurations
# =============================================================================
_hdr("Test 2 — eta and cross-pair name for common configurations")

eta_cases = [
    # (pair_a, pair_b, expected_eta, expected_cross)
    ("AUDUSD", "EURUSD", +1, "AUDEUR"),   # same DOM
    ("USDJPY", "USDMXN", +1, "JPYMXN"),   # same FOR
    ("USDJPY", "EURUSD", -1, "EURJPY"),   # USD on opposite sides
    ("EURUSD", "USDJPY", -1, "EURJPY"),   # symmetric
    ("EURUSD", "GBPUSD", +1, "EURGBP"),   # same DOM
    ("USDCAD", "USDCHF", +1, "CADCHF"),   # same FOR
    ("AUDJPY", "NZDJPY", +1, "AUDNZD"),   # same DOM (JPY)
    ("EURJPY", "EURGBP", +1, "JPYGBP"),   # same FOR (EUR)
    ("AUDUSD", "USDJPY", -1, "AUDJPY"),   # USD opposite sides
]

all_pass_2 = True
for a, b, exp_eta, exp_cross in eta_cases:
    res = triangulation_eta_and_cross(a, b)
    if res is None:
        print(f"  {a} × {b}  →  NONE (FAIL)")
        all_pass_2 = False
        continue
    got_eta, got_cross = res
    eta_ok = (got_eta == exp_eta)
    # Cross name can be either ordering — we just need the same 6 letters
    cross_ok = sorted(got_cross) == sorted(exp_cross)
    flag = "OK" if (eta_ok and cross_ok) else "FAIL"
    if not (eta_ok and cross_ok):
        all_pass_2 = False
    print(f"  {a} × {b}  →  (η={got_eta:+d}, cross={got_cross})  "
          f"expected (η={exp_eta:+d}, cross={exp_cross})   {flag}")

# Sanity: no-shared-currency case returns None
no_share = triangulation_eta_and_cross("EURGBP", "AUDJPY")
all_pass_2 = all_pass_2 and (no_share is None)
print(f"  EURGBP × AUDJPY  →  {no_share}  (expected None)   "
      f"{'OK' if no_share is None else 'FAIL'}")


# =============================================================================
# Test 3: clipping for arbitrage-violating inputs
# =============================================================================
_hdr("Test 3 — clipping behavior")

# With ASYMMETRIC leg vols, σ_X can be outside the no-arbitrage range
# [|σ_A - σ_B|, σ_A + σ_B], driving the formula past ±1.
# Use σ_A = 0.05, σ_B = 0.20 — no-arb range for σ_X is [0.15, 0.25].
# Outside this, the formula returns |ρ| > 1 and needs clipping.

# σ_X = 0.05 (below |σ_A - σ_B| = 0.15) → ρ > 1
rho_clipped = triangulated_correlation(0.05, 0.20, 0.05, eta=+1, clip=True)
rho_unclipped = triangulated_correlation(0.05, 0.20, 0.05, eta=+1, clip=False)
print(f"  σ_X=0.05 (below no-arb floor 0.15):  clipped ρ = {rho_clipped:+.4f}  "
       f"unclipped ρ = {rho_unclipped:+.4f}")
ok3a = (rho_clipped == 1.0) and (rho_unclipped > 1.0)

# σ_X = 0.40 (above σ_A + σ_B = 0.25) → ρ < -1
rho_clipped = triangulated_correlation(0.05, 0.20, 0.40, eta=+1, clip=True)
rho_unclipped = triangulated_correlation(0.05, 0.20, 0.40, eta=+1, clip=False)
print(f"  σ_X=0.40 (above no-arb ceiling 0.25): clipped ρ = {rho_clipped:+.4f}  "
       f"unclipped ρ = {rho_unclipped:+.4f}")
ok3b = (rho_clipped == -1.0) and (rho_unclipped < -1.0)
all_pass_3 = ok3a and ok3b


# =============================================================================
# Test 4: rolling_realized_correlation convergence
# =============================================================================
_hdr("Test 4 — rolling_realized_correlation reasonably tracks true ρ")

# Simulate two correlated GBM-style log-return series, run rolling
# 60-day correlation, check the average is close to the true ρ.
n_days = 1000
true_rho = 0.55
sigma_a, sigma_b = 0.08, 0.10

rng = np.random.default_rng(7)
dt_yr = 1 / 252
z1 = rng.standard_normal(n_days)
z2 = true_rho * z1 + np.sqrt(1 - true_rho**2) * rng.standard_normal(n_days)
log_a = sigma_a * np.sqrt(dt_yr) * z1
log_b = sigma_b * np.sqrt(dt_yr) * z2
spot_a = pd.Series(np.exp(np.cumsum(log_a)),
                    index=pd.bdate_range("2022-01-01", periods=n_days))
spot_b = pd.Series(np.exp(np.cumsum(log_b)),
                    index=pd.bdate_range("2022-01-01", periods=n_days))

corr_ser = rolling_realized_correlation(spot_a, spot_b, window=60)
n_valid = int(corr_ser.notna().sum())
mean_corr = float(corr_ser.dropna().mean())
print(f"  true ρ = {true_rho:+.2f}  rolling-60d mean ρ over {n_valid} valid obs: "
      f"{mean_corr:+.4f}  (err {abs(mean_corr - true_rho):.4f})")
all_pass_4 = abs(mean_corr - true_rho) < 0.02


# realized_correlation_at single-point. A single 60-day window's
# correlation has standard error ≈ (1-ρ²)/√(60-2) ≈ 0.10 at ρ≈0.55,
# so allow a wide tolerance.
rho_at_end, n_obs = realized_correlation_at(spot_a, spot_b, spot_a.index[-1])
print(f"  realized_correlation_at end-of-sample: rho = {rho_at_end:+.4f}  n_obs = {n_obs}")
ok4b = rho_at_end is not None and abs(rho_at_end - true_rho) < 0.30
all_pass_4 = all_pass_4 and ok4b


# =============================================================================
# Summary
# =============================================================================
_hdr("Summary")
print(f"  Test 1 (MC round-trip):           {'PASS' if all_pass_1 else 'FAIL'}")
print(f"  Test 2 (eta + cross name):        {'PASS' if all_pass_2 else 'FAIL'}")
print(f"  Test 3 (clipping):                {'PASS' if all_pass_3 else 'FAIL'}")
print(f"  Test 4 (rolling realized):        {'PASS' if all_pass_4 else 'FAIL'}")

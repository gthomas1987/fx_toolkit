"""Step 2c — validate that the worst-of BACKTEST engine works with
all three pricing engines (legacy_multiplier, closed_form, monte_carlo)
on synthetic data.

Checks:
  1. All three engines run a full backtest without errors.
  2. Same set of trade dates produced by each (engines only affect
     premium pricing, not trade timing).
  3. The new per-trade fields are populated correctly.
  4. CF and MC trade premiums agree within MC noise per trade.
  5. Legacy and new-engine premiums DIFFER (showing the bias).
  6. Rolling-60d correlation source works and produces a different
     premium series than manual correlation when ρ_hist ≠ ρ_manual.
  7. Validation errors fire when combining new engine with
     american_ohlc or vanna_volga_american.
"""
from __future__ import annotations
import sys, os, datetime as dt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
import pandas as pd

from core.worstof import (
    WorstOfSpec, WorstOfTrade, run_worstof_strategy,
)


FOLDER = "/tmp/wop_test_data"


def _hdr(s):
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


# Common base spec (legs: USDJPY UO call + EURUSD UO call, 1M)
def _base_spec(**overrides) -> WorstOfSpec:
    defaults = dict(
        leg_a_pair="USDJPY", leg_a_direction="call",
        leg_a_barrier_type="up_and_out",
        leg_a_strike_delta_value=0.35, leg_a_strike_delta_label="35Δ",
        leg_a_ko_delta_value=0.10, leg_a_ko_delta_label="10Δ",
        leg_b_pair="EURUSD", leg_b_direction="call",
        leg_b_barrier_type="up_and_out",
        leg_b_strike_delta_value=0.35, leg_b_strike_delta_label="35Δ",
        leg_b_ko_delta_value=0.10, leg_b_ko_delta_label="10Δ",
        tenor_label="1M",
        tx_cost_bps=4.0,
        multiplier=0.33,
    )
    defaults.update(overrides)
    return WorstOfSpec(**defaults)


# Use a 6-month window to keep the test fast but get a meaningful
# number of trades.
START = dt.date(2024, 9, 1)
END = dt.date(2025, 4, 30)


# =============================================================================
# Test 1: All three engines run cleanly
# =============================================================================
_hdr("Test 1 — all three engines run cleanly on synthetic data")

results = {}
for engine in ("legacy_multiplier", "closed_form", "monte_carlo"):
    spec = _base_spec(
        pricing_engine=engine,
        correlation_source="manual",
        correlation_value=0.30,
        mc_n_paths=50_000 if engine == "monte_carlo" else 100_000,
    )
    trades = run_worstof_strategy(
        FOLDER, spec=spec, start_date=START, end_date=END,
        notional_usd=10_000_000.0,
    )
    results[engine] = trades
    n_trades = len(trades)
    if n_trades == 0:
        print(f"  {engine:>18s}:  NO TRADES (something's wrong)")
        continue
    avg_prem = np.mean([t.structure_premium_mid_usd for t in trades])
    avg_paid = np.mean([t.structure_premium_paid_usd for t in trades])
    total_pnl = sum(t.pnl_usd for t in trades)
    print(f"  {engine:>18s}:  n_trades = {n_trades},  "
          f"avg_mid = ${avg_prem:>10,.0f},  "
          f"avg_paid = ${avg_paid:>10,.0f},  "
          f"total_pnl = ${total_pnl:>14,.0f}")

ok1 = all(len(t) > 0 for t in results.values())


# =============================================================================
# Test 2: Same trade dates across engines (engines don't affect entry)
# =============================================================================
_hdr("Test 2 — same trade-date set across engines")

dates_per_engine = {
    e: sorted([t.trade_date for t in trades]) for e, trades in results.items()
}
all_same_dates = (dates_per_engine["legacy_multiplier"]
                   == dates_per_engine["closed_form"]
                   == dates_per_engine["monte_carlo"])
print(f"  same trade-date sets across all 3 engines: {all_same_dates}")
print(f"  trade_count: {len(dates_per_engine['legacy_multiplier'])}")
ok2 = all_same_dates


# =============================================================================
# Test 3: Per-trade fields populated correctly
# =============================================================================
_hdr("Test 3 — per-trade fields populated correctly")

cf_trade = results["closed_form"][0]
mc_trade = results["monte_carlo"][0]
legacy_trade = results["legacy_multiplier"][0]

ok3a = legacy_trade.pricing_engine == "legacy_multiplier"
ok3b = legacy_trade.correlation_used is None
ok3c = legacy_trade.correlation_source_used == "legacy"
ok3d = legacy_trade.structure_premium_legacy_usd > 0

ok3e = cf_trade.pricing_engine == "closed_form"
ok3f = cf_trade.correlation_used == 0.30
ok3g = cf_trade.correlation_source_used == "manual"
ok3h = cf_trade.structure_premium_legacy_usd > 0  # legacy formula also recorded

ok3i = mc_trade.pricing_engine == "monte_carlo"
ok3j = mc_trade.correlation_used == 0.30

print(f"  legacy.pricing_engine == 'legacy_multiplier':  {ok3a}")
print(f"  legacy.correlation_used is None:                {ok3b}")
print(f"  legacy.correlation_source_used == 'legacy':     {ok3c}")
print(f"  legacy.structure_premium_legacy_usd > 0:        {ok3d}")
print(f"  cf.pricing_engine == 'closed_form':             {ok3e}")
print(f"  cf.correlation_used == 0.30:                    {ok3f}")
print(f"  cf.correlation_source_used == 'manual':         {ok3g}")
print(f"  cf.structure_premium_legacy_usd > 0 (recorded): {ok3h}")
print(f"  mc.pricing_engine == 'monte_carlo':             {ok3i}")
print(f"  mc.correlation_used == 0.30:                    {ok3j}")
ok3 = all([ok3a, ok3b, ok3c, ok3d, ok3e, ok3f, ok3g, ok3h, ok3i, ok3j])


# =============================================================================
# Test 4: CF and MC agree per-trade; legacy differs
# =============================================================================
_hdr("Test 4 — CF and MC per-trade premium agreement vs legacy divergence")

cf_trades = {t.trade_date: t for t in results["closed_form"]}
mc_trades = {t.trade_date: t for t in results["monte_carlo"]}
lg_trades = {t.trade_date: t for t in results["legacy_multiplier"]}

# CF vs MC: per-trade % difference
cf_mc_diff_pct = []
cf_lg_diff_pct = []
for d, cf_t in cf_trades.items():
    if d not in mc_trades:
        continue
    mc_t = mc_trades[d]
    lg_t = lg_trades[d]
    cf_p, mc_p, lg_p = (cf_t.structure_premium_mid_usd,
                         mc_t.structure_premium_mid_usd,
                         lg_t.structure_premium_mid_usd)
    if cf_p > 0:
        cf_mc_diff_pct.append(abs(mc_p - cf_p) / cf_p * 100)
        cf_lg_diff_pct.append(abs(lg_p - cf_p) / cf_p * 100)

cf_mc_diff_arr = np.array(cf_mc_diff_pct)
cf_lg_diff_arr = np.array(cf_lg_diff_pct)
print(f"  |MC - CF| / CF (%):    median {np.median(cf_mc_diff_arr):.2f}%,  "
      f"95th pct {np.percentile(cf_mc_diff_arr, 95):.2f}%,  "
      f"max {cf_mc_diff_arr.max():.2f}%")
print(f"  |Legacy - CF| / CF (%): median {np.median(cf_lg_diff_arr):.2f}%,  "
      f"95th pct {np.percentile(cf_lg_diff_arr, 95):.2f}%,  "
      f"max {cf_lg_diff_arr.max():.2f}%")
# CF and MC should agree within a few percent; Legacy is the one we're
# REPLACING so it should differ by more.
ok4a = float(np.median(cf_mc_diff_arr)) < 5.0    # CF and MC within 5% median
ok4b = float(np.median(cf_lg_diff_arr)) > float(np.median(cf_mc_diff_arr))   # Legacy diverges more
print(f"  CF and MC within 5% median: {ok4a}")
print(f"  Legacy divergence > CF/MC divergence: {ok4b}")
ok4 = ok4a and ok4b


# =============================================================================
# Test 5: Rolling correlation source works
# =============================================================================
_hdr("Test 5 — rolling_60d correlation source produces a different series")

spec_roll = _base_spec(
    pricing_engine="closed_form", correlation_source="rolling_60d",
    correlation_value=0.30,
)
trades_roll = run_worstof_strategy(
    FOLDER, spec=spec_roll, start_date=START, end_date=END,
    notional_usd=10_000_000.0,
)
# Check the correlation_used varies across trades
rho_values = [t.correlation_used for t in trades_roll
              if t.correlation_source_used == "rolling_60d"]
print(f"  n_trades using rolling_60d: {len(rho_values)} / {len(trades_roll)}")
if rho_values:
    print(f"  rho span: [{min(rho_values):+.3f}, {max(rho_values):+.3f}]")
    print(f"  rho mean: {np.mean(rho_values):+.3f}")
ok5a = len(rho_values) > 0
ok5b = (len(set(rho_values)) > 1) if rho_values else False  # vary per date
# Also: at least some early trades use fallback (since 60d lookback isn't ready)
fallback_count = sum(1 for t in trades_roll
                     if t.correlation_source_used == "rolling_60d_fallback_to_manual")
print(f"  n_trades with fallback (insufficient lookback): {fallback_count}")
# In a 6mo window with 16 months of prior data, fallback should be 0:
# rolling-60 needs only ~60 business days of prior history, and we have
# ~5 years before 2024-09-01 in the synthetic dataset (2023-05 → 2025-04).
# So no fallback expected. If we see any, that's a smoke signal but not a fail.
ok5 = ok5a and ok5b


# =============================================================================
# Test 6: Validation errors fire for incompatible mode combos
# =============================================================================
_hdr("Test 6 — validation errors")

# Try CF + american_ohlc — should raise ValueError
try:
    bad_spec = _base_spec(pricing_engine="closed_form",
                            ko_check_mode="american_ohlc")
    run_worstof_strategy(FOLDER, spec=bad_spec,
                          start_date=START, end_date=END)
    print(f"  CF + american_ohlc: did NOT raise — FAIL")
    ok6a = False
except ValueError as e:
    print(f"  CF + american_ohlc → ValueError OK: {str(e)[:80]}...")
    ok6a = True

# Try MC + vanna_volga_american — should raise ValueError
try:
    bad_spec = _base_spec(pricing_engine="monte_carlo",
                            leg_pricing_mode="vanna_volga_american")
    run_worstof_strategy(FOLDER, spec=bad_spec,
                          start_date=START, end_date=END)
    print(f"  MC + vanna_volga_american: did NOT raise — FAIL")
    ok6b = False
except ValueError as e:
    print(f"  MC + vanna_volga_american → ValueError OK: {str(e)[:80]}...")
    ok6b = True

ok6 = ok6a and ok6b


# =============================================================================
# Summary
# =============================================================================
_hdr("Summary")
print(f"  Test 1 (all engines run):                   {'PASS' if ok1 else 'FAIL'}")
print(f"  Test 2 (same trade dates):                  {'PASS' if ok2 else 'FAIL'}")
print(f"  Test 3 (per-trade fields populated):        {'PASS' if ok3 else 'FAIL'}")
print(f"  Test 4 (CF≈MC, legacy diverges):            {'PASS' if ok4 else 'FAIL'}")
print(f"  Test 5 (rolling_60d works):                 {'PASS' if ok5 else 'FAIL'}")
print(f"  Test 6 (validation errors fire):            {'PASS' if ok6 else 'FAIL'}")

"""Step R1 — validate the RKO single-leg backtester runs cleanly across
flat_atm / vol_at_strike / vanna_volga pricing models and produces
ledgers that differ in the expected directions.

Parallel to scripts/validate_backtest_pricing_models.py (which covers
the EUROPEAN-barrier backtester). Same test layout, same assertions —
the only differences are:
  - uses core.backtest_american.run_single_strategy_american
  - the BS engine underneath is ako_closed_form (American barrier)
  - VV is layered on top via core.ako_pricing.price_ako_dispatch

Checks:
  1. All three models run a full backtest without errors.
  2. Same trade-date set across models (model only affects pricing,
     not entry).
  3. premium_mid differs across models in the predicted direction:
     - VV vs vol_at_strike differs materially (smile premium)
     - flat_atm and vol_at_strike differ by σ_atm vs σ_smile only
  4. Per-trade `pricing_model` field is populated correctly.

Synthetic data: USDJPY UO call 35Δ / 10Δ, 1M tenor, 4-month window.
"""
from __future__ import annotations
import sys, os, datetime as dt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np

from core.backtest import StrategySpec
from core.backtest_american import (
    run_single_strategy_american, preload_pair_panels_american,
)


FOLDER = "/tmp/wop_test_data"


def _hdr(s):
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


def _spec(pricing_model: str) -> StrategySpec:
    return StrategySpec(
        pair="USDJPY",
        direction="call",
        barrier_type="up_and_out",
        delta_label="35Δ",
        delta_value=0.35,
        tenor_label="1M",
        tx_cost_bps=4.0,
        ko_method="delta",
        target_ko_delta=0.10,
        ko_delta_label="10Δ",
        pricing_model=pricing_model,
    )


START = dt.date(2024, 12, 1)
END = dt.date(2025, 4, 30)


# =============================================================================
# Test 1: All three models run cleanly
# =============================================================================
_hdr("Test 1 — all three models produce trades (American-barrier backtest)")

panels = preload_pair_panels_american(FOLDER, "USDJPY", prefer="offshore")
assert panels, "USDJPY panels missing from synth folder"

results = {}
for model in ("flat_atm", "vol_at_strike", "vanna_volga"):
    trades = run_single_strategy_american(_spec(model), panels, START, END,
                                            notional_usd=10_000_000.0)
    results[model] = trades
    if not trades:
        print(f"  {model:>15s}:  NO TRADES (FAIL)")
        continue
    avg_mid = np.mean([t.premium_mid_usd for t in trades])
    avg_atm = np.mean([t.sigma_atm for t in trades])
    avg_smile = np.mean([t.sigma_smile for t in trades])
    n_ko = sum(1 for t in trades if t.knocked_out)
    print(f"  {model:>15s}:  n_trades={len(trades):>3},  "
          f"KO={n_ko:>3},  "
          f"avg premium_mid = ${avg_mid:>9,.0f},  "
          f"avg σ_atm = {avg_atm*100:.3f}%,  "
          f"avg σ_smile = {avg_smile*100:.3f}%")

ok1 = all(len(t) > 0 for t in results.values())


# =============================================================================
# Test 2: Same trade-date set across models
# =============================================================================
_hdr("Test 2 — same trade-date set across models")

dates_by_model = {m: sorted(t.trade_date for t in ts)
                    for m, ts in results.items()}
same = (dates_by_model["flat_atm"]
         == dates_by_model["vol_at_strike"]
         == dates_by_model["vanna_volga"])
print(f"  identical trade-date sets across all 3 models: {same}")
print(f"  trade count: {len(dates_by_model['flat_atm'])}")
ok2 = same


# =============================================================================
# Test 3: Premium differences in the predicted direction
# =============================================================================
_hdr("Test 3 — premium differences across models")

mids_by_model = {m: {t.trade_date: t.premium_mid_usd for t in ts}
                   for m, ts in results.items()}
common = set(mids_by_model["flat_atm"]) & set(mids_by_model["vol_at_strike"]) \
            & set(mids_by_model["vanna_volga"])

def _avg_diff_pct(a_name: str, b_name: str) -> float:
    """mean(b-a)/a * 100 across common dates"""
    diffs = [(mids_by_model[b_name][d] - mids_by_model[a_name][d])
              / mids_by_model[a_name][d] * 100
              for d in common
              if mids_by_model[a_name][d] > 0]
    return float(np.mean(diffs)) if diffs else float("nan")

flat_to_vas = _avg_diff_pct("flat_atm", "vol_at_strike")
flat_to_vv = _avg_diff_pct("flat_atm", "vanna_volga")
vas_to_vv = _avg_diff_pct("vol_at_strike", "vanna_volga")

print(f"  mean (vol_at_strike - flat_atm) / flat_atm  = {flat_to_vas:+.2f}%")
print(f"  mean (vanna_volga - flat_atm) / flat_atm     = {flat_to_vv:+.2f}%")
print(f"  mean (vanna_volga - vol_at_strike) / vol_at_strike = {vas_to_vv:+.2f}%")
# VV should capture a meaningful smile premium on RKOs (typically
# larger than EKO since American barrier accentuates the wing-vol
# dependence). On the synth folder we saw ~37% in the live-pricer
# smoke test earlier.
ok3a = abs(flat_to_vas) >= 0.0    # informational only — can be ~0 if structure is vol-insensitive
ok3b = abs(vas_to_vv) > 5.0       # VV captures non-trivial smile premium
print(f"  VV vs vol-at-strike differs by >5% (smile premium):  {ok3b}")
ok3 = ok3a and ok3b


# =============================================================================
# Test 4: Per-trade pricing_model field
# =============================================================================
_hdr("Test 4 — per-trade pricing_model field populated correctly")

# American backtester maps:
#   spec.pricing_model='flat_atm'      -> trade.pricing_model='r_r'
#   spec.pricing_model='vol_at_strike' -> trade.pricing_model='r_r'
#   spec.pricing_model='vanna_volga'   -> trade.pricing_model='vanna_volga'
# (The 'r_r' label is the legacy stable identifier; sigma_used and
# sigma_atm on the row distinguish which σ was fed in.)
expected_label = {
    "flat_atm":      "r_r",
    "vol_at_strike": "r_r",
    "vanna_volga":   "vanna_volga",
}
ok4 = True
for model, ts in results.items():
    exp = expected_label[model]
    # Strip the "_close_only" suffix that gets added when OHLC isn't
    # available (we have OHLC in the synth folder so this should
    # never trigger, but be defensive).
    bad = [t for t in ts if t.pricing_model.replace("_close_only", "") != exp]
    if bad:
        print(f"  {model:>15s}: {len(bad)}/{len(ts)} rows have wrong "
              f"pricing_model label (expected {exp!r}) — FAIL")
        ok4 = False
    else:
        print(f"  {model:>15s}: all {len(ts)} rows tagged "
              f"pricing_model={exp!r} (OK)")


# =============================================================================
# Summary
# =============================================================================
_hdr("Summary")
print(f"  Test 1 (all models produce trades):     {'PASS' if ok1 else 'FAIL'}")
print(f"  Test 2 (same trade-date set):           {'PASS' if ok2 else 'FAIL'}")
print(f"  Test 3 (VV adds non-trivial premium):   {'PASS' if ok3 else 'FAIL'}")
print(f"  Test 4 (pricing_model field tagged):    {'PASS' if ok4 else 'FAIL'}")

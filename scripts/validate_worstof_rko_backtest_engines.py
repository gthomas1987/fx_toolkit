"""Validate the Worst-of backtest engine's American-barrier extension
(Step R3).

Mirrors `validate_worstof_backtest_engines.py` but exercises the two
new engines: `cf_approx_american` and `monte_carlo_american`. These
require `ko_check_mode='american_ohlc'` and produce structure premiums
materially different from the legacy multiplier formula (which is the
whole point — accurate correlation-aware pricing for RKO worst-ofs).

Checks:
  1. All four configurations run: legacy_multiplier + american_ohlc,
     cf_approx_american + american_ohlc, monte_carlo_american +
     american_ohlc, and one validation-error case.
  2. Same trade-date set across engines (engine only changes pricing,
     not entry logic).
  3. Per-trade fields populated: pricing_engine tagged correctly,
     structure_premium_legacy_usd computed alongside the engine's
     premium so users can A/B compare.
  4. CF-approx and MC agree to within 50% (the same loose tolerance
     used in the standalone pricer validation — these are tight
     synth-folder RKOs where the CF is known low-biased).
  5. Validation errors fire for incompatible configurations.

Synthetic data: USDJPY × EURUSD, 1M tenor, 5-month backtest window.
"""
from __future__ import annotations
import sys, os, datetime as dt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np

from core.worstof import WorstOfSpec, run_worstof_strategy


FOLDER = "/tmp/wop_test_data"
START = dt.date(2024, 12, 1)
END = dt.date(2025, 4, 30)


def _base(**overrides) -> WorstOfSpec:
    """Common WorstOfSpec base. Overrides fields per-test."""
    cfg = dict(
        leg_a_pair="USDJPY",
        leg_a_direction="call",
        leg_a_barrier_type="up_and_out",
        leg_a_strike_delta_label="35Δ",
        leg_a_strike_delta_value=0.35,
        leg_a_ko_delta_label="10Δ",
        leg_a_ko_delta_value=0.10,
        leg_b_pair="EURUSD",
        leg_b_direction="call",
        leg_b_barrier_type="up_and_out",
        leg_b_strike_delta_label="35Δ",
        leg_b_strike_delta_value=0.35,
        leg_b_ko_delta_label="10Δ",
        leg_b_ko_delta_value=0.10,
        tenor_label="1M",
        tx_cost_bps=4.0,
        multiplier=0.40,    # RKO default
        prefer="offshore",
        trade_mode="stack",
    )
    cfg.update(overrides)
    return WorstOfSpec(**cfg)


def _hdr(s):
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


# =============================================================================
# Test 1: All three configurations run
# =============================================================================
_hdr("Test 1 — all three American-barrier engines run")

configs = [
    ("legacy_multiplier", "american_ohlc"),
    ("cf_approx_american", "american_ohlc"),
    ("monte_carlo_american", "american_ohlc"),
]
results = {}
for engine, ko_mode in configs:
    spec = _base(
        pricing_engine=engine,
        ko_check_mode=ko_mode,
        leg_pricing_mode="european",   # required by new engines
        correlation_source="manual",
        correlation_value=0.30,
        mc_n_paths=20_000,             # smaller for test speed
    )
    trades = run_worstof_strategy(FOLDER, spec=spec,
                                     start_date=START, end_date=END)
    results[engine] = trades
    if trades:
        avg = float(np.mean([t.structure_premium_mid_usd for t in trades]))
        avg_legacy = float(np.mean(
            [t.structure_premium_legacy_usd or 0 for t in trades]
        ))
        avg_corr = float(np.mean([t.correlation_used or 0 for t in trades]))
        print(f"  {engine:>22s}: n_trades={len(trades):>3},  "
               f"avg struct_mid_usd = ${avg:>9,.0f},  "
               f"avg legacy_usd = ${avg_legacy:>9,.0f},  "
               f"avg ρ = {avg_corr:+.3f}")
    else:
        print(f"  {engine:>22s}: NO TRADES (FAIL)")

ok1 = all(len(t) > 0 for t in results.values())


# =============================================================================
# Test 2: Same trade-date set across engines
# =============================================================================
_hdr("Test 2 — same trade-date set across engines")

dates_by_engine = {e: sorted(t.trade_date for t in ts)
                    for e, ts in results.items()}
same = (dates_by_engine["legacy_multiplier"]
         == dates_by_engine["cf_approx_american"]
         == dates_by_engine["monte_carlo_american"])
print(f"  identical trade-date sets across all 3 engines: {same}")
print(f"  trade count: {len(dates_by_engine['legacy_multiplier'])}")
ok2 = same


# =============================================================================
# Test 3: Per-trade fields populated
# =============================================================================
_hdr("Test 3 — per-trade engine / correlation / legacy fields populated")

ok3 = True
for engine, ts in results.items():
    t0 = ts[0]
    print(f"  {engine}:")
    print(f"    pricing_engine          = {t0.pricing_engine!r}")
    print(f"    correlation_source_used = {t0.correlation_source_used!r}")
    print(f"    correlation_used        = {t0.correlation_used}")
    print(f"    structure_premium_legacy_usd = {t0.structure_premium_legacy_usd}")
    print(f"    structure_premium_mid_usd    = {t0.structure_premium_mid_usd}")
    if engine == "legacy_multiplier":
        # Legacy mode: pricing_engine == "legacy_multiplier", legacy
        # field may equal mid (they're the same number).
        if t0.pricing_engine != "legacy_multiplier":
            ok3 = False
    else:
        # Engine mode: pricing_engine tagged correctly, legacy_usd
        # computed alongside so users can A/B.
        if t0.pricing_engine != engine:
            ok3 = False
        if t0.structure_premium_legacy_usd is None:
            print(f"    ⚠ legacy_usd is None — should be populated for A/B")
            ok3 = False


# =============================================================================
# Test 4: CF-approx vs MC consistency
# =============================================================================
_hdr("Test 4 — CF-approx and MC give consistent premiums")

cf_premiums = {t.trade_date: t.structure_premium_mid_usd
                for t in results["cf_approx_american"]}
mc_premiums = {t.trade_date: t.structure_premium_mid_usd
                for t in results["monte_carlo_american"]}
common = sorted(set(cf_premiums) & set(mc_premiums))

errs_pct = []
for d in common:
    cf = cf_premiums[d]
    mc = mc_premiums[d]
    if mc > 0:
        errs_pct.append(abs(cf - mc) / mc * 100)

avg_cf = np.mean([cf_premiums[d] for d in common])
avg_mc = np.mean([mc_premiums[d] for d in common])
median_err = float(np.median(errs_pct)) if errs_pct else float("nan")
max_err = float(np.max(errs_pct)) if errs_pct else float("nan")
print(f"  Average CF premium:           ${avg_cf:,.0f}")
print(f"  Average MC premium:           ${avg_mc:,.0f}")
print(f"  Median |CF-MC|/MC per trade:  {median_err:.1f}%")
print(f"  Max    |CF-MC|/MC per trade:  {max_err:.1f}%")
# Same tolerance as standalone pricer validation.
ok4 = median_err < 50.0


# =============================================================================
# Test 5: Validation errors fire on incompatible configurations
# =============================================================================
_hdr("Test 5 — validation errors fire on incompatible configs")

# Cases that should raise ValueError:
bad_cases = [
    ("cf_approx_american with european_at_expiry",
     dict(pricing_engine="cf_approx_american",
          ko_check_mode="european_at_expiry",
          leg_pricing_mode="european")),
    ("monte_carlo_american with european_at_expiry",
     dict(pricing_engine="monte_carlo_american",
          ko_check_mode="european_at_expiry",
          leg_pricing_mode="european")),
    ("closed_form (European) with american_ohlc",
     dict(pricing_engine="closed_form",
          ko_check_mode="american_ohlc",
          leg_pricing_mode="european")),
]

ok5 = True
for label, cfg in bad_cases:
    spec = _base(**cfg)
    try:
        run_worstof_strategy(FOLDER, spec=spec,
                              start_date=START, end_date=END)
        print(f"  {label:>50s}: NO ERROR (FAIL — should have raised)")
        ok5 = False
    except ValueError as e:
        print(f"  {label:>50s}: ValueError ✓")
    except Exception as e:
        print(f"  {label:>50s}: Wrong error type: {type(e).__name__}: {e} (FAIL)")
        ok5 = False


# =============================================================================
# Summary
# =============================================================================
_hdr("Summary")
print(f"  Test 1 (all engines run):           {'PASS' if ok1 else 'FAIL'}")
print(f"  Test 2 (same trade dates):          {'PASS' if ok2 else 'FAIL'}")
print(f"  Test 3 (per-trade fields):          {'PASS' if ok3 else 'FAIL'}")
print(f"  Test 4 (CF ≈ MC within tolerance):  {'PASS' if ok4 else 'FAIL'}")
print(f"  Test 5 (validation errors fire):    {'PASS' if ok5 else 'FAIL'}")

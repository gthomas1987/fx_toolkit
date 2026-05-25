"""Step 2d — verify the worst-of backtest engine supports
correlation_source='triangulation'.

Synth data is configured so that the cross-vol of EURJPY is consistent
with ρ = 0.30 between USDJPY and EURUSD (with daily jitter of ~±0.05).
The backtest should produce per-trade rhos near 0.30 with std ≈ 0.05
when correlation_source='triangulation' is used.

Compares the three correlation sources on the same backtest:
  - manual = 0.30
  - rolling_60d (realized, backward-looking)
  - triangulation (implied from cross vol, forward-looking)
"""
from __future__ import annotations
import sys, os, datetime as dt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
from core.worstof import WorstOfSpec, run_worstof_strategy


FOLDER = "/tmp/wop_test_data"


def _base(**overrides):
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
        pricing_engine="closed_form",
        correlation_value=0.30,
    )
    defaults.update(overrides)
    return WorstOfSpec(**defaults)


def _hdr(s):
    print()
    print("=" * 78)
    print(s)
    print("=" * 78)


START = dt.date(2024, 9, 1)
END = dt.date(2025, 4, 30)
TRUE_RHO = 0.30   # baked into the cross vol panel


# =============================================================================
# Test 1: triangulation source runs and recovers baked-in ρ
# =============================================================================
_hdr("Test 1 — triangulation source recovers baked-in ρ in per-trade rhos")

spec_tri = _base(correlation_source="triangulation")
trades_tri = run_worstof_strategy(
    FOLDER, spec=spec_tri, start_date=START, end_date=END,
    notional_usd=10_000_000.0,
)
n = len(trades_tri)
rhos = [t.correlation_used for t in trades_tri
        if t.correlation_source_used == "triangulation"]
n_fb = sum(1 for t in trades_tri
            if t.correlation_source_used.startswith("triangulation_fallback"))

mean_rho = float(np.mean(rhos)) if rhos else float("nan")
std_rho = float(np.std(rhos)) if rhos else float("nan")
print(f"  n_trades = {n}, of which using triangulation: {len(rhos)}, "
      f"fallback: {n_fb}")
print(f"  per-trade ρ mean = {mean_rho:+.4f}  std = {std_rho:.4f}")
print(f"  baked-in true ρ = {TRUE_RHO:+.4f} (jitter std ≈ 0.05)")
ok1a = len(rhos) > 0
ok1b = abs(mean_rho - TRUE_RHO) < 0.02
ok1c = 0.03 < std_rho < 0.08          # roughly matches the synth jitter
print(f"  triangulation actually used:  {ok1a}")
print(f"  |mean ρ - true ρ| < 0.02:     {ok1b}")
print(f"  0.03 < std ρ < 0.08:          {ok1c}")
ok1 = ok1a and ok1b and ok1c


# =============================================================================
# Test 2: same trade-date set across the three sources
# =============================================================================
_hdr("Test 2 — same trade-date set across all 3 correlation sources")

spec_man = _base(correlation_source="manual")
spec_roll = _base(correlation_source="rolling_60d")
trades_man = run_worstof_strategy(FOLDER, spec=spec_man,
                                     start_date=START, end_date=END)
trades_roll = run_worstof_strategy(FOLDER, spec=spec_roll,
                                      start_date=START, end_date=END)
dates_man = sorted(t.trade_date for t in trades_man)
dates_roll = sorted(t.trade_date for t in trades_roll)
dates_tri = sorted(t.trade_date for t in trades_tri)
same = (dates_man == dates_roll == dates_tri)
print(f"  manual: {len(dates_man)}, rolling_60d: {len(dates_roll)}, "
      f"triangulation: {len(dates_tri)}")
print(f"  identical date sets: {same}")
ok2 = same


# =============================================================================
# Test 3: structure premium differs across sources (different ρs → different prices)
# =============================================================================
_hdr("Test 3 — structure premiums differ across correlation sources")

# Match by trade_date for a clean per-date comparison
def _premiums_by_date(trades):
    return {t.trade_date: t.structure_premium_mid_usd for t in trades}

p_man = _premiums_by_date(trades_man)
p_roll = _premiums_by_date(trades_roll)
p_tri = _premiums_by_date(trades_tri)
common = set(p_man) & set(p_roll) & set(p_tri)
print(f"  Avg premium by source over {len(common)} common dates:")
print(f"    manual (ρ=0.30):   ${np.mean([p_man[d] for d in common]):,.0f}")
print(f"    rolling 60d:       ${np.mean([p_roll[d] for d in common]):,.0f}")
print(f"    triangulation:     ${np.mean([p_tri[d] for d in common]):,.0f}")
# rolling_60d had different rho (mean ~+0.11 in this sample) so its
# premium differs from manual (ρ=0.30). triangulation should be
# CLOSE to manual since its mean ρ is ~0.30 by construction.
diff_man_roll = np.mean([abs(p_man[d] - p_roll[d]) for d in common])
diff_man_tri = np.mean([abs(p_man[d] - p_tri[d]) for d in common])
print(f"  Mean |manual − rolling|     = ${diff_man_roll:,.0f}")
print(f"  Mean |manual − triangulation| = ${diff_man_tri:,.0f}")
ok3a = diff_man_roll > diff_man_tri      # rolling diverges more (true rho different)
ok3b = diff_man_tri > 0                  # but tri isn't identical to manual either
print(f"  rolling diverges more from manual than tri does: {ok3a}")
print(f"  triangulation is non-trivially different from manual: {ok3b}")
ok3 = ok3a and ok3b


# =============================================================================
# Test 4: triangulation falls back gracefully when cross vol is missing
# =============================================================================
_hdr("Test 4 — triangulation falls back when no cross vol panel exists")

# NZDUSD isn't in the synth folder; the cross NZDJPY can't be read.
# Expect 0 trades. The test confirms NO CRASH on missing data.
# (We used AUDUSD here originally, but AUDUSD was later added to the
# synth folder when the Portfolio Analyzer page was ported in — needs
# a different missing pair.)
spec_bad = _base(
    leg_a_pair="NZDUSD",  # not in synth folder
    leg_b_pair="USDJPY",
    correlation_source="triangulation",
)
trades_bad = run_worstof_strategy(FOLDER, spec=spec_bad,
                                     start_date=START, end_date=END)
print(f"  NZDUSD (missing pair) × USDJPY:  n_trades = {len(trades_bad)} "
      f"(expected 0, no crash)")
ok4 = (len(trades_bad) == 0)


# =============================================================================
# Summary
# =============================================================================
_hdr("Summary")
print(f"  Test 1 (triangulation recovers baked-in ρ):  {'PASS' if ok1 else 'FAIL'}")
print(f"  Test 2 (same trade-date set):                {'PASS' if ok2 else 'FAIL'}")
print(f"  Test 3 (premiums differ across sources):     {'PASS' if ok3 else 'FAIL'}")
print(f"  Test 4 (graceful fallback on missing data):  {'PASS' if ok4 else 'FAIL'}")

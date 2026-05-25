"""EKO Pricer — European Knock-Out Option Pricer & Backtester.

Sub-app of the FX Toolkit. Reached via the landing page or sidebar
nav; not run directly. (Formerly: apps/9_ko_pricer.py.)

# Tabs
1. Pricer       — single-trade pricing + premium grid.
2. Backtest     — daily-rolling backtest of (pair × delta × tenor × direction).
3. Drilldown    — equity, drawdown, monthly/annual stats and downloadable
                  ledger for one selected strategy from the latest backtest.

# Conventions (see core/* docstrings for full detail)
- Date schedule: trade -> spot+2bd -> settlement+tenor -> expiry-2bd.
- Rates: USD/USOSFR, JPY/JYSO, EUR/EESWE, KRW/KWCDC+KWSWO, all via
  `_index.csv` lookup by Bloomberg ticker. Linear interp in T.
- Strike: vanilla Δ (closed form). Barrier: payout ratio = max_payoff /
  premium (leverage), wide-branch solution; falls back to closest
  achievable when target < ratio_min.
- Backtest PnL: % of foreign notional. European fixing only. No delta hedge.
"""
from __future__ import annotations

import io
import os
import sys
import time
from datetime import date as _date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.data_loader import (
    discovery_summary, load_panel, load_by_ticker, get_pair_value_at_T,
)
from core.ko import ko_price, ki_price, ko_probability, ko_spot_delta
from core.ko_solvers import solve_strike
from core.vanilla import vanilla_price, vanilla_spot_delta
from core.vanna_volga import vv_price_ko
from core.worstof_pricer import (
    WorstOfLeg, worstof_eko_price_cf, worstof_eko_price_mc,
)
from core.conventions import get_pip_scale
from core.calendar import compute_option_dates
from core.rates import RATE_TICKERS, load_rates_panel, get_rate_at
from core.backtest import (
    StrategySpec, build_strategy_grid, run_grid,
    trades_to_df, summarize_strategy, compute_equity_and_drawdown,
    monthly_pnl_table, annual_summary_table,
    compute_mtm_curves, summarize_mtm,
    export_strategy_time_series,
)


# =============================================================================
# Page config + styling
# =============================================================================
st.set_page_config(
    page_title="EKO Pricer & Backtester",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Card / tag styling — pulled into shared/style.py so EKO + RKO pages
# stay in sync if the dark theme is ever updated. The base CSS tightens
# typography across the toolkit.
from shared.style import inject_base_css, inject_card_css
inject_base_css()
inject_card_css()


# =============================================================================
# Shared sidebar — data folder
# Uses the same shared helper as the Vol Dashboard so the folder selection
# persists across pages via st.session_state["data_dir"].
# =============================================================================
from core.ui import data_dir_input as _data_dir_input
st.sidebar.markdown("### Data source")
folder = _data_dir_input(default="market_data")
if folder is None:
    st.info("Specify the market data folder in the sidebar.")
    st.stop()

# Phase 3: auto-discover and register per-pair HMM regime panels from
# `<folder>/regimes/*.csv`. After this call, the `hmm_state_*` and
# `hmm_prob_state_*_gt_*` gates work in the gate multi-selects for any
# pair that has a saved regime file. Pairs without a file fall through
# gracefully (gate returns all-False, no crash). Regenerate the files
# any time via app 10's "Fit & save per-pair regimes" tab.
from core.regimes import set_regime_folder, list_registered_pairs
set_regime_folder(folder)
_reg_pairs = list_registered_pairs()
if _reg_pairs:
    st.sidebar.caption(
        f"🧬 Regime panels loaded for: {', '.join(_reg_pairs)}"
    )

with st.sidebar.expander("Discovered files", expanded=False):
    s = discovery_summary(folder)
    st.caption(f"Mode: `{s['mode']}`  ·  {s['n_pairs']} pairs across {s['n_files']} files")
    if s["categories"]:
        for k, v in sorted(s["categories"].items()):
            st.caption(f"  {k}: {v} pairs")


# =============================================================================
# Sidebar — Worst-of approximation multiplier
# =============================================================================
# Used in the Worst-of tab. There's no closed form for worst-of KO, so we
# use the rule-of-thumb:
#     premium_worstof  ≈  multiplier × min(premium_leg1, premium_leg2)
# where the multiplier captures how much cheaper the worst-of is vs the
# cheaper single leg. European-barrier worst-of is typically priced
# closer to the cheaper single leg (less monitoring → fewer ways to KO),
# so we default to 50%. Use 40% for less-correlated legs.
st.sidebar.markdown("### Worst-of approximation")
wo_multiplier_pct = st.sidebar.radio(
    "Multiplier",
    options=[33, 40, 50],
    index=2,            # 50% default for App 9 (European barrier)
    horizontal=True,
    format_func=lambda x: f"{x}%",
    key="eko_wo_multiplier_pct",
    help=("Premium ≈ multiplier × min(price_leg1, price_leg2). "
           "European-barrier worst-of typically prices closer to the "
           "cheaper single leg than American-barrier — default 50% here, "
           "vs 40% for App 12. Adjust by leg correlation."),
)
wo_multiplier = wo_multiplier_pct / 100.0


# =============================================================================
# Single-leg pricing model
# =============================================================================
# Three choices control how the smile enters the EKO price:
#
# - 'flat_atm'      : ignore the smile entirely; use σ_atm at all strikes.
#                     Useful as a debug baseline ("what does no smile look
#                     like?"). Equivalent to the pre-smile-era pricer.
# - 'vol_at_strike' : use σ_smile(K) — the vol interpolated off the smile
#                     at the option's own strike — as a single σ in the
#                     KO closed form. This is the historical default in
#                     this app (matches existing backtests).
# - 'vanna_volga'   : Castagna-Mercurio first-order Vanna-Volga correction
#                     on top of the flat-vol KO closed form. Replicates
#                     the exotic's vega/vanna/volga with three reference
#                     vanillas (ATM + 25Δ wings) at their smile vols.
#                     Matches Bloomberg OVML's 'Vanna-Volga' model within
#                     ~0.5% on validated cases. Heavier compute
#                     (~5 closed-form calls per price) but captures the
#                     full smile premium that vol-at-strike misses on
#                     ATM-strike trades with wing-region barriers.
#
# Stored on st.session_state under 'eko_pricing_model'. Default kept at
# 'vol_at_strike' to preserve backwards compatibility with saved
# backtests; switch to 'vanna_volga' on a per-trade basis to see the
# smile-corrected mid.
st.sidebar.markdown("### Single-leg pricing model")
_pricing_model_options = ["flat_atm", "vol_at_strike", "vanna_volga"]
_pricing_model_labels = {
    "flat_atm": "Flat BS (σ_atm)",
    "vol_at_strike": "Vol-at-strike σ_smile(K)",
    "vanna_volga": "Vanna-Volga",
}
pricing_model = st.sidebar.radio(
    "Model",
    options=_pricing_model_options,
    index=1,            # default: vol_at_strike (= legacy behavior)
    format_func=lambda x: _pricing_model_labels[x],
    key="eko_pricing_model",
    help=(
        "How the smile enters the EKO price. "
        "Flat BS ignores the smile (debug baseline). "
        "Vol-at-strike uses σ_smile(K) — the historical app default. "
        "Vanna-Volga adds the first-order Castagna-Mercurio smile "
        "correction (matches Bloomberg OVML). "
        "Greeks shown below are still computed at σ_smile in all three "
        "modes — VV-consistent Greeks are a follow-up step."
    ),
)


def _price_eko_dispatch(option_type: str, barrier_type: str,
                        S: float, K: float, H: float, T: float,
                        sigma_atm: float, sigma_smile: float,
                        rr_25: float, bf_25: float,
                        r_d: float, r_f: float,
                        model: str) -> "tuple[float, dict]":
    """Thin alias around core.eko_pricing.price_eko_dispatch.

    Kept as a name-stable wrapper so all in-page call sites continue
    to import it from this module; the actual logic was moved to
    core.eko_pricing so the single-leg backtester (core/backtest.py)
    can share it. Any future model addition should land in that
    module — both callers will pick it up.
    """
    from core.eko_pricing import price_eko_dispatch
    return price_eko_dispatch(
        option_type, barrier_type, S, K, H, T,
        sigma_atm, sigma_smile, rr_25, bf_25, r_d, r_f, model,
    )


# =============================================================================
# Shared helpers
# =============================================================================
TENOR_LIST = ['1M', '6W', '2M', '10W', '3M']
DELTA_CHOICES = {"ATM": 0.0, "45Δ": 0.45, "40Δ": 0.40,
                 "35Δ": 0.35, "30Δ": 0.30, "25Δ": 0.25}
PAYOUT_CHOICES = {"4:1": 4.0, "8:1": 8.0, "16:1": 16.0, "32:1": 32.0}
KO_DELTA_CHOICES = {"5Δ": 0.05, "10Δ": 0.10, "15Δ": 0.15, "20Δ": 0.20, "25Δ": 0.25}
DIRECTIONS = {
    "Call (up-and-out)": ("call", "up_and_out"),
    "Put (down-and-out)": ("put", "down_and_out"),
}

FEASIBILITY_HELP = (
    "Feasibility = whether the target payout ratio is achievable at the "
    "strike. ratio(H) is U-shaped, so each strike has an achievable "
    "minimum (e.g. 25Δ in normal vol regimes can't get below ~12×). "
    "When target < min, the trade still happens but at the closest "
    "achievable structure (H at the U-shape minimum). `feasible=False` "
    "rows in the ledger are 'best-we-can-do' trades, not target-matching."
)


# =============================================================================
# Preset loading (Phase 4.5 — bridges app 10's Barrier guidance → here).
# This is the bridge: app 10 tab 4 writes a JSON preset; app 9 reads it
# and auto-populates the bulk-runner controls. Users can still hand-edit
# any control after loading.
#
# Streamlit gotcha: you can't programmatically modify `st.session_state[k]`
# for a widget that's already been instantiated in the same script run.
# So we use a two-step pattern:
#   1. User picks a preset → we stash it under `_pending_preset_*` keys
#   2. On the next rerun (triggered by the button click), we copy from
#      `_pending_preset_*` into the actual widget keys BEFORE the widgets
#      are instantiated.
# This is a well-known Streamlit pattern for one-click "fill from saved
# config" UIs.
# =============================================================================
from core import presets as _presets


def _apply_pending_preset(tab: str) -> None:
    """Copy pending preset values into widget keys BEFORE widgets render.

    Call this at the top of each bulk-runner tab, before any widgets
    are created. `tab` is 'bt' (Backtest) or 'wo' (Worst-of). If no
    pending preset for this tab, this is a no-op.

    IMPORTANT subtleties:
    - Gate multi-selects store GATE KEYS (e.g. "hmm_state_0") or the
      literal string "(no gate)", NOT human-readable labels. The
      format_func just changes display.
    - `wo_combos` is a st.data_editor, not a multiselect. We pass a
      DataFrame with the right columns; the editor picks it up.
    """
    key = f"_pending_preset_{tab}"
    if key not in st.session_state:
        return
    preset_data = st.session_state.pop(key)
    if tab == "bt":
        bg = preset_data.get("backtest_grid", {})
        if bg.get("pairs"):
            st.session_state["bt_pairs"] = bg["pairs"]
        if bg.get("deltas"):
            st.session_state["bt_deltas"] = bg["deltas"]
        if bg.get("tenors"):
            st.session_state["bt_tenors"] = bg["tenors"]
        if bg.get("direction"):
            st.session_state["bt_directions"] = [bg["direction"]]
        if bg.get("ko_method") == "delta":
            st.session_state["bt_ko_method"] = "KO delta"
            if bg.get("ko_deltas"):
                st.session_state["bt_ko_delta"] = bg["ko_deltas"]
        if "gate_keys" in bg:
            # Gates are stored as GATE KEYS, with "(no gate)" for the
            # null option (NOT labels — the widget uses format_func to
            # display them).
            st.session_state["bt_gate_keys"] = [
                g if g else "(no gate)" for g in bg["gate_keys"]
            ]
    elif tab == "wo":
        wg = preset_data.get("worstof_grid", {})
        if wg.get("pair_combos"):
            # `wo_combos` is a st.data_editor — and Streamlit does NOT
            # allow setting its value via session_state at any point
            # (StreamlitValueAssignmentNotAllowedError). The
            # ValueAssignment rule for data_editor is stricter than for
            # multiselect / selectbox. So we stash the desired combos
            # in a NON-widget key `_pending_wo_combos` and rely on the
            # data_editor construction site (below in this file) to
            # consume that slot as its `value=` argument when present.
            # This sidesteps the rule entirely — Streamlit only forbids
            # writing to a widget's OWN session_state key.
            st.session_state["_pending_wo_combos"] = pd.DataFrame(
                wg["pair_combos"], columns=["Pair A", "Pair B"]
            )
        if wg.get("tenors"):
            st.session_state["wo_tenors"] = wg["tenors"]
        if wg.get("direction"):
            st.session_state["wo_direction"] = wg["direction"]
        # Phase WF-C bugfix: in WF mode the preset stores `["DYN"]` as
        # the strike/KO band placeholder because the engine reads
        # levels from the schedule, not the multi-selects. But "DYN"
        # isn't a valid choice in DELTA_CHOICES, so the multi-select
        # would render EMPTY and the spec-grid count would be 0 — the
        # backtest wouldn't run. Fix: in WF mode, populate the multi-
        # selects with a single valid placeholder ("35Δ" for strike,
        # "10Δ" for KO). The engine ignores these values when a
        # schedule is attached; they just need to be present so the
        # grid produces exactly one spec per (tenor × gate) cell.
        is_wf_preset = preset_data.get("dynamic_schedule") is not None
        if is_wf_preset:
            st.session_state["wo_sd_a"] = ["35Δ"]
            st.session_state["wo_kd_a"] = ["10Δ"]
            st.session_state["wo_sd_b"] = ["35Δ"]
            st.session_state["wo_kd_b"] = ["10Δ"]
        else:
            if wg.get("leg_a_strike_deltas"):
                st.session_state["wo_sd_a"] = wg["leg_a_strike_deltas"]
            if wg.get("leg_a_ko_deltas"):
                st.session_state["wo_kd_a"] = wg["leg_a_ko_deltas"]
            if wg.get("leg_b_strike_deltas"):
                st.session_state["wo_sd_b"] = wg["leg_b_strike_deltas"]
            if wg.get("leg_b_ko_deltas"):
                st.session_state["wo_kd_b"] = wg["leg_b_ko_deltas"]
        if "gates_a" in wg:
            st.session_state["wo_gate_a"] = [
                g if g else "(no gate)" for g in wg["gates_a"]
            ]
        if "gates_b" in wg:
            st.session_state["wo_gate_b"] = [
                g if g else "(no gate)" for g in wg["gates_b"]
            ]
        # Phase WF-C: stash the dynamic schedule for the bulk runner.
        # When this is non-None, the run path will attach it to every
        # spec produced by build_worstof_grid. Cleared when a static
        # preset (without dynamic_schedule) is loaded — so switching
        # presets works as expected.
        st.session_state["_active_wo_schedule"] = preset_data.get(
            "dynamic_schedule"
        )
        st.session_state["_active_wo_schedule_label"] = (
            preset_data.get("label") if preset_data.get("dynamic_schedule")
            else None
        )
        # Adaptive: strike strategy is in metadata. Default to cheapest.
        st.session_state["_active_wo_strike_strategy"] = (
            preset_data.get("metadata", {})
            .get("strike_strategy", "cheapest")
        )


def _render_preset_loader(tab: str, folder: str) -> None:
    """Render the '📥 Load preset' expander at the top of a bulk-runner
    tab. Call this AFTER `_apply_pending_preset(tab)` and AFTER the
    folder is validated, but BEFORE any control widgets are created.

    Workflow: user expands → picks a preset → clicks Apply →
    `_pending_preset_<tab>` is set in session state → `st.rerun()` is
    invoked → on next run, `_apply_pending_preset()` copies values into
    widget state → widgets render with preset defaults.
    """
    presets_list = _presets.list_presets(folder)
    if not presets_list:
        return   # No presets to load — silent (folder may have none yet).
    with st.expander(
        f"📥 Load preset from app 10 ({len(presets_list)} available)",
        expanded=False,
    ):
        st.caption(
            "Auto-populates the controls below with the analytical "
            "suggestions exported from app 10's Barrier guidance tab. "
            "You can still hand-edit anything before clicking Run. "
            "Existing values will be overwritten when you click Apply."
        )
        # Build display labels — "USDJPY × USDKRW · 1M · 95% cluster 0 · 2026-05-11 08:45"
        opt_labels = []
        for p in presets_list:
            # Trim microseconds from timestamp for readability
            ts = (p["generated_at"][:16].replace("T", " ")
                  if p["generated_at"] else "")
            opt_labels.append(f"{p['label']} · {ts}")
        chosen_label = st.selectbox(
            "Preset",
            options=opt_labels,
            key=f"preset_choice_{tab}",
        )
        chosen_idx = opt_labels.index(chosen_label)
        chosen_preset_meta = presets_list[chosen_idx]
        loaded = _presets.load_preset(chosen_preset_meta["path"])
        if loaded is None:
            st.error("Preset file is corrupted or has an unrecognised "
                      "version. Skip this one or regenerate from app 10.")
            return
        # Preview what will be applied
        if tab == "bt":
            target = loaded.get("backtest_grid", {})
            st.markdown("**Will apply:**")
            st.json({
                "pairs": target.get("pairs"),
                "deltas": target.get("deltas"),
                "tenors": target.get("tenors"),
                "ko_method": target.get("ko_method"),
                "ko_deltas": target.get("ko_deltas"),
                "direction": target.get("direction"),
                "gate_keys": target.get("gate_keys"),
            }, expanded=False)
        else:
            target = loaded.get("worstof_grid", {})
            st.markdown("**Will apply:**")
            st.json({
                "pair_combos": target.get("pair_combos"),
                "tenors": target.get("tenors"),
                "leg_a_strike_deltas": target.get("leg_a_strike_deltas"),
                "leg_a_ko_deltas": target.get("leg_a_ko_deltas"),
                "leg_b_strike_deltas": target.get("leg_b_strike_deltas"),
                "leg_b_ko_deltas": target.get("leg_b_ko_deltas"),
                "gates_a": target.get("gates_a"),
                "gates_b": target.get("gates_b"),
            }, expanded=False)
        # Show preset metadata
        meta = loaded.get("metadata", {})
        if meta:
            with st.expander("Preset metadata (generated context)",
                              expanded=False):
                st.json(meta, expanded=False)
        if st.button(
            "✅ Apply preset", key=f"apply_preset_{tab}",
            help="Sets the controls below from this preset. You can "
                  "still edit anything before clicking Run.",
        ):
            st.session_state[f"_pending_preset_{tab}"] = loaded
            st.rerun()


def _render_batch_run_all_presets(folder: str) -> None:
    """Render a 'Run all WF presets' UI for the worst-of tab.

    Loops over every preset in `<folder>/presets/`, runs the worst-of
    bulk backtest for each (using the run-time settings from session
    state — notional, tx cost, date range, prefer_em), and pools the
    results into a single combined view sorted by Sharpe.

    Only shows when at least one WF preset is in the folder. Each
    preset becomes ONE spec (the schedule determines K, H per trade
    date; the multi-selects are placeholders). The grid for each
    preset is sized by len(tenors) × len(gates_a) × len(gates_b) — in
    typical use that's just 1 spec per preset.
    """
    presets_list = _presets.list_presets(folder)
    if not presets_list:
        return
    # Filter to those with a dynamic_schedule attached — static
    # presets are also valid but the "Apply each, run, compare" loop
    # is mainly motivated by the WF workflow (each WF preset is a
    # different cluster's schedule). We include static too so users can
    # mix them in the same comparison.
    loaded_set = []
    for p in presets_list:
        data = _presets.load_preset(p["path"])
        if data is None:
            continue
        loaded_set.append((p, data))
    if not loaded_set:
        return

    n_wf = sum(1 for _, d in loaded_set
                  if d.get("dynamic_schedule") is not None)
    n_static = len(loaded_set) - n_wf

    with st.expander(
        f"⚡ Run all presets ({len(loaded_set)} available — "
        f"{n_wf} WF, {n_static} static)",
        expanded=False,
    ):
        st.caption(
            "Runs **every preset** in the folder back-to-back, using "
            "the run-time settings below (notional, tx cost, date "
            "range, direction). Each preset contributes a small "
            "sub-grid (typically one spec for WF presets, more for "
            "static ones with delta bands). All results are pooled "
            "into the comparison table — sort by Sharpe to find the "
            "best (cluster, tenor, mode) combination across all "
            "presets in one click."
        )
        # Show what's about to run
        preset_summary = pd.DataFrame([
            {
                "Preset": d.get("label", "?"),
                "Pair A": d.get("pair_a", "?"),
                "Pair B": d.get("pair_b", "?"),
                "Tenor": d.get("tenor", "?"),
                "Cluster": d.get("target_cluster", "?"),
                "Mode": ("WF" if d.get("dynamic_schedule") else "static"),
                "Gate": (d.get("worstof_grid", {})
                            .get("gates_a", [None])[0] or "(none)"),
            }
            for _, d in loaded_set
        ])
        st.dataframe(preset_summary, use_container_width=True,
                      hide_index=True)

        # Run-time settings come from the regular widgets below; we
        # read them from session_state. On the FIRST visit to this tab
        # those widgets haven't rendered yet, so the keys may be
        # missing. In that case we tell the user to scroll down and
        # configure once first, then come back.
        required_keys = ["wo_notional", "wo_tx", "wo_start", "wo_end",
                          "wo_prefer", "wo_direction"]
        missing = [k for k in required_keys
                     if k not in st.session_state]
        if missing:
            st.info(
                "⏳ Scroll down once to render the run-time controls "
                "(notional, dates, etc.), then come back here. "
                "Streamlit only registers widget values after they've "
                "been displayed once."
            )
            return

        col_run = st.columns([3, 1])
        with col_run[1]:
            run_all_clicked = st.button(
                "▶ Apply & run all",
                key="wo_run_all_presets",
                type="primary", use_container_width=True,
                help=("Loops through every preset, runs the worst-of "
                      "backtest for each, pools results into a single "
                      "comparison view."),
            )
        if run_all_clicked:
            _execute_run_all_presets(folder, loaded_set)


def _execute_run_all_presets(folder: str,
                                 loaded_set: list[tuple[dict, dict]]) -> None:
    """Execute the loop: for each preset, build a small spec grid and
    run the bulk backtest. Aggregate results into a single combined
    summary table in session state.

    The original `wo_results`/`wo_specs` slots are populated with the
    UNION of trades across all preset runs, so the existing summary
    table rendering code in the WO tab automatically displays them
    without changes. Each spec's name is prefixed with the preset
    label so rows in the table are unambiguous.
    """
    from core.worstof import (
        build_worstof_grid, run_worstof_grid,
    )

    # Pull run-time settings from session state (validated above)
    notional = float(st.session_state["wo_notional"])
    tx_cost = float(st.session_state["wo_tx"])
    start_dt = st.session_state["wo_start"]
    end_dt = st.session_state["wo_end"]
    prefer = st.session_state["wo_prefer"]
    direction_label = st.session_state["wo_direction"]
    direction_a, btype_a = DIRECTIONS[direction_label]
    direction_b, btype_b = DIRECTIONS[direction_label]

    combined_results = {}
    combined_specs = {}
    error_rows = []

    overall_progress = st.progress(0.0, text="Starting batch run…")
    n_total = len(loaded_set)
    for i, (preset_meta, preset_data) in enumerate(loaded_set):
        overall_progress.progress(
            i / n_total,
            text=f"Running preset {i+1}/{n_total}: "
                  f"{preset_data.get('label', '?')[:60]}",
        )
        wg = preset_data.get("worstof_grid", {})
        is_wf = preset_data.get("dynamic_schedule") is not None
        # Build the small grid for this preset.
        pair_combos = [tuple(c) for c in wg.get("pair_combos", [])]
        if not pair_combos:
            error_rows.append({
                "Preset": preset_data.get("label", "?"),
                "Status": "skipped: no pair_combos in preset",
            })
            continue
        tenors = wg.get("tenors", ["1M"])
        # Strike/KO bands: in WF mode the preset has placeholders;
        # substitute with a single valid delta so build_worstof_grid
        # produces specs. The engine ignores these in WF mode.
        if is_wf:
            sd_a, kd_a = ["35Δ"], ["10Δ"]
            sd_b, kd_b = ["35Δ"], ["10Δ"]
        else:
            sd_a = wg.get("leg_a_strike_deltas", ["35Δ"])
            kd_a = wg.get("leg_a_ko_deltas", ["10Δ"])
            sd_b = wg.get("leg_b_strike_deltas", ["35Δ"])
            kd_b = wg.get("leg_b_ko_deltas", ["10Δ"])
        # Resolve gates: preset stores keys (or None); the grid builder
        # accepts None entries.
        gates_a = [g for g in wg.get("gates_a", [None])]
        gates_b = [g for g in wg.get("gates_b", [None])]

        # Build deltas in their numeric form (the grid builder accepts
        # label/value pairs as the per-leg lists).
        def _to_pairs(labels: list[str], choices: dict[str, float]):
            out = []
            for lab in labels:
                if lab in choices:
                    out.append((lab, choices[lab]))
            return out
        sd_a_resolved = _to_pairs(sd_a, DELTA_CHOICES)
        kd_a_resolved = _to_pairs(kd_a, KO_DELTA_CHOICES)
        sd_b_resolved = _to_pairs(sd_b, DELTA_CHOICES)
        kd_b_resolved = _to_pairs(kd_b, KO_DELTA_CHOICES)

        try:
            specs = build_worstof_grid(
                pair_combos=pair_combos,
                tenors=tenors,
                leg_a_directions=[(direction_a, btype_a)],
                leg_b_directions=[(direction_b, btype_b)],
                leg_a_strike_deltas=sd_a_resolved,
                leg_b_strike_deltas=sd_b_resolved,
                leg_a_ko_deltas=kd_a_resolved,
                leg_b_ko_deltas=kd_b_resolved,
                gates_a=gates_a,
                gates_b=gates_b,
                tx_cost_bps=tx_cost,
                prefer=prefer,
                multiplier=wo_multiplier,
            )
        except Exception as e:
            error_rows.append({
                "Preset": preset_data.get("label", "?"),
                "Status": f"grid-build failed: {e}",
            })
            continue
        if not specs:
            error_rows.append({
                "Preset": preset_data.get("label", "?"),
                "Status": "skipped: empty grid after KO<strike filter",
            })
            continue
        # Attach the dynamic schedule (in WF mode) to every spec
        if is_wf:
            schedule = preset_data.get("dynamic_schedule")
            # Pull strike strategy from preset metadata (default cheapest)
            strike_strategy = (
                preset_data.get("metadata", {})
                .get("strike_strategy", "cheapest")
            )
            for s in specs:
                s.dynamic_schedule = schedule
                s._adaptive_strike_strategy = strike_strategy
        # Run this preset's sub-grid
        try:
            sub_results = run_worstof_grid(
                folder, specs, start_dt, end_dt,
                notional_usd=notional,
            )
        except Exception as e:
            error_rows.append({
                "Preset": preset_data.get("label", "?"),
                "Status": f"run failed: {e}",
            })
            continue
        # Merge results into the combined collection. Prefix each
        # spec name with the preset label so the table is unambiguous
        # when the same strategy structure appears in multiple presets.
        preset_short = preset_data.get("label", "?").split(" · ")
        # E.g. "USDJPY × USDKRW · 1M · 95% cluster 0 · WF" → "c0 · WF"
        # The short prefix saves table width — full preset label is
        # already shown in the summary above.
        try:
            tag_parts = []
            for p in preset_short[2:]:   # skip pair pair + tenor
                tag_parts.append(p.replace("95% cluster ", "c"))
            short_tag = " · ".join(tag_parts) or preset_data.get("label", "?")
        except Exception:
            short_tag = preset_data.get("label", "?")
        for spec_name, trades in sub_results.items():
            unique_name = f"[{short_tag}] {spec_name}"
            combined_results[unique_name] = trades
            # Find the matching spec object for the specs cache
            for s in specs:
                if s.name == spec_name:
                    combined_specs[unique_name] = s
                    break
    overall_progress.empty()

    # Collect aggregate info from the loaded set so the summary
    # caption renders meaningfully (otherwise it shows "direction: None
    # · tenors:" which is confusing).
    all_tenors = sorted(set(
        d.get("tenor") for _, d in loaded_set if d.get("tenor")
    ))
    all_pair_combos = sorted(set(
        tuple(combo)
        for _, d in loaded_set
        for combo in d.get("worstof_grid", {}).get("pair_combos", [])
    ))

    # Stash the combined results in the same slots the regular run path
    # uses, so the existing summary/dl/breakdown code displays them
    # without modification.
    st.session_state["wo_results"] = combined_results
    st.session_state["wo_specs"] = combined_specs
    st.session_state["wo_meta"] = {
        "start": start_dt, "end": end_dt,
        "notional_usd": notional, "tx_cost_bps": tx_cost,
        "prefer": prefer,
        "direction_label": direction_label,
        "tenors": all_tenors,
        "pair_combos": list(all_pair_combos),
        "n_specs": len(combined_results),
        "n_trades": sum(len(t) for t in combined_results.values()),
        "elapsed": 0.0,
        # batch_run_all signals to the summary section to show a
        # persistent banner explaining where these results came from.
        # Survives reruns because it's in session state.
        "batch_run_all": True,
        "batch_n_presets": len(loaded_set),
        "batch_n_errors": len(error_rows),
    }
    n_ok = len(combined_results)
    n_err = len(error_rows)
    if n_ok > 0:
        st.success(
            f"✅ Batch complete: {n_ok} strategies executed across "
            f"{len(loaded_set)} presets. **Scroll down past the "
            f"configuration controls to see the summary table** — "
            f"those controls below are for a *new* manual run and "
            f"don't reflect the batch."
        )
    if error_rows:
        with st.expander(f"⚠️ {n_err} preset(s) had errors",
                          expanded=False):
            st.dataframe(pd.DataFrame(error_rows),
                          use_container_width=True, hide_index=True)


def _fmt_px(pair, x):
    if pair in ("USDJPY", "EURJPY", "GBPJPY", "AUDJPY"):
        return f"{x:.3f}"
    if pair in ("USDIDR", "USDKRW", "USDTWD"):
        return f"{x:.2f}"
    return f"{x:.4f}"


def _fmt_usd(x):
    if not np.isfinite(x):
        return "∞"
    sign = "-" if x < 0 else ""
    a = abs(x)
    if a >= 1e6:
        return f"{sign}${a/1e6:.2f}M"
    if a >= 1e3:
        return f"{sign}${a/1e3:.1f}K"
    return f"{sign}${a:.0f}"


def _fmt_date(d):
    return d.strftime('%a %d %b %Y') if d is not None else "—"


@st.cache_data(show_spinner=False)
def _list_pairs(folder: str) -> list[str]:
    df = load_panel(folder, "SPOT", None)
    return sorted(df.columns.tolist())


@st.cache_data(show_spinner="Loading rates…")
def _load_rates_panel_cached(folder: str, currency: str):
    return load_rates_panel(folder, currency, load_by_ticker)


def _rdylgn_bg(v, vmax: float = 2.0) -> str:
    """Map a numeric value to an RdYlGn cell background CSS string,
    no matplotlib needed. v in [-vmax, +vmax] is mapped Red→Yellow→Green.

    Anchor colours come from ColorBrewer RdYlGn (5-class):
      red    = rgb(215, 25, 28)
      yellow = rgb(255, 255, 191)
      green  = rgb( 26,150, 65)
    """
    if pd.isna(v):
        return ""
    t = max(-1.0, min(1.0, float(v) / vmax))
    if t < 0:
        f = t + 1.0   # 0 (red) -> 1 (yellow)
        r = int(215 + (255 - 215) * f)
        g = int( 25 + (255 -  25) * f)
        b = int( 28 + (191 -  28) * f)
    else:
        f = t           # 0 (yellow) -> 1 (green)
        r = int(255 + ( 26 - 255) * f)
        g = int(255 + (150 - 255) * f)
        b = int(191 + ( 65 - 191) * f)
    # Dark text reads well on these mid-range pastels
    return f"background-color: rgb({r},{g},{b}); color: #111;"


# =============================================================================
# Tabs
# =============================================================================
tab_pricer, tab_backtest, tab_drilldown, tab_worstof_pricer, \
    tab_worstof, tab_wo_drill, \
    tab_eko_port, tab_eko_drill, tab_wo_port, tab_wo_drill_port = st.tabs(
    ["💰  Pricer", "📊  Backtest", "🔍  Strategy drilldown",
     "🪢💰  Worst-of Pricer",
     "🪢  Worst-of", "🔬  Worst-of strategy drilldown",
     "📦  EKO Portfolio", "🔎  EKO Portfolio drilldown",
     "🪢📦  WO EKO Portfolio", "🪢🔎  WO EKO Portfolio drilldown"]
)


# -----------------------------------------------------------------------------
# TAB 1: PRICER
# -----------------------------------------------------------------------------
def render_pricer_tab():
    pairs_avail = _list_pairs(folder)
    if not pairs_avail:
        st.error("No SPOT data found.")
        return

    with st.container():
        c_in, c_out = st.columns([1, 2.6], gap="medium")

    with c_in:
        st.markdown("**Trade**")
        tenor_label = st.selectbox("Tenor", TENOR_LIST, index=0,
                                     key="pr_tenor")
        default_pair = ("USDJPY" if "USDJPY" in pairs_avail else
                          ("EURUSD" if "EURUSD" in pairs_avail else pairs_avail[0]))
        pair = st.selectbox("Currency pair", pairs_avail,
                              index=pairs_avail.index(default_pair), key="pr_pair")
        asia_em = pair in {"USDCNH", "USDCNY", "USDINR", "USDIDR", "USDKRW",
                            "USDMYR", "USDPHP", "USDTHB", "USDTWD"}
        prefer = st.radio("Variant", ["offshore", "onshore"], index=0,
                            horizontal=True, key="pr_prefer") if asia_em else "offshore"

        direction_label = st.radio("Direction", list(DIRECTIONS.keys()),
                                     index=0, key="pr_dir")
        option_type, barrier_type = DIRECTIONS[direction_label]

        strike_delta_label = st.radio("Strike Δ",
                                        list(DELTA_CHOICES.keys()), index=0,
                                        horizontal=True, key="pr_delta")
        strike_delta = DELTA_CHOICES[strike_delta_label]

        ko_method_label = st.radio(
            "KO method", ("Payout ratio", "KO delta"),
            index=0, horizontal=True, key="pr_ko_method",
            help=("• **Payout ratio**: solves H so that max_payoff / premium "
                   "hits the target leverage.\n• **KO delta**: places H at a "
                   "vanilla-Δ wing strike (same convention as the strike Δ).")
        )
        ko_method = "ratio" if ko_method_label == "Payout ratio" else "delta"

        if ko_method == "ratio":
            payout_label = st.radio("Payout ratio (max payoff / premium)",
                                      list(PAYOUT_CHOICES.keys()), index=1,
                                      horizontal=True, key="pr_payout")
            payout_ratio = PAYOUT_CHOICES[payout_label]
            ko_delta_label, ko_delta_value = None, None
        else:
            ko_delta_label = st.radio(
                "KO Δ (vanilla wing)",
                list(KO_DELTA_CHOICES.keys()), index=1,  # default 10Δ
                horizontal=True, key="pr_ko_delta",
                help=("Barrier H is placed at the strike where the same-side "
                       "vanilla option has this Δ. Smaller Δ ⇒ deeper OTM ⇒ "
                       "barrier further from spot ⇒ lower KO probability ⇒ "
                       "higher premium.")
            )
            ko_delta_value = KO_DELTA_CHOICES[ko_delta_label]
            payout_label, payout_ratio = None, None

        notional_usd = st.number_input(
            "Notional (USD)", min_value=100_000.0, max_value=200_000_000.0,
            value=10_000_000.0, step=1_000_000.0, format="%.0f",
            key="pr_notional",
        )

    # Spot
    spot_df = load_panel(folder, "SPOT", None, prefer=prefer, pairs=(pair,))
    if spot_df.empty or pair not in spot_df.columns:
        c_out.error(f"No SPOT for {pair}.")
        return
    spot_ts = spot_df[pair].dropna()
    if spot_ts.empty:
        c_out.error(f"Empty SPOT for {pair}.")
        return

    with c_in:
        val_date = st.date_input(
            "Trade date", value=spot_ts.index.max().date(),
            min_value=spot_ts.index.min().date(),
            max_value=spot_ts.index.max().date(),
            key="pr_val_date",
        )

    opt_dates = compute_option_dates(val_date, tenor_label)
    T = opt_dates.T_years
    val_ts = pd.Timestamp(val_date)

    # Spot at val date
    valid = spot_ts.loc[:val_ts]
    if valid.empty:
        c_out.error(f"No spot data at or before {val_date}.")
        return
    S = float(valid.iloc[-1])

    # Vol & forward across tenors
    sigma_pct = get_pair_value_at_T(folder, pair, prefer, "VOL_ATM", T, val_date)
    if sigma_pct is None:
        c_out.error(f"No VOL_ATM data for {pair}.")
        return
    sigma_atm = sigma_pct / 100.0

    # Smile inputs — optional (fall back to flat ATM if absent)
    rr_pct = get_pair_value_at_T(folder, pair, prefer, "VOL_25R", T, val_date)
    bf_pct = get_pair_value_at_T(folder, pair, prefer, "VOL_25B", T, val_date)
    rr_25 = (rr_pct / 100.0) if rr_pct is not None else 0.0
    bf_25 = (bf_pct / 100.0) if bf_pct is not None else 0.0
    smile_available = (rr_pct is not None) and (bf_pct is not None)

    # Backwards-compat alias used by the rest of the function
    sigma = sigma_atm

    fwd_pts_at_T = get_pair_value_at_T(folder, pair, prefer, "FWD_POINTS",
                                          T, val_date)
    pip = get_pip_scale(pair)
    if fwd_pts_at_T is None:
        F_market = S
        fwd_avail = False
    else:
        F_market = S + fwd_pts_at_T * pip
        fwd_avail = True

    # Rates
    foreign_ccy, domestic_ccy = pair[:3].upper(), pair[3:].upper()
    f_panel = _load_rates_panel_cached(folder, foreign_ccy)
    d_panel = _load_rates_panel_cached(folder, domestic_ccy)
    r_f_market = get_rate_at(f_panel, T, val_date)
    r_d_market = get_rate_at(d_panel, T, val_date)

    with c_in:
        st.markdown("---")
        st.markdown("**Rates**")
        if r_f_market is None:
            r_f = st.number_input(
                f"{foreign_ccy} (%, no data — manual)",
                min_value=-2.0, max_value=20.0, value=3.0, step=0.05,
                format="%.3f", key="pr_rf",
            ) / 100.0
            r_f_source = "manual"
        else:
            st.caption(f"{foreign_ccy} rate: **{r_f_market*100:.3f}%** "
                        f"(interp at T={T:.4f}y)")
            r_f = r_f_market
            r_f_source = f"{foreign_ccy} OIS interp"

        if r_d_market is not None:
            st.caption(f"{domestic_ccy} rate: **{r_d_market*100:.3f}%** "
                        f"(interp at T={T:.4f}y)")
            r_d = r_d_market
            r_d_source = f"{domestic_ccy} OIS interp"
        else:
            if fwd_avail:
                r_d = r_f + np.log(F_market / S) / T
                r_d_source = "implied from forward (CIP)"
                st.caption(f"{domestic_ccy} rate (CIP-implied): "
                            f"**{r_d*100:.3f}%**")
            else:
                r_d = r_f
                r_d_source = "= r_f (no fwd or domestic OIS)"

    F_implied_by_rates = S * np.exp((r_d - r_f) * T)

    # Solve K and H
    # Solve K and H (smile-aware if RR/BF available, flat-vol otherwise)
    K, H, info = solve_strike(
        option_type, barrier_type, strike_delta,
        S, T, sigma_atm, r_d, r_f,
        target_ratio=payout_ratio,
        target_ko_delta=ko_delta_value,
        ko_method=ko_method,
        rr_25=rr_25, bf_25=bf_25,
    )
    sigma_smile = float(info.get("sigma_smile", sigma_atm))

    # Price
    # Mid price — dispatches on the sidebar 'pricing_model' selection.
    # Greeks (delta, prob) below stay at σ_smile for now; VV-consistent
    # Greeks require FD on vv_price_ko and are a follow-up step.
    ko_per_unit, _price_detail = _price_eko_dispatch(
        option_type, barrier_type, S, K, H, T,
        sigma_atm, sigma_smile, rr_25, bf_25, r_d, r_f,
        pricing_model,
    )
    vanilla_per_unit = vanilla_price(option_type, S, K, T, sigma_smile, r_d, r_f)
    ko_prob = ko_probability(barrier_type, S, H, T, sigma_smile, r_d, r_f)
    ko_delta_signed = ko_spot_delta(option_type, barrier_type, S, K, H, T,
                                       sigma_smile, r_d, r_f)
    vanilla_delta_signed = vanilla_spot_delta(option_type, S, K, T, sigma_smile,
                                                 r_d, r_f)

    ko_usd = ko_per_unit / S * notional_usd
    vanilla_usd = vanilla_per_unit / S * notional_usd
    max_payoff_per_unit = abs(H - K) if (option_type, barrier_type) in (
        ("call", "up_and_out"), ("put", "down_and_out")) else (
        K if option_type == "put" else float("inf"))
    max_payoff_usd = (max_payoff_per_unit / S * notional_usd
                       if np.isfinite(max_payoff_per_unit) else float("inf"))

    # ----- Output side -----
    with c_out:
        st.markdown(
            f"### {pair}  ·  Buy "
            f"<span class='tag-{option_type}'>{option_type.upper()}</span>  "
            f"with KO "
            f"<span class='tag-ko'>{barrier_type.replace('_', '-').upper()}</span>  ·  "
            f"{tenor_label}  ·  strike {strike_delta_label}  ·  "
            + (f"target leverage {payout_label}" if ko_method == "ratio"
                else f"KO @ {ko_delta_label} (vanilla)"),
            unsafe_allow_html=True,
        )

        cc1, cc2, cc3, cc4, cc5 = st.columns(5)
        with cc1:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-title'>KO Premium</div>"
                f"<div class='metric-value'>{_fmt_usd(ko_usd)}</div>"
                f"<div class='metric-sub'>{abs(ko_usd)/notional_usd*100:.3f}% notl</div>"
                f"</div>", unsafe_allow_html=True)
        with cc2:
            lev = (max_payoff_usd / max(ko_usd, 1)
                   if np.isfinite(max_payoff_usd) else float("inf"))
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-title'>Max Payoff</div>"
                f"<div class='metric-value'>{_fmt_usd(max_payoff_usd)}</div>"
                f"<div class='metric-sub'>{lev:.1f}× leverage</div>"
                f"</div>", unsafe_allow_html=True)
        with cc3:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-title'>KO Probability</div>"
                f"<div class='metric-value'>{ko_prob*100:.1f}%</div>"
                f"<div class='metric-sub'>at expiry, risk-neutral</div>"
                f"</div>", unsafe_allow_html=True)
        with cc4:
            cheap = (1 - ko_per_unit / vanilla_per_unit) * 100 if vanilla_per_unit > 0 else 0
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-title'>vs Vanilla</div>"
                f"<div class='metric-value'>{cheap:.0f}% cheaper</div>"
                f"<div class='metric-sub'>vanilla = {_fmt_usd(vanilla_usd)}</div>"
                f"</div>", unsafe_allow_html=True)
        with cc5:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-title'>KO Spot Δ</div>"
                f"<div class='metric-value'>{ko_delta_signed*100:+.1f}%</div>"
                f"<div class='metric-sub'>vanilla Δ = {vanilla_delta_signed*100:+.1f}%</div>"
                f"</div>", unsafe_allow_html=True)

        # Solver feedback
        achieved = info.get("achieved_ratio", float("nan"))
        ratio_min = info.get("ratio_min", float("nan"))
        if info.get("note"):
            st.warning(info["note"])
        if ko_method == "ratio" and np.isfinite(achieved):
            feasible = abs(achieved - payout_ratio) < 0.01
            if feasible:
                st.caption(f"✓ target {payout_ratio:.0f}× leverage achieved "
                            f"({achieved:.2f}×). Min at this strike: "
                            f"{ratio_min:.2f}×.")
            else:
                st.caption(f"⚠ target {payout_ratio:.0f}× INFEASIBLE — "
                            f"min at this strike is {ratio_min:.2f}×; using H "
                            f"that gives the minimum.")
        elif ko_method == "delta" and np.isfinite(achieved):
            st.caption(f"✓ Barrier H placed at {ko_delta_label} vanilla wing "
                        f"strike. Achieved leverage: {achieved:.2f}× (depends on "
                        f"vol/rates — varies in backtest).")

        # Pricing-model badge + (for VV) breakdown of the smile correction.
        # We always show the active model so the user can tell at a glance
        # which pricer produced the displayed mid; the VV expander is only
        # rendered when 'vanna_volga' is selected and the correction is
        # non-trivial.
        st.caption(
            f"Pricing model: **{_pricing_model_labels[pricing_model]}** "
            f"(change in sidebar)."
        )
        if pricing_model == "vanna_volga" and _price_detail.get("vv_detail"):
            vv_d = _price_detail["vv_detail"]
            corr = _price_detail["correction"]
            p_bs = _price_detail["price_bs"]
            corr_pct = (corr / p_bs * 100) if abs(p_bs) > 1e-12 else float("nan")
            with st.expander("Vanna-Volga correction detail"):
                st.markdown(
                    f"- **Flat-vol BS price (σ_atm):** "
                    f"`{p_bs:.6f}` per FOR unit "
                    f"(`{_fmt_usd(p_bs / S * notional_usd)}` on notional)\n"
                    f"- **VV correction:** `{corr:+.6f}` per FOR unit "
                    f"(`{corr_pct:+.2f}%` of BS)\n"
                    f"- **VV mid:** `{ko_per_unit:.6f}` per FOR unit "
                    f"(`{_fmt_usd(ko_usd)}`)"
                )
                if vv_d.get("weights") is not None:
                    w_atm, w_25c, w_25p = vv_d["weights"]
                    K_atm, K_25c, K_25p = vv_d["reference_strikes"]
                    s_atm, s_25c, s_25p = vv_d["reference_vols"]
                    sc = vv_d["smile_costs"]
                    cond = vv_d["condition_number"]
                    st.markdown(
                        f"**Hedge weights / reference vanillas**\n"
                        f"- ATM: K=`{K_atm:.4f}`, σ=`{s_atm*100:.3f}%`, "
                        f"w=`{w_atm:+.3f}`, smile-cost=`{sc[0]:+.6f}`\n"
                        f"- 25Δ Call: K=`{K_25c:.4f}`, σ=`{s_25c*100:.3f}%`, "
                        f"w=`{w_25c:+.3f}`, smile-cost=`{sc[1]:+.6f}`\n"
                        f"- 25Δ Put:  K=`{K_25p:.4f}`, σ=`{s_25p*100:.3f}%`, "
                        f"w=`{w_25p:+.3f}`, smile-cost=`{sc[2]:+.6f}`\n"
                        f"\nCondition number of hedge system: "
                        f"`{cond:.2e}` "
                        f"({'OK' if cond < 1e8 else '⚠ ill-conditioned'})"
                    )
                elif vv_d.get("note"):
                    st.info(vv_d["note"])
                st.caption(
                    "VV correction = Σᵢ wᵢ × [C(Kᵢ, σ_smile_ᵢ) − C(Kᵢ, σ_atm)]. "
                    "The weights solve the 3×3 vega/vanna/volga hedge system "
                    "for the exotic. Greeks shown above use σ_smile; "
                    "VV-consistent Greeks coming in a follow-up step."
                )

        st.markdown("---")
        cd1, cd2 = st.columns(2)
        with cd1:
            st.markdown("**Date schedule**")
            st.markdown(
                f"- Trade date:&nbsp;&nbsp;`{_fmt_date(opt_dates.trade_date)}`\n"
                f"- Spot settlement:&nbsp;`{_fmt_date(opt_dates.spot_settlement)}` (T+2)\n"
                f"- Option settlement:&nbsp;`{_fmt_date(opt_dates.option_settlement)}` (spot + {tenor_label})\n"
                f"- Option expiry:&nbsp;`{_fmt_date(opt_dates.option_expiry)}` (settle − 2bd)\n"
                f"- T = `{T:.4f}`y ({(opt_dates.option_expiry - opt_dates.trade_date).days}d)"
            )
        with cd2:
            st.markdown("**Levels**")
            if not fwd_avail:
                st.warning(f"No FWD_POINTS at {tenor_label}: F = spot.")
            from core.smile import wing_vols_25d
            if smile_available:
                v_25c, v_25p = wing_vols_25d(sigma_atm, rr_25, bf_25)
                vol_lines = (
                    f"- ATM Vol: `{sigma_atm*100:.3f}%`  ·  "
                    f"25Δ RR: `{rr_25*100:+.3f}%`  ·  "
                    f"25Δ BF: `{bf_25*100:+.3f}%`\n"
                    f"  → σ_25C = `{v_25c*100:.3f}%`  ·  "
                    f"σ_25P = `{v_25p*100:.3f}%`\n"
                    f"- **σ_smile @ K** = `{sigma_smile*100:.3f}%` "
                    f"({(sigma_smile-sigma_atm)*100:+.3f} pp vs ATM) "
                    f"— used for pricing\n"
                )
            else:
                vol_lines = (
                    f"- ATM Vol: `{sigma_atm*100:.3f}%` "
                    f"(no RR/BF — flat-vol mode)\n"
                )
            st.markdown(
                f"- Spot: `{_fmt_px(pair, S)}`\n"
                f"- F_market: `{_fmt_px(pair, F_market)}` ({(F_market/S-1)*100:+.2f}%)\n"
                f"- F_implied (rates): `{_fmt_px(pair, F_implied_by_rates)}`\n"
                + vol_lines +
                f"- Strike K: `{_fmt_px(pair, K)}` ({(K/S-1)*100:+.2f}%)\n"
                f"- Barrier H: `{_fmt_px(pair, H)}` ({(H/S-1)*100:+.2f}%)"
            )

        basis_per_year = (np.log(F_market / F_implied_by_rates) / T
                            if (F_market > 0 and F_implied_by_rates > 0 and T > 0)
                            else 0.0)
        st.markdown("**Rates / forward consistency**")
        st.markdown(
            f"- r_f ({foreign_ccy}): `{r_f*100:.4f}%`  ({r_f_source})\n"
            f"- r_d ({domestic_ccy}): `{r_d*100:.4f}%`  ({r_d_source})\n"
            f"- FX basis = log(F_mkt / F_implied) / T ≈ "
            f"`{basis_per_year*100:.3f}%` per year"
        )
        if abs(basis_per_year) > 0.005:
            st.caption("⚠ Non-trivial gap between F_market and F_implied. The "
                       "pricer uses r_d and r_f directly, so its implicit "
                       "forward is F_implied, not F_market.")

        # Payoff diagram
        st.markdown("---")
        st.markdown("### Payoff at expiry")
        S_lo = (min(S * 0.85, K * 0.95, H * 0.95)
                if barrier_type == "down_and_out"
                else min(S * 0.85, K * 0.95))
        S_hi = (max(S * 1.15, K * 1.05, H * 1.05)
                if barrier_type == "up_and_out"
                else max(S * 1.15, K * 1.05))
        S_grid = np.linspace(S_lo, S_hi, 400)
        if option_type == "call":
            payoff = np.maximum(S_grid - K, 0.0)
        else:
            payoff = np.maximum(K - S_grid, 0.0)
        if barrier_type == "up_and_out":
            payoff = np.where(S_grid >= H, 0.0, payoff)
        else:
            payoff = np.where(S_grid <= H, 0.0, payoff)
        payoff_usd = payoff / S * notional_usd

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=S_grid, y=payoff_usd - ko_usd, mode="lines",
                                  name="Net P&L",
                                  line=dict(color="#38bdf8", width=2.6)))
        fig.add_trace(go.Scatter(x=S_grid, y=payoff_usd, mode="lines",
                                  name="Gross payoff",
                                  line=dict(color="#86efac", width=1.4, dash="dot")))
        fig.add_hline(y=-ko_usd, line=dict(color="#fb923c", dash="dash", width=1),
                       annotation_text=f"−Premium = {_fmt_usd(-ko_usd)}",
                       annotation_position="bottom right")
        fig.add_vline(x=S, line=dict(color="#9aa1ad", dash="dot", width=1),
                       annotation_text=f"S = {_fmt_px(pair, S)}",
                       annotation_position="top")
        fig.add_vline(x=K, line=dict(color="#facc15", dash="dot", width=1),
                       annotation_text=f"K = {_fmt_px(pair, K)}",
                       annotation_position="top")
        fig.add_vline(x=H, line=dict(color="#ef4444", dash="solid", width=1.6),
                       annotation_text=f"H = {_fmt_px(pair, H)}",
                       annotation_position="top")
        fig.update_layout(
            yaxis=dict(title="USD P&L", gridcolor="rgba(255,255,255,0.08)",
                        zeroline=True, zerolinecolor="rgba(255,255,255,0.3)"),
            xaxis=dict(title=f"{pair} spot at expiry"),
            height=380, template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            margin=dict(l=20, r=20, t=20, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                          xanchor="left", x=0,
                          font=dict(size=11, color="#cbd5e1")),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Premium grid
        st.markdown("---")
        smile_note = (" (smile-aware via 25Δ RR/BF)"
                       if smile_available else " (flat ATM — no RR/BF data)")
        if ko_method == "ratio":
            st.markdown("### Premium grid (strike Δ × leverage)")
            st.caption(f"Each cell solves K from vanilla Δ at σ_atm, then H "
                        f"from leverage target at σ_smile(K)" + smile_note +
                        ". Premium shown at σ_smile(K). Cells marked ⚠ are "
                        f"infeasible; the achieved leverage shown is the "
                        f"minimum the structure can deliver at that strike.")
            grid_rows = []
            for d_label, d_val in DELTA_CHOICES.items():
                row = {"Strike Δ": d_label}
                for r_label, r_val in PAYOUT_CHOICES.items():
                    K_g, H_g, info_g = solve_strike(
                        option_type, barrier_type, d_val,
                        S, T, sigma_atm, r_d, r_f,
                        target_ratio=r_val, ko_method="ratio",
                        rr_25=rr_25, bf_25=bf_25,
                    )
                    sg = info_g.get("sigma_smile", sigma_atm)
                    p, _ = _price_eko_dispatch(
                        option_type, barrier_type, S, K_g, H_g, T,
                        sigma_atm, sg, rr_25, bf_25, r_d, r_f,
                        pricing_model,
                    )
                    prem_usd = p / S * notional_usd
                    ach = info_g.get("achieved_ratio", float("nan"))
                    infeasible = (abs(ach - r_val) > 0.01
                                    if np.isfinite(ach) else True)
                    row[r_label] = (f"⚠ {_fmt_usd(prem_usd)} ({ach:.1f}× min)"
                                     if infeasible else _fmt_usd(prem_usd))
                grid_rows.append(row)
        else:
            st.markdown("### Premium grid (strike Δ × KO Δ)")
            st.caption(f"Each cell solves K from strike Δ and H from KO Δ, "
                        f"both at σ_atm" + smile_note + ". Premium shown at "
                        f"σ_smile(K). Cells marked ⚠ are degenerate (H lands "
                        f"on the wrong side of K — KO Δ ≥ strike Δ for "
                        f"up-out call, etc.). Number after premium is the "
                        f"achieved leverage that falls out from this geometry.")
            grid_rows = []
            for d_label, d_val in DELTA_CHOICES.items():
                row = {"Strike Δ": d_label}
                for kd_label, kd_val in KO_DELTA_CHOICES.items():
                    K_g, H_g, info_g = solve_strike(
                        option_type, barrier_type, d_val,
                        S, T, sigma_atm, r_d, r_f,
                        target_ko_delta=kd_val, ko_method="delta",
                        rr_25=rr_25, bf_25=bf_25,
                    )
                    sg = info_g.get("sigma_smile", sigma_atm)
                    p, _ = _price_eko_dispatch(
                        option_type, barrier_type, S, K_g, H_g, T,
                        sigma_atm, sg, rr_25, bf_25, r_d, r_f,
                        pricing_model,
                    )
                    prem_usd = p / S * notional_usd
                    ach = info_g.get("achieved_ratio", float("nan"))
                    if "note" in info_g:
                        row[kd_label] = "⚠ degenerate"
                    elif np.isfinite(ach):
                        row[kd_label] = f"{_fmt_usd(prem_usd)} ({ach:.1f}×)"
                    else:
                        row[kd_label] = _fmt_usd(prem_usd)
                grid_rows.append(row)
        st.dataframe(pd.DataFrame(grid_rows), use_container_width=True,
                      hide_index=True)


# -----------------------------------------------------------------------------
# TAB 2: BACKTEST
# -----------------------------------------------------------------------------
def render_backtest_tab():
    pairs_avail = _list_pairs(folder)
    if not pairs_avail:
        st.error("No SPOT data found.")
        return

    # Phase 4.5: apply pending preset (if any) BEFORE widgets are created.
    # Then render the loader expander so user can pick another preset.
    _apply_pending_preset("bt")
    _render_preset_loader("bt", folder)

    st.markdown("### Backtest configuration")
    st.caption("Cross-product of (pair × strike Δ × tenor × direction) is run "
                "as one strategy each. Same payout ratio and tx cost across "
                "all strategies.")

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        default_pairs = [p for p in ("USDJPY", "USDKRW") if p in pairs_avail]
        if not default_pairs:
            default_pairs = pairs_avail[:1]
        pairs_sel = st.multiselect(
            "Currency pairs", pairs_avail, default=default_pairs,
            key="bt_pairs",
        )
        deltas_sel = st.multiselect(
            "Strike Δ list", list(DELTA_CHOICES.keys()), default=["40Δ"],
            key="bt_deltas",
        )
        tenors_sel = st.multiselect(
            "Tenors", TENOR_LIST, default=["1M"], key="bt_tenors",
        )
    with cc2:
        directions_sel = st.multiselect(
            "Direction(s)", list(DIRECTIONS.keys()),
            default=["Call (up-and-out)"],
            key="bt_directions",
        )
        ko_method_bt_label = st.radio(
            "KO method", ("Payout ratio", "KO delta"),
            index=0, horizontal=True, key="bt_ko_method",
            help=("• **Payout ratio**: H solved so max_payoff/premium hits "
                   "target leverage (closest if infeasible).\n"
                   "• **KO delta**: H placed at vanilla-Δ wing strike "
                   "(same convention as the strike Δ).")
        )
        ko_method_bt = ("ratio" if ko_method_bt_label == "Payout ratio"
                          else "delta")
        if ko_method_bt == "ratio":
            payout_labels_bt = st.multiselect(
                "Payout ratio(s)",
                list(PAYOUT_CHOICES.keys()),
                default=["8:1"], key="bt_payout",
                help="Each selected ratio becomes a separate strategy.",
            )
            payout_ratios_bt = [PAYOUT_CHOICES[lbl]
                                  for lbl in payout_labels_bt]
            ko_delta_labels_bt = []
            ko_delta_values_bt = []
            st.caption("If target ratio is below the strike's achievable "
                        "minimum, the closest achievable structure is used "
                        "(H at the U-shape min). Trades flagged "
                        "`feasible=False`.")
        else:
            ko_delta_labels_bt = st.multiselect(
                "KO Δ (vanilla wing)",
                list(KO_DELTA_CHOICES.keys()),
                default=["10Δ"], key="bt_ko_delta",
                help="Each selected KO Δ becomes a separate strategy.",
            )
            ko_delta_values_bt = [KO_DELTA_CHOICES[lbl]
                                    for lbl in ko_delta_labels_bt]
            payout_labels_bt = []
            payout_ratios_bt = []
            st.caption(f"Barrier H placed at the same-side vanilla wing. "
                        f"Trades flagged `feasible=False` only when "
                        f"KO Δ ≥ strike Δ (degenerate). Achieved leverage "
                        f"varies trade by trade — surfaced in the ledger.")

        tx_cost_bps = st.slider(
            "Transaction cost (bps of notional)", 0.0, 20.0, 2.0, 0.5,
            help=("Flat bps markup on the foreign notional, added to the "
                   "mid premium. 2 bps on $10M notional = $2,000."),
            key="bt_txcost",
        )

        # ---- Entry gate(s) ----
        from core.gates import GATE_REGISTRY
        gate_options_bt = ["(no gate)"] + list(GATE_REGISTRY.keys())
        gate_labels_bt = st.multiselect(
            "Gate(s)",
            gate_options_bt,
            default=["(no gate)"],
            key="bt_gate_keys",
            format_func=lambda k: ("(no gate)" if k == "(no gate)"
                                   else GATE_REGISTRY[k][0]),
            help=("Pick any combination of gates plus '(no gate)' to test "
                   "both gated and ungated variants in one run. Each "
                   "selection becomes its own strategy in the grid."),
        )
        gate_keys_bt = [None if k == "(no gate)" else k
                          for k in gate_labels_bt]
    with cc3:
        # Date range — derive from intersection of available data
        date_max = _date.today()
        date_min = _date(2020, 1, 1)
        try:
            spot_all = load_panel(folder, "SPOT", None)
            if not spot_all.empty:
                date_max = spot_all.index.max().date()
                date_min = max(spot_all.index.min().date(),
                                 date_max - timedelta(days=365 * 10))
        except Exception:
            pass

        default_start = max(date_min, date_max - timedelta(days=365 * 2))
        start_date = st.date_input(
            "Start date", value=default_start,
            min_value=date_min, max_value=date_max, key="bt_start",
        )
        end_date = st.date_input(
            "End date", value=date_max,
            min_value=date_min, max_value=date_max, key="bt_end",
        )
        prefer_em = st.radio(
            "EM pairs variant", ["offshore", "onshore"], index=0,
            horizontal=True, key="bt_prefer_em",
        )
        notional_usd_bt = st.number_input(
            "Notional (USD)", min_value=100_000.0, max_value=200_000_000.0,
            value=10_000_000.0, step=1_000_000.0, format="%.0f",
            key="bt_notional",
            help="Foreign-currency notional. All USD figures (premium, "
                  "payout, PnL, drawdown, tx cost) are computed at this "
                  "size and stored on each trade.",
        )
        enable_mtm = st.checkbox(
            "Mark-to-market mode (daily)",
            value=False, key="bt_mtm",
            help=("Off (default): equity curve is a step function — each "
                   "trade contributes once on its expiry date. "
                   "On: every business day, all open positions are marked "
                   "to mid-vol fair value, so the equity curve moves daily. "
                   "Total PnL is identical in both modes; only the timing "
                   "of paper P&L differs, which materially changes Sharpe "
                   "and max drawdown. Adds ~3-5s per pair to the run."),
        )
        trade_mode_bt = st.radio(
            "Trade mode",
            ["stack", "single"],
            index=0, horizontal=True, key="bt_trade_mode",
            help=("**stack** (default): open a new trade on every "
                   "eligible date — overlapping book, daily rebalancing. "
                   "**single**: open at most one trade per pair at a "
                   "time — next entry only on/after the prior trade's "
                   "expiry. Much lower turnover, cleaner equity curve."),
        )

    # Validate and count strategies
    n_pairs = len(pairs_sel)
    n_deltas = len(deltas_sel)
    n_tenors = len(tenors_sel)
    n_dirs = len(directions_sel)
    # KO axis: number of payout ratios OR number of KO deltas, depending
    # on mode. Gate axis: each selected entry (including "(no gate)") is
    # one strategy variant.
    if ko_method_bt == "ratio":
        n_ko = len(payout_ratios_bt)
        ko_axis_label = "payout ratios"
    else:
        n_ko = len(ko_delta_values_bt)
        ko_axis_label = "KO deltas"
    n_gates = max(len(gate_keys_bt), 1)
    n_specs = n_pairs * n_deltas * n_tenors * n_dirs * n_ko * n_gates

    st.caption(f"**{n_specs}** strategies will run "
                f"({n_pairs} pairs × {n_deltas} deltas × {n_tenors} tenors × "
                f"{n_dirs} directions × {n_ko} {ko_axis_label} × "
                f"{n_gates} gates) "
                f"over {(end_date - start_date).days} calendar days.")

    can_run = (n_specs > 0 and pairs_sel and deltas_sel and tenors_sel
                  and directions_sel and n_ko > 0 and len(gate_keys_bt) > 0)
    run_clicked = st.button("▶ Run backtest", type="primary",
                              disabled=not can_run)

    # ----- Execute on click -----
    if run_clicked:
        # Build specs: cross-product across the per-strategy axes
        # (payout-ratio or KO-delta × gate). The base grid builder
        # handles pair × delta × tenor × direction; we loop the rest
        # here so each (ratio/delta, gate) combination becomes its own
        # named strategy. gate_keys_bt may contain None (= "(no gate)"
        # variant) alongside named gates.
        specs = []
        gate_axis = gate_keys_bt   # already includes None for "(no gate)"
        if ko_method_bt == "ratio":
            ko_axis = [(r, None, None) for r in payout_ratios_bt]
        else:
            ko_axis = [(None, v, lbl)
                         for v, lbl in zip(ko_delta_values_bt,
                                              ko_delta_labels_bt)]
        for ratio_v, kdv, kdl in ko_axis:
            for gk in gate_axis:
                specs += build_strategy_grid(
                    pairs=pairs_sel,
                    deltas=[(d, DELTA_CHOICES[d]) for d in deltas_sel],
                    tenors=tenors_sel,
                    directions=[DIRECTIONS[d] for d in directions_sel],
                    tx_cost_bps=tx_cost_bps,
                    prefer=prefer_em,
                    ko_method=ko_method_bt,
                    payout_ratio=ratio_v,
                    target_ko_delta=kdv,
                    ko_delta_label=kdl,
                    entry_gate=gk,
                    trade_mode=trade_mode_bt,
                    pricing_model=pricing_model,
                )

        progress_bar = st.progress(0.0, text="Starting…")
        t0 = time.time()
        last_update_time = [t0]

        def cb(p, name):
            now = time.time()
            if now - last_update_time[0] > 0.1 or p >= 1.0:
                progress_bar.progress(min(p, 1.0),
                                      text=f"Running: {name} ({p*100:.0f}%)")
                last_update_time[0] = now

        results = run_grid(folder, specs, start_date, end_date,
                            notional_usd=notional_usd_bt,
                            progress_cb=cb)

        # Optional MTM equity curves
        mtm_curves = None
        if enable_mtm:
            def cb_mtm(p, name):
                progress_bar.progress(min(p, 1.0),
                                       text=f"MTM: {name} ({p*100:.0f}%)")
            mtm_curves = compute_mtm_curves(folder, specs, results,
                                              progress_cb=cb_mtm)

        elapsed = time.time() - t0
        progress_bar.empty()

        st.session_state["eko_backtest_results"] = results
        st.session_state["backtest_mtm_curves"] = mtm_curves
        st.session_state["backtest_specs"] = specs
        st.session_state["backtest_elapsed"] = elapsed
        st.session_state["eko_backtest_meta"] = {
            "start": start_date, "end": end_date,
            "ko_method": ko_method_bt,
            "payout_ratios": payout_ratios_bt,
            "payout_labels": payout_labels_bt,
            "ko_delta_labels": ko_delta_labels_bt,
            "ko_delta_values": ko_delta_values_bt,
            "tx_cost_bps": tx_cost_bps,
            "entry_gates": gate_keys_bt,
            "prefer": prefer_em,
            "n_specs": n_specs, "notional_usd": notional_usd_bt,
            "mtm_enabled": enable_mtm,
        }

        st.success(
            f"Done in {elapsed:.1f}s — {n_specs} strategies, "
            f"{sum(len(t) for t in results.values())} trades total."
        )

    # ----- Show summary if results exist -----
    if "eko_backtest_results" in st.session_state:
        st.markdown("---")
        st.markdown("### Summary across strategies")
        results = st.session_state["eko_backtest_results"]
        mtm_curves = st.session_state.get("backtest_mtm_curves")
        meta = st.session_state.get("eko_backtest_meta", {})
        mtm_on = meta.get("mtm_enabled", False) and mtm_curves is not None

        ko_method_meta = meta.get("ko_method", "ratio")
        if ko_method_meta == "delta":
            kd_labels = meta.get("ko_delta_labels", [])
            ko_str = ("KO @ " + ", ".join(kd_labels) + " (vanilla)"
                       if kd_labels else "KO Δ unset")
        else:
            payout_labels = meta.get("payout_labels", [])
            ko_str = ("target leverage " + ", ".join(payout_labels)
                       if payout_labels else "leverage unset")
        from core.gates import gate_label as _gate_lbl
        gate_keys_meta = meta.get("entry_gates", [])
        if gate_keys_meta:
            gate_str = ", ".join(_gate_lbl(g) for g in gate_keys_meta)
        else:
            gate_str = "(none)"
        st.caption(
            f"Run period: {meta.get('start')} → {meta.get('end')}  ·  "
            f"notional ${meta.get('notional_usd', 0):,.0f}  ·  "
            f"{ko_str}  ·  "
            f"tx cost {meta.get('tx_cost_bps', 0):.1f} bps  ·  "
            f"entry gate(s): **{gate_str}**  ·  "
            f"PnL mode: **{'MTM (daily)' if mtm_on else 'Realized at expiry'}**  ·  "
            f"elapsed {st.session_state.get('backtest_elapsed', 0):.1f}s"
        )
        st.caption(
            "💡 Click any column header to sort. For a single-shot "
            "**consistency-aware ranking**, sort by `Sharpe Score` "
            "descending (= μ − σ; rewards high mean AND low yearly "
            "swing). For finer-grained views: `%Pos Yrs` ↓ then "
            "`Sharpe(y) min` ↓ then `Sharpe(y) σ` ↑. `Calmar` and "
            "`G2P` add drawdown-pain and asymmetry views. Note: "
            "`Sharpe(y) μ` alone is NOT a concentration penalty — "
            "see column definitions."
        )

        rows = []
        for name, trades in results.items():
            df = trades_to_df(trades)
            if df.empty:
                rows.append({"Strategy": name, "n trades": 0})
                continue
            s = summarize_strategy(df)
            # When MTM is on, swap in MTM-based Sharpe and DD; total PnL
            # is identical so we keep that.
            if mtm_on:
                mtm_eq = mtm_curves.get(name)
                sm = summarize_mtm(mtm_eq) if mtm_eq is not None else {}
                sharpe_label = "Sharpe (d)"
                sharpe_val = sm.get("sharpe_daily_mtm", 0.0)
                maxdd_val = sm.get("max_drawdown_usd_mtm",
                                     s.get("max_drawdown_usd", 0))
            else:
                sharpe_label = "Sharpe (m)"
                sharpe_val = s["sharpe_monthly"]
                maxdd_val = s.get("max_drawdown_usd", 0)
            rows.append({
                "Strategy": name,
                "n": s["n_trades"],
                "Feas%": f"{s['feasibility_pct']:.0f}",
                "KO%": f"{s['ko_rate_pct']:.0f}",
                "Win%": f"{s['win_rate_pct']:.0f}",
                "Σ Premium": _fmt_usd(s.get("total_premium_usd", 0)),
                "Σ TX Cost": _fmt_usd(s.get("total_transaction_cost_usd", 0)),
                "Σ Payout": _fmt_usd(s.get("total_payout_usd", 0)),
                "Σ PnL": _fmt_usd(s.get("total_pnl_usd", 0)),
                sharpe_label: f"{sharpe_val:+.2f}",
                "Max DD": _fmt_usd(maxdd_val),
                "Recovery%": f"{s['premium_recovery_pct']:.0f}",
                # --- Cross-year consistency block ---
                "Yrs": s.get("n_years", 0),
                "%Pos Yrs": f"{s.get('pct_positive_years', 0):.0f}",
                "Min Ann $": _fmt_usd(s.get("min_annual_pnl_usd", 0)),
                "Sharpe(y) μ": f"{s.get('annual_sharpe_mean', 0):+.2f}",
                "Sharpe(y) min": f"{s.get('annual_sharpe_min', 0):+.2f}",
                "Sharpe(y) σ": f"{s.get('annual_sharpe_std', 0):.2f}",
                "Sharpe(y) CV": (f"{s.get('annual_sharpe_cv', 0):+.2f}"
                                   if s.get('annual_sharpe_cv', 0) != 0
                                   else "—"),
                "Sharpe Score": f"{s.get('annual_sharpe_score', 0):+.2f}",
                "Calmar": f"{s.get('calmar', 0):+.2f}",
                "G2P": (f"{s.get('gain_to_pain', 0):.2f}"
                         if s.get("gain_to_pain", 0) != float("inf")
                         else "∞"),
                "Ulcer": f"{s.get('ulcer_index', 0):.2f}",
            })

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                          hide_index=True)
            with st.expander("Column definitions", expanded=False):
                st.markdown(
                    f"**Headline columns**\n"
                    f"- **Feas%** — {FEASIBILITY_HELP}\n"
                    "- **KO%** — share of trades that knocked out at expiry "
                    "(payoff = 0).\n"
                    "- **Win%** — share of trades with PnL > 0 (net of tx cost).\n"
                    "- **Σ Premium / Σ TX / Σ Payout / Σ PnL** — totals over "
                    "the run, USD-denominated.\n"
                    "- **Sharpe (m)** — realised mode: mean(monthly PnL) / "
                    "std(monthly PnL) × √12. **Sharpe (d)** — same but daily, "
                    "MTM mode.\n"
                    "- **Max DD** — peak-to-trough on the equity curve.\n"
                    "- **Recovery%** — Σ Payout / Σ Premium × 100.\n\n"
                    "**Cross-year consistency columns** (orthogonal info — "
                    "use to rank strategies by 'works across multiple years' "
                    "rather than 'great in one or two years'):\n"
                    "- **Yrs** — calendar years observed.\n"
                    "- **%Pos Yrs** — fraction of those years with positive "
                    "PnL. The most direct measure of cross-year consistency.\n"
                    "- **Min Ann $** — worst calendar-year PnL in USD. Flags "
                    "blow-up years in absolute terms.\n"
                    "- **Sharpe(y) μ** — unweighted average of per-year "
                    "Sharpes. ⚠️ Treats each year equally regardless of "
                    "magnitude or variance, so '9 years of Sharpe=0 + "
                    "1 year of Sharpe=10' and '10 years of Sharpe=1' "
                    "produce *similar* values (~1.0). Useful as a "
                    "scale-invariant average — NOT as a concentration "
                    "penalty. For that, look at Sharpe(y) σ or "
                    "Sharpe(y) min instead.\n"
                    "- **Sharpe(y) min** — worst per-year Sharpe. Directly "
                    "answers 'what's the worst year I'd have lived through?'. "
                    "In the example above: Sharpe=0 vs Sharpe=1 — a clean "
                    "1-unit gap that separates concentrated from consistent.\n"
                    "- **Sharpe(y) σ** — standard deviation of per-year "
                    "Sharpes. Directly measures cross-year stability — "
                    "high σ means risk-adjusted returns swing widely "
                    "year-to-year. In the example: σ ≈ 3.2 (concentrated) "
                    "vs σ ≈ 0.8 (consistent). Low σ + high μ is the goal.\n"
                    "- **Sharpe(y) CV** — coefficient of variation = σ/μ "
                    "(signed). Magnitude tells you how much yearly Sharpes "
                    "swing relative to their average; sign tells you if μ "
                    "is positive (good) or negative (bad). For the same |μ|, "
                    "a smaller |CV| means a more reliable strategy. Shown as "
                    "'—' when |μ| ≈ 0 (CV is undefined).\n"
                    "- **Sharpe Score** — composite from Yavuz Akbay's "
                    "framework: `μ × (1 − CV) = μ − σ`. Rewards high mean "
                    "AND low variance in one number — closer to a 'pick "
                    "the best strategy' ranking criterion than any "
                    "single metric. Use this to sort when you want both "
                    "signals at once.\n"
                    "- **Calmar** — annualised return / |max DD|. "
                    "Pain-adjusted return — punishes deep DDs that "
                    "Sharpe (a stdev) smooths over.\n"
                    "- **G2P** — gain-to-pain: Σ positive monthly returns / "
                    "|Σ negative|. Outlier-robust alternative to Sharpe. "
                    "> 1 = profitable; > 2 = good; > 3 = rare.\n"
                    "- **Ulcer** — RMS drawdown depth across the run. "
                    "Captures both DD depth AND duration; long shallow "
                    "underwater stretches that Sharpe ignores get penalised.\n\n"
                    "⚠️ **Small-sample caveat:** with only 2-3 years of data "
                    "all annual metrics are noisy. A `Sharpe(y) min` of "
                    "−2 based on one year is not the same evidence as one "
                    "based on ten."
                )

        # =====================================================================
        # Downloads: summary table + per-strategy time-series for downstream
        # apps to recreate equity/drawdown/monthly/annual charts.
        #
        # The downloaded summary CSV uses a CANONICAL schema (shared with
        # the Worst-of tab's download) so both can be stacked / analyzed
        # in the same downstream app. The on-screen table above is a
        # formatted view; the CSV is raw numeric.
        # =====================================================================
        # =====================================================================
        # Phase 4: regime-conditioned breakdown.
        # Available when an HMM regime panel is registered for the
        # strategy's pair (set up via app 10 → Tab 7). Lets you see
        # whether a strategy's PnL comes mostly from one regime vs
        # spread across both.
        # =====================================================================
        from core.backtest import summarize_by_regime as _bt_by_regime
        from core.regimes import get_regime_panel, list_registered_pairs
        if list_registered_pairs():
            with st.expander("🧬 Regime breakdown (per HMM state)",
                              expanded=False):
                st.caption(
                    "For each strategy, attributes every trade to the "
                    "HMM-decoded state of its **pair on the trade entry "
                    "date**, then aggregates per state. Tells you whether "
                    "the strategy works because it captures the dominant "
                    "regime (state 0), the broken regime (state 1), or "
                    "both. Only strategies whose pair has a registered "
                    "regime panel appear here — fit & save per-pair "
                    "regimes in app 10's Tab 7 to populate."
                )
                regime_brk_rows = []
                for name, trades in results.items():
                    df_b = trades_to_df(trades)
                    if df_b.empty or "pair" not in df_b.columns:
                        continue
                    pair_ = df_b["pair"].iloc[0]
                    panel = get_regime_panel(pair_)
                    if panel is None:
                        continue
                    brk = _bt_by_regime(df_b, panel)
                    if brk.empty:
                        continue
                    for _, r in brk.iterrows():
                        regime_brk_rows.append({
                            "Strategy": name,
                            "Pair": pair_,
                            "State": int(r["state"]),
                            "n trades": int(r["n_trades"]),
                            "Share %": f"{r['share_of_trades_pct']:.1f}%",
                            "Win %": f"{r['win_rate_pct']:.1f}%",
                            "KO %": f"{r['ko_rate_pct']:.1f}%",
                            "Total PnL $": _fmt_usd(r["total_pnl_usd"]),
                            "Mean PnL/trade $": _fmt_usd(r["mean_pnl_usd"]),
                        })
                if regime_brk_rows:
                    st.dataframe(pd.DataFrame(regime_brk_rows),
                                  use_container_width=True, hide_index=True)
                    st.caption(
                        "💡 State 0 = HMM's dominant cluster (highest stationary "
                        "probability). Sort by `Strategy` → `State` to read "
                        "each strategy's PnL split. A strategy that earns "
                        "mostly in state 0 is regime-dependent; one that "
                        "earns evenly across both is regime-agnostic. "
                        "Use this to validate the `hmm_state_X` gates: a "
                        "gated strategy should show ~100% of trades in the "
                        "gate's target state."
                    )
                else:
                    st.caption("_No strategies have a registered regime "
                                "panel for their pair._")

        st.markdown("### Downloads")
        st.caption(
            "Bulk run results are now download-only — the per-strategy "
            "drilldown tab still shows full charts for any single strategy. "
            "The **time series CSV** contains daily (per-expiry-date), "
            "monthly, and annual rows with `pnl_usd`, `equity_usd`, and "
            "`drawdown_usd` at each period end — enough to recompute any "
            "Sharpe-, drawdown-, or PnL-based ratio at any granularity. "
            "Schema matches the Worst-of tab's CSV (via the `strategy_type` "
            "column) so both can be stacked in the same downstream app."
        )

        # Canonical summary frame (raw numeric, shared schema with WO)
        summary_canon_rows = []
        for name, trades in results.items():
            df_s = trades_to_df(trades)
            if df_s.empty:
                continue
            s = summarize_strategy(df_s)
            g2p = s.get("gain_to_pain", 0.0)
            summary_canon_rows.append({
                # Identifier
                "strategy_name": name,
                "strategy_type": "single",
                "n_trades": int(s.get("n_trades", 0)),
                # Money totals
                "notional_usd": s.get("notional_usd", 0.0),
                "total_premium_paid_usd": s.get("total_premium_usd", 0.0),
                "total_tx_cost_usd": s.get("total_transaction_cost_usd", 0.0),
                "total_payout_usd": s.get("total_payout_usd", 0.0),
                "total_pnl_usd": s.get("total_pnl_usd", 0.0),
                "max_drawdown_usd": s.get("max_drawdown_usd", 0.0),
                # Rates / ratios
                "win_rate_pct": s.get("win_rate_pct", 0.0),
                "premium_recovery_pct": s.get("premium_recovery_pct", 0.0),
                # Sharpe block (all six)
                "sharpe_monthly": s.get("sharpe_monthly", 0.0),
                "annual_sharpe_mean": s.get("annual_sharpe_mean", 0.0),
                "annual_sharpe_min": s.get("annual_sharpe_min", 0.0),
                "annual_sharpe_std": s.get("annual_sharpe_std", 0.0),
                "annual_sharpe_cv": s.get("annual_sharpe_cv", 0.0),
                "annual_sharpe_score": s.get("annual_sharpe_score", 0.0),
                # Cross-year consistency
                "n_years": int(s.get("n_years", 0)),
                "pct_positive_years": s.get("pct_positive_years", 0.0),
                "min_annual_pnl_usd": s.get("min_annual_pnl_usd", 0.0),
                "calmar": s.get("calmar", 0.0),
                "gain_to_pain": (g2p if g2p != float("inf") else np.nan),
                "ulcer_index": s.get("ulcer_index", 0.0),
                # Single-leg-specific (kept at end so common cols line up
                # when stacked with worst-of CSV — WO will have these as NaN)
                "feasibility_pct": s.get("feasibility_pct", 0.0),
                "ko_rate_pct": s.get("ko_rate_pct", 0.0),
                # Worst-of-specific placeholders (NaN for single-leg)
                "leg_a_ko_rate_pct": np.nan,
                "leg_b_ko_rate_pct": np.nan,
                "both_survive_rate_pct": np.nan,
                "structure_vs_min_leg_pct": np.nan,
            })
        summary_canon_df = (pd.DataFrame(summary_canon_rows)
                              if summary_canon_rows else pd.DataFrame())

        # Time-series CSV — long-format daily/monthly/annual for every strategy.
        # When a regime panel is registered for the pair, augment with a
        # `state` column on every row (Phase 4). This is what lets the
        # downstream app build "equity curve by regime" views.
        from core.backtest import augment_time_series_with_regime
        from core.regimes import get_regime_panel
        ts_frames = []
        for name, trades in results.items():
            df_ts = trades_to_df(trades)
            if df_ts.empty:
                continue
            ts = export_strategy_time_series(df_ts)
            if ts.empty:
                continue
            # Look up the pair for this strategy (one pair per single-leg)
            pair_for_strat = (df_ts["pair"].iloc[0]
                                 if "pair" in df_ts.columns else None)
            if pair_for_strat:
                ts = augment_time_series_with_regime(
                    ts, get_regime_panel(pair_for_strat),
                    column_name="state",
                )
            ts.insert(0, "strategy_name", name)
            ts.insert(1, "strategy_type", "single")
            ts_frames.append(ts)
        ts_combined = (pd.concat(ts_frames, ignore_index=True)
                         if ts_frames else pd.DataFrame())

        cdl_a, cdl_b = st.columns(2)
        with cdl_a:
            if not summary_canon_df.empty:
                st.download_button(
                    label=(f"⬇ Download summary table "
                             f"({len(summary_canon_df)} rows, CSV)"),
                    data=summary_canon_df.to_csv(index=False).encode("utf-8"),
                    file_name="eko_backtest_bulk_summary.csv",
                    mime="text/csv",
                    help=("Canonical schema: strategy_name, strategy_type, "
                           "n_trades, money totals, Sharpe block (including "
                           "annual_sharpe_cv and annual_sharpe_score), "
                           "consistency block, then strategy-specific. "
                           "Worst-of-specific columns are NaN here."),
                    use_container_width=True,
                    key="bt_summary_dl",
                )
            else:
                st.caption("_No summary rows yet._")
        with cdl_b:
            if not ts_combined.empty:
                n_strats = ts_combined["strategy_name"].nunique()
                st.download_button(
                    label=(f"⬇ Download time series — {n_strats} strategies × "
                             f"{len(ts_combined):,} rows (CSV)"),
                    data=ts_combined.to_csv(index=False).encode("utf-8"),
                    file_name="eko_backtest_bulk_timeseries.csv",
                    mime="text/csv",
                    help=("Long-format: each row has period_type "
                           "('daily'|'monthly'|'annual'), period_end date, "
                           "pnl_usd, equity_usd, drawdown_usd. Monthly/annual "
                           "rows now carry end-of-period equity & drawdown "
                           "snapshots so DD-based ratios (Calmar, ulcer) "
                           "are recomputable at any granularity."),
                    use_container_width=True,
                    key="bt_timeseries_dl",
                )
            else:
                st.caption("_No time-series data yet._")


# -----------------------------------------------------------------------------
# TAB 3: STRATEGY DRILLDOWN
# -----------------------------------------------------------------------------
def render_drilldown_tab():
    if "eko_backtest_results" not in st.session_state:
        st.info("Run a backtest first (Backtest tab).")
        return

    results = st.session_state["eko_backtest_results"]
    mtm_curves = st.session_state.get("backtest_mtm_curves")
    meta = st.session_state.get("eko_backtest_meta", {})
    mtm_on = meta.get("mtm_enabled", False) and mtm_curves is not None
    notional_label = (f"Notional: ${meta.get('notional_usd', 0):,.0f}"
                       if meta.get("notional_usd") else "")

    names = [n for n in results if results[n]]
    if not names:
        st.warning("All strategies in the latest run produced zero trades.")
        return

    selected = st.selectbox("Strategy", names, index=0, key="dr_select")
    trades = results[selected]
    df = trades_to_df(trades)
    if df.empty:
        st.warning("Empty trade ledger for this strategy.")
        return

    summary = summarize_strategy(df)
    eq = compute_equity_and_drawdown(df)
    has_usd = "pnl_usd" in df.columns

    mtm_eq = mtm_curves.get(selected) if mtm_on else None
    mtm_summary = summarize_mtm(mtm_eq) if (mtm_on and mtm_eq is not None) else {}

    # Diagnostic: see the analogous block in the basket-detail renderer.
    # Without this warning, an MTM-on run with a missing curve shows
    # "Sharpe (d) +0.00 daily MTM × √252" because mtm_summary.get returns
    # the default 0 — extremely confusing.
    if mtm_on and (mtm_eq is None or mtm_eq.empty):
        avail_keys = sorted(mtm_curves.keys()) if mtm_curves else []
        st.warning(
            "MTM toggle is on but no MTM equity curve is available for "
            f"**{selected}** — Sharpe and Max DD will show 0 / NaN. "
            f"Curves keyed in session: {len(avail_keys)} "
            f"(`{', '.join(avail_keys[:3])}"
            f"{'…' if len(avail_keys) > 3 else ''}`). Re-run the "
            "backtest, or inspect `compute_mtm_curves` for the failing "
            "strategy."
        )

    # ----- Headline metrics -----
    st.markdown(f"### {selected}")
    mode_tag = ("<span class='tag-ko'>MTM mode</span>" if mtm_on
                else "<span class='tag-call'>Realized at expiry</span>")
    st.markdown(
        f"{mode_tag}  &nbsp;&nbsp;<span style='color:#8b9bb4'>{notional_label}</span>",
        unsafe_allow_html=True,
    )

    cs = st.columns(6)
    if has_usd:
        if mtm_on:
            sharpe_lbl = "Sharpe (d)"
            sharpe_val = f"{mtm_summary.get('sharpe_daily_mtm', 0):+.2f}"
            sharpe_sub = "daily MTM × √252"
            maxdd_val = _fmt_usd(mtm_summary.get("max_drawdown_usd_mtm",
                                                   summary["max_drawdown_usd"]))
            maxdd_sub = "MTM-based"
        else:
            sharpe_lbl = "Sharpe (m)"
            sharpe_val = f"{summary['sharpe_monthly']:+.2f}"
            sharpe_sub = "monthly × √12"
            maxdd_val = _fmt_usd(summary["max_drawdown_usd"])
            maxdd_sub = "realised, by expiry"

        metrics = [
            ("Trades", f"{summary['n_trades']}", ""),
            ("Total PnL", _fmt_usd(summary["total_pnl_usd"]),
             f"{summary['total_pnl_pct']:+.2f}% notl"),
            (sharpe_lbl, sharpe_val, sharpe_sub),
            ("Max DD", maxdd_val, maxdd_sub),
            ("Win rate", f"{summary['win_rate_pct']:.0f}%",
             f"{int(summary['n_trades'] * summary['win_rate_pct']/100)} winners"),
            ("KO rate", f"{summary['ko_rate_pct']:.0f}%", "barrier hit at expiry"),
        ]
    else:
        metrics = [
            ("Trades", f"{summary['n_trades']}", ""),
            ("Total PnL %", f"{summary['total_pnl_pct']:+.2f}", ""),
            ("Sharpe (m)", f"{summary['sharpe_monthly']:+.2f}", ""),
            ("Max DD %", f"{summary['max_drawdown_pct']:.2f}", ""),
            ("Win rate", f"{summary['win_rate_pct']:.0f}%", ""),
            ("KO rate", f"{summary['ko_rate_pct']:.0f}%", ""),
        ]
    for col, (lbl, val, sub) in zip(cs, metrics):
        col.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>{lbl}</div>"
            f"<div class='metric-value'>{val}</div>"
            f"<div class='metric-sub'>{sub}</div>"
            f"</div>", unsafe_allow_html=True)

    cs2 = st.columns(6)
    if has_usd:
        tx_share = summary.get("tx_cost_share_of_premium_pct", 0)
        metrics2 = [
            ("Σ Premium", _fmt_usd(summary["total_premium_usd"]),
             "at σ_used (mid+tx)"),
            ("Σ TX Cost", _fmt_usd(summary["total_transaction_cost_usd"]),
             f"{tx_share:.1f}% of premium"),
            ("Σ Payout", _fmt_usd(summary.get("total_payout_usd", 0)),
             "realised at expiry"),
            ("Recovery", f"{summary['premium_recovery_pct']:.0f}%",
             "Σ Payout / Σ Premium"),
            ("Best trade", _fmt_usd(summary["best_trade_usd"]), ""),
            ("Worst trade", _fmt_usd(summary["worst_trade_usd"]), ""),
        ]
    else:
        metrics2 = [
            ("Σ Premium %", f"{summary['total_premium_pct']:.2f}", ""),
            ("Σ Payout %", f"{summary['total_payout_pct']:.2f}", ""),
            ("Recovery %", f"{summary['premium_recovery_pct']:.0f}", ""),
            ("Avg prem %", f"{summary['avg_premium_pct']:.3f}", ""),
            ("Best trade %", f"{summary['best_trade_pct']:+.3f}", ""),
            ("Worst trade %", f"{summary['worst_trade_pct']:+.3f}", ""),
        ]
    for col, (lbl, val, sub) in zip(cs2, metrics2):
        col.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>{lbl}</div>"
            f"<div class='metric-value'>{val}</div>"
            f"<div class='metric-sub'>{sub}</div>"
            f"</div>", unsafe_allow_html=True)

    # Feasibility tooltip — visible inline in case user is curious
    if summary["feasibility_pct"] < 100:
        st.caption(
            f"ℹ︎ {summary['feasibility_pct']:.0f}% of trades hit the target "
            f"payout ratio exactly; the rest used the closest achievable "
            f"structure (`feasible=False` in the ledger). "
            f"_{FEASIBILITY_HELP}_"
        )

    # ----- Equity + drawdown chart (USD), realised or MTM -----
    st.markdown("---")
    if mtm_on and mtm_eq is not None and not mtm_eq.empty:
        st.markdown("#### Equity and drawdown — daily MTM")
        eq_x = mtm_eq.index
        eq_y = mtm_eq["equity_usd"]
        dd_y = mtm_eq["drawdown_usd"]
    elif not eq.empty:
        st.markdown("#### Equity and drawdown — realised at expiry")
        eq_x = eq.index
        eq_y = eq["equity_usd"] if has_usd else eq["equity_pct"]
        dd_y = eq["drawdown_usd"] if has_usd else eq["drawdown_pct"]
    else:
        eq_x = eq_y = dd_y = None

    if eq_x is not None and len(eq_x) > 0:
        unit = "USD" if has_usd else "%"
        eq_fmt = "$,.0f" if has_usd else "+.2f"
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq_x, y=eq_y, mode="lines", name="Equity",
            line=dict(color="#38bdf8", width=2), yaxis="y1",
            hovertemplate=("%{x|%Y-%m-%d}<br>Equity: " +
                            ("$%{y:,.0f}" if has_usd else "%{y:+.2f}%") +
                            "<extra></extra>"),
        ))
        fig.add_trace(go.Scatter(
            x=eq_x, y=dd_y, mode="lines", name="Drawdown",
            fill="tozeroy", line=dict(color="#ef4444", width=1.5),
            yaxis="y2",
            hovertemplate=("%{x|%Y-%m-%d}<br>DD: " +
                            ("$%{y:,.0f}" if has_usd else "%{y:.2f}%") +
                            "<extra></extra>"),
        ))
        dd_min = float(dd_y.min())
        fig.update_layout(
            yaxis=dict(title=f"Equity ({unit})",
                        gridcolor="rgba(255,255,255,0.08)",
                        zeroline=True, zerolinecolor="rgba(255,255,255,0.3)",
                        tickformat=eq_fmt),
            yaxis2=dict(title=f"Drawdown ({unit})", overlaying="y", side="right",
                          range=[dd_min * 1.1, 0] if dd_min < 0 else [-1, 0],
                          gridcolor="rgba(0,0,0,0)",
                          tickformat=eq_fmt),
            xaxis=dict(title="Date" if mtm_on else "Expiry date"),
            height=420, template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                          xanchor="left", x=0,
                          font=dict(size=11, color="#cbd5e1")),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ----- Annual + Monthly bar charts (side by side) -----
    annual = annual_summary_table(df)
    cl, cr = st.columns(2)

    with cl:
        st.markdown("#### Annual PnL")
        if not annual.empty:
            yvals = (annual["total_pnl_usd"] if has_usd
                     else annual["total_pnl_pct"])
            tx_vals = (annual["total_tx_cost_usd"]
                        if has_usd and "total_tx_cost_usd" in annual.columns
                        else None)
            fig_y = go.Figure()
            fig_y.add_trace(go.Bar(
                x=annual.index.astype(str), y=yvals,
                marker_color=["#86efac" if v >= 0 else "#fca5a5" for v in yvals],
                name="PnL (net of tx)",
                hovertemplate=("%{x}<br>PnL: " +
                                ("$%{y:,.0f}" if has_usd else "%{y:+.2f}%") +
                                "<extra></extra>"),
                text=[(_fmt_usd(v) if has_usd else f"{v:+.2f}%") for v in yvals],
                textposition="outside",
            ))
            if tx_vals is not None:
                # TX cost as a separate orange bar (negative-direction so
                # it visually stacks with negative PnL)
                fig_y.add_trace(go.Bar(
                    x=annual.index.astype(str), y=-tx_vals,
                    marker_color="#fb923c",
                    name="TX cost (paid)",
                    hovertemplate="%{x}<br>TX cost: $%{customdata:,.0f}<extra></extra>",
                    customdata=tx_vals,
                    opacity=0.7,
                ))
            fig_y.update_layout(
                barmode="group",
                yaxis=dict(title=("PnL (USD)" if has_usd else "PnL (%)"),
                            gridcolor="rgba(255,255,255,0.08)",
                            zeroline=True,
                            zerolinecolor="rgba(255,255,255,0.3)",
                            tickformat=("$,.0f" if has_usd else "+.1f")),
                xaxis=dict(title="Year"),
                height=320, template="plotly_dark",
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                              xanchor="left", x=0,
                              font=dict(size=10, color="#cbd5e1")),
            )
            st.plotly_chart(fig_y, use_container_width=True)

    with cr:
        st.markdown("#### Monthly PnL")
        if not eq.empty:
            m_col = "pnl_usd" if has_usd else "pnl_pct"
            m_pnl = eq[m_col].resample("ME").sum()
            if not m_pnl.empty:
                fig_m = go.Figure()
                fig_m.add_trace(go.Bar(
                    x=m_pnl.index, y=m_pnl.values,
                    marker_color=["#86efac" if v >= 0 else "#fca5a5"
                                    for v in m_pnl.values],
                    hovertemplate=("%{x|%b %Y}<br>PnL: " +
                                    ("$%{y:,.0f}" if has_usd else "%{y:+.2f}%") +
                                    "<extra></extra>"),
                ))
                fig_m.update_layout(
                    yaxis=dict(title=("PnL (USD)" if has_usd else "PnL (%)"),
                                gridcolor="rgba(255,255,255,0.08)",
                                zeroline=True,
                                zerolinecolor="rgba(255,255,255,0.3)",
                                tickformat=("$,.0f" if has_usd else "+.1f")),
                    xaxis=dict(title="Month"),
                    height=320, template="plotly_dark",
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False,
                )
                st.plotly_chart(fig_m, use_container_width=True)

    # ----- Annual stats table (the secondary/non-PnL stuff) -----
    st.markdown("#### Annual stats")
    if not annual.empty:
        cols_to_show = ["n_trades", "win_rate_pct", "ko_rate_pct",
                          "feasibility_pct", "sharpe_monthly"]
        if has_usd:
            cols_to_show = ["n_trades", "total_pnl_usd",
                              "total_premium_usd", "total_tx_cost_usd",
                              "total_payout_usd",
                              "win_rate_pct", "ko_rate_pct",
                              "feasibility_pct", "sharpe_monthly"]
        sub = annual[cols_to_show].copy()
        rename = {
            "n_trades": "n",
            "total_pnl_usd": "Σ PnL",
            "total_premium_usd": "Σ Premium",
            "total_tx_cost_usd": "Σ TX Cost",
            "total_payout_usd": "Σ Payout",
            "win_rate_pct": "Win%",
            "ko_rate_pct": "KO%",
            "feasibility_pct": "Feas%",
            "sharpe_monthly": "Sharpe (m)",
        }
        sub = sub.rename(columns=rename)
        # Format the USD columns with _fmt_usd, percent columns as numbers
        if has_usd:
            for c in ("Σ PnL", "Σ Premium", "Σ TX Cost", "Σ Payout"):
                if c in sub.columns:
                    sub[c] = sub[c].apply(_fmt_usd)
        for c in ("Win%", "KO%", "Feas%"):
            if c in sub.columns:
                sub[c] = sub[c].round(0).astype(int).astype(str)
        if "Sharpe (m)" in sub.columns:
            sub["Sharpe (m)"] = sub["Sharpe (m)"].round(2)
        st.dataframe(sub, use_container_width=True)

    # ----- Cumulative TX cost trajectory (separate chart, USD) -----
    if has_usd and "cum_tx_cost_usd" in eq.columns:
        st.markdown("#### Cumulative transaction cost paid")
        st.caption("How much of the equity-curve performance went to "
                    "transaction fees (i.e. premium paid above mid-vol fair "
                    "value).")
        fig_tx = go.Figure()
        fig_tx.add_trace(go.Scatter(
            x=eq.index, y=eq["cum_tx_cost_usd"], mode="lines",
            fill="tozeroy",
            line=dict(color="#fb923c", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>Cum TX: $%{y:,.0f}<extra></extra>",
        ))
        fig_tx.update_layout(
            yaxis=dict(title="Cumulative TX cost (USD)",
                        gridcolor="rgba(255,255,255,0.08)",
                        zeroline=True, zerolinecolor="rgba(255,255,255,0.3)",
                        tickformat="$,.0f"),
            xaxis=dict(title="Expiry date"),
            height=240, template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
        )
        st.plotly_chart(fig_tx, use_container_width=True)

    # ----- Spot + gate indicator + entry markers -----
    st.markdown("#### Spot, gate indicator & trade entries")
    pair_dr = df["pair"].iloc[0]
    gate_key_dr = (df["entry_gate"].iloc[0]
                    if "entry_gate" in df.columns else None)
    prefer_dr = meta.get("prefer", "offshore")  # for EM pairs

    # Load the pair's spot panel covering the trade window + 1y buffer
    # (200DMA / 252d-pct gates need at least 200-252 prior business days)
    from datetime import timedelta as _td
    chart_start = df["trade_date"].min() - _td(days=300)
    chart_end = df["expiry_date"].max() + _td(days=5)
    spot_panel = load_panel(folder, "SPOT", None, prefer=prefer_dr,
                              pairs=(pair_dr,))
    if not spot_panel.empty and pair_dr in spot_panel.columns:
        spot_full = spot_panel[pair_dr].dropna()
        spot_s = spot_full.loc[(spot_full.index >= pd.Timestamp(chart_start))
                                  & (spot_full.index <= pd.Timestamp(chart_end))]

        from core.gates import (gate_chart_layers, gate_label as _glab,
                                   compute_gate_mask)
        layers = gate_chart_layers(spot_full, gate_key_dr)
        # Crop ALL line series to chart window (both panels)
        for ln in layers["price_lines"] + layers["subplot_lines"]:
            ln["series"] = ln["series"].reindex(spot_s.index)
        gate_mask_full = layers["mask"]
        gate_mask_window = (gate_mask_full.reindex(spot_s.index)
                             if gate_mask_full is not None else None)

        entry_dates = pd.to_datetime(df["trade_date"]).dt.normalize()
        entry_spots = pd.Series(spot_s).reindex(entry_dates).dropna()

        # Build figure — single panel for trend gates, 2-row subplot for
        # vol-regime AND combo gates (combos overlay MA on spot panel AND
        # show realized vol in the indicator subplot).
        from plotly.subplots import make_subplots
        needs_subplot = layers["panel"] in ("subplot", "both")
        if needs_subplot:
            fig_g = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                     row_heights=[0.65, 0.35],
                                     vertical_spacing=0.05)
            spot_row = 1
            ind_row = 2
        else:
            fig_g = go.Figure()
            spot_row = ind_row = None

        # Shade gate-active regions on both panels (works for ALL gates
        # since `mask` is gate-agnostic). The end index lands on the FIRST
        # 0-day after a run, so the rect extends 1 business day past the
        # actual transition — a cosmetic overshoot we accept for clarity.
        if gate_key_dr and gate_mask_window is not None:
            on = gate_mask_window.fillna(False).astype(int)
            edges = on.diff().fillna(on)
            start_idx = on.index[edges == 1].tolist()
            end_idx = on.index[edges == -1].tolist()
            if not on.empty and on.iloc[-1] == 1:
                end_idx.append(on.index[-1])
            for s_, e_ in zip(start_idx, end_idx):
                if needs_subplot:
                    fig_g.add_vrect(x0=s_, x1=e_, fillcolor="#22c55e",
                                      opacity=0.10, line_width=0,
                                      layer="below", row=1, col=1)
                    fig_g.add_vrect(x0=s_, x1=e_, fillcolor="#22c55e",
                                      opacity=0.10, line_width=0,
                                      layer="below", row=2, col=1)
                else:
                    fig_g.add_vrect(x0=s_, x1=e_, fillcolor="#22c55e",
                                      opacity=0.10, line_width=0,
                                      layer="below")

        # Spot line + entries always on top panel
        def _add(trace, row=None):
            if row is not None:
                fig_g.add_trace(trace, row=row, col=1)
            else:
                fig_g.add_trace(trace)

        _add(go.Scatter(
            x=spot_s.index, y=spot_s.values, mode="lines", name="Spot",
            line=dict(color="#38bdf8", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>Spot: %{y:.4f}<extra></extra>",
        ), row=spot_row)

        # Price-overlay lines (MAs) — go on spot panel
        for ln in layers["price_lines"]:
            _add(go.Scatter(
                x=ln["series"].index, y=ln["series"].values, mode="lines",
                name=ln["name"],
                line=dict(color=ln["color"], width=1.5, dash=ln["dash"]),
                hovertemplate="%{x|%Y-%m-%d}<br>" + ln["name"] +
                                ": %{y:.4f}<extra></extra>",
            ), row=spot_row)

        # Indicator-subplot lines (RV) — go on bottom panel
        for ln in layers["subplot_lines"]:
            _add(go.Scatter(
                x=ln["series"].index, y=ln["series"].values, mode="lines",
                name=ln["name"],
                line=dict(color=ln["color"], width=1.5, dash=ln["dash"]),
                hovertemplate="%{x|%Y-%m-%d}<br>" + ln["name"] +
                                ": %{y:.4f}<extra></extra>",
            ), row=ind_row)

        # Entry markers on the spot panel
        if len(entry_spots) > 0:
            _add(go.Scatter(
                x=entry_spots.index, y=entry_spots.values, mode="markers",
                name=f"Entries ({len(entry_spots)})",
                marker=dict(color="#a3e635", size=6, symbol="diamond",
                              line=dict(color="#365314", width=0.5)),
                hovertemplate=("%{x|%Y-%m-%d}<br>Spot @ entry: %{y:.4f}"
                                 "<extra></extra>"),
            ), row=spot_row)

        gate_caption = (f"Green shading = `{_glab(gate_key_dr)}` "
                          f"satisfied (entries allowed)."
                          if gate_key_dr
                          else "No entry gate — trades on every business day.")
        st.caption(f"Pair: **{pair_dr}**  ·  {gate_caption}  ·  "
                    f"trades shown: **{len(entry_spots)}** of {len(df)} "
                    f"(some entries may pre-date the spot buffer).")

        if needs_subplot:
            fig_g.update_yaxes(title=f"Spot ({pair_dr})",
                                  gridcolor="rgba(255,255,255,0.08)",
                                  row=1, col=1)
            fig_g.update_yaxes(title=layers["subplot_title"],
                                  gridcolor="rgba(255,255,255,0.08)",
                                  ticksuffix="%", row=2, col=1)
            fig_g.update_xaxes(title="Trade date",
                                  gridcolor="rgba(255,255,255,0.08)",
                                  row=2, col=1)
            fig_g.update_layout(height=460)
        else:
            fig_g.update_layout(
                xaxis=dict(title="Trade date",
                              gridcolor="rgba(255,255,255,0.08)"),
                yaxis=dict(title=f"Spot ({pair_dr})",
                              gridcolor="rgba(255,255,255,0.08)"),
                height=320,
            )
        fig_g.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                          xanchor="left", x=0,
                          font=dict(size=11, color="#cbd5e1")),
        )
        st.plotly_chart(fig_g, use_container_width=True)
    else:
        st.info(f"No spot panel available for {pair_dr} — chart skipped.")

    # ----- Per-trade distribution + scatter (USD) -----
    cl, cr = st.columns(2)
    with cl:
        st.markdown("#### Per-trade PnL distribution")
        col_x = "pnl_usd" if has_usd else "pnl_pct"
        x_label = "PnL (USD)" if has_usd else "PnL (%)"
        x_fmt = "$,.0f" if has_usd else "+.2f"
        mean_v = float(df[col_x].mean())
        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(
            x=df[col_x], nbinsx=50, marker_color="#38bdf8", opacity=0.85,
        ))
        fig_h.add_vline(x=0, line=dict(color="#9aa1ad", dash="dot"))
        fig_h.add_vline(x=mean_v, line=dict(color="#facc15", dash="dash"),
                          annotation_text=("mean = " +
                                            (_fmt_usd(mean_v) if has_usd
                                             else f"{mean_v:+.3f}%")))
        fig_h.update_layout(
            xaxis=dict(title=x_label, tickformat=x_fmt),
            yaxis=dict(title="Count"),
            height=300, template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            margin=dict(l=20, r=20, t=20, b=20), showlegend=False,
        )
        st.plotly_chart(fig_h, use_container_width=True)
    with cr:
        st.markdown("#### Premium vs realised payout")
        x_col = "premium_usd" if has_usd else "premium_pct"
        y_col = "actual_payoff_usd" if has_usd else "actual_payoff_pct"
        unit = "USD" if has_usd else "%"
        s_fmt = "$,.0f" if has_usd else "+.2f"
        ko_mask = df["knocked_out"]
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            x=df.loc[~ko_mask, x_col], y=df.loc[~ko_mask, y_col],
            mode="markers", name="not KO",
            marker=dict(color="#86efac", size=4, opacity=0.5),
        ))
        fig_s.add_trace(go.Scatter(
            x=df.loc[ko_mask, x_col], y=df.loc[ko_mask, y_col],
            mode="markers", name="KO",
            marker=dict(color="#fca5a5", size=4, opacity=0.5),
        ))
        max_v = max(df[x_col].max(), df[y_col].max())
        fig_s.add_trace(go.Scatter(
            x=[0, max_v], y=[0, max_v], mode="lines",
            line=dict(color="#fb923c", dash="dash", width=1),
            name="break-even (payout = premium)",
        ))
        fig_s.update_layout(
            xaxis=dict(title=f"Premium paid ({unit})", tickformat=s_fmt),
            yaxis=dict(title=f"Realised payoff ({unit})", tickformat=s_fmt),
            height=300, template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3,
                          xanchor="left", x=0,
                          font=dict(size=10, color="#cbd5e1")),
        )
        st.plotly_chart(fig_s, use_container_width=True)

    # ----- Trade ledger + download -----
    st.markdown("---")
    st.markdown("#### Trade ledger")
    st.caption(
        "**PnL accounting:**  "
        "`premium_paid = premium_mid + tx_cost`,  "
        "`pnl = payoff − premium_paid`  (already net of tx cost — "
        "do **not** subtract tx_cost again).  "
        "USD columns first; % columns retained for reference. "
        "The downloadable CSV contains the full record (every column, all fields)."
    )
    if has_usd:
        display_cols = [
            "trade_date", "expiry_date", "pair", "tenor_label", "delta_label",
            "ko_method", "target_payout_ratio", "target_ko_delta",
            "spot", "strike", "barrier", "spot_at_expiry", "knocked_out",
            "achieved_payout_ratio", "feasible",
            "premium_mid_usd", "transaction_cost_usd", "premium_usd",
            "max_payoff_usd", "actual_payoff_usd", "pnl_usd",
            "sigma_atm", "rr_25", "bf_25", "sigma_smile", "r_d", "r_f",
        ]
        # Friendlier display labels — clarifies premium_usd = paid (mid + tx)
        display_rename = {
            "premium_mid_usd":       "premium_mid_usd  (a)",
            "transaction_cost_usd":  "tx_cost_usd  (b)",
            "premium_usd":           "premium_paid_usd  (a+b)",
            "actual_payoff_usd":     "payoff_usd  (c)",
            "pnl_usd":               "pnl_usd  (= c − a − b)",
        }
    else:
        display_cols = [
            "trade_date", "expiry_date", "pair", "tenor_label", "delta_label",
            "ko_method", "target_payout_ratio", "target_ko_delta",
            "spot", "strike", "barrier", "spot_at_expiry", "knocked_out",
            "achieved_payout_ratio", "feasible",
            "premium_pct", "max_payoff_pct", "actual_payoff_pct", "pnl_pct",
            "sigma_atm", "rr_25", "bf_25", "sigma_smile", "r_d", "r_f",
        ]
        display_rename = {}
    # Drop columns not present (e.g. older trade records may lack ko_method)
    display_cols = [c for c in display_cols if c in df.columns]
    show = df[display_cols].copy()
    for c in ("sigma_atm", "rr_25", "bf_25", "sigma_smile", "sigma_mid",
                "sigma_used"):
        if c in show.columns:
            show[c] = (show[c] * 100).round(3)
    show["r_d"] = (show["r_d"] * 100).round(4)
    show["r_f"] = (show["r_f"] * 100).round(4)
    for c in ("spot", "strike", "barrier", "spot_at_expiry"):
        show[c] = show[c].round(4)
    if "achieved_payout_ratio" in show.columns:
        show["achieved_payout_ratio"] = show["achieved_payout_ratio"].round(2)
    for c in ("premium_mid_usd", "premium_usd", "transaction_cost_usd",
                "max_payoff_usd", "actual_payoff_usd", "pnl_usd",
                "premium_pct", "max_payoff_pct", "actual_payoff_pct",
                "pnl_pct"):
        if c in show.columns:
            show[c] = show[c].round(2 if "usd" in c else 4)

    show = show.rename(columns=display_rename)
    st.dataframe(show, use_container_width=True, hide_index=True, height=360)

    csv = df.to_csv(index=False).encode("utf-8")
    safe_name = selected.replace(" ", "_").replace(":", "x").replace("×", "x")
    st.download_button(
        label="⬇ Download full ledger (CSV)",
        data=csv,
        file_name=f"eko_backtest_{safe_name}.csv",
        mime="text/csv",
    )



# -----------------------------------------------------------------------------
# TAB 4: WORST-OF PRICER (single-trade, correlation-aware)
# -----------------------------------------------------------------------------
# Replaces the legacy `multiplier × min(P_A, P_B)` approximation with the
# new closed-form (semi-CF via 1D quadrature) and Monte Carlo joint
# pricers from `core.worstof_pricer`. Designed for live valuation; the
# legacy multiplier is still in use in the Backtest engine and will be
# replaced in a subsequent step.
#
# Per-leg vol uses σ_smile(K) (the "vol-at-strike" treatment). The
# structure-level Vanna-Volga correction for worst-of is a separate
# follow-up.
# -----------------------------------------------------------------------------
def _wop_resolve_leg(folder: str, leg_label: str,
                      pair: str, prefer: str,
                      direction_label: str, strike_delta_label: str,
                      ko_delta_label: str,
                      val_date, tenor_label: str,
                      ) -> "dict | None":
    """Load market data + solve strike/barrier for one worst-of leg.

    Returns a dict with all the inputs the WorstOfLeg constructor needs
    plus diagnostics for the per-leg display block, or None on data
    error (with an error pushed to streamlit).

    Uses the delta-based KO method (vanilla wing strike for the
    barrier) since that's the dominant worst-of convention.
    """
    # Spot
    spot_df = load_panel(folder, "SPOT", None, prefer=prefer, pairs=(pair,))
    if spot_df.empty or pair not in spot_df.columns:
        st.error(f"Leg {leg_label} ({pair}): no SPOT data.")
        return None
    spot_ts = spot_df[pair].dropna()
    val_ts = pd.Timestamp(val_date)
    valid = spot_ts.loc[:val_ts]
    if valid.empty:
        st.error(f"Leg {leg_label} ({pair}): no spot at or before {val_date}.")
        return None
    S = float(valid.iloc[-1])

    # Tenor T
    opt_dates = compute_option_dates(val_date, tenor_label)
    T = opt_dates.T_years

    # Vols
    sigma_pct = get_pair_value_at_T(folder, pair, prefer, "VOL_ATM", T, val_date)
    if sigma_pct is None:
        st.error(f"Leg {leg_label} ({pair}): no VOL_ATM at {tenor_label}.")
        return None
    sigma_atm = sigma_pct / 100.0
    rr_pct = get_pair_value_at_T(folder, pair, prefer, "VOL_25R", T, val_date)
    bf_pct = get_pair_value_at_T(folder, pair, prefer, "VOL_25B", T, val_date)
    rr_25 = (rr_pct / 100.0) if rr_pct is not None else 0.0
    bf_25 = (bf_pct / 100.0) if bf_pct is not None else 0.0
    smile_available = (rr_pct is not None) and (bf_pct is not None)

    # Rates
    for_ccy, dom_ccy = pair[:3].upper(), pair[3:].upper()
    f_panel = _load_rates_panel_cached(folder, for_ccy)
    d_panel = _load_rates_panel_cached(folder, dom_ccy)
    r_f_market = get_rate_at(f_panel, T, val_date)
    r_d_market = get_rate_at(d_panel, T, val_date)

    # Fallback chain — same logic as the single-leg pricer, kept inline
    # so this helper is self-contained. If a rate isn't available we
    # use the other leg's data or punt to 3% to avoid blocking.
    if r_f_market is None:
        r_f = 0.03
        r_f_source = "manual default (3%) — no FOR OIS data"
    else:
        r_f, r_f_source = r_f_market, f"{for_ccy} OIS interp"
    if r_d_market is not None:
        r_d, r_d_source = r_d_market, f"{dom_ccy} OIS interp"
    else:
        # Try CIP from forward
        fwd_pts = get_pair_value_at_T(folder, pair, prefer, "FWD_POINTS",
                                          T, val_date)
        if fwd_pts is not None:
            F_market = S + fwd_pts * get_pip_scale(pair)
            r_d = r_f + np.log(F_market / S) / T
            r_d_source = "implied from forward (CIP)"
        else:
            r_d, r_d_source = r_f, "= r_f (no fwd or DOM OIS)"

    # Resolve direction and per-leg deltas
    option_type, barrier_type = DIRECTIONS[direction_label]
    strike_delta = DELTA_CHOICES[strike_delta_label]
    ko_delta = KO_DELTA_CHOICES[ko_delta_label]

    # Solve K (vanilla Δ) and H (KO Δ wing) — smile-aware if RR/BF data exists
    K, H, info = solve_strike(
        option_type, barrier_type, strike_delta,
        S, T, sigma_atm, r_d, r_f,
        target_ko_delta=ko_delta,
        ko_method="delta",
        rr_25=rr_25, bf_25=bf_25,
    )
    sigma_smile = float(info.get("sigma_smile", sigma_atm))

    # Single-leg EKO mid at σ_smile (the "vol-at-strike" treatment).
    ko_per_unit = ko_price(option_type, barrier_type, S, K, H, T,
                            sigma_smile, r_d, r_f)
    ko_prob = ko_probability(barrier_type, S, H, T, sigma_smile, r_d, r_f)

    return {
        "pair": pair,
        "prefer": prefer,
        "for_ccy": for_ccy,
        "dom_ccy": dom_ccy,
        "S": S, "K": K, "H": H,
        "T": T,
        "sigma_atm": sigma_atm,
        "rr_25": rr_25, "bf_25": bf_25, "sigma_smile": sigma_smile,
        "smile_available": smile_available,
        "r_d": r_d, "r_f": r_f,
        "r_d_source": r_d_source, "r_f_source": r_f_source,
        "option_type": option_type, "barrier_type": barrier_type,
        "strike_delta_label": strike_delta_label,
        "ko_delta_label": ko_delta_label,
        "ko_per_unit": float(ko_per_unit),
        "ko_prob": float(ko_prob),
        "spot_ts": spot_ts,
        "info": info,
    }


def _wop_pair_share_ccy(pair_a: str, pair_b: str) -> bool:
    """True iff pair_a and pair_b share a currency (DOM or FOR)."""
    if len(pair_a) != 6 or len(pair_b) != 6:
        return False
    a = {pair_a[:3].upper(), pair_a[3:].upper()}
    b = {pair_b[:3].upper(), pair_b[3:].upper()}
    return bool(a & b)


def _wop_compute_rolling_corr(spot_a: pd.Series, spot_b: pd.Series,
                                val_date, window: int = 60
                                ) -> "tuple[float | None, int]":
    """Thin wrapper around core.correlation.realized_correlation_at,
    kept for callers that already use this name."""
    from core.correlation import realized_correlation_at
    return realized_correlation_at(spot_a, spot_b, val_date, window=window)


def render_worstof_pricer_tab():
    """Single-trade worst-of two-leg EKO pricer.

    Uses the new correlation-aware closed-form (semi-CF via 1D
    quadrature) and Monte Carlo pricers from core.worstof_pricer.
    Shown alongside the legacy multiplier approximations for direct
    comparison.

    Out of scope for this initial version:
        - Worst-of Greeks (delta, vega, ρ-sensitivity)
        - Structure-level Vanna-Volga smile correction
        - FX-triangulation implied correlation
        - Replacing the multiplier inside the Backtest engine
    """
    pairs_avail = _list_pairs(folder)
    if len(pairs_avail) < 2:
        st.error("Need at least 2 pairs in the data folder.")
        return

    st.markdown("### Worst-of two-leg EKO — single-trade pricer")
    st.caption(
        "Joint correlation-aware pricing of two-leg worst-of European "
        "barrier KO structures. Each leg is priced under its natural "
        "FX measure; the structure is discounted at leg A's DOM rate "
        "(the assumed numeraire). The legacy "
        "`multiplier × min(P_A, P_B)` approximation is shown alongside "
        "for comparison. Strike Δ defines K and KO Δ defines H, both "
        "via vanilla wing-strike solver (same convention as the "
        "single-leg Pricer tab)."
    )

    # =========================================================================
    # Shared controls — tenor, notional
    # =========================================================================
    cc1, cc2 = st.columns([1, 1])
    with cc1:
        tenor_label = st.selectbox("Tenor (shared across legs)",
                                     TENOR_LIST, index=0, key="wop_tenor")
    with cc2:
        notional_usd = st.number_input(
            "Notional (USD, per leg)",
            min_value=100_000.0, max_value=200_000_000.0,
            value=10_000_000.0, step=1_000_000.0, format="%.0f",
            key="wop_notional",
        )

    # =========================================================================
    # Two leg input blocks side-by-side
    # =========================================================================
    leg_a_col, leg_b_col = st.columns(2)

    def _leg_inputs(label: str, col, default_pair_priority: list[str], prefix: str):
        """Render one leg's input widgets. Returns a config dict."""
        with col:
            st.markdown(f"#### Leg {label}")
            default_pair = next(
                (p for p in default_pair_priority if p in pairs_avail),
                pairs_avail[0],
            )
            pair = st.selectbox(
                "Pair", pairs_avail,
                index=pairs_avail.index(default_pair),
                key=f"{prefix}pair",
            )
            asia_em = pair in {"USDCNH", "USDCNY", "USDINR", "USDIDR",
                                "USDKRW", "USDMYR", "USDPHP", "USDTHB",
                                "USDTWD"}
            prefer = (st.radio("Variant", ["offshore", "onshore"], index=0,
                                horizontal=True, key=f"{prefix}prefer")
                      if asia_em else "offshore")
            direction_label = st.radio("Direction", list(DIRECTIONS.keys()),
                                          index=0, key=f"{prefix}dir")
            strike_delta_label = st.radio(
                "Strike Δ", list(DELTA_CHOICES.keys()),
                index=0, horizontal=True, key=f"{prefix}strike_delta",
            )
            ko_delta_label = st.radio(
                "KO Δ (vanilla wing)", list(KO_DELTA_CHOICES.keys()),
                index=1, horizontal=True, key=f"{prefix}ko_delta",
                help="Smaller Δ ⇒ deeper OTM ⇒ barrier further from spot.",
            )
        return {
            "pair": pair, "prefer": prefer,
            "direction_label": direction_label,
            "strike_delta_label": strike_delta_label,
            "ko_delta_label": ko_delta_label,
        }

    leg_a_cfg = _leg_inputs("A", leg_a_col,
                              ["USDJPY", "EURUSD", "USDMXN"], "wop_a_")
    leg_b_cfg = _leg_inputs("B", leg_b_col,
                              ["USDMXN", "AUDUSD", "EURUSD", "USDJPY"],
                              "wop_b_")

    if leg_a_cfg["pair"] == leg_b_cfg["pair"]:
        st.warning(
            "Both legs use the same pair. The structure degenerates "
            "to the cheaper of two single-leg options at correlation = 1. "
            "Set correlation = 1 (manual) to verify."
        )

    # =========================================================================
    # Trade date — use intersection of leg A and leg B spot availability
    # =========================================================================
    spot_a_df = load_panel(folder, "SPOT", None,
                              prefer=leg_a_cfg["prefer"],
                              pairs=(leg_a_cfg["pair"],))
    spot_b_df = load_panel(folder, "SPOT", None,
                              prefer=leg_b_cfg["prefer"],
                              pairs=(leg_b_cfg["pair"],))
    if (spot_a_df.empty or leg_a_cfg["pair"] not in spot_a_df.columns or
            spot_b_df.empty or leg_b_cfg["pair"] not in spot_b_df.columns):
        st.error("Missing SPOT data for one or both legs.")
        return

    spot_a_ts = spot_a_df[leg_a_cfg["pair"]].dropna()
    spot_b_ts = spot_b_df[leg_b_cfg["pair"]].dropna()
    common_idx = spot_a_ts.index.intersection(spot_b_ts.index)
    if len(common_idx) == 0:
        st.error("No overlapping spot dates between leg A and leg B.")
        return
    min_date = common_idx.min().date()
    max_date = common_idx.max().date()
    val_date = st.date_input(
        "Trade date",
        value=max_date, min_value=min_date, max_value=max_date,
        key="wop_val_date",
    )

    # =========================================================================
    # Resolve each leg (load vols, rates, solve K/H, compute single-leg EKO)
    # =========================================================================
    leg_a_resolved = _wop_resolve_leg(
        folder, "A",
        leg_a_cfg["pair"], leg_a_cfg["prefer"],
        leg_a_cfg["direction_label"],
        leg_a_cfg["strike_delta_label"],
        leg_a_cfg["ko_delta_label"],
        val_date, tenor_label,
    )
    leg_b_resolved = _wop_resolve_leg(
        folder, "B",
        leg_b_cfg["pair"], leg_b_cfg["prefer"],
        leg_b_cfg["direction_label"],
        leg_b_cfg["strike_delta_label"],
        leg_b_cfg["ko_delta_label"],
        val_date, tenor_label,
    )
    if leg_a_resolved is None or leg_b_resolved is None:
        return  # error already shown

    # =========================================================================
    # Per-leg detail cards
    # =========================================================================
    cd_a, cd_b = st.columns(2)
    for (col, lr, cfg, label) in [
        (cd_a, leg_a_resolved, leg_a_cfg, "A"),
        (cd_b, leg_b_resolved, leg_b_cfg, "B"),
    ]:
        with col:
            ko_usd = lr["ko_per_unit"] / lr["S"] * notional_usd
            smile_tag = "" if lr["smile_available"] else " (flat ATM — no RR/BF data)"
            st.markdown(
                f"**Leg {label}: {lr['pair']} "
                f"{lr['option_type']} {lr['barrier_type'].replace('_', '-')}** \n"
                f"&nbsp;&nbsp;S = `{lr['S']:.5f}`  ·  K = `{lr['K']:.5f}`  ·  "
                f"H = `{lr['H']:.5f}` \n"
                f"&nbsp;&nbsp;σ_atm = `{lr['sigma_atm']*100:.3f}%`{smile_tag}  ·  "
                f"σ_smile(K) = `{lr['sigma_smile']*100:.3f}%` \n"
                f"&nbsp;&nbsp;r_d ({lr['dom_ccy']}) = `{lr['r_d']*100:.3f}%`  ·  "
                f"r_f ({lr['for_ccy']}) = `{lr['r_f']*100:.3f}%` \n"
                f"&nbsp;&nbsp;Single-leg EKO mid: **{_fmt_usd(ko_usd)}**  "
                f"({abs(ko_usd)/notional_usd*100:.3f}% notl)  ·  "
                f"P(KO) = `{lr['ko_prob']*100:.1f}%`"
            )

    # Numeraire check — warn if leg DOMs differ
    if leg_a_resolved["dom_ccy"] != leg_b_resolved["dom_ccy"]:
        st.warning(
            f"⚠ Numeraire mismatch: leg A DOM = `{leg_a_resolved['dom_ccy']}`, "
            f"leg B DOM = `{leg_b_resolved['dom_ccy']}`. "
            f"Pricer treats leg A's `r_d` (`{leg_a_resolved['dom_ccy']}`) as the "
            f"structure-level discount/numeraire rate. Each leg drifts under "
            f"its own natural-measure (quanto-ignored). For short tenors "
            f"(<6M) this typically introduces <5% pricing error; for longer "
            f"tenors or large σ√T, consider inverting one of the pairs to "
            f"match DOMs."
        )

    # =========================================================================
    # Correlation
    # =========================================================================
    st.markdown("---")
    st.markdown("**Correlation**")
    ccol1, ccol2 = st.columns([1, 2])
    with ccol1:
        corr_mode = st.radio(
            "Source",
            ["Manual", "Historical 60d", "Triangulation (cross vol)"],
            index=0, horizontal=False, key="wop_corr_mode",
            help=(
                "**Manual**: slider input.  \n"
                "**Historical 60d**: backward-looking rolling 60-day "
                "log-return correlation of the two pairs.  \n"
                "**Triangulation (cross vol)**: forward-looking implied "
                "correlation from the cross-pair's ATM vol, via "
                "σ²_X = σ²_A + σ²_B ± 2ρ·σ_A·σ_B. Only available when "
                "the two pairs share a currency AND the cross's "
                "VOL_ATM panel is present in the data folder."
            ),
        )

    # Pre-compute reference values that we display regardless of source
    hist_corr, n_obs_corr = _wop_compute_rolling_corr(
        leg_a_resolved["spot_ts"], leg_b_resolved["spot_ts"], val_date,
        window=60,
    )
    # Triangulation lookup (may be None if cross-vol not in folder OR pairs
    # share no currency)
    from core.correlation import implied_correlation_at_T
    tri_res = implied_correlation_at_T(
        folder, leg_a_cfg["pair"], leg_b_cfg["pair"],
        leg_a_resolved["T"], val_date,
        prefer_a=leg_a_cfg["prefer"], prefer_b=leg_b_cfg["prefer"],
        prefer_cross="offshore",
    )

    with ccol2:
        if corr_mode == "Manual":
            rho = st.slider(
                "ρ (log-return correlation)",
                min_value=-0.99, max_value=0.99, value=0.30, step=0.05,
                key="wop_rho_manual",
            )
            ref_bits = []
            if hist_corr is not None:
                ref_bits.append(f"60d realized = `{hist_corr:+.3f}` "
                                  f"(n_obs = {n_obs_corr})")
            if tri_res is not None:
                ref_bits.append(f"triangulation via {tri_res.cross_pair} = "
                                  f"`{tri_res.rho_implied:+.3f}`")
            if ref_bits:
                st.caption("For reference: " + "  ·  ".join(ref_bits) + ".")

        elif corr_mode == "Historical 60d":
            if hist_corr is None:
                st.error(
                    f"Not enough overlapping data (n={n_obs_corr}) to compute "
                    "60d rolling correlation. Switching to manual = 0.30."
                )
                rho = 0.30
            else:
                rho = hist_corr
                cap_bits = [
                    f"Historical 60d log-return correlation at "
                    f"{val_date}: **{rho:+.3f}** (n_obs = {n_obs_corr})"
                ]
                if tri_res is not None:
                    cap_bits.append(
                        f"triangulation says `{tri_res.rho_implied:+.3f}`"
                    )
                st.caption(".  ".join(cap_bits) + ".")

        else:   # Triangulation
            if tri_res is None:
                share = _wop_pair_share_ccy(
                    leg_a_cfg["pair"], leg_b_cfg["pair"]
                )
                if not share:
                    st.error(
                        f"{leg_a_cfg['pair']} and {leg_b_cfg['pair']} share "
                        f"no currency — triangulation undefined. Switching "
                        f"to manual = 0.30."
                    )
                else:
                    st.error(
                        f"Cross-pair vol panel for the implied cross of "
                        f"{leg_a_cfg['pair']} × {leg_b_cfg['pair']} is not "
                        f"in the data folder. Switching to manual = 0.30."
                    )
                rho = 0.30
            else:
                rho = tri_res.rho_implied
                cap = (
                    f"Triangulation via **{tri_res.cross_pair}** "
                    f"(η = {tri_res.eta:+d}) at T = {leg_a_resolved['T']:.4f}y: "
                    f"σ_A = `{tri_res.sigma_a*100:.3f}%`  "
                    f"σ_B = `{tri_res.sigma_b*100:.3f}%`  "
                    f"σ_X = `{tri_res.sigma_cross*100:.3f}%`  →  "
                    f"**ρ = {rho:+.4f}**"
                )
                if tri_res.clipped:
                    cap += f"  ⚠ {tri_res.notes}"
                if hist_corr is not None:
                    cap += f".  Historical 60d says `{hist_corr:+.3f}`."
                st.caption(cap)

    # =========================================================================
    # Pricing engine
    # =========================================================================
    st.markdown("---")
    ecol1, ecol2 = st.columns([1, 1])
    with ecol1:
        engine = st.radio(
            "Pricing engine",
            ["Both (CF + MC)", "CF only", "MC only"],
            index=0, horizontal=True, key="wop_engine",
            help=(
                "CF = closed form via 1D Gauss-Legendre quadrature "
                "(deterministic, ~1-3 ms). "
                "MC = correlated bivariate-normal terminal draws "
                "(stochastic, ~5-15 ms at 200k paths). "
                "Both = run both side-by-side for cross-validation."
            ),
        )
    with ecol2:
        if engine != "CF only":
            n_paths_mc = st.select_slider(
                "MC paths",
                options=[50_000, 100_000, 200_000, 500_000, 1_000_000],
                value=200_000, key="wop_n_paths",
            )
        else:
            n_paths_mc = 200_000  # unused

    # =========================================================================
    # Build legs + price
    #
    # IMPORTANT: each leg's raw intrinsic is in its own DOM units (e.g. JPY
    # per USD for USDJPY, USD per EUR for EURUSD). Taking min() across two
    # legs in different currencies is meaningless, so we follow the same
    # convention as core/worstof.py: scale each leg by 1/S_init so the
    # payoff becomes "% of initial spot" (dimensionless return). After the
    # scaling, min() makes sense and the pricer output is in % of notional;
    # multiplying by notional_usd gives the USD premium.
    #
    # The GBM dynamics are scale-invariant: (S/c) follows GBM with the same
    # drift and vol as S, with strike/barrier also scaled by 1/c. So this
    # is exact, not an approximation.
    # =========================================================================
    S_a_init = leg_a_resolved["S"]
    S_b_init = leg_b_resolved["S"]
    wol_a = WorstOfLeg(
        S=1.0,
        K=leg_a_resolved["K"] / S_a_init,
        H=leg_a_resolved["H"] / S_a_init,
        sigma=leg_a_resolved["sigma_smile"],  # vol-at-strike per leg
        r_d=leg_a_resolved["r_d"], r_f=leg_a_resolved["r_f"],
        opt=leg_a_resolved["option_type"],
        bar_dir=leg_a_resolved["barrier_type"],
    )
    wol_b = WorstOfLeg(
        S=1.0,
        K=leg_b_resolved["K"] / S_b_init,
        H=leg_b_resolved["H"] / S_b_init,
        sigma=leg_b_resolved["sigma_smile"],
        r_d=leg_b_resolved["r_d"], r_f=leg_b_resolved["r_f"],
        opt=leg_b_resolved["option_type"],
        bar_dir=leg_b_resolved["barrier_type"],
    )
    T = leg_a_resolved["T"]   # shared tenor by construction
    r_d_discount = leg_a_resolved["r_d"]   # numeraire = leg A's DOM

    cf_out = None
    mc_out = None
    if engine in ("Both (CF + MC)", "CF only"):
        cf_out = worstof_eko_price_cf(wol_a, wol_b, T, rho, r_d_discount,
                                        n_quad=80)
    if engine in ("Both (CF + MC)", "MC only"):
        mc_out = worstof_eko_price_mc(wol_a, wol_b, T, rho, r_d_discount,
                                        n_paths=n_paths_mc, seed=42)

    # =========================================================================
    # Results table
    # =========================================================================
    st.markdown("---")
    st.markdown("### Structure premium")

    # Per-leg single-leg premiums (FOR-unit -> USD via leg's own spot)
    p_a_per_unit = leg_a_resolved["ko_per_unit"]
    p_b_per_unit = leg_b_resolved["ko_per_unit"]
    p_a_usd = p_a_per_unit / leg_a_resolved["S"] * notional_usd
    p_b_usd = p_b_per_unit / leg_b_resolved["S"] * notional_usd

    # The pricer's output is already in "% of notional" (because we
    # normalized each leg's spot to 1.0). Multiply by notional_usd
    # to get USD premium.
    rows = []
    if cf_out is not None:
        prem_usd = cf_out["price"] * notional_usd
        rows.append({
            "Method": "Closed form (1D quadrature)",
            "% of notional": f"{cf_out['price'] * 100:.4f}%",
            "Premium (USD)": _fmt_usd(prem_usd),
            "Note": "n_quad=80",
        })
    if mc_out is not None:
        prem_usd_mc = mc_out["price"] * notional_usd
        se_usd_mc = mc_out["std_err"] * notional_usd
        rows.append({
            "Method": "Monte Carlo",
            "% of notional": (f"{mc_out['price'] * 100:.4f}% ± "
                                f"{1.96 * mc_out['std_err'] * 100:.4f}%"),
            "Premium (USD)": (f"{_fmt_usd(prem_usd_mc)} ± "
                                f"{_fmt_usd(1.96 * se_usd_mc)}"),
            "Note": f"n_paths={mc_out['n_paths']:,}, antithetic",
        })

    # Legacy multipliers for comparison
    leg_a_usd = p_a_usd
    leg_b_usd = p_b_usd
    min_leg_usd = min(leg_a_usd, leg_b_usd)
    for mult in (0.33, 0.40, 0.50):
        rows.append({
            "Method": f"Legacy: {mult:.2f} × min(P_A, P_B)",
            "% of notional": f"{mult * min_leg_usd / notional_usd * 100:.4f}%",
            "Premium (USD)": _fmt_usd(mult * min_leg_usd),
            "Note": "(approximation we are replacing)",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # =========================================================================
    # Diagnostic block: survival probabilities, per-leg context
    # =========================================================================
    st.markdown("**Diagnostics**")
    diag_src = cf_out if cf_out is not None else mc_out
    if diag_src is not None:
        ddc1, ddc2, ddc3, ddc4 = st.columns(4)
        with ddc1:
            st.metric("P(leg A barrier alive)",
                       f"{diag_src['p_alive_leg1']*100:.1f}%")
        with ddc2:
            st.metric("P(leg B barrier alive)",
                       f"{diag_src['p_alive_leg2']*100:.1f}%")
        with ddc3:
            st.metric("P(both alive, joint)",
                       f"{diag_src['p_alive_joint']*100:.1f}%")
        with ddc4:
            st.metric("P(both alive AND both ITM)",
                       f"{diag_src['p_both_itm_and_alive']*100:.1f}%")

    # Side context — per-leg single-leg EKO prices for reading the table
    st.caption(
        f"Per-leg single-leg EKO mids (at σ_smile): "
        f"P_A = **{_fmt_usd(p_a_usd)}**, P_B = **{_fmt_usd(p_b_usd)}**, "
        f"min(P_A, P_B) = **{_fmt_usd(min_leg_usd)}**.  "
        f"ρ = `{rho:+.3f}`  ·  T = `{T:.4f}y` ({tenor_label})  ·  "
        f"discount r_d = `{r_d_discount*100:.3f}%` ({leg_a_resolved['dom_ccy']})."
    )

    if cf_out is not None and mc_out is not None:
        # Cross-validation: CF should fall inside MC 95% CI
        ci_lo = mc_out["ci_95_lower"]
        ci_hi = mc_out["ci_95_upper"]
        cf_p = cf_out["price"]
        in_ci = ci_lo <= cf_p <= ci_hi
        flag = "✓" if in_ci else "⚠"
        st.caption(
            f"{flag} CF vs MC cross-check: CF = `{cf_p*100:.4f}%`, "
            f"MC 95% CI = [`{ci_lo*100:.4f}%`, `{ci_hi*100:.4f}%`].  "
            f"{'In CI — agree to MC noise.' if in_ci else 'Outside CI — possible bias; consider increasing MC paths or CF quadrature nodes.'}"
        )

    # =========================================================================
    # Step G3 — Greeks (finite differences on CF pricer)
    # =========================================================================
    # Computed via central FD over the same WorstOfLeg objects used by
    # the engine above. CF is fast (~40 ms for full set) so we auto-
    # compute. For MC-based Greeks we'd add a button; deferred to keep
    # the tab snappy on first load.
    st.markdown("---")
    st.markdown("**Greeks** (per leg-A notional, FD on CF pricer)")
    st.caption(
        "Δ = per-spot sensitivity (dimensionless when each leg's spot "
        "is normalized to 1).  Γ = spot convexity.  ν = vega per 1 "
        "vol point (×0.01 in σ).  ∂V/∂ρ = correlation sensitivity per "
        "1 rho point (×0.01 in ρ).  Θ = -∂V/∂T per calendar day "
        "(negative = value decreases with time).  "
        "**USD Greeks** scale by notional ($"
        f"{notional_usd:,.0f}) per leg."
    )
    from core.worstof_greeks import worstof_greeks_fd

    g_cf = worstof_greeks_fd(
        wol_a, wol_b, T, rho, r_d=r_d_discount,
        pricer=worstof_eko_price_cf,
    )

    # Scale for display: convert per-leg vega and rho sensitivity to
    # the trader-friendly "per 1 vol point" / "per 1 rho point" basis.
    S_a_resolved = leg_a_resolved["S"]
    S_b_resolved = leg_b_resolved["S"]
    vega_a_per_vp = g_cf.vega_a * 0.01
    vega_b_per_vp = g_cf.vega_b * 0.01
    rho_sens_per_rp = g_cf.rho_sensitivity * 0.01

    greeks_rows = [
        {
            "Greek": "Δ_A (per 1% leg-A spot)",
            "% of notional": f"{g_cf.delta_a * S_a_resolved * 0.01 * 100:+.3f}%",
            "USD": f"${g_cf.delta_a * S_a_resolved * 0.01 * notional_usd:+,.0f}",
            "Interpretation": "Long Δ → hedge by shorting leg-A spot",
        },
        {
            "Greek": "Δ_B (per 1% leg-B spot)",
            "% of notional": f"{g_cf.delta_b * S_b_resolved * 0.01 * 100:+.3f}%",
            "USD": f"${g_cf.delta_b * S_b_resolved * 0.01 * notional_usd:+,.0f}",
            "Interpretation": "Long Δ → hedge by shorting leg-B spot",
        },
        {
            "Greek": "Γ_A (per 1% × 1% spot²)",
            "% of notional": (f"{g_cf.gamma_a * (S_a_resolved*0.01)**2 * 100:+.4f}%"),
            "USD": (f"${g_cf.gamma_a * (S_a_resolved*0.01)**2 * notional_usd:+,.0f}"),
            "Interpretation": "PnL from 1% spot move beyond linear Δ",
        },
        {
            "Greek": "Γ_B (per 1% × 1% spot²)",
            "% of notional": (f"{g_cf.gamma_b * (S_b_resolved*0.01)**2 * 100:+.4f}%"),
            "USD": (f"${g_cf.gamma_b * (S_b_resolved*0.01)**2 * notional_usd:+,.0f}"),
            "Interpretation": "Convexity in leg-B spot",
        },
        {
            "Greek": "ν_A (per 1 vol point)",
            "% of notional": f"{vega_a_per_vp * 100:+.4f}%",
            "USD": f"${vega_a_per_vp * notional_usd:+,.0f}",
            "Interpretation": "PnL if leg-A vol up by 1pt (e.g. 8% → 9%)",
        },
        {
            "Greek": "ν_B (per 1 vol point)",
            "% of notional": f"{vega_b_per_vp * 100:+.4f}%",
            "USD": f"${vega_b_per_vp * notional_usd:+,.0f}",
            "Interpretation": "PnL if leg-B vol up by 1pt",
        },
        {
            "Greek": "∂V/∂ρ (per 1 rho point)",
            "% of notional": f"{rho_sens_per_rp * 100:+.4f}%",
            "USD": f"${rho_sens_per_rp * notional_usd:+,.0f}",
            "Interpretation": "PnL if ρ moves up by 0.01 — "
                                "DISTINCTIVE worst-of risk",
        },
        {
            "Greek": "Θ (per calendar day)",
            "% of notional": f"{g_cf.theta_per_day * 100:+.4f}%",
            "USD": f"${g_cf.theta_per_day * notional_usd:+,.0f}",
            "Interpretation": ("Daily time decay (negative = value lost)"
                                 if g_cf.theta_per_day < 0
                                 else "Daily time gain (barrier-dominated "
                                       "structure)"),
        },
    ]
    st.dataframe(pd.DataFrame(greeks_rows), use_container_width=True,
                   hide_index=True)
    st.caption(
        f"Greeks via central finite differences on the CF pricer "
        f"({g_cf.method}). Δ/Γ use ±{g_cf.bump_sizes['spot_frac']*100:.1f}% "
        f"spot bumps; ν use ±{g_cf.bump_sizes['sigma_abs']*100:.0f} vol "
        f"point; ρ uses ±{g_cf.bump_sizes['rho_abs']:.2f}; Θ uses "
        f"1-day forward bump."
    )




# -----------------------------------------------------------------------------
# TAB 5: WORST-OF (BULK STRATEGY RUNNER) — uses the legacy multiplier
# approximation. The new correlation-aware pricer (TAB 4 above) is for
# single-trade live valuation only at this step; replacing the multiplier
# inside this bulk-runner engine is a follow-up.
# -----------------------------------------------------------------------------
def render_worstof_tab():
    """Bulk worst-of pricer.

    Mirrors the Backtest tab pattern: cross-product across multi-select
    axes, single 'Run' button, summary table of all strategies plus
    equity overlay and a multi-strategy CSV download. Drill into any
    single strategy on the next tab (Worst-of strategy drilldown).
    """
    from core.worstof import (
        WorstOfSpec, build_worstof_grid, run_worstof_grid,
        worstof_trades_to_df, worstof_summarize, worstof_equity_curve,
        worstof_export_time_series,
    )
    from core.gates import GATE_REGISTRY, gate_label

    pairs_avail = _list_pairs(folder)
    if len(pairs_avail) < 2:
        st.error("Need at least 2 pairs in the data folder.")
        return

    # Phase 4.5: apply pending preset BEFORE widgets are created
    _apply_pending_preset("wo")
    _render_preset_loader("wo", folder)
    # New: batch "Run all presets in folder" — pools every preset's
    # backtest into one combined comparison table. Only the worst-of
    # tab gets this for now since presets are exported with a worst-of
    # grid by default.
    _render_batch_run_all_presets(folder)

    # Phase WF-C: banner if a dynamic-mode preset is currently active.
    # This is informational; the strike Δ / KO Δ multi-selects below
    # will show "DYN" placeholder labels and the engine will read
    # levels from the schedule per trade date.
    _active_schedule = st.session_state.get("_active_wo_schedule")
    _active_schedule_label = st.session_state.get("_active_wo_schedule_label")
    if _active_schedule:
        st.warning(
            f"🔄 **Walk-forward schedule active** — preset "
            f"`{_active_schedule_label}` is loaded with "
            f"{len(_active_schedule)} monthly entries spanning "
            f"{_active_schedule[0]['valid_from']} → "
            f"{_active_schedule[-1]['valid_to']}. Strike & barrier "
            f"LEVELS will come from this schedule on each trade date — "
            f"the strike Δ / KO Δ selections below are ignored in this "
            f"mode. Sweep tenor and gate to compare; strikes are fixed "
            f"by the schedule. Clear by loading a static preset or "
            f"clicking the 'Clear walk-forward schedule' button below."
        )
        col_clear = st.columns([4, 1])
        with col_clear[1]:
            if st.button("✖ Clear schedule", key="wo_clear_schedule",
                          use_container_width=True):
                st.session_state.pop("_active_wo_schedule", None)
                st.session_state.pop("_active_wo_schedule_label", None)
                st.rerun()

    st.markdown("### Worst-of two-pair KO — bulk runner")
    st.caption(
        "Specify discrete pair pairings (one row per A-B combo) and one or "
        "more values on each axis (tenor, strike Δ per leg, KO Δ per leg, "
        "gate per leg). The cross-product becomes the strategy grid. "
        "Per-leg KO Δ values that aren't strictly less than the strike Δ "
        "are auto-filtered (barrier must be further OTM than strike). "
        "Structure premium = `min(P_A, P_B) / 3 + tx_cost`; "
        "payoff = `min(payoff_A, payoff_B)` at expiry."
    )

    # =========================================================================
    # Common controls (apply to all strategies in the grid)
    # =========================================================================
    cc1, cc2, cc3 = st.columns([1, 1, 1])
    with cc1:
        notional_usd_wo = st.number_input(
            "Notional (USD)",
            min_value=100_000.0, max_value=200_000_000.0,
            value=10_000_000.0, step=1_000_000.0, format="%.0f",
            key="wo_notional",
        )
        tx_cost_bps_wo = st.slider(
            "Transaction cost (bps of notional)",
            0.0, 20.0, 4.0, 0.5,
            help=("Flat structure-level markup. 4 bps on $10M = $4,000 "
                   "per trade. Individual leg premiums are at mid (no "
                   "tx applied to legs)."),
            key="wo_tx",
        )
    with cc2:
        # Date range derived from spot panel intersection
        date_max = _date.today()
        date_min = _date(2020, 1, 1)
        try:
            spot_all = load_panel(folder, "SPOT", None)
            if not spot_all.empty:
                date_max = spot_all.index.max().date()
                date_min = max(spot_all.index.min().date(),
                                 date_max - timedelta(days=365 * 10))
        except Exception:
            pass
        start_date_wo = st.date_input(
            "Start date",
            value=max(date_min, date_max - timedelta(days=365 * 2)),
            min_value=date_min, max_value=date_max, key="wo_start",
        )
        end_date_wo = st.date_input(
            "End date", value=date_max,
            min_value=date_min, max_value=date_max, key="wo_end",
        )
    with cc3:
        prefer_em_wo = st.radio(
            "EM preference (if applicable)", ["offshore", "onshore"],
            index=0, horizontal=True, key="wo_prefer",
        )
        direction_wo_label = st.selectbox(
            "Direction (shared across both legs)",
            list(DIRECTIONS.keys()), index=0, key="wo_direction",
            help="Both legs use the same direction. For mixed call/put "
                  "worst-of structures, see the standalone Pricer tab.",
        )
        direction_a, btype_a = DIRECTIONS[direction_wo_label]
        direction_b, btype_b = DIRECTIONS[direction_wo_label]

    # =========================================================================
    # Structure-level pricing engine (Step 2c)
    # =========================================================================
    st.markdown("---")
    st.markdown("**Structure pricing engine**")
    with st.expander("Engine settings", expanded=False):
        st.caption(
            "Choose how the worst-of structure premium is computed. The "
            "**legacy multiplier** (default) preserves historical behaviour "
            "for backwards-compatible backtests; **closed form** and "
            "**Monte Carlo** use the new correlation-aware pricer "
            "(`core.worstof_pricer`). Both new engines require "
            "`ko_check_mode='european_at_expiry'` and "
            "`leg_pricing_mode='european'`."
        )
        wec1, wec2 = st.columns([1, 1])
        with wec1:
            wo_engine_label = st.radio(
                "Engine",
                ["Legacy multiplier (default)",
                 "Closed form (CF)",
                 "Monte Carlo (MC)"],
                index=0,
                key="wo_pricing_engine",
                help=(
                    "Legacy: `multiplier × min(P_A, P_B)`. "
                    "CF: 1D-quadrature joint pricer (~1–3 ms/trade). "
                    "MC: terminal correlated-GBM simulator "
                    "(~5–15 ms/trade at default paths). CF and MC are "
                    "equivalent in expectation; CF is faster + deterministic."
                ),
            )
            _engine_map = {
                "Legacy multiplier (default)": "legacy_multiplier",
                "Closed form (CF)": "closed_form",
                "Monte Carlo (MC)": "monte_carlo",
            }
            wo_pricing_engine = _engine_map[wo_engine_label]
        with wec2:
            wo_correlation_source = "manual"
            wo_correlation_value = 0.30
            wo_mc_n_paths = 100_000
            if wo_pricing_engine != "legacy_multiplier":
                wo_correlation_source_label = st.radio(
                    "Correlation source",
                    ["Manual (single ρ)",
                     "Historical 60d rolling",
                     "Triangulation (cross vol)"],
                    index=1,  # 60d rolling is the more useful default
                    key="wo_correlation_source",
                    help=(
                        "**Manual**: same ρ used for every trade date.  \n"
                        "**Historical 60d**: rolling 60-business-day "
                        "realized log-return correlation, computed once "
                        "per backtest per pair-combo.  \n"
                        "**Triangulation**: forward-looking implied "
                        "correlation from the cross-pair's ATM vol — "
                        "free in FX since cross vols are quoted. "
                        "Requires the cross pair's VOL_ATM panel in the "
                        "data folder.  \n\n"
                        "All non-manual sources fall back to Manual on "
                        "dates where the source's value is missing."
                    ),
                )
                _src_map = {
                    "Manual (single ρ)": "manual",
                    "Historical 60d rolling": "rolling_60d",
                    "Triangulation (cross vol)": "triangulation",
                }
                wo_correlation_source = _src_map[wo_correlation_source_label]
                wo_correlation_value = st.slider(
                    ("ρ (Manual value; fallback when 60d/triangulation "
                     "data is unavailable)"),
                    min_value=-0.95, max_value=0.95,
                    value=0.30, step=0.05,
                    key="wo_correlation_value",
                )
                if wo_pricing_engine == "monte_carlo":
                    wo_mc_n_paths = st.select_slider(
                        "MC paths per trade",
                        options=[20_000, 50_000, 100_000, 200_000, 500_000],
                        value=100_000, key="wo_mc_n_paths",
                        help=("Std error per trade scales as 1/√n. "
                               "100k → ~1bp; 500k → ~0.5bp."),
                    )



    # Sensible defaults: a few common combos if the pairs exist, else
    # just the first two available pairs.
    default_combo_candidates = [
        ("AUDUSD", "NZDUSD"),
        ("EURUSD", "GBPUSD"),
        ("USDJPY", "USDKRW"),
    ]
    default_combos = [(a, b) for (a, b) in default_combo_candidates
                        if a in pairs_avail and b in pairs_avail]
    if not default_combos:
        # Fall back: first pair × second pair
        default_combos = [(pairs_avail[0], pairs_avail[1])]

    combo_df_default = pd.DataFrame(default_combos[:1],
                                       columns=["Pair A", "Pair B"])

    # Phase 4.5: if a preset was just applied, it stashed the desired
    # pair-combos in `_pending_wo_combos` (a NON-widget key, because
    # data_editor refuses session_state assignment on its own key). We
    # consume it here as the editor's initial value. We also have to
    # delete the existing widget state under `wo_combos` so the editor
    # rebuilds with the new value rather than restoring its prior
    # state. Pop on consumption so this doesn't clobber subsequent
    # user edits.
    pending_combos = st.session_state.pop("_pending_wo_combos", None)
    if pending_combos is not None:
        combo_df_default = pending_combos
        # The widget's own state must be discarded; otherwise Streamlit
        # would restore its prior contents and ignore our new value=.
        st.session_state.pop("wo_combos", None)
    combo_df = st.data_editor(
        combo_df_default,
        column_config={
            "Pair A": st.column_config.SelectboxColumn(
                "Pair A", options=pairs_avail, required=True),
            "Pair B": st.column_config.SelectboxColumn(
                "Pair B", options=pairs_avail, required=True),
        },
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        key="wo_combos",
    )

    # Sanitize: drop any rows with missing values, deduplicate, warn on
    # self-pairs (no diversification benefit).
    raw_combos = []
    for _, r in combo_df.iterrows():
        a, b = r.get("Pair A"), r.get("Pair B")
        if pd.isna(a) or pd.isna(b) or not a or not b:
            continue
        raw_combos.append((a, b))
    pair_combos: list[tuple[str, str]] = list(dict.fromkeys(raw_combos))
    self_pairs = [c for c in pair_combos if c[0] == c[1]]
    if self_pairs:
        st.warning(f"{len(self_pairs)} pair combo(s) use the same pair on "
                    f"both legs — the worst-of reduces to a single-leg "
                    f"structure (correlation = 1, no diversification). "
                    f"These will still run but won't match the spec name "
                    f"convention.")

    # =========================================================================
    # Tenor + per-leg strike/KO/gate axes
    # =========================================================================
    st.markdown("---")
    st.markdown("**Per-leg axes** — cross-product across all selections.")

    cax1, cax2, cax3 = st.columns([1, 1, 1])
    with cax1:
        st.markdown("**Common**")
        tenors_wo = st.multiselect(
            "Tenor(s)", TENOR_LIST, default=["1M"], key="wo_tenors",
            help="Tenor is shared between both legs of a given strategy. "
                  "Each selected tenor becomes its own row in the grid.",
        )
    delta_choices_no_atm = {k: v for k, v in DELTA_CHOICES.items()
                              if k != "ATM"}
    with cax2:
        st.markdown("**Leg A**")
        sd_a_labels = st.multiselect(
            "Strike Δ (Leg A)", list(DELTA_CHOICES.keys()),
            default=["35Δ"], key="wo_sd_a",
            help="ATM strike (Δ=0) is allowed and bypasses the "
                  "KO-Δ-must-be-lower filter (the wing KO is always "
                  "OTM relative to ATM).",
        )
        kd_a_labels = st.multiselect(
            "KO Δ (Leg A)", list(KO_DELTA_CHOICES.keys()),
            default=["10Δ"], key="wo_kd_a",
            help="Combos where KO Δ ≥ Strike Δ are filtered out "
                  "(barrier would be inside / equal to the strike — "
                  "degenerate). ATM strikes bypass this filter.",
        )
        # Per-leg gate, with explicit (none) option
        gate_a_options = ["(no gate)"] + list(GATE_REGISTRY.keys())
        gate_a_labels = st.multiselect(
            "Gate(s) (Leg A)", gate_a_options,
            default=["(no gate)"], key="wo_gate_a",
            format_func=lambda k: ("(no gate)" if k == "(no gate)"
                                   else GATE_REGISTRY[k][0]),
            help="A trade enters only when this gate passes for pair A "
                  "AND Leg B's gate passes for pair B on the same date. "
                  "Include '(no gate)' alongside named gates to test "
                  "both gated and ungated variants in one run.",
        )
        gates_a_resolved: list = [None if k == "(no gate)" else k
                                     for k in gate_a_labels]
    with cax3:
        st.markdown("**Leg B**")
        sd_b_labels = st.multiselect(
            "Strike Δ (Leg B)", list(DELTA_CHOICES.keys()),
            default=["35Δ"], key="wo_sd_b",
        )
        kd_b_labels = st.multiselect(
            "KO Δ (Leg B)", list(KO_DELTA_CHOICES.keys()),
            default=["10Δ"], key="wo_kd_b",
        )
        gate_b_options = ["(no gate)"] + list(GATE_REGISTRY.keys())
        gate_b_labels = st.multiselect(
            "Gate(s) (Leg B)", gate_b_options,
            default=["(no gate)"], key="wo_gate_b",
            format_func=lambda k: ("(no gate)" if k == "(no gate)"
                                   else GATE_REGISTRY[k][0]),
        )
        gates_b_resolved: list = [None if k == "(no gate)" else k
                                     for k in gate_b_labels]

    # =========================================================================
    # Build specs preview & validate
    # =========================================================================
    sd_a_resolved = [(lbl, DELTA_CHOICES[lbl]) for lbl in sd_a_labels]
    sd_b_resolved = [(lbl, DELTA_CHOICES[lbl]) for lbl in sd_b_labels]
    kd_a_resolved = [(lbl, KO_DELTA_CHOICES[lbl]) for lbl in kd_a_labels]
    kd_b_resolved = [(lbl, KO_DELTA_CHOICES[lbl]) for lbl in kd_b_labels]

    # Pre-build to get the post-filter count (same call signature used
    # on the run path so the count is exact, not estimated).
    if (pair_combos and tenors_wo and sd_a_resolved and sd_b_resolved
            and kd_a_resolved and kd_b_resolved
            and gates_a_resolved and gates_b_resolved):
        preview_specs = build_worstof_grid(
            pair_combos=pair_combos,
            tenors=tenors_wo,
            leg_a_directions=[(direction_a, btype_a)],
            leg_b_directions=[(direction_b, btype_b)],
            leg_a_strike_deltas=sd_a_resolved,
            leg_b_strike_deltas=sd_b_resolved,
            leg_a_ko_deltas=kd_a_resolved,
            leg_b_ko_deltas=kd_b_resolved,
            gates_a=gates_a_resolved,
            gates_b=gates_b_resolved,
            tx_cost_bps=tx_cost_bps_wo,
            prefer=prefer_em_wo,
            multiplier=wo_multiplier,
            # Step 2c
            pricing_engine=wo_pricing_engine,
            correlation_source=wo_correlation_source,
            correlation_value=wo_correlation_value,
            mc_n_paths=wo_mc_n_paths,
        )
        # Phase WF-C: if a dynamic schedule is active, attach it to
        # every spec. The engine then reads (K, H) levels from the
        # schedule per trade date and ignores the strike/KO deltas.
        # This is the LAST step in spec construction so that filtering
        # in build_worstof_grid (KO Δ < strike Δ) still runs — though
        # in dynamic mode the deltas are placeholders, the structural
        # validity is enforced by the schedule's levels which are
        # consistent (H further OTM than K by construction).
        if _active_schedule is not None:
            _active_strike_strategy = st.session_state.get(
                "_active_wo_strike_strategy", "cheapest"
            )
            for s in preview_specs:
                s.dynamic_schedule = _active_schedule
                s._adaptive_strike_strategy = _active_strike_strategy
    else:
        preview_specs = []

    # Pre-filter naive count, post-filter actual count
    naive_count = (len(pair_combos) * len(tenors_wo) *
                     len(sd_a_resolved) * len(sd_b_resolved) *
                     len(kd_a_resolved) * len(kd_b_resolved) *
                     len(gates_a_resolved) * len(gates_b_resolved))
    n_specs = len(preview_specs)
    filtered = naive_count - n_specs
    filter_str = (f"  (filtered {filtered} where KO Δ ≥ strike Δ)"
                   if filtered > 0 else "")

    st.markdown("---")
    st.caption(
        f"**{n_specs}** strategies will run "
        f"({len(pair_combos)} pair combos × {len(tenors_wo)} tenors × "
        f"{len(sd_a_resolved)} strike-Δ-A × {len(kd_a_resolved)} KO-Δ-A × "
        f"{len(sd_b_resolved)} strike-Δ-B × {len(kd_b_resolved)} KO-Δ-B × "
        f"{len(gates_a_resolved)} gate-A × {len(gates_b_resolved)} gate-B)"
        f"{filter_str}  ·  "
        f"{(end_date_wo - start_date_wo).days} calendar days."
    )
    # If batch results are already loaded, clarify that this count is
    # for the NEXT manual run — separate from what's already in the
    # summary table below.
    if st.session_state.get("wo_meta", {}).get("batch_run_all"):
        st.caption(
            f"_↑ This is the count for the **next manual run** if you "
            f"click the button below. The {st.session_state['wo_meta'].get('n_specs', '?')} "
            f"strategies from your most recent batch are already loaded "
            f"in the summary table further down._"
        )

    can_run = n_specs > 0
    if not can_run:
        st.info("Pick at least one value on every axis. If the count is 0 "
                  "but all axes have selections, all combos were filtered "
                  "out by the KO < strike rule (try widening either axis).")

    run_clicked_wo = st.button("▶ Run worst-of bulk backtest",
                                  type="primary",
                                  disabled=not can_run, key="wo_run")

    # =========================================================================
    # Execute
    # =========================================================================
    if run_clicked_wo:
        progress_bar = st.progress(0.0, text="Starting…")
        last_t = [time.time()]

        def cb(p, name):
            now = time.time()
            if now - last_t[0] > 0.1 or p >= 1.0:
                progress_bar.progress(min(p, 1.0),
                                       text=f"Running: {name} ({p*100:.0f}%)")
                last_t[0] = now

        t0 = time.time()
        results = run_worstof_grid(
            folder, preview_specs, start_date_wo, end_date_wo,
            notional_usd=notional_usd_wo, progress_cb=cb,
        )
        elapsed = time.time() - t0
        progress_bar.empty()

        # Persist results by name; the drilldown tab looks them up
        st.session_state["wo_results"] = results
        st.session_state["wo_specs"] = {s.name: s for s in preview_specs}
        st.session_state["wo_meta"] = {
            "start": start_date_wo, "end": end_date_wo,
            "notional_usd": notional_usd_wo,
            "tx_cost_bps": tx_cost_bps_wo,
            "prefer": prefer_em_wo,
            "direction_label": direction_wo_label,
            "tenors": tenors_wo,
            "pair_combos": pair_combos,
            "n_specs": n_specs,
            "elapsed": elapsed,
        }
        total_trades = sum(len(t) for t in results.values())
        st.success(f"Done in {elapsed:.1f}s — {n_specs} strategies, "
                     f"{total_trades} trades total.")

    # =========================================================================
    # Summary
    # =========================================================================
    if "wo_results" not in st.session_state:
        st.info("Configure axes above and click **Run** to see a summary "
                  "table. Drill into any one strategy on the "
                  "*Worst-of strategy drilldown* tab.")
        return

    results = st.session_state["wo_results"]
    meta = st.session_state.get("wo_meta", {})

    # Persistent banner when results came from "Apply & run all
    # presets" (Phase 4.5 batch path). Survives reruns because it's
    # keyed off `wo_meta.batch_run_all` in session state. The reason
    # for this banner: after a batch run, the configuration controls
    # above show "1 strategies will run" for the NEXT manual click —
    # not the batch count. Without this banner, users see the disconnect
    # and assume something's broken.
    if meta.get("batch_run_all"):
        n_presets = meta.get("batch_n_presets", "?")
        n_err = meta.get("batch_n_errors", 0)
        n_trades = meta.get("n_trades", 0)
        err_str = f" ({n_err} errored)" if n_err else ""
        st.info(
            f"📦 **Batch results loaded** — {meta.get('n_specs', '?')} "
            f"strategies pooled from {n_presets} preset(s){err_str}, "
            f"with {n_trades:,} trades total across the period "
            f"{meta.get('start')} → {meta.get('end')}. The "
            f"configuration controls *above* are for a new manual "
            f"run — they don't reflect this batch. Sort the table "
            f"below by **Sharpe** (or any metric) to compare. To "
            f"clear and start a fresh single run, click the regular "
            f"Run button above."
        )

    st.markdown("---")
    st.markdown("### Summary across worst-of strategies")
    pair_combos_str = ", ".join(f"{a}-{b}" for a, b in
                                   meta.get("pair_combos", []))
    st.caption(
        f"Run period: {meta.get('start')} → {meta.get('end')}  ·  "
        f"notional ${meta.get('notional_usd', 0):,.0f}  ·  "
        f"tx cost {meta.get('tx_cost_bps', 0):.1f} bps  ·  "
        f"direction: **{meta.get('direction_label')}**  ·  "
        f"tenors: {', '.join(meta.get('tenors', []))}  ·  "
        f"pair combos: {pair_combos_str}  ·  "
        f"elapsed {meta.get('elapsed', 0):.1f}s"
    )

    rows = []
    for name, trades in results.items():
        df = worstof_trades_to_df(trades)
        if df.empty:
            rows.append({"Strategy": name, "n": 0})
            continue
        s = worstof_summarize(df)
        rows.append({
            "Strategy": name,
            "n": s["n_trades"],
            "Win%": f"{s['win_rate']*100:.0f}",
            "A KO%": f"{s['leg_a_ko_rate']*100:.0f}",
            "B KO%": f"{s['leg_b_ko_rate']*100:.0f}",
            "Both surv%": f"{s['both_survive_rate']*100:.0f}",
            "Σ Premium": _fmt_usd(s["total_premium_paid_usd"]),
            "Σ TX Cost": _fmt_usd(s["total_tx_cost_usd"]),
            "Σ Payout": _fmt_usd(s.get("total_payout_usd", 0)),
            "Σ PnL": _fmt_usd(s["total_pnl_usd"]),
            "Sharpe (m)": f"{s.get('sharpe_monthly', 0):+.2f}",
            "Max DD": _fmt_usd(s["max_drawdown_usd"]),
            "Recovery%": f"{s['premium_recovery_pct']:.0f}",
            "Struct/Min%": f"{s['structure_vs_min_leg_pct']:.0f}",
            # --- Cross-year consistency block (matches Backtest tab) ---
            "Yrs": s.get("n_years", 0),
            "%Pos Yrs": f"{s.get('pct_positive_years', 0):.0f}",
            "Min Ann $": _fmt_usd(s.get("min_annual_pnl_usd", 0)),
            "Sharpe(y) μ": f"{s.get('annual_sharpe_mean', 0):+.2f}",
            "Sharpe(y) min": f"{s.get('annual_sharpe_min', 0):+.2f}",
            "Sharpe(y) σ": f"{s.get('annual_sharpe_std', 0):.2f}",
            "Sharpe(y) CV": (f"{s.get('annual_sharpe_cv', 0):+.2f}"
                               if s.get('annual_sharpe_cv', 0) != 0
                               else "—"),
            "Sharpe Score": f"{s.get('annual_sharpe_score', 0):+.2f}",
            "Calmar": f"{s.get('calmar', 0):+.2f}",
            "G2P": (f"{s.get('gain_to_pain', 0):.2f}"
                     if s.get("gain_to_pain", 0) != float("inf")
                     else "∞"),
            "Ulcer": f"{s.get('ulcer_index', 0):.2f}",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                       hide_index=True)
        st.caption(
            "💡 Click any column header to sort. For a single-shot "
            "**consistency-aware ranking**, sort by `Sharpe Score` "
            "descending (= μ − σ; rewards high mean AND low yearly "
            "swing). For finer-grained views: `%Pos Yrs` ↓ then "
            "`Sharpe(y) min` ↓ then `Sharpe(y) σ` ↑. `Calmar` and "
            "`G2P` add drawdown-pain and asymmetry views. Note: "
            "`Sharpe(y) μ` alone is NOT a concentration penalty — "
            "see column definitions."
        )
        with st.expander("Column definitions", expanded=False):
            st.markdown(
                "**Headline columns**\n"
                "- **n** — number of trades opened.\n"
                "- **Win%** — share of trades with positive worst-of payoff "
                "(both legs survive AND finish ITM).\n"
                "- **A KO% / B KO%** — share of trades where that leg's "
                "barrier was hit at expiry.\n"
                "- **Both surv%** — share of trades where neither leg "
                "KO'd (worst-of can still be 0 if one leg finishes OTM).\n"
                "- **Σ Premium** — total paid = mid (min/3) + tx.\n"
                "- **Σ TX Cost** — bps markup × notional × n_trades.\n"
                "- **Σ Payout** — sum of `min(payoff_A, payoff_B)`.\n"
                "- **Σ PnL** — Σ Payout − Σ Premium.\n"
                "- **Sharpe (m)** — monthly PnL Sharpe: mean / std × √12.\n"
                "- **Max DD** — realised peak-to-trough on the equity "
                "curve (at-expiry, not MTM).\n"
                "- **Recovery%** — Σ Payout / Σ Premium × 100.\n"
                "- **Struct/Min%** — structure premium as % of the min "
                "leg premium (should average ~33.3% by construction).\n\n"
                "**Cross-year consistency columns** (same definitions as "
                "the Backtest tab — use to rank strategies that work in "
                "many years rather than ones that earn it all in 1-2 "
                "years):\n"
                "- **Yrs** — calendar years observed.\n"
                "- **%Pos Yrs** — fraction of those years with positive "
                "PnL. The most direct measure of cross-year consistency.\n"
                "- **Min Ann $** — worst calendar-year PnL in USD. Flags "
                "blow-up years in absolute terms.\n"
                "- **Sharpe(y) μ** — unweighted average of per-year "
                "Sharpes. ⚠️ Treats each year equally regardless of "
                "magnitude or variance, so '9 years of Sharpe=0 + "
                "1 year of Sharpe=10' and '10 years of Sharpe=1' "
                "produce *similar* values. Useful as a scale-invariant "
                "average — NOT as a concentration penalty. For that, "
                "look at Sharpe(y) σ or Sharpe(y) min instead.\n"
                "- **Sharpe(y) min** — worst per-year Sharpe. Directly "
                "answers 'what's the worst year I'd have lived through?'.\n"
                "- **Sharpe(y) σ** — standard deviation of per-year "
                "Sharpes. Directly measures cross-year stability — "
                "high σ means risk-adjusted returns swing widely "
                "year-to-year. Low σ + high μ is the goal.\n"
                "- **Sharpe(y) CV** — coefficient of variation = σ/μ "
                "(signed). Magnitude tells you how much yearly Sharpes "
                "swing relative to their average; sign tells you if μ "
                "is positive or negative. Shown as '—' when |μ| ≈ 0.\n"
                "- **Sharpe Score** — composite from Yavuz Akbay's "
                "framework: `μ × (1 − CV) = μ − σ`. Rewards high mean "
                "AND low variance in one number — use to rank "
                "strategies when you want both signals at once.\n"
                "- **Calmar** — annualised return / |max DD|. "
                "Pain-adjusted return — punishes deep DDs that "
                "Sharpe smooths over.\n"
                "- **G2P** — gain-to-pain: Σ positive monthly returns / "
                "|Σ negative|. Outlier-robust alternative to Sharpe. "
                "> 1 = profitable; > 2 = good; > 3 = rare.\n"
                "- **Ulcer** — RMS drawdown depth across the run. "
                "Captures both DD depth AND duration; long shallow "
                "underwater stretches that Sharpe ignores get penalised.\n\n"
                "⚠️ **Small-sample caveat:** with only 2-3 years of data "
                "all annual metrics are noisy. A `Sharpe(y) min` of "
                "−2 based on one year is not the same evidence as one "
                "based on ten."
            )

    # =========================================================================
    # Downloads: summary table + per-strategy time series (and optionally the
    # full trade-level combined ledger). The equity overlay chart is gone —
    # use the time-series CSV in another app to recreate charts; or use the
    # Worst-of strategy drilldown tab for an interactive single-strategy view.
    #
    # The downloaded summary CSV uses a CANONICAL schema shared with the
    # regular Backtest tab — both can be stacked in the same downstream app
    # via the `strategy_type` column ('single' vs 'worst_of').
    # =========================================================================
    # =========================================================================
    # Phase 4: regime breakdowns for worst-of.
    # For each strategy, attribute every trade to the (state_a, state_b)
    # of the underlying pair pair on its trade-entry date. Shows three
    # views (per-leg-A, per-leg-B, joint) — pick whichever answers your
    # question. The joint view can be sparse when both pairs have many
    # states; the per-leg views are always dense.
    # =========================================================================
    from core.worstof import worstof_summarize_by_regime as _wo_by_regime
    from core.regimes import get_regime_panel, list_registered_pairs
    if list_registered_pairs():
        st.markdown("---")
        with st.expander("🧬 Regime breakdown (per HMM state, per leg)",
                          expanded=False):
            st.caption(
                "For each strategy, attributes every trade to the "
                "HMM-decoded state of **each leg's pair on the trade "
                "entry date**, then aggregates. Three views: per-leg-A, "
                "per-leg-B, and joint (state_a × state_b). The joint "
                "view is the most informative but can be sparse. Only "
                "strategies whose pair pair has registered regime "
                "panels appear here — fit & save in app 10's Tab 7."
            )
            wo_brk_a, wo_brk_b, wo_brk_joint = [], [], []
            for name, trades in results.items():
                df_b = worstof_trades_to_df(trades)
                if (df_b.empty or "leg_a_pair" not in df_b.columns
                        or "leg_b_pair" not in df_b.columns):
                    continue
                pair_a_ = df_b["leg_a_pair"].iloc[0]
                pair_b_ = df_b["leg_b_pair"].iloc[0]
                pan_a = get_regime_panel(pair_a_)
                pan_b = get_regime_panel(pair_b_)
                if pan_a is None and pan_b is None:
                    continue
                brk = _wo_by_regime(df_b, pan_a, pan_b)
                for _, r in brk["by_state_a"].iterrows():
                    wo_brk_a.append({
                        "Strategy": name, "Pair A": pair_a_,
                        "State A": int(r["state"]),
                        "n trades": int(r["n_trades"]),
                        "Share %": f"{r['share_of_trades_pct']:.1f}%",
                        "Win %": f"{r['win_rate_pct']:.1f}%",
                        "KO %": f"{r['structure_ko_rate_pct']:.1f}%",
                        "Total PnL $": _fmt_usd(r["total_pnl_usd"]),
                        "Mean PnL $": _fmt_usd(r["mean_pnl_usd"]),
                    })
                for _, r in brk["by_state_b"].iterrows():
                    wo_brk_b.append({
                        "Strategy": name, "Pair B": pair_b_,
                        "State B": int(r["state"]),
                        "n trades": int(r["n_trades"]),
                        "Share %": f"{r['share_of_trades_pct']:.1f}%",
                        "Win %": f"{r['win_rate_pct']:.1f}%",
                        "KO %": f"{r['structure_ko_rate_pct']:.1f}%",
                        "Total PnL $": _fmt_usd(r["total_pnl_usd"]),
                        "Mean PnL $": _fmt_usd(r["mean_pnl_usd"]),
                    })
                for _, r in brk["joint"].iterrows():
                    wo_brk_joint.append({
                        "Strategy": name,
                        "State A": int(r["state_a"]),
                        "State B": int(r["state_b"]),
                        "n trades": int(r["n_trades"]),
                        "Share %": f"{r['share_of_trades_pct']:.1f}%",
                        "Win %": f"{r['win_rate_pct']:.1f}%",
                        "KO %": f"{r['structure_ko_rate_pct']:.1f}%",
                        "Total PnL $": _fmt_usd(r["total_pnl_usd"]),
                        "Mean PnL $": _fmt_usd(r["mean_pnl_usd"]),
                    })
            if wo_brk_joint:
                st.markdown("**Joint breakdown** (state_a × state_b)")
                st.dataframe(pd.DataFrame(wo_brk_joint),
                              use_container_width=True, hide_index=True)
            wo_view_cols = st.columns(2)
            with wo_view_cols[0]:
                if wo_brk_a:
                    st.markdown("**Per-leg-A** (pair A state only)")
                    st.dataframe(pd.DataFrame(wo_brk_a),
                                  use_container_width=True, hide_index=True)
            with wo_view_cols[1]:
                if wo_brk_b:
                    st.markdown("**Per-leg-B** (pair B state only)")
                    st.dataframe(pd.DataFrame(wo_brk_b),
                                  use_container_width=True, hide_index=True)
            if not (wo_brk_a or wo_brk_b or wo_brk_joint):
                st.caption("_No strategies have registered regime panels "
                            "for both legs._")
            else:
                st.caption(
                    "💡 Read the joint view as: 'in this strategy, X% of "
                    "trades opened when pair A was in state SA and pair B "
                    "was in state SB; their realised win rate was Y% and "
                    "structure-KO rate was Z%.' The combination where both "
                    "pairs are in their dominant cluster (state 0, state 0) "
                    "is typically where the structure has the best risk-"
                    "adjusted edge — confirm in the data."
                )

    st.markdown("---")
    st.markdown("### Downloads")
    st.caption(
        "Bulk run results are now download-only — the per-strategy "
        "drilldown tab still shows full charts for any single strategy. "
        "The **time series CSV** contains daily (per-expiry-date), "
        "monthly, and annual rows with `pnl_usd`, `equity_usd`, and "
        "`drawdown_usd` at each period end — enough to recompute any "
        "Sharpe-, drawdown-, or PnL-based ratio at any granularity. "
        "Schema matches the regular Backtest tab's CSV (via the "
        "`strategy_type` column) so both can be stacked in the same "
        "downstream app."
    )

    # Canonical summary frame (raw numeric, identical column set to the
    # Backtest tab's CSV). Single-leg-specific columns are NaN for WO;
    # WO-specific columns are populated.
    summary_canon_rows_wo = []
    for name, trades in results.items():
        df_s = worstof_trades_to_df(trades)
        if df_s.empty:
            continue
        s = worstof_summarize(df_s)
        g2p = s.get("gain_to_pain", 0.0)
        summary_canon_rows_wo.append({
            # Identifier
            "strategy_name": name,
            "strategy_type": "worst_of",
            "n_trades": int(s.get("n_trades", 0)),
            # Money totals (engine uses `total_premium_paid_usd` for WO,
            # `total_premium_usd` for single — alias to the same canonical key)
            "notional_usd": s.get("notional_usd", 0.0),
            "total_premium_paid_usd": s.get("total_premium_paid_usd", 0.0),
            "total_tx_cost_usd": s.get("total_tx_cost_usd", 0.0),
            "total_payout_usd": s.get("total_payout_usd", 0.0),
            "total_pnl_usd": s.get("total_pnl_usd", 0.0),
            "max_drawdown_usd": s.get("max_drawdown_usd", 0.0),
            # Rates / ratios (WO uses `win_rate` as a fraction 0-1; convert)
            "win_rate_pct": float(s.get("win_rate", 0.0)) * 100,
            "premium_recovery_pct": s.get("premium_recovery_pct", 0.0),
            # Sharpe block (all six)
            "sharpe_monthly": s.get("sharpe_monthly", 0.0),
            "annual_sharpe_mean": s.get("annual_sharpe_mean", 0.0),
            "annual_sharpe_min": s.get("annual_sharpe_min", 0.0),
            "annual_sharpe_std": s.get("annual_sharpe_std", 0.0),
            "annual_sharpe_cv": s.get("annual_sharpe_cv", 0.0),
            "annual_sharpe_score": s.get("annual_sharpe_score", 0.0),
            # Cross-year consistency
            "n_years": int(s.get("n_years", 0)),
            "pct_positive_years": s.get("pct_positive_years", 0.0),
            "min_annual_pnl_usd": s.get("min_annual_pnl_usd", 0.0),
            "calmar": s.get("calmar", 0.0),
            "gain_to_pain": (g2p if g2p != float("inf") else np.nan),
            "ulcer_index": s.get("ulcer_index", 0.0),
            # Single-leg placeholders (NaN for WO)
            "feasibility_pct": np.nan,
            "ko_rate_pct": np.nan,
            # WO-specific (populated)
            "leg_a_ko_rate_pct": float(s.get("leg_a_ko_rate", 0.0)) * 100,
            "leg_b_ko_rate_pct": float(s.get("leg_b_ko_rate", 0.0)) * 100,
            "both_survive_rate_pct": float(s.get("both_survive_rate", 0.0)) * 100,
            "structure_vs_min_leg_pct": s.get("structure_vs_min_leg_pct", 0.0),
        })
    summary_canon_df_wo = (pd.DataFrame(summary_canon_rows_wo)
                              if summary_canon_rows_wo else pd.DataFrame())

    # Time-series CSV (long-format daily/monthly/annual for every strategy).
    # Augment with state_a / state_b columns when regime panels are
    # registered (Phase 4). Worst-of has two pairs so two state cols.
    from core.backtest import augment_time_series_with_regime
    from core.regimes import get_regime_panel
    ts_frames_wo = []
    for name, trades in results.items():
        df_ts = worstof_trades_to_df(trades)
        if df_ts.empty:
            continue
        ts = worstof_export_time_series(df_ts)
        if ts.empty:
            continue
        pair_a_for_strat = (df_ts["leg_a_pair"].iloc[0]
                              if "leg_a_pair" in df_ts.columns else None)
        pair_b_for_strat = (df_ts["leg_b_pair"].iloc[0]
                              if "leg_b_pair" in df_ts.columns else None)
        if pair_a_for_strat:
            ts = augment_time_series_with_regime(
                ts, get_regime_panel(pair_a_for_strat),
                column_name="state_a",
            )
        if pair_b_for_strat:
            ts = augment_time_series_with_regime(
                ts, get_regime_panel(pair_b_for_strat),
                column_name="state_b",
            )
        ts.insert(0, "strategy_name", name)
        ts.insert(1, "strategy_type", "worst_of")
        ts_frames_wo.append(ts)
    ts_combined_wo = (pd.concat(ts_frames_wo, ignore_index=True)
                        if ts_frames_wo else pd.DataFrame())

    cdl_a, cdl_b = st.columns(2)
    with cdl_a:
        if not summary_canon_df_wo.empty:
            st.download_button(
                label=(f"⬇ Download summary table "
                         f"({len(summary_canon_df_wo)} rows, CSV)"),
                data=summary_canon_df_wo.to_csv(index=False).encode("utf-8"),
                file_name="eko_worstof_bulk_summary.csv",
                mime="text/csv",
                help=("Canonical schema: strategy_name, strategy_type, "
                       "n_trades, money totals, Sharpe block (including "
                       "annual_sharpe_cv and annual_sharpe_score), "
                       "consistency block, then strategy-specific. "
                       "Single-leg-specific columns are NaN here; "
                       "WO-specific columns are populated."),
                use_container_width=True,
                key="wo_summary_dl",
            )
        else:
            st.caption("_No summary rows yet._")
    with cdl_b:
        if not ts_combined_wo.empty:
            n_strats = ts_combined_wo["strategy_name"].nunique()
            st.download_button(
                label=(f"⬇ Download time series — {n_strats} strategies × "
                         f"{len(ts_combined_wo):,} rows (CSV)"),
                data=ts_combined_wo.to_csv(index=False).encode("utf-8"),
                file_name="eko_worstof_bulk_timeseries.csv",
                mime="text/csv",
                help=("Long-format: each row has period_type "
                       "('daily'|'monthly'|'annual'), period_end date, "
                       "pnl_usd, equity_usd, drawdown_usd. Monthly/annual "
                       "rows now carry end-of-period equity & drawdown "
                       "snapshots so DD-based ratios (Calmar, ulcer) "
                       "are recomputable at any granularity."),
                use_container_width=True,
                key="wo_timeseries_dl",
            )
        else:
            st.caption("_No time-series data yet._")

    # Trade-level combined ledger — heavyweight, advanced users only.
    with st.expander("Trade-level combined ledger (advanced)", expanded=False):
        st.caption("Every individual trade across all strategies in one "
                    "frame, prefixed with `strategy_name`. Useful for "
                    "audit / debugging or building your own analytics.")
        trade_frames = []
        for name, trades in results.items():
            df = worstof_trades_to_df(trades)
            if not df.empty:
                cols = ["strategy_name"] + [c for c in df.columns
                                              if c != "strategy_name"]
                trade_frames.append(df[cols])
        if trade_frames:
            combined = pd.concat(trade_frames, ignore_index=True)
            st.download_button(
                label=(f"⬇ Download combined trade ledger — "
                         f"{len(trade_frames)} strategies × "
                         f"{len(combined):,} trades (CSV)"),
                data=combined.to_csv(index=False).encode("utf-8"),
                file_name="eko_worstof_bulk_trades.csv",
                mime="text/csv",
                key="wo_trades_dl",
            )


# -----------------------------------------------------------------------------
# TAB 5: WORST-OF STRATEGY DRILLDOWN
# -----------------------------------------------------------------------------
def render_worstof_drilldown_tab():
    """Single-strategy detail view for any strategy in the latest WO run.

    Mirrors the single-strategy view that used to live on the WO tab:
    KPI cards, equity & drawdown, annual + monthly PnL, per-leg spot
    charts with gate shading, full trade ledger, per-strategy CSV.
    """
    from core.worstof import (
        worstof_trades_to_df, worstof_summarize, worstof_equity_curve,
        worstof_monthly_pnl, worstof_annual_summary,
    )
    from core.gates import gate_label, gate_chart_layers

    if "wo_results" not in st.session_state:
        st.info("Run a worst-of bulk backtest first (Worst-of tab).")
        return

    results = st.session_state["wo_results"]
    specs_by_name = st.session_state.get("wo_specs", {})
    meta_wo = st.session_state.get("wo_meta", {})

    names = [n for n in results if results[n]]
    if not names:
        st.warning("All strategies in the latest run produced zero trades. "
                    "Widen the date range, loosen the gates, or try other "
                    "strike/KO combinations.")
        return

    selected = st.selectbox("Select strategy", names, index=0,
                              key="wo_drill_select")
    trades_wo = results[selected]
    spec_wo = specs_by_name.get(selected)
    df = worstof_trades_to_df(trades_wo)
    if df.empty:
        st.warning("Empty trade ledger for this strategy.")
        return
    if spec_wo is None:
        st.warning("Spec metadata missing for this strategy — chart and "
                    "gate views may be limited.")
        return

    # ---- Header ----
    st.markdown(f"### {spec_wo.name}")
    gate_a_lbl = (gate_label(spec_wo.entry_gate_a)
                    if spec_wo.entry_gate_a else "(none)")
    gate_b_lbl = (gate_label(spec_wo.entry_gate_b)
                    if spec_wo.entry_gate_b else "(none)")
    st.caption(
        f"Run period: {meta_wo.get('start')} → {meta_wo.get('end')}  ·  "
        f"notional ${meta_wo.get('notional_usd', 0):,.0f}  ·  "
        f"tx {meta_wo.get('tx_cost_bps', 0):.1f} bps  ·  "
        f"entry gate A: **{gate_a_lbl}**  ·  "
        f"entry gate B: **{gate_b_lbl}**"
    )

    # ---- Adaptive context: show what the engine picked per date ----
    # Populated only when the spec ran in adaptive mode (build_adaptive_
    # schedule path). The fields are None otherwise.
    if ("adaptive_cluster_index" in df.columns
            and df["adaptive_cluster_index"].notna().any()):
        st.markdown("##### 🔄 Adaptive mode — per-trade decisions")
        st.caption(
            "Each trade picked its own cluster (nearest to current "
            "spot) and tenor (per the spec's tenor strategy). The "
            "distributions below summarise what the engine actually "
            "did across the backtest."
        )
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            tenor_counts = df["adaptive_chosen_tenor"].value_counts()
            tenor_lines = [
                f"<b>{t}</b>: {n} ({n/len(df)*100:.0f}%)"
                for t, n in tenor_counts.items()
            ]
            st.markdown(
                "<div class='metric-card'>"
                "<div class='metric-title'>Tenor distribution</div>"
                f"<div style='font-size:14px;line-height:1.6;'>"
                f"{'<br>'.join(tenor_lines)}</div></div>",
                unsafe_allow_html=True,
            )
        with ac2:
            cluster_counts = df["adaptive_cluster_index"].value_counts()
            cluster_lines = [
                f"<b>c{int(c)}</b>: {n} ({n/len(df)*100:.0f}%)"
                for c, n in cluster_counts.items()
            ]
            st.markdown(
                "<div class='metric-card'>"
                "<div class='metric-title'>Cluster distribution</div>"
                f"<div style='font-size:14px;line-height:1.6;'>"
                f"{'<br>'.join(cluster_lines)}</div></div>",
                unsafe_allow_html=True,
            )
        with ac3:
            mean_soj = df["adaptive_cluster_sojourn_days"].mean()
            mean_dist = df["adaptive_cluster_distance_from_spot"].mean()
            st.markdown(
                "<div class='metric-card'>"
                "<div class='metric-title'>Avg regime stats</div>"
                f"<div style='font-size:14px;line-height:1.6;'>"
                f"<b>Sojourn</b>: {mean_soj:.0f} days<br>"
                f"<b>Spot-cluster distance</b>: {mean_dist:.2f}"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    s = worstof_summarize(df)

    # ---- KPI row 1 ----
    cs1, cs2, cs3, cs4, cs5, cs6 = st.columns(6)
    with cs1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-title'>Trades</div>"
            f"<div class='metric-value'>{s['n_trades']}</div>"
            f"<div class='metric-sub'>both legs entered</div></div>",
            unsafe_allow_html=True,
        )
    with cs2:
        st.markdown(
            f"<div class='metric-card'><div class='metric-title'>Σ PnL</div>"
            f"<div class='metric-value'>{_fmt_usd(s['total_pnl_usd'])}</div>"
            f"<div class='metric-sub'>recovery {s['premium_recovery_pct']:.0f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with cs3:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Σ Premium paid</div>"
            f"<div class='metric-value'>{_fmt_usd(s['total_premium_paid_usd'])}</div>"
            f"<div class='metric-sub'>tx {_fmt_usd(s['total_tx_cost_usd'])}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with cs4:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Σ Payoff</div>"
            f"<div class='metric-value'>{_fmt_usd(s.get('total_payout_usd', 0))}</div>"
            f"<div class='metric-sub'>worst-of at expiry</div></div>",
            unsafe_allow_html=True,
        )
    with cs5:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Win rate</div>"
            f"<div class='metric-value'>{s['win_rate']*100:.1f}%</div>"
            f"<div class='metric-sub'>both ITM, neither KO</div></div>",
            unsafe_allow_html=True,
        )
    with cs6:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Max drawdown</div>"
            f"<div class='metric-value'>{_fmt_usd(s['max_drawdown_usd'])}</div>"
            f"<div class='metric-sub'>realized at expiry</div></div>",
            unsafe_allow_html=True,
        )

    # ---- KPI row 2 ----
    cs7, cs8, cs9, cs10 = st.columns(4)
    with cs7:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Leg A KO rate</div>"
            f"<div class='metric-value'>{s['leg_a_ko_rate']*100:.1f}%</div>"
            f"<div class='metric-sub'>{spec_wo.leg_a_pair} · "
            f"avg P_mid {_fmt_usd(s['avg_leg_a_premium_mid_usd'])}</div></div>",
            unsafe_allow_html=True,
        )
    with cs8:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Leg B KO rate</div>"
            f"<div class='metric-value'>{s['leg_b_ko_rate']*100:.1f}%</div>"
            f"<div class='metric-sub'>{spec_wo.leg_b_pair} · "
            f"avg P_mid {_fmt_usd(s['avg_leg_b_premium_mid_usd'])}</div></div>",
            unsafe_allow_html=True,
        )
    with cs9:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Both survive</div>"
            f"<div class='metric-value'>{s['both_survive_rate']*100:.1f}%</div>"
            f"<div class='metric-sub'>neither KO'd</div></div>",
            unsafe_allow_html=True,
        )
    with cs10:
        avg_paid = s['avg_premium_paid_usd']
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Avg structure premium</div>"
            f"<div class='metric-value'>{_fmt_usd(avg_paid)}</div>"
            f"<div class='metric-sub'>≈ {s['structure_vs_min_leg_pct']:.0f}% of min leg</div></div>",
            unsafe_allow_html=True,
        )

    # ---- Equity + drawdown ----
    eq = worstof_equity_curve(df)
    if not eq.empty:
        st.markdown("---")
        st.markdown("#### Equity & drawdown (realized at expiry)")
        from plotly.subplots import make_subplots
        fig_eq = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  row_heights=[0.7, 0.3], vertical_spacing=0.04)
        fig_eq.add_trace(go.Scatter(
            x=eq.index, y=eq["equity_usd"], mode="lines",
            line=dict(color="#22c55e", width=2),
            name="Equity",
            hovertemplate="%{x|%Y-%m-%d}<br>Equity: $%{y:,.0f}<extra></extra>",
        ), row=1, col=1)
        fig_eq.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)",
                                            dash="dot"), row=1, col=1)
        fig_eq.add_trace(go.Scatter(
            x=eq.index, y=eq["drawdown_usd"], mode="lines",
            fill="tozeroy", line=dict(color="#ef4444", width=1),
            fillcolor="rgba(239,68,68,0.20)", name="Drawdown",
            hovertemplate="%{x|%Y-%m-%d}<br>DD: $%{y:,.0f}<extra></extra>",
        ), row=2, col=1)
        fig_eq.update_layout(
            height=420, template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            margin=dict(l=20, r=20, t=20, b=20), showlegend=False,
        )
        fig_eq.update_yaxes(title="Equity (USD)", tickformat="$,.0f", row=1, col=1)
        fig_eq.update_yaxes(title="DD (USD)", tickformat="$,.0f", row=2, col=1)
        fig_eq.update_xaxes(title="Expiry date", row=2, col=1)
        st.plotly_chart(fig_eq, use_container_width=True)

    # ---- Annual + monthly bars ----
    st.markdown("#### Annual & monthly PnL")
    annual = worstof_annual_summary(df)
    monthly = worstof_monthly_pnl(df)
    cap, cm = st.columns(2)
    with cap:
        if not annual.empty:
            colors = ["#22c55e" if v >= 0 else "#ef4444"
                      for v in annual["total_pnl_usd"]]
            fig_a = go.Figure()
            fig_a.add_trace(go.Bar(
                x=annual.index.astype(str), y=annual["total_pnl_usd"],
                marker_color=colors, name="Annual PnL",
                hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
            ))
            fig_a.update_layout(
                height=280, template="plotly_dark",
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                margin=dict(l=20, r=20, t=20, b=20), showlegend=False,
                yaxis=dict(title="PnL (USD)", tickformat="$,.0f"),
                xaxis=dict(title="Year"),
            )
            st.plotly_chart(fig_a, use_container_width=True)
    with cm:
        if not monthly.empty:
            colors = ["#22c55e" if v >= 0 else "#ef4444" for v in monthly]
            fig_m = go.Figure()
            fig_m.add_trace(go.Bar(
                x=monthly.index, y=monthly.values, marker_color=colors,
                name="Monthly PnL",
                hovertemplate="%{x|%b %Y}<br>$%{y:,.0f}<extra></extra>",
            ))
            fig_m.update_layout(
                height=280, template="plotly_dark",
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                margin=dict(l=20, r=20, t=20, b=20), showlegend=False,
                yaxis=dict(title="PnL (USD)", tickformat="$,.0f"),
                xaxis=dict(title="Month"),
            )
            st.plotly_chart(fig_m, use_container_width=True)

    # ---- Spot + per-leg gate charts ----
    st.markdown("---")
    st.markdown("#### Spot, per-leg gate indicator & trade entries")
    st.caption(
        "Each chart shows ONE leg's pair with ITS OWN gate shaded. "
        "Trade entries (diamonds) only fall where BOTH legs' gates are "
        "satisfied simultaneously — so entries lie in the intersection "
        "of green windows across the two charts."
    )
    chart_legs = [
        (spec_wo.leg_a_pair, spec_wo.entry_gate_a, "Leg A"),
        (spec_wo.leg_b_pair, spec_wo.entry_gate_b, "Leg B"),
    ]
    seen_pairs: set[str] = set()
    for pair_chart, leg_gate, leg_lbl in chart_legs:
        if pair_chart in seen_pairs:
            st.caption(f"_(Leg B uses the same pair as Leg A — chart "
                        f"shown above. Note Leg B's gate ({gate_label(leg_gate)}) "
                        f"may differ from Leg A's.)_")
            continue
        seen_pairs.add(pair_chart)
        leg_gate_lbl = gate_label(leg_gate) if leg_gate else "(none)"
        st.markdown(f"**{leg_lbl}: {pair_chart}** — gate: *{leg_gate_lbl}*")

        chart_start = df["trade_date"].min() - timedelta(days=300)
        chart_end = df["expiry_date"].max() + timedelta(days=5)
        spot_panel = load_panel(folder, "SPOT", None,
                                  prefer=meta_wo.get("prefer", "offshore"),
                                  pairs=(pair_chart,))
        if spot_panel.empty or pair_chart not in spot_panel.columns:
            st.info(f"No spot panel for {pair_chart}.")
            continue
        spot_full = spot_panel[pair_chart].dropna()
        spot_s = spot_full.loc[
            (spot_full.index >= pd.Timestamp(chart_start)) &
            (spot_full.index <= pd.Timestamp(chart_end))
        ]

        # Per-leg gate layers (this leg's gate only)
        layers = gate_chart_layers(spot_full, leg_gate)
        for ln in layers["price_lines"] + layers["subplot_lines"]:
            ln["series"] = ln["series"].reindex(spot_s.index)
        gate_mask_full = layers["mask"]
        gate_mask_window = (gate_mask_full.reindex(spot_s.index)
                             if gate_mask_full is not None else None)

        entry_dates = pd.to_datetime(df["trade_date"]).dt.normalize()
        entry_spots = pd.Series(spot_s).reindex(entry_dates).dropna()

        from plotly.subplots import make_subplots
        needs_subplot = layers["panel"] in ("subplot", "both")
        if needs_subplot:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                   row_heights=[0.65, 0.35],
                                   vertical_spacing=0.05)
            spot_row, ind_row = 1, 2
        else:
            fig = go.Figure()
            spot_row = ind_row = None

        if leg_gate and gate_mask_window is not None:
            on = gate_mask_window.fillna(False).astype(int)
            edges = on.diff().fillna(on)
            start_idx = on.index[edges == 1].tolist()
            end_idx = on.index[edges == -1].tolist()
            if not on.empty and on.iloc[-1] == 1:
                end_idx.append(on.index[-1])
            for s_, e_ in zip(start_idx, end_idx):
                if needs_subplot:
                    fig.add_vrect(x0=s_, x1=e_, fillcolor="#22c55e",
                                    opacity=0.10, line_width=0,
                                    layer="below", row=1, col=1)
                    fig.add_vrect(x0=s_, x1=e_, fillcolor="#22c55e",
                                    opacity=0.10, line_width=0,
                                    layer="below", row=2, col=1)
                else:
                    fig.add_vrect(x0=s_, x1=e_, fillcolor="#22c55e",
                                    opacity=0.10, line_width=0,
                                    layer="below")

        def _add_wo(trace, row=None):
            if row is not None:
                fig.add_trace(trace, row=row, col=1)
            else:
                fig.add_trace(trace)

        _add_wo(go.Scatter(
            x=spot_s.index, y=spot_s.values, mode="lines", name="Spot",
            line=dict(color="#38bdf8", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>Spot: %{y:.4f}<extra></extra>",
        ), row=spot_row)

        for ln in layers["price_lines"]:
            _add_wo(go.Scatter(
                x=ln["series"].index, y=ln["series"].values, mode="lines",
                name=ln["name"],
                line=dict(color=ln["color"], width=1.5, dash=ln["dash"]),
                hovertemplate="%{x|%Y-%m-%d}<br>" + ln["name"] +
                                ": %{y:.4f}<extra></extra>",
            ), row=spot_row)

        for ln in layers["subplot_lines"]:
            _add_wo(go.Scatter(
                x=ln["series"].index, y=ln["series"].values, mode="lines",
                name=ln["name"],
                line=dict(color=ln["color"], width=1.5, dash=ln["dash"]),
                hovertemplate="%{x|%Y-%m-%d}<br>" + ln["name"] +
                                ": %{y:.4f}<extra></extra>",
            ), row=ind_row)

        if len(entry_spots) > 0:
            _add_wo(go.Scatter(
                x=entry_spots.index, y=entry_spots.values, mode="markers",
                name=f"Entries ({len(entry_spots)})",
                marker=dict(color="#a3e635", size=6, symbol="diamond",
                              line=dict(color="#365314", width=0.5)),
                hovertemplate=("%{x|%Y-%m-%d}<br>Spot @ entry: %{y:.4f}"
                                 "<extra></extra>"),
            ), row=spot_row)

        if needs_subplot:
            fig.update_yaxes(title=f"Spot ({pair_chart})",
                                gridcolor="rgba(255,255,255,0.08)",
                                row=1, col=1)
            fig.update_yaxes(title=layers["subplot_title"],
                                gridcolor="rgba(255,255,255,0.08)",
                                ticksuffix="%", row=2, col=1)
            fig.update_xaxes(title="Trade date",
                                gridcolor="rgba(255,255,255,0.08)",
                                row=2, col=1)
            fig.update_layout(height=440)
        else:
            fig.update_layout(
                xaxis=dict(title="Trade date",
                              gridcolor="rgba(255,255,255,0.08)"),
                yaxis=dict(title=f"Spot ({pair_chart})",
                              gridcolor="rgba(255,255,255,0.08)"),
                height=300,
            )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            margin=dict(l=20, r=20, t=10, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.30,
                          xanchor="left", x=0,
                          font=dict(size=11, color="#cbd5e1")),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Trade ledger ----
    st.markdown("---")
    st.markdown("#### Trade ledger")
    st.caption(
        "**Structure accounting:**  "
        "`structure_premium_mid = min(P_A_mid, P_B_mid) / 3`,  "
        "`paid = mid + tx`,  "
        "`worst_of_payoff = min(payoff_A, payoff_B)`,  "
        "`pnl = worst_of_payoff − paid`. "
        "Each leg premium and payoff is at its own pair's spot/vol."
    )

    display_cols = [
        "trade_date", "expiry_date",
    ]
    # Adaptive columns (only if populated — non-adaptive trades have
    # these as None and they'd just clutter the table)
    if ("adaptive_cluster_index" in df.columns
            and df["adaptive_cluster_index"].notna().any()):
        display_cols.extend([
            "adaptive_cluster_index", "adaptive_chosen_tenor",
            "adaptive_cluster_sojourn_days",
        ])
    display_cols.extend([
        # Leg A
        "leg_a_pair", "leg_a_strike_delta_label", "leg_a_ko_delta_label",
        "leg_a_spot", "leg_a_strike", "leg_a_barrier",
        "leg_a_spot_at_expiry", "leg_a_knocked_out",
        "leg_a_premium_mid_usd", "leg_a_payoff_usd",
        # Leg B
        "leg_b_pair", "leg_b_strike_delta_label", "leg_b_ko_delta_label",
        "leg_b_spot", "leg_b_strike", "leg_b_barrier",
        "leg_b_spot_at_expiry", "leg_b_knocked_out",
        "leg_b_premium_mid_usd", "leg_b_payoff_usd",
        # Structure
        "structure_premium_mid_usd", "tx_cost_usd",
        "structure_premium_paid_usd", "worst_of_payoff_usd", "pnl_usd",
    ])
    display_rename = {
        "leg_a_premium_mid_usd": "P_A_mid_usd",
        "leg_b_premium_mid_usd": "P_B_mid_usd",
        "structure_premium_mid_usd": "struct_mid_usd  (= min/3)",
        "tx_cost_usd": "tx_usd",
        "structure_premium_paid_usd": "paid_usd",
        "worst_of_payoff_usd": "worst_of_payoff_usd",
        "pnl_usd": "pnl_usd  (= payoff − paid)",
    }
    show = df[display_cols].copy()
    for c in ("leg_a_spot", "leg_a_strike", "leg_a_barrier",
                "leg_a_spot_at_expiry", "leg_b_spot", "leg_b_strike",
                "leg_b_barrier", "leg_b_spot_at_expiry"):
        show[c] = show[c].round(4)
    for c in ("leg_a_premium_mid_usd", "leg_a_payoff_usd",
                "leg_b_premium_mid_usd", "leg_b_payoff_usd",
                "structure_premium_mid_usd", "tx_cost_usd",
                "structure_premium_paid_usd", "worst_of_payoff_usd",
                "pnl_usd"):
        show[c] = show[c].round(2)
    show = show.rename(columns=display_rename)
    st.dataframe(show, use_container_width=True, hide_index=True, height=360)

    # CSV download (full record for this strategy)
    csv = df.to_csv(index=False).encode("utf-8")
    safe_name = (spec_wo.name.replace(" ", "_").replace("[", "")
                    .replace("]", "").replace("∧", "and").replace("·", "")
                    .replace("/", "-").replace("@", "at"))
    st.download_button(
        label="⬇ Download ledger for this strategy (CSV)",
        data=csv,
        file_name=f"eko_worstof_{safe_name}.csv",
        mime="text/csv",
        key="wo_drill_dl",
    )



# =============================================================================
# TAB 6: EKO PORTFOLIO  +  TAB 7: EKO PORTFOLIO DRILLDOWN
# =============================================================================
# A "basket" portfolio backtest. One STRATEGY = one (tenor × direction
# × strike-Δ × KO-Δ × gate) combination applied UNIFORMLY to every
# pair in the selected basket. With 7 pairs × 1 tenor × 1 dir × 1 K ×
# 1 KO × 1 gate, that's 1 strategy holding 7 pair-positions per
# trading day. The basket is the unit; individual pairs are sub-rows
# within each strategy's pooled ledger.
#
# Notional convention: equal-capital-per-pair. The UI notional applies
# in full to EVERY pair inside the basket. So $10M × 7 pairs = $70M of
# capital deployed per trading day per strategy.
#
# Implementation: under the hood we still call build_strategy_grid /
# run_grid which return per-pair specs and per-pair trade lists. The
# basket layer pools the per-pair trade lists under a basket-level
# strategy name. The engine sees only single-pair specs; aggregation
# is a presentation concern.
#
# session_state keys:
#   eko_port_results    : {basket_name: list[Trade]}  — POOLED ledger
#   eko_port_specs      : {basket_name: list[StrategySpec]} — the
#                         per-pair specs that built this basket
#   eko_port_mtm_curves : {basket_name: DataFrame}    — SUMMED MTM
#                         curve across pairs (or None)
#   eko_port_meta       : run-level inputs for the header strip

EKO_PORT_DEFAULT_PAIRS = [
    "USDCNH", "USDINR", "USDKRW", "USDJPY",
    "USDSGD", "USDTHB", "USDTWD",
]


def _basket_strategy_name(tenor: str, direction_label: str,
                            strike_label: str, ko_label: str,
                            gate_key) -> str:
    """Concise basket-strategy name. Pair list deliberately omitted —
    the meta dict carries the full basket; this string is the on-screen
    selector label so we keep it tight."""
    from core.gates import gate_label
    gate_str = (f"  [{gate_label(gate_key)}]" if gate_key else "")
    # Direction labels are 'Call (up-and-out)' or 'Put (down-and-out)' —
    # compress to 'Call-UO' / 'Put-DO' for readability.
    dir_short = (direction_label.replace("Call (up-and-out)", "Call-UO")
                    .replace("Put (down-and-out)", "Put-DO"))
    return (f"BASKET  {dir_short}  {tenor}  {strike_label}/H@{ko_label}"
              f"{gate_str}")


def _wo_basket_strategy_name(tenor: str, direction_label: str,
                                sd_a_label: str, kd_a_label: str,
                                gate_a_key,
                                sd_b_label: str, kd_b_label: str,
                                gate_b_key) -> str:
    """Concise WO basket name capturing per-leg params (uniform across
    all 2-leg crosses inside the basket)."""
    from core.gates import gate_label
    dir_short = (direction_label.replace("Call (up-and-out)", "Call-UO")
                    .replace("Put (down-and-out)", "Put-DO"))
    ga_str = f"[{gate_label(gate_a_key)}]" if gate_a_key else "[no gate]"
    gb_str = f"[{gate_label(gate_b_key)}]" if gate_b_key else "[no gate]"
    return (f"WO-BASKET  {dir_short}  {tenor}  "
              f"A:{sd_a_label}/H@{kd_a_label} {ga_str}  ∧  "
              f"B:{sd_b_label}/H@{kd_b_label} {gb_str}")


def _sum_mtm_curves(curves: list) -> "pd.DataFrame | None":
    """Sum a list of per-pair MTM equity curves into a single basket
    curve. Aligns on date index, forward-fills holes from pair-specific
    missing days, recomputes peak/drawdown from the summed equity (the
    per-pair peak is NOT the basket peak).

    Empty inputs return None — caller handles that as "no MTM"."""
    valid = [c for c in curves if c is not None and not c.empty]
    if not valid:
        return None
    # Equity is the cumulative net P&L per pair. Summing it gives the
    # basket's cumulative net P&L by date. Forward-fill so a pair that
    # has no trades yet contributes zero (after fillna(0) below).
    eq_list = [c["equity_usd"] for c in valid]
    wide = pd.concat(eq_list, axis=1).sort_index()
    wide = wide.ffill().fillna(0.0)
    basket_eq = wide.sum(axis=1)
    out = pd.DataFrame({"equity_usd": basket_eq})
    out["pnl_usd"] = out["equity_usd"].diff().fillna(out["equity_usd"].iloc[0])
    out["peak_usd"] = out["equity_usd"].cummax()
    out["drawdown_usd"] = out["equity_usd"] - out["peak_usd"]
    out["drawdown_usd_pos"] = -out["drawdown_usd"]
    return out


def render_eko_portfolio_tab():
    pairs_avail = _list_pairs(folder)
    if not pairs_avail:
        st.error("No SPOT data found.")
        return

    st.markdown("### EKO Portfolio configuration")
    st.caption(
        "A **basket** portfolio backtest. One **strategy** = one "
        "`(tenor × direction × strike Δ × KO Δ × gate)` combination, "
        "applied **uniformly to every pair** in the basket. With 7 "
        "pairs × 1 tenor × 1 strike × 1 KO × 1 gate, that's 1 strategy "
        "holding 7 pair-positions per trading day. Notional applies "
        "PER PAIR (so $10M × 7 pairs = $70M deployed)."
    )

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        default_pairs = [p for p in EKO_PORT_DEFAULT_PAIRS
                         if p in pairs_avail]
        if not default_pairs:
            default_pairs = pairs_avail[:1]
        pairs_sel = st.multiselect(
            "Currency pairs (the basket)",
            pairs_avail,
            default=default_pairs,
            key="ep_pairs",
            help=("This is the BASKET. Every strategy in the run "
                   "applies its (tenor, strike, KO, gate, direction) "
                   "to all of these pairs simultaneously."),
        )
        tenors_sel = st.multiselect(
            "Tenors", TENOR_LIST, default=["1M"], key="ep_tenors",
            help=("Each selected tenor produces its own basket "
                   "strategy. Same tenor used across all pairs in "
                   "that strategy."),
        )
        directions_sel = st.multiselect(
            "Direction(s)", list(DIRECTIONS.keys()),
            default=["Call (up-and-out)"],
            key="ep_directions",
            help=("Each selected direction is its own basket strategy."),
        )

    with cc2:
        deltas_sel = st.multiselect(
            "Strike Δ list",
            list(DELTA_CHOICES.keys()),
            default=["ATM"], key="ep_deltas",
            help=("Each selected Δ is its own basket strategy. "
                   "ATM (Δ=0) bypasses the KO-vs-strike filter."),
        )
        ko_delta_labels = st.multiselect(
            "KO Δ list",
            list(KO_DELTA_CHOICES.keys()),
            default=["20Δ"], key="ep_ko_deltas",
            help=("Each selected KO Δ is its own basket strategy. "
                   "Combos where KO Δ ≥ strike Δ (non-ATM) are "
                   "auto-filtered by the engine."),
        )

        from core.gates import GATE_REGISTRY
        gate_options = ["(no gate)"] + list(GATE_REGISTRY.keys())
        gate_labels = st.multiselect(
            "Gate(s)",
            gate_options,
            default=["(no gate)"],
            key="ep_gates",
            format_func=lambda k: ("(no gate)" if k == "(no gate)"
                                   else GATE_REGISTRY[k][0]),
            help=("Each selected gate is its own basket strategy. "
                   "Same gate applied to every pair in the basket."),
        )
        gate_keys = [None if k == "(no gate)" else k
                     for k in gate_labels]

        trade_mode = st.radio(
            "Trade mode",
            ["stack", "single"],
            index=0, horizontal=True, key="ep_trade_mode",
            help=("**stack**: each pair opens a new trade on every "
                   "eligible date — overlapping per-pair book. "
                   "**single**: each pair holds at most one open "
                   "trade at a time."),
        )

    with cc3:
        date_max_avail = _date.today()
        date_min_avail = _date(2018, 1, 1)
        try:
            spot_all = load_panel(folder, "SPOT", None)
            if not spot_all.empty:
                date_max_avail = spot_all.index.max().date()
                date_min_avail = spot_all.index.min().date()
        except Exception:
            pass
        default_start = max(_date(2023, 1, 1), date_min_avail)
        if default_start >= date_max_avail:
            default_start = date_min_avail
        start_date = st.date_input(
            "Start date", value=default_start,
            min_value=date_min_avail, max_value=date_max_avail,
            key="ep_start",
        )
        end_date = st.date_input(
            "End date", value=date_max_avail,
            min_value=date_min_avail, max_value=date_max_avail,
            key="ep_end",
        )
        tx_cost_bps = st.number_input(
            "Transaction cost (bps of notional)",
            min_value=0.0, max_value=20.0, value=2.0, step=0.5,
            key="ep_tx",
        )
        prefer_em = st.radio(
            "EM pairs variant", ["offshore", "onshore"], index=0,
            horizontal=True, key="ep_prefer_em",
        )
        notional_usd = st.number_input(
            "Notional (USD, per pair)",
            min_value=100_000.0, max_value=200_000_000.0,
            value=10_000_000.0, step=1_000_000.0, format="%.0f",
            key="ep_notional",
            help=("Applied PER PAIR. $10M with 7 pairs = $70M total "
                   "capital deployed per trading day per strategy."),
        )
        enable_mtm = st.checkbox(
            "Mark-to-market mode (daily)",
            value=False, key="ep_mtm",
            help=("On = daily mark-to-mid; off = realized at expiry. "
                   "MTM curves are summed across pairs to form the "
                   "basket curve."),
        )

    # Basket-strategy count = product of axis lengths (NOT including
    # pairs — pairs are aggregated, not faceted).
    n_baskets = (len(tenors_sel) * len(directions_sel) * len(deltas_sel)
                  * len(ko_delta_labels) * max(len(gate_keys), 1))
    st.caption(
        f"**{n_baskets}** basket strategies will run across "
        f"**{len(pairs_sel)} pairs** ({len(tenors_sel)} tenors × "
        f"{len(directions_sel)} dirs × {len(deltas_sel)} strike Δs × "
        f"{len(ko_delta_labels)} KO Δs × {max(len(gate_keys), 1)} "
        f"gates). Each strategy holds all {len(pairs_sel)} pair-"
        f"positions per trading day. Trade mode: `{trade_mode}`."
    )

    can_run = (pairs_sel and tenors_sel and directions_sel and deltas_sel
               and ko_delta_labels and len(gate_keys) > 0)
    run_clicked = st.button(
        "▶ Run EKO portfolio backtest", type="primary",
        disabled=not can_run, key="ep_run_btn",
    )

    if run_clicked:
        # Cross the basket axes (NOT pairs). For each axis combination
        # we construct per-pair specs, run them, and pool the resulting
        # trade lists under a single basket strategy name.
        from itertools import product

        all_basket_results: dict[str, list] = {}
        all_basket_specs: dict[str, list] = {}
        # Specs assembled for compute_mtm_curves (flat per-pair list);
        # we'll group their MTM curves into basket curves after.
        all_flat_specs: list = []
        # Per-basket lookup of which flat-spec names belong to it
        basket_member_names: dict[str, list[str]] = {}

        axis_combos = list(product(
            tenors_sel, directions_sel, deltas_sel,
            ko_delta_labels, gate_keys,
        ))
        n_baskets_actual = len(axis_combos)
        progress_bar = st.progress(0.0, text="Starting basket runs…")
        t0 = time.time()
        last_update = [t0]

        def cb(p, name):
            """Per-strategy progress callback from run_grid. Reused
            across basket axes — we scale the within-basket progress
            to the global progress."""
            now = time.time()
            if now - last_update[0] > 0.1 or p >= 1.0:
                progress_bar.progress(min(p, 1.0),
                                      text=f"Running: {name} "
                                            f"({p*100:.0f}%)")
                last_update[0] = now

        for axis_i, (tenor, dir_label, strike_label, ko_label, gk) in \
                enumerate(axis_combos):
            basket_name = _basket_strategy_name(
                tenor, dir_label, strike_label, ko_label, gk,
            )
            # Build the per-pair specs sharing these uniform params.
            specs_this_basket = build_strategy_grid(
                pairs=pairs_sel,
                deltas=[(strike_label, DELTA_CHOICES[strike_label])],
                tenors=[tenor],
                directions=[DIRECTIONS[dir_label]],
                tx_cost_bps=tx_cost_bps,
                prefer=prefer_em,
                ko_method="delta",
                target_ko_delta=KO_DELTA_CHOICES[ko_label],
                ko_delta_label=ko_label,
                entry_gate=gk,
                trade_mode=trade_mode,
                pricing_model=pricing_model,
            )
            if not specs_this_basket:
                # All-filtered (e.g. KO ≥ strike for every pair —
                # impossible normally but defensive)
                all_basket_results[basket_name] = []
                all_basket_specs[basket_name] = []
                basket_member_names[basket_name] = []
                continue

            # Run only this axis-combo's per-pair specs (cheap: pairs
            # are preloaded once per run_grid call so multiple axis
            # combos sharing pairs will re-preload — acceptable cost
            # given the per-pair pricing is the heavy step).
            sub_results = run_grid(
                folder, specs_this_basket, start_date, end_date,
                notional_usd=notional_usd,
                progress_cb=lambda p, name, _i=axis_i: cb(
                    (_i + p) / n_baskets_actual,
                    f"[basket {_i+1}/{n_baskets_actual}] {name}",
                ),
            )
            # Pool all per-pair trade lists into one basket trade list.
            pooled: list = []
            for s in specs_this_basket:
                pooled.extend(sub_results.get(s.name, []))
            all_basket_results[basket_name] = pooled
            all_basket_specs[basket_name] = specs_this_basket
            all_flat_specs.extend(specs_this_basket)
            basket_member_names[basket_name] = [s.name
                                                  for s in specs_this_basket]

        # MTM: compute per-pair curves on the flat spec list, then sum
        # within each basket. We rebuild a per-name results dict that
        # compute_mtm_curves expects.
        mtm_curves_basket = None
        if enable_mtm and all_flat_specs:
            # Reconstruct flat results dict {per_pair_spec_name: trades}
            flat_results: dict[str, list] = {}
            for basket_name, specs_in_basket in all_basket_specs.items():
                # We didn't save per-pair sub_results separately; rebuild
                # by partitioning the pooled list by spec name. Each
                # Trade carries strategy_name = its per-pair spec name.
                pooled = all_basket_results[basket_name]
                for spec in specs_in_basket:
                    flat_results[spec.name] = [
                        t for t in pooled if t.strategy_name == spec.name
                    ]
            progress_bar.progress(0.5, text="MTM aggregation…")
            per_pair_mtm = compute_mtm_curves(
                folder, all_flat_specs, flat_results,
                progress_cb=lambda p, name: cb(p, f"MTM {name}"),
            )
            mtm_curves_basket = {}
            for basket_name, member_names in basket_member_names.items():
                per_pair_list = [per_pair_mtm.get(nm)
                                 for nm in member_names]
                mtm_curves_basket[basket_name] = _sum_mtm_curves(
                    per_pair_list
                )

        elapsed = time.time() - t0
        progress_bar.empty()

        st.session_state["eko_port_results"] = all_basket_results
        st.session_state["eko_port_specs"] = all_basket_specs
        st.session_state["eko_port_mtm_curves"] = mtm_curves_basket
        st.session_state["eko_port_meta"] = {
            "pairs": pairs_sel,
            "tenors": tenors_sel,
            "directions": directions_sel,
            "strike_deltas": deltas_sel,
            "ko_deltas": ko_delta_labels,
            "gates": gate_keys,
            "trade_mode": trade_mode,
            "start": start_date, "end": end_date,
            "tx_cost_bps": tx_cost_bps,
            "prefer_em": prefer_em,
            "notional_usd": notional_usd,
            "mtm_enabled": enable_mtm,
            "n_baskets": n_baskets_actual,
            "elapsed": elapsed,
        }
        total_trades = sum(len(t) for t in all_basket_results.values())
        st.success(
            f"Done in {elapsed:.1f}s — {n_baskets_actual} basket "
            f"strategies, {total_trades} pooled trades total "
            f"({len(pairs_sel)} pairs per basket). Switch to the EKO "
            f"Portfolio drilldown tab to inspect."
        )

    # --- Summary table for the latest run ---
    if "eko_port_results" not in st.session_state:
        st.info("Configure axes above and click **Run** to see a "
                  "summary. Drill into any one basket strategy on the "
                  "*EKO Portfolio drilldown* tab.")
        return

    results = st.session_state["eko_port_results"]
    mtm_curves = st.session_state.get("eko_port_mtm_curves")
    meta = st.session_state.get("eko_port_meta", {})
    mtm_on = meta.get("mtm_enabled", False) and mtm_curves is not None

    st.markdown("---")
    st.markdown("### Latest run — basket strategies")
    pairs_str = ", ".join(meta.get("pairs", []))
    st.caption(
        f"**Basket:** {pairs_str}  ·  "
        f"period {meta.get('start')} → {meta.get('end')}  ·  "
        f"${meta.get('notional_usd', 0):,.0f} per pair  ·  "
        f"tx {meta.get('tx_cost_bps', 0):.1f} bps  ·  "
        f"mode `{meta.get('trade_mode', 'stack')}`  ·  "
        f"**{'MTM (daily)' if mtm_on else 'Realized at expiry'}**"
    )

    rows = []
    for name, trades_list in results.items():
        if not trades_list:
            rows.append({"Basket strategy": name, "n trades": 0})
            continue
        sdf = trades_to_df(trades_list)
        s = summarize_strategy(sdf)
        if mtm_on and mtm_curves and mtm_curves.get(name) is not None:
            sm = summarize_mtm(mtm_curves[name])
            sharpe_lbl = "Sharpe (d)"
            sharpe_val = sm.get("sharpe_daily_mtm", 0.0)
            maxdd_val = sm.get("max_drawdown_usd_mtm",
                               s.get("max_drawdown_usd", 0))
        else:
            sharpe_lbl = "Sharpe (m)"
            sharpe_val = s["sharpe_monthly"]
            maxdd_val = s.get("max_drawdown_usd", 0)
        rows.append({
            "Basket strategy": name,
            "n": s["n_trades"],
            "Pairs": sdf["pair"].nunique() if "pair" in sdf.columns else 0,
            "Win %": f"{s['win_rate_pct']:.0f}",
            "KO %": f"{s['ko_rate_pct']:.0f}",
            "Σ Premium": _fmt_usd(s.get("total_premium_usd", 0)),
            "Σ Payoff": _fmt_usd(s.get("total_payout_usd", 0)),
            "PnL": _fmt_usd(s.get("total_pnl_usd", 0)),
            sharpe_lbl: f"{sharpe_val:+.2f}",
            "Max DD": _fmt_usd(maxdd_val),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

    # =====================================================================
    # Downloads — canonical schema matching the Backtest tab
    # =====================================================================
    # The CSVs feed a downstream app that stacks single-leg, worst-of,
    # AND now EKO/WO basket runs side-by-side. To stay stackable they
    # share the same column set, identified by `strategy_type` —
    # 'single' (Backtest), 'worst_of' (Worst-of), or 'eko_basket' /
    # 'wo_basket' (the two portfolio tabs). The downstream app keys off
    # `strategy_type` when a divergence in interpretation matters
    # (e.g. a basket trade has a `pair` field but no `feasibility_pct`,
    # since basket = many pairs).
    #
    # Two CSVs per tab:
    #   1. Summary  — one row per basket strategy, headline metrics
    #   2. Time     — long-format daily/monthly/annual rows per strategy,
    #                  pnl_usd / equity_usd / drawdown_usd at each
    #                  period end. Sufficient to recompute Sharpe,
    #                  drawdown, Calmar at any granularity downstream.
    st.markdown("---")
    st.markdown("### Downloads")
    st.caption(
        "Bulk run results in CSV form. **Summary** = one row per basket "
        "strategy with headline metrics. **Time series** = long-format "
        "daily/monthly/annual rows with `pnl_usd`, `equity_usd`, "
        "`drawdown_usd` per period end. Schema matches the regular "
        "Backtest tab (via `strategy_type='eko_basket'`) so basket "
        "results can be stacked with single-leg results in the "
        "downstream app."
    )

    # --- Build canonical summary frame ---
    summary_canon_rows = []
    for name, trades_list in results.items():
        if not trades_list:
            continue
        sdf = trades_to_df(trades_list)
        s = summarize_strategy(sdf)
        g2p = s.get("gain_to_pain", 0.0)
        n_pairs_strat = (sdf["pair"].nunique()
                          if "pair" in sdf.columns else 0)
        # Pull MTM-based stats if MTM mode was on
        max_dd_canon = s.get("max_drawdown_usd", 0.0)
        if mtm_on and mtm_curves and mtm_curves.get(name) is not None:
            sm = summarize_mtm(mtm_curves[name])
            max_dd_canon = sm.get("max_drawdown_usd_mtm",
                                    s.get("max_drawdown_usd", 0.0))
        summary_canon_rows.append({
            # Identifier
            "strategy_name": name,
            "strategy_type": "eko_basket",
            "n_trades": int(s.get("n_trades", 0)),
            # Money totals — same canonical keys as Backtest CSV
            "notional_usd": s.get("notional_usd", 0.0),
            "total_premium_paid_usd": s.get("total_premium_usd", 0.0),
            "total_tx_cost_usd": s.get("total_transaction_cost_usd", 0.0),
            "total_payout_usd": s.get("total_payout_usd", 0.0),
            "total_pnl_usd": s.get("total_pnl_usd", 0.0),
            "max_drawdown_usd": max_dd_canon,
            # Rates / ratios
            "win_rate_pct": s.get("win_rate_pct", 0.0),
            "premium_recovery_pct": s.get("premium_recovery_pct", 0.0),
            # Sharpe block
            "sharpe_monthly": s.get("sharpe_monthly", 0.0),
            "annual_sharpe_mean": s.get("annual_sharpe_mean", 0.0),
            "annual_sharpe_min": s.get("annual_sharpe_min", 0.0),
            "annual_sharpe_std": s.get("annual_sharpe_std", 0.0),
            "annual_sharpe_cv": s.get("annual_sharpe_cv", 0.0),
            "annual_sharpe_score": s.get("annual_sharpe_score", 0.0),
            # Cross-year consistency
            "n_years": int(s.get("n_years", 0)),
            "pct_positive_years": s.get("pct_positive_years", 0.0),
            "min_annual_pnl_usd": s.get("min_annual_pnl_usd", 0.0),
            "calmar": s.get("calmar", 0.0),
            "gain_to_pain": (g2p if g2p != float("inf") else np.nan),
            "ulcer_index": s.get("ulcer_index", 0.0),
            # Single-leg fields — populated for EKO basket strategies
            # (each trade IS single-leg; basket aggregation is at the
            # presentation layer). Feasibility is always 100% in delta-
            # KO mode (every trade is structurally valid by construction).
            "feasibility_pct": s.get("feasibility_pct", 100.0),
            "ko_rate_pct": s.get("ko_rate_pct", 0.0),
            # Worst-of placeholders (NaN for basket too)
            "leg_a_ko_rate_pct": np.nan,
            "leg_b_ko_rate_pct": np.nan,
            "both_survive_rate_pct": np.nan,
            "structure_vs_min_leg_pct": np.nan,
            # Basket-specific extras (NaN in non-basket CSVs)
            "basket_n_pairs": int(n_pairs_strat),
            "basket_pairs": ",".join(sorted(sdf["pair"].unique())
                                       if "pair" in sdf.columns else []),
            "trade_mode": meta.get("trade_mode", "stack"),
        })
    summary_canon_df = (pd.DataFrame(summary_canon_rows)
                         if summary_canon_rows else pd.DataFrame())

    # --- Build canonical time-series frame ---
    # We export the pooled basket ledger as if it were a single strategy
    # ledger — pnl_usd / equity_usd / drawdown_usd are at the BASKET
    # level (summed across pairs), which is what the downstream app
    # cares about for risk-aggregated views. Per-pair drilldowns can
    # still be reconstructed from the trade-level ledger if needed.
    # Daily rows are dropped on purpose — the user works at monthly/
    # annual cadence; daily granularity is recomputable from the
    # trade ledger if ever needed.
    ts_frames = []
    for name, trades_list in results.items():
        if not trades_list:
            continue
        sdf = trades_to_df(trades_list)
        ts = export_strategy_time_series(sdf)
        if ts.empty:
            continue
        # Keep monthly + annual only
        ts = ts[ts["period_type"].isin(["monthly", "annual"])].copy()
        if ts.empty:
            continue
        ts.insert(0, "strategy_name", name)
        ts.insert(1, "strategy_type", "eko_basket")
        ts_frames.append(ts)
    ts_combined = (pd.concat(ts_frames, ignore_index=True)
                     if ts_frames else pd.DataFrame())

    cdl_a, cdl_b = st.columns(2)
    with cdl_a:
        if not summary_canon_df.empty:
            st.download_button(
                label=(f"⬇ Download summary table "
                         f"({len(summary_canon_df)} rows, CSV)"),
                data=summary_canon_df.to_csv(index=False).encode("utf-8"),
                file_name=_build_portfolio_download_filename(
                    meta, "EKO", "summary"),
                mime="text/csv",
                help=("Canonical schema: strategy_name, strategy_type "
                       "('eko_basket'), n_trades, money totals, Sharpe "
                       "block, consistency block, then strategy- and "
                       "basket-specific columns (basket_n_pairs, "
                       "basket_pairs, trade_mode)."),
                use_container_width=True,
                key="ep_summary_dl",
            )
        else:
            st.caption("_No summary rows yet._")
    with cdl_b:
        if not ts_combined.empty:
            n_strats = ts_combined["strategy_name"].nunique()
            st.download_button(
                label=(f"⬇ Download time series — {n_strats} strategies × "
                         f"{len(ts_combined):,} rows (CSV)"),
                data=ts_combined.to_csv(index=False).encode("utf-8"),
                file_name=_build_portfolio_download_filename(
                    meta, "EKO", "timeseries"),
                mime="text/csv",
                help=("Long-format: period_type ('monthly'|'annual'), "
                       "period_end, pnl_usd, equity_usd, drawdown_usd "
                       "at each period end. BASKET-LEVEL (pooled "
                       "across pairs). Daily rows are excluded — "
                       "available from the trade ledger if needed."),
                use_container_width=True,
                key="ep_ts_dl",
            )
        else:
            st.caption("_No time-series rows yet._")


def _build_portfolio_download_filename(meta: dict, prefix: str,
                                              kind: str) -> str:
    """Build a descriptive filename for portfolio bulk-download CSVs.

    Pattern: `{PREFIX}_{pair_tokens}_T{tenor}_S{strike}_B{barrier}_{kind}.csv`

    Examples (basket: USDJPY/USDKRW/USDTHB, tenors=[2M], strikes=[ATM,45D],
    barriers=[20D,15D]):
      EKO_JPY_KRW_THB_T2M_SATM_S45D_B20D_B15D_summary.csv
      WO-EKO_JPY_KRW_THB_T2M_SATM_S45D_B20D_B15D_timeseries.csv

    Pair tokens strip the USD prefix (USDJPY → JPY). Strike/barrier Δ
    labels strip the trailing Δ ('20Δ' → '20D'). Tenor tokens stay
    as-is. Mirror of the same helper in apps/_app12_tabs.py — inlined
    here because App 9 doesn't import from App 12.
    """
    pairs = meta.get("pairs") or []
    tenors = meta.get("tenors") or []
    strikes = meta.get("strike_deltas") or []
    barriers = meta.get("ko_deltas") or []

    def _strip_usd(p):
        return p[3:] if p.upper().startswith("USD") else p

    def _strip_delta(lbl):
        return lbl.replace("Δ", "D")

    tokens = [prefix]
    tokens.extend(_strip_usd(p) for p in pairs)
    tokens.extend(f"T{t}" for t in tenors)
    tokens.extend(f"S{_strip_delta(s)}" for s in strikes)
    tokens.extend(f"B{_strip_delta(b)}" for b in barriers)
    tokens.append(kind)
    return "_".join(tokens) + ".csv"


def _annual_sharpe_per_year(eq: "pd.DataFrame") -> "pd.Series":
    """Per-calendar-year Sharpe ratio from an equity curve.

    Reproduces the algorithm in core.backtest._consistency_metrics:
    monthly-resample the daily P&L stream, group by calendar year, then
    Sharpe_y = mean(monthly_pnl) / std(monthly_pnl) × √12 for each
    year `y` (using the engine's monthly-Sharpe convention).

    Why monthly basis: matches the per-strategy `annual_sharpe_*`
    metrics in the summary CSVs, so the drilldown's per-year column
    is directly comparable to the headline `annual_sharpe_min/mean`
    on the summary table.

    A year with fewer than 2 valid monthly observations or zero std
    returns NaN (not 0) so the table can render '—' for those years
    rather than misleadingly showing 0.

    `eq` must contain a `pnl_usd` column (and any DateTime index).
    Sharpe is unit-invariant — USD vs % gives the same number — so
    this helper works for both single-leg (`pnl_pct`-based engine)
    and worst-of (`pnl_usd`-based engine) equity curves.
    """
    if eq is None or eq.empty or "pnl_usd" not in eq.columns:
        return pd.Series(dtype=float)
    monthly = eq["pnl_usd"].resample("ME").sum()
    if monthly.empty:
        return pd.Series(dtype=float)
    out: dict[int, float] = {}
    for yr, sub in monthly.groupby(monthly.index.year):
        if len(sub) > 1 and sub.std() > 0:
            out[int(yr)] = float(sub.mean() / sub.std() * np.sqrt(12))
        else:
            out[int(yr)] = float("nan")
    return pd.Series(out).sort_index()


def _render_pnl_by_year_chart(yearly: "pd.Series",
                                 title: str = "P&L by year") -> None:
    """Bar chart + table for yearly PnL (USD). Used by both drilldowns."""
    if yearly.empty:
        st.caption("(no trades — no yearly PnL)")
        return
    fig = go.Figure()
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in yearly.values]
    fig.add_trace(go.Bar(
        x=[str(y) for y in yearly.index], y=yearly.values,
        marker_color=colors,
        text=[_fmt_usd(v) for v in yearly.values],
        textposition="auto",
        hovertemplate="%{x}<br>PnL: $%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=title, height=340, margin=dict(l=10, r=10, t=40, b=10),
        yaxis_tickprefix="$", yaxis_tickformat=",.0f",
        showlegend=False, xaxis_title="Year",
    )
    st.plotly_chart(fig, width='stretch')


def _render_pnl_by_pair_chart(by_pair: "pd.Series",
                                  pair_label: str = "Pair") -> None:
    """Bar chart for PnL by pair (or cross). Used by both drilldowns."""
    if by_pair.empty:
        st.caption(f"(no trades by {pair_label.lower()})")
        return
    by_pair_sorted = by_pair.sort_values(ascending=False)
    fig = go.Figure()
    colors = ["#22c55e" if v >= 0 else "#ef4444"
              for v in by_pair_sorted.values]
    fig.add_trace(go.Bar(
        x=by_pair_sorted.index, y=by_pair_sorted.values,
        marker_color=colors,
        text=[_fmt_usd(v) for v in by_pair_sorted.values],
        textposition="auto",
        hovertemplate="%{x}<br>PnL: $%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"P&L by {pair_label.lower()}",
        height=380, margin=dict(l=10, r=10, t=40, b=10),
        yaxis_tickprefix="$", yaxis_tickformat=",.0f",
        showlegend=False, xaxis_title=pair_label,
    )
    st.plotly_chart(fig, width='stretch')


def _render_pnl_heatmap(year_pair: "pd.DataFrame",
                           pair_label: str = "Pair") -> None:
    """Year × pair PnL heatmap. Rows = year (descending), cols = pair."""
    if year_pair.empty:
        st.caption("(no trades for the heatmap)")
        return
    # Sort columns by total PnL (best on left), rows by year (newest top)
    col_order = year_pair.sum(axis=0).sort_values(ascending=False).index
    yp = year_pair[col_order].sort_index(ascending=False)
    vmax = float(np.nanmax(np.abs(yp.values))) if yp.size else 1.0
    fig = go.Figure(data=go.Heatmap(
        z=yp.values,
        x=yp.columns.tolist(),
        y=[str(y) for y in yp.index],
        colorscale="RdYlGn", zmid=0, zmin=-vmax, zmax=vmax,
        text=[[_fmt_usd(v) if not pd.isna(v) else "" for v in row]
              for row in yp.values],
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate=(f"{pair_label}: %{{x}}<br>Year: %{{y}}<br>"
                          "PnL: $%{z:,.0f}<extra></extra>"),
        colorbar=dict(title="PnL ($)"),
    ))
    fig.update_layout(
        title=f"P&L heatmap — year × {pair_label.lower()}",
        height=max(280, 38 * len(yp.index) + 100),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, width='stretch')


def render_eko_portfolio_drilldown_tab():
    if "eko_port_results" not in st.session_state:
        st.info("Run a portfolio backtest first (EKO Portfolio tab).")
        return

    results = st.session_state["eko_port_results"]
    specs_by_basket = st.session_state.get("eko_port_specs", {})
    mtm_curves = st.session_state.get("eko_port_mtm_curves")
    meta = st.session_state.get("eko_port_meta", {})
    mtm_on = meta.get("mtm_enabled", False) and mtm_curves is not None

    names = [n for n in results if results[n]]
    if not names:
        st.warning("All basket strategies produced zero trades. Widen "
                    "the date range, loosen the gates, or try other "
                    "strike/KO combinations.")
        return

    selected = st.selectbox(
        "Select basket strategy", names, index=0, key="ep_drill_select",
    )
    trades_list = results[selected]
    df = trades_to_df(trades_list)
    if df.empty:
        st.warning("Empty pooled trade ledger for this basket.")
        return

    summary = summarize_strategy(df)
    eq = compute_equity_and_drawdown(df)
    mtm_eq = mtm_curves.get(selected) if mtm_on else None
    mtm_summary = (summarize_mtm(mtm_eq)
                   if (mtm_on and mtm_eq is not None) else {})

    # Diagnostic: if the user enabled MTM but no curve is available for
    # this basket, the headline metrics will SILENTLY fall back to the
    # realized-by-expiry numbers (Sharpe (m) / monthly × √12 / realized
    # MDD). Without this warning the user sees no change after toggling
    # MTM on and has no idea why. The most common causes are:
    #   • `_sum_mtm_curves` returned None because every per-pair MTM
    #     curve in the basket was empty (failed pricing / missing
    #     panels for the underlying pair).
    #   • The basket name → curve key lookup missed (rename / cache
    #     mismatch between `st.session_state` and the current selection).
    if mtm_on and (mtm_eq is None or mtm_eq.empty):
        avail_keys = sorted(mtm_curves.keys()) if mtm_curves else []
        st.warning(
            "MTM toggle is on but no MTM equity curve is available for "
            f"**{selected}** — falling back to realized-at-expiry "
            "metrics. This is usually caused by an empty per-pair MTM "
            "curve (failed pricing or missing panels) or a basket-name "
            "key miss. "
            f"Curves keyed in session: {len(avail_keys)} "
            f"(`{', '.join(avail_keys[:3])}"
            f"{'…' if len(avail_keys) > 3 else ''}`). Re-run the "
            "portfolio backtest, or inspect `_sum_mtm_curves` / "
            "`compute_mtm_curves` for the failing pair."
        )

    # --- Header ---
    n_pairs = df["pair"].nunique() if "pair" in df.columns else 0
    pair_list_str = ", ".join(sorted(df["pair"].unique())
                              if "pair" in df.columns else [])
    st.markdown(f"### {selected}")
    st.caption(
        f"**Pairs in basket** ({n_pairs}): {pair_list_str}  ·  "
        f"notional ${meta.get('notional_usd', 0):,.0f} per pair  ·  "
        f"trade mode `{meta.get('trade_mode', 'stack')}`  ·  "
        f"PnL accounting: "
        f"**{'MTM (daily)' if mtm_on else 'Realized at expiry'}**"
    )

    # --- Headline metrics ---
    cs = st.columns(6)
    if mtm_on and mtm_summary:
        sharpe_lbl = "Sharpe (d)"
        sharpe_val = f"{mtm_summary.get('sharpe_daily_mtm', 0):+.2f}"
        sharpe_sub = "daily MTM × √252"
        maxdd_val = _fmt_usd(mtm_summary.get(
            "max_drawdown_usd_mtm", summary["max_drawdown_usd"]
        ))
        maxdd_sub = "MTM-based"
    else:
        sharpe_lbl = "Sharpe (m)"
        sharpe_val = f"{summary['sharpe_monthly']:+.2f}"
        sharpe_sub = "monthly × √12"
        maxdd_val = _fmt_usd(summary["max_drawdown_usd"])
        maxdd_sub = "realized, by expiry"
    metrics = [
        ("Trades (pooled)", f"{summary['n_trades']}",
         f"{n_pairs} pairs"),
        ("Total PnL", _fmt_usd(summary["total_pnl_usd"]),
         f"{summary['total_pnl_pct']:+.2f}% notl"),
        (sharpe_lbl, sharpe_val, sharpe_sub),
        ("Max DD", maxdd_val, maxdd_sub),
        ("Win rate", f"{summary['win_rate_pct']:.0f}%",
         f"{int(summary['n_trades'] * summary['win_rate_pct'] / 100)} winners"),
        ("KO rate", f"{summary['ko_rate_pct']:.0f}%",
         "barrier hit at expiry"),
    ]
    for col, (lbl, val, sub) in zip(cs, metrics):
        col.metric(lbl, val, sub)

    st.divider()

    # --- Equity curve ---
    st.markdown("#### Equity & drawdown (basket)")
    from plotly.subplots import make_subplots
    eq_to_chart = mtm_eq if (mtm_on and mtm_eq is not None) else eq
    chart_title = ("MTM equity & drawdown (daily, summed across pairs)"
                   if mtm_on
                   else "Realized equity & drawdown (by expiry, pooled)")
    fig_eq = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.07,
        subplot_titles=("Cumulative P&L (USD)", "Drawdown (USD)"),
    )
    fig_eq.add_trace(go.Scatter(
        x=eq_to_chart.index, y=eq_to_chart["equity_usd"],
        mode="lines", line=dict(color="#22c55e", width=2),
        showlegend=False,
        hovertemplate=("%{x|%Y-%m-%d}<br>"
                        "Equity: $%{y:,.0f}<extra></extra>"),
    ), row=1, col=1)
    if "drawdown_usd" in eq_to_chart.columns:
        fig_eq.add_trace(go.Scatter(
            x=eq_to_chart.index, y=eq_to_chart["drawdown_usd"],
            mode="lines", line=dict(color="#ef4444", width=1.5),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.25)",
            showlegend=False,
            hovertemplate=("%{x|%Y-%m-%d}<br>"
                            "DD: $%{y:,.0f}<extra></extra>"),
        ), row=2, col=1)
    fig_eq.update_layout(
        height=520, margin=dict(l=10, r=10, t=50, b=10),
        title_text=chart_title,
    )
    fig_eq.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig_eq, width='stretch')

    st.divider()

    # --- PnL breakdowns ---
    # Realized PnL is grouped by expiry-date year. Each trade's expiry
    # is the date at which its PnL crystalises into the ledger.
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    df["_year"] = df["expiry_date"].dt.year

    st.markdown("#### P&L by year")
    yearly = df.groupby("_year")["pnl_usd"].sum().sort_index()
    _render_pnl_by_year_chart(yearly, "P&L by expiry year")
    # Add per-year Sharpe alongside the realized PnL.
    # Use MTM curve if MTM mode was on (monthly resampling captures
    # intra-trade variability — closer to a "real desk" Sharpe view).
    # Otherwise use the realized-at-expiry equity curve.
    eq_for_sharpe = mtm_eq if (mtm_on and mtm_eq is not None) else eq
    sharpe_by_year = _annual_sharpe_per_year(eq_for_sharpe)
    yearly_df = yearly.reset_index().rename(
        columns={"_year": "Year", "pnl_usd": "PnL (USD)"}
    )
    yearly_df["Sharpe (m)"] = yearly_df["Year"].map(
        lambda y: sharpe_by_year.get(int(y), float("nan"))
    )
    yearly_df = yearly_df.assign(**{
        "PnL (USD)": yearly_df["PnL (USD)"].apply(_fmt_usd),
        "Sharpe (m)": yearly_df["Sharpe (m)"].apply(
            lambda v: f"{v:+.2f}" if pd.notna(v) else "—"
        ),
    })
    st.dataframe(yearly_df, hide_index=True, width='stretch')
    st.caption(
        "**Sharpe (m)** is the monthly-basis annualized Sharpe ratio "
        "computed *within* each calendar year — same formula as the "
        "headline `annual_sharpe_*` columns in the summary CSV: "
        "`mean(monthly_pnl) / std(monthly_pnl) × √12`, monthly stream "
        + ("from MTM equity curve. " if (mtm_on and mtm_eq is not None)
            else "from realized-at-expiry equity curve. ")
        + "Years with fewer than 2 valid monthly observations show '—'."
    )

    st.divider()

    st.markdown("#### P&L by currency pair")
    by_pair = df.groupby("pair")["pnl_usd"].sum()
    _render_pnl_by_pair_chart(by_pair, pair_label="Pair")
    # Full per-pair breakdown table
    pair_tbl = df.groupby("pair").agg(
        n_trades=("pnl_usd", "size"),
        total_pnl_usd=("pnl_usd", "sum"),
        total_premium_usd=("premium_usd", "sum"),
        total_payoff_usd=("actual_payoff_usd", "sum"),
        ko_rate=("knocked_out",
                 lambda s: 100.0 * s.astype(bool).sum() / len(s)),
        win_rate=("pnl_usd",
                  lambda s: 100.0 * (s > 0).sum() / len(s)),
    ).reset_index().sort_values("total_pnl_usd", ascending=False)
    pair_tbl_disp = pair_tbl.copy()
    pair_tbl_disp["total_pnl_usd"] = (
        pair_tbl_disp["total_pnl_usd"].apply(_fmt_usd)
    )
    pair_tbl_disp["total_premium_usd"] = (
        pair_tbl_disp["total_premium_usd"].apply(_fmt_usd)
    )
    pair_tbl_disp["total_payoff_usd"] = (
        pair_tbl_disp["total_payoff_usd"].apply(_fmt_usd)
    )
    pair_tbl_disp["ko_rate"] = (
        pair_tbl_disp["ko_rate"].apply(lambda x: f"{x:.0f}%")
    )
    pair_tbl_disp["win_rate"] = (
        pair_tbl_disp["win_rate"].apply(lambda x: f"{x:.0f}%")
    )
    pair_tbl_disp.columns = ["Pair", "n trades", "PnL",
                              "Σ Premium", "Σ Payoff",
                              "KO %", "Win %"]
    st.dataframe(pair_tbl_disp, hide_index=True, width='stretch')

    st.divider()

    # ---- Monthly P&L heatmap (year × month) — matches App 12's RKO drilldown ----
    st.markdown("#### Monthly P&L heatmap — year × month (USD)")
    monthly_df = monthly_pnl_table(df, value_col="pnl_usd")
    if not monthly_df.empty:
        month_labels = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                          7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        monthly_df = monthly_df.rename(
            columns=lambda c: month_labels.get(c, str(c)))
        # Matplotlib-free diverging palette (inlined; mirrors the helper
        # in apps/_app12_tabs.py so the two apps look identical)
        arr = monthly_df.to_numpy(dtype=float, na_value=np.nan)
        finite = arr[np.isfinite(arr)]
        vmax = float(np.max(np.abs(finite))) if finite.size else 0.0
        def _ryg(v):
            if pd.isna(v) or not np.isfinite(v) or vmax == 0.0:
                return ""
            x = max(-1.0, min(1.0, v / vmax))
            if x >= 0:
                r, g, b = int(255 + (60 - 255) * x), int(255 + (170 - 255) * x), int(180 + (80 - 180) * x)
            else:
                xm = -x
                r, g, b = int(255 + (200 - 255) * xm), int(255 + (60 - 255) * xm), int(180 + (60 - 180) * xm)
            return f"background-color: rgb({r},{g},{b}); color: #1a1a1a;"
        st.dataframe(
            monthly_df.style.format("${:,.0f}", na_rep="").map(_ryg),
            use_container_width=True,
        )

    st.divider()

    st.markdown("#### P&L heatmap — year × pair")
    year_pair = df.groupby(["_year", "pair"])["pnl_usd"].sum().unstack(
        fill_value=0.0
    )
    _render_pnl_heatmap(year_pair, pair_label="Pair")

    st.divider()

    # ---- Bulk per-pair breakdown CSV download ----
    st.markdown("#### Per-pair breakdown — CSV download")
    st.caption(
        "One row per (strategy_name, pair) across all basket strategies "
        "in this run. The downstream analyzer ingests this alongside the "
        "main summary CSV to reproduce the per-pair table for any basket."
    )
    per_pair_rows = []
    for name, t in results.items():
        if not t:
            continue
        df_one = trades_to_df(t)
        if df_one.empty or "pair" not in df_one.columns:
            continue
        b = df_one.groupby("pair").agg(
            n_trades=("pnl_usd", "size"),
            total_pnl_usd=("pnl_usd", "sum"),
            total_premium_usd=("premium_usd", "sum"),
            total_payoff_usd=("actual_payoff_usd", "sum"),
            ko_rate_pct=("knocked_out",
                            lambda s: 100.0 * s.astype(bool).sum() / len(s)),
            win_rate_pct=("pnl_usd",
                            lambda s: 100.0 * (s > 0).sum() / len(s)),
        ).reset_index().rename(columns={"pair": "Pair"})
        if b.empty:
            continue
        b = b.sort_values("total_pnl_usd", ascending=False)
        b.insert(0, "strategy_name", name)
        b.insert(1, "strategy_type", "eko_basket")
        per_pair_rows.append(b)
    if per_pair_rows:
        per_pair_df = pd.concat(per_pair_rows, ignore_index=True)
        st.download_button(
            label=(f"⬇ Download per-pair breakdown "
                     f"({len(per_pair_df)} rows, CSV)"),
            data=per_pair_df.to_csv(index=False).encode("utf-8"),
            file_name="eko_portfolio_per_pair.csv",
            mime="text/csv",
            use_container_width=True, key="ep_per_pair_dl",
        )
    else:
        st.caption("_No per-pair rows yet._")

    st.divider()

    # --- Full ledger expander ---
    with st.expander(
        f"📜 Full pooled trade ledger ({summary['n_trades']} trades)",
        expanded=False,
    ):
        led_cols_default = [
            "pair", "trade_date", "expiry_date", "tenor_label",
            "spot", "strike", "barrier",
            "premium_usd", "actual_payoff_usd", "pnl_usd",
            "knocked_out",
        ]
        led_cols = [c for c in led_cols_default if c in df.columns]
        st.dataframe(
            df[led_cols].sort_values(["pair", "trade_date"]),
            hide_index=True, width='stretch',
        )


# =============================================================================
# TAB 8: WO EKO PORTFOLIO  +  TAB 9: WO EKO PORTFOLIO DRILLDOWN
# =============================================================================
# Same basket model as EKO Portfolio but with worst-of structures.
# One STRATEGY = one (tenor × dir × strike-A × KO-A × gate-A ×
# strike-B × KO-B × gate-B) combination, applied uniformly to EVERY
# 2-leg cross derived from the basket. With 7 pairs you get 21 crosses
# (C(7,2)) per strategy.
#
# session_state keys:
#   wo_port_results  : {basket_name: list[WorstOfTrade]}
#   wo_port_specs    : {basket_name: list[WorstOfSpec]}
#   wo_port_meta     : run-level inputs


def _parse_eko_strategy_name(name: str) -> "dict | None":
    """Parse a single-leg EKO strategy_name back into its fields.

    Format examples:
        'USDTHB CALL-upout  ATM  1M  H@5Δ'
        'USDTHB CALL-upout  ATM  1M  H@5Δ  [Spot > 50DMA]'

    Returns a dict {pair, direction, strike_label, tenor_label,
    ko_label, gate_label_disp} on success, None on failure.
    The `gate_label_disp` is the human-readable label shown in
    GATE_REGISTRY (e.g. 'Spot > 50DMA'); None if no gate.
    """
    import re
    pattern = re.compile(
        r"^(?P<pair>\S+)\s+"
        r"(?P<direction>CALL-upout|PUT-downout)\s+"
        r"(?P<strike>\S+)\s+"
        r"(?P<tenor>\S+)\s+"
        r"H@(?P<ko>\S+)"
        r"(?:\s+\[(?P<gate>[^\]]+)\])?"
        r"$"
    )
    m = pattern.match(name.strip())
    if not m:
        return None
    return {
        "pair": m["pair"],
        "direction": m["direction"],
        "strike_label": m["strike"],
        "tenor_label": m["tenor"],
        "ko_label": m["ko"],
        "gate_label_disp": m["gate"],
    }


def _gate_key_from_display(disp_label: "str | None") -> "str | None":
    """Reverse-lookup a gate's REGISTRY key from its display label.

    GATE_REGISTRY[key] = (display_label, callable). The summary CSV
    contains display labels (e.g. 'Spot > 50DMA'); the engine needs
    the registry KEY (e.g. 'spot_above_50dma'). None if no gate or
    the label isn't in the registry."""
    if not disp_label:
        return None
    from core.gates import GATE_REGISTRY
    for k, (lbl, _fn) in GATE_REGISTRY.items():
        if lbl == disp_label:
            return k
    return None


# Tenor ordering for the "lower tenor" rule. Matches TENOR_LIST order
# already defined above (1M < 6W < 2M < 10W < 3M by calendar days).
_TENOR_RANK = {t: i for i, t in enumerate(TENOR_LIST)}


def _lower_tenor(a: str, b: str) -> str:
    """Return whichever of (a, b) is the lower (shorter-duration)
    tenor per _TENOR_RANK. Falls back to `a` if either tenor is
    unknown to the rank table."""
    if a not in _TENOR_RANK or b not in _TENOR_RANK:
        return a
    return a if _TENOR_RANK[a] <= _TENOR_RANK[b] else b


# Set of metrics from the EKO summary CSV that make sense to optimize on.
# Direction tells the picker which way is "better" (max vs min). The
# label is what shows up in the UI.
_OPTIMIZE_METRICS = [
    # (column,                 display label,                  better)
    ("sharpe_monthly",         "Sharpe (monthly)",             "max"),
    ("annual_sharpe_mean",     "Annual Sharpe (mean)",         "max"),
    ("annual_sharpe_min",      "Annual Sharpe (min — worst yr)",  "max"),
    ("annual_sharpe_score",    "Annual Sharpe score (μ − σ)",  "max"),
    ("calmar",                 "Calmar",                       "max"),
    ("gain_to_pain",           "Gain-to-pain",                 "max"),
    ("total_pnl_usd",          "Total PnL (USD)",              "max"),
    ("win_rate_pct",           "Win rate (%)",                 "max"),
    ("premium_recovery_pct",   "Premium recovery (%)",         "max"),
    ("pct_positive_years",     "% positive years",             "max"),
    ("max_drawdown_usd",       "Max drawdown (USD)  — less neg better",
                                                                "max"),
    ("ulcer_index",            "Ulcer index — lower better",   "min"),
    ("ko_rate_pct",            "KO rate (%) — lower better",   "min"),
]


def _render_wo_optimized_basket_mode(pairs_avail: list[str],
                                       engine_cfg: dict) -> None:
    """Optimized-basket mode for WO Portfolio. Distinct from the
    sweep mode: produces ONE basket strategy whose constituent
    per-cross specs are heterogeneous (each cross gets its own
    optimal params, derived from an uploaded EKO summary CSV).

    `engine_cfg` carries the structure-pricing-engine choices from
    `_wo_portfolio_engine_controls()` (rendered by the calling page
    so both modes share them). The four keys —  pricing_engine,
    correlation_source, correlation_value, mc_n_paths — are applied
    to every per-cross WorstOfSpec built here.

    Saves to the same session_state keys as the sweep mode so the
    drilldown tab works for either source."""
    import re
    from core.worstof import (
        WorstOfSpec, build_worstof_grid, run_worstof_grid,
    )
    from core.gates import GATE_REGISTRY
    from itertools import combinations

    st.caption(
        "Upload a single-leg EKO summary CSV (the file produced by the "
        "**Backtest** tab's `Download summary table` button). For each "
        "currency, the tab finds the row that's best on a chosen "
        "metric, then auto-builds a worst-of basket where every 2-leg "
        "cross uses each leg's pair-optimal (strike, KO, gate) combo. "
        "Tenor defaults per cross to the **lower** (shorter-duration) "
        "of the two legs' optimal tenors. **Per-cross overrides** for "
        "everything are available in the editor below."
    )

    # ---- 1) Upload + parse ----
    upload = st.file_uploader(
        "Upload EKO summary CSV",
        type=["csv"],
        key="wp_opt_upload",
        help=("Schema: must contain `strategy_name` plus numeric "
               "metric columns. The strategy_name is parsed as "
               "`{PAIR} CALL-upout  {STRIKE}  {TENOR}  H@{KO}  "
               "[{GATE}]` (gate optional)."),
    )
    if upload is None:
        st.info("Upload a CSV to continue. The expected schema matches "
                 "the Backtest tab's summary download.")
        return

    try:
        eko_df = pd.read_csv(upload)
    except Exception as e:
        st.error(f"Couldn't read CSV: {e}")
        return

    if "strategy_name" not in eko_df.columns:
        st.error("CSV missing required `strategy_name` column.")
        return

    # Parse names — drop unparseable rows but warn
    parsed_rows = []
    for _, row in eko_df.iterrows():
        parsed = _parse_eko_strategy_name(str(row["strategy_name"]))
        if not parsed:
            continue
        # Carry through every numeric metric we recognize
        new_row = dict(row)
        new_row.update(parsed)
        parsed_rows.append(new_row)

    if not parsed_rows:
        st.error("No strategy names matched the expected format. "
                  "Check that the CSV is a Backtest tab summary CSV.")
        return

    eko_parsed = pd.DataFrame(parsed_rows)
    n_dropped = len(eko_df) - len(eko_parsed)
    pairs_in_csv = sorted(eko_parsed["pair"].unique())
    st.caption(
        f"Loaded **{len(eko_parsed)}** strategies across "
        f"**{len(pairs_in_csv)}** pairs: {', '.join(pairs_in_csv)}"
        + (f"  ·  ({n_dropped} unparseable rows skipped)"
           if n_dropped else "")
    )

    # ---- 2) Pick metric + pairs ----
    cc_opt1, cc_opt2 = st.columns([1, 1])
    with cc_opt1:
        # Filter metrics to those actually present in the CSV
        available_metrics = [
            (col, lbl, direction)
            for (col, lbl, direction) in _OPTIMIZE_METRICS
            if col in eko_parsed.columns
        ]
        if not available_metrics:
            st.error("CSV has none of the recognized optimization "
                      "metric columns.")
            return
        metric_labels = [lbl for (_c, lbl, _d) in available_metrics]
        metric_to_pick_idx = st.selectbox(
            "Optimization metric",
            range(len(metric_labels)),
            index=2,  # Annual Sharpe (min) — a robust default
            format_func=lambda i: metric_labels[i],
            key="wp_opt_metric",
            help=("Each pair's 'optimal' (strike, tenor, KO, gate) "
                   "config is the row that maximizes (or minimizes, "
                   "for cost-style metrics) this metric."),
        )
        metric_col, metric_label, metric_dir = (
            available_metrics[metric_to_pick_idx]
        )

    with cc_opt2:
        # Pair selection — intersection of available data folder pairs
        # and pairs in the CSV. Default = the intersection.
        eligible = sorted(set(pairs_in_csv) & set(pairs_avail))
        if len(eligible) < 2:
            st.error(f"Need at least 2 pairs present in BOTH the CSV "
                      f"and the data folder. CSV has "
                      f"{len(pairs_in_csv)}, folder has "
                      f"{len(pairs_avail)}, intersection has "
                      f"{len(eligible)}.")
            return
        pairs_sel = st.multiselect(
            "Pairs to include (basket)",
            eligible,
            default=eligible,
            key="wp_opt_pairs",
            help=("Each cross C(N,2) becomes one position in the "
                   "basket. Default = every pair present in both the "
                   "CSV and the data folder."),
        )

    if len(pairs_sel) < 2:
        st.warning("Pick at least 2 pairs.")
        return

    # ---- 3) Find optimal per-pair config ----
    # For each pair, restrict to rows for that pair, drop rows where
    # the metric is NaN, and idxmax/idxmin per `metric_dir`. We keep
    # the picked row's *parsed* params (strike/tenor/KO/gate) so the
    # picker is faithful to the source data.
    pair_optima = {}
    for pair in pairs_sel:
        sub = eko_parsed[eko_parsed["pair"] == pair].copy()
        sub = sub[pd.notna(sub[metric_col])]
        if sub.empty:
            st.warning(f"No rows with a valid `{metric_col}` for {pair}. "
                        f"Skipping.")
            continue
        if metric_dir == "max":
            best_idx = sub[metric_col].idxmax()
        else:
            best_idx = sub[metric_col].idxmin()
        best_row = sub.loc[best_idx]
        # pandas returns NaN (not None) for missing string values, so
        # coerce here to keep downstream gate-lookup simple.
        gate_disp_raw = best_row.get("gate_label_disp")
        gate_disp = (None if (gate_disp_raw is None
                                or (isinstance(gate_disp_raw, float)
                                    and pd.isna(gate_disp_raw)))
                     else str(gate_disp_raw))
        pair_optima[pair] = {
            "strike_label": best_row["strike_label"],
            "tenor_label": best_row["tenor_label"],
            "ko_label": best_row["ko_label"],
            "gate_label_disp": gate_disp,
            "metric_value": float(best_row[metric_col]),
            "strategy_name": best_row["strategy_name"],
        }

    if len(pair_optima) < 2:
        st.error("Not enough pairs survived the metric filter. "
                  "Either the metric has NaN for too many pairs, or "
                  "the CSV is missing those pairs entirely.")
        return

    # Show the per-pair optimum table — auditable view of what the
    # picker actually decided per pair.
    st.markdown("#### Per-pair optimum (by selected metric)")
    opt_rows = []
    for pair, o in pair_optima.items():
        opt_rows.append({
            "Pair": pair,
            "Strike Δ": o["strike_label"],
            "Tenor": o["tenor_label"],
            "KO Δ": o["ko_label"],
            "Gate": o["gate_label_disp"] or "(no gate)",
            metric_label: f"{o['metric_value']:.4f}",
            "Source strategy": o["strategy_name"],
        })
    st.dataframe(pd.DataFrame(opt_rows), hide_index=True,
                 width='stretch')

    # ---- 4) Build the cross editor (per-cross overrides) ----
    # Generate canonical pair_combos (sorted within each combo) from
    # the pairs with valid optima. Default tenor per cross = lower of
    # the two pair-tenors (per user spec).
    valid_pairs = [p for p in pairs_sel if p in pair_optima]
    pair_combos = [tuple(c) for c in combinations(sorted(valid_pairs), 2)]
    n_combos = len(pair_combos)
    st.caption(f"Generated **{n_combos}** crosses from "
                f"{len(valid_pairs)} pairs.")

    # Allowed value lists for the editor's dropdowns
    strike_choices = list(DELTA_CHOICES.keys())
    ko_choices = list(KO_DELTA_CHOICES.keys())
    gate_choices = ["(no gate)"] + [
        GATE_REGISTRY[k][0] for k in GATE_REGISTRY
    ]

    st.markdown("#### Per-cross specs (override as needed)")
    st.caption(
        "Each row is one cross. Tenor defaults to the **lower** of the "
        "two legs' optimal tenors; strike/KO/gate per leg default to "
        "that leg's pair-optimal. Edit any cell directly to override."
    )

    editor_rows = []
    unknown_gates_in_optima: set[str] = set()
    for (pa, pb) in pair_combos:
        oa = pair_optima[pa]
        ob = pair_optima[pb]
        default_tenor = _lower_tenor(oa["tenor_label"],
                                      ob["tenor_label"])
        # Coerce per-leg gate to either a valid editor option or "(no gate)".
        # If the CSV references a gate that no longer exists in this
        # app's GATE_REGISTRY, we drop the default to "(no gate)" rather
        # than silently letting data_editor pick something arbitrary.
        ga_raw = oa["gate_label_disp"]
        gb_raw = ob["gate_label_disp"]
        if ga_raw and ga_raw not in gate_choices:
            unknown_gates_in_optima.add(ga_raw)
            ga_disp = "(no gate)"
        else:
            ga_disp = ga_raw or "(no gate)"
        if gb_raw and gb_raw not in gate_choices:
            unknown_gates_in_optima.add(gb_raw)
            gb_disp = "(no gate)"
        else:
            gb_disp = gb_raw or "(no gate)"
        editor_rows.append({
            "Cross": f"{pa}×{pb}",
            "Tenor": default_tenor,
            "Strike A": oa["strike_label"],
            "KO A": oa["ko_label"],
            "Gate A": ga_disp,
            "Strike B": ob["strike_label"],
            "KO B": ob["ko_label"],
            "Gate B": gb_disp,
        })

    if unknown_gates_in_optima:
        st.warning(
            "The following gate(s) from the uploaded CSV aren't in "
            "this app's gate registry — they've been replaced with "
            "**(no gate)** in the editor. Pick a registered gate if "
            f"you want a gate applied: {sorted(unknown_gates_in_optima)}"
        )

    edited = st.data_editor(
        pd.DataFrame(editor_rows),
        hide_index=True,
        use_container_width=True,
        key="wp_opt_editor",
        column_config={
            "Cross": st.column_config.TextColumn(
                "Cross", disabled=True,
                help="Read-only — cross identifier",
            ),
            "Tenor": st.column_config.SelectboxColumn(
                "Tenor", options=TENOR_LIST, required=True,
                help="Defaults to lower of the two legs' optimal tenors.",
            ),
            "Strike A": st.column_config.SelectboxColumn(
                "Strike A", options=strike_choices, required=True,
            ),
            "KO A": st.column_config.SelectboxColumn(
                "KO A", options=ko_choices, required=True,
            ),
            "Gate A": st.column_config.SelectboxColumn(
                "Gate A", options=gate_choices, required=True,
            ),
            "Strike B": st.column_config.SelectboxColumn(
                "Strike B", options=strike_choices, required=True,
            ),
            "KO B": st.column_config.SelectboxColumn(
                "KO B", options=ko_choices, required=True,
            ),
            "Gate B": st.column_config.SelectboxColumn(
                "Gate B", options=gate_choices, required=True,
            ),
        },
    )

    # ---- 5) Run controls ----
    st.markdown("---")
    crd1, crd2, crd3 = st.columns(3)
    with crd1:
        date_max_avail = _date.today()
        date_min_avail = _date(2018, 1, 1)
        try:
            spot_all = load_panel(folder, "SPOT", None)
            if not spot_all.empty:
                date_max_avail = spot_all.index.max().date()
                date_min_avail = spot_all.index.min().date()
        except Exception:
            pass
        default_start = max(_date(2023, 1, 1), date_min_avail)
        if default_start >= date_max_avail:
            default_start = date_min_avail
        start_date = st.date_input(
            "Start date", value=default_start,
            min_value=date_min_avail, max_value=date_max_avail,
            key="wp_opt_start",
        )
        end_date = st.date_input(
            "End date", value=date_max_avail,
            min_value=date_min_avail, max_value=date_max_avail,
            key="wp_opt_end",
        )
    with crd2:
        tx_cost_bps = st.number_input(
            "Transaction cost (bps of notional)",
            min_value=0.0, max_value=20.0, value=2.0, step=0.5,
            key="wp_opt_tx",
        )
        prefer_em = st.radio(
            "EM pairs variant", ["offshore", "onshore"], index=0,
            horizontal=True, key="wp_opt_prefer_em",
        )
    with crd3:
        notional_usd = st.number_input(
            "Notional (USD, per cross)",
            min_value=100_000.0, max_value=200_000_000.0,
            value=10_000_000.0, step=1_000_000.0, format="%.0f",
            key="wp_opt_notional",
        )
        trade_mode = st.radio(
            "Trade mode",
            ["stack", "single"],
            index=0, horizontal=True, key="wp_opt_trade_mode",
        )

    run_clicked = st.button(
        "▶ Run optimized WO basket", type="primary",
        disabled=(n_combos == 0), key="wp_opt_run_btn",
    )

    if not run_clicked:
        return

    # ---- 6) Build per-cross specs from the edited table ----
    # The basket is one strategy whose constituent crosses have
    # potentially different tenors / strikes / KOs / gates. We build
    # each cross's WorstOfSpec individually (rather than via
    # build_worstof_grid, which assumes uniform per-leg axes).
    direction_label = "Call (up-and-out)"
    dir_a_pair = DIRECTIONS[direction_label]
    dir_b_pair = DIRECTIONS[direction_label]
    specs_per_cross: list[WorstOfSpec] = []
    edited_rows_records = edited.to_dict(orient="records")
    for row_dict in edited_rows_records:
        cross_label = row_dict["Cross"]
        pa, pb = cross_label.split("×")
        tenor = row_dict["Tenor"]
        sd_a = row_dict["Strike A"]
        kd_a = row_dict["KO A"]
        ga_disp = row_dict["Gate A"]
        sd_b = row_dict["Strike B"]
        kd_b = row_dict["KO B"]
        gb_disp = row_dict["Gate B"]
        ga_key = (None if ga_disp == "(no gate)"
                  else _gate_key_from_display(ga_disp))
        gb_key = (None if gb_disp == "(no gate)"
                  else _gate_key_from_display(gb_disp))

        spec = WorstOfSpec(
            leg_a_pair=pa,
            leg_a_direction=dir_a_pair[0],
            leg_a_barrier_type=dir_a_pair[1],
            leg_a_strike_delta_label=sd_a,
            leg_a_strike_delta_value=DELTA_CHOICES[sd_a],
            leg_a_ko_delta_label=kd_a,
            leg_a_ko_delta_value=KO_DELTA_CHOICES[kd_a],
            leg_b_pair=pb,
            leg_b_direction=dir_b_pair[0],
            leg_b_barrier_type=dir_b_pair[1],
            leg_b_strike_delta_label=sd_b,
            leg_b_strike_delta_value=DELTA_CHOICES[sd_b],
            leg_b_ko_delta_label=kd_b,
            leg_b_ko_delta_value=KO_DELTA_CHOICES[kd_b],
            tenor_label=tenor,
            tx_cost_bps=tx_cost_bps,
            entry_gate_a=ga_key,
            entry_gate_b=gb_key,
            prefer=prefer_em,
            trade_mode=trade_mode,
            # Step 2c+2d — structure-level engine, propagated to every
            # per-cross spec so the whole basket is repriced under the
            # same correlation-aware engine.
            pricing_engine=engine_cfg["pricing_engine"],
            correlation_source=engine_cfg["correlation_source"],
            correlation_value=engine_cfg["correlation_value"],
            mc_n_paths=engine_cfg["mc_n_paths"],
        )
        specs_per_cross.append(spec)

    if not specs_per_cross:
        st.error("No valid per-cross specs built. Check the editor.")
        return

    # ---- 7) Run + pool into ONE basket strategy ----
    progress_bar = st.progress(0.0, text="Starting optimized basket run…")
    t0 = time.time()
    last_update = [t0]

    def cb(p, name):
        now = time.time()
        if now - last_update[0] > 0.1 or p >= 1.0:
            progress_bar.progress(min(p, 1.0),
                                  text=f"Running: {name} ({p*100:.0f}%)")
            last_update[0] = now

    sub_results = run_worstof_grid(
        folder, specs_per_cross, start_date, end_date,
        notional_usd=notional_usd, progress_cb=cb,
    )
    elapsed = time.time() - t0
    progress_bar.empty()

    # Pool every cross's trade list into ONE basket
    basket_name = (f"OPTIMIZED-WO  {metric_label}  "
                    f"({len(specs_per_cross)} crosses, "
                    f"{trade_mode})")
    pooled = []
    for s in specs_per_cross:
        pooled.extend(sub_results.get(s.name, []))

    # Persist on the same session_state keys as sweep mode so the
    # drilldown works unchanged.
    st.session_state["wo_port_results"] = {basket_name: pooled}
    st.session_state["wo_port_specs"] = {basket_name: specs_per_cross}
    st.session_state["wo_port_meta"] = {
        "mode": "optimized",
        "pairs": sorted(valid_pairs),
        "pair_combos": pair_combos,
        # Tenors actually used (per-cross), not a single global tenor
        "tenors_used": sorted({s.tenor_label for s in specs_per_cross}),
        "tenors": sorted({s.tenor_label for s in specs_per_cross}),
        "directions": [direction_label],
        "trade_mode": trade_mode,
        "start": start_date, "end": end_date,
        "tx_cost_bps": tx_cost_bps,
        "prefer_em": prefer_em,
        "notional_usd": notional_usd,
        "n_baskets": 1,
        "n_crosses": len(pair_combos),
        "optimization_metric_label": metric_label,
        "optimization_metric_col": metric_col,
        "optimization_direction": metric_dir,
        "elapsed": elapsed,
    }
    n_trades = len(pooled)
    st.success(
        f"Done in {elapsed:.1f}s — 1 optimized basket strategy "
        f"({len(specs_per_cross)} crosses, {n_trades} pooled trades). "
        f"Switch to the WO EKO Portfolio drilldown tab to inspect."
    )


def _wo_portfolio_engine_controls() -> dict:
    """Render the structure-pricing-engine controls used by BOTH the
    Worst-of EKO Portfolio's sweep and optimized modes, and return a
    dict of the captured values.

    Returns
    -------
    dict with keys:
      - pricing_engine: 'legacy_multiplier' | 'closed_form' | 'monte_carlo'
      - correlation_source: 'manual' | 'rolling_60d' | 'triangulation'
      - correlation_value: float, used in 'manual' mode and as a
        fallback when the requested source has no data on a given
        trade date.
      - mc_n_paths: int, only used when pricing_engine='monte_carlo'.

    The values default to the legacy multiplier formula so existing
    workflows are unaffected. Selecting CF or MC routes structure
    premium through `core.worstof_pricer` and enables correlation-source
    controls. Same UI pattern as the single-trade Worst-of Pricer
    (TAB 4) and the bulk Worst-of (TAB 5).
    """
    with st.expander("Structure pricing engine", expanded=False):
        st.caption(
            "How the worst-of structure premium is computed. **Legacy "
            "multiplier** (default) preserves historical behaviour. "
            "**Closed form** and **Monte Carlo** use the correlation-"
            "aware pricer in `core.worstof_pricer`. Both require "
            "`ko_check_mode='european_at_expiry'` and "
            "`leg_pricing_mode='european'`."
        )
        engine_label = st.radio(
            "Engine",
            ["Legacy multiplier (default)",
             "Closed form (CF)",
             "Monte Carlo (MC)"],
            index=0,
            key="wp_pricing_engine",
            help=(
                "Legacy: `multiplier × min(P_A, P_B)`. "
                "CF: 1D-quadrature joint pricer (~1–3 ms/trade). "
                "MC: terminal correlated-GBM simulator "
                "(~5–15 ms/trade at default paths). CF and MC are "
                "equivalent in expectation; CF is faster + deterministic."
            ),
        )
        _engine_map = {
            "Legacy multiplier (default)": "legacy_multiplier",
            "Closed form (CF)":            "closed_form",
            "Monte Carlo (MC)":            "monte_carlo",
        }
        pricing_engine = _engine_map[engine_label]

        correlation_source = "manual"
        correlation_value = 0.30
        mc_n_paths = 100_000
        if pricing_engine != "legacy_multiplier":
            corr_src_label = st.radio(
                "Correlation source",
                ["Manual (single ρ)",
                 "Historical 60d rolling",
                 "Triangulation (cross vol)"],
                index=1,
                key="wp_correlation_source",
                help=(
                    "**Manual**: same ρ used for every trade date.  \n"
                    "**Historical 60d**: rolling 60-business-day "
                    "realized log-return correlation, computed once "
                    "per backtest per pair-combo.  \n"
                    "**Triangulation**: forward-looking implied "
                    "correlation from the cross-pair's ATM vol. "
                    "Requires the cross pair's VOL_ATM panel in the "
                    "data folder.  \n\n"
                    "All non-manual sources fall back to Manual on "
                    "dates where the source's value is missing."
                ),
            )
            _src_map = {
                "Manual (single ρ)":         "manual",
                "Historical 60d rolling":    "rolling_60d",
                "Triangulation (cross vol)": "triangulation",
            }
            correlation_source = _src_map[corr_src_label]
            correlation_value = st.slider(
                ("ρ (Manual value; fallback when 60d/triangulation "
                 "data is unavailable)"),
                min_value=-0.95, max_value=0.95,
                value=0.30, step=0.05,
                key="wp_correlation_value",
            )
            if pricing_engine == "monte_carlo":
                mc_n_paths = st.select_slider(
                    "MC paths per trade",
                    options=[20_000, 50_000, 100_000, 200_000, 500_000],
                    value=100_000, key="wp_mc_n_paths",
                    help=("Std error per trade scales as 1/√n. "
                           "100k → ~1bp; 500k → ~0.5bp."),
                )
    return dict(
        pricing_engine=pricing_engine,
        correlation_source=correlation_source,
        correlation_value=correlation_value,
        mc_n_paths=mc_n_paths,
    )


def render_wo_eko_portfolio_tab():
    from itertools import combinations, product
    from core.worstof import build_worstof_grid, run_worstof_grid
    from core.gates import GATE_REGISTRY

    pairs_avail = _list_pairs(folder)
    if len(pairs_avail) < 2:
        st.error("Need at least 2 pairs in the data folder.")
        return

    st.markdown("### WO EKO Portfolio configuration")

    # --- Mode selector ---------------------------------------------------
    # Two distinct workflows:
    #   * sweep      — the original cross-product mode (every axis is a
    #                  multiselect; the engine runs every combination as
    #                  its own basket strategy).
    #   * optimized  — upload a pre-computed single-leg EKO summary CSV,
    #                  pick the best (strike, tenor, KO, gate) per pair
    #                  by a chosen metric, then build ONE basket strategy
    #                  whose per-cross specs are heterogeneous (each
    #                  cross gets its own params).
    #
    # The two modes share the drilldown — both write to wo_port_results /
    # wo_port_specs / wo_port_meta and the drilldown is parameter-aware
    # via meta.get(...).
    run_mode = st.radio(
        "Run mode",
        ["Cross-product sweep", "Optimized basket (from summary CSV)"],
        index=0, horizontal=True, key="wp_run_mode",
        help=("**Sweep**: original mode — every axis (tenor, strike, "
               "KO, gate) is a multiselect, full cross-product. "
               "**Optimized**: upload an EKO summary CSV, auto-pick the "
               "best per-pair config by a chosen metric, then build "
               "ONE basket whose cross-specs are heterogeneous "
               "(per-cross tenor / strike / KO / gate)."),
    )

    if run_mode == "Optimized basket (from summary CSV)":
        engine_cfg = _wo_portfolio_engine_controls()
        _render_wo_optimized_basket_mode(pairs_avail, engine_cfg)
        return

    # ---- Original sweep mode (engine controls + cross-product axes) ----
    engine_cfg = _wo_portfolio_engine_controls()

    # ---- Original sweep mode (unchanged below) -------------------------
    st.caption(
        "A **basket** worst-of portfolio. One **strategy** = one "
        "`(tenor × dir × strike-A × KO-A × gate-A × strike-B × KO-B "
        "× gate-B)` combination, applied uniformly to **every 2-leg "
        "cross** derived from the basket (C(N,2) crosses). With 7 "
        "pairs that's 21 cross-positions per strategy per trading day. "
        "Notional applies PER CROSS."
    )

    cc1, cc2, cc3 = st.columns([1, 1, 1])
    with cc1:
        default_pairs = [p for p in EKO_PORT_DEFAULT_PAIRS
                         if p in pairs_avail]
        if len(default_pairs) < 2:
            default_pairs = pairs_avail[:min(3, len(pairs_avail))]
        pairs_sel = st.multiselect(
            "Currency pairs (basket)",
            pairs_avail,
            default=default_pairs,
            key="wp_pairs",
            help=("The basket. All C(N, 2) two-pair crosses are "
                   "auto-generated and traded as part of every basket "
                   "strategy."),
        )
        tenors_sel = st.multiselect(
            "Tenor(s)", TENOR_LIST,
            default=["1M"], key="wp_tenors",
            help="Each selected tenor is its own basket strategy.",
        )
        direction_labels = st.multiselect(
            "Direction(s) (shared across both legs)",
            list(DIRECTIONS.keys()),
            default=["Call (up-and-out)"], key="wp_directions",
        )

    with cc2:
        st.markdown("**Leg A**")
        sd_a_labels = st.multiselect(
            "Strike Δ (Leg A)", list(DELTA_CHOICES.keys()),
            default=["ATM"], key="wp_sd_a",
        )
        kd_a_labels = st.multiselect(
            "KO Δ (Leg A)", list(KO_DELTA_CHOICES.keys()),
            default=["20Δ"], key="wp_kd_a",
        )
        gate_a_options = ["(no gate)"] + list(GATE_REGISTRY.keys())
        gate_a_labels = st.multiselect(
            "Gate(s) (Leg A)", gate_a_options,
            default=["(no gate)"], key="wp_gate_a",
            format_func=lambda k: ("(no gate)" if k == "(no gate)"
                                   else GATE_REGISTRY[k][0]),
        )
        gates_a_resolved = [None if k == "(no gate)" else k
                            for k in gate_a_labels]

    with cc3:
        st.markdown("**Leg B**")
        sd_b_labels = st.multiselect(
            "Strike Δ (Leg B)", list(DELTA_CHOICES.keys()),
            default=["ATM"], key="wp_sd_b",
        )
        kd_b_labels = st.multiselect(
            "KO Δ (Leg B)", list(KO_DELTA_CHOICES.keys()),
            default=["20Δ"], key="wp_kd_b",
        )
        gate_b_options = ["(no gate)"] + list(GATE_REGISTRY.keys())
        gate_b_labels = st.multiselect(
            "Gate(s) (Leg B)", gate_b_options,
            default=["(no gate)"], key="wp_gate_b",
            format_func=lambda k: ("(no gate)" if k == "(no gate)"
                                   else GATE_REGISTRY[k][0]),
        )
        gates_b_resolved = [None if k == "(no gate)" else k
                            for k in gate_b_labels]

    st.markdown("---")
    cd1, cd2, cd3 = st.columns(3)
    with cd1:
        date_max_avail = _date.today()
        date_min_avail = _date(2018, 1, 1)
        try:
            spot_all = load_panel(folder, "SPOT", None)
            if not spot_all.empty:
                date_max_avail = spot_all.index.max().date()
                date_min_avail = spot_all.index.min().date()
        except Exception:
            pass
        default_start = max(_date(2023, 1, 1), date_min_avail)
        if default_start >= date_max_avail:
            default_start = date_min_avail
        start_date = st.date_input(
            "Start date", value=default_start,
            min_value=date_min_avail, max_value=date_max_avail,
            key="wp_start",
        )
        end_date = st.date_input(
            "End date", value=date_max_avail,
            min_value=date_min_avail, max_value=date_max_avail,
            key="wp_end",
        )
    with cd2:
        tx_cost_bps = st.number_input(
            "Transaction cost (bps of notional)",
            min_value=0.0, max_value=20.0, value=2.0, step=0.5,
            key="wp_tx",
        )
        prefer_em = st.radio(
            "EM pairs variant", ["offshore", "onshore"], index=0,
            horizontal=True, key="wp_prefer_em",
        )
    with cd3:
        notional_usd = st.number_input(
            "Notional (USD, per cross)",
            min_value=100_000.0, max_value=200_000_000.0,
            value=10_000_000.0, step=1_000_000.0, format="%.0f",
            key="wp_notional",
            help="Applied PER CROSS. $10M × 21 crosses = $210M total "
                  "capital deployed per trading day per strategy.",
        )
        trade_mode = st.radio(
            "Trade mode",
            ["stack", "single"],
            index=0, horizontal=True, key="wp_trade_mode",
        )
        # MTM toggle — disabled. WO engine doesn't yet have a daily MTM
        # pricer for the worst-of structure (the single-leg engine has
        # `compute_mtm_curves` but there's no analogue in core/worstof).
        # Locking the toggle prevents the silent no-op of letting users
        # think they enabled it. Default off matches realized-at-expiry.
        st.checkbox(
            "Mark-to-market mode (daily)",
            value=False,
            disabled=True,
            key="wp_mtm",
            help=("DISABLED — daily MTM for worst-of structures is "
                   "not yet implemented in the engine. Worst-of P&L "
                   "is computed at expiry only. The realized equity "
                   "curve in the drilldown remains accurate; only "
                   "intra-trade daily MTM is unavailable."),
        )
        enable_mtm = False

    # Auto-compute pair combos and the basket-strategy axis cardinality
    pair_combos = [tuple(c) for c in combinations(sorted(pairs_sel), 2)]
    n_baskets = (len(tenors_sel) * len(direction_labels)
                  * len(sd_a_labels) * len(kd_a_labels)
                  * max(len(gates_a_resolved), 1)
                  * len(sd_b_labels) * len(kd_b_labels)
                  * max(len(gates_b_resolved), 1))

    st.caption(
        f"**{n_baskets}** basket strategies will run across "
        f"**{len(pair_combos)} crosses** ({len(tenors_sel)} tenors × "
        f"{len(direction_labels)} dirs × {len(sd_a_labels)}·"
        f"{len(kd_a_labels)}·{max(len(gates_a_resolved), 1)} leg-A × "
        f"{len(sd_b_labels)}·{len(kd_b_labels)}·"
        f"{max(len(gates_b_resolved), 1)} leg-B). Each strategy holds "
        f"all {len(pair_combos)} cross-positions. Trade mode: "
        f"`{trade_mode}`."
    )

    if pair_combos:
        preview_n = min(10, len(pair_combos))
        preview = ", ".join(f"{a}×{b}" for a, b in pair_combos[:preview_n])
        more = (f" … (+{len(pair_combos) - preview_n} more)"
                if len(pair_combos) > preview_n else "")
        st.caption(f"**Crosses:** {preview}{more}")

    can_run = (n_baskets > 0 and pair_combos and tenors_sel
               and direction_labels and sd_a_labels and kd_a_labels
               and sd_b_labels and kd_b_labels
               and gates_a_resolved and gates_b_resolved)
    run_clicked = st.button(
        "▶ Run WO portfolio backtest", type="primary",
        disabled=not can_run, key="wp_run_btn",
    )

    if run_clicked:
        all_basket_results: dict[str, list] = {}
        all_basket_specs: dict[str, list] = {}
        axis_combos = list(product(
            tenors_sel, direction_labels,
            sd_a_labels, kd_a_labels, gates_a_resolved,
            sd_b_labels, kd_b_labels, gates_b_resolved,
        ))
        n_baskets_actual = len(axis_combos)

        progress_bar = st.progress(0.0, text="Starting WO basket runs…")
        t0 = time.time()
        last_update = [t0]

        def cb(p, name):
            now = time.time()
            if now - last_update[0] > 0.1 or p >= 1.0:
                progress_bar.progress(min(p, 1.0),
                                      text=f"Running: {name} "
                                            f"({p*100:.0f}%)")
                last_update[0] = now

        for axis_i, (tenor, dir_label, sd_a, kd_a, ga,
                       sd_b, kd_b, gb) in enumerate(axis_combos):
            basket_name = _wo_basket_strategy_name(
                tenor, dir_label, sd_a, kd_a, ga, sd_b, kd_b, gb,
            )
            # Build per-cross specs sharing these uniform params.
            specs_this = build_worstof_grid(
                pair_combos=pair_combos,
                tenors=[tenor],
                leg_a_directions=[DIRECTIONS[dir_label]],
                leg_b_directions=[DIRECTIONS[dir_label]],
                leg_a_strike_deltas=[(sd_a, DELTA_CHOICES[sd_a])],
                leg_b_strike_deltas=[(sd_b, DELTA_CHOICES[sd_b])],
                leg_a_ko_deltas=[(kd_a, KO_DELTA_CHOICES[kd_a])],
                leg_b_ko_deltas=[(kd_b, KO_DELTA_CHOICES[kd_b])],
                gates_a=[ga], gates_b=[gb],
                tx_cost_bps=tx_cost_bps,
                prefer=prefer_em,
                trade_mode=trade_mode,
                multiplier=wo_multiplier,
                pricing_engine=engine_cfg["pricing_engine"],
                correlation_source=engine_cfg["correlation_source"],
                correlation_value=engine_cfg["correlation_value"],
                mc_n_paths=engine_cfg["mc_n_paths"],
            )
            if not specs_this:
                all_basket_results[basket_name] = []
                all_basket_specs[basket_name] = []
                continue

            sub_results = run_worstof_grid(
                folder, specs_this, start_date, end_date,
                notional_usd=notional_usd,
                progress_cb=lambda p, name, _i=axis_i: cb(
                    (_i + p) / n_baskets_actual,
                    f"[basket {_i+1}/{n_baskets_actual}] {name}",
                ),
            )
            pooled = []
            for s in specs_this:
                pooled.extend(sub_results.get(s.name, []))
            all_basket_results[basket_name] = pooled
            all_basket_specs[basket_name] = specs_this

        elapsed = time.time() - t0
        progress_bar.empty()

        st.session_state["wo_port_results"] = all_basket_results
        st.session_state["wo_port_specs"] = all_basket_specs
        # Union of leg-A + leg-B selections, preserving the user-selection
        # order (leg A first, then any extras from leg B). The download-
        # filename builder doesn't differentiate the two legs in the name,
        # matching the user's filename spec; preserving order keeps the
        # filename stable regardless of internal `KO_DELTA_CHOICES` ordering.
        def _dedupe_preserve_order(*iters):
            seen, out = set(), []
            for it in iters:
                for v in it:
                    if v not in seen:
                        seen.add(v); out.append(v)
            return out
        st.session_state["wo_port_meta"] = {
            "pairs": sorted(pairs_sel),
            "pair_combos": pair_combos,
            "tenors": tenors_sel,
            "directions": direction_labels,
            "strike_deltas": _dedupe_preserve_order(sd_a_labels, sd_b_labels),
            "ko_deltas": _dedupe_preserve_order(kd_a_labels, kd_b_labels),
            "trade_mode": trade_mode,
            "start": start_date, "end": end_date,
            "tx_cost_bps": tx_cost_bps,
            "prefer_em": prefer_em,
            "notional_usd": notional_usd,
            "n_baskets": n_baskets_actual,
            "n_crosses": len(pair_combos),
            "elapsed": elapsed,
        }
        n_trades = sum(len(t) for t in all_basket_results.values())
        st.success(
            f"Done in {elapsed:.1f}s — {n_baskets_actual} basket "
            f"strategies, {n_trades} pooled trades total "
            f"({len(pair_combos)} crosses per basket). Switch to the "
            f"WO EKO Portfolio drilldown tab to inspect."
        )

    # --- Summary table ---
    if "wo_port_results" not in st.session_state:
        st.info("Configure axes above and click **Run** to see a "
                  "summary table.")
        return

    from core.worstof import worstof_trades_to_df, worstof_summarize
    results = st.session_state["wo_port_results"]
    meta = st.session_state.get("wo_port_meta", {})

    st.markdown("---")
    st.markdown("### Latest run — WO basket strategies")
    pairs_str = ", ".join(meta.get("pairs", []))
    st.caption(
        f"**Basket:** {pairs_str} ({meta.get('n_crosses', 0)} crosses)  ·  "
        f"period {meta.get('start')} → {meta.get('end')}  ·  "
        f"${meta.get('notional_usd', 0):,.0f} per cross  ·  "
        f"tx {meta.get('tx_cost_bps', 0):.1f} bps  ·  "
        f"mode `{meta.get('trade_mode', 'stack')}`"
    )

    rows = []
    for name, trades_list in results.items():
        if not trades_list:
            rows.append({"Basket strategy": name, "n trades": 0})
            continue
        sdf = worstof_trades_to_df(trades_list)
        s = worstof_summarize(sdf)
        rows.append({
            "Basket strategy": name,
            "n": s["n_trades"],
            "Crosses": sdf.apply(
                lambda r: f"{r['leg_a_pair']}×{r['leg_b_pair']}", axis=1
            ).nunique(),
            "Win %": f"{s.get('win_rate', 0) * 100:.0f}",
            "KO %": f"{s.get('any_ko_rate', 0) * 100:.0f}",
            "Σ Premium": _fmt_usd(s.get("total_premium_paid_usd", 0)),
            "Σ Payoff": _fmt_usd(s.get("total_payout_usd", 0)),
            "PnL": _fmt_usd(s.get("total_pnl_usd", 0)),
            "Sharpe (m)": f"{s.get('sharpe_monthly', 0):+.2f}",
            "Max DD": _fmt_usd(s.get("max_drawdown_usd", 0)),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

    # =====================================================================
    # Downloads — canonical schema matching the Worst-of bulk tab
    # =====================================================================
    st.markdown("---")
    st.markdown("### Downloads")
    st.caption(
        "Bulk run results in CSV form. **Summary** = one row per WO "
        "basket strategy. **Time series** = long-format daily/monthly/"
        "annual rows with `pnl_usd`, `equity_usd`, `drawdown_usd` per "
        "period end. Schema matches the regular Worst-of tab (via "
        "`strategy_type='wo_basket'`) so results stack with the "
        "single-leg / worst-of CSVs in the downstream app."
    )

    # --- Canonical summary frame (same columns as WO bulk CSV) ---
    summary_canon_rows_wo = []
    for name, trades_list in results.items():
        if not trades_list:
            continue
        sdf = worstof_trades_to_df(trades_list)
        s = worstof_summarize(sdf)
        g2p = s.get("gain_to_pain", 0.0)
        n_crosses_strat = (sdf.apply(
            lambda r: f"{r['leg_a_pair']}×{r['leg_b_pair']}", axis=1
        ).nunique() if len(sdf) else 0)
        crosses_str = ",".join(sorted(
            f"{r['leg_a_pair']}×{r['leg_b_pair']}"
            for _, r in sdf.iterrows()
        )) if len(sdf) else ""
        # Dedup the crosses_str (each cross appears N times in trades)
        if crosses_str:
            unique_crosses = sorted(set(crosses_str.split(",")))
            crosses_str = ",".join(unique_crosses)

        summary_canon_rows_wo.append({
            # Identifier
            "strategy_name": name,
            "strategy_type": "wo_basket",
            "n_trades": int(s.get("n_trades", 0)),
            # Money totals
            "notional_usd": s.get("notional_usd", 0.0),
            "total_premium_paid_usd": s.get("total_premium_paid_usd",
                                              0.0),
            "total_tx_cost_usd": s.get("total_tx_cost_usd", 0.0),
            "total_payout_usd": s.get("total_payout_usd", 0.0),
            "total_pnl_usd": s.get("total_pnl_usd", 0.0),
            "max_drawdown_usd": s.get("max_drawdown_usd", 0.0),
            # Rates / ratios — convert 0-1 fractions to percent
            "win_rate_pct": float(s.get("win_rate", 0.0)) * 100,
            "premium_recovery_pct": s.get("premium_recovery_pct", 0.0),
            # Sharpe block
            "sharpe_monthly": s.get("sharpe_monthly", 0.0),
            "annual_sharpe_mean": s.get("annual_sharpe_mean", 0.0),
            "annual_sharpe_min": s.get("annual_sharpe_min", 0.0),
            "annual_sharpe_std": s.get("annual_sharpe_std", 0.0),
            "annual_sharpe_cv": s.get("annual_sharpe_cv", 0.0),
            "annual_sharpe_score": s.get("annual_sharpe_score", 0.0),
            # Cross-year consistency
            "n_years": int(s.get("n_years", 0)),
            "pct_positive_years": s.get("pct_positive_years", 0.0),
            "min_annual_pnl_usd": s.get("min_annual_pnl_usd", 0.0),
            "calmar": s.get("calmar", 0.0),
            "gain_to_pain": (g2p if g2p != float("inf") else np.nan),
            "ulcer_index": s.get("ulcer_index", 0.0),
            # Single-leg placeholders (NaN for WO basket)
            "feasibility_pct": np.nan,
            "ko_rate_pct": np.nan,
            # WO-specific — populated
            "leg_a_ko_rate_pct": float(s.get("leg_a_ko_rate", 0.0)) * 100,
            "leg_b_ko_rate_pct": float(s.get("leg_b_ko_rate", 0.0)) * 100,
            "both_survive_rate_pct": float(
                s.get("both_survive_rate", 0.0)
            ) * 100,
            "structure_vs_min_leg_pct": s.get("structure_vs_min_leg_pct",
                                                 0.0),
            # Basket-specific extras
            "basket_n_crosses": int(n_crosses_strat),
            "basket_crosses": crosses_str,
            "trade_mode": meta.get("trade_mode", "stack"),
        })
    summary_canon_df_wo = (pd.DataFrame(summary_canon_rows_wo)
                              if summary_canon_rows_wo else pd.DataFrame())

    # --- Canonical time-series frame ---
    # Pooled ledger → long-format. Daily rows are dropped — monthly +
    # annual only, since that's the cadence the downstream app uses.
    from core.worstof import worstof_export_time_series
    ts_frames_wo = []
    for name, trades_list in results.items():
        if not trades_list:
            continue
        sdf = worstof_trades_to_df(trades_list)
        ts = worstof_export_time_series(sdf)
        if ts.empty:
            continue
        # Keep monthly + annual only
        ts = ts[ts["period_type"].isin(["monthly", "annual"])].copy()
        if ts.empty:
            continue
        ts.insert(0, "strategy_name", name)
        ts.insert(1, "strategy_type", "wo_basket")
        ts_frames_wo.append(ts)
    ts_combined_wo = (pd.concat(ts_frames_wo, ignore_index=True)
                        if ts_frames_wo else pd.DataFrame())

    cdl_a, cdl_b = st.columns(2)
    with cdl_a:
        if not summary_canon_df_wo.empty:
            st.download_button(
                label=(f"⬇ Download summary table "
                         f"({len(summary_canon_df_wo)} rows, CSV)"),
                data=summary_canon_df_wo.to_csv(index=False).encode("utf-8"),
                file_name=_build_portfolio_download_filename(
                    meta, "WO-EKO", "summary"),
                mime="text/csv",
                help=("Canonical schema: strategy_name, strategy_type "
                       "('wo_basket'), n_trades, money totals, Sharpe "
                       "block, consistency block, then strategy- and "
                       "basket-specific columns (basket_n_crosses, "
                       "basket_crosses, trade_mode). Stacks with the "
                       "Backtest and Worst-of CSVs."),
                use_container_width=True,
                key="wp_summary_dl",
            )
        else:
            st.caption("_No summary rows yet._")
    with cdl_b:
        if not ts_combined_wo.empty:
            n_strats = ts_combined_wo["strategy_name"].nunique()
            st.download_button(
                label=(f"⬇ Download time series — {n_strats} strategies × "
                         f"{len(ts_combined_wo):,} rows (CSV)"),
                data=ts_combined_wo.to_csv(index=False).encode("utf-8"),
                file_name=_build_portfolio_download_filename(
                    meta, "WO-EKO", "timeseries"),
                mime="text/csv",
                help=("Long-format: period_type ('monthly'|'annual'), "
                       "period_end, pnl_usd, equity_usd, drawdown_usd "
                       "at each period end. BASKET-LEVEL (pooled "
                       "across crosses). Daily rows are excluded — "
                       "available from the trade ledger if needed."),
                use_container_width=True,
                key="wp_ts_dl",
            )
        else:
            st.caption("_No time-series rows yet._")


def render_wo_eko_portfolio_drilldown_tab():
    if "wo_port_results" not in st.session_state:
        st.info("Run a portfolio backtest first (WO EKO Portfolio tab).")
        return

    from core.worstof import worstof_trades_to_df, worstof_summarize, worstof_equity_curve
    from plotly.subplots import make_subplots

    results = st.session_state["wo_port_results"]
    specs_by_basket = st.session_state.get("wo_port_specs", {})
    meta = st.session_state.get("wo_port_meta", {})

    names = [n for n in results if results[n]]
    if not names:
        st.warning("All basket strategies produced zero trades.")
        return

    selected = st.selectbox(
        "Select basket strategy", names, index=0,
        key="wp_drill_select",
    )
    trades_list = results[selected]
    df = worstof_trades_to_df(trades_list)
    if df.empty:
        st.warning("Empty pooled trade ledger for this basket.")
        return

    s = worstof_summarize(df)
    # Build the cross identifier for grouping
    df["cross"] = df.apply(
        lambda r: f"{r['leg_a_pair']}×{r['leg_b_pair']}", axis=1,
    )
    n_crosses = df["cross"].nunique()

    # --- Header ---
    cross_list_str = ", ".join(sorted(df["cross"].unique()))
    st.markdown(f"### {selected}")
    st.caption(
        f"**Crosses in basket** ({n_crosses}): {cross_list_str}  ·  "
        f"notional ${meta.get('notional_usd', 0):,.0f} per cross  ·  "
        f"trade mode `{meta.get('trade_mode', 'stack')}`"
    )

    # --- Headline metrics ---
    notional_usd_v = s.get("notional_usd", 0)
    total_pnl_v = s.get("total_pnl_usd", 0)
    total_pnl_pct = (total_pnl_v / notional_usd_v * 100
                     if notional_usd_v > 0 else 0.0)
    win_rate_pct = s.get("win_rate", 0.0) * 100
    ko_rate_pct = s.get("any_ko_rate", 0.0) * 100

    cs = st.columns(6)
    metrics = [
        ("Trades (pooled)", f"{s['n_trades']}",
         f"{n_crosses} crosses"),
        ("Total PnL", _fmt_usd(total_pnl_v),
         f"{total_pnl_pct:+.2f}% notl"),
        ("Sharpe (m)", f"{s.get('sharpe_monthly', 0):+.2f}",
         "monthly × √12"),
        ("Max DD", _fmt_usd(s.get("max_drawdown_usd", 0)),
         "realized, by expiry"),
        ("Win rate", f"{win_rate_pct:.0f}%",
         f"{int(s['n_trades'] * win_rate_pct / 100)} winners"),
        ("KO rate (any leg)", f"{ko_rate_pct:.0f}%",
         "either leg knocked"),
    ]
    for col, (lbl, val, sub) in zip(cs, metrics):
        col.metric(lbl, val, sub)

    st.divider()

    # --- Equity curve ---
    st.markdown("#### Equity & drawdown (basket)")
    eq = worstof_equity_curve(df)
    fig_eq = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.07,
        subplot_titles=("Cumulative P&L (USD)", "Drawdown (USD)"),
    )
    fig_eq.add_trace(go.Scatter(
        x=eq.index, y=eq["equity_usd"],
        mode="lines", line=dict(color="#22c55e", width=2),
        showlegend=False,
        hovertemplate=("%{x|%Y-%m-%d}<br>"
                        "Equity: $%{y:,.0f}<extra></extra>"),
    ), row=1, col=1)
    fig_eq.add_trace(go.Scatter(
        x=eq.index, y=eq["drawdown_usd"],
        mode="lines", line=dict(color="#ef4444", width=1.5),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.25)",
        showlegend=False,
        hovertemplate=("%{x|%Y-%m-%d}<br>"
                        "DD: $%{y:,.0f}<extra></extra>"),
    ), row=2, col=1)
    fig_eq.update_layout(
        height=520, margin=dict(l=10, r=10, t=50, b=10),
        title_text="Realized equity & drawdown (pooled across crosses)",
    )
    fig_eq.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig_eq, width='stretch')

    st.divider()

    # --- PnL breakdowns: year / cross / year × cross ---
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    df["_year"] = df["expiry_date"].dt.year

    st.markdown("#### P&L by year")
    yearly = df.groupby("_year")["pnl_usd"].sum().sort_index()
    _render_pnl_by_year_chart(yearly, "P&L by expiry year")
    # Add per-year Sharpe column from the realized equity curve.
    # WO has no MTM mode (engine doesn't price the structure daily),
    # so `eq` is the only equity stream available.
    sharpe_by_year = _annual_sharpe_per_year(eq)
    yearly_df = yearly.reset_index().rename(
        columns={"_year": "Year", "pnl_usd": "PnL (USD)"}
    )
    yearly_df["Sharpe (m)"] = yearly_df["Year"].map(
        lambda y: sharpe_by_year.get(int(y), float("nan"))
    )
    yearly_df = yearly_df.assign(**{
        "PnL (USD)": yearly_df["PnL (USD)"].apply(_fmt_usd),
        "Sharpe (m)": yearly_df["Sharpe (m)"].apply(
            lambda v: f"{v:+.2f}" if pd.notna(v) else "—"
        ),
    })
    st.dataframe(yearly_df, hide_index=True, width='stretch')
    st.caption(
        "**Sharpe (m)** is the monthly-basis annualized Sharpe ratio "
        "*within* each calendar year — same formula as the headline "
        "`annual_sharpe_*` columns in the summary CSV: "
        "`mean(monthly_pnl) / std(monthly_pnl) × √12`, monthly stream "
        "from the realized-at-expiry equity curve. Years with fewer "
        "than 2 valid monthly observations show '—'."
    )

    st.divider()

    st.markdown("#### P&L by cross")
    by_cross = df.groupby("cross")["pnl_usd"].sum()
    _render_pnl_by_pair_chart(by_cross, pair_label="Cross")
    # Full per-cross breakdown
    # Precompute structure_ko = leg_a OR leg_b. For typical worst-of
    # structures (all UO or all DO), the structure pays
    # min(payoff_A, payoff_B); if either leg knocks out, the min is 0
    # and the structure has KO'd. This matches the headline `any_ko_rate`
    # metric. We add this as a tagged column rather than computing
    # inside .agg() so it sorts/formats predictably.
    df_with_struct = df.copy()
    df_with_struct["_struct_ko"] = (
        df_with_struct["leg_a_knocked_out"].astype(bool)
        | df_with_struct["leg_b_knocked_out"].astype(bool)
    )
    cross_tbl = df_with_struct.groupby("cross").agg(
        n_trades=("pnl_usd", "size"),
        total_pnl_usd=("pnl_usd", "sum"),
        total_premium_usd=("structure_premium_paid_usd", "sum"),
        total_payoff_usd=("worst_of_payoff_usd", "sum"),
        leg_a_ko_rate=("leg_a_knocked_out",
                       lambda s: 100.0 * s.astype(bool).sum() / len(s)),
        leg_b_ko_rate=("leg_b_knocked_out",
                       lambda s: 100.0 * s.astype(bool).sum() / len(s)),
        structure_ko_rate=("_struct_ko",
                           lambda s: 100.0 * s.sum() / len(s)),
        win_rate=("pnl_usd",
                  lambda s: 100.0 * (s > 0).sum() / len(s)),
    ).reset_index().sort_values("total_pnl_usd", ascending=False)
    cross_tbl_disp = cross_tbl.copy()
    cross_tbl_disp["total_pnl_usd"] = (
        cross_tbl_disp["total_pnl_usd"].apply(_fmt_usd)
    )
    cross_tbl_disp["total_premium_usd"] = (
        cross_tbl_disp["total_premium_usd"].apply(_fmt_usd)
    )
    cross_tbl_disp["total_payoff_usd"] = (
        cross_tbl_disp["total_payoff_usd"].apply(_fmt_usd)
    )
    cross_tbl_disp["leg_a_ko_rate"] = (
        cross_tbl_disp["leg_a_ko_rate"].apply(lambda x: f"{x:.0f}%")
    )
    cross_tbl_disp["leg_b_ko_rate"] = (
        cross_tbl_disp["leg_b_ko_rate"].apply(lambda x: f"{x:.0f}%")
    )
    cross_tbl_disp["structure_ko_rate"] = (
        cross_tbl_disp["structure_ko_rate"].apply(lambda x: f"{x:.0f}%")
    )
    cross_tbl_disp["win_rate"] = (
        cross_tbl_disp["win_rate"].apply(lambda x: f"{x:.0f}%")
    )
    cross_tbl_disp.columns = ["Cross", "n trades", "PnL",
                                "Σ Premium", "Σ Payoff",
                                "Leg A KO %", "Leg B KO %",
                                "Structure KO %", "Win %"]
    st.dataframe(cross_tbl_disp, hide_index=True, width='stretch')
    st.caption(
        "**Structure KO %** = % of trades where the worst-of "
        "structure knocked out (either leg's barrier was hit). "
        "For a typical worst-of with both legs in the same direction, "
        "if either leg knocks, `min(payoff_A, payoff_B) = 0` so the "
        "structure paid nothing — that's the trade-level KO event. "
        "Should equal **max(Leg A KO %, Leg B KO %)** when the legs' "
        "KO events overlap perfectly, and **min(Leg A KO % + Leg B KO %, 100%)** "
        "in the opposite-extreme case where their KOs are disjoint."
    )

    st.divider()

    # ---- Monthly P&L heatmap (year × month) ----
    # New: matches App 12's WO-RKO drilldown for consistency. Per-month
    # rollup with a YTD column and a diverging RYG palette so seasonality
    # / regime-change visible at a glance.
    st.markdown("#### Monthly P&L heatmap — year × month (USD)")
    from core.worstof import worstof_monthly_pnl
    monthly_usd = worstof_monthly_pnl(df)
    if not monthly_usd.empty:
        mdf = monthly_usd.copy()
        mdf.index = pd.to_datetime(mdf.index)
        pivot = mdf.groupby([mdf.index.year, mdf.index.month]).sum().unstack(
            fill_value=0)
        month_labels = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                          7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        pivot = pivot.rename(columns=month_labels)
        pivot["YTD"] = pivot.sum(axis=1)
        # Matplotlib-free diverging palette (same idea as App 12)
        arr = pivot.to_numpy(dtype=float, na_value=np.nan)
        finite = arr[np.isfinite(arr)]
        vmax = float(np.max(np.abs(finite))) if finite.size else 0.0
        def _ryg(v):
            if pd.isna(v) or not np.isfinite(v) or vmax == 0.0:
                return ""
            x = max(-1.0, min(1.0, v / vmax))
            if x >= 0:
                r, g, b = int(255 + (60 - 255) * x), int(255 + (170 - 255) * x), int(180 + (80 - 180) * x)
            else:
                xm = -x
                r, g, b = int(255 + (200 - 255) * xm), int(255 + (60 - 255) * xm), int(180 + (60 - 180) * xm)
            return f"background-color: rgb({r},{g},{b}); color: #1a1a1a;"
        st.dataframe(
            pivot.style.format("${:,.0f}", na_rep="").map(_ryg),
            use_container_width=True,
        )

    st.divider()

    # ---- Field-selector monthly heatmap ----
    # Sibling to the P&L heatmap above. User picks a field (Premium /
    # Payoff / Tx cost / etc.) from a dropdown; we re-roll the pooled
    # WO trade ledger by (year × month-of-expiry) and render the same
    # diverging-RYG palette. Aggregation differs by field: sums for
    # USD totals, means for rates, count for trade counts. YTD column
    # is added for sum-based fields only (re-summing a monthly mean
    # doesn't give a meaningful annual mean).
    st.markdown("#### Monthly heatmap — pick a field")
    st.caption(
        "Same year-by-month layout as the P&L heatmap above, but for "
        "whichever trade-level field you want to slice. Premium / Payoff "
        "/ Tx cost aggregate as **sums** (a YTD column is added). KO and "
        "win rates aggregate as **means within the (year, month) bucket** "
        "weighted by trades; the **Year** column re-computes the rate "
        "over the full year for a correct annual figure. **Trade count** "
        "shows how many trades expired in each month."
    )

    # (column, agg, formatter_fn, year_total_mode) per option.
    # year_total_mode: 'sum' = re-sum, 'mean' = re-mean over the full year
    # at the trade level (NOT mean of monthly means), 'count' = re-count,
    # 'none' = no Year column.
    field_options = {
        "Σ Premium (USD)": (
            "structure_premium_paid_usd", "sum",
            lambda v: f"${v:,.0f}" if pd.notna(v) and v != 0 else "—",
            "sum",
        ),
        "Σ Payoff (USD)": (
            "worst_of_payoff_usd", "sum",
            lambda v: f"${v:,.0f}" if pd.notna(v) and v != 0 else "—",
            "sum",
        ),
        "Σ Transaction cost (USD)": (
            "tx_cost_usd", "sum",
            lambda v: f"${v:,.0f}" if pd.notna(v) and v != 0 else "—",
            "sum",
        ),
        "Σ P&L (USD)": (
            "pnl_usd", "sum",
            lambda v: f"${v:,.0f}" if pd.notna(v) else "—",
            "sum",
        ),
        "# Trades": (
            "pnl_usd", "size",
            lambda v: f"{int(v):,}" if pd.notna(v) and v > 0 else "—",
            "count",
        ),
        "Either-leg KO rate %": (
            "_any_ko", "mean",
            lambda v: f"{v*100:.0f}%" if pd.notna(v) else "—",
            "mean",
        ),
        "Both-survive rate %": (
            "_both_survive", "mean",
            lambda v: f"{v*100:.0f}%" if pd.notna(v) else "—",
            "mean",
        ),
        "Win rate %": (
            "_is_winner", "mean",
            lambda v: f"{v*100:.0f}%" if pd.notna(v) else "—",
            "mean",
        ),
    }

    field_choice = st.selectbox(
        "Field to display", list(field_options.keys()),
        index=0,   # Σ Premium — what the user asked about
        key="wp_field_heatmap_select",
    )
    col, agg, fmt, year_mode = field_options[field_choice]

    # Build derived helper columns lazily so we don't pay the cost if
    # the user picks a basic field.
    work = df.copy()
    work["_year_m"] = pd.to_datetime(work["expiry_date"])
    work["_y"] = work["_year_m"].dt.year
    work["_m"] = work["_year_m"].dt.month
    if col == "_any_ko":
        work[col] = (work["leg_a_knocked_out"].astype(bool)
                       | work["leg_b_knocked_out"].astype(bool)).astype(float)
    elif col == "_both_survive":
        work[col] = ((~work["leg_a_knocked_out"].astype(bool))
                       & (~work["leg_b_knocked_out"].astype(bool))).astype(float)
    elif col == "_is_winner":
        work[col] = (work["pnl_usd"] > 0).astype(float)

    grouped = work.groupby(["_y", "_m"])[col]
    if agg == "size":
        # Count — use grouped.size() which ignores column, but we want
        # per-bucket trade count which is exactly what .size() gives us.
        pivot_field = grouped.size().unstack(fill_value=0)
    elif agg == "mean":
        pivot_field = grouped.mean().unstack(fill_value=np.nan)
    else:   # 'sum'
        pivot_field = grouped.sum().unstack(fill_value=0)

    if not pivot_field.empty:
        month_labels_full = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                                 7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        pivot_field = pivot_field.rename(columns=month_labels_full)
        # Reorder columns Jan..Dec (groupby returns numeric order which
        # already is 1..12, but renaming preserves whatever the original
        # column order was; explicitly reindex to be safe).
        present = [m for m in month_labels_full.values() if m in pivot_field.columns]
        pivot_field = pivot_field[present]

        # Year column — recomputed correctly per agg type
        if year_mode == "sum":
            pivot_field["Year"] = pivot_field.sum(axis=1)
        elif year_mode == "count":
            pivot_field["Year"] = pivot_field.sum(axis=1).astype(int)
        elif year_mode == "mean":
            # Re-aggregate at the trade level over the full year, NOT
            # mean of monthly means (which would weight equally regardless
            # of trade volume in each month).
            year_means = work.groupby("_y")[col].mean()
            pivot_field["Year"] = pivot_field.index.to_series().map(year_means)
        # year_mode == 'none' → no Year column

        # Diverging RYG palette around the mid-point.
        # For rate fields (0..1) and counts, "zero" isn't the meaningful
        # mid-point — anchor around the median instead so the colour
        # band actually communicates variation. For USD fields, anchor
        # around 0 so red = loss / spend, green = gain.
        arr = pivot_field.to_numpy(dtype=float, na_value=np.nan)
        finite = arr[np.isfinite(arr)]
        if year_mode == "sum" and col == "pnl_usd":
            mid = 0.0
            vmax = float(np.max(np.abs(finite))) if finite.size else 0.0
            v_lo, v_hi = -vmax, vmax
        elif year_mode == "sum":
            # Premium / Payoff / Tx cost — typically all same sign;
            # use min..max so the palette spans the actual range.
            mid = float(np.median(finite)) if finite.size else 0.0
            v_lo = float(np.min(finite)) if finite.size else 0.0
            v_hi = float(np.max(finite)) if finite.size else 1.0
        else:   # mean / count
            mid = float(np.median(finite)) if finite.size else 0.5
            v_lo = float(np.min(finite)) if finite.size else 0.0
            v_hi = float(np.max(finite)) if finite.size else 1.0

        def _ryg2(v):
            if pd.isna(v) or not np.isfinite(v):
                return ""
            if v_hi == v_lo:
                return "background-color: rgb(255,255,180); color: #1a1a1a;"
            if v >= mid:
                t = (v - mid) / (v_hi - mid) if v_hi > mid else 0.0
                t = max(0.0, min(1.0, t))
                r = int(255 + (60 - 255) * t)
                g = int(255 + (170 - 255) * t)
                b = int(180 + (80 - 180) * t)
            else:
                t = (mid - v) / (mid - v_lo) if mid > v_lo else 0.0
                t = max(0.0, min(1.0, t))
                r = int(255 + (200 - 255) * t)
                g = int(255 + (60 - 255) * t)
                b = int(180 + (60 - 180) * t)
            return f"background-color: rgb({r},{g},{b}); color: #1a1a1a;"

        st.dataframe(
            pivot_field.style.format(fmt, na_rep="—").map(_ryg2),
            use_container_width=True,
        )

    st.divider()

    st.markdown("#### P&L heatmap — year × cross")
    year_cross = df.groupby(["_year", "cross"])["pnl_usd"].sum().unstack(
        fill_value=0.0
    )
    _render_pnl_heatmap(year_cross, pair_label="Cross")

    st.divider()

    # ---- Bulk per-cross breakdown CSV download ----
    # One row per (strategy_name, cross). Same columns as the per-cross
    # display table so the downstream analyzer can reproduce it offline.
    st.markdown("#### Per-cross breakdown — CSV download")
    st.caption(
        "One row per (strategy_name, cross) across all basket strategies "
        "in this run. The downstream analyzer ingests this alongside the "
        "main summary CSV to reproduce the per-cross table for any "
        "basket without re-running the backtest."
    )
    per_cross_rows = []
    for name, t in results.items():
        if not t:
            continue
        df_one = worstof_trades_to_df(t)
        if df_one.empty:
            continue
        work = df_one.copy()
        work["cross"] = work["leg_a_pair"] + "×" + work["leg_b_pair"]
        work["_struct_ko"] = (work["leg_a_knocked_out"].astype(bool)
                                | work["leg_b_knocked_out"].astype(bool))
        b = work.groupby("cross").agg(
            n_trades=("pnl_usd", "size"),
            total_pnl_usd=("pnl_usd", "sum"),
            total_premium_usd=("structure_premium_paid_usd", "sum"),
            total_payoff_usd=("worst_of_payoff_usd", "sum"),
            leg_a_ko_rate_pct=("leg_a_knocked_out",
                                  lambda s: 100.0 * s.astype(bool).sum() / len(s)),
            leg_b_ko_rate_pct=("leg_b_knocked_out",
                                  lambda s: 100.0 * s.astype(bool).sum() / len(s)),
            structure_ko_rate_pct=("_struct_ko",
                                      lambda s: 100.0 * s.sum() / len(s)),
            win_rate_pct=("pnl_usd",
                            lambda s: 100.0 * (s > 0).sum() / len(s)),
        ).reset_index().rename(columns={"cross": "Cross"})
        if b.empty:
            continue
        b = b.sort_values("total_pnl_usd", ascending=False)
        b.insert(0, "strategy_name", name)
        b.insert(1, "strategy_type", "wo_basket")
        per_cross_rows.append(b)
    if per_cross_rows:
        per_cross_df = pd.concat(per_cross_rows, ignore_index=True)
        st.download_button(
            label=(f"⬇ Download per-cross breakdown "
                     f"({len(per_cross_df)} rows, CSV)"),
            data=per_cross_df.to_csv(index=False).encode("utf-8"),
            file_name="wo_eko_portfolio_per_cross.csv",
            mime="text/csv",
            use_container_width=True, key="wp_per_cross_dl",
        )
    else:
        st.caption("_No per-cross rows yet._")

    st.divider()

    # --- Full ledger expander ---
    with st.expander(
        f"📜 Full pooled WO trade ledger ({s['n_trades']} trades)",
        expanded=False,
    ):
        led_cols_default = [
            "cross", "trade_date", "expiry_date", "tenor_label",
            "leg_a_strike", "leg_a_barrier",
            "leg_b_strike", "leg_b_barrier",
            "structure_premium_paid_usd", "worst_of_payoff_usd",
            "pnl_usd",
            "leg_a_knocked_out", "leg_b_knocked_out",
        ]
        led_cols = [c for c in led_cols_default if c in df.columns]
        st.dataframe(
            df[led_cols].sort_values(["cross", "trade_date"]),
            hide_index=True, width='stretch',
        )


# =============================================================================
# Render tabs
# =============================================================================
with tab_pricer:
    render_pricer_tab()

with tab_backtest:
    render_backtest_tab()

with tab_drilldown:
    render_drilldown_tab()

with tab_worstof_pricer:
    render_worstof_pricer_tab()

with tab_worstof:
    render_worstof_tab()

with tab_wo_drill:
    render_worstof_drilldown_tab()

with tab_eko_port:
    render_eko_portfolio_tab()

with tab_eko_drill:
    render_eko_portfolio_drilldown_tab()

with tab_wo_port:
    render_wo_eko_portfolio_tab()

with tab_wo_drill_port:
    render_wo_eko_portfolio_drilldown_tab()

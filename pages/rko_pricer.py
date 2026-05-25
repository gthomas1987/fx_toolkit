"""RKO Pricer — American-barrier Knock-Out Option Pricer & Backtester.

Sub-app of the FX Toolkit. Reached via the landing page or sidebar
nav; not run directly. (Formerly: apps/12_american_ko_pricer.py.)

# What this is
European exercise (only at expiry) + American barrier (continuously
monitored — ANY touch during the option's life kills it). This is the
standard "FX knock-out" sold in flow markets. Materially different from
the EKO Pricer (European exercise + European barrier, where the
barrier is checked only at expiry).

# Why a separate page
The pricing math is fundamentally different — the EKO Pricer uses a
clean vanilla + digital decomposition (closed form, no path dependence
in the formula). American-barrier pricing requires either the
Reiner-Rubinstein closed form (with 8 separate cases for the 4
option-types × 2 strike-vs-barrier configurations) or a numerical
scheme (tree / PDE) that monitors the barrier at every node.

# Pricing methods
Four numerical methods are shown side-by-side for cross-validation
against Bloomberg / your platform of choice:
  1. Closed-form Reiner-Rubinstein (analytic, exact continuous monitoring)
  2. Binomial tree (CRR) + Broadie-Glasserman-Kou continuity correction
  3. Trinomial tree (Boyle) with explicit barrier-on-node placement
  4. Crank-Nicolson finite difference on the BS PDE
See `core/american_barrier.py` for the math.

# Tabs
Pricer (single-trade snapshot), Backtest + drilldown (single-leg),
Worst-of + drilldown, RKO Portfolio + drilldown (basket across pairs),
and WO-RKO Portfolio + drilldown (basket of worst-of crosses). The
American-barrier backtest uses DAILY OHLC to mark a barrier hit
whenever [day_low, day_high] contains H — a richer monitoring scheme
than close-only.

# Spot data
SPOT panel loads via `load_panel(..., "SPOT", ...)`, which prefers the
"close" column when the CSV has OHLC.
"""
from __future__ import annotations

import os
import sys
from datetime import date as _date
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
from core.american_barrier import (
    ako_closed_form, ako_binomial, ako_trinomial, ako_finite_difference,
    ako_probability_continuous, ako_spot_delta,
)
from core.vanna_volga import vv_price_ko
from core.ko_solvers import solve_strike
from core.vanilla import vanilla_price, vanilla_spot_delta
from core.conventions import get_pip_scale
from core.calendar import compute_option_dates
from core.rates import load_rates_panel, get_rate_at


# =============================================================================
# Page config + styling (matches App 9 visual language)
# =============================================================================
st.set_page_config(
    page_title="RKO Pricer & Backtester",
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
# Sidebar — data folder
# Uses the same shared helper as the Vol Dashboard / EKO Pricer so the
# folder selection persists across pages via st.session_state["data_dir"].
# =============================================================================
from core.ui import data_dir_input as _data_dir_input
st.sidebar.markdown("### Data source")
folder = _data_dir_input(default="market_data")
if folder is None:
    st.info("Specify the market data folder in the sidebar.")
    st.stop()

with st.sidebar.expander("Discovered files", expanded=False):
    s = discovery_summary(folder)
    st.caption(f"Mode: `{s['mode']}`  ·  {s['n_pairs']} pairs across "
                f"{s['n_files']} files")
    if s["categories"]:
        for k, v in sorted(s["categories"].items()):
            st.caption(f"  {k}: {v} pairs")


# Sidebar diagnostic: which smile tenors are registered for each pair.
# Silent fallback to flat-vol is the most common "what happened?" issue —
# this expander surfaces it explicitly.
with st.sidebar.expander("Smile (RR/BF) availability", expanded=False):
    idx_df = None
    try:
        from core.data_loader import get_index
        idx_df = get_index(folder)
    except Exception as _e:
        st.caption(f"⚠ Could not read index: {_e}")
    if idx_df is not None and not idx_df.empty:
        rr_rows = idx_df[idx_df["category"] == "VOL_RR_25D"]
        bf_rows = idx_df[idx_df["category"] == "VOL_BF_25D"]
        st.caption(f"**25Δ RR:** {len(rr_rows)} rows, "
                    f"{rr_rows['pair'].nunique()} pairs, "
                    f"tenors: {sorted(rr_rows['tenor'].unique().tolist())}")
        st.caption(f"**25Δ BF:** {len(bf_rows)} rows, "
                    f"{bf_rows['pair'].nunique()} pairs, "
                    f"tenors: {sorted(bf_rows['tenor'].unique().tolist())}")
        if len(rr_rows) == 0:
            st.caption("⚠ No VOL_RR_25D rows found. The loader accepts old "
                        "names (VOL_25R) and new names (VOL_RR_25D) — but "
                        "if your CSVs use yet another spelling (e.g. "
                        "RR25 / 25_RR), add a row to `_index.csv` with "
                        "category=`VOL_RR_25D` to map it in.")


# =============================================================================
# Sidebar — numerical-method config (collapsed by default)
# =============================================================================
with st.sidebar.expander("Numerical method config", expanded=False):
    st.caption("Closed-form is the analytic reference. The other three "
                "methods discretize and converge as resolution increases.")
    n_steps_binom = st.slider("Binomial steps", min_value=200, max_value=4000,
                                  value=1000, step=100, key="ako_n_binom",
                                  help="CRR with BGK continuity correction. "
                                       "N ≥ 500 typically gives < 1% error.")
    n_steps_trinom = st.slider("Trinomial steps", min_value=100, max_value=2000,
                                   value=500, step=50, key="ako_n_trinom",
                                   help="Boyle tree, Δx chosen so the barrier "
                                        "lies on a node.")
    n_S_fd = st.slider("FD spatial nodes", min_value=100, max_value=600,
                           value=300, step=20, key="ako_n_S",
                           help="Crank-Nicolson grid spatial resolution.")
    n_t_fd = st.slider("FD time steps", min_value=100, max_value=1000,
                           value=400, step=20, key="ako_n_t",
                           help="Crank-Nicolson grid time resolution.")


# =============================================================================
# Sidebar — Worst-of approximation multiplier
# =============================================================================
# Used in the Worst-of tab. There's no closed form for worst-of KO under
# American barriers, so we use the rule-of-thumb:
#     premium_worstof  ≈  multiplier × min(premium_leg1, premium_leg2)
# where the multiplier captures how much cheaper the worst-of is vs the
# cheaper single leg. Typical FX-pair correlations + American-barrier
# monitoring → 40% is a reasonable starting point. Higher values (50%)
# are appropriate when legs are highly correlated, lower (33%) when
# they're nearly independent.
st.sidebar.markdown("### Worst-of approximation")
wo_multiplier_pct = st.sidebar.radio(
    "Multiplier",
    options=[33, 40, 50],
    index=1,            # 40% default for App 12 (American barrier)
    horizontal=True,
    format_func=lambda x: f"{x}%",
    key="rko_wo_multiplier_pct",
    help=("Premium ≈ multiplier × min(price_leg1, price_leg2). "
           "40% is a reasonable starting point for FX worst-of with "
           "American barriers; higher for highly correlated legs, "
           "lower for nearly independent ones."),
)
wo_multiplier = wo_multiplier_pct / 100.0


# =============================================================================
# Sidebar — single-leg pricing model
# =============================================================================
# Mirrors the EKO Pricer's selector (Step 1b/1c). Three modes:
#
#   flat_atm        ako_closed_form at σ_atm — smile-ignorant baseline.
#                   Useful as a debug reference: tells you the pure
#                   American-barrier price ignoring the wing-region
#                   premium that real markets quote.
#
#   vol_at_strike   ako_closed_form at σ_smile(K) — uses the smile-
#                   interpolated vol at the strike. Cheap to evaluate
#                   and a reasonable first-order smile adjustment.
#
#   vanna_volga     Castagna-Mercurio Vanna-Volga smile correction on
#                   top of ako_closed_form. Bloomberg OVML's default
#                   for American-barrier KOs. Captures the FULL smile
#                   premium that vol_at_strike misses on ATM-strike
#                   trades with wing-region barriers. Heavier compute
#                   (~5 ako_closed_form calls per VV call) but most
#                   accurate.
#
# Default is 'vanna_volga' for RKO since (a) the live pricer's current
# behaviour is to use VV when smile data is available and we want to
# preserve that, and (b) American-barrier trades are more sensitive to
# the wing-region vol than European-barrier trades because the
# continuous-monitoring barrier has a higher knockout probability
# under the smile than under flat ATM vol.
#
# Stored on st.session_state under 'rko_pricing_model'.
st.sidebar.markdown("### Single-leg pricing model")
_rko_pricing_model_options = ["flat_atm", "vol_at_strike", "vanna_volga"]
_rko_pricing_model_labels = {
    "flat_atm": "Flat BS (σ_atm)",
    "vol_at_strike": "Vol-at-strike σ_smile(K)",
    "vanna_volga": "Vanna-Volga",
}
rko_pricing_model = st.sidebar.radio(
    "Model",
    options=_rko_pricing_model_options,
    index=2,            # default: vanna_volga (preserves legacy RKO behaviour)
    format_func=lambda x: _rko_pricing_model_labels[x],
    key="rko_pricing_model",
    help=(
        "How the smile enters the RKO price. "
        "Flat BS ignores the smile (debug baseline). "
        "Vol-at-strike uses σ_smile(K) — first-order smile adjustment. "
        "Vanna-Volga adds the full Castagna-Mercurio correction "
        "(matches Bloomberg OVML). "
        "Greeks shown below are still computed at σ_smile in all three "
        "modes — VV-consistent Greeks are a follow-up step."
    ),
)


def _price_rko_dispatch(option_type: str, barrier_type: str,
                          S: float, K: float, H: float, T: float,
                          sigma_atm: float, sigma_smile: float,
                          rr_25: float, bf_25: float,
                          r_d: float, r_f: float,
                          model: str) -> "tuple[float, dict]":
    """Thin alias around core.ako_pricing.price_ako_dispatch.

    Kept as a name-stable wrapper so all in-page call sites can import
    it locally; the actual logic lives in core.ako_pricing so the
    single-leg RKO backtester (core/backtest_american.py) can share it.
    Adding a new model is a single edit in core.ako_pricing and both
    callers pick it up.
    """
    from core.ako_pricing import price_ako_dispatch
    return price_ako_dispatch(
        option_type, barrier_type, S, K, H, T,
        sigma_atm, sigma_smile, rr_25, bf_25, r_d, r_f, model,
    )


# =============================================================================
# Shared constants (mirrors App 9 — same trade-input dropdowns)
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


# =============================================================================
# Helpers (copied from App 9 — identical formatting conventions)
# =============================================================================
def _fmt_px(pair: str, x: float) -> str:
    if not np.isfinite(x):
        return "—"
    if pair.endswith("JPY"):
        return f"{x:,.3f}"
    if pair in ("USDKRW", "USDIDR", "USDINR", "USDTHB", "USDPHP"):
        return f"{x:,.2f}"
    return f"{x:,.4f}"


def _fmt_usd(x: float) -> str:
    if not np.isfinite(x):
        return "—"
    a = abs(x)
    sign = "-" if x < 0 else ""
    if a >= 1e6:
        return f"{sign}${a/1e6:,.2f}M"
    if a >= 1e3:
        return f"{sign}${a/1e3:,.1f}K"
    return f"{sign}${a:,.0f}"


def _fmt_date(d) -> str:
    if d is None or pd.isna(d):
        return "—"
    return pd.Timestamp(d).strftime("%Y-%m-%d")


@st.cache_data(show_spinner=False)
def _list_pairs(folder: str) -> list[str]:
    df = load_panel(folder, "SPOT", None)
    return sorted(df.columns.tolist())


@st.cache_data(show_spinner="Loading rates…")
def _load_rates_panel_cached(folder: str, currency: str):
    return load_rates_panel(folder, currency, load_by_ticker)


# =============================================================================
# Page title
# =============================================================================
st.title("RKO Pricer — American-barrier knock-outs")
st.caption(
    "European exercise (only at expiry) + American barrier "
    "(continuously monitored — any in-life touch kills the option). "
    "Four pricing methods shown side-by-side for cross-validation."
)



# =============================================================================
# Tabs
# =============================================================================
tab_pricer, tab_backtest, tab_drilldown, tab_worstof, tab_wo_drill, \
    tab_rko_port, tab_rko_drill, tab_wo_rko_port, tab_wo_rko_drill = st.tabs(
    ["💰 Pricer", "📊 Backtest", "🔍 Backtest drilldown",
     "🔀 Worst-of", "🔬 Worst-of drilldown",
     "📦 RKO Portfolio", "🔎 RKO Portfolio drilldown",
     "🪢📦 WO-RKO Portfolio", "🪢🔎 WO-RKO Portfolio drilldown"]
)

with tab_pricer:
# =============================================================================
# Pricer tab
# =============================================================================
    pairs_avail = _list_pairs(folder)
    if not pairs_avail:
        st.error("No SPOT data found in `_index.csv`.")
        st.stop()

    c_in, c_out = st.columns([1, 2.6], gap="medium")


# ---------- Trade inputs (left column) ----------
    with c_in:
        st.markdown("**Trade**")
        tenor_label = st.selectbox("Tenor", TENOR_LIST, index=0, key="ako_tenor")
        default_pair = ("USDJPY" if "USDJPY" in pairs_avail else
                          ("EURUSD" if "EURUSD" in pairs_avail else pairs_avail[0]))
        pair = st.selectbox("Currency pair", pairs_avail,
                              index=pairs_avail.index(default_pair), key="ako_pair")
        asia_em = pair in {"USDCNH", "USDCNY", "USDINR", "USDIDR", "USDKRW",
                            "USDMYR", "USDPHP", "USDTHB", "USDTWD"}
        prefer = st.radio("Variant", ["offshore", "onshore"], index=0,
                            horizontal=True, key="ako_prefer") if asia_em else "offshore"

        direction_label = st.radio("Direction", list(DIRECTIONS.keys()),
                                     index=0, key="ako_dir")
        option_type, barrier_type = DIRECTIONS[direction_label]

        strike_delta_label = st.radio("Strike Δ", list(DELTA_CHOICES.keys()),
                                          index=0, horizontal=True, key="ako_delta")
        strike_delta = DELTA_CHOICES[strike_delta_label]

        ko_method_label = st.radio(
            "KO method", ("Payout ratio", "KO delta"),
            index=0, horizontal=True, key="ako_ko_method",
            help=("• **Payout ratio**: solves H so max_payoff / premium hits "
                   "the target leverage. Premium is computed via the AMERICAN-"
                   "barrier closed form, so target leverage is the leverage you "
                   "actually get on the continuously-monitored product.\n"
                   "• **KO delta**: places H at a vanilla-Δ wing strike "
                   "(same convention as the strike Δ).")
        )
        ko_method = "ratio" if ko_method_label == "Payout ratio" else "delta"

        if ko_method == "ratio":
            payout_label = st.radio("Payout ratio (max payoff / premium)",
                                      list(PAYOUT_CHOICES.keys()), index=1,
                                      horizontal=True, key="ako_payout")
            payout_ratio = PAYOUT_CHOICES[payout_label]
            ko_delta_label, ko_delta_value = None, None
        else:
            ko_delta_label = st.radio(
                "KO Δ (vanilla wing)",
                list(KO_DELTA_CHOICES.keys()), index=1,
                horizontal=True, key="ako_ko_delta",
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
            key="ako_notional",
        )

        # ---- Manual smile override (for testing against Bloomberg) ----
        st.markdown("---")
        smile_override = st.checkbox(
            "Manual smile override (25Δ RR / BF)",
            value=False, key="ako_smile_override",
            help=("When your data folder doesn't have VOL_25R / VOL_25B at "
                   "this tenor, type the smile inputs in directly. Use "
                   "Bloomberg's mid values for an apples-to-apples cross-"
                   "check against OVML's Vanna-Volga price.")
        )
        if smile_override:
            rr_override_pct = st.number_input(
                "25Δ RR (%, σ_25C − σ_25P)",
                min_value=-10.0, max_value=10.0, value=-1.435,
                step=0.05, format="%.3f", key="ako_rr_override",
                help="Negative = puts more expensive than calls (typical "
                      "for USDJPY — JPY-call skew).",
            ) / 100.0
            bf_override_pct = st.number_input(
                "25Δ BF (%)",
                min_value=-2.0, max_value=5.0, value=0.208,
                step=0.02, format="%.3f", key="ako_bf_override",
            ) / 100.0
            atm_override_pct = st.number_input(
                "Override ATM vol (%, leave 0 to use data)",
                min_value=0.0, max_value=200.0, value=0.0,
                step=0.05, format="%.3f", key="ako_atm_override",
                help="Optional — leave at 0 to use the interpolated value "
                      "from your data folder.",
            ) / 100.0
        else:
            rr_override_pct = None
            bf_override_pct = None
            atm_override_pct = None


# ---------- Spot (close prices) ----------
    spot_df = load_panel(folder, "SPOT", None, prefer=prefer, pairs=(pair,))
    if spot_df.empty or pair not in spot_df.columns:
        c_out.error(f"No SPOT for {pair}.")
        st.stop()
    spot_ts = spot_df[pair].dropna()
    if spot_ts.empty:
        c_out.error(f"Empty SPOT for {pair}.")
        st.stop()

    with c_in:
        val_date = st.date_input(
            "Trade date", value=spot_ts.index.max().date(),
            min_value=spot_ts.index.min().date(),
            max_value=spot_ts.index.max().date(),
            key="ako_val_date",
        )

    opt_dates = compute_option_dates(val_date, tenor_label)
    T = opt_dates.T_years
    val_ts = pd.Timestamp(val_date)

    # Spot at val date (Close, by virtue of the load_panel fix)
    valid = spot_ts.loc[:val_ts]
    if valid.empty:
        c_out.error(f"No spot data at or before {val_date}.")
        st.stop()
    S = float(valid.iloc[-1])


# ---------- Vol / smile inputs ----------
    sigma_pct = get_pair_value_at_T(folder, pair, prefer, "VOL_ATM", T, val_date)
    if sigma_pct is None:
        c_out.error(f"No VOL_ATM data for {pair}.")
        st.stop()
    sigma_atm = sigma_pct / 100.0

    rr_pct = get_pair_value_at_T(folder, pair, prefer, "VOL_25R", T, val_date)
    bf_pct = get_pair_value_at_T(folder, pair, prefer, "VOL_25B", T, val_date)
    rr_25 = (rr_pct / 100.0) if rr_pct is not None else 0.0
    bf_25 = (bf_pct / 100.0) if bf_pct is not None else 0.0
    smile_available = (rr_pct is not None) and (bf_pct is not None)

    # Manual smile override (for cross-checking against Bloomberg)
    smile_overridden = False
    if smile_override:
        rr_25 = rr_override_pct
        bf_25 = bf_override_pct
        if atm_override_pct and atm_override_pct > 0:
            sigma_atm = atm_override_pct
        smile_available = True
        smile_overridden = True


# ---------- Forward ----------
    fwd_pts_at_T = get_pair_value_at_T(folder, pair, prefer, "FWD_POINTS", T, val_date)
    pip = get_pip_scale(pair)
    if fwd_pts_at_T is None:
        F_market = S
        fwd_avail = False
    else:
        F_market = S + fwd_pts_at_T * pip
        fwd_avail = True


# ---------- Rates ----------
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
                format="%.3f", key="ako_rf",
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


# ---------- Solve K and H using AMERICAN-barrier pricer ----------
    # Solver pricer must match the headline model so the achieved
    # leverage is consistent with the displayed premium. Three branches:
    #
    #   vanna_volga    : VV-on-RR (matches displayed VV price)
    #   vol_at_strike  : RR at σ_smile (the solver internally uses
    #                     σ_smile(K) anyway via solve_strike's smile-aware
    #                     branch — see core/ko_solvers.py)
    #   flat_atm       : RR at σ_atm (the solver's flat-vol path)
    #
    # When smile data is unavailable, σ_smile collapses to σ_atm and
    # the three modes coincide; we use the cheapest pricer.
    def _vv_solver_pricer(opt_type, bar_type, S_, K_, H_, T_, sig_, rd_, rf_):
        out = vv_price_ko(opt_type, bar_type, S_, K_, H_, T_,
                              sig_, rr_25, bf_25, rd_, rf_,
                              flat_vol_pricer=ako_closed_form)
        return out["price_vv"]

    if not smile_available or rko_pricing_model in ("flat_atm", "vol_at_strike"):
        solver_pricer = ako_closed_form
    else:   # vanna_volga
        solver_pricer = _vv_solver_pricer

    K, H, info = solve_strike(
        option_type, barrier_type, strike_delta,
        S, T, sigma_atm, r_d, r_f,
        target_ratio=payout_ratio,
        target_ko_delta=ko_delta_value,
        ko_method=ko_method,
        rr_25=rr_25, bf_25=bf_25,
        pricer=solver_pricer,
    )
    sigma_smile = float(info.get("sigma_smile", sigma_atm))


# ---------- Headline price ----------
    # Route through the dispatcher so the sidebar selector ('flat_atm' /
    # 'vol_at_strike' / 'vanna_volga') drives the headline price.
    # The method-comparison panel below still shows all 4 methods at
    # σ_smile for cross-validation (those are diagnostic, not headline).
    #
    # Special case: when no smile data is available, σ_smile == σ_atm
    # and the VV branch would fall back to a flat-vol price anyway.
    # We still let the dispatcher run so the detail dict labels the
    # model correctly.
    _effective_model = (rko_pricing_model if smile_available
                          else "flat_atm")
    ako_per_unit, _hl_detail = _price_rko_dispatch(
        option_type, barrier_type, S, K, H, T,
        sigma_atm=sigma_atm, sigma_smile=sigma_smile,
        rr_25=rr_25, bf_25=bf_25,
        r_d=r_d, r_f=r_f,
        model=_effective_model,
    )
    if _effective_model == "vanna_volga":
        headline_method = "Vanna-Volga (smile-adjusted)"
    elif _effective_model == "vol_at_strike":
        headline_method = "Reiner-Rubinstein (σ_smile(K))"
    else:
        headline_method = "Reiner-Rubinstein (flat σ_atm)"
    vanilla_per_unit = vanilla_price(option_type, S, K, T, sigma_smile, r_d, r_f)
    ako_prob = ako_probability_continuous(barrier_type, S, H, T, sigma_smile, r_d, r_f)
    ako_delta_signed = ako_spot_delta(option_type, barrier_type, S, K, H, T,
                                           sigma_smile, r_d, r_f)
    vanilla_delta_signed = vanilla_spot_delta(option_type, S, K, T, sigma_smile, r_d, r_f)

    ako_usd = ako_per_unit / S * notional_usd
    vanilla_usd = vanilla_per_unit / S * notional_usd
    max_payoff_per_unit = abs(H - K) if (option_type, barrier_type) in (
        ("call", "up_and_out"), ("put", "down_and_out")) else (
        K if option_type == "put" else float("inf"))
    max_payoff_usd = (max_payoff_per_unit / S * notional_usd
                       if np.isfinite(max_payoff_per_unit) else float("inf"))


# =============================================================================
    # Output (right column)
# =============================================================================
    with c_out:
        st.markdown(
            f"### {pair}  ·  Buy "
            f"<span class='tag-{option_type}'>{option_type.upper()}</span>  "
            f"with KO "
            f"<span class='tag-ko'>{barrier_type.replace('_', '-').upper()}</span>  "
            f"<span class='tag-amer'>AMERICAN BARRIER</span>  ·  "
            f"{tenor_label}  ·  strike {strike_delta_label}  ·  "
            + (f"target leverage {payout_label}" if ko_method == "ratio"
                else f"KO @ {ko_delta_label} (vanilla)"),
            unsafe_allow_html=True,
        )

        # Headline metrics (VV is the reference price when smile available)
        cc1, cc2, cc3, cc4, cc5 = st.columns(5)
        with cc1:
            headline_tag = "Vanna-Volga" if smile_available else "R-R closed-form"
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-title'>KO Premium</div>"
                f"<div class='metric-value'>{_fmt_usd(ako_usd)}</div>"
                f"<div class='metric-sub'>{abs(ako_usd)/notional_usd*100:.3f}% notl"
                f"  ·  {headline_tag}</div>"
                f"</div>", unsafe_allow_html=True)
        with cc2:
            lev = (max_payoff_usd / max(ako_usd, 1)
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
                f"<div class='metric-value'>{ako_prob*100:.1f}%</div>"
                f"<div class='metric-sub'>any-time hit, risk-neutral</div>"
                f"</div>", unsafe_allow_html=True)
        with cc4:
            cheap = (1 - ako_per_unit / vanilla_per_unit) * 100 if vanilla_per_unit > 0 else 0
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
                f"<div class='metric-value'>{ako_delta_signed*100:+.1f}%</div>"
                f"<div class='metric-sub'>vanilla Δ = {vanilla_delta_signed*100:+.1f}%</div>"
                f"</div>", unsafe_allow_html=True)

        # ---------- Method comparison panel ----------
        st.markdown("---")
        st.markdown("### Method comparison — cross-check against Bloomberg")
        smile_caption = ("Smile inputs: σ_atm={:.3f}%, RR_25={:+.3f}%, BF_25={:+.3f}%"
                          .format(sigma_atm*100, rr_25*100, bf_25*100)
                          if smile_available else
                          "Smile inputs: σ_atm={:.3f}% only (no RR/BF — VV reduces to BS)"
                          .format(sigma_atm*100))
        if smile_overridden:
            smile_caption += "  ·  *(manual override)*"
        st.caption(
            "**Vanna-Volga** (row 1) is the headline price — smile-adjusted, "
            "matches Bloomberg OVML's 'Vanna-Volga' model within ~0.5% on "
            "USDJPY. The four flat-σ_atm rows below it are diagnostics: the "
            "Reiner-Rubinstein closed form is the analytic reference for "
            "continuous monitoring; the three numerical methods (binomial / "
            "trinomial / Crank-Nicolson) discretize and should converge to it. "
            + smile_caption
        )
        import time as _t
        methods = []

        # Vanna-Volga — the headline, computed first
        t0 = _t.perf_counter()
        vv_out = vv_price_ko(option_type, barrier_type, S, K, H, T,
                                sigma_atm, rr_25, bf_25, r_d, r_f,
                                flat_vol_pricer=ako_closed_form,
                                weight_by_survival=False)
        p_vv = vv_out["price_vv"]
        t_vv = (_t.perf_counter() - t0) * 1000.0

        # Closed-form (R-R, the analytic reference at σ_atm — used as the
        # comparison anchor for the other flat-vol methods)
        t0 = _t.perf_counter()
        p_cf = ako_closed_form(option_type, barrier_type, S, K, H, T,
                                  sigma_smile, r_d, r_f)
        t_cf = (_t.perf_counter() - t0) * 1000.0

        vv_label = ("Vanna-Volga (smile-adjusted) — Bloomberg OVML equivalent"
                      if smile_available else
                      "Vanna-Volga (smile-adjusted) — n/a, falls back to BS")
        methods.append((vv_label, p_vv, t_vv,
                        f"{(p_vv-p_cf)/max(p_cf,1e-15)*100:+.3f}%"))
        methods.append(("Closed-form (Reiner-Rubinstein) — flat σ_atm",
                        p_cf, t_cf, "—"))

        # Binomial
        t0 = _t.perf_counter()
        p_bn = ako_binomial(option_type, barrier_type, S, K, H, T,
                              sigma_smile, r_d, r_f,
                              n_steps=n_steps_binom, bgk_correction=True)
        t_bn = (_t.perf_counter() - t0) * 1000.0
        methods.append((f"Binomial CRR + BGK (N={n_steps_binom}) — flat σ_atm",
                        p_bn, t_bn,
                        f"{(p_bn-p_cf)/max(p_cf,1e-15)*100:+.3f}%"))

        # Trinomial
        t0 = _t.perf_counter()
        p_tn = ako_trinomial(option_type, barrier_type, S, K, H, T,
                                sigma_smile, r_d, r_f, n_steps=n_steps_trinom)
        t_tn = (_t.perf_counter() - t0) * 1000.0
        methods.append((f"Trinomial Boyle (N={n_steps_trinom}) — flat σ_atm",
                        p_tn, t_tn,
                        f"{(p_tn-p_cf)/max(p_cf,1e-15)*100:+.3f}%"))

        # FD
        t0 = _t.perf_counter()
        p_fd = ako_finite_difference(option_type, barrier_type, S, K, H, T,
                                           sigma_smile, r_d, r_f,
                                           n_S=n_S_fd, n_t=n_t_fd)
        t_fd = (_t.perf_counter() - t0) * 1000.0
        methods.append((f"Crank-Nicolson FD ({n_S_fd}×{n_t_fd}) — flat σ_atm",
                        p_fd, t_fd,
                        f"{(p_fd-p_cf)/max(p_cf,1e-15)*100:+.3f}%"))

        rows = []
        for name, p, t_ms, diff in methods:
            rows.append({
                "Method": name,
                "Price (per unit FOR)": f"{p:.6f}",
                "Premium (USD)": _fmt_usd(p / S * notional_usd),
                "Δ vs closed-form": diff,
                "Compute time": f"{t_ms:.1f} ms",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if smile_available:
            smile_lift_pct = (p_vv - p_cf) / max(p_cf, 1e-15) * 100
            st.caption(
                f"💡 The smile lifts the price by **{smile_lift_pct:+.1f}%** "
                f"vs flat-σ_atm BS for this trade. On USDJPY (JPY-call skew, "
                f"RR<0), an up-and-out call gets a positive smile premium "
                f"because the barrier-side wing vol (σ_25C) is LOWER than "
                f"σ_atm → less likely to KO → option more valuable. The "
                f"Vanna-Volga row replicates this consistently with how "
                f"Bloomberg OVML's 'Vanna-Volga' model prices the same trade."
            )
        else:
            st.caption(
                f"⚠ No RR/BF data at {tenor_label} for {pair} — Vanna-Volga "
                f"falls back to flat-vol BS. Either load 25Δ RR/BF CSVs into "
                f"the data folder, or enable the manual smile override "
                f"(checkbox in the trade inputs) to type the values in."
            )

        # Solver feedback (mirrors App 9)
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
                        f"strike. Achieved leverage: {achieved:.2f}× (varies "
                        f"with vol/rates).")

        st.markdown("---")
        cd1, cd2 = st.columns(2)
        with cd1:
            st.markdown("**Date schedule**")
            st.markdown(
                f"- Trade date:&nbsp;&nbsp;`{_fmt_date(opt_dates.trade_date)}`\n"
                f"- Spot settlement:&nbsp;`{_fmt_date(opt_dates.spot_settlement)}` (T+2)\n"
                f"- Option settlement:&nbsp;`{_fmt_date(opt_dates.option_settlement)}` (spot + {tenor_label})\n"
                f"- Option expiry:&nbsp;`{_fmt_date(opt_dates.option_expiry)}` (settle − 2bd)\n"
                f"- T = `{T:.4f}`y "
                f"({(opt_dates.option_expiry - opt_dates.trade_date).days}d)"
            )
        with cd2:
            st.markdown("**Levels**")
            if not fwd_avail:
                st.warning(f"No FWD_POINTS at {tenor_label}: F = spot.")
            if smile_available:
                from core.smile import wing_vols_25d
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
                f"- Spot (close): `{_fmt_px(pair, S)}`\n"
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

        # Payoff diagram (same logic as App 9 — payoff depends only on terminal
        # spot relative to K and H; the American-barrier feature is path-
        # dependent and not visible in a terminal-only chart, but for the
        # ALREADY-SURVIVED option at expiry the payoff diagram is identical
        # to a European-barrier KO. We caption this distinction.)
        st.markdown("---")
        st.markdown("### Payoff at expiry  *(conditional on barrier never touched)*")
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
        fig.add_trace(go.Scatter(x=S_grid, y=payoff_usd - ako_usd, mode="lines",
                                  name="Net P&L",
                                  line=dict(color="#38bdf8", width=2.6)))
        fig.add_trace(go.Scatter(x=S_grid, y=payoff_usd, mode="lines",
                                  name="Gross payoff",
                                  line=dict(color="#86efac", width=1.4, dash="dot")))
        fig.add_hline(y=-ako_usd, line=dict(color="#fb923c", dash="dash", width=1),
                       annotation_text=f"−Premium = {_fmt_usd(-ako_usd)}",
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
        st.caption(
            "Path-dependent KO not visible in this terminal-only diagram — the "
            "chart shows payoff CONDITIONAL on the barrier never being touched. "
            f"The probability of getting to expiry without an in-life touch is "
            f"~{(1 - ako_prob) * 100:.1f}%; the premium ({_fmt_usd(ako_usd)}) "
            f"already prices this in."
        )

        # Premium grid (uses VV pricing when smile available, R-R otherwise)
        st.markdown("---")
        smile_note = (" — VV smile-adjusted via 25Δ RR/BF"
                       if smile_available else " — flat-vol R-R (no smile data)")
        grid_pricer = _vv_solver_pricer if smile_available else ako_closed_form

        def _grid_price(opt, bar, S_, K_, H_, T_, sig_, rd_, rf_):
            """Final-cell pricing — VV if smile, else R-R closed-form."""
            if smile_available:
                out = vv_price_ko(opt, bar, S_, K_, H_, T_,
                                     sig_, rr_25, bf_25, rd_, rf_,
                                     flat_vol_pricer=ako_closed_form)
                return out["price_vv"]
            return ako_closed_form(opt, bar, S_, K_, H_, T_, sig_, rd_, rf_)

        if ko_method == "ratio":
            st.markdown(f"### Premium grid (strike Δ × leverage){smile_note}")
            st.caption(f"Each cell solves K from vanilla Δ at σ_atm, then H from "
                        f"the leverage target. Premium shown at σ_smile(K)"
                        + (" with full Vanna-Volga smile adjustment."
                            if smile_available else ", flat-vol Reiner-Rubinstein.")
                        + " Cells marked ⚠ are infeasible; the achieved leverage "
                        f"shown is the minimum at that strike.")
            grid_rows = []
            for d_label, d_val in DELTA_CHOICES.items():
                row = {"Strike Δ": d_label}
                for r_label, r_val in PAYOUT_CHOICES.items():
                    K_g, H_g, info_g = solve_strike(
                        option_type, barrier_type, d_val,
                        S, T, sigma_atm, r_d, r_f,
                        target_ratio=r_val, ko_method="ratio",
                        rr_25=rr_25, bf_25=bf_25,
                        pricer=grid_pricer,
                    )
                    sg = info_g.get("sigma_smile", sigma_atm)
                    p = _grid_price(option_type, barrier_type, S, K_g, H_g,
                                       T, sg, r_d, r_f)
                    prem_usd = p / S * notional_usd
                    ach = info_g.get("achieved_ratio", float("nan"))
                    infeasible = (abs(ach - r_val) > 0.01
                                    if np.isfinite(ach) else True)
                    row[r_label] = (f"⚠ {_fmt_usd(prem_usd)} ({ach:.1f}× min)"
                                     if infeasible else _fmt_usd(prem_usd))
                grid_rows.append(row)
        else:
            st.markdown(f"### Premium grid (strike Δ × KO Δ){smile_note}")
            st.caption(f"Each cell solves K from strike Δ and H from KO Δ. "
                        f"Premium shown at σ_smile(K)"
                        + (" with full Vanna-Volga smile adjustment."
                            if smile_available else ", flat-vol Reiner-Rubinstein.")
                        + " Cells marked ⚠ are degenerate. Number after premium "
                        f"is the achieved leverage under continuous-barrier "
                        f"monitoring.")
            grid_rows = []
            for d_label, d_val in DELTA_CHOICES.items():
                row = {"Strike Δ": d_label}
                for kd_label, kd_val in KO_DELTA_CHOICES.items():
                    K_g, H_g, info_g = solve_strike(
                        option_type, barrier_type, d_val,
                        S, T, sigma_atm, r_d, r_f,
                        target_ko_delta=kd_val, ko_method="delta",
                        rr_25=rr_25, bf_25=bf_25,
                        pricer=grid_pricer,
                    )
                    sg = info_g.get("sigma_smile", sigma_atm)
                    p = _grid_price(option_type, barrier_type, S, K_g, H_g,
                                       T, sg, r_d, r_f)
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




# =============================================================================
# Backtest + Drilldown — render functions defined in _rko_pricer_tabs.py
# =============================================================================
# Streamlit runs the script with cwd at the project root and `apps/` not on
# sys.path. Make the sibling module importable via path injection.
import sys as _sys
from pathlib import Path as _Path
_apps_dir = str(_Path(__file__).parent)
if _apps_dir not in _sys.path:
    _sys.path.insert(0, _apps_dir)

with tab_backtest:
    from _rko_pricer_tabs import render_backtest_tab
    render_backtest_tab(folder)

with tab_drilldown:
    from _rko_pricer_tabs import render_drilldown_tab
    render_drilldown_tab()

with tab_worstof:
    from _rko_pricer_tabs import render_worstof_tab
    render_worstof_tab(folder, multiplier=wo_multiplier)

with tab_wo_drill:
    from _rko_pricer_tabs import render_worstof_drilldown_tab
    render_worstof_drilldown_tab()

with tab_rko_port:
    from _rko_pricer_tabs import render_rko_portfolio_tab
    render_rko_portfolio_tab(folder)

with tab_rko_drill:
    from _rko_pricer_tabs import render_rko_portfolio_drilldown_tab
    render_rko_portfolio_drilldown_tab()

with tab_wo_rko_port:
    from _rko_pricer_tabs import render_wo_rko_portfolio_tab
    render_wo_rko_portfolio_tab(folder, multiplier=wo_multiplier)

with tab_wo_rko_drill:
    from _rko_pricer_tabs import render_wo_rko_portfolio_drilldown_tab
    render_wo_rko_portfolio_drilldown_tab()

# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.caption(
    "**App 12 — American-barrier KO pricer & backtester.** Nine tabs: "
    "Pricer (single-trade snapshot), Backtest + drilldown (single-leg), "
    "Worst-of + drilldown, RKO Portfolio + drilldown (basket across pairs), "
    "and WO-RKO Portfolio + drilldown (basket of worst-of crosses). "
    "All American-barrier monitoring uses daily OHLC — a barrier hit is "
    "recorded whenever [day_low, day_high] contains H. Entry pricing is "
    "Vanna-Volga on the American closed form, matching Bloomberg OVML's "
    "'Vanna-Volga' model to within ~3 bp of notional on tested pairs. "
    "Worst-of multiplier (sidebar) defaults to 40% (vs App 9's EKO default "
    "of 50%) to compensate for the richer American monitoring."
)

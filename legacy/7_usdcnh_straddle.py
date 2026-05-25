"""App 7 — USDCNH 25Δ strangle vol calendar.

Run with:
    streamlit run apps/7_usdcnh_straddle.py

Trigger:
    Open new structure when 1Y USDCNH ATM vol percentile within trailing
    3Y window falls below `trigger_pct` (default 0% — multi-year low).

Structure (one at a time):
    - Long 1Y 25Δ strangle (USDCNH offshore) — separate call + put legs
    - Short 1M 25Δ strangle, rolled monthly — separate call + put legs
    - Last 1M roll capped at 1Y expiry so all 4 legs finish together
    - Daily delta hedge via spot
    - User-specified take-profit and stop-loss on cumulative net P&L

Sizing:
    Same USD notional applied to every leg (1Y call, 1Y put,
    every 1M short call/put roll).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from core.ts_loader import load_panel
from core.ui import data_dir_input
from core.conventions import get_pip_scale
from core.strategy_engine_7 import (StrategyConfig7, run_strategy_7,
                                     summary_stats_7)
from core.strategy_dashboard_7 import render_dashboard_7
from core.strategy_dashboard import inject_dashboard_css


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="7 · USDCNH Vol Calendar",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_dashboard_css()

st.title("7 · USDCNH  ·  Long 1Y / short rolling 1M  25Δ strangle")
st.caption(
    "Trigger: **1Y USDCNH ATM vol at 3Y multi-year low**  ·  "
    "25Δ strangles (separate call & put legs)  ·  daily delta hedge  ·  "
    "monthly 1M roll  ·  one structure at a time"
)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
folder = data_dir_input()
if folder is None:
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("**Trigger**")
trigger_pct = st.sidebar.number_input(
    "Trigger pct (≤)", min_value=0.0, max_value=50.0, value=0.0, step=1.0,
    help="Open new structure when 1Y vol percentile rank within trailing "
         "3Y window is at-or-below this. Default 0 = at the 3-year low.",
)
trigger_lookback_years = st.sidebar.slider(
    "Lookback (years)", min_value=1, max_value=10, value=3,
)

st.sidebar.markdown("**Sizing**")
notional_usd = st.sidebar.number_input(
    "Notional per leg (USD)", min_value=1_000_000.0, max_value=200_000_000.0,
    value=10_000_000.0, step=1_000_000.0, format="%.0f",
    help="Same USD notional applied to every leg: 1Y call, 1Y put, "
         "and every 1M short call/put roll.",
)

st.sidebar.markdown("**Risk management**")
use_tp = st.sidebar.checkbox("Take profit", value=False)
take_profit_usd = (
    st.sidebar.number_input(
        "TP (USD)", min_value=10_000.0, max_value=10_000_000.0,
        value=500_000.0, step=50_000.0, format="%.0f")
    if use_tp else None
)
use_stop = st.sidebar.checkbox("Stop loss", value=False)
stop_loss_usd = (
    st.sidebar.number_input(
        "Stop (USD)", min_value=10_000.0, max_value=10_000_000.0,
        value=500_000.0, step=50_000.0, format="%.0f")
    if use_stop else None
)
st.sidebar.markdown("**Hedging**")
hedge_mode = st.sidebar.selectbox(
    "Hedge mode",
    options=["daily", "threshold", "spot_move", "none"],
    index=0,
    help=(
        "**daily**: rebalance every leg back to its −delta every day.  "
        "**threshold**: rebalance only when |structure delta| > X% of notional, "
        "bringing residual back to Y%.  "
        "**spot_move**: rebalance a leg only when spot has crossed into a new "
        "k×Z% band relative to that leg's entry spot.  "
        "**none**: run fully unhedged."
    ),
)

# Mode-specific params (only the relevant ones are shown)
if hedge_mode == "threshold":
    threshold_trigger_pct = st.sidebar.slider(
        "X — trigger threshold (% of notional)",
        min_value=0.5, max_value=20.0, value=5.0, step=0.5,
        help="Hedge fires when |structure delta| / notional exceeds this.",
    )
    threshold_target_pct = st.sidebar.slider(
        "Y — post-hedge target (% of notional)",
        min_value=0.0, max_value=10.0, value=1.0, step=0.5,
        help="Residual is brought back to this magnitude (sign preserved).",
    )
    spot_move_pct = 1.0
else:
    threshold_trigger_pct = 5.0
    threshold_target_pct = 1.0

if hedge_mode == "spot_move":
    spot_move_pct = st.sidebar.slider(
        "Z — band width (% spot move from entry)",
        min_value=0.1, max_value=5.0, value=1.0, step=0.1,
        help="A leg is hedged whenever spot crosses a new k·Z% band from its entry.",
    )
elif hedge_mode != "threshold":
    spot_move_pct = 1.0

st.sidebar.markdown("**Costs / pricing**")
vol_bid_ask_pts = st.sidebar.slider(
    "Vol bid-ask (vol pts)",
    min_value=0.05, max_value=1.0, value=0.25, step=0.05,
    help="Spread between bid and offer in vol points (e.g. 0.25 = 25 bp of vol)."
)
spot_cost_bps = st.sidebar.slider(
    "Spot cost (bps)", min_value=0.0, max_value=10.0, value=1.0, step=0.5,
)
usd_rate_pct = st.sidebar.slider(
    "USD funding (%)", min_value=0.0, max_value=8.0, value=3.0, step=0.25,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Variant**")
variant = st.sidebar.radio(
    "Onshore/Offshore", ["offshore", "onshore"], index=0, horizontal=True,
)


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading USDCNH market data…")
def load_usdcnh(folder: str, prefer: str
                ) -> dict[str, pd.Series]:
    """Returns dict with keys: spot, vol_1m, vol_1y, forward_1y."""
    pair = "USDCNH"
    out: dict[str, pd.Series] = {}

    spot_df = load_panel(folder, "SPOT", tenor=None, prefer=prefer, pairs=(pair,))
    if not spot_df.empty and pair in spot_df.columns:
        out["spot"] = spot_df[pair].dropna()

    v1m_df = load_panel(folder, "VOL_ATM", tenor="1M", prefer=prefer, pairs=(pair,))
    if not v1m_df.empty and pair in v1m_df.columns:
        out["vol_1m"] = v1m_df[pair].dropna()

    v1y_df = load_panel(folder, "VOL_ATM", tenor="1Y", prefer=prefer, pairs=(pair,))
    if not v1y_df.empty and pair in v1y_df.columns:
        out["vol_1y"] = v1y_df[pair].dropna()

    fwd_df = load_panel(folder, "FWD_POINTS", tenor="12M", prefer=prefer, pairs=(pair,))
    if not fwd_df.empty and pair in fwd_df.columns:
        out["fwd_pts_1y"] = fwd_df[pair].dropna()

    return out


data = load_usdcnh(folder, variant)

required = {"spot", "vol_1m", "vol_1y", "fwd_pts_1y"}
missing = required - set(data.keys())
if missing:
    st.error(
        "Missing USDCNH data in `_index.csv`: " + ", ".join(sorted(missing)) + ". "
        f"Need rows with pair=USDCNH, onshore_offshore={variant.upper()}, and "
        "categories SPOT / VOL_ATM (tenor 1M and 1Y) / FWD_POINTS (tenor 12M)."
    )
    st.stop()

spot = data["spot"]
v1m = data["vol_1m"]
v1y = data["vol_1y"]
fwd_pts_1y = data["fwd_pts_1y"]

# Auto-detect vol units (Bloomberg quotes vol in vol points e.g. 6.0 = 6%;
# our engine wants fractions — divide by 100 if values look like percentages)
if v1m.max() > 1.0:
    v1m = v1m / 100.0
if v1y.max() > 1.0:
    v1y = v1y / 100.0

# Convert forward points to forward levels.  Convention:
#   forward = spot + fwd_points * pip_scale
pip_scale = get_pip_scale("USDCNH")
forward_1y = spot + fwd_pts_1y.reindex(spot.index).ffill() * pip_scale

# -----------------------------------------------------------------------------
# Run backtest
# -----------------------------------------------------------------------------
config = StrategyConfig7(
    pair="USDCNH",
    trigger_pct=float(trigger_pct),
    trigger_lookback_days=int(trigger_lookback_years * 365),
    notional_usd=float(notional_usd),
    take_profit_usd=float(take_profit_usd) if take_profit_usd is not None else None,
    stop_loss_usd=float(stop_loss_usd) if stop_loss_usd is not None else None,
    hedge_mode=str(hedge_mode),
    threshold_trigger_pct=float(threshold_trigger_pct),
    threshold_target_pct=float(threshold_target_pct),
    spot_move_pct=float(spot_move_pct),
    vol_bid_ask_pts=float(vol_bid_ask_pts),
    spot_cost_bps=float(spot_cost_bps),
    usd_rate_pct=float(usd_rate_pct),
)


@st.cache_data(show_spinner="Running USDCNH vol-calendar backtest…")
def _run(folder: str, variant: str,
         trigger_pct: float, lookback_days: int,
         notional_usd: float,
         take_profit_usd: float | None, stop_loss_usd: float | None,
         hedge_mode: str,
         threshold_trigger_pct: float, threshold_target_pct: float,
         spot_move_pct: float,
         vol_bid_ask_pts: float, spot_cost_bps: float, usd_rate_pct: float):
    cfg = StrategyConfig7(
        pair="USDCNH",
        trigger_pct=trigger_pct, trigger_lookback_days=lookback_days,
        notional_usd=notional_usd,
        take_profit_usd=take_profit_usd, stop_loss_usd=stop_loss_usd,
        hedge_mode=hedge_mode,
        threshold_trigger_pct=threshold_trigger_pct,
        threshold_target_pct=threshold_target_pct,
        spot_move_pct=spot_move_pct,
        vol_bid_ask_pts=vol_bid_ask_pts, spot_cost_bps=spot_cost_bps,
        usd_rate_pct=usd_rate_pct,
    )
    d = load_usdcnh(folder, variant)
    sp = d["spot"]
    vm = d["vol_1m"]
    vy = d["vol_1y"]
    fp = d["fwd_pts_1y"]
    if vm.max() > 1.0: vm = vm / 100.0
    if vy.max() > 1.0: vy = vy / 100.0
    fwd = sp + fp.reindex(sp.index).ffill() * get_pip_scale("USDCNH")
    res = run_strategy_7(sp, vm, vy, fwd, cfg)
    return res, summary_stats_7(res)


r, stats = _run(folder, variant,
                config.trigger_pct, config.trigger_lookback_days,
                config.notional_usd,
                config.take_profit_usd, config.stop_loss_usd,
                config.hedge_mode,
                config.threshold_trigger_pct, config.threshold_target_pct,
                config.spot_move_pct,
                config.vol_bid_ask_pts, config.spot_cost_bps,
                config.usd_rate_pct)


# -----------------------------------------------------------------------------
# Render
# -----------------------------------------------------------------------------
render_dashboard_7(r, stats)


st.markdown("---")
st.caption(
    f"Data: `{folder}`  ·  Variant: **{variant}**  ·  "
    f"Date range: {r.spot.index[0].strftime('%Y-%m-%d')} → "
    f"{r.spot.index[-1].strftime('%Y-%m-%d')}  ·  "
    f"Trigger: 1Y vol pct ≤ **{config.trigger_pct:g}** within {trigger_lookback_years}Y"
)

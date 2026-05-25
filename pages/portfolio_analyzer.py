"""Portfolio Analyzer — Leveraged FX Exotic Options Risk Monitor.

Sub-app of the FX Toolkit. Reached via the landing page or sidebar nav;
not run directly.

Monitors a portfolio of European-style FX options: vanillas, call
spreads, call flys, EKOs, and dual EKOs. Risk dashboards aligned to
what actually matters for an exotic book: bucketed Greeks, barrier
and path risk, correlation risk, and scenario cubes.

Six tabs:
  1. Portfolio Overview — book-level summary + per-trade table.
  2. Greeks Dashboard   — bucketed Δ/Γ/Vega/Theta by pair/tenor.
  3. Barrier & Path     — barrier proximity, knockout probabilities,
                          time-to-barrier scenarios.
  4. Scenario Risk Cube — multi-axis (spot × vol × time) MTM cubes,
                          per-trade and aggregated.
  5. Correlation Risk   — dual-EKO trade exposures by joint move.
  6. Trade Drill-down   — full per-trade detail: pricing, Greeks,
                          MTM history since booking.

Data folder convention is the toolkit standard (`_index.csv` + per-pair
SPOT / VOL_ATM / VOL_RR / VOL_BF / FWD_POINTS / RATE_* CSVs); the
sidebar's data-folder input is shared across all toolkit pages via
session state.

Originally fx_levels_monitor's `apps/app_11.py`; ported into the
toolkit with no math changes — only the data-folder resolution and
the page-level chrome (st.set_page_config, app_header) were updated
to match the toolkit's convention.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sibling top-level packages (core/, shared/) importable when
# Streamlit executes this file out of the project root. Same pattern
# as the other toolkit pages.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from core.data_loader import available_dates, snapshot
from core.portfolio import build_portfolio, Trade
from core.pricing_engine import (
    price_trade, compute_greeks, barrier_diagnostics,
    risk_cube, portfolio_risk_cube,
)
from core.conventions import quote_decimals
from core.ui import data_dir_input, app_header
from shared.style import inject_base_css


# =============================================================================
# Trade ticker / display helpers
# =============================================================================

def _tenor_from_dates(booked: pd.Timestamp, expiry: pd.Timestamp) -> str:
    """Approximate tenor label from booking → expiry date span."""
    days = (pd.Timestamp(expiry) - pd.Timestamp(booked)).days
    if days <= 10:
        return f"{days}D"
    if days <= 20:
        return f"{round(days / 7)}W"
    months = round(days / 30)
    return f"{months}M"


def _fmt_strike(K: float, pair: str) -> str:
    """Format a strike to the pair's natural decimal precision."""
    nd = quote_decimals(pair)
    return f"{K:.{nd}f}"


def _fmt_money(x: float, sig: int = 3) -> str:
    """Compact money display: $1.23M / $456k / $12 etc.

    Picks units that keep the number short and readable in metric cards,
    where Streamlit truncates wide values. Used for book-level Greeks
    where the magnitudes vary from a few hundred (Charm) to millions
    (Delta). Negatives carry the sign in the right place: "$-1.2M".
    """
    if x is None or pd.isna(x):
        return "—"
    ax = abs(x)
    sign = "-" if x < 0 else ""
    if ax >= 1e9:
        return f"${sign}{ax/1e9:.{sig}g}B"
    if ax >= 1e6:
        return f"${sign}{ax/1e6:.{sig}g}M"
    if ax >= 1e3:
        return f"${sign}{ax/1e3:.{sig}g}k"
    return f"${sign}{ax:.0f}"


def format_trade_ticker(t: Trade) -> str:
    """Human-readable ticker capturing the structure & key levels.

    Examples:
      vanilla      → "USDJPY 1M 156.50 C"
      call_spread  → "EURUSD 2M 1.1000/1.1300 call spread"
      call_fly     → "AUDUSD 2M 0.6650/0.6850/0.7050 call fly"
      eko          → "USDJPY 4M 152.00 C  UO@162.00 EKO"
      dual_eko     → "WO-CALL 4M · USDJPY 153.00 UO@161.00 / "
                     "USDCNH 7.2300 UO@7.4000"
    """
    tenor = _tenor_from_dates(t.booking_date, t.expiry_date)
    side_arrow = "" if t.side == "buy" else " (SOLD)"

    if t.structure == "vanilla":
        leg = t.legs[0]
        return (f"{t.pair} {tenor} {_fmt_strike(leg.K, t.pair)} "
                f"{leg.opt[0].upper()}{side_arrow}")
    if t.structure == "call_spread":
        ks = "/".join(_fmt_strike(l.K, t.pair) for l in t.legs)
        return f"{t.pair} {tenor} {ks} call spread{side_arrow}"
    if t.structure == "call_fly":
        ks = "/".join(_fmt_strike(l.K, t.pair) for l in t.legs)
        return f"{t.pair} {tenor} {ks} call fly{side_arrow}"
    if t.structure == "eko":
        leg = t.legs[0]
        bar_tag = ("UO" if leg.bar_dir == "up_and_out"
                   else "DO" if leg.bar_dir == "down_and_out"
                   else leg.bar_dir or "")
        return (f"{t.pair} {tenor} {_fmt_strike(leg.K, t.pair)} "
                f"{leg.opt[0].upper()} {bar_tag}@"
                f"{_fmt_strike(leg.H, t.pair)} EKO{side_arrow}")
    if t.structure == "dual_eko":
        l1, l2 = t.legs[0], t.legs2[0]
        b1 = ("UO" if l1.bar_dir == "up_and_out"
              else "DO" if l1.bar_dir == "down_and_out" else "")
        b2 = ("UO" if l2.bar_dir == "up_and_out"
              else "DO" if l2.bar_dir == "down_and_out" else "")
        kind_label = {
            "wo_call": "WO-CALL", "wo_put": "WO-PUT",
            "bo_call": "BO-CALL", "bo_put": "BO-PUT",
        }.get(t.structure_kind, "DUAL")
        return (f"{kind_label} {tenor} · "
                f"{t.pair} {_fmt_strike(l1.K, t.pair)} "
                f"{b1}@{_fmt_strike(l1.H, t.pair)} / "
                f"{t.pair2} {_fmt_strike(l2.K, t.pair2)} "
                f"{b2}@{_fmt_strike(l2.H, t.pair2)}{side_arrow}")
    return f"{t.pair} {t.structure}{side_arrow}"


# =============================================================================
# Page config & caching
# =============================================================================

st.set_page_config(
    page_title="Portfolio Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_base_css()

# Sidebar: data-folder picker (shared across all toolkit pages via
# session state). We resolve it BEFORE the cached helpers so the
# helpers can take a path arg and remain cacheable.
DATA_DIR = data_dir_input(default="market_data")
if DATA_DIR is None:
    # data_dir_input has already shown the "Enter a data folder…"
    # info message. Halt the page; no point continuing without data.
    st.stop()


# Cached helpers — these take the folder as an explicit arg so the
# cache key includes it (re-running with a different folder will
# correctly invalidate). Keeping them at module scope (rather than
# closing over DATA_DIR) preserves the original app_11 behaviour.
@st.cache_data(show_spinner=False)
def _available_dates_cached(data_dir: str) -> list[pd.Timestamp]:
    return available_dates(data_dir)


@st.cache_data(show_spinner=False)
def _snapshot_cached(asof_iso: str, data_dir: str) -> dict:
    return snapshot(data_dir, pd.Timestamp(asof_iso))


@st.cache_data(show_spinner=False)
def _price_all_cached(asof_iso: str, data_dir: str) -> pd.DataFrame:
    """Price the full portfolio and return a tidy DataFrame."""
    asof = pd.Timestamp(asof_iso)
    snap = _snapshot_cached(asof_iso, data_dir)
    trades = build_portfolio()
    rows = []
    for t in trades:
        try:
            r = price_trade(t, snap, asof)
            rows.append({
                "trade_id": t.trade_id,
                "ticker": format_trade_ticker(t),
                "structure": t.structure,
                "pair": t.pair if not t.pair2 else f"{t.pair}/{t.pair2}",
                "primary_pair": t.pair,
                "side": t.side,
                "notional_usd": t.notional_usd,
                "premium_paid_usd": t.premium_paid_usd,
                "booking_date": t.booking_date,
                "expiry_date": t.expiry_date,
                "days_to_expiry": t.days_to_expiry(asof),
                "T_yrs": r["T_yrs"],
                "mtm_usd": r["mtm_usd"],
                "intrinsic_usd": r["intrinsic_usd"],
                "pnl_vs_premium_usd": r["mtm_usd"] - t.premium_paid_usd,
                "notes": t.notes,
                "is_dual": r.get("is_dual", False),
                "p_alive": r.get("p_alive", np.nan),
                "structure_kind": t.structure_kind,
            })
        except Exception as e:
            rows.append({
                "trade_id": t.trade_id,
                "ticker": format_trade_ticker(t),
                "structure": t.structure,
                "pair": t.pair, "mtm_usd": np.nan, "error": str(e),
            })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _greeks_all_cached(asof_iso: str, data_dir: str) -> pd.DataFrame:
    """Compute Greeks across the book — one row per (trade, pair) for duals."""
    asof = pd.Timestamp(asof_iso)
    snap = _snapshot_cached(asof_iso, data_dir)
    trades = build_portfolio()
    rows = []
    for t in trades:
        try:
            g = compute_greeks(t, snap, asof)
            for pair, gp in g["by_pair"].items():
                rows.append({
                    "trade_id": t.trade_id,
                    "structure": t.structure,
                    "structure_kind": t.structure_kind,
                    "pair": pair,
                    "is_dual": (t.structure == "dual_eko"),
                    "days_to_expiry": t.days_to_expiry(asof),
                    "tenor_bucket": _tenor_bucket(t.days_to_expiry(asof)),
                    "notional_usd": t.notional_usd,
                    "base_mtm_usd": g["base_mtm"],
                    "delta_usd_per_1pct": gp["delta_usd_per_1pct"],
                    "gamma_usd_per_1pct2": gp["gamma_usd_per_1pct2"],
                    "vega_usd_per_volpt": gp["vega_usd_per_volpt"],
                    "volga_usd_per_volpt2": gp["volga_usd_per_volpt2"],
                    "vanna_usd_per_1pct_x_volpt": gp["vanna_usd_per_1pct_x_volpt"],
                    "charm_usd_per_day": gp["charm_usd_per_day"],
                    "theta_usd_per_day": g["theta_usd_per_day"],
                    "cega_usd_per_5pct_rho": g.get("cega_usd_per_5pct_rho", np.nan),
                })
        except Exception as e:
            rows.append({"trade_id": t.trade_id, "error": str(e)})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _barriers_all_cached(asof_iso: str, data_dir: str) -> pd.DataFrame:
    asof = pd.Timestamp(asof_iso)
    snap = _snapshot_cached(asof_iso, data_dir)
    trades = build_portfolio()
    rows = []
    for t in trades:
        bars = barrier_diagnostics(t, snap, asof)
        for b in bars:
            rows.append({
                "trade_id": t.trade_id, "structure": t.structure,
                "structure_kind": t.structure_kind,
                **b,
            })
    return pd.DataFrame(rows)


def _tenor_bucket(days: int) -> str:
    if days <= 0:
        return "expired"
    if days <= 14:
        return "0-2W"
    if days <= 45:
        return "2W-1.5M"
    if days <= 90:
        return "1.5M-3M"
    if days <= 180:
        return "3M-6M"
    return ">6M"


# =============================================================================
# Sidebar — date selector & market snapshot
# =============================================================================

st.sidebar.title("⚙️  Controls")

all_dates = _available_dates_cached(DATA_DIR)
default_idx = len(all_dates) - 1  # latest

asof = st.sidebar.selectbox(
    "As-of date",
    options=all_dates,
    index=default_idx,
    format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d (%a)"),
)
asof = pd.Timestamp(asof)
asof_iso = asof.strftime("%Y-%m-%d")

st.sidebar.caption(
    f"Earliest: {all_dates[0].strftime('%Y-%m-%d')} · "
    f"Latest: {all_dates[-1].strftime('%Y-%m-%d')}"
)

snap = _snapshot_cached(asof_iso, DATA_DIR)
st.sidebar.subheader("📈 Market snapshot")
mkt_rows = []
for pair, sp in snap.items():
    nd = quote_decimals(pair)
    mkt_rows.append({
        "Pair": pair,
        "Spot": (f"{sp['spot']:.{nd}f}"
                 if sp.get("spot") is not None else "—"),
        "ATM 1M": f"{sp['vols'].get('1M', np.nan)*100:.2f}%"
                  if "1M" in sp["vols"] else "—",
        "ATM 3M": f"{sp['vols'].get('3M', np.nan)*100:.2f}%"
                  if "3M" in sp["vols"] else "—",
        "RR25 3M": f"{sp['rr25'].get('3M', 0)*100:+.2f}"
                  if "3M" in sp["rr25"] else "—",
    })
st.sidebar.dataframe(pd.DataFrame(mkt_rows), hide_index=True,
                      width='stretch')

st.sidebar.divider()
st.sidebar.caption(
    "Portfolio: 10 trades booked Mar–Apr 2026.  \n"
    "All options are **European**.  \n"
    "EKO = barrier observed at expiry only."
)


# =============================================================================
# Main header
# =============================================================================

app_header(
    "🎯 Portfolio Analyzer — Leveraged FX Exotic Options",
    f"As-of: `{asof.strftime('%Y-%m-%d')}`  ·  "
    f"Portfolio: 10 trades  ·  "
    f"Structures: vanilla · call spread · call fly · EKO · dual EKO",
)


# =============================================================================
# Tabs
# =============================================================================

TAB_OVERVIEW, TAB_GREEKS, TAB_BARRIER, TAB_CUBE, TAB_CORR, TAB_DRILL = st.tabs([
    "📋 Portfolio Overview",
    "📊 Greeks Dashboard",
    "🚧 Barrier & Path Risk",
    "🔥 Scenario Risk Cube",
    "🔗 Correlation Risk",
    "🔍 Trade Drill-down",
])


# -----------------------------------------------------------------------------
# Tab: Portfolio Overview
# -----------------------------------------------------------------------------

with TAB_OVERVIEW:
    with st.expander("💡 What does this view tell me?", expanded=False):
        st.markdown("""
        **Purpose.** The single 'what do I own and what is it worth today' view.
        Every other tab decomposes the risk; this one tells you the size of the
        plate.
        
        **Why it matters for an exotic book.** In a vanilla portfolio, marks
        are roughly linear in spot — a quick look at spot moves gives you a
        ballpark MTM. In an exotic book, marks move discontinuously near
        barriers, structurally with vol regimes, and non-linearly with
        correlation. You need an authoritative MTM that:
        
        - Uses the **same engine** for live pricing and for backtesting/risk
          (no Vanilla-BS in one place and Vanna-Volga in another — that
          mis-state goes straight into hedging error)
        - Aggregates **P&L vs premium**, not just MTM — premium-vs-MTM is your
          actual stake at risk; MTM is what you'd liquidate at
        - Flags **survival probability** for any structure with a live barrier
          — a $500k MTM with a 30% chance of knocking is not the same thing as
          a $500k MTM at 95% survival
        
        **What's NOT in this view.** Greek bucketing (Greeks tab), distance
        to barriers in vol-adjusted std-devs (Barrier tab), scenario stresses
        (Risk Cube tab), and correlation exposure (Correlation tab). The risks
        on each tab need different tooling because they hurt you differently.
        """)

    df = _price_all_cached(asof_iso, DATA_DIR)

    # ---- top-level metrics ----
    total_mtm = df["mtm_usd"].sum()
    total_prem = df["premium_paid_usd"].sum()
    total_pnl = df["pnl_vs_premium_usd"].sum()
    total_notional = df["notional_usd"].sum()
    n_at_risk = ((df["is_dual"]) & (df["p_alive"] < 0.85)).sum() + \
                ((df["structure"] == "eko") & (df["p_alive"].isna())).sum()
    # Count EKO trades close to barrier
    bdf = _barriers_all_cached(asof_iso, DATA_DIR)
    near_bar = bdf[bdf["distance_vol_sd"] < 1.0]["trade_id"].nunique() if not bdf.empty else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total MTM", f"${total_mtm/1e6:+.2f}M")
    c2.metric("Premium paid", f"${total_prem/1e6:.2f}M")
    c3.metric("P&L vs Premium", f"${total_pnl/1e6:+.2f}M",
              delta=f"{(total_pnl/total_prem*100) if total_prem else 0:+.0f}%")
    c4.metric("Total notional", f"${total_notional/1e6:.0f}M")
    c5.metric("Trades within 1σ of barrier", f"{near_bar}",
              help="EKO / dual-EKO legs whose barrier is < 1 vol-adjusted "
                   "std dev away from current spot. These need the most "
                   "watching — a small spot move can vaporise the option.")

    st.divider()

    # ---- positions table ----
    st.subheader("Positions")
    st.caption(
        "**Premium** is shown as a **negative cash flow** (we are long "
        "all options — premium paid out). **MTM** is the option's "
        "current value (positive for in-the-money long positions). "
        "**P&L = MTM + Premium** (premium is already signed). All "
        "trades sized at a uniform **$100M notional** to make MTM/P&L "
        "directly comparable across the book."
    )
    disp = df.copy()
    disp["MTM ($M)"] = disp["mtm_usd"] / 1e6
    # Cash-outflow convention: long premium paid → negative cash flow
    disp["Premium ($M)"] = -disp["premium_paid_usd"] / 1e6
    disp["P&L ($M)"] = disp["pnl_vs_premium_usd"] / 1e6
    disp["Notional ($M)"] = disp["notional_usd"] / 1e6
    disp["Booked"] = disp["booking_date"].dt.strftime("%Y-%m-%d")
    disp["Expiry"] = disp["expiry_date"].dt.strftime("%Y-%m-%d")
    disp["Days left"] = disp["days_to_expiry"]
    disp["Tenor bucket"] = disp["days_to_expiry"].apply(_tenor_bucket)
    disp["Survival %"] = disp["p_alive"].apply(
        lambda x: f"{x*100:.0f}%" if pd.notna(x) else "—"
    )

    show_cols = ["trade_id", "ticker", "structure", "pair", "side",
                 "Notional ($M)", "Booked", "Expiry", "Days left",
                 "MTM ($M)", "Premium ($M)", "P&L ($M)",
                 "Tenor bucket", "Survival %"]
    st.dataframe(
        disp[show_cols].rename(columns={
            "trade_id": "ID", "ticker": "Ticker",
            "structure": "Structure", "pair": "Pair", "side": "Side",
        }),
        hide_index=True, width='stretch',
        column_config={
            "MTM ($M)": st.column_config.NumberColumn(
                format="%.3f",
                help="Option current value (positive for ITM long)"),
            "Premium ($M)": st.column_config.NumberColumn(
                format="%.3f",
                help="Cash flow at booking — negative because we paid "
                     "premium (long all options)"),
            "P&L ($M)": st.column_config.NumberColumn(
                format="%.3f",
                help="= MTM + Premium (both signed)"),
            "Notional ($M)": st.column_config.NumberColumn(format="%.0f"),
        },
    )

    st.divider()

    # ---- P&L attribution charts ----
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("P&L by trade")
        fig = px.bar(
            disp.sort_values("pnl_vs_premium_usd"),
            x="trade_id", y="pnl_vs_premium_usd",
            color=disp["pnl_vs_premium_usd"] >= 0,
            color_discrete_map={True: "#2ca02c", False: "#d62728"},
            text=disp["pnl_vs_premium_usd"].apply(
                lambda v: f"${v/1e6:+.2f}M"
            ),
            title="P&L vs Premium (MTM − Premium)",
        )
        fig.update_layout(showlegend=False, height=380,
                          xaxis_title="", yaxis_title="USD")
        fig.update_yaxes(tickformat="$,.0f")
        fig.add_hline(y=0, line=dict(color="black", width=1))
        st.plotly_chart(fig, width='stretch')

    with col_b:
        st.subheader("MTM by structure type")
        by_struct = (disp.groupby("structure")["mtm_usd"].sum() / 1e6).reset_index()
        by_struct.columns = ["Structure", "MTM ($M)"]
        fig = px.bar(
            by_struct, x="Structure", y="MTM ($M)",
            color="Structure",
            text=by_struct["MTM ($M)"].apply(lambda v: f"${v:+.2f}M"),
            title="MTM aggregated by structure type",
        )
        fig.update_layout(showlegend=False, height=380)
        fig.add_hline(y=0, line=dict(color="black", width=1))
        st.plotly_chart(fig, width='stretch')

    # ---- premium burn / time decay chart ----
    st.subheader("Premium vs MTM by trade")
    plot_df = disp[["trade_id", "premium_paid_usd", "mtm_usd"]].copy()
    plot_df["Premium"] = plot_df["premium_paid_usd"]
    plot_df["MTM"] = plot_df["mtm_usd"]
    long = plot_df.melt(id_vars="trade_id", value_vars=["Premium", "MTM"],
                        var_name="Kind", value_name="USD")
    fig = px.bar(long, x="trade_id", y="USD", color="Kind", barmode="group",
                 color_discrete_map={"Premium": "#1f77b4", "MTM": "#ff7f0e"},
                 title="Premium paid (blue) vs current MTM (orange)")
    fig.update_layout(height=380, xaxis_title="Trade")
    fig.update_yaxes(tickformat="$,.0f")
    st.plotly_chart(fig, width='stretch')

    # ---- trade notes ----
    with st.expander("📝 Trade rationale notes", expanded=False):
        for _, row in disp.iterrows():
            st.markdown(f"**{row['trade_id']}** ({row['structure']}, "
                        f"{row['pair']}): {row['notes']}")


# -----------------------------------------------------------------------------
# Placeholder tabs (filled in later phases)
# -----------------------------------------------------------------------------

with TAB_GREEKS:
    with st.expander("💡 What does this view tell me?", expanded=False):
        st.markdown("""
        **Purpose.** All Greeks aggregated by **currency**, **tenor bucket**,
        and **structure type** — so you can see "what's my AUD vega 1M-3M"
        in one number rather than chasing through 10 individual trades.

        **What's first-order vs second-order — and why both matter:**

        | Greek | Interpretation | Why it matters |
        |-------|---------------|---------------|
        | **Δ Delta** | USD per 1% spot move | Linear hedge size |
        | **Γ Gamma** | USD per (1% spot)² | How much delta moves when spot moves — the gap between linear and reality |
        | **V Vega** | USD per 1 vol point | First-order vol exposure |
        | **Volga** | USD per (1vol)² | How vega changes with vol — material for OTM strikes (vol-of-vol) |
        | **Vanna** | USD per (1% spot × 1vol) | How delta moves with vol — the killer Greek in RR-heavy structures |
        | **Charm** | USD per day (Δ decay) | Hedge-rebalancing pressure as time passes |
        | **Theta** | USD per day (total) | Time decay of the whole structure |

        **Vanna deserves special attention in FX.** The spot-vol correlation
        in FX (RR) means that when spot moves, the vol surface twists in a
        predictable direction — your delta then moves too, *before* spot has
        moved any further. Long-RR structures (USDJPY upside, EM USD-calls)
        have positive vanna and benefit; short-RR structures bleed.

        **For barrier options, gamma flips sign near the barrier.** A reverse-
        KO call (ITM strike, OUT barrier above spot) is long-gamma far from the
        barrier and **short-gamma** near it — because the barrier wipes the
        payoff. This view shows you the *current* sign; the Barrier Map tab
        shows you where the flip happens.

        **Aggregation choice.** Greeks are aggregated **per currency** (not
        per pair) for duals — so a USDCNH leg of a worst-of dual contributes
        to USDCNH bucket, not to some "USDJPY/USDCNH" bucket. This matches
        how you'd actually hedge — by going to the spot desk for each ccy.
        """)

    gdf = _greeks_all_cached(asof_iso, DATA_DIR)
    if "error" in gdf.columns and gdf["error"].notna().any():
        st.error(f"Greeks failed for some trades: "
                 f"{gdf[gdf['error'].notna()]['trade_id'].tolist()}")
        gdf = gdf[gdf["error"].isna()] if "error" in gdf.columns else gdf

    # ---- top-line book Greeks ----
    st.subheader("Book-level Greeks (USD)")
    # Theta is trade-level (not per-pair), so dedup before summing
    theta_total = (gdf.drop_duplicates("trade_id")["theta_usd_per_day"].sum()
                   if not gdf.empty else 0)
    book = {
        "Δ (per 1% spot)": gdf["delta_usd_per_1pct"].sum(),
        "Γ (per 1%²)": gdf["gamma_usd_per_1pct2"].sum(),
        "V Vega (per volpt)": gdf["vega_usd_per_volpt"].sum(),
        "Volga (per volpt²)": gdf["volga_usd_per_volpt2"].sum(),
        "Vanna (per 1%·volpt)": gdf["vanna_usd_per_1pct_x_volpt"].sum(),
        "Charm (per day)": gdf["charm_usd_per_day"].sum(),
        "Θ Theta (per day)": theta_total,
    }
    # 4 + 3 grid keeps metric columns wide enough to show full values on
    # standard screens — the 7-in-a-row layout truncates to ellipses.
    row1 = st.columns(4)
    row2 = st.columns(4)
    cells = list(row1) + list(row2)
    for cell, (label, val) in zip(cells, book.items()):
        cell.metric(label, _fmt_money(val))

    st.caption(
        "Δ positive → book gains on +1% spot rally (per pair convention: each "
        "leg signed by its own pair's spot move). Θ is summed across distinct "
        "trades (trade-level Greek, not per-pair)."
    )

    st.divider()

    # ---- Greeks by currency ----
    st.subheader("Greeks bucketed by currency")
    greek_choice = st.radio(
        "Greek to display",
        ["Delta", "Gamma", "Vega", "Vanna", "Volga", "Charm"],
        horizontal=True, key="greek_by_ccy",
    )
    col_map = {
        "Delta": "delta_usd_per_1pct",
        "Gamma": "gamma_usd_per_1pct2",
        "Vega": "vega_usd_per_volpt",
        "Vanna": "vanna_usd_per_1pct_x_volpt",
        "Volga": "volga_usd_per_volpt2",
        "Charm": "charm_usd_per_day",
    }
    col = col_map[greek_choice]

    ccy_agg = gdf.groupby("pair")[col].sum().reset_index()
    ccy_agg.columns = ["Pair", greek_choice]
    fig = px.bar(
        ccy_agg, x="Pair", y=greek_choice,
        color=ccy_agg[greek_choice] >= 0,
        color_discrete_map={True: "#2ca02c", False: "#d62728"},
        text=ccy_agg[greek_choice].apply(lambda v: f"${v:,.0f}"),
        title=f"{greek_choice} aggregated by currency pair",
    )
    fig.update_layout(showlegend=False, height=360, yaxis_title="USD")
    fig.update_yaxes(tickformat="$,.0f")
    fig.add_hline(y=0, line=dict(color="black", width=1))
    st.plotly_chart(fig, width='stretch')

    st.divider()

    # ---- Greeks heatmap: ccy × tenor ----
    st.subheader("Greeks heatmap — currency × tenor bucket")
    tenor_order = ["0-2W", "2W-1.5M", "1.5M-3M", "3M-6M", ">6M"]
    pivot = gdf.pivot_table(
        index="pair", columns="tenor_bucket", values=col, aggfunc="sum",
        fill_value=0,
    )
    pivot = pivot.reindex(columns=[c for c in tenor_order if c in pivot.columns])

    if not pivot.empty:
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="RdBu_r",
            zmid=0,
            text=[[f"${v:,.0f}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            colorbar=dict(title="USD"),
        ))
        fig.update_layout(
            height=320, xaxis_title="Tenor bucket", yaxis_title="Pair",
            title=f"{greek_choice} concentration map",
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.info("No Greeks to plot at this date.")

    st.divider()

    # ---- Greeks by structure type ----
    st.subheader("Greeks bucketed by structure")
    by_struct = gdf.groupby("structure")[
        ["delta_usd_per_1pct", "gamma_usd_per_1pct2",
         "vega_usd_per_volpt", "vanna_usd_per_1pct_x_volpt",
         "volga_usd_per_volpt2", "charm_usd_per_day"]
    ].sum().reset_index()
    by_struct.columns = ["Structure", "Δ", "Γ", "V", "Vanna", "Volga", "Charm"]
    st.dataframe(
        by_struct, hide_index=True, width='stretch',
        column_config={c: st.column_config.NumberColumn(format="$%,.0f")
                       for c in ["Δ", "Γ", "V", "Vanna", "Volga", "Charm"]},
    )

    with st.expander("🔬 Per-(trade × pair) Greeks table", expanded=False):
        per_trade = gdf[[
            "trade_id", "structure", "pair", "tenor_bucket",
            "delta_usd_per_1pct", "gamma_usd_per_1pct2",
            "vega_usd_per_volpt", "vanna_usd_per_1pct_x_volpt",
            "volga_usd_per_volpt2", "charm_usd_per_day",
        ]].rename(columns={
            "trade_id": "ID", "structure": "Structure", "pair": "Pair",
            "tenor_bucket": "Tenor",
            "delta_usd_per_1pct": "Δ",
            "gamma_usd_per_1pct2": "Γ",
            "vega_usd_per_volpt": "V",
            "vanna_usd_per_1pct_x_volpt": "Vanna",
            "volga_usd_per_volpt2": "Volga",
            "charm_usd_per_day": "Charm",
        })
        st.dataframe(per_trade, hide_index=True, width='stretch',
                     column_config={c: st.column_config.NumberColumn(format="$%,.0f")
                                    for c in ["Δ", "Γ", "V", "Vanna", "Volga", "Charm"]})

    st.divider()

    # ---- Per-pair Greeks vs spot (aggregated across positions) -------
    # For each pair in the book, sweep its spot ±15% holding all other
    # pairs constant, and plot MTM / Δ / Γ / V on the same axis. Strike
    # levels of every leg touching that pair are overlaid as vertical
    # reference lines so you can see exactly where the curvature lives.
    #
    # WHY this view: the book-level Greeks at the top are point
    # estimates at current spot. If USDJPY rallies 3%, what does my
    # delta become? My gamma? My vega? That's a "Greeks profile" view,
    # and it's what option desks actually trade off of intraday.
    #
    # Implementation note: we re-price every trade at each spot grid
    # point and aggregate. For 25 grid points × ~6 trades touching the
    # pair × 1 vanilla re-price each = ~150 re-prices per pair. Adding
    # MC-priced dual EKOs would balloon this — for now we exclude dual
    # EKOs from this view (they show up in the joint surface in
    # drill-down instead, which is the proper way to see their geometry).
    st.subheader("Aggregated Greeks profile by pair")
    st.caption(
        "For each currency pair, sweep its spot ±15% and aggregate the "
        "MTM + key Greeks across **every position touching that pair**. "
        "Dashed vertical lines = current spot; dotted lines = strikes "
        "(and barriers) of legs in this pair. Dual-EKO trades are "
        "excluded from this view — their joint geometry is better seen "
        "in the per-trade drill-down surface."
    )

    @st.cache_data(show_spinner="Computing Greeks profiles…")
    def _build_pair_greeks_profile(asof_iso: str, pair: str,
                                       n_grid: int = 25,
                                       sweep_pct: float = 0.15) -> dict:
        """Sweep `pair`'s spot, aggregate MTM/Δ/Γ/V across trades that
        touch `pair`. Excludes dual EKO trades (slow, and joint
        geometry is best viewed in TAB_DRILL surface instead)."""
        asof_ts = pd.Timestamp(asof_iso)
        snap = _snapshot_cached(asof_iso, DATA_DIR)
        trades_all = build_portfolio()
        # Single-pair only — exclude duals from the aggregated profile
        relevant = [t for t in trades_all
                    if t.structure != "dual_eko" and t.pair == pair]
        if not relevant:
            return {}
        S0 = snap[pair]["spot"]
        if S0 is None:
            return {}
        shocks = np.linspace(-sweep_pct, sweep_pct, n_grid)
        S_grid = S0 * (1.0 + shocks)
        mtm_curve = np.zeros(n_grid)
        delta_curve = np.zeros(n_grid)
        gamma_curve = np.zeros(n_grid)
        vega_curve = np.zeros(n_grid)
        for i, S_target in enumerate(S_grid):
            dS_offset = S_target - S0
            # Aggregate across all trades touching this pair
            for t in relevant:
                base_bump = {"spot": {pair: dS_offset}}
                mtm_at = price_trade(t, snap, asof_ts,
                                       bump=base_bump)["mtm_usd"]
                mtm_curve[i] += mtm_at
                # Δ via central diff around S_target
                ds_loc = S_target * 0.005
                up = price_trade(t, snap, asof_ts,
                                 bump={"spot": {pair: dS_offset + ds_loc}}
                                 )["mtm_usd"]
                dn = price_trade(t, snap, asof_ts,
                                 bump={"spot": {pair: dS_offset - ds_loc}}
                                 )["mtm_usd"]
                # Δ in USD per 1% spot move at S_target
                delta_curve[i] += (up - dn) / (2 * ds_loc) * S_target * 0.01
                # Γ in USD per (1% spot)^2 at S_target
                gamma_curve[i] += ((up + dn - 2 * mtm_at) / (ds_loc ** 2)
                                       * (S_target * 0.01) ** 2)
                # Vega — bump vol by 0.5 volpts around base spot
                vu = price_trade(t, snap, asof_ts,
                                 bump={"spot": {pair: dS_offset},
                                       "vol": {pair: +0.005}})["mtm_usd"]
                vd = price_trade(t, snap, asof_ts,
                                 bump={"spot": {pair: dS_offset},
                                       "vol": {pair: -0.005}})["mtm_usd"]
                vega_curve[i] += (vu - vd) / (2 * 0.005) * 0.01
        # Collect strike / barrier levels for vertical reference lines
        levels = []
        for t in relevant:
            for leg in t.legs:
                levels.append((leg.K, "K", t.trade_id))
                if leg.H is not None:
                    levels.append((leg.H, "H", t.trade_id))
        return {
            "S_grid": S_grid, "S0": S0, "pair": pair,
            "mtm": mtm_curve, "delta": delta_curve,
            "gamma": gamma_curve, "vega": vega_curve,
            "levels": levels,
            "n_trades": len(relevant),
            "trade_ids": [t.trade_id for t in relevant],
        }

    pair_choice = st.selectbox(
        "Pair to profile",
        sorted({t.pair for t in build_portfolio()
                if t.structure != "dual_eko"}),
        key="greek_profile_pair",
    )
    profile = _build_pair_greeks_profile(asof_iso, pair_choice)

    if not profile:
        st.info(f"No single-pair trades on {pair_choice}.")
    else:
        st.caption(
            f"Aggregating across {profile['n_trades']} trade(s): "
            f"{', '.join(profile['trade_ids'])}."
        )
        # 2x2 grid of charts: MTM, Δ, Γ, V — all on same x-axis (spot)
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("MTM (USD)", "Δ (USD per 1% spot)",
                            "Γ (USD per 1%²)", "Vega (USD per volpt)"),
            shared_xaxes=True,
            vertical_spacing=0.12, horizontal_spacing=0.10,
        )
        x = profile["S_grid"]
        nd = quote_decimals(pair_choice)

        traces = [
            ("mtm", "#22c55e", 1, 1),
            ("delta", "#3b82f6", 1, 2),
            ("gamma", "#f59e0b", 2, 1),
            ("vega", "#a855f7", 2, 2),
        ]
        for key, color, r, c in traces:
            fig.add_trace(go.Scatter(
                x=x, y=profile[key], mode="lines",
                line=dict(color=color, width=2.5),
                showlegend=False,
                hovertemplate=(f"{pair_choice}: %{{x:.{nd}f}}<br>"
                                  f"{key}: $%{{y:+,.0f}}<extra></extra>"),
            ), row=r, col=c)

        # All overlay lines (current spot, zero baseline, strikes, barriers)
        # are added in a single update_layout(shapes=..., annotations=...)
        # call. Previously we used many sequential add_vline() calls inside
        # a nested loop, which caused Plotly to recompute layout on each
        # call — under Streamlit's incremental rendering this surfaced as
        # the chart appearing multiple times in the UI in progressively
        # more-populated states. One bulk shapes assignment fixes it.
        #
        # The subplot axis naming convention for a 2x2 make_subplots is:
        #   (1,1) → xref "x",  yref "y"
        #   (1,2) → xref "x2", yref "y2"
        #   (2,1) → xref "x3", yref "y3"
        #   (2,2) → xref "x4", yref "y4"
        # Using yref "y domain" anchors the line vertically across the
        # subplot regardless of that subplot's data range.
        subplot_refs = [("x", "y"), ("x2", "y2"),
                          ("x3", "y3"), ("x4", "y4")]
        shapes = []
        annotations = []

        # Current-spot dashed grey vlines (all 4 subplots, no annotation)
        for xref, yref in subplot_refs:
            shapes.append(dict(
                type="line", xref=xref, yref=f"{yref} domain",
                x0=profile["S0"], x1=profile["S0"], y0=0, y1=1,
                line=dict(color="#94a3b8", width=1.5, dash="dash"),
            ))
            # Zero baseline (horizontal) on each subplot
            shapes.append(dict(
                type="line", xref=f"{xref} domain", yref=yref,
                x0=0, x1=1, y0=0, y1=0,
                line=dict(color="#475569", width=1),
            ))

        # Strike / barrier overlays — dedup by (rounded level, kind) and
        # annotate ONLY on the top-left panel. To prevent annotation text
        # from overlapping when multiple strikes sit close together
        # (e.g. an AUDUSD call fly with strikes 0.665 / 0.685 / 0.705),
        # we stagger the annotations vertically by sorted-position index.
        unique_levels = []
        seen = set()
        for level, kind, tid in profile["levels"]:
            if not (profile["S_grid"][0] <= level <= profile["S_grid"][-1]):
                continue
            key = (round(level, 4), kind)
            if key in seen:
                continue
            seen.add(key)
            unique_levels.append((level, kind))

        # Sort left-to-right so adjacent-x annotations stagger predictably
        unique_levels.sort(key=lambda lk: lk[0])
        for i, (level, kind) in enumerate(unique_levels):
            color = "#60a5fa" if kind == "K" else "#ef4444"
            # Vertical line on all four sub-plots
            for xref, yref in subplot_refs:
                shapes.append(dict(
                    type="line", xref=xref, yref=f"{yref} domain",
                    x0=level, x1=level, y0=0, y1=1,
                    line=dict(color=color, width=1, dash="dot"),
                    opacity=0.65,
                ))
            # Annotation only on top-left subplot, stagger y by index
            # so close-together strikes don't print on top of each other
            y_stagger = 1.05 - 0.06 * (i % 3)
            annotations.append(dict(
                xref="x", yref="y domain",
                x=level, y=y_stagger,
                text=f"{kind}={level:.{nd}f}",
                showarrow=False,
                xanchor="center", yanchor="bottom",
                font=dict(color=color, size=9),
            ))

        fig.update_xaxes(title_text=f"{pair_choice} spot", row=2, col=1)
        fig.update_xaxes(title_text=f"{pair_choice} spot", row=2, col=2)
        fig.update_layout(
            height=620,
            margin=dict(l=10, r=10, t=60, b=10),
            title_text=(f"Greeks profile vs {pair_choice} spot · "
                          f"current = {profile['S0']:.{nd}f} · "
                          f"blue dotted = strikes, red dotted = barriers"),
            shapes=shapes,
            annotations=list(fig.layout.annotations) + annotations,
        )
        # Format all y-axes as currency
        fig.update_yaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig, width='stretch')

        st.caption(
            "**How to read this**: each curve is the **aggregated** "
            "MTM/Greek across all trades on this pair, swept through "
            "spot. The vertical dashed line is current spot — where you "
            "are now. Blue dotted lines = strikes (where intrinsic "
            "kicks in / spreads turn over); red dotted lines = barriers "
            "(where KO trades go to zero — watch for delta jumps and "
            "negative gamma right before them on reverse-KOs)."
        )

with TAB_BARRIER:
    with st.expander("💡 What does this view tell me?", expanded=False):
        st.markdown("""
        **Why barriers need their own dashboard.** A vanilla P&L moves
        linearly. A barrier option's P&L moves linearly until it doesn't —
        when the barrier hits, the option goes from valuable to zero in a
        single tick. Vega, gamma, and delta all do strange things near the
        barrier. You cannot treat this risk inside a normal Greek bucket.

        **The three measures of "how close" matters:**
        - **Distance in %**: the headline number — the easiest to communicate
        - **Distance in pips** (or quote units): what spot moves you'd need
        - **Distance in vol-adjusted std devs** = `|ln(H/S)| / (σ·√T)` —
          the *actual* probability-relevant metric. A 2% distance with 2 weeks
          to go is much riskier than 2% with 6 months to go, even though the
          first two numbers look identical. The std-dev distance reads off
          how many "shocks" of typical size sit between you and oblivion.

        **What "survival probability" tells you.** Risk-neutral P(barrier
        not breached at expiry) — the chance you collect anything. For a
        European-monitoring KO this is just `N(d₂)` against the barrier (with
        the right sign for UO/DO). **Note**: this is risk-neutral, not
        physical — it incorporates the forward drift, not your view.

        **The gamma flip near reverse-KO barriers.** A reverse-KO call has:
        - Far below H: positive gamma (it behaves like a vanilla call)
        - Near H: **negative gamma** (the upside is capped by knockout, so
          rallying spot reduces value)
        - At H: discontinuous — drops to zero

        This is the single biggest source of P&L surprise in a reverse-KO
        book. The "P&L slope to barrier" column quantifies it: it's the MTM
        change if spot moves halfway to the barrier from here.

        **What "what kills you" looks like:**
        - **EKO**: spot drifts to barrier → option vaporizes at expiry
        - **Dual EKO worst-of**: *either* underlying touches its barrier →
          payoff goes to zero. Effective survival probability is much lower
          than each leg's marginal survival.
        """)

    bdf = _barriers_all_cached(asof_iso, DATA_DIR)

    if bdf.empty:
        st.info("No barrier options in the book at this date.")
    else:
        # ---- distance dashboard ----
        st.subheader("Distance-to-barrier dashboard")
        disp = bdf.copy()
        disp["Distance %"] = disp["distance_pct"] * 100
        disp["Distance σ√T"] = disp["distance_vol_sd"]
        disp["Survival %"] = disp["survival_prob"] * 100
        disp["Days left"] = (disp["T_yrs"] * 365).round().astype(int)
        disp["ATM vol"] = (disp["sigma"] * 100).round(2).astype(str) + "%"

        show = disp[[
            "trade_id", "structure", "pair", "S", "K", "H", "bar_dir",
            "Distance %", "Distance σ√T", "Survival %", "Days left", "ATM vol",
        ]].rename(columns={
            "trade_id": "ID", "structure": "Structure", "pair": "Pair",
            "S": "Spot", "K": "Strike", "H": "Barrier", "bar_dir": "Direction",
        })

        # color-flag closeness. NB: must set both background AND text
        # color — Streamlit's dark theme defaults text to white, which
        # is unreadable on the light-yellow/light-red backgrounds.
        def _color_rows(row):
            sd = row["Distance σ√T"]
            if sd < 0.5:
                return ["background-color: #ffcccc; color: #1f2937"] * len(row)
            if sd < 1.0:
                return ["background-color: #fff2cc; color: #1f2937"] * len(row)
            return [""] * len(row)

        st.dataframe(
            show.style.apply(_color_rows, axis=1).format({
                "Spot": "{:.4f}", "Strike": "{:.4f}", "Barrier": "{:.4f}",
                "Distance %": "{:+.2f}%", "Distance σ√T": "{:.2f}",
                "Survival %": "{:.1f}%",
            }),
            hide_index=True, width='stretch',
        )
        st.caption(
            "🟥 < 0.5 σ√T (alarm — small spot move triggers KO)  ·  "
            "🟨 0.5-1.0 σ√T (watch — meaningful breach risk)  ·  "
            "⬜ > 1.0 σ√T"
        )

        st.divider()

        # ---- distance bubble chart ----
        st.subheader("Barrier proximity map")
        st.caption(
            "Each bubble = one barrier in the book. **Y-axis is the proximity "
            "metric you actually care about** (vol-adjusted std devs). "
            "Bubble size scales with notional, colour with survival probability."
        )
        plot_df = disp.copy()
        plot_df["abs_dist"] = plot_df["Distance σ√T"]

        # Need notional from main df
        pdf = _price_all_cached(asof_iso, DATA_DIR)[["trade_id", "notional_usd"]]
        plot_df = plot_df.merge(pdf, on="trade_id", how="left")

        fig = px.scatter(
            plot_df,
            x="Days left", y="abs_dist",
            size="notional_usd",
            color="Survival %",
            color_continuous_scale="RdYlGn",
            range_color=[0, 100],
            hover_data={
                "trade_id": True, "structure": True, "pair": True,
                "Distance %": ":.2f", "Distance σ√T": ":.2f",
                "Survival %": ":.1f", "notional_usd": ":$,.0f",
            },
            title="Days to expiry × distance to barrier (vol-adjusted)",
        )
        fig.add_hline(y=0.5, line=dict(color="red", width=1, dash="dash"),
                      annotation_text="0.5 σ√T — alarm zone")
        fig.add_hline(y=1.0, line=dict(color="orange", width=1, dash="dash"),
                      annotation_text="1.0 σ√T — watch zone")
        fig.update_layout(
            height=420, xaxis_title="Days to expiry",
            yaxis_title="Distance to barrier (σ√T)",
        )
        st.plotly_chart(fig, width='stretch')

        st.divider()

        # ---- P&L slope to barrier for each EKO trade ----
        st.subheader("P&L profile vs spot — barrier trades only")
        st.caption(
            "How MTM responds to spot moves. The dashed line is the barrier. "
            "Near-vertical drops are the gamma cliff."
        )

        snap_now = _snapshot_cached(asof_iso, DATA_DIR)
        trades_obj = build_portfolio()
        barrier_trades = [t for t in trades_obj
                          if t.structure in ("eko", "dual_eko")]

        # Compute MTM along a spot ladder for each barrier trade
        n_pts = 41
        ladders = []
        for t in barrier_trades:
            S0 = snap_now[t.pair]["spot"]
            if S0 is None:
                # No spot data for this pair at as-of — skip the ladder
                # rather than blowing up the whole tab. Surfaces as the
                # trade simply not appearing in the scenarios chart.
                continue
            shocks = np.linspace(-0.10, 0.10, n_pts)  # ±10%
            mtms = []
            for sh in shocks:
                bump = {"spot": {t.pair: S0 * sh}}
                try:
                    mtms.append(price_trade(t, snap_now, asof,
                                             bump=bump)["mtm_usd"])
                except Exception:
                    mtms.append(np.nan)
            # For duals, also report barriers on both pairs (use pair1 barrier
            # for the dashed line in this plot — the other pair held constant
            # at its current spot)
            primary_bars = [l.H for l in t.legs if l.H is not None]
            ladders.append({
                "trade_id": t.trade_id, "structure": t.structure,
                "pair": t.pair, "S0": S0,
                "spot_shocks": shocks, "spots": S0 * (1 + shocks),
                "mtm_usd": mtms,
                "barriers": primary_bars,
                "side": t.side,
                "label": (f"{t.trade_id} ({t.structure} {t.pair}"
                          + (" worst-of" if t.structure_kind else "") + ")"),
            })

        # Plot grid
        n_cols = 2
        for i in range(0, len(ladders), n_cols):
            cols_pl = st.columns(n_cols)
            for j, c in enumerate(cols_pl):
                if i + j >= len(ladders):
                    break
                L = ladders[i + j]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=L["spots"], y=L["mtm_usd"], mode="lines",
                    line=dict(width=2.5, color="#1f77b4"),
                    name="MTM",
                ))
                # mark current spot
                fig.add_vline(x=L["S0"], line=dict(color="black", width=1,
                                                    dash="dot"),
                              annotation_text=f"S = {L['S0']:.4f}")
                # mark each barrier on primary pair
                for H in L["barriers"]:
                    fig.add_vline(x=H, line=dict(color="red", width=2,
                                                  dash="dash"),
                                  annotation_text=f"H = {H:.4f}")
                fig.update_layout(
                    title=L["label"], height=320,
                    xaxis_title=f"Spot ({L['pair']})",
                    yaxis_title="MTM (USD)",
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                fig.update_yaxes(tickformat="$,.0f")
                fig.add_hline(y=0, line=dict(color="grey", width=1))
                c.plotly_chart(fig, width='stretch')

        with st.expander("📐 P&L slope decomposition near the barrier",
                          expanded=False):
            st.markdown("""
            The slope of these P&L lines *is* the trade's delta (in USD per
            unit of spot). Where the line is flat → no spot sensitivity.
            Where it rolls over and drops to zero → that's the gamma flip
            zone for a reverse-KO. The vertical drop at H is the "barrier
            cliff": the structural payout loss if spot reaches H at expiry.

            **Reading the chart for hedging:**
            - **Slope** = effective delta for the next move
            - **Curvature** = effective gamma; positive curvature means you'll
              be over-hedged on a rally, negative means you'll be under-hedged
            - **Distance from current spot to the cliff** = how much spot
              movement you can tolerate before the structure is wiped
            """)
with TAB_CUBE:
    with st.expander("💡 What does this view tell me?", expanded=False):
        st.markdown("""
        **Why a scenario cube and not just Greeks.** Greeks are local —
        they tell you about a small spot move from here. Stress moves
        (a +5% USDJPY rally, a vol-spike) go outside the local
        approximation; second-order Greeks (gamma, volga, vanna) help, but
        only up to a point. For a leveraged exotic book, you need to
        re-price under the actual stressed market state.

        **Reading the cube.**
        - Rows = vol shifts (vol-points added to *every* pair's surface)
        - Columns = spot shocks (proportional move applied to *every* pair)
        - Cell = USD MTM of the stressed book at that scenario

        **What this is NOT.** It's not a stress test of individual pair
        moves (USDJPY +5% while EURUSD flat); it's the *common* shock — a
        risk-off / dollar-rally / vol-spike scenario. For per-pair stress,
        use the per-trade cubes below.

        **Where the cube reveals exotic-specific risk:**
        - **Barrier knockouts under spot shocks**: large negative numbers
          in the columns where a barrier gets breached. A vanilla book
          would just have a smooth P&L surface; an exotic book has cliffs.
        - **Vol-shift × spot-shift cross term** (vanna): the row × column
          interaction. If row +2vol and column +5% produce a number that's
          NOT just (row +2vol col 0) + (row 0 col +5%) − (row 0 col 0), the
          gap is the realized vanna contribution.
        - **Vol-volatility (volga)**: row symmetry. If row +5vol and row
          -5vol both lose money relative to row 0, you're short volga
          (short OTM optionality).
        """)

    st.subheader("Portfolio scenario cube")

    c1, c2, c3 = st.columns([1, 1, 1])
    max_spot = c1.slider("Max spot shock (±%)", 1, 15, 8, key="ms")
    n_spot = c2.slider("Spot grid points", 5, 13, 9, key="ns")
    max_vol = c3.slider("Max vol shock (±vol pts)", 1, 10, 5, key="mv")
    n_vol = st.slider("Vol grid points", 3, 9, 5, key="nv")

    spot_shocks = list(np.linspace(-max_spot/100, max_spot/100, n_spot))
    vol_shocks = list(np.linspace(-max_vol/100, max_vol/100, n_vol))

    snap_now = _snapshot_cached(asof_iso, DATA_DIR)
    trades_obj = build_portfolio()

    @st.cache_data(show_spinner="Building portfolio cube …")
    def _port_cube_cached(asof_iso, spot_tup, vol_tup):
        snap = _snapshot_cached(asof_iso, DATA_DIR)
        trades = build_portfolio()
        return portfolio_risk_cube(
            trades, snap, pd.Timestamp(asof_iso),
            list(spot_tup), list(vol_tup),
        )

    cube = _port_cube_cached(asof_iso, tuple(spot_shocks), tuple(vol_shocks))

    # heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cube.values,
        x=cube.columns.tolist(),
        y=cube.index.tolist(),
        colorscale="RdBu_r",
        zmid=cube.iloc[len(cube)//2, len(cube.columns)//2],  # centre on base
        text=[[f"${v/1e6:+.2f}M" for v in row] for row in cube.values],
        texttemplate="%{text}",
        colorbar=dict(title="MTM ($)", tickformat="$,.0f"),
    ))
    fig.update_layout(
        height=420,
        xaxis_title="Spot shock (applied to every pair)",
        yaxis_title="Vol shock (applied to every surface)",
        title="Portfolio MTM under joint spot × vol shocks",
    )
    st.plotly_chart(fig, width='stretch')

    # P&L vs base
    base_idx_r = len(cube) // 2
    base_idx_c = len(cube.columns) // 2
    base = cube.iloc[base_idx_r, base_idx_c]
    pnl_cube = cube - base

    st.subheader("P&L change vs unstressed book")
    fig = go.Figure(data=go.Heatmap(
        z=pnl_cube.values,
        x=pnl_cube.columns.tolist(),
        y=pnl_cube.index.tolist(),
        colorscale="RdYlGn",
        zmid=0,
        text=[[f"${v/1e6:+.2f}M" for v in row] for row in pnl_cube.values],
        texttemplate="%{text}",
        colorbar=dict(title="ΔMTM ($)", tickformat="$,.0f"),
    ))
    fig.update_layout(
        height=420,
        xaxis_title="Spot shock",
        yaxis_title="Vol shock",
        title="Book P&L change vs base (MTM − unstressed MTM)",
    )
    st.plotly_chart(fig, width='stretch')

    # ---- per-trade cubes (collapsible) ----
    st.divider()
    st.subheader("Per-trade cubes")
    st.caption("Drill into the trades that drive the picture.")

    @st.cache_data(show_spinner=False)
    def _trade_cube_cached(asof_iso, trade_id, spot_tup, vol_tup):
        snap = _snapshot_cached(asof_iso, DATA_DIR)
        trades = build_portfolio()
        t = next(t for t in trades if t.trade_id == trade_id)
        return risk_cube(t, snap, pd.Timestamp(asof_iso),
                          list(spot_tup), list(vol_tup))

    pdf = _price_all_cached(asof_iso, DATA_DIR)
    trade_choices = pdf["trade_id"].tolist()
    pick = st.multiselect(
        "Show cubes for these trades:",
        options=trade_choices, default=["T7", "T9"],
    )
    n_cols = 2
    for i in range(0, len(pick), n_cols):
        cols_pl = st.columns(n_cols)
        for j, c in enumerate(cols_pl):
            if i + j >= len(pick):
                break
            tid = pick[i + j]
            cube_t = _trade_cube_cached(asof_iso, tid,
                                         tuple(spot_shocks), tuple(vol_shocks))
            fig = go.Figure(data=go.Heatmap(
                z=cube_t.values,
                x=cube_t.columns.tolist(),
                y=cube_t.index.tolist(),
                colorscale="RdYlGn",
                zmid=0,
                text=[[f"${v/1e3:+.0f}k" for v in row]
                      for row in cube_t.values],
                texttemplate="%{text}",
                colorbar=dict(title="MTM ($)"),
            ))
            t_info = pdf[pdf["trade_id"] == tid].iloc[0]
            fig.update_layout(
                title=f"{tid} — {t_info['structure']} {t_info['pair']}",
                height=320, margin=dict(l=10, r=10, t=40, b=10),
                xaxis_title="Spot", yaxis_title="Vol",
            )
            c.plotly_chart(fig, width='stretch')
with TAB_CORR:
    with st.expander("💡 What does this view tell me?", expanded=False):
        st.markdown("""
        **Why correlation deserves its own tab.** For any multi-asset
        structure — dual EKO, worst-of, best-of, basket — the price depends
        on the **correlation** between the two underlyings as much as on the
        marginal vols. And correlation is the most mismarked Greek in any
        exotic book because:

        1. It's hard to hedge directly (you can buy correlation swaps but
           liquidity is thin).
        2. Implied correlation can decouple from realised correlation for
           weeks.
        3. In stress, correlation *moves* — a "correlation regime shift"
           where pairs that used to be 0.4 become 0.8 (or -0.4 → +0.7) is a
           well-documented phenomenon during risk-off events.

        **Cega = ∂Price / ∂ρ.** Quoted here as USD per +0.05 correlation
        move (a small but realistic regime shift). For:

        - **Worst-of calls/puts**: typically **positive cega** for the buyer
          — higher correlation means both underlyings move together, so the
          "worst" one is closer to the "best" one, so the payoff is larger
          on average. The seller of a worst-of is short correlation.
        - **Best-of**: opposite sign.

        **The correlation curve below** shows the dual-EKO MTM at this
        moment as we vary the assumed correlation across [-0.95, +0.95],
        holding everything else constant. Bumps in this curve highlight
        non-trivial regime risk: the worst-of price is a non-linear function
        of ρ near the extremes.

        **Stress idea worth pondering.** Around the original trade date
        (March 2026), you assumed ρ ≈ 0.45 for the USDJPY × USDCNH worst-of.
        What if the realised correlation runs at 0.8 instead? Your live MTM
        is wrong by cega × 0.07 ≈ noticeably more than you'd think. And if a
        risk-off event hits, ρ could jump to 0.9 in days.
        """)

    pdf = _price_all_cached(asof_iso, DATA_DIR)
    duals = pdf[pdf["is_dual"]]

    if duals.empty:
        st.info("No multi-asset / correlation-sensitive trades in the book.")
    else:
        # ---- top-line cega ----
        gdf = _greeks_all_cached(asof_iso, DATA_DIR)
        cega = (gdf[gdf["is_dual"]]
                  .drop_duplicates("trade_id")
                  [["trade_id", "structure_kind", "cega_usd_per_5pct_rho"]])
        cega_total = cega["cega_usd_per_5pct_rho"].sum()

        c1, c2 = st.columns(2)
        c1.metric("Total book cega (USD per +0.05 ρ)",
                   f"${cega_total:+,.0f}")
        c2.metric("# multi-asset trades", str(len(duals)))

        st.dataframe(
            cega.rename(columns={
                "trade_id": "ID", "structure_kind": "Kind",
                "cega_usd_per_5pct_rho": "Cega ($ / +0.05 ρ)",
            }),
            hide_index=True, width='stretch',
            column_config={"Cega ($ / +0.05 ρ)":
                            st.column_config.NumberColumn(format="$%,.0f")},
        )

        st.divider()

        # ---- ρ-sweep MTM curve for each dual ----
        st.subheader("MTM vs correlation — sweep curves")

        snap_now = _snapshot_cached(asof_iso, DATA_DIR)
        trades_obj = build_portfolio()
        rho_grid = np.linspace(-0.95, 0.95, 21)

        @st.cache_data(show_spinner=False)
        def _rho_sweep(asof_iso, trade_id):
            snap = _snapshot_cached(asof_iso, DATA_DIR)
            trades = build_portfolio()
            t = next(tr for tr in trades if tr.trade_id == trade_id)
            asof = pd.Timestamp(asof_iso)
            vals = []
            for r in rho_grid:
                bump = {"rho": (r - (t.rho_traded or 0.5))}
                vals.append(price_trade(t, snap, asof, bump=bump)["mtm_usd"])
            return rho_grid, np.array(vals), (t.rho_traded or 0.5)

        n_cols = 2
        dual_ids = duals["trade_id"].tolist()
        for i in range(0, len(dual_ids), n_cols):
            cols_pl = st.columns(n_cols)
            for j, c in enumerate(cols_pl):
                if i + j >= len(dual_ids):
                    break
                tid = dual_ids[i + j]
                xs, ys, rho_traded = _rho_sweep(asof_iso, tid)
                t_info = duals[duals["trade_id"] == tid].iloc[0]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(width=2.5, color="#1f77b4"),
                    name="MTM",
                ))
                fig.add_vline(x=rho_traded, line=dict(color="orange", width=2,
                                                       dash="dash"),
                              annotation_text=f"ρ traded = {rho_traded:.2f}")
                # current ρ (the price uses rho_traded as the "current" assumption)
                fig.add_hline(y=0, line=dict(color="grey", width=1))
                fig.update_layout(
                    title=f"{tid} — {t_info['structure_kind']} "
                          f"{t_info['pair']}",
                    height=320,
                    xaxis_title="Correlation ρ",
                    yaxis_title="MTM (USD)",
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                fig.update_yaxes(tickformat="$,.0f")
                c.plotly_chart(fig, width='stretch')

        st.divider()

        # ---- correlation regime stress table ----
        st.subheader("Correlation regime stress")
        st.caption(
            "Book MTM at common stress correlations. ρ=0 mimics a regime "
            "where pairs decouple; ρ=0.9 mimics a global-USD or risk-off "
            "regime where everything moves together."
        )

        regimes = [-0.5, 0.0, 0.3, 0.5, 0.7, 0.9]
        rows = []
        for r in regimes:
            total = 0.0
            for tid in dual_ids:
                t = next(tr for tr in trades_obj if tr.trade_id == tid)
                rho_d = (r - (t.rho_traded or 0.5))
                v = price_trade(t, snap_now, asof, bump={"rho": rho_d})["mtm_usd"]
                total += v
            rows.append({"ρ": r, "Dual-book MTM": total})
        regime_df = pd.DataFrame(rows)
        base_rho_mtm = duals["mtm_usd"].sum()
        regime_df["ΔMTM vs base"] = regime_df["Dual-book MTM"] - base_rho_mtm

        st.dataframe(
            regime_df, hide_index=True, width='stretch',
            column_config={
                "ρ": st.column_config.NumberColumn(format="%.2f"),
                "Dual-book MTM": st.column_config.NumberColumn(format="$%,.0f"),
                "ΔMTM vs base": st.column_config.NumberColumn(format="$%,.0f"),
            },
        )
with TAB_DRILL:
    with st.expander("💡 What does this view tell me?", expanded=False):
        st.markdown("""
        Single-trade view with everything in one place: pricing inputs, full
        Greeks, barrier diagnostics, and an MTM history. Use this when one
        of the aggregate dashboards flags a trade — drill in here to
        understand exactly what's driving it.
        """)

    pdf = _price_all_cached(asof_iso, DATA_DIR)
    ticker_map = pdf.set_index("trade_id")["ticker"].to_dict()
    tid = st.selectbox(
        "Pick a trade",
        options=pdf["trade_id"].tolist(),
        format_func=lambda x: f"{x} — {ticker_map.get(x, x)}",
    )

    trades_obj = build_portfolio()
    trade = next(t for t in trades_obj if t.trade_id == tid)
    snap_now = _snapshot_cached(asof_iso, DATA_DIR)

    # ---- Trade card ----
    info = pdf[pdf["trade_id"] == tid].iloc[0]
    st.markdown(f"### {info['trade_id']} — {info['ticker']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MTM",
              f"${info['mtm_usd']/1e3:+,.0f}k",
              help="Option current value (positive for ITM long)")
    # Premium displayed as negative cash flow (long → paid out)
    c2.metric("Premium (cash flow)",
              f"$-{info['premium_paid_usd']/1e3:,.0f}k",
              help="Negative because we paid premium at booking")
    c3.metric("P&L (MTM + Premium)",
              f"${info['pnl_vs_premium_usd']/1e3:+,.0f}k",
              help="= MTM + Premium (premium already signed)")
    c4.metric("Notional", f"${info['notional_usd']/1e6:.0f}M")

    st.markdown(
        f"**Booked:** {info['booking_date'].strftime('%Y-%m-%d')}  ·  "
        f"**Expiry:** {info['expiry_date'].strftime('%Y-%m-%d')}  ·  "
        f"**Days to expiry:** {info['days_to_expiry']}  ·  "
        f"**Side:** {info['side']}"
    )
    st.markdown(f"_{info['notes']}_")

    # ---- Legs table ----
    st.subheader("Legs")
    leg_rows = []
    for i, leg in enumerate(trade.legs):
        leg_rows.append({
            "Pair": trade.pair, "Leg #": i+1, "Type": leg.opt,
            "Strike": leg.K, "Qty": leg.qty,
            "Barrier": leg.H if leg.H else "—",
            "Direction": leg.bar_dir or "—",
        })
    if trade.legs2:
        for i, leg in enumerate(trade.legs2):
            leg_rows.append({
                "Pair": trade.pair2, "Leg #": i+1, "Type": leg.opt,
                "Strike": leg.K, "Qty": leg.qty,
                "Barrier": leg.H if leg.H else "—",
                "Direction": leg.bar_dir or "—",
            })
    st.dataframe(pd.DataFrame(leg_rows), hide_index=True,
                  width='stretch')

    # ---- Greeks ----
    st.subheader("Greeks (USD)")
    g = compute_greeks(trade, snap_now, asof)
    greek_rows = []
    for pair, gp in g["by_pair"].items():
        greek_rows.append({
            "Pair": pair,
            "Δ /1%": gp["delta_usd_per_1pct"],
            "Γ /1%²": gp["gamma_usd_per_1pct2"],
            "V /volpt": gp["vega_usd_per_volpt"],
            "Vanna": gp["vanna_usd_per_1pct_x_volpt"],
            "Volga": gp["volga_usd_per_volpt2"],
            "Charm /day": gp["charm_usd_per_day"],
        })
    greek_df = pd.DataFrame(greek_rows)
    st.dataframe(
        greek_df, hide_index=True, width='stretch',
        column_config={c: st.column_config.NumberColumn(format="$%,.0f")
                       for c in greek_df.columns if c != "Pair"},
    )
    cc1, cc2 = st.columns(2)
    cc1.metric("Θ Theta (per day)", f"${g['theta_usd_per_day']:,.0f}")
    if "cega_usd_per_5pct_rho" in g:
        cc2.metric("Cega (per +0.05 ρ)",
                    f"${g['cega_usd_per_5pct_rho']:+,.0f}")

    # ---- Barrier diagnostics ----
    bars = barrier_diagnostics(trade, snap_now, asof)
    if bars:
        st.subheader("Barrier diagnostics")
        rows = []
        for b in bars:
            rows.append({
                "Pair": b["pair"], "Spot": b["S"], "Strike": b["K"],
                "Barrier": b["H"], "Direction": b["bar_dir"],
                "Distance %": b["distance_pct"]*100,
                "σ√T units": b["distance_vol_sd"],
                "Survival %": b["survival_prob"]*100,
            })
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, width='stretch',
            column_config={
                "Spot": st.column_config.NumberColumn(format="%.4f"),
                "Strike": st.column_config.NumberColumn(format="%.4f"),
                "Barrier": st.column_config.NumberColumn(format="%.4f"),
                "Distance %": st.column_config.NumberColumn(format="%+.2f%%"),
                "σ√T units": st.column_config.NumberColumn(format="%.2f"),
                "Survival %": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

    # ---- Dual-EKO joint-spot MTM surface (contour + 3D toggle) ------
    # A single-pair P&L line (like the Scenarios tab) is misleading for a
    # worst-of: pair2's spot is held constant, which hides the joint
    # geometry. A 2D shock surface captures the actual structure:
    #   - Two cliffs (one per pair's UO/DO barrier) — wherever either
    #     pair touches its barrier, the cell goes to zero/intrinsic
    #   - The "worst-of" payoff floor — the value is dragged toward the
    #     weaker leg even when the stronger leg is deep ITM
    #   - The correlation effect — visible as diagonal vs orthogonal
    #     iso-payoff contours (high ρ → diagonal, low ρ → orthogonal)
    if trade.structure == "dual_eko":
        st.subheader("Joint-spot MTM surface")
        st.caption(
            f"How MTM responds when **both** spots move independently. "
            f"The single-line P&L profile in the Scenarios tab holds "
            f"pair2 constant, which masks the joint geometry. This "
            f"surface shocks {trade.pair} and {trade.pair2} together "
            f"on a ±10% grid (15×15 = 225 re-prices, 10k MC paths each, "
            f"cached). Each spot's barrier shows as a dashed cliff; the "
            f"intersection is the 'safe corner' where neither pair has "
            f"knocked. Contour by default; 3D surface in the expander "
            f"below."
        )

        @st.cache_data(show_spinner="Computing joint-spot surface "
                                     "(225 MC re-prices)…")
        def _dual_mtm_surface(trade_id: str, asof_iso: str,
                              n_grid: int = 15,
                              shock_pct: float = 0.10) -> dict:
            """2D MTM surface for a dual-EKO trade across joint spot
            shocks. Returns axes, surface, current spots, barrier/strike
            levels."""
            asof_ts = pd.Timestamp(asof_iso)
            snap_local = _snapshot_cached(asof_iso, DATA_DIR)
            t_obj = next(t for t in build_portfolio()
                         if t.trade_id == trade_id)
            S1_0 = snap_local[t_obj.pair]["spot"]
            S2_0 = snap_local[t_obj.pair2]["spot"]
            if S1_0 is None or S2_0 is None:
                return {}
            shocks = np.linspace(-shock_pct, shock_pct, n_grid)
            S1_axis = S1_0 * (1.0 + shocks)
            S2_axis = S2_0 * (1.0 + shocks)
            # Build the surface — note np.meshgrid w/ indexing='ij'
            # would invert axes vs plotly's expected orientation. We use
            # surface[j, i] = MTM(S1_axis[i], S2_axis[j]) so y=S2, x=S1.
            surface = np.full((n_grid, n_grid), np.nan)
            for i, s1 in enumerate(S1_axis):
                for j, s2 in enumerate(S2_axis):
                    bump = {"spot": {t_obj.pair: s1 - S1_0,
                                     t_obj.pair2: s2 - S2_0}}
                    try:
                        r = price_trade(t_obj, snap_local, asof_ts,
                                        bump=bump)
                        surface[j, i] = r["mtm_usd"]
                    except Exception:
                        surface[j, i] = np.nan
            return {
                "S1_axis": S1_axis, "S2_axis": S2_axis,
                "surface_usd": surface,
                "S1_0": S1_0, "S2_0": S2_0,
                "pair1": t_obj.pair, "pair2": t_obj.pair2,
                "K1": t_obj.legs[0].K, "K2": t_obj.legs2[0].K,
                "H1": t_obj.legs[0].H, "H2": t_obj.legs2[0].H,
                "bar1": t_obj.legs[0].bar_dir,
                "bar2": t_obj.legs2[0].bar_dir,
                "rho": t_obj.rho_traded,
            }

        surf = _dual_mtm_surface(trade.trade_id, asof_iso)
        if not surf:
            st.warning(
                "Spot data missing for one of the legs at as-of date — "
                "can't compute the surface."
            )
        else:
            mtm_m = surf["surface_usd"] / 1e6  # USD millions
            S1_axis = surf["S1_axis"]
            S2_axis = surf["S2_axis"]
            nd1 = quote_decimals(surf["pair1"])
            nd2 = quote_decimals(surf["pair2"])

            # --- Contour plot (default view) ---
            fig_c = go.Figure(data=go.Contour(
                x=S1_axis, y=S2_axis, z=mtm_m,
                colorscale="RdYlGn",
                contours=dict(
                    showlabels=True,
                    labelfont=dict(size=10, color="black"),
                ),
                colorbar=dict(title="MTM ($M)"),
                hovertemplate=(
                    f"{surf['pair1']}: %{{x:.{nd1}f}}<br>"
                    f"{surf['pair2']}: %{{y:.{nd2}f}}<br>"
                    "MTM: $%{z:+.3f}M<extra></extra>"
                ),
            ))
            # Current spot crosshair
            fig_c.add_vline(
                x=surf["S1_0"],
                line=dict(color="black", width=1.5, dash="dot"),
                annotation_text=f"S₁={surf['S1_0']:.{nd1}f}",
                annotation_position="top",
            )
            fig_c.add_hline(
                y=surf["S2_0"],
                line=dict(color="black", width=1.5, dash="dot"),
                annotation_text=f"S₂={surf['S2_0']:.{nd2}f}",
                annotation_position="right",
            )
            # Barriers (only annotate if within the displayed range)
            if S1_axis.min() <= surf["H1"] <= S1_axis.max():
                fig_c.add_vline(
                    x=surf["H1"],
                    line=dict(color="red", width=2, dash="dash"),
                    annotation_text=f"H₁={surf['H1']:.{nd1}f}",
                    annotation_position="bottom",
                )
            if S2_axis.min() <= surf["H2"] <= S2_axis.max():
                fig_c.add_hline(
                    y=surf["H2"],
                    line=dict(color="red", width=2, dash="dash"),
                    annotation_text=f"H₂={surf['H2']:.{nd2}f}",
                    annotation_position="left",
                )
            # Strikes (lighter dashes; only if within range)
            if S1_axis.min() <= surf["K1"] <= S1_axis.max():
                fig_c.add_vline(
                    x=surf["K1"],
                    line=dict(color="blue", width=1, dash="dashdot"),
                    opacity=0.7,
                )
            if S2_axis.min() <= surf["K2"] <= S2_axis.max():
                fig_c.add_hline(
                    y=surf["K2"],
                    line=dict(color="blue", width=1, dash="dashdot"),
                    opacity=0.7,
                )
            rho_str = (f"ρ_traded={surf['rho']:.2f}"
                       if surf["rho"] is not None else "ρ unset")
            fig_c.update_layout(
                title=(f"{trade.trade_id} MTM ($M) — {surf['pair1']} × "
                       f"{surf['pair2']} joint-spot shock surface · "
                       f"{rho_str}"),
                xaxis_title=f"{surf['pair1']} spot",
                yaxis_title=f"{surf['pair2']} spot",
                height=560,
                margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig_c, width='stretch')

            st.caption(
                f"**Reading the surface**: black dotted crosshair = "
                f"current spot ({surf['pair1']} {surf['S1_0']:.{nd1}f}, "
                f"{surf['pair2']} {surf['S2_0']:.{nd2}f}). Red dashed = "
                f"barriers (H₁={surf['H1']:.{nd1}f}, "
                f"H₂={surf['H2']:.{nd2}f}). Blue dash-dot = strikes "
                f"(K₁={surf['K1']:.{nd1}f}, K₂={surf['K2']:.{nd2}f}). "
                f"Green = positive MTM, red = negative or near-zero "
                f"(knocked). The corner where green concentrates is "
                f"the structure's payoff sweet spot."
            )

            # --- 3D surface (in expander) ---
            with st.expander("🌄 View as 3D surface", expanded=False):
                st.caption(
                    "Same data, perspective rendering. Hold-and-drag to "
                    "rotate; scroll to zoom. The 3D view makes barrier "
                    "cliffs visually striking but contours are easier "
                    "for reading off precise MTM values."
                )
                fig_3d = go.Figure(data=go.Surface(
                    x=S1_axis, y=S2_axis, z=mtm_m,
                    colorscale="RdYlGn",
                    colorbar=dict(title="MTM ($M)"),
                    hovertemplate=(
                        f"{surf['pair1']}: %{{x:.{nd1}f}}<br>"
                        f"{surf['pair2']}: %{{y:.{nd2}f}}<br>"
                        "MTM: $%{z:+.3f}M<extra></extra>"
                    ),
                    contours=dict(
                        z=dict(show=True, usecolormap=True,
                               highlightcolor="white", project_z=True),
                    ),
                ))
                # Mark current spot in 3D with a vertical line
                z_min = float(np.nanmin(mtm_m))
                z_max = float(np.nanmax(mtm_m))
                # Find the MTM at current spot (closest grid point)
                i_curr = int(np.argmin(np.abs(S1_axis - surf["S1_0"])))
                j_curr = int(np.argmin(np.abs(S2_axis - surf["S2_0"])))
                z_curr = float(mtm_m[j_curr, i_curr])
                fig_3d.add_trace(go.Scatter3d(
                    x=[surf["S1_0"], surf["S1_0"]],
                    y=[surf["S2_0"], surf["S2_0"]],
                    z=[z_min, z_max],
                    mode="lines+markers",
                    line=dict(color="black", width=4),
                    marker=dict(size=[3, 6], color="black"),
                    name=f"current ($-{abs(z_curr):.2f}M MTM)"
                          if z_curr < 0
                          else f"current (+${z_curr:.2f}M MTM)",
                    hovertemplate=(
                        f"current spot<br>"
                        f"{surf['pair1']}: {surf['S1_0']:.{nd1}f}<br>"
                        f"{surf['pair2']}: {surf['S2_0']:.{nd2}f}<br>"
                        f"MTM: $%{{z:+.3f}}M<extra></extra>"
                    ),
                ))
                fig_3d.update_layout(
                    scene=dict(
                        xaxis_title=surf["pair1"],
                        yaxis_title=surf["pair2"],
                        zaxis_title="MTM ($M)",
                        camera=dict(eye=dict(x=1.6, y=-1.4, z=0.9)),
                    ),
                    height=620,
                    margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig_3d, width='stretch')

        # ---- Joint Greek surfaces (Δ, Γ, V per pair) ----------------
        # The MTM surface above tells you what the book is worth at any
        # (S1, S2). These tell you what your *risk profile* is at any
        # (S1, S2) — i.e. what would my Greeks become if both spots
        # rallied 5%? That answers questions like "where in the joint
        # spot plane does my delta flip sign? Where is gamma maximised?
        # Where am I shortest vega?". These flip surfaces are where the
        # exotic-specific risk really shows: near-barrier negative
        # gamma cliffs are the textbook example.
        #
        # WHY a coarser grid: each cell here is 4-6 MC re-prices (one
        # for each finite-diff bump in spot and vol), versus 1 for MTM.
        # A 9×9 grid means ~324 MC re-prices vs ~225 for MTM — still
        # cached, but capped to keep first-load time under ~60s.
        if trade.structure == "dual_eko":
            st.subheader("Joint Greek surfaces")
            st.caption(
                f"Same joint-spot grid as the MTM surface above, but "
                f"shows your **risk profile** at every (S₁, S₂) instead "
                f"of MTM. The Δ surface answers: 'if both spots rally "
                f"5%, what's my new delta in each leg?'. The Γ and V "
                f"surfaces show where curvature lives — typically a "
                f"sharp negative-gamma ridge right before each barrier. "
                f"9×9 grid (each cell = 4 MC re-prices, cached)."
            )

            @st.cache_data(show_spinner="Computing joint Greek surfaces "
                                          "(≈324 MC re-prices)…")
            def _dual_greek_surfaces(trade_id: str, asof_iso: str,
                                       n_grid: int = 9,
                                       shock_pct: float = 0.10) -> dict:
                """Δ/Γ/V per pair on the joint spot grid. Returns
                axes + 6 surfaces keyed by (greek, pair_label)."""
                asof_ts = pd.Timestamp(asof_iso)
                snap_local = _snapshot_cached(asof_iso, DATA_DIR)
                t_obj = next(t for t in build_portfolio()
                             if t.trade_id == trade_id)
                S1_0 = snap_local[t_obj.pair]["spot"]
                S2_0 = snap_local[t_obj.pair2]["spot"]
                if S1_0 is None or S2_0 is None:
                    return {}
                shocks = np.linspace(-shock_pct, shock_pct, n_grid)
                S1_axis = S1_0 * (1.0 + shocks)
                S2_axis = S2_0 * (1.0 + shocks)
                # Surfaces — indexing same as MTM surface
                # (surface[j, i] = G(S1_axis[i], S2_axis[j]))
                shape = (n_grid, n_grid)
                surf = {
                    ("delta", t_obj.pair): np.full(shape, np.nan),
                    ("delta", t_obj.pair2): np.full(shape, np.nan),
                    ("gamma", t_obj.pair): np.full(shape, np.nan),
                    ("gamma", t_obj.pair2): np.full(shape, np.nan),
                    ("vega", t_obj.pair): np.full(shape, np.nan),
                    ("vega", t_obj.pair2): np.full(shape, np.nan),
                }
                ds_pct = 0.005   # 0.5% relative bump for Δ/Γ
                dv = 0.005       # 0.5 volpt bump for V
                for i, s1 in enumerate(S1_axis):
                    for j, s2 in enumerate(S2_axis):
                        b1, b2 = s1 - S1_0, s2 - S2_0
                        # Base price at this point
                        base_b = {"spot": {t_obj.pair: b1,
                                            t_obj.pair2: b2}}
                        try:
                            base = price_trade(t_obj, snap_local,
                                                 asof_ts,
                                                 bump=base_b)["mtm_usd"]
                            # Δ/Γ for pair1 — bump pair1's spot only
                            dS1 = s1 * ds_pct
                            up1 = price_trade(t_obj, snap_local,
                                              asof_ts,
                                              bump={"spot": {
                                                  t_obj.pair: b1 + dS1,
                                                  t_obj.pair2: b2,
                                              }})["mtm_usd"]
                            dn1 = price_trade(t_obj, snap_local,
                                              asof_ts,
                                              bump={"spot": {
                                                  t_obj.pair: b1 - dS1,
                                                  t_obj.pair2: b2,
                                              }})["mtm_usd"]
                            surf[("delta", t_obj.pair)][j, i] = (
                                (up1 - dn1) / (2 * dS1) * s1 * 0.01
                            )
                            surf[("gamma", t_obj.pair)][j, i] = (
                                (up1 + dn1 - 2 * base) / (dS1 ** 2)
                                * (s1 * 0.01) ** 2
                            )
                            # Δ/Γ for pair2 — bump pair2's spot only
                            dS2 = s2 * ds_pct
                            up2 = price_trade(t_obj, snap_local,
                                              asof_ts,
                                              bump={"spot": {
                                                  t_obj.pair: b1,
                                                  t_obj.pair2: b2 + dS2,
                                              }})["mtm_usd"]
                            dn2 = price_trade(t_obj, snap_local,
                                              asof_ts,
                                              bump={"spot": {
                                                  t_obj.pair: b1,
                                                  t_obj.pair2: b2 - dS2,
                                              }})["mtm_usd"]
                            surf[("delta", t_obj.pair2)][j, i] = (
                                (up2 - dn2) / (2 * dS2) * s2 * 0.01
                            )
                            surf[("gamma", t_obj.pair2)][j, i] = (
                                (up2 + dn2 - 2 * base) / (dS2 ** 2)
                                * (s2 * 0.01) ** 2
                            )
                            # Vega per pair
                            vu1 = price_trade(t_obj, snap_local,
                                              asof_ts,
                                              bump={"spot": {
                                                      t_obj.pair: b1,
                                                      t_obj.pair2: b2},
                                                  "vol": {t_obj.pair: +dv}}
                                              )["mtm_usd"]
                            vd1 = price_trade(t_obj, snap_local,
                                              asof_ts,
                                              bump={"spot": {
                                                      t_obj.pair: b1,
                                                      t_obj.pair2: b2},
                                                  "vol": {t_obj.pair: -dv}}
                                              )["mtm_usd"]
                            surf[("vega", t_obj.pair)][j, i] = (
                                (vu1 - vd1) / (2 * dv) * 0.01
                            )
                            vu2 = price_trade(t_obj, snap_local,
                                              asof_ts,
                                              bump={"spot": {
                                                      t_obj.pair: b1,
                                                      t_obj.pair2: b2},
                                                  "vol": {t_obj.pair2: +dv}}
                                              )["mtm_usd"]
                            vd2 = price_trade(t_obj, snap_local,
                                              asof_ts,
                                              bump={"spot": {
                                                      t_obj.pair: b1,
                                                      t_obj.pair2: b2},
                                                  "vol": {t_obj.pair2: -dv}}
                                              )["mtm_usd"]
                            surf[("vega", t_obj.pair2)][j, i] = (
                                (vu2 - vd2) / (2 * dv) * 0.01
                            )
                        except Exception:
                            pass
                return {
                    "S1_axis": S1_axis, "S2_axis": S2_axis,
                    "S1_0": S1_0, "S2_0": S2_0,
                    "pair1": t_obj.pair, "pair2": t_obj.pair2,
                    "H1": t_obj.legs[0].H, "H2": t_obj.legs2[0].H,
                    "K1": t_obj.legs[0].K, "K2": t_obj.legs2[0].K,
                    "surfaces": surf,
                }

            gsurf = _dual_greek_surfaces(trade.trade_id, asof_iso)
            if not gsurf:
                st.warning("Spot data missing — can't compute "
                            "Greek surfaces.")
            else:
                # Greek picker — let user choose which Greek to display
                greek_pick = st.radio(
                    "Greek to display",
                    ["Δ Delta", "Γ Gamma", "V Vega"],
                    horizontal=True, key="dual_greek_pick",
                )
                greek_key = {"Δ Delta": "delta", "Γ Gamma": "gamma",
                             "V Vega": "vega"}[greek_pick]
                unit_label = {
                    "delta": "USD per 1% spot",
                    "gamma": "USD per 1%² spot",
                    "vega": "USD per volpt",
                }[greek_key]

                # Two side-by-side contours (one per pair)
                from plotly.subplots import make_subplots
                fig_g = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=(
                        f"{greek_pick} on {gsurf['pair1']}",
                        f"{greek_pick} on {gsurf['pair2']}",
                    ),
                    horizontal_spacing=0.14,
                )
                nd1 = quote_decimals(gsurf["pair1"])
                nd2 = quote_decimals(gsurf["pair2"])

                for col_idx, (pair_lbl, ndp) in enumerate(
                    [(gsurf["pair1"], nd1), (gsurf["pair2"], nd2)],
                    start=1,
                ):
                    z = gsurf["surfaces"][(greek_key, pair_lbl)] / 1e3
                    fig_g.add_trace(
                        go.Contour(
                            x=gsurf["S1_axis"], y=gsurf["S2_axis"], z=z,
                            colorscale="RdBu",
                            zmid=0,
                            contours=dict(showlabels=True,
                                            labelfont=dict(size=9,
                                                            color="black")),
                            colorbar=dict(
                                title=f"{greek_pick} ($k)<br>"
                                          f"vs {pair_lbl}",
                                x=0.46 if col_idx == 1 else 1.02,
                                len=0.85,
                            ),
                            hovertemplate=(
                                f"{gsurf['pair1']}: %{{x:.{nd1}f}}<br>"
                                f"{gsurf['pair2']}: %{{y:.{nd2}f}}<br>"
                                f"{greek_pick}: $%{{z:+.1f}}k"
                                "<extra></extra>"
                            ),
                        ),
                        row=1, col=col_idx,
                    )
                    # Current spot, barriers, strikes
                    fig_g.add_vline(
                        x=gsurf["S1_0"],
                        line=dict(color="black", width=1.5, dash="dot"),
                        row=1, col=col_idx,
                    )
                    fig_g.add_hline(
                        y=gsurf["S2_0"],
                        line=dict(color="black", width=1.5, dash="dot"),
                        row=1, col=col_idx,
                    )
                    if (gsurf["S1_axis"].min() <= gsurf["H1"]
                          <= gsurf["S1_axis"].max()):
                        fig_g.add_vline(
                            x=gsurf["H1"],
                            line=dict(color="red", width=1.5,
                                        dash="dash"),
                            row=1, col=col_idx,
                        )
                    if (gsurf["S2_axis"].min() <= gsurf["H2"]
                          <= gsurf["S2_axis"].max()):
                        fig_g.add_hline(
                            y=gsurf["H2"],
                            line=dict(color="red", width=1.5,
                                        dash="dash"),
                            row=1, col=col_idx,
                        )
                fig_g.update_xaxes(title_text=gsurf["pair1"])
                fig_g.update_yaxes(title_text=gsurf["pair2"])
                fig_g.update_layout(
                    height=480,
                    margin=dict(l=10, r=10, t=50, b=10),
                    title_text=(f"{trade.trade_id} {greek_pick} on the "
                                  f"joint-spot grid · units: {unit_label}"),
                )
                st.plotly_chart(fig_g, width='stretch')

                st.caption(
                    "**How to read these surfaces**: each panel shows "
                    "the Greek **with respect to one pair's spot**, "
                    f"plotted across the joint (S₁, S₂) grid. Blue = "
                    f"negative, red = positive, white ~ zero. The "
                    f"black dotted crosshair is current spot; red "
                    f"dashed lines are the UO/DO barriers. **What to "
                    f"look for**: (1) sign flips — places where "
                    f"crossing a contour changes which way you'd "
                    f"want spot to move; (2) sharp ridges or cliffs "
                    f"near barriers — that's where the exotic risk "
                    f"lives; (3) asymmetry between the two panels — "
                    f"if Δ on pair1 looks very different from Δ on "
                    f"pair2 at the same point, that's the worst-of "
                    f"structure picking sides."
                )

                with st.expander(
                    f"🌄 View {greek_pick} as 3D surfaces "
                    f"(both pairs)",
                    expanded=False,
                ):
                    st.caption(
                        "Same data, perspective rendering. Hold-and-"
                        "drag to rotate. Useful for spotting the "
                        "gamma cliffs / vega ridges near barriers — "
                        "the contours hide vertical scale, the 3D "
                        "view exaggerates it."
                    )
                    fig_g3d = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=(
                            f"{greek_pick} on {gsurf['pair1']}",
                            f"{greek_pick} on {gsurf['pair2']}",
                        ),
                        specs=[[{"type": "surface"},
                                {"type": "surface"}]],
                        horizontal_spacing=0.05,
                    )
                    for col_idx, pair_lbl in enumerate(
                        [gsurf["pair1"], gsurf["pair2"]], start=1,
                    ):
                        z = (gsurf["surfaces"][(greek_key, pair_lbl)]
                                / 1e3)
                        fig_g3d.add_trace(
                            go.Surface(
                                x=gsurf["S1_axis"],
                                y=gsurf["S2_axis"],
                                z=z,
                                colorscale="RdBu",
                                cmid=0,
                                showscale=(col_idx == 2),
                                colorbar=dict(
                                    title=f"{greek_pick} ($k)"
                                ) if col_idx == 2 else None,
                                hovertemplate=(
                                    f"{gsurf['pair1']}: %{{x:.{nd1}f}}<br>"
                                    f"{gsurf['pair2']}: %{{y:.{nd2}f}}<br>"
                                    f"{greek_pick}: $%{{z:+.1f}}k"
                                    "<extra></extra>"
                                ),
                            ),
                            row=1, col=col_idx,
                        )
                    fig_g3d.update_layout(
                        height=540,
                        margin=dict(l=0, r=0, t=40, b=0),
                        scene=dict(
                            xaxis_title=gsurf["pair1"],
                            yaxis_title=gsurf["pair2"],
                            zaxis_title=f"{greek_pick} ($k)",
                            camera=dict(eye=dict(x=1.6, y=-1.4, z=0.9)),
                        ),
                        scene2=dict(
                            xaxis_title=gsurf["pair1"],
                            yaxis_title=gsurf["pair2"],
                            zaxis_title=f"{greek_pick} ($k)",
                            camera=dict(eye=dict(x=1.6, y=-1.4, z=0.9)),
                        ),
                    )
                    st.plotly_chart(fig_g3d, width='stretch')

    # ---- MTM history since booking ----
    st.subheader("MTM history since booking")
    hist_dates = [d for d in _available_dates_cached(DATA_DIR)
                   if trade.booking_date <= d <= asof]
    if len(hist_dates) > 1:
        # subsample if too many to keep it fast
        if len(hist_dates) > 60:
            step = len(hist_dates) // 60
            hist_dates = hist_dates[::step]
        hist_rows = []
        for d in hist_dates:
            try:
                sp = _snapshot_cached(d.strftime("%Y-%m-%d"), DATA_DIR)
                v = price_trade(trade, sp, d)["mtm_usd"]
                hist_rows.append({"date": d, "mtm_usd": v})
            except Exception:
                pass
        if hist_rows:
            hist = pd.DataFrame(hist_rows)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist["date"], y=hist["mtm_usd"], mode="lines",
                line=dict(width=2.5, color="#1f77b4"),
            ))
            fig.add_hline(y=trade.premium_paid_usd,
                           line=dict(color="orange", width=1.5, dash="dash"),
                           annotation_text="Premium paid")
            fig.add_hline(y=0, line=dict(color="grey", width=1))
            fig.update_layout(
                height=320, xaxis_title="Date", yaxis_title="MTM (USD)",
                title=f"MTM since {trade.booking_date.strftime('%Y-%m-%d')}",
            )
            fig.update_yaxes(tickformat="$,.0f")
            st.plotly_chart(fig, width='stretch')
    else:
        st.caption("Not enough dates between booking and as-of for history.")

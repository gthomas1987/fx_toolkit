"""Shared helper for apps 3-6 (VIX-shock systematic strategies).

Each of those apps is a 1-line caller that delegates the entire UI +
backtest workflow to `run_strategy_app`. The strategy:

    On any business day t where VIX z-score over a trailing 252-day
    window exceeds `threshold`, enter a position in `pair` for the
    next `horizon` business days. Position direction = `default_sign`:
      -1 → short the pair (typical risk-off response for AUDUSD, NZDUSD,
            GBPUSD, USDNOK — i.e. EUR/USD-ish risk currencies decline
            on VIX shocks)
      +1 → long the pair

    Multiple overlapping entries allowed (the simple version) — each
    entry/exit is independent so daily P&L aggregates across however
    many trades are open.

This is a "VIX-as-signal" study, not a tradable strategy — no costs,
no slippage, no funding. The metric of interest is whether the signal
direction historically had positive expected return on `pair`.

Public API:
    run_strategy_app(pair, app_number, default_sign,
                       default_threshold, default_horizon)

Data requirements:
    SPOT for `pair` (used to compute returns)
    SPOT for "VIX" — special pair name in the data folder. App will
    error gracefully if VIX isn't present.

If you don't have a VIX panel, drop a `VIX_SPOT.csv` (date + close
columns) into the data folder — the loader will pick it up.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.ts_loader import load_panel
from core.ui import data_dir_input, app_header


def run_strategy_app(pair: str,
                          app_number: str,
                          default_sign: int = -1,
                          default_threshold: float = 1.5,
                          default_horizon: int = 2) -> None:
    """Top-level entry point for apps 3-6. Fully self-contained:
    renders the page, loads data, runs the backtest, displays results.

    Args:
        pair: the FX pair to trade (e.g. "AUDUSD", "USDNOK")
        app_number: string for the title (e.g. "3", "4")
        default_sign: -1 (short pair on VIX spike) or +1 (long)
        default_threshold: VIX z-score threshold for entry
        default_horizon: holding period in business days
    """
    st.set_page_config(
        page_title=f"{app_number} · {pair} VIX shock",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    direction_word = "short" if default_sign < 0 else "long"
    app_header(
        f"{app_number} · {pair} VIX-shock systematic strategy",
        f"Default: go **{direction_word} {pair}** for {default_horizon} "
        f"business days after VIX z-score (252d) > {default_threshold:g}",
    )

    folder = data_dir_input()
    if folder is None:
        st.stop()

    # ----- Sidebar controls -----
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Signal parameters**")
    sign_choice = st.sidebar.radio(
        f"Trade direction on shock",
        ["Short " + pair, "Long " + pair],
        index=0 if default_sign < 0 else 1,
        help=("On a VIX shock, do we expect the pair to rally or sell off? "
               "For risk-off pairs (AUDUSD, NZDUSD, GBPUSD, USDNOK as "
               "USD-weakness pairs) the historical pattern is sell-off "
               "after a VIX shock."),
    )
    sign = -1 if sign_choice.startswith("Short") else 1

    threshold = st.sidebar.slider(
        "VIX z-score threshold",
        min_value=0.5, max_value=4.0, value=float(default_threshold),
        step=0.1,
        help=("z = (VIX − rolling_mean) / rolling_std. Threshold ~1.5 "
               "≈ top decile shocks; higher = rarer, more extreme."),
    )
    z_window = st.sidebar.slider(
        "z-score window (days)",
        min_value=63, max_value=504, value=252, step=21,
        help="Trailing window for VIX z-score normalisation.",
    )
    horizon = st.sidebar.slider(
        "Holding horizon (business days)",
        min_value=1, max_value=20, value=int(default_horizon),
    )

    st.sidebar.markdown("**Date range**")
    # Date range — load spot first to discover available range
    spot_df = load_panel(folder, "SPOT", None, pairs=(pair,))
    if spot_df.empty or pair not in spot_df.columns:
        st.error(f"No SPOT data for {pair} in `{folder}`. Need a row "
                  f"in `_index.csv` with pair={pair}, category=SPOT.")
        return
    pair_spot = spot_df[pair].dropna()

    vix_df = load_panel(folder, "SPOT", None, pairs=("VIX",))
    if vix_df.empty or "VIX" not in vix_df.columns:
        st.error(
            "No VIX data found. Drop a VIX CSV file into the data folder "
            "(name: `VIX_SPOT.csv` or `VIX.csv`) and re-run, or add a "
            "row to `_index.csv` with pair=VIX, category=SPOT."
        )
        st.caption("Hint: VIX daily close from Yahoo Finance or CBOE works.")
        return
    vix = vix_df["VIX"].dropna()

    # Common date range across pair + VIX
    common = pair_spot.index.intersection(vix.index)
    if len(common) < z_window + horizon + 50:
        st.error(f"Not enough overlap between {pair} and VIX "
                  f"({len(common)} days). Need at least "
                  f"{z_window + horizon + 50}.")
        return
    pair_spot = pair_spot.reindex(common)
    vix = vix.reindex(common)

    date_min, date_max = common.min().date(), common.max().date()
    sel = st.sidebar.date_input(
        "Backtest range", value=(date_min, date_max),
        min_value=date_min, max_value=date_max,
    )
    if isinstance(sel, tuple) and len(sel) == 2:
        start, end = pd.Timestamp(sel[0]), pd.Timestamp(sel[1])
        pair_spot = pair_spot.loc[start:end]
        vix = vix.loc[start:end]
        common = pair_spot.index

    # ----- Backtest -----
    # 1. Compute trailing-z-score of VIX
    rolling_mean = vix.rolling(z_window, min_periods=max(20, z_window // 4)).mean()
    rolling_std = vix.rolling(z_window, min_periods=max(20, z_window // 4)).std()
    z = (vix - rolling_mean) / rolling_std
    z = z.dropna()

    # 2. Find shock days (z > threshold). Use crossing semantics —
    # only fire on the day z FIRST exceeds threshold, not every day
    # while z stays above (avoids stacking 10 trades on a 5-day spike).
    shock_mask = (z > threshold) & (z.shift(1) <= threshold)
    shock_dates = z.index[shock_mask.fillna(False).values]

    # 3. For each shock, compute pair return over `horizon` bdays
    # signed by the chosen direction. Returns are log-returns for
    # additive aggregation; trades are independent (no compounding
    # across overlapping trades).
    trades = []
    log_pair = np.log(pair_spot.astype(float))
    for sd in shock_dates:
        try:
            i_open = pair_spot.index.get_loc(sd)
        except KeyError:
            continue
        i_close = i_open + horizon
        if i_close >= len(pair_spot):
            continue
        open_dt = pair_spot.index[i_open]
        close_dt = pair_spot.index[i_close]
        open_px = float(pair_spot.iloc[i_open])
        close_px = float(pair_spot.iloc[i_close])
        log_ret = float(log_pair.iloc[i_close] - log_pair.iloc[i_open])
        signed_ret_pct = sign * log_ret * 100.0
        trades.append({
            "open_date": open_dt,
            "close_date": close_dt,
            "vix_z_at_open": float(z.loc[open_dt]),
            "vix_at_open": float(vix.loc[open_dt]),
            "open_spot": open_px,
            "close_spot": close_px,
            "raw_log_return_pct": log_ret * 100.0,
            "signed_return_pct": signed_ret_pct,
        })
    trades_df = pd.DataFrame(trades)

    # ----- Render -----
    st.markdown(f"### Strategy: **{'Short' if sign < 0 else 'Long'} "
                f"{pair}** for **{horizon} bday(s)** when **VIX z > "
                f"{threshold:.2f}** ({z_window}d window)")

    if trades_df.empty:
        st.warning("No shock events in the selected range with these "
                   "parameters. Try lowering the threshold or widening "
                   "the date range.")
        return

    # Headline metrics
    n_trades = len(trades_df)
    win_rate = (trades_df["signed_return_pct"] > 0).mean() * 100
    mean_ret = trades_df["signed_return_pct"].mean()
    median_ret = trades_df["signed_return_pct"].median()
    total_ret = trades_df["signed_return_pct"].sum()
    sharpe_per_trade = (trades_df["signed_return_pct"].mean()
                          / trades_df["signed_return_pct"].std()
                          if trades_df["signed_return_pct"].std() > 0 else 0)
    annual_factor = np.sqrt(252.0 / max(horizon, 1))
    sharpe_annual = sharpe_per_trade * annual_factor

    cs = st.columns(6)
    cs[0].metric("Shocks", f"{n_trades}",
                  f"~{n_trades / max(len(common), 1) * 252:.1f}/yr")
    cs[1].metric("Win rate", f"{win_rate:.0f}%")
    cs[2].metric("Mean return", f"{mean_ret:+.2f}%",
                  f"per trade ({horizon}bd)")
    cs[3].metric("Median", f"{median_ret:+.2f}%")
    cs[4].metric("Total", f"{total_ret:+.1f}%",
                  "sum of all trades")
    cs[5].metric("Sharpe (ann.)", f"{sharpe_annual:+.2f}",
                  f"per-trade × √(252/{horizon})")

    st.divider()

    # Cumulative equity curve (simple: cumulative sum of signed log returns)
    # Place each trade's return on its CLOSE date for a "P&L crystallisation" view.
    pnl_by_close = (trades_df.set_index("close_date")["signed_return_pct"]
                      .sort_index())
    cum = pnl_by_close.cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pair_spot.index, y=pair_spot.values,
        mode="lines", name=f"{pair} spot",
        line=dict(color="#1f1f1f", width=1.2),
        yaxis="y1",
    ))
    # Shock markers
    if len(trades_df):
        fig.add_trace(go.Scatter(
            x=trades_df["open_date"],
            y=[pair_spot.loc[d] for d in trades_df["open_date"]],
            mode="markers", name="VIX shock entries",
            marker=dict(color="#d62728", size=8, symbol="triangle-up"),
            yaxis="y1",
        ))
    fig.add_trace(go.Scatter(
        x=cum.index, y=cum.values, mode="lines",
        name="Cumulative strategy P&L (%)",
        line=dict(color="#2ca02c", width=2),
        yaxis="y2", fill="tozeroy",
        fillcolor="rgba(44,160,44,0.10)",
    ))
    fig.update_layout(
        height=500,
        yaxis=dict(title=f"{pair} spot", side="left"),
        yaxis2=dict(title="Cum strategy P&L (%)", side="right",
                    overlaying="y", showgrid=False, zeroline=True),
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # VIX z + threshold chart
    z_fig = go.Figure()
    z_fig.add_trace(go.Scatter(
        x=z.index, y=z.values, mode="lines",
        line=dict(color="#1f77b4", width=1.4),
        name=f"VIX z (window={z_window}d)",
    ))
    z_fig.add_hline(y=threshold, line_color="#d62728",
                     line_dash="dash", line_width=1.4,
                     annotation_text=f"threshold = {threshold:.2f}",
                     annotation_position="right")
    z_fig.add_hline(y=0, line_color="rgba(0,0,0,0.3)", line_width=0.6)
    z_fig.update_layout(
        height=240,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="VIX z-score",
        template="plotly_white", showlegend=False,
    )
    st.plotly_chart(z_fig, use_container_width=True)

    st.divider()

    # Return distribution
    cd1, cd2 = st.columns(2)
    with cd1:
        st.markdown("##### Per-trade return distribution")
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(
            x=trades_df["signed_return_pct"], nbinsx=30,
            marker_color=("#2ca02c" if mean_ret > 0 else "#d62728"),
            opacity=0.75,
        ))
        hist_fig.add_vline(x=0, line_color="black", line_width=1)
        hist_fig.add_vline(x=mean_ret, line_color="#d62728",
                            line_dash="dash",
                            annotation_text=f"mean = {mean_ret:+.2f}%",
                            annotation_position="top")
        hist_fig.update_layout(
            height=320, margin=dict(l=10, r=10, t=30, b=10),
            template="plotly_white", showlegend=False,
            xaxis_title="Signed return (%)", yaxis_title="count", bargap=0.05,
        )
        st.plotly_chart(hist_fig, use_container_width=True)
    with cd2:
        st.markdown("##### Trade ledger")
        disp = trades_df.copy()
        disp["open_date"] = disp["open_date"].dt.strftime("%Y-%m-%d")
        disp["close_date"] = disp["close_date"].dt.strftime("%Y-%m-%d")
        disp = disp[["open_date", "close_date", "vix_at_open",
                     "vix_z_at_open", "raw_log_return_pct",
                     "signed_return_pct"]]
        disp.columns = ["Open", "Close", "VIX", "VIX z", "Raw %", "Strat %"]
        st.dataframe(
            disp.style.format({
                "VIX": "{:.2f}", "VIX z": "{:+.2f}",
                "Raw %": "{:+.2f}", "Strat %": "{:+.2f}",
            }).map(
                lambda v: ("background-color: rgba(44,160,44,0.18)"
                            if isinstance(v, (int, float)) and v > 0
                            else "background-color: rgba(214,39,40,0.18)"
                            if isinstance(v, (int, float)) and v < 0
                            else ""),
                subset=["Strat %"],
            ),
            use_container_width=True, hide_index=True, height=320,
        )

    st.markdown("---")
    st.caption(
        f"Data: `{folder}` · Pair: **{pair}** · Direction: "
        f"**{'short' if sign < 0 else 'long'}** · "
        f"Threshold: **{threshold:.2f}** · Horizon: **{horizon}**bd · "
        f"Trades counted on the FIRST day z crosses above threshold (not "
        f"on each subsequent day above). Returns are log-returns × 100. "
        f"No costs / slippage."
    )

"""RKO Pricer — Backtest, Drilldown, Worst-of, and Portfolio tab impls.

Kept in a sibling module so `pages/rko_pricer.py` stays focused on
the single-trade Pricer tab. All tab functions are passed the data
folder and read shared session state. (Formerly: apps/_app12_tabs.py.)

# Architecture

- **Backtest tab**: cross-product spec inputs → runs `run_grid_american`
  → stores results in `st.session_state["rko_bt_results"]` and a meta
  dict for context.

- **Drilldown tab**: picks one strategy from the latest results →
  shows equity, drawdown, monthly/annual P&L, trade ledger with CSV
  export.

- **Worst-of + drilldown**: same shape, but on two-pair WO baskets.
  Results land in `rko_wo_results`.

- **RKO Portfolio + drilldown**: basket across pairs. Results in
  `rko_rp_results`.

- **WO-RKO Portfolio + drilldown**: basket of worst-of crosses.
  Results in `rko_wrp_results`.

We reuse the EKO Pricer ledger/summary helpers (`trades_to_df`,
`summarize_strategy`, `compute_equity_and_drawdown`, `monthly_pnl_table`,
`annual_summary_table`) — the column structure of `Trade` is identical
across both apps, so these helpers work on American trades unchanged.
"""
from __future__ import annotations

import time
from datetime import date as _date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from core.backtest import (
    StrategySpec,
    annual_summary_table,
    build_strategy_grid,
    compute_equity_and_drawdown,
    monthly_pnl_table,
    summarize_strategy,
    trades_to_df,
)
from core.backtest_american import run_grid_american
from core.data_loader import load_panel


# =============================================================================
# Color helper for monthly heatmaps — matplotlib-free
# =============================================================================
# `pandas.Styler.background_gradient` requires matplotlib, which we don't
# want as a hard dependency just for a small visual nicety. This helper
# returns a Styler with the same diverging red-yellow-green effect by
# computing CSS background colors directly from cell values.
#
# Behaviour:
#   - Symmetric diverging palette around 0
#   - Most-negative cell → strong red
#   - 0 → pale yellow (near-white)
#   - Most-positive cell → strong green
#   - NaN cells are left unstyled
# Saturation is normalised by the absolute max in the dataframe so the
# scale is per-table (same as the matplotlib cmap='RdYlGn' default).

def _diverging_red_yellow_green(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Return df.style with a diverging RYG background based on cell values."""
    # Flatten to find absolute max, ignoring NaN
    arr = df.to_numpy(dtype=float, na_value=np.nan)
    finite = arr[np.isfinite(arr)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 0.0
    if vmax == 0.0:
        return df.style  # nothing to colour

    def colour_for(v: float) -> str:
        if pd.isna(v) or not np.isfinite(v):
            return ""
        # x in [-1, 1] after normalisation
        x = max(-1.0, min(1.0, v / vmax))
        if x >= 0:
            # interpolate yellow (255,255,180) → green (60,170,80)
            r = int(255 + (60 - 255) * x)
            g = int(255 + (170 - 255) * x)
            b = int(180 + (80 - 180) * x)
        else:
            # interpolate yellow (255,255,180) → red (200,60,60)
            xm = -x
            r = int(255 + (200 - 255) * xm)
            g = int(255 + (60 - 255) * xm)
            b = int(180 + (60 - 180) * xm)
        # Dark text on these light backgrounds
        return f"background-color: rgb({r},{g},{b}); color: #1a1a1a;"

    return df.style.map(colour_for)


# =============================================================================
# Shared chart + breakdown helpers — drilldown views
# =============================================================================
# Used by both single-leg-basket (RKO Portfolio) and worst-of-basket
# (WO-RKO Portfolio) drilldowns. Mirrors the equivalent helpers in
# apps/9_ko_pricer.py so the two apps' WO drilldowns look identical.

def _fmt_usd(x) -> str:
    """Compact USD formatting: $1.23M / $12.3K / $123 / -$123."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v):
        return ""
    sign = "-" if v < 0 else ""
    a = abs(v)
    if a >= 1e9:
        return f"{sign}${a/1e9:.2f}B"
    if a >= 1e6:
        return f"{sign}${a/1e6:.2f}M"
    if a >= 1e3:
        return f"{sign}${a/1e3:.2f}K"
    return f"{sign}${a:,.0f}"


def _annual_sharpe_per_year(eq: "pd.DataFrame") -> "pd.Series":
    """Per-calendar-year Sharpe ratio from an equity curve.

    Same formula as core.backtest._consistency_metrics — monthly-resample
    the daily P&L stream, group by calendar year, then
    Sharpe_y = mean(monthly_pnl) / std(monthly_pnl) × √12 for each year y.
    Years with fewer than 2 valid monthly observations or zero std return
    NaN so the table can render '—' rather than misleading zeros.
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
    """Green/red bar chart for yearly PnL (USD)."""
    import plotly.graph_objects as go
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
    st.plotly_chart(fig, use_container_width=True)


def _render_pnl_by_pair_chart(by_pair: "pd.Series",
                                  pair_label: str = "Pair") -> None:
    """Green/red bar chart for PnL by pair (or cross)."""
    import plotly.graph_objects as go
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
    st.plotly_chart(fig, use_container_width=True)


def _render_pnl_heatmap(year_pair: "pd.DataFrame",
                            pair_label: str = "Pair") -> None:
    """Year × pair (or cross) PnL heatmap. Rows = year descending, cols = pair."""
    import plotly.graph_objects as go
    if year_pair.empty:
        st.caption("(no trades for the heatmap)")
        return
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
    st.plotly_chart(fig, use_container_width=True)


def _build_per_pair_breakdown(df: "pd.DataFrame") -> "pd.DataFrame":
    """Single-leg basket per-pair breakdown — 7 columns matching App 9's
    EKO Portfolio drilldown table. Returns a numeric DataFrame (USD as
    floats, rates as percentage values 0-100); caller formats for display
    or CSV as needed."""
    if df.empty or "pair" not in df.columns:
        return pd.DataFrame()
    out = df.groupby("pair").agg(
        n_trades=("pnl_usd", "size"),
        total_pnl_usd=("pnl_usd", "sum"),
        total_premium_usd=("premium_usd", "sum"),
        total_payoff_usd=("actual_payoff_usd", "sum"),
        ko_rate_pct=("knocked_out",
                       lambda s: 100.0 * s.astype(bool).sum() / len(s)),
        win_rate_pct=("pnl_usd",
                        lambda s: 100.0 * (s > 0).sum() / len(s)),
    ).reset_index().rename(columns={"pair": "Pair"})
    return out.sort_values("total_pnl_usd", ascending=False)


def _build_per_cross_breakdown(df: "pd.DataFrame") -> "pd.DataFrame":
    """Worst-of basket per-cross breakdown — 9 columns matching App 9's
    WO EKO Portfolio drilldown table (the screenshot). Includes the
    structure-level KO rate (either leg knocked) which is the cleanest
    way to read 'this trade KO'd' for downstream consumers."""
    if df.empty or "leg_a_pair" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work["cross"] = work["leg_a_pair"] + "×" + work["leg_b_pair"]
    work["_struct_ko"] = (work["leg_a_knocked_out"].astype(bool)
                            | work["leg_b_knocked_out"].astype(bool))
    out = work.groupby("cross").agg(
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
    return out.sort_values("total_pnl_usd", ascending=False)


def _format_breakdown_for_display(df: "pd.DataFrame") -> "pd.DataFrame":
    """Pretty-format the numeric breakdown for st.dataframe display.
    USD columns → compact strings ($1.23M); rate columns → '12%'."""
    if df.empty:
        return df
    disp = df.copy()
    for col in disp.columns:
        if col.endswith("_usd"):
            disp[col] = disp[col].apply(_fmt_usd)
        elif col.endswith("_pct"):
            disp[col] = disp[col].apply(
                lambda x: f"{x:.0f}%" if pd.notna(x) else "—")
    # Friendly column headers for display
    rename = {
        "n_trades": "n trades",
        "total_pnl_usd": "PnL",
        "total_premium_usd": "Σ Premium",
        "total_payoff_usd": "Σ Payoff",
        "leg_a_ko_rate_pct": "Leg A KO %",
        "leg_b_ko_rate_pct": "Leg B KO %",
        "structure_ko_rate_pct": "Structure KO %",
        "ko_rate_pct": "KO %",
        "win_rate_pct": "Win %",
    }
    return disp.rename(columns=rename)


def _build_portfolio_download_filename(meta: dict, prefix: str,
                                              kind: str) -> str:
    """Build a descriptive filename for portfolio bulk-download CSVs.

    Pattern: `{PREFIX}_{pair_tokens}_T{tenor1}_T{tenor2}_S{strike1}_S{strike2}_
    B{barrier1}_B{barrier2}_{kind}.csv`

    Examples (basket: USDJPY/USDKRW/USDTHB, tenors=[2M], strikes=[ATM,45D],
    barriers=[20D,15D]):
      RKO_JPY_KRW_THB_T2M_SATM_S45D_B20D_B15D_summary.csv
      EKO_JPY_KRW_THB_T2M_SATM_S45D_B20D_B15D_timeseries.csv
      WO-RKO_JPY_KRW_THB_T2M_SATM_S45D_B20D_B15D_per_cross.csv

    Pair tokens strip the USD prefix (so USDJPY → JPY). Strike/barrier
    Δ labels strip the trailing Δ character (so '20Δ' → '20D'). Tenor
    tokens stay as-is. Kind is appended verbatim ('summary',
    'timeseries', 'per_pair', 'per_cross').
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


# =============================================================================
# Canonical-CSV builders + download buttons (App 12 analogues of App 9's flow)
# =============================================================================
# These produce the SAME canonical CSV schema used by App 9's bulk-download
# buttons so a single downstream analyzer can ingest both single-leg and
# worst-of CSVs from either app interchangeably. Schema (29 columns):
#   strategy_name, strategy_type, n_trades, notional_usd,
#   total_premium_paid_usd, total_tx_cost_usd, total_payout_usd,
#   total_pnl_usd, max_drawdown_usd, win_rate_pct, premium_recovery_pct,
#   sharpe_monthly, annual_sharpe_{mean,min,std,cv,score},
#   n_years, pct_positive_years, min_annual_pnl_usd, calmar,
#   gain_to_pain, ulcer_index, feasibility_pct, ko_rate_pct,
#   leg_a_ko_rate_pct, leg_b_ko_rate_pct, both_survive_rate_pct,
#   structure_vs_min_leg_pct
# Single-leg rows leave the WO-specific columns NaN; worst-of rows leave
# the single-leg-specific columns NaN. Same row layout in both cases.
#
# Time-series CSV is long-format (monthly + annual rows; daily omitted at
# user request) with columns:
#   strategy_name, strategy_type, period_type, period_end,
#   pnl_usd, equity_usd, drawdown_usd
# Worst-of variant adds the same plus the two regime-state columns the
# downstream analyzer can read (state_a, state_b).


def _canonical_summary_row_single(name: str, s: dict) -> dict:
    """Build one row of the canonical summary CSV for a single-leg strategy.
    `s` is the dict returned by `summarize_strategy`. WO columns are NaN."""
    g2p = s.get("gain_to_pain", 0.0)
    return {
        "strategy_name": name,
        "strategy_type": "single",
        "n_trades": int(s.get("n_trades", 0)),
        "notional_usd": s.get("notional_usd", 0.0),
        "total_premium_paid_usd": s.get("total_premium_usd", 0.0),
        "total_tx_cost_usd": s.get("total_transaction_cost_usd", 0.0),
        "total_payout_usd": s.get("total_payout_usd", 0.0),
        "total_pnl_usd": s.get("total_pnl_usd", 0.0),
        "max_drawdown_usd": s.get("max_drawdown_usd", 0.0),
        "win_rate_pct": s.get("win_rate_pct", 0.0),
        "premium_recovery_pct": s.get("premium_recovery_pct", 0.0),
        "sharpe_monthly": s.get("sharpe_monthly", 0.0),
        "annual_sharpe_mean": s.get("annual_sharpe_mean", 0.0),
        "annual_sharpe_min": s.get("annual_sharpe_min", 0.0),
        "annual_sharpe_std": s.get("annual_sharpe_std", 0.0),
        "annual_sharpe_cv": s.get("annual_sharpe_cv", 0.0),
        "annual_sharpe_score": s.get("annual_sharpe_score", 0.0),
        "n_years": int(s.get("n_years", 0)),
        "pct_positive_years": s.get("pct_positive_years", 0.0),
        "min_annual_pnl_usd": s.get("min_annual_pnl_usd", 0.0),
        "calmar": s.get("calmar", 0.0),
        "gain_to_pain": (g2p if g2p != float("inf") else np.nan),
        "ulcer_index": s.get("ulcer_index", 0.0),
        "feasibility_pct": s.get("feasibility_pct", 0.0),
        "ko_rate_pct": s.get("ko_rate_pct", 0.0),
        "leg_a_ko_rate_pct": np.nan,
        "leg_b_ko_rate_pct": np.nan,
        "both_survive_rate_pct": np.nan,
        "structure_vs_min_leg_pct": np.nan,
    }


def _canonical_summary_row_worstof(name: str, s: dict) -> dict:
    """One row of the canonical summary CSV for a worst-of strategy.
    `s` is `worstof_summarize` output. Single-leg columns are NaN."""
    g2p = s.get("gain_to_pain", 0.0)
    return {
        "strategy_name": name,
        "strategy_type": "worst_of",
        "n_trades": int(s.get("n_trades", 0)),
        "notional_usd": s.get("notional_usd", 0.0),
        "total_premium_paid_usd": s.get("total_premium_paid_usd", 0.0),
        "total_tx_cost_usd": s.get("total_tx_cost_usd", 0.0),
        "total_payout_usd": s.get("total_payout_usd", 0.0),
        "total_pnl_usd": s.get("total_pnl_usd", 0.0),
        "max_drawdown_usd": s.get("max_drawdown_usd", 0.0),
        "win_rate_pct": float(s.get("win_rate", 0.0)) * 100,
        "premium_recovery_pct": s.get("premium_recovery_pct", 0.0),
        "sharpe_monthly": s.get("sharpe_monthly", 0.0),
        "annual_sharpe_mean": s.get("annual_sharpe_mean", 0.0),
        "annual_sharpe_min": s.get("annual_sharpe_min", 0.0),
        "annual_sharpe_std": s.get("annual_sharpe_std", 0.0),
        "annual_sharpe_cv": s.get("annual_sharpe_cv", 0.0),
        "annual_sharpe_score": s.get("annual_sharpe_score", 0.0),
        "n_years": int(s.get("n_years", 0)),
        "pct_positive_years": s.get("pct_positive_years", 0.0),
        "min_annual_pnl_usd": s.get("min_annual_pnl_usd", 0.0),
        "calmar": s.get("calmar", 0.0),
        "gain_to_pain": (g2p if g2p != float("inf") else np.nan),
        "ulcer_index": s.get("ulcer_index", 0.0),
        "feasibility_pct": np.nan,
        "ko_rate_pct": np.nan,
        "leg_a_ko_rate_pct": float(s.get("leg_a_ko_rate", 0.0)) * 100,
        "leg_b_ko_rate_pct": float(s.get("leg_b_ko_rate", 0.0)) * 100,
        "both_survive_rate_pct": float(s.get("both_survive_rate", 0.0)) * 100,
        "structure_vs_min_leg_pct": s.get("structure_vs_min_leg_pct", 0.0),
    }


def _filter_monthly_annual(ts: pd.DataFrame) -> pd.DataFrame:
    """Filter export_strategy_time_series / worstof_export_time_series output
    to monthly + annual rows only (daily omitted, per user request)."""
    if ts.empty or "period_type" not in ts.columns:
        return ts
    return ts[ts["period_type"].isin(("monthly", "annual"))].reset_index(drop=True)


def _render_download_buttons_single(
        results: dict, file_prefix: str, key_prefix: str,
        ) -> None:
    """Render the two-column download buttons (summary + timeseries) for a
    single-leg backtest results dict {strategy_name: [Trade, ...]}.

    file_prefix: used in the suggested download filenames (e.g. 'backtest')
    key_prefix: streamlit widget key prefix (must be unique per tab)
    """
    from core.backtest import (
        export_strategy_time_series, augment_time_series_with_regime,
    )
    from core.regimes import get_regime_panel

    # ---- Summary ----
    summary_rows = []
    for name, trades in results.items():
        df = trades_to_df(trades)
        if df.empty:
            continue
        s = summarize_strategy(df)
        summary_rows.append(_canonical_summary_row_single(name, s))
    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()

    # ---- Time series (monthly + annual only) ----
    ts_frames = []
    for name, trades in results.items():
        df_t = trades_to_df(trades)
        if df_t.empty:
            continue
        ts = export_strategy_time_series(df_t)
        if ts.empty:
            continue
        ts = _filter_monthly_annual(ts)
        pair = df_t["pair"].iloc[0] if "pair" in df_t.columns else None
        if pair:
            ts = augment_time_series_with_regime(
                ts, get_regime_panel(pair), column_name="state")
        ts.insert(0, "strategy_name", name)
        ts.insert(1, "strategy_type", "single")
        ts_frames.append(ts)
    ts_combined = (pd.concat(ts_frames, ignore_index=True)
                     if ts_frames else pd.DataFrame())

    cdl_a, cdl_b = st.columns(2)
    with cdl_a:
        if not summary_df.empty:
            st.download_button(
                label=f"⬇ Download summary table ({len(summary_df)} rows, CSV)",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{file_prefix}_summary.csv",
                mime="text/csv",
                help=("Canonical schema: strategy_name, strategy_type, "
                       "n_trades, money totals, Sharpe block, consistency "
                       "block, then strategy-specific. Compatible with the "
                       "downstream analyzer that ingests App 9's CSVs."),
                use_container_width=True,
                key=f"{key_prefix}_summary_dl",
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
                file_name=f"{file_prefix}_timeseries.csv",
                mime="text/csv",
                help=("Long-format: monthly + annual rows only (daily "
                       "omitted). period_end is the right-edge date; "
                       "equity_usd and drawdown_usd are end-of-period "
                       "snapshots so DD-based ratios are recomputable."),
                use_container_width=True,
                key=f"{key_prefix}_timeseries_dl",
            )
        else:
            st.caption("_No time-series data yet._")


def _render_download_buttons_worstof(
        results: dict, file_prefix: str, key_prefix: str,
        ) -> None:
    """Worst-of analogue of `_render_download_buttons_single`. Uses the
    worst-of summary + time-series functions and emits the canonical
    schema with WO columns populated."""
    from core.worstof import (
        worstof_trades_to_df, worstof_summarize, worstof_export_time_series,
    )
    from core.backtest import augment_time_series_with_regime
    from core.regimes import get_regime_panel

    summary_rows = []
    for name, trades in results.items():
        df = worstof_trades_to_df(trades)
        if df.empty:
            continue
        s = worstof_summarize(df)
        summary_rows.append(_canonical_summary_row_worstof(name, s))
    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()

    ts_frames = []
    for name, trades in results.items():
        df_t = worstof_trades_to_df(trades)
        if df_t.empty:
            continue
        ts = worstof_export_time_series(df_t)
        if ts.empty:
            continue
        ts = _filter_monthly_annual(ts)
        pair_a = (df_t["leg_a_pair"].iloc[0]
                    if "leg_a_pair" in df_t.columns else None)
        pair_b = (df_t["leg_b_pair"].iloc[0]
                    if "leg_b_pair" in df_t.columns else None)
        if pair_a:
            ts = augment_time_series_with_regime(
                ts, get_regime_panel(pair_a), column_name="state_a")
        if pair_b:
            ts = augment_time_series_with_regime(
                ts, get_regime_panel(pair_b), column_name="state_b")
        ts.insert(0, "strategy_name", name)
        ts.insert(1, "strategy_type", "worst_of")
        ts_frames.append(ts)
    ts_combined = (pd.concat(ts_frames, ignore_index=True)
                     if ts_frames else pd.DataFrame())

    cdl_a, cdl_b = st.columns(2)
    with cdl_a:
        if not summary_df.empty:
            st.download_button(
                label=f"⬇ Download summary table ({len(summary_df)} rows, CSV)",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{file_prefix}_summary.csv",
                mime="text/csv",
                help=("Worst-of canonical schema (same column layout as "
                       "single-leg, with WO-specific columns populated)."),
                use_container_width=True,
                key=f"{key_prefix}_summary_dl",
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
                file_name=f"{file_prefix}_timeseries.csv",
                mime="text/csv",
                help=("Monthly + annual rows only (daily omitted). With "
                       "state_a / state_b regime columns when regime "
                       "panels are registered for the underlying pairs."),
                use_container_width=True,
                key=f"{key_prefix}_timeseries_dl",
            )
        else:
            st.caption("_No time-series data yet._")


# =============================================================================
# Shared constants — drive the Backtest, Worst-of, RKO Portfolio, and
# WO-RKO Portfolio tabs. The Pricer tab in 12_american_ko_pricer.py has
# its own slightly larger set (includes 25Δ for one-off snapshot pricing).
# =============================================================================
TENOR_LIST = ["1M", "6W", "2M", "10W", "3M"]

DELTA_CHOICES = {
    "ATM": 0.0, "45Δ": 0.45, "40Δ": 0.40, "35Δ": 0.35, "30Δ": 0.30,
}
KO_DELTA_CHOICES = {
    "20Δ": 0.20, "15Δ": 0.15, "10Δ": 0.10, "5Δ": 0.05,
}
PAYOUT_CHOICES = {
    "4:1": 4.0, "6:1": 6.0, "8:1": 8.0, "10:1": 10.0,
    "15:1": 15.0, "20:1": 20.0,
}
DIRECTIONS = {
    "Call (up-and-out)":  ("call", "up_and_out"),
    "Put (down-and-out)": ("put", "down_and_out"),
}

# Backtest defaults — applied uniformly across the four backtest tabs.
DEFAULT_TENOR = "2M"
DEFAULT_STRIKE_DELTA_LABEL = "ATM"
DEFAULT_KO_DELTA_LABEL = "20Δ"
DEFAULT_START_DATE = _date(2023, 1, 1)


def _list_pairs(folder: str) -> "list[str]":
    """Pairs that have SPOT data — same logic as the Pricer tab."""
    df = load_panel(folder, "SPOT", None)
    return sorted(df.columns.tolist())


# =============================================================================
# Backtest tab
# =============================================================================
def render_backtest_tab(folder: str) -> None:
    pairs_avail = _list_pairs(folder)
    if not pairs_avail:
        st.error("No SPOT data found in the data folder.")
        return

    st.markdown("### Backtest configuration")
    st.caption(
        "Cross-product of (pair × strike Δ × tenor × direction × payout/KO Δ "
        "× gate) is run as one strategy each. Entry premium uses **Vanna-Volga** "
        "when smile data is available, R-R closed-form otherwise. Barrier "
        "monitoring is **American** — daily OHLC range check; the option "
        "knocks out the first day the barrier sits within [Low, High]."
    )

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        default_pairs = [p for p in ("USDJPY", "USDKRW") if p in pairs_avail]
        if not default_pairs:
            default_pairs = pairs_avail[:1]
        pairs_sel = st.multiselect(
            "Currency pairs", pairs_avail, default=default_pairs,
            key="rko_bt_pairs",
        )
        deltas_sel = st.multiselect(
            "Strike Δ list", list(DELTA_CHOICES.keys()),
            default=[DEFAULT_STRIKE_DELTA_LABEL],
            key="rko_bt_deltas",
        )
        tenors_sel = st.multiselect(
            "Tenors", TENOR_LIST, default=[DEFAULT_TENOR],
            key="rko_bt_tenors",
        )

    with cc2:
        directions_sel = st.multiselect(
            "Direction(s)", list(DIRECTIONS.keys()),
            default=["Call (up-and-out)"],
            key="rko_bt_directions",
        )
        ko_method_label = st.radio(
            "KO method", ("Payout ratio", "KO delta"),
            index=1, horizontal=True, key="rko_bt_ko_method",
            help=("• **Payout ratio**: solve H so max_payoff/premium hits "
                   "the target leverage.\n"
                   "• **KO delta**: H placed at vanilla-Δ wing strike."),
        )
        ko_method = ("ratio" if ko_method_label == "Payout ratio" else "delta")
        if ko_method == "ratio":
            payout_labels = st.multiselect(
                "Payout ratio(s)", list(PAYOUT_CHOICES.keys()),
                default=["8:1"], key="rko_bt_payout",
            )
            payout_ratios = [PAYOUT_CHOICES[lbl] for lbl in payout_labels]
            ko_delta_labels, ko_delta_values = [], []
            st.caption(
                "American-barrier KOs cost ~half as much as European-barrier "
                "for the same K, H — i.e. an 8× leverage on European "
                "becomes ~4× on American. Adjust expectations accordingly."
            )
        else:
            ko_delta_labels = st.multiselect(
                "KO Δ (vanilla wing)", list(KO_DELTA_CHOICES.keys()),
                default=[DEFAULT_KO_DELTA_LABEL], key="rko_bt_ko_delta",
            )
            ko_delta_values = [KO_DELTA_CHOICES[lbl] for lbl in ko_delta_labels]
            payout_labels, payout_ratios = [], []
            st.caption(
                "Barrier H placed at the vanilla-wing strike (same Δ "
                "convention as the strike). Achieved leverage varies trade by "
                "trade and is surfaced in the ledger."
            )

        tx_cost_bps = st.slider(
            "Transaction cost (bps of notional)", 0.0, 20.0, 2.0, 0.5,
            help="Flat bps markup on the foreign notional, added to the "
                  "mid VV premium. 2 bps on $10M notional = $2,000.",
            key="rko_bt_txcost",
        )

        from core.gates import GATE_REGISTRY
        gate_options = ["(no gate)"] + list(GATE_REGISTRY.keys())
        gate_labels = st.multiselect(
            "Gate(s)", gate_options, default=["(no gate)"],
            key="rko_bt_gate_keys",
            format_func=lambda k: ("(no gate)" if k == "(no gate)"
                                   else GATE_REGISTRY[k][0]),
            help="Each selection becomes its own strategy variant.",
        )
        gate_keys = [None if k == "(no gate)" else k for k in gate_labels]

    with cc3:
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

        # Default start = 1 Jan 2023 (per spec), clamped to the available data range
        default_start = min(max(DEFAULT_START_DATE, date_min), date_max)
        start_date = st.date_input(
            "Start date", value=default_start,
            min_value=date_min, max_value=date_max, key="rko_bt_start",
        )
        end_date = st.date_input(
            "End date", value=date_max,
            min_value=date_min, max_value=date_max, key="rko_bt_end",
        )
        prefer_em = st.radio(
            "EM pairs variant", ["offshore", "onshore"], index=0,
            horizontal=True, key="rko_bt_prefer_em",
        )
        notional_usd = st.number_input(
            "Notional (USD)", min_value=100_000.0, max_value=200_000_000.0,
            value=10_000_000.0, step=1_000_000.0, format="%.0f",
            key="rko_bt_notional",
            help="Foreign-currency notional. All USD figures (premium, "
                  "payout, PnL, drawdown) are computed at this size.",
        )
        trade_mode = st.radio(
            "Trade mode",
            ["stack", "single"],
            index=0, horizontal=True, key="rko_bt_trade_mode",
            help=("**stack**: open a new trade every eligible date — "
                   "overlapping book. **single**: at most one trade per pair "
                   "open at a time — next entry only on/after prior expiry."),
        )

    # ---- Validate + count strategies ----
    n_pairs = len(pairs_sel)
    n_deltas = len(deltas_sel)
    n_tenors = len(tenors_sel)
    n_dirs = len(directions_sel)
    if ko_method == "ratio":
        n_ko = len(payout_ratios)
        ko_axis_label = "payout ratios"
    else:
        n_ko = len(ko_delta_values)
        ko_axis_label = "KO deltas"
    n_gates = max(len(gate_keys), 1)
    n_specs = n_pairs * n_deltas * n_tenors * n_dirs * n_ko * n_gates

    st.caption(
        f"**{n_specs}** strategies will run "
        f"({n_pairs} pairs × {n_deltas} deltas × {n_tenors} tenors × "
        f"{n_dirs} directions × {n_ko} {ko_axis_label} × {n_gates} gates) "
        f"over {(end_date - start_date).days} calendar days. "
        f"Expect roughly 1-2 minutes for a single-pair / single-tenor "
        f"strategy over 2y, scaling linearly with the cross-product size."
    )

    can_run = (n_specs > 0 and pairs_sel and deltas_sel and tenors_sel
                and directions_sel and n_ko > 0 and len(gate_keys) > 0)
    run_clicked = st.button("▶ Run backtest", type="primary",
                              disabled=not can_run, key="rko_bt_run")

    # ---- Execute on click ----
    if run_clicked:
        # Build cross-product specs (same pattern as App 9)
        specs = []
        if ko_method == "ratio":
            ko_axis = [(r, None, None) for r in payout_ratios]
        else:
            ko_axis = [(None, v, lbl)
                         for v, lbl in zip(ko_delta_values, ko_delta_labels)]
        # Step R1 — pass the sidebar's single-leg pricing model (set
        # in pages/rko_pricer.py via st.session_state['rko_pricing_model']).
        # Default to 'vanna_volga' to preserve legacy RKO backtest
        # behaviour for callers that haven't set it.
        _pm = st.session_state.get("rko_pricing_model", "vanna_volga")
        for ratio_v, kdv, kdl in ko_axis:
            for gk in gate_keys:
                specs += build_strategy_grid(
                    pairs=pairs_sel,
                    deltas=[(d, DELTA_CHOICES[d]) for d in deltas_sel],
                    tenors=tenors_sel,
                    directions=[DIRECTIONS[d] for d in directions_sel],
                    tx_cost_bps=tx_cost_bps,
                    prefer=prefer_em,
                    ko_method=ko_method,
                    payout_ratio=ratio_v,
                    target_ko_delta=kdv,
                    ko_delta_label=kdl,
                    entry_gate=gk,
                    trade_mode=trade_mode,
                    pricing_model=_pm,
                )

        progress_bar = st.progress(0.0, text="Loading panels…")
        t0 = time.time()
        last_update = [t0]

        def cb(p, name):
            now = time.time()
            if now - last_update[0] > 0.1 or p >= 1.0:
                progress_bar.progress(min(p, 1.0),
                                       text=f"Running: {name} ({p*100:.0f}%)")
                last_update[0] = now

        try:
            results = run_grid_american(
                folder, specs, start_date, end_date,
                notional_usd=notional_usd, progress_cb=cb,
            )
        except Exception as e:
            progress_bar.empty()
            st.error(f"Backtest failed: {e}")
            return

        elapsed = time.time() - t0
        progress_bar.empty()

        st.session_state["rko_bt_results"] = results
        st.session_state["rko_bt_specs"] = specs
        st.session_state["rko_bt_elapsed"] = elapsed
        st.session_state["rko_bt_meta"] = {
            "start": start_date, "end": end_date,
            "ko_method": ko_method,
            "payout_labels": payout_labels,
            "ko_delta_labels": ko_delta_labels,
            "tx_cost_bps": tx_cost_bps,
            "entry_gates": gate_keys,
            "prefer": prefer_em,
            "n_specs": n_specs,
            "notional_usd": notional_usd,
            "trade_mode": trade_mode,
        }

        n_trades = sum(len(t) for t in results.values())
        st.success(
            f"Done in {elapsed:.1f}s — "
            f"{n_specs} {'strategy' if n_specs == 1 else 'strategies'}, "
            f"{n_trades} trades total."
        )

    # ---- Render summary if results exist ----
    if "rko_bt_results" in st.session_state:
        _render_summary_table()


def _render_summary_table() -> None:
    """Strategy-level summary table — one row per spec."""
    st.markdown("---")
    st.markdown("### Summary across strategies")

    results = st.session_state["rko_bt_results"]
    meta = st.session_state.get("rko_bt_meta", {})

    # Context caption
    if meta.get("ko_method") == "delta":
        kd_labels = meta.get("ko_delta_labels", [])
        ko_str = (f"KO @ {', '.join(kd_labels)} (vanilla)" if kd_labels
                   else "KO Δ unset")
    else:
        pl = meta.get("payout_labels", [])
        ko_str = (f"leverage {', '.join(pl)}" if pl else "leverage unset")
    from core.gates import gate_label as _gate_lbl
    gks = meta.get("entry_gates", [])
    gate_str = ", ".join(_gate_lbl(g) for g in gks) if gks else "(none)"

    st.caption(
        f"Run: {meta.get('start')} → {meta.get('end')}  ·  "
        f"notional ${meta.get('notional_usd', 0):,.0f}  ·  "
        f"{ko_str}  ·  tx cost {meta.get('tx_cost_bps', 0):.1f} bps  ·  "
        f"gates: **{gate_str}**  ·  trade mode: **{meta.get('trade_mode', 'stack')}**  ·  "
        f"elapsed {st.session_state.get('rko_bt_elapsed', 0):.1f}s"
    )

    # Build summary rows
    rows = []
    for name, trades in results.items():
        df = trades_to_df(trades)
        if df.empty:
            rows.append({
                "Strategy": name,
                "n trades": 0,
                "KO %": float("nan"),
                "Win %": float("nan"),
                "Avg premium": "",
                "Total PnL ($)": "",
                "Total PnL (%)": float("nan"),
                "Max DD ($)": "",
                "Sharpe": float("nan"),
                "Annual Sharpe μ": float("nan"),
                "%Pos Yrs": float("nan"),
                "Calmar": float("nan"),
                "G2P": float("nan"),
                "Recovery %": float("nan"),
            })
            continue
        s = summarize_strategy(df)
        rows.append({
            "Strategy": name,
            "n trades": s["n_trades"],
            "KO %": s["ko_rate_pct"],
            "Win %": s["win_rate_pct"],
            "Avg premium": f"${s.get('avg_premium_usd', 0):,.0f}",
            "Total PnL ($)": f"${s.get('total_pnl_usd', 0):,.0f}",
            "Total PnL (%)": s["total_pnl_pct"],
            "Max DD ($)": f"${s.get('max_drawdown_usd', 0):,.0f}",
            "Sharpe": s["sharpe_monthly"],
            "Annual Sharpe μ": s.get("annual_sharpe_mean", float("nan")),
            "%Pos Yrs": s.get("pct_positive_years", float("nan")),
            "Calmar": s.get("calmar", float("nan")),
            "G2P": s.get("gain_to_pain", float("nan")),
            "Recovery %": s["premium_recovery_pct"],
        })

    sdf = pd.DataFrame(rows)

    # Sort by Sharpe descending (NaN to bottom)
    sdf = sdf.sort_values("Sharpe", ascending=False, na_position="last")

    st.caption(
        "💡 Click any column header to sort. `Sharpe` is monthly P&L Sharpe; "
        "`Annual Sharpe μ` averages yearly Sharpes (more robust to "
        "concentration); `%Pos Yrs` measures consistency. `Recovery %` = "
        "payouts received / premium paid — over 100% means net winning "
        "(before tx cost)."
    )
    st.dataframe(
        sdf.style.format({
            "KO %": "{:.1f}",
            "Win %": "{:.1f}",
            "Total PnL (%)": "{:+.2f}",
            "Sharpe": "{:.2f}",
            "Annual Sharpe μ": "{:.2f}",
            "%Pos Yrs": "{:.0f}",
            "Calmar": "{:.2f}",
            "G2P": "{:.2f}",
            "Recovery %": "{:.0f}",
        }, na_rep="—"),
        use_container_width=True, hide_index=True,
    )

    n_total = sum(len(t) for t in results.values())
    st.caption(
        f"_{len(results)} strategies, {n_total} trades total. "
        f"→ Switch to the **Backtest drilldown** tab to inspect a single "
        f"strategy's equity curve, monthly P&L, and trade ledger._"
    )

    # ---- Bulk download (summary + monthly/annual time series) ----
    st.markdown("---")
    st.markdown("#### Bulk download")
    st.caption(
        "Two CSVs in the same canonical schema the downstream analyzer "
        "ingests. The summary CSV is one row per strategy with full Sharpe "
        "/ Calmar / consistency blocks; the time-series CSV has monthly + "
        "annual rows for every strategy (daily omitted by design)."
    )
    _render_download_buttons_single(results, file_prefix="rko_backtest",
                                          key_prefix="bt12")


# =============================================================================
# Backtest drilldown tab
# =============================================================================
def render_drilldown_tab() -> None:
    if "rko_bt_results" not in st.session_state:
        st.info(
            "No backtest results yet. Run a backtest in the **Backtest** "
            "tab first, then come back here to inspect a single strategy."
        )
        return

    results = st.session_state["rko_bt_results"]
    meta = st.session_state.get("rko_bt_meta", {})

    # Strategy picker
    names = list(results.keys())
    if not names:
        st.warning("No strategies in the latest backtest.")
        return

    # Default to the strategy with most trades (more interesting to drill into)
    default_idx = max(range(len(names)), key=lambda i: len(results[names[i]]))
    sel = st.selectbox(
        "Strategy", names, index=default_idx, key="rko_dd_strategy",
    )
    trades = results.get(sel, [])
    df = trades_to_df(trades)
    if df.empty:
        st.warning(f"Strategy `{sel}` produced no trades.")
        return

    s = summarize_strategy(df)
    eq = compute_equity_and_drawdown(df)

    # ---- Top-line metrics ----
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Trades", f"{s['n_trades']}")
    c2.metric("KO rate", f"{s['ko_rate_pct']:.1f}%")
    c3.metric("Win rate", f"{s['win_rate_pct']:.1f}%")
    c4.metric("Total P&L", f"${s.get('total_pnl_usd', 0):,.0f}")
    c5.metric("Max drawdown", f"${s.get('max_drawdown_usd', 0):,.0f}")

    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("Sharpe (monthly)", f"{s['sharpe_monthly']:.2f}")
    c7.metric("Calmar", f"{s.get('calmar', float('nan')):.2f}")
    c8.metric("Gain-to-Pain", f"{s.get('gain_to_pain', float('nan')):.2f}")
    c9.metric("Premium recovery",
                f"{s['premium_recovery_pct']:.0f}%")
    c10.metric("Avg premium",
                 f"${s.get('avg_premium_usd', 0):,.0f}")

    st.markdown("---")

    # ---- Equity & drawdown ----
    st.markdown("#### Equity curve and drawdown (USD)")
    if not eq.empty and "equity_usd" in eq.columns:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.65, 0.35], vertical_spacing=0.05,
            subplot_titles=("Cumulative P&L (USD)", "Drawdown (USD)"),
        )
        fig.add_trace(
            go.Scatter(x=eq.index, y=eq["equity_usd"], mode="lines",
                          line=dict(color="#1f77b4", width=2),
                          name="Equity"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=eq.index, y=eq["drawdown_usd"], mode="lines",
                          line=dict(color="#d62728", width=1.5),
                          fill="tozeroy",
                          fillcolor="rgba(214, 39, 40, 0.15)",
                          name="Drawdown"),
            row=2, col=1,
        )
        fig.update_layout(height=520, showlegend=False, hovermode="x unified",
                            margin=dict(l=10, r=10, t=40, b=10))
        fig.update_yaxes(title_text="USD", row=1, col=1)
        fig.update_yaxes(title_text="USD", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No equity curve data available.")

    st.markdown("---")

    # ---- Monthly heatmap ----
    st.markdown("#### Monthly P&L heatmap (USD)")
    monthly_df = monthly_pnl_table(df, value_col="pnl_usd")
    if not monthly_df.empty:
        # Months come back as ints (1-12) plus a string 'YTD'. Convert to
        # short string labels so the column index is single-typed (avoids
        # an Arrow conversion warning) and easier to read.
        month_labels = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May",
                          6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct",
                          11: "Nov", 12: "Dec"}
        monthly_df = monthly_df.rename(
            columns=lambda c: month_labels.get(c, str(c)))
        # Display as a colored dataframe with our matplotlib-free
        # diverging palette (background_gradient requires matplotlib).
        st.dataframe(
            _diverging_red_yellow_green(monthly_df).format("${:,.0f}", na_rep=""),
            use_container_width=True,
        )
    else:
        st.info("Not enough data for monthly summary.")

    st.markdown("---")

    # ---- Annual summary ----
    st.markdown("#### Annual summary")
    annual_df = annual_summary_table(df)
    if not annual_df.empty:
        # Identify numeric vs string columns for formatting
        fmt_map = {}
        for col in annual_df.columns:
            ser = annual_df[col]
            if pd.api.types.is_numeric_dtype(ser):
                if "pct" in col.lower() or "%" in col:
                    fmt_map[col] = "{:.1f}%"
                elif "sharpe" in col.lower() or "calmar" in col.lower():
                    fmt_map[col] = "{:.2f}"
                elif "usd" in col.lower() or "$" in col:
                    fmt_map[col] = "${:,.0f}"
                else:
                    fmt_map[col] = "{:.2f}"
        st.dataframe(annual_df.style.format(fmt_map, na_rep="—"),
                       use_container_width=True, hide_index=False)
    else:
        st.info("Not enough data for annual breakdown.")

    st.markdown("---")

    # ---- Trade ledger + CSV export ----
    st.markdown("#### Trade ledger")
    st.caption(
        f"{len(df)} trades. American-barrier KOs: column `knockout_date` "
        f"shows the day the barrier was first touched (within [Low, High]); "
        f"`knockout_spot` is the close on that day."
    )

    # Columns most useful in the drilldown view
    cols_show = [
        "trade_date", "expiry_date", "knockout_date", "knocked_out",
        "spot", "strike", "barrier", "knockout_spot", "spot_at_expiry",
        "sigma_atm", "sigma_smile", "rr_25", "bf_25",
        "achieved_payout_ratio", "feasible",
        "premium_usd", "actual_payoff_usd", "pnl_usd",
        "pricing_model",
    ]
    cols_have = [c for c in cols_show if c in df.columns]
    df_view = df[cols_have].copy()

    # Format
    fmt_pct_cols = [c for c in df_view.columns
                       if any(k in c for k in ["sigma", "rr_25", "bf_25"])]
    for c in fmt_pct_cols:
        df_view[c] = df_view[c].apply(
            lambda x: f"{x*100:+.3f}%" if pd.notna(x) else "")
    for c in ["spot", "strike", "barrier", "knockout_spot", "spot_at_expiry"]:
        if c in df_view.columns:
            df_view[c] = df_view[c].apply(
                lambda x: f"{x:,.4f}" if pd.notna(x) else "")
    for c in ["premium_usd", "actual_payoff_usd", "pnl_usd"]:
        if c in df_view.columns:
            df_view[c] = df_view[c].apply(
                lambda x: f"${x:,.0f}" if pd.notna(x) else "")
    if "achieved_payout_ratio" in df_view.columns:
        df_view["achieved_payout_ratio"] = df_view["achieved_payout_ratio"].apply(
            lambda x: f"{x:.2f}×" if pd.notna(x) else "")
    st.dataframe(df_view, use_container_width=True, hide_index=True)

    # Full CSV export (raw numeric)
    csv_bytes = df.to_csv(index=False).encode()
    safe_name = sel.replace("/", "_").replace(" ", "_").replace(",", "")
    st.download_button(
        f"⬇ Download full ledger (CSV)",
        data=csv_bytes,
        file_name=f"rko_backtest_{safe_name}.csv",
        mime="text/csv",
        key="rko_dd_csv",
    )


# =============================================================================
# Worst-of tab — pricer + backtest
# =============================================================================
# A worst-of structure is two single-leg KOs that share a structure-level
# premium. The premium approximation is `multiplier × min(prem_A, prem_B)`
# where multiplier is set in the sidebar (App 12 default 40%). The
# structure dies if EITHER leg's barrier is hit on any day (daily-OHLC
# check for App 12), and at expiry pays `min(payoff_A, payoff_B)`.
#
# We use the existing core/worstof.py engine with `ko_check_mode=
# 'american_ohlc'` so the daily-OHLC scan replaces the European
# at-expiry check. The pricer used inside the engine is still the
# European `ko_price`; the multiplier choice compensates for the
# difference between European and American leg-level pricing.


def render_worstof_tab(folder: str, multiplier: float) -> None:
    """Worst-of backtest tab for App 12.

    `multiplier` is sourced from the sidebar `wo_multiplier` variable
    (a float in {0.33, 0.40, 0.50}).
    """
    pairs_avail = _list_pairs(folder)
    if len(pairs_avail) < 2:
        st.error("Worst-of needs at least two pairs with SPOT data.")
        return

    st.markdown("### Worst-of configuration")
    st.caption(
        f"Premium ≈ **{int(multiplier*100)}% × min(leg_A, leg_B)** + tx cost. "
        f"Each leg is priced with **full Vanna-Volga on the American-barrier "
        f"closed form** (matches Bloomberg OVML's 'Vanna-Volga' model within "
        f"~0.5% on USDJPY), so the leg premiums entering the min() are "
        f"smile-adjusted. Structure dies if **either** leg's barrier sits "
        f"in the day's [Low, High] range. At expiry: payoff = "
        f"min(intrinsic_A, intrinsic_B), floored at zero. Both legs use "
        f"the same strike Δ, tenor, direction, and KO target. Change the "
        f"multiplier in the sidebar."
    )

    # ---- Pricer card (single-snapshot view) ----
    # Sits at the top of the Worst-of tab so users can see one structure
    # priced live before kicking off a multi-year backtest. Uses
    # latest-available date in the data folder. Mirrors the Pricer tab
    # but for two legs at once.
    with st.expander("💱 Worst-of pricer (single snapshot)", expanded=False):
        _render_worstof_pricer_card(folder, multiplier, pairs_avail)
    st.markdown("---")

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.markdown("**Pair combinations**")
        default_pair_a = "USDJPY" if "USDJPY" in pairs_avail else pairs_avail[0]
        default_pair_b = ("USDKRW" if "USDKRW" in pairs_avail
                            else pairs_avail[1] if len(pairs_avail) > 1
                            else pairs_avail[0])
        pairs_a = st.multiselect(
            "Leg A pairs", pairs_avail, default=[default_pair_a],
            key="rko_wo_pairs_a",
        )
        pairs_b = st.multiselect(
            "Leg B pairs", pairs_avail, default=[default_pair_b],
            key="rko_wo_pairs_b",
        )
        st.caption(
            "Cross-product: each leg-A pair runs against each leg-B pair. "
            "Same-pair combinations (e.g. USDJPY × USDJPY) are skipped."
        )

    with cc2:
        st.markdown("**Structure (shared across legs)**")
        deltas_sel = st.multiselect(
            "Strike Δ list", list(DELTA_CHOICES.keys()),
            default=[DEFAULT_STRIKE_DELTA_LABEL],
            key="rko_wo_deltas",
        )
        tenors_sel = st.multiselect(
            "Tenors", TENOR_LIST, default=[DEFAULT_TENOR],
            key="rko_wo_tenors",
        )
        directions_sel = st.multiselect(
            "Direction(s)", list(DIRECTIONS.keys()),
            default=["Call (up-and-out)"],
            key="rko_wo_directions",
        )
        ko_delta_labels = st.multiselect(
            "KO Δ (vanilla wing) per leg",
            list(KO_DELTA_CHOICES.keys()),
            default=[DEFAULT_KO_DELTA_LABEL], key="rko_wo_ko_delta",
            help="Each leg's barrier sits at this vanilla-Δ wing strike. "
                  "Shared between legs A and B (worst-of conventions).",
        )
        tx_cost_bps = st.slider(
            "Transaction cost (bps of notional)", 0.0, 20.0, 2.0, 0.5,
            help="Flat bps markup on foreign notional, applied at the "
                  "structure level only — legs stay at mid.",
            key="rko_wo_txcost",
        )

    with cc3:
        st.markdown("**Run parameters**")
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
        # Default start = 1 Jan 2023 (per spec), clamped to the available data range
        default_start = min(max(DEFAULT_START_DATE, date_min), date_max)
        start_date = st.date_input(
            "Start date", value=default_start,
            min_value=date_min, max_value=date_max, key="rko_wo_start",
        )
        end_date = st.date_input(
            "End date", value=date_max,
            min_value=date_min, max_value=date_max, key="rko_wo_end",
        )
        prefer_em = st.radio(
            "EM pairs variant", ["offshore", "onshore"], index=0,
            horizontal=True, key="rko_wo_prefer_em",
        )
        notional_usd = st.number_input(
            "Notional (USD)", min_value=100_000.0, max_value=200_000_000.0,
            value=10_000_000.0, step=1_000_000.0, format="%.0f",
            key="rko_wo_notional",
            help="Foreign-currency notional, same on each leg.",
        )
        trade_mode = st.radio(
            "Trade mode", ["stack", "single"],
            index=0, horizontal=True, key="rko_wo_trade_mode",
        )

    # ---- Build pair combos (cross product, skip same-pair) ----
    pair_combos = [(a, b) for a in pairs_a for b in pairs_b if a != b]
    n_combos = len(pair_combos)
    n_deltas = len(deltas_sel)
    n_tenors = len(tenors_sel)
    n_dirs = len(directions_sel)
    n_ko = len(ko_delta_labels)
    n_specs = n_combos * n_deltas * n_tenors * n_dirs * n_ko

    st.caption(
        f"**{n_specs}** worst-of strategies will run "
        f"({n_combos} pair combos × {n_deltas} deltas × {n_tenors} tenors "
        f"× {n_dirs} directions × {n_ko} KO deltas)  ·  "
        f"multiplier = **{int(multiplier*100)}%**"
    )
    if n_combos == 0:
        st.warning("Need at least one (leg-A, leg-B) pair combination — "
                    "your selected pair lists overlap or are empty.")

    # ---- Structure pricing engine controls (Step R3) -----------------------
    # Same UI as the EKO bulk worst-of tab. Three engines:
    #
    #   Legacy multiplier (default):  multiplier × min(leg_A_mid, leg_B_mid)
    #                                 — historical behaviour. Leg-level VV
    #                                 American pricing. Preserves existing
    #                                 user backtests.
    #
    #   CF-approx American:           core.worstof_pricer_american.
    #                                 worstof_rko_price_cf_approx —
    #                                 ratio-scaled European worst-of, ~2 ms
    #                                 per trade. Biased low on tight
    #                                 barriers; very accurate on wide.
    #
    #   MC American:                  core.worstof_pricer_american.
    #                                 worstof_rko_price_mc — daily-step
    #                                 correlated GBM with Brownian-bridge
    #                                 sub-step correction. Canonical pricer;
    #                                 ~400 ms per trade at 100k paths.
    #
    # Non-legacy engines REQUIRE leg_pricing_mode='european' (the European
    # single-leg pricer is what the worst-of engine consumes internally
    # for its joint calc). When the user picks CF or MC, we switch the
    # leg-pricing mode accordingly so leg-level mids on the trade rows
    # are consistent with the structure premium calculation.
    with st.expander("Structure pricing engine", expanded=False):
        st.caption(
            "How the worst-of structure premium is computed. **Legacy "
            "multiplier** (default) preserves historical behaviour. "
            "**CF-approx American** and **MC American** use the new "
            "correlation-aware pricers in `core.worstof_pricer_american`."
        )
        engine_label = st.radio(
            "Engine",
            ["Legacy multiplier (default)",
             "CF-approx American (fast)",
             "Monte Carlo American (canonical)"],
            index=0,
            key="rko_wo_pricing_engine",
            help=(
                "Legacy: `multiplier × min(P_A, P_B)`. "
                "CF-approx: ~2 ms/trade, low-biased on tight barriers. "
                "MC: ~400 ms/trade at 100k paths, canonical pricer."
            ),
        )
        _engine_map = {
            "Legacy multiplier (default)":      "legacy_multiplier",
            "CF-approx American (fast)":         "cf_approx_american",
            "Monte Carlo American (canonical)":  "monte_carlo_american",
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
                key="rko_wo_correlation_source",
                help=(
                    "**Manual**: same ρ used for every trade date. "
                    "**Historical 60d**: rolling 60-business-day realized "
                    "log-return correlation. "
                    "**Triangulation**: forward-looking implied ρ from "
                    "the cross-pair's ATM vol. Falls back to Manual when "
                    "the source's value is missing."
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
                 "data unavailable)"),
                min_value=-0.95, max_value=0.95,
                value=0.30, step=0.05,
                key="rko_wo_correlation_value",
            )
            if pricing_engine == "monte_carlo_american":
                mc_n_paths = st.select_slider(
                    "MC paths per trade",
                    options=[20_000, 50_000, 100_000, 200_000, 500_000],
                    value=100_000, key="rko_wo_mc_n_paths",
                    help=("Std error per trade scales as 1/√n. "
                           "100k → ~1-2bp; 500k → ~0.5bp on tight RKOs."),
                )

    can_run = (n_specs > 0 and pairs_a and pairs_b and deltas_sel
                and tenors_sel and directions_sel and ko_delta_labels)
    run_clicked = st.button("▶ Run worst-of backtest", type="primary",
                              disabled=not can_run, key="rko_wo_run")

    if run_clicked:
        from core.worstof import build_worstof_grid, run_worstof_grid

        # Non-legacy American engines require leg_pricing_mode='european'.
        # This changes the recorded leg-level mids on each trade row
        # (they'll be European single-leg KO prices, not the VV-American
        # ones the tab uses by default). The STRUCTURE premium uses the
        # new engine; the leg-level numbers are informational.
        if pricing_engine == "legacy_multiplier":
            _leg_pricing_mode = "vanna_volga_american"
        else:
            _leg_pricing_mode = "european"

        specs = []
        for d_label in deltas_sel:
            sd_resolved = [(d_label, DELTA_CHOICES[d_label])]
            for kd_label in ko_delta_labels:
                kd_resolved = [(kd_label, KO_DELTA_CHOICES[kd_label])]
                for tenor in tenors_sel:
                    for dir_label in directions_sel:
                        dir_, btype = DIRECTIONS[dir_label]
                        specs += build_worstof_grid(
                            pair_combos=pair_combos,
                            tenors=[tenor],
                            leg_a_directions=[(dir_, btype)],
                            leg_b_directions=[(dir_, btype)],
                            leg_a_strike_deltas=sd_resolved,
                            leg_b_strike_deltas=sd_resolved,
                            leg_a_ko_deltas=kd_resolved,
                            leg_b_ko_deltas=kd_resolved,
                            gates_a=[None], gates_b=[None],
                            tx_cost_bps=tx_cost_bps,
                            prefer=prefer_em,
                            trade_mode=trade_mode,
                            multiplier=multiplier,
                            ko_check_mode="american_ohlc",
                            leg_pricing_mode=_leg_pricing_mode,
                            pricing_engine=pricing_engine,
                            correlation_source=correlation_source,
                            correlation_value=correlation_value,
                            mc_n_paths=mc_n_paths,
                        )

        if not specs:
            st.error(
                "No valid worst-of specs built — check the KO Δ < strike Δ "
                "filter (KO must be further OTM than strike on each leg)."
            )
            return

        progress_bar = st.progress(0.0, text="Loading panels…")
        t0 = time.time()
        last_update = [t0]

        def cb(p, name):
            now = time.time()
            if now - last_update[0] > 0.1 or p >= 1.0:
                progress_bar.progress(min(p, 1.0),
                                       text=f"Running: {name} ({p*100:.0f}%)")
                last_update[0] = now

        try:
            results = run_worstof_grid(folder, specs, start_date, end_date,
                                          notional_usd=notional_usd,
                                          progress_cb=cb)
        except Exception as e:
            progress_bar.empty()
            st.error(f"Worst-of backtest failed: {e}")
            return

        elapsed = time.time() - t0
        progress_bar.empty()

        st.session_state["rko_wo_results"] = results
        st.session_state["rko_wo_specs"] = specs
        st.session_state["rko_wo_elapsed"] = elapsed
        st.session_state["rko_wo_meta"] = {
            "start": start_date, "end": end_date,
            "tx_cost_bps": tx_cost_bps,
            "prefer": prefer_em,
            "n_specs": len(specs),
            "notional_usd": notional_usd,
            "trade_mode": trade_mode,
            "multiplier": multiplier,
        }

        n_trades = sum(len(t) for t in results.values())
        n_with_trades = sum(1 for t in results.values() if t)
        st.success(
            f"Done in {elapsed:.1f}s — "
            f"{len(specs)} {'strategy' if len(specs) == 1 else 'strategies'} "
            f"({n_with_trades} produced trades), {n_trades} trades total."
        )

    # ---- Summary table ----
    if "rko_wo_results" in st.session_state:
        _render_worstof_summary()


def _render_worstof_summary() -> None:
    """Render the worst-of summary table from the latest backtest."""
    st.markdown("---")
    st.markdown("### Summary across worst-of strategies")

    from core.worstof import worstof_trades_to_df, worstof_summarize

    results = st.session_state["rko_wo_results"]
    meta = st.session_state.get("rko_wo_meta", {})

    st.caption(
        f"Run: {meta.get('start')} → {meta.get('end')}  ·  "
        f"notional ${meta.get('notional_usd', 0):,.0f}  ·  "
        f"multiplier **{int(meta.get('multiplier', 0.40)*100)}%**  ·  "
        f"tx cost {meta.get('tx_cost_bps', 0):.1f} bps  ·  "
        f"trade mode: **{meta.get('trade_mode', 'stack')}**  ·  "
        f"elapsed {st.session_state.get('rko_wo_elapsed', 0):.1f}s"
    )

    rows = []
    for name, trades in results.items():
        df = worstof_trades_to_df(trades)
        if df.empty:
            rows.append({
                "Strategy": name, "n trades": 0,
                "Either leg KO%": float("nan"),
                "Both survive%": float("nan"),
                "Win %": float("nan"),
                "Avg structure premium": "",
                "Avg min single leg": "",
                "Implied mult": float("nan"),
                "Total P&L ($)": "",
                "Max DD ($)": "",
                "Sharpe": float("nan"),
                "Recovery %": float("nan"),
            })
            continue
        s = worstof_summarize(df)
        min_leg_mid = df[["leg_a_premium_mid_usd",
                            "leg_b_premium_mid_usd"]].min(axis=1).mean()
        rows.append({
            "Strategy": name,
            "n trades": s["n_trades"],
            "Either leg KO%": s["any_ko_rate"] * 100,
            "Both survive%": s["both_survive_rate"] * 100,
            "Win %": s["win_rate"] * 100,
            "Avg structure premium": f"${s.get('avg_premium_paid_usd', 0):,.0f}",
            "Avg min single leg": f"${min_leg_mid:,.0f}",
            "Implied mult": (
                s.get("avg_premium_paid_usd", 0) / min_leg_mid * 100
                if min_leg_mid else float("nan")),
            "Total P&L ($)": f"${s.get('total_pnl_usd', 0):,.0f}",
            "Max DD ($)": f"${s.get('max_drawdown_usd', 0):,.0f}",
            "Sharpe": s.get("sharpe_monthly", float("nan")),
            "Recovery %": s.get("premium_recovery_pct", float("nan")),
        })

    sdf = pd.DataFrame(rows)
    sdf = sdf.sort_values("Sharpe", ascending=False, na_position="last")

    st.caption(
        "💡 `Implied mult` is what the structure premium turned out to be "
        "as a % of the cheaper single leg — should equal the multiplier "
        "set in the sidebar, including the small tx-cost markup. "
        "`Either leg KO%` is the structure-level knockout rate (one leg "
        "is enough to kill it). `Recovery %` = payouts / premiums paid."
    )
    st.dataframe(
        sdf.style.format({
            "Either leg KO%": "{:.1f}",
            "Both survive%": "{:.1f}",
            "Win %": "{:.1f}",
            "Implied mult": "{:.0f}%",
            "Sharpe": "{:.2f}",
            "Recovery %": "{:.0f}",
        }, na_rep="—"),
        use_container_width=True, hide_index=True,
    )

    n_total = sum(len(t) for t in results.values())
    st.caption(
        f"_{len(results)} strategies, {n_total} trades total. "
        f"→ Switch to **Worst-of drilldown** to inspect a single strategy._"
    )

    # ---- Bulk download (summary + monthly/annual time series) ----
    st.markdown("---")
    st.markdown("#### Bulk download")
    st.caption(
        "Worst-of CSVs in the same canonical schema. Single-leg-specific "
        "columns (ko_rate_pct, feasibility_pct) are NaN; WO-specific "
        "columns (leg_a_ko_rate_pct, both_survive_rate_pct, etc.) are "
        "populated."
    )
    _render_download_buttons_worstof(results, file_prefix="rko_worstof",
                                            key_prefix="wo12")


# =============================================================================
# Worst-of drilldown tab
# =============================================================================
def render_worstof_drilldown_tab() -> None:
    if "rko_wo_results" not in st.session_state:
        st.info(
            "No worst-of backtest results yet. Run a backtest in the "
            "**Worst-of** tab first, then come back here to inspect a "
            "single strategy."
        )
        return

    from core.worstof import (worstof_trades_to_df, worstof_summarize,
                                  worstof_equity_curve, worstof_monthly_pnl,
                                  worstof_annual_summary)

    results = st.session_state["rko_wo_results"]
    meta = st.session_state.get("rko_wo_meta", {})

    names = list(results.keys())
    if not names:
        st.warning("No strategies in the latest worst-of backtest.")
        return

    default_idx = max(range(len(names)), key=lambda i: len(results[names[i]]))
    sel = st.selectbox(
        "Strategy", names, index=default_idx, key="rko_wod_strategy",
    )
    trades = results.get(sel, [])
    df = worstof_trades_to_df(trades)
    if df.empty:
        st.warning(f"Strategy `{sel}` produced no trades.")
        return

    s = worstof_summarize(df)
    eq = worstof_equity_curve(df)

    # ---- Engine / correlation banner (Step R3) ----
    # Surface the pricing-engine context above the metrics so users
    # immediately see which engine produced this ledger. Crucial when
    # comparing legacy vs CF/MC runs side by side.
    if "pricing_engine" in df.columns:
        engine_used = df["pricing_engine"].iloc[0]
        engine_label = {
            "legacy_multiplier":     "Legacy multiplier (multiplier × min)",
            "cf_approx_american":    "CF-approx American (~2 ms/trade)",
            "monte_carlo_american":  "Monte Carlo American (canonical)",
            "closed_form":           "Closed-form (European)",   # not for this page, but defensive
            "monte_carlo":           "Monte Carlo (European)",
        }.get(engine_used, engine_used)
        corr_used_avg = (df["correlation_used"].dropna().mean()
                          if "correlation_used" in df.columns else None)
        corr_src = (df["correlation_source_used"].iloc[0]
                     if "correlation_source_used" in df.columns else "—")
        if engine_used != "legacy_multiplier":
            avg_legacy = (df["structure_premium_legacy_usd"].dropna().mean()
                            if "structure_premium_legacy_usd" in df.columns
                            else None)
            avg_engine = df["structure_premium_mid_usd"].mean()
            ratio_str = (f"  ·  engine/legacy ratio = "
                          f"{avg_engine/avg_legacy:.2f}"
                          if avg_legacy and avg_legacy > 0 else "")
            corr_str = (f"  ·  avg ρ = {corr_used_avg:+.3f}  ·  "
                         f"source = {corr_src}"
                         if corr_used_avg is not None else "")
            st.info(
                f"**Engine**: {engine_label}{corr_str}{ratio_str}  ·  "
                f"avg legacy: ${avg_legacy:,.0f}  ·  "
                f"avg engine: ${avg_engine:,.0f}"
            )
        else:
            st.info(f"**Engine**: {engine_label}")

    # ---- Top-line metrics ----
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Trades", f"{s['n_trades']}")
    c2.metric("Either leg KO", f"{s['any_ko_rate']*100:.1f}%")
    c3.metric("Both survived", f"{s['both_survive_rate']*100:.1f}%")
    c4.metric("Total P&L", f"${s.get('total_pnl_usd', 0):,.0f}")
    c5.metric("Max drawdown", f"${s.get('max_drawdown_usd', 0):,.0f}")

    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("Win rate", f"{s['win_rate']*100:.1f}%")
    c7.metric("Sharpe (monthly)", f"{s.get('sharpe_monthly', 0):.2f}")
    c8.metric("Premium recovery", f"{s.get('premium_recovery_pct', 0):.0f}%")
    c9.metric("Multiplier",
                 f"{int(df['multiplier'].iloc[0]*100)}%")
    c10.metric("Leg A KO rate", f"{s['leg_a_ko_rate']*100:.1f}%")

    st.markdown("---")

    # ---- Equity & drawdown ----
    st.markdown("#### Equity curve and drawdown (USD)")
    if not eq.empty and "equity_usd" in eq.columns:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.65, 0.35], vertical_spacing=0.05,
            subplot_titles=("Cumulative P&L (USD)", "Drawdown (USD)"),
        )
        fig.add_trace(
            go.Scatter(x=eq.index, y=eq["equity_usd"], mode="lines",
                          line=dict(color="#1f77b4", width=2),
                          name="Equity"),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=eq.index, y=eq["drawdown_usd"], mode="lines",
                          line=dict(color="#d62728", width=1.5),
                          fill="tozeroy",
                          fillcolor="rgba(214, 39, 40, 0.15)",
                          name="Drawdown"),
            row=2, col=1,
        )
        fig.update_layout(height=520, showlegend=False, hovermode="x unified",
                            margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No equity curve data available.")

    st.markdown("---")

    # ---- Monthly P&L heatmap ----
    st.markdown("#### Monthly P&L heatmap (USD)")
    monthly_usd = worstof_monthly_pnl(df)
    if not monthly_usd.empty:
        # Pivot to year × month
        mdf = monthly_usd.copy()
        mdf.index = pd.to_datetime(mdf.index)
        pivot = mdf.groupby([mdf.index.year, mdf.index.month]).sum().unstack(
            fill_value=0)
        # Rename columns to short month labels
        month_labels = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                          7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        pivot = pivot.rename(columns=month_labels)
        # Add YTD column
        pivot["YTD"] = pivot.sum(axis=1)
        st.dataframe(
            _diverging_red_yellow_green(pivot).format("${:,.0f}", na_rep=""),
            use_container_width=True,
        )
    else:
        st.info("Not enough data for monthly summary.")

    st.markdown("---")

    # ---- Annual summary ----
    st.markdown("#### Annual summary")
    annual_df = worstof_annual_summary(df)
    if not annual_df.empty:
        fmt_map = {}
        for col in annual_df.columns:
            ser = annual_df[col]
            if pd.api.types.is_numeric_dtype(ser):
                if "pct" in col.lower() or "%" in col:
                    fmt_map[col] = "{:.1f}%"
                elif "sharpe" in col.lower() or "calmar" in col.lower():
                    fmt_map[col] = "{:.2f}"
                elif "usd" in col.lower() or "$" in col:
                    fmt_map[col] = "${:,.0f}"
                else:
                    fmt_map[col] = "{:.2f}"
        st.dataframe(annual_df.style.format(fmt_map, na_rep="—"),
                       use_container_width=True, hide_index=False)
    else:
        st.info("Not enough data for annual breakdown.")

    st.markdown("---")

    # ---- Trade ledger ----
    st.markdown("#### Trade ledger")
    st.caption(
        f"{len(df)} trades. American-barrier KO check: each leg scans "
        f"daily [Low, High] for barrier touch. `_a_` columns refer to "
        f"leg A; `_b_` to leg B."
    )

    cols_show = [
        "trade_date", "expiry_date",
        "leg_a_pair", "leg_a_spot", "leg_a_strike", "leg_a_barrier",
        "leg_a_knocked_out", "leg_a_payoff_usd", "leg_a_premium_mid_usd",
        "leg_b_pair", "leg_b_spot", "leg_b_strike", "leg_b_barrier",
        "leg_b_knocked_out", "leg_b_payoff_usd", "leg_b_premium_mid_usd",
        "structure_premium_paid_usd", "worst_of_payoff_usd", "pnl_usd",
        "multiplier",
    ]
    cols_have = [c for c in cols_show if c in df.columns]
    df_view = df[cols_have].copy()

    for c in ["leg_a_spot", "leg_a_strike", "leg_a_barrier",
                "leg_b_spot", "leg_b_strike", "leg_b_barrier"]:
        if c in df_view.columns:
            df_view[c] = df_view[c].apply(
                lambda x: f"{x:,.4f}" if pd.notna(x) else "")
    for c in ["leg_a_payoff_usd", "leg_a_premium_mid_usd",
                "leg_b_payoff_usd", "leg_b_premium_mid_usd",
                "structure_premium_paid_usd", "worst_of_payoff_usd",
                "pnl_usd"]:
        if c in df_view.columns:
            df_view[c] = df_view[c].apply(
                lambda x: f"${x:,.0f}" if pd.notna(x) else "")
    if "multiplier" in df_view.columns:
        df_view["multiplier"] = df_view["multiplier"].apply(
            lambda x: f"{x*100:.0f}%" if pd.notna(x) else "")

    st.dataframe(df_view, use_container_width=True, hide_index=True)

    csv_bytes = df.to_csv(index=False).encode()
    safe_name = sel.replace("/", "_").replace(" ", "_").replace(",", "")
    st.download_button(
        f"⬇ Download full worst-of ledger (CSV)",
        data=csv_bytes,
        file_name=f"rko_worstof_{safe_name}.csv",
        mime="text/csv",
        key="rko_wod_csv",
    )


# =============================================================================
# Worst-of pricer card — single-snapshot view at the top of the Worst-of tab
# =============================================================================
def _render_worstof_pricer_card(folder: str, multiplier: float,
                                       pairs_avail: list) -> None:
    """One-shot pricer for a worst-of structure on the latest available
    market data. Mirrors the Pricer tab UX but for two legs simultaneously:
    user picks two pairs, direction, strike Δ, KO Δ, tenor → we price each
    leg with VV-on-American, then show:

      - per-leg metrics (K, H, σ_atm, premium)
      - structure-level metrics (premium, max payoff, hit-the-cap leverage)
      - a small comparison table: how the structure premium scales across
        all three multipliers
    """
    from datetime import date as _date
    from core.calendar import compute_option_dates
    from core.data_loader import load_panel, load_by_ticker
    from core.rates import load_rates_panel, get_rate_at
    from core.ko_solvers import solve_strike_from_delta, solve_barrier_from_delta
    from core.american_barrier import ako_closed_form
    from core.vanna_volga import vv_price_ko
    from core.smile import smile_vol_at_strike
    from core.backtest import _interp_panels_at_T
    from core.conventions import get_pip_scale

    if len(pairs_avail) < 2:
        st.warning("Worst-of pricer needs at least two pairs.")
        return

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        default_a = "USDJPY" if "USDJPY" in pairs_avail else pairs_avail[0]
        default_b = ("USDKRW" if "USDKRW" in pairs_avail and "USDKRW" != default_a
                       else (pairs_avail[1] if len(pairs_avail) > 1 else pairs_avail[0]))
        pair_a = st.selectbox("Leg A pair", pairs_avail,
                                  index=pairs_avail.index(default_a),
                                  key="rko_wop_pair_a")
        # Bias default leg B index to first that's not pair_a
        pair_b_options = [p for p in pairs_avail if p != pair_a]
        if not pair_b_options:
            st.warning("Pick distinct pairs for the two legs.")
            return
        default_b_idx = (pair_b_options.index(default_b)
                            if default_b in pair_b_options else 0)
        pair_b = st.selectbox("Leg B pair", pair_b_options,
                                  index=default_b_idx, key="rko_wop_pair_b")
        prefer = st.radio("EM variant", ["offshore", "onshore"], index=0,
                              horizontal=True, key="rko_wop_prefer")

    with cc2:
        direction_label = st.selectbox(
            "Direction", list(DIRECTIONS.keys()), index=0, key="rko_wop_dir")
        direction, barrier_type = DIRECTIONS[direction_label]
        strike_label = st.selectbox("Strike Δ", list(DELTA_CHOICES.keys()),
                                         index=0, key="rko_wop_strike_delta")
        strike_delta = DELTA_CHOICES[strike_label]
        ko_label = st.selectbox("KO Δ (vanilla wing)", list(KO_DELTA_CHOICES.keys()),
                                     index=1, key="rko_wop_ko_delta")
        ko_delta = KO_DELTA_CHOICES[ko_label]

    with cc3:
        tenor_label = st.selectbox("Tenor", TENOR_LIST,
                                        index=TENOR_LIST.index("1M"),
                                        key="rko_wop_tenor")
        notional_usd = st.number_input(
            "Notional (USD, per leg)", min_value=100_000.0,
            max_value=200_000_000.0, value=10_000_000.0, step=1_000_000.0,
            format="%.0f", key="rko_wop_notional",
            help="Same foreign notional on each leg.")
        tx_cost_bps = st.slider(
            "Transaction cost (bps)", 0.0, 20.0, 2.0, 0.5,
            key="rko_wop_txcost",
            help="Flat bps markup on foreign notional at the structure "
                  "level (legs stay at mid).")

    # ---- Load latest market data for both legs ----
    # Use latest common business day across the two pairs.
    try:
        spot_a_df = load_panel(folder, "SPOT", None, prefer=prefer,
                                   pairs=(pair_a,))
        spot_b_df = load_panel(folder, "SPOT", None, prefer=prefer,
                                   pairs=(pair_b,))
    except Exception as e:
        st.error(f"Could not load spot data: {e}")
        return
    if spot_a_df.empty or pair_a not in spot_a_df.columns:
        st.error(f"No SPOT data for {pair_a}.")
        return
    if spot_b_df.empty or pair_b not in spot_b_df.columns:
        st.error(f"No SPOT data for {pair_b}.")
        return

    spot_a = spot_a_df[pair_a].dropna()
    spot_b = spot_b_df[pair_b].dropna()
    if spot_a.empty or spot_b.empty:
        st.error("Spot series came back empty after dropping NaNs.")
        return

    # Latest common date
    common_dates = spot_a.index.intersection(spot_b.index)
    if len(common_dates) == 0:
        st.error("No common business days between the two spot series.")
        return
    td_ts = common_dates.max()
    td = td_ts.date()

    S_a = float(spot_a.loc[td_ts])
    S_b = float(spot_b.loc[td_ts])

    opt_dates = compute_option_dates(td, tenor_label)
    T = opt_dates.T_years

    # Per-leg pricing helper
    def _price_leg(pair: str, S: float, strike_delta_v: float,
                       ko_delta_v: float) -> dict | None:
        # Vol panels (use the standard tenor set; smile_vol_at_strike handles
        # the smile interpolation downstream)
        vol_df = load_panel(folder, "VOL_ATM", tenor_label, prefer=prefer,
                                pairs=(pair,))
        if vol_df.empty or pair not in vol_df.columns:
            return None
        sigma_atm_pct = vol_df[pair].asof(td_ts)
        if pd.isna(sigma_atm_pct):
            return None
        sigma_atm = float(sigma_atm_pct) / 100.0

        rr_df = load_panel(folder, "VOL_RR_25D", tenor_label, prefer=prefer,
                              pairs=(pair,))
        bf_df = load_panel(folder, "VOL_BF_25D", tenor_label, prefer=prefer,
                              pairs=(pair,))
        rr_25 = bf_25 = 0.0
        smile_avail = False
        if (not rr_df.empty and pair in rr_df.columns
                and not bf_df.empty and pair in bf_df.columns):
            rr_v = rr_df[pair].asof(td_ts)
            bf_v = bf_df[pair].asof(td_ts)
            if pd.notna(rr_v) and pd.notna(bf_v):
                rr_25 = float(rr_v) / 100.0
                bf_25 = float(bf_v) / 100.0
                smile_avail = True

        fwd_df = load_panel(folder, "FWD_POINTS", tenor_label, prefer=prefer,
                                pairs=(pair,))
        pip = get_pip_scale(pair)
        F_market = S
        if not fwd_df.empty and pair in fwd_df.columns:
            fwd_v = fwd_df[pair].asof(td_ts)
            if pd.notna(fwd_v):
                F_market = S + float(fwd_v) * pip

        foreign, domestic = pair[:3], pair[3:]
        f_panel = load_rates_panel(folder, foreign, load_by_ticker)
        d_panel = load_rates_panel(folder, domestic, load_by_ticker)
        r_f = get_rate_at(f_panel, T, td)
        r_d = get_rate_at(d_panel, T, td)
        if r_f is None and r_d is None:
            return None
        if r_f is None:
            r_f = r_d - np.log(F_market / S) / T
        if r_d is None:
            r_d = r_f + np.log(F_market / S) / T

        # Solve K from strike Δ at σ_atm. ATM-fwd if delta_v == 0.
        if strike_delta_v == 0.0:
            K = S * np.exp((r_d - r_f) * T)
        else:
            K = solve_strike_from_delta(direction, strike_delta_v,
                                             S, T, sigma_atm, r_d, r_f)
        # Solve H from KO Δ at σ_atm
        H = solve_barrier_from_delta(barrier_type, ko_delta_v,
                                          S, T, sigma_atm, r_d, r_f)
        # Smile vol at K (used as a reference; VV uses ATM + RR + BF
        # directly, not σ_smile(K))
        sigma_smile = smile_vol_at_strike(S, K, T, sigma_atm, rr_25, bf_25,
                                              r_d, r_f)

        # VV-on-American-barrier
        vv_out = vv_price_ko(direction, barrier_type, S, K, H, T,
                                sigma_atm, rr_25, bf_25, r_d, r_f,
                                flat_vol_pricer=ako_closed_form)
        prem_per_unit = vv_out["price_vv"]
        prem_flat_per_unit = vv_out["price_bs"]

        max_pay_per_unit = abs(H - K)
        return {
            "S": S, "K": K, "H": H,
            "sigma_atm": sigma_atm, "rr_25": rr_25, "bf_25": bf_25,
            "sigma_smile": sigma_smile, "r_d": r_d, "r_f": r_f,
            "prem_per_unit": prem_per_unit,
            "prem_flat_per_unit": prem_flat_per_unit,
            "prem_usd": prem_per_unit / S * notional_usd,
            "prem_flat_usd": prem_flat_per_unit / S * notional_usd,
            "max_pay_per_unit": max_pay_per_unit,
            "max_pay_usd": max_pay_per_unit / S * notional_usd,
            "smile_avail": smile_avail,
        }

    leg_a = _price_leg(pair_a, S_a, strike_delta, ko_delta)
    leg_b = _price_leg(pair_b, S_b, strike_delta, ko_delta)
    if leg_a is None or leg_b is None:
        st.warning(
            "Could not price one of the legs at this snapshot — "
            "check vol / rates / forward data for the chosen tenor."
        )
        return

    # ---- Structure-level math ----
    structure_mid_usd = multiplier * min(leg_a["prem_usd"], leg_b["prem_usd"])
    tx_cost_usd = tx_cost_bps * notional_usd / 10_000
    structure_paid_usd = structure_mid_usd + tx_cost_usd
    # Max payoff: floored by the smaller-leg max
    structure_max_pay_usd = min(leg_a["max_pay_usd"], leg_b["max_pay_usd"])
    structure_leverage = (structure_max_pay_usd / structure_paid_usd
                           if structure_paid_usd > 0 else float("nan"))

    # ---- Date + spot context ----
    smile_status = ("smile-adjusted (VV)"
                      if (leg_a["smile_avail"] and leg_b["smile_avail"])
                      else "VV falls back to flat-vol on missing-smile legs")
    st.caption(
        f"As of **{td}**  ·  spot_A {pair_a}={leg_a['S']:.4f}  ·  "
        f"spot_B {pair_b}={leg_b['S']:.4f}  ·  T={T:.4f}y  ·  "
        f"{smile_status}"
    )

    # ---- Structure metrics row ----
    cm1, cm2, cm3, cm4 = st.columns(4)
    cm1.metric("Structure premium",
                  f"${structure_paid_usd:,.0f}",
                  f"{structure_paid_usd/notional_usd*100:.3f}% notl")
    cm2.metric(f"Structure mid (×{int(multiplier*100)}%)",
                  f"${structure_mid_usd:,.0f}",
                  f"vs min(leg_A, leg_B)=${min(leg_a['prem_usd'], leg_b['prem_usd']):,.0f}")
    cm3.metric("Max payoff (cap)",
                  f"${structure_max_pay_usd:,.0f}",
                  f"min of leg caps")
    cm4.metric("Leverage (max / paid)",
                  f"{structure_leverage:.1f}×" if np.isfinite(structure_leverage)
                  else "—")

    # ---- Per-leg detail table ----
    st.markdown("**Per-leg breakdown**")
    leg_rows = []
    for pair, leg in [(pair_a, leg_a), (pair_b, leg_b)]:
        leg_rows.append({
            "Leg": pair,
            "S": f"{leg['S']:.4f}",
            "K": f"{leg['K']:.4f}",
            "H": f"{leg['H']:.4f}",
            "σ_atm": f"{leg['sigma_atm']*100:.3f}%",
            "σ_smile(K)": f"{leg['sigma_smile']*100:.3f}%",
            "RR_25": f"{leg['rr_25']*100:+.3f}%",
            "BF_25": f"{leg['bf_25']*100:+.3f}%",
            "Premium (VV)": f"${leg['prem_usd']:,.0f}",
            "Premium (flat)": f"${leg['prem_flat_usd']:,.0f}",
            "Smile lift": (f"{(leg['prem_usd']/leg['prem_flat_usd']-1)*100:+.2f}%"
                            if leg["prem_flat_usd"] > 0 else "—"),
            "Max payoff": f"${leg['max_pay_usd']:,.0f}",
        })
    st.dataframe(pd.DataFrame(leg_rows), use_container_width=True,
                   hide_index=True)

    # ---- Multiplier comparison table ----
    st.markdown("**Multiplier sweep** — structure premium across all three "
                 "settings, holding leg pricing fixed")
    sweep_rows = []
    min_leg = min(leg_a["prem_usd"], leg_b["prem_usd"])
    for m_pct in [33, 40, 50]:
        m = m_pct / 100.0
        mid = m * min_leg
        paid = mid + tx_cost_usd
        lev = structure_max_pay_usd / paid if paid > 0 else float("nan")
        row = {
            "Multiplier": f"{m_pct}%",
            "Structure mid": f"${mid:,.0f}",
            "Structure paid": f"${paid:,.0f}",
            "% of notl": f"{paid/notional_usd*100:.3f}%",
            "Leverage": f"{lev:.1f}×" if np.isfinite(lev) else "—",
        }
        if m_pct == int(multiplier * 100):
            row["Multiplier"] = f"**{m_pct}%** (current)"
        sweep_rows.append(row)
    st.dataframe(pd.DataFrame(sweep_rows), use_container_width=True,
                   hide_index=True)
    st.caption(
        "💡 Switch the multiplier in the sidebar to change the default for "
        "the backtest below. The leg premiums above are smile-adjusted "
        "(VV-on-American); the flat-vol column is shown for cross-check "
        "against Bloomberg's BS-only price."
    )

    # =========================================================================
    # Step R5 — Engine comparison: legacy multiplier vs CF-approx vs MC
    # =========================================================================
    # Three engines priced from the SAME per-leg inputs as above:
    #
    #   Legacy multiplier  : multiplier × min(leg_A_VV, leg_B_VV)
    #                        (the value already shown in the metrics row)
    #
    #   CF-approx American : core.worstof_pricer_american
    #                        .worstof_rko_price_cf_approx — ~2 ms,
    #                        ρ-aware via the European worst-of CF.
    #
    #   MC American (BB)   : core.worstof_pricer_american
    #                        .worstof_rko_price_mc with Brownian-bridge
    #                        sub-step touch correction — canonical
    #                        pricer for live/forward pricing where the
    #                        future OHLC isn't known yet. ~100 ms at
    #                        20k paths, ~400 ms at 100k.
    #
    # Correlation source matches the bulk/portfolio tabs:
    #   - Manual          : user-set ρ slider
    #   - Rolling 60d     : realized log-return correlation over the
    #                        last 60 business days from the spot panels
    #   - Triangulation   : implied ρ from the cross-pair's ATM vol at
    #                        the same tenor (falls back to Manual when
    #                        the cross-pair's vol panel isn't present)
    st.markdown("---")
    st.markdown("**Engine comparison** — same trade, three pricers")
    st.caption(
        "Compare the legacy multiplier formula against the correlation-"
        "aware pricers in `core.worstof_pricer_american`. Useful for "
        "sanity-checking the multiplier choice on tight/wide barriers "
        "and across different ρ regimes."
    )

    ec1, ec2 = st.columns([1, 1])
    with ec1:
        corr_src_label = st.radio(
            "ρ source",
            ["Manual (slider)", "Realized 60d", "Triangulation"],
            index=0, horizontal=True,
            key="rko_wop_corr_source",
            help=(
                "**Manual**: ρ from the slider below.  \n"
                "**Realized 60d**: rolling 60-business-day log-return "
                "correlation of (spot_A, spot_B) at the snapshot date.  \n"
                "**Triangulation**: implied ρ from the cross-pair's "
                "ATM vol at the same tenor. Falls back to Manual when "
                "the cross-pair's VOL_ATM panel is missing."
            ),
        )
    with ec2:
        rho_manual = st.slider(
            "Manual ρ (used directly, or as fallback)",
            min_value=-0.95, max_value=0.95, value=0.30, step=0.05,
            key="rko_wop_rho_manual",
        )

    # Resolve ρ for the snapshot date based on the chosen source
    rho_used = float(rho_manual)
    rho_source_used = "manual"
    if corr_src_label == "Realized 60d":
        from core.correlation import realized_correlation_at
        r60, _n = realized_correlation_at(spot_a, spot_b, td_ts, window=60)
        if r60 is not None and not pd.isna(r60):
            rho_used = float(r60)
            rho_source_used = "rolling_60d"
        else:
            rho_source_used = "rolling_60d_fallback_to_manual"
    elif corr_src_label == "Triangulation":
        from core.correlation import implied_correlation_at_T
        try:
            tri = implied_correlation_at_T(
                folder, pair_a, pair_b, T, td_ts,
                prefer_a=prefer, prefer_b=prefer, prefer_cross=prefer,
            )
            if (tri is not None and tri.rho_implied is not None
                    and not np.isnan(tri.rho_implied)):
                rho_used = float(tri.rho_implied)
                rho_source_used = "triangulation"
            else:
                rho_source_used = "triangulation_fallback_to_manual"
        except Exception:
            rho_source_used = "triangulation_fallback_to_manual"

    # Build WorstOfLeg objects (engine convention: each leg's spot
    # normalized to 1.0 with K, H rescaled).
    from core.worstof_pricer import WorstOfLeg
    from core.worstof_pricer_american import (
        worstof_rko_price_cf_approx, worstof_rko_price_mc,
    )
    import time as _t

    wol_a = WorstOfLeg(
        S=1.0, K=leg_a["K"] / leg_a["S"], H=leg_a["H"] / leg_a["S"],
        sigma=leg_a["sigma_smile"], r_d=leg_a["r_d"], r_f=leg_a["r_f"],
        opt=direction, bar_dir=barrier_type,
    )
    wol_b = WorstOfLeg(
        S=1.0, K=leg_b["K"] / leg_b["S"], H=leg_b["H"] / leg_b["S"],
        sigma=leg_b["sigma_smile"], r_d=leg_b["r_d"], r_f=leg_b["r_f"],
        opt=direction, bar_dir=barrier_type,
    )
    # Discount at leg A's DOM (mixed-measure convention — same as engine)
    r_d_struct = leg_a["r_d"]

    # CF-approx — always run (fast)
    t0 = _t.perf_counter()
    cf_out = worstof_rko_price_cf_approx(wol_a, wol_b, T, rho_used,
                                            r_d=r_d_struct, n_quad=60)
    cf_ms = (_t.perf_counter() - t0) * 1000
    cf_struct_mid_usd = cf_out["price"] * notional_usd

    # MC — gated behind a button (slower). Use 20k paths for the
    # snapshot pricer as the trade-off between accuracy and UX.
    mc_btn_col, mc_paths_col = st.columns([1, 1])
    with mc_btn_col:
        run_mc = st.button("▶ Run MC (Brownian-bridge)",
                              key="rko_wop_run_mc", type="primary")
    with mc_paths_col:
        mc_n_paths_snap = st.select_slider(
            "MC paths",
            options=[10_000, 20_000, 50_000, 100_000, 200_000],
            value=20_000, key="rko_wop_mc_paths",
            help=("Live snapshot uses fewer paths than backtest defaults "
                   "for UX. SE scales as 1/√n."),
        )

    mc_out = None
    mc_ms = None
    if run_mc:
        with st.spinner(f"Running MC ({mc_n_paths_snap:,} paths, "
                          f"Brownian-bridge monitoring)…"):
            t0 = _t.perf_counter()
            mc_out = worstof_rko_price_mc(
                wol_a, wol_b, T, rho_used, r_d=r_d_struct,
                n_paths=mc_n_paths_snap, seed=42,
                monitoring="brownian_bridge",
            )
            mc_ms = (_t.perf_counter() - t0) * 1000
            mc_struct_mid_usd = mc_out["price"] * notional_usd
        # Cache so re-runs with same params don't always recompute. We
        # don't bother with full memoisation — fresh run each click is
        # the simplest behaviour and signals the MC noise visually.

    # ---- Resolved ρ caption + comparison table ----
    src_to_label = {
        "manual":                            "Manual (slider)",
        "rolling_60d":                       "Realized 60d",
        "rolling_60d_fallback_to_manual":    "Realized 60d (no data → Manual)",
        "triangulation":                     "Triangulation",
        "triangulation_fallback_to_manual":  "Triangulation (no data → Manual)",
    }
    st.caption(
        f"**ρ = {rho_used:+.4f}**  ·  source: "
        f"{src_to_label.get(rho_source_used, rho_source_used)}  ·  "
        f"discount rate (leg A DOM): r_d = {r_d_struct*100:.3f}%"
    )

    engine_rows = [
        {
            "Engine": "Legacy multiplier",
            "Structure mid": f"${structure_mid_usd:,.0f}",
            "vs legacy": "1.00×",
            "Timing": "—",
            "Detail": f"multiplier = {int(multiplier*100)}%  ·  "
                       f"× min(leg_A, leg_B) = "
                       f"${min(leg_a['prem_usd'], leg_b['prem_usd']):,.0f}",
        },
        {
            "Engine": "CF-approx American",
            "Structure mid": f"${cf_struct_mid_usd:,.0f}",
            "vs legacy": (f"{cf_struct_mid_usd/structure_mid_usd:.2f}×"
                            if structure_mid_usd > 0 else "—"),
            "Timing": f"{cf_ms:.1f} ms",
            "Detail": f"P_EKO_WO = ${cf_out['p_eko_wo']*notional_usd:,.0f}  ·  "
                       f"ratio_A = {cf_out['ratio_a']:.3f}  ·  "
                       f"ratio_B = {cf_out['ratio_b']:.3f}",
        },
    ]
    if mc_out is not None:
        mc_struct_mid_usd = mc_out["price"] * notional_usd
        mc_se_usd = mc_out["std_err"] * notional_usd
        engine_rows.append({
            "Engine": "Monte Carlo American (BB)",
            "Structure mid": f"${mc_struct_mid_usd:,.0f}",
            "vs legacy": (f"{mc_struct_mid_usd/structure_mid_usd:.2f}×"
                            if structure_mid_usd > 0 else "—"),
            "Timing": f"{mc_ms:.0f} ms",
            "Detail": (f"SE = ±${mc_se_usd:,.0f}  ·  "
                        f"p_surv_A = {mc_out['p_survive_a']:.3f}  ·  "
                        f"p_surv_B = {mc_out['p_survive_b']:.3f}  ·  "
                        f"p_surv_joint = {mc_out['p_survive_joint']:.3f}"),
        })
    else:
        engine_rows.append({
            "Engine": "Monte Carlo American (BB)",
            "Structure mid": "(click ▶ Run MC)",
            "vs legacy": "—",
            "Timing": "—",
            "Detail": "Brownian-bridge sub-step monitoring; "
                       "canonical pricer for live valuation.",
        })

    st.dataframe(pd.DataFrame(engine_rows), use_container_width=True,
                   hide_index=True)

    # ---- Interpretation hint ----
    if mc_out is not None:
        # Direction of CF vs MC bias
        ratio_cf_mc = cf_struct_mid_usd / max(mc_struct_mid_usd, 1e-9)
        if abs(ratio_cf_mc - 1.0) < 0.10:
            bias_note = "(CF and MC agree within 10% — high confidence)"
        elif ratio_cf_mc < 1.0:
            bias_note = (f"(CF is {(1 - ratio_cf_mc)*100:.0f}% below MC "
                          f"— tight barrier regime, MC is the reference)")
        else:
            bias_note = (f"(CF is {(ratio_cf_mc - 1)*100:.0f}% above MC "
                          f"— unusual; cross-check inputs)")
        st.caption(bias_note)

    # =========================================================================
    # Greeks — finite-difference Greeks on the CF-approx pricer
    # =========================================================================
    # Computed via central finite differences (core.worstof_greeks).
    # Bumped over the same WorstOfLeg objects used by the engine
    # comparison above, so the Greeks correspond to the CF-approx
    # price shown. The CF Greeks are fast (~40 ms total) so we
    # auto-compute them. MC Greeks (5-10× slower) are behind a button.
    st.markdown("---")
    st.markdown("**Greeks** (per leg-A notional, FD on CF-approx pricer)")
    st.caption(
        "Δ = per-spot sensitivity (dimensionless when each leg's spot is "
        "normalized to 1).  Γ = spot convexity.  ν = vega per 1 vol "
        "point (×0.01 in σ).  ∂V/∂ρ = correlation sensitivity per 1 "
        "rho point (×0.01 in ρ).  Θ = -∂V/∂T per calendar day "
        "(negative = value decreases with time).  "
        "**USD Greeks** scale by notional ($"
        f"{notional_usd:,.0f}) per leg."
    )
    from core.worstof_greeks import worstof_greeks_fd

    g_cf = worstof_greeks_fd(
        wol_a, wol_b, T, rho_used, r_d=r_d_struct,
        pricer=worstof_rko_price_cf_approx,
    )

    # Scale for display: convert per-leg vega and rho sensitivity to
    # the trader-friendly "per 1 vol point" / "per 1 rho point" basis.
    vega_a_per_vp = g_cf.vega_a * 0.01
    vega_b_per_vp = g_cf.vega_b * 0.01
    rho_sens_per_rp = g_cf.rho_sensitivity * 0.01

    # USD scaling: per-unit values are in "% of leg notional", so
    # multiply by notional_usd.
    greeks_rows = [
        {
            "Greek": "Δ_A (per 1% leg-A spot)",
            "% of notional": f"{g_cf.delta_a * leg_a['S'] * 0.01 * 100:+.3f}%",
            "USD": f"${g_cf.delta_a * leg_a['S'] * 0.01 * notional_usd:+,.0f}",
            "Interpretation": "Long Δ → hedge by shorting leg-A spot",
        },
        {
            "Greek": "Δ_B (per 1% leg-B spot)",
            "% of notional": f"{g_cf.delta_b * leg_b['S'] * 0.01 * 100:+.3f}%",
            "USD": f"${g_cf.delta_b * leg_b['S'] * 0.01 * notional_usd:+,.0f}",
            "Interpretation": "Long Δ → hedge by shorting leg-B spot",
        },
        {
            "Greek": "Γ_A (per 1% × 1% spot²)",
            "% of notional": (f"{g_cf.gamma_a * (leg_a['S']*0.01)**2 * 100:+.4f}%"),
            "USD": (f"${g_cf.gamma_a * (leg_a['S']*0.01)**2 * notional_usd:+,.0f}"),
            "Interpretation": "PnL from 1% spot move beyond linear Δ",
        },
        {
            "Greek": "Γ_B (per 1% × 1% spot²)",
            "% of notional": (f"{g_cf.gamma_b * (leg_b['S']*0.01)**2 * 100:+.4f}%"),
            "USD": (f"${g_cf.gamma_b * (leg_b['S']*0.01)**2 * notional_usd:+,.0f}"),
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
                                 else "Daily time gain (rare; barrier "
                                       "domination)"),
        },
    ]
    st.dataframe(pd.DataFrame(greeks_rows), use_container_width=True,
                   hide_index=True)
    st.caption(
        f"Greeks via central finite differences on the CF-approx pricer "
        f"({g_cf.method}). Δ/Γ use ±{g_cf.bump_sizes['spot_frac']*100:.1f}% "
        f"spot bumps; ν use ±{g_cf.bump_sizes['sigma_abs']*100:.0f} vol "
        f"point; ρ uses ±{g_cf.bump_sizes['rho_abs']:.2f}; Θ uses "
        f"1-day forward bump. **For MC-noise-free Greeks under the canonical "
        f"MC pricer, click ▶ Run MC above and the MC price will use the "
        f"same CRN seed.**"
    )


# =============================================================================
# RKO Portfolio tab (American-barrier basket — App 12 equivalent of App 9 EKO)
# =============================================================================
# Same model as App 9's EKO Portfolio: pick a basket of pairs + one
# (tenor × direction × strike Δ × KO Δ × gate) combo, run the spec on
# every pair in the basket, pool the per-pair trade lists into one
# basket strategy. Difference: barriers are American (daily OHLC scan)
# and entry pricing is Vanna-Volga (smile-adjusted, matches BBG OVML).
#
# Notional is per-pair: 7 pairs × $10M = $70M deployed. The pooled
# trade ledger looks like a single mega-strategy with `pair` varying
# across rows.

RKO_PORT_DEFAULT_PAIRS = [
    "USDCNH", "USDINR", "USDKRW", "USDJPY",
    "USDSGD", "USDTHB", "USDTWD",
]


def _rko_basket_strategy_name(tenor: str, direction_label: str,
                                    strike_label: str, ko_label: str,
                                    gate_key) -> str:
    """Concise basket-strategy name for RKO. 'RKO' prefix distinguishes
    from EKO (European-barrier) baskets in mixed CSV ingestion."""
    from core.gates import gate_label
    gate_str = (f"  [{gate_label(gate_key)}]" if gate_key else "")
    dir_short = (direction_label.replace("Call (up-and-out)", "Call-UO")
                    .replace("Put (down-and-out)", "Put-DO"))
    return (f"RKO-BASKET  {dir_short}  {tenor}  {strike_label}/H@{ko_label}"
              f"{gate_str}")


def render_rko_portfolio_tab(folder: str) -> None:
    """App 12 RKO (Reverse/American-barrier KO) Portfolio. Mirrors App 9
    EKO Portfolio but uses `core.backtest_american.run_grid_american` so
    every leg gets Vanna-Volga-on-American pricing and daily OHLC KO
    monitoring."""
    pairs_avail = _list_pairs(folder)
    if not pairs_avail:
        st.error("No SPOT data found.")
        return

    st.markdown("### RKO Portfolio configuration")
    st.caption(
        "A **basket** portfolio backtest. One **strategy** = one "
        "`(tenor × direction × strike Δ × KO Δ × gate)` combination, "
        "applied **uniformly to every pair** in the basket. With 7 "
        "pairs × 1 tenor × 1 strike × 1 KO × 1 gate, that's 1 strategy "
        "holding 7 pair-positions per trading day. Notional applies "
        "PER PAIR (so $10M × 7 pairs = $70M deployed). "
        "**Pricing: Vanna-Volga on American closed-form** (matches "
        "Bloomberg OVML's 'Vanna-Volga' model). KO monitoring is "
        "**daily OHLC** — the structure dies on the first day any pair's "
        "[Low, High] range touches its barrier."
    )

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        default_pairs = [p for p in RKO_PORT_DEFAULT_PAIRS
                          if p in pairs_avail]
        if not default_pairs:
            default_pairs = pairs_avail[:1]
        pairs_sel = st.multiselect(
            "Currency pairs (the basket)", pairs_avail,
            default=default_pairs, key="rko_rp_pairs",
            help=("This is the BASKET. Every strategy in the run "
                   "applies its (tenor, strike, KO, gate, direction) "
                   "to all of these pairs simultaneously."),
        )
        tenors_sel = st.multiselect(
            "Tenors", TENOR_LIST, default=[DEFAULT_TENOR],
            key="rko_rp_tenors",
            help="Each selected tenor → its own basket strategy.",
        )
        directions_sel = st.multiselect(
            "Direction(s)", list(DIRECTIONS.keys()),
            default=["Call (up-and-out)"], key="rko_rp_directions",
            help="Each selected direction → its own basket strategy.",
        )

    with cc2:
        deltas_sel = st.multiselect(
            "Strike Δ list", list(DELTA_CHOICES.keys()),
            default=[DEFAULT_STRIKE_DELTA_LABEL], key="rko_rp_deltas",
            help=("Each selected Δ → its own basket strategy. "
                   "ATM (Δ=0) bypasses the KO-vs-strike filter."),
        )
        ko_delta_labels = st.multiselect(
            "KO Δ list", list(KO_DELTA_CHOICES.keys()),
            default=[DEFAULT_KO_DELTA_LABEL], key="rko_rp_ko_deltas",
            help=("Each selected KO Δ → its own basket strategy. Must be "
                   "more OTM than strike Δ except for ATM strikes."),
        )
        tx_cost_bps = st.slider(
            "Transaction cost (bps of notional)", 0.0, 20.0, 2.0, 0.5,
            key="rko_rp_txcost",
            help="Flat bps markup on the foreign notional, applied at "
                  "each per-pair trade level.",
        )
        from core.gates import GATE_REGISTRY
        gate_options = ["(no gate)"] + list(GATE_REGISTRY.keys())
        gate_labels_sel = st.multiselect(
            "Gate(s)", gate_options, default=["(no gate)"],
            key="rko_rp_gates",
            format_func=lambda k: ("(no gate)" if k == "(no gate)"
                                   else GATE_REGISTRY[k][0]),
            help="Each selection becomes its own basket strategy.",
        )
        gate_keys = [None if k == "(no gate)" else k for k in gate_labels_sel]

    with cc3:
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
        # Default start = 1 Jan 2023 (per spec), clamped to the available data range
        default_start = min(max(DEFAULT_START_DATE, date_min), date_max)
        start_date = st.date_input(
            "Start date", value=default_start,
            min_value=date_min, max_value=date_max, key="rko_rp_start",
        )
        end_date = st.date_input(
            "End date", value=date_max,
            min_value=date_min, max_value=date_max, key="rko_rp_end",
        )
        prefer_em = st.radio(
            "EM pairs variant", ["offshore", "onshore"], index=0,
            horizontal=True, key="rko_rp_prefer_em",
        )
        notional_usd = st.number_input(
            "Notional (USD, PER PAIR)", min_value=100_000.0,
            max_value=200_000_000.0, value=10_000_000.0, step=1_000_000.0,
            format="%.0f", key="rko_rp_notional",
            help=("Per-pair foreign notional. Total deployed = this × "
                  "n_pairs in the basket."),
        )
        trade_mode = st.radio(
            "Trade mode", ["stack", "single"], index=0, horizontal=True,
            key="rko_rp_trade_mode",
        )

    from itertools import product
    n_pairs = len(pairs_sel)
    axis_combos = list(product(tenors_sel, directions_sel, deltas_sel,
                                  ko_delta_labels, gate_keys))
    n_strategies = len(axis_combos)
    st.caption(
        f"**{n_strategies}** basket strategies will run "
        f"({len(tenors_sel)} tenors × {len(directions_sel)} dirs × "
        f"{len(deltas_sel)} strike Δs × {len(ko_delta_labels)} KO Δs × "
        f"{max(len(gate_keys), 1)} gates), each holding **{n_pairs} "
        f"pair-positions** per trading day. Total per-pair specs: "
        f"**{n_strategies * n_pairs}**."
    )

    can_run = (n_strategies > 0 and pairs_sel and tenors_sel
                and directions_sel and deltas_sel and ko_delta_labels)
    run_clicked = st.button("▶ Run RKO Portfolio", type="primary",
                              disabled=not can_run, key="rko_rp_run")

    if run_clicked:
        from core.backtest import StrategySpec, build_strategy_grid
        from core.backtest_american import run_grid_american

        all_basket_results: dict[str, list] = {}
        all_basket_specs: dict[str, list] = {}

        progress_bar = st.progress(0.0, text="Starting basket runs…")
        t0 = time.time()
        last_update = [t0]

        def cb(p, name):
            now = time.time()
            if now - last_update[0] > 0.1 or p >= 1.0:
                progress_bar.progress(min(p, 1.0),
                                       text=f"Running: {name} ({p*100:.0f}%)")
                last_update[0] = now

        # Step R1 — single-leg pricing model from sidebar.
        _pm = st.session_state.get("rko_pricing_model", "vanna_volga")
        for axis_i, (tenor, dir_label, strike_label, ko_label, gk) in \
                enumerate(axis_combos):
            basket_name = _rko_basket_strategy_name(
                tenor, dir_label, strike_label, ko_label, gk)
            specs_this = build_strategy_grid(
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
                pricing_model=_pm,
            )
            if not specs_this:
                all_basket_results[basket_name] = []
                all_basket_specs[basket_name] = []
                continue

            sub_results = run_grid_american(
                folder, specs_this, start_date, end_date,
                notional_usd=notional_usd,
                progress_cb=lambda p, name, _i=axis_i: cb(
                    (_i + p) / n_strategies,
                    f"[basket {_i+1}/{n_strategies}] {name}"),
            )
            pooled: list = []
            for s in specs_this:
                pooled.extend(sub_results.get(s.name, []))
            all_basket_results[basket_name] = pooled
            all_basket_specs[basket_name] = specs_this

        elapsed = time.time() - t0
        progress_bar.empty()

        st.session_state["rko_rp_results"] = all_basket_results
        st.session_state["rko_rp_specs"] = all_basket_specs
        st.session_state["rko_rp_meta"] = {
            "pairs": pairs_sel, "tenors": tenors_sel,
            "directions": directions_sel, "strike_deltas": deltas_sel,
            "ko_deltas": ko_delta_labels, "gates": gate_keys,
            "trade_mode": trade_mode,
            "start": start_date, "end": end_date,
            "tx_cost_bps": tx_cost_bps, "prefer_em": prefer_em,
            "notional_usd": notional_usd,
            "n_strategies": n_strategies, "elapsed": elapsed,
        }
        total_trades = sum(len(t) for t in all_basket_results.values())
        st.success(
            f"Done in {elapsed:.1f}s — {n_strategies} basket "
            f"{'strategy' if n_strategies == 1 else 'strategies'}, "
            f"{total_trades} pooled trades total ({n_pairs} pairs per "
            f"basket). Switch to the RKO Portfolio drilldown tab to inspect."
        )

    # ---- Summary table ----
    if "rko_rp_results" not in st.session_state:
        st.info("Configure axes above and click **Run** to see a summary. "
                  "Drill into any one basket strategy on the *RKO Portfolio "
                  "drilldown* tab.")
        return

    results = st.session_state["rko_rp_results"]
    meta = st.session_state.get("rko_rp_meta", {})

    st.markdown("---")
    st.markdown("### Latest run — basket strategies")
    pairs_str = ", ".join(meta.get("pairs", []))
    st.caption(
        f"**Basket:** {pairs_str}  ·  "
        f"period {meta.get('start')} → {meta.get('end')}  ·  "
        f"${meta.get('notional_usd', 0):,.0f} per pair  ·  "
        f"tx {meta.get('tx_cost_bps', 0):.1f} bps  ·  "
        f"mode `{meta.get('trade_mode', 'stack')}`  ·  "
        f"**VV-on-American pricing**"
    )

    rows = []
    for name, trades_list in results.items():
        if not trades_list:
            rows.append({"Basket strategy": name, "n trades": 0})
            continue
        sdf = trades_to_df(trades_list)
        s = summarize_strategy(sdf)
        rows.append({
            "Basket strategy": name,
            "n": s["n_trades"],
            "Pairs": sdf["pair"].nunique() if "pair" in sdf.columns else 0,
            "Win %": f"{s['win_rate_pct']:.0f}",
            "KO %": f"{s['ko_rate_pct']:.0f}",
            "Σ Premium": f"${s.get('total_premium_usd', 0):,.0f}",
            "Σ Payoff": f"${s.get('total_payout_usd', 0):,.0f}",
            "PnL": f"${s.get('total_pnl_usd', 0):,.0f}",
            "Sharpe (m)": f"{s['sharpe_monthly']:+.2f}",
            "Max DD": f"${s.get('max_drawdown_usd', 0):,.0f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ---- Bulk download ----
    st.markdown("---")
    st.markdown("#### Bulk download")
    st.caption(
        "Canonical schema with `strategy_type='rko_basket'`. Each row is "
        "a basket strategy with pooled trade ledger across all pairs in "
        "the basket."
    )

    # Build canonical summary frame — augment with rko_basket type
    from core.backtest import (
        export_strategy_time_series, augment_time_series_with_regime,
    )
    from core.regimes import get_regime_panel

    summary_rows = []
    for name, trades_list in results.items():
        if not trades_list:
            continue
        sdf = trades_to_df(trades_list)
        s = summarize_strategy(sdf)
        row = _canonical_summary_row_single(name, s)
        row["strategy_type"] = "rko_basket"   # override
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()

    ts_frames = []
    for name, trades_list in results.items():
        if not trades_list:
            continue
        sdf = trades_to_df(trades_list)
        # For a basket strategy, the pooled ledger has rows from many
        # pairs. Build the time series on the pooled ledger directly —
        # period-end equity already reflects all pair contributions.
        ts = export_strategy_time_series(sdf)
        if ts.empty:
            continue
        ts = _filter_monthly_annual(ts)
        # No regime augmentation here — the basket spans multiple pairs
        # so a single `state` column wouldn't be well-defined.
        ts.insert(0, "strategy_name", name)
        ts.insert(1, "strategy_type", "rko_basket")
        ts_frames.append(ts)
    ts_combined = (pd.concat(ts_frames, ignore_index=True)
                     if ts_frames else pd.DataFrame())

    cdl_a, cdl_b = st.columns(2)
    with cdl_a:
        if not summary_df.empty:
            st.download_button(
                label=f"⬇ Download summary table ({len(summary_df)} rows, CSV)",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name=_build_portfolio_download_filename(
                    meta, "RKO", "summary"),
                mime="text/csv",
                use_container_width=True,
                key="rko_rp_summary_dl",
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
                    meta, "RKO", "timeseries"),
                mime="text/csv",
                use_container_width=True,
                key="rko_rp_timeseries_dl",
            )
        else:
            st.caption("_No time-series data yet._")


def render_rko_portfolio_drilldown_tab() -> None:
    if "rko_rp_results" not in st.session_state:
        st.info("Run a portfolio backtest first (**RKO Portfolio** tab).")
        return

    from core.backtest import compute_equity_and_drawdown, monthly_pnl_table
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    results = st.session_state["rko_rp_results"]
    meta = st.session_state.get("rko_rp_meta", {})

    names = [n for n in results if results[n]]
    if not names:
        st.warning("All basket strategies produced zero trades. Widen "
                    "the date range, loosen gates, or try other strike/KO "
                    "combinations.")
        return

    sel = st.selectbox("Select basket strategy", names, index=0,
                          key="rko_rp_drill_select")
    trades_list = results[sel]
    df = trades_to_df(trades_list)
    if df.empty:
        st.warning("Empty pooled trade ledger for this basket.")
        return

    s = summarize_strategy(df)
    eq = compute_equity_and_drawdown(df)

    # ---- Header ----
    n_pairs = df["pair"].nunique() if "pair" in df.columns else 0
    pair_list_str = ", ".join(sorted(df["pair"].unique())
                                if "pair" in df.columns else [])
    st.markdown(f"### {sel}")
    st.caption(
        f"**Pairs in basket** ({n_pairs}): {pair_list_str}  ·  "
        f"notional ${meta.get('notional_usd', 0):,.0f} per pair  ·  "
        f"trade mode `{meta.get('trade_mode', 'stack')}`  ·  "
        f"**VV-on-American pricing**"
    )

    # ---- Headline metrics ----
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Trades (pooled)", f"{s['n_trades']}", f"{n_pairs} pairs")
    c2.metric("Total P&L", _fmt_usd(s.get("total_pnl_usd", 0)),
                 f"{s['total_pnl_pct']:+.2f}% notl")
    c3.metric("Sharpe (m)", f"{s['sharpe_monthly']:+.2f}",
                 "monthly × √12")
    c4.metric("Max DD", _fmt_usd(s.get("max_drawdown_usd", 0)),
                 "realized, by expiry")
    c5.metric("Win rate", f"{s['win_rate_pct']:.0f}%",
                 f"{int(s['n_trades'] * s['win_rate_pct'] / 100)} winners")

    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("KO rate", f"{s['ko_rate_pct']:.0f}%", "barrier hit at expiry")
    c7.metric("Σ Premium", _fmt_usd(s.get("total_premium_usd", 0)))
    c8.metric("Σ Payoff", _fmt_usd(s.get("total_payout_usd", 0)))
    c9.metric("Calmar", f"{s.get('calmar', float('nan')):.2f}")
    c10.metric("Premium recovery",
                  f"{s['premium_recovery_pct']:.0f}%")

    st.divider()

    # ---- Equity & drawdown ----
    st.markdown("#### Equity curve and drawdown (basket — pooled USD)")
    if not eq.empty and "equity_usd" in eq.columns:
        fig_eq = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.65, 0.35], vertical_spacing=0.07,
            subplot_titles=("Cumulative P&L (USD)", "Drawdown (USD)"),
        )
        fig_eq.add_trace(go.Scatter(
            x=eq.index, y=eq["equity_usd"], mode="lines",
            line=dict(color="#22c55e", width=2), showlegend=False,
            hovertemplate=("%{x|%Y-%m-%d}<br>"
                              "Equity: $%{y:,.0f}<extra></extra>"),
        ), row=1, col=1)
        fig_eq.add_trace(go.Scatter(
            x=eq.index, y=eq["drawdown_usd"], mode="lines",
            line=dict(color="#ef4444", width=1.5),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.25)",
            showlegend=False,
            hovertemplate=("%{x|%Y-%m-%d}<br>"
                              "DD: $%{y:,.0f}<extra></extra>"),
        ), row=2, col=1)
        fig_eq.update_layout(
            height=520, margin=dict(l=10, r=10, t=50, b=10),
            title_text="Realized equity & drawdown (pooled across pairs)",
        )
        fig_eq.update_yaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig_eq, use_container_width=True)

    st.divider()

    # ---- P&L by year (chart + per-year Sharpe table) ----
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    df["_year"] = df["expiry_date"].dt.year

    st.markdown("#### P&L by year")
    yearly = df.groupby("_year")["pnl_usd"].sum().sort_index()
    _render_pnl_by_year_chart(yearly, "P&L by expiry year")
    sharpe_by_year = _annual_sharpe_per_year(eq)
    yearly_df = yearly.reset_index().rename(
        columns={"_year": "Year", "pnl_usd": "PnL (USD)"})
    yearly_df["Sharpe (m)"] = yearly_df["Year"].map(
        lambda y: sharpe_by_year.get(int(y), float("nan")))
    yearly_df_disp = yearly_df.assign(**{
        "PnL (USD)": yearly_df["PnL (USD)"].apply(_fmt_usd),
        "Sharpe (m)": yearly_df["Sharpe (m)"].apply(
            lambda v: f"{v:+.2f}" if pd.notna(v) else "—"),
    })
    st.dataframe(yearly_df_disp, hide_index=True,
                   use_container_width=True)
    st.caption(
        "**Sharpe (m)** is the monthly-basis annualized Sharpe within "
        "each calendar year — matches the `annual_sharpe_*` block on "
        "the summary CSV. Years with fewer than 2 monthly observations "
        "show '—'."
    )

    st.divider()

    # ---- P&L by pair (chart + 7-column per-pair breakdown table) ----
    st.markdown("#### P&L by currency pair")
    by_pair = df.groupby("pair")["pnl_usd"].sum()
    _render_pnl_by_pair_chart(by_pair, pair_label="Pair")

    pair_tbl = _build_per_pair_breakdown(df)
    pair_tbl_disp = _format_breakdown_for_display(pair_tbl)
    st.dataframe(pair_tbl_disp, hide_index=True,
                   use_container_width=True)

    st.divider()

    # ---- Monthly P&L heatmap (year × month) ----
    st.markdown("#### Monthly P&L heatmap — year × month (USD)")
    monthly_df = monthly_pnl_table(df, value_col="pnl_usd")
    if not monthly_df.empty:
        month_labels = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                          7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        monthly_df = monthly_df.rename(
            columns=lambda c: month_labels.get(c, str(c)))
        st.dataframe(
            _diverging_red_yellow_green(monthly_df).format("${:,.0f}", na_rep=""),
            use_container_width=True,
        )

    st.divider()

    # ---- Year × pair P&L heatmap ----
    st.markdown("#### P&L heatmap — year × pair")
    year_pair = df.groupby(["_year", "pair"])["pnl_usd"].sum().unstack(
        fill_value=0.0)
    _render_pnl_heatmap(year_pair, pair_label="Pair")

    st.divider()

    # ---- Per-pair breakdown CSV download ----
    st.markdown("#### Per-pair breakdown — CSV download")
    st.caption(
        "One row per (strategy_name, pair) across all basket strategies "
        "in this run. Lets the downstream analyzer reproduce the per-pair "
        "table for any basket without re-running the backtest."
    )
    per_pair_rows = []
    for name, t in results.items():
        if not t:
            continue
        df_one = trades_to_df(t)
        if df_one.empty:
            continue
        b = _build_per_pair_breakdown(df_one)
        if b.empty:
            continue
        b.insert(0, "strategy_name", name)
        b.insert(1, "strategy_type", "rko_basket")
        per_pair_rows.append(b)
    if per_pair_rows:
        per_pair_df = pd.concat(per_pair_rows, ignore_index=True)
        st.download_button(
            label=(f"⬇ Download per-pair breakdown "
                     f"({len(per_pair_df)} rows, CSV)"),
            data=per_pair_df.to_csv(index=False).encode("utf-8"),
            file_name="rko_rko_portfolio_per_pair.csv",
            mime="text/csv",
            use_container_width=True, key="rko_rp_per_pair_dl",
        )
    else:
        st.caption("_No per-pair rows yet._")

    st.divider()

    # ---- Full pooled ledger expander ----
    with st.expander(
            f"📜 Full pooled trade ledger ({s['n_trades']} trades)",
            expanded=False):
        led_cols_default = [
            "pair", "trade_date", "expiry_date", "tenor_label",
            "spot", "strike", "barrier",
            "premium_usd", "actual_payoff_usd", "pnl_usd",
            "knocked_out",
        ]
        led_cols = [c for c in led_cols_default if c in df.columns]
        st.dataframe(
            df[led_cols].sort_values(["pair", "trade_date"]),
            hide_index=True, use_container_width=True,
        )

    # ---- Full ledger CSV download ----
    csv_bytes = df.to_csv(index=False).encode()
    safe_name = sel.replace("/", "_").replace(" ", "_").replace(",", "")
    st.download_button(
        f"⬇ Download full pooled ledger (CSV)",
        data=csv_bytes,
        file_name=f"rko_rko_basket_{safe_name}.csv",
        mime="text/csv", key="rko_rp_ledger_dl",
    )


# =============================================================================
# WO-RKO Portfolio tab (worst-of basket on American barriers — App 12)
# =============================================================================
# Like the App 9 WO-EKO Portfolio: a basket of UNIQUE PAIR CROSSES
# (e.g. JPY+KRW, JPY+THB, KRW+THB, ...) each running a 2-leg worst-of
# with shared (tenor, direction, strike Δ, KO Δ, gates_A, gates_B).
# Per-cross worst-of trades get pooled into one basket strategy.
# Pricing: VV-on-American per leg (`leg_pricing_mode='vanna_volga_american'`).
# Barrier monitoring: daily OHLC (`ko_check_mode='american_ohlc'`).
# Multiplier comes from the sidebar (33/40/50%).

def _wo_rko_basket_strategy_name(tenor: str, direction_label: str,
                                          strike_a_label: str, ko_a_label: str,
                                          gate_a_key,
                                          strike_b_label: str, ko_b_label: str,
                                          gate_b_key,
                                          multiplier: float) -> str:
    """Concise WO-RKO basket name. Includes multiplier so different
    multiplier choices are distinguishable in downstream CSV ingestion."""
    from core.gates import gate_label
    dir_short = (direction_label.replace("Call (up-and-out)", "Call-UO")
                    .replace("Put (down-and-out)", "Put-DO"))
    ga = f"[{gate_label(gate_a_key)}]" if gate_a_key else "[no gate]"
    gb = f"[{gate_label(gate_b_key)}]" if gate_b_key else "[no gate]"
    return (f"WO-RKO-BASKET  {dir_short}  {tenor}  "
              f"A:{strike_a_label}/H@{ko_a_label} {ga}  ∧  "
              f"B:{strike_b_label}/H@{ko_b_label} {gb}  "
              f"[m={int(multiplier*100)}%]")


def _wo_rko_portfolio_engine_controls() -> dict:
    """Render the structure-pricing-engine controls used by the
    Worst-of RKO Portfolio tab, and return a dict of the captured
    values.

    Returns
    -------
    dict with keys:
      - pricing_engine: 'legacy_multiplier' | 'cf_approx_american' |
                        'monte_carlo_american'
      - correlation_source: 'manual' | 'rolling_60d' | 'triangulation'
      - correlation_value: float, used in 'manual' mode and as a
        fallback when the requested source has no data on a trade date.
      - mc_n_paths: int, only used when pricing_engine='monte_carlo_american'.

    Same UI pattern as `_wo_portfolio_engine_controls` in the EKO
    page, but the engine codes are the American-barrier ones from
    Step R3. Defaults to legacy_multiplier so existing presets and
    behaviour are preserved.

    Streamlit-key prefix is `rwp_` to keep namespaces distinct from
    the bulk Worst-of RKO tab (`rko_wo_*`) and the EKO portfolio's
    `wp_*` keys.
    """
    with st.expander("Structure pricing engine", expanded=False):
        st.caption(
            "How the worst-of structure premium is computed. **Legacy "
            "multiplier** (default) preserves historical behaviour. "
            "**CF-approx American** and **MC American** use the "
            "correlation-aware pricers in "
            "`core.worstof_pricer_american`."
        )
        engine_label = st.radio(
            "Engine",
            ["Legacy multiplier (default)",
             "CF-approx American (fast)",
             "Monte Carlo American (canonical)"],
            index=0,
            key="rwp_pricing_engine",
            help=(
                "Legacy: `multiplier × min(P_A, P_B)`. "
                "CF-approx: ~2 ms/trade, low-biased on tight barriers. "
                "MC: ~400 ms/trade at 100k paths, canonical pricer. "
                "Non-legacy engines force leg_pricing_mode='european'."
            ),
        )
        _engine_map = {
            "Legacy multiplier (default)":      "legacy_multiplier",
            "CF-approx American (fast)":         "cf_approx_american",
            "Monte Carlo American (canonical)":  "monte_carlo_american",
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
                key="rwp_correlation_source",
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
                key="rwp_correlation_value",
            )
            if pricing_engine == "monte_carlo_american":
                mc_n_paths = st.select_slider(
                    "MC paths per trade",
                    options=[20_000, 50_000, 100_000, 200_000, 500_000],
                    value=100_000, key="rwp_mc_n_paths",
                    help=("Std error per trade scales as 1/√n. "
                           "100k → ~1-2bp; 500k → ~0.5bp."),
                )
    return dict(
        pricing_engine=pricing_engine,
        correlation_source=correlation_source,
        correlation_value=correlation_value,
        mc_n_paths=mc_n_paths,
    )


def render_wo_rko_portfolio_tab(folder: str, multiplier: float) -> None:
    """Worst-of basket on American barriers — App 12 analog of App 9
    WO EKO Portfolio."""
    pairs_avail = _list_pairs(folder)
    if len(pairs_avail) < 2:
        st.error("WO-RKO needs at least two pairs.")
        return

    st.markdown("### WO-RKO Portfolio configuration")
    st.caption(
        "A **basket of worst-of structures**. The basket consists of "
        "**unique pair crosses** (e.g. USDJPY×USDKRW, USDJPY×USDTHB, ...). "
        "Each cross runs a 2-leg worst-of with shared (tenor, direction, "
        "strike Δ, KO Δ) — both legs use the same parameters. Per-cross "
        "worst-of trades get pooled into one basket strategy. "
        f"Multiplier (sidebar): **{int(multiplier*100)}%**. "
        f"Pricing: **VV-on-American per leg**. KO monitoring: **daily OHLC** "
        f"(structure dies on the first day either leg's barrier sits "
        f"in-range)."
    )

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        default_pairs = [p for p in RKO_PORT_DEFAULT_PAIRS
                          if p in pairs_avail]
        if not default_pairs:
            default_pairs = pairs_avail[:2]
        pairs_sel = st.multiselect(
            "Currency pairs (pool for cross-combinations)", pairs_avail,
            default=default_pairs, key="rko_wrp_pairs",
            help=("All unique unordered pairs from this list become a "
                   "cross in the basket. n_crosses = n_pairs * (n_pairs-1) / 2."),
        )
        tenors_sel = st.multiselect(
            "Tenors", TENOR_LIST, default=[DEFAULT_TENOR],
            key="rko_wrp_tenors",
        )
        directions_sel = st.multiselect(
            "Direction(s)", list(DIRECTIONS.keys()),
            default=["Call (up-and-out)"], key="rko_wrp_directions",
        )

    with cc2:
        deltas_sel = st.multiselect(
            "Strike Δ list (same on both legs)", list(DELTA_CHOICES.keys()),
            default=[DEFAULT_STRIKE_DELTA_LABEL], key="rko_wrp_deltas",
        )
        ko_delta_labels = st.multiselect(
            "KO Δ list (same on both legs)", list(KO_DELTA_CHOICES.keys()),
            default=[DEFAULT_KO_DELTA_LABEL], key="rko_wrp_ko_deltas",
        )
        tx_cost_bps = st.slider(
            "Transaction cost (bps of notional, structure level)",
            0.0, 20.0, 2.0, 0.5, key="rko_wrp_txcost",
        )
        from core.gates import GATE_REGISTRY
        gate_options = ["(no gate)"] + list(GATE_REGISTRY.keys())
        gate_labels_sel = st.multiselect(
            "Gates (applied to BOTH legs identically)", gate_options,
            default=["(no gate)"], key="rko_wrp_gates",
            format_func=lambda k: ("(no gate)" if k == "(no gate)"
                                   else GATE_REGISTRY[k][0]),
        )
        gate_keys = [None if k == "(no gate)" else k for k in gate_labels_sel]

    with cc3:
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
        # Default start = 1 Jan 2023 (per spec), clamped to the available data range
        default_start = min(max(DEFAULT_START_DATE, date_min), date_max)
        start_date = st.date_input(
            "Start date", value=default_start,
            min_value=date_min, max_value=date_max, key="rko_wrp_start",
        )
        end_date = st.date_input(
            "End date", value=date_max,
            min_value=date_min, max_value=date_max, key="rko_wrp_end",
        )
        prefer_em = st.radio(
            "EM pairs variant", ["offshore", "onshore"], index=0,
            horizontal=True, key="rko_wrp_prefer_em",
        )
        notional_usd = st.number_input(
            "Notional (USD, per worst-of leg)", min_value=100_000.0,
            max_value=200_000_000.0, value=10_000_000.0, step=1_000_000.0,
            format="%.0f", key="rko_wrp_notional",
        )
        trade_mode = st.radio(
            "Trade mode", ["stack", "single"], index=0, horizontal=True,
            key="rko_wrp_trade_mode",
        )

    # Build unique pair crosses
    from itertools import combinations, product as _product
    pair_combos = list(combinations(pairs_sel, 2))  # unordered, distinct
    n_combos = len(pair_combos)
    axis_combos = list(_product(
        tenors_sel, directions_sel, deltas_sel, ko_delta_labels, gate_keys))
    n_strategies = len(axis_combos)

    st.caption(
        f"**{n_strategies}** basket strategies will run, each pooling "
        f"worst-of trades across **{n_combos} pair crosses**. Crosses: "
        + (", ".join(f"{a}×{b}" for a, b in pair_combos[:6])
            + ("…" if n_combos > 6 else "")
            if pair_combos else "(none — pick at least 2 pairs)")
    )

    # ---- Engine controls (Step R4) ----
    # Captured once at the top of the tab; passed into every per-axis
    # call to build_worstof_grid below. Defaults to legacy_multiplier
    # so existing WO-RKO Portfolio presets reproduce bit-for-bit.
    engine_cfg = _wo_rko_portfolio_engine_controls()

    can_run = (n_strategies > 0 and n_combos > 0
                and tenors_sel and directions_sel and deltas_sel
                and ko_delta_labels)
    run_clicked = st.button("▶ Run WO-RKO Portfolio", type="primary",
                              disabled=not can_run, key="rko_wrp_run")

    if run_clicked:
        from core.worstof import build_worstof_grid, run_worstof_grid

        # Non-legacy American engines require leg_pricing_mode='european'.
        # Switch when the user opts into CF/MC; otherwise preserve the
        # historical VV-on-American leg-pricing for legacy backtests.
        if engine_cfg["pricing_engine"] == "legacy_multiplier":
            _leg_pricing_mode = "vanna_volga_american"
        else:
            _leg_pricing_mode = "european"

        all_basket_results: dict[str, list] = {}

        progress_bar = st.progress(0.0, text="Starting worst-of basket runs…")
        t0 = time.time()
        last_update = [t0]

        def cb(p, name):
            now = time.time()
            if now - last_update[0] > 0.1 or p >= 1.0:
                progress_bar.progress(min(p, 1.0),
                                       text=f"Running: {name} ({p*100:.0f}%)")
                last_update[0] = now

        for axis_i, (tenor, dir_label, strike_label, ko_label, gk) in \
                enumerate(axis_combos):
            dir_, btype = DIRECTIONS[dir_label]
            basket_name = _wo_rko_basket_strategy_name(
                tenor, dir_label, strike_label, ko_label, gk,
                strike_label, ko_label, gk, multiplier)
            # Build per-cross WO specs (each cross is one spec)
            specs_this = build_worstof_grid(
                pair_combos=pair_combos, tenors=[tenor],
                leg_a_directions=[(dir_, btype)],
                leg_b_directions=[(dir_, btype)],
                leg_a_strike_deltas=[(strike_label, DELTA_CHOICES[strike_label])],
                leg_b_strike_deltas=[(strike_label, DELTA_CHOICES[strike_label])],
                leg_a_ko_deltas=[(ko_label, KO_DELTA_CHOICES[ko_label])],
                leg_b_ko_deltas=[(ko_label, KO_DELTA_CHOICES[ko_label])],
                gates_a=[gk], gates_b=[gk],
                tx_cost_bps=tx_cost_bps, prefer=prefer_em,
                trade_mode=trade_mode,
                multiplier=multiplier,
                ko_check_mode="american_ohlc",
                leg_pricing_mode=_leg_pricing_mode,
                pricing_engine=engine_cfg["pricing_engine"],
                correlation_source=engine_cfg["correlation_source"],
                correlation_value=engine_cfg["correlation_value"],
                mc_n_paths=engine_cfg["mc_n_paths"],
            )
            if not specs_this:
                all_basket_results[basket_name] = []
                continue

            sub_results = run_worstof_grid(
                folder, specs_this, start_date, end_date,
                notional_usd=notional_usd,
                progress_cb=lambda p, name, _i=axis_i: cb(
                    (_i + p) / n_strategies,
                    f"[basket {_i+1}/{n_strategies}] {name}"),
            )
            pooled = []
            for s in specs_this:
                pooled.extend(sub_results.get(s.name, []))
            all_basket_results[basket_name] = pooled

        elapsed = time.time() - t0
        progress_bar.empty()

        st.session_state["rko_wrp_results"] = all_basket_results
        st.session_state["rko_wrp_meta"] = {
            "pairs": pairs_sel, "pair_combos": pair_combos,
            "tenors": tenors_sel, "directions": directions_sel,
            "strike_deltas": deltas_sel, "ko_deltas": ko_delta_labels,
            "gates": gate_keys, "trade_mode": trade_mode,
            "start": start_date, "end": end_date,
            "tx_cost_bps": tx_cost_bps, "prefer_em": prefer_em,
            "notional_usd": notional_usd, "multiplier": multiplier,
            "n_strategies": n_strategies, "elapsed": elapsed,
        }
        total_trades = sum(len(t) for t in all_basket_results.values())
        st.success(
            f"Done in {elapsed:.1f}s — {n_strategies} basket "
            f"{'strategy' if n_strategies == 1 else 'strategies'}, "
            f"{total_trades} pooled WO trades total ({n_combos} crosses "
            f"per basket). Switch to the WO-RKO Portfolio drilldown tab "
            f"to inspect."
        )

    # ---- Summary ----
    if "rko_wrp_results" not in st.session_state:
        st.info("Configure axes above and click **Run** to see a summary.")
        return

    results = st.session_state["rko_wrp_results"]
    meta = st.session_state.get("rko_wrp_meta", {})

    st.markdown("---")
    st.markdown("### Latest run — basket strategies")
    n_combos = len(meta.get("pair_combos", []))
    st.caption(
        f"**{n_combos} crosses** in basket  ·  "
        f"period {meta.get('start')} → {meta.get('end')}  ·  "
        f"${meta.get('notional_usd', 0):,.0f} per leg  ·  "
        f"tx {meta.get('tx_cost_bps', 0):.1f} bps  ·  "
        f"multiplier **{int(meta.get('multiplier', 0.4)*100)}%**  ·  "
        f"mode `{meta.get('trade_mode', 'stack')}`"
    )

    from core.worstof import worstof_trades_to_df, worstof_summarize

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
            "Crosses": (sdf[["leg_a_pair","leg_b_pair"]].drop_duplicates().shape[0]
                          if not sdf.empty else 0),
            "Either KO %": f"{s.get('any_ko_rate', 0)*100:.0f}",
            "Both survive %": f"{s.get('both_survive_rate', 0)*100:.0f}",
            "Win %": f"{s.get('win_rate', 0)*100:.0f}",
            "Σ Premium": f"${s.get('total_premium_paid_usd', 0):,.0f}",
            "PnL": f"${s.get('total_pnl_usd', 0):,.0f}",
            "Sharpe (m)": f"{s.get('sharpe_monthly', 0):+.2f}",
            "Max DD": f"${s.get('max_drawdown_usd', 0):,.0f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ---- Bulk download ----
    st.markdown("---")
    st.markdown("#### Bulk download")
    st.caption(
        "Canonical schema with `strategy_type='wo_rko_basket'`. WO-specific "
        "columns populated; single-leg-specific columns NaN."
    )

    from core.worstof import worstof_export_time_series
    from core.backtest import augment_time_series_with_regime
    from core.regimes import get_regime_panel

    summary_rows = []
    for name, trades_list in results.items():
        if not trades_list:
            continue
        sdf = worstof_trades_to_df(trades_list)
        s = worstof_summarize(sdf)
        row = _canonical_summary_row_worstof(name, s)
        row["strategy_type"] = "wo_rko_basket"
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()

    ts_frames = []
    for name, trades_list in results.items():
        if not trades_list:
            continue
        sdf = worstof_trades_to_df(trades_list)
        ts = worstof_export_time_series(sdf)
        if ts.empty:
            continue
        ts = _filter_monthly_annual(ts)
        ts.insert(0, "strategy_name", name)
        ts.insert(1, "strategy_type", "wo_rko_basket")
        ts_frames.append(ts)
    ts_combined = (pd.concat(ts_frames, ignore_index=True)
                     if ts_frames else pd.DataFrame())

    cdl_a, cdl_b = st.columns(2)
    with cdl_a:
        if not summary_df.empty:
            st.download_button(
                label=f"⬇ Download summary table ({len(summary_df)} rows, CSV)",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name=_build_portfolio_download_filename(
                    meta, "WO-RKO", "summary"),
                mime="text/csv",
                use_container_width=True, key="rko_wrp_summary_dl",
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
                    meta, "WO-RKO", "timeseries"),
                mime="text/csv",
                use_container_width=True, key="rko_wrp_timeseries_dl",
            )
        else:
            st.caption("_No time-series data yet._")


def render_wo_rko_portfolio_drilldown_tab() -> None:
    if "rko_wrp_results" not in st.session_state:
        st.info("Run a WO-RKO portfolio backtest first.")
        return

    from core.worstof import (worstof_trades_to_df, worstof_summarize,
                                  worstof_equity_curve, worstof_monthly_pnl)
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    results = st.session_state["rko_wrp_results"]
    meta = st.session_state.get("rko_wrp_meta", {})

    names = [n for n in results if results[n]]
    if not names:
        st.warning("All basket strategies produced zero trades.")
        return

    sel = st.selectbox("Select basket strategy", names, index=0,
                          key="rko_wrp_drill_select")
    trades_list = results[sel]
    df = worstof_trades_to_df(trades_list)
    if df.empty:
        st.warning("Empty pooled trade ledger for this basket.")
        return

    s = worstof_summarize(df)
    eq = worstof_equity_curve(df)

    # ---- Engine / correlation banner (Step R4) ----
    # Same banner as the bulk Worst-of RKO drilldown. Surfaces the
    # pricing engine, average ρ and source, and engine-vs-legacy ratio
    # for A/B comparison.
    if "pricing_engine" in df.columns:
        engine_used = df["pricing_engine"].iloc[0]
        engine_label = {
            "legacy_multiplier":     "Legacy multiplier (multiplier × min)",
            "cf_approx_american":    "CF-approx American (~2 ms/trade)",
            "monte_carlo_american":  "Monte Carlo American (canonical)",
        }.get(engine_used, engine_used)
        if engine_used != "legacy_multiplier":
            corr_used_avg = (df["correlation_used"].dropna().mean()
                              if "correlation_used" in df.columns else None)
            corr_src = (df["correlation_source_used"].iloc[0]
                         if "correlation_source_used" in df.columns else "—")
            avg_legacy = (df["structure_premium_legacy_usd"].dropna().mean()
                          if "structure_premium_legacy_usd" in df.columns
                          else None)
            avg_engine = df["structure_premium_mid_usd"].mean()
            ratio_str = (f"  ·  engine/legacy ratio = "
                          f"{avg_engine/avg_legacy:.2f}"
                          if avg_legacy and avg_legacy > 0 else "")
            corr_str = (f"  ·  avg ρ = {corr_used_avg:+.3f}  ·  "
                         f"source = {corr_src}"
                         if corr_used_avg is not None else "")
            st.info(
                f"**Engine**: {engine_label}{corr_str}{ratio_str}  ·  "
                f"avg legacy: ${avg_legacy:,.0f}  ·  "
                f"avg engine: ${avg_engine:,.0f}"
            )
        else:
            st.info(f"**Engine**: {engine_label}")

    # ---- Header ----
    df["cross"] = df["leg_a_pair"] + "×" + df["leg_b_pair"]
    n_crosses = df["cross"].nunique()
    cross_list_str = ", ".join(sorted(df["cross"].unique()))
    multiplier = float(df["multiplier"].iloc[0]) if "multiplier" in df.columns else None
    st.markdown(f"### {sel}")
    header_bits = [
        f"**Crosses in basket** ({n_crosses}): {cross_list_str}",
        f"notional ${meta.get('notional_usd', 0):,.0f} per leg",
    ]
    if multiplier is not None:
        header_bits.append(f"multiplier **{int(multiplier*100)}%**")
    header_bits.append(f"trade mode `{meta.get('trade_mode', 'stack')}`")
    header_bits.append("**VV-on-American pricing**")
    st.caption("  ·  ".join(header_bits))

    # ---- Headline metrics — unified set, 10 cells ----
    notional_usd_v = s.get("notional_usd", 0)
    total_pnl_v = s.get("total_pnl_usd", 0)
    total_pnl_pct = (total_pnl_v / notional_usd_v * 100
                       if notional_usd_v > 0 else 0.0)
    win_rate_pct = s.get("win_rate", 0.0) * 100
    any_ko_pct = s.get("any_ko_rate", 0.0) * 100
    both_surv_pct = s.get("both_survive_rate", 0.0) * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Trades (pooled)", f"{s['n_trades']}", f"{n_crosses} crosses")
    c2.metric("Total P&L", _fmt_usd(total_pnl_v),
                 f"{total_pnl_pct:+.2f}% notl")
    c3.metric("Sharpe (m)", f"{s.get('sharpe_monthly', 0):+.2f}",
                 "monthly × √12")
    c4.metric("Max DD", _fmt_usd(s.get("max_drawdown_usd", 0)),
                 "realized, by expiry")
    c5.metric("Win rate", f"{win_rate_pct:.0f}%",
                 f"{int(s['n_trades'] * win_rate_pct / 100)} winners")

    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("Either leg KO", f"{any_ko_pct:.1f}%",
                 "structure-level KO rate")
    c7.metric("Both survive", f"{both_surv_pct:.1f}%",
                 "neither leg knocked")
    c8.metric("Σ Premium", _fmt_usd(s.get("total_premium_paid_usd", 0)))
    c9.metric("Premium recovery",
                 f"{s.get('premium_recovery_pct', 0):.0f}%")
    c10.metric("Multiplier",
                  f"{int(multiplier*100)}%" if multiplier is not None else "—")

    st.divider()

    # ---- Equity & drawdown ----
    st.markdown("#### Equity curve and drawdown (basket — pooled USD)")
    if not eq.empty and "equity_usd" in eq.columns:
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
        st.plotly_chart(fig_eq, use_container_width=True)

    st.divider()

    # ---- P&L by year (chart + per-year Sharpe table) ----
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    df["_year"] = df["expiry_date"].dt.year

    st.markdown("#### P&L by year")
    yearly = df.groupby("_year")["pnl_usd"].sum().sort_index()
    _render_pnl_by_year_chart(yearly, "P&L by expiry year")
    sharpe_by_year = _annual_sharpe_per_year(eq)
    yearly_df = yearly.reset_index().rename(
        columns={"_year": "Year", "pnl_usd": "PnL (USD)"})
    yearly_df["Sharpe (m)"] = yearly_df["Year"].map(
        lambda y: sharpe_by_year.get(int(y), float("nan")))
    yearly_df_disp = yearly_df.assign(**{
        "PnL (USD)": yearly_df["PnL (USD)"].apply(_fmt_usd),
        "Sharpe (m)": yearly_df["Sharpe (m)"].apply(
            lambda v: f"{v:+.2f}" if pd.notna(v) else "—"),
    })
    st.dataframe(yearly_df_disp, hide_index=True,
                   use_container_width=True)
    st.caption(
        "**Sharpe (m)** is the monthly-basis annualized Sharpe within each "
        "calendar year — same formula as the `annual_sharpe_*` columns in "
        "the summary CSV: `mean(monthly_pnl) / std(monthly_pnl) × √12`. "
        "Years with fewer than 2 monthly observations show '—'."
    )

    st.divider()

    # ---- P&L by cross (chart + 9-column breakdown table) ----
    st.markdown("#### P&L by cross")
    by_cross = df.groupby("cross")["pnl_usd"].sum()
    _render_pnl_by_pair_chart(by_cross, pair_label="Cross")

    cross_tbl = _build_per_cross_breakdown(df)
    cross_tbl_disp = _format_breakdown_for_display(cross_tbl)
    st.dataframe(cross_tbl_disp, hide_index=True,
                   use_container_width=True)
    st.caption(
        "**Structure KO %** = % of trades where the worst-of structure "
        "knocked out (either leg's barrier was hit). For same-direction "
        "worst-of structures this equals 1 − both_survive_rate at the "
        "trade level. Should sit between max(Leg A, Leg B) and min(Leg A "
        "+ Leg B, 100%) depending on how independent the two legs' KOs "
        "are."
    )

    st.divider()

    # ---- Monthly P&L heatmap (year × month) ----
    st.markdown("#### Monthly P&L heatmap — year × month (USD)")
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
        st.dataframe(
            _diverging_red_yellow_green(pivot).format("${:,.0f}", na_rep=""),
            use_container_width=True,
        )

    st.divider()

    # ---- Year × cross P&L heatmap ----
    st.markdown("#### P&L heatmap — year × cross")
    year_cross = df.groupby(["_year", "cross"])["pnl_usd"].sum().unstack(
        fill_value=0.0)
    _render_pnl_heatmap(year_cross, pair_label="Cross")

    st.divider()

    # ---- Bulk per-cross breakdown download ----
    st.markdown("#### Per-cross breakdown — CSV download")
    st.caption(
        "One row per (strategy_name, cross) pair across all basket "
        "strategies in this run. Lets the downstream analyzer "
        "reproduce the per-cross table above for any of the basket "
        "strategies without re-running the backtest."
    )
    per_cross_rows = []
    for name, t in results.items():
        if not t:
            continue
        df_one = worstof_trades_to_df(t)
        if df_one.empty:
            continue
        b = _build_per_cross_breakdown(df_one)
        if b.empty:
            continue
        b.insert(0, "strategy_name", name)
        b.insert(1, "strategy_type", "wo_rko_basket")
        per_cross_rows.append(b)
    if per_cross_rows:
        per_cross_df = pd.concat(per_cross_rows, ignore_index=True)
        st.download_button(
            label=(f"⬇ Download per-cross breakdown "
                     f"({len(per_cross_df)} rows, CSV)"),
            data=per_cross_df.to_csv(index=False).encode("utf-8"),
            file_name="rko_wo_rko_portfolio_per_cross.csv",
            mime="text/csv",
            use_container_width=True, key="rko_wrp_per_cross_dl",
        )
    else:
        st.caption("_No per-cross rows yet._")

    st.divider()

    # ---- Full ledger expander ----
    with st.expander(
            f"📜 Full pooled WO trade ledger ({s['n_trades']} trades)",
            expanded=False):
        led_cols_default = [
            "cross", "trade_date", "expiry_date", "tenor_label",
            "leg_a_strike", "leg_a_barrier",
            "leg_b_strike", "leg_b_barrier",
            "structure_premium_paid_usd", "worst_of_payoff_usd", "pnl_usd",
            "leg_a_knocked_out", "leg_b_knocked_out",
            "multiplier", "leg_pricing_mode",
        ]
        led_cols = [c for c in led_cols_default if c in df.columns]
        st.dataframe(
            df[led_cols].sort_values(["cross", "trade_date"]),
            hide_index=True, use_container_width=True,
        )

    # ---- Full ledger CSV download ----
    csv_bytes = df.to_csv(index=False).encode()
    safe_name = sel.replace("/", "_").replace(" ", "_").replace(",", "")[:80]
    st.download_button(
        f"⬇ Download full pooled WO-RKO ledger (CSV)",
        data=csv_bytes,
        file_name=f"rko_wo_rko_basket_{safe_name}.csv",
        mime="text/csv", key="rko_wrp_ledger_dl",
    )

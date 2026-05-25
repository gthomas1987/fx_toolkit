"""App 2 — India macro dashboard → USDINR directional bias.

Run with:
    streamlit run apps/2_india.py

Z-scores 43 macro indicators across 5 categories (External Drivers, Domestic
Rates, Equity & Flows, Macro Fundamentals, Liquidity), applies a sign
convention so that each signed z-score contributes in the "bullish USDINR
(bearish INR)" direction, and aggregates equal-weighted within and across
categories into a composite directional signal.

Tabs:
  📡 Overview     — composite gauge, USDINR overlay, category sub-composites
  🔢 Components  — full table of all indicators with z-scores & contributions
  🕰️ History     — per-category and per-indicator z-score time series
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.india_loader import load_india_data
from core.india_signals import (
    SIGNALS, MANUAL_SIGNALS, CATEGORY_ORDER, EXCLUDED_TICKERS_PREFIXES,
    is_excluded, resolve_signals, get_series_for_signal,
    native_z_score, latest_z_score, build_signed_z_table, composite_history,
)
from core.charts import time_series_chart, signal_gauge, percentile_heatmap


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="2 — India Macro → USDINR Bias",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("2 · India Macro → USDINR Directional Bias")
st.caption(
    "Z-scores **{}** macro indicators across **{}** categories  ·  signed so a "
    "**positive composite ⇒ bullish USDINR / bearish INR**".format(
        sum(1 for _ in SIGNALS) + len(MANUAL_SIGNALS), len(CATEGORY_ORDER)
    )
)


# -----------------------------------------------------------------------------
# Sidebar — data path & lookback
# -----------------------------------------------------------------------------
def resolve_default_data_file() -> str:
    """Look in common locations."""
    env = os.environ.get("INDIA_DATA_FILE")
    if env and Path(env).exists():
        return env
    candidates = [
        Path("/mnt/user-data/uploads/india_data.csv"),
        ROOT.parent / "india_data.csv",
        ROOT / "india_data.csv",
        Path.cwd() / "india_data.csv",
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return str(ROOT / "india_data.csv")


if "india_data_file" not in st.session_state:
    st.session_state["india_data_file"] = resolve_default_data_file()

data_file = st.sidebar.text_input(
    "India data file",
    value=st.session_state["india_data_file"],
    key="india_data_file_input",
    help="Path to india_data.csv. Override with INDIA_DATA_FILE env var.",
)
st.session_state["india_data_file"] = data_file
if not Path(data_file).exists():
    st.sidebar.error(f"File not found: {data_file}")
    st.stop()

LOOKBACK_OPTS = {"1Y": 365, "2Y": 730, "3Y": 1095, "5Y": 1825, "Full": 10**9}
lb_choice = st.sidebar.selectbox(
    "Z-score lookback",
    list(LOOKBACK_OPTS.keys()),
    index=1,   # default 2Y
    key="india_lb",
    help="Window over which each indicator's z-score is computed (on its "
         "native daily/weekly/monthly frequency).",
)
LOOKBACK_DAYS = LOOKBACK_OPTS[lb_choice]


# -----------------------------------------------------------------------------
# Load data + resolve signals
# -----------------------------------------------------------------------------
df, meta_list = load_india_data(data_file)
columns = list(df.columns)
signals = resolve_signals(meta_list, columns)

# Get USDINR spot for overlay
usdinr = df["USDINR Curncy"].dropna() if "USDINR Curncy" in columns else pd.Series(dtype=float)


@st.cache_data(show_spinner="Computing z-scores & composite history…")
def cached_composite(data_file: str, lookback_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (signed_z_table, composite_history_df)."""
    d, m = load_india_data(data_file)
    sigs = resolve_signals(m, list(d.columns))
    z_table = build_signed_z_table(d, list(d.columns), sigs, lookback_days=lookback_days)
    hist = composite_history(d, list(d.columns), sigs, lookback_days=lookback_days)
    return z_table, hist


z_table, comp_hist = cached_composite(data_file, LOOKBACK_DAYS)


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_over, tab_comp, tab_hist = st.tabs(
    ["📡 Overview", "🔢 Components", "🕰️ History"]
)


# ============================================================================
# TAB 1 — Overview
# ============================================================================
with tab_over:
    # Latest values
    latest_comp = float(comp_hist["Composite"].iloc[-1]) if not comp_hist.empty else float("nan")
    cat_latest = {cat: float(comp_hist[cat].iloc[-1])
                  for cat in CATEGORY_ORDER if cat in comp_hist.columns}
    last_date = comp_hist.index[-1] if not comp_hist.empty else None
    usdinr_last = float(usdinr.iloc[-1]) if not usdinr.empty else float("nan")
    usdinr_chg_1d = float(usdinr.diff().iloc[-1]) if len(usdinr) > 1 else float("nan")
    usdinr_chg_1w = float(usdinr.diff(5).iloc[-1]) if len(usdinr) > 5 else float("nan")

    # Top metric strip
    m1, m2, m3, m4 = st.columns([2, 1, 1, 1])
    with m1:
        st.markdown(
            f"##### As of **{last_date.strftime('%Y-%m-%d') if last_date is not None else '—'}**  "
            f"·  z-score lookback **{lb_choice}**"
        )
    m2.metric("USDINR spot", f"{usdinr_last:,.4f}",
              f"{usdinr_chg_1d:+.4f}" if pd.notna(usdinr_chg_1d) else "—",
              help="Daily change in spot")
    m3.metric("USDINR 1W change", f"{usdinr_chg_1w:+.4f}" if pd.notna(usdinr_chg_1w) else "—")
    m4.metric("Composite signal", f"{latest_comp:+.2f}",
              "bullish USDINR" if latest_comp > 0 else "bearish USDINR",
              delta_color="off")

    st.markdown("---")

    # Two-column layout: composite gauge + USDINR with composite overlay
    c_left, c_right = st.columns([1, 2], gap="medium")
    with c_left:
        st.markdown("##### Composite directional signal")
        # Use composite range for the gauge bounds — robust
        comp_clean = comp_hist["Composite"].dropna()
        gauge_range = max(2.0, float(comp_clean.abs().quantile(0.99)) * 1.1) \
                      if not comp_clean.empty else 3.0
        fig_gauge = signal_gauge(
            latest_comp,
            title="",
            range_min=-gauge_range, range_max=gauge_range,
            height=300,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        if latest_comp > 0:
            st.caption("🔴 Bias: **bullish USDINR / bearish INR**  "
                       f"(reading {latest_comp:+.2f}, gauge ±{gauge_range:.1f}).")
        else:
            st.caption("🟢 Bias: **bullish INR / bearish USDINR**  "
                       f"(reading {latest_comp:+.2f}, gauge ±{gauge_range:.1f}).")

    with c_right:
        st.markdown("##### USDINR with composite signal overlay")
        if not usdinr.empty and not comp_hist.empty:
            # Restrict to the composite history's date range to align
            common_idx = comp_hist.index.intersection(usdinr.index)
            spot_aligned = usdinr.reindex(common_idx)
            comp_aligned = comp_hist["Composite"].reindex(common_idx)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=spot_aligned.index, y=spot_aligned.values,
                mode="lines", name="USDINR",
                line=dict(color="#1a1a1a", width=1.8),
                yaxis="y1",
            ))
            fig.add_trace(go.Scatter(
                x=comp_aligned.index, y=comp_aligned.values,
                mode="lines", name="Composite signal",
                line=dict(color="#d62728", width=1.4),
                yaxis="y2",
                fill="tozeroy", fillcolor="rgba(214, 39, 40, 0.10)",
            ))
            fig.update_layout(
                yaxis=dict(title="USDINR spot", side="left"),
                yaxis2=dict(title="Composite signal (z)", side="right",
                            overlaying="y", showgrid=False,
                            zeroline=True, zerolinecolor="rgba(0,0,0,0.2)"),
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                template="plotly_white",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "USDINR (left axis, black) vs composite signal (right axis, red). "
                "Sustained positive composite → expect USDINR up; sustained negative → INR strength."
            )
        else:
            st.warning("Insufficient data for chart.")

    st.markdown("---")
    st.markdown("##### Category sub-composites  (signed z-score, equal-weighted within category)")

    # 5 mini gauges per category
    cat_cols = st.columns(len(CATEGORY_ORDER))
    for col, cat in zip(cat_cols, CATEGORY_ORDER):
        with col:
            v = cat_latest.get(cat, float("nan"))
            cat_clean = comp_hist[cat].dropna() if cat in comp_hist.columns else pd.Series(dtype=float)
            cat_range = max(2.0, float(cat_clean.abs().quantile(0.99)) * 1.1) \
                        if not cat_clean.empty else 3.0
            fig_c = signal_gauge(
                v, title=cat,
                range_min=-cat_range, range_max=cat_range,
                height=200,
            )
            st.plotly_chart(fig_c, use_container_width=True)


# ============================================================================
# TAB 2 — Components
# ============================================================================
with tab_comp:
    st.markdown(f"##### All {len(z_table)} indicators · z-score over **{lb_choice}** lookback")

    show = z_table.copy()
    show["abs_signed_z"] = show["Signed z"].abs()
    show["Direction"] = show["Signed z"].apply(
        lambda v: "↑ USDINR" if pd.notna(v) and v > 0 else
                  ("↓ USDINR" if pd.notna(v) and v < 0 else "—")
    )
    show["Sign"] = show["Sign"].apply(lambda x: f"{x:+d}")
    show["Last update"] = show["Last update"].dt.strftime("%Y-%m-%d")

    # Filters
    f1, f2, f3 = st.columns([2, 2, 2])
    with f1:
        cat_filter = st.multiselect(
            "Category", CATEGORY_ORDER, default=[], key="comp_cat_filter")
    with f2:
        sort_by = st.selectbox(
            "Sort by",
            ["|Signed z| (high to low)", "Signed z (USDINR-bullish first)",
             "Signed z (INR-bullish first)", "Category · Indicator"],
            index=0, key="comp_sort")
    with f3:
        only_extreme = st.checkbox(
            "Only show |signed z| > 1.5",
            value=False, key="comp_extreme")

    sub = show
    if cat_filter:
        sub = sub[sub["Category"].isin(cat_filter)]
    if only_extreme:
        sub = sub[sub["abs_signed_z"] > 1.5]
    if sort_by == "|Signed z| (high to low)":
        sub = sub.sort_values("abs_signed_z", ascending=False, na_position="last")
    elif sort_by == "Signed z (USDINR-bullish first)":
        sub = sub.sort_values("Signed z", ascending=False, na_position="last")
    elif sort_by == "Signed z (INR-bullish first)":
        sub = sub.sort_values("Signed z", ascending=True, na_position="last")
    else:
        sub = sub.sort_values(["Category", "Indicator"])

    display_cols = ["Category", "Indicator", "Sign", "Direction",
                    "Last value", "Last update", "Raw z", "Signed z"]
    sub_display = sub[display_cols]

    def style_signed_z(v):
        if pd.isna(v):
            return ""
        if v > 1.5:
            return "background-color: rgba(214, 39, 40, 0.30); font-weight: bold;"
        if v < -1.5:
            return "background-color: rgba(44, 160, 44, 0.30); font-weight: bold;"
        if v > 0.5:
            return "background-color: rgba(214, 39, 40, 0.12);"
        if v < -0.5:
            return "background-color: rgba(44, 160, 44, 0.12);"
        return ""

    styled = (sub_display
                .style
                .format({"Last value": "{:,.4g}",
                         "Raw z": "{:+.2f}", "Signed z": "{:+.2f}"})
                .map(style_signed_z, subset=["Signed z"]))
    st.dataframe(styled, use_container_width=True, hide_index=True, height=560)

    st.caption(
        "**Sign**: ±1, applied so signed z is in 'bullish USDINR' direction.  "
        "**Raw z** = (latest − mean) / std over the lookback window on the indicator's "
        "native frequency.  **Signed z** = Raw z × Sign.  "
        "Red shading = currently leans bullish USDINR; green = leans bullish INR."
    )


# ============================================================================
# TAB 3 — History
# ============================================================================
with tab_hist:
    if comp_hist.empty:
        st.warning("No composite history available.")
    else:
        # 1) Composite + categories overlaid
        st.markdown("##### Composite & category sub-composites")
        cat_cols_in_hist = [c for c in CATEGORY_ORDER if c in comp_hist.columns]
        plot_cols = ["Composite"] + cat_cols_in_hist

        # Restrict displayed window to the lookback (or full sample for 'Full')
        end = comp_hist.index[-1]
        cutoff = end - pd.Timedelta(days=LOOKBACK_DAYS) if LOOKBACK_DAYS < 1e8 else comp_hist.index[0]
        plot_df = comp_hist[comp_hist.index >= cutoff][plot_cols]

        fig = go.Figure()
        # Background "extreme bands" at ±1.5 and ±2 z
        for y in [1.5, -1.5]:
            fig.add_hline(y=y, line_color="#888", line_dash="dot", line_width=1)
        fig.add_hline(y=0, line_color="black", line_width=0.6, opacity=0.6)
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df["Composite"],
            mode="lines", name="Composite",
            line=dict(color="#1a1a1a", width=2.4),
        ))
        from core.charts import COLOR_PALETTE
        for i, cat in enumerate(cat_cols_in_hist):
            fig.add_trace(go.Scatter(
                x=plot_df.index, y=plot_df[cat],
                mode="lines", name=cat,
                line=dict(color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                          width=1.2, dash="solid"),
                opacity=0.8,
                visible="legendonly" if i > 0 else True,
            ))
        fig.update_layout(
            height=420,
            yaxis_title="Signed z-score",
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Composite (black) = equal-weighted average of the 5 category sub-composites. "
            "Click legend entries to toggle categories on/off. Dotted lines at ±1.5 z."
        )

        # 2) Per-indicator inspection
        st.markdown("##### Per-indicator z-score inspection")
        inspect_signals = sorted(signals, key=lambda s: (s.category, s.description))
        labels = [f"[{s.category}]  {s.description}  ({'+' if s.sign > 0 else '-'})"
                  for s in inspect_signals]
        idx = st.selectbox("Indicator", range(len(labels)),
                           format_func=lambda i: labels[i],
                           key="hist_indicator")
        sig = inspect_signals[idx]
        s_raw = get_series_for_signal(df, columns, sig).dropna()
        z_native = native_z_score(s_raw, lookback_days=LOOKBACK_DAYS)
        z_native_clean = z_native.dropna()

        if s_raw.empty:
            st.warning("No data for this indicator.")
        else:
            cc1, cc2 = st.columns(2)
            with cc1:
                fig_lvl = go.Figure()
                fig_lvl.add_trace(go.Scatter(
                    x=s_raw.index, y=s_raw.values,
                    mode="lines+markers" if sig.ticker.startswith("__col_") or
                                            len(s_raw) < 200 else "lines",
                    name=sig.description,
                    line=dict(color="#1f77b4", width=1.6),
                    marker=dict(size=4),
                ))
                fig_lvl.update_layout(
                    title=f"Level — {sig.description}", height=320,
                    template="plotly_white", showlegend=False,
                    hovermode="x unified",
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig_lvl, use_container_width=True)
            with cc2:
                fig_z = go.Figure()
                if not z_native_clean.empty:
                    z_signed = z_native_clean * sig.sign
                    fig_z.add_trace(go.Scatter(
                        x=z_signed.index, y=z_signed.values,
                        mode="lines",
                        line=dict(color="#d62728", width=1.4),
                        fill="tozeroy",
                        fillcolor="rgba(214, 39, 40, 0.12)",
                        name="Signed z",
                    ))
                    fig_z.add_hline(y=0, line_color="black", line_width=0.6, opacity=0.6)
                    fig_z.add_hline(y=1.5, line_color="#888", line_dash="dot", line_width=1)
                    fig_z.add_hline(y=-1.5, line_color="#888", line_dash="dot", line_width=1)
                fig_z.update_layout(
                    title=f"Signed z-score (sign = {sig.sign:+d})",
                    height=320, template="plotly_white", showlegend=False,
                    hovermode="x unified",
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig_z, use_container_width=True)
            if sig.note:
                st.caption(f"💡 {sig.note}")


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    f"Data: `{data_file}` · "
    f"Z-score lookback: **{lb_choice}** · "
    f"Sign convention: **+ve composite ⇒ bullish USDINR / bearish INR**"
)

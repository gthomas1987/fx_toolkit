"""App 1c — FX Forward-Point Calendar Spreads.

Run with:
    streamlit run apps/1c_fwd_spreads.py

Spreads:  1W-1M,  1M-3M,  3M-6M,  3M-1Y   (long minus short, in pip units).

Tabs:
    1. Time series — selected spreads with reference quantile bands
    2. Snapshot — current values + percentiles for all spreads
    3. Distribution — per-spread histogram with current marker
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from core.conventions import FWD_TENORS
from core.ts_loader import load_panel, list_available_pairs
from core.percentiles import (current_percentile, expanding_percentile,
                              reference_quantiles)
from core.charts import (time_series_chart, time_series_with_quantile_bands,
                         percentile_path_chart, histogram_with_marker)
from core.ui import data_dir_input, lookback_selector, app_header, format_value
from core.fwd_structures import all_spreads, SPREAD_SPECS


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="1c — FX Forward Spreads",
    layout="wide",
    initial_sidebar_state="expanded",
)
app_header(
    "1c · FX Forward-Point Calendar Spreads",
    "Spreads:  1W-1M  ·  1M-3M  ·  3M-6M  ·  3M-1Y   (long minus short, in pip units)",
)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
folder = data_dir_input()
if folder is None:
    st.stop()

available_pairs = list_available_pairs(folder, "FWD_POINTS")
if not available_pairs:
    st.error("No FWD_POINTS tickers found.")
    st.stop()

default_pair = "EURUSD" if "EURUSD" in available_pairs else available_pairs[0]
pair = st.sidebar.selectbox(
    "Currency pair",
    available_pairs,
    index=available_pairs.index(default_pair),
    key="fs_pair",
)

asia_em = pair in {"USDCNH", "USDCNY", "USDINR", "USDIDR", "USDKRW",
                   "USDMYR", "USDPHP", "USDTHB", "USDTWD"}
prefer = "offshore"
if asia_em:
    prefer = st.sidebar.radio(
        "Variant", ["offshore", "onshore"], index=0,
        horizontal=True, key="fs_prefer",
    )

lookback_label, lookback_days = lookback_selector(default="3Y")


# -----------------------------------------------------------------------------
# Load fwd points panels and build spreads
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading forward points and building spreads…")
def load_and_build(folder: str, pair: str, prefer: str
                   ) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    fwd_panels: dict[str, pd.Series] = {}
    for tenor in FWD_TENORS:
        df = load_panel(folder, category="FWD_POINTS", tenor=tenor,
                        prefer=prefer, pairs=(pair,))
        if not df.empty and pair in df.columns:
            s = df[pair].dropna()
            if not s.empty:
                fwd_panels[tenor] = s
    return fwd_panels, all_spreads(fwd_panels)


fwd_panels, spreads = load_and_build(folder, pair, prefer)
if not spreads:
    st.warning(
        f"No spreads could be built for {pair}. "
        f"Available fwd tenors: {sorted(fwd_panels.keys())}"
    )
    st.stop()

spread_labels = [lbl for (lbl, *_) in SPREAD_SPECS if lbl in spreads]


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_ts, tab_snap, tab_dist = st.tabs(
    ["📈 Time series", "🔢 Snapshot", "📉 Distribution"]
)


# ============================================================================
# TAB 1 — Time series
# ============================================================================
with tab_ts:
    sel = st.multiselect(
        "Spreads to chart",
        spread_labels,
        default=spread_labels,
        key="fs_ts_sel",
    )
    if not sel:
        st.info("Select at least one spread.")
        st.stop()

    cutoff = max((spreads[l].index[-1] for l in sel)) - pd.Timedelta(days=lookback_days)
    wide = pd.concat({l: spreads[l] for l in sel}, axis=1)
    wide_cut = wide[wide.index >= cutoff]

    fig = time_series_chart(
        wide_cut,
        title=f"{pair}  ·  Calendar spreads  ·  trailing {lookback_label}",
        yaxis_title="Spread (pip units)",
        height=460,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Per-spread: value + reference quantiles + percentile path")
    for label in sel:
        s = spreads[label]
        s_cut = s[s.index >= cutoff]
        quantiles = reference_quantiles(s, lookback_days=lookback_days)
        pct_now = current_percentile(s, lookback_days)

        c1, c2 = st.columns([3, 1])
        with c1:
            fig_v = time_series_with_quantile_bands(
                s_cut, quantiles,
                title=f"{label}  ·  current = {format_value(s.iloc[-1], 2)} "
                      f"(pct {format_value(pct_now, 1)})",
                yaxis_title="Spread (pips)",
                height=240,
            )
            st.plotly_chart(fig_v, use_container_width=True)
        with c2:
            pct_path = expanding_percentile(s, lookback_days=lookback_days)
            fig_p = percentile_path_chart(
                pct_path[pct_path.index >= cutoff],
                title="", height=240,
            )
            st.plotly_chart(fig_p, use_container_width=True)


# ============================================================================
# TAB 2 — Snapshot
# ============================================================================
with tab_snap:
    st.markdown(f"##### Snapshot  ·  {pair}  ·  {lookback_label} lookback")
    rows = []
    for label in spread_labels:
        s = spreads[label]
        cur = float(s.iloc[-1])
        pct = current_percentile(s, lookback_days)
        cutoff = s.index[-1] - pd.Timedelta(days=lookback_days)
        window = s[s.index >= cutoff]
        rows.append({
            "Spread": label,
            "Current (pips)": cur,
            "Percentile": pct,
            "Min": float(window.min()) if not window.empty else float("nan"),
            "p10": float(window.quantile(0.10)) if len(window) >= 5 else float("nan"),
            "Median": float(window.median()) if not window.empty else float("nan"),
            "p90": float(window.quantile(0.90)) if len(window) >= 5 else float("nan"),
            "Max": float(window.max()) if not window.empty else float("nan"),
        })
    df = pd.DataFrame(rows)

    def style_pct(v):
        if pd.isna(v):
            return ""
        if v > 99 or v < 1:
            return "background-color: rgba(214, 39, 40, 0.25); font-weight: bold;"
        if v > 90 or v < 10:
            return "background-color: rgba(255, 127, 14, 0.15);"
        return ""

    styled = (df.style
                .format({c: "{:,.2f}" for c in df.columns if c != "Spread"})
                .map(style_pct, subset=["Percentile"]))
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ============================================================================
# TAB 3 — Distribution
# ============================================================================
with tab_dist:
    sel_label = st.selectbox("Spread", spread_labels, key="fs_dist_sel")
    if sel_label and sel_label in spreads:
        s = spreads[sel_label]
        cutoff = s.index[-1] - pd.Timedelta(days=lookback_days)
        window = s[s.index >= cutoff].dropna()
        cur = float(s.iloc[-1])
        pct = current_percentile(s, lookback_days)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current (pips)", format_value(cur, 2))
        m2.metric("Percentile", format_value(pct, 1))
        m3.metric("Mean", format_value(float(window.mean()) if not window.empty else float("nan"), 2))
        m4.metric("Std", format_value(float(window.std()) if not window.empty else float("nan"), 2))

        fig_h = histogram_with_marker(
            window, cur,
            title=f"{pair}  ·  {sel_label}  ·  trailing {lookback_label}",
            height=380,
        )
        st.plotly_chart(fig_h, use_container_width=True)


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    f"Data: `{folder}` · Pair: **{pair}** · "
    f"Lookback: **{lookback_label}**"
    + (f" · Variant: **{prefer}**" if asia_em else "")
    + "  ·  Construction: linear (long-tenor minus short-tenor)"
)

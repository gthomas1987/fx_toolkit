"""App 1b — FX Forward Points across Tenors.

Run with:
    streamlit run apps/1b_fwd_points.py

Shows forward points for tenors 1W, 1M, 2M, 3M, 6M, 9M, 1Y for any pair,
with 2Y/3Y selectable lookback (per spec) and broader lookbacks available.

Tabs:
    1. Time series — all tenors overlaid + per-tenor percentile path
    2. Term structure — current curve vs historical curves
    3. Snapshot — current values, percentiles, history bands per tenor
    4. Distribution — histogram per tenor with current marker
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from core.conventions import FWD_TENORS, tenor_sort_key
from core.ts_loader import load_panel, list_available_pairs, list_available_tenors
from core.percentiles import (current_percentile, expanding_percentile,
                              reference_quantiles)
from core.charts import (time_series_chart, percentile_path_chart,
                         histogram_with_marker, term_structure_chart)
from core.ui import data_dir_input, lookback_selector, app_header, format_value


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="1b — FX Forward Points",
    layout="wide",
    initial_sidebar_state="expanded",
)
app_header(
    "1b · FX Forward Points — Historical Levels",
    "Tenors: 1W, 1M, 2M, 3M, 6M, 9M, 1Y  ·  default lookback 3Y (selectable)",
)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
folder = data_dir_input()
if folder is None:
    st.stop()

available_pairs = list_available_pairs(folder, "FWD_POINTS")
if not available_pairs:
    st.error("No FWD_POINTS tickers found in `_index.csv`.")
    st.stop()

default_pair = "EURUSD" if "EURUSD" in available_pairs else available_pairs[0]
pair = st.sidebar.selectbox(
    "Currency pair",
    available_pairs,
    index=available_pairs.index(default_pair),
    key="fp_pair",
)

asia_em = pair in {"USDCNH", "USDCNY", "USDINR", "USDIDR", "USDKRW",
                   "USDMYR", "USDPHP", "USDTHB", "USDTWD"}
prefer = "offshore"
if asia_em:
    prefer = st.sidebar.radio(
        "Variant", ["offshore", "onshore"], index=0,
        horizontal=True, key="fp_prefer",
    )

# 2Y/3Y per spec, plus the others for flexibility
lookback_label, lookback_days = lookback_selector(
    default="3Y",
    options=["2Y", "3Y", "5Y", "Full", "1Y"],
)


# -----------------------------------------------------------------------------
# Load forward points panels for this pair
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading forward points…")
def load_fwd_panels(folder: str, pair: str, prefer: str
                    ) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for tenor in FWD_TENORS:
        df = load_panel(folder, category="FWD_POINTS", tenor=tenor,
                        prefer=prefer, pairs=(pair,))
        if df.empty or pair not in df.columns:
            continue
        s = df[pair].dropna()
        if not s.empty:
            out[tenor] = s
    return out


panels = load_fwd_panels(folder, pair, prefer)
if not panels:
    st.warning(f"No FWD_POINTS data found for {pair}.")
    st.stop()

available_tenors = sorted(panels.keys(), key=tenor_sort_key)


# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_ts, tab_curve, tab_snap, tab_dist = st.tabs(
    ["📈 Time series", "📊 Term structure", "🔢 Snapshot", "📉 Distribution"]
)


# ============================================================================
# TAB 1 — Time series
# ============================================================================
with tab_ts:
    sel_tenors = st.multiselect(
        "Tenors to chart",
        available_tenors,
        default=available_tenors,
        key="fp_ts_tenors",
    )
    if not sel_tenors:
        st.info("Select at least one tenor.")
        st.stop()

    # Restrict to lookback window for cleaner viewing
    cutoff = max((panels[t].index[-1] for t in sel_tenors)) - pd.Timedelta(days=lookback_days)
    wide = pd.concat({t: panels[t] for t in sel_tenors}, axis=1)
    wide_cut = wide[wide.index >= cutoff]

    fig = time_series_chart(
        wide_cut,
        title=f"{pair}  ·  Forward points  ·  trailing {lookback_label}",
        yaxis_title="Forward points (pip units)",
        height=460,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Point-in-time percentile path  (each tenor)")
    for tenor in sel_tenors:
        s = panels[tenor]
        pct_now = current_percentile(s, lookback_days)
        pct_path = expanding_percentile(s, lookback_days=lookback_days)
        # Restrict displayed pct path to lookback for visual clarity
        pct_path_cut = pct_path[pct_path.index >= cutoff]
        fig_pct = percentile_path_chart(
            pct_path_cut,
            title=f"{tenor}  ·  current = {format_value(s.iloc[-1], 2)} "
                  f"(pct {format_value(pct_now, 1)})",
            height=170,
        )
        st.plotly_chart(fig_pct, use_container_width=True)


# ============================================================================
# TAB 2 — Term structure
# ============================================================================
with tab_curve:
    st.markdown(f"##### {pair}  ·  Forward-points term structure")
    ref_offsets_days = [0, 30, 90, 180, 365]
    ref_labels = ["Today", "1M ago", "3M ago", "6M ago", "1Y ago"]
    curves: dict[str, dict[str, float]] = {}
    for offset, label in zip(ref_offsets_days, ref_labels):
        curve = {}
        for tenor in available_tenors:
            s = panels[tenor]
            target_date = s.index[-1] - pd.Timedelta(days=offset)
            vals = s[s.index <= target_date]
            if not vals.empty:
                curve[tenor] = float(vals.iloc[-1])
        if curve:
            curves[label] = curve

    fig = term_structure_chart(
        curves,
        tenors_ordered=available_tenors,
        title="",
        yaxis_title="Forward points (pip units)",
        height=460,
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# TAB 3 — Snapshot
# ============================================================================
with tab_snap:
    st.markdown(f"##### Current values & {lookback_label} percentiles  ·  {pair}")
    rows = []
    for tenor in available_tenors:
        s = panels[tenor]
        cur = float(s.iloc[-1])
        pct = current_percentile(s, lookback_days)
        cutoff = s.index[-1] - pd.Timedelta(days=lookback_days)
        window = s[s.index >= cutoff]
        rows.append({
            "Tenor": tenor,
            "Current (pips)": cur,
            "Percentile": pct,
            "Min": float(window.min()) if not window.empty else float("nan"),
            "p10": float(window.quantile(0.10)) if len(window) >= 5 else float("nan"),
            "Median": float(window.median()) if not window.empty else float("nan"),
            "p90": float(window.quantile(0.90)) if len(window) >= 5 else float("nan"),
            "Max": float(window.max()) if not window.empty else float("nan"),
            "_tenor_order": tenor_sort_key(tenor),
        })
    df = pd.DataFrame(rows).sort_values("_tenor_order").drop(columns=["_tenor_order"])

    def style_pct(v):
        if pd.isna(v):
            return ""
        if v > 99 or v < 1:
            return "background-color: rgba(214, 39, 40, 0.25); font-weight: bold;"
        if v > 90 or v < 10:
            return "background-color: rgba(255, 127, 14, 0.15);"
        return ""

    styled = (df.style
                .format({c: "{:,.2f}" for c in df.columns if c != "Tenor"})
                .map(style_pct, subset=["Percentile"]))
    st.dataframe(styled, use_container_width=True, hide_index=True)

    n_extreme = int(((df["Percentile"] > 99) | (df["Percentile"] < 1)).sum())
    st.caption(f"{n_extreme} tenor(s) currently outside [1, 99] pct.")


# ============================================================================
# TAB 4 — Distribution
# ============================================================================
with tab_dist:
    dist_tenor = st.selectbox("Tenor", available_tenors, key="fp_dist_tenor")
    if dist_tenor and dist_tenor in panels:
        s = panels[dist_tenor]
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
            title=f"{pair}  ·  {dist_tenor} fwd points  ·  trailing {lookback_label}",
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
)

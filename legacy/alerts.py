"""Alerts Dashboard — current extremes across the FX vol surface and forwards.

Run with:
    streamlit run apps/alerts.py

Scans every (pair × metric × tenor) covered by apps 1a, 1b, and 1c, and
flags those whose CURRENT value sits outside [low_pct, high_pct] of the
trailing lookback window. Defaults to <1 / >99 pct on a 3Y lookback per
spec.

Output: a single sortable, filterable table — pair, metric class, metric,
tenor, current value, percentile, lookback summary stats. Each row
clickable to expand and inspect the full time series + percentile path.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from core.conventions import (FWD_TENORS, VOL_TENORS, VOL_CATEGORIES,
                              VOL_CATEGORY_ORDER, tenor_sort_key,
                              is_tradeable_fx_pair)
from core.ts_loader import load_panel, list_available_pairs
from core.percentiles import (current_percentile, expanding_percentile,
                              reference_quantiles, extremity_distance)
from core.charts import (time_series_with_quantile_bands,
                         percentile_path_chart, histogram_with_marker)
from core.ui import data_dir_input, lookback_selector, app_header, format_value
from core.fwd_structures import all_spreads, SPREAD_SPECS


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Alerts — FX Levels Extremes",
    layout="wide",
    initial_sidebar_state="expanded",
)
app_header(
    "🚨 Alerts — FX Levels Extremes",
    "Cross-pair scan of vol surface (1a) · forward points (1b) · spreads (1c)  ·  "
    "Highlights current values outside [low, high] percentile",
)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
folder = data_dir_input()
if folder is None:
    st.stop()

lookback_label, lookback_days = lookback_selector(default="3Y")

c1, c2 = st.sidebar.columns(2)
with c1:
    low_pct = st.number_input(
        "Low pct", min_value=0.0, max_value=20.0, value=1.0, step=0.5,
        key="alerts_low",
    )
with c2:
    high_pct = st.number_input(
        "High pct", min_value=80.0, max_value=100.0, value=99.0, step=0.5,
        key="alerts_high",
    )

st.sidebar.markdown("---")
st.sidebar.markdown("**Metric classes to scan**")
scan_vol = st.sidebar.checkbox("1a · Vol surface (ATM, RR, BF)", value=True,
                               key="alerts_scan_vol")
scan_fwd = st.sidebar.checkbox("1b · Forward points", value=True,
                               key="alerts_scan_fwd")
scan_sprd = st.sidebar.checkbox("1c · Calendar spreads", value=True,
                                key="alerts_scan_sprd")

prefer = st.sidebar.radio(
    "Asia EM variant", ["offshore", "onshore"], index=0,
    horizontal=True, key="alerts_prefer",
)

# Pair filter (optional — default all)
all_pairs_set: set[str] = set()
for cat in (["FWD_POINTS"] + VOL_CATEGORY_ORDER):
    all_pairs_set.update(list_available_pairs(folder, cat))
all_pairs = sorted(all_pairs_set)
sel_pairs = st.sidebar.multiselect(
    "Pairs (empty = all)",
    all_pairs,
    default=[],
    key="alerts_pairs",
)
scan_pairs = sel_pairs if sel_pairs else all_pairs


# -----------------------------------------------------------------------------
# Core scan
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Scanning for extremes…")
def scan_universe(folder: str, pairs: tuple[str, ...], prefer: str,
                  lookback_days: int, scan_vol: bool, scan_fwd: bool,
                  scan_sprd: bool) -> pd.DataFrame:
    """Build a DataFrame of (pair × metric class × metric × tenor) with
    current value, percentile, and lookback summary stats."""

    rows: list[dict] = []

    # --- 1a: Vol surface (5 metric classes × 6 tenors per pair) -------------
    if scan_vol:
        for cat in VOL_CATEGORY_ORDER:
            for tenor in VOL_TENORS:
                df = load_panel(folder, category=cat, tenor=tenor,
                                prefer=prefer, pairs=pairs)
                if df.empty:
                    continue
                cutoff_per_col = {
                    p: df[p].dropna().index[-1] - pd.Timedelta(days=lookback_days)
                    for p in df.columns if not df[p].dropna().empty
                }
                for pair in df.columns:
                    s = df[pair].dropna()
                    if s.empty:
                        continue
                    cur = float(s.iloc[-1])
                    pct = current_percentile(s, lookback_days)
                    cutoff = cutoff_per_col[pair]
                    win = s[s.index >= cutoff]
                    rows.append({
                        "_class": "1a Vol",
                        "Class": "Vol",
                        "Metric": VOL_CATEGORIES.get(cat, cat),
                        "Tenor": tenor,
                        "Pair": pair,
                        "Current": cur,
                        "Percentile": pct,
                        "p1":  float(win.quantile(0.01)) if len(win) >= 5 else float("nan"),
                        "p99": float(win.quantile(0.99)) if len(win) >= 5 else float("nan"),
                        "Min": float(win.min()) if not win.empty else float("nan"),
                        "Max": float(win.max()) if not win.empty else float("nan"),
                        "AsOf": s.index[-1].strftime("%Y-%m-%d"),
                        "_cat_key": cat,
                        "_tenor_key": tenor_sort_key(tenor),
                    })

    # --- 1b: Forward points (7 tenors per pair) -----------------------------
    if scan_fwd:
        for tenor in FWD_TENORS:
            df = load_panel(folder, category="FWD_POINTS", tenor=tenor,
                            prefer=prefer, pairs=pairs)
            if df.empty:
                continue
            for pair in df.columns:
                s = df[pair].dropna()
                if s.empty:
                    continue
                cur = float(s.iloc[-1])
                pct = current_percentile(s, lookback_days)
                cutoff = s.index[-1] - pd.Timedelta(days=lookback_days)
                win = s[s.index >= cutoff]
                rows.append({
                    "_class": "1b Fwd",
                    "Class": "Fwd Pts",
                    "Metric": "Forward points",
                    "Tenor": tenor,
                    "Pair": pair,
                    "Current": cur,
                    "Percentile": pct,
                    "p1":  float(win.quantile(0.01)) if len(win) >= 5 else float("nan"),
                    "p99": float(win.quantile(0.99)) if len(win) >= 5 else float("nan"),
                    "Min": float(win.min()) if not win.empty else float("nan"),
                    "Max": float(win.max()) if not win.empty else float("nan"),
                    "AsOf": s.index[-1].strftime("%Y-%m-%d"),
                    "_cat_key": "FWD",
                    "_tenor_key": tenor_sort_key(tenor),
                })

    # --- 1c: Calendar spreads (4 spreads per pair) ----------------------------
    if scan_sprd:
        # Need to load all fwd-points panels per pair, then build spreads
        # Do a per-pair pass to keep memory reasonable
        for pair in pairs:
            fwd_panels: dict[str, pd.Series] = {}
            for tenor in FWD_TENORS:
                df = load_panel(folder, category="FWD_POINTS", tenor=tenor,
                                prefer=prefer, pairs=(pair,))
                if df.empty or pair not in df.columns:
                    continue
                s = df[pair].dropna()
                if not s.empty:
                    fwd_panels[tenor] = s
            if not fwd_panels:
                continue
            spreads = all_spreads(fwd_panels)
            for label, s in spreads.items():
                if s.empty:
                    continue
                cur = float(s.iloc[-1])
                pct = current_percentile(s, lookback_days)
                cutoff = s.index[-1] - pd.Timedelta(days=lookback_days)
                win = s[s.index >= cutoff]
                rows.append({
                    "_class": "1c Spread",
                    "Class": "Spread",
                    "Metric": "Calendar spread",
                    "Tenor": label,
                    "Pair": pair,
                    "Current": cur,
                    "Percentile": pct,
                    "p1":  float(win.quantile(0.01)) if len(win) >= 5 else float("nan"),
                    "p99": float(win.quantile(0.99)) if len(win) >= 5 else float("nan"),
                    "Min": float(win.min()) if not win.empty else float("nan"),
                    "Max": float(win.max()) if not win.empty else float("nan"),
                    "AsOf": s.index[-1].strftime("%Y-%m-%d"),
                    "_cat_key": "SPREAD",
                    "_tenor_key": 0,
                })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


with st.spinner("Scanning…"):
    full_df = scan_universe(
        folder, tuple(scan_pairs), prefer, lookback_days,
        scan_vol, scan_fwd, scan_sprd,
    )

if full_df.empty:
    st.info("No metrics scanned (check that data classes are enabled and data exists).")
    st.stop()

# Derived columns
full_df["Extremity"] = full_df["Percentile"].apply(extremity_distance)
full_df["IsExtreme"] = full_df["Percentile"].apply(
    lambda p: pd.notna(p) and (p < low_pct or p > high_pct)
)
full_df["Direction"] = full_df["Percentile"].apply(
    lambda p: "↑ HIGH" if pd.notna(p) and p > high_pct
              else ("↓ LOW" if pd.notna(p) and p < low_pct else "—")
)


# -----------------------------------------------------------------------------
# Header summary metrics
# -----------------------------------------------------------------------------
n_total = len(full_df)
n_extreme = int(full_df["IsExtreme"].sum())
n_high = int((full_df["Percentile"] > high_pct).sum())
n_low  = int((full_df["Percentile"] < low_pct).sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Metrics scanned", f"{n_total:,}")
m2.metric("🚨 Currently extreme", f"{n_extreme:,}")
m3.metric("↑ HIGH (above " + str(high_pct) + " pct)", f"{n_high:,}")
m4.metric("↓ LOW (below " + str(low_pct) + " pct)", f"{n_low:,}")


# -----------------------------------------------------------------------------
# Two tabs: Extremes (default), All metrics
# -----------------------------------------------------------------------------
tab_ex, tab_all = st.tabs([f"🚨 Extremes ({n_extreme})", f"All scanned ({n_total})"])


def style_alerts(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Conditional formatting on Percentile column."""
    def style_pct(v):
        if pd.isna(v):
            return ""
        if v > high_pct:
            return "background-color: rgba(214, 39, 40, 0.30); font-weight: bold;"
        if v < low_pct:
            return "background-color: rgba(31, 119, 180, 0.30); font-weight: bold;"
        if v > 90:
            return "background-color: rgba(255, 127, 14, 0.15);"
        if v < 10:
            return "background-color: rgba(31, 119, 180, 0.10);"
        return ""

    return (df.style
              .format({
                  "Current": "{:.4f}", "Percentile": "{:.1f}",
                  "p1": "{:.4f}", "p99": "{:.4f}",
                  "Min": "{:.4f}", "Max": "{:.4f}",
                  "Extremity": "{:.1f}",
              })
              .map(style_pct, subset=["Percentile"]))


display_cols = ["Pair", "Class", "Metric", "Tenor", "Direction",
                "Current", "Percentile", "p1", "p99", "Min", "Max", "AsOf"]


# ============================================================================
# TAB 1 — Extremes only
# ============================================================================
with tab_ex:
    sub = full_df[full_df["IsExtreme"]].sort_values(
        "Extremity", ascending=False
    ).reset_index(drop=True)
    if sub.empty:
        st.success(
            f"✓ No metrics currently outside [{low_pct}, {high_pct}] pct over "
            f"{lookback_label} lookback."
        )
    else:
        st.dataframe(
            style_alerts(sub[display_cols]),
            use_container_width=True, hide_index=True, height=480,
        )

        # Drill-down — pick one row to inspect
        st.markdown("---")
        st.markdown("##### 🔍 Inspect a flagged metric")
        sub["_label"] = (
            sub["Pair"] + "  ·  " + sub["Metric"] + "  ·  " + sub["Tenor"]
            + "  →  " + sub["Direction"] + " (pct " + sub["Percentile"].round(1).astype(str) + ")"
        )
        choice = st.selectbox("Metric", sub["_label"].tolist(), key="alerts_drill")
        if choice:
            row = sub[sub["_label"] == choice].iloc[0]

            # Reload the underlying series
            cls = row["_class"]
            pair = row["Pair"]
            tenor = row["Tenor"]
            if cls == "1a Vol":
                # Recover category from Metric label
                cat = next(
                    (c for c, lbl in VOL_CATEGORIES.items() if lbl == row["Metric"]),
                    None,
                )
                df_ = load_panel(folder, category=cat, tenor=tenor,
                                 prefer=prefer, pairs=(pair,))
                s = df_[pair].dropna() if (not df_.empty and pair in df_.columns) else pd.Series(dtype=float)
                ylab = "Vol level (%)" if cat == "VOL_ATM" else "Vol points"
            elif cls == "1b Fwd":
                df_ = load_panel(folder, category="FWD_POINTS", tenor=tenor,
                                 prefer=prefer, pairs=(pair,))
                s = df_[pair].dropna() if (not df_.empty and pair in df_.columns) else pd.Series(dtype=float)
                ylab = "Forward points (pips)"
            elif cls == "1c Spread":
                fwd_panels: dict[str, pd.Series] = {}
                for t in FWD_TENORS:
                    df_ = load_panel(folder, category="FWD_POINTS", tenor=t,
                                     prefer=prefer, pairs=(pair,))
                    if not df_.empty and pair in df_.columns:
                        fwd_panels[t] = df_[pair].dropna()
                spreads = all_spreads(fwd_panels)
                s = spreads.get(tenor, pd.Series(dtype=float))
                ylab = "Spread (pips)"
            else:
                s = pd.Series(dtype=float)
                ylab = ""

            if not s.empty:
                cutoff = s.index[-1] - pd.Timedelta(days=lookback_days)
                s_cut = s[s.index >= cutoff]
                quantiles = reference_quantiles(s, lookback_days=lookback_days)
                pct_path = expanding_percentile(s, lookback_days=lookback_days)
                pct_path_cut = pct_path[pct_path.index >= cutoff]

                c1, c2 = st.columns([3, 2])
                with c1:
                    fig_v = time_series_with_quantile_bands(
                        s_cut, quantiles,
                        title=f"{pair} · {row['Metric']} · {tenor}",
                        yaxis_title=ylab, height=360,
                    )
                    st.plotly_chart(fig_v, use_container_width=True)
                with c2:
                    fig_h = histogram_with_marker(
                        s_cut, float(s.iloc[-1]),
                        title=f"Distribution · {lookback_label}",
                        height=360,
                    )
                    st.plotly_chart(fig_h, use_container_width=True)

                fig_p = percentile_path_chart(
                    pct_path_cut,
                    title="Point-in-time percentile path",
                    height=200,
                )
                st.plotly_chart(fig_p, use_container_width=True)


# ============================================================================
# TAB 2 — All scanned
# ============================================================================
with tab_all:
    # Filters
    f1, f2, f3 = st.columns(3)
    with f1:
        flt_class = st.multiselect(
            "Class filter", sorted(full_df["Class"].unique()),
            default=[], key="alerts_flt_class")
    with f2:
        flt_pair = st.multiselect(
            "Pair filter", sorted(full_df["Pair"].unique()),
            default=[], key="alerts_flt_pair")
    with f3:
        sort_by = st.selectbox(
            "Sort by",
            ["Extremity (high to low)", "Percentile (high to low)",
             "Percentile (low to high)", "Pair · Metric"],
            index=0, key="alerts_sort",
        )

    sub = full_df.copy()
    if flt_class:
        sub = sub[sub["Class"].isin(flt_class)]
    if flt_pair:
        sub = sub[sub["Pair"].isin(flt_pair)]

    if sort_by == "Extremity (high to low)":
        sub = sub.sort_values("Extremity", ascending=False)
    elif sort_by == "Percentile (high to low)":
        sub = sub.sort_values("Percentile", ascending=False, na_position="last")
    elif sort_by == "Percentile (low to high)":
        sub = sub.sort_values("Percentile", ascending=True, na_position="last")
    else:
        sub = sub.sort_values(["Pair", "Class", "_cat_key", "_tenor_key"])

    st.dataframe(
        style_alerts(sub[display_cols]),
        use_container_width=True, hide_index=True, height=560,
    )


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    f"Data: `{folder}`  ·  Lookback: **{lookback_label}**  ·  "
    f"Extreme thresholds: **<{low_pct}** or **>{high_pct}** pct  ·  "
    f"Asia EM variant: **{prefer}**"
)

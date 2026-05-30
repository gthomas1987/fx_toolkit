"""Vol Dashboard — FX Vol Surface Historical Percentiles.

Sub-app of the FX Toolkit. Reached via the landing page or
sidebar nav; not run directly.

First tab: a 6-up grid of normalised vol smiles (one chart per tenor:
ON, 1W, 1M, 3M, 6M, 1Y). Each chart plots vol(strike)/ATM across delta
strikes [10Δ Put, 25Δ Put, ATM, 25Δ Call, 10Δ Call] with:
    - shaded band between the configurable low/high percentiles (default 1/99)
    - dashed line for the historical mean smile
    - bold red line for the current smile (prominent)

Other tabs (deeper drill-down):
    Implied Vol (ATM time series per tenor + term structure + distribution),
    Heatmap (percentile grid + clickable vol cards + drill-down),
    Alerts (vol-surface extremes).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `core/` importable when the app is run from project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from core.conventions import (VOL_TENORS, VOL_CATEGORIES, VOL_CATEGORY_ORDER,
                              tenor_sort_key)
from core.ts_loader import load_panel, list_available_pairs
from core.percentiles import (current_percentile, expanding_percentile,
                              reference_quantiles, extremity_distance)
from core.charts import (time_series_chart, percentile_path_chart,
                         histogram_with_marker, term_structure_chart,
                         smile_chart, percentile_heatmap,
                         time_series_with_quantile_bands,
                         time_series_with_pct_band)
from core.smile import (compute_smile_panel, compute_absolute_smile_panel,
                        DELTA_STRIKES)
from core.ui import data_dir_input, lookback_selector, app_header, format_value


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FX Vol Surface",
    layout="wide",
    initial_sidebar_state="expanded",
)
from shared.style import inject_base_css
inject_base_css()
app_header(
    "FX Vol Surface — Historical Percentiles",
    "Normalised smile  ·  ATM · 25Δ RR · 10Δ RR · 25Δ BF · 10Δ BF  ·  "
    "Tenors: ON, 1W, 1M, 3M, 6M, 1Y",
)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
folder = data_dir_input(default="market_data")
if folder is None:
    st.stop()

# Discover available pairs across all vol categories
all_pairs: set[str] = set()
for cat in VOL_CATEGORY_ORDER:
    all_pairs.update(list_available_pairs(folder, cat))
all_pairs_sorted = sorted(all_pairs)
if not all_pairs_sorted:
    st.error("No vol-surface tickers found in _index.csv. "
             "Categories expected: " + ", ".join(VOL_CATEGORY_ORDER))
    st.stop()

default_pair = (
    "USDJPY" if "USDJPY" in all_pairs_sorted
    else "EURUSD" if "EURUSD" in all_pairs_sorted
    else all_pairs_sorted[0]
)
pair = st.sidebar.selectbox(
    "Currency pair",
    all_pairs_sorted,
    index=all_pairs_sorted.index(default_pair),
    key="vs_pair",
)

# Asia EM: surface onshore/offshore toggle
asia_em = pair in {"USDCNH", "USDCNY", "USDINR", "USDIDR", "USDKRW",
                   "USDMYR", "USDPHP", "USDTHB", "USDTWD"}
prefer = "offshore"
if asia_em:
    prefer = st.sidebar.radio(
        "Variant", ["offshore", "onshore"], index=0,
        horizontal=True, key="vs_prefer",
    )

lookback_label, lookback_days = lookback_selector(default="1Y")

st.sidebar.markdown("**Skew band percentiles**")
b1, b2 = st.sidebar.columns(2)
with b1:
    low_pct = st.number_input(
        "Lower", min_value=0.1, max_value=49.9, value=1.0, step=0.5,
        key="vs_low_pct",
        help="Lower edge of the shaded band on the smile chart.",
    )
with b2:
    high_pct = st.number_input(
        "Upper", min_value=50.1, max_value=99.9, value=99.0, step=0.5,
        key="vs_high_pct",
        help="Upper edge of the shaded band on the smile chart.",
    )


# -----------------------------------------------------------------------------
# Load all metric × tenor panels for this pair (used by every tab)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading vol surface…")
def load_all_panels(folder: str, pair: str, prefer: str
                    ) -> dict[tuple[str, str], pd.Series]:
    """Returns {(category, tenor): Series(date → value)} for one pair."""
    out: dict[tuple[str, str], pd.Series] = {}
    for cat in VOL_CATEGORY_ORDER:
        for tenor in VOL_TENORS:
            df = load_panel(folder, category=cat, tenor=tenor,
                            prefer=prefer, pairs=(pair,))
            if df.empty or pair not in df.columns:
                continue
            s = df[pair].dropna()
            if not s.empty:
                out[(cat, tenor)] = s
    return out


@st.cache_data(show_spinner="Building smile panels…")
def build_smile_panels(folder: str, pair: str, prefer: str
                       ) -> dict[str, pd.DataFrame]:
    """Returns {tenor: smile_df} where smile_df has columns DELTA_STRIKES."""
    out: dict[str, pd.DataFrame] = {}
    for tenor in VOL_TENORS:
        atm_df = load_panel(folder, "VOL_ATM", tenor=tenor,
                            prefer=prefer, pairs=(pair,))
        if atm_df.empty or pair not in atm_df.columns:
            continue
        atm = atm_df[pair].dropna()
        if atm.empty:
            continue

        def _maybe(cat: str):
            df = load_panel(folder, cat, tenor=tenor,
                            prefer=prefer, pairs=(pair,))
            if df.empty or pair not in df.columns:
                return None
            s = df[pair].dropna()
            return s if not s.empty else None

        smile_df = compute_smile_panel(
            atm=atm,
            rr_25=_maybe("VOL_RR_25D"),
            bf_25=_maybe("VOL_BF_25D"),
            rr_10=_maybe("VOL_RR_10D"),
            bf_10=_maybe("VOL_BF_10D"),
        )
        if not smile_df.empty:
            out[tenor] = smile_df
    return out


@st.cache_data(show_spinner="Building absolute vol panels…")
def build_absolute_smile_panels(folder: str, pair: str, prefer: str
                                 ) -> dict[str, pd.DataFrame]:
    """Returns {tenor: smile_df} with ABSOLUTE vol levels (not normalised)."""
    out: dict[str, pd.DataFrame] = {}
    for tenor in VOL_TENORS:
        atm_df = load_panel(folder, "VOL_ATM", tenor=tenor,
                            prefer=prefer, pairs=(pair,))
        if atm_df.empty or pair not in atm_df.columns:
            continue
        atm = atm_df[pair].dropna()
        if atm.empty:
            continue

        def _maybe(cat: str):
            df = load_panel(folder, cat, tenor=tenor,
                            prefer=prefer, pairs=(pair,))
            if df.empty or pair not in df.columns:
                return None
            s = df[pair].dropna()
            return s if not s.empty else None

        abs_df = compute_absolute_smile_panel(
            atm=atm,
            rr_25=_maybe("VOL_RR_25D"),
            bf_25=_maybe("VOL_BF_25D"),
            rr_10=_maybe("VOL_RR_10D"),
            bf_10=_maybe("VOL_BF_10D"),
        )
        if not abs_df.empty:
            out[tenor] = abs_df
    return out


panels = load_all_panels(folder, pair, prefer)
if not panels:
    st.warning(f"No vol-surface data found for {pair}.")
    st.stop()

available_categories = sorted({k[0] for k in panels.keys()},
                              key=VOL_CATEGORY_ORDER.index)
available_tenors_per_cat = {
    cat: sorted({k[1] for k in panels.keys() if k[0] == cat}, key=tenor_sort_key)
    for cat in available_categories
}


# -----------------------------------------------------------------------------
# Tabs (Skew is the new landing tab — first position)
# -----------------------------------------------------------------------------
(tab_skew, tab_iv, tab_heat, tab_alerts,
 tab_vol3m, tab_carry, tab_dnt, tab_binary) = st.tabs([
    "📊 Skew", "📈 Implied Vol", "🌡️ Heatmap", "🚨 Alerts",
    "🎯 3m Vol Screen", "💰 Static Carry",
    "🚫 DNT", "🎲 Binary 10:1",
])


# ============================================================================
# TAB 0 — Skew (landing) — 6-up grid of normalised smiles
# ============================================================================
with tab_skew:
    smile_panels = build_smile_panels(folder, pair, prefer)
    if not smile_panels:
        st.warning(
            "Couldn't construct smile panels — need at least ATM + (RR_25D, BF_25D)."
        )
    else:
        tenors_in_order = sorted(smile_panels.keys(), key=tenor_sort_key)

        # Compute shared y-range so the 6 mini-charts are visually comparable
        all_lo, all_hi = [], []
        for tenor in tenors_in_order:
            sdf = smile_panels[tenor]
            # When `lookback_days is None` (the "Full" option in the
            # sidebar), use the entire panel — skip the cutoff filter
            # entirely. pd.Timedelta has a ~292-year ceiling and would
            # raise OutOfBoundsTimedelta on the previous 10^9-day sentinel.
            if lookback_days is None:
                win = sdf
            else:
                cutoff = sdf.index[-1] - pd.Timedelta(days=lookback_days)
                win = sdf[sdf.index >= cutoff]
            if len(win) < 5:
                continue
            for col in DELTA_STRIKES:
                if col not in sdf.columns or win[col].notna().sum() == 0:
                    continue
                lo_v = float(win[col].quantile(low_pct / 100.0))
                hi_v = float(win[col].quantile(high_pct / 100.0))
                cur_v = float(sdf[col].iloc[-1])
                if pd.notna(lo_v): all_lo.append(lo_v)
                if pd.notna(hi_v): all_hi.append(hi_v)
                if pd.notna(cur_v):
                    all_lo.append(cur_v); all_hi.append(cur_v)
        if all_lo and all_hi:
            ymin = min(all_lo) - 0.01
            ymax = max(all_hi) + 0.01
            yrange = (ymin, ymax)
        else:
            yrange = None

        # Caption strip
        st.markdown(
            f"**{pair}** "
            + (f"  ·  *{prefer}*" if asia_em else "")
            + f"  ·  Band: **{low_pct:g}–{high_pct:g} pct** over **{lookback_label}**"
            + f"  ·  As of: **{max(s.index[-1] for s in smile_panels.values()).strftime('%Y-%m-%d')}**"
        )

        # Standalone legend strip — replaces the embedded legend that
        # used to sit inside the first chart. Inline SVG glyphs mirror
        # the chart trace styling exactly (red solid line + circle for
        # Current, dashed grey + small circle for Historical mean,
        # translucent blue rectangle for the band). Keeping the legend
        # external lets every chart use its full plotting area, and
        # avoids the awkward asymmetry of one chart being taller than
        # the other five.
        legend_html = (
            f"""<div style="display:flex;gap:28px;align-items:center;
                       flex-wrap:wrap;margin:6px 0 14px 0;font-size:14px;">
              <div style="display:flex;align-items:center;gap:8px;">
                <svg width="36" height="14" style="overflow:visible;">
                  <line x1="2" y1="7" x2="34" y2="7"
                        stroke="#d62728" stroke-width="2.5"/>
                  <circle cx="18" cy="7" r="4" fill="#d62728"/>
                </svg>
                <span><b>Current</b></span>
              </div>
              <div style="display:flex;align-items:center;gap:8px;">
                <svg width="36" height="14" style="overflow:visible;">
                  <line x1="2" y1="7" x2="34" y2="7"
                        stroke="#888" stroke-width="1.6"
                        stroke-dasharray="5,3"/>
                  <circle cx="18" cy="7" r="2.5" fill="#888"/>
                </svg>
                <span>Historical mean</span>
              </div>
              <div style="display:flex;align-items:center;gap:8px;">
                <span style="display:inline-block;width:34px;height:12px;
                              background:rgba(31,119,180,0.30);
                              border:1px solid rgba(31,119,180,0.55);">
                </span>
                <span>[{low_pct:g}, {high_pct:g}] pct band</span>
              </div>
            </div>"""
        )
        st.markdown(legend_html, unsafe_allow_html=True)

        # 2-column grid (was 3 wide; now 2 wide so each chart has more
        # horizontal room to breathe). With 6 tenors we now get 3 rows
        # of 2 instead of 2 rows of 3.
        n_per_row = 2
        for row_start in range(0, len(tenors_in_order), n_per_row):
            cols = st.columns(n_per_row)
            for i, tenor in enumerate(tenors_in_order[row_start:row_start + n_per_row]):
                with cols[i]:
                    fig = smile_chart(
                        smile_panels[tenor],
                        lookback_days=lookback_days,
                        low_pct=low_pct, high_pct=high_pct,
                        title=f"{tenor}",
                        height=320,
                        yrange=yrange,
                        show_legend=False,    # legend lives above the grid
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Each chart: vol(strike) ÷ ATM at 5 delta points. "
            "**Red line** = current; **dashed grey** = historical mean; "
            f"**shaded band** = [{low_pct:g}, {high_pct:g}] percentile range over "
            f"the {lookback_label} window. ATM column is 1.0 by construction."
        )

        # --------------------------------------------------------------
        # Download Data — full time-series panels for the selected pair,
        # in wide format. Columns: date, then one column per
        # (category, tenor) pair (e.g. VOL_ATM_3M, VOL_RR_25D_6M, ...).
        # Useful for taking the data into a separate analysis. Avoids
        # re-loading anything — uses `panels` already in memory.
        # --------------------------------------------------------------
        st.markdown("---")
        download_frame = pd.concat(
            {f"{cat}_{tenor}": s for (cat, tenor), s in panels.items()},
            axis=1,
        ).sort_index()
        download_frame.index.name = "date"
        csv_bytes = download_frame.to_csv().encode("utf-8")
        download_name = (
            f"{pair}_vol_surface"
            + (f"_{prefer}" if asia_em else "")
            + f"_{download_frame.index[-1].strftime('%Y%m%d')}.csv"
        )
        st.download_button(
            label="⬇  Download Data (CSV)",
            data=csv_bytes,
            file_name=download_name,
            mime="text/csv",
            help=(
                f"Full time-series panels for {pair} — all available "
                "(category, tenor) combinations in wide format, dates as "
                "the index."
            ),
            key="skew_download_data",
        )


# ============================================================================
# TAB 1 — Implied Vol — ATM vol time series per tenor (6 charts, banded)
# ============================================================================
# One chart per tenor (ON, 1W, 1M, 3M, 6M, 1Y). Each shows the ATM
# implied vol history with a shaded percentile band at the user-selected
# [low_pct, high_pct] levels. The band is time-varying — at each point
# the band edges are the trailing-lookback quantile cutoffs computed
# strictly without look-ahead (`reference_quantiles` enforces that).
#
# Note on interpretation: "implied vol at each tenor" is shown as the
# ATM vol for that tenor — the cleanest single-number reading of "vol
# level" for the tenor. To inspect non-ATM strikes (10Δ Put, 25Δ Call,
# etc.), the Heatmap tab's clickable cells route to the same chart
# type but for the chosen (tenor, strike) pair.
with tab_iv:
    iv_tenors = [t for t in VOL_TENORS if ("VOL_ATM", t) in panels]
    if not iv_tenors:
        st.warning(f"No ATM vol data available for {pair}.")
    else:
        st.markdown(
            f"**{pair} · ATM implied vol** "
            + (f"  ·  *{prefer}*" if asia_em else "")
            + f"  ·  Shaded band: **[{low_pct:g}, {high_pct:g}] pct** "
              f"over **{lookback_label}** trailing window"
            + f"  ·  As of: **{max(panels[('VOL_ATM', t)].index[-1] for t in iv_tenors).strftime('%Y-%m-%d')}**"
        )
        st.caption(
            "Each chart: full ATM-vol history for one tenor, with a "
            "shaded percentile band at the cutoffs you picked in the "
            "sidebar. Band edges are **rolling-window** quantile cutoffs "
            "— at each date they reflect only data available up to that "
            "point, so the shading widens or contracts as vol regimes "
            "shift. The red dot marks the current value; the dashed line "
            "is the rolling median."
        )

        # 2-up grid — same aesthetic choice as the Skew tab. With 6
        # tenors this gives a clean 3-row layout. Each chart is tall
        # enough to read the y-axis without crowding.
        n_per_row = 2
        for row_start in range(0, len(iv_tenors), n_per_row):
            cols = st.columns(n_per_row)
            for i, tenor in enumerate(iv_tenors[row_start:row_start + n_per_row]):
                with cols[i]:
                    s = panels[("VOL_ATM", tenor)]
                    # Trim to the user-selected lookback for the chart
                    # x-axis (the band edges themselves are computed
                    # over the same window via reference_quantiles).
                    if lookback_days is None:
                        s_disp = s.dropna()
                    else:
                        cutoff = s.index[-1] - pd.Timedelta(days=lookback_days)
                        s_disp = s[s.index >= cutoff].dropna()
                    pct_now = current_percentile(s, lookback_days)
                    cur_v = float(s.iloc[-1])
                    pct_str = format_value(pct_now, 1)
                    fig = time_series_with_pct_band(
                        s_disp,
                        lookback_days=lookback_days,
                        low_pct=low_pct, high_pct=high_pct,
                        title=(f"{tenor}  ·  current = "
                                  f"{format_value(cur_v, 3)} "
                                  f"(pct {pct_str})"),
                        yaxis_title="ATM vol (%)",
                        height=300,
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # --------------------------------------------------------------
        # Term-structure section (was previously its own tab).
        # Lets the user pick any vol metric (ATM, RR_25, BF_25, RR_10,
        # BF_10) and see the current curve across tenors plus four
        # historical reference dates (1M / 3M / 6M / 1Y ago). The
        # selector defaults to ATM so it pairs naturally with the
        # six-chart grid above.
        # --------------------------------------------------------------
        st.markdown("---")
        st.markdown("##### Term structure")
        c1_ts, _ = st.columns([1, 1])
        with c1_ts:
            curve_cat = st.selectbox(
                "Metric",
                available_categories,
                format_func=lambda c: VOL_CATEGORIES.get(c, c),
                key="iv_curve_cat",
            )
        avail_t_curve = available_tenors_per_cat.get(curve_cat, [])
        if not avail_t_curve:
            st.warning(f"No tenors for {VOL_CATEGORIES[curve_cat]}.")
        else:
            # Pull "today" + four historical curves so the user can see
            # how the term structure has shifted. Offsets are calendar
            # days, not trading days — close enough for vol reads.
            ref_offsets_days = [0, 30, 90, 180, 365]
            ref_labels = ["Today", "1M ago", "3M ago", "6M ago", "1Y ago"]
            curves: dict[str, dict[str, float]] = {}
            for offset, label in zip(ref_offsets_days, ref_labels):
                curve = {}
                for tenor in avail_t_curve:
                    s = panels[(curve_cat, tenor)]
                    if s.empty:
                        continue
                    target_date = s.index[-1] - pd.Timedelta(days=offset)
                    vals = s[s.index <= target_date]
                    if not vals.empty:
                        curve[tenor] = float(vals.iloc[-1])
                if curve:
                    curves[label] = curve

            fig_curve = term_structure_chart(
                curves,
                tenors_ordered=avail_t_curve,
                title=(f"{pair}  ·  {VOL_CATEGORIES[curve_cat]}  ·  "
                          "Term structure"),
                yaxis_title=("Vol level (%)" if curve_cat == "VOL_ATM"
                                else "Vol points"),
                height=420,
            )
            st.plotly_chart(fig_curve, use_container_width=True)

        # --------------------------------------------------------------
        # Distribution section (was previously its own tab).
        # Histogram of a single (metric, tenor) over the lookback window
        # with a vertical line at the current value, plus 4 summary
        # metrics above. Two independent dropdowns let the user pick
        # any combination — keys are namespaced `iv_dist_*` so they
        # don't collide with the Heatmap drill-down's selectors.
        # --------------------------------------------------------------
        st.markdown("---")
        st.markdown("##### Distribution")
        c1_d, c2_d = st.columns([1, 1])
        with c1_d:
            dist_cat = st.selectbox(
                "Metric",
                available_categories,
                format_func=lambda c: VOL_CATEGORIES.get(c, c),
                key="iv_dist_cat",
            )
        avail_t_dist = available_tenors_per_cat.get(dist_cat, [])
        with c2_d:
            dist_tenor = st.selectbox(
                "Tenor", avail_t_dist, key="iv_dist_tenor",
            )

        if dist_tenor and (dist_cat, dist_tenor) in panels:
            s = panels[(dist_cat, dist_tenor)]
            # `lookback_days is None` ⇒ Full lookback ⇒ window is the
            # whole series; .dropna() still applies so the histogram
            # only counts observations.
            if lookback_days is None:
                window = s.dropna()
            else:
                cutoff = s.index[-1] - pd.Timedelta(days=lookback_days)
                window = s[s.index >= cutoff].dropna()
            cur = float(s.iloc[-1])
            pct = current_percentile(s, lookback_days)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Current", format_value(cur, 4))
            m2.metric("Percentile", format_value(pct, 1))
            m3.metric(
                "Mean",
                format_value(float(window.mean())
                                if not window.empty else float("nan"), 4),
            )
            m4.metric(
                "Std",
                format_value(float(window.std())
                                if not window.empty else float("nan"), 4),
            )

            fig_h = histogram_with_marker(
                window, cur,
                title=(f"{pair}  ·  {VOL_CATEGORIES[dist_cat]}  ·  "
                          f"{dist_tenor}  ·  trailing {lookback_label}"),
                height=380,
            )
            st.plotly_chart(fig_h, use_container_width=True)


# ============================================================================
# TAB 2 — Heatmap — clickable vol cards + percentile heatmap + drill-down
# ============================================================================
# Two displays, top to bottom:
#   1. PERCENTILE — (tenor × strike) grid as a coloured heatmap table
#      (red → yellow → green on fixed 0..100 scale, cell text centred).
#      Use this for a fast visual scan of which cells are currently
#      extreme.
#   2. ABSOLUTE VOL LEVEL — same grid rendered as a row of clickable
#      cards with the current vol value as the label (e.g. "9.5%").
#      Clicking a card seeds the Tenor/Strike dropdowns below.
#
# Drill-down section below: two dropdowns (Tenor, Strike) seeded by the
# last card click, a `time_series_with_pct_band` chart of the history
# with the user-selected percentile band, and a Distribution-style block
# (4 metrics + histogram) below the chart. All vols rendered as % with
# 1 decimal place — internal data stays in decimal, ×100 at display.
with tab_heat:
    abs_panels = build_absolute_smile_panels(folder, pair, prefer)
    if not abs_panels:
        st.warning(
            "Couldn't construct absolute-vol panels — need at least "
            "ATM + (RR_25D, BF_25D)."
        )
    else:
        # Build the (tenor × strike) grids — abs vols are in DECIMAL
        # units (post the `_convert_smile_inputs` unit-handling fix);
        # we'll multiply by 100 at the display layer to render % values.
        present_tenors = [t for t in VOL_TENORS if t in abs_panels]
        pct_rows: dict[str, dict[str, float]] = {}
        lvl_rows: dict[str, dict[str, float]] = {}
        for tenor in present_tenors:
            adf = abs_panels[tenor]
            row_pct: dict[str, float] = {}
            row_lvl: dict[str, float] = {}
            for strike in DELTA_STRIKES:
                if strike not in adf.columns or adf[strike].notna().sum() < 5:
                    row_pct[strike] = float("nan")
                    row_lvl[strike] = float("nan")
                    continue
                row_pct[strike] = current_percentile(adf[strike], lookback_days)
                row_lvl[strike] = float(adf[strike].dropna().iloc[-1])
            pct_rows[tenor] = row_pct
            lvl_rows[tenor] = row_lvl

        levels_df = pd.DataFrame(lvl_rows).T.reindex(present_tenors)[DELTA_STRIKES]
        pct_df = pd.DataFrame(pct_rows).T.reindex(present_tenors)[DELTA_STRIKES]

        as_of = max(s.index[-1] for s in abs_panels.values())
        st.markdown(
            f"**{pair}** "
            + (f"  ·  *{prefer}*" if asia_em else "")
            + f"  ·  Current absolute vol vs **{lookback_label}** lookback"
            + f"  ·  As of: **{as_of.strftime('%Y-%m-%d')}**"
        )

        # ---- Percentile heatmap table — fast visual scan for which
        # (tenor, strike) cells are currently extreme. Sits ABOVE the
        # card grid so a glance at this table tells you where to click
        # in the grid below.
        st.markdown("")
        st.markdown("##### Percentile in lookback window")
        st.caption(
            f"Cell value: current percentile in the trailing-"
            f"{lookback_label} window. Background colour is on a fixed "
            "0–100 scale (red → yellow → green), so a 99 pct cell looks "
            "the same regardless of tenor."
        )

        pct_display = pct_df.copy()
        pct_display.insert(0, "Tenor", pct_display.index)
        pct_display = pct_display.reset_index(drop=True)

        def _abs_ryg(row):
            """Fixed-scale 0..100 diverging palette for percentiles.
            Cell text centred for cleaner reading."""
            out = []
            for col in pct_display.columns:
                v = row[col]
                if col == "Tenor":
                    # Tenor column — no fill, but centre the label
                    out.append("text-align: center; font-weight: 600;")
                    continue
                if pd.isna(v):
                    out.append("text-align: center;")
                    continue
                t = max(0.0, min(1.0, float(v) / 100.0))
                if t < 0.5:
                    x = t / 0.5
                    r = int(220 + (255 - 220) * x)
                    g = int(80 + (220 - 80) * x)
                    b = int(80 + (130 - 80) * x)
                else:
                    x = (t - 0.5) / 0.5
                    r = int(255 + (90 - 255) * x)
                    g = int(220 + (180 - 220) * x)
                    b = int(130 + (90 - 130) * x)
                out.append(
                    f"background-color: rgb({r},{g},{b}); color: #1a1a1a; "
                    "text-align: center;"
                )
            return out

        styled_pct = (pct_display.style
                          .apply(_abs_ryg, axis=1)
                          .format({c: "{:.0f}" for c in DELTA_STRIKES},
                                    na_rep="—")
                          # Centre the column headers too — the per-cell
                          # text-align above only catches the data cells,
                          # not the <th> row.
                          .set_table_styles([
                              {"selector": "th",
                                "props": [("text-align", "center"),
                                          ("font-weight", "600")]},
                          ]))
        st.dataframe(
            styled_pct, use_container_width=True, hide_index=True,
            key="heat_pct_table",
        )

        # Quick-scan summary of the percentile grid
        flat_pct = pct_df.stack().dropna()
        n_extreme = int(((flat_pct < low_pct) | (flat_pct > high_pct)).sum())
        n_high = int((flat_pct > 90).sum())
        n_low = int((flat_pct < 10).sum())
        st.caption(
            f"**{n_extreme}** cell(s) currently outside [{low_pct:g}, "
            f"{high_pct:g}] pct  ·  {n_high} above 90  ·  {n_low} below 10."
        )

        # Inject minimal CSS to make the grid buttons feel like cards
        # (rounded, subtle border, uniform). Affects all st.button on
        # the page; we keep it neutral so the small effect on other
        # tabs' buttons is benign — slightly more padding and
        # rounded corners.
        st.markdown(
            """
            <style>
            div.stButton > button {
                border-radius: 8px;
                font-weight: 600;
                font-size: 1.05rem;
                padding: 0.55rem 0.4rem;
                line-height: 1.1;
                white-space: nowrap;
            }
            div.stButton > button:hover {
                border-color: #d62728;
                color: #d62728;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # ---- Card grid: absolute vol level (current), clickable.
        # Sits BELOW the percentile heatmap so the user can spot
        # extremes there first, then click the corresponding cell here
        # to drill in. Layout: header row of strike labels, then one
        # row per tenor. Each cell is a button labelled "X.X%" (1 dp).
        # Click sets session state so the dropdowns + chart below
        # auto-update.
        st.markdown("##### Absolute vol level (current)")
        st.caption(
            "**Click any card** to load that (tenor, strike) into the "
            "selectors below and update the history chart + distribution "
            "stats."
        )

        # Header row: empty corner cell + 5 strike labels
        hdr_cols = st.columns([1.0] + [1.5] * len(DELTA_STRIKES))
        hdr_cols[0].markdown("&nbsp;", unsafe_allow_html=True)
        for j, strike in enumerate(DELTA_STRIKES):
            hdr_cols[j + 1].markdown(
                f"<div style='text-align:center; font-weight:600; "
                f"color:#888; padding:4px 0;'>{strike}</div>",
                unsafe_allow_html=True,
            )

        # One row per tenor — left label + 5 cards
        for tenor in present_tenors:
            row_cols = st.columns([1.0] + [1.5] * len(DELTA_STRIKES))
            row_cols[0].markdown(
                f"<div style='text-align:right; font-weight:600; "
                f"padding-top:14px; padding-right:8px;'>{tenor}</div>",
                unsafe_allow_html=True,
            )
            for j, strike in enumerate(DELTA_STRIKES):
                v = levels_df.loc[tenor, strike]
                with row_cols[j + 1]:
                    if pd.isna(v):
                        st.button("—", key=f"heatcard_{tenor}_{strike}",
                                     disabled=True, use_container_width=True)
                    else:
                        # Vol × 100, 1 dp + "%" — matches the requested
                        # 7.5% display convention. Hover tooltip carries
                        # the percentile for extra context.
                        label = f"{v * 100:.1f}%"
                        pct_v = pct_df.loc[tenor, strike]
                        help_txt = (f"Tenor {tenor} · {strike} · "
                                          f"pct {pct_v:.0f}"
                                          if pd.notna(pct_v) else
                                          f"Tenor {tenor} · {strike}")

                        def _on_card_click(_t=tenor, _s=strike):
                            # Write directly to the dropdown widget keys.
                            # Streamlit's selectbox honours `index=` only
                            # on its very first render; once a widget key
                            # is set in session_state, it wins on every
                            # subsequent rerun. So to make a card-click
                            # propagate to the dropdowns we must update
                            # the dropdown keys themselves — not a
                            # separate "card selection" key.
                            st.session_state["heat_drill_tenor"] = _t
                            st.session_state["heat_drill_strike"] = _s

                        st.button(
                            label, key=f"heatcard_{tenor}_{strike}",
                            help=help_txt,
                            use_container_width=True,
                            on_click=_on_card_click,
                        )

        # ============================================================
        # Drill-down section — driven by two dropdowns. Card clicks
        # write directly to these widgets' session_state keys, so the
        # click + dropdown share a single source of truth.
        # ============================================================
        st.markdown("---")
        st.markdown("##### Drill-down — historical vol & distribution")

        # First-load defaults: ATM at the first available tenor.
        # On any rerun after a card click or a dropdown change, the
        # widget's own session_state key wins (Streamlit ignores the
        # `index=` argument once the key has been set).
        first_tenor = present_tenors[0] if present_tenors else None
        if "heat_drill_tenor" not in st.session_state and first_tenor:
            st.session_state["heat_drill_tenor"] = first_tenor
        if "heat_drill_strike" not in st.session_state:
            st.session_state["heat_drill_strike"] = "ATM"

        # Guard against stale state if the pair changed (different
        # available tenors). Reset to a valid value.
        if st.session_state.get("heat_drill_tenor") not in present_tenors:
            st.session_state["heat_drill_tenor"] = first_tenor

        d1, d2 = st.columns(2)
        with d1:
            sel_tenor = st.selectbox(
                "Tenor", present_tenors,
                key="heat_drill_tenor",
            )
        with d2:
            sel_strike = st.selectbox(
                "Strike", DELTA_STRIKES,
                key="heat_drill_strike",
            )

        # Pull the underlying series
        adf = abs_panels.get(sel_tenor)
        if adf is None or sel_strike not in adf.columns:
            st.warning(f"No data for {pair} {sel_tenor} {sel_strike}.")
        else:
            s_raw = adf[sel_strike].dropna()
            if s_raw.empty:
                st.warning("Series is empty.")
            else:
                # Window for chart x-axis (band edges use the same
                # window via reference_quantiles inside the helper).
                if lookback_days is None:
                    s_window = s_raw
                else:
                    cutoff = s_raw.index[-1] - pd.Timedelta(days=lookback_days)
                    s_window = s_raw[s_raw.index >= cutoff]

                pct_now = current_percentile(s_raw, lookback_days)
                cur_v = float(s_raw.iloc[-1])

                # Multiply by 100 for display — vol shown as % with 1 dp
                # everywhere. The internal data stays in decimal so
                # `current_percentile` and the like work unchanged.
                s_display = s_window * 100.0

                # ---- Chart: history with user-selected percentile band
                fig_hist = time_series_with_pct_band(
                    s_display,
                    lookback_days=lookback_days,
                    low_pct=low_pct, high_pct=high_pct,
                    title=(f"{pair}  ·  {sel_tenor} {sel_strike}  ·  "
                              f"current = {cur_v * 100:.1f}% "
                              f"(pct {pct_now:.1f})"),
                    yaxis_title="Vol (%)",
                    height=360,
                )
                # Override hover format to show 1 dp + "%" since the
                # values are now in percent rather than decimal.
                for trace in fig_hist.data:
                    if (trace.hovertemplate
                            and "y:.4f" in trace.hovertemplate):
                        trace.hovertemplate = (
                            "%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>"
                        )
                st.plotly_chart(fig_hist, use_container_width=True)

                # ---- Distribution stats below the chart. Mirrors the
                # Distribution tab's layout (4 metrics + histogram), but
                # with values rendered as percentages.
                st.markdown("##### Distribution")
                mean_v = (float(s_window.mean())
                              if not s_window.empty else float("nan"))
                std_v = (float(s_window.std())
                              if not s_window.empty else float("nan"))
                mt1, mt2, mt3, mt4 = st.columns(4)
                mt1.metric(
                    "Current",
                    f"{cur_v * 100:.1f}%" if pd.notna(cur_v) else "—",
                )
                mt2.metric(
                    "Percentile",
                    f"{pct_now:.1f}" if pd.notna(pct_now) else "—",
                )
                mt3.metric(
                    "Mean",
                    f"{mean_v * 100:.1f}%" if pd.notna(mean_v) else "—",
                )
                mt4.metric(
                    "Std",
                    f"{std_v * 100:.1f}%" if pd.notna(std_v) else "—",
                )

                # Histogram of values (in percent) with the current
                # value marked. `histogram_with_marker` doesn't enforce
                # any axis format, so showing the % series directly is
                # fine — bin edges and marker scale together.
                fig_hbar = histogram_with_marker(
                    s_window * 100.0,
                    float(cur_v * 100.0),
                    title=(f"{pair}  ·  {sel_tenor} {sel_strike}  ·  "
                              f"trailing {lookback_label}"),
                    height=320,
                )
                # Adjust hover / axis to read as percent values
                for trace in fig_hbar.data:
                    if trace.hovertemplate and "{x}" in trace.hovertemplate:
                        # Bin labels are already strings like "0.07 – 0.08"
                        # produced by histogram_with_marker — leave them
                        # alone, just retitle the x-axis below.
                        pass
                fig_hbar.update_xaxes(title_text="Vol (%)")
                st.plotly_chart(fig_hbar, use_container_width=True)




# ============================================================================
# TAB 3 — Alerts — vol-surface extremes for the selected pair
# ============================================================================
# Scans the (5 vol categories × 6 tenors) panel set already loaded in
# memory for the selected pair, flagging any series whose CURRENT value
# sits outside [low_pct, high_pct] of its trailing lookback window.
# Reuses the sidebar pair/lookback/band-percentile selections; no
# additional UI controls (deliberately — this is a focused alerts view,
# not a separate dashboard). For the multi-pair / multi-metric-class
# version see apps/alerts.py at the project level.
with tab_alerts:
    # Build the alert rows directly from `panels` — already in memory,
    # no need to re-load. Each panel entry is a (category, tenor) →
    # Series mapping for the currently selected pair.
    alert_rows: list[dict] = []
    for (cat, tenor), s in panels.items():
        if s is None or s.empty:
            continue
        cur = float(s.iloc[-1])
        pct = current_percentile(s, lookback_days)
        # Trailing-window stats — used for the table's reference cols
        # (p1, p99, min, max). When the user picks "Full" the window
        # is the whole series; otherwise it's the trailing lookback.
        if lookback_days is None:
            win = s.dropna()
        else:
            cutoff = s.index[-1] - pd.Timedelta(days=lookback_days)
            win = s[s.index >= cutoff].dropna()
        alert_rows.append({
            "Metric": VOL_CATEGORIES.get(cat, cat),
            "Tenor": tenor,
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

    if not alert_rows:
        st.info(f"No vol-surface data available for {pair}.")
    else:
        full_df = pd.DataFrame(alert_rows)
        # Derived classification — does this metric currently sit
        # outside the [low_pct, high_pct] band?
        full_df["Extremity"] = full_df["Percentile"].apply(extremity_distance)
        full_df["IsExtreme"] = full_df["Percentile"].apply(
            lambda p: pd.notna(p) and (p < low_pct or p > high_pct)
        )
        full_df["Direction"] = full_df["Percentile"].apply(
            lambda p: "↑ HIGH" if pd.notna(p) and p > high_pct
                       else ("↓ LOW" if pd.notna(p) and p < low_pct else "—")
        )

        n_total = len(full_df)
        n_extreme = int(full_df["IsExtreme"].sum())
        n_high = int((full_df["Percentile"] > high_pct).sum())
        n_low = int((full_df["Percentile"] < low_pct).sum())

        # Header strip — sets context that this view is scoped to the
        # currently-selected pair only. The summary metrics mirror the
        # standalone alerts app but for one pair instead of many.
        st.markdown(
            f"**{pair}** "
            + (f"  ·  *{prefer}*" if asia_em else "")
            + f"  ·  Band: **{low_pct:g}–{high_pct:g} pct** over "
              f"**{lookback_label}**"
            + f"  ·  As of: **{full_df['AsOf'].max()}**"
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Metrics scanned", f"{n_total}")
        m2.metric("🚨 Currently extreme", f"{n_extreme}")
        m3.metric(f"↑ HIGH (> {high_pct:g} pct)", f"{n_high}")
        m4.metric(f"↓ LOW (< {low_pct:g} pct)", f"{n_low}")

        st.markdown("---")

        # Conditional formatting — red for HIGH, blue for LOW, soft
        # orange/blue for "watch list" (>90 / <10 but not yet flagged).
        def _style_pct(v):
            if pd.isna(v):
                return ""
            if v > high_pct:
                return ("background-color: rgba(214, 39, 40, 0.30); "
                          "font-weight: bold;")
            if v < low_pct:
                return ("background-color: rgba(31, 119, 180, 0.30); "
                          "font-weight: bold;")
            if v > 90:
                return "background-color: rgba(255, 127, 14, 0.15);"
            if v < 10:
                return "background-color: rgba(31, 119, 180, 0.10);"
            return ""

        display_cols = ["Metric", "Tenor", "Direction",
                          "Current", "Percentile",
                          "p1", "p99", "Min", "Max", "AsOf"]

        def _style_alerts(df):
            return (df.style
                       .format({
                           "Current": "{:.4f}", "Percentile": "{:.1f}",
                           "p1": "{:.4f}", "p99": "{:.4f}",
                           "Min": "{:.4f}", "Max": "{:.4f}",
                       })
                       .map(_style_pct, subset=["Percentile"]))

        # --- Sub-tabs: Extremes (focused) | All scanned (overview) ---
        sub_ex, sub_all = st.tabs(
            [f"🚨 Extremes ({n_extreme})",
             f"All scanned ({n_total})"]
        )

        with sub_ex:
            sub = (full_df[full_df["IsExtreme"]]
                       .sort_values("Extremity", ascending=False)
                       .reset_index(drop=True))
            if sub.empty:
                st.success(
                    f"✓ No vol-surface metrics for {pair} currently "
                    f"outside [{low_pct:g}, {high_pct:g}] pct over the "
                    f"{lookback_label} lookback."
                )
            else:
                st.dataframe(
                    _style_alerts(sub[display_cols]),
                    use_container_width=True, hide_index=True, height=320,
                )

                # Drill-down — pick one flagged metric to inspect.
                # Three charts side-by-side: time series with quantile
                # bands, distribution histogram with current marker, and
                # below them the point-in-time percentile path. Mirrors
                # the layout of the standalone alerts app's drill-down
                # so users see a consistent experience between the two.
                st.markdown("---")
                st.markdown("##### 🔍 Inspect a flagged metric")
                sub["_label"] = (
                    sub["Metric"] + "  ·  " + sub["Tenor"]
                    + "  →  " + sub["Direction"]
                    + " (pct " + sub["Percentile"].round(1).astype(str) + ")"
                )
                choice = st.selectbox(
                    "Metric", sub["_label"].tolist(),
                    key="alerts_drill_select",
                )
                if choice:
                    row = sub[sub["_label"] == choice].iloc[0]
                    cat = next(
                        (c for c, lbl in VOL_CATEGORIES.items()
                         if lbl == row["Metric"]),
                        None,
                    )
                    s = panels.get((cat, row["Tenor"]), pd.Series(dtype=float))
                    if s is None or s.empty:
                        st.warning("Underlying series not available.")
                    else:
                        # Window for chart display (full series if Full,
                        # else trailing lookback) and percentile path
                        if lookback_days is None:
                            s_cut = s.dropna()
                        else:
                            cutoff = s.index[-1] - pd.Timedelta(days=lookback_days)
                            s_cut = s[s.index >= cutoff].dropna()
                        quantiles = reference_quantiles(
                            s, lookback_days=lookback_days)
                        pct_path = expanding_percentile(
                            s, lookback_days=lookback_days)
                        if lookback_days is None:
                            pct_path_cut = pct_path
                        else:
                            pct_path_cut = pct_path[pct_path.index >= cutoff]

                        ylab = ("Vol level (%)" if cat == "VOL_ATM"
                                  else "Vol points")
                        cc1, cc2 = st.columns([3, 2])
                        with cc1:
                            fig_v = time_series_with_quantile_bands(
                                s_cut, quantiles,
                                title=(f"{pair} · {row['Metric']} · "
                                          f"{row['Tenor']}"),
                                yaxis_title=ylab, height=340,
                            )
                            st.plotly_chart(fig_v, use_container_width=True)
                        with cc2:
                            fig_h = histogram_with_marker(
                                s_cut, float(s.iloc[-1]),
                                title=f"Distribution · {lookback_label}",
                                height=340,
                            )
                            st.plotly_chart(fig_h, use_container_width=True)

                        fig_p = percentile_path_chart(
                            pct_path_cut,
                            title="Point-in-time percentile path",
                            height=200,
                        )
                        st.plotly_chart(fig_p, use_container_width=True)

        with sub_all:
            # Full table — keep the same vol-category × tenor ordering
            # used elsewhere in the app so it's easy to scan visually.
            sub = full_df.sort_values(["_cat_key", "_tenor_key"]).reset_index(
                drop=True)
            st.dataframe(
                _style_alerts(sub[display_cols]),
                use_container_width=True, hide_index=True, height=520,
            )

        st.caption(
            f"Scope: vol surface only (ATM · 25Δ RR · 25Δ BF · 10Δ RR · "
            f"10Δ BF) for **{pair}**. For a cross-pair scan that also "
            f"covers forward points and calendar spreads, see the "
            f"standalone `apps/alerts.py` app."
        )





# ============================================================================
# TAB 4 — 3m Vol Screen — Goldman 'Best 3m Vol Screen' (image 6)
# ============================================================================
# SCOPE: Multi-pair scan — ignores the sidebar pair selector for the
# table/scatter (shows every USD pair we have 3m ATM vol for). The
# sidebar pair is used only as the *default drill-down* below.
#
# Layout:
#   1. Summary metrics row (pairs scanned, cheap vs rich counts)
#   2. Sortable table: Cross | 3m Implied | 3m Realized | Diff |
#      2y Low | 2y High | Percentile. Diff cells coloured green
#      (cheap implied vs realized → candidate BUY) / red (rich →
#      candidate SELL). Percentile cells heat-mapped 0..100.
#   3. Scatter: percentile of current implied (x) vs Diff (y) —
#      same framing as Goldman's 'Entry Point vs Richness' chart.
#   4. Drill-down: pair dropdown (defaults to sidebar pair) → time
#      series of 3m Implied vs 3m Realized over the lookback window.
with tab_vol3m:
    import plotly.express as px
    import plotly.graph_objects as go

    from core.screens import (scan_3m_vol, scan_3m_vol_history,
                                ASIA_EM_PAIRS)

    @st.cache_data(show_spinner="Scanning 3m vols…")
    def _scan_3m_vol_cached(folder: str,
                            pairs: tuple[str, ...],
                            prefer_em: str) -> pd.DataFrame:
        return scan_3m_vol(folder, pairs, lookback_years=2,
                            prefer_em=prefer_em)

    @st.cache_data(show_spinner="Loading vol history…")
    def _scan_3m_vol_history_cached(folder: str,
                                     pair: str,
                                     prefer_em: str) -> pd.DataFrame:
        return scan_3m_vol_history(folder, pair, prefer_em=prefer_em)

    # All pairs with 3m ATM vol data. `list_available_pairs` is the
    # same helper the sidebar uses to populate the pair selector.
    all_vol_pairs = sorted(list_available_pairs(folder, "VOL_ATM"))
    if not all_vol_pairs:
        st.warning("No VOL_ATM data available in `_index.csv`.")
    else:
        # Use the sidebar's onshore/offshore choice as the EM default
        # (it's only set when an Asia EM pair is selected).
        em_pref = prefer if asia_em else "offshore"
        df_scan = _scan_3m_vol_cached(folder, tuple(all_vol_pairs), em_pref)

        if df_scan.empty:
            st.warning("Couldn't build the screen — no usable data.")
        else:
            # ---- Header
            st.markdown(
                f"**3m Vol Screen**  ·  Implied vs Realized across "
                f"**{len(df_scan)}** pair(s)  ·  EM preference: "
                f"**{em_pref}**"
            )

            n_cheap = int((df_scan["Diff"] < 0).sum())
            n_rich = int((df_scan["Diff"] > 0).sum())
            n_low_pct = int((df_scan["Percentile"] < 25).sum())
            n_high_pct = int((df_scan["Percentile"] > 75).sum())
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Pairs scanned", f"{len(df_scan)}")
            m2.metric("Implied < Realized", f"{n_cheap}",
                          help="Candidates to BUY vol")
            m3.metric("Implied > Realized", f"{n_rich}",
                          help="Candidates to SELL vol")
            m4.metric("Pct < 25  /  > 75",
                          f"{n_low_pct}  /  {n_high_pct}")

            # ---- Table
            st.markdown("##### Best 3m Vol Screen")
            st.caption(
                "Diff = 3m Implied − 3m Realized (% vol points). "
                "**Green** Diff = implied is cheap vs realized → buy "
                "candidate; **red** = implied is rich → sell candidate. "
                "Percentile is current implied vs trailing-2y history "
                "of implied. Sorted ascending by Diff (biggest "
                "buy candidates on top)."
            )

            def _style_diff(v):
                """Green for negative (cheap), red for positive (rich).
                Saturation scales with |v| capped at 5 vol points."""
                if pd.isna(v):
                    return ""
                t = min(1.0, abs(float(v)) / 5.0)
                if v < 0:
                    g = int(220 + (160 - 220) * t)
                    return (f"background-color: rgb(180, {g+40}, 180); "
                            "color: #1a1a1a; text-align: right;")
                else:
                    r = int(255 + (220 - 255) * t)
                    return (f"background-color: rgb({r}, {180-int(t*40)},"
                            f" {180-int(t*40)}); color: #1a1a1a; "
                            "text-align: right;")

            def _style_pct(v):
                """Red→Yellow→Green on fixed 0..100 scale."""
                if pd.isna(v):
                    return "text-align: center;"
                t = max(0.0, min(1.0, float(v) / 100.0))
                if t < 0.5:
                    x = t / 0.5
                    r = int(220 + (255 - 220) * x)
                    g = int(80 + (220 - 80) * x)
                    b = int(80 + (130 - 80) * x)
                else:
                    x = (t - 0.5) / 0.5
                    r = int(255 + (90 - 255) * x)
                    g = int(220 + (180 - 220) * x)
                    b = int(130 + (90 - 130) * x)
                return (f"background-color: rgb({r},{g},{b}); "
                        "color: #1a1a1a; text-align: center; "
                        "font-weight: 600;")

            styled = (df_scan.style
                        .format({
                            "3m Implied": "{:.1f}",
                            "3m Realized": "{:.1f}",
                            "Diff": "{:+.1f}",
                            "2y Low": "{:.1f}",
                            "2y High": "{:.1f}",
                            "Percentile": "{:.0f}",
                        }, na_rep="—")
                        .map(_style_diff, subset=["Diff"])
                        .map(_style_pct, subset=["Percentile"])
                        .set_table_styles([
                            {"selector": "th",
                                "props": [("text-align", "center"),
                                          ("font-weight", "600")]},
                        ]))
            st.dataframe(styled, use_container_width=True,
                            hide_index=True, height=min(560, 38 * (len(df_scan) + 1)))

            # ---- Scatter: Entry Point vs Richness
            st.markdown("---")
            st.markdown("##### Entry Point vs Richness")
            st.caption(
                "X-axis: percentile of current 3m implied in its 2y "
                "history (low = cheap entry). Y-axis: Implied − "
                "Realized (positive = implied trading rich vs delivered, "
                "expensive to buy). **Lower-left** corner is the sweet "
                "spot for buying vol; **upper-right** for selling vol."
            )
            df_scatter = df_scan.copy().dropna(
                subset=["Percentile", "Diff"])
            df_scatter["Region"] = df_scatter["Cross"].map(
                lambda c: "EM Asia" if c in ASIA_EM_PAIRS else "G10"
            )
            fig_sc = px.scatter(
                df_scatter,
                x="Percentile", y="Diff",
                color="Region", text="Cross",
                hover_data={
                    "3m Implied": ":.1f",
                    "3m Realized": ":.1f",
                    "Diff": ":+.1f",
                    "Percentile": ":.0f",
                    "Region": False,
                },
                color_discrete_map={"G10": "#1f77b4",
                                       "EM Asia": "#2ca02c"},
                height=460,
            )
            fig_sc.update_traces(textposition="top center",
                                    marker=dict(size=11,
                                                line=dict(width=0.5,
                                                            color="#333")))
            fig_sc.add_hline(y=0, line_dash="dash",
                                line_color="#888", opacity=0.5)
            fig_sc.add_vline(x=50, line_dash="dot",
                                line_color="#aaa", opacity=0.4)
            fig_sc.update_layout(
                xaxis_title="Percentile of current 3m implied (2y)",
                yaxis_title="Implied − Realized (vol pts)",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom",
                                y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

            # ---- Drill-down: one pair, history
            st.markdown("---")
            st.markdown(
                "##### Drill-down — 3m Implied vs Realized history")

            scan_pairs = df_scan["Cross"].tolist()
            default_idx = (scan_pairs.index(pair)
                              if pair in scan_pairs else 0)
            drill_pair = st.selectbox(
                "Pair", scan_pairs, index=default_idx,
                key="vol3m_drill_pair",
            )

            hist = _scan_3m_vol_history_cached(
                folder, drill_pair, em_pref)
            if hist.empty:
                st.warning(f"No history for {drill_pair}.")
            else:
                # Apply sidebar lookback for chart x-range
                if lookback_days is None:
                    hist_disp = hist
                else:
                    cutoff = (hist.index[-1]
                                 - pd.Timedelta(days=lookback_days))
                    hist_disp = hist[hist.index >= cutoff]

                cur_imp = hist["3m Implied"].dropna().iloc[-1]
                cur_rv_series = hist["3m Realized"].dropna()
                cur_rv = (cur_rv_series.iloc[-1]
                              if not cur_rv_series.empty else float("nan"))

                fig_h = go.Figure()
                fig_h.add_trace(go.Scatter(
                    x=hist_disp.index,
                    y=hist_disp["3m Implied"],
                    mode="lines", name="3m Implied",
                    line=dict(color="#d62728", width=2),
                    hovertemplate=("%{x|%Y-%m-%d}<br>"
                                       "Implied: %{y:.2f}%<extra></extra>"),
                ))
                fig_h.add_trace(go.Scatter(
                    x=hist_disp.index,
                    y=hist_disp["3m Realized"],
                    mode="lines", name="3m Realized (63-day)",
                    line=dict(color="#1f77b4", width=2),
                    hovertemplate=("%{x|%Y-%m-%d}<br>"
                                       "Realized: %{y:.2f}%<extra></extra>"),
                ))
                title = (f"{drill_pair}  ·  current implied "
                            f"{cur_imp:.1f}%  ·  current realized "
                            f"{cur_rv:.1f}%  ·  trailing {lookback_label}")
                fig_h.update_layout(
                    title=title,
                    xaxis_title="", yaxis_title="Vol (%)",
                    height=380,
                    margin=dict(l=10, r=10, t=50, b=10),
                    legend=dict(orientation="h", yanchor="bottom",
                                    y=1.02, xanchor="right", x=1),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_h, use_container_width=True)

            # ---- Download
            st.markdown("---")
            csv_bytes = df_scan.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇  Download screen (CSV)",
                data=csv_bytes,
                file_name=f"3m_vol_screen_{pd.Timestamp.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="vol3m_download",
            )




# ============================================================================
# TAB 5 — Static Carry — Goldman 'Best Carry Screen' (image 5)
# ============================================================================
# SCOPE: Multi-pair scan. The structure is an ATMF/ATMS put spread
# (e.g. buy 1 ATMF put, sell 1 ATMS put) — for a "Long" trade we'd
# reverse to a call spread, but the static-carry ratio is symmetric.
#
# Static carry = |F - S| / |Put_ATMF_FV - Put_ATMS_FV|, i.e. the
# payoff at expiry assuming spot rolls to spot, divided by the premium
# paid. Larger ratio = better risk-reward. Direction follows fwd vs
# spot (which side is the high-yielder).
#
# Caveat vs Goldman: we use ATM vol only; they use the full smile.
# Expect ratio differences of 0.1-0.4x for the same date. Ranking and
# direction should agree closely though.
with tab_carry:
    import plotly.express as px
    import plotly.graph_objects as go

    from core.screens import (scan_static_carry, scan_static_carry_history,
                                ASIA_EM_PAIRS)

    @st.cache_data(show_spinner="Scanning static carry…")
    def _scan_carry_cached(folder: str,
                            pairs: tuple[str, ...],
                            tenor: str,
                            prefer_em: str) -> pd.DataFrame:
        return scan_static_carry(folder, pairs, tenor=tenor,
                                  prefer_em=prefer_em)

    @st.cache_data(show_spinner="Loading carry history…")
    def _carry_history_cached(folder: str,
                               pair: str,
                               tenor: str,
                               prefer_em: str) -> pd.DataFrame:
        return scan_static_carry_history(folder, pair, tenor=tenor,
                                          prefer_em=prefer_em)

    # Tenor selector — Goldman defaults to 3m. We expose 1M/3M/6M so
    # users can compare short-dated vs longer-dated carry.
    cc1, cc2 = st.columns([1, 3])
    with cc1:
        carry_tenor = st.selectbox(
            "Tenor", ["1M", "3M", "6M"], index=1, key="carry_tenor",
        )

    all_pairs = sorted(list_available_pairs(folder, "VOL_ATM"))
    if not all_pairs:
        st.warning("No VOL_ATM data in `_index.csv`.")
    else:
        em_pref = prefer if asia_em else "offshore"
        df_scan = _scan_carry_cached(folder, tuple(all_pairs),
                                       carry_tenor, em_pref)

        if df_scan.empty:
            st.warning("Couldn't build the screen — no usable data.")
        else:
            st.markdown(
                f"**Static Carry Screen**  ·  {carry_tenor} ATMF/ATMS "
                f"put-spread  ·  **{len(df_scan)}** pair(s)  ·  EM "
                f"preference: **{em_pref}**"
            )
            st.caption(
                "Static Carry = |F − S| / |Put_ATMF − Put_ATMS| "
                "(forward values). Higher = better static "
                "risk-reward. Direction follows fwd vs spot. 1m Return "
                "is the smoothed return of the **non-USD currency** "
                "(positive = appreciated). Goldman uses the full vol "
                "smile; we use ATM only, so ratios may differ by "
                "~0.1-0.4x but rankings should align."
            )

            n_long = int((df_scan["Direction"] == "Long").sum())
            n_short = int((df_scan["Direction"] == "Short").sum())
            top_ratio = (float(df_scan["Static Carry"].iloc[0])
                            if not df_scan.empty else float("nan"))
            top_pair = (df_scan["Pair"].iloc[0]
                            if not df_scan.empty else "—")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Pairs scanned", f"{len(df_scan)}")
            m2.metric("Long carry trades", f"{n_long}")
            m3.metric("Short carry trades", f"{n_short}")
            m4.metric("Top ratio", f"{top_ratio:.2f}x", top_pair)

            # ---- Table
            def _style_carry(v):
                """Green-saturating for high ratios; red for low."""
                if pd.isna(v):
                    return ""
                t = max(0.0, min(1.0, (float(v) - 0.5) / 2.5))
                if t > 0.5:
                    x = (t - 0.5) / 0.5
                    r = int(220 + (40 - 220) * x)
                    g = int(255 + (160 - 255) * x)
                    b = int(220 + (40 - 220) * x)
                else:
                    x = t / 0.5
                    r = int(255 + (220 - 255) * x)
                    g = int(200 + (255 - 200) * x)
                    b = int(200 + (220 - 200) * x)
                return (f"background-color: rgb({r},{g},{b}); "
                        "color: #1a1a1a; text-align: right; "
                        "font-weight: 600;")

            def _style_dir(v):
                if v == "Long":
                    return ("background-color: rgba(31, 119, 180, 0.20); "
                              "text-align: center; font-weight: 600;")
                elif v == "Short":
                    return ("background-color: rgba(214, 39, 40, 0.20); "
                              "text-align: center; font-weight: 600;")
                return "text-align: center;"

            def _style_ret(v):
                if pd.isna(v):
                    return "text-align: right;"
                if v > 0:
                    return "color: #2ca02c; text-align: right;"
                if v < 0:
                    return "color: #d62728; text-align: right;"
                return "text-align: right;"

            # Per-pair Spot/Forward format — JPY/IDR/KRW/HUF need
            # different precision than EURUSD/GBPUSD
            def _fmt_px(v, pair):
                if pd.isna(v):
                    return "—"
                if pair in ("USDJPY", "USDKRW", "USDHUF"):
                    return f"{v:,.2f}"
                if pair == "USDIDR":
                    return f"{v:,.0f}"
                return f"{v:.4f}"

            df_display = df_scan.copy()
            df_display["Spot"] = df_display.apply(
                lambda r: _fmt_px(r["Spot"], r["Pair"]), axis=1)
            df_display["Forward"] = df_display.apply(
                lambda r: _fmt_px(r["Forward"], r["Pair"]), axis=1)

            styled = (df_display.style
                        .format({
                            "Implied Vol": "{:.1f}",
                            "Static Carry": "{:.2f}x",
                            "1m Return": "{:+.1f}%",
                        }, na_rep="—")
                        .map(_style_carry, subset=["Static Carry"])
                        .map(_style_dir, subset=["Direction"])
                        .map(_style_ret, subset=["1m Return"])
                        .set_table_styles([
                            {"selector": "th",
                                "props": [("text-align", "center"),
                                          ("font-weight", "600")]},
                        ]))
            st.dataframe(styled, use_container_width=True,
                            hide_index=True,
                            height=min(560, 38 * (len(df_scan) + 1)))

            # ---- Scatter: carry vs spot return (Goldman's right-hand
            # panel of the screen)
            st.markdown("---")
            st.markdown("##### Static Carry vs 1m Spot Return")
            st.caption(
                "X-axis: static-carry ratio. Y-axis: smoothed 1m return "
                "of the non-USD currency. **Upper-right** = high static "
                "payoff plus recent appreciation (momentum + carry "
                "aligned). **Lower-right** = high carry but currency "
                "weakened recently (carry trade with adverse spot "
                "momentum)."
            )
            df_sc = df_scan.copy().dropna(
                subset=["Static Carry", "1m Return"])
            df_sc["Region"] = df_sc["Pair"].map(
                lambda c: "EM Asia" if c in ASIA_EM_PAIRS else "G10"
            )
            fig_sc = px.scatter(
                df_sc,
                x="Static Carry", y="1m Return",
                color="Region", text="Pair", symbol="Direction",
                hover_data={
                    "Implied Vol": ":.1f",
                    "Static Carry": ":.2f",
                    "1m Return": ":+.2f",
                    "Direction": True,
                    "Region": False,
                },
                color_discrete_map={"G10": "#1f77b4",
                                       "EM Asia": "#2ca02c"},
                symbol_map={"Long": "circle",
                              "Short": "triangle-down"},
                height=460,
            )
            fig_sc.update_traces(
                textposition="top center",
                marker=dict(size=11,
                              line=dict(width=0.5, color="#333")))
            fig_sc.add_hline(y=0, line_dash="dash",
                                line_color="#888", opacity=0.5)
            fig_sc.update_layout(
                xaxis_title="Static Carry ratio (x)",
                yaxis_title="1m Spot Return (%, non-USD ccy)",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom",
                                y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

            # ---- Drill-down: history of carry ratio for one pair
            st.markdown("---")
            st.markdown(
                "##### Drill-down — Static Carry history")

            scan_pairs = df_scan["Pair"].tolist()
            default_idx = (scan_pairs.index(pair)
                              if pair in scan_pairs else 0)
            drill_pair = st.selectbox(
                "Pair", scan_pairs, index=default_idx,
                key="carry_drill_pair",
            )

            hist = _carry_history_cached(folder, drill_pair,
                                            carry_tenor, em_pref)
            if hist.empty:
                st.warning(f"No carry history for {drill_pair}.")
            else:
                if lookback_days is None:
                    hist_disp = hist
                else:
                    cutoff = (hist.index[-1]
                                 - pd.Timedelta(days=lookback_days))
                    hist_disp = hist[hist.index >= cutoff]

                ratio_s = hist["Static Carry"].dropna()
                cur_ratio = (float(ratio_s.iloc[-1])
                                if not ratio_s.empty else float("nan"))
                pct = (float((ratio_s <= cur_ratio).sum())
                          / float(len(ratio_s)) * 100.0
                          if not ratio_s.empty else float("nan"))

                fig_h = go.Figure()
                fig_h.add_trace(go.Scatter(
                    x=hist_disp.index,
                    y=hist_disp["Static Carry"],
                    mode="lines", name="Static Carry",
                    line=dict(color="#d62728", width=2),
                    hovertemplate=("%{x|%Y-%m-%d}<br>"
                                       "Carry: %{y:.2f}x<extra></extra>"),
                ))
                # Reference: mean over the displayed window
                if not hist_disp["Static Carry"].dropna().empty:
                    mean_ratio = float(
                        hist_disp["Static Carry"].dropna().mean())
                    fig_h.add_hline(
                        y=mean_ratio, line_dash="dash",
                        line_color="#888",
                        annotation_text=f"mean {mean_ratio:.2f}x",
                        annotation_position="bottom right",
                    )
                title = (f"{drill_pair} {carry_tenor}  ·  current "
                            f"{cur_ratio:.2f}x  ·  pct {pct:.0f}  ·  "
                            f"trailing {lookback_label}")
                fig_h.update_layout(
                    title=title,
                    xaxis_title="", yaxis_title="Static Carry (x)",
                    height=380,
                    margin=dict(l=10, r=10, t=50, b=10),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_h, use_container_width=True)

            # ---- Download
            st.markdown("---")
            csv_bytes = df_scan.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇  Download screen (CSV)",
                data=csv_bytes,
                file_name=(f"static_carry_{carry_tenor.lower()}_"
                              f"{pd.Timestamp.today().strftime('%Y%m%d')}"
                              ".csv"),
                mime="text/csv",
                key="carry_download",
            )




# ============================================================================
# TAB 6 — DNT — Goldman 'Best DNT Screen' (image 3)
# ============================================================================
# SCOPE: Multi-pair scan. For each pair, construct a symmetric barrier
# corridor around current spot (half-width = 1.2 × max excursion over
# lookback window) and compute the THEORETICAL Black-Scholes survival
# probability under continuous monitoring.
#
# Important caveat: Goldman backs out IMPLIED survival probability from
# market DNT QUOTES. We compute the THEORETICAL probability under flat-
# vol GBM with continuous monitoring. Market DNTs trade at a large
# spread, so expect their numbers to be 3-5x higher than ours. Use this
# screen to find pairs where the theoretical probability is LOW (range
# is tight relative to vol) — those are the cleanest candidates to BUY
# DNTs vs the market, modulo the market spread.
with tab_dnt:
    import plotly.express as px
    import plotly.graph_objects as go

    from core.screens import (scan_dnt, scan_dnt_history, ASIA_EM_PAIRS)

    @st.cache_data(show_spinner="Scanning DNT corridors…")
    def _scan_dnt_cached(folder: str,
                          pairs: tuple[str, ...],
                          tenor: str,
                          lookback: int,
                          widen_pct: float,
                          prefer_em: str) -> pd.DataFrame:
        return scan_dnt(folder, pairs, tenor=tenor,
                          lookback_days=lookback,
                          widen_pct=widen_pct,
                          prefer_em=prefer_em)

    @st.cache_data(show_spinner="Loading DNT history…")
    def _dnt_history_cached(folder: str,
                              pair: str,
                              tenor: str,
                              lookback: int,
                              widen_pct: float,
                              prefer_em: str) -> pd.DataFrame:
        return scan_dnt_history(folder, pair, tenor=tenor,
                                  lookback_days=lookback,
                                  widen_pct=widen_pct,
                                  prefer_em=prefer_em)

    # Controls row — three knobs: tenor, range lookback, widening %
    cc1, cc2, cc3 = st.columns([1, 1, 1])
    with cc1:
        dnt_tenor = st.selectbox(
            "Tenor", ["1M", "3M", "6M"], index=1, key="dnt_tenor",
        )
    with cc2:
        dnt_lookback = st.selectbox(
            "Range lookback (BDays)",
            options=[21, 42, 63, 126, 252],
            index=2,
            help="Window for computing max(|S_t − S_now|/S_now). "
                  "Default 63 ≈ 3m, matching the DNT tenor.",
            key="dnt_lookback",
        )
    with cc3:
        dnt_widen = st.number_input(
            "Widening %", min_value=0.0, max_value=100.0,
            value=20.0, step=5.0,
            help="Half-width = (1 + widening/100) × max recent excursion. "
                  "Goldman uses 20%.",
            key="dnt_widen",
        )

    all_pairs = sorted(list_available_pairs(folder, "VOL_ATM"))
    if not all_pairs:
        st.warning("No VOL_ATM data in `_index.csv`.")
    else:
        em_pref = prefer if asia_em else "offshore"
        df_scan = _scan_dnt_cached(folder, tuple(all_pairs),
                                     dnt_tenor, dnt_lookback,
                                     float(dnt_widen), em_pref)

        if df_scan.empty:
            st.warning("Couldn't build the screen — no usable data.")
        else:
            st.markdown(
                f"**DNT Screen**  ·  {dnt_tenor} double-no-touch  ·  "
                f"range from {dnt_lookback}-BDay history × "
                f"{dnt_widen:g}% widening  ·  **{len(df_scan)}** pair(s)"
            )
            st.warning(
                "⚠️ Survival Prob is THEORETICAL Black-Scholes (continuous "
                "monitoring, flat ATM vol, no drift). Goldman's published "
                "screen backs out IMPLIED probability from market DNT "
                "quotes — those will be 3-5× higher because DNTs trade at "
                "wide spreads. Use this for ranking which corridors are "
                "tight relative to vol, not as a direct price comparison."
            )

            mean_p = float(df_scan["Survival Prob"].mean())
            n_low = int((df_scan["Survival Prob"] < 10).sum())
            n_high = int((df_scan["Survival Prob"] > 50).sum())
            tightest = (df_scan.iloc[0]["Cross"]
                         if not df_scan.empty else "—")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Pairs scanned", f"{len(df_scan)}")
            m2.metric("Mean survival prob", f"{mean_p:.1f}%")
            m3.metric("Prob < 10%", f"{n_low}",
                          help="Tightest corridors (most likely to KO)")
            m4.metric("Tightest", tightest)

            # ---- Table
            st.markdown("##### Best DNT Screen")

            def _style_prob(v):
                """Green for low prob (tight corridor = KO likely);
                red for high prob (wide corridor = unlikely to KO)."""
                if pd.isna(v):
                    return ""
                t = max(0.0, min(1.0, float(v) / 100.0))
                if t < 0.5:
                    x = t / 0.5
                    r = int(40 + (255 - 40) * x)
                    g = int(160 + (220 - 160) * x)
                    b = int(40 + (130 - 40) * x)
                else:
                    x = (t - 0.5) / 0.5
                    r = int(255 + (220 - 255) * x)
                    g = int(220 + (80 - 220) * x)
                    b = int(130 + (80 - 130) * x)
                return (f"background-color: rgb({r},{g},{b}); "
                          "color: #1a1a1a; text-align: center; "
                          "font-weight: 600;")

            def _fmt_px_dnt(v, pair):
                if pd.isna(v):
                    return "—"
                if pair in ("USDJPY", "USDKRW"):
                    return f"{v:,.2f}"
                if pair == "USDIDR":
                    return f"{v:,.0f}"
                return f"{v:.4f}"

            df_display = df_scan.copy()
            for col in ("Spot", "Lower KO", "Upper KO"):
                df_display[col] = df_display.apply(
                    lambda r, c=col: _fmt_px_dnt(r[c], r["Cross"]), axis=1)

            styled = (df_display.style
                        .format({
                            "Range": "{:.2f}%",
                            "Implied Vol": "{:.1f}",
                            "Realized Vol": "{:.1f}",
                            "Survival Prob": "{:.1f}%",
                        }, na_rep="—")
                        .map(_style_prob, subset=["Survival Prob"])
                        .set_table_styles([
                            {"selector": "th",
                                "props": [("text-align", "center"),
                                          ("font-weight", "600")]},
                        ]))
            st.dataframe(styled, use_container_width=True,
                            hide_index=True,
                            height=min(560, 38 * (len(df_scan) + 1)))

            # ---- Scatter: vol richness vs survival prob (Goldman's
            # 'DNT Probability vs Vol Richness' panel)
            st.markdown("---")
            st.markdown("##### Survival Prob vs Vol Richness")
            st.caption(
                "X-axis: theoretical survival probability. Y-axis: implied "
                "minus realized vol (positive = vol trading rich). "
                "**Upper-left** = low DNT prob AND rich vol — best "
                "candidates to SELL the DNT. **Lower-right** = wide "
                "corridor and cheap vol — least interesting."
            )
            df_sc = df_scan.copy()
            df_sc["Vol Richness"] = (df_sc["Implied Vol"]
                                       - df_sc["Realized Vol"])
            df_sc = df_sc.dropna(subset=["Survival Prob", "Vol Richness"])
            df_sc["Region"] = df_sc["Cross"].map(
                lambda c: "EM Asia" if c in ASIA_EM_PAIRS else "G10"
            )
            fig_sc = px.scatter(
                df_sc,
                x="Survival Prob", y="Vol Richness",
                color="Region", text="Cross",
                hover_data={
                    "Range": ":.2f",
                    "Implied Vol": ":.1f",
                    "Realized Vol": ":.1f",
                    "Survival Prob": ":.1f",
                    "Region": False,
                },
                color_discrete_map={"G10": "#1f77b4",
                                       "EM Asia": "#2ca02c"},
                height=460,
            )
            fig_sc.update_traces(textposition="top center",
                                    marker=dict(size=11,
                                                  line=dict(width=0.5,
                                                              color="#333")))
            fig_sc.add_hline(y=0, line_dash="dash",
                                line_color="#888", opacity=0.5)
            fig_sc.update_layout(
                xaxis_title="Survival Probability (%)",
                yaxis_title="Implied − Realized Vol (pts)",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom",
                                y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

            # ---- Drill-down: pair → spot + KO levels over time
            st.markdown("---")
            st.markdown("##### Drill-down — Spot and KO levels")

            scan_pairs = df_scan["Cross"].tolist()
            default_idx = (scan_pairs.index(pair)
                              if pair in scan_pairs else 0)
            drill_pair = st.selectbox(
                "Pair", scan_pairs, index=default_idx,
                key="dnt_drill_pair",
            )

            hist = _dnt_history_cached(folder, drill_pair, dnt_tenor,
                                          dnt_lookback, float(dnt_widen),
                                          em_pref)
            if hist.empty:
                st.warning(f"No spot history for {drill_pair}.")
            else:
                if lookback_days is None:
                    hist_disp = hist
                else:
                    cutoff = (hist.index[-1]
                                 - pd.Timedelta(days=lookback_days))
                    hist_disp = hist[hist.index >= cutoff]

                row = df_scan[df_scan["Cross"] == drill_pair].iloc[0]
                cur_prob = float(row["Survival Prob"])

                fig_h = go.Figure()
                fig_h.add_trace(go.Scatter(
                    x=hist_disp.index, y=hist_disp["Upper KO"],
                    mode="lines", name="Upper KO",
                    line=dict(color="#d62728", width=1.5, dash="dash"),
                    hovertemplate="%{x|%Y-%m-%d}<br>U: %{y:.4f}<extra></extra>",
                ))
                fig_h.add_trace(go.Scatter(
                    x=hist_disp.index, y=hist_disp["Lower KO"],
                    mode="lines", name="Lower KO",
                    line=dict(color="#d62728", width=1.5, dash="dash"),
                    fill="tonexty",
                    fillcolor="rgba(214, 39, 40, 0.08)",
                    hovertemplate="%{x|%Y-%m-%d}<br>L: %{y:.4f}<extra></extra>",
                ))
                fig_h.add_trace(go.Scatter(
                    x=hist_disp.index, y=hist_disp["Spot"],
                    mode="lines", name="Spot",
                    line=dict(color="#1f77b4", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>S: %{y:.4f}<extra></extra>",
                ))
                title = (f"{drill_pair} {dnt_tenor} DNT  ·  current "
                            f"survival prob {cur_prob:.1f}%  ·  "
                            f"trailing {lookback_label}")
                fig_h.update_layout(
                    title=title,
                    xaxis_title="", yaxis_title="Level",
                    height=400,
                    margin=dict(l=10, r=10, t=50, b=10),
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom",
                                  y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_h, use_container_width=True)
                st.caption(
                    "Dashed bands = KO levels computed point-in-time "
                    f"from {dnt_lookback}-BDay trailing spot range × "
                    f"{dnt_widen:g}% widening. If the spot line breached "
                    "the band in the past, a hypothetical DNT placed at "
                    "that moment would have knocked out."
                )

            # ---- Download
            st.markdown("---")
            csv_bytes = df_scan.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇  Download screen (CSV)",
                data=csv_bytes,
                file_name=(f"dnt_screen_{dnt_tenor.lower()}_"
                              f"{pd.Timestamp.today().strftime('%Y%m%d')}"
                              ".csv"),
                mime="text/csv",
                key="dnt_download",
            )


# ============================================================================
# TAB 7 — Binary 10:1 — Goldman 'Binary OTMS' (image 4)
# ============================================================================
# SCOPE: For each USD pair, solve for the strike where a 3m (configurable)
# binary needs to be placed for a payout ratio of approximately 10:1
# (premium $1, payout $10 if struck, breakeven prob ≈ 10%).
#
# Goldman shows 4 panels (USD/EUR × Call/Put × G10/EM). We have USD pairs
# only, so we ship USD_CALL and USD_PUT side-by-side. Each shows G10 and
# EM Asia in one table, sorted by tightness (smallest |% OTMS| on top).
# Add EUR-cross pairs (EURJPY, EURGBP, EURCAD etc) to your data folder
# to extend.
#
# Caveat: GS uses the full vol smile; we use ATM vol. For skewed pairs
# expect ~0.5-1pp absolute discrepancy, ranking preserved.
with tab_binary:
    import plotly.express as px

    from core.screens import scan_binary_otms, ASIA_EM_PAIRS

    @st.cache_data(show_spinner="Scanning binary strikes…")
    def _scan_binary_cached(folder: str,
                              pairs: tuple[str, ...],
                              tenor: str,
                              direction: str,
                              payout_ratio: float,
                              prefer_em: str) -> pd.DataFrame:
        return scan_binary_otms(folder, pairs, tenor=tenor,
                                  direction=direction,
                                  payout_ratio=payout_ratio,
                                  prefer_em=prefer_em)

    # Controls
    cc1, cc2 = st.columns([1, 1])
    with cc1:
        bin_tenor = st.selectbox(
            "Tenor", ["1M", "3M", "6M"], index=1, key="bin_tenor",
        )
    with cc2:
        bin_ratio = st.number_input(
            "Payout ratio (X:1)",
            min_value=2.0, max_value=50.0, value=10.0, step=1.0,
            help=("Higher ratio = farther OTMS strike. 10:1 means $1 "
                    "premium pays $10 if struck (breakeven prob = 10%)."),
            key="bin_ratio",
        )

    all_pairs = sorted(list_available_pairs(folder, "VOL_ATM"))
    if not all_pairs:
        st.warning("No VOL_ATM data in `_index.csv`.")
    else:
        em_pref = prefer if asia_em else "offshore"

        df_call = _scan_binary_cached(folder, tuple(all_pairs),
                                          bin_tenor, "USD_CALL",
                                          float(bin_ratio), em_pref)
        df_put = _scan_binary_cached(folder, tuple(all_pairs),
                                        bin_tenor, "USD_PUT",
                                        float(bin_ratio), em_pref)

        st.markdown(
            f"**Binary {bin_ratio:.0f}:1 Screen**  ·  {bin_tenor} cash-or-"
            f"nothing options  ·  breakeven prob = {1/bin_ratio*100:.1f}%"
        )
        st.caption(
            "% OTMS = (K/S − 1) × 100. Normalized = |% OTMS| ÷ realized "
            "vol (a vol-adjusted distance — tighter strike in vol-time "
            "units = more attractive). Goldman uses the smile; we use "
            "ATM. Add EUR-cross pairs to extend to the EUR side."
        )

        def _style_norm(v):
            """Green for small (tight strike), red for large."""
            if pd.isna(v):
                return ""
            t = max(0.0, min(1.0, (float(v) - 0.3) / 1.0))
            if t < 0.5:
                x = t / 0.5
                r = int(40 + (255 - 40) * x)
                g = int(160 + (220 - 160) * x)
                b = int(40 + (130 - 40) * x)
            else:
                x = (t - 0.5) / 0.5
                r = int(255 + (220 - 255) * x)
                g = int(220 + (80 - 220) * x)
                b = int(130 + (80 - 130) * x)
            return (f"background-color: rgb({r},{g},{b}); "
                      "color: #1a1a1a; text-align: right; font-weight: 600;")

        def _fmt_px_bin(v, pair):
            if pd.isna(v):
                return "—"
            if pair in ("USDJPY", "USDKRW"):
                return f"{v:,.2f}"
            if pair == "USDIDR":
                return f"{v:,.0f}"
            return f"{v:.4f}"

        # ---- Two columns: Call panel | Put panel
        col_call, col_put = st.columns(2)

        with col_call:
            st.markdown("##### USD Calls (USD strengthens)")
            if df_call.empty:
                st.info("No data.")
            else:
                # Region tagging for display
                df_call_disp = df_call.copy()
                df_call_disp["Region"] = df_call_disp["Pair"].map(
                    lambda p: "EM" if p in ASIA_EM_PAIRS else "G10")
                df_call_disp["Spot"] = df_call_disp.apply(
                    lambda r: _fmt_px_bin(r["Spot"], r["Pair"]), axis=1)
                df_call_disp["Strike"] = df_call_disp.apply(
                    lambda r: _fmt_px_bin(r["Strike"], r["Pair"]), axis=1)

                styled = (df_call_disp[["Currency", "Region", "Spot",
                                          "Strike", "% OTMS", "Normalized"]]
                            .style
                            .format({
                                "% OTMS": "{:+.2f}%",
                                "Normalized": "{:.2f}x",
                            }, na_rep="—")
                            .map(_style_norm, subset=["Normalized"])
                            .set_table_styles([
                                {"selector": "th",
                                  "props": [("text-align", "center"),
                                              ("font-weight", "600")]},
                            ]))
                st.dataframe(styled, use_container_width=True,
                                hide_index=True,
                                height=min(560, 38 * (len(df_call) + 1)))

        with col_put:
            st.markdown("##### USD Puts (USD weakens)")
            if df_put.empty:
                st.info("No data.")
            else:
                df_put_disp = df_put.copy()
                df_put_disp["Region"] = df_put_disp["Pair"].map(
                    lambda p: "EM" if p in ASIA_EM_PAIRS else "G10")
                df_put_disp["Spot"] = df_put_disp.apply(
                    lambda r: _fmt_px_bin(r["Spot"], r["Pair"]), axis=1)
                df_put_disp["Strike"] = df_put_disp.apply(
                    lambda r: _fmt_px_bin(r["Strike"], r["Pair"]), axis=1)

                styled = (df_put_disp[["Currency", "Region", "Spot",
                                          "Strike", "% OTMS", "Normalized"]]
                            .style
                            .format({
                                "% OTMS": "{:+.2f}%",
                                "Normalized": "{:.2f}x",
                            }, na_rep="—")
                            .map(_style_norm, subset=["Normalized"])
                            .set_table_styles([
                                {"selector": "th",
                                  "props": [("text-align", "center"),
                                              ("font-weight", "600")]},
                            ]))
                st.dataframe(styled, use_container_width=True,
                                hide_index=True,
                                height=min(560, 38 * (len(df_put) + 1)))

        # ---- Scatter: Calls vs Puts asymmetry
        # Plots (OTMS_call + OTMS_put)/2 vs (OTMS_call - |OTMS_put|).
        # Positive Y = call needs to be farther OTMS than put (RR skewed
        # to USD calls = market prices USD upside as more likely).
        if not df_call.empty and not df_put.empty:
            st.markdown("---")
            st.markdown("##### Call vs Put Skew")
            st.caption(
                "X-axis: average distance to 10:1 strike (lower = vol "
                "is cheaper to buy upside). Y-axis: call OTMS minus "
                "|put OTMS| (positive = USD calls need to be farther OTMS, "
                "meaning the market prices USD downside as more likely "
                "= bearish-USD risk-reversal)."
            )
            merged = df_call[["Currency", "Pair", "% OTMS"]].merge(
                df_put[["Pair", "% OTMS"]],
                on="Pair", suffixes=("_call", "_put"),
            )
            merged["AvgDist"] = (merged["% OTMS_call"]
                                   + merged["% OTMS_put"].abs()) / 2.0
            merged["Skew"] = (merged["% OTMS_call"]
                                + merged["% OTMS_put"])  # put is neg, so this = call - |put|
            merged["Region"] = merged["Pair"].map(
                lambda p: "EM Asia" if p in ASIA_EM_PAIRS else "G10")

            fig_sc = px.scatter(
                merged, x="AvgDist", y="Skew",
                color="Region", text="Currency",
                color_discrete_map={"G10": "#1f77b4",
                                       "EM Asia": "#2ca02c"},
                height=420,
            )
            fig_sc.update_traces(textposition="top center",
                                    marker=dict(size=11,
                                                  line=dict(width=0.5,
                                                              color="#333")))
            fig_sc.add_hline(y=0, line_dash="dash",
                                line_color="#888", opacity=0.5)
            fig_sc.update_layout(
                xaxis_title="Avg distance to 10:1 strike (% OTMS)",
                yaxis_title="Call OTMS − |Put OTMS| (skew)",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom",
                                y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        # ---- Download
        st.markdown("---")
        d1, d2 = st.columns(2)
        with d1:
            if not df_call.empty:
                csv_bytes = df_call.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇  Calls (CSV)", data=csv_bytes,
                    file_name=(f"binary_calls_{bin_tenor.lower()}_"
                                  f"{pd.Timestamp.today().strftime('%Y%m%d')}"
                                  ".csv"),
                    mime="text/csv",
                    key="bin_call_download",
                )
        with d2:
            if not df_put.empty:
                csv_bytes = df_put.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇  Puts (CSV)", data=csv_bytes,
                    file_name=(f"binary_puts_{bin_tenor.lower()}_"
                                  f"{pd.Timestamp.today().strftime('%Y%m%d')}"
                                  ".csv"),
                    mime="text/csv",
                    key="bin_put_download",
                )


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    f"Data: `{folder}` · Pair: **{pair}** · Lookback: **{lookback_label}** · "
    f"Skew band: **{low_pct:g}–{high_pct:g} pct**"
    + (f" · Variant: **{prefer}**" if asia_em else "")
)

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
tab_skew, tab_iv, tab_heat, tab_alerts = st.tabs(
    ["📊 Skew", "📈 Implied Vol", "🌡️ Heatmap", "🚨 Alerts"]
)


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


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption(
    f"Data: `{folder}` · Pair: **{pair}** · Lookback: **{lookback_label}** · "
    f"Skew band: **{low_pct:g}–{high_pct:g} pct**"
    + (f" · Variant: **{prefer}**" if asia_em else "")
)

"""
WO Knockout Portfolio Comparison — Streamlit app
================================================
Upload backtest CSVs (one summary + one timeseries per strategy). Filename
pattern parsed automatically:
    WO-<EKO|RKO>_<CCY>_<CCY>_..._T<tenor>_S<strike>_B<barrier>_<summary|timeseries>.csv
e.g. WO-EKO_JPY_KRW_TWD_T2M_SATM_B10D_summary.csv

Each portfolio (= unique leg-set) is shown as a row of two cards: EKO on the
left, RKO on the right. Each card has the key stats plus a small annual-PnL
bar chart with the annual Sharpe overlaid on a secondary axis.

Run:
    pip install streamlit pandas plotly numpy
    streamlit run app.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="WO Knockout Comparison",
    page_icon="📊",
    layout="wide",
)

# Colors for EKO / RKO bars
EKO_COLOR = "#2563eb"   # blue
RKO_COLOR = "#dc2626"   # red

# ============================================================================
# Filename parsing
# ============================================================================

_STRUCTURE_RE = re.compile(r"^WO-(EKO|RKO)_(.+)$", re.IGNORECASE)
_TENOR_RE     = re.compile(r"_T(\d+[MWY])(?=_|$)", re.IGNORECASE)
_STRIKE_RE    = re.compile(r"_S([\w+\-.]+?)(?=_B|$)", re.IGNORECASE)
_BARRIER_RE   = re.compile(r"_B([\w+\-.@]+?)$", re.IGNORECASE)


def parse_filename(name: str) -> dict:
    """Return dict with structure / legs / tenor / strike / barrier / kind."""
    info = {
        "structure": None, "legs": None, "tenor": None,
        "strike": None, "barrier": None, "kind": None,
    }
    stem = Path(name).stem
    for suffix in ("_summary", "_timeseries"):
        if stem.lower().endswith(suffix):
            info["kind"] = suffix[1:]
            stem = stem[: -len(suffix)]
            break

    m = _STRUCTURE_RE.match(stem)
    if not m:
        return info
    info["structure"] = m.group(1).upper()
    body = m.group(2)

    # Tenor anchors the parse — leg currencies before it, strike/barrier after.
    # This avoids matching '_S' inside currency codes like 'SGD' or '_B' inside 'BRL'.
    t = _TENOR_RE.search(body)
    if not t:
        return info
    info["tenor"] = t.group(1).upper()

    legs = [x for x in body[: t.start()].split("_") if x]
    if legs:
        info["legs"] = "/".join(legs)

    tail = body[t.end():]  # everything after the tenor token
    s = _STRIKE_RE.search(tail)
    b = _BARRIER_RE.search(tail)
    if s:
        info["strike"]  = s.group(1).upper()
    if b:
        info["barrier"] = b.group(1).upper()
    return info


def detect_kind_from_columns(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if {"period_end", "equity_usd"}.issubset(cols):
        return "timeseries"
    if {"strategy_name", "total_pnl_usd"}.issubset(cols):
        return "summary"
    return "unknown"


# ============================================================================
# Data combine
# ============================================================================

def combine(meta: pd.DataFrame, frames: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    s_rows, t_rows = [], []
    for _, r in meta.iterrows():
        if not r["structure"] or not r["legs"]:
            continue
        df = frames[r["filename"]].copy()
        df["structure"] = r["structure"]
        df["legs"]      = r["legs"]
        df["tenor"]     = r["tenor"]
        df["strike"]    = r["strike"]
        df["barrier"]   = r["barrier"]
        if r["kind"] == "summary":
            s_rows.append(df)
        elif r["kind"] == "timeseries":
            t_rows.append(df)

    summary = (
        pd.concat(s_rows, ignore_index=True, sort=False) if s_rows else pd.DataFrame()
    )
    ts = pd.concat(t_rows, ignore_index=True, sort=False) if t_rows else pd.DataFrame()
    if not ts.empty:
        ts["period_end"] = pd.to_datetime(ts["period_end"])
    return summary, ts


def annual_from_timeseries(ts_sub: pd.DataFrame) -> pd.DataFrame:
    """Aggregate monthly timeseries into annual PnL + annual Sharpe."""
    if ts_sub.empty:
        return pd.DataFrame(columns=["year", "pnl", "sharpe"])
    df = ts_sub.copy()
    df["year"] = df["period_end"].dt.year
    grp = (
        df.groupby("year")
        .agg(pnl=("pnl_usd", "sum"),
             pnl_mean=("pnl_usd", "mean"),
             pnl_std=("pnl_usd", "std"),
             n=("pnl_usd", "count"))
        .reset_index()
    )
    grp["sharpe"] = np.where(
        (grp["pnl_std"] > 0) & (grp["n"] >= 3),
        (grp["pnl_mean"] / grp["pnl_std"]) * np.sqrt(12),
        np.nan,
    )
    return grp[["year", "pnl", "sharpe"]]


# ============================================================================
# Sidebar — upload + filters
# ============================================================================

st.sidebar.title("⚙️  Controls")
st.sidebar.subheader("Upload CSV files")
st.sidebar.caption(
    "Filename pattern (auto-parsed):  \n"
    "`WO-<EKO|RKO>_<CCY>_..._T<tenor>_S<strike>_B<barrier>_<summary|timeseries>.csv`"
)

uploads = st.sidebar.file_uploader(
    "Drop summary + timeseries CSVs", type="csv", accept_multiple_files=True
)

if not uploads:
    st.info(
        "📥  **Upload CSVs in the sidebar to begin.**\n\n"
        "Example filenames:\n"
        "- `WO-EKO_JPY_KRW_THB_T2M_SATM_B10D_summary.csv`\n"
        "- `WO-EKO_JPY_KRW_THB_T2M_SATM_B10D_timeseries.csv`\n"
        "- `WO-RKO_CNH_INR_SGD_TWD_T2M_SATM_B10D_summary.csv`\n"
        "- `WO-RKO_CNH_INR_SGD_TWD_T2M_SATM_B10D_timeseries.csv`\n"
    )
    st.stop()

# Read each file once
frames: dict[str, pd.DataFrame] = {}
meta_rows: list[dict] = []
for uf in uploads:
    try:
        df = pd.read_csv(uf)
    except Exception as e:
        st.sidebar.error(f"Could not read `{uf.name}`: {e}")
        continue
    info = parse_filename(uf.name)
    if info["kind"] not in ("summary", "timeseries"):
        info["kind"] = detect_kind_from_columns(df)
    frames[uf.name] = df
    meta_rows.append({**info, "filename": uf.name, "rows": len(df)})

meta = pd.DataFrame(meta_rows)

with st.sidebar.expander("📄 Files detected", expanded=False):
    st.dataframe(
        meta[["filename", "structure", "legs", "tenor", "strike", "barrier", "kind", "rows"]],
        hide_index=True, use_container_width=True,
    )

summary, timeseries = combine(meta, frames)

if summary.empty or timeseries.empty:
    st.error(
        "Need at least one summary AND one timeseries CSV with the expected "
        "filename pattern. See the **Files detected** panel for parse results."
    )
    st.stop()

# ---- Filters ----
st.sidebar.subheader("Filters")


def _multi(df, col, label):
    opts = sorted(df[col].dropna().unique().tolist())
    return st.sidebar.multiselect(label, opts, default=opts) if opts else opts


strike_sel  = _multi(summary, "strike",  "Strike")
barrier_sel = _multi(summary, "barrier", "Barrier")
tenor_sel   = _multi(summary, "tenor",   "Tenor")

mask = (
    summary["strike"].isin(strike_sel)
    & summary["barrier"].isin(barrier_sel)
    & summary["tenor"].isin(tenor_sel)
)
summary_f = summary[mask].copy()

ts_mask = (
    timeseries["strike"].isin(strike_sel)
    & timeseries["barrier"].isin(barrier_sel)
    & timeseries["tenor"].isin(tenor_sel)
)
ts_f = timeseries[ts_mask].copy()


# ============================================================================
# Main — header KPIs
# ============================================================================

st.title("📊  Worst-of Knockout Portfolio Comparison")
st.caption("EKO (European knockout) vs RKO (American knockout) — Asia FX baskets")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Strategies", len(summary_f))
k2.metric("Portfolios", summary_f["legs"].nunique())
k3.metric("Total PnL", f"${summary_f['total_pnl_usd'].sum() / 1e6:,.1f}M")
k4.metric("Avg ann. Sharpe", f"{summary_f['annual_sharpe_mean'].mean():.2f}")

st.divider()


# ============================================================================
# Per-portfolio cards (EKO left, RKO right)
# ============================================================================

def _render_card_body(row: pd.Series, ts_sub: pd.DataFrame, color: str) -> None:
    """Stats grid + tiny PnL/Sharpe chart inside a card."""
    pnl_m  = row["total_pnl_usd"]   / 1e6
    dd_m   = row["max_drawdown_usd"] / 1e6
    sharpe = row["annual_sharpe_mean"]
    calmar = row["calmar"]
    win    = row["win_rate_pct"]
    rec    = row["premium_recovery_pct"]

    # Stats in two rows of 3
    r1c1, r1c2, r1c3 = st.columns(3)
    r1c1.metric("Total PnL",      f"${pnl_m:,.1f}M")
    r1c2.metric("Sharpe (ann.)",  f"{sharpe:.2f}")
    r1c3.metric("Calmar",         f"{calmar:.2f}")
    r2c1, r2c2, r2c3 = st.columns(3)
    r2c1.metric("Max drawdown",   f"${dd_m:,.1f}M")
    r2c2.metric("Win rate",       f"{win:.1f}%")
    r2c3.metric("Prem. recovery", f"{rec:.0f}%")

    # Two small side-by-side charts: annual PnL on the left, annual Sharpe on the right
    ann = annual_from_timeseries(ts_sub)
    if ann.empty:
        st.caption("No timeseries available for this strategy")
        return

    chart_l, chart_r = st.columns(2)

    with chart_l:
        fig_pnl = go.Figure()
        fig_pnl.add_trace(
            go.Bar(
                x=ann["year"], y=ann["pnl"],
                marker_color=color, opacity=0.85,
                hovertemplate="<b>%{x}</b><br>PnL: $%{y:,.0f}<extra></extra>",
            )
        )
        fig_pnl.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)
        fig_pnl.update_yaxes(tickformat="$,.2s")
        fig_pnl.update_xaxes(type="category")
        fig_pnl.update_layout(
            title=dict(text="Annual PnL", font=dict(size=12), x=0.5, xanchor="center"),
            height=220,
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False,
            bargap=0.25,
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

    with chart_r:
        fig_sharpe = go.Figure()
        fig_sharpe.add_trace(
            go.Bar(
                x=ann["year"], y=ann["sharpe"],
                marker_color="#f59e0b", opacity=0.9,
                hovertemplate="<b>%{x}</b><br>Sharpe: %{y:.2f}<extra></extra>",
            )
        )
        fig_sharpe.add_hline(y=0, line_dash="dot", line_color="grey", line_width=1)
        fig_sharpe.update_xaxes(type="category")
        fig_sharpe.update_layout(
            title=dict(text="Annual Sharpe", font=dict(size=12), x=0.5, xanchor="center"),
            height=220,
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False,
            bargap=0.25,
        )
        st.plotly_chart(fig_sharpe, use_container_width=True)


def _render_empty(label: str) -> None:
    st.markdown(f"**{label}**")
    st.info("No data for this portfolio.")


# Order portfolios by combined PnL across structures (best first)
order = (
    summary_f.groupby("legs")["total_pnl_usd"].sum().sort_values(ascending=False).index.tolist()
)

st.subheader("Portfolio comparison — EKO  vs  RKO")

for legs in order:
    sub = summary_f[summary_f["legs"] == legs]
    eko = sub[sub["structure"] == "EKO"]
    rko = sub[sub["structure"] == "RKO"]

    st.markdown(f"#### `{legs}`")
    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown(
                f"<span style='color:{EKO_COLOR};font-weight:600;font-size:1.05rem'>"
                f"● EKO &nbsp;·&nbsp; European knockout</span>",
                unsafe_allow_html=True,
            )
            if not eko.empty:
                row = eko.iloc[0]
                tsub = ts_f[(ts_f["legs"] == legs) & (ts_f["structure"] == "EKO")]
                _render_card_body(row, tsub, EKO_COLOR)
            else:
                _render_empty("EKO")

    with right:
        with st.container(border=True):
            st.markdown(
                f"<span style='color:{RKO_COLOR};font-weight:600;font-size:1.05rem'>"
                f"● RKO &nbsp;·&nbsp; American knockout</span>",
                unsafe_allow_html=True,
            )
            if not rko.empty:
                row = rko.iloc[0]
                tsub = ts_f[(ts_f["legs"] == legs) & (ts_f["structure"] == "RKO")]
                _render_card_body(row, tsub, RKO_COLOR)
            else:
                _render_empty("RKO")

st.divider()


# ============================================================================
# Detail table — column selector
# ============================================================================

st.subheader("Detail table")

ALL_COLS = [
    c for c in summary_f.columns if c not in {"strategy_name", "strategy_type"}
]
# Put parsed/identifier cols first
ID_COLS = ["structure", "legs", "tenor", "strike", "barrier"]
ALL_COLS = ID_COLS + [c for c in ALL_COLS if c not in ID_COLS]

DEFAULT_COLS = [
    "structure", "legs", "tenor", "strike", "barrier",
    "n_trades", "total_premium_paid_usd", "total_pnl_usd",
    "max_drawdown_usd", "win_rate_pct", "premium_recovery_pct",
    "annual_sharpe_mean", "calmar", "gain_to_pain",
    "leg_a_ko_rate_pct", "leg_b_ko_rate_pct",
]
DEFAULT_COLS = [c for c in DEFAULT_COLS if c in ALL_COLS]

cols = st.multiselect(
    "Columns to display",
    options=ALL_COLS,
    default=DEFAULT_COLS,
    help="Pick the metrics you want in the detail table below.",
)

if not cols:
    st.info("Pick at least one column above to render the table.")
    st.stop()


def _rdylgn_css(s: pd.Series, reverse: bool = False) -> list[str]:
    """Pure-CSS red→yellow→green gradient (no matplotlib)."""
    vals = pd.to_numeric(s, errors="coerce")
    finite = vals.dropna()
    if finite.empty or finite.min() == finite.max():
        return ["" for _ in s]
    lo, hi = float(finite.min()), float(finite.max())
    out: list[str] = []
    for v in vals:
        if pd.isna(v):
            out.append("")
            continue
        t = (float(v) - lo) / (hi - lo)
        if reverse:
            t = 1.0 - t
        if t < 0.5:
            k = t * 2.0
            r = int(round(215 + (255 - 215) * k))
            g = int(round( 25 + (255 -  25) * k))
            b = int(round( 28 + (191 -  28) * k))
        else:
            k = (t - 0.5) * 2.0
            r = int(round(255 + ( 26 - 255) * k))
            g = int(round(255 + (150 - 255) * k))
            b = int(round(191 + ( 65 - 191) * k))
        out.append(f"background-color: rgb({r},{g},{b}); color: #111")
    return out


def build_styler(df: pd.DataFrame, cols: list[str]):
    fmt: dict[str, str] = {}
    for c in cols:
        if c.endswith("_usd"):
            fmt[c] = "${:,.0f}"
        elif c.endswith("_pct"):
            fmt[c] = "{:.1f}"
        elif pd.api.types.is_float_dtype(df[c]):
            fmt[c] = "{:.2f}"

    styler = df[cols].style.format(fmt, na_rep="—")

    # Apply gradient to numeric, higher-is-better cols
    higher_good = [c for c in cols if c in {
        "total_pnl_usd", "annual_sharpe_mean", "calmar", "gain_to_pain",
        "win_rate_pct", "premium_recovery_pct", "pct_positive_years",
        "both_survive_rate_pct",
    }]
    if higher_good:
        styler = styler.apply(_rdylgn_css, subset=higher_good)

    # Lower-is-better
    lower_good = [c for c in cols if c in {
        "max_drawdown_usd", "ulcer_index", "annual_sharpe_cv",
        "total_premium_paid_usd", "total_tx_cost_usd",
        "leg_a_ko_rate_pct", "leg_b_ko_rate_pct",
    }]
    if lower_good:
        styler = styler.apply(lambda s: _rdylgn_css(s, reverse=True), subset=lower_good)

    return styler


# Sort by structure then legs for a clean visual
table_df = summary_f.sort_values(["structure", "legs"]).reset_index(drop=True)
st.dataframe(build_styler(table_df, cols), use_container_width=True, hide_index=True)

dlc1, dlc2 = st.columns([1, 5])
dlc1.download_button(
    "⬇️  Download CSV",
    table_df[cols].to_csv(index=False).encode("utf-8"),
    file_name="filtered_summary.csv",
    mime="text/csv",
    use_container_width=True,
)

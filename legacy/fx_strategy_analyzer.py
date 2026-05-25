"""
FX Option Strategy Analyzer
============================
A Streamlit app for finding the optimal combination of strike, KO strike, tenor
and gate across an FX option backtest CSV (e.g. USDJPY_All.csv).

Usage:
    streamlit run fx_strategy_analyzer.py
"""

from __future__ import annotations

import hmac
import re
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FX Option Strategy Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Optional password gate. Activates only when an "app_password" secret is set
# in Streamlit secrets (.streamlit/secrets.toml locally, or the Secrets pane
# in the Community Cloud UI). If no secret is set, the app is open.
# ---------------------------------------------------------------------------
def _check_password() -> bool:
    if "app_password" not in st.secrets:
        return True  # no password configured -> open access
    if st.session_state.get("auth_ok"):
        return True

    st.title("🔒 FX Option Strategy Analyzer")
    pwd = st.text_input("Password", type="password")
    if pwd and hmac.compare_digest(pwd, str(st.secrets["app_password"])):
        st.session_state["auth_ok"] = True
        st.rerun()
    elif pwd:
        st.error("Incorrect password.")
    return False


if not _check_password():
    st.stop()


# ---------------------------------------------------------------------------
# Where to find preloaded result files. Drop CSVs into ./data/ in the repo
# and they'll show up in the dropdown automatically.
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"


def list_preloaded() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("*.csv"))


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
TENOR_TO_DAYS = {"1W": 7, "2W": 14, "3W": 21, "1M": 30, "6W": 42,
                 "2M": 60, "10W": 70, "3M": 90, "4M": 120, "6M": 180,
                 "9M": 270, "1Y": 365}


def parse_dollar(value) -> float:
    """Parse '$197.23M', '-$10.27M', '$1.2B', '$500K' into a float (USD)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    sign = -1 if s.startswith("-") else 1
    s = s.replace("-", "").replace("$", "").replace(",", "").strip()
    mult = 1.0
    if s.endswith("B"):
        mult, s = 1e9, s[:-1]
    elif s.endswith("M"):
        mult, s = 1e6, s[:-1]
    elif s.endswith("K") or s.endswith("k"):
        mult, s = 1e3, s[:-1]
    try:
        return sign * float(s) * mult
    except ValueError:
        return np.nan


def parse_pct(value) -> float:
    """Parse '41' or '41%' into 41.0."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def strike_to_numeric(strike: str) -> float:
    """ATM -> 50, '25Δ' -> 25, '45Δ' -> 45. Higher number = closer to spot."""
    if not isinstance(strike, str):
        return np.nan
    s = strike.strip()
    if s.upper() == "ATM":
        return 50.0
    m = re.match(r"(\d+(?:\.\d+)?)\s*[Δ∆D]", s)
    return float(m.group(1)) if m else np.nan


def ko_to_numeric(ko: str) -> float:
    """'H@5Δ' -> 5."""
    if not isinstance(ko, str):
        return np.nan
    m = re.search(r"@\s*(\d+(?:\.\d+)?)\s*[Δ∆D]", ko)
    return float(m.group(1)) if m else np.nan


def tenor_to_days(t: str) -> float:
    return TENOR_TO_DAYS.get(str(t).strip().upper(), np.nan)


STRATEGY_RE = re.compile(
    r"""^\s*
        (?P<pair>[A-Z]{6})\s+
        (?P<type>\S+)\s+
        (?P<strike>ATM|\d+(?:\.\d+)?[Δ∆D])\s+
        (?P<tenor>\d+[WwMmYy])\s+
        (?P<ko>H@\d+(?:\.\d+)?[Δ∆D])
        (?:\s+\[(?P<gate>[^\]]+)\])?\s*$
    """,
    re.VERBOSE,
)


def parse_strategy(strategy: str) -> dict:
    """Parse a strategy string like:
        'USDJPY CALL-upout  ATM  1M  H@5Δ'
        'USDJPY CALL-upout  ATM  1M  H@5Δ  [Spot > 50DMA]'
    """
    if not isinstance(strategy, str):
        return {"Pair": None, "OptType": None, "Strike": None,
                "Tenor": None, "KO": None, "Gate": "None"}
    m = STRATEGY_RE.match(strategy)
    if not m:
        # Best-effort fallback: split first 5 tokens on whitespace
        parts = strategy.strip().split(None, 4)
        gate = "None"
        if len(parts) >= 5 and "[" in parts[4]:
            head, _, rest = parts[4].partition("[")
            parts[4] = head.strip()
            gate = rest.strip(" ]")
        return {
            "Pair": parts[0] if len(parts) > 0 else None,
            "OptType": parts[1] if len(parts) > 1 else None,
            "Strike": parts[2] if len(parts) > 2 else None,
            "Tenor": parts[3] if len(parts) > 3 else None,
            "KO": parts[4] if len(parts) > 4 else None,
            "Gate": gate,
        }
    d = m.groupdict()
    return {
        "Pair": d["pair"],
        "OptType": d["type"],
        "Strike": d["strike"],
        "Tenor": d["tenor"],
        "KO": d["ko"],
        "Gate": d["gate"] if d["gate"] else "None",
    }


# ---------------------------------------------------------------------------
# Loader / cleaner
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes: bytes) -> pd.DataFrame:
    """Read the raw CSV bytes and return a cleaned, parsed DataFrame."""
    df = pd.read_csv(BytesIO(file_bytes), encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    # Parse strategy column into separate columns
    parsed = df["Strategy"].apply(parse_strategy).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)

    # Numeric conversions
    pct_cols = ["Feas%", "KO%", "Win%", "Recovery%", "%Pos Yrs"]
    dollar_cols = ["Σ Premium", "Σ TX Cost", "Σ Payout", "Σ PnL",
                   "Max DD", "Min Ann $"]
    float_cols = ["n", "Sharpe (m)", "n trades", "Yrs",
                  "Sharpe(y) μ", "Sharpe(y) min", "Calmar", "G2P", "Ulcer"]

    for c in pct_cols:
        if c in df.columns:
            df[c] = df[c].apply(parse_pct)
    for c in dollar_cols:
        if c in df.columns:
            df[c] = df[c].apply(parse_dollar)
    for c in float_cols:
        if c in df.columns:
            # to_numeric handles signed floats like "+0.84", "-3.60"
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derived numeric helpers
    df["Strike_num"] = df["Strike"].apply(strike_to_numeric)
    df["KO_num"] = df["KO"].apply(ko_to_numeric)
    df["Tenor_days"] = df["Tenor"].apply(tenor_to_days)

    # Useful derived ratios (guard against div-by-zero)
    df["PnL/Premium"] = np.where(df["Σ Premium"] > 0,
                                 df["Σ PnL"] / df["Σ Premium"], np.nan)
    df["PnL/|DD|"] = np.where(df["Max DD"] < 0,
                              df["Σ PnL"] / df["Max DD"].abs(), np.nan)
    df["TX/Premium %"] = np.where(df["Σ Premium"] > 0,
                                  100 * df["Σ TX Cost"] / df["Σ Premium"],
                                  np.nan)
    return df


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt_usd(x) -> str:
    if pd.isna(x):
        return "—"
    a = abs(x)
    sign = "-" if x < 0 else ""
    if a >= 1e9:
        return f"{sign}${a/1e9:.2f}B"
    if a >= 1e6:
        return f"{sign}${a/1e6:.2f}M"
    if a >= 1e3:
        return f"{sign}${a/1e3:.1f}K"
    return f"{sign}${a:,.0f}"


def fmt_pct(x) -> str:
    return "—" if pd.isna(x) else f"{x:.1f}%"


def fmt_float(x, n=2) -> str:
    return "—" if pd.isna(x) else f"{x:.{n}f}"


# ---------------------------------------------------------------------------
# Single source of truth for metric semantics. Every tab pulls from this so
# new columns get picked up consistently.
# Format kind: "usd", "pct", "float", "int"
# ---------------------------------------------------------------------------
METRIC_SPECS: dict[str, dict] = {
    # Return / quality (higher better)
    "Σ PnL":         {"label": "Σ PnL",                  "higher": True,  "fmt": "usd"},
    "Sharpe (m)":    {"label": "Sharpe (monthly)",       "higher": True,  "fmt": "float"},
    "Sharpe(y) μ":   {"label": "Sharpe (yearly, mean)",  "higher": True,  "fmt": "float"},
    "Sharpe(y) min": {"label": "Sharpe (yearly, worst)", "higher": True,  "fmt": "float"},
    "Calmar":        {"label": "Calmar",                 "higher": True,  "fmt": "float"},
    "G2P":           {"label": "Gain-to-Pain",           "higher": True,  "fmt": "float"},
    "Win%":          {"label": "Win %",                  "higher": True,  "fmt": "pct"},
    "%Pos Yrs":      {"label": "% Positive Years",       "higher": True,  "fmt": "pct"},
    "Recovery%":     {"label": "Recovery %",             "higher": True,  "fmt": "pct"},
    "PnL/Premium":   {"label": "PnL ÷ Premium",          "higher": True,  "fmt": "float"},
    "PnL/|DD|":      {"label": "PnL ÷ |MaxDD|",          "higher": True,  "fmt": "float"},
    # Loss-side (less negative is better, so higher=True)
    "Max DD":        {"label": "Max DD (less neg=better)",     "higher": True, "fmt": "usd"},
    "Min Ann $":     {"label": "Worst annual $ (higher=better)", "higher": True, "fmt": "usd"},
    # Risk / cost (lower better)
    "KO%":           {"label": "KO % (lower=better)",        "higher": False, "fmt": "pct"},
    "Ulcer":         {"label": "Ulcer Index (lower=better)", "higher": False, "fmt": "float"},
    "TX/Premium %":  {"label": "TX % of Premium (lower=better)", "higher": False, "fmt": "pct"},
}


def available_metrics(df: pd.DataFrame,
                      only: Optional[list[str]] = None) -> dict[str, dict]:
    """Return a dict of metric_name -> spec for metrics that exist in df
    and have at least one non-NaN value. Optionally restrict to a subset."""
    out = {}
    for k, spec in METRIC_SPECS.items():
        if only is not None and k not in only:
            continue
        if k in df.columns and df[k].notna().any():
            out[k] = spec
    return out


def style_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        spec = METRIC_SPECS.get(c)
        if spec is None:
            continue
        kind = spec["fmt"]
        if kind == "usd":
            out[c] = out[c].apply(fmt_usd)
        elif kind == "pct":
            out[c] = out[c].apply(fmt_pct)
        elif kind == "float":
            out[c] = out[c].apply(lambda v: fmt_float(v, 2))
    # Feas% isn't a "metric" per se but format it anyway
    if "Feas%" in out.columns:
        out["Feas%"] = out["Feas%"].apply(fmt_pct)
    return out


# Canonical column order for tables — only columns actually present are kept
TABLE_COLS_ORDER = [
    "Strike", "Tenor", "KO", "Gate", "n", "Yrs", "Feas%", "KO%", "Win%",
    "%Pos Yrs", "Σ PnL", "Σ Premium", "Σ TX Cost", "Σ Payout",
    "Sharpe (m)", "Sharpe(y) μ", "Sharpe(y) min", "Calmar", "G2P",
    "Max DD", "Min Ann $", "Ulcer", "Recovery%",
    "PnL/Premium", "PnL/|DD|", "TX/Premium %",
]


def table_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in TABLE_COLS_ORDER if c in df.columns]


# ---------------------------------------------------------------------------
# Sidebar — data source: dropdown of preloaded files OR upload
# ---------------------------------------------------------------------------
st.sidebar.title("📁 Data")

preloaded = list_preloaded()
options: dict[str, Path | None] = {p.stem: p for p in preloaded}
options["⬆️ Upload my own CSV"] = None  # always offer upload as fallback

choice = st.sidebar.selectbox(
    "Dataset",
    list(options.keys()),
    index=0 if preloaded else len(options) - 1,
    help="Pick a preloaded backtest, or upload a CSV with the same schema.",
)

file_bytes: bytes | None = None
file_label: str = ""

if options[choice] is None:
    uploaded = st.sidebar.file_uploader("Upload backtest CSV", type=["csv"])
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        file_label = uploaded.name
else:
    path = options[choice]
    file_bytes = path.read_bytes()
    file_label = path.name

if file_bytes is None:
    st.title("📈 FX Option Strategy Analyzer")
    if preloaded:
        st.markdown(
            f"Select a preloaded dataset in the sidebar "
            f"(**{len(preloaded)} available**), or upload your own CSV."
        )
    else:
        st.markdown(
            """
            Upload an FX option backtest CSV in the sidebar to begin.

            Expected schema (e.g. **USDJPY_All.csv**):

            ```
            Strategy, n, Feas%, KO%, Win%, Σ Premium, Σ TX Cost,
            Σ Payout, Σ PnL, Sharpe (m), Max DD, Recovery%, n trades
            ```

            The **Strategy** column is parsed into its components:
            *Pair · OptionType · Strike · Tenor · KO · Gate*.
            """
        )
    st.stop()

data = load_and_clean(file_bytes)
pair = data["Pair"].mode().iat[0] if not data["Pair"].isna().all() else "—"

st.title(f"📈 FX Option Strategy Analyzer — {pair}")
st.caption(
    f"📄 `{file_label}` · "
    f"**{len(data):,}** strategies · "
    f"{data['Strike'].nunique()} strikes · "
    f"{data['Tenor'].nunique()} tenors · "
    f"{data['KO'].nunique()} KO levels · "
    f"{data['Gate'].nunique()} gates"
)


# Sidebar filters
st.sidebar.title("🔎 Filters")

with st.sidebar.expander("Dimension filters", expanded=True):
    strikes = sorted(data["Strike"].dropna().unique().tolist(),
                     key=strike_to_numeric)
    tenors = sorted(data["Tenor"].dropna().unique().tolist(),
                    key=lambda t: TENOR_TO_DAYS.get(t.upper(), 9999))
    kos = sorted(data["KO"].dropna().unique().tolist(), key=ko_to_numeric)
    gates = sorted(data["Gate"].dropna().unique().tolist())

    sel_strikes = st.multiselect("Strike", strikes, default=strikes)
    sel_tenors = st.multiselect("Tenor", tenors, default=tenors)
    sel_kos = st.multiselect("KO", kos, default=kos)
    sel_gates = st.multiselect("Gate", gates, default=gates)

with st.sidebar.expander("Performance constraints", expanded=False):
    min_n = int(np.nanmin(data["n"])) if data["n"].notna().any() else 0
    max_n = int(np.nanmax(data["n"])) if data["n"].notna().any() else 1
    n_min = st.number_input("Min sample size (n)", min_value=0,
                            max_value=max_n, value=min_n, step=10)
    max_ko = st.slider("Max KO% (knockout rate)", 0, 100, 100)
    min_sharpe = st.number_input("Min Sharpe", value=-5.0, step=0.1)

f = data[
    data["Strike"].isin(sel_strikes)
    & data["Tenor"].isin(sel_tenors)
    & data["KO"].isin(sel_kos)
    & data["Gate"].isin(sel_gates)
    & (data["n"].fillna(0) >= n_min)
    & (data["KO%"].fillna(0) <= max_ko)
    & (data["Sharpe (m)"].fillna(-99) >= min_sharpe)
].copy()

if f.empty:
    st.warning("No strategies match the current filters. Loosen them in the sidebar.")
    st.stop()

st.sidebar.markdown(f"**{len(f):,} / {len(data):,}** strategies after filters")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_top, tab_dim, tab_heat, tab_gate, tab_pareto, tab_compare, tab_data = st.tabs(
    [
        "🏆 Optimal & Top-N",
        "📊 Per-dimension impact",
        "🔥 Heatmaps",
        "🚪 Gate uplift",
        "⚖️ Pareto frontier",
        "🆚 Side-by-side compare",
        "📄 Data",
    ]
)


# ===========================================================================
# Tab: Optimal & Top-N
# ===========================================================================
with tab_top:
    st.subheader("Optimal strategy")

    metric_options = available_metrics(f)
    cA, cB = st.columns([2, 1])
    metric = cA.selectbox(
        "Optimise on",
        list(metric_options.keys()),
        format_func=lambda k: metric_options[k]["label"],
        index=0,
    )
    higher_better = metric_options[metric]["higher"]
    top_n = cB.slider("Show top N", 5, 50, 10)

    ranked = f.dropna(subset=[metric]).sort_values(
        metric, ascending=not higher_better
    )
    if ranked.empty:
        st.warning(f"No rows have a value for {metric}.")
    else:
        best = ranked.iloc[0]

        # Headline cards - row 1: dimensions
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Strike", best["Strike"])
        k2.metric("Tenor", best["Tenor"])
        k3.metric("KO", best["KO"])
        k4.metric("Gate", best["Gate"] if best["Gate"] != "None" else "— none —")
        k5.metric("Σ PnL", fmt_usd(best.get("Σ PnL", np.nan)))
        k6.metric("Sharpe (m)", fmt_float(best.get("Sharpe (m)", np.nan)))

        # Headline cards - row 2: key risk/quality
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Win %", fmt_pct(best.get("Win%", np.nan)))
        k2.metric("KO %", fmt_pct(best.get("KO%", np.nan)))
        k3.metric("Max DD", fmt_usd(best.get("Max DD", np.nan)))
        k4.metric("Recovery %", fmt_pct(best.get("Recovery%", np.nan)))
        k5.metric("Σ Premium", fmt_usd(best.get("Σ Premium", np.nan)))
        k6.metric("n", f"{int(best['n']):,}" if pd.notna(best["n"]) else "—")

        # Headline cards - row 3: annual / risk-adjusted (only if present)
        extra = [c for c in ["Sharpe(y) μ", "Sharpe(y) min", "Calmar",
                             "G2P", "%Pos Yrs", "Min Ann $", "Ulcer", "Yrs"]
                 if c in best.index]
        if extra:
            cols = st.columns(min(6, len(extra)))
            for i, c in enumerate(extra[:len(cols)]):
                spec = METRIC_SPECS.get(c, {"fmt": "float"})
                v = best.get(c, np.nan)
                if spec["fmt"] == "usd":
                    s = fmt_usd(v)
                elif spec["fmt"] == "pct":
                    s = fmt_pct(v)
                elif c == "Yrs":
                    s = f"{int(v)}" if pd.notna(v) else "—"
                else:
                    s = fmt_float(v)
                cols[i].metric(c, s)

        st.markdown("---")
        st.subheader(f"Top {top_n} by {metric_options[metric]['label']}")
        cols_to_show = table_cols(ranked)
        st.dataframe(style_table(ranked[cols_to_show].head(top_n)),
                     use_container_width=True, hide_index=True)

        # Quick visual — color by Sharpe(y) μ when present, else Sharpe (m)
        topdf = ranked.head(top_n).copy()
        topdf["Label"] = (topdf["Strike"] + " · " + topdf["Tenor"]
                          + " · " + topdf["KO"] + " · "
                          + topdf["Gate"].str.replace(r"\s+", " ", regex=True))
        color_col = ("Sharpe(y) μ" if "Sharpe(y) μ" in topdf.columns
                     and topdf["Sharpe(y) μ"].notna().any()
                     else "Sharpe (m)")
        hover = [c for c in ["Σ PnL", "Win%", "KO%", "Max DD", "Recovery%",
                             "Calmar", "G2P", "Ulcer", "Min Ann $", "%Pos Yrs"]
                 if c in topdf.columns]
        fig = px.bar(
            topdf[::-1], x=metric, y="Label", orientation="h",
            color=color_col, color_continuous_scale="RdYlGn",
            hover_data=hover,
            title=f"Top {top_n} strategies by {metric_options[metric]['label']}",
        )
        fig.update_layout(height=max(360, 24 * len(topdf)),
                          yaxis_title="", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# Tab: Per-dimension impact
# ===========================================================================
with tab_dim:
    st.subheader("Per-dimension impact")
    st.caption(
        "Aggregates over all other dimensions to show how each individual "
        "axis (Strike / Tenor / KO / Gate) affects performance. Use this to "
        "isolate which knob is doing the work."
    )

    _avail = available_metrics(f)
    impact_metric = st.selectbox(
        "Metric",
        list(_avail.keys()),
        format_func=lambda k: _avail[k]["label"],
        index=0, key="dim_metric",
    )
    agg_func = st.radio("Aggregate", ["mean", "median"],
                        horizontal=True, key="dim_agg")

    dims = [("Strike", strike_to_numeric),
            ("Tenor", lambda t: TENOR_TO_DAYS.get(str(t).upper(), 9999)),
            ("KO", ko_to_numeric),
            ("Gate", lambda g: g)]
    # Skip the Gate dimension when there are no real gates in the data
    if f["Gate"].nunique() <= 1:
        dims = [d for d in dims if d[0] != "Gate"]

    cols = st.columns(2)
    for i, (dim, key) in enumerate(dims):
        with cols[i % 2]:
            grp = (f.groupby(dim)[impact_metric]
                     .agg(agg_func).reset_index()
                     .dropna(subset=[impact_metric]))
            if grp.empty:
                st.info(f"No data for {dim}.")
                continue
            grp = grp.sort_values(dim, key=lambda s: s.map(key))
            grp["_color"] = grp[impact_metric]
            fig = px.bar(
                grp, x=dim, y=impact_metric,
                color="_color", color_continuous_scale="RdYlGn",
                title=f"{agg_func.title()} {impact_metric} by {dim}",
            )
            fig.update_layout(
                showlegend=False, coloraxis_showscale=False,
                height=320, margin=dict(l=10, r=10, t=50, b=10),
                xaxis_title="", yaxis_title=impact_metric,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Marginal sensitivity: range vs. mean")
    st.caption(
        "Range (max − min) across each dimension's aggregate — bigger range "
        "= that dimension matters more for the chosen metric."
    )
    ranges = []
    _hb = METRIC_SPECS.get(impact_metric, {"higher": True})["higher"]
    for dim, key in dims:
        grp = (f.groupby(dim)[impact_metric]
                 .agg(agg_func).dropna())
        if len(grp) >= 2:
            ranges.append({"Dimension": dim,
                           "Range": grp.max() - grp.min(),
                           "Min": grp.min(), "Max": grp.max(),
                           "Best level": grp.idxmax() if _hb else grp.idxmin()})
    if ranges:
        rdf = pd.DataFrame(ranges).sort_values("Range", ascending=False)
        rfig = px.bar(rdf, x="Range", y="Dimension", orientation="h",
                      color="Range", color_continuous_scale="Viridis",
                      hover_data=["Min", "Max", "Best level"],
                      title=f"Which dimension moves {impact_metric} the most?")
        rfig.update_layout(height=300, coloraxis_showscale=False,
                           margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(rfig, use_container_width=True)


# ===========================================================================
# Tab: Heatmaps
# ===========================================================================
with tab_heat:
    st.subheader("Heatmaps — two-dimensional interactions")

    axes = ["Strike", "Tenor", "KO"]
    if f["Gate"].nunique() > 1:
        axes.append("Gate")

    c1, c2, c3, c4 = st.columns(4)
    x_axis = c1.selectbox("X-axis", axes, index=0, key="hm_x")
    y_axis = c2.selectbox(
        "Y-axis",
        [a for a in ["Tenor", "Strike", "KO", "Gate"] if a in axes],
        index=0, key="hm_y",
    )
    _avail = available_metrics(f)
    heat_metric = c3.selectbox(
        "Metric",
        list(_avail.keys()),
        format_func=lambda k: _avail[k]["label"],
        index=0, key="hm_metric",
    )
    heat_agg = c4.radio("Aggregate", ["mean", "median", "max"],
                        horizontal=True, key="hm_agg")

    if x_axis == y_axis:
        st.warning("Choose two different axes.")
    else:
        key_map = {"Strike": strike_to_numeric,
                   "KO": ko_to_numeric,
                   "Tenor": lambda t: TENOR_TO_DAYS.get(str(t).upper(), 9999),
                   "Gate": lambda g: g}
        pivot = (f.pivot_table(values=heat_metric, index=y_axis,
                               columns=x_axis, aggfunc=heat_agg))
        if pivot.empty:
            st.warning("Not enough data for this combination.")
        else:
            pivot = pivot.reindex(
                index=sorted(pivot.index, key=key_map[y_axis]),
                columns=sorted(pivot.columns, key=key_map[x_axis]),
            )
            # Lower-is-better metrics get a reversed color scale
            reverse = not METRIC_SPECS.get(heat_metric, {"higher": True})["higher"]
            scale = "RdYlGn_r" if reverse else "RdYlGn"
            fig = px.imshow(
                pivot, text_auto=".2f", aspect="auto",
                color_continuous_scale=scale,
                labels=dict(color=heat_metric),
                title=f"{heat_agg.title()} {heat_metric}: {y_axis} × {x_axis}",
            )
            fig.update_layout(height=max(360, 32 * len(pivot.index) + 100),
                              margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Show numeric pivot"):
                st.dataframe(pivot.round(2), use_container_width=True)


# ===========================================================================
# Tab: Gate uplift
# ===========================================================================
with tab_gate:
    st.subheader("Gate uplift vs. ungated baseline")
    st.caption(
        "For each (Strike, Tenor, KO), compute the metric without a gate "
        "and with each gate, then show the **average uplift** the gate adds."
    )

    has_baseline = "None" in f["Gate"].unique()
    has_real_gates = (f["Gate"] != "None").any()
    if not has_baseline:
        st.info("No ungated baseline strategies in current filter. "
                "Include Gate = 'None' to enable this view.")
    elif not has_real_gates:
        st.info("This dataset contains only ungated strategies — "
                "no gates to compare. (Load a file with gate variants "
                "to see uplift analysis.)")
    else:
        _avail = available_metrics(
            f, only=["Sharpe (m)", "Σ PnL", "Win%", "%Pos Yrs", "Recovery%",
                     "Sharpe(y) μ", "Sharpe(y) min", "Calmar", "G2P",
                     "PnL/Premium", "PnL/|DD|", "Min Ann $",
                     "KO%", "Ulcer", "Max DD"],
        )
        gate_metric = st.selectbox(
            "Metric",
            list(_avail.keys()),
            format_func=lambda k: _avail[k]["label"],
            index=0, key="gate_metric",
        )
        key_cols = ["Strike", "Tenor", "KO"]
        baseline = (f[f["Gate"] == "None"]
                    .set_index(key_cols)[gate_metric]
                    .rename("baseline"))
        merged = (f[f["Gate"] != "None"]
                  .merge(baseline, on=key_cols, how="left"))
        merged["uplift"] = merged[gate_metric] - merged["baseline"]
        merged["uplift_pct"] = np.where(
            merged["baseline"].abs() > 1e-12,
            100 * (merged[gate_metric] - merged["baseline"]) / merged["baseline"].abs(),
            np.nan,
        )

        gate_summary = (merged.groupby("Gate")
                              .agg(avg_uplift=("uplift", "mean"),
                                   med_uplift=("uplift", "median"),
                                   avg_uplift_pct=("uplift_pct", "mean"),
                                   pct_better=("uplift",
                                               lambda s: 100 * (s > 0).mean()),
                                   n=("uplift", "count"))
                              .reset_index()
                              .sort_values("avg_uplift", ascending=False))

        reverse = not METRIC_SPECS.get(gate_metric, {"higher": True})["higher"]
        if reverse:
            gate_summary = gate_summary.sort_values("avg_uplift")

        fig = px.bar(
            gate_summary, x="avg_uplift", y="Gate", orientation="h",
            color="avg_uplift",
            color_continuous_scale="RdYlGn_r" if reverse else "RdYlGn",
            hover_data=["med_uplift", "avg_uplift_pct", "pct_better", "n"],
            title=f"Average uplift in {gate_metric} vs. ungated baseline",
        )
        fig.update_layout(height=max(300, 36 * len(gate_summary)),
                          margin=dict(l=10, r=10, t=50, b=10),
                          coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        show = gate_summary.copy()
        _is_dollar = gate_metric in ["Σ PnL", "Max DD", "Min Ann $"]
        show["avg_uplift"] = show["avg_uplift"].apply(
            lambda v: fmt_usd(v) if _is_dollar else fmt_float(v, 3))
        show["med_uplift"] = show["med_uplift"].apply(
            lambda v: fmt_usd(v) if _is_dollar else fmt_float(v, 3))
        show["avg_uplift_pct"] = show["avg_uplift_pct"].apply(fmt_pct)
        show["pct_better"] = show["pct_better"].apply(fmt_pct)
        show = show.rename(columns={
            "avg_uplift": "Avg uplift",
            "med_uplift": "Median uplift",
            "avg_uplift_pct": "Avg uplift %",
            "pct_better": "% of (S,T,KO) where gate helped",
            "n": "# comparisons",
        })
        st.dataframe(show, use_container_width=True, hide_index=True)


# ===========================================================================
# Tab: Pareto frontier
# ===========================================================================
with tab_pareto:
    st.subheader("Risk-return Pareto frontier")
    c1, c2 = st.columns(2)

    # Risk-side metrics: Max DD and Min Ann $ are signed (negative = worse) and
    # need sign-flipping for the "minimize x" Pareto logic.
    risk_candidates = ["Max DD", "Min Ann $", "KO%", "Ulcer", "TX/Premium %"]
    risk_axes = [c for c in risk_candidates
                 if c in f.columns and f[c].notna().any()]
    reward_candidates = ["Σ PnL", "Sharpe (m)", "Sharpe(y) μ", "Sharpe(y) min",
                         "Calmar", "G2P", "Win%", "%Pos Yrs", "Recovery%",
                         "PnL/Premium", "PnL/|DD|"]
    reward_axes = [c for c in reward_candidates
                   if c in f.columns and f[c].notna().any()]

    px_x = c1.selectbox("X (cost / risk)", risk_axes, index=0)
    px_y = c2.selectbox("Y (reward)", reward_axes, index=0)

    plot_df = f.dropna(subset=[px_x, px_y]).copy()
    if plot_df.empty:
        st.warning("Not enough data.")
    else:
        # Signed-negative risk metrics: less negative = better, so flip sign
        # for the minimize-x Pareto sweep.
        flip_x = px_x in {"Max DD", "Min Ann $", "Sharpe(y) min"}
        x_vals = -plot_df[px_x] if flip_x else plot_df[px_x]
        y_vals = plot_df[px_y]

        # Pareto: minimize x (after flip if needed), maximize y
        # We want non-dominated points: no other has lower x AND higher y.
        order = np.lexsort((-y_vals.values, x_vals.values))  # asc x, desc y
        best_y = -np.inf
        is_pareto = np.zeros(len(plot_df), dtype=bool)
        for idx in order:
            if y_vals.iloc[idx] > best_y:
                best_y = y_vals.iloc[idx]
                is_pareto[idx] = True
        plot_df["Pareto"] = is_pareto

        plot_df["Label"] = (plot_df["Strike"] + " · " + plot_df["Tenor"]
                            + " · " + plot_df["KO"] + " · "
                            + plot_df["Gate"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df.loc[~plot_df["Pareto"], px_x],
            y=plot_df.loc[~plot_df["Pareto"], px_y],
            mode="markers",
            marker=dict(size=6, color="lightgray", opacity=0.55),
            name="Dominated",
            text=plot_df.loc[~plot_df["Pareto"], "Label"],
            hovertemplate="%{text}<br>" + px_x + ": %{x}<br>"
                          + px_y + ": %{y}<extra></extra>",
        ))
        pareto = plot_df[plot_df["Pareto"]].copy()
        # Sort the frontier line: for flipped risk axes, higher x = better/safer
        pareto = pareto.sort_values(px_x, ascending=not flip_x)
        fig.add_trace(go.Scatter(
            x=pareto[px_x], y=pareto[px_y],
            mode="markers+lines",
            marker=dict(size=10, color="crimson",
                        line=dict(width=1, color="black")),
            line=dict(color="crimson", width=2, dash="dot"),
            name="Pareto-efficient",
            text=pareto["Label"],
            hovertemplate="%{text}<br>" + px_x + ": %{x}<br>"
                          + px_y + ": %{y}<extra></extra>",
        ))
        fig.update_layout(
            title=f"{px_y} vs. {px_x} — Pareto frontier in red",
            xaxis_title=px_x, yaxis_title=px_y, height=560,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Pareto-efficient strategies**")
        st.dataframe(style_table(pareto[table_cols(pareto)]),
                     use_container_width=True, hide_index=True)


# ===========================================================================
# Tab: Side-by-side compare
# ===========================================================================
with tab_compare:
    st.subheader("Side-by-side comparison")
    st.caption("Pick any two strategies from the filtered set and diff them.")
    f2 = f.copy()
    f2["Label"] = (f2["Strike"] + " · " + f2["Tenor"]
                   + " · " + f2["KO"] + " · " + f2["Gate"])
    options = f2["Label"].tolist()
    c1, c2 = st.columns(2)
    a = c1.selectbox("Strategy A", options, index=0, key="cmp_a")
    b_idx = 1 if len(options) > 1 else 0
    b = c2.selectbox("Strategy B", options, index=b_idx, key="cmp_b")

    A = f2[f2["Label"] == a].iloc[0]
    B = f2[f2["Label"] == b].iloc[0]

    # Show every metric that exists in the data, in a sensible order
    metric_order = ["n", "Yrs", "Feas%", "KO%", "Win%", "%Pos Yrs",
                    "Σ Premium", "Σ TX Cost", "Σ Payout", "Σ PnL",
                    "Sharpe (m)", "Sharpe(y) μ", "Sharpe(y) min",
                    "Calmar", "G2P", "Ulcer",
                    "Max DD", "Min Ann $", "Recovery%",
                    "PnL/Premium", "PnL/|DD|", "TX/Premium %"]
    metrics = [m for m in metric_order if m in f.columns]

    rows = []
    dollar_set = {"Σ Premium", "Σ TX Cost", "Σ Payout", "Σ PnL",
                  "Max DD", "Min Ann $"}
    pct_set = {"Feas%", "KO%", "Win%", "%Pos Yrs", "Recovery%", "TX/Premium %"}
    for m in metrics:
        va, vb = A.get(m, np.nan), B.get(m, np.nan)
        if pd.isna(va) and pd.isna(vb):
            continue
        diff = (vb - va) if (pd.notna(va) and pd.notna(vb)) else np.nan
        if m in dollar_set:
            fa, fb, fd = fmt_usd(va), fmt_usd(vb), fmt_usd(diff)
        elif m in pct_set:
            fa, fb, fd = fmt_pct(va), fmt_pct(vb), (
                "—" if pd.isna(diff) else f"{diff:+.1f} pp")
        elif m in ("n", "Yrs"):
            fa = f"{int(va):,}" if pd.notna(va) else "—"
            fb = f"{int(vb):,}" if pd.notna(vb) else "—"
            fd = "—" if pd.isna(diff) else f"{int(diff):+,}"
        else:
            fa, fb = fmt_float(va, 3), fmt_float(vb, 3)
            fd = "—" if pd.isna(diff) else f"{diff:+.3f}"
        rows.append({"Metric": m, "A": fa, "B": fb, "B − A": fd})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ===========================================================================
# Tab: Data
# ===========================================================================
with tab_data:
    st.subheader("Filtered data")
    cols = table_cols(f)
    st.dataframe(style_table(f[cols]), use_container_width=True,
                 hide_index=True, height=600)

    extra = [c for c in ["Pair", "OptType"] if c in f.columns]
    csv_bytes = f[cols + extra].to_csv(index=False).encode()
    st.download_button("⬇️ Download filtered CSV", data=csv_bytes,
                       file_name=f"{pair}_filtered.csv", mime="text/csv")

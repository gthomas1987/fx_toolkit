"""Currency Screener — Worst-Of / Dual-Digital pair screener.

Sub-app of the FX Toolkit. Reached via the landing page or sidebar nav;
not run directly.

Screens FX cross-pairs for the correlation structure that makes them
attractive worst-of and dual-digital option candidates:

  * HIGH multi-day correlation        -> assets drift together
  * LOW daily correlation             -> day-to-day moves decorrelated
  * Cointegration                     -> shared stochastic trend
  * Mean-reverting log-spread (VR<1)  -> noise dominates daily, signal at horizon

The wedge between long-horizon and daily correlation is the trade: dealer
vol surfaces calibrate off daily correlation, but terminal payoff depends
on multi-day co-movement.

Four tabs: Ranked table · Pair matrix · Pair drill-down · Learn.

Originally a standalone Streamlit app; ported into the toolkit with no
math changes — only the data-folder resolution (now uses the toolkit's
shared `data_dir_input`) and the page chrome were updated.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sibling top-level packages (core/, shared/) importable when
# Streamlit executes this file out of the project root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import coint

from core.ui import data_dir_input, app_header
from shared.style import inject_base_css

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
# Default list of pairs to consider when no `_index.csv` is available.
# When an index IS present (the toolkit convention), the pair universe
# is auto-discovered from its SPOT rows instead — see `_resolve_pairs`.
DEFAULT_PAIRS = [
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD",
    "USDCHF", "USDJPY", "USDCAD", "USDNOK", "USDSEK",
    "USDCNH", "USDINR", "USDIDR", "USDPHP", "USDKRW",
    "USDTHB", "USDHKD", "USDSGD", "USDMYR", "USDTWD",
]

HORIZONS = [1, 5, 10, 20, 60]

st.set_page_config(
    page_title="Currency Screener",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_base_css()


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_index(data_dir: str) -> dict[str, str]:
    """
    Read _index.csv from the market_data folder and build a {pair: filename}
    map for SPOT rows only.

    Expected schema (Currency Dashboard 2 convention):
        bbg_ticker, csv_filename, pair, region, onshore_offshore,
        category, tenor, field, in_data, n_rows, first_date, last_date

    Returns {} if the file is missing or unparseable, in which case callers
    fall back to a direct {pair}.csv lookup.
    """
    idx_path = Path(data_dir) / "_index.csv"
    if not idx_path.exists():
        return {}
    try:
        df = pd.read_csv(idx_path)
    except Exception:
        return {}
    if df.empty:
        return {}

    # Be tolerant if column names have unexpected casing/whitespace
    df.columns = [str(c).strip() for c in df.columns]
    needed = {"pair", "csv_filename", "category"}
    if not needed.issubset({c.lower() for c in df.columns}):
        return {}
    rename = {c: c.lower() for c in df.columns if c.lower() in
              {"pair", "csv_filename", "category", "in_data"}}
    df = df.rename(columns=rename)

    spots = df[df["category"].astype(str).str.upper() == "SPOT"].copy()
    if "in_data" in spots.columns:
        # Drop rows flagged as not actually present on disk
        spots = spots[spots["in_data"].astype(str).str.lower().isin(
            ("true", "1", "yes", "y"))]

    mapping: dict[str, str] = {}
    for _, row in spots.iterrows():
        pair = str(row["pair"]).strip().upper()
        fname = str(row["csv_filename"]).strip()
        if not pair or pair == "NAN" or not fname or fname.lower() == "nan":
            continue
        if not fname.lower().endswith(".csv"):
            fname = fname + ".csv"
        mapping[pair] = fname
    return mapping


@st.cache_data(show_spinner=False)
def load_spot(pair: str, data_dir: str, filename: str | None = None) -> pd.Series | None:
    """
    Load a spot price series for `pair` from `data_dir`.

    Always reads the `Close` column (per user instruction), regardless of
    whether the file has Date+Close only or full Date,Open,High,Low,Close.

    `filename` is the resolved CSV filename (typically from _index.csv).
    If omitted, falls back to `{pair}.csv`.
    """
    if filename:
        path = Path(data_dir) / filename
        if not path.exists():
            # final fallback if the index points to a missing file
            alt = Path(data_dir) / f"{pair}.csv"
            if alt.exists():
                path = alt
            else:
                return None
    else:
        path = Path(data_dir) / f"{pair}.csv"
        if not path.exists():
            return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None

    df.columns = [str(c).strip() for c in df.columns]

    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    # Always prefer the Close column. Bloomberg PX_LAST exports may use other
    # names, so accept a small whitelist of synonyms as fallback.
    close_col = next((c for c in df.columns if c.lower() == "close"), None)
    if close_col is None:
        close_col = next(
            (c for c in df.columns
             if c.lower() in ("px_last", "pxlast", "last", "price", "value")),
            None,
        )
    if date_col is None or close_col is None:
        return None

    # Date parsing: the toolkit convention is ISO 8601 (YYYY-MM-DD) for
    # all CSVs in `_index.csv` folders. Note: do NOT pass `dayfirst=True`
    # here — that would misinterpret ISO 8601 dates like '2023-05-01' as
    # January 5, 2023.
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    s = pd.to_numeric(df[close_col], errors="coerce").rename(pair).dropna()
    if s.empty:
        return None
    return s[~s.index.duplicated(keep="last")]


@st.cache_data(show_spinner=True)
def load_all_spots(
    data_dir: str,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """
    Load every SPOT pair found in `data_dir`'s `_index.csv`. If no index
    is present, fall back to looking for the conventional set of pairs
    as direct `{pair}.csv` files.

    Returns
    -------
    spots : DataFrame of close prices, one column per pair
    missing : list of pairs that could not be loaded
    index_map : the {pair: filename} mapping used (empty dict if no
                _index.csv was found — useful for diagnostics in the UI)
    """
    index_map = load_index(data_dir)
    # Pair universe: if _index.csv is present, use ALL of its SPOT pairs
    # (the toolkit convention — your folder defines what you analyze).
    # Otherwise fall back to the conventional G10 + Asia EM list.
    pairs_to_try = list(index_map.keys()) if index_map else list(DEFAULT_PAIRS)
    found: dict[str, pd.Series] = {}
    missing: list[str] = []
    for p in pairs_to_try:
        fname = index_map.get(p)  # None → load_spot falls back to {p}.csv
        s = load_spot(p, data_dir, fname)
        if s is not None and len(s) > 0:
            found[p] = s
        else:
            missing.append(p)
    if not found:
        return pd.DataFrame(), missing, index_map
    return pd.DataFrame(found).sort_index(), missing, index_map


# -----------------------------------------------------------------------------
# Stat functions
# -----------------------------------------------------------------------------
def horizon_corr(r1: pd.Series, r2: pd.Series, h: int) -> float:
    """Pearson correlation of h-day overlapping log returns."""
    if h == 1:
        return float(r1.corr(r2))
    r1h = r1.rolling(h).sum()
    r2h = r2.rolling(h).sum()
    return float(r1h.corr(r2h))


def variance_ratio(spread: pd.Series, q: int) -> tuple[float, float]:
    """
    Lo-MacKinlay (1988) heteroscedasticity-consistent variance ratio test.

    Returns
    -------
    (VR, z_stat)
        VR < 1  => mean-reverting spread
        VR ~ 1  => random walk
        |z| > 1.96 => statistically distinguishable from RW at 5%
    """
    s = spread.dropna()
    if len(s) < q * 4:
        return np.nan, np.nan
    r = s.diff().dropna()
    T = len(r)
    rq = s.diff(q).dropna()
    var_1 = float(r.var(ddof=1))
    if var_1 <= 0:
        return np.nan, np.nan
    vr = float(rq.var(ddof=1) / (q * var_1))

    eps = (r - r.mean()).values
    eps_sq = eps ** 2
    sum_eps_sq = float(eps_sq.sum())
    if sum_eps_sq <= 0:
        return vr, np.nan
    theta = 0.0
    for j in range(1, q):
        delta_j = T * float((eps_sq[j:] * eps_sq[:-j]).sum()) / (sum_eps_sq ** 2)
        theta += ((2 * (q - j) / q) ** 2) * delta_j
    if theta <= 0:
        return vr, np.nan
    z = (vr - 1.0) / np.sqrt(theta / T)
    return vr, float(z)


def engle_granger(p1: pd.Series, p2: pd.Series) -> tuple[float, float]:
    """Engle-Granger cointegration test on log-prices. Returns (p-value, beta)."""
    df = pd.concat([np.log(p1), np.log(p2)], axis=1).dropna()
    if len(df) < 100:
        return np.nan, np.nan
    y = df.iloc[:, 0].values
    x = df.iloc[:, 1].values
    try:
        _, pvalue, _ = coint(y, x, autolag="AIC")
        pvalue = float(pvalue)
    except Exception:
        pvalue = np.nan
    beta = float(np.polyfit(x, y, 1)[0])
    return pvalue, beta


def screen_pair(spots: pd.DataFrame, a: str, b: str) -> dict | None:
    """Compute all stats for one cross-pair (a, b) of FX pairs."""
    p1 = spots[a].dropna()
    p2 = spots[b].dropna()
    common = p1.index.intersection(p2.index)
    if len(common) < 250:
        return None
    p1c = p1.loc[common]
    p2c = p2.loc[common]
    r1 = np.log(p1c).diff().dropna()
    r2 = np.log(p2c).diff().dropna()
    common_r = r1.index.intersection(r2.index)
    r1, r2 = r1.loc[common_r], r2.loc[common_r]

    out: dict = {"Pair_A": a, "Pair_B": b, "N": len(common_r)}
    for h in HORIZONS:
        out[f"rho_{h}d"] = horizon_corr(r1, r2, h)
    out["score"] = out["rho_20d"] - out["rho_1d"]
    out["abs_score"] = abs(out["score"])

    pval, beta = engle_granger(p1c, p2c)
    out["coint_pval"] = pval
    out["beta_log"] = beta

    if beta is not None and not np.isnan(beta):
        log_spread = np.log(p1c) - beta * np.log(p2c)
        vr, z = variance_ratio(log_spread, 20)
        out["VR_20"] = vr
        out["VR_z"] = z
    else:
        out["VR_20"] = np.nan
        out["VR_z"] = np.nan
    return out


@st.cache_data(show_spinner=False)
def run_screener(
    spots: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Run screener across all C(n,2) pair combinations within [start, end]."""
    sl = spots.loc[start:end]
    rows = []
    combos = list(combinations(sl.columns, 2))
    progress = st.progress(0.0, text="Screening pairs…")
    for i, (a, b) in enumerate(combos):
        r = screen_pair(sl, a, b)
        if r is not None:
            rows.append(r)
        if i % 5 == 0:
            progress.progress((i + 1) / len(combos))
    progress.empty()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values("score", ascending=False).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def plot_term_structure(r1: pd.Series, r2: pd.Series, a: str, b: str) -> go.Figure:
    rhos = [horizon_corr(r1, r2, h) for h in HORIZONS]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=HORIZONS, y=rhos, mode="lines+markers",
                   line=dict(width=3), marker=dict(size=10))
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title=f"Correlation term structure  —  {a} vs {b}",
        xaxis_title="Horizon (days, log scale)",
        yaxis_title="Pearson ρ of overlapping log returns",
        xaxis_type="log",
        height=380,
        template="plotly_white",
    )
    return fig


def plot_normalized_prices(p1: pd.Series, p2: pd.Series, a: str, b: str) -> go.Figure:
    df = pd.concat([p1, p2], axis=1).dropna()
    df = df / df.iloc[0] * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[a], name=a))
    fig.add_trace(go.Scatter(x=df.index, y=df[b], name=b))
    fig.update_layout(
        title="Normalized levels (start = 100)",
        height=320, template="plotly_white",
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def plot_spread(p1: pd.Series, p2: pd.Series, beta: float, a: str, b: str) -> go.Figure:
    df = pd.concat([np.log(p1), np.log(p2)], axis=1).dropna()
    spread = df.iloc[:, 0] - beta * df.iloc[:, 1]
    z = (spread - spread.rolling(252, min_periods=63).mean()) / \
        spread.rolling(252, min_periods=63).std()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=[
                            f"Log-spread:  log({a}) − {beta:.3f}·log({b})",
                            "Spread z-score (rolling 1Y)"],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=spread.index, y=spread.values, name="spread"), 1, 1)
    fig.add_hline(y=spread.mean(), line_dash="dot", line_color="gray", row=1, col=1)
    fig.add_trace(go.Scatter(x=z.index, y=z.values, name="z"), 2, 1)
    for level, color in [(2, "red"), (-2, "red"), (0, "gray")]:
        fig.add_hline(y=level, line_dash="dot", line_color=color, row=2, col=1)
    fig.update_layout(height=520, template="plotly_white", showlegend=False)
    return fig


def plot_rolling_corr(r1: pd.Series, r2: pd.Series, a: str, b: str) -> go.Figure:
    win_d, win_w = 60, 60  # 60-day rolling windows
    daily = r1.rolling(win_d).corr(r2)
    r1_5 = r1.rolling(5).sum()
    r2_5 = r2.rolling(5).sum()
    weekly = r1_5.rolling(win_w).corr(r2_5)
    r1_20 = r1.rolling(20).sum()
    r2_20 = r2.rolling(20).sum()
    monthly = r1_20.rolling(win_w).corr(r2_20)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.index, y=daily.values, name="ρ(1d), 60d win"))
    fig.add_trace(go.Scatter(x=weekly.index, y=weekly.values, name="ρ(5d), 60d win"))
    fig.add_trace(go.Scatter(x=monthly.index, y=monthly.values, name="ρ(20d), 60d win"))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="Rolling realized correlation by horizon",
        yaxis_title="ρ", height=380, template="plotly_white",
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def plot_score_heatmap(results: pd.DataFrame, metric: str = "score") -> go.Figure:
    # Use whichever pairs actually appear in the results (not a hard-coded
    # list). Sorted for stable axis ordering.
    pairs_in_results = sorted(
        set(results["Pair_A"]).union(set(results["Pair_B"]))
    )
    mat = pd.DataFrame(index=pairs_in_results, columns=pairs_in_results,
                        dtype=float)
    for _, row in results.iterrows():
        a, b = row["Pair_A"], row["Pair_B"]
        mat.loc[a, b] = row[metric]
        mat.loc[b, a] = row[metric]
    fig = px.imshow(
        mat.values, x=mat.columns, y=mat.index,
        color_continuous_scale="RdBu_r",
        zmin=-mat.abs().max().max(),
        zmax=mat.abs().max().max(),
        aspect="auto",
    )
    fig.update_layout(
        title=f"Pair matrix: {metric}",
        height=600, template="plotly_white",
    )
    return fig


# -----------------------------------------------------------------------------
# Educational content — help expanders
# -----------------------------------------------------------------------------
def help_overview() -> None:
    with st.expander("📚  **Start here** — What this screener does and why", expanded=False):
        st.markdown(r"""
### The thesis in one paragraph

Some pairs of FX underlyings **drift in the same direction over weeks and months**
but their **day-to-day moves are largely independent**. When a dealer prices a
worst-of option or a dual digital, they typically calibrate correlation from
**daily** returns, because that's where the liquid data is. For an option that
pays at expiry (1M, 3M …) what actually matters is the **terminal** correlation
of the two assets. If terminal correlation is materially higher than implied
daily correlation, the option is mispriced cheap — buying it is positive expected
value.

This screener systematically scans all $\binom{19}{2} = 171$ combinations of the
G10 + EM Asia universe to find the candidates with the biggest wedge.

### What you'll see

For each pair $(A, B)$ the screener computes:

1. **Correlation at multiple horizons** — $\rho(1d), \rho(5d), \rho(10d),
   \rho(20d), \rho(60d)$. The **shape** of this curve is the core signal.
2. **Score** — defined as $\rho(20d) - \rho(1d)$. The "wedge" between
   one-month and daily realized correlation. Bigger is better.
3. **Cointegration test** — formal statistical test that the two
   log-price series share a common stochastic trend (not just coincidence).
4. **Variance ratio** — measures whether the constructed log-spread
   mean-reverts. A mean-reverting spread is corroborating evidence that
   the "drift together" relationship is real.

### Why this works mechanically (preview, math below)

If two log-prices share a common I(1) random-walk component plus stationary
idiosyncratic noise:

$$\log P^A_t = X_t + \epsilon^A_t, \qquad \log P^B_t = \beta X_t + \epsilon^B_t$$

where $X_t$ is a random walk and $\epsilon^A, \epsilon^B$ are mean-reverting,
then the variance of the $h$-day return decomposes as:

$$\text{Var}\!\left[r^A_t(h)\right] = h\,\sigma^2_{\Delta X} +
\text{Var}\!\left[\epsilon^A_{t+h} - \epsilon^A_t\right]$$

The first term **grows linearly in $h$**; the second term is **bounded**
(because $\epsilon$ is stationary, so $\epsilon_{t+h} - \epsilon_t$ has finite
variance). So as $h$ grows, the **shared random-walk component dominates the
variance**, and the correlation $\rho(h) \to \pm 1$. At $h=1$, the
idiosyncratic noise is still loud relative to the common trend → $\rho(1)$ is
small. That's exactly the term-structure shape we screen for.
""")


def help_table_columns() -> None:
    with st.expander("📖  How to read the table — column by column"):
        st.markdown(r"""
| Column | Definition | What to look for |
|---|---|---|
| `Pair_A`, `Pair_B` | The two FX underlyings | — |
| `N` | Number of overlapping observations | $\geq 500$ ideally for inference |
| `rho_1d` | $\text{Corr}(r^A_t, r^B_t)$ on daily log returns | **Low** for good candidates |
| `rho_5d` | Correlation of 5-day overlapping log returns | Intermediate |
| `rho_10d` | Same at 10 days | Intermediate |
| `rho_20d` | Same at 20 days (~ 1 month) | **High** for good candidates |
| `rho_60d` | Same at 60 days (~ 3 months) | Confirmation of long-horizon co-movement |
| `score` | $\rho(20d) - \rho(1d)$ — the wedge | **As high as possible** |
| `abs_score` | Magnitude of score (useful for sorting either-sign trades) | — |
| `coint_pval` | Engle-Granger cointegration p-value | **Low** ($< 0.05$ ideal, $< 0.20$ minimum) |
| `beta_log` | OLS hedge ratio on log-prices | Magnitude tells you how to construct spread |
| `VR_20` | Variance ratio of the log-spread at $q=20$ | **$< 1$** indicates mean-reverting spread |
| `VR_z` | Het-consistent z-stat for $H_0: VR=1$ | $|z| > 1.96$ is significant at 5% |

**The "perfect" candidate has:**
- High $|\rho(20d)|$, low $|\rho(1d)|$ → big `score`
- `coint_pval` $< 0.05$ → genuine shared trend, not coincidence
- `VR_20` well below 1 with significant `VR_z` → tradable mean-reverting spread
""")


def help_math_correlation() -> None:
    with st.expander("📐  The math — Correlation term structure $\\rho(h)$"):
        st.markdown(r"""
### Definition

Let $r_t = \log(P_t / P_{t-1})$ be the daily log return. The $h$-day
overlapping log return is just the sum:

$$r_t(h) = \sum_{i=0}^{h-1} r_{t-i} = \log(P_t / P_{t-h})$$

The $h$-horizon correlation between assets $A$ and $B$ is:

$$\rho(h) = \frac{\text{Cov}\!\left[r^A_t(h),\, r^B_t(h)\right]}
{\sqrt{\text{Var}\!\left[r^A_t(h)\right]\,\text{Var}\!\left[r^B_t(h)\right]}}$$

### Why ρ(h) varies with h

For two **independent random walks**, $\rho(h) = 0$ for all $h$.

For two **perfectly correlated** processes, $\rho(h) = 1$ for all $h$.

The interesting cases — and the ones this screener cares about — are in
between. They generically arise from **cointegration**: two log-price series
that share a common stochastic trend but have independent stationary idiosyncratic
noise.

### The cointegration case in detail

Decompose log-prices as:

$$\log P^A_t = X_t + \epsilon^A_t$$
$$\log P^B_t = \beta X_t + \epsilon^B_t$$

where $X_t$ is a random walk with $\Delta X_t \sim$ iid $(0, \sigma^2_X)$,
and $\epsilon^A_t$, $\epsilon^B_t$ are independent stationary AR(1) processes
with $|\phi| < 1$ and innovation variance $\sigma^2_\epsilon$.

**Daily returns** are:
$$r^A_t = \Delta X_t + \Delta \epsilon^A_t$$

The daily covariance:
$$\text{Cov}(r^A_t, r^B_t) = \beta\,\sigma^2_X$$

The daily variance:
$$\text{Var}(r^A_t) = \sigma^2_X + \frac{2\sigma^2_\epsilon}{1+\phi}$$

So $\rho(1)$ is a number between 0 and 1, with the **idiosyncratic noise
contaminating the correlation**.

**$h$-period returns**:
$$r^A_t(h) = (X_t - X_{t-h}) + (\epsilon^A_t - \epsilon^A_{t-h})$$

The first term has variance $h\,\sigma^2_X$ (random walk increment).
The second term has variance bounded by $2\sigma^2_\epsilon / (1-\phi^2)$
(stationary — **bounded as $h \to \infty$**).

So:
$$\text{Var}(r^A_t(h)) = h\,\sigma^2_X + O(1)$$
$$\text{Cov}(r^A_t(h), r^B_t(h)) = h\,\beta\,\sigma^2_X$$

And:
$$\rho(h) \to \frac{\beta}{|\beta|} = \pm 1 \quad \text{as } h \to \infty$$

**This is the formal statement of "drift together, decorrelate daily."**
The bigger the idiosyncratic noise relative to the common trend, the steeper
the rise of $\rho(h)$ from $\rho(1)$ toward $\pm 1$.

### Visualisation — what ρ(h) curves look like in each case

| $h$ | $\rho(h)$ for "good" cointegrated pair | $\rho(h)$ for unrelated pair | $\rho(h)$ for already-correlated pair |
|---|---|---|---|
| 1 | 0.10 | 0.02 | 0.85 |
| 5 | 0.35 | -0.01 | 0.86 |
| 20 | 0.65 | 0.03 | 0.87 |
| 60 | 0.80 | 0.00 | 0.88 |

The third column (already-correlated pair) is **not interesting** — dealers
already price these tightly. The first column is the prize.

### Implementation note: overlapping returns

When we compute $\rho(20d)$ over $T$ days of data, we use **overlapping**
20-day windows, giving roughly $T - 19$ observations. Overlapping windows
**reuse data**, so they inflate apparent statistical significance. Treat
the headline p-values as suggestive, not definitive. For real inference,
use non-overlapping windows or HAC standard errors (Newey-West).
""")


def help_math_cointegration() -> None:
    with st.expander("🔬  The math — Cointegration (Engle-Granger test)"):
        st.markdown(r"""
### Why we need a separate test

The rising shape of $\rho(h)$ is consistent with cointegration but **not
proof of it**. Two trending series with similar drift can show rising
correlation at long horizons without sharing a stochastic trend
(this is sometimes called "common growth" rather than cointegration).

Cointegration is a stronger and cleaner claim: it says there exists a
linear combination $\log P^A_t - \beta \log P^B_t$ that is **stationary**
(mean-reverting), even though each series individually is non-stationary
(a random walk, or I(1)).

### Engle-Granger procedure

**Step 1.** OLS regress one log-price on the other:
$$\log P^A_t = \alpha + \beta \log P^B_t + e_t$$

Estimate $\hat\beta$ and recover residuals $\hat e_t$.

**Step 2.** Test whether $\hat e_t$ is stationary using an Augmented
Dickey-Fuller (ADF) unit-root test:

$$\Delta \hat e_t = \gamma \hat e_{t-1}
+ \sum_{i=1}^{p} \delta_i \Delta \hat e_{t-i} + u_t$$

Null hypothesis $H_0: \gamma = 0$ (residuals have unit root, **not** cointegrated).
Reject $H_0$ if the test statistic is sufficiently negative.

Critical values for the EG test are **not** the standard ADF critical values —
they're more conservative because $\hat\beta$ was estimated in Step 1. The
`statsmodels.tsa.stattools.coint()` function uses MacKinnon p-values that
account for this.

### Interpretation

| `coint_pval` | Interpretation |
|---|---|
| $< 0.01$ | Very strong evidence of cointegration |
| $0.01 - 0.05$ | Strong evidence |
| $0.05 - 0.10$ | Suggestive — worth a closer look |
| $0.10 - 0.20$ | Weak — needs more sample or a different window |
| $> 0.20$ | No reliable evidence of cointegration |

### What if cointegration fails?

If `coint_pval` is high but `score` is high anyway, several things might be true:

1. The pair really isn't cointegrated and the correlation term structure is
   coincidence over the sample → don't trade.
2. The pair is cointegrated but your sample is too short or contains a
   regime break → check on a different window.
3. The pair is cointegrated nonlinearly (e.g. with a structural break in
   $\beta$) → Engle-Granger misses it. Try Johansen, or split the sample.

### Important caveat — spurious regression

If you run OLS on log-prices and the pair is **not** cointegrated, the
$\hat\beta$ you get is a **spurious regression** estimate. The residuals will
appear stationary in-sample (by construction, OLS minimizes residual variance),
but this is misleading — the "spread" you built is not really mean-reverting in
the population.

**This is why you must trust the `coint_pval` before trusting the `VR_20`
or the spread chart.**
""")


def help_math_variance_ratio() -> None:
    with st.expander("🔄  The math — Variance ratio test (Lo-MacKinlay 1988)"):
        st.markdown(r"""
### Idea

If $\{X_t\}$ is a random walk, then variances of multi-period increments
scale **linearly with horizon**: $\text{Var}(X_t - X_{t-q}) = q \cdot
\text{Var}(X_t - X_{t-1})$. So the **variance ratio**:

$$\text{VR}(q) = \frac{\text{Var}(X_t - X_{t-q})}{q \cdot \text{Var}(X_t - X_{t-1})}$$

equals 1 under the random walk null. Deviations from 1 indicate
**serial correlation** in increments.

| VR(q) | Interpretation |
|---|---|
| $\approx 1$ | Random walk / no autocorrelation |
| $< 1$ | **Negative serial correlation** — mean reversion |
| $> 1$ | **Positive serial correlation** — momentum / trending |

### Application here

We construct the log-spread using the OLS hedge ratio:
$$s_t = \log P^A_t - \hat\beta \log P^B_t$$

and compute $\text{VR}(20)$ on $\{s_t\}$. If the pair is truly cointegrated,
the spread is stationary by construction, and its multi-period variance grows
**sublinearly** — so $\text{VR}(20)$ should be materially less than 1.

### Test statistic (heteroscedasticity-consistent)

The basic test statistic (Lo-MacKinlay 1988, equation (15)):

$$z^*(q) = \sqrt{T}\,(\text{VR}(q) - 1)\,\big/\,\sqrt{\theta^*(q)}$$

where the heteroscedasticity-consistent variance is:

$$\theta^*(q) = \sum_{j=1}^{q-1} \left[\frac{2(q-j)}{q}\right]^2 \delta(j)$$

with

$$\delta(j) = \frac{T \cdot \sum_{t=j+1}^{T} (\Delta s_t - \bar{\mu})^2
(\Delta s_{t-j} - \bar{\mu})^2}
{\left[\sum_{t=1}^{T} (\Delta s_t - \bar{\mu})^2\right]^2}$$

Under the null of a random walk, $z^*(q)$ is asymptotically standard normal.
$|z| > 1.96$ rejects the random-walk null at 5%.

### Interpretation

| `VR_20` value | What it tells you |
|---|---|
| $< 0.5$ with $|z| > 2$ | Strongly mean-reverting spread |
| $0.5 - 0.8$ with $|z| > 2$ | Mean-reverting but with substantial residual noise |
| $0.8 - 1.2$ | Indistinguishable from random walk → spread doesn't mean-revert reliably |
| $> 1.2$ with $|z| > 2$ | Spread is **trending** — bad sign for spread mean reversion |

### Why this matters for the trade

A mean-reverting spread is the empirical signature of cointegration. If
$\text{VR}(20) \ll 1$ **and** `coint_pval` is small, both tests are saying
the same thing: the relationship between $A$ and $B$ has a real, exploitable
mean-reverting component. Convergence over the option's tenor is more likely
than the dealer's daily-correlation-based pricing implies.
""")


def help_what_makes_good() -> None:
    with st.expander("🎯  What makes a good candidate? — practical decision rules"):
        st.markdown(r"""
### Tiered decision rules

**Tier 1 — Strong candidate (worth pitching a trade):**
- `score` $> 0.20$
- `coint_pval` $< 0.05$
- `VR_20` $< 0.7$ with $|VR\_z| > 2$
- `N` $> 1000$ (at least 4 years of overlap)
- Behaviour is **persistent in the rolling-correlation chart** (not driven
  by one regime / crisis episode)

**Tier 2 — Worth investigating further:**
- `score` $\in [0.10, 0.20]$
- `coint_pval` $\in [0.05, 0.20]$
- `VR_20` $\in [0.7, 0.9]$

These need more analysis: check rolling stability, look at different windows,
consider regime classification.

**Tier 3 — Pass:**
- `score` $< 0.10$, OR
- `coint_pval` $> 0.20$, OR
- `VR_20` close to 1, OR
- Rolling correlation shows the relationship breaks down in normal regimes
  (only "works" in crises)

### Common false positives

1. **Both pairs trend similarly because both are USD pairs** — e.g.
   `USDKRW vs USDTWD` during a DXY-strong period. Both went up, but it
   was just "USD strength," not a genuine cointegrating relationship.
   Confirmed by checking what happens across DXY regimes.
2. **One outlier episode dominates the sample** — e.g. COVID-March-2020 or
   the 2022 dollar surge. The rolling-correlation chart should reveal this.
3. **Structural break** — a pair that was cointegrated before a regime
   change (e.g. KRW post-WGBI inclusion) may not be anymore.
4. **Mechanical similarity** — pegs and bands (USDCNH, USDSGD, USDHKD) can
   produce high apparent cointegration with anything because they barely
   move. Watch for $|\beta|$ very far from 1 or near 0.
""")


def help_heatmap() -> None:
    with st.expander("📊  How to read this heatmap"):
        st.markdown(r"""
The matrix shows the selected metric (default: `score`) for every pair.
The color scale is **diverging** (red = positive, blue = negative, white =
zero), so you can immediately spot:

- **Hot red cells** — strong positive wedge candidates (drift together,
  decorrelate daily). These are worst-of and dual-digital "both up" or "both
  down" candidates.
- **Hot blue cells** — strong **negative** wedge candidates. These are pairs
  that move oppositely at long horizons but daily moves are less anticorrelated.
  Useful for cross-direction structures (long EURUSD + short USDKRW dual digital
  for "USD weakness" plays).
- **Clusters** — if you see a block of red across e.g. all USD-Asia pairs
  vs each other, it's the Asia FX cluster co-driving with regional flows.

Try switching the metric to `rho_20d` to see the raw 1-month correlation, or
`VR_20` to see which pairs have the most strongly mean-reverting spreads.
""")


def help_drilldown_charts() -> None:
    with st.expander("🔍  What to look for in the drill-down charts"):
        st.markdown(r"""
### 1. Normalized prices (top chart)

Both series rebased to 100 at the start of the window. **What you want to see:**
the two lines drifting roughly in parallel over multi-month moves, with
short-term wiggles that don't line up. If the lines diverge dramatically with
no convergence, the "cointegration" is probably an in-sample artifact.

### 2. Correlation term structure (left)

The shape of the curve is the core signal. **What you want to see:**
- A **monotonically rising** curve from $\rho(1d)$ to $\rho(60d)$.
- The rise should be at least 0.2 in magnitude (i.e. `score` > 0.2).
- $\rho(60d)$ should be in the ballpark of 0.7+ for a strong candidate.

**Bad shapes to avoid:**
- Flat curve at moderate values → no wedge to exploit.
- Hump shape (rises then falls) → relationship is regime-dependent at long horizons.
- Curve that rises then plateaus around 0.4-0.5 → some shared trend but a lot
  of residual noise that won't be ironed out in your option's tenor.

### 3. Rolling realized correlation (right)

Shows $\rho(1d)$, $\rho(5d)$, $\rho(20d)$ over rolling 60-day windows.
**What you want to see:**
- The lines should be **persistently separated** (long-horizon line above
  short-horizon line, most of the time).
- Avoid pairs where the wedge only appears in one or two short episodes.

This is the most important diagnostic for ruling out **regime-dependent
relationships**. If the wedge only opens during crises (2008, 2020) and
collapses to zero in calm regimes, it's not a tradable opportunity — you
can't time your option entry around regime classification cleanly.

### 4. Log-spread (bottom)

If cointegration is real, the log-spread $\log P^A - \hat\beta \log P^B$ should
look mean-reverting around a roughly constant level. The bottom panel z-scores
the spread on a rolling 1-year window; values outside $\pm 2$ are unusual
deviations and (historically) tend to revert.

**Good sign:** spread oscillates around a flat mean.
**Bad sign:** spread itself has a trend over the sample — that means $\hat\beta$
was a spurious-regression artifact, and the cointegration test should have failed.
""")


def help_trading_apps() -> None:
    with st.expander("💼  Trading applications — Worst-of, Dual Digitals, KO"):
        st.markdown(r"""
### Worst-of options

**Payoff** (worst-of call on $N$ underlyings):
$$\text{payoff} = \max\left[\min_i \big(S_i(T)/S_i(0) - K\big),\ 0\right]$$

For two underlyings $A, B$ and ATM strike ($K=0$):
$$\text{payoff} = \max\big[\min(r^A_T, r^B_T),\ 0\big]$$

where $r^A_T, r^B_T$ are the returns over the option's life.

**Pricing intuition:**
- If $A$ and $B$ are **perfectly positively correlated** at expiry, the
  worst-of call is identical to a vanilla call on the smaller-moving asset
  (no diversification, no payoff drag).
- If $A$ and $B$ are **uncorrelated**, the probability that **both** end ITM
  is the product of individual ITM probabilities. The worst-of pays only when
  both are ITM, so it's much cheaper than a vanilla.
- If they're **anti-correlated**, the worst-of is approximately worthless
  (one is almost always OTM).

So **higher correlation $\to$ higher worst-of price**. Dealer prices it using
their copula calibrated to **daily** realized correlation. If your view is
that **terminal** correlation is materially higher than daily implies,
buying the worst-of is positive EV.

**Pair from screener:** if you find `USDKRW / USDTWD` with $\rho(1d) = 0.2$
but $\rho(20d) = 0.7$, the dealer might price the worst-of call using a
correlation of ~0.3 (with their copula adjustments). The true 1M terminal
correlation is closer to 0.7. Joint ITM probability is much higher than the
dealer's model thinks — you're getting a cheap call.

### Dual digitals

**Payoff:** $\mathbb{1}[S^A_T > K^A] \cdot \mathbb{1}[S^B_T > K^B]$ — pays 1 if
both conditions are met at expiry, 0 otherwise.

**Pricing:** the probability of joint ITM under the risk-neutral measure.
Under bivariate Gaussian returns with correlation $\rho$:
$$P(S^A_T > K^A,\ S^B_T > K^B) = \Phi_2(d^A, d^B; \rho)$$

where $\Phi_2$ is the bivariate normal CDF and $d^A, d^B$ are the (Black-Scholes)
$d_2$ values.

This is **monotonically increasing in $\rho$** when both digitals are OTM
($d^A, d^B < 0$) or both ITM ($d^A, d^B > 0$).

**Edge case worth knowing:** if you're constructing a "diversifying" dual
digital (one OTM call + one OTM put), the payoff probability is monotonically
**decreasing** in $\rho$ — high correlation hurts you. Screener flips the score
sign for these structures (filter by "Opposite direction").

### Knock-out (KO) options — the real prize structure

A KO option ceases to exist if the barrier is hit during the option's life.
KO discount makes the option much cheaper than a vanilla.

**The double-magic of low daily correlation + high terminal correlation for
KO structures:**

1. **Daily correlation determines barrier-hit probability.** If $A$ and $B$
   move independently day-to-day, the probability of **either** hitting its
   KO barrier on any given day is roughly the sum of individual hit
   probabilities, not the product of survival. But the **joint** hit
   probability (both KO simultaneously) is low.
2. **Terminal correlation determines payoff conditional on survival.** Given
   neither has KO'd, the option pays based on terminal levels. High terminal
   correlation → both more likely to be deep ITM together.

In a "worst-of with American KO" structure, the dealer prices the survival
probability using a daily-correlation-based barrier model. If true daily
correlation is lower than the dealer's calibration (e.g. dealer extrapolates
from a 1M window that includes a stressed period), you get more KO discount
than you should. Combined with higher terminal payoff probability, this is a
double-edge structure.

### Example trade construction (illustrative, USDKRW / USDTWD)

- View: 1M USD strength against Asia, low daily co-movement between KRW and
  TWD (KRW driven by tech-cycle and BoK FX management, TWD by semi-cycle
  and CBC interventions — orthogonal day-to-day).
- Screener output: `score = +0.35`, `coint_pval = 0.02`, `VR_20 = 0.4`.
- Trade: long 1M ATM worst-of call on USDKRW + USDTWD (both up = both Asia
  currencies weaken). Strike ATM.
- Why: dealer pricing this off $\rho \approx 0.3$ but true terminal corr more
  like 0.65. Joint upside probability much higher than priced.
- Add KO ~3% out of the money to cheapen further. Daily decorrelation means
  joint barrier hit is rare; survival probability higher than dealer's vol
  surface implies.
""")


def help_caveats() -> None:
    with st.expander("⚠️  Caveats & pitfalls — what can go wrong"):
        st.markdown(r"""
### 1. The Epps effect (microstructure)

The reason daily correlations are mechanically lower than weekly correlations
isn't always cointegration — part of it is the **Epps effect**: non-synchronous
trading, bid-ask bounce, and stale prices cause daily correlations to be biased
down at high frequencies. As sampling frequency falls, the bias decays.

This effect is **strongest for FX pairs with different trading-hour profiles**
(e.g. USDKRW vs EURUSD: Asia-centric vs Europe-centric trading) and **weakest**
when both pairs trade in the same time zone with deep liquidity.

**Mitigation:** Don't trade pairs where the screener wedge is small (`score <
0.10`) and the pairs have very different trading-hour profiles — much of that
wedge is probably Epps, not tradable. Bigger wedges (`score > 0.25`) and same
time-zone pairs are more likely to be real signal.

### 2. Spurious regression

Already covered in detail above. Always trust `coint_pval` before trusting
`VR_20` or the spread chart. **Hard rule:** if `coint_pval > 0.20`, ignore
the spread.

### 3. Regime dependence

A pair's relationship can change dramatically across regimes:
- INR's daily volatility regime changed after RBI's 2024 communication shift.
- KRW's correlation structure shifted after WGBI inclusion (2025).
- TWD had a structural break around the May 2025 CBC episode.
- USDCNH's dynamics depend on PBoC fixing policy regime.

**Mitigation:** Run the screener on multiple windows (5Y, 3Y, 1Y) and require
the candidate to show up in all of them. Use the rolling-correlation chart to
visually confirm persistence.

### 4. Implied vs realized correlation

Dealer marks don't price strictly off realized daily correlation — they apply
adjustments (correlation skew, term structure of implied correlation). For
liquid pairs with active worst-of and basket markets, much of this edge may
**already be priced in**. The cleanest edges are typically in **less liquid
pairs and less standard structures** — e.g. EM Asia FX worst-of with bespoke
strikes, or dual digitals with non-standard barrier conditions.

### 5. Sample size and multiple testing

Testing 171 pairs at the 5% level gives an expected **~9 false positives**
even if no pair is genuinely cointegrated. Don't trust the headline p-values
naively. Two practical fixes:
- Bonferroni-adjust: require $p < 0.05 / 171 \approx 0.0003$ for true 5%
  family-wise error rate (very conservative).
- Use the table as a **hypothesis generator**, then validate each candidate
  out-of-sample on a hold-out window before pitching it.

### 6. Overlapping returns inflate significance

The screener uses overlapping windows for $\rho(h)$. This is fine for getting
the **shape** of the term structure right, but it artificially inflates
apparent statistical significance of any one correlation. For final
inference, recompute on non-overlapping windows.

### 7. Cointegration ≠ tradable

Even if a pair is genuinely cointegrated with a strongly mean-reverting
spread, the **time scale** of mean reversion matters. If the half-life of
the spread is 3 months but your option is 1 month, the option will likely
expire before convergence happens. Compute the half-life of mean reversion
from the AR(1) coefficient on the spread:

$$\hat\phi = \frac{\sum s_t s_{t-1}}{\sum s_{t-1}^2}, \qquad
\text{half-life} = \frac{\ln(0.5)}{\ln(\hat\phi)}$$

Match your option tenor to roughly 1× to 2× the half-life for cleanest
convergence trades.
""")


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
def main() -> None:
    app_header(
        "Currency Screener",
        "Find pairs of FX underlyings whose terminal correlation is materially "
        "higher than their daily correlation — the wedge dealer vol surfaces miss.",
    )

    # Always-available overview help at top
    help_overview()

    # -------- Sidebar --------
    # Use the toolkit's shared data-folder picker. It persists across
    # toolkit pages via st.session_state["data_dir"], so set the folder
    # once and every page (this one, EKO Pricer, Portfolio Analyzer, etc.)
    # uses the same path.
    data_dir = data_dir_input(default="market_data")
    if data_dir is None:
        # data_dir_input already showed the "Enter a data folder…" hint
        st.stop()

    with st.sidebar:
        st.header("Data")
        spots, missing, index_map = load_all_spots(data_dir)
        if spots.empty:
            st.error(f"No SPOT data loaded from `{data_dir}/`. Check the path "
                     "and that `_index.csv` (or per-pair `{pair}.csv` files) exist.")
            st.stop()
        st.success(f"Loaded {len(spots.columns)} pairs · "
                   f"{spots.index.min().date()} → {spots.index.max().date()}")
        if index_map:
            with st.expander(f"📑 Loaded via _index.csv  ({len(index_map)} mappings)"):
                st.dataframe(
                    pd.DataFrame(
                        [(p, f) for p, f in index_map.items()],
                        columns=["pair", "csv_filename"],
                    ),
                    use_container_width=True, height=240, hide_index=True,
                )
        else:
            st.caption("`_index.csv` not found — using direct `{pair}.csv` lookup.")
        if missing:
            with st.expander(f"⚠️ {len(missing)} pairs missing"):
                st.write(", ".join(missing))

        st.header("Window")
        min_d = spots.index.min().date()
        max_d = spots.index.max().date()
        default_start = max(min_d, (pd.Timestamp(max_d) - pd.DateOffset(years=5)).date())
        start = st.date_input("Start", value=default_start,
                              min_value=min_d, max_value=max_d)
        end = st.date_input("End", value=max_d, min_value=min_d, max_value=max_d)

        st.header("Filters")
        sign_filter = st.radio(
            "Co-movement direction",
            ["Same direction (positive ρ)", "Opposite direction (negative ρ)", "Any"],
            help="Worst-of and dual digitals on both-up or both-down structures "
                 "need positive ρ on quoted underlyings. Some convention plays "
                 "(e.g. EURUSD vs USDKRW for 'USD weakness') want negative ρ.",
        )
        min_abs_rho20 = st.slider("Min |ρ(20d)|", 0.0, 1.0, 0.30, 0.05)
        min_score = st.slider("Min score (ρ(20d) − ρ(1d))", -0.3, 0.5, 0.05, 0.01)
        max_coint_p = st.slider("Max cointegration p-value", 0.0, 1.0, 0.20, 0.05)

    # -------- Run screener --------
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if st.button("▶ Run screener", type="primary"):
        st.session_state["results"] = run_screener(spots, start_ts, end_ts)

    if "results" not in st.session_state:
        st.info("Click **Run screener** to compute statistics across all pair "
                "combinations.")
        return

    results: pd.DataFrame = st.session_state["results"].copy()
    if results.empty:
        st.warning("No pairs passed the minimum data requirements.")
        return

    # Apply sign filter. For "Opposite direction" we redefine score on
    # absolute correlations so that "bigger long-horizon wedge" still ranks high.
    if sign_filter.startswith("Same"):
        results = results[results["rho_20d"] > 0]
    elif sign_filter.startswith("Opposite"):
        results = results[results["rho_20d"] < 0]
        results = results.assign(
            score=results["rho_20d"].abs() - results["rho_1d"].abs()
        )

    # Apply numerical filters
    results = results[
        (results["rho_20d"].abs() >= min_abs_rho20)
        & (results["score"] >= min_score)
        & ((results["coint_pval"].isna()) | (results["coint_pval"] <= max_coint_p))
    ].sort_values("score", ascending=False).reset_index(drop=True)

    # -------- Tabs --------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Ranked table", "Pair matrix", "Pair drill-down", "📚 Learn"]
    )

    with tab1:
        st.subheader(f"Top candidates ({len(results)} pairs after filters)")

        # Educational expanders for the table view
        help_table_columns()
        help_math_correlation()
        help_math_cointegration()
        help_math_variance_ratio()
        help_what_makes_good()

        if results.empty:
            st.info("No pairs pass current filters. Loosen thresholds.")
        else:
            disp = results.copy()
            for c in [c for c in disp.columns if c.startswith("rho_") or c == "score"
                      or c == "abs_score"]:
                disp[c] = disp[c].round(3)
            disp["coint_pval"] = disp["coint_pval"].round(3)
            disp["VR_20"] = disp["VR_20"].round(3)
            disp["VR_z"] = disp["VR_z"].round(2)
            disp["beta_log"] = disp["beta_log"].round(3)
            st.dataframe(disp, use_container_width=True, height=500)

            csv = results.to_csv(index=False).encode()
            st.download_button("⬇ Download CSV", csv,
                               file_name="worst_of_screener_results.csv")

    with tab2:
        st.subheader("Pair matrix")
        help_heatmap()
        metric = st.selectbox("Metric", ["score", "rho_1d", "rho_5d", "rho_20d",
                                          "rho_60d", "VR_20"], index=0)
        if not st.session_state["results"].empty:
            st.plotly_chart(plot_score_heatmap(st.session_state["results"], metric),
                           use_container_width=True)

    with tab3:
        st.subheader("Pair drill-down")
        help_drilldown_charts()
        if results.empty:
            st.info("No pairs to inspect. Adjust filters.")
        else:
            options = [f"{r.Pair_A} / {r.Pair_B}" for r in results.itertuples()]
            sel = st.selectbox("Select a pair", options)
            a, b = sel.split(" / ")

            p1 = spots[a].loc[start_ts:end_ts].dropna()
            p2 = spots[b].loc[start_ts:end_ts].dropna()
            common = p1.index.intersection(p2.index)
            p1, p2 = p1.loc[common], p2.loc[common]
            r1 = np.log(p1).diff().dropna()
            r2 = np.log(p2).diff().dropna()

            # Metric row
            row = results[(results["Pair_A"] == a) & (results["Pair_B"] == b)].iloc[0]
            cols = st.columns(6)
            cols[0].metric("ρ(1d)", f"{row['rho_1d']:.3f}")
            cols[1].metric("ρ(5d)", f"{row['rho_5d']:.3f}")
            cols[2].metric("ρ(20d)", f"{row['rho_20d']:.3f}")
            cols[3].metric("score", f"{row['score']:.3f}")
            cols[4].metric("coint p", f"{row['coint_pval']:.3f}"
                           if not np.isnan(row['coint_pval']) else "n/a")
            cols[5].metric("VR(20)", f"{row['VR_20']:.3f}"
                           if not np.isnan(row['VR_20']) else "n/a")

            st.plotly_chart(plot_normalized_prices(p1, p2, a, b),
                           use_container_width=True)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_term_structure(r1, r2, a, b),
                               use_container_width=True)
            with c2:
                st.plotly_chart(plot_rolling_corr(r1, r2, a, b),
                               use_container_width=True)
            if not np.isnan(row["beta_log"]):
                st.plotly_chart(plot_spread(p1, p2, row["beta_log"], a, b),
                               use_container_width=True)

    with tab4:
        st.subheader("Learn the framework end-to-end")
        st.markdown(
            "These are the same expanders surfaced in the other tabs, "
            "grouped here in study order — read top to bottom."
        )
        st.markdown("### 1. The thesis & how to read results")
        help_overview()
        help_table_columns()
        help_what_makes_good()

        st.markdown("### 2. The math")
        help_math_correlation()
        help_math_cointegration()
        help_math_variance_ratio()

        st.markdown("### 3. Interpreting the visualisations")
        help_heatmap()
        help_drilldown_charts()

        st.markdown("### 4. From candidate to trade")
        help_trading_apps()

        st.markdown("### 5. Things that can go wrong")
        help_caveats()


# Streamlit multi-page apps execute each page script directly when the
# user navigates to it; __name__ is the page module name, NOT "__main__".
# Call main() unconditionally so the page renders. Keeping the function
# encapsulated (rather than inlining at module level) preserves the
# screener's original structure for future editing.
main()

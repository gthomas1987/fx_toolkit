"""Joint Pair Analyzer — joint distribution analysis for two FX pairs.

Sub-app of the FX Toolkit. Reached via the landing page or sidebar nav;
not run directly.

Standalone implementation of a four-layer statistical framework for
understanding where two FX pairs cluster jointly, how stable those
clusters are, and what that implies for barrier-option design.

Pipeline:
  1. Empirical joint density via 2D KDE.
  2. Cluster decomposition via Gaussian Mixture Model (with BIC sweep).
  3. Regime dynamics via Hidden Markov Model (sojourn time, Viterbi
     decoding, stationary distribution).
  4. Barrier guidance via Mahalanobis confidence ellipses on the
     dominant cluster.
  5. Stationarity diagnostics (rolling cluster anchor, d² vs χ²(2)).
  6. Summary with downloadable model parameters.

Originally `apps/10_joint_distribution.py` from the fx_levels_monitor
project; ported into the toolkit with no math changes — only the
data-folder resolution (now via `data_dir_input`) and the page-level
chrome were updated.
"""
from __future__ import annotations

import io
import json
import os
import sys
from datetime import date, timedelta
from itertools import permutations
from pathlib import Path

# Make sibling top-level packages (core/, shared/) importable when
# Streamlit executes this file out of the project root. Same pattern
# as the other toolkit pages.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import chi2, norm

from core.data_loader import load_panel, discovery_summary
from core.ui import data_dir_input, app_header
from shared.style import inject_base_css

# Optional dependencies
try:
    from sklearn.mixture import GaussianMixture
except ImportError:
    GaussianMixture = None  # type: ignore

try:
    from hmmlearn.hmm import GaussianHMM
    HMMLEARN_OK = True
except ImportError:
    HMMLEARN_OK = False


# =============================================================================
# Page / theme
# =============================================================================
st.set_page_config(layout="wide", page_title="Joint Pair Analyzer")
inject_base_css()

# Subtle dark-mode-friendly chart palette (consistent with the EKO Pricer)
CHART_BG = "#0e1117"
GRID = "rgba(255,255,255,0.08)"
CLUSTER_COLORS = ["#ef4444", "#3b82f6", "#10b981", "#f59e0b", "#a855f7", "#06b6d4"]


# =============================================================================
# Header
# =============================================================================
app_header(
    "🪐 Joint Pair Analyzer",
    "Joint distribution analysis for two FX pairs (default: USDJPY × USDKRW). "
    "Four-layer framework: 2D KDE → GMM clusters → HMM regime dynamics → "
    "Mahalanobis-based barrier guidance. Each tab includes a "
    "'How to read this' panel.",
)

if GaussianMixture is None:
    st.error("This app requires scikit-learn. Install with: "
              "`pip install scikit-learn`")
    st.stop()


# =============================================================================
# Theory primer (collapsible) — keeps the app self-contained
# =============================================================================
with st.expander("📖 Theory primer (click to expand)", expanded=False):
    st.markdown(
        """
**The four-layer framework**

This app implements a stack of statistical tools for understanding where
two FX pairs cluster jointly, how stable those clusters are, and what
that implies for barrier-option design.

1. **Joint density (KDE)** — answers *“where has the pair lived?”*. A
   non-parametric smoothed version of the 2D histogram, $f(x,y)$.
   Bright regions = high density = the market spent a lot of time
   there.

2. **Cluster decomposition (GMM)** — answers *“what regimes generated
   this density?”*. Fits $f(x,y) = \\sum_k \\pi_k \\mathcal{N}(\\mu_k, \\Sigma_k)$
   where each component is a regime. The number of clusters $K$ is
   chosen by minimising the **Bayesian Information Criterion** (BIC).

3. **Regime dynamics (HMM)** — answers *“how long does each regime
   last?”*. Same Gaussian components as the GMM, but with a transition
   matrix $A$ between hidden states. The expected sojourn time in state
   $k$ is $\\tau_k = 1/(1-A_{kk})$ trading days. The Viterbi algorithm
   labels each historical date with its most-likely regime.

4. **Barrier placement (Mahalanobis ellipses)** — answers *“where
   should I put the strikes and barriers?”*. For each cluster, the
   set $\\{x: d^2(x;\\mu_k,\\Sigma_k) \\le c\\}$ is an ellipse whose
   semi-axes are $\\sqrt{c \\lambda_i}$ along the eigenvectors of
   $\\Sigma_k$. Standard rule: place barriers at $d^2 \\approx 6$
   (95% confidence) on the cluster you're betting will hold.

**Key trade-relevant numbers** (produced by this app):
- $\\pi_0$ = fraction of time in the dominant regime
- $\\mu_0$ = anchor levels (USDJPY, USDKRW) of the dominant regime
- $\\sqrt{\\lambda_1}, \\sqrt{\\lambda_2}$ = major/minor axis dispersion
- $\\tau_0$ = expected sojourn in days
- Sojourn-vs-tenor health check: $\\tau_0 \\ge 2T$ means the regime is
  hedgeable at tenor $T$
        """
    )


# =============================================================================
# Sidebar — data source
# =============================================================================
# Use the toolkit's shared data-folder picker. It persists across all
# toolkit pages via st.session_state["data_dir"], so users set the
# folder once and every page (this one, EKO Pricer, Vol Dashboard,
# etc.) reads from it. The default ("market_data") matches the rest
# of the toolkit.
folder = data_dir_input(default="market_data")
if folder is None:
    # data_dir_input has already shown the "Enter a data folder…" hint
    # in the sidebar. Halt the page; no point continuing without data.
    st.stop()

try:
    summ = discovery_summary(folder)
except Exception as e:
    st.error(f"Failed to read folder: {e}")
    st.stop()
st.sidebar.caption(f"{summ['n_pairs']} pairs · {summ['n_files']} files")


# =============================================================================
# Controls — pair, date range, K, seed
# =============================================================================
st.markdown("---")

# Discover available pairs from SPOT panel
spot_all = load_panel(folder, "SPOT", None)
all_pairs = sorted(spot_all.columns.tolist()) if not spot_all.empty else []

c1, c2, c3 = st.columns([2, 2, 2])
with c1:
    pair_a = st.selectbox(
        "Pair A",
        all_pairs,
        index=all_pairs.index("USDJPY") if "USDJPY" in all_pairs else 0,
        help="First currency pair. Default USDJPY when available.",
    )
    pair_b_options = [p for p in all_pairs if p != pair_a]
    default_b = ("USDKRW" if "USDKRW" in pair_b_options
                  else pair_b_options[0] if pair_b_options else None)
    pair_b = st.selectbox(
        "Pair B",
        pair_b_options,
        index=pair_b_options.index(default_b) if default_b else 0,
        help="Second currency pair. Default USDKRW when available.",
    )

# Common date range
common_idx = spot_all[[pair_a, pair_b]].dropna().index if not spot_all.empty else []
if len(common_idx) == 0:
    st.error(f"No overlapping spot data for {pair_a} × {pair_b}.")
    st.stop()
date_min = common_idx.min().date()
date_max = common_idx.max().date()

with c2:
    # Default to last 5 years (or the whole range if shorter)
    default_start = max(date_min, date_max - timedelta(days=365 * 5))
    start_date = st.date_input(
        "Start date",
        value=default_start,
        min_value=date_min,
        max_value=date_max,
    )
    end_date = st.date_input(
        "End date",
        value=date_max,
        min_value=date_min,
        max_value=date_max,
    )

with c3:
    K_mode = st.radio(
        "Number of clusters K",
        ["Auto (BIC)", "Manual"],
        horizontal=True,
        help="Auto picks K by minimising BIC across K=1..6. Manual lets "
              "you override after looking at the BIC sweep.",
    )
    if K_mode == "Manual":
        K_manual = st.slider("K", 1, 6, 2, key="jd_K_manual")
    else:
        K_manual = None
    seed = st.number_input(
        "Random seed",
        min_value=0, max_value=99999, value=42,
        help="Controls reproducibility of EM initialisations.",
    )

prefer_em = st.sidebar.radio(
    "EM convention (where applicable)",
    ["offshore", "onshore"],
    index=0,
    help="For pairs with both onshore and offshore listings (e.g. KRW), "
          "which to prefer. USDJPY ignores this.",
    key="jd_prefer",
)


# =============================================================================
# Data loading
# =============================================================================
@st.cache_data(show_spinner=False)
def load_spot_history(folder, pair_a, pair_b, start, end, prefer):
    """Load the spot panel for the two pairs over the requested window.

    Returns a DataFrame indexed by date with columns [pair_a, pair_b],
    dropped to common business days where both have data.
    """
    panel = load_panel(folder, "SPOT", None, prefer=prefer,
                        pairs=(pair_a, pair_b))
    if panel.empty:
        return pd.DataFrame()
    df = panel.loc[(panel.index.date >= start) & (panel.index.date <= end)]
    df = df[[pair_a, pair_b]].dropna()
    return df


data = load_spot_history(folder, pair_a, pair_b, start_date, end_date,
                            prefer_em)
if data.empty or len(data) < 50:
    st.warning(f"Not enough data: {len(data)} observations. Widen the date "
                f"range or check that both pairs have data.")
    st.stop()

X = data.values
n = len(X)


# =============================================================================
# Cached fits
# =============================================================================
@st.cache_data(show_spinner="Sweeping BIC across K…")
def cached_bic_sweep(X_bytes, K_max, n_init, seed):
    X_arr = np.frombuffer(X_bytes, dtype=np.float64).reshape(-1, 2)
    out = {"K": [], "BIC": [], "AIC": [], "loglik": []}
    for K in range(1, K_max + 1):
        g = GaussianMixture(n_components=K, covariance_type="full",
                              n_init=n_init, random_state=seed,
                              reg_covar=1e-4).fit(X_arr)
        out["K"].append(K)
        out["BIC"].append(g.bic(X_arr))
        out["AIC"].append(g.aic(X_arr))
        out["loglik"].append(g.score(X_arr) * len(X_arr))
    return out


@st.cache_data(show_spinner="Fitting GMM…")
def cached_gmm(X_bytes, K, n_init, seed):
    X_arr = np.frombuffer(X_bytes, dtype=np.float64).reshape(-1, 2)
    g = GaussianMixture(n_components=K, covariance_type="full",
                          n_init=n_init, random_state=seed,
                          reg_covar=1e-4).fit(X_arr)
    # Sort clusters by mixing weight descending — cluster 0 = dominant
    order = np.argsort(-g.weights_)
    return {
        "weights": g.weights_[order],
        "means": g.means_[order],
        "covariances": g.covariances_[order],
        "labels": np.array([list(order).index(l) for l in g.predict(X_arr)]),
        "loglik": float(g.score(X_arr) * len(X_arr)),
        "n_parameters": g._n_parameters(),
        "bic": float(g.bic(X_arr)),
        "aic": float(g.aic(X_arr)),
    }


@st.cache_data(show_spinner="Fitting HMM…")
def cached_hmm(X_bytes, K, n_iter, seed, gmm_means_bytes):
    if not HMMLEARN_OK:
        return None
    X_arr = np.frombuffer(X_bytes, dtype=np.float64).reshape(-1, 2)
    gmm_means = np.frombuffer(gmm_means_bytes, dtype=np.float64).reshape(-1, 2)

    # Run a few HMM fits and keep the one with the highest likelihood,
    # then align states to GMM clusters by means-matching.
    best_h, best_ll = None, -np.inf
    for s in range(seed, seed + 5):
        try:
            h = GaussianHMM(n_components=K, covariance_type="full",
                              n_iter=n_iter, random_state=s,
                              tol=1e-4)
            h.fit(X_arr)
            ll = h.score(X_arr)
            if ll > best_ll:
                best_ll = ll
                best_h = h
        except Exception:
            continue
    if best_h is None:
        return None

    # Permute HMM states to match GMM cluster ordering (so cluster 0 in
    # GMM == state 0 in HMM, etc.). Choose the permutation minimising
    # total distance between means.
    K_ = best_h.n_components
    best_perm, best_d = None, np.inf
    for perm in permutations(range(K_)):
        d = sum(np.linalg.norm(best_h.means_[perm[k]] - gmm_means[k])
                  for k in range(K_))
        if d < best_d:
            best_d = d
            best_perm = perm
    perm = np.array(best_perm)
    invperm = np.argsort(perm)   # invperm[k] = HMM-original-label for GMM-label k

    # Reorder transition matrix, means, covariances, startprob
    A = best_h.transmat_[np.ix_(perm, perm)]
    means = best_h.means_[perm]
    covs = best_h.covars_[perm]
    startprob = best_h.startprob_[perm]

    # Decoded states (in HMM's native labelling) → map to GMM labelling
    raw_states = best_h.predict(X_arr)
    decoded = invperm[raw_states]

    # Filtered probabilities (each row sums to 1) — also relabel
    raw_filt = best_h.predict_proba(X_arr)
    filt = raw_filt[:, perm]

    # Stationary distribution
    eigvals, eigvecs = np.linalg.eig(A.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    sd = np.real(eigvecs[:, idx])
    sd = sd / sd.sum()
    sd = np.maximum(sd, 0.0)
    sd = sd / sd.sum()

    return {
        "A": A,
        "means": means,
        "covariances": covs,
        "startprob": startprob,
        "decoded_states": decoded,
        "filtered_probs": filt,
        "stationary": sd,
        "loglik": float(best_ll),
    }


X_bytes = X.tobytes()

# BIC sweep
bic_sweep = cached_bic_sweep(X_bytes, K_max=6, n_init=10, seed=int(seed))
K_auto = int(bic_sweep["K"][int(np.argmin(bic_sweep["BIC"]))])
K_used = K_manual if K_manual is not None else K_auto

# GMM fit at K_used
gmm = cached_gmm(X_bytes, K_used, n_init=10, seed=int(seed))

# HMM fit at K_used
hmm = cached_hmm(X_bytes, K_used, n_iter=500, seed=int(seed),
                    gmm_means_bytes=gmm["means"].tobytes())


# =============================================================================
# Utilities
# =============================================================================
def ellipse_xy(mu: np.ndarray, cov: np.ndarray, d2_threshold: float = 5.99,
                n_pts: int = 200) -> np.ndarray:
    """Generate (x, y) points on a Mahalanobis ellipse boundary.

    The ellipse is the level set d²(x; μ, Σ) = d2_threshold. Default
    d2=5.99 is the 95% boundary of a χ²(2) — the standard barrier rule.
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigvals are ascending; eigvecs columns are corresponding directions
    t = np.linspace(0, 2 * np.pi, n_pts)
    # Unit circle stretched by sqrt(d2 * λ_i) along each axis
    pts_local = np.column_stack([
        np.sqrt(d2_threshold * eigvals[0]) * np.cos(t),
        np.sqrt(d2_threshold * eigvals[1]) * np.sin(t),
    ])
    # Rotate into data frame: eigvecs maps local→data
    pts = pts_local @ eigvecs.T
    return pts + mu


def cluster_axes(cov: np.ndarray) -> dict:
    """Eigendecomposition summary of a cluster covariance."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_idx = int(np.argmax(eigvals))
    minor_idx = 1 - major_idx
    major_vec = eigvecs[:, major_idx]
    angle_deg = float(np.degrees(np.arctan2(major_vec[1], major_vec[0])))
    return {
        "major_sigma": float(np.sqrt(eigvals[major_idx])),
        "minor_sigma": float(np.sqrt(eigvals[minor_idx])),
        "angle_deg": angle_deg,
        "anisotropy": float(np.sqrt(eigvals[major_idx] / eigvals[minor_idx])),
    }


def mahalanobis_d2(X: np.ndarray, mu: np.ndarray,
                      cov: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distance of each row of X to (mu, cov)."""
    diff = X - mu
    cov_inv = np.linalg.inv(cov)
    return np.sum((diff @ cov_inv) * diff, axis=1)


def call_delta(S: float, K: float, T: float, sigma: float,
                r: float = 0.0, q: float = 0.0) -> float:
    """Garman-Kohlhagen call delta — simple approximation for converting
    barrier levels to deltas. Uses single-vol (no smile)."""
    if sigma <= 0 or T <= 0:
        return float(K <= S)   # degenerate
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    return float(np.exp(-q * T) * norm.cdf(d1))


def plotly_dark(fig, height=420, legend_below=True):
    """Apply consistent dark theme to a Plotly figure."""
    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    if legend_below:
        fig.update_layout(legend=dict(
            orientation="h", yanchor="bottom", y=-0.25,
            xanchor="left", x=0,
            font=dict(size=10, color="#cbd5e1"),
        ))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def kde_grid(X: np.ndarray, n: int = 150,
                pad: float = 0.06) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 2D KDE on a grid over X's bounding box (padded)."""
    xmin, xmax = X[:, 0].min(), X[:, 0].max()
    ymin, ymax = X[:, 1].min(), X[:, 1].max()
    dx, dy = xmax - xmin, ymax - ymin
    xmin -= pad * dx; xmax += pad * dx
    ymin -= pad * dy; ymax += pad * dy
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, n), np.linspace(ymin, ymax, n))
    kde = stats.gaussian_kde(X.T)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return xx, yy, zz


def _render_adaptive_audit(
    schedule: list[dict],
    spot_panel: pd.DataFrame,
    folder: str,
    pair_a: str, pair_b: str,
    confidence_pct: int,
    K_used: int,
    prefer_em: str,
) -> None:
    """Render the per-month audit UI for an adaptive schedule.

    Shows training data, all clusters with parameters, daily spot in
    the selected month with per-day cluster picks, and the
    level→delta conversion for a representative trade. All values
    are computed live from the schedule + spot panel + ATM vol
    panels; nothing extra needs to be persisted in the preset JSON.
    """
    from core.wf_schedule import (
        get_training_slice_for_entry, select_cluster_and_tenor,
        _ADAPTIVE_TENORS,
    )
    if not schedule:
        st.warning("Empty schedule — nothing to audit.")
        return

    # 1. Month selector — schedule entries' valid_from
    month_options = [e["valid_from"] for e in schedule]
    month_labels = [
        f"{e['valid_from']} → {e['valid_to']}  "
        f"(trained on {e['n_training_days']} days through "
        f"{e['fit_end_date']})"
        for e in schedule
    ]
    sel_idx = st.selectbox(
        "Select month",
        options=list(range(len(month_options))),
        format_func=lambda i: month_labels[i],
        key="audit_month_select",
        # Default to the most recent month
        index=len(month_options) - 1,
    )
    entry = schedule[sel_idx]

    # 2. Header card — training window + fit metadata
    train_slice = get_training_slice_for_entry(
        spot_panel, pair_a, pair_b, entry
    )
    hcol1, hcol2 = st.columns(2)
    with hcol1:
        window_days = entry.get("training_window_days")
        window_str = (f"Rolling: {window_days} days"
                          if window_days else "Expanding (all prior data)")
        st.markdown(
            "<div class='metric-card'>"
            "<div class='metric-title'>Training window</div>"
            f"<div style='font-size:14px;line-height:1.7;'>"
            f"<b>Window mode</b>: {window_str}<br>"
            f"<b>Start</b>: {train_slice.index.min().date()}<br>"
            f"<b>End (fit_end)</b>: {entry['fit_end_date']}<br>"
            f"<b>Observations</b>: {len(train_slice):,}<br>"
            f"<b>Schedule valid from</b>: {entry['valid_from']}<br>"
            f"<b>Schedule valid to</b>: {entry['valid_to']}"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    with hcol2:
        st.markdown(
            "<div class='metric-card'>"
            "<div class='metric-title'>Fit metadata</div>"
            f"<div style='font-size:14px;line-height:1.7;'>"
            f"<b>K (clusters)</b>: {entry['K']}<br>"
            f"<b>Confidence %</b>: {entry['confidence_pct']}<br>"
            f"<b>Sojourn threshold</b>: ≥ {entry['sojourn_threshold']:.1f}× tenor<br>"
            f"<b>Tenor strategy</b>: <code>{entry['tenor_strategy']}</code><br>"
            f"<b>Pair A / B</b>: {pair_a} / {pair_b}"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    # 3. Cluster summary table
    clusters = entry["clusters"]
    st.markdown("---")
    st.markdown("##### Clusters from this month's fit")
    cluster_rows = []
    for c in clusters:
        ratios = c.get("tenor_sojourn_ratios", {})
        cluster_rows.append({
            "Cluster": f"c{c['cluster_index']}",
            f"μ_a ({pair_a})": f"{c['mu_a']:.3f}",
            f"μ_b ({pair_b})": f"{c['mu_b']:.2f}",
            f"σ_a": f"{c['sigma_a']:.3f}",
            f"σ_b": f"{c['sigma_b']:.2f}",
            "Weight": f"{c['weight']:.2%}",
            "Sojourn (days)": (
                f"{c['sojourn_days']:.0f}"
                if c['sojourn_days'] is not None else "—"
            ),
            **{
                f"r_{t}": f"{ratios.get(t, 0):.2f}×"
                for t, _ in _ADAPTIVE_TENORS
            },
        })
    st.dataframe(pd.DataFrame(cluster_rows),
                  use_container_width=True, hide_index=True)
    st.caption(
        "`r_<tenor>` = sojourn / tenor_days. ≥ 2 (`green`) means the "
        "regime is expected to last at least 2× the option life — the "
        "engine will consider this tenor for this cluster. Lower than "
        "2 means the tenor is filtered out."
    )

    # 4. KDE + cluster geometry chart
    st.markdown("##### Joint distribution + cluster geometry")
    st.caption(
        "KDE of joint spot during the training window, with all K "
        "cluster ellipses (95% Mahalanobis). The scatter shows daily "
        "spot during the selected month, colored by which cluster the "
        "engine picked that day. The chosen cluster's ellipse is "
        "highlighted."
    )
    X = train_slice.values
    try:
        xx, yy, zz = kde_grid(X, n=120)
        # Per-day cluster selection for spots IN this month
        month_start = pd.Timestamp(entry["valid_from"])
        month_end = pd.Timestamp(entry["valid_to"])
        month_spot = spot_panel[
            (spot_panel.index >= month_start)
            & (spot_panel.index <= month_end)
        ][[pair_a, pair_b]].dropna()
        per_day_picks = []
        for ts, row in month_spot.iterrows():
            dec = select_cluster_and_tenor(
                entry, float(row[pair_a]), float(row[pair_b])
            )
            per_day_picks.append({
                "date": ts.date().isoformat(),
                "spot_a": float(row[pair_a]),
                "spot_b": float(row[pair_b]),
                "cluster": (dec["cluster_index"] if dec else None),
                "tenor": (dec["chosen_tenor"] if dec else "(no green)"),
                "green_tenors": (",".join(dec["green_tenors"])
                                       if dec else ""),
                "decision": (dec["decision_log"] if dec else "skipped"),
            })

        fig = go.Figure()
        # KDE contour
        fig.add_contour(
            x=xx[0], y=yy[:, 0], z=zz,
            colorscale=[[0, "#0a1628"], [0.3, "#1f3a5f"],
                          [0.7, "#3a6ea5"], [1.0, "#a8c8ec"]],
            showscale=False, opacity=0.6,
            contours=dict(coloring="fill"),
        )
        # Each cluster's ellipse
        d2_thresh = stats.chi2(df=2).ppf(confidence_pct / 100.0)
        cluster_colors = ["#ff6b9d", "#52d273", "#ffd966",
                            "#a78bfa", "#f97316", "#06b6d4"]
        # Figure out which cluster(s) were actually picked this month
        picked_clusters = set(
            p["cluster"] for p in per_day_picks
            if p["cluster"] is not None
        )
        for c in clusters:
            ck = c["cluster_index"]
            color = cluster_colors[ck % len(cluster_colors)]
            mu = np.array([c["mu_a"], c["mu_b"]])
            cov = np.array([[c["cov_aa"], c["cov_ab"]],
                              [c["cov_ba"], c["cov_bb"]]])
            # ellipse_xy returns an (N, 2) array — index columns
            ellipse_pts = ellipse_xy(mu, cov, d2_thresh)
            xe = ellipse_pts[:, 0]
            ye = ellipse_pts[:, 1]
            is_picked = ck in picked_clusters
            fig.add_scatter(
                x=xe, y=ye, mode="lines",
                line=dict(color=color,
                            width=4 if is_picked else 2,
                            dash="solid" if is_picked else "dot"),
                name=f"c{ck}" + (" ★ picked" if is_picked else ""),
                showlegend=True,
            )
            fig.add_scatter(
                x=[mu[0]], y=[mu[1]], mode="markers+text",
                marker=dict(color=color, size=12, symbol="x"),
                text=[f"c{ck}"], textposition="top center",
                textfont=dict(color=color, size=12),
                showlegend=False,
            )
        # Daily spots for the selected month, colored by pick
        if per_day_picks:
            picks_df = pd.DataFrame(per_day_picks)
            for ck in picks_df["cluster"].dropna().unique():
                sub = picks_df[picks_df["cluster"] == ck]
                color = cluster_colors[int(ck) % len(cluster_colors)]
                fig.add_scatter(
                    x=sub["spot_a"], y=sub["spot_b"], mode="markers",
                    marker=dict(color=color, size=8,
                                  line=dict(color="white", width=1)),
                    name=f"month spot · picked c{int(ck)}",
                    customdata=sub[["date", "tenor"]].values,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        f"{pair_a}: %{{x:.3f}}<br>"
                        f"{pair_b}: %{{y:.2f}}<br>"
                        f"picked c{int(ck)} · tenor %{{customdata[1]}}"
                        "<extra></extra>"
                    ),
                )
            # Dates with no green at all
            skipped = picks_df[picks_df["cluster"].isna()]
            if not skipped.empty:
                fig.add_scatter(
                    x=skipped["spot_a"], y=skipped["spot_b"],
                    mode="markers",
                    marker=dict(color="#888", size=8, symbol="x",
                                  line=dict(color="white", width=1)),
                    name="month spot · skipped (no green)",
                    hovertemplate=(
                        f"%{{x:.3f}}, %{{y:.2f}}<br>no green tenor"
                        "<extra></extra>"
                    ),
                )
        fig.update_layout(
            xaxis_title=pair_a, yaxis_title=pair_b,
            height=560, hovermode="closest",
        )
        plotly_dark(fig, height=560)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"KDE/chart render failed: {e}")

    # 5. Per-day decision table
    st.markdown("##### Per-day decisions in this month")
    if per_day_picks:
        decisions_df = pd.DataFrame(per_day_picks)
        decisions_df.rename(columns={
            "date": "Date",
            "spot_a": f"Spot {pair_a}",
            "spot_b": f"Spot {pair_b}",
            "cluster": "Cluster",
            "tenor": "Tenor",
            "green_tenors": "Green tenors",
            "decision": "Decision",
        }, inplace=True)
        st.dataframe(decisions_df, use_container_width=True,
                      hide_index=True)
    else:
        st.info("No spot data in this month — likely the schedule "
                  "covers a future month not yet observed.")

    # 6. Level → delta conversion for one representative trade
    st.markdown("---")
    st.markdown("##### Strike & barrier selection (representative trade)")
    rep_picks = [p for p in per_day_picks
                    if p["cluster"] is not None]
    if not rep_picks:
        st.info("No trade-eligible date in this month — every day "
                  "was skipped because no cluster had a green tenor.")
    else:
        rep = rep_picks[0]
        rep_date = pd.Timestamp(rep["date"])
        rep_cluster = next(c for c in clusters
                               if c["cluster_index"] == rep["cluster"])
        rep_tenor = rep["tenor"]
        # Look up ATM vol for that tenor on the rep date
        try:
            atm_p = load_panel(folder, "VOL_ATM", rep_tenor,
                                 prefer=prefer_em,
                                 pairs=(pair_a, pair_b))
            sigma_a = float(atm_p[pair_a].asof(rep_date)) / 100.0
            sigma_b = float(atm_p[pair_b].asof(rep_date)) / 100.0
            vol_ok = True
        except Exception as e:
            sigma_a, sigma_b = float("nan"), float("nan")
            vol_ok = False
            vol_err = str(e)
        tenor_days_lookup = dict(_ADAPTIVE_TENORS)
        T_years = tenor_days_lookup[rep_tenor] / 252.0
        S_a = rep["spot_a"]
        S_b = rep["spot_b"]
        mu_a, mu_b = rep_cluster["mu_a"], rep_cluster["mu_b"]
        dx_a = rep_cluster["ellipse_dx_a"]
        dy_b = rep_cluster["ellipse_dy_b"]
        upper_a = mu_a + dx_a
        upper_b = mu_b + dy_b

        if vol_ok:
            from core.wf_schedule import (
                select_strikes_and_barriers, _call_delta_gk,
                ALLOWED_STRIKE_DELTAS, ALLOWED_KO_DELTAS,
                MIN_STRIKE_KO_GAP, SNAP_TOLERANCE,
            )
            # Show all three strategies side-by-side for transparency
            strategies = ["cheapest", "balanced", "max_payoff"]
            ko_a_raw = _call_delta_gk(S_a, upper_a, T_years, sigma_a)
            ko_b_raw = _call_delta_gk(S_b, upper_b, T_years, sigma_b)
        else:
            strategies = ["cheapest"]
            ko_a_raw = ko_b_raw = float("nan")

        st.caption(
            f"Trade date: **{rep['date']}** · "
            f"cluster picked: **c{rep['cluster']}** · "
            f"tenor picked: **{rep_tenor}** "
            f"({tenor_days_lookup[rep_tenor]} business days) · "
            f"T = {T_years:.4f} years"
        )

        # ---- Step 1: cluster geometry → raw KO delta ----
        st.markdown(f"**Step 1**: cluster upper-edge → raw KO Δ candidate")
        step1_tbl = pd.DataFrame([
            [f"Cluster μ_a ({pair_a})", f"{mu_a:.4f}"],
            [f"Spot at trade ({pair_a})", f"{S_a:.4f}"],
            [f"Ellipse half-width δx", f"{dx_a:.4f}"],
            [f"Upper edge ({pair_a}) = μ_a + δx", f"{upper_a:.4f}"],
            [f"ATM vol ({pair_a}, {rep_tenor})",
              f"{sigma_a*100:.2f}%" if vol_ok else "n/a"],
            [f"Raw KO Δ candidate ({pair_a}) at upper edge",
              f"{ko_a_raw:.4f}" if vol_ok else "n/a"],
            ["─", "─"],
            [f"Cluster μ_b ({pair_b})", f"{mu_b:.3f}"],
            [f"Spot at trade ({pair_b})", f"{S_b:.3f}"],
            [f"Ellipse half-width δy", f"{dy_b:.3f}"],
            [f"Upper edge ({pair_b}) = μ_b + δy", f"{upper_b:.3f}"],
            [f"ATM vol ({pair_b}, {rep_tenor})",
              f"{sigma_b*100:.2f}%" if vol_ok else "n/a"],
            [f"Raw KO Δ candidate ({pair_b}) at upper edge",
              f"{ko_b_raw:.4f}" if vol_ok else "n/a"],
        ], columns=["Quantity", "Value"])
        st.dataframe(step1_tbl, use_container_width=True, hide_index=True)

        if vol_ok and (S_a > upper_a or S_b > upper_b):
            st.error(
                f"🚫 Trade **would be skipped** — spot is above the "
                f"cluster's upper edge on at least one leg "
                f"(regime broken). "
                f"{pair_a}: spot={S_a:.4f}, upper_edge={upper_a:.4f} "
                f"({'BROKEN' if S_a > upper_a else 'OK'}). "
                f"{pair_b}: spot={S_b:.3f}, upper_edge={upper_b:.3f} "
                f"({'BROKEN' if S_b > upper_b else 'OK'})."
            )
        elif vol_ok:
            # ---- Step 2: snap to grid ----
            st.markdown(
                f"**Step 2**: snap KO Δ to nearest allowed in "
                f"{{20Δ, 15Δ, 10Δ, 5Δ}} (within ±{SNAP_TOLERANCE*100:.0f}Δ)"
            )
            # ---- Step 3: enumerate valid strikes ----
            # ---- Step 4: pick by strategy ----
            # ---- Step 5: solve back to levels ----
            strat_rows = []
            for strat in strategies:
                res = select_strikes_and_barriers(
                    rep_cluster, spot_a=S_a, spot_b=S_b,
                    T_years=T_years, sigma_a=sigma_a, sigma_b=sigma_b,
                    strike_strategy=strat,
                )
                if res is None:
                    strat_rows.append({
                        "Strategy": strat,
                        f"KO Δ ({pair_a})": "skipped",
                        f"Strike Δ ({pair_a})": "—",
                        f"K_a level": "—",
                        f"H_a level": "—",
                        f"KO Δ ({pair_b})": "skipped",
                        f"Strike Δ ({pair_b})": "—",
                        f"K_b level": "—",
                        f"H_b level": "—",
                    })
                    continue
                strat_rows.append({
                    "Strategy": strat,
                    f"KO Δ ({pair_a})": res["ko_label_a"],
                    f"Strike Δ ({pair_a})": res["strike_label_a"],
                    f"K_a level": f"{res['K_a']:.4f}",
                    f"H_a level": f"{res['H_a']:.4f}",
                    f"KO Δ ({pair_b})": res["ko_label_b"],
                    f"Strike Δ ({pair_b})": res["strike_label_b"],
                    f"K_b level": f"{res['K_b']:.3f}",
                    f"H_b level": f"{res['H_b']:.3f}",
                })
            st.markdown("**Steps 3-5**: enumerate valid strikes, pick "
                         "by strategy, solve back to levels")
            st.dataframe(pd.DataFrame(strat_rows),
                          use_container_width=True, hide_index=True)
            st.caption(
                f"Strict gap constraint: strike Δ − KO Δ ≥ "
                f"{MIN_STRIKE_KO_GAP*100:.0f}Δ. "
                f"`cheapest` picks the lowest strike Δ; `max_payoff` "
                f"the highest; `balanced` the middle. The schedule's "
                f"`strike_strategy` setting determines which row the "
                f"engine actually used on this date."
            )

        if not vol_ok:
            st.warning(
                f"ATM vol for tenor `{rep_tenor}` not available — "
                f"delta values shown as n/a. The engine would also "
                f"skip this trade date. Error: {vol_err}"
            )

    # 7. Plain-English explanation
    st.markdown("---")
    st.markdown("##### How the math works")
    st.markdown(
        f"""
At the start of each month, the engine fits a fresh **Gaussian Mixture
Model** with K = {K_used} clusters on the training window (strictly
causal — no future data used). Each cluster has μ (centre) and Σ
(covariance, capturing shape and orientation). The schedule for this
month stores all K clusters' parameters plus HMM-derived sojourn ratios.

Then **at each trade date** during the month:

1. **Pick the cluster nearest to current spot** by Euclidean distance.

2. **Find the green tenor list** for that cluster — tenors where
   sojourn ≥ 2× tenor. Tenors considered:
   {", ".join(t for t, _ in _ADAPTIVE_TENORS)}.

3. **Skip if no green tenor exists** — the green filter IS the gate.

4. **Pick tenor** per the spec's tenor strategy (e.g. `shortest_green`
   for buyers).

5. **Strike/KO selection with grid constraints**:
    - Compute cluster's **upper edge** = μ + δx (95% ellipse)
    - **Skip if spot > upper edge** (regime broken)
    - Convert upper edge to delta at current spot/vol/T → "raw KO Δ"
    - **Snap to nearest allowed KO** in {{20Δ, 15Δ, 10Δ, 5Δ}};
      skip if outside ±{SNAP_TOLERANCE*100:.0f}Δ tolerance
    - Filter strike Δ choices in {{ATM, 45Δ, 40Δ, 35Δ}} to those with
      gap ≥ {MIN_STRIKE_KO_GAP*100:.0f}Δ vs the snapped KO
    - Pick one strike per the spec's `strike_strategy`:
      `cheapest` (lowest Δ, most OTM, lowest premium — best for buyers),
      `max_payoff` (highest Δ), or `balanced` (middle)
    - Solve back to spot levels: K such that Φ(d1) = strike Δ,
      H such that Φ(d1) = KO Δ

The engine prices using levels (K, H). Deltas are the constraints; the
spot levels are what hit the pricer.
""")


# =============================================================================
# Run-summary banner
# =============================================================================
hmm_status = "✓" if hmm is not None else ("⚠ hmmlearn missing" if not HMMLEARN_OK
                                                 else "⚠ HMM fit failed")
st.markdown(
    f"**Loaded:** {pair_a} × {pair_b} · "
    f"{n} obs · {data.index.min().date()} → {data.index.max().date()} · "
    f"**K (auto)** = {K_auto} · **K (used)** = {K_used} · HMM: {hmm_status}"
)


# =============================================================================
# Tabs
# =============================================================================
(tab_market, tab_kde, tab_gmm, tab_hmm, tab_barrier, tab_stat, tab_summary,
 tab_fit_save) = st.tabs([
    "📍 Market state",
    "📊 Joint density (KDE)",
    "🎯 Clusters (GMM)",
    "⏱ Regime dynamics (HMM)",
    "🛡 Barrier guidance",
    "🔬 Stationarity diagnostics",
    "📋 Summary",
    "🧬 Fit & save per-pair regimes (Phase 3)",
])

# -----------------------------------------------------------------------------
# Tab 0 — Market state (current snapshot, time-traveled or live)
# -----------------------------------------------------------------------------
# Standalone "where are we now, where should I trade?" view. INDEPENDENT
# of the page-level pair/date/K controls — has its own complete set of
# inputs so changing anything here doesn't disturb the other tabs.
#
# Inputs:
#   - Pair A, Pair B (defaults USDJPY × USDKRW where available)
#   - Regime lookback (1Y, 2Y, 3Y, 5Y, ALL — from the as-of date going back)
#   - K, seed, confidence
#   - As-of date (time-travel — pretend today is this date; use only data ≤ this)
#
# Outputs (in render order):
#   1. Status bar showing the actual fit window and HMM availability
#   2. 5 metric cards: spot A, spot B, current cluster, sojourn, in-regime?
#   3. KDE + cluster ellipses chart with current spot star and recent trail
#   4. Per-pair "level ladder" charts — recent price with cluster μ/edges as
#      horizontal reference lines (visual highlight of strikes & KOs)
#   5. All-clusters geometric levels table — anchors, edges, sojourn, distance
#   6. Advisable trades matrix — for each (cluster, green tenor, strategy)
#      combination, the snapped strike/KO levels in both price and delta
# -----------------------------------------------------------------------------
with tab_market:
    # Local imports — keep the new tab self-contained
    from core.wf_schedule import (
        _ADAPTIVE_TENORS, select_strikes_and_barriers, _call_delta_gk,
        ALLOWED_STRIKE_DELTAS, ALLOWED_KO_DELTAS,
        MIN_STRIKE_KO_GAP, SNAP_TOLERANCE,
    )

    st.markdown("### 📍 Market state — current snapshot")
    st.caption(
        "Quick read of where the joint spot sits right now and what "
        "trades the cluster geometry suggests. Use the as-of date to "
        "time-travel and audit what the framework would have shown on "
        "any historical date. **All controls here are independent** of "
        "the page-level controls — set anything you like without "
        "affecting the other tabs."
    )

    # === Controls Row 1: pair A, pair B, as-of date =========================
    msc1, msc2, msc3 = st.columns([2, 2, 2])
    with msc1:
        ms_pair_a = st.selectbox(
            "Currency A",
            all_pairs,
            index=all_pairs.index("USDJPY") if "USDJPY" in all_pairs else 0,
            key="ms_pair_a",
            help="First FX pair.",
        )
    with msc2:
        ms_pair_b_options = [p for p in all_pairs if p != ms_pair_a]
        if not ms_pair_b_options:
            st.error("Need at least two distinct pairs in the data folder.")
            st.stop()
        ms_default_b = ("USDKRW" if "USDKRW" in ms_pair_b_options
                            else ms_pair_b_options[0])
        ms_pair_b = st.selectbox(
            "Currency B",
            ms_pair_b_options,
            index=ms_pair_b_options.index(ms_default_b),
            key="ms_pair_b",
            help="Second FX pair (must differ from A).",
        )
    with msc3:
        # Recompute the common index for THIS pair combination so as-of
        # bounds reflect what's actually available for these two pairs.
        ms_common = spot_all[[ms_pair_a, ms_pair_b]].dropna().index
        if len(ms_common) == 0:
            st.error(f"No overlapping spot data for {ms_pair_a} × {ms_pair_b}.")
            st.stop()
        ms_date_min = ms_common.min().date()
        ms_date_max = ms_common.max().date()
        ms_asof = st.date_input(
            "As-of date (time-travel)",
            value=ms_date_max,
            min_value=ms_date_min,
            max_value=ms_date_max,
            key="ms_asof",
            help="Treat this as 'today'. The fit uses only data on or "
                  "before this date. Pick a historical date to audit "
                  "what the framework would have flagged at that point.",
        )

    # === Controls Row 2: lookback, K, seed, confidence ======================
    msc4, msc5, msc6, msc7 = st.columns([2, 1, 1, 2])
    with msc4:
        LOOKBACK_OPTS = ["1Y", "2Y", "3Y", "5Y", "ALL"]
        LOOKBACK_DAYS_MAP = {"1Y": 252, "2Y": 504, "3Y": 756,
                                  "5Y": 1260, "ALL": None}
        # Pick a sensible default — 3Y if available, else use what's there
        n_years_avail = (ms_date_max - ms_date_min).days / 365.25
        if n_years_avail >= 3.0:
            default_lb = "3Y"
        elif n_years_avail >= 2.0:
            default_lb = "2Y"
        else:
            default_lb = "ALL"
        ms_lookback = st.select_slider(
            "Regime lookback (from as-of, going back)",
            options=LOOKBACK_OPTS,
            value=default_lb,
            key="ms_lookback",
            help="How far back from the as-of date to use for the cluster "
                  "fit. Shorter = tracks current regime closely; longer = "
                  "more stable parameter estimates but mixes defunct "
                  "regimes. ALL uses everything available before as-of.",
        )
    with msc5:
        ms_K = st.slider(
            "# clusters K",
            min_value=2, max_value=6, value=int(K_used),
            key="ms_K",
            help="Number of GMM regimes to fit.",
        )
    with msc6:
        ms_seed = st.number_input(
            "Seed",
            min_value=0, max_value=99999, value=int(seed),
            key="ms_seed",
            help="Random seed for the GMM/HMM EM initialisations.",
        )
    with msc7:
        ms_conf = st.select_slider(
            "Confidence (ellipse mass)",
            options=[68, 80, 90, 95, 99],
            value=95,
            key="ms_conf",
            help="Mass of in-regime moves kept inside the Mahalanobis "
                  "ellipse. 95% (d² ≈ 6) is the standard barrier rule.",
        )

    # === Load data, restrict to as-of, apply lookback =======================
    LOOKBACK_DAYS = LOOKBACK_DAYS_MAP[ms_lookback]
    full_panel = load_spot_history(
        folder, ms_pair_a, ms_pair_b,
        ms_date_min, ms_date_max, prefer_em,
    )
    if full_panel.empty:
        st.error(f"No spot data for {ms_pair_a} × {ms_pair_b}.")
        st.stop()
    asof_ts = pd.Timestamp(ms_asof)
    panel = full_panel[full_panel.index <= asof_ts]
    if LOOKBACK_DAYS is not None and len(panel) > LOOKBACK_DAYS:
        panel = panel.iloc[-LOOKBACK_DAYS:]
    if len(panel) < 60:
        st.warning(
            f"Only {len(panel)} observations after lookback/as-of filter "
            f"— need at least 60 for a meaningful cluster fit. Widen the "
            f"lookback or pick a later as-of date."
        )
        st.stop()

    # === Fit GMM and HMM on the restricted panel ============================
    ms_X = panel.values
    ms_X_bytes = ms_X.tobytes()
    ms_gmm = cached_gmm(ms_X_bytes, ms_K, n_init=10, seed=int(ms_seed))
    ms_hmm = cached_hmm(
        ms_X_bytes, ms_K, n_iter=500, seed=int(ms_seed),
        gmm_means_bytes=ms_gmm["means"].tobytes(),
    )

    # === Current spot snapshot ==============================================
    spot_a_now = float(panel[ms_pair_a].iloc[-1])
    spot_b_now = float(panel[ms_pair_b].iloc[-1])
    asof_actual = panel.index[-1].date()

    def _ms_pct_change(series: pd.Series, n: int):
        if len(series) <= n:
            return None
        return (series.iloc[-1] / series.iloc[-1 - n] - 1.0) * 100.0

    spot_a_5d = _ms_pct_change(panel[ms_pair_a], 5)
    spot_a_21d = _ms_pct_change(panel[ms_pair_a], 21)
    spot_b_5d = _ms_pct_change(panel[ms_pair_b], 5)
    spot_b_21d = _ms_pct_change(panel[ms_pair_b], 21)

    # === Per-cluster metrics ================================================
    d2_ms = float(chi2(df=2).ppf(ms_conf / 100.0))
    spot_arr = np.array([[spot_a_now, spot_b_now]])
    cluster_info = []
    for k in range(ms_K):
        mu = ms_gmm["means"][k]
        cov = ms_gmm["covariances"][k]
        euclid = float(np.sqrt(
            (spot_a_now - mu[0]) ** 2 + (spot_b_now - mu[1]) ** 2
        ))
        mahal_d2 = float(mahalanobis_d2(spot_arr, mu, cov)[0])
        # Sojourn from HMM diagonal
        if ms_hmm is not None:
            A_kk = float(ms_hmm["A"][k, k])
            if A_kk >= 0.99999:
                sojourn = float("inf")
                sojourn_str = "∞ (absorbing)"
                sojourn_for_calc = 5000.0
            else:
                sojourn = 1.0 / (1.0 - A_kk)
                sojourn_str = f"{sojourn:.0f}"
                sojourn_for_calc = sojourn
        else:
            sojourn = None
            sojourn_str = "—"
            sojourn_for_calc = None
        # Ellipse half-widths along each spot axis (the engine's
        # ellipse_dx_a / ellipse_dy_b convention)
        dx = float(np.sqrt(d2_ms * cov[0, 0]))
        dy = float(np.sqrt(d2_ms * cov[1, 1]))
        # Green tenors: sojourn / tenor_days ≥ 2
        green = []
        all_ratios = {}
        if sojourn is not None:
            for t_label, t_days in _ADAPTIVE_TENORS:
                ratio = sojourn / t_days
                all_ratios[t_label] = ratio
                if ratio >= 2.0:
                    green.append((t_label, t_days, ratio))
        cluster_info.append({
            "k": k,
            "mu_a": float(mu[0]), "mu_b": float(mu[1]),
            "cov": cov,
            "weight": float(ms_gmm["weights"][k]),
            "sojourn": sojourn,
            "sojourn_str": sojourn_str,
            "sojourn_for_calc": sojourn_for_calc,
            "euclid_dist": euclid,
            "mahal_d2": mahal_d2,
            "dx": dx, "dy": dy,
            "upper_a": float(mu[0] + dx), "lower_a": float(mu[0] - dx),
            "upper_b": float(mu[1] + dy), "lower_b": float(mu[1] - dy),
            "green_tenors": green,
            "all_ratios": all_ratios,
            "color": CLUSTER_COLORS[k % len(CLUSTER_COLORS)],
        })

    # Current cluster — nearest Euclidean (mirrors wf_schedule logic)
    current_k = min(range(ms_K), key=lambda k: cluster_info[k]["euclid_dist"])
    current_cluster = cluster_info[current_k]
    in_regime = current_cluster["mahal_d2"] <= d2_ms

    # === Helper to build cluster_dict for select_strikes_and_barriers =======
    def _ms_cluster_to_dict(ci: dict) -> dict:
        return {
            "mu_a": ci["mu_a"], "mu_b": ci["mu_b"],
            "ellipse_dx_a": ci["dx"], "ellipse_dy_b": ci["dy"],
            "cluster_index": ci["k"],
            "sigma_a": float(np.sqrt(ci["cov"][0, 0])),
            "sigma_b": float(np.sqrt(ci["cov"][1, 1])),
            "weight": ci["weight"],
            "sojourn_days": (ci["sojourn_for_calc"]
                                  if ci["sojourn_for_calc"] is not None
                                  else 5000.0),
        }

    # === Status bar =========================================================
    fit_status = (
        f"**Fit window**: {panel.index.min().date()} → "
        f"{panel.index.max().date()} · {len(panel):,} obs · "
        f"K = {ms_K} · confidence = {ms_conf}% (d² = {d2_ms:.2f}) · "
        f"HMM: {'✓' if ms_hmm is not None else '⚠ failed'}"
    )
    if asof_actual != ms_asof:
        fit_status += (
            f"  \n*Requested as-of {ms_asof} → fell on a non-business "
            f"day or pre-data; using last available trading date "
            f"{asof_actual} instead.*"
        )
    st.info(fit_status)

    # === Snapshot metric row ================================================
    st.markdown("#### 🧭 Snapshot at as-of date")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    with mc1:
        delta_str = ""
        if spot_a_5d is not None and spot_a_21d is not None:
            delta_str = f"{spot_a_5d:+.2f}% (5d) | {spot_a_21d:+.2f}% (21d)"
        st.metric(f"{ms_pair_a}", f"{spot_a_now:.2f}", delta_str)
    with mc2:
        delta_str = ""
        if spot_b_5d is not None and spot_b_21d is not None:
            delta_str = f"{spot_b_5d:+.2f}% (5d) | {spot_b_21d:+.2f}% (21d)"
        st.metric(f"{ms_pair_b}", f"{spot_b_now:.2f}", delta_str)
    with mc3:
        st.metric(
            "Nearest cluster",
            f"c{current_k}",
            f"π = {current_cluster['weight']:.0%} · "
            f"d_euclid = {current_cluster['euclid_dist']:.3f}",
        )
    with mc4:
        st.metric(
            "Sojourn (expected)",
            f"{current_cluster['sojourn_str']} bd",
            f"Mahal d² = {current_cluster['mahal_d2']:.2f}",
        )
    with mc5:
        if in_regime:
            st.metric("In-regime?", "✅ Inside",
                      f"d² ≤ {d2_ms:.2f}")
        else:
            st.metric("In-regime?", "⚠️ Outside",
                      f"d² > {d2_ms:.2f}")
    if not in_regime:
        st.warning(
            f"Current spot is OUTSIDE the {ms_conf}% ellipse of every "
            f"cluster (nearest is c{current_k} at d² = "
            f"{current_cluster['mahal_d2']:.2f}). Treat the regime as "
            f"broken — the geometry is unreliable for placing barriers "
            f"on the dominant cluster until the spot re-enters one of "
            f"the ellipses."
        )

    st.markdown("---")

    # === Joint density + cluster geometry chart =============================
    st.markdown("#### 🌍 Joint density + cluster geometry")
    st.caption(
        "KDE backdrop = where the joint spot lived during the lookback "
        f"window. Each cluster is a {ms_conf}% Mahalanobis ellipse. "
        f"The current (nearest) cluster c{current_k} has its ellipse "
        f"drawn thick. The yellow star is the spot at as-of "
        f"({asof_actual}); the faint trail is the prior ~60 trading "
        f"days. Hover over any element for details."
    )
    try:
        xx, yy, zz = kde_grid(ms_X, n=140)
        fig_kde = go.Figure()
        fig_kde.add_trace(go.Contour(
            x=xx[0], y=yy[:, 0], z=zz,
            colorscale=[[0, "#0a1628"], [0.3, "#1f3a5f"],
                          [0.7, "#3a6ea5"], [1.0, "#a8c8ec"]],
            showscale=False, opacity=0.55,
            contours=dict(coloring="fill"),
            hoverinfo="skip",
            name="KDE",
            showlegend=False,
        ))
        # Ellipses + means
        for ci in cluster_info:
            color = ci["color"]
            is_curr = ci["k"] == current_k
            ellipse_pts = ellipse_xy(
                np.array([ci["mu_a"], ci["mu_b"]]), ci["cov"], d2_ms,
            )
            label = (f"c{ci['k']} ★ current (τ={ci['sojourn_str']})"
                          if is_curr
                          else f"c{ci['k']} (τ={ci['sojourn_str']})")
            fig_kde.add_trace(go.Scatter(
                x=ellipse_pts[:, 0], y=ellipse_pts[:, 1],
                mode="lines",
                line=dict(color=color,
                            width=4 if is_curr else 2,
                            dash="solid" if is_curr else "dash"),
                name=label,
                hoverinfo="skip",
            ))
            fig_kde.add_trace(go.Scatter(
                x=[ci["mu_a"]], y=[ci["mu_b"]],
                mode="markers+text",
                marker=dict(symbol="x", size=14, color=color,
                            line=dict(width=2)),
                text=[f"c{ci['k']}"],
                textposition="bottom center",
                textfont=dict(color=color, size=11),
                showlegend=False,
                hovertemplate=(
                    f"<b>c{ci['k']}</b><br>"
                    f"μ_a = {ci['mu_a']:.2f}<br>"
                    f"μ_b = {ci['mu_b']:.2f}<br>"
                    f"π = {ci['weight']:.2%}<br>"
                    f"sojourn = {ci['sojourn_str']} bd"
                    "<extra></extra>"
                ),
            ))
        # Recent path trail
        trail_n = min(60, len(panel))
        trail = panel.iloc[-trail_n:]
        fig_kde.add_trace(go.Scatter(
            x=trail[ms_pair_a].values, y=trail[ms_pair_b].values,
            mode="lines+markers",
            line=dict(color="rgba(255,255,255,0.35)", width=1),
            marker=dict(size=3, color="rgba(255,255,255,0.5)"),
            name=f"trail (last {trail_n} bd)",
            hovertemplate=(f"{ms_pair_a}: %{{x:.2f}}<br>"
                              f"{ms_pair_b}: %{{y:.2f}}<extra></extra>"),
        ))
        # Current spot
        fig_kde.add_trace(go.Scatter(
            x=[spot_a_now], y=[spot_b_now],
            mode="markers+text",
            marker=dict(symbol="star", size=24, color="#fbbf24",
                          line=dict(color="white", width=2)),
            text=[f"  {asof_actual}"],
            textposition="middle right",
            textfont=dict(color="#fbbf24", size=11),
            name="current spot",
            hovertemplate=(
                f"<b>{asof_actual}</b><br>"
                f"{ms_pair_a}: %{{x:.2f}}<br>"
                f"{ms_pair_b}: %{{y:.2f}}<br>"
                f"nearest = c{current_k}<br>"
                f"Mahal d² = {current_cluster['mahal_d2']:.2f}"
                "<extra></extra>"
            ),
        ))
        fig_kde.update_layout(
            xaxis_title=ms_pair_a, yaxis_title=ms_pair_b,
            hovermode="closest",
        )
        plotly_dark(fig_kde, height=580)
        st.plotly_chart(fig_kde, use_container_width=True)
    except Exception as e:
        st.error(f"KDE / cluster chart failed: {e}")

    # === Per-leg level ladders ==============================================
    st.markdown("---")
    st.markdown("#### 📏 Per-leg level ladders")
    st.caption(
        "Each panel shows the recent spot history for one leg with "
        "horizontal reference lines marking every cluster's geometry: "
        "**μ** (anchor, solid) and **upper/lower edges** at the chosen "
        "confidence (dashed). Current cluster is bold; other clusters "
        "are dimmed. This is the same information as the joint chart "
        "above but viewed one leg at a time — useful when reading off "
        "approximate strike/KO levels by eye."
    )
    ladder_lookback_n = min(252, len(panel))   # ~1Y display window
    ladder_panel = panel.iloc[-ladder_lookback_n:]

    def _ladder_chart(pair_label: str, axis_key: str) -> go.Figure:
        """One leg's ladder. axis_key is 'a' or 'b' to pick μ/edges."""
        mu_key = f"mu_{axis_key}"
        up_key = f"upper_{axis_key}"
        dn_key = f"lower_{axis_key}"
        fig = go.Figure()
        # Recent spot history
        fig.add_trace(go.Scatter(
            x=ladder_panel.index, y=ladder_panel[pair_label],
            mode="lines",
            line=dict(color="#e2e8f0", width=1.5),
            name=pair_label,
            hovertemplate=(f"%{{x|%Y-%m-%d}}<br>"
                              f"{pair_label}: %{{y:.2f}}<extra></extra>"),
        ))
        # Cluster reference lines — annotate ONLY the current cluster's
        # three lines (μ, +δ, −δ) to keep the chart readable at higher K.
        # Non-current clusters are drawn (color-coded, dimmed) but without
        # text labels — the user can identify them by color from the KDE
        # chart above.
        #
        # NB: Plotly's add_hline renders the default placeholder "new text"
        # when annotation_text is passed as None. To suppress the
        # annotation entirely, the kwarg must be OMITTED, not set to None.
        # That's why current vs non-current clusters take separate code
        # paths below.
        for ci in cluster_info:
            color = ci["color"]
            is_curr = ci["k"] == current_k
            opacity_solid = 1.0 if is_curr else 0.40
            opacity_dash = 0.95 if is_curr else 0.25
            width_solid = 3 if is_curr else 1.4
            width_dash = 2 if is_curr else 1

            if is_curr:
                # Current cluster — annotate all three lines with full labels
                fig.add_hline(
                    y=ci[mu_key],
                    line=dict(color=color, width=width_solid),
                    opacity=opacity_solid,
                    annotation_text=f"c{ci['k']} ★ μ",
                    annotation_position="right",
                    annotation_font=dict(color=color, size=11),
                )
                fig.add_hline(
                    y=ci[up_key],
                    line=dict(color=color, width=width_dash, dash="dash"),
                    opacity=opacity_dash,
                    annotation_text=f"c{ci['k']} ★ +δ",
                    annotation_position="right",
                    annotation_font=dict(color=color, size=10),
                )
                fig.add_hline(
                    y=ci[dn_key],
                    line=dict(color=color, width=width_dash, dash="dash"),
                    opacity=opacity_dash,
                    annotation_text=f"c{ci['k']} ★ −δ",
                    annotation_position="right",
                    annotation_font=dict(color=color, size=10),
                )
            else:
                # Non-current — label μ only (so the user can still tell
                # which color is which cluster); +δ / −δ lines drawn
                # silently without any annotation kwargs.
                fig.add_hline(
                    y=ci[mu_key],
                    line=dict(color=color, width=width_solid),
                    opacity=opacity_solid,
                    annotation_text=f"c{ci['k']} μ",
                    annotation_position="right",
                    annotation_font=dict(color=color, size=9),
                )
                fig.add_hline(
                    y=ci[up_key],
                    line=dict(color=color, width=width_dash, dash="dash"),
                    opacity=opacity_dash,
                )
                fig.add_hline(
                    y=ci[dn_key],
                    line=dict(color=color, width=width_dash, dash="dash"),
                    opacity=opacity_dash,
                )
        # Current spot dot
        fig.add_trace(go.Scatter(
            x=[ladder_panel.index[-1]],
            y=[ladder_panel[pair_label].iloc[-1]],
            mode="markers",
            marker=dict(symbol="star", size=16, color="#fbbf24",
                          line=dict(color="white", width=1.5)),
            name=f"{asof_actual}",
            showlegend=False,
            hovertemplate=(
                f"<b>{asof_actual}</b><br>"
                f"{pair_label}: %{{y:.2f}}<extra></extra>"
            ),
        ))
        fig.update_layout(
            title=f"{pair_label} — recent {ladder_lookback_n}bd "
                    f"with cluster levels",
            yaxis_title=pair_label,
            xaxis_title="date",
        )
        plotly_dark(fig, height=380, legend_below=False)
        return fig

    lc1, lc2 = st.columns(2)
    with lc1:
        st.plotly_chart(_ladder_chart(ms_pair_a, "a"),
                          use_container_width=True)
    with lc2:
        st.plotly_chart(_ladder_chart(ms_pair_b, "b"),
                          use_container_width=True)

    # === All-clusters geometric levels table ================================
    st.markdown("---")
    st.markdown("#### 📋 All clusters — geometric levels (always available)")
    st.caption(
        "Raw output of the cluster geometry — no smile, no snapping, no "
        "vol assumption. These are the **untreated** μ ± δ levels that "
        "the framework recommends as strike (K = μ) and barrier "
        "(H = μ + δ) anchors. The Advisable Trades section below snaps "
        "these to the engine's allowed delta grid using current ATM vol."
    )
    levels_rows = []
    for ci in cluster_info:
        is_curr = ci["k"] == current_k
        green_str = (", ".join(t for t, _, _ in ci["green_tenors"])
                          if ci["green_tenors"] else "—")
        levels_rows.append({
            "Cluster": (f"c{ci['k']} ★" if is_curr else f"c{ci['k']}"),
            "π (weight)": f"{ci['weight']:.2%}",
            f"μ {ms_pair_a}": f"{ci['mu_a']:.2f}",
            f"μ {ms_pair_b}": f"{ci['mu_b']:.2f}",
            f"K_A range ({ms_pair_a})":
                f"[{ci['lower_a']:.2f}, {ci['upper_a']:.2f}]",
            f"K_B range ({ms_pair_b})":
                f"[{ci['lower_b']:.2f}, {ci['upper_b']:.2f}]",
            "Sojourn (bd)": ci["sojourn_str"],
            "Mahal d² (to spot)": f"{ci['mahal_d2']:.2f}",
            "Green tenors": green_str,
        })
    st.dataframe(pd.DataFrame(levels_rows), use_container_width=True,
                  hide_index=True)

    # === Advisable trades matrix ============================================
    # CURRENT CLUSTER ONLY — other clusters are hypothetical regimes,
    # not directly actionable. Pricing uses the engine's standard FX
    # smile convention:
    #   1. ATM vol drives the strike-from-delta solve (avoids the
    #      "delta-depends-on-vol-depends-on-strike" chicken-and-egg —
    #      same convention as core/smile.py and the adaptive engine).
    #   2. The smile vol at the snapped strike (linearly interpolated
    #      across the 25Δ RR/BF surface) drives the KO premium.
    #
    # Vol-tenor handling: ATM is quoted at 1W/1M/2M/3M/6M/1Y in the
    # data layout; RR_25 and BF_25 at 1M/3M/6M/1Y only. For the engine's
    # adaptive tenors (1M, 6W, 2M, 10W, 3M), we interpolate linearly in
    # T (years) across whatever bracketing standard tenors are available.
    # E.g. 6W ATM interpolates 1M↔2M; 2M RR/BF interpolates 1M↔3M.
    #
    # The "Ratio" column = Max payoff / Premium = the trade's gross
    # leverage if both legs finish ITM with barriers intact. Both legs'
    # premium and max-payoff are aggregated to the worst-of structure
    # level via the engine's convention:
    #   structure_premium = min(leg_premia) / 3
    #   structure_max_pay = min(leg_max_pays)
    # (see core/worstof.py; the /3 is the heuristic for the joint
    # probability discount vs the cheaper leg standalone.)
    #
    # Premium and max payoff are reported in bps of notional, which is
    # invariant of the USD notional choice — convenient for comparing
    # trades across different sizes.

    # Local imports — pricing utilities
    from core.smile import smile_vol_at_strike
    from core.ko import ko_price
    from core.rates import TENOR_YEARS as RATES_TENOR_YEARS

    st.markdown("---")
    st.markdown(
        f"#### 🎯 Advisable trades for current cluster c{current_k}"
    )
    st.caption(
        f"Trades shown for the **current (nearest) cluster only** — other "
        f"clusters are hypothetical regimes and not directly actionable. "
        f"Pricing uses smile-aware single-vol Garman-Kohlhagen: **ATM "
        f"vol** solves the strike-from-delta grid, then **σ@K** (the "
        f"smile vol at the snapped strike, interpolated from the 25Δ "
        f"RR/BF surface) prices the KO. Tenors not directly quoted in "
        f"the vol panels (6W, 10W; also 2M for RR/BF) are linearly "
        f"interpolated in T between bracketing standard tenors. Premium "
        f"and Max payoff are in **bps of notional** (notional-invariant). "
        f"**Ratio = Max payoff / Premium** is the trade's gross leverage "
        f"if both legs finish ITM with barriers intact."
    )

    # ----- Load all available vol/RR/BF panels (.asof) -----------------
    VOL_TRY_TENORS = ("1W", "1M", "2M", "3M", "6M", "1Y")
    RRBF_TRY_TENORS = ("1M", "3M", "6M", "1Y")

    def _ms_load_pair_panels(category, tenors):
        """Return ({pa_tenor: val}, {pb_tenor: val}) at-or-before as-of.
        Values stored as raw CSV numbers; divide by 100 at use time
        (matches engine convention)."""
        out_a, out_b = {}, {}
        for tn in tenors:
            try:
                p = load_panel(folder, category, tn, prefer=prefer_em,
                               pairs=(ms_pair_a, ms_pair_b))
            except Exception:
                continue
            if p.empty:
                continue
            p_asof = p[p.index <= asof_ts]
            if p_asof.empty:
                continue
            if ms_pair_a in p_asof.columns:
                s = p_asof[ms_pair_a].dropna()
                if not s.empty:
                    out_a[tn] = float(s.iloc[-1])
            if ms_pair_b in p_asof.columns:
                s = p_asof[ms_pair_b].dropna()
                if not s.empty:
                    out_b[tn] = float(s.iloc[-1])
        return out_a, out_b

    atm_a, atm_b = _ms_load_pair_panels("VOL_ATM", VOL_TRY_TENORS)
    rr_a, rr_b = _ms_load_pair_panels("VOL_25R", RRBF_TRY_TENORS)
    bf_a, bf_b = _ms_load_pair_panels("VOL_25B", RRBF_TRY_TENORS)

    def _interp_vol_in_T(panel_dict, T_target):
        """Linear interp across {tenor_label: value} dict in T (years).
        Tenor → T via TENOR_YEARS from core.rates. Returns None if dict
        empty. Flat extrapolation outside the bracketing range."""
        if not panel_dict:
            return None
        pts = [(RATES_TENOR_YEARS.get(tn), v)
                for tn, v in panel_dict.items()
                if RATES_TENOR_YEARS.get(tn) is not None]
        if not pts:
            return None
        pts.sort()
        Ts = [p[0] for p in pts]
        vs = [p[1] for p in pts]
        if T_target <= Ts[0]:
            return vs[0]
        if T_target >= Ts[-1]:
            return vs[-1]
        return float(np.interp(T_target, Ts, vs))

    def _get_smile_inputs(T_target):
        """Return (σ_atm_a, σ_atm_b, rr_a, rr_b, bf_a, bf_b) at T_target,
        all in decimal. RR/BF default to 0 if no panels available.
        Returns None if ATM is missing for either pair."""
        atm_a_pct = _interp_vol_in_T(atm_a, T_target)
        atm_b_pct = _interp_vol_in_T(atm_b, T_target)
        if atm_a_pct is None or atm_b_pct is None:
            return None
        rr_a_pct = _interp_vol_in_T(rr_a, T_target)
        rr_b_pct = _interp_vol_in_T(rr_b, T_target)
        bf_a_pct = _interp_vol_in_T(bf_a, T_target)
        bf_b_pct = _interp_vol_in_T(bf_b, T_target)
        return (
            atm_a_pct / 100.0,
            atm_b_pct / 100.0,
            (rr_a_pct / 100.0) if rr_a_pct is not None else 0.0,
            (rr_b_pct / 100.0) if rr_b_pct is not None else 0.0,
            (bf_a_pct / 100.0) if bf_a_pct is not None else 0.0,
            (bf_b_pct / 100.0) if bf_b_pct is not None else 0.0,
        )

    # ----- Build trades matrix for current cluster only -----------------
    trades_rows = []
    skipped_reasons = []
    strategies = ["cheapest", "balanced", "max_payoff"]
    cluster_dict = _ms_cluster_to_dict(current_cluster)

    # Fundamental blockers first — these affect the whole cluster
    if not current_cluster["green_tenors"]:
        if current_cluster["all_ratios"]:
            max_ratio = max(current_cluster["all_ratios"].values())
            st.warning(
                f"Current cluster **c{current_k}** has NO green tenors "
                f"(max sojourn ratio = {max_ratio:.2f}× vs required ≥ "
                f"2.00×). Regime too short-lived to support any tenor "
                f"in {{1M, 6W, 2M, 10W, 3M}} — no advisable trades."
            )
        else:
            st.warning(
                f"Current cluster **c{current_k}**: HMM unavailable — "
                f"sojourn can't be estimated. No advisable trades."
            )
    elif (spot_a_now > current_cluster["upper_a"]
          or spot_b_now > current_cluster["upper_b"]):
        st.warning(
            f"Current cluster **c{current_k}**: spot is ALREADY above "
            f"the cluster's upper edge ({ms_pair_a}: "
            f"{spot_a_now:.2f} vs upper "
            f"{current_cluster['upper_a']:.2f}; {ms_pair_b}: "
            f"{spot_b_now:.2f} vs upper "
            f"{current_cluster['upper_b']:.2f}). Regime is broken — a "
            f"call up-and-out can't be priced from this cluster's "
            f"geometry. No advisable trades."
        )
    else:
        # Per-tenor pricing
        for t_label, t_days, _ in current_cluster["green_tenors"]:
            T_years = t_days / 252.0
            v_inputs = _get_smile_inputs(T_years)
            if v_inputs is None:
                skipped_reasons.append(
                    f"{t_label}: ATM vol panel missing for one or "
                    f"both pairs"
                )
                continue
            (sigma_atm_a, sigma_atm_b,
             rr_25_a, rr_25_b, bf_25_a, bf_25_b) = v_inputs

            for strat in strategies:
                # Strike solve uses ATM vol (matches engine convention)
                result = select_strikes_and_barriers(
                    cluster_dict, spot_a_now, spot_b_now,
                    T_years, sigma_atm_a, sigma_atm_b,
                    strike_strategy=strat,
                )
                if result is None:
                    if strat == "cheapest":
                        raw_a = _call_delta_gk(
                            spot_a_now, current_cluster["upper_a"],
                            T_years, sigma_atm_a,
                        )
                        raw_b = _call_delta_gk(
                            spot_b_now, current_cluster["upper_b"],
                            T_years, sigma_atm_b,
                        )
                        skipped_reasons.append(
                            f"{t_label}: snap failed — raw KO Δ "
                            f"A={raw_a:.0%}, B={raw_b:.0%} (outside "
                            f"±{int(SNAP_TOLERANCE * 100)}Δ of "
                            f"5Δ/10Δ/15Δ/20Δ grid)"
                        )
                    continue
                K_a, K_b = result["K_a"], result["K_b"]
                H_a, H_b = result["H_a"], result["H_b"]

                # Smile vol at the snapped strike (single σ per leg —
                # standard FX desk convention). r_d = r_f = 0 to stay
                # consistent with the strike solver.
                sigma_K_a = smile_vol_at_strike(
                    spot_a_now, K_a, T_years, sigma_atm_a,
                    rr_25_a, bf_25_a, 0.0, 0.0,
                )
                sigma_K_b = smile_vol_at_strike(
                    spot_b_now, K_b, T_years, sigma_atm_b,
                    rr_25_b, bf_25_b, 0.0, 0.0,
                )

                # KO price per leg (call up-and-out)
                prem_a_per_unit = ko_price(
                    "call", "up_and_out",
                    spot_a_now, K_a, H_a, T_years, sigma_K_a, 0.0, 0.0,
                )
                prem_b_per_unit = ko_price(
                    "call", "up_and_out",
                    spot_b_now, K_b, H_b, T_years, sigma_K_b, 0.0, 0.0,
                )

                # Per-leg bps of notional (notional-invariant):
                # bps = premium_DOM_per_unit_FOR / S * 10000
                prem_a_bps = prem_a_per_unit / spot_a_now * 10000.0
                prem_b_bps = prem_b_per_unit / spot_b_now * 10000.0
                max_pay_a_bps = abs(H_a - K_a) / spot_a_now * 10000.0
                max_pay_b_bps = abs(H_b - K_b) / spot_b_now * 10000.0

                # Worst-of structure aggregation (matches worstof.py):
                #   premium = min(leg_premia) / 3
                #   max_pay = min(leg_max_pays)
                struct_prem_bps = min(prem_a_bps, prem_b_bps) / 3.0
                struct_max_pay_bps = min(max_pay_a_bps, max_pay_b_bps)
                if struct_prem_bps > 1e-9:
                    payoff_ratio = struct_max_pay_bps / struct_prem_bps
                    ratio_str = f"{payoff_ratio:.2f}×"
                else:
                    payoff_ratio = float("inf")
                    ratio_str = "∞"

                trades_rows.append({
                    "_t_days": t_days,           # for sorting
                    "_strat": strat,             # for sorting
                    "_ratio_num": payoff_ratio,  # for sorting tie-break
                    "Tenor": t_label,
                    "Strategy": strat,
                    f"K {ms_pair_a}":
                        f"{K_a:.2f} ({result['strike_label_a']})",
                    f"H {ms_pair_a}":
                        f"{H_a:.2f} ({result['ko_label_a']})",
                    f"K {ms_pair_b}":
                        f"{K_b:.2f} ({result['strike_label_b']})",
                    f"H {ms_pair_b}":
                        f"{H_b:.2f} ({result['ko_label_b']})",
                    "ATM σ A": f"{sigma_atm_a * 100:.2f}%",
                    "σ@K_A": f"{sigma_K_a * 100:.2f}%",
                    "ATM σ B": f"{sigma_atm_b * 100:.2f}%",
                    "σ@K_B": f"{sigma_K_b * 100:.2f}%",
                    "Premium (bps)": f"{struct_prem_bps:.1f}",
                    "Max payoff (bps)": f"{struct_max_pay_bps:.1f}",
                    "Ratio (payoff/prem)": ratio_str,
                })

    # ----- Render the table with best-trade highlight ----------------
    if trades_rows:
        # Sort by tenor days asc, then strategy (cheapest first)
        strat_order = {"cheapest": 0, "balanced": 1, "max_payoff": 2}
        trades_rows.sort(
            key=lambda r: (r["_t_days"], strat_order[r["_strat"]])
        )
        # Best per framework rules = shortest_green tenor + cheapest
        # strike strategy. After sort, that's the first 'cheapest' row.
        best_idx = next(
            (i for i, r in enumerate(trades_rows)
              if r["_strat"] == "cheapest"),
            None,
        )
        for i, r in enumerate(trades_rows):
            r["Pick"] = "★ BEST" if i == best_idx else ""

        # Final column order — Pick first
        ordered_cols = [
            "Pick", "Tenor", "Strategy",
            f"K {ms_pair_a}", f"H {ms_pair_a}",
            f"K {ms_pair_b}", f"H {ms_pair_b}",
            "ATM σ A", "σ@K_A", "ATM σ B", "σ@K_B",
            "Premium (bps)", "Max payoff (bps)", "Ratio (payoff/prem)",
        ]
        trades_df = pd.DataFrame(trades_rows)[ordered_cols]
        st.dataframe(trades_df, use_container_width=True, hide_index=True)

        # ---- Best-trade explanation ----
        if best_idx is not None:
            best = trades_rows[best_idx]
            shortest_t = best["Tenor"]
            shortest_days = best["_t_days"]
            shortest_ratio = current_cluster["all_ratios"].get(
                shortest_t, 0
            )
            st.success(
                f"### ★ Best trade — {shortest_t} cheapest on c{current_k}\n\n"
                f"**Why this is the framework's pick**, per the default "
                f"buyer-of-EKO rules baked into the adaptive engine:\n\n"
                f"1. **Shortest green tenor: {shortest_t} "
                f"({shortest_days} business days).** Of the tenors where "
                f"sojourn / tenor ≥ 2 (= the regime is expected to "
                f"outlast the option by at least 2×), this is the "
                f"shortest. **Shortest tenor = lowest premium per trade**, "
                f"which is the buyer's bias (cheap optionality, not "
                f"maximum vega exposure). Sojourn ratio at {shortest_t}: "
                f"**{shortest_ratio:.2f}×**.\n\n"
                f"2. **Cheapest strike (most OTM, lowest Δ).** Within "
                f"the engine's grid constraints (KO Δ ∈ {{20Δ, 15Δ, 10Δ, "
                f"5Δ}}, strike Δ ∈ {{ATM, 45Δ, 40Δ, 35Δ}}, gap ≥ 25Δ), "
                f"`cheapest` picks the lowest valid strike Δ. Lowest "
                f"strike Δ = furthest OTM = lowest premium — same buyer "
                f"bias as point 1.\n\n"
                f"3. **Levels**: {best[f'K {ms_pair_a}']} / "
                f"{best[f'H {ms_pair_a}']} on **{ms_pair_a}**, "
                f"{best[f'K {ms_pair_b}']} / "
                f"{best[f'H {ms_pair_b}']} on **{ms_pair_b}** (smile σ "
                f"used: {best['σ@K_A']} / {best['σ@K_B']}).\n\n"
                f"4. **Premium = {best['Premium (bps)']} bps of notional**, "
                f"**Max payoff = {best['Max payoff (bps)']} bps**, "
                f"**leverage = {best['Ratio (payoff/prem)']}** if both "
                f"legs finish ITM with barriers intact.\n\n"
                f"*Premium and max payoff use the worst-of structure "
                f"convention: min(leg_A, leg_B) / 3 for premium, "
                f"min(leg_A, leg_B) for max payoff. Smile-aware single-vol "
                f"GK; no correlation model — actual worst-of premium would "
                f"be slightly lower under a copula or full MC.*"
            )
    elif not current_cluster["green_tenors"] or (
        spot_a_now > current_cluster["upper_a"]
        or spot_b_now > current_cluster["upper_b"]
    ):
        # The warnings above already explained why; nothing more to add
        pass
    else:
        st.info(
            f"No feasible tenor / strategy combination for c{current_k} "
            f"at as-of date."
        )

    # Skip reasons (current cluster only)
    if skipped_reasons:
        with st.expander(
            f"Show skip reasons ({len(skipped_reasons)} entries)",
            expanded=False,
        ):
            for r in skipped_reasons:
                st.markdown(f"- {r}")

    # === Bottom note ========================================================
    st.markdown("---")
    with st.expander("📖 How to read this tab", expanded=False):
        st.markdown(
            f"""
**What the framework is telling you, in plain English:**

1. **Where am I right now?** — the yellow star in the joint density
   chart. The "Mahal d² to spot" column tells you how close the spot
   is to each cluster *in cluster-shape units* — small = inside.

2. **What regime am I in?** — the *nearest cluster* by Euclidean
   distance (`c{current_k}` here). The decision rule matches the
   adaptive engine in app 9.

3. **Is the regime stable enough to trade?** — sojourn (in business
   days) is the expected time the market stays in this cluster before
   transitioning. A tenor is "green" if sojourn ≥ 2× tenor days.
   `c{current_k}` has these green tenors:
   `{', '.join(t for t, _, _ in current_cluster['green_tenors']) or 'NONE'}`.
   *A sojourn shown as `∞ (absorbing)` means the regime was observed
   entering but never leaving in the training window — interpret as
   "very sticky in-sample" but be aware the estimate is upward-biased
   when the sample is short or the regime is the latest one.*

4. **What strikes and KOs should I do?** — the snapped strike/KO
   levels in the Advisable Trades table, shown for the **current
   cluster only** (other clusters are hypothetical regimes — not
   actionable). Three strategies are shown side-by-side: `cheapest`
   (lowest premium, most OTM), `balanced` (middle of valid strike
   grid), `max_payoff` (most ATM, biggest in-the-money window).
   Pricing is **smile-aware**: ATM solves the strike-from-delta
   grid, then σ@K (smile vol at the snapped strike, from 25Δ RR/BF)
   prices the KO. Vols at non-quoted tenors (6W, 10W; also 2M for
   RR/BF in some pairs) are linearly interpolated in T.

5. **Which one should I actually trade?** — the row marked **★ BEST**
   is the framework's default pick: shortest green tenor (lowest
   premium per trade) at the cheapest strike (most OTM, lowest Δ).
   The explanation under the table breaks down the reasoning. The
   **Ratio** column = Max payoff ÷ Premium = the gross leverage if
   both legs finish ITM with barriers intact.

6. **Time-travel** — change the as-of date to audit what the
   framework would have flagged on any historical date. The fit is
   re-done using only data on or before that date, so there's no
   look-ahead. Useful for sanity-checking regime calls during
   historical stress periods.

**Knob intuitions:**

- **Regime lookback** — shorter window tracks current regime
  closely but is noisier (parameters jump as data drops off the
  back). Longer window is stable but mixes defunct regimes.
  Try 1Y vs 3Y vs ALL and watch the cluster ellipses move.
- **K** — more clusters always lower BIC up to a point, then
  parameters become unidentified. If two clusters overlap visibly
  you've gone too high.
- **Confidence** — wider ellipse → wider strike/barrier range →
  larger snapped KO Δ in the trades table. 95% is the standard.
            """
        )


# -----------------------------------------------------------------------------
# Tab 1 — Joint density (KDE)
# -----------------------------------------------------------------------------
with tab_kde:
    st.markdown("### Empirical joint density")
    st.caption(
        f"Where {pair_a} × {pair_b} has lived over the selected window. "
        f"Brightness = how often the pair was in that neighbourhood. "
        f"Multi-modal structure (separate bright blobs) suggests "
        f"distinct regimes — which we will decompose in the next tab."
    )

    xx, yy, zz = kde_grid(X, n=150)

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.78, 0.22],
        row_heights=[0.22, 0.78],
        horizontal_spacing=0.02, vertical_spacing=0.02,
        shared_xaxes=True, shared_yaxes=True,
    )

    # Marginal histograms
    fig.add_trace(go.Histogram(
        x=X[:, 0], nbinsx=40, marker_color="#7dd3fc",
        showlegend=False, hovertemplate=f"{pair_a}: %{{x:.4f}}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Histogram(
        y=X[:, 1], nbinsy=40, marker_color="#86efac",
        showlegend=False, hovertemplate=f"{pair_b}: %{{y:.4f}}<extra></extra>",
    ), row=2, col=2)

    # KDE contour (filled)
    fig.add_trace(go.Contour(
        x=xx[0, :], y=yy[:, 0], z=zz,
        colorscale="Viridis", showscale=True,
        contours=dict(showlines=True, showlabels=False),
        line=dict(color="rgba(255,255,255,0.25)", width=0.5),
        colorbar=dict(title="density", thickness=12, len=0.7,
                        x=1.04, y=0.4),
        hovertemplate=(f"{pair_a}: %{{x:.4f}}<br>"
                         f"{pair_b}: %{{y:.4f}}<br>"
                         f"density: %{{z:.4e}}<extra></extra>"),
    ), row=2, col=1)

    # Scatter overlay (so individual outliers are visible)
    fig.add_trace(go.Scatter(
        x=X[:, 0], y=X[:, 1], mode="markers",
        marker=dict(size=2.5, color="rgba(255,255,255,0.35)"),
        showlegend=False,
        hovertemplate=(f"{pair_a}: %{{x:.4f}}<br>"
                         f"{pair_b}: %{{y:.4f}}<extra></extra>"),
    ), row=2, col=1)

    fig.update_xaxes(title=pair_a, row=2, col=1)
    fig.update_yaxes(title=pair_b, row=2, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=2)
    plotly_dark(fig, height=620, legend_below=False)
    fig.update_layout(showlegend=False, bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("How to read this", expanded=False):
        st.markdown(
            f"""
- The big panel shows the joint density $\\hat f(x, y)$ of
  (`{pair_a}`, `{pair_b}`).
- The top panel is the marginal histogram of `{pair_a}` alone; the
  right panel is the marginal histogram of `{pair_b}` alone.
- **Marginals lose the joint information.** A single tilted blob in
  the joint means the pairs co-move; you can't see that from the
  marginals.
- **Number of bright blobs ≈ number of regimes.** If you see one
  dominant blob and a smaller secondary one, that's typical "defended
  + post-break" Asian-FX behaviour.
- **Elongation direction of a blob** = the co-movement axis within
  that regime. A diagonal tilt means both pairs strengthen / weaken
  together.

**Limitations of KDE alone:** it gives a picture but doesn't *name*
the regimes or tell you how stable they are. The next two tabs do that.
            """
        )


# -----------------------------------------------------------------------------
# Tab 2 — Clusters (GMM)
# -----------------------------------------------------------------------------
with tab_gmm:
    st.markdown("### Cluster decomposition (Gaussian Mixture Model)")
    st.caption(
        f"Fits $f(x,y) = \\sum_k \\pi_k \\mathcal{{N}}(\\mu_k, \\Sigma_k)$ "
        f"with K chosen by BIC. Each cluster is a regime: $\\pi_k$ is its "
        f"long-run weight, $\\mu_k$ its anchor, and $\\Sigma_k$ its "
        f"shape (visualised as ellipses below)."
    )

    # BIC sweep chart
    bic_fig = go.Figure()
    bic_fig.add_trace(go.Scatter(
        x=bic_sweep["K"], y=bic_sweep["BIC"], mode="lines+markers",
        line=dict(color="#ef4444", width=2.5), marker=dict(size=8),
        name="BIC (preferred)",
    ))
    bic_fig.add_trace(go.Scatter(
        x=bic_sweep["K"], y=bic_sweep["AIC"], mode="lines+markers",
        line=dict(color="#3b82f6", width=2, dash="dash"), marker=dict(size=7),
        name="AIC",
    ))
    bic_fig.add_vline(
        x=K_auto, line=dict(color="#22c55e", dash="dot", width=2),
        annotation_text=f"Auto K = {K_auto}",
        annotation_position="top right",
    )
    if K_manual is not None and K_manual != K_auto:
        bic_fig.add_vline(
            x=K_manual, line=dict(color="#f59e0b", dash="dash", width=2),
            annotation_text=f"Manual K = {K_manual}",
            annotation_position="top left",
        )
    bic_fig.update_layout(
        title=f"BIC / AIC sweep across K — argmin BIC at K = {K_auto}",
        xaxis_title="Number of clusters K",
        yaxis_title="Information criterion (lower is better)",
    )
    plotly_dark(bic_fig, height=320)
    st.plotly_chart(bic_fig, use_container_width=True)

    # Cluster scatter with ellipses
    scatter_fig = go.Figure()
    # KDE backdrop (faint contour for context)
    xx_g, yy_g, zz_g = kde_grid(X, n=120)
    scatter_fig.add_trace(go.Contour(
        x=xx_g[0, :], y=yy_g[:, 0], z=zz_g, showscale=False,
        contours=dict(coloring="lines"),
        line=dict(color="rgba(255,255,255,0.10)", width=0.5),
        opacity=0.5, hoverinfo="skip",
    ))
    for k in range(K_used):
        mask = gmm["labels"] == k
        c = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        scatter_fig.add_trace(go.Scatter(
            x=X[mask, 0], y=X[mask, 1], mode="markers",
            marker=dict(size=4, color=c, opacity=0.55),
            name=f"Cluster {k} · π={gmm['weights'][k]:.2f}",
            hovertemplate=(f"{pair_a}: %{{x:.4f}}<br>"
                             f"{pair_b}: %{{y:.4f}}<br>"
                             f"cluster {k}<extra></extra>"),
        ))
        # Ellipses at 1σ, 2σ, 3σ (Mahalanobis)
        for d2_val, dash in [(2.30, "dot"), (5.99, "dash"), (11.83, "solid")]:
            pts = ellipse_xy(gmm["means"][k], gmm["covariances"][k], d2_val)
            scatter_fig.add_trace(go.Scatter(
                x=pts[:, 0], y=pts[:, 1], mode="lines",
                line=dict(color=c, width=1.5, dash=dash),
                showlegend=False, hoverinfo="skip",
            ))
        # Centre marker
        scatter_fig.add_trace(go.Scatter(
            x=[gmm["means"][k, 0]], y=[gmm["means"][k, 1]],
            mode="markers", marker=dict(symbol="x", size=14, color=c,
                                          line=dict(width=2)),
            showlegend=False,
            hovertemplate=(f"μ_{k} = ({gmm['means'][k, 0]:.4f}, "
                             f"{gmm['means'][k, 1]:.4f})<extra></extra>"),
        ))
    scatter_fig.update_layout(
        title=f"GMM fit at K = {K_used} (dotted=68%, dashed=95%, solid=99.7%)",
        xaxis_title=pair_a, yaxis_title=pair_b,
    )
    plotly_dark(scatter_fig, height=560)
    st.plotly_chart(scatter_fig, use_container_width=True)

    # Per-cluster parameter table
    st.markdown("#### Cluster parameters")
    rows = []
    for k in range(K_used):
        ax = cluster_axes(gmm["covariances"][k])
        n_in_cluster = int((gmm["labels"] == k).sum())
        rows.append({
            "Cluster": k,
            "π (weight)": f"{gmm['weights'][k]:.3f}",
            f"μ {pair_a}": f"{gmm['means'][k, 0]:.4f}",
            f"μ {pair_b}": f"{gmm['means'][k, 1]:.4f}",
            "Major-axis σ": f"{ax['major_sigma']:.4f}",
            "Minor-axis σ": f"{ax['minor_sigma']:.4f}",
            "Anisotropy": f"{ax['anisotropy']:.2f}",
            "Major-axis angle (°)": f"{ax['angle_deg']:.1f}",
            "n points": n_in_cluster,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                  hide_index=True)

    with st.expander("How to read this", expanded=False):
        st.markdown(
            f"""
- **BIC / AIC sweep**: lower is better. Both criteria balance fit
  (likelihood) against complexity (parameters). BIC has a stronger
  complexity penalty so it tends to pick fewer clusters; for regime
  identification BIC is preferred because we want each named cluster
  to be reliably present in the data.
- **π (weight)**: long-run fraction of time the market spent in this
  regime. Cluster 0 (the dominant cluster) has the highest π by
  convention in this app.
- **μ ({pair_a}, {pair_b})**: the anchor levels — read these as
  "the central bank's defended level for {pair_a} was ~{gmm['means'][0,0]:.2f}
  with {pair_b} ~ {gmm['means'][0,1]:.2f} during this regime."
- **Major / minor-axis σ**: dispersion *along the natural axes of the
  cluster*, not along the spot axes. Major-axis = the "more variable"
  direction (typically co-movement); minor-axis = the "more controlled"
  direction (typically relative value).
- **Anisotropy** = major/minor ratio. Values near 1 mean a roughly
  round cluster (USDJPY and USDKRW move independently inside the
  regime); large values mean a strongly co-moving regime.
- **Angle**: orientation of the major axis in degrees from the x-axis.
  ~45° means "both pairs co-move 1-for-1 in level space" (modulo
  scale differences).
            """
        )

    # Eigendecomposition educational box
    with st.expander("📖 Eigendecomposition refresher (click to expand)",
                     expanded=False):
        st.markdown(
            """
Given a 2×2 covariance matrix
$\\Sigma = \\begin{pmatrix} \\sigma_x^2 & \\rho\\sigma_x\\sigma_y \\\\
\\rho\\sigma_x\\sigma_y & \\sigma_y^2 \\end{pmatrix}$, the eigendecomposition is

$$\\Sigma = V D V^T, \\quad D = \\text{diag}(\\lambda_1, \\lambda_2)$$

where:
- The columns of $V$ are unit-length **eigenvectors** $v_1, v_2$ — the
  ellipse's axis directions.
- The **eigenvalues** $\\lambda_1, \\lambda_2$ are the squared
  semi-axis lengths along those directions. The $1\\sigma$ ellipse has
  semi-axes $\\sqrt{\\lambda_1}, \\sqrt{\\lambda_2}$.

The "Major-axis σ" column above is just $\\sqrt{\\lambda_{\\max}}$;
"Minor-axis σ" is $\\sqrt{\\lambda_{\\min}}$.
            """
        )


# -----------------------------------------------------------------------------
# Tab 3 — Regime dynamics (HMM)
# -----------------------------------------------------------------------------
with tab_hmm:
    st.markdown("### Regime dynamics (Hidden Markov Model)")
    st.caption(
        "Fits the same K Gaussian components, but also learns a "
        "transition matrix A. Tells you not just *where* the regimes "
        "are but how long each visit lasts and when historical regime "
        "switches occurred."
    )

    if not HMMLEARN_OK:
        st.warning("This section requires the `hmmlearn` package. "
                    "Install with `pip install hmmlearn`.")
    elif hmm is None:
        st.error("HMM fit failed for this configuration. Try a different "
                  "K or random seed.")
    else:
        # ------ Transition matrix display ------
        st.markdown("#### Transition matrix A")
        st.caption("$A_{ij}$ = probability of moving from state $i$ at "
                   "day $t$ to state $j$ at day $t+1$. Diagonal entries "
                   "are 'stickiness' — bigger = longer expected sojourn.")
        A = hmm["A"]
        # Show A as a heatmap-style table. We avoid pandas' built-in
        # `background_gradient` because it pulls in matplotlib just for
        # the colormap. A pure-CSS linear interpolation between two hex
        # colors achieves the same visual without the heavy dep.
        A_df = pd.DataFrame(
            A,
            index=[f"From state {i}" for i in range(K_used)],
            columns=[f"To state {j}" for j in range(K_used)],
        )
        def _ylgnbu_bg(v: float, vmin: float = 0.0,
                          vmax: float = 1.0) -> str:
            """Map v ∈ [vmin, vmax] to a YlGnBu-ish color via linear
            interpolation in RGB space between three anchor colors:
            yellow (255, 255, 217) → green (127, 205, 187) → blue
            (29, 145, 192). Returns a CSS background-color rule."""
            if pd.isna(v):
                return ""
            t = max(0.0, min(1.0, (float(v) - vmin) / (vmax - vmin + 1e-9)))
            anchors = [(1.0, 1.0, 0.85), (0.50, 0.80, 0.73), (0.11, 0.57, 0.75)]
            if t < 0.5:
                u = t / 0.5
                r = anchors[0][0] + u * (anchors[1][0] - anchors[0][0])
                g = anchors[0][1] + u * (anchors[1][1] - anchors[0][1])
                b = anchors[0][2] + u * (anchors[1][2] - anchors[0][2])
            else:
                u = (t - 0.5) / 0.5
                r = anchors[1][0] + u * (anchors[2][0] - anchors[1][0])
                g = anchors[1][1] + u * (anchors[2][1] - anchors[1][1])
                b = anchors[1][2] + u * (anchors[2][2] - anchors[1][2])
            R, G, B = int(r * 255), int(g * 255), int(b * 255)
            # Pick text color for contrast (dark text on light bg, vice versa)
            text = "#000000" if (R * 0.299 + G * 0.587 + B * 0.114) > 150 else "#ffffff"
            return f"background-color: rgb({R},{G},{B}); color: {text};"
        # Pandas added DataFrame.style.map in 2.1; .applymap before that.
        # Use whichever is available.
        styler = A_df.style.format("{:.4f}")
        if hasattr(styler, "map"):
            styler = styler.map(_ylgnbu_bg)
        else:
            styler = styler.applymap(_ylgnbu_bg)
        st.dataframe(styler, use_container_width=True)

        # ------ Expected sojourn / stationary distribution ------
        st.markdown("#### Sojourn times and stationary distribution")
        cols = st.columns(K_used)
        for k in range(K_used):
            with cols[k]:
                A_kk = A[k, k]
                # Sojourn = 1/(1-A_kk). When A_kk is numerically 1
                # (HMM has effectively absorbed this state — common when
                # a regime appears once and never leaves in the sample),
                # display as ∞ rather than a misleading huge number.
                if A_kk >= 0.99999:
                    tau_str = "∞ (absorbing in sample)"
                else:
                    tau_str = f"{1.0 / (1.0 - A_kk):.0f} trading days"
                pi_inf = hmm["stationary"][k]
                pi_gmm = gmm["weights"][k]
                c = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
                st.markdown(
                    f"<div style='border-left: 4px solid {c}; "
                    f"padding-left: 10px;'>"
                    f"<b style='color:{c}'>State {k}</b><br>"
                    f"A<sub>{k}{k}</sub> = <b>{A_kk:.4f}</b><br>"
                    f"E[sojourn] = <b>{tau_str}</b><br>"
                    f"π<sub>∞</sub> = <b>{pi_inf:.3f}</b><br>"
                    f"GMM π = <b>{pi_gmm:.3f}</b><br>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        st.caption(
            "GMM π and HMM π<sub>∞</sub> should be roughly similar — they "
            "are two estimates of the same thing (long-run regime weight). "
            "A large gap suggests your sample is too short for the dynamics "
            "to have reached steady-state.",
            unsafe_allow_html=True,
        )

        # ------ Spot path with Viterbi-decoded state shading ------
        st.markdown("#### Spot history with Viterbi-decoded regime")
        st.caption(
            "Each historical day is labelled by its most-likely regime. "
            "The background shading shows the regime; you can read off "
            "regime switches by eye."
        )
        states = hmm["decoded_states"]
        path_fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    vertical_spacing=0.04,
                                    subplot_titles=[pair_a, pair_b])
        for k in range(K_used):
            in_state = states == k
            if not in_state.any():
                continue
            # Find contiguous runs of state k
            edges = np.diff(np.concatenate([[0], in_state.astype(int), [0]]))
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0] - 1
            c = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
            for s_idx, e_idx in zip(starts, ends):
                x0 = data.index[s_idx]
                x1 = data.index[min(e_idx, len(data) - 1)]
                for r in [1, 2]:
                    path_fig.add_vrect(
                        x0=x0, x1=x1, fillcolor=c, opacity=0.12,
                        layer="below", line_width=0, row=r, col=1,
                    )
        # Spot lines
        path_fig.add_trace(go.Scatter(
            x=data.index, y=data[pair_a], mode="lines",
            line=dict(color="#e0e7ff", width=1.5),
            name=pair_a, showlegend=False,
        ), row=1, col=1)
        path_fig.add_trace(go.Scatter(
            x=data.index, y=data[pair_b], mode="lines",
            line=dict(color="#e0e7ff", width=1.5),
            name=pair_b, showlegend=False,
        ), row=2, col=1)
        # Cluster anchors as horizontal lines
        for k in range(K_used):
            c = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
            path_fig.add_hline(y=gmm["means"][k, 0], row=1, col=1,
                                line=dict(color=c, dash="dot", width=1),
                                annotation_text=f"μ_{k}",
                                annotation_position="right",
                                annotation=dict(font_size=10))
            path_fig.add_hline(y=gmm["means"][k, 1], row=2, col=1,
                                line=dict(color=c, dash="dot", width=1),
                                annotation_text=f"μ_{k}",
                                annotation_position="right",
                                annotation=dict(font_size=10))
        plotly_dark(path_fig, height=520, legend_below=False)
        path_fig.update_layout(showlegend=False)
        path_fig.update_xaxes(title="Date", row=2, col=1)
        path_fig.update_yaxes(title=pair_a, row=1, col=1)
        path_fig.update_yaxes(title=pair_b, row=2, col=1)
        st.plotly_chart(path_fig, use_container_width=True)

        # ------ Empirical sojourn distribution ------
        st.markdown("#### Empirical sojourn distribution")
        st.caption(
            "Histogram of actual contiguous-run lengths in the "
            "Viterbi-decoded sequence, compared against the geometric "
            "distribution implied by $A_{kk}$. A close match means the "
            "Markov assumption is reasonable; systematic deviations "
            "(e.g. very long runs more frequent than predicted) suggest "
            "higher-order dependence."
        )
        # Compute run-lengths per state
        sojourn_runs = {k: [] for k in range(K_used)}
        prev = states[0]
        run = 1
        for s in states[1:]:
            if s == prev:
                run += 1
            else:
                sojourn_runs[prev].append(run)
                prev = s
                run = 1
        sojourn_runs[prev].append(run)

        sojourn_cols = st.columns(K_used)
        for k in range(K_used):
            runs_k = sojourn_runs[k]
            with sojourn_cols[k]:
                if len(runs_k) < 2:
                    st.caption(f"State {k}: too few transitions to plot.")
                    continue
                sj_fig = go.Figure()
                max_t = max(runs_k) + 5
                # Empirical histogram
                hist, edges = np.histogram(runs_k,
                                              bins=min(20, max(5, len(runs_k))),
                                              density=True)
                centres = 0.5 * (edges[:-1] + edges[1:])
                c = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
                sj_fig.add_trace(go.Bar(
                    x=centres, y=hist, marker_color=c, opacity=0.55,
                    name="empirical",
                ))
                # Geometric reference
                days = np.arange(1, max_t)
                p = (1 - A[k, k]) * (A[k, k] ** (days - 1))
                sj_fig.add_trace(go.Scatter(
                    x=days, y=p, mode="lines",
                    line=dict(color="white", width=2),
                    name="geometric model",
                ))
                sj_fig.update_layout(
                    title=f"State {k}: {len(runs_k)} runs, "
                          f"mean = {np.mean(runs_k):.0f} d, "
                          f"E[geo] = {1/(1-A[k,k]):.0f} d",
                    xaxis_title="Sojourn length (days)",
                    yaxis_title="probability",
                )
                plotly_dark(sj_fig, height=300)
                sj_fig.update_layout(showlegend=False)
                st.plotly_chart(sj_fig, use_container_width=True)

        with st.expander("How to read this", expanded=False):
            st.markdown(
                """
- **Transition matrix**: rows sum to 1. The diagonal is the
  "stickiness" — probability that today's regime persists into
  tomorrow.
- **Expected sojourn** = 1 / (1 − A_kk). This is the expected length
  of a single visit to that regime. Compare it to your option tenor
  in the next tab.
- **Stationary distribution π∞** is the long-run fraction of time
  spent in each state, derived purely from A. It should roughly
  match the GMM mixing weights π_k. Material disagreement is a sign
  of finite-sample noise or non-stationarity.
- **Viterbi shading**: each band shows when the model thinks the
  market was in each regime. Use this to spot whether the labels
  match your trader-intuition (e.g. defended periods vs known
  policy-shift dates).
- **Geometric reference**: under the Markov assumption, sojourn
  lengths are geometrically distributed. If the empirical histogram
  has fatter tails than the geometric curve, the model is
  *under-estimating* how long regimes actually last (and vice versa).
                """
            )


# -----------------------------------------------------------------------------
# Tab 4 — Barrier guidance
# -----------------------------------------------------------------------------
with tab_barrier:
    st.markdown("### Barrier-placement guidance from the dominant cluster")
    st.caption(
        "Translates the dominant cluster's geometry into concrete spot "
        "levels (and approximate deltas) where strikes and barriers "
        "should sit for a worst-of structure."
    )

    cb1, cb2, cb3 = st.columns(3)
    with cb1:
        target_cluster = st.selectbox(
            "Target cluster",
            list(range(K_used)),
            index=0,
            format_func=lambda k: f"Cluster {k} (π = {gmm['weights'][k]:.2f})",
            help="The regime you're betting will hold for the option's "
                  "life. Default: cluster 0 (highest weight).",
        )
    with cb2:
        confidence = st.select_slider(
            "Confidence (mass of in-regime moves kept inside)",
            options=[68, 90, 95, 99],
            value=95,
            help="Where to place the barrier on the cluster's "
                  "Mahalanobis ellipse. 95% (d² ≈ 6) is the standard rule.",
        )
        d2_thresh = chi2(df=2).ppf(confidence / 100.0)
    with cb3:
        tenor_choice = st.selectbox(
            "Tenor (for sojourn health check and delta conversion)",
            ["1M", "2M", "3M", "6M"],
            index=0,
        )
        tenor_days = {"1M": 21, "2M": 42, "3M": 63, "6M": 126}[tenor_choice]
        tenor_years = {"1M": 1/12, "2M": 2/12, "3M": 0.25, "6M": 0.5}[tenor_choice]

    mu_t = gmm["means"][target_cluster]
    cov_t = gmm["covariances"][target_cluster]
    ax_t = cluster_axes(cov_t)

    # Extreme points on the ellipse along each axis (in spot space)
    eigvals, eigvecs = np.linalg.eigh(cov_t)
    # Ellipse extent on x-axis is sqrt(d2 * Sigma_xx); same for y
    delta_x = np.sqrt(d2_thresh * cov_t[0, 0])
    delta_y = np.sqrt(d2_thresh * cov_t[1, 1])

    # Spot today (last observation)
    S_a = float(data[pair_a].iloc[-1])
    S_b = float(data[pair_b].iloc[-1])

    # Suggested call-up-and-out barriers: in the direction further from
    # spot toward the OTHER cluster (proxied by "above" the dominant
    # anchor). User should sense-check direction in production.
    H_a_up = float(mu_t[0] + delta_x)
    H_b_up = float(mu_t[1] + delta_y)
    H_a_dn = float(mu_t[0] - delta_x)
    H_b_dn = float(mu_t[1] - delta_y)

    # Strikes at the cluster anchor (= "ATM relative to the regime")
    K_a = float(mu_t[0])
    K_b = float(mu_t[1])

    st.markdown(f"#### Geometric output (cluster {target_cluster}, "
                f"d² = {d2_thresh:.2f} = {confidence}%)")
    st.markdown(
        f"""
The {confidence}% Mahalanobis ellipse for cluster {target_cluster} has
the following extent:

- **{pair_a}**: cluster anchor $\\mu = {mu_t[0]:.4f}$; ellipse reaches
  to $\\mu \\pm {delta_x:.4f}$ → range **[{H_a_dn:.4f}, {H_a_up:.4f}]**
- **{pair_b}**: cluster anchor $\\mu = {mu_t[1]:.4f}$; ellipse reaches
  to $\\mu \\pm {delta_y:.4f}$ → range **[{H_b_dn:.4f}, {H_b_up:.4f}]**
- Major-axis dispersion: {ax_t['major_sigma']:.4f}
  (at {ax_t['angle_deg']:.1f}° from horizontal)
- Minor-axis dispersion: {ax_t['minor_sigma']:.4f}

**Current spot**: {pair_a} = {S_a:.4f}, {pair_b} = {S_b:.4f}
        """
    )

    # ATM-vol-based delta conversion (rough, single-vol GK)
    st.markdown("#### Approximate delta equivalents")
    st.caption(
        "Maps the ellipse extents to call deltas at the selected tenor "
        "using your stored ATM vol panel (single-vol Garman-Kohlhagen — "
        "approximate). For exact smile-adjusted deltas, plug the spot "
        "levels above into the Pricer tab of app 9."
    )

    # Pull ATM vol at the chosen tenor for each pair
    try:
        atm_panel = load_panel(folder, "VOL_ATM", tenor_choice,
                                  prefer=prefer_em, pairs=(pair_a, pair_b))
        if not atm_panel.empty:
            sigma_a = float(atm_panel[pair_a].dropna().iloc[-1]) / 100.0
            sigma_b = float(atm_panel[pair_b].dropna().iloc[-1]) / 100.0
            vol_avail = True
        else:
            vol_avail = False
    except Exception:
        vol_avail = False

    if vol_avail:
        # Strike at anchor — its call delta
        K_a_delta = call_delta(S_a, K_a, tenor_years, sigma_a)
        K_b_delta = call_delta(S_b, K_b, tenor_years, sigma_b)
        # Barrier above (up-and-out call)
        H_a_delta = call_delta(S_a, H_a_up, tenor_years, sigma_a)
        H_b_delta = call_delta(S_b, H_b_up, tenor_years, sigma_b)
        delta_table = pd.DataFrame({
            "Leg": [f"{pair_a}", f"{pair_b}"],
            "S (today)": [f"{S_a:.4f}", f"{S_b:.4f}"],
            f"ATM vol ({tenor_choice})": [f"{sigma_a*100:.2f}%",
                                              f"{sigma_b*100:.2f}%"],
            "Strike K (= μ_anchor)": [f"{K_a:.4f}", f"{K_b:.4f}"],
            "K call-Δ": [f"{K_a_delta*100:.0f}Δ", f"{K_b_delta*100:.0f}Δ"],
            "Barrier H (μ + ellipse)": [f"{H_a_up:.4f}", f"{H_b_up:.4f}"],
            "H call-Δ": [f"{H_a_delta*100:.0f}Δ", f"{H_b_delta*100:.0f}Δ"],
        })
        st.dataframe(delta_table, use_container_width=True, hide_index=True)

        st.markdown(
            f"**Suggested grid for the bulk runner:** sweep strike Δ ∈ "
            f"[{max(5, int(K_a_delta*100)-5)}Δ, {int(K_a_delta*100)+5}Δ] "
            f"and KO Δ ∈ [{max(5, int(H_a_delta*100)-5)}Δ, "
            f"{int(H_a_delta*100)+5}Δ] on leg A, with analogous bands on "
            f"leg B. This focuses the grid on the ellipse-implied "
            f"optimum."
        )
    else:
        st.info("ATM vol panel not available for this tenor / pair combo. "
                  "Use the spot levels above directly in app 9's Pricer "
                  "tab.")

    # Sojourn-tenor health check
    sojourn_health = "na"
    sojourn_days_val = None
    if hmm is not None:
        tau = 1.0 / max(1e-9, 1.0 - hmm["A"][target_cluster, target_cluster])
        sojourn_days_val = float(tau)
        ratio = tau / tenor_days
        st.markdown("#### Sojourn-tenor health check")
        if ratio >= 2.0:
            sojourn_health = "ok"
            st.success(
                f"✅ **Hedgeable**: expected sojourn in cluster "
                f"{target_cluster} is {tau:.0f} trading days, which is "
                f"{ratio:.1f}× your selected {tenor_choice} tenor "
                f"({tenor_days} trading days). The regime is stable "
                f"enough to support this tenor."
            )
        elif ratio >= 1.0:
            sojourn_health = "warn"
            st.warning(
                f"⚠️ **Marginal**: expected sojourn is {tau:.0f} days vs "
                f"tenor {tenor_days} days (ratio {ratio:.2f}). Roughly "
                f"30–50% of trades will span a regime change. Consider "
                f"a shorter tenor or tighter barriers."
            )
        else:
            sojourn_health = "fail"
            st.error(
                f"❌ **Unstable**: expected sojourn is only {tau:.0f} days, "
                f"shorter than your {tenor_choice} tenor ({tenor_days} days). "
                f"The barrier will almost certainly be tested within the "
                f"option's life. Pick a shorter tenor or accept that the "
                f"structure will frequently KO."
            )

    # =====================================================================
    # Export to bulk runner — Phase 4.5 wiring.
    # Writes a small JSON preset that app 9's Backtest and Worst-of
    # tabs can load to auto-populate strike Δ / KO Δ / tenor / gate
    # multi-selects with a band around this analytical optimum. The
    # user can still edit anything before clicking Run.
    # =====================================================================
    st.markdown("---")
    st.markdown("#### 💾 Export to bulk runner")
    if not vol_avail:
        st.info(
            "Preset export needs an ATM vol panel to convert ellipse spot "
            "levels into discrete deltas. Either load a folder with vol "
            "data, or transcribe the spot levels above into app 9's "
            "controls manually."
        )
    else:
        # Constants must match app 9's DELTA_CHOICES / KO_DELTA_CHOICES.
        # Kept in sync with apps/9_ko_pricer.py.
        DELTA_CHOICES_LOCAL = {"ATM": 0.0, "45Δ": 0.45, "40Δ": 0.40,
                                 "35Δ": 0.35, "30Δ": 0.30, "25Δ": 0.25}
        KO_DELTA_CHOICES_LOCAL = {"5Δ": 0.05, "10Δ": 0.10, "15Δ": 0.15,
                                      "20Δ": 0.20, "25Δ": 0.25}

        # Use the average of leg-A and leg-B analytical deltas for the
        # band centre — the ellipse gives one optimum per axis, and
        # symmetric strikes are the most common worst-of setup. Users
        # can widen or shift either leg in app 9 after loading.
        target_strike_delta = float((K_a_delta + K_b_delta) / 2)
        target_ko_delta = float((H_a_delta + H_b_delta) / 2)

        # =================================================================
        # Phase WF-C: mode toggle — Static (in-sample, current behaviour)
        # or Walk-forward monthly (causal, refits μ and Σ on first biz
        # day of each month using only prior data).
        # =================================================================
        export_mode = st.radio(
            "**Export mode**",
            options=[
                "Static (in-sample)",
                "Walk-forward monthly (causal — WF-C)",
                "Adaptive (one preset, pick cluster + tenor per date)",
            ],
            horizontal=False,
            key="bg_mode",
            help=(
                "**Static**: bakes today's GMM fit into a single (K, H) "
                "level band. Strike Δ and KO Δ stay fixed across the "
                "whole backtest. Has parameter-level look-ahead.\n\n"
                "**Walk-forward monthly**: rebuilds the cluster fit at "
                "the start of every month using only prior data, then "
                "applies that month's (K, H) to all trades opening in "
                "that month. One preset per (cluster, tenor) combination "
                "— you have to pick a cluster up front.\n\n"
                "**Adaptive**: ONE preset that picks the nearest cluster "
                "to current spot AND the shortest green tenor per trade "
                "date. Strike, barrier, AND tenor are all dynamic. No "
                "separate gate needed — the green-sojourn filter is the "
                "gate. Best for buyers of cheap KOs (shortest tenor = "
                "lowest premium per trade)."
            ),
        )
        is_dynamic_mode = export_mode.startswith("Walk-forward")
        is_adaptive_mode = export_mode.startswith("Adaptive")

        ce1, ce2, ce3 = st.columns([2, 2, 2])
        with ce1:
            if is_adaptive_mode:
                # Adaptive mode: cluster picked per date, tenor picked
                # per date — none of the static-mode "band width" knobs
                # apply. Show the WF refit knobs (K and min training)
                # plus the tenor strategy.
                wf_K = st.slider(
                    "K (clusters per monthly refit)",
                    min_value=2, max_value=6, value=K_used,
                    help="Same K used at each monthly refit. All "
                          "clusters get included in the schedule — "
                          "the engine picks the one nearest to current "
                          "spot per trade date.",
                    key="bg_wf_K",
                )
                wf_min_train = st.number_input(
                    "Minimum training days",
                    min_value=126, max_value=1260,
                    value=252, step=21,
                    help="Skip monthly refits with fewer than this "
                          "many prior days.",
                    key="bg_wf_min_train",
                )
                training_window_mode = st.selectbox(
                    "Training window",
                    options=[
                        "Expanding (use ALL prior data)",
                        "Rolling: last 2 years (504 days)",
                        "Rolling: last 3 years (756 days)",
                        "Rolling: last 5 years (1260 days)",
                    ],
                    index=1,  # default rolling 2y — most appropriate
                    help=(
                        "EXPANDING: each refit uses every prior obs. "
                        "More data → tighter parameter estimates, but "
                        "old defunct regimes still anchor the fit "
                        "(can cause stale cluster labels in multi-"
                        "epoch data — e.g. a pre-2022 'strong Won' "
                        "cluster that no longer exists).\n\n"
                        "ROLLING: each refit uses only the last N "
                        "days. Tracks current regime structure; "
                        "forgets defunct historical regimes. Recommended "
                        "default for an adaptive framework on long "
                        "multi-regime data."
                    ),
                    key="bg_training_window_mode",
                )
                _window_lookup = {
                    "Expanding (use ALL prior data)": None,
                    "Rolling: last 2 years (504 days)": 504,
                    "Rolling: last 3 years (756 days)": 756,
                    "Rolling: last 5 years (1260 days)": 1260,
                }
                training_window_days = _window_lookup[training_window_mode]
                tenor_strategy_label = st.selectbox(
                    "Tenor strategy (from green list)",
                    options=["shortest_green", "median_green",
                              "longest_green"],
                    index=0,
                    help="From the tenors where sojourn ≥ 2× tenor "
                          "(green): shortest = cheapest premium (best "
                          "for KO BUYERS); longest = max vega; median "
                          "= middle ground.",
                    key="bg_tenor_strategy",
                )
                strike_strategy_label = st.selectbox(
                    "Strike strategy (within KO/Δ grid)",
                    options=["cheapest", "balanced", "max_payoff"],
                    index=0,
                    help=(
                        "Once KO Δ is fixed by snapping the cluster's "
                        "upper edge to the nearest {20Δ,15Δ,10Δ,5Δ}, "
                        "the strike Δ is picked from {ATM,45Δ,40Δ,35Δ} "
                        "satisfying a strict 25Δ minimum gap.\n\n"
                        "**cheapest** = lowest strike Δ (most OTM, "
                        "lowest premium — recommended for buyers).\n\n"
                        "**max_payoff** = highest strike Δ (most ATM, "
                        "biggest payoff window).\n\n"
                        "**balanced** = middle of the valid list."
                    ),
                    key="bg_strike_strategy",
                )
                band_width = 0  # not used
            elif is_dynamic_mode:
                # No band width — the strike and KO are levels from the
                # schedule, not deltas, so band-width is meaningless.
                # Instead expose the WF-fit knobs.
                wf_K = st.slider(
                    "K (clusters per monthly refit)",
                    min_value=2, max_value=4, value=K_used,
                    help="Same K used at each monthly refit. Keep "
                          "consistent with the BIC sweep in Tab 2.",
                    key="bg_wf_K",
                )
                wf_min_train = st.number_input(
                    "Minimum training days",
                    min_value=126, max_value=1260,
                    value=252, step=21,
                    help="Skip monthly refits with fewer than this many "
                          "prior days. ~252 = 1 year is the standard "
                          "minimum for stable cluster fits.",
                    key="bg_wf_min_train",
                )
                band_width = 0  # not used; engine reads schedule
                tenor_strategy_label = "shortest_green"  # not used
                strike_strategy_label = "cheapest"  # not used
                training_window_days = None  # not used
            else:
                band_width = st.slider(
                    "Grid width (steps each side of optimum)",
                    min_value=0, max_value=3, value=1,
                    help="0 = just the single nearest delta. 1 = ±1 "
                          "grid step on each side (typical). Larger = "
                          "wider sweep but more strategies to evaluate.",
                    key="bg_band_width",
                )
                wf_K = K_used
                wf_min_train = 252
                tenor_strategy_label = "shortest_green"  # not used
                strike_strategy_label = "cheapest"  # not used
                training_window_days = None  # not used
        with ce2:
            if is_adaptive_mode:
                # G1 design — green-sojourn filter IS the gate.
                # Show informational text instead of a checkbox.
                st.info(
                    "**No separate gate needed** — in adaptive mode, "
                    "the green-sojourn filter (sojourn ≥ 2× tenor) is "
                    "the gate. Dates where no cluster has a green "
                    "tenor are automatically skipped."
                )
                include_hmm_gate = False  # gate keys = [None] downstream
            else:
                include_hmm_gate = st.checkbox(
                    "Include `hmm_dominant` gate",
                    value=True,
                    help="Adds the **label-robust** dominant-regime gate "
                          "to the preset. This gate fires when the market "
                          "is currently in the prevailing regime — "
                          "regardless of which integer label is attached. "
                          "Works correctly even when the HMM fit assigned "
                          "non-zero labels to the currently-dominant "
                          "regime (which happens in trending markets). "
                          "Requires regime CSVs (Tab 7) but no re-fitting.",
                    key="bg_include_gate",
                )
        with ce3:
            export_dir = st.text_input(
                "Output folder (presets/ subdir will be created)",
                value=folder,
                help="Defaults to your market data folder. App 9 scans "
                      "`<this folder>/presets/*.json` for available presets.",
                key="bg_export_dir",
            )

        # Preview what will be written
        try:
            from core.presets import build_preset, save_preset, \
                delta_band_around, nearest_delta_label
            if is_adaptive_mode or is_dynamic_mode:
                # Both WF and adaptive modes build a per-month schedule.
                # Preview the expected entry count.
                from core.wf_schedule import first_business_days_of_month
                hint_start = pd.Timestamp(data.index.min())
                hint_end = pd.Timestamp(data.index.max())
                n_months_total = len(
                    first_business_days_of_month(hint_start, hint_end)
                )
                est_skip = max(0, int(np.ceil(wf_min_train / 21)))
                est_entries = max(0, n_months_total - est_skip)
                strike_band_preview = None
                ko_band_preview = None
            else:
                strike_band_preview = delta_band_around(
                    target_strike_delta, DELTA_CHOICES_LOCAL,
                    n_each_side=band_width,
                )
                ko_band_preview = delta_band_around(
                    target_ko_delta, KO_DELTA_CHOICES_LOCAL,
                    n_each_side=band_width,
                )
                est_entries = None
        except Exception as e:
            strike_band_preview = []
            ko_band_preview = []
            est_entries = None
            st.error(f"Preview failed: {e}")

        gate_str = "hmm_dominant" if include_hmm_gate else "(no gate)"
        if is_adaptive_mode and est_entries is not None:
            st.caption(
                f"**Preview (adaptive mode)**: will fit a fresh GMM + "
                f"HMM at the start of each month, producing ~"
                f"**{est_entries}** schedule entries with ALL {wf_K} "
                f"clusters' parameters per entry. At each trade date, "
                f"the engine picks the cluster nearest to current spot "
                f"and the **{tenor_strategy_label}** tenor from "
                f"{{1M, 6W, 2M, 10W, 3M}} where sojourn ≥ 2× tenor. "
                f"No separate gate (green-sojourn filter is the gate). "
                f"This produces **ONE strategy** with fully dynamic "
                f"tenor + strike + barrier per trade date. "
                f"⚠️ Building this schedule fits {est_entries}+ "
                f"GMMs and HMMs — expect 30-90 seconds."
            )
            do_save = True
        elif is_dynamic_mode and est_entries is not None:
            st.caption(
                f"**Preview (walk-forward mode)**: will fit a fresh GMM "
                f"at the start of each month, producing approximately "
                f"**{est_entries}** schedule entries spanning "
                f"{hint_start.date()} → {hint_end.date()} "
                f"(skipping the first ~{est_skip} months for training). "
                f"Each entry contains the (K_a, K_b, H_a, H_b) spot "
                f"levels at 95% Mahalanobis confidence for cluster "
                f"{target_cluster}. Tenor: `{tenor_choice}`, gate: "
                f"`{gate_str}`. ⚠️ Building this schedule fits "
                f"{est_entries}+ GMMs and HMMs — expect 30-90 seconds."
            )
            do_save = True
        elif strike_band_preview and ko_band_preview:
            n_combos = len(strike_band_preview) * len(ko_band_preview)
            st.caption(
                f"**Preview**: strike Δ ∈ {strike_band_preview} · "
                f"KO Δ ∈ {ko_band_preview} · gate: `{gate_str}` · "
                f"tenor: `{tenor_choice}` · "
                f"= **{n_combos}** per-leg combinations "
                f"({n_combos*n_combos} worst-of specs in a symmetric sweep)"
            )
            do_save = True
        else:
            do_save = False

        if do_save:
            colsx = st.columns([3, 1])
            with colsx[1]:
                save_clicked = st.button(
                    "💾 Save preset", type="primary",
                    use_container_width=True,
                    key="bg_save_preset",
                )
            if save_clicked:
                # In dynamic/adaptive mode, build the schedule first
                # (the expensive step — fits one GMM per month).
                schedule = None
                if is_adaptive_mode:
                    from core.wf_schedule import build_adaptive_schedule
                    with st.spinner(
                        f"Building ADAPTIVE schedule with {wf_K} "
                        f"clusters (one GMM + one HMM per month, "
                        f"all clusters retained)…"
                    ):
                        try:
                            schedule = build_adaptive_schedule(
                                data, pair_a, pair_b,
                                confidence_pct=int(confidence),
                                K=int(wf_K),
                                backtest_start=hint_start,
                                backtest_end=hint_end,
                                min_training_days=int(wf_min_train),
                                seed=int(seed),
                                sojourn_threshold=2.0,
                                tenor_strategy=tenor_strategy_label,
                                training_window_days=training_window_days,
                            )
                        except Exception as e:
                            st.error(f"Adaptive schedule build failed: {e}")
                            schedule = None
                    if not schedule:
                        st.error(
                            "Adaptive schedule is empty — either the "
                            "data window is too short or all monthly "
                            "fits failed. Widen the date range or "
                            "lower the minimum training window."
                        )
                        st.stop()
                    # Stats for the success banner
                    n_entries = len(schedule)
                    n_with_green = sum(
                        1 for e in schedule
                        if any(
                            r >= 2.0
                            for c in e["clusters"]
                            for r in c["tenor_sojourn_ratios"].values()
                        )
                    )
                    st.info(
                        f"Built **{n_entries}** monthly entries "
                        f"({n_with_green} have at least one cluster "
                        f"with a green tenor). Saving preset…"
                    )
                elif is_dynamic_mode:
                    from core.wf_schedule import (
                        build_monthly_schedule, annotate_schedule_with_tenor,
                    )
                    with st.spinner(
                        f"Building walk-forward schedule for cluster "
                        f"{target_cluster} (this fits one GMM per month)…"
                    ):
                        try:
                            schedule = build_monthly_schedule(
                                data, pair_a, pair_b,
                                target_cluster=int(target_cluster),
                                confidence_pct=int(confidence),
                                K=int(wf_K),
                                backtest_start=hint_start,
                                backtest_end=hint_end,
                                min_training_days=int(wf_min_train),
                                seed=int(seed),
                                fit_hmm=True,
                            )
                        except Exception as e:
                            st.error(f"Schedule build failed: {e}")
                            schedule = None
                    if not schedule:
                        st.error(
                            "Schedule is empty — either the data window "
                            "is too short or all monthly fits failed. "
                            "Widen the date range or lower the minimum "
                            "training window."
                        )
                        st.stop()
                    schedule = annotate_schedule_with_tenor(
                        schedule, tenor_days
                    )
                    n_ok = sum(1 for e in schedule
                                 if e["sojourn_health"] == "ok")
                    st.info(
                        f"Built **{len(schedule)}** monthly entries "
                        f"({n_ok} green at {tenor_choice} tenor). "
                        f"Saving preset…"
                    )

                preset = build_preset(
                    pair_a=pair_a, pair_b=pair_b, tenor=tenor_choice,
                    confidence_pct=int(confidence),
                    target_cluster=int(target_cluster),
                    direction_label="Call (up-and-out)",
                    target_strike_delta=target_strike_delta,
                    target_ko_delta=target_ko_delta,
                    delta_choices=DELTA_CHOICES_LOCAL,
                    ko_delta_choices=KO_DELTA_CHOICES_LOCAL,
                    metadata={
                        "current_spot_a": float(S_a),
                        "current_spot_b": float(S_b),
                        "cluster_mu_a": float(mu_t[0]),
                        "cluster_mu_b": float(mu_t[1]),
                        "cluster_sigma_major": float(ax_t["major_sigma"]),
                        "cluster_sigma_minor": float(ax_t["minor_sigma"]),
                        "ellipse_half_width_a": float(delta_x),
                        "ellipse_half_width_b": float(delta_y),
                        "analytical_strike_delta_a": float(K_a_delta),
                        "analytical_strike_delta_b": float(K_b_delta),
                        "analytical_ko_delta_a": float(H_a_delta),
                        "analytical_ko_delta_b": float(H_b_delta),
                        "expected_sojourn_days": sojourn_days_val,
                        "sojourn_health": sojourn_health,
                        "wf_K": (int(wf_K)
                                    if (is_dynamic_mode or is_adaptive_mode)
                                    else None),
                        "wf_min_training_days": (int(wf_min_train)
                                                       if (is_dynamic_mode or is_adaptive_mode)
                                                       else None),
                        "tenor_strategy": (tenor_strategy_label
                                                if is_adaptive_mode else None),
                        "strike_strategy": (strike_strategy_label
                                                 if is_adaptive_mode else None),
                        "n_schedule_entries": (len(schedule)
                                                  if schedule else None),
                        "mode": ("adaptive" if is_adaptive_mode
                                    else "walk_forward_monthly"
                                    if is_dynamic_mode else "in_sample"),
                        "notes": (
                            f"Generated from app 10's Barrier guidance "
                            f"tab. "
                            + (
                                f"ADAPTIVE — engine picks nearest cluster + "
                                f"{tenor_strategy_label} tenor per trade "
                                f"date from {{1M, 6W, 2M, 10W, 3M}}. "
                                f"K={wf_K} clusters, "
                                f"{confidence}% Mahalanobis ellipse."
                                if is_adaptive_mode
                                else
                                f"Target cluster {target_cluster} "
                                f"({confidence}% Mahalanobis ellipse). "
                                f"Mode: "
                                f"{'walk-forward monthly (causal)' if is_dynamic_mode else 'in-sample'}."
                            )
                            + f" Spot at export: ({S_a:.4f}, {S_b:.4f})."
                        ),
                    },
                    gate_keys=(["hmm_dominant"]
                                  if (include_hmm_gate
                                        and not is_adaptive_mode)
                                  else [None]),
                    band_width=band_width,
                    dynamic_schedule=schedule,
                )
                try:
                    out_path = save_preset(export_dir, preset)
                    mode_tag = (
                        "adaptive (one preset, dynamic 5-D)"
                        if is_adaptive_mode
                        else "walk-forward (causal)"
                        if is_dynamic_mode
                        else "in-sample"
                    )
                    st.success(
                        f"✅ Saved {mode_tag} preset to `{out_path}`. "
                        f"Open app 9 (Backtest or Worst-of tab), expand "
                        f"the **'📥 Load preset from app 10'** section "
                        f"at the top, pick this preset, click Apply."
                    )
                except Exception as e:
                    st.error(f"Save failed: {e}")

        # =================================================================
        # Batch-generate: one preset per (cluster × tenor) combination
        # where the sojourn-tenor health check passes (ratio ≥ 2.0). Uses
        # the SAME band_width / gate / confidence / output folder
        # settings the user already picked above — the only thing that
        # changes across iterations is the (cluster, tenor) pair.
        # =================================================================
        st.markdown("---")
        if is_adaptive_mode:
            # =============================================================
            # Per-month audit (adaptive mode only)
            # =============================================================
            # Lets the user verify the framework on a chosen schedule
            # month: see the training data, all clusters, daily spot,
            # the cluster the engine picked each day, and the level →
            # delta conversion. The adaptive schedule is built on
            # first click and cached in session state keyed by the
            # current control settings, so subsequent month-clicks are
            # instant.
            # =============================================================
            st.markdown("##### 🔬 Per-month audit (adaptive mode)")
            st.caption(
                "Drill into any month's GMM fit. Shows the training "
                "window used, all K clusters' parameters, the daily "
                "spot path through the month, which cluster the "
                "engine picked each day, and the full level → delta "
                "conversion math. Build the audit cache first; "
                "afterward each month-click is instant."
            )
            # Cache key — invalidate if any input changes
            audit_key = (
                "_audit_adaptive",
                pair_a, pair_b,
                int(wf_K), int(confidence),
                int(wf_min_train), int(seed),
                tenor_strategy_label,
                # Training window mode (None or rolling-N) — fits
                # differ if this changes
                training_window_days,
                # Data window: only the endpoints matter for the
                # adaptive schedule's coverage
                pd.Timestamp(data.index.min()).isoformat(),
                pd.Timestamp(data.index.max()).isoformat(),
            )
            cache_state_key = "audit_adaptive_cache"
            existing = st.session_state.get(cache_state_key)
            cache_is_valid = (existing is not None
                                  and existing.get("key") == audit_key)
            colsa1, colsa2 = st.columns([3, 1])
            with colsa2:
                if not cache_is_valid:
                    btn_label = "🔨 Build audit cache"
                else:
                    btn_label = "🔄 Rebuild audit cache"
                build_audit_clicked = st.button(
                    btn_label, key="bg_audit_build",
                    use_container_width=True,
                    help=("Builds a fresh adaptive schedule using the "
                          "current settings. ~30-90 seconds. Cached "
                          "until any input changes."),
                )
            with colsa1:
                if cache_is_valid:
                    st.success(
                        f"Audit cache ready — {len(existing['schedule'])} "
                        f"monthly entries available."
                    )
                else:
                    st.info(
                        "Click **Build audit cache** to build a fresh "
                        "adaptive schedule for this pair pair and K. "
                        "This takes ~30-90 seconds and only needs to "
                        "happen once per setting change."
                    )
            if build_audit_clicked:
                from core.wf_schedule import build_adaptive_schedule
                with st.spinner(
                    f"Fitting {wf_K}-cluster GMM + HMM per month…"
                ):
                    sched = build_adaptive_schedule(
                        data, pair_a, pair_b,
                        confidence_pct=int(confidence),
                        K=int(wf_K),
                        backtest_start=pd.Timestamp(data.index.min()),
                        backtest_end=pd.Timestamp(data.index.max()),
                        min_training_days=int(wf_min_train),
                        seed=int(seed),
                        sojourn_threshold=2.0,
                        tenor_strategy=tenor_strategy_label,
                        training_window_days=training_window_days,
                    )
                if sched:
                    st.session_state[cache_state_key] = {
                        "key": audit_key, "schedule": sched,
                    }
                    st.rerun()
                else:
                    st.error("Schedule build returned empty. Widen "
                                "the date range or lower min training.")

            # Render the audit UI once cache is valid
            if cache_is_valid:
                _render_adaptive_audit(
                    schedule=existing["schedule"],
                    spot_panel=data, folder=folder,
                    pair_a=pair_a, pair_b=pair_b,
                    confidence_pct=int(confidence),
                    K_used=int(wf_K),
                    prefer_em=prefer_em,
                )
        else:
            st.markdown("##### 🟢 Batch: generate presets for all green combinations")
            # Caption adapts to the currently-selected mode so the user sees
            # what the batch will do BEFORE they click.
            if is_dynamic_mode:
                st.caption(
                    "**Walk-forward mode is active.** Iterates every "
                    "(cluster × tenor) combination, builds a fresh "
                    "walk-forward schedule per cluster (one schedule reused "
                    "across all tenors for that cluster — schedule is "
                    "tenor-independent), then saves a preset for each "
                    "(cluster, tenor) where the in-sample sojourn is "
                    "**green** at that tenor. Each schedule takes ~30–60 "
                    "seconds to build — full batch can take several minutes."
                )
            else:
                st.caption(
                    "Iterates every (cluster × tenor) combination, "
                    "recomputes the analytical ellipse + delta conversion "
                    "+ sojourn health check, and saves a static preset for "
                    "each combination that comes out **green** (expected "
                    "sojourn ≥ 2× the tenor). Skips amber and red. Uses "
                    "the band width, gate, and output folder settings "
                    "above. Tenors swept: 1M, 2M, 3M, 6M."
                )
        if is_adaptive_mode:
            pass  # batch not applicable — info shown above
        elif hmm is None:
            st.info(
                "Batch generation needs HMM transition probabilities for "
                "the sojourn check. Either install `hmmlearn` (see Tab 7 "
                "for instructions) or generate presets individually."
            )
        else:
            colsb = st.columns([3, 1])
            with colsb[1]:
                batch_clicked = st.button(
                    "🟢 Generate all green",
                    type="primary",
                    use_container_width=True,
                    key="bg_batch",
                    help=(
                        "Saves one preset per (cluster, tenor) where "
                        "E[sojourn] ≥ 2 × tenor. Re-running creates "
                        "additional dated files (no overwrites). "
                        "Output mode (static vs walk-forward) follows "
                        "the Export Mode radio above."
                    ),
                )
            if batch_clicked:
                # Tenors to sweep.
                _BATCH_TENORS = {
                    "1M": (21, 1 / 12),
                    "2M": (42, 2 / 12),
                    "3M": (63, 0.25),
                    "6M": (126, 0.5),
                }
                rows = []
                progress = st.progress(0.0, text="Starting batch…")
                total_combos = K_used * len(_BATCH_TENORS)
                idx = 0

                # In WF mode, building a schedule is expensive (~30-60s
                # per cluster) but depends only on the cluster — not on
                # tenor. So we cache schedules by cluster index and
                # reuse across the 4 tenors. This makes batch generation
                # of (K_used × 4) presets cost K_used schedule builds.
                wf_schedule_cache: dict[int, "list[dict]"] = {}
                if is_dynamic_mode:
                    from core.wf_schedule import (
                        build_monthly_schedule,
                        annotate_schedule_with_tenor,
                    )

                for k_iter in range(K_used):
                    for t_label, (t_days, t_years) in _BATCH_TENORS.items():
                        idx += 1
                        progress.progress(
                            idx / total_combos,
                            text=f"Cluster {k_iter} · {t_label}",
                        )

                        # Sojourn check FIRST — short-circuits cheap combos.
                        # NOTE: uses the *in-sample* HMM's transition matrix
                        # for the gate decision, so it stays consistent with
                        # the single-export path. In WF mode we *additionally*
                        # report per-month sojourns inside the schedule for
                        # auditability.
                        try:
                            A_kk = hmm["A"][k_iter, k_iter]
                            tau_iter = 1.0 / max(1e-9, 1.0 - A_kk)
                            ratio = tau_iter / t_days
                            if ratio >= 2.0:
                                health = "ok"
                            elif ratio >= 1.0:
                                health = "warn"
                            else:
                                health = "fail"
                        except Exception as e:
                            rows.append({
                                "Cluster": k_iter, "Tenor": t_label,
                                "Mode": ("WF" if is_dynamic_mode
                                            else "static"),
                                "E[sojourn] (d)": "—",
                                "Tenor (d)": t_days,
                                "Ratio": "—", "Health": "error",
                                "Status": f"sojourn calc failed: {e}",
                            })
                            continue
                        if health != "ok":
                            rows.append({
                                "Cluster": k_iter, "Tenor": t_label,
                                "Mode": ("WF" if is_dynamic_mode
                                            else "static"),
                                "E[sojourn] (d)": f"{tau_iter:.0f}",
                                "Tenor (d)": t_days,
                                "Ratio": f"{ratio:.2f}×",
                                "Health": health,
                                "Status": "skipped (not green)",
                            })
                            continue

                        # Sojourn passed — compute ellipse geometry +
                        # ATM-vol deltas for THIS (cluster, tenor).
                        mu_iter = gmm["means"][k_iter]
                        cov_iter = gmm["covariances"][k_iter]
                        ax_iter = cluster_axes(cov_iter)
                        dx_iter = float(np.sqrt(d2_thresh * cov_iter[0, 0]))
                        dy_iter = float(np.sqrt(d2_thresh * cov_iter[1, 1]))
                        H_a_iter = float(mu_iter[0] + dx_iter)
                        H_b_iter = float(mu_iter[1] + dy_iter)
                        K_a_iter = float(mu_iter[0])
                        K_b_iter = float(mu_iter[1])

                        # Pull ATM vol for THIS tenor.
                        try:
                            atm_p = load_panel(
                                folder, "VOL_ATM", t_label,
                                prefer=prefer_em,
                                pairs=(pair_a, pair_b),
                            )
                            if (atm_p.empty
                                    or pair_a not in atm_p.columns
                                    or pair_b not in atm_p.columns):
                                rows.append({
                                    "Cluster": k_iter, "Tenor": t_label,
                                    "Mode": ("WF" if is_dynamic_mode
                                                else "static"),
                                    "E[sojourn] (d)": f"{tau_iter:.0f}",
                                    "Tenor (d)": t_days,
                                    "Ratio": f"{ratio:.2f}×",
                                    "Health": health,
                                    "Status": "skipped (no ATM vol panel)",
                                })
                                continue
                            sigma_a_i = float(
                                atm_p[pair_a].dropna().iloc[-1]
                            ) / 100.0
                            sigma_b_i = float(
                                atm_p[pair_b].dropna().iloc[-1]
                            ) / 100.0
                        except Exception as e:
                            rows.append({
                                "Cluster": k_iter, "Tenor": t_label,
                                "Mode": ("WF" if is_dynamic_mode
                                            else "static"),
                                "E[sojourn] (d)": f"{tau_iter:.0f}",
                                "Tenor (d)": t_days,
                                "Ratio": f"{ratio:.2f}×",
                                "Health": health,
                                "Status": f"vol load failed: {e}",
                            })
                            continue

                        K_a_d_i = call_delta(S_a, K_a_iter, t_years, sigma_a_i)
                        K_b_d_i = call_delta(S_b, K_b_iter, t_years, sigma_b_i)
                        H_a_d_i = call_delta(S_a, H_a_iter, t_years, sigma_a_i)
                        H_b_d_i = call_delta(S_b, H_b_iter, t_years, sigma_b_i)
                        target_strike_d = float((K_a_d_i + K_b_d_i) / 2)
                        target_ko_d = float((H_a_d_i + H_b_d_i) / 2)

                        # If in WF mode, ensure we have a schedule for
                        # this cluster. Build it once, then reuse across
                        # all tenors for this cluster. annotate_schedule_
                        # with_tenor mutates the schedule in place to
                        # add per-entry sojourn_health for THIS tenor;
                        # since the same dict is reused across tenors, we
                        # MUST deep-copy before annotating, or each tenor's
                        # preset would carry only the most recent tenor's
                        # health labels.
                        schedule_for_preset = None
                        if is_dynamic_mode:
                            if k_iter not in wf_schedule_cache:
                                progress.progress(
                                    idx / total_combos,
                                    text=(f"Cluster {k_iter} · "
                                            f"building schedule "
                                            f"(~30-60s)…"),
                                )
                                try:
                                    sched = build_monthly_schedule(
                                        data, pair_a, pair_b,
                                        target_cluster=int(k_iter),
                                        confidence_pct=int(confidence),
                                        K=int(K_used),
                                        backtest_start=pd.Timestamp(
                                            data.index.min()),
                                        backtest_end=pd.Timestamp(
                                            data.index.max()),
                                        min_training_days=252,
                                        seed=int(seed),
                                        fit_hmm=True,
                                    )
                                    if not sched:
                                        wf_schedule_cache[k_iter] = []
                                    else:
                                        wf_schedule_cache[k_iter] = sched
                                except Exception as e:
                                    rows.append({
                                        "Cluster": k_iter, "Tenor": t_label,
                                        "Mode": "WF",
                                        "E[sojourn] (d)": f"{tau_iter:.0f}",
                                        "Tenor (d)": t_days,
                                        "Ratio": f"{ratio:.2f}×",
                                        "Health": health,
                                        "Status": (f"schedule build "
                                                     f"failed: {e}"),
                                    })
                                    continue
                            base_sched = wf_schedule_cache[k_iter]
                            if not base_sched:
                                rows.append({
                                    "Cluster": k_iter, "Tenor": t_label,
                                    "Mode": "WF",
                                    "E[sojourn] (d)": f"{tau_iter:.0f}",
                                    "Tenor (d)": t_days,
                                    "Ratio": f"{ratio:.2f}×",
                                    "Health": health,
                                    "Status": ("skipped (empty WF "
                                                "schedule — too little "
                                                "training data?)"),
                                })
                                continue
                            # Deep-copy the cached schedule + annotate
                            # with THIS tenor's health labels.
                            import copy
                            schedule_for_preset = copy.deepcopy(base_sched)
                            schedule_for_preset = (
                                annotate_schedule_with_tenor(
                                    schedule_for_preset, t_days
                                )
                            )

                        # Build & save preset. The presence of
                        # dynamic_schedule flips build_preset into WF
                        # mode (sets look_ahead_mode, embeds the schedule).
                        try:
                            preset_iter = build_preset(
                                pair_a=pair_a, pair_b=pair_b,
                                tenor=t_label,
                                confidence_pct=int(confidence),
                                target_cluster=int(k_iter),
                                direction_label="Call (up-and-out)",
                                target_strike_delta=target_strike_d,
                                target_ko_delta=target_ko_d,
                                delta_choices=DELTA_CHOICES_LOCAL,
                                ko_delta_choices=KO_DELTA_CHOICES_LOCAL,
                                metadata={
                                    "current_spot_a": float(S_a),
                                    "current_spot_b": float(S_b),
                                    "cluster_mu_a": float(mu_iter[0]),
                                    "cluster_mu_b": float(mu_iter[1]),
                                    "cluster_sigma_major": float(
                                        ax_iter["major_sigma"]),
                                    "cluster_sigma_minor": float(
                                        ax_iter["minor_sigma"]),
                                    "ellipse_half_width_a": float(dx_iter),
                                    "ellipse_half_width_b": float(dy_iter),
                                    "analytical_strike_delta_a": float(
                                        K_a_d_i),
                                    "analytical_strike_delta_b": float(
                                        K_b_d_i),
                                    "analytical_ko_delta_a": float(H_a_d_i),
                                    "analytical_ko_delta_b": float(H_b_d_i),
                                    "expected_sojourn_days": float(
                                        tau_iter),
                                    "sojourn_health": health,
                                    "batch_generated": True,
                                    "n_schedule_entries": (
                                        len(schedule_for_preset)
                                        if schedule_for_preset else None
                                    ),
                                    "notes": (
                                        f"Batch-generated "
                                        f"({'WF' if is_dynamic_mode else 'static'}). "
                                        f"Cluster {k_iter} "
                                        f"({confidence}% ellipse). "
                                        f"Sojourn {tau_iter:.0f}d / "
                                        f"tenor {t_days}d = {ratio:.2f}×."
                                    ),
                                },
                                gate_keys=(["hmm_dominant"]
                                              if include_hmm_gate
                                              else [None]),
                                band_width=band_width,
                                dynamic_schedule=schedule_for_preset,
                            )
                            out_p = save_preset(export_dir, preset_iter)
                            rows.append({
                                "Cluster": k_iter, "Tenor": t_label,
                                "Mode": ("WF" if is_dynamic_mode
                                            else "static"),
                                "E[sojourn] (d)": f"{tau_iter:.0f}",
                                "Tenor (d)": t_days,
                                "Ratio": f"{ratio:.2f}×",
                                "Health": health,
                                "Status": f"saved: {out_p.name}",
                            })
                        except Exception as e:
                            rows.append({
                                "Cluster": k_iter, "Tenor": t_label,
                                "Mode": ("WF" if is_dynamic_mode
                                            else "static"),
                                "E[sojourn] (d)": f"{tau_iter:.0f}",
                                "Tenor (d)": t_days,
                                "Ratio": f"{ratio:.2f}×",
                                "Health": health,
                                "Status": f"build/save failed: {e}",
                            })

                progress.empty()

                # Summary table — one row per attempted combo, with
                # status saying whether it was saved or why it was
                # skipped. Color-coded for legibility.
                if rows:
                    df_summary = pd.DataFrame(rows)
                    n_saved = sum(1 for r in rows
                                    if r["Status"].startswith("saved"))
                    n_skipped = sum(1 for r in rows
                                      if r["Status"].startswith("skipped"))
                    n_errored = sum(1 for r in rows
                                      if "failed" in r["Status"]
                                      or r["Status"].startswith("error"))
                    st.markdown(
                        f"**Batch complete:** {n_saved} saved · "
                        f"{n_skipped} skipped (not green or no vol) · "
                        f"{n_errored} errored — out of {len(rows)} "
                        f"total combinations attempted."
                    )

                    # Apply per-row coloring based on Status
                    def _row_color(row):
                        s = row["Status"]
                        if s.startswith("saved"):
                            return ["background-color: rgba(34,197,94,0.15)"
                                      ] * len(row)
                        if s.startswith("skipped"):
                            return ["background-color: rgba(245,158,11,0.10)"
                                      ] * len(row)
                        return ["background-color: rgba(239,68,68,0.15)"
                                  ] * len(row)
                    st.dataframe(
                        df_summary.style.apply(_row_color, axis=1),
                        use_container_width=True, hide_index=True,
                    )
                    if n_saved > 0:
                        st.info(
                            "💡 Open app 9 (Backtest or Worst-of tab) → "
                            "expand **'📥 Load preset from app 10'** at "
                            "the top — all newly-saved presets will be "
                            "in the dropdown, sorted newest-first. "
                            "Load and run each one to compare across "
                            "regimes and tenors."
                        )
                    if n_skipped == len(rows):
                        st.warning(
                            "No green combinations found. The dominant "
                            "regime may have an expected sojourn shorter "
                            "than 2× any of the swept tenors — consider "
                            "shorter tenors, or accept amber combinations "
                            "by generating presets individually with the "
                            "single-export button above."
                        )
                else:
                    st.warning("No combinations attempted — something "
                                "unexpected happened.")

    # Visual: dominant cluster ellipse + KO zones + current spot marker
    st.markdown("#### Visual placement")
    bar_fig = go.Figure()
    # KDE background
    xx_b, yy_b, zz_b = kde_grid(X, n=120)
    bar_fig.add_trace(go.Contour(
        x=xx_b[0, :], y=yy_b[:, 0], z=zz_b, showscale=False,
        colorscale="Blues", contours=dict(coloring="fill"),
        line=dict(color="rgba(255,255,255,0.0)"),
        opacity=0.55, hoverinfo="skip",
    ))
    # Target cluster 95% ellipse
    pts_t = ellipse_xy(mu_t, cov_t, d2_thresh)
    c_t = CLUSTER_COLORS[target_cluster % len(CLUSTER_COLORS)]
    bar_fig.add_trace(go.Scatter(
        x=pts_t[:, 0], y=pts_t[:, 1], mode="lines",
        line=dict(color=c_t, width=3),
        name=f"Cluster {target_cluster} · {confidence}% ellipse",
    ))
    # Other clusters' 95% ellipses (for context)
    for k in range(K_used):
        if k == target_cluster:
            continue
        pts_o = ellipse_xy(gmm["means"][k], gmm["covariances"][k], 5.99)
        c = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        bar_fig.add_trace(go.Scatter(
            x=pts_o[:, 0], y=pts_o[:, 1], mode="lines",
            line=dict(color=c, width=1.5, dash="dot"),
            name=f"Cluster {k} · 95%",
        ))
    # Current spot
    bar_fig.add_trace(go.Scatter(
        x=[S_a], y=[S_b], mode="markers",
        marker=dict(symbol="star", size=18, color="#fde047",
                      line=dict(color="black", width=1)),
        name="Current spot",
        hovertemplate=(f"Today: ({S_a:.4f}, {S_b:.4f})<extra></extra>"),
    ))
    # Strike lines (purple dotted)
    bar_fig.add_vline(x=K_a, line=dict(color="#a855f7", dash="dot", width=1.5),
                       annotation_text=f"K_{pair_a}",
                       annotation_position="top")
    bar_fig.add_hline(y=K_b, line=dict(color="#a855f7", dash="dot", width=1.5),
                       annotation_text=f"K_{pair_b}",
                       annotation_position="right")
    # Barrier lines (red dashed, up direction)
    bar_fig.add_vline(x=H_a_up, line=dict(color="#ef4444", dash="dash", width=2),
                       annotation_text=f"H_{pair_a}={H_a_up:.3f}",
                       annotation_position="top right")
    bar_fig.add_hline(y=H_b_up, line=dict(color="#ef4444", dash="dash", width=2),
                       annotation_text=f"H_{pair_b}={H_b_up:.3f}",
                       annotation_position="right")
    bar_fig.update_layout(
        title=f"Cluster {target_cluster} {confidence}% ellipse + barrier placement",
        xaxis_title=pair_a, yaxis_title=pair_b,
    )
    plotly_dark(bar_fig, height=560)
    st.plotly_chart(bar_fig, use_container_width=True)

    with st.expander("How to read this", expanded=False):
        st.markdown(
            """
- **Standard rule**: place barriers at the 95% Mahalanobis boundary of
  the cluster you're betting will hold. Inside this ellipse you have
  95% of in-regime moves; the barrier rarely triggers on normal
  in-regime drift, but KOs efficiently when the market leaves the regime.
- **Strike at the anchor** (K = μ) means the option is roughly ATM
  relative to *the regime's centre*, not relative to today's spot. If
  today's spot has drifted away from μ, you may want to set K closer
  to today's spot for a more typical ATM option.
- **Up vs down barriers**: the framework is direction-agnostic — the
  ellipse extends in both directions. For an up-and-out call you take
  the *upper* end of the ellipse; for a down-and-out put you take the
  *lower* end.
- **Delta conversion** is a single-vol GK approximation. The exact
  delta you'll trade at depends on the vol smile; use app 9's Pricer
  tab to get the precise number, and the bulk runner to evaluate the
  realised PnL of a range of structures around this suggestion.
            """
        )


# -----------------------------------------------------------------------------
# Tab 5 — Stationarity diagnostics
# -----------------------------------------------------------------------------
with tab_stat:
    st.markdown("### Stationarity diagnostics")
    st.caption(
        "Tests whether the GMM/HMM assumptions hold over the sample window. "
        "Two checks: (1) is the cluster anchor stable, or does it drift "
        "over time? (2) Are the residuals around the cluster centre "
        "actually bivariate normal, as the model assumes?"
    )

    # ---- Rolling cluster anchor ----
    st.markdown("#### Rolling cluster anchor")
    window = st.slider(
        "Rolling window (trading days)",
        min_value=60, max_value=min(500, max(120, n // 3)),
        value=min(120, max(60, n // 5)),
        help="Width of the rolling window used to compute a local "
              "estimate of the dominant cluster's anchor μ.",
    )

    # Simple proxy: rolling mean of the spot itself (the cluster
    # decomposition is too expensive to refit per-window for visual
    # purposes — and the rolling mean is the dominant cluster's anchor
    # in expectation when the dominant cluster carries most weight).
    roll_a = data[pair_a].rolling(window, min_periods=window // 2).mean()
    roll_b = data[pair_b].rolling(window, min_periods=window // 2).mean()

    drift_fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 vertical_spacing=0.04)
    drift_fig.add_trace(go.Scatter(
        x=data.index, y=data[pair_a], mode="lines",
        line=dict(color="#94a3b8", width=1), name=f"{pair_a} spot",
    ), row=1, col=1)
    drift_fig.add_trace(go.Scatter(
        x=roll_a.index, y=roll_a.values, mode="lines",
        line=dict(color="#ef4444", width=2),
        name=f"rolling {window}d mean",
    ), row=1, col=1)
    drift_fig.add_hline(y=gmm["means"][0, 0],
                          line=dict(color="white", dash="dot", width=1),
                          annotation_text="full-sample μ_0",
                          annotation_position="right", row=1, col=1)
    drift_fig.add_trace(go.Scatter(
        x=data.index, y=data[pair_b], mode="lines",
        line=dict(color="#94a3b8", width=1), name=f"{pair_b} spot",
        showlegend=False,
    ), row=2, col=1)
    drift_fig.add_trace(go.Scatter(
        x=roll_b.index, y=roll_b.values, mode="lines",
        line=dict(color="#ef4444", width=2),
        name=f"rolling {window}d mean", showlegend=False,
    ), row=2, col=1)
    drift_fig.add_hline(y=gmm["means"][0, 1],
                          line=dict(color="white", dash="dot", width=1),
                          annotation_text="full-sample μ_0",
                          annotation_position="right", row=2, col=1)
    drift_fig.update_yaxes(title=pair_a, row=1, col=1)
    drift_fig.update_yaxes(title=pair_b, row=2, col=1)
    drift_fig.update_xaxes(title="Date", row=2, col=1)
    plotly_dark(drift_fig, height=460, legend_below=False)
    drift_fig.update_layout(showlegend=False)
    st.plotly_chart(drift_fig, use_container_width=True)

    # Drift statistic: how much did the rolling mean wander vs the cluster σ?
    roll_drift_a = float(roll_a.max() - roll_a.min())
    roll_drift_b = float(roll_b.max() - roll_b.min())
    ax_0 = cluster_axes(gmm["covariances"][0])
    drift_ratio_a = roll_drift_a / (2 * ax_0["major_sigma"] + 1e-9)
    drift_ratio_b = roll_drift_b / (2 * ax_0["major_sigma"] + 1e-9)

    cs1, cs2 = st.columns(2)
    with cs1:
        if drift_ratio_a < 1.0:
            st.success(
                f"✅ {pair_a}: rolling-mean drift = "
                f"{roll_drift_a:.4f}, which is {drift_ratio_a:.2f}× the "
                f"dominant cluster's 1σ. Anchor is roughly stationary."
            )
        elif drift_ratio_a < 2.0:
            st.warning(
                f"⚠️ {pair_a}: rolling-mean drift = "
                f"{roll_drift_a:.4f} ({drift_ratio_a:.2f}× 1σ). "
                f"Moderate non-stationarity — refit on rolling windows."
            )
        else:
            st.error(
                f"❌ {pair_a}: rolling-mean drift = "
                f"{roll_drift_a:.4f} ({drift_ratio_a:.2f}× 1σ). "
                f"Strong non-stationarity — the full-sample model "
                f"averages over different regimes."
            )
    with cs2:
        if drift_ratio_b < 1.0:
            st.success(
                f"✅ {pair_b}: rolling-mean drift "
                f"= {roll_drift_b:.4f} ({drift_ratio_b:.2f}× 1σ)."
            )
        elif drift_ratio_b < 2.0:
            st.warning(
                f"⚠️ {pair_b}: rolling-mean drift = "
                f"{roll_drift_b:.4f} ({drift_ratio_b:.2f}× 1σ)."
            )
        else:
            st.error(
                f"❌ {pair_b}: rolling-mean drift = "
                f"{roll_drift_b:.4f} ({drift_ratio_b:.2f}× 1σ)."
            )

    # ---- Mahalanobis d² distribution vs χ²(2) ----
    st.markdown("#### Mahalanobis d² vs χ²(2) — bivariate normal goodness-of-fit")
    st.caption(
        "For each point, compute d² to its assigned cluster centre. "
        "If the bivariate normal assumption holds, these should follow "
        "a χ²(2) distribution. Heavier-than-χ²(2) tails mean the "
        "model is *under-estimating* tail risk — barriers placed at the "
        "95% d² boundary may be tested more often than 5% of the time."
    )

    # Compute d² for every point against its cluster
    d2_all = []
    for k in range(K_used):
        mask = gmm["labels"] == k
        if mask.sum() < 5:
            continue
        d2 = mahalanobis_d2(X[mask], gmm["means"][k],
                              gmm["covariances"][k])
        d2_all.append(d2)
    if d2_all:
        d2_concat = np.concatenate(d2_all)
        # Histogram + chi2(2) overlay
        chi2_fig = make_subplots(rows=1, cols=2,
                                    subplot_titles=["PDF comparison",
                                                       "Q-Q plot vs χ²(2)"])
        hist, edges = np.histogram(d2_concat, bins=40, density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        chi2_fig.add_trace(go.Bar(
            x=centres, y=hist, marker_color="#7dd3fc", opacity=0.6,
            name="empirical d²",
        ), row=1, col=1)
        xs = np.linspace(0, max(d2_concat.max(), 12), 200)
        chi2_fig.add_trace(go.Scatter(
            x=xs, y=chi2(df=2).pdf(xs), mode="lines",
            line=dict(color="#ef4444", width=2.5),
            name="χ²(2) reference",
        ), row=1, col=1)
        # Q-Q plot
        sample_sorted = np.sort(d2_concat)
        n_d = len(sample_sorted)
        theo = chi2(df=2).ppf((np.arange(1, n_d + 1) - 0.5) / n_d)
        chi2_fig.add_trace(go.Scatter(
            x=theo, y=sample_sorted, mode="markers",
            marker=dict(size=4, color="#86efac"),
            name="empirical quantile", showlegend=False,
        ), row=1, col=2)
        lim_max = max(theo.max(), sample_sorted.max()) * 1.05
        chi2_fig.add_trace(go.Scatter(
            x=[0, lim_max], y=[0, lim_max], mode="lines",
            line=dict(color="white", dash="dash", width=1.5),
            name="45° (perfect fit)", showlegend=False,
        ), row=1, col=2)
        chi2_fig.update_xaxes(title="d²", row=1, col=1)
        chi2_fig.update_yaxes(title="density", row=1, col=1)
        chi2_fig.update_xaxes(title="χ²(2) theoretical", row=1, col=2)
        chi2_fig.update_yaxes(title="empirical", row=1, col=2)
        plotly_dark(chi2_fig, height=400, legend_below=False)
        chi2_fig.update_layout(showlegend=True)
        st.plotly_chart(chi2_fig, use_container_width=True)

        # Tail concentration metric: how many points beyond χ²(2) 95%?
        pct_beyond_95 = (d2_concat > 5.99).mean() * 100
        pct_beyond_99 = (d2_concat > 9.21).mean() * 100
        st.markdown(
            f"""
**Tail statistics:**
- Fraction of points with d² > 5.99 (95% threshold under χ²(2)):
  **{pct_beyond_95:.1f}%** (expected: 5.0%)
- Fraction with d² > 9.21 (99% threshold): **{pct_beyond_99:.1f}%**
  (expected: 1.0%)

If empirical tail mass exceeds the expected χ² values by a meaningful
margin, the bivariate normal assumption under-estimates tail risk and
you should be more conservative with barrier placement (e.g. use the
99% rather than 95% ellipse, or accept that 95% barriers will KO
more often than the model predicts).
            """
        )

    with st.expander("How to read this", expanded=False):
        st.markdown(
            """
**Two diagnostics, two failure modes:**

- **Rolling anchor drift**: if the rolling mean wanders by more than
  1σ over the sample, your "single defended cluster" is really
  multiple regimes glued together. Either (a) shorten the sample to
  the most recent stable period, or (b) accept that the model
  parameters reflect an average over different policy regimes and
  use the most recent rolling window's parameters for barrier sizing.
- **d² vs χ²(2)**: the Mahalanobis distance is the natural unit-free
  distance under the bivariate normal model. If empirical d² values
  match χ²(2) closely, the model is well-specified. If empirical
  d² has fatter tails, your *barriers will be tested more often than
  the model predicts*. Diagnostically:
  - Mismatch in the body (small d²): cluster has the wrong shape
    (e.g. correlation isn't quite right).
  - Mismatch in the tail (large d²): real distribution is heavier-tailed
    than Gaussian (e.g. needs a t-distribution component, or fat-tail
    jump regime that isn't captured).
            """
        )


# -----------------------------------------------------------------------------
# Tab 6 — Summary
# -----------------------------------------------------------------------------
with tab_summary:
    st.markdown("### Summary")

    # Build a JSON-serialisable dict of the full model
    model_json = {
        "pair_a": pair_a,
        "pair_b": pair_b,
        "start_date": str(data.index.min().date()),
        "end_date": str(data.index.max().date()),
        "n_observations": int(n),
        "K_auto_BIC": int(K_auto),
        "K_used": int(K_used),
        "gmm": {
            "weights": gmm["weights"].tolist(),
            "means": gmm["means"].tolist(),
            "covariances": gmm["covariances"].tolist(),
            "bic": gmm["bic"],
            "aic": gmm["aic"],
            "loglik": gmm["loglik"],
        },
        "hmm": None if hmm is None else {
            "A": hmm["A"].tolist(),
            "means": hmm["means"].tolist(),
            "covariances": hmm["covariances"].tolist(),
            "stationary": hmm["stationary"].tolist(),
            "expected_sojourns_days": [
                1.0 / max(1e-9, 1.0 - hmm["A"][k, k]) for k in range(K_used)
            ],
            "loglik": hmm["loglik"],
        },
        "bic_sweep": bic_sweep,
    }

    # Headline numbers
    ax_dom = cluster_axes(gmm["covariances"][0])
    st.markdown(
        f"""
**Headline numbers for the dominant cluster:**

- Weight π₀ = **{gmm['weights'][0]:.3f}** (≈ {gmm['weights'][0]*100:.0f}% of time)
- Anchor (μ_{pair_a}, μ_{pair_b}) = **({gmm['means'][0, 0]:.4f}, {gmm['means'][0, 1]:.4f})**
- Major-axis σ = **{ax_dom['major_sigma']:.4f}**, minor-axis σ = **{ax_dom['minor_sigma']:.4f}**
- Anisotropy (major/minor) = **{ax_dom['anisotropy']:.2f}**
- Major-axis angle = **{ax_dom['angle_deg']:.1f}°**
        """
    )
    if hmm is not None:
        tau_0 = 1.0 / max(1e-9, 1.0 - hmm["A"][0, 0])
        st.markdown(
            f"""
- Expected sojourn in dominant regime = **{tau_0:.0f} trading days**
- Tenor compatibility (rule: ≥2× tenor):
    - 1M: **{'✅' if tau_0 >= 42 else '⚠️' if tau_0 >= 21 else '❌'}** ({tau_0/21:.1f}× 1M tenor)
    - 2M: **{'✅' if tau_0 >= 84 else '⚠️' if tau_0 >= 42 else '❌'}** ({tau_0/42:.1f}× 2M tenor)
    - 3M: **{'✅' if tau_0 >= 126 else '⚠️' if tau_0 >= 63 else '❌'}** ({tau_0/63:.1f}× 3M tenor)
            """
        )

    # Downloads
    st.markdown("#### Downloads")
    cd1, cd2 = st.columns(2)
    with cd1:
        st.download_button(
            "⬇ Download fitted model parameters (JSON)",
            data=json.dumps(model_json, indent=2).encode("utf-8"),
            file_name=f"joint_dist_{pair_a}_{pair_b}_K{K_used}.json",
            mime="application/json",
            use_container_width=True,
            help="Includes GMM weights/means/covariances, HMM transition "
                  "matrix, BIC sweep, and metadata. Suitable for loading "
                  "into a downstream pricing or analytics app.",
        )
    with cd2:
        # Decoded-state CSV
        if hmm is not None:
            decode_df = pd.DataFrame({
                "date": data.index,
                pair_a: X[:, 0],
                pair_b: X[:, 1],
                "viterbi_state": hmm["decoded_states"],
            })
            for k in range(K_used):
                decode_df[f"filt_prob_state_{k}"] = hmm["filtered_probs"][:, k]
            st.download_button(
                "⬇ Download decoded state series (CSV)",
                data=decode_df.to_csv(index=False).encode("utf-8"),
                file_name=f"joint_dist_states_{pair_a}_{pair_b}.csv",
                mime="text/csv",
                use_container_width=True,
                help="Every historical date labelled with its "
                      "Viterbi-decoded regime and filtered probabilities. "
                      "Use as a regime gate in the bulk runner.",
            )

    st.markdown("---")
    st.markdown(
        """
**Next steps** (referencing the implementation roadmap in the tutorial doc):

1. **Phase 2 — barrier guidance**: take the suggested (K, H) levels
   from Tab 4 to app 9's Pricer (for one-off pricing) or to the bulk
   runner (focus the strike-Δ / KO-Δ grid around those values).
2. **Phase 3 — regime-aware gates**: download the decoded state CSV
   above and add a new gate to `core/gates.py` that fires when the
   filtered probability of being in the dominant regime exceeds, say,
   0.7. This gives the bulk runner a model-driven gate that mirrors
   what the central bank is actually doing.
3. **Phase 4 — regime-conditioned reporting**: extend the bulk-runner
   summary with per-state breakdowns of Win%, KO%, and PnL.
        """
    )

    # Quick model-summary text for copy-paste
    with st.expander("Plain-text summary (for copy-paste)", expanded=False):
        lines = [
            f"Joint distribution analysis: {pair_a} × {pair_b}",
            f"Window: {data.index.min().date()} → {data.index.max().date()}  "
            f"({n} obs)",
            f"BIC-optimal K = {K_auto}; used K = {K_used}",
            "",
            "GMM clusters (sorted by weight):",
        ]
        for k in range(K_used):
            ax_k = cluster_axes(gmm["covariances"][k])
            lines.append(
                f"  Cluster {k}: π={gmm['weights'][k]:.3f}  "
                f"μ=({gmm['means'][k,0]:.4f}, {gmm['means'][k,1]:.4f})  "
                f"σ_major={ax_k['major_sigma']:.4f}  "
                f"σ_minor={ax_k['minor_sigma']:.4f}  "
                f"angle={ax_k['angle_deg']:.1f}°"
            )
        if hmm is not None:
            lines.extend([
                "",
                "HMM transition matrix:",
                np.array2string(hmm["A"], precision=4),
                "",
                "Expected sojourns (days): " + ", ".join(
                    f"state {k}={1/max(1e-9, 1-hmm['A'][k,k]):.0f}"
                    for k in range(K_used)
                ),
                "Stationary distribution: " + ", ".join(
                    f"state {k}={hmm['stationary'][k]:.3f}"
                    for k in range(K_used)
                ),
            ])
        st.code("\n".join(lines), language="text")


# -----------------------------------------------------------------------------
# Tab 7 — Fit & save per-pair regimes (Phase 3 — feeds the regime gates
# in core/gates.py)
# -----------------------------------------------------------------------------
with tab_fit_save:
    st.markdown("### Fit per-pair HMMs and save as regime gates")
    st.caption(
        "This tab fits a **univariate** HMM on each selected pair's spot "
        "history and saves the result as `market_data/regimes/<pair>.csv`. "
        "These files are auto-loaded by the bulk-runner app (app 9), where "
        "they power the `hmm_state_0`, `hmm_state_1`, and "
        "`hmm_prob_state_X_gt_Y` gates. Unlike Tab 3's joint analysis, "
        "this is per-pair (each leg of a worst-of structure gates on its "
        "own pair's regime)."
    )

    from core import regimes as _reg

    if not _reg.HMMLEARN_OK:
        st.error("This tab requires `hmmlearn`. "
                  "Install with `pip install hmmlearn`.")
    else:
        # =====================================================================
        # Configuration
        # =====================================================================
        st.markdown("#### Step 1 — select pairs and fit settings")
        cf1, cf2, cf3 = st.columns(3)
        with cf1:
            pairs_to_fit = st.multiselect(
                "Pairs to fit",
                all_pairs,
                default=[p for p in [pair_a, pair_b] if p in all_pairs],
                help="One HMM per pair. The defaults are the two pairs "
                      "you analysed jointly in the tabs above.",
                key="ts_pairs",
            )
        with cf2:
            K_fit = st.slider(
                "K (states per pair)",
                min_value=2, max_value=4, value=2,
                help="Number of regimes to fit per pair. Use the BIC sweep "
                      "below to pick K rigorously — typically 2 for "
                      "managed FX pairs.",
                key="ts_K",
            )
            n_restarts = st.slider(
                "EM restarts", min_value=3, max_value=15, value=5,
                help="Multiple restarts from different random inits. "
                      "EM is locally optimal; more restarts = more "
                      "robust fit, but slower.",
                key="ts_restarts",
            )
        with cf3:
            fit_mode = st.radio(
                "Fit mode",
                ["In-sample (full history, fast)",
                 "Walk-forward (refit periodically, causal)"],
                index=0,
                help="In-sample fits the HMM ONCE on the full history — "
                      "fast and good for diagnostics, but has parameter-"
                      "level look-ahead. Walk-forward refits "
                      "periodically using only prior data — "
                      "causally clean but ~refit_every× slower.",
                key="ts_mode",
            )
            wf_initial = wf_refit = None
            if fit_mode.startswith("Walk-forward"):
                wf_initial = st.number_input(
                    "Initial training window (days)",
                    min_value=126, max_value=1260, value=252, step=21,
                    help="Use this many days as the initial fit window. "
                          "No regime labels produced before this many "
                          "days of history.",
                    key="ts_wf_initial",
                )
                wf_refit = st.number_input(
                    "Refit cadence (days)",
                    min_value=5, max_value=126, value=21, step=1,
                    help="Refit the HMM every N trading days. 21 ≈ "
                          "monthly; 5 ≈ weekly (much slower).",
                    key="ts_wf_refit",
                )

        # =====================================================================
        # Optional BIC sweep — helps user pick K
        # =====================================================================
        with st.expander("📉 BIC sweep per pair (optional — helps pick K)",
                          expanded=False):
            st.caption(
                "Fits univariate HMMs at K = 1..4 per pair and reports BIC. "
                "Lower BIC = better fit-vs-complexity tradeoff. Use the "
                "minimum to pick K above."
            )
            if st.button("Run BIC sweep", key="ts_bic_run"):
                bic_rows = []
                bic_prog = st.progress(0.0, text="…")
                for i, p in enumerate(pairs_to_fit):
                    ser = data[p] if p in (pair_a, pair_b) else (
                        load_panel(folder, "SPOT", None, prefer=prefer_em,
                                     pairs=(p,))[p].dropna()
                    )
                    bic_prog.progress((i + 0.5) / max(len(pairs_to_fit), 1),
                                       text=f"BIC sweep: {p}")
                    res = _reg.bic_sweep(ser, K_max=4, seed=int(seed),
                                            n_restarts=3)
                    for k_, b in zip(res["K"], res["BIC"]):
                        bic_rows.append({"pair": p, "K": k_,
                                           "BIC": round(b, 1)})
                bic_prog.empty()
                if bic_rows:
                    bic_df = pd.DataFrame(bic_rows).pivot(
                        index="K", columns="pair", values="BIC")
                    st.dataframe(bic_df.style.highlight_min(axis=0,
                                                                color="#1e3a8a"),
                                  use_container_width=True)
                    st.caption("Highlighted cell = argmin BIC for that pair.")

        # =====================================================================
        # Fit + save
        # =====================================================================
        st.markdown("#### Step 2 — fit and save")
        existing = _reg.available_regime_pairs(folder)
        if existing:
            st.caption(
                f"Currently saved regime files in `{folder}/regimes/`: "
                + ", ".join(f"`{p}`" for p in existing)
            )
        cfb1, cfb2 = st.columns([3, 1])
        with cfb1:
            st.markdown(
                "Click **Fit & save all** to fit one HMM per pair and "
                "write each result to `market_data/regimes/<pair>.csv`. "
                "App 9 (the bulk runner) auto-loads these files when "
                "the data folder is set, so the HMM gates appear in the "
                "gate multi-select for any pair with a saved file."
            )
        with cfb2:
            fit_clicked = st.button("▶ Fit & save all",
                                       type="primary", key="ts_fit_all",
                                       use_container_width=True,
                                       disabled=not pairs_to_fit)

        if fit_clicked:
            progress = st.progress(0.0, text="Starting…")
            results: dict[str, pd.DataFrame] = {}
            for i, p in enumerate(pairs_to_fit):
                progress.progress(i / max(len(pairs_to_fit), 1),
                                    text=f"Fitting {p}…")
                try:
                    if p in (pair_a, pair_b):
                        ser = data[p]
                    else:
                        ser = load_panel(
                            folder, "SPOT", None, prefer=prefer_em,
                            pairs=(p,),
                        )[p].dropna()
                except Exception as e:
                    st.warning(f"Couldn't load {p}: {e}")
                    continue

                if fit_mode.startswith("Walk-forward"):
                    panel_p = _reg.fit_pair_regimes_walk_forward(
                        ser, K=K_fit, seed=int(seed),
                        initial_window=int(wf_initial),
                        refit_every=int(wf_refit),
                        n_restarts=n_restarts,
                    )
                else:
                    panel_p = _reg.fit_pair_regimes(
                        ser, K=K_fit, seed=int(seed),
                        n_restarts=n_restarts,
                    )
                if panel_p is None:
                    st.warning(f"Fit failed for {p}. Check data length / "
                                f"try a different K.")
                    continue
                _reg.save_regime_csv(folder, p, panel_p)
                _reg.register_regime_panel(p, panel_p)
                results[p] = panel_p
            progress.progress(1.0, text="Done.")
            progress.empty()
            if results:
                st.success(
                    f"✓ Fit & saved {len(results)} pair"
                    f"{'s' if len(results) > 1 else ''}: "
                    + ", ".join(results.keys())
                    + ". Restart app 9 (or click 'Rerun' on app 9) to "
                    "pick up the new regimes in the gate dropdowns."
                )

        # =====================================================================
        # Preview saved regimes
        # =====================================================================
        st.markdown("---")
        st.markdown("#### Step 3 — preview saved regimes")
        saved_now = _reg.available_regime_pairs(folder)
        if not saved_now:
            st.info("No saved regime files yet. Run the fit above.")
        else:
            preview_pair = st.selectbox(
                "Pair to preview",
                saved_now,
                key="ts_preview_pair",
            )
            panel_pv = _reg.load_regime_csv(folder, preview_pair)
            if panel_pv is None or panel_pv.empty:
                st.warning(f"Couldn't load regime file for {preview_pair}.")
            else:
                K_pv = int(panel_pv["n_states"].iloc[0])

                # Stats
                stats_rows = []
                for k in range(K_pv):
                    in_state = (panel_pv["state"] == k)
                    n_in = int(in_state.sum())
                    pct = n_in / len(panel_pv) * 100
                    mu = float(panel_pv[f"mu_state_{k}"].iloc[0])
                    sigma = float(panel_pv[f"sigma_state_{k}"].iloc[0])

                    # Empirical sojourn from contiguous runs
                    runs = []
                    cur = 0
                    for v in in_state.values:
                        if v:
                            cur += 1
                        elif cur > 0:
                            runs.append(cur)
                            cur = 0
                    if cur > 0:
                        runs.append(cur)
                    avg_sojourn = (sum(runs) / len(runs)) if runs else 0
                    stats_rows.append({
                        "State": k,
                        "n days": n_in,
                        "% of sample": f"{pct:.1f}%",
                        "μ (initial fit)": f"{mu:.4f}",
                        "σ (initial fit)": f"{sigma:.4f}",
                        "avg sojourn": f"{avg_sojourn:.0f} d",
                        "n runs": len(runs),
                    })
                st.dataframe(pd.DataFrame(stats_rows),
                              use_container_width=True, hide_index=True)

                # Spot path with state shading
                try:
                    spot_pv = load_panel(folder, "SPOT", None,
                                            prefer=prefer_em,
                                            pairs=(preview_pair,))[preview_pair]
                    spot_pv = spot_pv.loc[panel_pv.index.min():
                                              panel_pv.index.max()]
                except Exception:
                    spot_pv = pd.Series(dtype=float)
                if not spot_pv.empty:
                    pv_fig = go.Figure()
                    # Shade state regions
                    state_series = panel_pv["state"]
                    for k in range(K_pv):
                        in_k = (state_series == k).reindex(spot_pv.index,
                                                              fill_value=False)
                        if not in_k.any():
                            continue
                        c = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
                        # Find contiguous runs
                        arr = in_k.astype(int).values
                        edges = np.diff(np.concatenate([[0], arr, [0]]))
                        starts = np.where(edges == 1)[0]
                        ends = np.where(edges == -1)[0] - 1
                        for s_idx, e_idx in zip(starts, ends):
                            x0 = spot_pv.index[s_idx]
                            x1 = spot_pv.index[min(e_idx,
                                                     len(spot_pv) - 1)]
                            pv_fig.add_vrect(
                                x0=x0, x1=x1, fillcolor=c, opacity=0.12,
                                layer="below", line_width=0,
                            )
                    pv_fig.add_trace(go.Scatter(
                        x=spot_pv.index, y=spot_pv.values, mode="lines",
                        line=dict(color="#e0e7ff", width=1.6),
                        name=f"{preview_pair} spot",
                    ))
                    pv_fig.update_layout(
                        title=f"{preview_pair} spot + decoded regime",
                        xaxis_title="Date",
                        yaxis_title=f"{preview_pair}",
                    )
                    plotly_dark(pv_fig, height=380, legend_below=False)
                    pv_fig.update_layout(showlegend=False)
                    st.plotly_chart(pv_fig, use_container_width=True)

                # Download
                csv_bytes = panel_pv.reset_index().to_csv(
                    index=False).encode("utf-8")
                st.download_button(
                    f"⬇ Download regime panel for {preview_pair} (CSV)",
                    data=csv_bytes,
                    file_name=f"regime_{preview_pair}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key=f"ts_dl_{preview_pair}",
                )

        # =====================================================================
        # Look-ahead caveats (always shown — they're important)
        # =====================================================================
        with st.expander("⚠️ Look-ahead caveats — read before using in backtests",
                          expanded=False):
            st.markdown(
                """
There are two distinct look-ahead concerns in HMM-based gating:

**1. Parameter look-ahead (IN-SAMPLE mode):** the HMM parameters
(means, transition matrix) are fit using the full sample. When you
then apply those parameters to label dates in the *middle* of the
sample, the labels are influenced by observations that came *after*
those dates. This is fine for diagnostics ("what regime structure
does this data have?") but contaminates a backtest if you treat the
regime label as a real-time tradable signal.

**2. Observation look-ahead (VITERBI only):** the Viterbi-decoded
state at date *t* uses observations after *t* to label *t*, because
Viterbi finds the most-likely *complete* path through all states.
The filtered probability $P(s_t \\mid y_1, \\dots, y_t)$ does not —
it only uses observations up to *t*.

**Implications for the gates:**
- `hmm_state_X` uses Viterbi labels → has observation look-ahead.
- `hmm_prob_state_X_gt_<thresh>` uses filtered probs → no observation
  look-ahead (but still has parameter look-ahead in in-sample mode).

**For look-ahead-free backtests:** use the **Walk-forward** fit mode
above (it refits the HMM periodically using only data up to each refit
date). This is the academically correct approach for backtesting; it's
significantly slower but produces a causally-clean signal series.

**Practical recommendation:**
- For exploratory grid search over many strategy variants, use
  in-sample fits to keep things fast.
- For the final "headline" backtest of a candidate strategy, re-fit
  in walk-forward mode and verify the result holds.
                """
            )

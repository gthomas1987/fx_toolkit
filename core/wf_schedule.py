"""Walk-forward monthly cluster schedule (Phase WF-C).

This module fits a fresh joint GMM at the first business day of every
month using only observations strictly before that date, then derives
the analytical strike & barrier LEVELS (μ ± Mahalanobis ellipse extent)
that should be used for every trade opening in that month.

The result is a list-of-dicts schedule that gets embedded into a preset
JSON. The bulk-runner engine looks up each trade date's schedule entry
and uses those levels directly, bypassing the static delta-based
strike solver. This is the only mode that produces a truly out-of-
sample backtest — both the gate (causal HMM) and the structure (causal
GMM, refit monthly) use only past data.

# Causality protocol

Refit on date `t` (= first biz day of month M) uses observations on
dates `< t`. So the entry for January 2024 trains on data through the
last business day of December 2023 — strictly causal.

# State labelling stability across refits

After fitting, clusters are sorted by mixing weight π descending so
"cluster 0" is the most common regime. To keep labels stable across
refits (so cluster-0 in March is the same regime as cluster-0 in April),
we apply nearest-mean matching against the previous month's μ. This is
the same pattern as `core.regimes.fit_pair_regimes_walk_forward`.

# Tenor compatibility

The schedule also reports the implied HMM sojourn for the target
cluster at each refit, so callers can flag refit dates where the regime
became too unstable for the chosen tenor. The schedule itself doesn't
filter — that's the caller's decision.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2

try:
    from sklearn.mixture import GaussianMixture
    from hmmlearn.hmm import GaussianHMM
    _ML_OK = True
except ImportError:
    _ML_OK = False


def first_business_days_of_month(
    start: pd.Timestamp, end: pd.Timestamp
) -> list[pd.Timestamp]:
    """Return the first business day (Mon-Fri) in each month spanning
    [start, end] inclusive.

    Holidays aren't excluded — we use weekday only. This is sufficient
    because the schedule's `valid_from` is the *target* date for the
    refit, not necessarily a trading day; the engine looks up the
    schedule entry valid on the trade date, which is always a trading
    day.
    """
    out = []
    cur_month_start = pd.Timestamp(start.year, start.month, 1)
    end_month_start = pd.Timestamp(end.year, end.month, 1)
    while cur_month_start <= end_month_start:
        d = cur_month_start
        while d.weekday() > 4:  # Sat=5, Sun=6
            d += pd.Timedelta(days=1)
        if start <= d <= end:
            out.append(d)
        # Advance to next month
        if cur_month_start.month == 12:
            cur_month_start = pd.Timestamp(
                cur_month_start.year + 1, 1, 1)
        else:
            cur_month_start = pd.Timestamp(
                cur_month_start.year, cur_month_start.month + 1, 1)
    return out


def _align_clusters_to_prev(
    means: np.ndarray, covs: np.ndarray, weights: np.ndarray,
    prev_means: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Permute clusters so cluster k corresponds to the same regime as
    last month's cluster k. Returns (means, covs, weights) in the
    matched order.

    First call (prev_means is None) sorts by descending weight so
    cluster 0 = most common regime. Subsequent calls use nearest-mean
    Hungarian matching (greedy over permutations — K is small).
    """
    K = len(means)
    if prev_means is None:
        # Initial labelling: sort by weight descending
        order = np.argsort(-weights)
        return means[order], covs[order], weights[order]
    # Nearest-mean matching: find permutation minimising total
    # Euclidean distance between today's μ_k and last month's μ_k.
    from itertools import permutations
    best_perm, best_d = None, np.inf
    for p in permutations(range(K)):
        d = sum(np.linalg.norm(means[p[k]] - prev_means[k])
                  for k in range(K))
        if d < best_d:
            best_d, best_perm = d, p
    perm = np.array(best_perm)
    return means[perm], covs[perm], weights[perm]


def build_monthly_schedule(
    spot_panel: pd.DataFrame,
    pair_a: str, pair_b: str,
    target_cluster: int,
    confidence_pct: int,
    K: int,
    backtest_start: pd.Timestamp,
    backtest_end: pd.Timestamp,
    min_training_days: int = 252,
    seed: int = 42,
    n_init: int = 10,
    fit_hmm: bool = True,
) -> Optional[list[dict]]:
    """Build the monthly walk-forward schedule.

    Parameters
    ----------
    spot_panel
        DataFrame indexed by date with columns [pair_a, pair_b]. Should
        cover the full window from at least `min_training_days` before
        `backtest_start` to `backtest_end`.
    target_cluster
        The cluster index to extract parameters for at each refit. This
        is the cluster the trader has chosen to bet on (e.g. 0 for the
        dominant regime). Labels are stabilised across refits so this
        index refers to the same regime month-to-month.
    confidence_pct
        Mahalanobis ellipse confidence level (68 / 90 / 95 / 99).
    K
        Number of clusters to fit each month. Pick by BIC sweep first
        (in tab 2) so K is stable across the window.
    backtest_start, backtest_end
        The trading-date window for which schedule entries should be
        produced. Months whose first business day is in this window
        get an entry.
    min_training_days
        Don't produce a schedule entry until we have this many days of
        prior observations to fit on. With Scott's-rule density
        estimation, 252 (~1 year) is a sensible minimum.
    fit_hmm
        If True, also fit an HMM at each refit and report the cluster's
        expected sojourn. Useful for the health-check column but the
        slowest part of the loop (5x in profiling). Set False to skip.

    Returns
    -------
    List of monthly entries, or None if `hmmlearn`/`sklearn` are
    missing. Each entry has the keys documented in `_make_entry`.
    Empty list if the data is too short to produce even one entry.
    """
    if not _ML_OK:
        return None
    panel = spot_panel[[pair_a, pair_b]].dropna().sort_index()
    if len(panel) < min_training_days + 1:
        return []

    refit_dates = first_business_days_of_month(
        pd.Timestamp(backtest_start), pd.Timestamp(backtest_end)
    )
    if not refit_dates:
        return []

    d2_thresh = chi2(df=2).ppf(confidence_pct / 100.0)
    schedule: list[dict] = []
    prev_means: Optional[np.ndarray] = None

    for i, refit_date in enumerate(refit_dates):
        # Training data: STRICTLY before refit_date (causal).
        train = panel[panel.index < refit_date]
        if len(train) < min_training_days:
            continue

        # Validity window: from this refit_date until the next refit
        # date (exclusive), or until backtest_end (inclusive) for the
        # final entry.
        if i + 1 < len(refit_dates):
            valid_to = refit_dates[i + 1] - pd.Timedelta(days=1)
        else:
            valid_to = pd.Timestamp(backtest_end)

        try:
            gmm = GaussianMixture(
                n_components=K, covariance_type="full",
                n_init=n_init, random_state=seed, reg_covar=1e-4,
            ).fit(train.values)
        except Exception:
            # Fit failure — skip this month, schedule has a gap
            continue

        means, covs, weights = _align_clusters_to_prev(
            gmm.means_, gmm.covariances_, gmm.weights_, prev_means
        )
        prev_means = means

        if target_cluster >= K:
            continue
        mu = means[target_cluster]
        cov = covs[target_cluster]
        pi = float(weights[target_cluster])

        # Ellipse half-widths along each spot axis
        dx = float(np.sqrt(d2_thresh * cov[0, 0]))
        dy = float(np.sqrt(d2_thresh * cov[1, 1]))

        # Eigendecomposition for cluster geometry (reporting only)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major_sigma = float(np.sqrt(eigvals[1]))
        minor_sigma = float(np.sqrt(eigvals[0]))

        # Sojourn check (optional — slowest part of the loop)
        sojourn_days = None
        sojourn_health = "na"
        if fit_hmm:
            try:
                hmm = GaussianHMM(
                    n_components=K, covariance_type="full",
                    n_iter=300, random_state=seed, tol=1e-4,
                ).fit(train.values)
                # Match HMM states to GMM clusters via nearest-mean
                from itertools import permutations
                best_perm, best_d = None, np.inf
                for p in permutations(range(K)):
                    d = sum(np.linalg.norm(
                        hmm.means_[p[k]] - means[k]
                    ) for k in range(K))
                    if d < best_d:
                        best_d, best_perm = d, p
                perm = np.array(best_perm)
                A = hmm.transmat_[np.ix_(perm, perm)]
                A_kk = float(A[target_cluster, target_cluster])
                sojourn_days = 1.0 / max(1e-9, 1.0 - A_kk)
            except Exception:
                pass

        schedule.append({
            "valid_from": refit_date.date().isoformat(),
            "valid_to": valid_to.date().isoformat(),
            "fit_end_date": train.index[-1].date().isoformat(),
            "n_training_days": int(len(train)),
            "cluster_weight": pi,
            "mu_a": float(mu[0]),
            "mu_b": float(mu[1]),
            "cluster_sigma_major": major_sigma,
            "cluster_sigma_minor": minor_sigma,
            "ellipse_K_a": float(mu[0]),
            "ellipse_K_b": float(mu[1]),
            # Up barriers (μ + extent). The bulk runner can apply these
            # for up_and_out calls; down_and_out puts can use μ − extent
            # (engine reads `ellipse_H_a_up`/`_down`).
            "ellipse_H_a_up": float(mu[0] + dx),
            "ellipse_H_b_up": float(mu[1] + dy),
            "ellipse_H_a_dn": float(mu[0] - dx),
            "ellipse_H_b_dn": float(mu[1] - dy),
            "ellipse_dx_a": dx,
            "ellipse_dy_b": dy,
            "sojourn_days": sojourn_days,
            "sojourn_health": sojourn_health,
            # Note: sojourn_health filled in by caller using the
            # tenor it's targeting; we report sojourn_days only here.
        })

    return schedule


# Tenor constants — must match the bulk runner's TENOR_LIST in app 9
# Each entry is (label, business_days). The engine's adaptive path
# iterates these from shortest to longest looking for the first green.
_ADAPTIVE_TENORS = [
    ("1M", 21), ("6W", 31), ("2M", 42), ("10W", 52), ("3M", 63),
]


def build_adaptive_schedule(
    spot_panel: pd.DataFrame,
    pair_a: str, pair_b: str,
    confidence_pct: int,
    K: int,
    backtest_start: pd.Timestamp,
    backtest_end: pd.Timestamp,
    min_training_days: int = 252,
    seed: int = 42,
    n_init: int = 10,
    sojourn_threshold: float = 2.0,
    tenor_strategy: str = "shortest_green",
    training_window_days: Optional[int] = None,
) -> Optional[list[dict]]:
    """Build a fully-adaptive monthly walk-forward schedule.

    Unlike `build_monthly_schedule` which is parameterised on a single
    target cluster and a single tenor, this produces a richer schedule
    where each monthly entry contains parameters for ALL clusters and
    the engine picks (cluster, tenor) per trade date based on:

      1. **Cluster selection**: at each trade date, find the cluster
         whose μ is nearest (Euclidean) to current spot. This is the
         "you are here" classification.

      2. **Tenor selection**: from the selected cluster's per-tenor
         sojourn ratios, pick the shortest tenor where sojourn ≥
         `sojourn_threshold × tenor_days` (the "green" filter). If
         no tenor is green, skip the trade date.

    Training window mode:
      - `training_window_days=None` (default): EXPANDING window —
        each refit uses ALL prior observations. More data, but old
        defunct regimes still anchor the fit.
      - `training_window_days=N` (e.g. 504 for 2 years): ROLLING
        window — each refit uses only the last N prior days. Tracks
        current regime structure; forgets defunct historical regimes.
        Smaller sample so cluster parameter estimates are noisier.

    The schedule is causal (each monthly refit uses only data strictly
    before its valid_from), so the result is a fully out-of-sample
    backtest.

    Parameters
    ----------
    sojourn_threshold
        Multiplier on tenor_days for the green filter. 2.0 = sojourn
        must be at least 2× the tenor (standard).
    tenor_strategy
        How to pick from the green list. Options:
          - "shortest_green": pick shortest passing tenor (minimum
            premium — recommended for KO buyers)
          - "longest_green": pick longest passing tenor (max premium —
            for sellers)
          - "median_green": middle of the green list

    Returns a list of monthly entry dicts with these keys:
        valid_from, valid_to             — ISO date strings
        fit_end_date, n_training_days
        K                                — number of clusters
        clusters: list of {
            mu: [mu_a, mu_b],
            cov: [[c00, c01], [c10, c11]],
            sigma_a, sigma_b,
            weight,
            sojourn_days,
            ellipse_dx_a, ellipse_dy_b,
            tenor_sojourn_ratios: {tenor_label: ratio}
        }
        # Note: per-trade-date cluster selection and tenor selection
        # happen in the engine path — not pre-baked into the entry.

    Returns None if hmmlearn/sklearn unavailable.
    """
    if not _ML_OK:
        return None
    panel = spot_panel[[pair_a, pair_b]].dropna().sort_index()
    if len(panel) < min_training_days + 1:
        return []

    refit_dates = first_business_days_of_month(
        pd.Timestamp(backtest_start), pd.Timestamp(backtest_end)
    )
    if not refit_dates:
        return []

    d2_thresh = chi2(df=2).ppf(confidence_pct / 100.0)
    schedule: list[dict] = []
    prev_means: Optional[np.ndarray] = None

    for i, refit_date in enumerate(refit_dates):
        # Causal training slice: everything strictly before refit_date
        full_train = panel[panel.index < refit_date]
        if len(full_train) < min_training_days:
            continue
        # Apply rolling window if requested
        if (training_window_days is not None
                and len(full_train) > training_window_days):
            train = full_train.iloc[-training_window_days:]
        else:
            train = full_train
        if len(train) < min_training_days:
            continue

        if i + 1 < len(refit_dates):
            valid_to = refit_dates[i + 1] - pd.Timedelta(days=1)
        else:
            valid_to = pd.Timestamp(backtest_end)

        # Fit GMM for cluster geometry + HMM for sojourn estimates.
        try:
            gmm = GaussianMixture(
                n_components=K, covariance_type="full",
                n_init=n_init, random_state=seed, reg_covar=1e-4,
            ).fit(train.values)
        except Exception:
            continue

        means, covs, weights = _align_clusters_to_prev(
            gmm.means_, gmm.covariances_, gmm.weights_, prev_means
        )
        prev_means = means

        # HMM transitions to estimate per-cluster sojourn.
        try:
            hmm = GaussianHMM(
                n_components=K, covariance_type="full",
                n_iter=300, random_state=seed, tol=1e-4,
            ).fit(train.values)
            # Match HMM states to GMM clusters by nearest mean.
            from itertools import permutations
            best_perm_match, best_d_match = None, np.inf
            for p in permutations(range(K)):
                d_match = sum(np.linalg.norm(hmm.means_[p[k]] - means[k])
                                 for k in range(K))
                if d_match < best_d_match:
                    best_d_match, best_perm_match = d_match, p
            perm_hmm = np.array(best_perm_match)
            A = hmm.transmat_[np.ix_(perm_hmm, perm_hmm)]
            sojourns = [
                # Cap at 5000 trading days (~20 years) to avoid
                # absorbing-state singularities polluting the logs.
                # An A_kk ≈ 1.0 means the cluster was never observed
                # transitioning OUT of itself in the training data;
                # treating that as "infinitely sticky" is misleading
                # since real markets aren't, and it distorts the
                # green ratios that follow.
                min(1.0 / max(1e-9, 1.0 - float(A[k, k])), 5000.0)
                for k in range(K)
            ]
        except Exception:
            sojourns = [None] * K

        # Build cluster records.
        clusters_out = []
        for k in range(K):
            mu_k = means[k]
            cov_k = covs[k]
            # Ellipse half-widths along each spot axis
            dx_k = float(np.sqrt(d2_thresh * cov_k[0, 0]))
            dy_k = float(np.sqrt(d2_thresh * cov_k[1, 1]))
            sigma_a_k = float(np.sqrt(cov_k[0, 0]))
            sigma_b_k = float(np.sqrt(cov_k[1, 1]))

            # Per-tenor sojourn ratios — engine picks shortest green
            soj = sojourns[k]
            tenor_ratios = {}
            if soj is not None:
                for t_label, t_days in _ADAPTIVE_TENORS:
                    tenor_ratios[t_label] = float(soj / t_days)
            clusters_out.append({
                "cluster_index": int(k),
                "mu_a": float(mu_k[0]),
                "mu_b": float(mu_k[1]),
                "cov_aa": float(cov_k[0, 0]),
                "cov_ab": float(cov_k[0, 1]),
                "cov_ba": float(cov_k[1, 0]),
                "cov_bb": float(cov_k[1, 1]),
                "sigma_a": sigma_a_k,
                "sigma_b": sigma_b_k,
                "weight": float(weights[k]),
                "sojourn_days": (float(soj) if soj is not None else None),
                "ellipse_dx_a": dx_k,
                "ellipse_dy_b": dy_k,
                "tenor_sojourn_ratios": tenor_ratios,
            })

        schedule.append({
            "valid_from": refit_date.date().isoformat(),
            "valid_to": valid_to.date().isoformat(),
            "fit_start_date": train.index[0].date().isoformat(),
            "fit_end_date": train.index[-1].date().isoformat(),
            "n_training_days": int(len(train)),
            "training_window_days": (int(training_window_days)
                                            if training_window_days
                                            else None),
            "K": int(K),
            "confidence_pct": int(confidence_pct),
            "sojourn_threshold": float(sojourn_threshold),
            "tenor_strategy": tenor_strategy,
            "clusters": clusters_out,
        })

    return schedule


def get_training_slice_for_entry(
    spot_panel: pd.DataFrame,
    pair_a: str, pair_b: str,
    schedule_entry: dict,
) -> pd.DataFrame:
    """Reload the training data that was used for a given schedule
    entry's monthly fit.

    Honors `fit_start_date` if present (rolling-window schedules) so
    the slice reflects exactly the window used at fit time. Falls
    back to "everything up to fit_end_date" for older schedules
    (expanding-window).
    """
    panel = spot_panel[[pair_a, pair_b]].dropna().sort_index()
    fit_end = pd.Timestamp(schedule_entry["fit_end_date"])
    sliced = panel[panel.index <= fit_end]
    fit_start_iso = schedule_entry.get("fit_start_date")
    if fit_start_iso:
        fit_start = pd.Timestamp(fit_start_iso)
        sliced = sliced[sliced.index >= fit_start]
    return sliced


def select_cluster_and_tenor(
    schedule_entry: dict,
    spot_a: float, spot_b: float,
) -> Optional[dict]:
    """Per-date decision for the adaptive engine path.

    Given a monthly schedule entry and current spot, returns:
        {
            "cluster_index": k,
            "cluster_mu_a": ..., "cluster_mu_b": ...,
            "cluster_sigma_a": ..., "cluster_sigma_b": ...,
            "cluster_sojourn_days": ...,
            "chosen_tenor": "1M" (e.g.),
            "tenor_days": 21,
            "ellipse_K_a": cluster μ_a,
            "ellipse_K_b": cluster μ_b,
            "ellipse_H_a_up": K_a + dx,
            "ellipse_H_b_up": K_b + dy,
            "ellipse_H_a_dn": K_a - dx,
            "ellipse_H_b_dn": K_b - dy,
            "decision_log": "nearest=k=2, sojourn=145d, green_tenors=[1M,6W,2M], picked=1M",
        }
    or None if no green tenor is available for the nearest cluster (in
    which case the engine skips this trade date).

    Decision rule:
      1. Pick cluster whose (μ_a, μ_b) is nearest to (spot_a, spot_b)
         by Euclidean distance.
      2. From that cluster's tenor_sojourn_ratios, take all tenors
         where ratio ≥ sojourn_threshold.
      3. Apply tenor_strategy to pick from the green list:
           - "shortest_green" → first (sorted by tenor_days asc)
           - "longest_green"  → last
           - "median_green"   → middle
    """
    clusters = schedule_entry.get("clusters", [])
    if not clusters:
        return None
    # 1. Nearest cluster (Euclidean in spot space)
    best_k, best_d = None, float("inf")
    for c in clusters:
        d_k = ((c["mu_a"] - spot_a) ** 2
                 + (c["mu_b"] - spot_b) ** 2) ** 0.5
        if d_k < best_d:
            best_d, best_k = d_k, c["cluster_index"]
    chosen_cluster = next(c for c in clusters
                              if c["cluster_index"] == best_k)

    # 2. Build green list (sorted by tenor_days ascending)
    threshold = schedule_entry.get("sojourn_threshold", 2.0)
    tenor_ratios = chosen_cluster.get("tenor_sojourn_ratios", {})
    # Map tenor labels back to days using _ADAPTIVE_TENORS as the
    # canonical list
    tenor_days_lookup = dict(_ADAPTIVE_TENORS)
    green = [
        (t_label, tenor_days_lookup[t_label])
        for t_label in tenor_ratios
        if (tenor_ratios[t_label] >= threshold
              and t_label in tenor_days_lookup)
    ]
    if not green:
        return None
    green.sort(key=lambda x: x[1])  # ascending by days

    # 3. Apply tenor strategy
    strategy = schedule_entry.get("tenor_strategy", "shortest_green")
    if strategy == "shortest_green":
        chosen_tenor, tenor_days = green[0]
    elif strategy == "longest_green":
        chosen_tenor, tenor_days = green[-1]
    elif strategy == "median_green":
        chosen_tenor, tenor_days = green[len(green) // 2]
    else:
        chosen_tenor, tenor_days = green[0]

    K_a = chosen_cluster["mu_a"]
    K_b = chosen_cluster["mu_b"]
    dx_a = chosen_cluster["ellipse_dx_a"]
    dy_b = chosen_cluster["ellipse_dy_b"]
    green_labels = [t for t, _ in green]

    return {
        "cluster_index": int(best_k),
        "cluster_mu_a": K_a,
        "cluster_mu_b": K_b,
        "cluster_sigma_a": chosen_cluster["sigma_a"],
        "cluster_sigma_b": chosen_cluster["sigma_b"],
        "cluster_weight": chosen_cluster["weight"],
        "cluster_sojourn_days": chosen_cluster["sojourn_days"],
        "cluster_distance_from_spot": float(best_d),
        "chosen_tenor": chosen_tenor,
        "tenor_days": int(tenor_days),
        "green_tenors": green_labels,
        "ellipse_K_a": K_a,
        "ellipse_K_b": K_b,
        "ellipse_H_a_up": K_a + dx_a,
        "ellipse_H_b_up": K_b + dy_b,
        "ellipse_H_a_dn": K_a - dx_a,
        "ellipse_H_b_dn": K_b - dy_b,
        "decision_log": (
            f"nearest=c{best_k} (d={best_d:.2f}), "
            f"sojourn={chosen_cluster['sojourn_days']:.0f}d, "
            f"green={green_labels}, picked={chosen_tenor}"
        ),
    }


def lookup_schedule_entry(
    schedule: list[dict], trade_date: date
) -> Optional[dict]:
    """Find the schedule entry valid on `trade_date`. Returns None if
    `trade_date` falls outside all entries' valid windows.

    Implementation note: schedule is sorted by valid_from ascending
    (the build loop produces it in order), so linear scan is fine for
    schedules of ~36 entries. Used by the engine on every trade date.
    """
    td_iso = trade_date.isoformat()
    for entry in schedule:
        if entry["valid_from"] <= td_iso <= entry["valid_to"]:
            return entry
    return None


# =============================================================================
# Strike / KO grid + selection logic (adaptive mode)
# =============================================================================
# Buyer-of-EKO constraints:
#   - Strike Δ in {ATM(=0.50), 45Δ, 40Δ, 35Δ}   (call up-and-out, OTM strikes)
#   - KO Δ in {20Δ, 15Δ, 10Δ, 5Δ}               (farther OTM than strike)
#   - Strike Δ − KO Δ ≥ 25 (strict gap requirement)
# Strike strategy (after KO is fixed) chooses among the strikes that
# satisfy the gap constraint:
#   - "cheapest"    — lowest strike Δ (most OTM, lowest premium)
#   - "max_payoff"  — highest strike Δ (most ATM, biggest payoff window)
#   - "balanced"    — middle strike Δ
ALLOWED_STRIKE_DELTAS = [
    ("ATM", 0.50), ("45Δ", 0.45), ("40Δ", 0.40), ("35Δ", 0.35),
]
ALLOWED_KO_DELTAS = [
    ("20Δ", 0.20), ("15Δ", 0.15), ("10Δ", 0.10), ("5Δ", 0.05),
]
MIN_STRIKE_KO_GAP = 0.25
SNAP_TOLERANCE = 0.10  # snap KO to nearest allowed within ±10Δ; else skip


def _call_delta_gk(S: float, K: float, T: float, sigma: float,
                       r_d: float = 0.0, r_f: float = 0.0) -> float:
    """Garman-Kohlhagen call delta (foreign-domestic FX option). Used
    purely for converting LEVELS↔DELTAS in the strike selector. Returns
    Φ(d1) discounted by foreign rate.
    """
    if sigma <= 0 or T <= 0:
        return float(K <= S)
    import math
    from scipy.stats import norm
    d1 = (math.log(S / K) + (r_d - r_f + 0.5 * sigma * sigma) * T) \
              / (sigma * math.sqrt(T))
    return float(math.exp(-r_f * T) * norm.cdf(d1))


def _snap_to_allowed_ko(target_delta: float) -> Optional[tuple[str, float]]:
    """Snap `target_delta` to the nearest allowed KO delta, but only
    if the nearest is within `SNAP_TOLERANCE`. Returns (label, value)
    or None if no allowed KO is within tolerance.

    The snap is **inclusive** of the boundary values: target=0.205
    snaps to 20Δ (distance 0.005 ≤ tolerance). target=0.31 doesn't
    snap (distance 0.11 from 20Δ > tolerance) — skip.
    """
    best_label, best_val, best_dist = None, None, float("inf")
    for label, val in ALLOWED_KO_DELTAS:
        d = abs(val - target_delta)
        if d < best_dist:
            best_dist, best_label, best_val = d, label, val
    if best_dist > SNAP_TOLERANCE:
        return None
    return (best_label, best_val)


def _valid_strikes_for_ko(ko_delta: float) -> list[tuple[str, float]]:
    """All strike-Δ choices satisfying the gap constraint with given KO."""
    return [(lbl, v) for lbl, v in ALLOWED_STRIKE_DELTAS
              if v - ko_delta >= MIN_STRIKE_KO_GAP]


def _pick_strike_by_strategy(
    valid: list[tuple[str, float]], strategy: str,
) -> Optional[tuple[str, float]]:
    """Pick one strike from the valid list per the chosen strategy.

    `valid` is sorted by ALLOWED_STRIKE_DELTAS order (descending delta:
    ATM first, 35Δ last). Returns (label, value) or None if empty.
    """
    if not valid:
        return None
    if strategy == "cheapest":
        # Lowest strike Δ — most OTM — cheapest premium
        return valid[-1]
    elif strategy == "max_payoff":
        # Highest strike Δ — most ATM — biggest payoff window
        return valid[0]
    elif strategy == "balanced":
        # Middle of the valid list
        return valid[len(valid) // 2]
    else:
        # Default — cheapest (buyer-friendly)
        return valid[-1]


def select_strikes_and_barriers(
    cluster: dict,
    spot_a: float, spot_b: float,
    T_years: float,
    sigma_a: float, sigma_b: float,
    strike_strategy: str = "cheapest",
) -> Optional[dict]:
    """Given a cluster + market state + strike strategy, return the
    (K_a, H_a, K_b, H_b) spot levels for a call up-and-out worst-of,
    or None if the constraints can't be satisfied.

    Per-leg algorithm:

      1. Compute the cluster's upper edge: `upper_edge = μ + δx` (where
         δx is the 95% Mahalanobis half-width along this axis).
      2. **Skip if spot is already above the upper edge** — regime is
         broken, an up-and-out call shouldn't be priced.
      3. Convert `upper_edge` to a call delta at current spot/vol/T.
         This is the "natural" KO delta the cluster geometry suggests.
      4. **Snap to nearest allowed KO Δ** in {20, 15, 10, 5}, but only
         if within `SNAP_TOLERANCE`. Otherwise skip.
      5. **Filter strike Δ choices** to those satisfying gap ≥ 25 with
         the snapped KO Δ.
      6. **Pick strike Δ** per `strike_strategy`.
      7. **Solve back to spot levels**: K such that call_delta(spot, K,
         T, σ) = strike_Δ; H such that call_delta(spot, H, T, σ) = ko_Δ.

    Both legs must succeed independently — if either leg fails any
    check, the whole trade is skipped (returns None). This matches the
    worst-of structure's all-or-nothing nature.

    Returns a dict with `K_a, H_a, K_b, H_b, strike_delta_a, ko_delta_a,
    strike_delta_b, ko_delta_b, strike_label_a, ko_label_a, ..., reason`
    on success. On failure returns None and the caller should look at
    debug fields in the cluster (not stored here for simplicity).
    """
    # Cluster geometry — upper edge for each axis
    mu_a, mu_b = cluster["mu_a"], cluster["mu_b"]
    dx_a, dy_b = cluster["ellipse_dx_a"], cluster["ellipse_dy_b"]
    upper_a, upper_b = mu_a + dx_a, mu_b + dy_b

    # Step 2 — spot must be inside the cluster's upper-edge envelope
    if spot_a > upper_a or spot_b > upper_b:
        return None

    # Step 3 — convert each upper edge to a call delta
    ko_d_a_raw = _call_delta_gk(spot_a, upper_a, T_years, sigma_a)
    ko_d_b_raw = _call_delta_gk(spot_b, upper_b, T_years, sigma_b)

    # Step 4 — snap to allowed KO, skip if out of tolerance
    snapped_a = _snap_to_allowed_ko(ko_d_a_raw)
    snapped_b = _snap_to_allowed_ko(ko_d_b_raw)
    if snapped_a is None or snapped_b is None:
        return None
    ko_label_a, ko_val_a = snapped_a
    ko_label_b, ko_val_b = snapped_b

    # Step 5 — valid strikes per leg under gap constraint
    valid_strikes_a = _valid_strikes_for_ko(ko_val_a)
    valid_strikes_b = _valid_strikes_for_ko(ko_val_b)
    if not valid_strikes_a or not valid_strikes_b:
        return None

    # Step 6 — apply strategy per leg
    strike_a_choice = _pick_strike_by_strategy(valid_strikes_a, strike_strategy)
    strike_b_choice = _pick_strike_by_strategy(valid_strikes_b, strike_strategy)
    if strike_a_choice is None or strike_b_choice is None:
        return None
    strike_label_a, strike_val_a = strike_a_choice
    strike_label_b, strike_val_b = strike_b_choice

    # Step 7 — solve back to spot levels via the engine's existing solvers
    from core.ko_solvers import solve_strike_from_delta
    try:
        K_a = solve_strike_from_delta(
            "call", strike_val_a, spot_a, T_years, sigma_a, 0.0, 0.0
        )
        H_a = solve_strike_from_delta(
            "call", ko_val_a, spot_a, T_years, sigma_a, 0.0, 0.0
        )
        K_b = solve_strike_from_delta(
            "call", strike_val_b, spot_b, T_years, sigma_b, 0.0, 0.0
        )
        H_b = solve_strike_from_delta(
            "call", ko_val_b, spot_b, T_years, sigma_b, 0.0, 0.0
        )
    except Exception:
        return None

    return {
        "K_a": float(K_a),
        "H_a": float(H_a),
        "K_b": float(K_b),
        "H_b": float(H_b),
        "strike_delta_a": strike_val_a,
        "strike_label_a": strike_label_a,
        "ko_delta_a": ko_val_a,
        "ko_label_a": ko_label_a,
        "strike_delta_b": strike_val_b,
        "strike_label_b": strike_label_b,
        "ko_delta_b": ko_val_b,
        "ko_label_b": ko_label_b,
        "cluster_upper_edge_delta_a": float(ko_d_a_raw),
        "cluster_upper_edge_delta_b": float(ko_d_b_raw),
    }


def annotate_schedule_with_tenor(
    schedule: list[dict], tenor_days: int
) -> list[dict]:
    """Add a `sojourn_health` flag to each entry based on the chosen
    tenor. Mutates entries in place and also returns the list. Pure
    post-processing — no model refitting."""
    for entry in schedule:
        soj = entry.get("sojourn_days")
        if soj is None:
            entry["sojourn_health"] = "na"
        else:
            ratio = soj / max(tenor_days, 1)
            if ratio >= 2.0:
                entry["sojourn_health"] = "ok"
            elif ratio >= 1.0:
                entry["sojourn_health"] = "warn"
            else:
                entry["sojourn_health"] = "fail"
    return schedule

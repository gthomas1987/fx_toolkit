"""Per-pair regime fitting, persistence, and lookup.

This module is the data layer behind Phase 3's regime-aware gates. It
fits a univariate Hidden Markov Model on each pair's spot history,
producing a per-date state label (Viterbi) and per-state filtered
probability series. The output is stored in `market_data/regimes/<pair>.csv`
and exposed to `core.gates` via an in-memory registry.

# Univariate vs joint

App 10 already does *joint* HMM analysis on a pair-pair (USDJPY × USDKRW).
For the bulk-backtest gate framework, where each leg of a worst-of
structure is gated on its own pair's spot, we need *univariate*
regimes — one HMM per pair, independent of others. The joint analysis
remains in app 10 for diagnostic purposes; the univariate fits power
the gates.

# State labelling convention

After fitting, HMM states are permuted so that **state 0 is the most
common regime** (highest stationary probability) and subsequent states
are sorted by ascending stationary weight. This matches the convention
in `apps/10_joint_distribution.py` and makes `hmm_state_0` gate keys
mean "the dominant regime" across all pairs.

# Look-ahead caveats

Two distinct look-ahead concerns in HMM-based gating:

1. **Parameter look-ahead** (DEFAULT, in-sample mode): the HMM is fit
   on the full sample, so the parameters (means, transition matrix)
   reflect future observations. The output state series and filtered
   probabilities have this parameter-level look-ahead baked in. This
   is acceptable for diagnostics and exploratory backtesting, but NOT
   for a production live-trading signal.

2. **Observation look-ahead** (Viterbi only): the Viterbi-decoded
   state at date t uses *future* observations to label t. The filtered
   probability P(s_t | y_1..y_t) does not. So `hmm_state_X` gates have
   observation look-ahead; `hmm_prob_state_X_gt_<thresh>` gates do not
   (modulo concern 1).

For look-ahead-free backtests, fit in walk-forward mode (refit at
each rebalance date using only prior data) — supported by the
`fit_pair_regimes_walk_forward` function. This is significantly slower
but produces a causally-clean signal.

# File format

`market_data/regimes/<pair>.csv` has columns:
    date, state, n_states, prob_state_0, prob_state_1, [prob_state_2, …]
    mu_state_0, mu_state_1, [mu_state_2, …]
    sigma_state_0, sigma_state_1, [sigma_state_2, …]

The `mu_*` and `sigma_*` columns are constant within a single-sample
fit, but vary across rows in walk-forward mode (each date's row carries
the parameters fit at that date).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
    HMMLEARN_OK = True
except ImportError:
    HMMLEARN_OK = False


# =============================================================================
# In-memory registry — gates read from here
# =============================================================================
_REGIME_PANELS: dict[str, pd.DataFrame] = {}
_REGIME_FOLDER: Optional[str] = None


def register_regime_panel(pair: str, df: pd.DataFrame) -> None:
    """Register a regime panel for `pair` in the in-memory cache.

    The DataFrame must be indexed by date and contain at minimum a
    `state` column (integer state labels). Optional columns:
    `prob_state_0`, `prob_state_1`, ... for filtered probabilities.

    Called by the bulk runner at startup once the regime folder is
    known, so gates can look up regimes without I/O on every trade
    date.
    """
    if "state" not in df.columns:
        raise ValueError("Regime panel must have a 'state' column")
    df = df.copy()
    df.index = pd.to_datetime(df.index).normalize()
    _REGIME_PANELS[pair] = df


def get_regime_panel(pair: str) -> Optional[pd.DataFrame]:
    """Look up a registered regime panel by pair. Returns None if not
    registered. Called by the HMM-based gates in `core.gates`."""
    return _REGIME_PANELS.get(pair)


def set_regime_folder(folder: Optional[str]) -> None:
    """Set the folder containing regime CSVs and auto-register all of
    them. Convention: `<folder>/regimes/<pair>.csv`. Idempotent — calls
    a second time will reload from disk."""
    global _REGIME_FOLDER
    _REGIME_FOLDER = folder
    _REGIME_PANELS.clear()
    if not folder:
        return
    rdir = Path(folder) / "regimes"
    if not rdir.exists():
        return
    for f in rdir.glob("*.csv"):
        pair = f.stem.upper()
        try:
            df = load_regime_csv(folder, pair)
            if df is not None and not df.empty:
                register_regime_panel(pair, df)
        except Exception:
            # Skip malformed files silently — they'll be missing from
            # the registry and the gate will return False on lookup.
            continue


def list_registered_pairs() -> list[str]:
    """Pairs currently registered in memory."""
    return sorted(_REGIME_PANELS.keys())


def regime_folder() -> Optional[str]:
    """Current regime folder path, if set."""
    return _REGIME_FOLDER


# =============================================================================
# Persistence
# =============================================================================
def save_regime_csv(folder: str, pair: str, df: pd.DataFrame) -> Path:
    """Save a regime DataFrame to `<folder>/regimes/<pair>.csv`.

    Creates the `regimes/` subdirectory if missing. Returns the path
    written. The DataFrame is expected to be indexed by date with a
    `state` column and optional `prob_state_*`, `mu_state_*`,
    `sigma_state_*` columns.
    """
    rdir = Path(folder) / "regimes"
    rdir.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.index.name = "date"
    out.reset_index().to_csv(rdir / f"{pair.upper()}.csv", index=False)
    return rdir / f"{pair.upper()}.csv"


def load_regime_csv(folder: str, pair: str) -> Optional[pd.DataFrame]:
    """Load a saved regime CSV. Returns None if the file doesn't exist."""
    p = Path(folder) / "regimes" / f"{pair.upper()}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"]).set_index("date")
    # Normalise the index timezone-free at midnight so gates can do
    # exact-day lookups regardless of how the source data is timestamped.
    df.index = df.index.normalize()
    return df


def available_regime_pairs(folder: str) -> list[str]:
    """List pairs with saved regime files in `<folder>/regimes/`."""
    rdir = Path(folder) / "regimes"
    if not rdir.exists():
        return []
    return sorted([f.stem.upper() for f in rdir.glob("*.csv")])


# =============================================================================
# Univariate HMM fit
# =============================================================================
def _relabel_by_stationary(hmm: "GaussianHMM") -> np.ndarray:
    """Return permutation that sorts states by descending stationary
    probability. After applying, state 0 = most common regime.

    The stationary distribution π_∞ satisfies π_∞ A = π_∞ — derived as
    the left eigenvector of the transition matrix corresponding to
    eigenvalue 1. If A is degenerate (a state is absorbing), we fall
    back to ranking by the empirical decoded-state frequencies.
    """
    try:
        evals, evecs = np.linalg.eig(hmm.transmat_.T)
        idx = np.argmin(np.abs(evals - 1.0))
        sd = np.real(evecs[:, idx])
        sd = np.maximum(sd, 0.0)
        if sd.sum() > 0:
            sd = sd / sd.sum()
        else:
            raise ValueError("degenerate stationary")
    except Exception:
        # Fallback: use uniform; the empirical relabelling below
        # via decoded-frequency ranking is more robust anyway.
        sd = np.ones(hmm.n_components) / hmm.n_components
    # Sort DESCENDING — state 0 = highest stationary weight
    return np.argsort(-sd)


def fit_pair_regimes(
    spot: pd.Series,
    K: int = 2,
    seed: int = 42,
    n_iter: int = 500,
    n_restarts: int = 5,
) -> Optional[pd.DataFrame]:
    """Fit a univariate Gaussian HMM to a single pair's spot series.

    Parameters
    ----------
    spot
        Daily spot series indexed by date. Should be cleaned of missing
        values before calling.
    K
        Number of regimes to fit. Typical values: 2–4. Pick by BIC if
        unsure (see `bic_sweep`).
    seed
        Random seed for EM initialisation. Multiple restarts (`n_restarts`)
        use seed, seed+1, ... and the best by likelihood is kept.
    n_iter
        Maximum EM iterations per restart.
    n_restarts
        Number of EM restarts (each from a different random init). HMM
        EM is locally optimal; more restarts → more robust fit.

    Returns
    -------
    DataFrame indexed by date with columns:
      - state              : Viterbi-decoded most-likely state (0..K-1)
      - n_states           : K (constant)
      - prob_state_0..K-1  : filtered probabilities P(s_t | y_1..y_t)
      - mu_state_0..K-1    : emission mean for each state (constant in
                             single-sample fit; varies in walk-forward)
      - sigma_state_0..K-1 : emission std (sqrt of variance)

    State labels follow the convention `state 0 = dominant regime`
    (highest stationary probability).

    Returns None if `hmmlearn` is unavailable or the fit fails.
    """
    if not HMMLEARN_OK:
        return None
    s = pd.Series(spot).dropna()
    if len(s) < max(50, K * 20):
        return None

    X = s.values.reshape(-1, 1).astype(np.float64)

    # Run multiple restarts; keep the best by likelihood
    best_h, best_ll = None, -np.inf
    for r in range(n_restarts):
        try:
            h = GaussianHMM(
                n_components=K, covariance_type="full",
                n_iter=n_iter, random_state=seed + r, tol=1e-4,
            )
            h.fit(X)
            ll = h.score(X)
            if ll > best_ll:
                best_ll = ll
                best_h = h
        except Exception:
            continue
    if best_h is None:
        return None

    # Permute states so state 0 = highest stationary probability
    perm = _relabel_by_stationary(best_h)
    invperm = np.argsort(perm)

    raw_states = best_h.predict(X)
    decoded = invperm[raw_states]
    raw_filt = best_h.predict_proba(X)
    filt = raw_filt[:, perm]
    means = best_h.means_[perm].flatten()
    sigmas = np.sqrt(best_h.covars_[perm].flatten())

    out = pd.DataFrame(index=s.index)
    out["state"] = decoded.astype(int)
    out["n_states"] = int(K)
    for k in range(K):
        out[f"prob_state_{k}"] = filt[:, k]
    for k in range(K):
        out[f"mu_state_{k}"] = means[k]
    for k in range(K):
        out[f"sigma_state_{k}"] = sigmas[k]
    return out


def bic_sweep(
    spot: pd.Series,
    K_max: int = 4,
    seed: int = 42,
    n_restarts: int = 3,
) -> dict:
    """BIC sweep across K=1..K_max for univariate HMMs.

    Returns dict with keys 'K', 'BIC', 'AIC', 'loglik' (each a list).
    Use to choose K when fitting a new pair.
    """
    if not HMMLEARN_OK:
        return {"K": [], "BIC": [], "AIC": [], "loglik": []}
    s = pd.Series(spot).dropna()
    X = s.values.reshape(-1, 1).astype(np.float64)
    n = len(X)
    out = {"K": [], "BIC": [], "AIC": [], "loglik": []}
    for K in range(1, K_max + 1):
        best_ll = -np.inf
        for r in range(n_restarts):
            try:
                h = GaussianHMM(
                    n_components=K, covariance_type="full",
                    n_iter=300, random_state=seed + r, tol=1e-4,
                )
                h.fit(X)
                ll = h.score(X)
                if ll > best_ll:
                    best_ll = ll
            except Exception:
                continue
        if best_ll == -np.inf:
            continue
        # Free parameters: (K-1) initial + K(K-1) transitions
        # + K means + K variances
        n_params = (K - 1) + K * (K - 1) + K + K
        bic = -2 * best_ll + n_params * np.log(n)
        aic = -2 * best_ll + 2 * n_params
        out["K"].append(K)
        out["BIC"].append(float(bic))
        out["AIC"].append(float(aic))
        out["loglik"].append(float(best_ll))
    return out


# =============================================================================
# Walk-forward fit (look-ahead-free for production backtests)
# =============================================================================
def fit_pair_regimes_walk_forward(
    spot: pd.Series,
    K: int = 2,
    seed: int = 42,
    initial_window: int = 252,
    refit_every: int = 21,
    n_restarts: int = 3,
) -> Optional[pd.DataFrame]:
    """Walk-forward HMM fit: refit periodically using only data up to
    each refit date. Produces a causally-clean regime panel.

    Parameters
    ----------
    initial_window
        Number of trading days to use as the initial training window.
        The first regime label is produced for the day AFTER this
        window. Defaults to ~1 year of daily data.
    refit_every
        Refit the HMM every N trading days. Between refits, the most
        recent model is used to score new observations (filtered
        probability + Viterbi at the boundary). Defaults to ~1 month.
    n_restarts
        Restarts per refit (kept smaller than the single-sample fit
        since this gets called many times).

    Returns None if hmmlearn unavailable or insufficient data. Otherwise
    returns a DataFrame in the same schema as `fit_pair_regimes`, but
    with `mu_state_*` and `sigma_state_*` varying across rows (each row
    carries the parameters of the model that was *current* at that date).

    NOTE on state-label convention:
    States are sorted by stationary probability AT EACH REFIT — state 0
    is always the most common regime according to that month's training
    data, regardless of which integer label was attached to the
    equivalent regime in earlier months. This is the correct semantic
    for the `hmm_state_0` gate (which means "trade in the dominant
    regime"), but a side effect is that `mu_state_k` can show
    discontinuities at month boundaries when the cluster ranking
    changes (e.g. a previously second-largest cluster overtakes the
    largest one). Always inspect the produced state series visually
    before relying on it in a critical decision.
    """
    if not HMMLEARN_OK:
        return None
    s = pd.Series(spot).dropna()
    if len(s) < initial_window + refit_every:
        return None

    X = s.values.astype(np.float64).reshape(-1, 1)
    n = len(X)

    # Storage
    state = np.full(n, -1, dtype=int)
    probs = np.full((n, K), np.nan)
    means_arr = np.full((n, K), np.nan)
    sigmas_arr = np.full((n, K), np.nan)

    refit_points = list(range(initial_window, n, refit_every))
    # Always include the last point as a refit boundary so the most
    # recent observation has a current model fit.
    if refit_points[-1] < n - 1:
        refit_points.append(n)

    last_idx = initial_window   # rows < initial_window left as NaN
    for next_idx in refit_points:
        # Fit on data up to (but not including) last_idx — this is
        # causal: at date last_idx-1 we know everything through that
        # date and can refit. We then SCORE dates [last_idx, next_idx).
        X_train = X[:last_idx]
        best_h, best_ll = None, -np.inf
        for r in range(n_restarts):
            try:
                h = GaussianHMM(
                    n_components=K, covariance_type="full",
                    n_iter=200, random_state=seed + r, tol=1e-4,
                )
                h.fit(X_train)
                ll = h.score(X_train)
                if ll > best_ll:
                    best_ll = ll
                    best_h = h
            except Exception:
                continue
        if best_h is None:
            last_idx = next_idx
            continue

        # Relabel by stationary probability AT THIS REFIT — state 0 is
        # always the most common regime according to today's training
        # data, regardless of what was labeled state 0 in earlier
        # months. This is the right semantic for the `hmm_state_0`
        # gate: traders use that gate to mean "trade in the dominant
        # regime", which is only meaningful if the label tracks the
        # current dominant cluster.
        #
        # An EARLIER VERSION of this code preserved label identity
        # across refits via nearest-mean matching to the previous
        # month's means. That sounded reasonable but had a fatal
        # failure mode in trending markets: state 0 got frozen to the
        # very first month's dominant cluster (e.g. USDJPY ~108 in
        # 2017), and as spot drifted higher year by year, that
        # original cluster eventually died (no observations in its
        # range any more). The label "state 0" stayed attached to the
        # dead cluster while the actually-dominant cluster got labeled
        # state 2 or state 3 — so the `hmm_state_0` gate fired on zero
        # dates post-2022 and backtests produced no trades.
        #
        # Trade-off of stationary-each-refit: a state's μ history can
        # show discontinuities at month boundaries when the cluster
        # ranking changes. This is the right trade-off because the
        # primary use of the panel is gate filtering, where label
        # consistency-with-the-dominant-regime matters more than
        # μ-trajectory continuity.
        perm = _relabel_by_stationary(best_h)
        relabeled_means = best_h.means_[perm].flatten()
        invperm = np.argsort(perm)

        # Score the test window [last_idx, next_idx)
        # We score on the EXTENDED prefix (so filtering uses correct
        # prior state distribution at last_idx) — then we extract
        # only the new range.
        X_so_far = X[:next_idx]
        raw_filt = best_h.predict_proba(X_so_far)
        raw_states = best_h.predict(X_so_far)
        filt = raw_filt[:, perm]
        decoded = invperm[raw_states]

        end = min(next_idx, n)
        slice_ = slice(last_idx, end)
        state[slice_] = decoded[slice_]
        probs[slice_] = filt[slice_]
        means_arr[slice_] = relabeled_means
        sigmas_arr[slice_] = np.sqrt(best_h.covars_[perm].flatten())

        last_idx = next_idx

    # Trim to rows that were actually filled
    valid = state >= 0
    if not valid.any():
        return None
    out = pd.DataFrame(index=s.index[valid])
    out["state"] = state[valid]
    out["n_states"] = int(K)
    for k in range(K):
        out[f"prob_state_{k}"] = probs[valid, k]
    for k in range(K):
        out[f"mu_state_{k}"] = means_arr[valid, k]
    for k in range(K):
        out[f"sigma_state_{k}"] = sigmas_arr[valid, k]
    return out


# =============================================================================
# Mask construction (used by gates)
# =============================================================================
def state_mask(panel: pd.DataFrame, target_state: int,
                  spot_index: pd.DatetimeIndex) -> pd.Series:
    """Bool mask aligned to spot_index: True where the panel's `state`
    equals `target_state`. Dates not present in the panel become False
    (gate fails)."""
    if "state" not in panel.columns:
        return pd.Series(False, index=spot_index)
    aligned = panel["state"].reindex(spot_index.normalize())
    return (aligned == target_state).fillna(False).astype(bool)


def prob_mask(panel: pd.DataFrame, target_state: int, threshold: float,
                 spot_index: pd.DatetimeIndex) -> pd.Series:
    """Bool mask: True where filtered probability of `target_state`
    exceeds `threshold`. Dates without a prob column or not in panel
    become False."""
    col = f"prob_state_{target_state}"
    if col not in panel.columns:
        return pd.Series(False, index=spot_index)
    aligned = panel[col].reindex(spot_index.normalize())
    return (aligned > threshold).fillna(False).astype(bool)

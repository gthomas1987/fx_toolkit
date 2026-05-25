"""Entry gates for the backtest engine.

A gate is a per-trade-date boolean filter. When a gate returns False on a
given date, no NEW trade is opened on that date; existing open positions
are unaffected (gate only sits between candidate-date and entry).

# Gate categories

**Trend filters** — align entry with prevailing direction. Useful for
KO-call buyers who want to ride a directional move without being
whipsawed by mean-reversion. Available:

  - spot_above_50dma       — fast trend reaction (spot > 50-day MA)
  - spot_above_200dma      — slow trend confirmation (spot > 200-day MA)
  - dma_20_above_50        — short-term trend (20DMA > 50DMA)
  - dma_50_above_200       — golden cross (50DMA > 200DMA), classic regime

**Vol-regime filters** — avoid high-RV environments where barriers
become more likely to KO. Available:

  - realized_vol_below_p75 — 20d realized vol below its 252d 75th pct
  - realized_vol_below_p50 — 20d realized vol below its 252d median

# Adding a new gate

Register it in GATE_REGISTRY:

    "key_snake_case": ("Display Label", compute_fn)

where compute_fn takes a spot Series and returns a bool Series indexed
by the same dates. NaN values (e.g. before a moving-window indicator
fills) become False so we don't trade on undefined indicators.

For gates that need more than spot (e.g. vol panel, RR/BF), generalize
the compute_fn signature when that need arises — for now spot is enough.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Built-in gates — Trend filters
# -----------------------------------------------------------------------------
def _spot_above_50dma(spot: pd.Series) -> pd.Series:
    """True when today's spot > 50-day moving average of spot.

    Fast trend reaction. Flips state often; suitable for capturing
    short-term momentum but prone to whipsaws.
    """
    dma = spot.rolling(window=50, min_periods=50).mean()
    return spot > dma


def _spot_above_200dma(spot: pd.Series) -> pd.Series:
    """True when today's spot > 200-day moving average of spot.

    Slow trend confirmation. Flips state rarely; used to identify
    durable bull/bear regimes. Spends ~12 months silent at start
    (200d burn-in).
    """
    dma = spot.rolling(window=200, min_periods=200).mean()
    return spot > dma


def _dma_20_above_50(spot: pd.Series) -> pd.Series:
    """True when 20-day MA > 50-day MA.

    Short-term trend signal. Less noisy than spot>50DMA, captures
    'is the recent average moving up?' Typically lags spot by 5-10d.
    """
    dma20 = spot.rolling(window=20, min_periods=20).mean()
    dma50 = spot.rolling(window=50, min_periods=50).mean()
    return dma20 > dma50


def _dma_50_above_200(spot: pd.Series) -> pd.Series:
    """True when 50-day MA > 200-day MA — the classic 'golden cross'.

    Most durable trend regime indicator. State changes are rare and
    typically signal multi-month directional moves. Spends ~12 months
    silent at start.
    """
    dma50 = spot.rolling(window=50, min_periods=50).mean()
    dma200 = spot.rolling(window=200, min_periods=200).mean()
    return dma50 > dma200


# -----------------------------------------------------------------------------
# Built-in gates — Vol-regime filters
# -----------------------------------------------------------------------------
def _realized_vol_20d(spot: pd.Series) -> pd.Series:
    """Helper: 20-day annualized realized vol from log returns."""
    log_ret = np.log(spot / spot.shift(1))
    return log_ret.rolling(window=20, min_periods=20).std() * np.sqrt(252)


def _realized_vol_below_p75(spot: pd.Series) -> pd.Series:
    """True when 20d realized vol is below its 252d 75th percentile.

    Filters out high-RV regimes — the top quartile of vol days where
    barriers are most likely to hit. Keeps ~75% of days when conditions
    are normal; tightens during vol spikes.
    """
    rv = _realized_vol_20d(spot)
    p75 = rv.rolling(window=252, min_periods=252).quantile(0.75)
    return rv < p75


def _realized_vol_below_p50(spot: pd.Series) -> pd.Series:
    """True when 20d realized vol is below its 252d median.

    Stricter version — only enters in the calmer half of the vol
    regime. Will roughly halve trade count vs. ungated.
    """
    rv = _realized_vol_20d(spot)
    p50 = rv.rolling(window=252, min_periods=252).quantile(0.50)
    return rv < p50


# -----------------------------------------------------------------------------
# Built-in gates — Combo (trend ∧ vol-regime)
# -----------------------------------------------------------------------------
def _spot_above_50dma_and_rv_below_p75(spot: pd.Series) -> pd.Series:
    """Fast trend AND not-spike-vol.

    The bread-and-butter combo: ride a short-term trend but avoid days
    when realized vol is in the top quartile (barriers more likely to
    hit, premium more likely to be expensive).
    """
    return _spot_above_50dma(spot) & _realized_vol_below_p75(spot)


def _golden_cross_and_rv_below_p75(spot: pd.Series) -> pd.Series:
    """Durable trend regime AND not-spike-vol.

    Looser than the strict combo but more selective than the fast-trend
    version — only enters during 50/200 bull regime AND avoids vol
    spikes. Captures macro-trend periods like extended USD-strength
    runs while sitting out vol panics.
    """
    return _dma_50_above_200(spot) & _realized_vol_below_p75(spot)


def _golden_cross_and_rv_below_p50(spot: pd.Series) -> pd.Series:
    """Strictest: durable trend regime AND below-median realized vol.

    Only enters during 50/200 bull regime AND in the calmer half of the
    vol distribution. Designed for premium-buyers who want maximum
    confidence in both directional and quiet-vol conditions; will
    significantly cut trade count vs ungated.
    """
    return _dma_50_above_200(spot) & _realized_vol_below_p50(spot)


# -----------------------------------------------------------------------------
# HMM regime helpers (Phase 3)
# -----------------------------------------------------------------------------
# These look up regime panels via `core.regimes.get_regime_panel(pair)`,
# which is populated by the caller (typically the bulk-runner app) via
# `core.regimes.set_regime_folder(folder)`. If no panel is registered
# for the pair, the gate fails closed — returns all False, no trades.
# That matches the "fail safe" behaviour of the other gates when their
# indicator is undefined.
#
# The `spot.name` of the input series identifies the pair. If a Series
# is passed in without a name (rare in this codebase), the gate logs a
# warning via NaN-becomes-False and produces no trades.
def _hmm_state_mask(spot: pd.Series, target_state: int) -> pd.Series:
    """True on dates where the pair's Viterbi-decoded HMM state equals
    `target_state`. False elsewhere (including dates with no regime
    data for this pair)."""
    from core.regimes import get_regime_panel, state_mask
    pair = getattr(spot, "name", None)
    if not pair:
        return pd.Series(False, index=spot.index)
    panel = get_regime_panel(pair)
    if panel is None:
        return pd.Series(False, index=spot.index)
    return state_mask(panel, target_state, spot.index)


def _hmm_prob_mask(spot: pd.Series, target_state: int,
                      threshold: float) -> pd.Series:
    """True on dates where the filtered probability of the pair being
    in `target_state` exceeds `threshold`."""
    from core.regimes import get_regime_panel, prob_mask
    pair = getattr(spot, "name", None)
    if not pair:
        return pd.Series(False, index=spot.index)
    panel = get_regime_panel(pair)
    if panel is None:
        return pd.Series(False, index=spot.index)
    return prob_mask(panel, target_state, threshold, spot.index)


def _hmm_dominant_mask(spot: pd.Series,
                          lookback_days: int = 30) -> pd.Series:
    """LABEL-ROBUST version of the `hmm_state_0` gate.

    The original `hmm_state_X` gate fires when Viterbi-decoded state ==
    X, which is fragile when state labels drift across walk-forward
    refits or when the in-sample fit attached label 0 to a regime that
    was historically dominant but isn't currently dominant (multi-
    epoch data problem).

    This gate is robust to that: it asks "is the market CURRENTLY in
    the dominant regime?" by:

    1. Computing `dominant_state_t` = argmax of filtered probabilities
       on date t.
    2. Computing `dominant_state_recent` = the modal value of
       dominant_state over the past `lookback_days` (≈ 1 month).
    3. Firing True when `dominant_state_t == dominant_state_recent` —
       i.e. "the regime we're in today is the regime we've been in
       lately."

    This captures the trader's intent ("trade in the prevailing
    regime") regardless of which integer label is attached to it, and
    works correctly for both in-sample and walk-forward fitted panels.

    Fails closed (returns all False) when no regime panel is registered
    for the pair, or when the panel lacks `prob_state_*` columns.
    """
    from core.regimes import get_regime_panel
    pair = getattr(spot, "name", None)
    if not pair:
        return pd.Series(False, index=spot.index)
    panel = get_regime_panel(pair)
    if panel is None:
        return pd.Series(False, index=spot.index)
    K = int(panel["n_states"].iloc[0]) if "n_states" in panel else None
    if K is None:
        return pd.Series(False, index=spot.index)
    prob_cols = [f"prob_state_{k}" for k in range(K)
                    if f"prob_state_{k}" in panel.columns]
    if not prob_cols:
        return pd.Series(False, index=spot.index)
    # Per-date dominant state by filtered prob argmax
    probs = panel[prob_cols].values
    panel_dom_state = probs.argmax(axis=1)
    panel_dom_ser = pd.Series(panel_dom_state, index=panel.index)
    # Recent-window dominant: rolling mode of dominant state over
    # lookback_days. Using the mode (most-common) keeps brief regime
    # excursions from changing the "prevailing" label.
    recent_dom = (panel_dom_ser
                    .rolling(window=lookback_days, min_periods=5)
                    .apply(lambda x: pd.Series(x).mode().iloc[0],
                              raw=False))
    # Gate fires when current dominant == recent prevailing dominant
    fires = (panel_dom_ser == recent_dom)
    # Reindex to spot's index, fill missing with False
    aligned = fires.reindex(spot.index.normalize()).fillna(False).astype(bool)
    aligned.index = spot.index
    return aligned


# -----------------------------------------------------------------------------
# Registry — `key: (label, compute_fn)`
# -----------------------------------------------------------------------------
GATE_REGISTRY: dict[str, tuple[str, Callable[[pd.Series], pd.Series]]] = {
    # Trend filters
    "spot_above_50dma":      ("Spot > 50DMA",          _spot_above_50dma),
    "spot_above_200dma":     ("Spot > 200DMA",         _spot_above_200dma),
    "dma_20_above_50":       ("20DMA > 50DMA",         _dma_20_above_50),
    "dma_50_above_200":      ("50DMA > 200DMA (golden cross)",
                                _dma_50_above_200),
    # Vol-regime filters
    "realized_vol_below_p75": ("Realized vol (20d) < 252d p75",
                                  _realized_vol_below_p75),
    "realized_vol_below_p50": ("Realized vol (20d) < 252d median",
                                  _realized_vol_below_p50),
    # Combo: trend ∧ vol-regime
    "spot_above_50dma_and_rv_below_p75": (
        "Spot > 50DMA  ∧  RV < p75",
        _spot_above_50dma_and_rv_below_p75,
    ),
    "golden_cross_and_rv_below_p75": (
        "Golden cross  ∧  RV < p75",
        _golden_cross_and_rv_below_p75,
    ),
    "golden_cross_and_rv_below_p50": (
        "Golden cross  ∧  RV < median (strict)",
        _golden_cross_and_rv_below_p50,
    ),
    # HMM regime gates — registered separately below so they can pull
    # the regime panel via `core.regimes`. Keys follow the convention:
    #   hmm_state_<k>              : Viterbi-decoded state == k
    #   hmm_prob_state_<k>_gt_<th> : filtered prob of state k > threshold
    # See `core.regimes` for the look-ahead caveats and labelling
    # convention (state 0 = dominant cluster).
    "hmm_state_0": ("HMM regime: in state 0 (dominant)",
                       lambda s: _hmm_state_mask(s, 0)),
    "hmm_state_1": ("HMM regime: in state 1",
                       lambda s: _hmm_state_mask(s, 1)),
    "hmm_dominant": (
        "HMM regime: currently in prevailing regime "
        "(label-robust — works for any HMM fit)",
        _hmm_dominant_mask,
    ),
    "hmm_prob_state_0_gt_0.5": (
        "HMM filtered P(state 0) > 0.5",
        lambda s: _hmm_prob_mask(s, 0, 0.5),
    ),
    "hmm_prob_state_0_gt_0.7": (
        "HMM filtered P(state 0) > 0.7",
        lambda s: _hmm_prob_mask(s, 0, 0.7),
    ),
    "hmm_prob_state_1_gt_0.5": (
        "HMM filtered P(state 1) > 0.5",
        lambda s: _hmm_prob_mask(s, 1, 0.5),
    ),
    "hmm_prob_state_1_gt_0.7": (
        "HMM filtered P(state 1) > 0.7",
        lambda s: _hmm_prob_mask(s, 1, 0.7),
    ),
}


# -----------------------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------------------
def gate_keys() -> list[str]:
    """All registered gate keys, in registration order."""
    return list(GATE_REGISTRY.keys())


def gate_label(key: Optional[str]) -> str:
    """Display label for a gate key. Returns '(none)' if key is None/empty."""
    if not key:
        return "(none)"
    if key in GATE_REGISTRY:
        return GATE_REGISTRY[key][0]
    return key  # unknown key — show raw


def compute_gate_mask(spot: pd.Series, gate_key: Optional[str]
                        ) -> Optional[pd.Series]:
    """Compute a bool mask aligned to spot.index, or None if no gate.

    NaN values in the underlying indicator become False (gate fails),
    so the engine simply doesn't trade on dates where the indicator is
    not yet defined.
    """
    if not gate_key:
        return None
    if gate_key not in GATE_REGISTRY:
        raise ValueError(
            f"Unknown gate {gate_key!r}. Known gates: {gate_keys()}"
        )
    _, fn = GATE_REGISTRY[gate_key]
    mask = fn(spot)
    return mask.fillna(False).astype(bool)


def gate_indicator_series(spot: pd.Series, gate_key: Optional[str]
                            ) -> Optional[pd.Series]:
    """Legacy single-line accessor. Returns the primary reference series
    that, when compared to spot, reproduces the gate (only meaningful
    for 'spot vs MA' gates). Returns None for other gates.

    Kept for backward compat. New chart code should use
    `gate_chart_layers` for richer plotting.
    """
    if gate_key == "spot_above_50dma":
        return spot.rolling(window=50, min_periods=50).mean()
    if gate_key == "spot_above_200dma":
        return spot.rolling(window=200, min_periods=200).mean()
    return None


def gate_chart_layers(spot: pd.Series, gate_key: Optional[str]
                        ) -> dict:
    """Per-gate chart configuration for the drilldown view.

    Returns a dict with:
      - 'panel': 'price' (overlay on spot chart only),
                  'subplot' (separate indicator panel only),
                  or 'both' (combos — spot overlay + indicator subplot)
      - 'price_lines': list of {name, series, color, dash} for the spot panel
      - 'subplot_lines': list of {name, series, color, dash} for the indicator panel
      - 'mask': bool Series — gate-active days (for green shading)
      - 'subplot_title': title for the indicator subplot (if any)

    All series are aligned to spot.index. NaN values are left in place
    so plotly draws gaps before the rolling window fills.

    Backward-compat note: the legacy keys 'lines' (alias of either
    price_lines or subplot_lines) is also populated for callers that
    haven't been updated yet.
    """
    if not gate_key:
        return {"panel": "price", "price_lines": [], "subplot_lines": [],
                  "lines": [], "mask": None, "subplot_title": ""}

    mask = compute_gate_mask(spot, gate_key)

    # Helper builders for reusable layer types
    def _ma_lines_50() -> list[dict]:
        dma50 = spot.rolling(50, min_periods=50).mean()
        return [{"name": "50DMA", "series": dma50,
                   "color": "#f97316", "dash": "dash"}]

    def _ma_lines_200() -> list[dict]:
        dma200 = spot.rolling(200, min_periods=200).mean()
        return [{"name": "200DMA", "series": dma200,
                   "color": "#f97316", "dash": "dash"}]

    def _ma_lines_20_50() -> list[dict]:
        dma20 = spot.rolling(20, min_periods=20).mean()
        dma50 = spot.rolling(50, min_periods=50).mean()
        return [{"name": "20DMA", "series": dma20,
                   "color": "#22d3ee", "dash": "solid"},
                  {"name": "50DMA", "series": dma50,
                   "color": "#f97316", "dash": "dash"}]

    def _ma_lines_50_200() -> list[dict]:
        dma50 = spot.rolling(50, min_periods=50).mean()
        dma200 = spot.rolling(200, min_periods=200).mean()
        return [{"name": "50DMA", "series": dma50,
                   "color": "#22d3ee", "dash": "solid"},
                  {"name": "200DMA", "series": dma200,
                   "color": "#f97316", "dash": "dash"}]

    def _rv_lines(quantile: float, label_q: str) -> list[dict]:
        rv = _realized_vol_20d(spot) * 100  # to percentage points
        thr = rv.rolling(252, min_periods=252).quantile(quantile)
        return [{"name": "20d realized vol (%)", "series": rv,
                   "color": "#a78bfa", "dash": "solid"},
                  {"name": f"{label_q} (252d)", "series": thr,
                   "color": "#f97316", "dash": "dash"}]

    # Pure trend
    trend_map = {
        "spot_above_50dma":   _ma_lines_50,
        "spot_above_200dma":  _ma_lines_200,
        "dma_20_above_50":    _ma_lines_20_50,
        "dma_50_above_200":   _ma_lines_50_200,
    }
    if gate_key in trend_map:
        price_lines = trend_map[gate_key]()
        return {"panel": "price", "price_lines": price_lines,
                  "subplot_lines": [], "lines": price_lines,
                  "mask": mask, "subplot_title": ""}

    # Pure vol-regime
    if gate_key == "realized_vol_below_p75":
        subplot_lines = _rv_lines(0.75, "75th pct")
        return {"panel": "subplot", "price_lines": [],
                  "subplot_lines": subplot_lines, "lines": subplot_lines,
                  "mask": mask, "subplot_title": "Realized vol regime"}
    if gate_key == "realized_vol_below_p50":
        subplot_lines = _rv_lines(0.50, "Median")
        return {"panel": "subplot", "price_lines": [],
                  "subplot_lines": subplot_lines, "lines": subplot_lines,
                  "mask": mask, "subplot_title": "Realized vol regime"}

    # Combos: trend overlay + vol subplot
    combo_map = {
        "spot_above_50dma_and_rv_below_p75": (_ma_lines_50, 0.75, "75th pct"),
        "golden_cross_and_rv_below_p75":     (_ma_lines_50_200, 0.75, "75th pct"),
        "golden_cross_and_rv_below_p50":     (_ma_lines_50_200, 0.50, "Median"),
    }
    if gate_key in combo_map:
        ma_fn, q, q_label = combo_map[gate_key]
        price_lines = ma_fn()
        subplot_lines = _rv_lines(q, q_label)
        return {"panel": "both", "price_lines": price_lines,
                  "subplot_lines": subplot_lines,
                  "lines": price_lines + subplot_lines,
                  "mask": mask, "subplot_title": "Realized vol regime"}

    return {"panel": "price", "price_lines": [], "subplot_lines": [],
              "lines": [], "mask": mask, "subplot_title": ""}

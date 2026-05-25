"""Worst-of two-pair KO option backtester.

# Structure

Two independent KO option legs (different pairs, possibly different
directions) opened simultaneously. Each leg is priced as a vanilla-Δ
strike with a vanilla-Δ wing barrier (delta-mode KO, same convention
as single-leg backtest in delta mode).

Pricing convention (smile-aware):

    P_A_mid = ko_price at σ_smile(K_A) on pair A
    P_B_mid = ko_price at σ_smile(K_B) on pair B
    structure_premium_mid = spec.multiplier × min(P_A_mid, P_B_mid)
    tx_cost              = bps × notional / 10_000        (USD)
    structure_premium    = structure_premium_mid + tx_cost

`spec.multiplier` defaults to 0.33 (≈ old hardcoded /3) but is set by
the calling app via the sidebar — App 9 defaults to 0.50 (European
barriers; structure prices closer to the cheaper leg) and App 12 to
0.40 (American barriers; cheaper structure relative to single legs).

Tx cost is applied at the structure level only — individual leg
premiums (P_A, P_B) are pure mid premiums at each leg's σ_smile.

# Payoff at expiry

For each leg independently:
- if S_expiry breaches the barrier → leg KOs (payoff = 0)
- otherwise → leg pays vanilla intrinsic, capped at |H − K|

Worst-of payoff = min(payoff_A, payoff_B). When EITHER leg KOs the
worst-of is 0; the structure pays only when BOTH legs survive AND both
finish ITM.

# Entry gate

Each leg can carry its OWN gate (or none). A trade enters only when
both per-leg gates are satisfied on the same date — i.e. gate_A passes
on pair A's spot AND gate_B passes on pair B's spot. A leg with
`entry_gate = None` contributes a trivially-true mask on that side, so
the AND reduces to "the other leg's gate". Existing positions are not
unwound when the gate later flips off.

This per-leg structure lets users mix gates (e.g. a 50DMA-trend gate
on the carry leg and a vol-floor gate on the funding leg) or treat
one leg as ungated while gating the other.

# Notional convention

Each leg has the same USD notional (= structure notional). For
USD-foreign pairs (USDxxx) this works directly with the shared engine
helpers. The summary reports everything in USD.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from typing import Optional, Callable

import numpy as np
import pandas as pd

from core.calendar import compute_option_dates
from core.ko import ko_price
from core.ko_solvers import solve_strike_from_delta, solve_barrier_from_delta
from core.smile import smile_vol_at_strike
from core.rates import get_rate_at
from core.gates import compute_gate_mask
from core.backtest import (
    preload_pair_panels,
    _spot_at_or_before,
    _interp_panels_at_T,
)


# -----------------------------------------------------------------------------
# Spec + Trade
# -----------------------------------------------------------------------------
@dataclass
class WorstOfSpec:
    # Leg A
    leg_a_pair: str
    leg_a_direction: str               # 'call' | 'put'
    leg_a_barrier_type: str            # 'up_and_out' | 'down_and_out'
    leg_a_strike_delta_value: float    # 0.35 for 35Δ
    leg_a_strike_delta_label: str      # '35Δ'
    leg_a_ko_delta_value: float        # 0.10 for 10Δ
    leg_a_ko_delta_label: str          # '10Δ'

    # Leg B
    leg_b_pair: str
    leg_b_direction: str
    leg_b_barrier_type: str
    leg_b_strike_delta_value: float
    leg_b_strike_delta_label: str
    leg_b_ko_delta_value: float
    leg_b_ko_delta_label: str

    # Common
    tenor_label: str
    tx_cost_bps: float = 4.0
    entry_gate_a: Optional[str] = None   # gate applied to leg A's pair
    entry_gate_b: Optional[str] = None   # gate applied to leg B's pair
    prefer: str = "offshore"

    # Phase WF-C: monthly walk-forward schedule of (K, H) LEVELS for
    # this trade's pair pair. When present, the engine looks up the
    # entry valid on each trade date and uses those levels directly
    # instead of solving (K, H) from the static deltas above. The
    # static `*_strike_delta_value` / `*_ko_delta_value` fields are
    # ignored on dates where the schedule has an entry.
    #
    # Schedule is a list of dicts as produced by
    # `core.wf_schedule.build_monthly_schedule`. Each entry has keys:
    #   valid_from, valid_to (ISO date strings)
    #   ellipse_K_a, ellipse_K_b               — strike spot levels
    #   ellipse_H_a_up, ellipse_H_b_up         — barriers for up_and_out
    #   ellipse_H_a_dn, ellipse_H_b_dn         — barriers for down_and_out
    # Trades on dates with no schedule entry are SKIPPED (return None
    # from the pricer, no trade recorded).
    dynamic_schedule: Optional[list[dict]] = None

    # Phase WF-C adaptive: which strike-selection strategy to use when
    # the spec is in adaptive mode. Choices: "cheapest" (lowest strike
    # Δ, most OTM — buyer-friendly), "max_payoff" (highest strike Δ,
    # most ATM — biggest payoff window), "balanced" (middle). The
    # underscore prefix marks it as "engine-internal" — not part of
    # the public delta/grid sweep.
    _adaptive_strike_strategy: str = "cheapest"

    # Trade mode — same semantics as core/backtest.StrategySpec.trade_mode:
    #   'stack'  (default) — open a new trade on every eligible date.
    #   'single'           — at most one open trade at a time; next
    #                        entry only on/after prior trade's expiry.
    # In worst-of, "expiry" refers to the structure's expiry (both legs
    # have the same option_expiry by construction since they share
    # spec.tenor_label or per_trade_tenor_label in adaptive mode).
    trade_mode: str = "stack"

    # Worst-of premium approximation:
    #   structure_premium_mid = multiplier × min(leg_a_premium, leg_b_premium)
    # 0.33 (≈ old hardcoded /3) is reasonable when legs are loosely
    # correlated; 0.40 is the App 12 default (American barriers), 0.50
    # the App 9 default (European barriers). Higher values suit highly
    # correlated legs that tend to KO together; lower for nearly
    # independent legs. The choice is the user's risk view, not a
    # well-defined market quantity.
    multiplier: float = 0.33

    # KO monitoring style. 'european_at_expiry' (default) checks each
    # leg's spot at the option_expiry only — appropriate for App 9.
    # 'american_ohlc' scans daily [Low, High] across the trade window
    # and marks the first day either leg's barrier sits in-range —
    # appropriate for App 12. Engine dispatches via this flag; spec
    # is the source of truth so a saved/loaded preset stays consistent.
    ko_check_mode: str = "european_at_expiry"

    # Leg-level pricing function. 'european' (default) prices each leg
    # with `ko_price` at σ_smile(K) — the historical behavior, matches
    # App 9's single-leg European-barrier pricer. 'vanna_volga_american'
    # prices each leg with `vv_price_ko` on top of the American closed
    # form, matching App 12's single-leg Pricer tab. The latter is
    # smile-adjusted (Bloomberg OVML equivalent) and bumps leg premiums
    # by typically 10–40% on right-skewed pairs like USDJPY — flowing
    # through to a correspondingly higher structure premium even at
    # the same multiplier.
    leg_pricing_mode: str = "european"

    # =========================================================================
    # STEP 2C: Structure-level pricing engine
    # =========================================================================
    # How to price the WORST-OF structure premium (not the per-leg mids,
    # which are still computed via leg_pricing_mode above).
    #
    #   'legacy_multiplier' (default) — historical behavior:
    #         structure_mid = spec.multiplier × min(leg_a_mid, leg_b_mid).
    #       The multiplier is a per-app convention (0.33/0.40/0.50) that
    #       crudely approximates the correlation effect. Kept as default
    #       for backwards compatibility with existing presets.
    #
    #   --- European-barrier engines (require ko_check_mode='european_at_expiry') ---
    #
    #   'closed_form' — correlation-aware semi-CF pricer
    #         (core.worstof_pricer.worstof_eko_price_cf). 1D Gauss-
    #         Legendre quadrature over leg-2's terminal log-spot,
    #         conditional capped-EKO payoff in closed form on leg-1.
    #         ~1–3 ms per trade.
    #
    #   'monte_carlo' — terminal-only MC
    #         (core.worstof_pricer.worstof_eko_price_mc). Stochastic but
    #         independent validator; useful for sanity checks. ~5–15 ms
    #         per trade at default mc_n_paths.
    #
    #   --- American-barrier engines (Step R3 — require ko_check_mode='american_ohlc') ---
    #
    #   'cf_approx_american' — fast CF approximation for worst-of RKOs
    #         (core.worstof_pricer_american.worstof_rko_price_cf_approx).
    #         Uses European worst-of CF scaled by the product of single-
    #         leg American/European KO ratios. ~2 ms per trade. Biased
    #         low ~10-50% on tight barriers; tracks MC to within 5% on
    #         wider barriers. See module docstring for the bias bound.
    #
    #   'monte_carlo_american' — exact MC for worst-of RKOs
    #         (core.worstof_pricer_american.worstof_rko_price_mc).
    #         Daily-step correlated GBM with Brownian-bridge sub-step
    #         touch correction. ~400 ms per trade at 100k paths.
    #         Canonical pricer; use for production backtests where
    #         accuracy matters more than throughput.
    #
    # IMPORTANT: each engine REQUIRES a specific ko_check_mode:
    #   - legacy_multiplier  : either mode allowed
    #   - closed_form / monte_carlo            : 'european_at_expiry' ONLY
    #   - cf_approx_american / monte_carlo_american : 'american_ohlc' ONLY
    # The engine raises ValueError on mismatch rather than producing
    # misleading numbers.
    pricing_engine: str = "legacy_multiplier"

    # Correlation handling — only meaningful when pricing_engine !=
    # 'legacy_multiplier'.
    #
    #   'manual'      : use spec.correlation_value for every trade date.
    #   'rolling_60d' : compute the rolling 60-business-day log-return
    #                   correlation of (spot_a, spot_b) and look up the
    #                   value AT EACH trade date. Trade dates with
    #                   insufficient lookback (first ~60 days) fall back
    #                   to spec.correlation_value.
    #
    # FX-triangulation-implied correlation (from cross vols) lands in
    # Step 2d as 'triangulation'.
    correlation_source: str = "manual"
    correlation_value: float = 0.30
    # MC paths PER trade. Default is lower than the live-pricer UI's
    # 200k since a 500-trade backtest at 200k × ~15 ms/trade = 7.5 s
    # is already snappy and the per-trade std error of ~2e-5 (= ~0.2 bp)
    # is far below realistic pricing uncertainty.
    mc_n_paths: int = 100_000

    @property
    def name(self) -> str:
        """A unique, human-readable strategy identifier.

        Both per-leg gates are inlined next to each leg, so two specs that
        differ only in which leg carries which gate produce distinct names.
        """
        from core.gates import gate_label
        a_dir = self.leg_a_barrier_type.replace("_and_out", "")
        b_dir = self.leg_b_barrier_type.replace("_and_out", "")
        # Per-leg gate suffix; '(none)' is rendered as a blank tag to keep
        # the all-gates-off case from cluttering the name.
        ga = f" [{gate_label(self.entry_gate_a)}]" if self.entry_gate_a else ""
        gb = f" [{gate_label(self.entry_gate_b)}]" if self.entry_gate_b else ""
        # WF-C dynamic mode: replace fixed delta labels with "DYN" so the
        # name reflects that K and H come from the walk-forward schedule
        # rather than a fixed delta. Sweeps over different schedules end
        # up with identical names — append a schedule identifier in that
        # case (or rely on the gate suffix to differentiate).
        # Adaptive mode (each schedule entry contains a "clusters" list)
        # is further distinguished by an "ADPT" suffix and a "DYN-Tenor"
        # tenor label, since the engine picks tenor per trade date.
        is_adaptive = False
        if self.dynamic_schedule and len(self.dynamic_schedule) > 0:
            is_adaptive = "clusters" in self.dynamic_schedule[0]
        if self.dynamic_schedule is not None:
            sd_a, kd_a = "DYN-K", "DYN-H"
            sd_b, kd_b = "DYN-K", "DYN-H"
        else:
            sd_a, kd_a = self.leg_a_strike_delta_label, self.leg_a_ko_delta_label
            sd_b, kd_b = self.leg_b_strike_delta_label, self.leg_b_ko_delta_label
        leg_a_str = (f"{self.leg_a_pair} {self.leg_a_direction[0].upper()}"
                       f"{a_dir[0]}o {sd_a}/H@{kd_a}{ga}")
        leg_b_str = (f"{self.leg_b_pair} {self.leg_b_direction[0].upper()}"
                       f"{b_dir[0]}o {sd_b}/H@{kd_b}{gb}")
        tenor_str = "DYN-Tenor [ADPT]" if is_adaptive else self.tenor_label
        return f"WO[{leg_a_str}  ∧  {leg_b_str}]  {tenor_str}"


@dataclass
class WorstOfTrade:
    # Context + dates
    strategy_name: str
    trade_date: date
    spot_settlement: date
    option_settlement: date
    expiry_date: date
    T_years: float

    # --- Leg A ---
    leg_a_pair: str
    leg_a_direction: str
    leg_a_barrier_type: str
    leg_a_strike_delta_label: str
    leg_a_ko_delta_label: str
    leg_a_spot: float
    leg_a_strike: float
    leg_a_barrier: float
    leg_a_sigma_atm: float
    leg_a_rr_25: float
    leg_a_bf_25: float
    leg_a_sigma_smile: float
    leg_a_r_d: float
    leg_a_r_f: float
    leg_a_premium_mid_usd: float    # at σ_smile, no tx
    leg_a_max_payoff_usd: float
    leg_a_spot_at_expiry: float
    leg_a_knocked_out: bool
    leg_a_payoff_usd: float

    # --- Leg B ---
    leg_b_pair: str
    leg_b_direction: str
    leg_b_barrier_type: str
    leg_b_strike_delta_label: str
    leg_b_ko_delta_label: str
    leg_b_spot: float
    leg_b_strike: float
    leg_b_barrier: float
    leg_b_sigma_atm: float
    leg_b_rr_25: float
    leg_b_bf_25: float
    leg_b_sigma_smile: float
    leg_b_r_d: float
    leg_b_r_f: float
    leg_b_premium_mid_usd: float
    leg_b_max_payoff_usd: float
    leg_b_spot_at_expiry: float
    leg_b_knocked_out: bool
    leg_b_payoff_usd: float

    # Structure-level
    notional_usd: float
    structure_premium_mid_usd: float    # = spec.multiplier × min(P_A_mid, P_B_mid)
    tx_cost_bps: float
    tx_cost_usd: float
    structure_premium_paid_usd: float   # = mid + tx
    worst_of_payoff_usd: float          # = min(payoff_A, payoff_B)
    pnl_usd: float                      # = worst_of − paid
    # Approximation multiplier used at trade time (0.33/0.40/0.50). Stored
    # on each trade so the ledger and CSV exports are self-describing —
    # users running multi-multiplier comparisons can tell which row used
    # which multiplier.
    multiplier: float

    entry_gate_a: Optional[str]    # gate applied to leg A's pair (or None)
    entry_gate_b: Optional[str]    # gate applied to leg B's pair (or None)

    # Leg-level pricing model used at trade time. 'european' = ko_price
    # at σ_smile(K); 'vanna_volga' = vv_price_ko on ako_closed_form.
    # Mirrors the per-trade `pricing_model` field on single-leg Trades
    # so multi-engine ledgers are self-describing.
    leg_pricing_mode: str = "european"

    # ---------- Step 2c: structure-level pricing engine context ----------
    # The engine used for THIS trade's structure premium. Persisted so
    # multi-engine ledgers / CSV exports / drilldown views can show
    # which engine produced which row. 'legacy_multiplier' means the
    # historical `multiplier × min(P_A, P_B)` formula was used (above
    # in this dataclass: see field `multiplier`). 'closed_form' or
    # 'monte_carlo' means the new core.worstof_pricer was used.
    pricing_engine: str = "legacy_multiplier"

    # The correlation actually used for this trade. None when engine is
    # 'legacy_multiplier' (multiplier formula doesn't take a rho).
    correlation_used: Optional[float] = None
    # The source the rho came from for this trade. 'manual' / 'rolling_60d'
    # / 'rolling_60d_fallback_to_manual' (when the lookback was too short).
    # 'legacy' when pricing_engine='legacy_multiplier'.
    correlation_source_used: str = "legacy"

    # Always-recorded legacy formula value for direct A/B comparison
    # vs the new engines, even when the new engine drove the trade
    # economics. Equals (multiplier × min(P_A_mid, P_B_mid)).
    structure_premium_legacy_usd: float = 0.0

    # Phase WF-C adaptive: per-trade context when the spec ran in
    # adaptive mode (cluster + tenor picked per trade date by
    # `select_cluster_and_tenor`). All None for non-adaptive trades.
    adaptive_cluster_index: Optional[int] = None
    adaptive_cluster_mu_a: Optional[float] = None
    adaptive_cluster_mu_b: Optional[float] = None
    adaptive_cluster_sigma_a: Optional[float] = None
    adaptive_cluster_sigma_b: Optional[float] = None
    adaptive_cluster_sojourn_days: Optional[float] = None
    adaptive_cluster_distance_from_spot: Optional[float] = None
    adaptive_chosen_tenor: Optional[str] = None
    adaptive_green_tenors: Optional[str] = None  # csv string for portability
    adaptive_decision_log: Optional[str] = None
    # Strike/KO grid context (set by the grid-aware selector). Show
    # which delta the engine snapped to versus what the cluster
    # geometry suggested raw.
    adaptive_strike_strategy: Optional[str] = None
    adaptive_strike_delta_a: Optional[float] = None
    adaptive_strike_label_a: Optional[str] = None
    adaptive_ko_delta_a: Optional[float] = None
    adaptive_ko_label_a: Optional[str] = None
    adaptive_strike_delta_b: Optional[float] = None
    adaptive_strike_label_b: Optional[str] = None
    adaptive_ko_delta_b: Optional[float] = None
    adaptive_ko_label_b: Optional[str] = None
    adaptive_cluster_upper_edge_delta_a: Optional[float] = None
    adaptive_cluster_upper_edge_delta_b: Optional[float] = None


# -----------------------------------------------------------------------------
# STEP 2C/2D HELPERS — correlation series + structure-premium dispatcher
# -----------------------------------------------------------------------------
# The rolling and triangulation correlation estimators live in
# core.correlation; this module just glues them into the per-trade
# backtest loop via a single "pre-computed correlation series" pattern
# (one Series object per (pair_a, pair_b, [tenor]) combination).

def _resolve_trade_correlation(
        spec: "WorstOfSpec", td_ts: pd.Timestamp,
        precomputed_corr_series: Optional[pd.Series],
) -> "tuple[float, str]":
    """Pick the correlation to use for this trade date.

    Returns (rho, source_label). Falls back to spec.correlation_value
    when the requested source is unavailable for this date.

    `precomputed_corr_series` is whichever Series was pre-built upstream
    based on spec.correlation_source — either the rolling-60d realized
    series (for 'rolling_60d') or the triangulation-implied series (for
    'triangulation'). For 'manual' the precomputed series is None and
    spec.correlation_value is used directly.
    """
    src = spec.correlation_source
    if src == "manual":
        return float(spec.correlation_value), "manual"

    if src in ("rolling_60d", "triangulation"):
        if (precomputed_corr_series is not None
                and td_ts in precomputed_corr_series.index):
            r = precomputed_corr_series.loc[td_ts]
            if pd.notna(r):
                return float(r), src
        return float(spec.correlation_value), f"{src}_fallback_to_manual"

    # Unknown source — be defensive and fall back to manual.
    return float(spec.correlation_value), f"unknown_source_{src}_fallback"


def _compute_structure_premium_via_engine(
        spec: "WorstOfSpec",
        leg_a: dict, leg_b: dict,
        T: float, rho: float, notional_usd: float,
) -> "tuple[float, dict]":
    """Dispatch to the correlation-aware pricer.

    `leg_a` and `leg_b` are the dicts returned by `_price_one_leg` —
    they contain the raw spot/strike/barrier/vol/rates for each leg.

    Each leg's spot is normalized to 1.0 (with K and H rescaled
    accordingly) before being handed to the pricer. This is the same
    convention as the historical multiplier formula — premiums are
    expressed as % of leg notional — and is what makes min() across
    two legs in different currencies meaningful.

    The pricer's output (% of notional) is multiplied by `notional_usd`
    to give a USD premium.

    Returns
    -------
    (structure_premium_mid_usd, pricer_meta)
        pricer_meta is the raw dict from `worstof_eko_price_*` for any
        downstream diagnostics (p_alive_*, std_err, etc.).
    """
    # Late import to avoid module-load cost when only the legacy engine
    # is in use.
    from core.worstof_pricer import (
        WorstOfLeg, worstof_eko_price_cf, worstof_eko_price_mc,
    )

    S_a, S_b = leg_a["S"], leg_b["S"]
    wol_a = WorstOfLeg(
        S=1.0,
        K=leg_a["K"] / S_a,
        H=leg_a["H"] / S_a,
        sigma=leg_a["sigma_smile"],
        r_d=leg_a["r_d"], r_f=leg_a["r_f"],
        opt=spec.leg_a_direction,
        bar_dir=spec.leg_a_barrier_type,
    )
    wol_b = WorstOfLeg(
        S=1.0,
        K=leg_b["K"] / S_b,
        H=leg_b["H"] / S_b,
        sigma=leg_b["sigma_smile"],
        r_d=leg_b["r_d"], r_f=leg_b["r_f"],
        opt=spec.leg_b_direction,
        bar_dir=spec.leg_b_barrier_type,
    )
    # Discount at leg A's DOM rate (numeraire = leg A's DOM). For
    # same-DOM legs this is the unique correct choice; for cross-DOM
    # legs this is the quanto-ignored mixed-measure convention — see
    # `core.worstof_pricer.WorstOfLeg` docstring.
    r_d_discount = leg_a["r_d"]

    if spec.pricing_engine == "closed_form":
        out = worstof_eko_price_cf(wol_a, wol_b, T, rho, r_d_discount,
                                     n_quad=60)
    elif spec.pricing_engine == "monte_carlo":
        # Deterministic seed per trade so the same backtest is
        # reproducible across runs.
        out = worstof_eko_price_mc(wol_a, wol_b, T, rho, r_d_discount,
                                     n_paths=spec.mc_n_paths, seed=42)
    elif spec.pricing_engine == "cf_approx_american":
        # Step R3 — fast CF-approximation for American-barrier worst-of.
        # Imports here (not at module load) since this is a relatively
        # new path and we don't want to pay the import cost for legacy
        # users.
        from core.worstof_pricer_american import worstof_rko_price_cf_approx
        out = worstof_rko_price_cf_approx(wol_a, wol_b, T, rho,
                                            r_d_discount, n_quad=60)
    elif spec.pricing_engine == "monte_carlo_american":
        # Step R3 — exact MC for American-barrier worst-of. Uses
        # Brownian-bridge sub-step touch correction (the right choice
        # for FORWARD pricing at trade-entry, since the future OHLC
        # isn't known yet — the realized-OHLC barrier check for in-
        # life KO is done independently by the engine's
        # `ko_check_mode='american_ohlc'` scan).
        from core.worstof_pricer_american import worstof_rko_price_mc
        out = worstof_rko_price_mc(
            wol_a, wol_b, T, rho, r_d_discount,
            n_paths=spec.mc_n_paths, seed=42,
            monitoring="brownian_bridge",
        )
    else:
        raise ValueError(
            f"_compute_structure_premium_via_engine called with engine="
            f"{spec.pricing_engine!r}; expected 'closed_form', "
            f"'monte_carlo', 'cf_approx_american', or "
            f"'monte_carlo_american' "
            f"(legacy_multiplier handled before dispatch)."
        )

    structure_mid_usd = out["price"] * notional_usd
    return float(structure_mid_usd), out


# -----------------------------------------------------------------------------
# Per-leg pricing helper
# -----------------------------------------------------------------------------
def _price_one_leg(
        panels: dict, td_ts: pd.Timestamp, td: date, T: float,
        direction: str, barrier_type: str,
        strike_delta_value: float, ko_delta_value: float,
        notional_usd: float,
        fixed_K: Optional[float] = None,
        fixed_H: Optional[float] = None,
        leg_pricing_mode: str = "european",
) -> Optional[dict]:
    """Price one leg of the worst-of structure on a given trade date.

    Returns a dict of leg fields, or None if data is missing or the
    structure is degenerate (KO Δ ≥ strike Δ ⇒ H on wrong side of K).

    If `fixed_K` and `fixed_H` are both provided (Phase WF-C dynamic
    schedule), they are used as the strike and barrier spot levels
    directly — the delta-based solver is skipped. The `strike_delta_value`
    and `ko_delta_value` are ignored in this case (kept in the signature
    for backward compat — callers that don't use dynamic levels pass
    them as before).
    """
    spot = panels["spot"]
    vol_panels = panels["vol_panels"]
    fwd_panels = panels["fwd_panels"]
    rr_panels = panels.get("rr_panels", {})
    bf_panels = panels.get("bf_panels", {})
    f_panel = panels["f_panel"]
    d_panel = panels["d_panel"]
    pip = panels["pip_scale"]

    S = _spot_at_or_before(spot, td_ts)
    if S is None:
        return None

    sigma_atm_pct = _interp_panels_at_T(vol_panels, T, td_ts)
    if sigma_atm_pct is None:
        return None
    sigma_atm = sigma_atm_pct / 100.0

    rr_pct = _interp_panels_at_T(rr_panels, T, td_ts) if rr_panels else None
    bf_pct = _interp_panels_at_T(bf_panels, T, td_ts) if bf_panels else None
    rr_25 = (rr_pct / 100.0) if rr_pct is not None else 0.0
    bf_25 = (bf_pct / 100.0) if bf_pct is not None else 0.0

    fwd_pts = _interp_panels_at_T(fwd_panels, T, td_ts)
    F_market = S + fwd_pts * pip if fwd_pts is not None else S

    r_f = get_rate_at(f_panel, T, td)
    r_d_rate = get_rate_at(d_panel, T, td)
    if r_f is None and r_d_rate is None:
        return None
    if r_f is None:
        r_f = r_d_rate - np.log(F_market / S) / T
    if r_d_rate is None:
        r_d_rate = r_f + np.log(F_market / S) / T

    # Solve K (vanilla Δ at σ_atm) and H (vanilla wing Δ at σ_atm),
    # OR use the fixed levels supplied by the caller (Phase WF-C
    # dynamic schedule path).
    if fixed_K is not None and fixed_H is not None:
        K = float(fixed_K)
        H = float(fixed_H)
    else:
        try:
            K = solve_strike_from_delta(direction, strike_delta_value,
                                           S, T, sigma_atm, r_d_rate, r_f)
            H = solve_barrier_from_delta(barrier_type, ko_delta_value,
                                             S, T, sigma_atm, r_d_rate, r_f)
        except Exception:
            return None

    # Validate barrier on the correct side
    if barrier_type == "up_and_out" and H <= K:
        return None
    if barrier_type == "down_and_out" and H >= K:
        return None

    sigma_smile = smile_vol_at_strike(S, K, T, sigma_atm, rr_25, bf_25,
                                        r_d_rate, r_f)

    # Leg pricing dispatch:
    #
    # 'european' (App 9): the historical Black-Scholes-Merton KO price
    # at the smile-adjusted vol-at-strike. Fast (closed form, no FD),
    # but assumes barrier monitored only at expiry. Smile enters
    # through σ_smile(K) — a single-vol approximation.
    #
    # 'vanna_volga_american' (App 12): full Vanna-Volga smile correction
    # on top of the American-barrier closed form. Matches Bloomberg
    # OVML's 'Vanna-Volga' model within ~0.5% on USDJPY KOs.
    # Computationally heavier (~5 ako_closed_form calls per VV call)
    # but captures the full smile premium — typically 10-40% higher
    # on right-skewed pairs.
    pricing_model = "european"
    if leg_pricing_mode == "vanna_volga_american":
        from core.vanna_volga import vv_price_ko
        from core.american_barrier import ako_closed_form
        vv_out = vv_price_ko(direction, barrier_type, S, K, H, T,
                                sigma_atm, rr_25, bf_25, r_d_rate, r_f,
                                flat_vol_pricer=ako_closed_form)
        premium_mid_per_unit = vv_out["price_vv"]
        pricing_model = "vanna_volga"
    else:
        premium_mid_per_unit = ko_price(direction, barrier_type,
                                          S, K, H, T, sigma_smile,
                                          r_d_rate, r_f)
    max_pay_per_unit = abs(H - K)

    return {
        "S": S, "K": K, "H": H,
        "sigma_atm": sigma_atm,
        "rr_25": rr_25, "bf_25": bf_25,
        "sigma_smile": sigma_smile,
        "r_d": r_d_rate, "r_f": r_f,
        "premium_mid_per_unit": premium_mid_per_unit,
        "premium_mid_usd": premium_mid_per_unit / S * notional_usd,
        "max_pay_per_unit": max_pay_per_unit,
        "max_pay_usd": max_pay_per_unit / S * notional_usd,
        "pricing_model": pricing_model,
    }


def _leg_payoff_at_expiry(
        spot: pd.Series, S_exp_ts: pd.Timestamp, leg: dict,
        direction: str, barrier_type: str, notional_usd: float,
) -> Optional[tuple[float, bool, float]]:
    """Compute realized payoff for one leg at expiry.

    Returns (payoff_usd, knocked_out, S_exp) or None if S_exp unavailable.
    """
    S_exp = _spot_at_or_before(spot, S_exp_ts)
    if S_exp is None:
        return None
    K, H = leg["K"], leg["H"]
    max_pay_per_unit = leg["max_pay_per_unit"]

    if barrier_type == "up_and_out":
        knocked_out = S_exp >= H
    else:
        knocked_out = S_exp <= H

    if knocked_out:
        payoff_per_unit = 0.0
    elif direction == "call":
        payoff_per_unit = max(S_exp - K, 0.0)
    else:
        payoff_per_unit = max(K - S_exp, 0.0)
    payoff_per_unit = min(payoff_per_unit, max_pay_per_unit)

    payoff_usd = payoff_per_unit / leg["S"] * notional_usd
    return payoff_usd, bool(knocked_out), float(S_exp)


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------
def run_worstof_strategy(
        folder: str, spec: WorstOfSpec,
        start_date: date, end_date: date,
        notional_usd: float = 10_000_000.0,
        progress_cb: Optional[Callable[[float, str], None]] = None,
) -> list[WorstOfTrade]:
    """Run the worst-of backtest for a single spec across [start, end].

    Each business day where BOTH pairs have spot data (and per-leg gates,
    if set, are satisfied for each leg's own pair) becomes a candidate
    trade date. The two legs are priced independently at their own
    σ_smile, the structure premium is `spec.multiplier × min(P_A, P_B) + tx_cost`, and
    at expiry the payoff is the minimum of the two legs' realized
    payoffs.

    This is a thin wrapper around `_run_worstof_with_panels` that loads
    the per-pair panels first. For bulk runs that share pairs, use
    `run_worstof_grid` instead — it preloads each unique pair once.
    """
    # Use the OHLC-aware loader when the spec asks for American
    # monitoring; otherwise the European-style loader (no OHLC) is fine.
    if spec.ko_check_mode == "american_ohlc":
        from core.backtest_american import preload_pair_panels_american
        panels_a = preload_pair_panels_american(folder, spec.leg_a_pair,
                                                       prefer=spec.prefer)
        panels_b = preload_pair_panels_american(folder, spec.leg_b_pair,
                                                       prefer=spec.prefer)
    else:
        panels_a = preload_pair_panels(folder, spec.leg_a_pair, prefer=spec.prefer)
        panels_b = preload_pair_panels(folder, spec.leg_b_pair, prefer=spec.prefer)
    if not panels_a or not panels_b:
        return []
    return _run_worstof_with_panels(
        spec, panels_a, panels_b, start_date, end_date,
        notional_usd=notional_usd, progress_cb=progress_cb,
        folder=folder,
    )


def _run_worstof_with_panels(
        spec: WorstOfSpec,
        panels_a: dict, panels_b: dict,
        start_date: date, end_date: date,
        notional_usd: float = 10_000_000.0,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        folder: Optional[str] = None,
) -> list[WorstOfTrade]:
    """Inner loop: runs the worst-of pricing day-by-day given preloaded
    panel dicts for each leg. Separated so a grid runner can preload
    pairs once and reuse them across many specs.

    `folder` is required ONLY when spec.correlation_source='triangulation'
    (we need to read the cross-pair vol panel). For 'manual' and
    'rolling_60d' it can be left None.
    """
    spot_a, spot_b = panels_a["spot"], panels_b["spot"]

    # =========================================================================
    # STEP 2C / R3: Validate engine + ko_check_mode compatibility
    # =========================================================================
    # Each non-legacy engine has a required ko_check_mode. Mixing is
    # disallowed because the engine would price one model and the
    # backtester would simulate a different one.
    _european_engines = ("closed_form", "monte_carlo")
    _american_engines = ("cf_approx_american", "monte_carlo_american")

    if (spec.pricing_engine in _european_engines
            and spec.ko_check_mode != "european_at_expiry"):
        raise ValueError(
            f"pricing_engine={spec.pricing_engine!r} requires "
            f"ko_check_mode='european_at_expiry', got "
            f"{spec.ko_check_mode!r}. The European closed-form / Monte-"
            f"Carlo pricers model terminal-only barrier monitoring; "
            f"combining with american_ohlc would price one model and "
            f"simulate a different one."
        )
    if (spec.pricing_engine in _american_engines
            and spec.ko_check_mode != "american_ohlc"):
        raise ValueError(
            f"pricing_engine={spec.pricing_engine!r} requires "
            f"ko_check_mode='american_ohlc', got "
            f"{spec.ko_check_mode!r}. The American CF-approx / Monte-"
            f"Carlo pricers model continuous in-life barrier monitoring; "
            f"combining with european_at_expiry would price the more "
            f"expensive structure and simulate the cheaper one."
        )
    if (spec.pricing_engine != "legacy_multiplier"
            and spec.leg_pricing_mode != "european"):
        # Per-leg σ_smile(K) is what the new engines consume. The VV-
        # American leg pricing path produces a different `premium_mid_usd`
        # but the new engines re-price legs from scratch internally
        # using the recorded sigma_smile — so this would create an
        # inconsistency. Keep things explicit rather than mixing modes.
        raise ValueError(
            f"pricing_engine={spec.pricing_engine!r} requires "
            f"leg_pricing_mode='european', got {spec.leg_pricing_mode!r}. "
            f"Structure-level VV for worst-of is a separate follow-up."
        )

    # =========================================================================
    # STEP 2D: Pre-compute the correlation series for the requested source
    # =========================================================================
    # Engine asks _resolve_trade_correlation per trade; that function
    # reads from this single Series (None for 'manual').
    precomputed_corr_series: Optional[pd.Series] = None
    if spec.pricing_engine != "legacy_multiplier":
        if spec.correlation_source == "rolling_60d":
            from core.correlation import rolling_realized_correlation
            precomputed_corr_series = rolling_realized_correlation(
                spot_a, spot_b, window=60,
            )
        elif spec.correlation_source == "triangulation":
            if folder is None:
                raise ValueError(
                    "correlation_source='triangulation' requires the "
                    "`folder` argument to be passed to "
                    "_run_worstof_with_panels (cross-pair vol comes from "
                    "the data folder)."
                )
            from core.correlation import implied_correlation_time_series
            precomputed_corr_series = implied_correlation_time_series(
                folder, spec.leg_a_pair, spec.leg_b_pair,
                tenor_label=spec.tenor_label,
                prefer_a=spec.prefer, prefer_b=spec.prefer,
                prefer_cross=spec.prefer,
            )
            if precomputed_corr_series.empty:
                # No cross-vol panel found — every trade falls back to
                # manual. Log via series being None so the resolver knows.
                precomputed_corr_series = None

    # Pre-validate the tenor label by attempting to compute dates for a
    # dummy trade date. compute_option_dates handles arbitrary NW/NM
    # labels (incl. 6W, 10W); it raises ValueError on unknown patterns.
    # T_years is recomputed per trade date below since it varies slightly
    # with weekends/holidays.
    try:
        compute_option_dates(date(2024, 1, 2), spec.tenor_label)
    except (ValueError, KeyError):
        return []

    # Per-leg gate masks. Each leg's mask is computed on its own pair's
    # spot panel. A trade requires the AND of (mask_a on date) AND
    # (mask_b on date): if either gate is None, that side contributes a
    # trivially-true mask. The intersection across calendars handles
    # holidays where one pair trades and the other doesn't.
    mask_a = compute_gate_mask(spot_a, spec.entry_gate_a)
    mask_b = compute_gate_mask(spot_b, spec.entry_gate_b)
    if mask_a is None and mask_b is None:
        gate_open_ts = None   # no filtering
    else:
        # Promote a missing gate to all-True over the matching index
        if mask_a is None:
            mask_a = pd.Series(True, index=spot_a.index)
        if mask_b is None:
            mask_b = pd.Series(True, index=spot_b.index)
        common = mask_a.index.intersection(mask_b.index)
        joint = (mask_a.reindex(common).fillna(False) &
                  mask_b.reindex(common).fillna(False))
        gate_open_ts = set(joint[joint].index.normalize())

    # Trade dates: intersection of business days where both spots exist,
    # in the requested window. Normalize to dates.
    common_dates = pd.DatetimeIndex(spot_a.index).normalize().intersection(
        pd.DatetimeIndex(spot_b.index).normalize()
    )
    in_range = common_dates[(common_dates >= pd.Timestamp(start_date)) &
                              (common_dates <= pd.Timestamp(end_date))]
    trade_dates = sorted(set(d.date() for d in in_range))
    if not trade_dates:
        return []

    last_data_ts = min(spot_a.index.max(), spot_b.index.max())

    trades: list[WorstOfTrade] = []
    # For trade_mode='single': track the most recent trade's expiry so
    # we can skip dates while an existing structure is still open.
    # Worst-of has a single expiry shared by both legs, so we only
    # track one value. None = no open trade yet.
    last_open_expiry: Optional[date] = None

    n_total = len(trade_dates)
    for i, td in enumerate(trade_dates):
        if progress_cb is not None and (i % 50 == 0 or i == n_total - 1):
            progress_cb(i / n_total, spec.name)

        td_ts = pd.Timestamp(td)

        # Single-mode block-out — done before any gate or schedule work
        # so we don't pay the lookup cost on blocked-out dates.
        if (spec.trade_mode == "single"
                and last_open_expiry is not None
                and td < last_open_expiry):
            continue

        if gate_open_ts is not None and td_ts not in gate_open_ts:
            continue

        # Phase WF-C: if the spec carries a dynamic schedule, look up
        # the entry valid on this trade date. No entry → skip this
        # date (it's before the schedule starts, after it ends, or in a
        # gap from a failed monthly fit). Has entry → use those fixed
        # spot levels for K and H on BOTH legs.
        #
        # Two flavors of dynamic schedule:
        #   1. STATIC-CLUSTER schedule (build_monthly_schedule):
        #      single (K_a, K_b, H_a, H_b) per entry, single tenor
        #      baked into spec.tenor_label.
        #   2. ADAPTIVE schedule (build_adaptive_schedule): each
        #      entry carries ALL clusters' parameters + per-tenor
        #      sojourn ratios; the engine selects nearest cluster +
        #      shortest-green tenor per trade date.
        # We distinguish by the presence of a "clusters" key in the
        # entry.
        fixed_K_a, fixed_H_a, fixed_K_b, fixed_H_b = None, None, None, None
        # Per-trade context (set in adaptive mode; carried into Trade)
        adaptive_ctx: Optional[dict] = None
        # Tenor label used for THIS trade — may differ from spec.tenor_label
        # in adaptive mode where each date can pick its own tenor.
        per_trade_tenor_label = spec.tenor_label

        if spec.dynamic_schedule is not None:
            from core.wf_schedule import lookup_schedule_entry
            sched_entry = lookup_schedule_entry(spec.dynamic_schedule, td)
            if sched_entry is None:
                continue
            is_adaptive = "clusters" in sched_entry
            if is_adaptive:
                # Adaptive path with strike/KO grid constraints.
                # Per-trade steps:
                #   1. Pick nearest cluster + shortest-green tenor
                #   2. Look up ATM vols at that tenor for both pairs
                #   3. Run the strike/KO selector under grid constraints
                #   4. Skip if any step fails
                S_a_today = _spot_at_or_before(spot_a, td_ts)
                S_b_today = _spot_at_or_before(spot_b, td_ts)
                if S_a_today is None or S_b_today is None:
                    continue
                from core.wf_schedule import (
                    select_cluster_and_tenor, select_strikes_and_barriers,
                )
                decision = select_cluster_and_tenor(
                    sched_entry, float(S_a_today), float(S_b_today)
                )
                if decision is None:
                    continue

                # Look up ATM vol for the chosen tenor.
                # The chosen_tenor label drives T and the σ_atm lookup
                # for the strike selector. The engine's _price_one_leg
                # will later look up σ_atm independently and should
                # agree with what we use here.
                per_trade_tenor_label = decision["chosen_tenor"]
                opt_dates_tmp = compute_option_dates(td, per_trade_tenor_label)
                T_tmp = opt_dates_tmp.T_years
                sigma_a_pct = _interp_panels_at_T(
                    panels_a["vol_panels"], T_tmp, td_ts
                )
                sigma_b_pct = _interp_panels_at_T(
                    panels_b["vol_panels"], T_tmp, td_ts
                )
                if sigma_a_pct is None or sigma_b_pct is None:
                    continue
                sigma_a_atm = sigma_a_pct / 100.0
                sigma_b_atm = sigma_b_pct / 100.0

                # Find the chosen cluster's full record (selector
                # returned its index but not its full dict — pull from
                # the schedule entry).
                chosen_cluster_dict = next(
                    (c for c in sched_entry["clusters"]
                       if c["cluster_index"] == decision["cluster_index"]),
                    None,
                )
                if chosen_cluster_dict is None:
                    continue

                # Determine strike strategy from spec metadata, with
                # cheapest as buyer-friendly default.
                strike_strategy = getattr(
                    spec, "_adaptive_strike_strategy", "cheapest"
                )

                strikes = select_strikes_and_barriers(
                    chosen_cluster_dict,
                    spot_a=float(S_a_today),
                    spot_b=float(S_b_today),
                    T_years=T_tmp,
                    sigma_a=sigma_a_atm,
                    sigma_b=sigma_b_atm,
                    strike_strategy=strike_strategy,
                )
                if strikes is None:
                    # Constraints not satisfiable — skip trade date
                    continue

                fixed_K_a = strikes["K_a"]
                fixed_H_a = strikes["H_a"]
                fixed_K_b = strikes["K_b"]
                fixed_H_b = strikes["H_b"]
                # Carry rich context into the trade record
                adaptive_ctx = {**decision, **strikes,
                                  "strike_strategy": strike_strategy}
                # Augment decision log with strike info
                adaptive_ctx["decision_log"] = (
                    f"{decision['decision_log']} · "
                    f"strikes={strikes['strike_label_a']}/"
                    f"{strikes['strike_label_b']} · "
                    f"KOs={strikes['ko_label_a']}/"
                    f"{strikes['ko_label_b']} "
                    f"(strategy={strike_strategy})"
                )
            else:
                # Static-cluster schedule — original WF-C path
                fixed_K_a = sched_entry["ellipse_K_a"]
                fixed_K_b = sched_entry["ellipse_K_b"]
                if spec.leg_a_barrier_type == "up_and_out":
                    fixed_H_a = sched_entry["ellipse_H_a_up"]
                else:
                    fixed_H_a = sched_entry["ellipse_H_a_dn"]
                if spec.leg_b_barrier_type == "up_and_out":
                    fixed_H_b = sched_entry["ellipse_H_b_up"]
                else:
                    fixed_H_b = sched_entry["ellipse_H_b_dn"]

        # In adaptive mode, recompute option_dates/T with the per-trade
        # tenor instead of spec.tenor_label.
        opt_dates = compute_option_dates(td, per_trade_tenor_label)
        T = opt_dates.T_years
        if pd.Timestamp(opt_dates.option_expiry) > last_data_ts:
            continue

        leg_a = _price_one_leg(
            panels_a, td_ts, td, T,
            spec.leg_a_direction, spec.leg_a_barrier_type,
            spec.leg_a_strike_delta_value, spec.leg_a_ko_delta_value,
            notional_usd,
            fixed_K=fixed_K_a, fixed_H=fixed_H_a,
            leg_pricing_mode=spec.leg_pricing_mode,
        )
        if leg_a is None:
            continue
        leg_b = _price_one_leg(
            panels_b, td_ts, td, T,
            spec.leg_b_direction, spec.leg_b_barrier_type,
            spec.leg_b_strike_delta_value, spec.leg_b_ko_delta_value,
            notional_usd,
            fixed_K=fixed_K_b, fixed_H=fixed_H_b,
            leg_pricing_mode=spec.leg_pricing_mode,
        )
        if leg_b is None:
            continue

        # Realized: pick KO check based on spec.ko_check_mode.
        #
        # 'european_at_expiry' (default, App 9): check each leg's close at
        # the expiry date against its barrier. The historic worst-of
        # behavior.
        #
        # 'american_ohlc' (App 12): scan each business day in the trade
        # window — if either leg's daily [Low, High] contains its barrier,
        # the structure dies on that day. Falls back to a close-only daily
        # check when OHLC isn't available for that pair's panels.
        S_exp_ts = pd.Timestamp(opt_dates.option_expiry)
        if spec.ko_check_mode == "american_ohlc":
            from core.backtest_american import (
                _first_barrier_hit_date, _first_barrier_hit_close_only,
            )
            ohlc_a = panels_a.get("spot_ohlc", pd.DataFrame())
            ohlc_b = panels_b.get("spot_ohlc", pd.DataFrame())
            has_ohlc_a = ("high" in ohlc_a.columns and "low" in ohlc_a.columns) \
                          if not ohlc_a.empty else False
            has_ohlc_b = ("high" in ohlc_b.columns and "low" in ohlc_b.columns) \
                          if not ohlc_b.empty else False

            if has_ohlc_a:
                ko_ts_a = _first_barrier_hit_date(
                    ohlc_a, spec.leg_a_barrier_type, leg_a["H"],
                    td, opt_dates.option_expiry)
            else:
                ko_ts_a = _first_barrier_hit_close_only(
                    spot_a, spec.leg_a_barrier_type, leg_a["H"],
                    td, opt_dates.option_expiry)
            if has_ohlc_b:
                ko_ts_b = _first_barrier_hit_date(
                    ohlc_b, spec.leg_b_barrier_type, leg_b["H"],
                    td, opt_dates.option_expiry)
            else:
                ko_ts_b = _first_barrier_hit_close_only(
                    spot_b, spec.leg_b_barrier_type, leg_b["H"],
                    td, opt_dates.option_expiry)

            # Resolve spot at expiry for the worst-of payoff calc when both
            # legs survive.
            S_exp_a = _spot_at_or_before(spot_a, S_exp_ts)
            S_exp_b = _spot_at_or_before(spot_b, S_exp_ts)
            if S_exp_a is None or S_exp_b is None:
                continue

            leg_a_ko = ko_ts_a is not None
            leg_b_ko = ko_ts_b is not None
            if leg_a_ko:
                leg_a_payoff_usd = 0.0
            else:
                # Same intrinsic-floored logic as European, applied to
                # the survivor.
                if spec.leg_a_direction == "call":
                    pay_a = max(S_exp_a - leg_a["K"], 0.0)
                else:
                    pay_a = max(leg_a["K"] - S_exp_a, 0.0)
                pay_a = min(pay_a, leg_a["max_pay_per_unit"])
                leg_a_payoff_usd = pay_a / leg_a["S"] * notional_usd
            if leg_b_ko:
                leg_b_payoff_usd = 0.0
            else:
                if spec.leg_b_direction == "call":
                    pay_b = max(S_exp_b - leg_b["K"], 0.0)
                else:
                    pay_b = max(leg_b["K"] - S_exp_b, 0.0)
                pay_b = min(pay_b, leg_b["max_pay_per_unit"])
                leg_b_payoff_usd = pay_b / leg_b["S"] * notional_usd
        else:
            a_pay = _leg_payoff_at_expiry(spot_a, S_exp_ts, leg_a,
                                             spec.leg_a_direction,
                                             spec.leg_a_barrier_type,
                                             notional_usd)
            b_pay = _leg_payoff_at_expiry(spot_b, S_exp_ts, leg_b,
                                             spec.leg_b_direction,
                                             spec.leg_b_barrier_type,
                                             notional_usd)
            if a_pay is None or b_pay is None:
                continue
            leg_a_payoff_usd, leg_a_ko, S_exp_a = a_pay
            leg_b_payoff_usd, leg_b_ko, S_exp_b = b_pay

        # Structure pricing: tx is bps × notional, applied at the structure
        # level only (legs stay at mid).
        #
        # The legacy formula always computes (multiplier × min(P_A_mid,
        # P_B_mid)) so it can be persisted on every trade for A/B
        # comparison regardless of which engine actually drove the
        # economics this trade.
        legacy_structure_premium_mid_usd = (
            spec.multiplier
            * min(leg_a["premium_mid_usd"], leg_b["premium_mid_usd"])
        )

        if spec.pricing_engine == "legacy_multiplier":
            structure_premium_mid_usd = legacy_structure_premium_mid_usd
            rho_used: Optional[float] = None
            rho_source_used = "legacy"
        else:
            rho_used, rho_source_used = _resolve_trade_correlation(
                spec, td_ts, precomputed_corr_series,
            )
            structure_premium_mid_usd, _pricer_meta = \
                _compute_structure_premium_via_engine(
                    spec, leg_a, leg_b, T, rho_used, notional_usd,
                )

        tx_cost_usd = spec.tx_cost_bps / 10_000.0 * notional_usd
        structure_premium_paid_usd = structure_premium_mid_usd + tx_cost_usd

        worst_of_payoff_usd = min(leg_a_payoff_usd, leg_b_payoff_usd)
        pnl_usd = worst_of_payoff_usd - structure_premium_paid_usd

        trades.append(WorstOfTrade(
            strategy_name=spec.name,
            trade_date=td,
            spot_settlement=opt_dates.spot_settlement,
            option_settlement=opt_dates.option_settlement,
            expiry_date=opt_dates.option_expiry,
            T_years=T,

            leg_a_pair=spec.leg_a_pair,
            leg_a_direction=spec.leg_a_direction,
            leg_a_barrier_type=spec.leg_a_barrier_type,
            leg_a_strike_delta_label=spec.leg_a_strike_delta_label,
            leg_a_ko_delta_label=spec.leg_a_ko_delta_label,
            leg_a_spot=leg_a["S"],
            leg_a_strike=leg_a["K"],
            leg_a_barrier=leg_a["H"],
            leg_a_sigma_atm=leg_a["sigma_atm"],
            leg_a_rr_25=leg_a["rr_25"],
            leg_a_bf_25=leg_a["bf_25"],
            leg_a_sigma_smile=leg_a["sigma_smile"],
            leg_a_r_d=leg_a["r_d"],
            leg_a_r_f=leg_a["r_f"],
            leg_a_premium_mid_usd=leg_a["premium_mid_usd"],
            leg_a_max_payoff_usd=leg_a["max_pay_usd"],
            leg_a_spot_at_expiry=S_exp_a,
            leg_a_knocked_out=leg_a_ko,
            leg_a_payoff_usd=leg_a_payoff_usd,

            leg_b_pair=spec.leg_b_pair,
            leg_b_direction=spec.leg_b_direction,
            leg_b_barrier_type=spec.leg_b_barrier_type,
            leg_b_strike_delta_label=spec.leg_b_strike_delta_label,
            leg_b_ko_delta_label=spec.leg_b_ko_delta_label,
            leg_b_spot=leg_b["S"],
            leg_b_strike=leg_b["K"],
            leg_b_barrier=leg_b["H"],
            leg_b_sigma_atm=leg_b["sigma_atm"],
            leg_b_rr_25=leg_b["rr_25"],
            leg_b_bf_25=leg_b["bf_25"],
            leg_b_sigma_smile=leg_b["sigma_smile"],
            leg_b_r_d=leg_b["r_d"],
            leg_b_r_f=leg_b["r_f"],
            leg_b_premium_mid_usd=leg_b["premium_mid_usd"],
            leg_b_max_payoff_usd=leg_b["max_pay_usd"],
            leg_b_spot_at_expiry=S_exp_b,
            leg_b_knocked_out=leg_b_ko,
            leg_b_payoff_usd=leg_b_payoff_usd,

            notional_usd=notional_usd,
            structure_premium_mid_usd=structure_premium_mid_usd,
            tx_cost_bps=spec.tx_cost_bps,
            tx_cost_usd=tx_cost_usd,
            structure_premium_paid_usd=structure_premium_paid_usd,
            worst_of_payoff_usd=worst_of_payoff_usd,
            pnl_usd=pnl_usd,
            multiplier=spec.multiplier,
            entry_gate_a=spec.entry_gate_a,
            entry_gate_b=spec.entry_gate_b,
            leg_pricing_mode=spec.leg_pricing_mode,
            # Step 2c: structure-level engine context
            pricing_engine=spec.pricing_engine,
            correlation_used=rho_used,
            correlation_source_used=rho_source_used,
            structure_premium_legacy_usd=legacy_structure_premium_mid_usd,
            # Adaptive context (None for non-adaptive trades)
            adaptive_cluster_index=(adaptive_ctx["cluster_index"]
                                       if adaptive_ctx else None),
            adaptive_cluster_mu_a=(adaptive_ctx["cluster_mu_a"]
                                      if adaptive_ctx else None),
            adaptive_cluster_mu_b=(adaptive_ctx["cluster_mu_b"]
                                      if adaptive_ctx else None),
            adaptive_cluster_sigma_a=(adaptive_ctx["cluster_sigma_a"]
                                          if adaptive_ctx else None),
            adaptive_cluster_sigma_b=(adaptive_ctx["cluster_sigma_b"]
                                          if adaptive_ctx else None),
            adaptive_cluster_sojourn_days=(
                adaptive_ctx["cluster_sojourn_days"]
                if adaptive_ctx else None
            ),
            adaptive_cluster_distance_from_spot=(
                adaptive_ctx["cluster_distance_from_spot"]
                if adaptive_ctx else None
            ),
            adaptive_chosen_tenor=(adaptive_ctx["chosen_tenor"]
                                       if adaptive_ctx else None),
            adaptive_green_tenors=(
                ",".join(adaptive_ctx["green_tenors"])
                if adaptive_ctx else None
            ),
            adaptive_decision_log=(adaptive_ctx["decision_log"]
                                       if adaptive_ctx else None),
            # Strike/KO grid context (populated only when grid-aware
            # selector was used — i.e. adaptive mode)
            adaptive_strike_strategy=(adaptive_ctx.get("strike_strategy")
                                            if adaptive_ctx else None),
            adaptive_strike_delta_a=(adaptive_ctx.get("strike_delta_a")
                                          if adaptive_ctx else None),
            adaptive_strike_label_a=(adaptive_ctx.get("strike_label_a")
                                          if adaptive_ctx else None),
            adaptive_ko_delta_a=(adaptive_ctx.get("ko_delta_a")
                                      if adaptive_ctx else None),
            adaptive_ko_label_a=(adaptive_ctx.get("ko_label_a")
                                      if adaptive_ctx else None),
            adaptive_strike_delta_b=(adaptive_ctx.get("strike_delta_b")
                                          if adaptive_ctx else None),
            adaptive_strike_label_b=(adaptive_ctx.get("strike_label_b")
                                          if adaptive_ctx else None),
            adaptive_ko_delta_b=(adaptive_ctx.get("ko_delta_b")
                                      if adaptive_ctx else None),
            adaptive_ko_label_b=(adaptive_ctx.get("ko_label_b")
                                      if adaptive_ctx else None),
            adaptive_cluster_upper_edge_delta_a=(
                adaptive_ctx.get("cluster_upper_edge_delta_a")
                if adaptive_ctx else None
            ),
            adaptive_cluster_upper_edge_delta_b=(
                adaptive_ctx.get("cluster_upper_edge_delta_b")
                if adaptive_ctx else None
            ),
        ))
        # Track expiry for single-mode block-out. No-op in stack mode.
        # Both legs share opt_dates.option_expiry by construction (same
        # tenor for both legs, or per_trade_tenor_label in adaptive
        # mode — either way both legs share the structure expiry).
        last_open_expiry = opt_dates.option_expiry

    if progress_cb is not None:
        progress_cb(1.0, spec.name)
    return trades


# -----------------------------------------------------------------------------
# Grid runner (bulk multi-spec evaluation)
# -----------------------------------------------------------------------------
def build_worstof_grid(
        pair_combos: list[tuple[str, str]],
        tenors: list[str],
        leg_a_directions: list[tuple[str, str]],
        leg_b_directions: list[tuple[str, str]],
        leg_a_strike_deltas: list[tuple[str, float]],
        leg_b_strike_deltas: list[tuple[str, float]],
        leg_a_ko_deltas: list[tuple[str, float]],
        leg_b_ko_deltas: list[tuple[str, float]],
        gates_a: list[Optional[str]],
        gates_b: list[Optional[str]],
        tx_cost_bps: float,
        prefer: str = "offshore",
        trade_mode: str = "stack",
        multiplier: float = 0.33,
        ko_check_mode: str = "european_at_expiry",
        leg_pricing_mode: str = "european",
        # Step 2c: structure-level engine. Defaults preserve historical
        # behavior; new presets can opt in to 'closed_form' / 'monte_carlo'.
        pricing_engine: str = "legacy_multiplier",
        correlation_source: str = "manual",
        correlation_value: float = 0.30,
        mc_n_paths: int = 100_000,
) -> list[WorstOfSpec]:
    """Build the cross-product of worst-of specs.

    Each axis is a list of pre-resolved tuples:
      - pair_combos: explicit (pair_a, pair_b) pairings — NOT a
        cross-product of pair lists, but the discrete combos the user
        asked for (e.g. [('AUDUSD','NZDUSD'), ('EURUSD','GBPUSD')]).
      - directions: list of (direction, barrier_type) tuples.
      - strike/KO deltas: list of (label, value) tuples.
      - gates: list of gate keys, where `None` means 'no gate on this
        leg'. Include `None` in the list to make the no-gate variant
        one of the spec axes.

    Invalid combos are filtered out:
      - KO Δ ≥ strike Δ on either leg (barrier closer to spot than the
        strike → trade would KO at inception or be degenerate). ATM
        strikes (Δ=0) bypass this filter since "at-the-money strike with
        OTM-wing KO" is a legitimate structure.
    """
    specs: list[WorstOfSpec] = []
    for pair_a, pair_b in pair_combos:
        for tenor in tenors:
            for (dir_a, btype_a) in leg_a_directions:
                for (dir_b, btype_b) in leg_b_directions:
                    for (sd_a_label, sd_a_val) in leg_a_strike_deltas:
                        for (kd_a_label, kd_a_val) in leg_a_ko_deltas:
                            # Filter: KO must be further OTM than strike.
                            # Skip if KO Δ ≥ strike Δ (numerically), except
                            # for the ATM strike case where Δ=0 means
                            # K≈forward and any wing KO is valid.
                            if sd_a_val > 0 and kd_a_val >= sd_a_val:
                                continue
                            for (sd_b_label, sd_b_val) in leg_b_strike_deltas:
                                for (kd_b_label, kd_b_val) in leg_b_ko_deltas:
                                    if sd_b_val > 0 and kd_b_val >= sd_b_val:
                                        continue
                                    for ga in gates_a:
                                        for gb in gates_b:
                                            specs.append(WorstOfSpec(
                                                leg_a_pair=pair_a,
                                                leg_a_direction=dir_a,
                                                leg_a_barrier_type=btype_a,
                                                leg_a_strike_delta_value=sd_a_val,
                                                leg_a_strike_delta_label=sd_a_label,
                                                leg_a_ko_delta_value=kd_a_val,
                                                leg_a_ko_delta_label=kd_a_label,
                                                leg_b_pair=pair_b,
                                                leg_b_direction=dir_b,
                                                leg_b_barrier_type=btype_b,
                                                leg_b_strike_delta_value=sd_b_val,
                                                leg_b_strike_delta_label=sd_b_label,
                                                leg_b_ko_delta_value=kd_b_val,
                                                leg_b_ko_delta_label=kd_b_label,
                                                tenor_label=tenor,
                                                tx_cost_bps=tx_cost_bps,
                                                entry_gate_a=ga,
                                                entry_gate_b=gb,
                                                prefer=prefer,
                                                trade_mode=trade_mode,
                                                multiplier=multiplier,
                                                ko_check_mode=ko_check_mode,
                                                leg_pricing_mode=leg_pricing_mode,
                                                pricing_engine=pricing_engine,
                                                correlation_source=correlation_source,
                                                correlation_value=correlation_value,
                                                mc_n_paths=mc_n_paths,
                                            ))
    return specs


def run_worstof_grid(
        folder: str, specs: list[WorstOfSpec],
        start_date: date, end_date: date,
        notional_usd: float = 10_000_000.0,
        progress_cb: Optional[Callable[[float, str], None]] = None,
) -> dict[str, list[WorstOfTrade]]:
    """Run all worst-of specs, preloading each unique pair once.

    Pair-loading dominates runtime when many specs share pairs (e.g. a
    KO-Δ sweep on a fixed AUDUSD/NZDUSD combo loads each pair once
    instead of `n_specs` times). Specs that reference an unloadable
    pair return an empty trade list rather than aborting the grid.

    The `prefer` field on each spec determines onshore vs offshore for
    EM pairs; pairs loaded with different `prefer` keys are cached
    separately."""
    # Cache panels per (pair, prefer, mode) — different prefer values
    # may yield different panels for EM pairs, and the OHLC-aware
    # loader is needed for any spec with ko_check_mode='american_ohlc'.
    panels_cache: dict[tuple[str, str, str], dict] = {}
    pairs_needed = set()
    for spec in specs:
        mode = "american" if spec.ko_check_mode == "american_ohlc" else "european"
        pairs_needed.add((spec.leg_a_pair, spec.prefer, mode))
        pairs_needed.add((spec.leg_b_pair, spec.prefer, mode))
    for (pair, prefer, mode) in pairs_needed:
        if mode == "american":
            from core.backtest_american import preload_pair_panels_american
            panels_cache[(pair, prefer, mode)] = preload_pair_panels_american(
                folder, pair, prefer=prefer)
        else:
            panels_cache[(pair, prefer, mode)] = preload_pair_panels(
                folder, pair, prefer=prefer)

    results: dict[str, list[WorstOfTrade]] = {}
    n_specs = len(specs)
    for i, spec in enumerate(specs):
        mode = "american" if spec.ko_check_mode == "american_ohlc" else "european"
        panels_a = panels_cache.get((spec.leg_a_pair, spec.prefer, mode), {})
        panels_b = panels_cache.get((spec.leg_b_pair, spec.prefer, mode), {})
        if not panels_a or not panels_b:
            results[spec.name] = []
            continue

        # Wrap the per-spec callback so overall progress is fraction
        # across the whole grid.
        def _cb(p, name, _i=i):
            if progress_cb is not None:
                overall = (_i + p) / max(n_specs, 1)
                progress_cb(min(overall, 1.0), name)

        results[spec.name] = _run_worstof_with_panels(
            spec, panels_a, panels_b, start_date, end_date,
            notional_usd=notional_usd, progress_cb=_cb,
            folder=folder,
        )

    if progress_cb is not None:
        progress_cb(1.0, "done")
    return results


# -----------------------------------------------------------------------------
# DataFrame + summary helpers
# -----------------------------------------------------------------------------
def worstof_trades_to_df(trades: list[WorstOfTrade]) -> pd.DataFrame:
    """Convert list[WorstOfTrade] to a wide DataFrame."""
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([asdict(t) for t in trades])


def _worstof_consistency_metrics(eq: pd.DataFrame, monthly_usd: pd.Series,
                                     notional_usd: float) -> dict:
    """Cross-year consistency / tail-pain metrics for worst-of runs.

    Parallel to `core.backtest._consistency_metrics` but USD-native: the
    worst-of equity curve is denominated in USD only, so pct fields are
    derived by dividing by `notional_usd`. Returns the same field names
    as the single-leg helper so the UI renders both consistently.

    Returns zeroes when there's insufficient data (e.g. <2 months, no
    drawdown, or empty equity curve).
    """
    out = {
        "annual_sharpe_mean": 0.0,
        "annual_sharpe_min": 0.0,
        "annual_sharpe_std": 0.0,
        "annual_sharpe_cv": 0.0,
        "annual_sharpe_score": 0.0,
        "pct_positive_years": 0.0,
        "min_annual_pnl_pct": 0.0,
        "min_annual_pnl_usd": 0.0,
        "n_years": 0,
        "calmar": 0.0,
        "gain_to_pain": 0.0,
        "ulcer_index": 0.0,
    }
    if eq.empty or len(monthly_usd) == 0 or notional_usd <= 0:
        return out

    # Sharpe is unit-free, so we can compute it directly from monthly USD
    by_year_usd = monthly_usd.groupby(monthly_usd.index.year)
    annual_sharpes: list[float] = []
    annual_pnls_usd: list[float] = []
    for _yr, sub in by_year_usd:
        annual_pnls_usd.append(float(sub.sum()))
        if len(sub) > 1 and sub.std() > 0:
            annual_sharpes.append(float(sub.mean() / sub.std() * np.sqrt(12)))
    n_years = len(annual_pnls_usd)

    if n_years > 0:
        out["n_years"] = int(n_years)
        arr_usd = np.array(annual_pnls_usd)
        out["pct_positive_years"] = float((arr_usd > 0).mean() * 100)
        out["min_annual_pnl_usd"] = float(arr_usd.min())
        out["min_annual_pnl_pct"] = float(arr_usd.min() / notional_usd * 100)
    if annual_sharpes:
        a_sh = np.array(annual_sharpes)
        mean_sh = float(a_sh.mean())
        std_sh = float(a_sh.std()) if len(a_sh) > 1 else 0.0
        out["annual_sharpe_mean"] = mean_sh
        out["annual_sharpe_min"] = float(a_sh.min())
        out["annual_sharpe_std"] = std_sh
        # CV = σ / μ (signed); score = μ × (1 − CV) = μ − σ. See
        # `core.backtest._consistency_metrics` for the rationale.
        if abs(mean_sh) > 1e-9:
            out["annual_sharpe_cv"] = std_sh / mean_sh
        out["annual_sharpe_score"] = mean_sh - std_sh

    # Calmar: annualised return % / |max DD %|. Convert via notional.
    if "pnl_usd" in eq.columns and "drawdown_usd" in eq.columns:
        total_pct = float(eq["pnl_usd"].sum()) / notional_usd * 100
        days = max((eq.index.max() - eq.index.min()).days, 1)
        years = days / 365.25
        cagr_pct = total_pct / max(years, 1e-6)
        max_dd_pct_abs = abs(float(eq["drawdown_usd"].min())) / notional_usd * 100
        if max_dd_pct_abs > 1e-9:
            out["calmar"] = float(cagr_pct / max_dd_pct_abs)

    # Gain-to-pain (monthly USD — ratio is unit-free)
    pos = float(monthly_usd[monthly_usd > 0].sum())
    neg_abs = float(abs(monthly_usd[monthly_usd < 0].sum()))
    if neg_abs > 1e-9:
        out["gain_to_pain"] = pos / neg_abs
    elif pos > 0:
        out["gain_to_pain"] = float("inf")

    # Ulcer index: RMS drawdown % (convert USD DD via notional)
    if "drawdown_usd" in eq.columns:
        dd_pct = (eq["drawdown_usd"].dropna() / notional_usd * 100)
        if len(dd_pct) > 0:
            out["ulcer_index"] = float(np.sqrt((dd_pct ** 2).mean()))

    return out


def worstof_summarize(df: pd.DataFrame) -> dict:
    """Headline summary for a worst-of run."""
    if df.empty:
        return {"n_trades": 0}
    n = len(df)
    paid = df["structure_premium_paid_usd"]
    payoff = df["worst_of_payoff_usd"]
    pnl = df["pnl_usd"]

    eq = worstof_equity_curve(df)
    max_dd = float(eq["drawdown_usd"].min()) if not eq.empty else 0.0

    out = {
        "n_trades": n,
        "notional_usd": float(df["notional_usd"].iloc[0]),
        "tx_cost_bps": float(df["tx_cost_bps"].iloc[0]),
        "win_rate": float((payoff > 0).mean()),
        "leg_a_ko_rate": float(df["leg_a_knocked_out"].mean()),
        "leg_b_ko_rate": float(df["leg_b_knocked_out"].mean()),
        "any_ko_rate": float(((df["leg_a_knocked_out"]) |
                                (df["leg_b_knocked_out"])).mean()),
        "both_survive_rate": float(((~df["leg_a_knocked_out"]) &
                                      (~df["leg_b_knocked_out"])).mean()),
        "total_premium_paid_usd": float(paid.sum()),
        "total_premium_mid_usd": float(df["structure_premium_mid_usd"].sum()),
        "total_tx_cost_usd": float(df["tx_cost_usd"].sum()),
        "total_payout_usd": float(payoff.sum()),
        "total_pnl_usd": float(pnl.sum()),
        "avg_premium_paid_usd": float(paid.mean()),
        "avg_payoff_usd": float(payoff.mean()),
        "best_trade_usd": float(pnl.max()),
        "worst_trade_usd": float(pnl.min()),
        "max_drawdown_usd": max_dd,
        "premium_recovery_pct": (float(payoff.sum() / paid.sum() * 100)
                                   if paid.sum() > 0 else 0.0),
        "avg_leg_a_premium_mid_usd": float(df["leg_a_premium_mid_usd"].mean()),
        "avg_leg_b_premium_mid_usd": float(df["leg_b_premium_mid_usd"].mean()),
        "structure_vs_min_leg_pct": (
            float(df["structure_premium_mid_usd"].sum() /
                    df[["leg_a_premium_mid_usd",
                          "leg_b_premium_mid_usd"]].min(axis=1).sum() * 100)
            if df[["leg_a_premium_mid_usd",
                     "leg_b_premium_mid_usd"]].min(axis=1).sum() > 0 else 0.0
        ),
    }

    # Cross-year consistency block — same field names as the single-leg
    # engine's `summarize_strategy` so the UI renders both tables uniformly.
    notional = float(df["notional_usd"].iloc[0]) if "notional_usd" in df.columns else 0.0
    monthly_usd = worstof_monthly_pnl(df)

    # Monthly Sharpe across the whole run — same metric as the single-leg
    # engine's `sharpe_monthly`. Unit-free, so USD and pct give identical
    # ratios; we compute on USD monthly series directly.
    if len(monthly_usd) > 1 and monthly_usd.std() > 0:
        out["sharpe_monthly"] = float(
            monthly_usd.mean() / monthly_usd.std() * np.sqrt(12)
        )
    else:
        out["sharpe_monthly"] = 0.0

    out.update(_worstof_consistency_metrics(eq, monthly_usd, notional))
    return out


def worstof_equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Realized equity & drawdown indexed by expiry date."""
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["expiry_date"] = pd.to_datetime(out["expiry_date"])
    daily = out.groupby("expiry_date")["pnl_usd"].sum().sort_index()
    eq = daily.cumsum()
    peak = eq.cummax()
    dd = eq - peak

    daily_paid = out.groupby("expiry_date")["structure_premium_paid_usd"].sum().sort_index()
    daily_pay = out.groupby("expiry_date")["worst_of_payoff_usd"].sum().sort_index()
    daily_tx = out.groupby("expiry_date")["tx_cost_usd"].sum().sort_index()

    return pd.DataFrame({
        "pnl_usd": daily,
        "equity_usd": eq,
        "peak_usd": peak,
        "drawdown_usd": dd,
        "drawdown_usd_pos": -dd,
        "premium_paid_usd": daily_paid,
        "payoff_usd": daily_pay,
        "tx_cost_usd": daily_tx,
        "cum_tx_cost_usd": daily_tx.cumsum(),
    })


def worstof_monthly_pnl(df: pd.DataFrame) -> pd.Series:
    """Monthly PnL (USD) indexed by month-end date."""
    if df.empty:
        return pd.Series(dtype=float)
    out = df.copy()
    out["expiry_date"] = pd.to_datetime(out["expiry_date"])
    return out.groupby(pd.Grouper(key="expiry_date", freq="ME"))["pnl_usd"].sum()


def worstof_summarize_by_regime(
    trades_df: pd.DataFrame,
    regime_panel_a: "pd.DataFrame | None" = None,
    regime_panel_b: "pd.DataFrame | None" = None,
) -> dict:
    """Per-regime breakdowns for a worst-of strategy.

    Returns a dict with up to three DataFrames:
      - 'by_state_a' : per-leg-A-state breakdown
      - 'by_state_b' : per-leg-B-state breakdown
      - 'joint'      : per (state_a, state_b) pair breakdown

    Each row gives n_trades, share_of_trades_pct, win_rate_pct,
    structure_ko_rate_pct (fraction where the worst-of payout was 0),
    total_pnl_usd, mean_pnl_usd.

    Trades whose `trade_date` is outside either panel's date range
    are dropped from the corresponding breakdown. If a regime panel
    is None, the corresponding breakdown is an empty DataFrame.

    For worst-of analysis, the per-leg breakdowns answer different
    questions:
      - by_state_a: "did this strategy work when USDJPY was in
                     its dominant regime?"
      - by_state_b: same for USDKRW
      - joint:     "did it work when both pairs were in their
                     dominant regimes?" (combinatorial — can be
                     sparse if K_a × K_b is large)
    """
    cols = ["state", "n_trades", "share_of_trades_pct",
              "win_rate_pct", "structure_ko_rate_pct",
              "total_pnl_usd", "mean_pnl_usd"]
    out = {
        "by_state_a": pd.DataFrame(columns=cols),
        "by_state_b": pd.DataFrame(columns=cols),
        "joint": pd.DataFrame(),
    }
    if trades_df.empty or "trade_date" not in trades_df.columns:
        return out

    trades = trades_df.copy()
    trades["trade_date"] = pd.to_datetime(trades["trade_date"]).dt.normalize()

    # Determine KO indicator: structure KO = worst-of payoff == 0 OR
    # either leg knocked out. Different engines store this differently;
    # robust path: payout==0 means structure KOed.
    if "payout_usd" in trades.columns:
        trades["_structure_ko"] = (trades["payout_usd"] == 0)
    elif "leg_a_knocked_out" in trades.columns:
        trades["_structure_ko"] = (trades.get("leg_a_knocked_out", False)
                                       | trades.get("leg_b_knocked_out", False))
    else:
        trades["_structure_ko"] = False

    def _summarize_by_col(col_name: str) -> pd.DataFrame:
        if col_name not in trades.columns or trades[col_name].isna().all():
            return pd.DataFrame(columns=cols)
        sub = trades.dropna(subset=[col_name]).copy()
        if sub.empty:
            return pd.DataFrame(columns=cols)
        sub[col_name] = sub[col_name].astype(int)
        total = len(sub)
        out_rows = []
        for st, g in sub.groupby(col_name):
            out_rows.append({
                "state": int(st),
                "n_trades": int(len(g)),
                "share_of_trades_pct": float(len(g) / total * 100),
                "win_rate_pct": float((g["pnl_usd"] > 0).mean() * 100),
                "structure_ko_rate_pct": float(g["_structure_ko"].mean() * 100),
                "total_pnl_usd": float(g["pnl_usd"].sum()),
                "mean_pnl_usd": float(g["pnl_usd"].mean()),
            })
        return pd.DataFrame(out_rows, columns=cols).sort_values("state").reset_index(drop=True)

    # Attribute states for each leg
    if regime_panel_a is not None and not regime_panel_a.empty:
        pa = regime_panel_a.copy()
        pa.index = pd.to_datetime(pa.index).normalize()
        trades["state_a"] = trades["trade_date"].map(pa["state"].to_dict())
        out["by_state_a"] = _summarize_by_col("state_a")

    if regime_panel_b is not None and not regime_panel_b.empty:
        pb = regime_panel_b.copy()
        pb.index = pd.to_datetime(pb.index).normalize()
        trades["state_b"] = trades["trade_date"].map(pb["state"].to_dict())
        out["by_state_b"] = _summarize_by_col("state_b")

    # Joint breakdown (state_a, state_b)
    if ("state_a" in trades.columns and "state_b" in trades.columns
            and trades["state_a"].notna().any()
            and trades["state_b"].notna().any()):
        sub = trades.dropna(subset=["state_a", "state_b"]).copy()
        if not sub.empty:
            sub["state_a"] = sub["state_a"].astype(int)
            sub["state_b"] = sub["state_b"].astype(int)
            total_joint = len(sub)
            joint_rows = []
            for (sa, sb), g in sub.groupby(["state_a", "state_b"]):
                joint_rows.append({
                    "state_a": int(sa),
                    "state_b": int(sb),
                    "n_trades": int(len(g)),
                    "share_of_trades_pct": float(len(g) / total_joint * 100),
                    "win_rate_pct": float((g["pnl_usd"] > 0).mean() * 100),
                    "structure_ko_rate_pct": float(g["_structure_ko"].mean() * 100),
                    "total_pnl_usd": float(g["pnl_usd"].sum()),
                    "mean_pnl_usd": float(g["pnl_usd"].mean()),
                })
            out["joint"] = (pd.DataFrame(joint_rows)
                              .sort_values(["state_a", "state_b"])
                              .reset_index(drop=True))
    return out


def worstof_export_time_series(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Long-format time-series export for one worst-of strategy.

    Parallel to `core.backtest.export_strategy_time_series` — same
    schema and same semantics. Monthly and annual rows include
    end-of-period `equity_usd` and `drawdown_usd` snapshots so
    downstream apps can recompute DD-based ratios (Calmar, ulcer) at
    any granularity without rebuilding from daily.

    Columns: period_type, period_end, pnl_usd, equity_usd, drawdown_usd.
    Caller prepends `strategy_name` when concatenating across strategies.
    """
    if trades_df.empty or "pnl_usd" not in trades_df.columns:
        return pd.DataFrame(columns=["period_type", "period_end",
                                       "pnl_usd", "equity_usd",
                                       "drawdown_usd"])

    eq = worstof_equity_curve(trades_df)
    if eq.empty:
        return pd.DataFrame(columns=["period_type", "period_end",
                                       "pnl_usd", "equity_usd",
                                       "drawdown_usd"])

    daily = pd.DataFrame({
        "period_type": "daily",
        "period_end": eq.index,
        "pnl_usd": eq["pnl_usd"].values,
        "equity_usd": eq["equity_usd"].values,
        "drawdown_usd": eq["drawdown_usd"].values,
    })

    # Monthly: end-of-month equity + drawdown snapshots, with PnL summed
    monthly_pnl = eq["pnl_usd"].resample("ME").sum()
    monthly_eq = eq["equity_usd"].resample("ME").last().ffill()
    monthly_dd = eq["drawdown_usd"].resample("ME").last().ffill()
    monthly_df = pd.DataFrame({
        "period_type": "monthly",
        "period_end": monthly_pnl.index,
        "pnl_usd": monthly_pnl.values,
        "equity_usd": monthly_eq.reindex(monthly_pnl.index).values,
        "drawdown_usd": monthly_dd.reindex(monthly_pnl.index).values,
    })

    # Annual: year-end snapshots
    annual_pnl = eq["pnl_usd"].groupby(eq.index.year).sum()
    annual_eq = eq["equity_usd"].groupby(eq.index.year).last()
    annual_dd = eq["drawdown_usd"].groupby(eq.index.year).last()
    year_ends = pd.to_datetime([f"{int(y)}-12-31" for y in annual_pnl.index])
    annual_df = pd.DataFrame({
        "period_type": "annual",
        "period_end": year_ends,
        "pnl_usd": annual_pnl.values,
        "equity_usd": annual_eq.reindex(annual_pnl.index).values,
        "drawdown_usd": annual_dd.reindex(annual_pnl.index).values,
    })

    return pd.concat([daily, monthly_df, annual_df], ignore_index=True)


def worstof_annual_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Annual summary: trades, premium paid, payoff, PnL, win rate per year."""
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["expiry_date"] = pd.to_datetime(out["expiry_date"])
    out["year"] = out["expiry_date"].dt.year
    grp = out.groupby("year")
    rows = []
    for yr, sub in grp:
        rows.append({
            "year": int(yr),
            "n_trades": int(len(sub)),
            "win_rate_pct": float((sub["worst_of_payoff_usd"] > 0).mean() * 100),
            "leg_a_ko_pct": float(sub["leg_a_knocked_out"].mean() * 100),
            "leg_b_ko_pct": float(sub["leg_b_knocked_out"].mean() * 100),
            "total_premium_paid_usd": float(sub["structure_premium_paid_usd"].sum()),
            "total_tx_cost_usd": float(sub["tx_cost_usd"].sum()),
            "total_payout_usd": float(sub["worst_of_payoff_usd"].sum()),
            "total_pnl_usd": float(sub["pnl_usd"].sum()),
        })
    return pd.DataFrame(rows).set_index("year")

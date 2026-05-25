"""VV-aware finite-difference Greeks for FX options.

Bloomberg's "Vanna-Volga" Greeks include the smile-sensitivity term,
i.e. they bump spot/vol/etc and recompute the FULL VV-corrected price.
Our plain BS-Δ at σ_smile(K) misses this and is materially off:

    Plain BS-Δ at σ_smile  : 50.37% (vanilla ATMF call USDJPY 1M)
    VV-Δ via FD            : 56.45%
    Bloomberg target       : 54.41%

VV-Δ is much closer (within 2%) than plain (7% off). The residual is
typically from smile dynamics (sticky-strike vs sticky-delta) — a
deeper modelling choice that we don't tackle here.

# API

Two helpers, one for each option category:

    vv_greeks_vanilla(opt, S, K, T, σ_atm, RR, BF, r_d, r_f, *, bumps)
    vv_greeks_ko(opt, bar_dir, S, K, H, T, σ_atm, RR, BF, r_d, r_f,
                  exercise_style='european', *, bumps)

Each returns a dict with `delta`, `gamma`, `vega`, `theta_per_year`.

# Implementation notes

- All Greeks computed by central FD on the FULL VV-corrected price
  (price-at-σ_smile for vanillas, BS+correction for KOs).
- `vega` is special: when bumping σ, we bump σ_atm (and hold RR/BF
  fixed). This is "parallel-shift" vega. The full VV machinery
  internally recomputes σ_smile.
- Theta uses one-sided FD because T can only decrease.
- Bump sizes: spot ±0.5% (default), vol ±1 vol pt, time -1 day.
- For KOs we recompute the smile vol at K each bumped-S since the
  smile depends on S (sticky-strike convention).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.vanilla import vanilla_price
from core.ko import ko_price
from core.american_barrier import ako_closed_form
from core.smile import smile_vol_at_strike
from core.vanna_volga import vv_correction


@dataclass
class GreeksResult:
    price: float            # the unbumped VV price (per unit FOR notional)
    delta: float            # ∂P/∂S
    gamma: float            # ∂²P/∂S²
    vega: float             # ∂P/∂σ_atm (per 1.0 in σ)
    theta_per_year: float   # ∂P/∂T (negative for long calls)


# =============================================================================
# Vanilla VV Greeks
# =============================================================================
def vv_price_vanilla(opt: str, S: float, K: float, T: float,
                       sigma_atm: float, rr_25: float, bf_25: float,
                       r_d: float, r_f: float) -> float:
    """Vanna-Volga vanilla price = BS at σ_atm + VV correction.

    Equivalent (up to tiny VV residuals) to the BS price at σ_smile(K),
    but includes the proper smile-sensitivity for Greeks.
    """
    p_bs_atm = vanilla_price(opt, S, K, T, sigma_atm, r_d, r_f)

    def closure(S_, sigma_):
        return vanilla_price(opt, S_, K, T, sigma_, r_d, r_f)

    vv_out = vv_correction(closure, S, T, sigma_atm, rr_25, bf_25,
                              r_d, r_f)
    return p_bs_atm + vv_out["correction"]


def vv_greeks_vanilla(opt: str, S: float, K: float, T: float,
                        sigma_atm: float, rr_25: float, bf_25: float,
                        r_d: float, r_f: float,
                        *,
                        bump_S_frac: float = 0.001,
                        bump_sigma: float = 0.001,
                        ) -> GreeksResult:
    """Greeks of a vanilla under the VV smile model, via FD."""
    def P(S_=S, sigma_atm_=sigma_atm, T_=T):
        return vv_price_vanilla(opt, S_, K, T_, sigma_atm_, rr_25, bf_25,
                                  r_d, r_f)

    p0 = P()

    # Δ + Γ : bump S
    hS = S * bump_S_frac
    p_up = P(S_=S + hS)
    p_dn = P(S_=S - hS)
    delta = (p_up - p_dn) / (2 * hS)
    gamma = (p_up - 2 * p0 + p_dn) / hS**2

    # Vega : bump σ_atm (parallel shift; RR/BF held fixed)
    p_vup = P(sigma_atm_=sigma_atm + bump_sigma)
    p_vdn = P(sigma_atm_=max(sigma_atm - bump_sigma, 1e-6))
    vega = (p_vup - p_vdn) / (2 * bump_sigma)

    # Theta : one-sided FD on T (T -> T - 1 day)
    h_T = 1.0 / 365.0
    if T - h_T > 1e-6:
        p_th = P(T_=T - h_T)
        # We want theta_per_year = ∂P/∂T (NEGATIVE for long calls
        # since the call LOSES value as T shrinks — i.e. less time
        # to expiry means lower price).
        # ∂P/∂T ≈ (P(T) - P(T-h)) / h is POSITIVE if longer T = more
        # value, so for a long call P(T) > P(T-h), making this
        # expression POSITIVE. That's the wrong sign — we want it
        # negative. So we negate.
        theta_per_year = -(p0 - p_th) / h_T
    else:
        theta_per_year = float("nan")

    return GreeksResult(price=p0, delta=delta, gamma=gamma, vega=vega,
                          theta_per_year=theta_per_year)


# =============================================================================
# KO VV Greeks (European or American)
# =============================================================================
def vv_price_ko_full(opt: str, bar_dir: str,
                      S: float, K: float, H: float, T: float,
                      sigma_atm: float, rr_25: float, bf_25: float,
                      r_d: float, r_f: float,
                      exercise_style: str = "european") -> float:
    """VV-corrected KO price — pricer dispatched by exercise style.

    `exercise_style` ∈ {'european', 'american'}.
    """
    flat_pricer = (ko_price if exercise_style == "european"
                    else ako_closed_form)
    p_bs = flat_pricer(opt, bar_dir, S, K, H, T, sigma_atm, r_d, r_f)

    def closure(S_, sigma_):
        return flat_pricer(opt, bar_dir, S_, K, H, T, sigma_, r_d, r_f)

    vv_out = vv_correction(closure, S, T, sigma_atm, rr_25, bf_25,
                              r_d, r_f)
    return max(0.0, p_bs + vv_out["correction"])


def vv_greeks_ko(opt: str, bar_dir: str,
                  S: float, K: float, H: float, T: float,
                  sigma_atm: float, rr_25: float, bf_25: float,
                  r_d: float, r_f: float,
                  *,
                  exercise_style: str = "european",
                  bump_S_frac: float = 0.001,
                  bump_sigma: float = 0.001,
                  ) -> GreeksResult:
    """Greeks of a KO under the VV smile model, via FD."""
    def P(S_=S, sigma_atm_=sigma_atm, T_=T):
        return vv_price_ko_full(opt, bar_dir, S_, K, H, T_,
                                   sigma_atm_, rr_25, bf_25, r_d, r_f,
                                   exercise_style=exercise_style)

    p0 = P()
    hS = S * bump_S_frac
    p_up = P(S_=S + hS)
    p_dn = P(S_=S - hS)
    delta = (p_up - p_dn) / (2 * hS)
    gamma = (p_up - 2 * p0 + p_dn) / hS**2

    p_vup = P(sigma_atm_=sigma_atm + bump_sigma)
    p_vdn = P(sigma_atm_=max(sigma_atm - bump_sigma, 1e-6))
    vega = (p_vup - p_vdn) / (2 * bump_sigma)

    h_T = 1.0 / 365.0
    if T - h_T > 1e-6:
        p_th = P(T_=T - h_T)
        # ∂P/∂T (negative for long options) — same logic as vanilla.
        theta_per_year = -(p0 - p_th) / h_T
    else:
        theta_per_year = float("nan")

    return GreeksResult(price=p0, delta=delta, gamma=gamma, vega=vega,
                          theta_per_year=theta_per_year)

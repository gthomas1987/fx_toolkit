"""Finite-difference Greeks for worst-of FX structures.

Computes Δ, Γ, ν, ∂V/∂ρ, Θ for two-leg worst-of structures (European
EKO OR American RKO). The method is generic — the same code dispatches
to whichever underlying pricer the caller passes in.

# Pricer-agnostic API

`worstof_greeks_fd(leg_a, leg_b, T, rho, r_d, pricer, *, pricer_kwargs,
                    bump_*)` returns a dict of Greeks. The `pricer`
argument is a function with signature `pricer(leg_a, leg_b, T, rho,
r_d, **pricer_kwargs) -> dict` matching the convention of
`worstof_eko_price_cf`, `worstof_eko_price_mc`,
`worstof_rko_price_cf_approx`, and `worstof_rko_price_mc`.

# Bump sizes (chosen for FX trade-by-trade pricing)

Per-leg spot   : ±0.5% (0.005 fractional)
Per-leg sigma  : ±0.01 absolute (1 vol point — FX convention)
Rho            : ±0.01 absolute
Time           : -1 calendar day = -1/365 yr (one-sided)

All bumps are CENTRAL differences (bump up + down, divide by 2*h)
except theta which is one-sided.

# MC noise control: common random numbers (CRN)

When pricing with the MC engines, every bumped repricing uses the SAME
`seed`. Path-level noise then cancels in the difference, so a sensible
Greek estimate is achievable even at modest path counts. The default
seed (`mc_greek_seed=42`) is fixed at the function level — callers can
override.

# Units convention

Greeks are returned in the SAME units as the pricer's price field
(typically "% of leg-A notional" if you used the engine-convention
normalization S=1). Per-leg deltas are unitless ratios (∂V/∂S_i) when
S_i is normalized to 1. Multiply by notional_usd at the call site to
get USD risks.

# What's NOT computed here

* Vanna (∂²V/∂S∂σ) and Volga (∂²V/∂σ²) per leg — second-order vol Greeks.
* Cross-gamma (∂²V/∂S_A∂S_B) — joint spot convexity.
* Smile-and-VV-aware structure Greeks — these would require a structure-
  level VV correction layer (separate follow-up).
* Pathwise / likelihood-ratio MC Greeks — they're more accurate than
  FD+CRN at the cost of considerable extra code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Optional

from core.worstof_pricer import WorstOfLeg


__all__ = ["worstof_greeks_fd", "WorstOfGreeks"]


@dataclass(frozen=True)
class WorstOfGreeks:
    """Container for worst-of Greeks. All values in the same units as
    the pricer's `price` field (typically % of normalized notional).

    Per-leg Greeks:
        delta_a, delta_b   : ∂V/∂S_i  (dimensionless when S_i is normalized to 1)
        gamma_a, gamma_b   : ∂²V/∂S_i²
        vega_a, vega_b     : ∂V/∂σ_i  (per 1.0 in σ, but reported here scaled
                                       per 1 vol point — see bump_sigma)
    Structure Greeks:
        rho_sensitivity    : ∂V/∂ρ   (per 1.0 in ρ — typically /100 to get
                                       "per 1% in ρ"; left raw here for the
                                       caller to interpret)
        theta_per_day      : -∂V/∂T  (per calendar day — positive means
                                       value DECREASES with time, the FX
                                       sign convention)

    Diagnostics:
        price_base         : the unbumped price (for reference)
        bump_sizes         : dict of bumps actually used
        method             : 'central_fd' or 'central_fd_with_crn'
    """
    delta_a: float
    delta_b: float
    gamma_a: float
    gamma_b: float
    vega_a: float
    vega_b: float
    rho_sensitivity: float
    theta_per_day: float
    price_base: float
    bump_sizes: dict
    method: str

    def to_dict(self) -> dict:
        return {
            "delta_a": self.delta_a,
            "delta_b": self.delta_b,
            "gamma_a": self.gamma_a,
            "gamma_b": self.gamma_b,
            "vega_a": self.vega_a,
            "vega_b": self.vega_b,
            "rho_sensitivity": self.rho_sensitivity,
            "theta_per_day": self.theta_per_day,
            "price_base": self.price_base,
            "bump_sizes": self.bump_sizes,
            "method": self.method,
        }


def worstof_greeks_fd(
        leg_a: WorstOfLeg, leg_b: WorstOfLeg,
        T: float, rho: float, r_d: float,
        pricer: Callable[..., dict],
        pricer_kwargs: Optional[dict] = None,
        *,
        bump_spot: float = 0.005,
        bump_sigma: float = 0.01,
        bump_rho: float = 0.01,
        bump_theta_days: float = 1.0,
        mc_greek_seed: int = 42,
) -> WorstOfGreeks:
    """Compute worst-of Greeks via central finite differences.

    Parameters
    ----------
    leg_a, leg_b : WorstOfLeg
        The two legs.
    T, rho, r_d : float
        Time to expiry, correlation, structure discount rate.
    pricer : callable
        One of (worstof_eko_price_cf, worstof_eko_price_mc,
        worstof_rko_price_cf_approx, worstof_rko_price_mc) or any
        function with the same signature returning a dict with a
        'price' key.
    pricer_kwargs : dict, optional
        Extra kwargs passed to every pricer call. For MC pricers,
        include `n_paths`; for `worstof_rko_price_mc`, include
        `monitoring` and any `daily_ohlc_*` arrays. The `seed` key,
        if present in pricer_kwargs, is OVERWRITTEN by `mc_greek_seed`
        on every call so all bumped reprices use common random numbers.
    bump_spot : float, default 0.005
        Fractional bump on each leg's spot (S → S * (1 ± h)).
    bump_sigma : float, default 0.01
        Absolute bump on each leg's sigma (σ → σ ± h). 0.01 = 1 vol pt.
    bump_rho : float, default 0.01
        Absolute bump on ρ. 0.01 = 1 "rho point".
    bump_theta_days : float, default 1.0
        Calendar days by which to advance time for theta. Result is
        per-day; multiply by 365 for annualized theta.
    mc_greek_seed : int, default 42
        Seed used in every pricer call to enforce common random
        numbers. Has no effect on CF pricers (deterministic).

    Returns
    -------
    WorstOfGreeks dataclass with all computed Greeks.

    Notes
    -----
    Time complexity:
      - 1 base call + 2 per leg-spot + 2 per leg-sigma + 2 for rho +
        1 for theta = 11 pricer calls.
      - With CF-approx (~2 ms/call): ~22 ms total.
      - With MC at 20k paths (~50 ms/call): ~550 ms total.
      - With MC at 100k paths (~400 ms/call): ~4.4 s total.
    """
    if pricer_kwargs is None:
        pricer_kwargs = {}

    # Detect whether the pricer accepts a 'seed' kwarg. If yes, we'll
    # force CRN; otherwise (CF pricers) we just call as-is.
    import inspect
    sig = inspect.signature(pricer)
    has_seed = "seed" in sig.parameters
    method = "central_fd_with_crn" if has_seed else "central_fd"

    def _price(legA, legB, T_, rho_) -> float:
        """Call the pricer with CRN-fixed seed and return the price."""
        kw = dict(pricer_kwargs)
        if has_seed:
            kw["seed"] = mc_greek_seed
        out = pricer(legA, legB, T_, rho_, r_d, **kw)
        return float(out["price"])

    # ---- Base price ----
    price_base = _price(leg_a, leg_b, T, rho)

    # ---- Spot deltas + gammas (central FD on each leg's S) ----
    # When the engine normalizes spots to S=1, K and H are passed through
    # already-rescaled by the caller. We need to bump S CONSISTENTLY: a
    # multiplicative bump on S and the corresponding multiplicative
    # rescale on K and H so the trade economics are preserved. Otherwise
    # bumping S alone would also change moneyness, contaminating delta
    # with vega-like effects.
    #
    # Actually no — the standard FX delta IS "change in V for change in
    # S holding K, H fixed". The bumped pricer should see a new spot
    # WITHOUT rescaling K, H. That's exactly what a real spot move
    # would do to a fixed strike/barrier trade. So we bump S directly.
    h_S = bump_spot
    leg_a_up = replace(leg_a, S=leg_a.S * (1 + h_S))
    leg_a_dn = replace(leg_a, S=leg_a.S * (1 - h_S))
    p_a_up = _price(leg_a_up, leg_b, T, rho)
    p_a_dn = _price(leg_a_dn, leg_b, T, rho)
    delta_a = (p_a_up - p_a_dn) / (2 * leg_a.S * h_S)
    gamma_a = (p_a_up - 2 * price_base + p_a_dn) / ((leg_a.S * h_S) ** 2)

    leg_b_up = replace(leg_b, S=leg_b.S * (1 + h_S))
    leg_b_dn = replace(leg_b, S=leg_b.S * (1 - h_S))
    p_b_up = _price(leg_a, leg_b_up, T, rho)
    p_b_dn = _price(leg_a, leg_b_dn, T, rho)
    delta_b = (p_b_up - p_b_dn) / (2 * leg_b.S * h_S)
    gamma_b = (p_b_up - 2 * price_base + p_b_dn) / ((leg_b.S * h_S) ** 2)

    # ---- Per-leg vegas (central FD on each leg's σ) ----
    # Bump σ by absolute amount h_σ. Output is "value change per 1.0
    # change in σ"; multiply by 0.01 at display time to get "per vol
    # point" (the FX desk convention).
    h_sigma = bump_sigma
    p_va_up = _price(replace(leg_a, sigma=leg_a.sigma + h_sigma), leg_b, T, rho)
    p_va_dn = _price(replace(leg_a, sigma=leg_a.sigma - h_sigma), leg_b, T, rho)
    vega_a = (p_va_up - p_va_dn) / (2 * h_sigma)

    p_vb_up = _price(leg_a, replace(leg_b, sigma=leg_b.sigma + h_sigma), T, rho)
    p_vb_dn = _price(leg_a, replace(leg_b, sigma=leg_b.sigma - h_sigma), T, rho)
    vega_b = (p_vb_up - p_vb_dn) / (2 * h_sigma)

    # ---- Correlation sensitivity (central FD on ρ) ----
    # Clip the bumped ρ to [-1, 1] in case the unbumped value is near
    # the boundary. Use one-sided if a central bump would push ρ
    # past ±1.
    h_rho = bump_rho
    rho_up = rho + h_rho
    rho_dn = rho - h_rho
    if rho_up > 1.0 - 1e-9:
        # near +1 — use backward difference
        p_rho_up = price_base
        p_rho_dn = _price(leg_a, leg_b, T, rho - h_rho)
        rho_sensitivity = (p_rho_up - p_rho_dn) / h_rho
    elif rho_dn < -1.0 + 1e-9:
        # near -1 — use forward difference
        p_rho_up = _price(leg_a, leg_b, T, rho + h_rho)
        p_rho_dn = price_base
        rho_sensitivity = (p_rho_up - p_rho_dn) / h_rho
    else:
        p_rho_up = _price(leg_a, leg_b, T, rho_up)
        p_rho_dn = _price(leg_a, leg_b, T, rho_dn)
        rho_sensitivity = (p_rho_up - p_rho_dn) / (2 * h_rho)

    # ---- Theta (one-sided FD on T) ----
    # FX convention: theta = -dV/dT, positive when value decreases as
    # time passes. We bump T DOWN by 1 day (T → T - 1/365). If T - 1d
    # would be non-positive (or vanishingly small), set theta to NaN
    # rather than producing a meaningless extrapolation.
    h_T = bump_theta_days / 365.0
    if T - h_T <= 1e-6:
        theta_per_day = float("nan")
    else:
        p_theta = _price(leg_a, leg_b, T - h_T, rho)
        # -dV/dT, per day:
        # dV/dT ≈ (V(T) - V(T-h)) / h
        # theta = -dV/dT * h_T (per day) = (V(T-h) - V(T)) / (h_T / h_T)
        # The standard "per day" theta is just (V(T-h_T) - V(T)) since
        # h_T = 1 day in years.
        theta_per_day = p_theta - price_base

    return WorstOfGreeks(
        delta_a=delta_a, delta_b=delta_b,
        gamma_a=gamma_a, gamma_b=gamma_b,
        vega_a=vega_a, vega_b=vega_b,
        rho_sensitivity=rho_sensitivity,
        theta_per_day=theta_per_day,
        price_base=price_base,
        bump_sizes={
            "spot_frac": bump_spot,
            "sigma_abs": bump_sigma,
            "rho_abs": bump_rho,
            "theta_days": bump_theta_days,
        },
        method=method,
    )

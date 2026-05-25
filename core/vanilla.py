"""Vanilla FX option pricing — Garman-Kohlhagen + full Greek set.

Convention:
    S = spot (DOM per 1 unit FOR)
    K = strike
    T = years to expiry
    sigma = annualised vol (decimal)
    r_d = domestic continuously-compounded rate
    r_f = foreign continuously-compounded rate
    F = S * exp((r_d - r_f) * T)

All prices and Greeks returned per 1 unit FOR notional (DOM units).
Convert to USD P&L by multiplying by FOR notional and (if needed)
dividing by spot for cross-pairs where DOM ≠ USD.

This module exposes BOTH the original ko_pricer API (vanilla_price
with positional `option_type`, vanilla_spot_delta, delta_to_strike,
atm_forward_strike, _d1d2) AND the newer app_11 API (d1d2 as a public
name, vanilla_delta, vanilla_gamma, vanilla_vega, vanilla_vanna,
vanilla_volga, vanilla_charm, vanilla_theta, strike_from_delta). The
two sets are aliases of the same underlying math — no duplication.
"""
from __future__ import annotations
from statistics import NormalDist
import numpy as np

_N = NormalDist()


def norm_cdf(x: float) -> float:
    return _N.cdf(x)


def norm_pdf(x: float) -> float:
    return _N.pdf(x)


def norm_ppf(p: float) -> float:
    """Inverse CDF with a small clip to avoid blow-ups at the tails."""
    return _N.inv_cdf(min(max(p, 1e-9), 1.0 - 1e-9))


def d1d2(S: float, K: float, T: float, sigma: float,
         r_d: float, r_f: float) -> tuple[float, float]:
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0
    sT = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma * sigma) * T) / sT
    d2 = d1 - sT
    return d1, d2


# Backwards-compat alias for existing ko_pricer modules that import the
# underscored name.
_d1d2 = d1d2


def vanilla_price(opt: str, S: float, K: float, T: float,
                  sigma: float, r_d: float, r_f: float) -> float:
    """Black-Scholes / Garman-Kohlhagen call/put price.

    `opt` is "call" or "put". Accepts the legacy name "option_type" via
    positional invocation — callers like `vanilla_price("call", ...)`
    work unchanged.
    """
    if T <= 0:
        return max(S - K, 0.0) if opt == "call" else max(K - S, 0.0)
    d1, d2 = d1d2(S, K, T, sigma, r_d, r_f)
    df_f = np.exp(-r_f * T)
    df_d = np.exp(-r_d * T)
    if opt == "call":
        return S * df_f * norm_cdf(d1) - K * df_d * norm_cdf(d2)
    return K * df_d * norm_cdf(-d2) - S * df_f * norm_cdf(-d1)


def vanilla_delta(opt: str, S: float, K: float, T: float, sigma: float,
                  r_d: float, r_f: float) -> float:
    """Spot delta (Garman-Kohlhagen)."""
    if T <= 0:
        if opt == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1, _ = d1d2(S, K, T, sigma, r_d, r_f)
    df_f = np.exp(-r_f * T)
    if opt == "call":
        return df_f * norm_cdf(d1)
    return df_f * (norm_cdf(d1) - 1.0)


# ko_pricer's historical name for the same function.
vanilla_spot_delta = vanilla_delta


def vanilla_gamma(S: float, K: float, T: float, sigma: float,
                  r_d: float, r_f: float) -> float:
    """∂²Price/∂S² — same for call and put."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = d1d2(S, K, T, sigma, r_d, r_f)
    df_f = np.exp(-r_f * T)
    return df_f * norm_pdf(d1) / (S * sigma * np.sqrt(T))


def vanilla_vega(S: float, K: float, T: float, sigma: float,
                 r_d: float, r_f: float) -> float:
    """∂Price/∂σ per 1.00 of σ (= 100 vol points). Divide by 100 for per-vol-point vega."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = d1d2(S, K, T, sigma, r_d, r_f)
    return S * np.exp(-r_f * T) * np.sqrt(T) * norm_pdf(d1)


def vanilla_vanna(S: float, K: float, T: float, sigma: float,
                  r_d: float, r_f: float) -> float:
    """∂²Price/∂S∂σ = ∂Delta/∂σ. Same for call and put."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = d1d2(S, K, T, sigma, r_d, r_f)
    return -np.exp(-r_f * T) * norm_pdf(d1) * d2 / sigma


def vanilla_volga(S: float, K: float, T: float, sigma: float,
                  r_d: float, r_f: float) -> float:
    """∂²Price/∂σ² (vomma). Same for call and put."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = d1d2(S, K, T, sigma, r_d, r_f)
    return vanilla_vega(S, K, T, sigma, r_d, r_f) * d1 * d2 / sigma


def vanilla_charm(opt: str, S: float, K: float, T: float, sigma: float,
                  r_d: float, r_f: float) -> float:
    """∂Delta/∂t — rate at which delta decays toward intrinsic delta.

    Returned per 1 year. Divide by 252 for per-day charm.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = d1d2(S, K, T, sigma, r_d, r_f)
    df_f = np.exp(-r_f * T)
    term1 = -r_f * (norm_cdf(d1) if opt == "call" else norm_cdf(d1) - 1.0)
    term2 = (norm_pdf(d1)
                * (2 * (r_d - r_f) * T - d2 * sigma * np.sqrt(T))
                / (2 * T * sigma * np.sqrt(T)))
    return df_f * (term1 + term2)


def vanilla_theta(opt: str, S: float, K: float, T: float, sigma: float,
                  r_d: float, r_f: float) -> float:
    """Per-year theta. Divide by 252 for per-day."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = d1d2(S, K, T, sigma, r_d, r_f)
    df_f = np.exp(-r_f * T)
    df_d = np.exp(-r_d * T)
    common = -S * df_f * norm_pdf(d1) * sigma / (2 * np.sqrt(T))
    if opt == "call":
        return (common
                - r_d * K * df_d * norm_cdf(d2)
                + r_f * S * df_f * norm_cdf(d1))
    return (common
            + r_d * K * df_d * norm_cdf(-d2)
            - r_f * S * df_f * norm_cdf(-d1))


def atm_forward_strike(S: float, T: float, r_d: float, r_f: float) -> float:
    """ATM-forward strike."""
    return S * np.exp((r_d - r_f) * T)


def strike_from_delta(opt: str, target_delta: float, S: float, T: float,
                      sigma: float, r_d: float, r_f: float) -> float:
    """Solve K such that |vanilla_delta(opt, S, K, T, σ)| = target_delta.

    target_delta convention: POSITIVE for both calls and puts. ATM if 0.
    """
    if target_delta <= 0:
        return S * np.exp((r_d - r_f) * T)
    if T <= 0 or sigma <= 0:
        return S
    sT = sigma * np.sqrt(T)
    if opt == "call":
        N_d1 = target_delta * np.exp(r_f * T)
    else:
        N_d1 = 1.0 - target_delta * np.exp(r_f * T)
    d1 = norm_ppf(N_d1)
    F = S * np.exp((r_d - r_f) * T)
    return F * np.exp(-d1 * sT + 0.5 * sigma * sigma * T)


# Backwards-compat alias for existing ko_pricer modules (ko.py,
# ko_solvers.py) that imported under the older name.
def delta_to_strike(option_type: str, target_delta: float,
                    S: float, T: float, sigma: float,
                    r_d: float, r_f: float) -> float:
    return strike_from_delta(option_type, abs(target_delta),
                              S, T, sigma, r_d, r_f)

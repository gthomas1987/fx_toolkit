"""European knock-out option pricing.

Barrier observed ONLY at expiry. Payoff = vanilla intrinsic × 1{barrier intact at T}.

Decomposition uses vanilla + cash-digital building blocks:
    UO call (K<H):  vanilla_call(K) - vanilla_call(H) - (H-K) * cash_call(H)
    UO call (K>=H): 0
    DO call (K>H):  vanilla_call(K)  (barrier non-binding for in-the-money call)
    DO call (K<=H): vanilla_call(H) + (H-K) * cash_call(H)
    UO put  (K<H):  vanilla_put(K)   (barrier non-binding)
    UO put  (K>=H): vanilla_put(H) + (K-H) * cash_put(H)
    DO put  (K>H):  vanilla_put(K) - vanilla_put(H) - (K-H) * cash_put(H)
    DO put  (K<=H): 0

In-out parity: KO + KI = vanilla (no rebate).
"""
from __future__ import annotations
import numpy as np
from .vanilla import d1d2, norm_cdf, vanilla_price


def cash_call(S: float, X: float, T: float, sigma: float,
              r_d: float, r_f: float) -> float:
    """PV of $1 (DOM) at T if S_T > X."""
    if T <= 0:
        return 1.0 if S > X else 0.0
    _, d2 = d1d2(S, X, T, sigma, r_d, r_f)
    return float(np.exp(-r_d * T) * norm_cdf(d2))


def cash_put(S: float, X: float, T: float, sigma: float,
             r_d: float, r_f: float) -> float:
    """PV of $1 (DOM) at T if S_T < X."""
    if T <= 0:
        return 1.0 if S < X else 0.0
    _, d2 = d1d2(S, X, T, sigma, r_d, r_f)
    return float(np.exp(-r_d * T) * norm_cdf(-d2))


def eko_price(opt: str, bar_dir: str, S: float, K: float, H: float, T: float,
              sigma: float, r_d: float, r_f: float) -> float:
    """European KO price.

    opt:     "call" | "put"
    bar_dir: "up_and_out" | "down_and_out"
    """
    if T <= 0:
        # at expiry: apply barrier + intrinsic directly
        if bar_dir == "up_and_out" and S >= H:
            return 0.0
        if bar_dir == "down_and_out" and S <= H:
            return 0.0
        return max(S - K, 0.0) if opt == "call" else max(K - S, 0.0)

    vc = lambda x: vanilla_price("call", S, x, T, sigma, r_d, r_f)
    vp = lambda x: vanilla_price("put", S, x, T, sigma, r_d, r_f)
    cc = lambda x: cash_call(S, x, T, sigma, r_d, r_f)
    cp = lambda x: cash_put(S, x, T, sigma, r_d, r_f)

    if opt == "call" and bar_dir == "up_and_out":
        if K >= H:
            return 0.0
        return max(vc(K) - vc(H) - (H - K) * cc(H), 0.0)
    if opt == "call" and bar_dir == "down_and_out":
        if K > H:
            return vc(K)
        return max(vc(H) + (H - K) * cc(H), 0.0)
    if opt == "put" and bar_dir == "up_and_out":
        if K < H:
            return vp(K)
        return max(vp(H) + (K - H) * cp(H), 0.0)
    if opt == "put" and bar_dir == "down_and_out":
        if K <= H:
            return 0.0
        return max(vp(K) - vp(H) - (K - H) * cp(H), 0.0)
    raise ValueError(f"Bad opt/bar_dir: {opt}/{bar_dir}")


def survival_prob(bar_dir: str, S: float, H: float, T: float, sigma: float,
                  r_d: float, r_f: float) -> float:
    """P(barrier NOT hit at expiry), risk-neutral. European-monitoring only."""
    if T <= 0:
        if bar_dir == "up_and_out":
            return 0.0 if S >= H else 1.0
        return 0.0 if S <= H else 1.0
    _, d2 = d1d2(S, H, T, sigma, r_d, r_f)
    # N(d2) = P(S_T > H) under risk-neutral.
    if bar_dir == "up_and_out":
        return float(norm_cdf(-d2))  # UO survives if S_T < H
    return float(norm_cdf(d2))       # DO survives if S_T > H

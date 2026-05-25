"""Monte Carlo validation of European KO option prices.

For European KO (barrier checked only at expiry), the MC is trivial:
simulate S_T directly via the lognormal closed form, then apply

    payoff = intrinsic(S_T, K) * 1{barrier check passes at S_T}

No path simulation needed.
"""
from __future__ import annotations
import numpy as np


def mc_ko_price(option_type: str, barrier_type: str,
                S: float, K: float, H: float, T: float, sigma: float,
                r_d: float, r_f: float,
                rebate: float = 0.0,
                n_paths: int = 1_000_000,
                seed: int = 42,
                antithetic: bool = True) -> dict:
    """European KO option Monte Carlo pricer.

    Direct one-shot simulation of S_T under risk-neutral GBM.
    Barrier is checked only at expiry, matching the closed-form spec.

    Returns:
        dict with price, std_err, ci_95_lower, ci_95_upper, ko_prob,
        n_paths.
    """
    rng = np.random.default_rng(seed)
    if antithetic:
        n_half = n_paths // 2
        Z = np.concatenate([rng.standard_normal(n_half),
                             -rng.standard_normal(n_half)])
        n_paths = 2 * n_half
    else:
        Z = rng.standard_normal(n_paths)

    drift = (r_d - r_f - 0.5 * sigma * sigma) * T
    diff = sigma * np.sqrt(T)
    S_T = S * np.exp(drift + diff * Z)

    if barrier_type == "up_and_out":
        knocked = S_T >= H
    elif barrier_type == "down_and_out":
        knocked = S_T <= H
    else:
        raise ValueError(f"Unknown barrier_type: {barrier_type}")

    if option_type == "call":
        intrinsic = np.maximum(S_T - K, 0.0)
    else:
        intrinsic = np.maximum(K - S_T, 0.0)

    payoff = np.where(knocked, rebate, intrinsic)
    discount = np.exp(-r_d * T)
    pv = discount * payoff

    return {
        "price": float(pv.mean()),
        "std_err": float(pv.std(ddof=1) / np.sqrt(n_paths)),
        "ci_95_lower": float(pv.mean() - 1.96 * pv.std(ddof=1) / np.sqrt(n_paths)),
        "ci_95_upper": float(pv.mean() + 1.96 * pv.std(ddof=1) / np.sqrt(n_paths)),
        "ko_prob": float(knocked.mean()),
        "n_paths": int(n_paths),
    }

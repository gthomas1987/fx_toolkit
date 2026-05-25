"""Dual-underlying European knock-out pricer — Monte Carlo.

Two correlated GBMs under risk-neutral measure. Barrier checks at expiry only.

Structures supported:
    - "wo_call"  : worst-of call. Payoff = max(min(S1_T-K1, S2_T-K2), 0)
                   subject to both barrier checks (each leg has its own H).
                   Each leg can be UO or DO independently.

Conventions:
    Each leg has its own (S, K, H, sigma, r_d, r_f, bar_dir).
    The shared correlation rho enters via the random shocks.
    Both legs are "knocked" together: if EITHER leg's barrier is breached,
    payoff = 0  (this is the standard worst-of EKO convention).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class Leg:
    """One leg of a dual structure."""
    S: float
    K: float
    H: float        # barrier (set far if none)
    sigma: float
    r_d: float
    r_f: float
    bar_dir: str    # "up_and_out" | "down_and_out" | "none"
    opt: str = "call"  # "call" | "put"


def _terminal_shocks(n: int, rho: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Antithetic correlated standard normals."""
    rng = np.random.default_rng(seed)
    half = n // 2
    z1 = rng.standard_normal(half)
    z2_indep = rng.standard_normal(half)
    z2 = rho * z1 + np.sqrt(max(1.0 - rho * rho, 0.0)) * z2_indep
    # antithetic
    return (np.concatenate([z1, -z1]),
            np.concatenate([z2, -z2]))


def _terminal_spots(leg: Leg, T: float, z: np.ndarray) -> np.ndarray:
    mu = (leg.r_d - leg.r_f - 0.5 * leg.sigma * leg.sigma) * T
    return leg.S * np.exp(mu + leg.sigma * np.sqrt(T) * z)


def _barrier_alive(leg: Leg, S_T: np.ndarray) -> np.ndarray:
    if leg.bar_dir == "up_and_out":
        return S_T < leg.H
    if leg.bar_dir == "down_and_out":
        return S_T > leg.H
    return np.ones_like(S_T, dtype=bool)


def dual_eko_price(leg1: Leg, leg2: Leg, structure: str, T: float, rho: float,
                   r_d_payoff: float | None = None,
                   n_paths: int = 50_000, seed: int = 42) -> dict:
    """Monte Carlo price + Greeks-friendly diagnostics for a dual EKO.

    structure: "wo_call"  (worst-of call), "wo_put", "bo_call", "bo_put".

    Returns:
        {
          "price_per_pair_unit": price expressed in DOM units assuming
              notional = 1 unit FOR per leg (in this prototype we report
              ccy-1 DOM units; for USD-denominated reporting use a
              translation step in the portfolio layer),
          "p_alive": probability both barriers survive,
          "p_alive_leg1": each leg alive marginally,
          "p_alive_leg2": ...,
          "rho_used": rho,
        }
    """
    if T <= 0:
        # at expiry — handle directly
        alive1 = _barrier_alive(leg1, np.array([leg1.S]))[0]
        alive2 = _barrier_alive(leg2, np.array([leg2.S]))[0]
        if not (alive1 and alive2):
            payoff = 0.0
        else:
            p1 = (leg1.S - leg1.K) if leg1.opt == "call" else (leg1.K - leg1.S)
            p2 = (leg2.S - leg2.K) if leg2.opt == "call" else (leg2.K - leg2.S)
            p1 = max(p1, 0.0)
            p2 = max(p2, 0.0)
            if structure.startswith("wo"):
                payoff = min(p1, p2)
            else:
                payoff = max(p1, p2)
        return {"price_per_pair_unit": payoff, "p_alive": float(alive1 and alive2),
                "p_alive_leg1": float(alive1), "p_alive_leg2": float(alive2),
                "rho_used": rho}

    z1, z2 = _terminal_shocks(n_paths, rho, seed)
    S1_T = _terminal_spots(leg1, T, z1)
    S2_T = _terminal_spots(leg2, T, z2)

    alive1 = _barrier_alive(leg1, S1_T)
    alive2 = _barrier_alive(leg2, S2_T)
    alive_both = alive1 & alive2

    p1 = (S1_T - leg1.K) if leg1.opt == "call" else (leg1.K - S1_T)
    p2 = (S2_T - leg2.K) if leg2.opt == "call" else (leg2.K - S2_T)
    p1 = np.maximum(p1, 0.0)
    p2 = np.maximum(p2, 0.0)

    if structure.startswith("wo"):
        payoff = np.minimum(p1, p2)
    else:
        payoff = np.maximum(p1, p2)

    payoff = np.where(alive_both, payoff, 0.0)

    # discount under leg1.r_d (assumes payoff is in leg1 DOM; in practice
    # payoff currency is contract-specific. For this prototype we treat the
    # MC price as representative of the structure value scaled by leg1 units.)
    df = np.exp(-leg1.r_d * T) if r_d_payoff is None else np.exp(-r_d_payoff * T)
    price = float(df * payoff.mean())

    return {
        "price_per_pair_unit": price,
        "p_alive": float(alive_both.mean()),
        "p_alive_leg1": float(alive1.mean()),
        "p_alive_leg2": float(alive2.mean()),
        "rho_used": rho,
    }

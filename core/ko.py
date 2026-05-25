"""European single-barrier knock-out option pricing.

The barrier is checked **only at expiry** (European-style monitoring). The
option pays the vanilla intrinsic value scaled by an indicator that the
barrier has not been breached at the final fixing.

This is materially different from American/continuous monitoring:
    - Easier to price (elementary linear combination of vanilla and
      digital options — no Reiner-Rubinstein, no BGK correction).
    - More valuable to the buyer of the option (the option survives any
      in-life touches of the barrier, only the final fixing matters).
    - Less common in FX flow markets where American KO is standard, but
      legitimate as an OTC structure when the buyer specifies it.

# Convention
    S, K, H = spot, strike, barrier (DOM per 1 FOR)
    T = years to expiry
    sigma = ATM vol (decimal)
    r_d, r_f = domestic / foreign continuously-compounded rates

# Payoffs (rebate = 0)
    European UO call: (S_T - K)+ * 1{S_T < H}
    European DO call: (S_T - K)+ * 1{S_T > H}
    European UO put:  (K - S_T)+ * 1{S_T < H}
    European DO put:  (K - S_T)+ * 1{S_T > H}

# Closed-form decompositions
    UO call, K <  H:  vanilla_call(K) - vanilla_call(H) - (H - K) * cash_call(H)
    UO call, K >= H:  0       [requires S_T > K >= H AND S_T < H -> empty set]
    DO call, K >  H:  vanilla_call(K)               [barrier non-binding]
    DO call, K <= H:  vanilla_call(H) + (H - K) * cash_call(H)
    UO put,  K <  H:  vanilla_put(K)                [barrier non-binding]
    UO put,  K >= H:  vanilla_put(H) + (K - H) * cash_put(H)
    DO put,  K >  H:  vanilla_put(K) - vanilla_put(H) - (K - H) * cash_put(H)
    DO put,  K <= H:  0       [requires S_T < K <= H AND S_T > H -> empty set]

where:
    cash_call(X) = e^(-r_d T) * N(d2(X))         [PV of $1 at T if S_T > X]
    cash_put (X) = e^(-r_d T) * N(-d2(X))        [PV of $1 at T if S_T < X]
    d2(X) = (ln(S/X) + (r_d - r_f - sigma^2/2) * T) / (sigma * sqrt(T))

# In-out parity
    For European KO with barrier check only at expiry:
        KO + KI = vanilla     (rebate = 0)
"""
from __future__ import annotations
import numpy as np

from core.vanilla import (
    norm_cdf, vanilla_price, _d1d2,
)


# -----------------------------------------------------------------------------
# Digital building blocks
# -----------------------------------------------------------------------------
def _cash_call(S: float, X: float, T: float, sigma: float,
               r_d: float, r_f: float) -> float:
    """Cash-or-nothing call: PV of $1 (DOM) paid at T if S_T > X."""
    if T <= 0:
        return 1.0 if S > X else 0.0
    _, d2 = _d1d2(S, X, T, sigma, r_d, r_f)
    return float(np.exp(-r_d * T) * norm_cdf(d2))


def _cash_put(S: float, X: float, T: float, sigma: float,
              r_d: float, r_f: float) -> float:
    """Cash-or-nothing put: PV of $1 (DOM) paid at T if S_T < X."""
    if T <= 0:
        return 1.0 if S < X else 0.0
    _, d2 = _d1d2(S, X, T, sigma, r_d, r_f)
    return float(np.exp(-r_d * T) * norm_cdf(-d2))


# -----------------------------------------------------------------------------
# European KO pricer
# -----------------------------------------------------------------------------
def ko_price(option_type: str, barrier_type: str,
             S: float, K: float, H: float, T: float, sigma: float,
             r_d: float, r_f: float,
             rebate: float = 0.0) -> float:
    """European knock-out option price.

    Args:
        option_type: "call" or "put"
        barrier_type: "up_and_out" or "down_and_out"
        S, K, H, T, sigma, r_d, r_f: standard FX option params
        rebate: cash paid at T if barrier breached at expiry (default 0)

    Returns:
        Price in DOM per 1 unit of FOR notional. To convert to USD-equivalent
        premium for FOR notional N (in FOR units): multiply by N. To convert
        from a USD-quoted notional N_usd: divide by S, multiply by N_usd.
    """
    if T <= 0:
        if option_type == "call":
            intrinsic = max(S - K, 0.0)
        else:
            intrinsic = max(K - S, 0.0)
        if barrier_type == "up_and_out" and S >= H:
            return rebate
        if barrier_type == "down_and_out" and S <= H:
            return rebate
        return intrinsic

    is_call = option_type == "call"
    is_up = barrier_type == "up_and_out"

    if is_call and is_up:               # ----- UO call -----
        if K >= H:
            ko_value = 0.0
        else:
            v_K = vanilla_price("call", S, K, T, sigma, r_d, r_f)
            v_H = vanilla_price("call", S, H, T, sigma, r_d, r_f)
            c_H = _cash_call(S, H, T, sigma, r_d, r_f)
            ko_value = max(v_K - v_H - (H - K) * c_H, 0.0)

    elif is_call and not is_up:          # ----- DO call -----
        if K > H:
            ko_value = vanilla_price("call", S, K, T, sigma, r_d, r_f)
        else:
            v_H = vanilla_price("call", S, H, T, sigma, r_d, r_f)
            c_H = _cash_call(S, H, T, sigma, r_d, r_f)
            ko_value = max(v_H + (H - K) * c_H, 0.0)

    elif (not is_call) and is_up:        # ----- UO put -----
        if K < H:
            ko_value = vanilla_price("put", S, K, T, sigma, r_d, r_f)
        else:
            v_H = vanilla_price("put", S, H, T, sigma, r_d, r_f)
            p_H = _cash_put(S, H, T, sigma, r_d, r_f)
            ko_value = max(v_H + (K - H) * p_H, 0.0)

    else:                                  # ----- DO put -----
        if K <= H:
            ko_value = 0.0
        else:
            v_K = vanilla_price("put", S, K, T, sigma, r_d, r_f)
            v_H = vanilla_price("put", S, H, T, sigma, r_d, r_f)
            p_H = _cash_put(S, H, T, sigma, r_d, r_f)
            ko_value = max(v_K - v_H - (K - H) * p_H, 0.0)

    if rebate > 0.0:
        prob_breach = ko_probability(barrier_type, S, H, T, sigma, r_d, r_f)
        ko_value += rebate * np.exp(-r_d * T) * prob_breach

    return ko_value


def ki_price(option_type: str, barrier_type_in: str,
             S: float, K: float, H: float, T: float, sigma: float,
             r_d: float, r_f: float,
             rebate: float = 0.0) -> float:
    """European knock-in price via in-out parity:  KI = vanilla - KO."""
    barrier_out = ("up_and_out" if barrier_type_in == "up_and_in"
                   else "down_and_out")
    vanilla = vanilla_price(option_type, S, K, T, sigma, r_d, r_f)
    ko = ko_price(option_type, barrier_out, S, K, H, T, sigma, r_d, r_f,
                  rebate=0.0)
    return max(vanilla - ko, 0.0)


# -----------------------------------------------------------------------------
# Knock-out probability (at expiry, European)
# -----------------------------------------------------------------------------
def ko_probability(barrier_type: str,
                   S: float, H: float, T: float, sigma: float,
                   r_d: float, r_f: float) -> float:
    """Risk-neutral probability that the barrier is breached at expiry.

    For European monitoring (check at T only):
        UO triggers if  S_T > H:  P = N(d2(H))
        DO triggers if  S_T < H:  P = N(-d2(H))
    """
    if T <= 0:
        if barrier_type == "up_and_out":
            return 1.0 if S >= H else 0.0
        return 1.0 if S <= H else 0.0

    _, d2 = _d1d2(S, H, T, sigma, r_d, r_f)
    if barrier_type == "up_and_out":
        return float(norm_cdf(d2))
    return float(norm_cdf(-d2))


# -----------------------------------------------------------------------------
# KO spot delta — via finite difference (handles branch boundaries cleanly)
# -----------------------------------------------------------------------------
def ko_spot_delta(option_type: str, barrier_type: str,
                  S: float, K: float, H: float, T: float, sigma: float,
                  r_d: float, r_f: float,
                  eps_rel: float = 1e-5) -> float:
    """Spot delta of a European KO option, via central finite difference."""
    eps = max(S * eps_rel, 1e-9)
    p_up = ko_price(option_type, barrier_type, S + eps, K, H, T, sigma, r_d, r_f)
    p_dn = ko_price(option_type, barrier_type, S - eps, K, H, T, sigma, r_d, r_f)
    return (p_up - p_dn) / (2.0 * eps)

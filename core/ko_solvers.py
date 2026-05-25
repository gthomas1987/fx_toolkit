"""Strike + barrier solver for European KO options.

# Workflow
  1. Pin K from a vanilla-Δ target (closed-form inversion of BS spot delta).
  2. Solve H from a target *payout ratio*, defined as:

         payout_ratio = max_payoff / ko_premium

     where max_payoff = |H - K| (for natural KO directions).

# Payout ratio interpretation (leverage)
A 1:1 payout means premium = max payoff (so the trade can at most double
the wager); 8:1 means max payoff is 8× the premium paid (an 8x leveraged
ticket if it pays out at the cap).

# U-shape and the wide branch
ratio(H) is a U-shaped function of H for fixed K:
  - As H → K⁺ (barrier collapses to strike): both max_payoff and
    ko_premium → 0, but premium goes faster (~ε² vs ε), so ratio → ∞.
  - As H → ∞: max_payoff → ∞ linearly; ko_premium → vanilla, so
    ratio → ∞.
  - There is a finite minimum ratio in between.

When `target_ratio < ratio_min`, the structure is infeasible at this
strike — premium is too high relative to max payoff. The solver returns
the H at the minimum and flags it. Otherwise it picks the **wide-branch**
solution (the H further from K) — the conventional KO structure where
the barrier is meaningfully outside spot.

# Supported directions
Natural KO structures only: up-and-out call and down-and-out put. For
unnatural directions (down-out call, up-out put), max_payoff is not
|H - K|, so the leverage definition doesn't apply. The solver returns
the vanilla-K with H placed at a wide default and notes the situation.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from core.ko import ko_price
from core.vanilla import vanilla_price, delta_to_strike as _delta_to_strike_vanilla
from core.smile import smile_vol_at_strike


# -----------------------------------------------------------------------------
# Strike solver: vanilla Δ
# -----------------------------------------------------------------------------
def solve_strike_from_delta(option_type: str, target_delta: float,
                              S: float, T: float, sigma: float,
                              r_d: float, r_f: float) -> float:
    """Solve K such that vanilla spot delta = target_delta.

    target_delta = 0  → ATM (forward).
    target_delta > 0  → for both calls (long delta) and puts (delta=-target_delta).
    """
    if abs(target_delta) < 1e-9:
        return float(S * np.exp((r_d - r_f + 0.5 * sigma * sigma) * T))
    signed_target = target_delta if option_type == "call" else -target_delta
    return float(_delta_to_strike_vanilla(option_type, signed_target,
                                            S, T, sigma, r_d, r_f))


# -----------------------------------------------------------------------------
# Barrier solver: payout ratio = max_payoff / ko_premium
# -----------------------------------------------------------------------------
def _ratio_fn(option_type: str, barrier_type: str,
                S: float, K: float, T: float, sigma: float,
                r_d: float, r_f: float,
                pricer=None):
    """Build the H-> ratio function for the given KO structure.

    `pricer` is a callable with the same signature as `ko_price`. Defaults
    to the European-barrier ko_price; pass ako_closed_form (or any of the
    American-barrier pricers) to solve leverage against continuously-
    monitored barriers instead. Used by app 12 to keep the same solver
    UI for both European and American barriers.
    """
    if pricer is None:
        pricer = ko_price
    if barrier_type == "up_and_out" and option_type == "call":
        max_payoff = lambda H: H - K
    elif barrier_type == "down_and_out" and option_type == "put":
        max_payoff = lambda H: K - H
    else:
        return None

    def ratio(H):
        p = pricer(option_type, barrier_type, S, K, H, T, sigma, r_d, r_f)
        mp = max_payoff(H)
        if p < 1e-15 or mp <= 0:
            return 1e15
        return mp / p
    return ratio


def solve_barrier_from_payout_ratio(option_type: str, barrier_type: str,
                                      K: float, S: float, T: float,
                                      sigma: float, r_d: float, r_f: float,
                                      target_ratio: float,
                                      pricer=None
                                      ) -> tuple[float, dict]:
    """Solve H such that payout_ratio (= max_payoff / ko_premium) = target.

    Returns (H, info_dict). info_dict has 'ratio_min' (the achievable
    minimum at this K) and possibly 'note' (warning text if target was
    infeasible or bisection had to clamp).

    Searches the WIDE branch (H far from K) — the conventional structure.

    `pricer` (optional): custom KO pricing function with the same
    signature as `core.ko.ko_price`. Defaults to European-barrier
    ko_price; pass `core.american_barrier.ako_closed_form` for the
    American-barrier solver.
    """
    ratio = _ratio_fn(option_type, barrier_type, S, K, T, sigma, r_d, r_f,
                         pricer=pricer)
    if ratio is None:
        # Unnatural KO direction — use vanilla K with a far barrier
        if barrier_type == "up_and_out":
            H_default = S * np.exp(5 * sigma * np.sqrt(T))
        else:
            H_default = S * np.exp(-5 * sigma * np.sqrt(T))
        return float(H_default), {
            "note": (f"Payout-ratio framework requires up-out call or "
                     f"down-out put; got {barrier_type} {option_type}. "
                     f"Used a wide default barrier.")
        }

    # Search bounds for the wide branch
    v = vanilla_price(option_type, S, K, T, sigma, r_d, r_f)
    if barrier_type == "up_and_out":
        H_lo = K * 1.00001
        H_far = max(S * np.exp(5 * sigma * np.sqrt(T)),
                     K * 1.5,
                     K + (target_ratio + 5) * max(v, S * 0.001))
        bounds = (H_lo, H_far)
    else:  # down_and_out
        H_far = K - (target_ratio + 5) * max(v, S * 0.001)
        H_far = min(H_far, S * np.exp(-5 * sigma * np.sqrt(T)),
                     K * 0.5)
        H_far = max(H_far, 0.01)
        H_hi = K * 0.99999
        bounds = (H_far, H_hi)

    # Find the bottom of the U
    res = minimize_scalar(ratio, bounds=bounds, method='bounded',
                            options={'xatol': max(S * 1e-5, 1e-5)})
    H_at_min = float(res.x)
    ratio_min = float(res.fun)

    info = {"ratio_min": ratio_min, "H_at_min": H_at_min}

    if target_ratio < ratio_min - 1e-3:
        info["note"] = (
            f"Target {target_ratio:.1f}× leverage is below the achievable "
            f"minimum of {ratio_min:.2f}× at this strike. Returned the H "
            f"that minimises ratio (H={H_at_min:.4f}, giving "
            f"~{ratio_min:.2f}×). Increase the strike Δ (move closer to ATM) "
            f"or lower the target ratio to make the structure feasible."
        )
        return H_at_min, info

    # Bisect on the wide branch (between H_at_min and the far end)
    if barrier_type == "up_and_out":
        bracket = (H_at_min, bounds[1])
    else:
        bracket = (bounds[0], H_at_min)

    f_lo = ratio(bracket[0]) - target_ratio
    f_hi = ratio(bracket[1]) - target_ratio
    if f_lo * f_hi > 0:
        # No sign change — try expanding the far bound
        for mult in (2.0, 5.0, 10.0, 50.0):
            if barrier_type == "up_and_out":
                H_try = bounds[1] * mult
                if ratio(H_try) - target_ratio > 0:
                    bracket = (H_at_min, H_try)
                    break
            else:
                H_try = max(0.01, bounds[0] / mult)
                if ratio(H_try) - target_ratio > 0:
                    bracket = (H_try, H_at_min)
                    break
        else:
            info["note"] = (f"Could not bracket target ratio. Used H_at_min "
                             f"({H_at_min:.4f}, {ratio_min:.2f}×).")
            return H_at_min, info

    try:
        H_solved = float(brentq(lambda h: ratio(h) - target_ratio,
                                  bracket[0], bracket[1], xtol=S * 1e-7))
        return H_solved, {"ratio_min": ratio_min, "H_at_min": H_at_min}
    except ValueError as e:
        info["note"] = f"Bisection failed ({e}); used H_at_min."
        return H_at_min, info


# -----------------------------------------------------------------------------
# Barrier solver: vanilla Δ at the wing
# -----------------------------------------------------------------------------
def solve_barrier_from_delta(barrier_type: str, target_delta: float,
                                S: float, T: float, sigma: float,
                                r_d: float, r_f: float) -> float:
    """Solve H such that the vanilla wing-option's spot delta = target_delta.

    Convention (target_delta is positive, e.g. 0.10 for 10Δ wing):
      - up_and_out:    H is the strike where vanilla CALL has +target_delta
                       (i.e. an OTM call wing strike; H > spot typically).
      - down_and_out:  H is the strike where vanilla PUT has -target_delta
                       (i.e. an OTM put wing strike; H < spot typically).

    Both branches just reuse the closed-form delta-to-strike inverter.
    Caller is responsible for checking H is on the correct side of K
    (i.e. H > K for up-out, H < K for down-out) — when it isn't, the
    structure is degenerate and a note is surfaced upstream.
    """
    if barrier_type == "up_and_out":
        return solve_strike_from_delta('call', target_delta, S, T, sigma, r_d, r_f)
    if barrier_type == "down_and_out":
        return solve_strike_from_delta('put', target_delta, S, T, sigma, r_d, r_f)
    raise ValueError(f"Unknown barrier_type: {barrier_type}")


# -----------------------------------------------------------------------------
# Combined entry point used by the app
# -----------------------------------------------------------------------------
def solve_strike(option_type: str, barrier_type: str, target_delta: float,
                  S: float, T: float, sigma: float,
                  r_d: float, r_f: float,
                  target_ratio: Optional[float] = None,
                  delta_interp: str = "vanilla",
                  rr_25: float = 0.0, bf_25: float = 0.0,
                  ko_method: str = "ratio",
                  target_ko_delta: Optional[float] = None,
                  pricer=None,
                  ) -> tuple[float, float, dict]:
    """Solve (K, H) given target vanilla Δ for the strike, and EITHER:
      - target *payout ratio* (max_payoff/premium leverage) — `ko_method='ratio'`
      - target *vanilla Δ* for the barrier strike       — `ko_method='delta'`

    Both modes solve K at σ_atm (the standard FX delta convention). When
    rr_25/bf_25 are non-zero, σ_smile(K) is computed and used for the H
    solve in ratio mode and for KO pricing in both modes.

    Ratio mode (default): H from the U-shape solver; falls back to
    H_at_min when target_ratio < achievable minimum ('feasible=False').

    Delta mode: H is the vanilla-Δ wing strike at σ_atm (same convention
    as the strike). H is then validated (must be on the correct side of
    K). The achieved payout ratio is reported but not constrained.

    info contains: 'sigma_smile', 'achieved_ratio', 'ko_method' plus
    mode-specific 'ratio_min'/'H_at_min' (ratio) or 'target_ko_delta'
    (delta), and an optional 'note' for any issue.

    `pricer` (optional): a callable with the same signature as
    `core.ko.ko_price`. Defaults to European-barrier pricing. Pass
    `core.american_barrier.ako_closed_form` for American-barrier
    solving — the strike (vanilla-Δ inversion) is unaffected, but the
    leverage solver and the achieved-ratio report use the supplied
    pricer.

    `delta_interp` is retained for backward compatibility — only vanilla
    is supported.
    """
    if pricer is None:
        pricer = ko_price
    # Step 1: K from delta at σ_atm (vanilla — pricer-independent)
    K = solve_strike_from_delta(option_type, target_delta, S, T, sigma, r_d, r_f)
    sigma_smile = smile_vol_at_strike(S, K, T, sigma, rr_25, bf_25, r_d, r_f)

    # Step 2: H from chosen method
    if ko_method == "delta":
        if target_ko_delta is None:
            raise ValueError("ko_method='delta' requires target_ko_delta")
        H = solve_barrier_from_delta(barrier_type, target_ko_delta,
                                        S, T, sigma, r_d, r_f)
        info: dict = {"ko_method": "delta",
                       "target_ko_delta": float(target_ko_delta)}
        # Validate H is on the correct side of K
        if barrier_type == "up_and_out" and H <= K:
            info["note"] = (
                f"KO Δ={target_ko_delta:.2f} → H={H:.4f} ≤ K={K:.4f}. "
                f"Barrier must be above strike for up-out call; choose a "
                f"smaller KO Δ (deeper OTM)."
            )
        elif barrier_type == "down_and_out" and H >= K:
            info["note"] = (
                f"KO Δ={target_ko_delta:.2f} → H={H:.4f} ≥ K={K:.4f}. "
                f"Barrier must be below strike for down-out put; choose a "
                f"smaller KO Δ (deeper OTM)."
            )
    elif ko_method == "ratio":
        if target_ratio is None:
            raise ValueError("ko_method='ratio' requires target_ratio")
        H, info = solve_barrier_from_payout_ratio(
            option_type, barrier_type, K, S, T, sigma_smile,
            r_d, r_f, target_ratio, pricer=pricer,
        )
        info["ko_method"] = "ratio"
    else:
        raise ValueError(f"Unknown ko_method: {ko_method!r} "
                          f"(expected 'ratio' or 'delta')")

    info["sigma_smile"] = float(sigma_smile)

    # Achieved payout ratio (always reported, computed via the supplied pricer)
    p = pricer(option_type, barrier_type, S, K, H, T, sigma_smile, r_d, r_f)
    if p > 1e-15:
        if barrier_type == "up_and_out" and option_type == "call":
            achieved = (H - K) / p
        elif barrier_type == "down_and_out" and option_type == "put":
            achieved = (K - H) / p
        else:
            achieved = float("nan")
        info["achieved_ratio"] = float(achieved)

    return float(K), float(H), info

"""European-exercise / American-barrier (continuously-monitored) knock-out
option pricing — four numerical methods for cross-validation.

# Product definition
    - Exercise: European (only at expiry T)
    - Barrier: American = monitored continuously over [0, T]
        ANY touch of H during the option's life kills the option.
    - Payoff at expiry: vanilla intrinsic × 1{barrier intact throughout}

This is the standard "FX knock-out" sold in flow markets. It's
materially different from the European-barrier KO priced in
`core/ko.py` (where the barrier is checked ONLY at expiry, so in-life
touches that recover are harmless). The American barrier is strictly
CHEAPER than the European barrier (more chances for the barrier to be
hit → lower survival probability → lower price).

# Four pricing methods provided

  1. Closed-form (Reiner-Rubinstein 1991, Haug's 8-formula table)
     Analytic, exact under continuous monitoring. Reference price.

  2. Binomial tree (Cox-Ross-Rubinstein) with Broadie-Glasserman-Kou
     (BGK 1997) continuity correction — shifts the discrete-monitored
     barrier inward by exp(±0.5826 σ √Δt) so the tree converges to the
     continuous-monitoring price as N → ∞.

  3. Trinomial tree (Boyle 1986) with explicit barrier-node placement.
     Δx is chosen so ln(H/S₀) is an integer multiple of Δx, putting
     the barrier exactly on a row of nodes — removes the largest
     source of discretization bias near the barrier.

  4. Finite difference (Crank-Nicolson on Black-Scholes PDE) with
     absorbing Dirichlet boundary at H. Slowest but most flexible —
     extends naturally to American exercise (not used here) or to a
     local-vol surface (future work).

All four expose the same call signature so they can be benchmarked
against each other on the same trade. Closed-form is the analytic
reference; the three numerical methods should converge to it as their
discretization is refined.

# Greeks
Spot delta is provided via central finite difference on the closed-form
pricer (the cheapest and most accurate route). Gamma/vega/theta can be
added in a later phase.

# What this module does NOT cover
- American EXERCISE (early-exercise option holder). The "American" in
  the name refers to barrier monitoring only; exercise is European.
- KNOCK-IN options. Use in-out parity: KI = vanilla − KO (rebate = 0).
- DOUBLE barriers. Single barrier only.
"""
from __future__ import annotations

import numpy as np

from core.vanilla import norm_cdf, vanilla_price


# =============================================================================
# Method 1 — Closed-form Reiner-Rubinstein
# =============================================================================
def _rr_components(S: float, K: float, H: float, T: float, sigma: float,
                       r_d: float, r_f: float,
                       eta: int, phi: int) -> "tuple[float, float, float, float]":
    """Reiner-Rubinstein building blocks A, B, C, D (rebate-free part).

    eta = +1 for down barriers, -1 for up barriers
    phi = +1 for call,           -1 for put

    These are the four components from Haug's "The Complete Guide to
    Option Pricing Formulas" — every barrier-option formula in the
    8-way table below is a linear combination of these four blocks.
    """
    b = r_d - r_f                       # cost of carry (FX convention)
    sT = sigma * np.sqrt(T)
    mu = (b - 0.5 * sigma * sigma) / (sigma * sigma)

    x1 = np.log(S / K) / sT + (1.0 + mu) * sT
    x2 = np.log(S / H) / sT + (1.0 + mu) * sT
    y1 = np.log(H * H / (S * K)) / sT + (1.0 + mu) * sT
    y2 = np.log(H / S) / sT + (1.0 + mu) * sT

    H_over_S = H / S
    pow1 = H_over_S ** (2.0 * (mu + 1.0))
    pow2 = H_over_S ** (2.0 * mu)

    disc_d = np.exp(-r_d * T)
    disc_carry = np.exp((b - r_d) * T)   # = e^(-r_f T) for FX

    A = (phi * S * disc_carry * norm_cdf(phi * x1)
          - phi * K * disc_d * norm_cdf(phi * x1 - phi * sT))
    B = (phi * S * disc_carry * norm_cdf(phi * x2)
          - phi * K * disc_d * norm_cdf(phi * x2 - phi * sT))
    C = (phi * S * disc_carry * pow1 * norm_cdf(eta * y1)
          - phi * K * disc_d * pow2 * norm_cdf(eta * y1 - eta * sT))
    D = (phi * S * disc_carry * pow1 * norm_cdf(eta * y2)
          - phi * K * disc_d * pow2 * norm_cdf(eta * y2 - eta * sT))

    return A, B, C, D


def _rebate_term(S: float, H: float, T: float, sigma: float,
                    r_d: float, r_f: float,
                    rebate: float, eta: int) -> float:
    """Rebate paid at HIT time (not expiry), Reiner-Rubinstein E5/E6.

    For an OUT option with eta = +1 (down) or -1 (up):
        F = K_reb × [(H/S)^(μ+λ) N(η z)
                       + (H/S)^(μ-λ) N(η z - 2η λ σ√T)]
    Setting rebate=0 makes this zero, which is the default for the
    standard FX KO product.
    """
    if rebate <= 0 or T <= 0:
        return 0.0
    b = r_d - r_f
    sT = sigma * np.sqrt(T)
    mu = (b - 0.5 * sigma * sigma) / (sigma * sigma)
    lam = np.sqrt(mu * mu + 2.0 * r_d / (sigma * sigma))
    z = np.log(H / S) / sT + lam * sT
    H_over_S = H / S
    F = rebate * (
        (H_over_S ** (mu + lam)) * norm_cdf(eta * z)
        + (H_over_S ** (mu - lam))
          * norm_cdf(eta * z - 2.0 * eta * lam * sT)
    )
    return float(F)


def ako_closed_form(option_type: str, barrier_type: str,
                        S: float, K: float, H: float, T: float, sigma: float,
                        r_d: float, r_f: float,
                        rebate: float = 0.0) -> float:
    """American-barrier KO price via Reiner-Rubinstein closed form.

    8 cases (4 option types × strike-above-or-below-barrier), per
    Haug's table:

        | type        | K vs H | formula              | always-dead? |
        |-------------|--------|----------------------|--------------|
        | down-out C  | K > H  | A − C + F            |              |
        | down-out C  | K ≤ H  | B − D + F            |              |
        | up-out   C  | K > H  | F                    | yes          |
        | up-out   C  | K ≤ H  | A − B + C − D + F    |              |
        | down-out P  | K > H  | A − B + C − D + F    |              |
        | down-out P  | K ≤ H  | F                    | yes          |
        | up-out   P  | K > H  | B − D + F            |              |
        | up-out   P  | K ≤ H  | A − C + F            |              |

    "always-dead" means: no path can give a positive payoff and avoid
    the barrier at the same time, so the option returns just the
    discounted rebate (zero if rebate=0).
    """
    # Degenerate edge cases (mirror core.ko.ko_price's behaviour)
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
    if sigma <= 0 or S <= 0 or K <= 0 or H <= 0:
        return 0.0
    if barrier_type == "up_and_out" and S >= H:
        return rebate
    if barrier_type == "down_and_out" and S <= H:
        return rebate

    is_call = (option_type == "call")
    is_up = (barrier_type == "up_and_out")
    phi = +1 if is_call else -1
    eta = -1 if is_up else +1

    F = _rebate_term(S, H, T, sigma, r_d, r_f, rebate, eta) if rebate > 0 else 0.0
    A, B, C, D = _rr_components(S, K, H, T, sigma, r_d, r_f, eta, phi)

    if is_call and not is_up:                       # down-and-out CALL
        price = (A - C + F) if K > H else (B - D + F)
    elif is_call and is_up:                         # up-and-out CALL
        price = F if K > H else (A - B + C - D + F)
    elif (not is_call) and not is_up:                # down-and-out PUT
        price = (A - B + C - D + F) if K > H else F
    else:                                             # up-and-out PUT
        price = (B - D + F) if K > H else (A - C + F)

    # Arbitrage floor — guard against tiny numerical leakage producing
    # negative prices in deep-OTM / near-barrier cases.
    return max(float(price), 0.0)


# =============================================================================
# Continuous-monitoring barrier-hit probability
# =============================================================================
def ako_probability_continuous(barrier_type: str,
                                     S: float, H: float,
                                     T: float, sigma: float,
                                     r_d: float, r_f: float) -> float:
    """Risk-neutral probability that the barrier is touched at LEAST
    ONCE during [0, T] under continuous monitoring.

    For a GBM with drift μ_log = (r_d - r_f - σ²/2):
        P(τ_H ≤ T) = N(-d₊) + (H/S)^(2μ_log/σ²) × N(-d₋)
      where
        d₊ = (ln(S/H) + μ_log T) / (σ√T)        [up barriers]
        d₋ = (ln(S/H) - μ_log T) / (σ√T)        [reflected term]
      with signs flipped for down barriers.

    This is materially HIGHER than the European-monitoring hit
    probability (`core.ko.ko_probability`), which only counts hits at
    expiry. For meaningful barriers the gap is often 2-4×.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or H <= 0:
        if barrier_type == "up_and_out":
            return 1.0 if S >= H else 0.0
        return 1.0 if S <= H else 0.0
    if barrier_type == "up_and_out" and S >= H:
        return 1.0
    if barrier_type == "down_and_out" and S <= H:
        return 1.0

    mu_log = r_d - r_f - 0.5 * sigma * sigma
    sT = sigma * np.sqrt(T)
    expo = 2.0 * mu_log / (sigma * sigma)
    H_over_S = H / S

    if barrier_type == "up_and_out":
        # P(max_t S_t ≥ H), H > S. Using ln(S/H) < 0 in d1, d2.
        # First-passage formula for drifted Brownian motion:
        #   P(τ_H ≤ T) = Φ(d1) + (H/S)^(2ν/σ²) Φ(d2)
        # where d1 = (ln(S/H) + ν T) / (σ√T)
        #       d2 = (ln(S/H) - ν T) / (σ√T)
        d1 = (np.log(S / H) + mu_log * T) / sT
        d2 = (np.log(S / H) - mu_log * T) / sT
        p = float(norm_cdf(d1) + (H_over_S ** expo) * norm_cdf(d2))
    else:
        # P(min_t S_t ≤ H), H < S. Mirror formulation:
        #   d1 = (ln(H/S) - ν T) / (σ√T)   (note: SIGN of νT flipped)
        #   d2 = (ln(H/S) + ν T) / (σ√T)
        d1 = (np.log(H / S) - mu_log * T) / sT
        d2 = (np.log(H / S) + mu_log * T) / sT
        p = float(norm_cdf(d1) + (H_over_S ** expo) * norm_cdf(d2))
    return max(0.0, min(1.0, p))


# =============================================================================
# Method 2 — Binomial (CRR) tree with BGK continuity correction
# =============================================================================
def ako_binomial(option_type: str, barrier_type: str,
                     S: float, K: float, H: float, T: float, sigma: float,
                     r_d: float, r_f: float,
                     n_steps: int = 1000,
                     bgk_correction: bool = True) -> float:
    """CRR binomial tree with Broadie-Glasserman-Kou (1997) continuity
    correction. With BGK on and N ≥ 500 the result is typically within
    0.5% of the closed-form for most non-pathological strikes/barriers.

    BGK shifts the EFFECTIVE barrier inward by exp(±0.5826 σ √Δt) so
    a discretely-monitored tree (which can only test at node times)
    converges to the continuously-monitored price. Without it, the
    discrete tree systematically OVERPRICES (the tree's barrier is
    "easier to avoid" than the continuous one).

    Args:
        n_steps: tree depth. 500-2000 is the practical range.
        bgk_correction: True (default) for accurate continuous-monitoring
            convergence. False for raw discrete-monitoring price
            (= once-per-day monitoring if N = trading days).
    """
    if T <= 0 or sigma <= 0 or n_steps < 1:
        return ako_closed_form(option_type, barrier_type, S, K, H, T,
                                   sigma, r_d, r_f)
    if barrier_type == "up_and_out" and S >= H:
        return 0.0
    if barrier_type == "down_and_out" and S <= H:
        return 0.0

    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    # Risk-neutral probability (FX: drift = r_d - r_f, discount = r_d)
    p = (np.exp((r_d - r_f) * dt) - d) / (u - d)
    if not (0.0 < p < 1.0):
        # Tree blew up (rare; very short T or extreme rates).
        # Fall back to closed-form.
        return ako_closed_form(option_type, barrier_type, S, K, H, T,
                                   sigma, r_d, r_f)
    disc = np.exp(-r_d * dt)

    # BGK barrier shift — move barrier INWARD (towards spot) so the
    # tree's discrete-fixing barrier acts like a continuous one.
    if bgk_correction:
        shift = np.exp(0.5826 * sigma * np.sqrt(dt))
        if barrier_type == "up_and_out":
            H_eff = H / shift     # shifted down (closer to spot)
        else:
            H_eff = H * shift     # shifted up   (closer to spot)
    else:
        H_eff = H

    # Build the terminal spot lattice (vectorised)
    j = np.arange(n_steps + 1)
    spots = S * (u ** (n_steps - j)) * (d ** j)
    # Terminal payoff
    if option_type == "call":
        values = np.maximum(spots - K, 0.0)
    else:
        values = np.maximum(K - spots, 0.0)
    # Apply barrier at terminal nodes
    if barrier_type == "up_and_out":
        values = np.where(spots >= H_eff, 0.0, values)
    else:
        values = np.where(spots <= H_eff, 0.0, values)

    # Backward induction with continuous-payoff-on-barrier check at every node
    for step in range(n_steps - 1, -1, -1):
        spots = spots[:-1] / u    # = S * u^(step-j) * d^j at this step
        values = disc * (p * values[:-1] + (1.0 - p) * values[1:])
        if barrier_type == "up_and_out":
            values = np.where(spots >= H_eff, 0.0, values)
        else:
            values = np.where(spots <= H_eff, 0.0, values)

    return max(float(values[0]), 0.0)


# =============================================================================
# Method 3 — Trinomial tree (Boyle) with barrier-node alignment
# =============================================================================
def ako_trinomial(option_type: str, barrier_type: str,
                      S: float, K: float, H: float, T: float, sigma: float,
                      r_d: float, r_f: float,
                      n_steps: int = 500) -> float:
    """Trinomial tree with Δx chosen so ln(H/S₀) is an integer multiple
    of Δx — the barrier lands exactly on a row of nodes, eliminating
    the dominant source of binomial discretization bias.

    Three branches at each node: up by Δx, flat, down by Δx. Risk-
    neutral probabilities are set to match the first two moments of
    log-spot under GBM:

        Δx² = σ² Δt + (νΔt)²,   where ν = r_d - r_f - σ²/2
        p_u = 0.5 × ((σ²Δt + (νΔt)²) / Δx²  +  νΔt / Δx)
        p_d = 0.5 × ((σ²Δt + (νΔt)²) / Δx²  −  νΔt / Δx)
        p_m = 1 - p_u - p_d

    To align the barrier on a node, Δx is set such that
        m := round(|ln(H/S₀)| / Δx_target)   is an integer ≥ 1
        Δx = |ln(H/S₀)| / m
    starting from a target Δx of σ√(3Δt) (Boyle's standard choice).
    """
    if T <= 0 or sigma <= 0 or n_steps < 1:
        return ako_closed_form(option_type, barrier_type, S, K, H, T,
                                   sigma, r_d, r_f)
    if barrier_type == "up_and_out" and S >= H:
        return 0.0
    if barrier_type == "down_and_out" and S <= H:
        return 0.0

    dt = T / n_steps
    nu = r_d - r_f - 0.5 * sigma * sigma
    sigma2_dt = sigma * sigma * dt
    nu_dt = nu * dt

    # Target Δx (Boyle's default)
    dx_target = sigma * np.sqrt(3.0 * dt)
    log_H_S = np.log(H / S)
    abs_log = abs(log_H_S)
    if abs_log < 1e-12:
        # Spot already on barrier — option is dead
        return 0.0
    m = max(1, int(round(abs_log / dx_target)))
    dx = abs_log / m
    # Index of the barrier row from spot (positive for up, negative for down)
    barrier_offset = m if log_H_S > 0 else -m

    # Probabilities (must satisfy 0 < p < 1)
    var = sigma2_dt + nu_dt * nu_dt
    p_u = 0.5 * (var / (dx * dx) + nu_dt / dx)
    p_d = 0.5 * (var / (dx * dx) - nu_dt / dx)
    p_m = 1.0 - p_u - p_d
    if not (0.0 < p_u < 1.0 and 0.0 < p_d < 1.0 and 0.0 < p_m < 1.0):
        # Probabilities invalid (rare; needs Δt tuning). Fall back.
        return ako_closed_form(option_type, barrier_type, S, K, H, T,
                                   sigma, r_d, r_f)
    disc = np.exp(-r_d * dt)

    # State space: log-spot offsets from log(S₀), in units of Δx.
    # At step n, possible offsets are -n .. +n (2n+1 nodes).
    # Build at terminal step, fold backward.
    j = np.arange(-n_steps, n_steps + 1)
    spots = S * np.exp(j * dx)

    # Terminal payoff
    if option_type == "call":
        values = np.maximum(spots - K, 0.0)
    else:
        values = np.maximum(K - spots, 0.0)
    # Barrier mask — exactly on the chosen row
    if barrier_type == "up_and_out":
        values = np.where(j >= barrier_offset, 0.0, values)
    else:
        values = np.where(j <= barrier_offset, 0.0, values)

    # Backward induction: at each step, output[k] = disc * (p_u * input[k+1]
    # + p_m * input[k] + p_d * input[k-1]).
    # The trinomial node array shrinks by 1 from each end per step backward.
    for step in range(n_steps - 1, -1, -1):
        # New indices range from -step to +step
        new_j = np.arange(-step, step + 1)
        # Old array has indices -step-1 .. +step+1 (2*step+3 entries)
        # We want to combine adjacent triples — slice [start_idx:end_idx]
        # of the old `values` to align with new_j.
        # Old index 0 corresponds to old_j = -(step+1), so old_j = k means
        # old array position k + step + 1.
        # New_j[i] = i - step, want old_j values = i-step-1, i-step, i-step+1
        # → old positions i, i+1, i+2 (i = 0..2*step).
        v_dn = values[0 : 2 * step + 1]
        v_md = values[1 : 2 * step + 2]
        v_up = values[2 : 2 * step + 3]
        values = disc * (p_u * v_up + p_m * v_md + p_d * v_dn)
        # Apply barrier at this step
        new_spots = S * np.exp(new_j * dx)
        if barrier_type == "up_and_out":
            values = np.where(new_j >= barrier_offset, 0.0, values)
        else:
            values = np.where(new_j <= barrier_offset, 0.0, values)

    return max(float(values[0]), 0.0)


# =============================================================================
# Method 4 — Crank-Nicolson finite difference on the BS PDE
# =============================================================================
def ako_finite_difference(option_type: str, barrier_type: str,
                                S: float, K: float, H: float, T: float,
                                sigma: float, r_d: float, r_f: float,
                                n_S: int = 300, n_t: int = 500) -> float:
    """Crank-Nicolson finite-difference solver for the Black-Scholes
    PDE with absorbing Dirichlet boundary at H:

        ∂V/∂t + (r_d - r_f) S ∂V/∂S + ½σ²S² ∂²V/∂S² − r_d V = 0

    Boundary conditions:
        - At expiry T:   V(S, T) = max(φ(S - K), 0),  φ = ±1
        - At barrier H:  V(H, t) = 0          ← absorbing (KO)
        - At far edge S → ∞ or S → 0:  Dirichlet from BS large-S limits
            for the non-barrier side

    Grid:
        - For up-and-out: S ∈ [S_min, H],  S_min near 0
        - For down-and-out: S ∈ [H, S_max], S_max far above spot
        n_S spatial points (uniform), n_t time steps.

    CN is unconditionally stable and second-order accurate in both
    time and space — ~10× more accurate per node than explicit FD.
    """
    if T <= 0 or sigma <= 0 or n_S < 10 or n_t < 1:
        return ako_closed_form(option_type, barrier_type, S, K, H, T,
                                   sigma, r_d, r_f)
    if barrier_type == "up_and_out" and S >= H:
        return 0.0
    if barrier_type == "down_and_out" and S <= H:
        return 0.0

    is_call = (option_type == "call")
    is_up = (barrier_type == "up_and_out")

    # Build the spatial grid: barrier on one boundary, "far edge" on the other
    if is_up:
        S_lo = max(1e-9, S * 0.001)
        S_hi = H
    else:
        S_lo = H
        S_hi = max(S * 3.0, K * 2.0, H * 2.0)
    S_grid = np.linspace(S_lo, S_hi, n_S + 1)
    dS = (S_hi - S_lo) / n_S
    dt = T / n_t

    # Terminal payoff
    if is_call:
        V = np.maximum(S_grid - K, 0.0)
    else:
        V = np.maximum(K - S_grid, 0.0)
    # Enforce barrier at terminal
    if is_up:
        V[-1] = 0.0
    else:
        V[0] = 0.0

    # Build CN coefficient vectors over interior nodes (i = 1 .. n_S - 1).
    # PDE in expanded form:
    #   ∂V/∂t = -½σ²Sᵢ² Vss - (r_d - r_f) Sᵢ Vs + r_d V
    # Centred FD:
    #   Vss_i = (V_{i+1} - 2V_i + V_{i-1}) / dS²
    #   Vs_i  = (V_{i+1} - V_{i-1}) / (2 dS)
    # Lump into V_{i-1}, V_i, V_{i+1} coefficients:
    i = np.arange(1, n_S)
    Si = S_grid[1:-1]
    alpha = 0.5 * sigma * sigma * Si * Si / (dS * dS)
    beta = 0.5 * (r_d - r_f) * Si / dS
    # ∂V/∂t = a_i V_{i-1} + b_i V_i + c_i V_{i+1}
    a = alpha - beta
    b = -2.0 * alpha - r_d
    c = alpha + beta

    # CN scheme: I - 0.5 dt L   on LHS,   I + 0.5 dt L   on RHS
    # where L is the spatial operator. Step BACKWARD in time (terminal → 0).
    lhs_l = -0.5 * dt * a
    lhs_d = 1.0 - 0.5 * dt * b
    lhs_u = -0.5 * dt * c
    rhs_l = 0.5 * dt * a
    rhs_d = 1.0 + 0.5 * dt * b
    rhs_u = 0.5 * dt * c

    # Tridiagonal solver (Thomas algorithm), pre-allocate scratch space
    cp = np.empty_like(lhs_u)
    dp = np.empty_like(lhs_d)

    for _ in range(n_t):
        # Build RHS = (I + 0.5 dt L) V_interior. Boundary contributions
        # come from V[0] and V[-1] which we enforce as Dirichlet below.
        rhs = (rhs_l * V[:-2] + rhs_d * V[1:-1] + rhs_u * V[2:])
        # Adjust RHS for known boundary values
        # Note: V[0] and V[-1] at this point are the values BEFORE the
        # step (terminal-side); we apply the SAME boundary values for
        # both sides since the problem is stationary-in-time on those.
        # For Dirichlet absorbing at barrier:
        #   V_barrier = 0  always
        # For far-edge: use unchanged terminal-style value (mild
        # approximation that's accurate when the far edge is set well
        # away from K and S; we can use the vanilla BS value here for
        # more accuracy, but for the KO this works since we put the
        # boundary far out).

        # Solve tridiagonal system lhs * V_new_interior = rhs
        n = len(rhs)
        cp[0] = lhs_u[0] / lhs_d[0]
        dp[0] = rhs[0] / lhs_d[0]
        for k in range(1, n):
            m = lhs_d[k] - lhs_l[k] * cp[k - 1]
            cp[k] = lhs_u[k] / m if k < n - 1 else 0.0
            dp[k] = (rhs[k] - lhs_l[k] * dp[k - 1]) / m

        V_new = np.empty_like(V)
        V_new[-1] = 0.0 if is_up else V[-1]
        V_new[0] = 0.0 if not is_up else V[0]
        V_new[n] = dp[n - 1]
        for k in range(n - 2, -1, -1):
            V_new[k + 1] = dp[k] - cp[k] * V_new[k + 2]

        V = V_new
        # Re-enforce barrier (Dirichlet)
        if is_up:
            V[-1] = 0.0
        else:
            V[0] = 0.0

    # Interpolate V at S₀
    return max(0.0, float(np.interp(S, S_grid, V)))


# =============================================================================
# Spot delta via central finite difference on the closed-form price
# =============================================================================
def ako_spot_delta(option_type: str, barrier_type: str,
                       S: float, K: float, H: float, T: float, sigma: float,
                       r_d: float, r_f: float,
                       eps_rel: float = 1e-5) -> float:
    """Spot delta of an American-barrier KO via central finite
    difference on the closed-form pricer. Matches the convention used
    in core.ko.ko_spot_delta (the European-barrier counterpart)."""
    eps = max(S * eps_rel, 1e-9)
    p_up = ako_closed_form(option_type, barrier_type, S + eps, K, H, T,
                                 sigma, r_d, r_f)
    p_dn = ako_closed_form(option_type, barrier_type, S - eps, K, H, T,
                                 sigma, r_d, r_f)
    return (p_up - p_dn) / (2.0 * eps)

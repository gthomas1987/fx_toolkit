"""Portfolio pricing engine.

Given a market snapshot and a list of trades, this module:
    - Marks each trade to market
    - Computes Greeks (Δ, Γ, V, vanna, volga, charm, θ, ρ via bumping)
    - Computes barrier proximity / survival probabilities
    - Computes correlation Greek (cega) for dual structures
    - Builds the risk cube (spot × vol grid scenario MTM)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import pandas as pd

from .vanilla import (vanilla_price, vanilla_delta, vanilla_gamma, vanilla_vega,
                       vanilla_vanna, vanilla_volga, vanilla_charm, vanilla_theta)
from .eko import eko_price, survival_prob
from .dual_eko import Leg as DualLeg, dual_eko_price
from .portfolio import Trade, TradeLeg, infer_rates
from .data_loader import interp_vol


# =============================================================================
# Core single-trade pricing
# =============================================================================

def _price_single_pair(pair_snap: dict, S: float, K: float, T: float, sigma: float,
                       r_d: float, r_f: float, H: float | None,
                       bar_dir: str | None, opt: str) -> float:
    """Price one leg in DOM per FOR unit."""
    if T <= 0 or sigma <= 0:
        # at-or-past expiry: intrinsic with barrier check
        if H is not None and bar_dir == "up_and_out" and S >= H:
            return 0.0
        if H is not None and bar_dir == "down_and_out" and S <= H:
            return 0.0
        if opt == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)
    if H is None or bar_dir is None:
        return vanilla_price(opt, S, K, T, sigma, r_d, r_f)
    return eko_price(opt, bar_dir, S, K, H, T, sigma, r_d, r_f)


def _usd_pnl_scaler(pair: str, S: float, notional_usd: float) -> float:
    """Convert DOM-per-FOR price into USD P&L for the given USD notional.

    For XXX/USD: DOM=USD, FOR=XXX. Multiply (DOM per FOR) by FOR notional.
    FOR notional = notional_usd / S (rough — assumes price is in USD per
    1 unit of XXX; price * (USD_notional/S) gives USD).
    But here notional_usd represents the USD face — the FOR face is then
    notional_usd / S since 1 XXX = S USD.

    For USD/XXX: DOM=XXX, FOR=USD. Multiply (XXX per USD) by USD notional
    = XXX P&L. Convert to USD by dividing by S (spot XXX per USD).
    """
    if len(pair) == 6 and pair[3:] == "USD":
        # XXX/USD pair → DOM per FOR = USD per XXX
        # USD P&L = (USD per XXX) * (FOR units) = price * (notional_usd / S)
        return notional_usd / S
    # USD/XXX pair → DOM per FOR = XXX per USD
    # XXX P&L = price * notional_usd ; USD P&L = (XXX P&L) / S
    return notional_usd / S


def price_trade(trade: Trade, snap: dict, asof: pd.Timestamp,
                bump: dict | None = None) -> dict:
    """Compute MTM and barrier info for one trade against the snapshot.

    `bump` is an optional dict of perturbations applied to the snapshot for
    Greek calculation:
        {'spot': {pair: delta_S}, 'vol': {pair: delta_sigma}, 'rho': delta_rho, 'dt': delta_T_years}
    """
    bump = bump or {}
    T_yrs = trade.days_to_expiry(asof) / 365.0
    T_yrs = T_yrs + bump.get("dt", 0.0)

    sign = +1.0 if trade.side == "buy" else -1.0

    # --- Branch by structure ---
    if trade.structure in ("vanilla", "call_spread", "call_fly", "eko"):
        sp = snap[trade.pair]
        S = sp["spot"] + bump.get("spot", {}).get(trade.pair, 0.0)
        # ATM vol bump applies uniformly to surface; we read interp_vol with bump applied
        vol_bump = bump.get("vol", {}).get(trade.pair, 0.0)
        r_d, r_f = infer_rates(trade.pair, sp, T_yrs)

        total_dom = 0.0
        total_intrinsic = 0.0
        legs_info = []
        for leg in trade.legs:
            sigma = interp_vol(sp, T_yrs, K=leg.K, S=S) + vol_bump
            sigma = max(sigma, 0.005)
            px = _price_single_pair(sp, S, leg.K, T_yrs, sigma, r_d, r_f,
                                     leg.H, leg.bar_dir, leg.opt)
            total_dom += leg.qty * px
            intr = max(S - leg.K, 0) if leg.opt == "call" else max(leg.K - S, 0)
            total_intrinsic += leg.qty * intr
            legs_info.append({"opt": leg.opt, "K": leg.K, "H": leg.H,
                              "bar_dir": leg.bar_dir, "qty": leg.qty,
                              "sigma": sigma, "px_dom": px})

        scaler = _usd_pnl_scaler(trade.pair, S, trade.notional_usd)
        mtm_usd = sign * total_dom * scaler
        intrinsic_usd = sign * total_intrinsic * scaler

        return {
            "mtm_usd": mtm_usd,
            "intrinsic_usd": intrinsic_usd,
            "T_yrs": T_yrs,
            "S": S,
            "r_d": r_d, "r_f": r_f,
            "legs_info": legs_info,
            "is_dual": False,
        }

    if trade.structure == "dual_eko":
        sp1 = snap[trade.pair]
        sp2 = snap[trade.pair2]
        S1 = sp1["spot"] + bump.get("spot", {}).get(trade.pair, 0.0)
        S2 = sp2["spot"] + bump.get("spot", {}).get(trade.pair2, 0.0)
        v1b = bump.get("vol", {}).get(trade.pair, 0.0)
        v2b = bump.get("vol", {}).get(trade.pair2, 0.0)

        r_d1, r_f1 = infer_rates(trade.pair, sp1, T_yrs)
        r_d2, r_f2 = infer_rates(trade.pair2, sp2, T_yrs)

        # one leg per pair (current portfolio convention)
        l1 = trade.legs[0]
        l2 = trade.legs2[0]
        sig1 = interp_vol(sp1, T_yrs, K=l1.K, S=S1) + v1b
        sig2 = interp_vol(sp2, T_yrs, K=l2.K, S=S2) + v2b

        rho = (trade.rho_traded or 0.5) + bump.get("rho", 0.0)
        rho = float(np.clip(rho, -0.99, 0.99))

        leg1 = DualLeg(S=S1, K=l1.K, H=(l1.H if l1.H else 1e9), sigma=max(sig1, 0.005),
                       r_d=r_d1, r_f=r_f1, bar_dir=(l1.bar_dir or "none"), opt=l1.opt)
        leg2 = DualLeg(S=S2, K=l2.K, H=(l2.H if l2.H else 1e9), sigma=max(sig2, 0.005),
                       r_d=r_d2, r_f=r_f2, bar_dir=(l2.bar_dir or "none"), opt=l2.opt)
        # for "wo_put" / similar, set H high values for puts barrier check
        # (we use the actual barrier; barrier types are set in TradeLeg)

        res = dual_eko_price(leg1, leg2, trade.structure_kind, T_yrs, rho,
                              n_paths=40_000)
        # We report the MC price in DOM units of leg1 per 1 FOR unit. Scale
        # to USD via leg1's pair scaler.
        scaler = _usd_pnl_scaler(trade.pair, S1, trade.notional_usd)
        mtm_usd = +1.0 * (1.0 if trade.side == "buy" else -1.0) * res["price_per_pair_unit"] * scaler

        # intrinsic
        if trade.structure_kind == "wo_call":
            p1 = max(S1 - l1.K, 0); p2 = max(S2 - l2.K, 0)
            intr = min(p1, p2)
        elif trade.structure_kind == "wo_put":
            p1 = max(l1.K - S1, 0); p2 = max(l2.K - S2, 0)
            intr = min(p1, p2)
        elif trade.structure_kind == "bo_call":
            p1 = max(S1 - l1.K, 0); p2 = max(S2 - l2.K, 0)
            intr = max(p1, p2)
        else:
            p1 = max(l1.K - S1, 0); p2 = max(l2.K - S2, 0)
            intr = max(p1, p2)
        intrinsic_usd = (1.0 if trade.side == "buy" else -1.0) * intr * scaler

        return {
            "mtm_usd": mtm_usd,
            "intrinsic_usd": intrinsic_usd,
            "T_yrs": T_yrs,
            "S1": S1, "S2": S2,
            "sigma1": sig1, "sigma2": sig2,
            "rho_used": rho,
            "p_alive": res["p_alive"],
            "p_alive_leg1": res["p_alive_leg1"],
            "p_alive_leg2": res["p_alive_leg2"],
            "is_dual": True,
        }

    raise ValueError(f"Unknown structure: {trade.structure}")


# =============================================================================
# Greeks via bumping
# =============================================================================

def compute_greeks(trade: Trade, snap: dict, asof: pd.Timestamp,
                   ds_pct: float = 0.005, dv: float = 0.005,
                   drho: float = 0.05, dt_days: float = 1.0) -> dict:
    """Finite-difference Greeks. Returns USD per relevant unit.

    Δ:    USD P&L for a 1% spot move (per pair for duals)
    Γ:    USD P&L from a 1% × 1% spot move (curvature)
    V:    USD P&L for a 1 vol-point move (per pair for duals)
    Vanna: USD P&L from a 1% spot × 1vol move
    Volga: USD P&L from a 1vol × 1vol move (curvature in vol)
    Charm: USD P&L from a 1-day passage (decay of delta)
    Cega: USD P&L for a +0.05 correlation move (duals only)
    Theta: USD P&L for 1-day passage
    """
    if trade.structure == "dual_eko":
        pairs = [trade.pair, trade.pair2]
    else:
        pairs = [trade.pair]

    base = price_trade(trade, snap, asof)
    base_mtm = base["mtm_usd"]

    g = {"base_mtm": base_mtm, "by_pair": {}}

    for p in pairs:
        S = snap[p]["spot"]
        dS = S * ds_pct
        up = price_trade(trade, snap, asof, bump={"spot": {p: +dS}})["mtm_usd"]
        dn = price_trade(trade, snap, asof, bump={"spot": {p: -dS}})["mtm_usd"]
        delta = (up - dn) / (2 * dS) * S * 0.01  # USD per 1% spot move
        gamma = (up + dn - 2 * base_mtm) / (dS ** 2) * (S * 0.01) ** 2  # per (1% spot)^2

        vu = price_trade(trade, snap, asof, bump={"vol": {p: +dv}})["mtm_usd"]
        vd = price_trade(trade, snap, asof, bump={"vol": {p: -dv}})["mtm_usd"]
        vega = (vu - vd) / (2 * dv) * 0.01  # USD per 1 vol pt (0.01 in σ)
        volga = (vu + vd - 2 * base_mtm) / (dv ** 2) * (0.01) ** 2  # per (1vol)^2

        # Vanna: cross-bump
        v_up_s_up = price_trade(trade, snap, asof,
                                bump={"spot": {p: +dS}, "vol": {p: +dv}})["mtm_usd"]
        v_up_s_dn = price_trade(trade, snap, asof,
                                bump={"spot": {p: -dS}, "vol": {p: +dv}})["mtm_usd"]
        v_dn_s_up = price_trade(trade, snap, asof,
                                bump={"spot": {p: +dS}, "vol": {p: -dv}})["mtm_usd"]
        v_dn_s_dn = price_trade(trade, snap, asof,
                                bump={"spot": {p: -dS}, "vol": {p: -dv}})["mtm_usd"]
        vanna = ((v_up_s_up - v_up_s_dn) - (v_dn_s_up - v_dn_s_dn)) \
                / (4 * dS * dv) * (S * 0.01) * 0.01  # per (1%spot × 1vol)

        # Charm: spot delta change over 1 day
        t_bump = -dt_days / 365.0  # time passes ⇒ T decreases
        up_t = price_trade(trade, snap, asof,
                           bump={"spot": {p: +dS}, "dt": t_bump})["mtm_usd"]
        dn_t = price_trade(trade, snap, asof,
                           bump={"spot": {p: -dS}, "dt": t_bump})["mtm_usd"]
        delta_t1 = (up_t - dn_t) / (2 * dS) * S * 0.01
        charm = delta_t1 - delta  # change in delta per day

        g["by_pair"][p] = {
            "delta_usd_per_1pct": delta,
            "gamma_usd_per_1pct2": gamma,
            "vega_usd_per_volpt": vega,
            "volga_usd_per_volpt2": volga,
            "vanna_usd_per_1pct_x_volpt": vanna,
            "charm_usd_per_day": charm,
        }

    # Theta (time decay)
    t_only = price_trade(trade, snap, asof, bump={"dt": -dt_days / 365.0})["mtm_usd"]
    g["theta_usd_per_day"] = t_only - base_mtm

    # Cega (correlation Greek)
    if trade.structure == "dual_eko":
        ru = price_trade(trade, snap, asof, bump={"rho": +drho})["mtm_usd"]
        rd = price_trade(trade, snap, asof, bump={"rho": -drho})["mtm_usd"]
        g["cega_usd_per_5pct_rho"] = (ru - rd) / 2  # USD per +0.05 rho

    return g


# =============================================================================
# Scenario risk cube
# =============================================================================

def risk_cube(trade: Trade, snap: dict, asof: pd.Timestamp,
              spot_shocks: list[float], vol_shocks: list[float]) -> pd.DataFrame:
    """USD MTM under (spot_pct, vol_pt) shocks. Shocks applied to all involved pairs."""
    pairs = [trade.pair] + ([trade.pair2] if trade.structure == "dual_eko" else [])
    rows = []
    for vs in vol_shocks:
        row = {}
        for ss in spot_shocks:
            bump = {"spot": {p: snap[p]["spot"] * ss for p in pairs},
                    "vol": {p: vs for p in pairs}}
            mtm = price_trade(trade, snap, asof, bump=bump)["mtm_usd"]
            row[f"{ss*100:+.1f}%"] = mtm
        rows.append(row)
    df = pd.DataFrame(rows, index=[f"{vs*100:+.1f}vol" for vs in vol_shocks])
    return df


def portfolio_risk_cube(trades: list[Trade], snap: dict, asof: pd.Timestamp,
                         spot_shocks: list[float], vol_shocks: list[float]) -> pd.DataFrame:
    """Aggregate cube across the portfolio (independently shocking each pair).

    For simplicity, applies the same shock to every pair simultaneously
    (a market-wide stress test). Use per-pair cube for granular work.
    """
    rows = []
    for vs in vol_shocks:
        row = {}
        for ss in spot_shocks:
            all_pairs = set()
            for t in trades:
                all_pairs.add(t.pair)
                if t.structure == "dual_eko":
                    all_pairs.add(t.pair2)
            bump = {"spot": {p: snap[p]["spot"] * ss for p in all_pairs},
                    "vol": {p: vs for p in all_pairs}}
            total = sum(price_trade(t, snap, asof, bump=bump)["mtm_usd"]
                        for t in trades)
            row[f"{ss*100:+.1f}%"] = total
        rows.append(row)
    df = pd.DataFrame(rows, index=[f"{vs*100:+.1f}vol" for vs in vol_shocks])
    return df


# =============================================================================
# Barrier diagnostics
# =============================================================================

def barrier_diagnostics(trade: Trade, snap: dict, asof: pd.Timestamp) -> list[dict]:
    """For each barrier leg in the trade, compute distance + survival.

    Returns:
        list of dicts with keys:
          pair, S, K, H, bar_dir, distance_abs, distance_pct,
          distance_vol_sd, survival_prob, T_yrs, leg_index
    """
    out = []
    T_yrs = trade.days_to_expiry(asof) / 365.0
    if T_yrs <= 0:
        return out

    if trade.structure == "eko":
        sp = snap[trade.pair]
        S = sp["spot"]
        r_d, r_f = infer_rates(trade.pair, sp, T_yrs)
        for i, leg in enumerate(trade.legs):
            if leg.H is None:
                continue
            sigma = interp_vol(sp, T_yrs, K=leg.K, S=S)
            d_abs = leg.H - S
            d_pct = d_abs / S
            sd = sigma * np.sqrt(T_yrs)
            d_sd = abs(np.log(leg.H / S)) / max(sd, 1e-9)
            out.append({
                "pair": trade.pair, "S": S, "K": leg.K, "H": leg.H,
                "bar_dir": leg.bar_dir, "distance_abs": d_abs,
                "distance_pct": d_pct, "distance_vol_sd": d_sd,
                "survival_prob": survival_prob(leg.bar_dir, S, leg.H, T_yrs,
                                                sigma, r_d, r_f),
                "T_yrs": T_yrs, "sigma": sigma, "leg_index": i,
            })
    elif trade.structure == "dual_eko":
        for pair_key, leg, legs_list in [(trade.pair, trade.legs[0], trade.legs),
                                          (trade.pair2, trade.legs2[0], trade.legs2)]:
            sp = snap[pair_key]
            S = sp["spot"]
            r_d, r_f = infer_rates(pair_key, sp, T_yrs)
            if leg.H is None:
                continue
            sigma = interp_vol(sp, T_yrs, K=leg.K, S=S)
            d_abs = leg.H - S
            d_pct = d_abs / S
            sd = sigma * np.sqrt(T_yrs)
            d_sd = abs(np.log(leg.H / S)) / max(sd, 1e-9)
            out.append({
                "pair": pair_key, "S": S, "K": leg.K, "H": leg.H,
                "bar_dir": leg.bar_dir, "distance_abs": d_abs,
                "distance_pct": d_pct, "distance_vol_sd": d_sd,
                "survival_prob": survival_prob(leg.bar_dir, S, leg.H, T_yrs,
                                                sigma, r_d, r_f),
                "T_yrs": T_yrs, "sigma": sigma, "leg_index": 0,
            })
    return out

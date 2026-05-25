"""Vanilla-option backtest harness.

A simple daily-rolling backtester for vanilla FX options that mirrors
the structure of `core.backtest.run_single_strategy` but for vanilla
calls / puts rather than EKOs.

# Why a separate module
The existing `core.backtest.run_single_strategy` is barrier-focused:
its `StrategySpec` requires a `barrier_type` and `payout_ratio` /
`target_ko_delta`. To avoid bending that interface to support
"barrierless" trades, we provide a parallel `VanillaSpec` and
`run_vanilla_strategy` here.

The return type — `list[VanillaTrade]` — is structurally compatible
with `Trade` in the key fields needed for portfolio aggregation
(`trade_date`, `expiry_date`, `pnl_usd`, `premium_usd`, `pair`,
`direction`, `tenor_label`, `notional_usd`). Tests use the same
column names where they overlap so downstream code that reads ledger
DataFrames (e.g. portfolio summaries) works on both.

# What it computes per trade
Daily loop over business days where SPOT data exists:
  1. Resolve K (ATM = forward, or strike-from-delta via Garman-Kohlhagen)
  2. Get σ_smile(K) via core.smile.smile_vol_at_strike
  3. Compute premium at σ_smile + tx_cost in bps
  4. At expiry, compute realized payoff from terminal spot
  5. P&L = realized_payoff - premium_paid

Calendar: uses `core.calendar_fx.compute_option_dates_for_pair` for
holiday-aware expiry dates.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date
from typing import Optional, Callable

import numpy as np
import pandas as pd

from core.calendar_fx import compute_option_dates_for_pair
from core.smile import smile_vol_at_strike
from core.vanilla import (
    vanilla_price, atm_forward_strike, strike_from_delta,
)


# =============================================================================
# Spec
# =============================================================================
@dataclass
class VanillaSpec:
    """Single-line specification for a vanilla backtest.

    Mirrors the relevant fields of `core.backtest.StrategySpec` but
    without barrier-specific fields.
    """
    pair: str
    direction: str            # 'call' | 'put'
    delta_label: str          # 'ATM', '25Δ', etc.
    delta_value: float        # 0.0 for ATM, 0.25 for 25Δ
    tenor_label: str          # '1M', '6W', etc.
    tx_cost_bps: float = 4.0
    prefer: str = "offshore"
    pricing_model: str = "vol_at_strike"  # for smile vol; matches StrategySpec

    @property
    def name(self) -> str:
        return (f"{self.pair} {self.direction.upper()}  "
                 f"{self.delta_label}  {self.tenor_label}  VANILLA")

    @property
    def short_name(self) -> str:
        return (f"{self.pair}_{self.direction}_{self.delta_label}_"
                 f"{self.tenor_label}_VAN")


# =============================================================================
# Trade record
# =============================================================================
@dataclass
class VanillaTrade:
    """Trade ledger row for a vanilla. Field names match `Trade` where
    they overlap (so aggregation code can be shared).
    """
    strategy_name: str
    pair: str
    direction: str
    delta_label: str
    tenor_label: str
    tx_cost_bps: float

    trade_date: date
    spot_settlement: date
    option_settlement: date
    expiry_date: date
    T_years: float

    spot: float
    sigma_atm: float
    rr_25: float
    bf_25: float
    sigma_smile: float
    r_d: float
    r_f: float
    fwd_market: float

    strike: float
    feasible: bool

    premium_pct: float
    premium_mid_pct: float
    transaction_cost_pct: float
    max_payoff_pct: float        # vanilla payoff is unbounded — kept for API parity

    spot_at_expiry: Optional[float]
    actual_payoff_pct: float
    pnl_pct: float
    pnl_gross_pct: float

    notional_usd: float
    premium_usd: float
    premium_mid_usd: float
    transaction_cost_usd: float
    max_payoff_usd: float
    actual_payoff_usd: float
    pnl_usd: float
    pnl_gross_usd: float

    pricing_model: str = "vol_at_strike"
    # Parity with `Trade.knocked_out` / `Trade.barrier_type` etc. so
    # aggregation code doesn't crash on missing columns. Vanillas have
    # no barrier, no KO concept — these are always populated as null.
    barrier_type: str = "none"
    knocked_out: Optional[bool] = False


def vanilla_trades_to_df(trades: "list[VanillaTrade]") -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([asdict(t) for t in trades])


# =============================================================================
# Single-strategy backtest
# =============================================================================
def run_vanilla_strategy(spec: VanillaSpec, panels: dict,
                            start_date: date, end_date: date,
                            notional_usd: float = 10_000_000.0,
                            progress_cb: Optional[Callable[[float], None]] = None,
                            ) -> "list[VanillaTrade]":
    """Run the daily-rolling vanilla backtest for `spec`.

    Uses preloaded pair panels (same format as `preload_pair_panels`).
    Returns a list of VanillaTrade records — one per business day in
    [start_date, end_date] where SPOT data is available.
    """
    spot = panels["spot"]
    vol_panels = panels["vol_panels"]
    fwd_panels = panels["fwd_panels"]
    rr_panels = panels.get("rr_panels", {})
    bf_panels = panels.get("bf_panels", {})
    f_panel = panels["f_panel"]
    d_panel = panels["d_panel"]
    pip = panels["pip_scale"]

    spot_dates = pd.DatetimeIndex(spot.index).normalize()
    in_range = spot_dates[(spot_dates >= pd.Timestamp(start_date))
                            & (spot_dates <= pd.Timestamp(end_date))]
    trade_dates = sorted(set(d.date() for d in in_range))
    if not trade_dates:
        return []

    # Foreign / domestic mapping
    foreign, domestic = spec.pair[:3], spec.pair[3:]

    # Build a fast lookup for spot
    spot_idx = pd.DatetimeIndex(spot.index).normalize()
    spot_arr = spot.values

    def _spot_asof(ts: pd.Timestamp) -> Optional[float]:
        pos = spot_idx.searchsorted(ts, side="right") - 1
        if pos < 0:
            return None
        v = spot_arr[pos]
        return float(v) if pd.notna(v) else None

    trades: list[VanillaTrade] = []
    n = len(trade_dates)
    for i, td in enumerate(trade_dates):
        if progress_cb is not None and i % 50 == 0:
            progress_cb(i / max(n, 1))
        ts = pd.Timestamp(td)
        S = _spot_asof(ts)
        if S is None:
            continue

        # Option dates with holiday-aware calendar
        try:
            od = compute_option_dates_for_pair(td, spec.tenor_label, spec.pair)
        except Exception:
            continue
        T = od.T_years
        if T <= 0:
            continue

        # Vol panels for THIS tenor
        vol_p = vol_panels.get(spec.tenor_label)
        if vol_p is None:
            continue
        sigma_atm_pct = vol_p.asof(ts)
        if pd.isna(sigma_atm_pct):
            continue
        sigma_atm = float(sigma_atm_pct) / 100.0

        rr_25 = bf_25 = 0.0
        rr_p = rr_panels.get(spec.tenor_label)
        bf_p = bf_panels.get(spec.tenor_label)
        if rr_p is not None and bf_p is not None:
            rr_v = rr_p.asof(ts)
            bf_v = bf_p.asof(ts)
            if pd.notna(rr_v) and pd.notna(bf_v):
                rr_25 = float(rr_v) / 100.0
                bf_25 = float(bf_v) / 100.0

        fwd_p = fwd_panels.get(spec.tenor_label)
        fwd_v = fwd_p.asof(ts) if fwd_p is not None else None
        F_market = S
        if fwd_v is not None and pd.notna(fwd_v):
            F_market = S + float(fwd_v) * pip

        # Rates at T
        from core.rates import get_rate_at
        r_f = get_rate_at(f_panel, T, td)
        r_d = get_rate_at(d_panel, T, td)
        if r_d is None and r_f is None:
            continue
        if r_d is None:
            r_d = r_f + np.log(F_market / S) / T
        if r_f is None:
            r_f = r_d - np.log(F_market / S) / T

        # Resolve strike
        if spec.delta_value == 0.0:
            # ATM = forward
            K = atm_forward_strike(S, T, r_d, r_f)
        else:
            K = strike_from_delta(spec.direction, spec.delta_value,
                                       S, T, sigma_atm, r_d, r_f)

        sigma_smile = smile_vol_at_strike(S, K, T, sigma_atm, rr_25, bf_25,
                                                r_d, r_f)

        # Premium
        p_per_unit = vanilla_price(spec.direction, S, K, T, sigma_smile,
                                         r_d, r_f)
        premium_mid_pct = p_per_unit / S * 100.0
        tx_pct = spec.tx_cost_bps / 100.0
        premium_pct = premium_mid_pct + tx_pct

        # Realized payoff at expiry
        expiry_ts = pd.Timestamp(od.option_expiry)
        S_T = _spot_asof(expiry_ts)
        actual_payoff_pct = 0.0
        pnl_pct = -premium_pct
        pnl_gross_pct = -premium_mid_pct
        if S_T is not None:
            if spec.direction == "call":
                payoff_per_unit = max(S_T - K, 0.0)
            else:
                payoff_per_unit = max(K - S_T, 0.0)
            actual_payoff_pct = payoff_per_unit / S * 100.0
            pnl_pct = actual_payoff_pct - premium_pct
            pnl_gross_pct = actual_payoff_pct - premium_mid_pct

        max_payoff_pct = 0.0   # vanilla is unbounded; conventional sentinel

        trade = VanillaTrade(
            strategy_name=spec.name,
            pair=spec.pair, direction=spec.direction,
            delta_label=spec.delta_label, tenor_label=spec.tenor_label,
            tx_cost_bps=spec.tx_cost_bps,
            trade_date=td,
            spot_settlement=od.spot_settlement,
            option_settlement=od.option_settlement,
            expiry_date=od.option_expiry,
            T_years=T,
            spot=S, sigma_atm=sigma_atm, rr_25=rr_25, bf_25=bf_25,
            sigma_smile=sigma_smile, r_d=r_d, r_f=r_f,
            fwd_market=F_market,
            strike=K, feasible=True,
            premium_pct=premium_pct, premium_mid_pct=premium_mid_pct,
            transaction_cost_pct=tx_pct,
            max_payoff_pct=max_payoff_pct,
            spot_at_expiry=S_T,
            actual_payoff_pct=actual_payoff_pct,
            pnl_pct=pnl_pct, pnl_gross_pct=pnl_gross_pct,
            notional_usd=notional_usd,
            premium_usd=premium_pct / 100.0 * notional_usd,
            premium_mid_usd=premium_mid_pct / 100.0 * notional_usd,
            transaction_cost_usd=tx_pct / 100.0 * notional_usd,
            max_payoff_usd=0.0,
            actual_payoff_usd=actual_payoff_pct / 100.0 * notional_usd,
            pnl_usd=pnl_pct / 100.0 * notional_usd,
            pnl_gross_usd=pnl_gross_pct / 100.0 * notional_usd,
            pricing_model=spec.pricing_model,
        )
        trades.append(trade)

    return trades

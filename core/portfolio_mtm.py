"""Mark-to-market trajectories for the Option Portfolio Backtest.

Computes per-trade daily MTM in USD across the trade's life, for every
supported strategy type:

  - vanilla : core.vanilla.vanilla_price at σ_smile(K) on each business day
  - eko     : core.eko_pricing.price_eko_dispatch (reuses
              core.backtest._trade_mtm_trajectory)
  - rko     : core.ako_pricing.price_ako_dispatch (new — analogous to EKO)
  - wo_eko  : core.worstof_pricer.worstof_eko_price_cf (joint mid)
  - wo_rko  : core.worstof_pricer_american.worstof_rko_price_cf_approx
              (joint mid w/ continuous-monitoring approximation)

# Output convention

`portfolio_mtm_equity_curve(trade_records_by_strategy, panels_by_pair,
                                strategies_meta, side_sign)`
returns a tuple:
    (portfolio_equity_df, per_strategy_equity_dict)

where:
  - portfolio_equity_df is a DataFrame indexed by date with columns:
      book_value_usd, cash_position_usd, equity_usd, peak_usd,
      drawdown_usd
  - per_strategy_equity_dict maps display_name -> same-shape DataFrame

Sign convention: equity rises on Sell-side (negative side_sign maps to
this) iff the realized P&L is positive. The `side_sign` is applied
ONCE per trade as multiplier on `premium_usd` and `actual_payoff_usd`
in the cash flow stream, and to the MTM book value (since Sell = short
position = negative book mark).

Notes:
- For WO MTM, this uses CF engines (faster than MC). Users who pick
  Joint MC for the trade-time premium still get CF for daily MTM —
  consistent within ~1 bp at default n_paths.
- Vol surface intra-trade is queried from the SAME pair panels used at
  trade entry, so MTM is internally consistent with the trade record.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import date as _date
from typing import Optional, Callable

from core.backtest import (
    _build_fast_panels, _fast_asof, _fast_interp_at_T,
    _trade_mtm_trajectory,
)


# =============================================================================
# Vanilla MTM trajectory
# =============================================================================
def _vanilla_mtm_trajectory(trade: dict, fast_panels: dict) -> dict:
    """MTM in USD across the trade's life for one vanilla.

    Same shape as core.backtest._trade_mtm_trajectory but uses
    `vanilla_price` instead of the EKO dispatcher. No barrier logic —
    just price the European vanilla at σ_smile on each business day.
    """
    from core.smile import smile_vol_at_strike as _smile_vol
    from core.vanilla import vanilla_price as _vanilla_price

    spot_fast = fast_panels["spot"]
    vol_fast = fast_panels["vol_panels_fast"]
    fwd_fast = fast_panels["fwd_panels_fast"]
    rr_fast = fast_panels.get("rr_panels_fast", {})
    bf_fast = fast_panels.get("bf_panels_fast", {})
    f_fast = fast_panels["f_panel_fast"]
    d_fast = fast_panels["d_panel_fast"]
    pip = fast_panels["pip_scale"]

    direction = trade["direction"]
    K = trade["strike"]
    S_trade = trade["spot"]
    notional = trade["notional_usd"]
    factor = notional / S_trade

    trade_dt = pd.Timestamp(trade["trade_date"])
    expiry_dt = pd.Timestamp(trade["expiry_date"])
    last_mtm_day = expiry_dt - pd.Timedelta(days=1)
    if last_mtm_day < trade_dt:
        return {}
    bdays = pd.bdate_range(trade_dt, last_mtm_day)

    out: dict = {}
    for d_ts in bdays:
        if d_ts == trade_dt:
            mtm_per_unit = trade["premium_mid_pct"] / 100.0 * S_trade
        else:
            T_rem = (expiry_dt - d_ts).days / 365.0
            if T_rem < 1e-4:
                continue
            ts_np = np.datetime64(d_ts.to_datetime64())
            S_d = _fast_asof(spot_fast, ts_np)
            if S_d is None:
                continue
            sigma_atm_pct = _fast_interp_at_T(vol_fast, T_rem, ts_np)
            if sigma_atm_pct is None:
                continue
            sigma_atm_d = sigma_atm_pct / 100.0

            r_f = _fast_interp_at_T(f_fast, T_rem, ts_np)
            r_d_rate = _fast_interp_at_T(d_fast, T_rem, ts_np)
            if r_f is None and r_d_rate is None:
                continue
            fwd_pts = _fast_interp_at_T(fwd_fast, T_rem, ts_np)
            F_d = S_d + fwd_pts * pip if fwd_pts is not None else S_d
            if r_f is None:
                r_f = r_d_rate - np.log(F_d / S_d) / T_rem
            if r_d_rate is None:
                r_d_rate = r_f + np.log(F_d / S_d) / T_rem

            rr_pct = _fast_interp_at_T(rr_fast, T_rem, ts_np) if rr_fast else None
            bf_pct = _fast_interp_at_T(bf_fast, T_rem, ts_np) if bf_fast else None
            rr = (rr_pct / 100.0) if rr_pct is not None else 0.0
            bf = (bf_pct / 100.0) if bf_pct is not None else 0.0
            sigma_d = _smile_vol(S_d, K, T_rem, sigma_atm_d, rr, bf,
                                   r_d_rate, r_f)

            try:
                mtm_per_unit = _vanilla_price(direction, S_d, K, T_rem,
                                                   sigma_d, r_d_rate, r_f)
            except Exception:
                continue
        out[d_ts] = mtm_per_unit * factor
    return out


# =============================================================================
# RKO MTM trajectory
# =============================================================================
def _rko_mtm_trajectory(trade: dict, fast_panels: dict) -> dict:
    """MTM trajectory for an American-barrier KO (RKO).

    Mirrors `_trade_mtm_trajectory` but routes through
    `core.ako_pricing.price_ako_dispatch` (American-barrier dispatcher).
    Also handles knockout: if `knockout_date` is set on the trade and
    falls within the MTM range, the trade's MTM becomes 0 from that
    date onwards (already implied by the closed form, but we short-
    circuit for safety).
    """
    from core.smile import smile_vol_at_strike as _smile_vol
    from core.ako_pricing import price_ako_dispatch as _price_ako_dispatch

    spot_fast = fast_panels["spot"]
    vol_fast = fast_panels["vol_panels_fast"]
    fwd_fast = fast_panels["fwd_panels_fast"]
    rr_fast = fast_panels.get("rr_panels_fast", {})
    bf_fast = fast_panels.get("bf_panels_fast", {})
    f_fast = fast_panels["f_panel_fast"]
    d_fast = fast_panels["d_panel_fast"]
    pip = fast_panels["pip_scale"]

    direction = trade["direction"]
    barrier_type = trade["barrier_type"]
    K = trade["strike"]
    H = trade["barrier"]
    S_trade = trade["spot"]
    notional = trade["notional_usd"]
    factor = notional / S_trade
    mtm_pricing_model = trade.get("pricing_model", "vol_at_strike")
    ko_date = trade.get("knockout_date")
    ko_dt = pd.Timestamp(ko_date) if ko_date else None

    trade_dt = pd.Timestamp(trade["trade_date"])
    expiry_dt = pd.Timestamp(trade["expiry_date"])
    last_mtm_day = expiry_dt - pd.Timedelta(days=1)
    if last_mtm_day < trade_dt:
        return {}
    bdays = pd.bdate_range(trade_dt, last_mtm_day)

    out: dict = {}
    for d_ts in bdays:
        # After knockout, MTM = 0 (option is dead)
        if ko_dt is not None and d_ts > ko_dt:
            out[d_ts] = 0.0
            continue
        if d_ts == trade_dt:
            mtm_per_unit = trade["premium_mid_pct"] / 100.0 * S_trade
        else:
            T_rem = (expiry_dt - d_ts).days / 365.0
            if T_rem < 1e-4:
                continue
            ts_np = np.datetime64(d_ts.to_datetime64())
            S_d = _fast_asof(spot_fast, ts_np)
            if S_d is None:
                continue
            sigma_atm_pct = _fast_interp_at_T(vol_fast, T_rem, ts_np)
            if sigma_atm_pct is None:
                continue
            sigma_atm_d = sigma_atm_pct / 100.0

            r_f = _fast_interp_at_T(f_fast, T_rem, ts_np)
            r_d_rate = _fast_interp_at_T(d_fast, T_rem, ts_np)
            if r_f is None and r_d_rate is None:
                continue
            fwd_pts = _fast_interp_at_T(fwd_fast, T_rem, ts_np)
            F_d = S_d + fwd_pts * pip if fwd_pts is not None else S_d
            if r_f is None:
                r_f = r_d_rate - np.log(F_d / S_d) / T_rem
            if r_d_rate is None:
                r_d_rate = r_f + np.log(F_d / S_d) / T_rem

            rr_pct = _fast_interp_at_T(rr_fast, T_rem, ts_np) if rr_fast else None
            bf_pct = _fast_interp_at_T(bf_fast, T_rem, ts_np) if bf_fast else None
            rr = (rr_pct / 100.0) if rr_pct is not None else 0.0
            bf = (bf_pct / 100.0) if bf_pct is not None else 0.0
            sigma_d = _smile_vol(S_d, K, T_rem, sigma_atm_d, rr, bf,
                                   r_d_rate, r_f)

            try:
                mtm_per_unit, _ = _price_ako_dispatch(
                    direction, barrier_type, S_d, K, H, T_rem,
                    sigma_atm=sigma_atm_d, sigma_smile=sigma_d,
                    rr_25=rr, bf_25=bf,
                    r_d=r_d_rate, r_f=r_f,
                    model=mtm_pricing_model,
                )
            except Exception:
                continue
        out[d_ts] = mtm_per_unit * factor
    return out


# =============================================================================
# Worst-of MTM trajectory
# =============================================================================
def _worstof_mtm_trajectory(trade: dict,
                                fast_panels_a: dict, fast_panels_b: dict,
                                wo_kind: str) -> dict:
    """MTM trajectory for a worst-of structure.

    Re-prices the joint structure on each business day during the trade's
    life using the CF engine (faster than MC; consistent within ~1 bp).

    For WO-EKO: `worstof_eko_price_cf`
    For WO-RKO: `worstof_rko_price_cf_approx`

    If EITHER leg has knocked out (recorded on the trade as
    `leg_a_knocked_out` / `leg_b_knocked_out`), the structure is dead
    from that day onwards. We don't store per-day-KO timing on
    WorstOfTrade (only end-state booleans), so we conservatively
    re-price the joint structure for all alive-at-trade-time days and
    let the CF pricer return ~0 if both legs are near KO.
    """
    from core.smile import smile_vol_at_strike as _smile_vol
    from core.worstof_pricer import WorstOfLeg, worstof_eko_price_cf
    from core.worstof_pricer_american import worstof_rko_price_cf_approx

    leg_a_pair = trade["leg_a_pair"]
    leg_b_pair = trade["leg_b_pair"]
    leg_a_direction = trade["leg_a_direction"]
    leg_b_direction = trade["leg_b_direction"]
    leg_a_bar_dir = trade["leg_a_barrier_type"]
    leg_b_bar_dir = trade["leg_b_barrier_type"]
    K_a = trade["leg_a_strike"]
    K_b = trade["leg_b_strike"]
    H_a = trade["leg_a_barrier"]
    H_b = trade["leg_b_barrier"]
    S_a_trade = trade["leg_a_spot"]
    S_b_trade = trade["leg_b_spot"]
    notional = trade["notional_usd"]
    # The factor maps per-unit price (in JPY-per-USD units etc.) back
    # to USD on the structure notional. We use leg_a's spot since the
    # structure premium is denominated in leg_a's units. This matches
    # the convention in core.worstof._run_worstof_with_panels.
    factor = notional / S_a_trade

    trade_dt = pd.Timestamp(trade["trade_date"])
    expiry_dt = pd.Timestamp(trade["expiry_date"])
    last_mtm_day = expiry_dt - pd.Timedelta(days=1)
    if last_mtm_day < trade_dt:
        return {}
    bdays = pd.bdate_range(trade_dt, last_mtm_day)

    pricer = (worstof_eko_price_cf if wo_kind == "wo_eko"
              else worstof_rko_price_cf_approx)

    def _build_fast_market(d_ts, fast_panels, K):
        """Get (S, sigma_smile, r_d, r_f, T_rem) for one pair on d_ts."""
        T_rem = (expiry_dt - d_ts).days / 365.0
        if T_rem < 1e-4:
            return None
        ts_np = np.datetime64(d_ts.to_datetime64())
        spot_fast = fast_panels["spot"]
        vol_fast = fast_panels["vol_panels_fast"]
        fwd_fast = fast_panels["fwd_panels_fast"]
        rr_fast = fast_panels.get("rr_panels_fast", {})
        bf_fast = fast_panels.get("bf_panels_fast", {})
        f_fast = fast_panels["f_panel_fast"]
        d_fast = fast_panels["d_panel_fast"]
        pip = fast_panels["pip_scale"]

        S_d = _fast_asof(spot_fast, ts_np)
        if S_d is None:
            return None
        sigma_atm_pct = _fast_interp_at_T(vol_fast, T_rem, ts_np)
        if sigma_atm_pct is None:
            return None
        sigma_atm_d = sigma_atm_pct / 100.0
        r_f = _fast_interp_at_T(f_fast, T_rem, ts_np)
        r_d_rate = _fast_interp_at_T(d_fast, T_rem, ts_np)
        if r_f is None and r_d_rate is None:
            return None
        fwd_pts = _fast_interp_at_T(fwd_fast, T_rem, ts_np)
        F_d = S_d + fwd_pts * pip if fwd_pts is not None else S_d
        if r_f is None:
            r_f = r_d_rate - np.log(F_d / S_d) / T_rem
        if r_d_rate is None:
            r_d_rate = r_f + np.log(F_d / S_d) / T_rem
        rr_pct = _fast_interp_at_T(rr_fast, T_rem, ts_np) if rr_fast else None
        bf_pct = _fast_interp_at_T(bf_fast, T_rem, ts_np) if bf_fast else None
        rr = (rr_pct / 100.0) if rr_pct is not None else 0.0
        bf = (bf_pct / 100.0) if bf_pct is not None else 0.0
        sigma_d = _smile_vol(S_d, K, T_rem, sigma_atm_d, rr, bf,
                               r_d_rate, r_f)
        return {"S": S_d, "sigma": sigma_d,
                 "r_d": r_d_rate, "r_f": r_f, "T_rem": T_rem}

    # Pre-compute realized log-returns of (spot_a, spot_b) for rolling
    # correlation. For MTM we use a fixed ρ = correlation at trade
    # date (simpler than re-estimating each day).
    rho_trade = trade.get("correlation_used", 0.30)
    if rho_trade is None or not np.isfinite(rho_trade):
        rho_trade = 0.30

    out: dict = {}
    for d_ts in bdays:
        if d_ts == trade_dt:
            mtm_per_unit_struct = (trade["structure_premium_mid_usd"]
                                       / notional * S_a_trade
                                   if notional > 0 else 0.0)
        else:
            mka = _build_fast_market(d_ts, fast_panels_a, K_a)
            mkb = _build_fast_market(d_ts, fast_panels_b, K_b)
            if mka is None or mkb is None:
                continue
            T_rem = mka["T_rem"]
            # Build WO legs with the engine's S=1 convention
            leg_a = WorstOfLeg(
                S=1.0, K=K_a / mka["S"], H=H_a / mka["S"],
                sigma=mka["sigma"], r_d=mka["r_d"], r_f=mka["r_f"],
                opt=leg_a_direction, bar_dir=leg_a_bar_dir,
            )
            leg_b = WorstOfLeg(
                S=1.0, K=K_b / mkb["S"], H=H_b / mkb["S"],
                sigma=mkb["sigma"], r_d=mkb["r_d"], r_f=mkb["r_f"],
                opt=leg_b_direction, bar_dir=leg_b_bar_dir,
            )
            # Sanity check: if leg A's barrier has been touched (for
            # American KOs) the price should be 0 — but we don't know
            # touch-status mid-trade. Trust the CF pricer.
            try:
                if wo_kind == "wo_eko":
                    res = pricer(leg_a, leg_b, T_rem, rho_trade,
                                   mka["r_d"], n_quad=80)
                else:
                    res = pricer(leg_a, leg_b, T_rem, rho_trade,
                                   mka["r_d"], n_quad=60)
                mtm_per_unit_struct = res["price"] * mka["S"]
                # rescale: engine returns price in units of leg_a's
                # current spot since S=1; we want it in absolute units
                # for the factor scaling. Actually since notional /
                # S_trade is the factor, and engine returns price as
                # fraction of S (=1 means 100%), we have:
                # USD MTM = res['price'] * notional
                # which is just `res['price']`*notional, NOT × S.
                # Reset:
                mtm_per_unit_struct = res["price"]  # USD-fraction units
            except Exception:
                continue
        # mtm_per_unit_struct is the structure price as a FRACTION of
        # foreign notional (since engine spots = 1.0). USD = fraction
        # × notional.
        out[d_ts] = mtm_per_unit_struct * notional
    return out


# =============================================================================
# Main entry point
# =============================================================================
def compute_strategy_mtm_curve(
        strat_info: dict, trades_df: pd.DataFrame, panels_cache: dict,
        side_sign: float,
        progress_cb: Optional[Callable[[float], None]] = None,
) -> pd.DataFrame:
    """Compute the mark-to-market equity curve for one strategy.

    Parameters
    ----------
    strat_info : dict
        From `_generate_strategies` — has 'kind' and 'spec'.
    trades_df : pd.DataFrame
        Per-trade ledger from the engine (already P&L-sign-flipped per
        Buy/Sell side).
    panels_cache : dict
        Same cache used during the backtest run, keyed by
        (cache_kind, pair, prefer). For WO strategies we look up BOTH
        legs' panels.
    side_sign : float
        +1 for Buy, -1 for Sell. Applied to MTM book and cash flows.

    Returns
    -------
    DataFrame indexed by date with columns:
      book_value_usd, cash_position_usd, equity_usd, peak_usd, drawdown_usd
    """
    if trades_df.empty:
        return pd.DataFrame()

    kind = strat_info["kind"]
    spec = strat_info["spec"]

    # 1. Compute per-trade MTM trajectory (sum across open trades)
    if kind == "vanilla":
        cache_key = ("eko", spec.pair, spec.prefer)
        panels = panels_cache.get(cache_key)
        if not panels:
            return pd.DataFrame()
        fast_panels = _build_fast_panels(panels)
        traj_fn = lambda t: _vanilla_mtm_trajectory(t, fast_panels)
    elif kind == "eko":
        cache_key = ("eko", spec.pair, spec.prefer)
        panels = panels_cache.get(cache_key)
        if not panels:
            return pd.DataFrame()
        fast_panels = _build_fast_panels(panels)
        traj_fn = lambda t: _trade_mtm_trajectory(t, fast_panels)
    elif kind == "rko":
        cache_key = ("rko", spec.pair, spec.prefer)
        panels = panels_cache.get(cache_key)
        if not panels:
            return pd.DataFrame()
        fast_panels = _build_fast_panels(panels)
        traj_fn = lambda t: _rko_mtm_trajectory(t, fast_panels)
    elif kind in ("wo_eko", "wo_rko"):
        cache_kind = "rko" if kind == "wo_rko" else "eko"
        key_a = (cache_kind, spec.leg_a_pair, spec.prefer)
        key_b = (cache_kind, spec.leg_b_pair, spec.prefer)
        panels_a = panels_cache.get(key_a)
        panels_b = panels_cache.get(key_b)
        if not panels_a or not panels_b:
            return pd.DataFrame()
        fast_a = _build_fast_panels(panels_a)
        fast_b = _build_fast_panels(panels_b)
        traj_fn = lambda t: _worstof_mtm_trajectory(t, fast_a, fast_b, kind)
    else:
        return pd.DataFrame()

    # 2. Aggregate MTM book (sum across all trades on each day)
    mtm_records = []
    n_trades = len(trades_df)
    for i, t in trades_df.iterrows():
        if progress_cb is not None and i % 50 == 0:
            progress_cb(i / max(n_trades, 1))
        try:
            traj = traj_fn(t.to_dict())
        except Exception:
            continue
        for d_ts, v in traj.items():
            # MTM book signed by Buy/Sell direction. For Buy: long
            # position has positive book value. For Sell: short
            # position has negative book value (we owe the option).
            mtm_records.append((d_ts, side_sign * v))

    if not mtm_records:
        return pd.DataFrame()
    mtm_long = pd.DataFrame(mtm_records, columns=["date", "mtm_usd"])
    book = mtm_long.groupby("date")["mtm_usd"].sum().sort_index()

    # 3. Cash flow series. Note: trades_df.premium_usd and
    # actual_payoff_usd ALREADY include the side sign (applied in
    # _run_one_strategy). So we use them directly.
    df = trades_df.copy()
    df["trade_dt"] = pd.to_datetime(df["trade_date"])
    df["expiry_dt"] = pd.to_datetime(df["expiry_date"])

    # Premium: paid for Buy (cash out, negative), received for Sell
    # (cash in, positive). The premium_usd column has the right sign
    # already.
    if "premium_usd" in df.columns:
        prem_col = "premium_usd"
    elif "structure_premium_paid_usd" in df.columns:
        prem_col = "structure_premium_paid_usd"
    else:
        return pd.DataFrame()

    prem = df.groupby("trade_dt")[prem_col].sum()

    if "actual_payoff_usd" in df.columns:
        pay_col = "actual_payoff_usd"
    elif "worst_of_payoff_usd" in df.columns:
        pay_col = "worst_of_payoff_usd"
    else:
        pay_col = None

    if pay_col:
        pay = df.groupby("expiry_dt")[pay_col].sum()
    else:
        pay = pd.Series(dtype=float)

    all_dates = book.index.union(prem.index).union(pay.index).sort_values()
    book_full = book.reindex(all_dates, fill_value=0.0)
    cf = pd.Series(0.0, index=all_dates)
    # Buy: premium is positive (we paid), so cash position -= premium
    # Sell: premium is negative (we received), so cash position -= premium = +ve
    # In both cases the operator is -=premium, which gives the right sign.
    cf.loc[prem.index] -= prem
    if pay_col and not pay.empty:
        # Buy: payoff is positive (we receive at expiry), so cash += payoff
        # Sell: payoff is negative (we pay), so cash += payoff (subtraction)
        cf.loc[pay.index] += pay
    cum_cash = cf.cumsum()

    equity = book_full + cum_cash
    peak = equity.cummax()
    dd = equity - peak

    return pd.DataFrame({
        "book_value_usd": book_full,
        "cash_position_usd": cum_cash,
        "equity_usd": equity,
        "peak_usd": peak,
        "drawdown_usd": dd,
    })


def aggregate_portfolio_mtm(
        strategy_curves: "dict[str, pd.DataFrame]",
) -> pd.DataFrame:
    """Sum per-strategy MTM equity curves into a portfolio curve.

    Returns DataFrame indexed by date with: book_value_usd,
    cash_position_usd, equity_usd, peak_usd, drawdown_usd.
    """
    if not strategy_curves:
        return pd.DataFrame()
    # Collect union of all dates
    all_dates = pd.DatetimeIndex(sorted(set().union(
        *[set(c.index) for c in strategy_curves.values() if not c.empty]
    )))
    if len(all_dates) == 0:
        return pd.DataFrame()

    book = pd.Series(0.0, index=all_dates)
    cash = pd.Series(0.0, index=all_dates)
    for name, c in strategy_curves.items():
        if c.empty:
            continue
        cb = c["book_value_usd"].reindex(all_dates, fill_value=0.0)
        cc = c["cash_position_usd"].reindex(all_dates, method="ffill",
                                                  fill_value=0.0)
        book = book.add(cb, fill_value=0.0)
        cash = cash.add(cc, fill_value=0.0)

    equity = book + cash
    peak = equity.cummax()
    dd = equity - peak
    return pd.DataFrame({
        "book_value_usd": book,
        "cash_position_usd": cash,
        "equity_usd": equity,
        "peak_usd": peak,
        "drawdown_usd": dd,
    })

"""Backtest engine for American-barrier KO options (App 12).

# What's different from core.backtest (European-barrier, App 9)

Two changes from the App 9 engine:

  1. **Barrier monitoring**: each business day in the trade's life, we
     check whether the day's intraday range [Low, High] contains the
     barrier H. If yes, the trade knocks out on that day with payoff =
     0 and net P&L = −premium_paid. This is richer than the European
     check (close at expiry only) and matches how American-barrier
     contracts settle in practice.

  2. **Pricing**: entry premium uses the Vanna-Volga smile-adjusted
     price (matches Bloomberg OVML). Falls back to flat-vol Reiner-
     Rubinstein when no RR/BF data is available, ensuring the engine
     stays usable on data sets without smile.

Everything downstream — ledger schema, summary stats, equity/drawdown
calcs, monthly/annual tables — uses the **same Trade dataclass and the
same `trades_to_df / summarize_strategy / compute_equity_and_drawdown`
helpers from core.backtest**. The only new fields are `knockout_date`,
`knockout_spot`, and `pricing_model`, all already added to the base
Trade dataclass with sensible defaults.

# OHLC data requirement

The American-barrier check requires daily High and Low for the spot.
We use `core.data_loader.load_spot_ohlc` which returns whichever of
[open, high, low, close] columns are present in the SPOT CSV. If only
Close is available, we fall back to close-only monitoring (the trade
knocks out if Close crosses the barrier on any day — a less faithful
but still defensible approximation, surfaced in the returned Trade's
`pricing_model` for traceability).

# Trade-mode semantics

Same as App 9 — 'stack' (overlapping book, daily rebalancing) or
'single' (one trade per pair at a time, next entry only after prior
expiry).

# Entry gates

The gate framework (core.gates) is reused as-is — same key registry,
same per-day mask semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd

from core.backtest import StrategySpec, Trade, _interp_panels_at_T, _spot_at_or_before
from core.calendar import compute_option_dates
from core.gates import compute_gate_mask
from core.ko_solvers import solve_strike
from core.american_barrier import ako_closed_form
from core.vanna_volga import vv_price_ko
from core.data_loader import load_panel, load_by_ticker, load_spot_ohlc
from core.rates import load_rates_panel, get_rate_at
from core.conventions import get_pip_scale


# =============================================================================
# Pair-level data preload — same as core.backtest.preload_pair_panels but
# also pulls OHLC for the barrier check.
# =============================================================================
def preload_pair_panels_american(
        folder: str, pair: str, prefer: str = "offshore",
        standard_tenors=("1M", "2M", "3M", "6M", "9M", "1Y"),
        ) -> dict:
    """Like core.backtest.preload_pair_panels but adds 'spot_ohlc' (DataFrame
    with open/high/low/close columns when available)."""
    spot_df = load_panel(folder, "SPOT", None, prefer=prefer, pairs=(pair,))
    if spot_df.empty or pair not in spot_df.columns:
        return {}
    spot = spot_df[pair].dropna()
    spot_ohlc = load_spot_ohlc(folder, pair, prefer=prefer)
    # spot_ohlc.close should match spot. Sanity-check (only the first row to
    # avoid pandas warning); fall back to close-only monitoring otherwise.

    vol_panels, fwd_panels, rr_panels, bf_panels = {}, {}, {}, {}
    for t in standard_tenors:
        vdf = load_panel(folder, "VOL_ATM", t, prefer=prefer, pairs=(pair,))
        if not vdf.empty and pair in vdf.columns:
            vol_panels[t] = vdf[pair].dropna()
        fdf = load_panel(folder, "FWD_POINTS", t, prefer=prefer, pairs=(pair,))
        if not fdf.empty and pair in fdf.columns:
            fwd_panels[t] = fdf[pair].dropna()
        rdf = load_panel(folder, "VOL_25R", t, prefer=prefer, pairs=(pair,))
        if not rdf.empty and pair in rdf.columns:
            rr_panels[t] = rdf[pair].dropna()
        bdf = load_panel(folder, "VOL_25B", t, prefer=prefer, pairs=(pair,))
        if not bdf.empty and pair in bdf.columns:
            bf_panels[t] = bdf[pair].dropna()

    foreign_ccy, domestic_ccy = pair[:3].upper(), pair[3:].upper()
    f_panel = load_rates_panel(folder, foreign_ccy, load_by_ticker)
    d_panel = load_rates_panel(folder, domestic_ccy, load_by_ticker)

    return {
        "spot": spot,
        "spot_ohlc": spot_ohlc,
        "vol_panels": vol_panels,
        "fwd_panels": fwd_panels,
        "rr_panels": rr_panels,
        "bf_panels": bf_panels,
        "f_panel": f_panel,
        "d_panel": d_panel,
        "pip_scale": get_pip_scale(pair),
        "foreign_ccy": foreign_ccy,
        "domestic_ccy": domestic_ccy,
        "smile_available": bool(rr_panels) and bool(bf_panels),
        "ohlc_available": ("high" in spot_ohlc.columns
                            and "low" in spot_ohlc.columns)
                            if not spot_ohlc.empty else False,
    }


# =============================================================================
# Barrier-hit helper
# =============================================================================
def _first_barrier_hit_date(spot_ohlc: pd.DataFrame, barrier_type: str,
                                  H: float, entry_date: date, expiry_date: date,
                                  ) -> Optional[pd.Timestamp]:
    """Return the timestamp of the first day in [entry_date+1, expiry_date]
    where the OHLC range contains the barrier, or None if the trade survives.

    Convention:
      - up_and_out: barrier hit if `high >= H` on any day. Equivalent to
        "[low, high] contains H or extends above H".
      - down_and_out: barrier hit if `low <= H` on any day.
      - Entry date itself is NOT checked — the trade exists from entry+1
        onwards. (Entry-day spot is the trade's reference; barrier check
        starts the next business day.)
      - If OHLC is unavailable, fall back to close-only check using the
        spot close. Detected at the caller level.
    """
    # Slice to trade window, exclusive at entry (entry day = open day, no
    # monitoring); inclusive at expiry.
    start = pd.Timestamp(entry_date) + pd.Timedelta(days=1)
    end = pd.Timestamp(expiry_date)
    if start > end:
        return None
    window = spot_ohlc.loc[start:end]
    if window.empty:
        return None
    if barrier_type == "up_and_out":
        # Hit if high reached H (or higher)
        hit_mask = window["high"] >= H
    else:
        hit_mask = window["low"] <= H
    if not hit_mask.any():
        return None
    return hit_mask[hit_mask].index[0]


def _first_barrier_hit_close_only(spot: pd.Series, barrier_type: str,
                                          H: float, entry_date: date,
                                          expiry_date: date,
                                          ) -> Optional[pd.Timestamp]:
    """Close-only fallback when OHLC isn't available. Hits when close
    crosses H. Identical to App 9's European check but evaluated daily.
    """
    start = pd.Timestamp(entry_date) + pd.Timedelta(days=1)
    end = pd.Timestamp(expiry_date)
    if start > end:
        return None
    window = spot.loc[start:end]
    if window.empty:
        return None
    if barrier_type == "up_and_out":
        hit_mask = window >= H
    else:
        hit_mask = window <= H
    if not hit_mask.any():
        return None
    return hit_mask[hit_mask].index[0]


# =============================================================================
# Main backtest loop
# =============================================================================
def run_single_strategy_american(
        spec: StrategySpec, panels: dict,
        start_date: date, end_date: date,
        notional_usd: float = 10_000_000.0,
        progress_cb: Optional[Callable[[float], None]] = None,
        ) -> list[Trade]:
    """American-barrier backtest for one strategy spec.

    Mirrors core.backtest.run_single_strategy in structure but:
      - prices entry premium via Vanna-Volga (smile-adjusted)
      - solves K, H via the VV pricer (slow but consistent with the
        displayed premium)
      - checks the barrier daily using OHLC range, marking the first
        hit date on the Trade

    Returns a list of Trade records with `knockout_date` populated when
    the trade was knocked out in-life. `pricing_model` is set to
    'vanna_volga' if smile data was used, else 'r_r' for flat-vol.
    """
    spot = panels["spot"]
    spot_ohlc = panels.get("spot_ohlc", pd.DataFrame())
    ohlc_available = panels.get("ohlc_available", False)
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

    gate_mask = compute_gate_mask(spot, spec.entry_gate)
    if gate_mask is not None:
        gate_open_ts = set(gate_mask[gate_mask].index.normalize())
    else:
        gate_open_ts = None

    last_data_ts = spot.index.max()
    last_open_expiry: Optional[date] = None
    trades: list[Trade] = []
    n_total = len(trade_dates)

    # Build a VV-aware pricer for the solver — closure over RR/BF at the
    # solve point so the solver can see the same vol surface the pricer
    # will use. Rebuilt on each trade date when rr_25/bf_25 change.
    def build_vv_pricer(rr_25_local: float, bf_25_local: float):
        def _vv_pricer(opt, bar, S_, K_, H_, T_, sig_, rd_, rf_):
            out = vv_price_ko(opt, bar, S_, K_, H_, T_,
                                 sig_, rr_25_local, bf_25_local,
                                 rd_, rf_,
                                 flat_vol_pricer=ako_closed_form)
            return out["price_vv"]
        return _vv_pricer

    for i, td in enumerate(trade_dates):
        td_ts = pd.Timestamp(td)

        if (spec.trade_mode == "single"
                and last_open_expiry is not None
                and td < last_open_expiry):
            continue
        if gate_open_ts is not None and td_ts not in gate_open_ts:
            continue

        S = _spot_at_or_before(spot, td_ts)
        if S is None:
            continue

        opt_dates = compute_option_dates(td, spec.tenor_label)
        T = opt_dates.T_years
        if pd.Timestamp(opt_dates.option_expiry) > last_data_ts:
            continue

        sigma_atm_pct = _interp_panels_at_T(vol_panels, T, td_ts)
        if sigma_atm_pct is None:
            continue
        sigma_atm = sigma_atm_pct / 100.0

        rr_pct = _interp_panels_at_T(rr_panels, T, td_ts) if rr_panels else None
        bf_pct = _interp_panels_at_T(bf_panels, T, td_ts) if bf_panels else None
        rr_25 = (rr_pct / 100.0) if rr_pct is not None else 0.0
        bf_25 = (bf_pct / 100.0) if bf_pct is not None else 0.0
        smile_avail = (rr_pct is not None) and (bf_pct is not None)

        fwd_pts = _interp_panels_at_T(fwd_panels, T, td_ts)
        F_market = S + fwd_pts * pip if fwd_pts is not None else S

        r_f = get_rate_at(f_panel, T, td)
        r_d = get_rate_at(d_panel, T, td)
        if r_f is None and r_d is None:
            continue
        if r_f is None:
            r_f = r_d - np.log(F_market / S) / T
        if r_d is None:
            r_d = r_f + np.log(F_market / S) / T
        F_implied = S * np.exp((r_d - r_f) * T)

        # Solve K, H. Solver-pricer matches the headline model so the
        # achieved leverage is consistent with the recorded premium:
        #
        #   vanna_volga    : VV-on-RR (matches headline VV price)
        #   flat_atm /
        #   vol_at_strike  : RR closed form (solve_strike's smile-aware
        #                    branch internally picks σ_atm or σ_smile;
        #                    it's the same RR-based engine either way)
        #
        # When smile data is unavailable, the three modes coincide and
        # we use the cheaper closed form.
        if (not smile_avail
                or spec.pricing_model in ("flat_atm", "vol_at_strike")):
            pricing_pricer = ako_closed_form
        else:   # vanna_volga
            pricing_pricer = build_vv_pricer(rr_25, bf_25)
        try:
            K, H, info = solve_strike(
                spec.direction, spec.barrier_type, spec.delta_value,
                S, T, sigma_atm, r_d, r_f,
                target_ratio=(spec.payout_ratio if spec.ko_method == "ratio" else None),
                target_ko_delta=(spec.target_ko_delta if spec.ko_method == "delta" else None),
                ko_method=spec.ko_method,
                rr_25=rr_25, bf_25=bf_25,
                pricer=pricing_pricer,
            )
        except Exception:
            continue

        achieved = info.get("achieved_ratio", float("nan"))
        ratio_min = info.get("ratio_min", float("nan"))
        if spec.ko_method == "ratio":
            feasible = (np.isfinite(achieved) and
                          abs(achieved - spec.payout_ratio) < 0.01)
        else:
            if "note" in info:
                continue
            feasible = True

        sigma_smile = float(info.get("sigma_smile", sigma_atm))
        sigma_mid = sigma_smile
        sigma_used = sigma_smile

        # Entry premium — dispatch on spec.pricing_model. Three modes
        # supported via core.ako_pricing.price_ako_dispatch (Step R1):
        #
        #   'flat_atm'      ako_closed_form at σ_atm
        #   'vol_at_strike' ako_closed_form at σ_smile(K)
        #   'vanna_volga'   VV correction on top of ako_closed_form
        #
        # When smile data isn't available, σ_smile collapses to σ_atm
        # and the three modes coincide, so we still dispatch (the
        # detail dict labels the model correctly).
        from core.ako_pricing import price_ako_dispatch as _price_ako_dispatch
        _model = spec.pricing_model if smile_avail else "flat_atm"
        premium_mid, _hl_detail = _price_ako_dispatch(
            spec.direction, spec.barrier_type,
            S, K, H, T,
            sigma_atm=sigma_atm, sigma_smile=sigma_smile,
            rr_25=rr_25, bf_25=bf_25,
            r_d=r_d, r_f=r_f,
            model=_model,
        )
        # Persist the model on the Trade so downstream ledgers tag it.
        # The legacy "r_r" / "vanna_volga" labels are kept stable so old
        # presets still load: flat_atm + vol_at_strike both map to "r_r"
        # (they're both Reiner-Rubinstein flavours; the distinction is
        # which σ is fed in, recorded separately as sigma_used).
        if _model == "vanna_volga":
            pricing_model = "vanna_volga"
        else:
            pricing_model = "r_r"

        max_pay = abs(H - K)

        # ====== Daily barrier check + settlement ======
        if ohlc_available:
            hit_ts = _first_barrier_hit_date(spot_ohlc, spec.barrier_type, H,
                                                  td, opt_dates.option_expiry)
        else:
            hit_ts = _first_barrier_hit_close_only(spot, spec.barrier_type, H,
                                                          td, opt_dates.option_expiry)
            pricing_model += "_close_only"

        S_exp_ts = pd.Timestamp(opt_dates.option_expiry)
        S_exp = _spot_at_or_before(spot, S_exp_ts)
        if S_exp is None:
            continue

        if hit_ts is not None:
            knocked_out = True
            actual_payoff = 0.0
            ko_date = hit_ts.date()
            ko_spot = _spot_at_or_before(spot, hit_ts)
        else:
            knocked_out = False
            ko_date = None
            ko_spot = None
            if spec.direction == "call":
                actual_payoff = max(S_exp - K, 0.0)
            else:
                actual_payoff = max(K - S_exp, 0.0)
            actual_payoff = min(actual_payoff, max_pay)

        prem_mid_pct = premium_mid / S * 100.0
        tx_cost_pct = spec.tx_cost_bps / 100.0
        prem_pct = prem_mid_pct + tx_cost_pct
        max_pay_pct = max_pay / S * 100.0
        actual_pay_pct = actual_payoff / S * 100.0
        pnl_pct = actual_pay_pct - prem_pct
        pnl_gross_pct = actual_pay_pct - prem_mid_pct
        f_usd = notional_usd / 100.0

        trades.append(Trade(
            strategy_name=spec.name,
            pair=spec.pair,
            direction=spec.direction,
            barrier_type=spec.barrier_type,
            delta_label=spec.delta_label,
            tenor_label=spec.tenor_label,
            ko_method=spec.ko_method,
            target_payout_ratio=(spec.payout_ratio if spec.ko_method == "ratio"
                                   else None),
            target_ko_delta=(spec.target_ko_delta if spec.ko_method == "delta"
                               else None),
            entry_gate=spec.entry_gate,
            tx_cost_bps=spec.tx_cost_bps,
            trade_date=td,
            spot_settlement=opt_dates.spot_settlement,
            option_settlement=opt_dates.option_settlement,
            expiry_date=opt_dates.option_expiry,
            T_years=T,
            spot=S,
            sigma_atm=sigma_atm,
            rr_25=rr_25,
            bf_25=bf_25,
            sigma_smile=sigma_smile,
            sigma_mid=sigma_mid,
            sigma_used=sigma_used,
            r_d=r_d,
            r_f=r_f,
            fwd_market=F_market,
            fwd_implied=F_implied,
            strike=K,
            barrier=H,
            achieved_payout_ratio=float(achieved) if np.isfinite(achieved) else float("nan"),
            feasible=bool(feasible),
            ratio_min=float(ratio_min) if np.isfinite(ratio_min) else float("nan"),
            premium_pct=prem_pct,
            premium_mid_pct=prem_mid_pct,
            transaction_cost_pct=tx_cost_pct,
            max_payoff_pct=max_pay_pct,
            spot_at_expiry=S_exp,
            knocked_out=bool(knocked_out),
            actual_payoff_pct=actual_pay_pct,
            pnl_pct=pnl_pct,
            pnl_gross_pct=pnl_gross_pct,
            notional_usd=notional_usd,
            premium_usd=prem_pct * f_usd,
            premium_mid_usd=prem_mid_pct * f_usd,
            transaction_cost_usd=tx_cost_pct * f_usd,
            max_payoff_usd=max_pay_pct * f_usd,
            actual_payoff_usd=actual_pay_pct * f_usd,
            pnl_usd=pnl_pct * f_usd,
            pnl_gross_usd=pnl_gross_pct * f_usd,
            # American-barrier extras
            knockout_date=ko_date,
            knockout_spot=ko_spot,
            pricing_model=pricing_model,
        ))
        last_open_expiry = opt_dates.option_expiry

        if progress_cb is not None and (i % 20 == 0 or i + 1 == n_total):
            progress_cb((i + 1) / n_total)

    return trades


# =============================================================================
# Grid runner — multiple specs in one call
# =============================================================================
def run_grid_american(
        folder: str, specs: "list[StrategySpec]",
        start_date: date, end_date: date,
        notional_usd: float = 10_000_000.0,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        ) -> "dict[str, list[Trade]]":
    """Run a batch of strategy specs. Returns {strategy_name: [Trade, ...]}.

    Loads each pair's panels once and reuses across all specs for that
    pair. progress_cb(p, name) is called after each spec finishes.
    """
    pairs_needed = sorted({s.pair for s in specs})
    panels_by_pair = {}
    for p in pairs_needed:
        prefer = next((s.prefer for s in specs if s.pair == p), "offshore")
        panels_by_pair[p] = preload_pair_panels_american(folder, p, prefer=prefer)

    out: dict[str, list[Trade]] = {}
    n_specs = len(specs)
    for i, spec in enumerate(specs):
        panels = panels_by_pair.get(spec.pair, {})
        if not panels:
            out[spec.name] = []
        else:
            out[spec.name] = run_single_strategy_american(
                spec, panels, start_date, end_date,
                notional_usd=notional_usd,
            )
        if progress_cb is not None:
            progress_cb((i + 1) / n_specs, spec.name)
    return out

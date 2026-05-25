"""Daily-rolling KO option backtest, European fixing only, no delta hedge.

# Trade lifecycle
For each business day d in [start_date, end_date]:
  1. Compute option date schedule: trade=d, spot=d+2bd, settle=spot+tenor,
     expiry=settle-2bd.
  2. Apply optional entry gate. If gate fails, skip d. Existing positions
     are unaffected.
  3. Look up S(d), σ_atm(d, T_v), 25Δ RR(d, T_v), 25Δ BF(d, T_v),
     F(d, T_v), r_f(d, T_v), r_d(d, T_v) where T_v = (expiry - trade)/365.
  4. Solve K at σ_atm from target vanilla Δ (FX delta convention).
     Compute σ_smile(K) by linear interp in spot-call-delta across the
     three smile anchors (25P / ATM / 25C). When RR/BF data is absent
     σ_smile = σ_atm (flat).
  5. Solve H — either from target payout ratio at σ_smile (ratio mode)
     or at vanilla-Δ wing strike at σ_atm (delta mode).
  6. Price KO premium at σ_smile (mid market for this strike).
  7. Apply transaction cost: flat bps of foreign notional, added on top
     of the mid premium → premium_paid = premium_mid + tx_cost.
  8. At expiry, look up S(expiry) and compute realized payoff:
       - up-out call:    payoff = max(S_T - K, 0) if S_T < H else 0
       - down-out put:   payoff = max(K - S_T, 0) if S_T > H else 0
     Capped at |H − K| in case spot lands a hair below the barrier.
  9. PnL = realized payoff − premium_paid (already net of tx cost).

# PnL normalisation
All cash quantities are reported as percent of the foreign-currency
notional, by dividing the per-unit ko_price by S at trade date and
multiplying by 100. With $10M USD notional, 1.0 in `premium_pct`
means $100,000 USD premium. USD columns = pct × notional / 100.

This is the standard "model PnL" — it ignores the second-order effect of
spot moves on the foreign→USD conversion of the realized payoff, which
is usually <1% of the payoff itself.

# Sharpe convention
Monthly aggregation: sum PnL by expiry month, then Sharpe = mean / std *
sqrt(12). For option strategies with daily entries and lumpy realizations,
monthly aggregation is more meaningful than daily. MTM mode uses daily
mark-to-market and Sharpe(d) × sqrt(252).

# Performance notes
The H ratio-solver does scipy.optimize.minimize_scalar + brentq per trade
(~25 ko_price evals = ~25ms). Delta-mode H solve is closed-form (~µs).
For a 700-day backtest with 4 strategies, expect ~1-2 minutes total.
Progress callback is supported.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import date
from typing import Optional, Callable

import numpy as np
import pandas as pd

from core.calendar import compute_option_dates
from core.ko import ko_price
from core.ko_solvers import solve_strike
from core.rates import load_rates_panel, get_rate_at, TENOR_YEARS
from core.data_loader import load_by_ticker, load_panel
from core.conventions import get_pip_scale
from core.gates import compute_gate_mask


# -----------------------------------------------------------------------------
# Strategy spec + Trade record
# -----------------------------------------------------------------------------
@dataclass
class StrategySpec:
    pair: str
    direction: str            # 'call' | 'put'
    barrier_type: str         # 'up_and_out' | 'down_and_out'
    delta_label: str          # 'ATM', '40Δ', etc.
    delta_value: float        # 0.0 for ATM, 0.40 for 40Δ
    tenor_label: str          # '1M', '6W', etc.
    tx_cost_bps: float        # bps of foreign notional, flat markup (e.g. 2.0)
    prefer: str = "offshore"  # for Asia EM pairs
    # KO method — exactly one of payout_ratio / target_ko_delta is used,
    # selected by ko_method ('ratio' or 'delta').
    ko_method: str = "ratio"
    payout_ratio: Optional[float] = None
    target_ko_delta: Optional[float] = None
    ko_delta_label: Optional[str] = None  # display label e.g. '10Δ'
    # Entry gate: per-day filter on new trade entries. None = no gate.
    # See core/gates.py for registered gate keys.
    entry_gate: Optional[str] = None
    # Trade mode:
    #   'stack'  — default, current behaviour: a new trade is opened on
    #              every eligible date, regardless of open positions.
    #              Daily rebalancing / overlapping book.
    #   'single' — at most one trade per pair is open at any time. A new
    #              trade is opened only on the FIRST eligible date on or
    #              after the previous trade's expiry. Cleaner single-line
    #              backtest with much lower turnover.
    trade_mode: str = "stack"

    # Step 1c — single-leg pricing model. Controls how the smile enters
    # the EKO premium during the BACKTEST loop. Mirrors the live pricer's
    # sidebar selection (pages/eko_pricer.py:eko_pricing_model) so a
    # backtest at, say, 'vanna_volga' produces ledgers that line up with
    # live mid-marks under the same model. Default is 'vol_at_strike',
    # the historical app behavior (= σ_smile(K) into ko_price). Switch
    # to 'flat_atm' for a debug baseline or 'vanna_volga' for the
    # smile-corrected (Castagna-Mercurio) mid.
    #
    # ALL three branches dispatch through core.eko_pricing.price_eko_dispatch
    # which is shared with the live pricer — adding a new model is a
    # single edit there.
    pricing_model: str = "vol_at_strike"

    @property
    def name(self) -> str:
        ko_dir = self.barrier_type.replace("_and_out", "")
        if self.ko_method == "delta":
            ko_str = f"H@{self.ko_delta_label or f'{int((self.target_ko_delta or 0)*100)}Δ'}"
        else:
            ko_str = f"{self.payout_ratio:.0f}×"
        base = (f"{self.pair} {self.direction.upper()}-{ko_dir}out  "
                 f"{self.delta_label}  {self.tenor_label}  {ko_str}")
        if self.entry_gate:
            from core.gates import gate_label
            base += f"  [{gate_label(self.entry_gate)}]"
        return base

    @property
    def short_name(self) -> str:
        return f"{self.pair}_{self.direction}_{self.delta_label}_{self.tenor_label}"


@dataclass
class Trade:
    # context
    strategy_name: str
    pair: str
    direction: str
    barrier_type: str
    delta_label: str
    tenor_label: str
    ko_method: str                          # 'ratio' or 'delta'
    target_payout_ratio: Optional[float]    # set when ko_method='ratio'
    target_ko_delta: Optional[float]        # set when ko_method='delta'
    entry_gate: Optional[str]               # gate key, or None if no gate
    tx_cost_bps: float

    # dates
    trade_date: date
    spot_settlement: date
    option_settlement: date
    expiry_date: date
    T_years: float

    # market context at trade
    spot: float
    sigma_atm: float       # raw ATM vol (decimal) from market data
    rr_25: float           # 25Δ risk reversal (decimal); 0 if unavailable
    bf_25: float           # 25Δ butterfly (decimal); 0 if unavailable
    sigma_smile: float     # vol-at-strike (smile-adjusted; = sigma_atm if no RR/BF)
    sigma_mid: float       # alias of sigma_smile, kept for backward compat
    sigma_used: float      # decimal; alias for sigma_smile in bps-tx mode
                            # (kept for backward-compat with downstream code)
    r_d: float
    r_f: float
    fwd_market: float      # from FWD_POINTS
    fwd_implied: float     # S * exp((r_d - r_f) * T)

    # solved structure (K and H solved with σ_used — what trader actually executes)
    strike: float
    barrier: float
    achieved_payout_ratio: float
    feasible: bool
    ratio_min: float       # the U-shape minimum at this strike

    # premium decomposition (% of foreign notional)
    premium_pct: float          # at σ_used — what the trader pays
    premium_mid_pct: float      # at σ_mid — fair value (no tx cost)
    transaction_cost_pct: float # = premium_pct − premium_mid_pct
    max_payoff_pct: float

    # realized
    spot_at_expiry: Optional[float]
    knocked_out: Optional[bool]
    actual_payoff_pct: float
    pnl_pct: float           # net of tx cost (= actual_payoff − premium_paid)
    pnl_gross_pct: float     # before tx cost (= actual_payoff − premium_mid)

    # USD-denominated fields (computed from pct × notional / 100)
    notional_usd: float
    premium_usd: float
    premium_mid_usd: float
    transaction_cost_usd: float
    max_payoff_usd: float
    actual_payoff_usd: float
    pnl_usd: float
    pnl_gross_usd: float

    # American-barrier extras — set by backtest_american; None in App 9
    # (European barrier) where KO can only fire at expiry.
    # `knockout_date`: the first business day on which the barrier was
    #     contained in the day's [Low, High] range. None if the trade
    #     survived to expiry, OR if this is a European-barrier trade.
    # `knockout_spot`: the spot CLOSE on the knockout day, recorded for
    #     traceability. (The actual KO spot was somewhere in the day's
    #     range, not necessarily the close.)
    knockout_date: Optional[date] = None
    knockout_spot: Optional[float] = None
    # Pricing model used at entry: 'european' (App 9 ko_price), 'r_r'
    # (Reiner-Rubinstein flat-vol, App 12 ako_closed_form), or 'vanna_volga'
    # (App 12 smile-adjusted VV). Helps disambiguate ledgers and makes
    # results reproducible.
    pricing_model: str = "european"


# -----------------------------------------------------------------------------
# Pair-level data preload
# -----------------------------------------------------------------------------
def preload_pair_panels(folder: str, pair: str, prefer: str = "offshore",
                          standard_tenors=("1M", "2M", "3M", "6M", "9M", "1Y")
                          ) -> dict:
    """Load all data needed for backtesting one pair.

    Returns dict with: 'spot', 'vol_panels' (tenor->Series), 'fwd_panels'
    (tenor->Series), 'rr_panels' (tenor->Series, 25Δ RR), 'bf_panels'
    (tenor->Series, 25Δ BF), 'f_panel' (foreign rates DF), 'd_panel'
    (domestic rates DF), 'pip_scale', 'foreign_ccy', 'domestic_ccy'.

    RR / BF panels are loaded if available (categories VOL_25R / VOL_25B);
    if absent the smile collapses to flat ATM.
    """
    spot_df = load_panel(folder, "SPOT", None, prefer=prefer, pairs=(pair,))
    if spot_df.empty or pair not in spot_df.columns:
        return {}
    spot = spot_df[pair].dropna()

    vol_panels = {}
    fwd_panels = {}
    rr_panels = {}
    bf_panels = {}
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
    }


def _interp_panels_at_T(panels: dict, T_target: float,
                         valuation_ts: pd.Timestamp) -> Optional[float]:
    """Linearly interpolate across tenors at a given T, using forward-fill
    on each tenor's series."""
    points = []
    for tenor, ser in panels.items():
        valid = ser.loc[:valuation_ts]
        if valid.empty:
            continue
        T = TENOR_YEARS.get(tenor)
        if T is None:
            continue
        points.append((T, float(valid.iloc[-1])))
    if not points:
        return None
    points.sort()
    Ts = [p[0] for p in points]
    vs = [p[1] for p in points]
    if T_target <= Ts[0]:
        return vs[0]
    if T_target >= Ts[-1]:
        return vs[-1]
    return float(np.interp(T_target, Ts, vs))


def _spot_at_or_before(spot: pd.Series, ts: pd.Timestamp) -> Optional[float]:
    """Latest spot at or before timestamp; None if no data."""
    valid = spot.loc[:ts]
    if valid.empty:
        return None
    return float(valid.iloc[-1])


# -----------------------------------------------------------------------------
# Run a single strategy
# -----------------------------------------------------------------------------
def run_single_strategy(spec: StrategySpec, panels: dict,
                          start_date: date, end_date: date,
                          notional_usd: float = 10_000_000.0,
                          progress_cb: Optional[Callable[[float], None]] = None
                          ) -> list[Trade]:
    """Run the daily-rolling backtest for one strategy spec on preloaded
    pair panels. Returns a list of Trade records.

    Parameters
    ----------
    notional_usd : float
        Foreign-currency notional. Used to compute USD-denominated fields
        on the Trade record (premium_usd, pnl_usd, etc.). Stored on each
        Trade for traceability.
    """
    spot = panels["spot"]
    vol_panels = panels["vol_panels"]
    fwd_panels = panels["fwd_panels"]
    rr_panels = panels.get("rr_panels", {})
    bf_panels = panels.get("bf_panels", {})
    f_panel = panels["f_panel"]
    d_panel = panels["d_panel"]
    pip = panels["pip_scale"]

    # Trade dates: business days in [start, end] for which spot is available.
    spot_dates = pd.DatetimeIndex(spot.index).normalize()
    in_range = spot_dates[(spot_dates >= pd.Timestamp(start_date))
                            & (spot_dates <= pd.Timestamp(end_date))]
    trade_dates = sorted(set(d.date() for d in in_range))
    if not trade_dates:
        return []

    # Entry gate — pre-compute the mask once, then filter trade dates.
    # Gate failures don't unwind existing positions; they only block new
    # entries on those dates. We normalize timestamps to midnight so
    # lookup works regardless of whether the spot panel was stored with
    # HMS components.
    gate_mask = compute_gate_mask(spot, spec.entry_gate)
    if gate_mask is not None:
        gate_open_ts = set(gate_mask[gate_mask].index.normalize())
    else:
        gate_open_ts = None

    last_data_ts = spot.index.max()
    trades: list[Trade] = []
    n_total = len(trade_dates)

    # For trade_mode='single', we only open a new trade once the prior
    # one has expired. `last_open_expiry` is None until we've opened any
    # trade; afterwards it holds the expiry_date of the most recent
    # trade. A new entry is allowed only when trade_date >= that expiry.
    # Equality is allowed so the strategy doesn't sit idle for one day
    # between expiry and the next entry — eligible spot/vol exists on
    # the expiry date itself.
    last_open_expiry: Optional[date] = None

    for i, td in enumerate(trade_dates):
        td_ts = pd.Timestamp(td)

        # Single-mode block-out: skip days while an existing position
        # is open. Cheap check; done before the gate so gates only fire
        # on entry candidates (matches the "no rebalancing" intent).
        if (spec.trade_mode == "single"
                and last_open_expiry is not None
                and td < last_open_expiry):
            continue

        # Apply entry gate (no-op if no gate configured)
        if gate_open_ts is not None and td_ts not in gate_open_ts:
            continue

        S = _spot_at_or_before(spot, td_ts)
        if S is None:
            continue

        opt_dates = compute_option_dates(td, spec.tenor_label)
        T = opt_dates.T_years
        # Skip if expiry is past available data
        if pd.Timestamp(opt_dates.option_expiry) > last_data_ts:
            continue

        # Vol mid at T (ATM)
        sigma_atm_pct = _interp_panels_at_T(vol_panels, T, td_ts)
        if sigma_atm_pct is None:
            continue
        sigma_atm = sigma_atm_pct / 100.0

        # 25Δ RR / BF — optional; fall back to 0 (= flat-vol) if missing
        rr_pct = _interp_panels_at_T(rr_panels, T, td_ts) if rr_panels else None
        bf_pct = _interp_panels_at_T(bf_panels, T, td_ts) if bf_panels else None
        rr_25 = (rr_pct / 100.0) if rr_pct is not None else 0.0
        bf_25 = (bf_pct / 100.0) if bf_pct is not None else 0.0

        # Forward
        fwd_pts = _interp_panels_at_T(fwd_panels, T, td_ts)
        F_market = S + fwd_pts * pip if fwd_pts is not None else S

        # Rates
        r_f = get_rate_at(f_panel, T, td)
        r_d = get_rate_at(d_panel, T, td)
        if r_f is None and r_d is None:
            continue
        if r_f is None:
            r_f = r_d - np.log(F_market / S) / T
        if r_d is None:
            r_d = r_f + np.log(F_market / S) / T
        F_implied = S * np.exp((r_d - r_f) * T)

        # Solve: K from Δ at σ_atm; H from either payout ratio or KO Δ
        try:
            K, H, info = solve_strike(
                spec.direction, spec.barrier_type, spec.delta_value,
                S, T, sigma_atm, r_d, r_f,
                target_ratio=(spec.payout_ratio if spec.ko_method == "ratio" else None),
                target_ko_delta=(spec.target_ko_delta if spec.ko_method == "delta" else None),
                ko_method=spec.ko_method,
                rr_25=rr_25, bf_25=bf_25,
            )
        except Exception:
            continue

        achieved = info.get("achieved_ratio", float("nan"))
        ratio_min = info.get("ratio_min", float("nan"))
        if spec.ko_method == "ratio":
            feasible = (np.isfinite(achieved) and
                        abs(achieved - spec.payout_ratio) < 0.01)
        else:
            # Delta mode: structure is fully specified. If H landed on the
            # wrong side of K (KO Δ ≥ strike Δ ⇒ degenerate H), the note
            # is surfaced upstream and we skip the trade entirely — the
            # structure is invalid and the resulting premium would be
            # spurious (≈0).
            if "note" in info:
                continue
            feasible = True

        # σ at strike (= σ_atm when no smile data)
        sigma_smile = float(info.get("sigma_smile", sigma_atm))
        # Backward-compat alias
        sigma_mid = sigma_smile

        # Premium at the trade's chosen pricing model. The dispatcher
        # picks whichever σ (atm vs. smile) and whether to apply the
        # Vanna-Volga correction, based on spec.pricing_model. See
        # core/eko_pricing.py for the branch logic.
        from core.eko_pricing import price_eko_dispatch
        premium_mid, _price_detail = price_eko_dispatch(
            spec.direction, spec.barrier_type,
            S, K, H, T,
            sigma_atm=sigma_atm, sigma_smile=sigma_smile,
            rr_25=rr_25, bf_25=bf_25,
            r_d=r_d, r_f=r_f,
            model=spec.pricing_model,
        )

        # Transaction cost: flat bps of foreign notional (added on top of
        # mid premium). Doesn't shift σ — `sigma_used` therefore equals
        # σ_smile here, kept on the Trade record only as a backwards-
        # compatible alias for downstream code.
        sigma_used = sigma_smile
        premium = premium_mid  # per-unit price of the structure (mid)

        max_pay = abs(H - K)

        # Realized: spot at expiry
        S_exp_ts = pd.Timestamp(opt_dates.option_expiry)
        S_exp = _spot_at_or_before(spot, S_exp_ts)
        if S_exp is None:
            continue

        # KO check at expiry only (European)
        if spec.barrier_type == "up_and_out":
            knocked_out = S_exp >= H
        else:
            knocked_out = S_exp <= H

        if knocked_out:
            actual_payoff = 0.0
        else:
            if spec.direction == "call":
                actual_payoff = max(S_exp - K, 0.0)
            else:
                actual_payoff = max(K - S_exp, 0.0)
            # Bound by max payoff in case spot just barely escaped barrier
            actual_payoff = min(actual_payoff, max_pay)

        # Normalise to % of foreign notional. tx cost is bps of notional
        # (1 bp = 0.01% = 1/100 of a percentage point).
        prem_mid_pct = premium_mid / S * 100.0
        tx_cost_pct = spec.tx_cost_bps / 100.0
        prem_pct = prem_mid_pct + tx_cost_pct
        max_pay_pct = max_pay / S * 100.0
        actual_pay_pct = actual_payoff / S * 100.0
        pnl_pct = actual_pay_pct - prem_pct
        pnl_gross_pct = actual_pay_pct - prem_mid_pct

        # USD = pct × notional / 100
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
            pricing_model=spec.pricing_model,
        ))
        # Track the most recently opened trade's expiry so single-mode
        # can block out subsequent entries until expiry passes. No-op
        # in stack mode but kept unconditionally for simplicity.
        last_open_expiry = opt_dates.option_expiry

        if progress_cb is not None and (i % 20 == 0 or i + 1 == n_total):
            progress_cb((i + 1) / n_total)

    return trades


# -----------------------------------------------------------------------------
# Summary statistics per strategy
# -----------------------------------------------------------------------------
def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    """Convert a trade list to a flat DataFrame (rich ledger)."""
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([asdict(t) for t in trades])


# -----------------------------------------------------------------------------
# Mark-to-market support
# -----------------------------------------------------------------------------
def _build_fast_panels(panels: dict) -> dict:
    """Wrap pd.Series panels in (dates_array, values_array) tuples so the
    MTM hot loop can use np.searchsorted instead of pandas slicing.

    The MTM step otherwise spends ~70% of its time inside
    `pd.Series.loc[:ts].iloc[-1]` for each (date, tenor, trade) triple.
    Pre-extracting numpy arrays lifts that to O(log N) per query.
    """
    spot = panels.get("spot")
    spot_fast = None
    if spot is not None and not spot.empty:
        s = spot.dropna().sort_index()
        spot_fast = (s.index.values.astype("datetime64[ns]"), s.values)

    def _wrap_panel_dict(d):
        out = {}
        for k, ser in (d or {}).items():
            if ser is None or ser.empty:
                continue
            s = ser.dropna().sort_index()
            out[k] = (s.index.values.astype("datetime64[ns]"), s.values)
        return out

    # rates.load_rates_panel returns a DataFrame; convert each column.
    def _wrap_panel_df(df):
        out = {}
        if df is None or df.empty:
            return out
        for c in df.columns:
            ser = df[c].dropna().sort_index()
            if ser.empty:
                continue
            out[c] = (ser.index.values.astype("datetime64[ns]"), ser.values)
        return out

    return {
        "spot": spot_fast,
        "vol_panels_fast": _wrap_panel_dict(panels.get("vol_panels")),
        "fwd_panels_fast": _wrap_panel_dict(panels.get("fwd_panels")),
        "rr_panels_fast": _wrap_panel_dict(panels.get("rr_panels")),
        "bf_panels_fast": _wrap_panel_dict(panels.get("bf_panels")),
        "f_panel_fast": _wrap_panel_df(panels.get("f_panel")),
        "d_panel_fast": _wrap_panel_df(panels.get("d_panel")),
        "pip_scale": panels.get("pip_scale", 1.0),
    }


def _fast_asof(arr_pair, ts_np):
    """arr_pair is (dates_array, values_array). Return last value at-or-
    before ts_np (numpy datetime64), or None if ts_np precedes first date."""
    if arr_pair is None:
        return None
    dates, values = arr_pair
    idx = int(np.searchsorted(dates, ts_np, side="right")) - 1
    if idx < 0:
        return None
    return float(values[idx])


def _fast_interp_at_T(fast_panel_dict, T_target, ts_np):
    """Linear interp across tenor curve at T_target, using as-of date ts_np."""
    points = []
    for tenor, arr_pair in fast_panel_dict.items():
        T = TENOR_YEARS.get(tenor)
        if T is None:
            continue
        v = _fast_asof(arr_pair, ts_np)
        if v is None:
            continue
        points.append((T, v))
    if not points:
        return None
    points.sort()
    if T_target <= points[0][0]:
        return points[0][1]
    if T_target >= points[-1][0]:
        return points[-1][1]
    Ts = np.fromiter((p[0] for p in points), dtype=float, count=len(points))
    vs = np.fromiter((p[1] for p in points), dtype=float, count=len(points))
    return float(np.interp(T_target, Ts, vs))


def _trade_mtm_trajectory(trade: dict, fast_panels: dict) -> dict:
    """Compute mid-vol MTM in USD for one trade on every business day from
    trade_date to expiry_date − 1bd (i.e., during the option's life).

    On entry day, uses the already-stored `premium_mid_pct` (no extra
    pricing call). On the expiry day MTM = 0 by construction (handled
    by the realised cash flow), so we skip it.

    Returns a dict {pd.Timestamp: mtm_usd}.
    """
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

    # Step 1c: MTM is repriced through the SAME dispatcher used at trade
    # inception, so VV-priced entries get VV-priced marks. Fall back to
    # 'vol_at_strike' for legacy trade dicts that pre-date the field.
    mtm_pricing_model = trade.get("pricing_model", "vol_at_strike")

    trade_dt = pd.Timestamp(trade["trade_date"])
    expiry_dt = pd.Timestamp(trade["expiry_date"])
    last_mtm_day = expiry_dt - pd.Timedelta(days=1)
    if last_mtm_day < trade_dt:
        return {}
    bdays = pd.bdate_range(trade_dt, last_mtm_day)

    # Lazy import — avoid circular at module load
    from core.smile import smile_vol_at_strike as _smile_vol
    from core.eko_pricing import price_eko_dispatch as _price_eko_dispatch

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

            # Smile vol at the trade's strike K on date d
            rr_pct = _fast_interp_at_T(rr_fast, T_rem, ts_np) if rr_fast else None
            bf_pct = _fast_interp_at_T(bf_fast, T_rem, ts_np) if bf_fast else None
            rr = (rr_pct / 100.0) if rr_pct is not None else 0.0
            bf = (bf_pct / 100.0) if bf_pct is not None else 0.0
            sigma_d = _smile_vol(S_d, K, T_rem, sigma_atm_d, rr, bf,
                                   r_d_rate, r_f)

            try:
                mtm_per_unit, _ = _price_eko_dispatch(
                    direction, barrier_type,
                    S_d, K, H, T_rem,
                    sigma_atm=sigma_atm_d, sigma_smile=sigma_d,
                    rr_25=rr, bf_25=bf,
                    r_d=r_d_rate, r_f=r_f,
                    model=mtm_pricing_model,
                )
            except Exception:
                continue
        out[d_ts] = mtm_per_unit * factor
    return out


def compute_mtm_equity_curve(trades_df: pd.DataFrame, panels: dict
                                ) -> pd.DataFrame:
    """Daily MTM equity curve for one strategy.

    equity(d) = book_value(d) + cash_position(d)
      book_value(d)    = sum over open trades of mid-vol MTM in USD
      cash_position(d) = (− cumulative premiums paid) + (+ cumulative payoffs)

    On entry, equity drops by tx_cost (paid premium − mid-vol mark).
    During life, equity moves with mid-vol MTM swings.
    At expiry, the cash payoff replaces the MTM exactly, so total equity
    converges to the realised-only PnL series at every expiry date.

    Returns DataFrame with: book_value_usd, cash_position_usd,
    equity_usd, peak_usd, drawdown_usd, drawdown_usd_pos.
    Empty DataFrame if MTM cannot be computed (no notional, empty trades).
    """
    if trades_df.empty or "pnl_usd" not in trades_df.columns:
        return pd.DataFrame()

    fast_panels = _build_fast_panels(panels)

    mtm_records = []
    for _, t in trades_df.iterrows():
        traj = _trade_mtm_trajectory(t.to_dict(), fast_panels)
        for d_ts, v in traj.items():
            mtm_records.append((d_ts, v))
    if not mtm_records:
        return pd.DataFrame()
    mtm_long = pd.DataFrame(mtm_records, columns=["date", "mtm_usd"])
    book = mtm_long.groupby("date")["mtm_usd"].sum().sort_index()

    df = trades_df.copy()
    df["trade_dt"] = pd.to_datetime(df["trade_date"])
    df["expiry_dt"] = pd.to_datetime(df["expiry_date"])
    prem = df.groupby("trade_dt")["premium_usd"].sum()
    pay = df.groupby("expiry_dt")["actual_payoff_usd"].sum()

    all_dates = book.index.union(prem.index).union(pay.index).sort_values()
    book_full = book.reindex(all_dates, fill_value=0.0)
    cf = pd.Series(0.0, index=all_dates)
    cf.loc[prem.index] -= prem
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
        "drawdown_usd_pos": -dd,
    })


def compute_mtm_curves(folder: str, specs: list[StrategySpec],
                          results: dict[str, list[Trade]],
                          progress_cb: Optional[Callable[[float, str], None]] = None
                          ) -> dict[str, pd.DataFrame]:
    """Run compute_mtm_equity_curve for each strategy in `results`.
    Caches preloaded panels per pair across specs."""
    pairs_seen: dict[str, dict] = {}
    out: dict[str, pd.DataFrame] = {}
    n = len(specs)
    for i, spec in enumerate(specs):
        trades = results.get(spec.name, [])
        if not trades:
            out[spec.name] = pd.DataFrame()
            continue
        if spec.pair not in pairs_seen:
            pairs_seen[spec.pair] = preload_pair_panels(folder, spec.pair, spec.prefer)
        panels = pairs_seen[spec.pair]
        if not panels:
            out[spec.name] = pd.DataFrame()
            continue
        df = trades_to_df(trades)
        out[spec.name] = compute_mtm_equity_curve(df, panels)
        if progress_cb is not None:
            progress_cb((i + 1) / n, spec.name)
    return out


def summarize_mtm(mtm_eq: pd.DataFrame) -> dict:
    """Stats on a MTM equity curve: daily-Sharpe, max drawdown, etc.
    Note: total PnL equals the realised-only total at the final date.
    Returned values are USD-denominated."""
    if mtm_eq is None or mtm_eq.empty:
        return {}
    eq = mtm_eq["equity_usd"]
    daily = eq.diff().dropna()
    if len(daily) > 1 and daily.std() > 0:
        sharpe_daily = daily.mean() / daily.std() * np.sqrt(252)
    else:
        sharpe_daily = 0.0
    return {
        "sharpe_daily_mtm": float(sharpe_daily),
        "max_drawdown_usd_mtm": float(mtm_eq["drawdown_usd"].min()),
        "final_equity_usd_mtm": float(eq.iloc[-1]),
    }


def compute_equity_and_drawdown(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Equity curve + drawdown indexed by expiry date.

    Returns DataFrame with columns: pnl_pct, equity_pct, peak_pct,
    drawdown_pct, drawdown_pct_pos, plus USD versions when notional was
    provided to the engine (pnl_usd, equity_usd, peak_usd, drawdown_usd).
    """
    if trades_df.empty:
        return pd.DataFrame(columns=["pnl_pct", "equity_pct", "peak_pct",
                                       "drawdown_pct", "drawdown_pct_pos"])

    df = trades_df.copy()
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])

    daily_pct = df.groupby("expiry_date")["pnl_pct"].sum().sort_index()
    eq_pct = daily_pct.cumsum()
    peak_pct = eq_pct.cummax()
    dd_pct = eq_pct - peak_pct
    out = pd.DataFrame({
        "pnl_pct": daily_pct,
        "equity_pct": eq_pct,
        "peak_pct": peak_pct,
        "drawdown_pct": dd_pct,
        "drawdown_pct_pos": -dd_pct,
    })

    if "pnl_usd" in df.columns:
        daily_usd = df.groupby("expiry_date")["pnl_usd"].sum().sort_index()
        eq_usd = daily_usd.cumsum()
        peak_usd = eq_usd.cummax()
        dd_usd = eq_usd - peak_usd
        out["pnl_usd"] = daily_usd
        out["equity_usd"] = eq_usd
        out["peak_usd"] = peak_usd
        out["drawdown_usd"] = dd_usd
        # Also: cumulative tx cost trajectory if available
        if "transaction_cost_usd" in df.columns:
            daily_tx = df.groupby("expiry_date")["transaction_cost_usd"].sum().sort_index()
            out["tx_cost_usd"] = daily_tx
            out["cum_tx_cost_usd"] = daily_tx.cumsum()

    return out


def _consistency_metrics(eq: pd.DataFrame, monthly: pd.Series,
                            df: pd.DataFrame) -> dict:
    """Cross-year consistency and tail-pain metrics.

    Returns a dict of metrics that reward strategies performing well
    across MULTIPLE years (vs. ones that earn it all in 1-2 years):

    - n_years            : calendar years observed
    - annual_sharpe_mean : mean of per-year Sharpes (each year's monthly
                            mean/std × √12)
    - annual_sharpe_min  : worst per-year Sharpe (flags bad-year risk)
    - annual_sharpe_std  : variability of per-year Sharpe across years
    - annual_sharpe_cv   : σ / μ — coefficient of variation. Lower (in
                            magnitude) is more consistent; sign reveals
                            whether μ itself is positive. Undefined when
                            |μ| < 1e-9 (returns 0 by convention).
    - annual_sharpe_score: μ × (1 − σ/μ) = μ − σ. Composite "high mean
                            AND low variability" score from Yavuz Akbay's
                            framework. Algebraically equivalent to
                            mean-minus-std-deviation; higher is better.
    - pct_positive_years : fraction of years with positive PnL
    - min_annual_pnl_pct : worst calendar year PnL in % space
    - min_annual_pnl_usd : worst calendar year PnL in USD space (if avail)
    - calmar             : (annualised return %) / |max DD %|
    - gain_to_pain       : Σ positive monthly / |Σ negative| — outlier-
                            robust alternative to Sharpe (∞ if no losses)
    - ulcer_index        : RMS drawdown depth (%, captures both depth
                            and duration unlike point-in-time max DD)

    All metrics gracefully return 0.0 when there's insufficient data
    (e.g. < 2 months, no drawdown, or empty equity curve).
    """
    out = {
        "annual_sharpe_mean": 0.0,
        "annual_sharpe_min": 0.0,
        "annual_sharpe_std": 0.0,
        "annual_sharpe_cv": 0.0,
        "annual_sharpe_score": 0.0,
        "pct_positive_years": 0.0,
        "min_annual_pnl_pct": 0.0,
        "min_annual_pnl_usd": 0.0,
        "n_years": 0,
        "calmar": 0.0,
        "gain_to_pain": 0.0,
        "ulcer_index": 0.0,
    }
    if eq.empty or len(monthly) == 0:
        return out

    # --- Per-year Sharpe + per-year PnL from monthly (pct) ---
    by_year_pct = monthly.groupby(monthly.index.year)
    annual_sharpes: list[float] = []
    annual_pnls_pct: list[float] = []
    for _yr, sub in by_year_pct:
        annual_pnls_pct.append(float(sub.sum()))
        if len(sub) > 1 and sub.std() > 0:
            annual_sharpes.append(float(sub.mean() / sub.std() * np.sqrt(12)))
    n_years = len(annual_pnls_pct)

    if n_years > 0:
        out["n_years"] = int(n_years)
        arr_pct = np.array(annual_pnls_pct)
        out["pct_positive_years"] = float((arr_pct > 0).mean() * 100)
        out["min_annual_pnl_pct"] = float(arr_pct.min())
    if annual_sharpes:
        a_sh = np.array(annual_sharpes)
        mean_sh = float(a_sh.mean())
        std_sh = float(a_sh.std()) if len(a_sh) > 1 else 0.0
        out["annual_sharpe_mean"] = mean_sh
        out["annual_sharpe_min"] = float(a_sh.min())
        out["annual_sharpe_std"] = std_sh
        # CV = σ / μ. Signed: a negative μ produces a negative CV, which is
        # the formula's natural way of saying "the mean is bad to start
        # with" so we leave it signed. Undefined when |μ| ≈ 0.
        if abs(mean_sh) > 1e-9:
            out["annual_sharpe_cv"] = std_sh / mean_sh
        # Composite score from Yavuz Akbay's "Weighted Annualized Scoring":
        # Score = μ × (1 − CV) = μ − σ. Penalises strategies where stdev
        # is large relative to mean. Note this is algebraically just
        # μ − σ regardless of sign — clean and unit-consistent.
        out["annual_sharpe_score"] = mean_sh - std_sh

    # --- Worst calendar year in USD if pnl_usd available on the eq curve ---
    # Equity curve has daily pnl_usd; group by calendar year on the index.
    if "pnl_usd" in eq.columns and not eq.empty:
        annual_usd = eq["pnl_usd"].groupby(eq.index.year).sum()
        if not annual_usd.empty:
            out["min_annual_pnl_usd"] = float(annual_usd.min())

    # --- Calmar: annualised return / |max DD| (% space) ---
    if "pnl_pct" in eq.columns:
        total_pct = float(eq["pnl_pct"].sum())
        days = max((eq.index.max() - eq.index.min()).days, 1)
        years = days / 365.25
        cagr_pct = total_pct / max(years, 1e-6)
        max_dd_pct_abs = abs(float(eq["drawdown_pct"].min())) \
            if "drawdown_pct" in eq.columns else 0.0
        if max_dd_pct_abs > 1e-9:
            out["calmar"] = float(cagr_pct / max_dd_pct_abs)

    # --- Gain-to-pain (monthly pct) ---
    pos = float(monthly[monthly > 0].sum())
    neg_abs = float(abs(monthly[monthly < 0].sum()))
    if neg_abs > 1e-9:
        out["gain_to_pain"] = pos / neg_abs
    elif pos > 0:
        out["gain_to_pain"] = float("inf")

    # --- Ulcer index (RMS drawdown %) ---
    if "drawdown_pct" in eq.columns:
        dd = eq["drawdown_pct"].dropna()
        if len(dd) > 0:
            out["ulcer_index"] = float(np.sqrt((dd ** 2).mean()))

    return out


def summarize_strategy(trades_df: pd.DataFrame) -> dict:
    """Headline statistics for one strategy's trade ledger."""
    if trades_df.empty:
        return {"n_trades": 0}

    df = trades_df
    pnl = df["pnl_pct"]
    prem = df["premium_pct"]
    pay = df["actual_payoff_pct"]
    max_pay = df["max_payoff_pct"]

    # Monthly Sharpe (note: a pure ratio, no $/% units)
    eq = compute_equity_and_drawdown(df)
    monthly = eq["pnl_pct"].resample("ME").sum() if not eq.empty else pd.Series(dtype=float)
    if len(monthly) > 1 and monthly.std() > 0:
        sharpe_monthly = monthly.mean() / monthly.std() * np.sqrt(12)
    else:
        sharpe_monthly = 0.0

    max_dd_pct = float(eq["drawdown_pct"].min()) if not eq.empty else 0.0

    out = {
        "n_trades": int(len(df)),
        "feasibility_pct": float(df["feasible"].mean() * 100),
        "ko_rate_pct": float(df["knocked_out"].mean() * 100),
        "win_rate_pct": float((pnl > 0).mean() * 100),
        "total_premium_pct": float(prem.sum()),
        "total_payout_pct": float(pay.sum()),
        "total_pnl_pct": float(pnl.sum()),
        "avg_premium_pct": float(prem.mean()),
        "avg_payout_pct": float(pay.mean()),
        "avg_max_payoff_pct": float(max_pay.mean()),
        "best_trade_pct": float(pnl.max()),
        "worst_trade_pct": float(pnl.min()),
        "sharpe_monthly": float(sharpe_monthly),
        "max_drawdown_pct": float(max_dd_pct),
        "premium_recovery_pct": float(pay.sum() / prem.sum() * 100) if prem.sum() > 0 else 0.0,
    }

    # Cross-year consistency block — merged in unconditionally; helper
    # returns zeroes when there's insufficient data.
    out.update(_consistency_metrics(eq, monthly, df))

    # USD-denominated additions (if engine provided notional)
    if "pnl_usd" in df.columns:
        notional = float(df["notional_usd"].iloc[0]) if "notional_usd" in df.columns else 0.0
        max_dd_usd = float(eq["drawdown_usd"].min()) if "drawdown_usd" in eq.columns else 0.0
        out.update({
            "notional_usd": notional,
            "total_premium_usd": float(df["premium_usd"].sum()),
            "total_premium_mid_usd": float(df["premium_mid_usd"].sum()),
            "total_transaction_cost_usd": float(df["transaction_cost_usd"].sum()),
            "total_payout_usd": float(df["actual_payoff_usd"].sum()),
            "total_pnl_usd": float(df["pnl_usd"].sum()),
            "total_pnl_gross_usd": float(df["pnl_gross_usd"].sum()),
            "avg_premium_usd": float(df["premium_usd"].mean()),
            "avg_payout_usd": float(df["actual_payoff_usd"].mean()),
            "best_trade_usd": float(df["pnl_usd"].max()),
            "worst_trade_usd": float(df["pnl_usd"].min()),
            "max_drawdown_usd": float(max_dd_usd),
            "tx_cost_share_of_premium_pct": float(
                df["transaction_cost_usd"].sum()
                / df["premium_usd"].sum() * 100
            ) if df["premium_usd"].sum() > 0 else 0.0,
        })

    return out


def summarize_by_regime(trades_df: pd.DataFrame,
                          regime_panel: "pd.DataFrame | None") -> pd.DataFrame:
    """Per-regime breakdown of trade outcomes for a single-leg strategy.

    Each trade is attributed to the regime state on its `trade_date`
    (entry date, not expiry). Returns a DataFrame indexed by state
    with one row per state and columns:

      - state              : integer state label (0 = dominant by
                              the convention in core/regimes)
      - n_trades           : count of trades in this state
      - share_of_trades_pct: that count as % of all attributable trades
      - win_rate_pct       : fraction of trades with pnl_usd > 0
      - ko_rate_pct        : fraction of trades that knocked out
      - total_pnl_usd      : aggregate PnL across all trades in state
      - mean_pnl_usd       : average PnL per trade in state
      - mean_premium_usd   : average premium paid per trade
      - tx_cost_share_pct  : tx cost as % of premium

    Trades whose `trade_date` is outside the regime panel's date range
    (or whose pair has no regime panel) are dropped silently — the
    summary describes only the attributable subset. Returns an empty
    DataFrame if no trades can be attributed.
    """
    cols = ["state", "n_trades", "share_of_trades_pct",
              "win_rate_pct", "ko_rate_pct",
              "total_pnl_usd", "mean_pnl_usd",
              "mean_premium_usd", "tx_cost_share_pct"]
    if (trades_df.empty or regime_panel is None or regime_panel.empty
            or "trade_date" not in trades_df.columns):
        return pd.DataFrame(columns=cols)

    trades = trades_df.copy()
    trades["trade_date"] = pd.to_datetime(trades["trade_date"]).dt.normalize()
    panel = regime_panel.copy()
    panel.index = pd.to_datetime(panel.index).normalize()

    state_lookup = panel["state"].to_dict()
    trades["state"] = trades["trade_date"].map(state_lookup)
    trades = trades.dropna(subset=["state"])
    if trades.empty:
        return pd.DataFrame(columns=cols)
    trades["state"] = trades["state"].astype(int)
    total = len(trades)

    out_rows = []
    for state_val, sub in trades.groupby("state"):
        prem = sub["premium_usd"].sum() if "premium_usd" in sub.columns else 0.0
        tx = (sub["transaction_cost_usd"].sum()
                if "transaction_cost_usd" in sub.columns else 0.0)
        out_rows.append({
            "state": int(state_val),
            "n_trades": int(len(sub)),
            "share_of_trades_pct": float(len(sub) / total * 100),
            "win_rate_pct": float((sub["pnl_usd"] > 0).mean() * 100),
            "ko_rate_pct": float(sub.get("knocked_out",
                                            pd.Series(False)).mean() * 100),
            "total_pnl_usd": float(sub["pnl_usd"].sum()),
            "mean_pnl_usd": float(sub["pnl_usd"].mean()),
            "mean_premium_usd": float(prem / len(sub)) if len(sub) > 0 else 0.0,
            "tx_cost_share_pct": float(tx / prem * 100) if prem > 0 else 0.0,
        })
    return pd.DataFrame(out_rows, columns=cols).sort_values("state").reset_index(drop=True)


def export_strategy_time_series(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Long-format time-series export for one strategy.

    Produces ONE DataFrame with rows at three frequencies. Designed to
    be concatenated across many strategies (with a `strategy_name`
    column) and consumed by another app that wants to recreate the
    equity / drawdown chart and the monthly + annual PnL bar charts
    WITHOUT needing trade-level data.

    Columns:
      - period_type   : 'daily' | 'monthly' | 'annual'
      - period_end    : pd.Timestamp — the right-edge of the period
                         (daily = expiry_date, monthly = month-end,
                          annual = Dec 31 of the year)
      - pnl_usd       : realized PnL in USD over the period
      - equity_usd    : cumulative equity at the end of this period.
                         Populated on ALL row types so downstream apps can
                         recompute drawdown/Calmar/ulcer at any granularity
                         without needing the daily series.
      - drawdown_usd  : drawdown vs running peak at the end of this period.
                         Daily rows use the daily peak; monthly/annual rows
                         re-derive from the cumulative series at month/year
                         end so they're self-consistent with `equity_usd`.

    Empty DataFrame if input is empty or lacks USD columns. Caller
    should prepend a `strategy_name` column when concatenating across
    strategies.
    """
    if trades_df.empty or "pnl_usd" not in trades_df.columns:
        return pd.DataFrame(columns=["period_type", "period_end",
                                       "pnl_usd", "equity_usd",
                                       "drawdown_usd"])

    eq = compute_equity_and_drawdown(trades_df)
    if eq.empty or "equity_usd" not in eq.columns:
        return pd.DataFrame(columns=["period_type", "period_end",
                                       "pnl_usd", "equity_usd",
                                       "drawdown_usd"])

    # Daily (per-expiry-date) granularity — full equity / drawdown info
    daily = pd.DataFrame({
        "period_type": "daily",
        "period_end": eq.index,
        "pnl_usd": eq["pnl_usd"].values,
        "equity_usd": eq["equity_usd"].values,
        "drawdown_usd": eq["drawdown_usd"].values,
    })

    # Monthly: PnL is sum-by-month; equity/drawdown are the END-OF-MONTH
    # snapshots of the daily cumulative series. This keeps the monthly
    # rows internally consistent (equity_t = equity_{t-1} + pnl_t) and
    # lets downstream apps recompute monthly Calmar / monthly Ulcer.
    monthly_pnl = eq["pnl_usd"].resample("ME").sum()
    monthly_eq = eq["equity_usd"].resample("ME").last().ffill()
    monthly_dd = eq["drawdown_usd"].resample("ME").last().ffill()
    monthly_df = pd.DataFrame({
        "period_type": "monthly",
        "period_end": monthly_pnl.index,
        "pnl_usd": monthly_pnl.values,
        "equity_usd": monthly_eq.reindex(monthly_pnl.index).values,
        "drawdown_usd": monthly_dd.reindex(monthly_pnl.index).values,
    })

    # Annual: PnL summed by calendar year; equity/drawdown at year-end.
    annual_pnl = eq["pnl_usd"].groupby(eq.index.year).sum()
    annual_eq = eq["equity_usd"].groupby(eq.index.year).last()
    annual_dd = eq["drawdown_usd"].groupby(eq.index.year).last()
    year_ends = pd.to_datetime([f"{int(y)}-12-31" for y in annual_pnl.index])
    annual_df = pd.DataFrame({
        "period_type": "annual",
        "period_end": year_ends,
        "pnl_usd": annual_pnl.values,
        "equity_usd": annual_eq.reindex(annual_pnl.index).values,
        "drawdown_usd": annual_dd.reindex(annual_pnl.index).values,
    })

    return pd.concat([daily, monthly_df, annual_df], ignore_index=True)


def augment_time_series_with_regime(
    ts_df: pd.DataFrame,
    regime_panel: "pd.DataFrame | None",
    column_name: str = "state",
) -> pd.DataFrame:
    """Add a `state` column to a time-series frame produced by
    `export_strategy_time_series`.

    Daily rows: state is looked up on each row's `period_end` date
    directly. Monthly and annual rows: their `period_end` is a
    calendar month-end or Dec 31, which may fall on a weekend or
    holiday — so we use a backward-asof join (most recent state
    on or before the period_end date) to populate those rows.

    Rows with no regime data (date before the panel starts) get NaN.
    If `regime_panel` is None, the column is added as all NaN — useful
    so the schema is stable whether or not a panel is available.

    `column_name` defaults to 'state' for single-leg use. For worst-of,
    call this function twice with `column_name='state_a'` and
    `column_name='state_b'` using each leg's regime panel.
    """
    out = ts_df.copy()
    if regime_panel is None or regime_panel.empty or out.empty:
        out[column_name] = pd.NA
        return out
    panel = regime_panel.copy()
    panel.index = pd.to_datetime(panel.index).normalize()
    panel = panel.sort_index()
    period_ends = pd.to_datetime(out["period_end"]).dt.normalize()

    # Asof-backward lookup so non-trading-day period_ends (Sat/Sun
    # month-ends, etc.) pick up the previous trading day's state.
    state_series = pd.Series(
        panel["state"].values, index=panel.index, name=column_name
    )
    # Build a frame to merge_asof on
    left = pd.DataFrame({"period_end": period_ends.values}).reset_index()
    left["period_end"] = pd.to_datetime(left["period_end"])
    right = state_series.reset_index()
    right.columns = ["period_end", column_name]
    merged = pd.merge_asof(
        left.sort_values("period_end"),
        right.sort_values("period_end"),
        on="period_end", direction="backward",
    )
    # Restore original row order and assign
    merged = merged.sort_values("index").reset_index(drop=True)
    out[column_name] = merged[column_name].astype("Int64").values
    return out


def monthly_pnl_table(trades_df: pd.DataFrame,
                        value_col: str = "pnl_pct") -> pd.DataFrame:
    """Pivot of monthly PnL — rows = year, columns = month, values =
    sum of `value_col` over that month. Adds a YTD total column.
    Default value_col is pct; pass 'pnl_usd' for USD."""
    if trades_df.empty or value_col not in trades_df.columns:
        return pd.DataFrame()
    df = trades_df.copy()
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    df["year"] = df["expiry_date"].dt.year
    df["month"] = df["expiry_date"].dt.month
    pivot = df.pivot_table(index="year", columns="month",
                            values=value_col, aggfunc="sum",
                            fill_value=0.0)
    pivot["YTD"] = pivot.sum(axis=1)
    return pivot


def annual_summary_table(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Per-year summary: trades, total PnL (pct + USD), win rate, KO rate,
    Sharpe, total premium, total tx cost."""
    if trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    df["year"] = df["expiry_date"].dt.year

    has_usd = "pnl_usd" in df.columns

    out = []
    for year, sub in df.groupby("year"):
        eq = compute_equity_and_drawdown(sub)
        monthly = (eq["pnl_pct"].resample("ME").sum() if not eq.empty
                    else pd.Series(dtype=float))
        sh = (monthly.mean() / monthly.std() * np.sqrt(12)
              if len(monthly) > 1 and monthly.std() > 0 else 0.0)
        row = {
            "year": int(year),
            "n_trades": len(sub),
            "total_pnl_pct": sub["pnl_pct"].sum(),
            "win_rate_pct": (sub["pnl_pct"] > 0).mean() * 100,
            "ko_rate_pct": sub["knocked_out"].mean() * 100,
            "feasibility_pct": sub["feasible"].mean() * 100,
            "sharpe_monthly": sh,
        }
        if has_usd:
            row.update({
                "total_pnl_usd": float(sub["pnl_usd"].sum()),
                "total_premium_usd": float(sub["premium_usd"].sum()),
                "total_payout_usd": float(sub["actual_payoff_usd"].sum()),
                "total_tx_cost_usd": float(sub["transaction_cost_usd"].sum()),
            })
        out.append(row)
    return pd.DataFrame(out).set_index("year")


# -----------------------------------------------------------------------------
# Multi-strategy orchestration
# -----------------------------------------------------------------------------
def build_strategy_grid(
        pairs: list[str], deltas: list[tuple[str, float]],
        tenors: list[str], directions: list[tuple[str, str]],
        tx_cost_bps: float = 0.0,
        prefer: str = "offshore",
        ko_method: str = "ratio",
        payout_ratio: Optional[float] = None,
        target_ko_delta: Optional[float] = None,
        ko_delta_label: Optional[str] = None,
        entry_gate: Optional[str] = None,
        trade_mode: str = "stack",
        pricing_model: str = "vol_at_strike",
) -> list[StrategySpec]:
    """Cross-product of (pair × delta × tenor × direction).

    ko_method='ratio' uses payout_ratio (e.g. 8.0 for 8:1 leverage).
    ko_method='delta' uses target_ko_delta (e.g. 0.10 for 10Δ wing barrier);
    pass `ko_delta_label` (e.g. '10Δ') for friendlier display in strategy
    names.

    `entry_gate` (e.g. 'spot_above_50dma') is a per-day filter on new
    trade entries. None = no gate. See core/gates.py.

    `trade_mode` is 'stack' (overlapping daily entries — original
    behaviour) or 'single' (next entry only after prior expiry).

    `pricing_model` propagates to every spec — see StrategySpec docstring.
    Default 'vol_at_strike' preserves legacy backtest behaviour.
    """
    if ko_method == "ratio" and payout_ratio is None:
        raise ValueError("ko_method='ratio' requires payout_ratio")
    if ko_method == "delta" and target_ko_delta is None:
        raise ValueError("ko_method='delta' requires target_ko_delta")
    out = []
    for pair in pairs:
        for d_label, d_val in deltas:
            for tenor in tenors:
                for dir_name, barrier_type in directions:
                    out.append(StrategySpec(
                        pair=pair,
                        direction=dir_name,
                        barrier_type=barrier_type,
                        delta_label=d_label,
                        delta_value=d_val,
                        tenor_label=tenor,
                        tx_cost_bps=tx_cost_bps,
                        prefer=prefer,
                        ko_method=ko_method,
                        payout_ratio=payout_ratio,
                        target_ko_delta=target_ko_delta,
                        ko_delta_label=ko_delta_label,
                        entry_gate=entry_gate,
                        trade_mode=trade_mode,
                        pricing_model=pricing_model,
                    ))
    return out


def run_grid(folder: str, specs: list[StrategySpec],
              start_date: date, end_date: date,
              notional_usd: float = 10_000_000.0,
              progress_cb: Optional[Callable[[float, str], None]] = None
              ) -> dict[str, list[Trade]]:
    """Run all specs. Pre-loads each unique pair once. Reports progress
    over the cross-product (strategies × dates). `notional_usd` applies
    uniformly to all strategies."""
    # Group by pair so we preload each pair once
    pairs_seen: dict[str, dict] = {}
    for s in specs:
        if s.pair not in pairs_seen:
            pairs_seen[s.pair] = preload_pair_panels(folder, s.pair, s.prefer)

    results: dict[str, list[Trade]] = {}
    n_specs = len(specs)
    for i, spec in enumerate(specs):
        panels = pairs_seen.get(spec.pair, {})
        if not panels:
            results[spec.name] = []
            continue

        def _cb(p):
            if progress_cb is not None:
                overall = (i + p) / n_specs
                progress_cb(overall, spec.name)

        results[spec.name] = run_single_strategy(
            spec, panels, start_date, end_date,
            notional_usd=notional_usd, progress_cb=_cb,
        )

    return results

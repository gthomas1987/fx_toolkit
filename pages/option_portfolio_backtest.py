"""Option Portfolio Backtest — basket backtester for multi-pair, multi-
type option strategies.

# Concept
Each strategy is one (pair, type) combination, all sharing the same
tenor / strike Δ / barrier offset / WO multiplier / side / notional.
The "portfolio" is the equal-notional sum of every strategy's daily-
rolling P&L.

# Supported types
- Vanilla call / Vanilla put          → core.vanilla_backtest
- EKO U&O / EKO D&O                    → core.backtest.run_single_strategy
- RKO U&O / RKO D&O                    → core.backtest_american.run_single_strategy_american
- WO-EKO                                → core.worstof._run_worstof_with_panels
- WO-RKO                                → core.worstof._run_worstof_with_panels
                                          (with ko_check_mode='american_ohlc')

# Strategy generation
For pairs `{P1, P2, P3}` and selected types `{Vanilla call, WO-EKO}`:
  - Vanilla call ⇒ 3 strategies (one per pair)
  - WO-EKO       ⇒ C(3,2)=3 strategies (pair combinations)
Total: 6 strategies, each with the same USD notional.

# Aggregation
After each strategy runs, its `Trade`/`VanillaTrade`/`WorstOfTrade`
list is collapsed into a daily P&L stream keyed by trade_date.
Equal-weighted sum produces the portfolio curve.

# Drilldown tabs
1. Portfolio P&L     — cumulative curve + metrics (Sharpe, MDD, win%, skew)
2. Per-strategy      — sortable table; click for trade-level details
3. Per-pair          — aggregate by currency pair
4. Per-type          — aggregate by strategy type (Vanilla/EKO/RKO/WO-*)

# What's NOT here
- Mark-to-market (we use the realized P&L at expiry per trade)
- Notional-sizing rules other than equal-notional (per user choice)
- Strategy types other than the seven listed (no spreads, flies, etc.)
- Per-strategy date ranges (one global date range applies to all)
"""
from __future__ import annotations

import sys
import itertools
from datetime import date as _date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.data_loader import discovery_summary, load_panel
from core.backtest import (
    StrategySpec, preload_pair_panels, run_single_strategy, trades_to_df,
    monthly_pnl_table,
)
from core.backtest_american import (
    run_single_strategy_american, preload_pair_panels_american,
)
from core.worstof import (
    WorstOfSpec, _run_worstof_with_panels,
)
from core.vanilla_backtest import (
    VanillaSpec, run_vanilla_strategy, vanilla_trades_to_df,
)


# =============================================================================
# Page config
# =============================================================================
st.set_page_config(page_title="Option Portfolio Backtest", layout="wide",
                       initial_sidebar_state="expanded")

from shared.style import inject_base_css, inject_card_css
inject_base_css()
inject_card_css()


# =============================================================================
# Sidebar — data folder
# =============================================================================
from core.ui import data_dir_input as _data_dir_input
st.sidebar.markdown("### Data source")
folder = _data_dir_input(default="market_data")
if folder is None:
    st.info("Specify the market data folder in the sidebar.")
    st.stop()

with st.sidebar.expander("Discovered files", expanded=False):
    s = discovery_summary(folder)
    st.caption(f"Mode: `{s['mode']}`  ·  {s['n_pairs']} pairs across "
                f"{s['n_files']} files")


# =============================================================================
# Constants
# =============================================================================
TENOR_LIST = ["1W", "2W", "3W", "1M", "2M", "3M", "6M", "9M", "1Y"]

# Strategy type definitions. Each maps a UI label to a (kind, direction,
# barrier_dir) triple. `barrier_dir` is "n/a" for vanillas.
STRATEGY_TYPES = {
    "Vanilla call":   ("vanilla", "call", None),
    "Vanilla put":    ("vanilla", "put",  None),
    "EKO Call U&O":   ("eko",     "call", "up_and_out"),
    "EKO Put D&O":    ("eko",     "put",  "down_and_out"),
    "RKO Call U&O":   ("rko",     "call", "up_and_out"),
    "RKO Put D&O":    ("rko",     "put",  "down_and_out"),
    "WO-EKO Call U&O": ("wo_eko", "call", "up_and_out"),
    "WO-EKO Put D&O":  ("wo_eko", "put",  "down_and_out"),
    "WO-RKO Call U&O": ("wo_rko", "call", "up_and_out"),
    "WO-RKO Put D&O":  ("wo_rko", "put",  "down_and_out"),
}

DELTA_CHOICES = {
    "ATM":  ("ATM",  0.00),
    "35Δ":  ("35Δ",  0.35),
    "25Δ":  ("25Δ",  0.25),
    "15Δ":  ("15Δ",  0.15),
}

KO_DELTA_CHOICES = {
    "10Δ":  ("10Δ",  0.10),
    "15Δ":  ("15Δ",  0.15),
    "20Δ":  ("20Δ",  0.20),
    "25Δ":  ("25Δ",  0.25),
}

# Worst-of pricing engines. The UI exposes a logical name ("Joint CF",
# "Joint MC", "Legacy multiplier") and we resolve to the correct engine
# name per WO-type:
#   WO-EKO uses closed_form / monte_carlo
#   WO-RKO uses cf_approx_american / monte_carlo_american
WO_ENGINE_CHOICES = ["Legacy multiplier", "Joint CF", "Joint MC"]

WO_CORRELATION_SOURCES = {
    "Manual (single ρ)":              "manual",
    "Historical 60d rolling":         "rolling_60d",
    "Triangulation (cross vol)":      "triangulation",
}


def _resolve_wo_engine(ui_choice: str, wo_kind: str) -> str:
    """Map the UI engine label + WO kind to the correct WorstOfSpec
    pricing_engine value.

    wo_kind ∈ {'wo_eko', 'wo_rko'}.
    """
    if ui_choice == "Legacy multiplier":
        return "legacy_multiplier"
    if wo_kind == "wo_eko":
        return "closed_form" if ui_choice == "Joint CF" else "monte_carlo"
    # wo_rko
    return ("cf_approx_american" if ui_choice == "Joint CF"
            else "monte_carlo_american")


ASIA_EM = {"USDCNH", "USDCNY", "USDINR", "USDIDR", "USDKRW", "USDMYR",
            "USDPHP", "USDTHB", "USDTWD"}


# =============================================================================
# Helpers
# =============================================================================
def _list_pairs(folder: str) -> "list[str]":
    try:
        ds = load_panel(folder, "SPOT", None)
        return sorted(ds.columns.tolist())
    except Exception:
        return []


def _data_date_range(folder: str, pairs: "list[str]") -> "tuple[_date, _date] | None":
    """Find the common spot data range across `pairs`. Returns None
    if pairs have disjoint data."""
    try:
        df = load_panel(folder, "SPOT", None, prefer="offshore", pairs=tuple(pairs))
        if df.empty:
            return None
        # Trim to rows where ALL pairs have data
        mask = df.notna().all(axis=1)
        if not mask.any():
            return None
        valid = df.index[mask]
        return (valid.min().date(), valid.max().date())
    except Exception:
        return None


def _fmt_usd(x) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"${x:,.0f}"


def _fmt_signed_usd(x) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"${x:+,.0f}"


def _fmt_signed_usd_compact(x) -> str:
    """Compact signed USD: $+1.23M, $-456K, $+78.

    Use this in narrow metric cards (e.g. the 6-column headline strip)
    where the full-precision `_fmt_signed_usd` value gets truncated by
    Streamlit's metric widget. Use `_fmt_signed_usd` everywhere else
    (tables, charts, two-column displays) where the extra precision is
    readable.
    """
    if x is None or not np.isfinite(x):
        return "—"
    sign = "+" if x >= 0 else "-"
    a = abs(x)
    if a >= 1e6:
        return f"${sign}{a/1e6:.2f}M"
    if a >= 1e3:
        return f"${sign}{a/1e3:.1f}K"
    return f"${sign}{a:.0f}"


def _fmt_pct(x) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.2f}%"


# =============================================================================
# Strategy generation
# =============================================================================
def _generate_strategies(
        pairs: "list[str]", strategy_types: "list[str]",
        tenor: str, strike_dlabel: str, strike_dvalue: float,
        ko_dlabel: str, ko_dvalue: float, multiplier: float,
        tx_cost_bps: float, prefer: str, side: str,
        wo_engine_ui: str = "Legacy multiplier",
        wo_corr_source: str = "manual",
        wo_corr_value: float = 0.30,
        wo_mc_n_paths: int = 100_000,
) -> "list[dict]":
    """Materialize a list of strategy specs.

    Returns a list of dicts, each with:
      - kind          : 'vanilla' | 'eko' | 'rko' | 'wo_eko' | 'wo_rko'
      - spec          : VanillaSpec | StrategySpec | WorstOfSpec
      - display_name  : human-readable label for tables
      - pair(s)       : pair (single-leg) or (pair_a, pair_b) (worst-of)
      - direction     : 'call' | 'put'

    WO engine arguments are only relevant for wo_eko / wo_rko types.
    `wo_engine_ui` ∈ {'Legacy multiplier', 'Joint CF', 'Joint MC'} —
    resolved per WO kind via `_resolve_wo_engine`.
    """
    out = []

    for type_label in strategy_types:
        kind, direction, bar_dir = STRATEGY_TYPES[type_label]

        if kind == "vanilla":
            for p in pairs:
                spec = VanillaSpec(
                    pair=p, direction=direction,
                    delta_label=strike_dlabel,
                    delta_value=strike_dvalue,
                    tenor_label=tenor,
                    tx_cost_bps=tx_cost_bps,
                    prefer=prefer,
                )
                out.append({
                    "kind": "vanilla", "spec": spec,
                    "display_name": f"{p} {direction.upper()} VAN {strike_dlabel}",
                    "pair": p, "direction": direction,
                    "type_label": type_label,
                })

        elif kind in ("eko", "rko"):
            for p in pairs:
                spec = StrategySpec(
                    pair=p, direction=direction,
                    barrier_type=bar_dir,
                    delta_label=strike_dlabel,
                    delta_value=strike_dvalue,
                    tenor_label=tenor,
                    tx_cost_bps=tx_cost_bps,
                    prefer=prefer,
                    ko_method="delta",
                    target_ko_delta=ko_dvalue,
                    ko_delta_label=ko_dlabel,
                )
                out.append({
                    "kind": kind, "spec": spec,
                    "display_name": (f"{p} {direction.upper()} {kind.upper()}"
                                       f"-{bar_dir} {strike_dlabel}/{ko_dlabel}"),
                    "pair": p, "direction": direction,
                    "type_label": type_label,
                })

        elif kind in ("wo_eko", "wo_rko"):
            if len(pairs) < 2:
                continue
            ko_check_mode = ("american_ohlc" if kind == "wo_rko"
                              else "european_at_expiry")
            pricing_engine = _resolve_wo_engine(wo_engine_ui, kind)
            for pa, pb in itertools.combinations(pairs, 2):
                spec = WorstOfSpec(
                    leg_a_pair=pa, leg_a_direction=direction,
                    leg_a_barrier_type=bar_dir,
                    leg_a_strike_delta_value=strike_dvalue,
                    leg_a_strike_delta_label=strike_dlabel,
                    leg_a_ko_delta_value=ko_dvalue,
                    leg_a_ko_delta_label=ko_dlabel,
                    leg_b_pair=pb, leg_b_direction=direction,
                    leg_b_barrier_type=bar_dir,
                    leg_b_strike_delta_value=strike_dvalue,
                    leg_b_strike_delta_label=strike_dlabel,
                    leg_b_ko_delta_value=ko_dvalue,
                    leg_b_ko_delta_label=ko_dlabel,
                    tenor_label=tenor,
                    tx_cost_bps=tx_cost_bps,
                    prefer=prefer,
                    multiplier=multiplier,
                    ko_check_mode=ko_check_mode,
                    pricing_engine=pricing_engine,
                    correlation_source=wo_corr_source,
                    correlation_value=wo_corr_value,
                    mc_n_paths=wo_mc_n_paths,
                )
                out.append({
                    "kind": kind, "spec": spec,
                    "display_name": (f"{pa}×{pb} {direction.upper()} "
                                       f"{kind.upper()}-{bar_dir} "
                                       f"{strike_dlabel}/{ko_dlabel}"),
                    "pair": f"{pa}×{pb}",
                    "direction": direction,
                    "type_label": type_label,
                })

    return out


# =============================================================================
# Backtest runner — dispatches to the right engine for each strategy
# =============================================================================
def _ledger_to_pnl_series(ledger_df: pd.DataFrame) -> pd.Series:
    """Convert a per-trade DataFrame to a daily P&L series indexed by
    expiry_date. Returns empty Series if ledger is empty.

    P&L is realized at expiry and BOOKED on the expiry_date — this is the
    same convention used by `core.backtest.compute_equity_and_drawdown`
    and keeps the Sharpe number on this page consistent with the EKO
    Pricer drilldowns. (Previously this was keyed on trade_date, which
    pulled each P&L forward ~30 calendar days and produced a slightly
    different monthly Sharpe than the EKO Pricer view on the same data.)
    """
    if ledger_df.empty or "pnl_usd" not in ledger_df.columns:
        return pd.Series(dtype=float)
    df = ledger_df.copy()
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    s = df.groupby("expiry_date")["pnl_usd"].sum().sort_index()
    s.index = pd.DatetimeIndex(s.index)
    return s


def _run_one_strategy(strat: dict, folder: str, start_date: _date,
                          end_date: _date, notional_usd: float,
                          side_sign: float,
                          panels_cache: dict,
                          ) -> "tuple[pd.DataFrame, pd.Series]":
    """Run a single strategy backtest. Returns (ledger_df, daily_pnl).

    `side_sign` = +1 for Buy, -1 for Sell. Applied to every P&L value:
    when you Sell premium, your P&L is the NEGATION of the buy-side
    P&L (you receive premium at trade entry and pay payoff at expiry).
    `panels_cache` is a dict of pair → panels so each unique pair is
    only preloaded once.
    """
    kind = strat["kind"]
    spec = strat["spec"]

    if kind == "vanilla":
        # cache key = pair (vanilla uses European panels)
        cache_key = ("eko", spec.pair, spec.prefer)
        panels = panels_cache.get(cache_key)
        if panels is None:
            panels = preload_pair_panels(folder, spec.pair, spec.prefer)
            panels_cache[cache_key] = panels
        if not panels:
            return pd.DataFrame(), pd.Series(dtype=float)
        trades = run_vanilla_strategy(
            spec, panels, start_date, end_date, notional_usd=notional_usd,
        )
        df = vanilla_trades_to_df(trades)

    elif kind == "eko":
        cache_key = ("eko", spec.pair, spec.prefer)
        panels = panels_cache.get(cache_key)
        if panels is None:
            panels = preload_pair_panels(folder, spec.pair, spec.prefer)
            panels_cache[cache_key] = panels
        if not panels:
            return pd.DataFrame(), pd.Series(dtype=float)
        trades = run_single_strategy(
            spec, panels, start_date, end_date, notional_usd=notional_usd,
        )
        df = trades_to_df(trades)

    elif kind == "rko":
        cache_key = ("rko", spec.pair, spec.prefer)
        panels = panels_cache.get(cache_key)
        if panels is None:
            panels = preload_pair_panels_american(folder, spec.pair, spec.prefer)
            panels_cache[cache_key] = panels
        if not panels:
            return pd.DataFrame(), pd.Series(dtype=float)
        trades = run_single_strategy_american(
            spec, panels, start_date, end_date, notional_usd=notional_usd,
        )
        df = trades_to_df(trades)

    elif kind in ("wo_eko", "wo_rko"):
        pa, pb = spec.leg_a_pair, spec.leg_b_pair
        # WO engine wants European-style panels for WO-EKO and OHLC
        # panels for WO-RKO
        preload_fn = (preload_pair_panels_american if kind == "wo_rko"
                       else preload_pair_panels)
        cache_kind = "rko" if kind == "wo_rko" else "eko"
        key_a = (cache_kind, pa, spec.prefer)
        key_b = (cache_kind, pb, spec.prefer)
        if key_a not in panels_cache:
            panels_cache[key_a] = preload_fn(folder, pa, spec.prefer)
        if key_b not in panels_cache:
            panels_cache[key_b] = preload_fn(folder, pb, spec.prefer)
        panels_a = panels_cache[key_a]
        panels_b = panels_cache[key_b]
        if not panels_a or not panels_b:
            return pd.DataFrame(), pd.Series(dtype=float)
        trades = _run_worstof_with_panels(
            spec, panels_a, panels_b, start_date, end_date,
            notional_usd=notional_usd, folder=folder,
        )
        from dataclasses import asdict
        df = pd.DataFrame([asdict(t) for t in trades]) if trades else pd.DataFrame()

    else:
        return pd.DataFrame(), pd.Series(dtype=float)

    # Apply side sign
    if not df.empty and "pnl_usd" in df.columns:
        df = df.copy()
        df["pnl_usd"] = df["pnl_usd"] * side_sign
        if "premium_usd" in df.columns:
            df["premium_usd"] = df["premium_usd"] * side_sign
        if "actual_payoff_usd" in df.columns:
            df["actual_payoff_usd"] = df["actual_payoff_usd"] * side_sign

    pnl_series = _ledger_to_pnl_series(df)
    return df, pnl_series


# =============================================================================
# Portfolio metrics
# =============================================================================
def _portfolio_metrics(pnl_daily: pd.Series, notional_per_strat: float,
                          n_strategies: int) -> dict:
    """Compute headline metrics on the portfolio daily P&L series."""
    if pnl_daily.empty:
        return {}
    total_pnl = float(pnl_daily.sum())
    total_notional = notional_per_strat * n_strategies
    # Equity curve
    eq = pnl_daily.cumsum()
    running_max = eq.cummax()
    dd = eq - running_max
    max_dd_usd = float(dd.min())
    max_dd_pct = float(max_dd_usd / total_notional * 100) if total_notional > 0 else 0.0

    # Daily stats
    daily = pnl_daily.copy()
    # Monthly resample for Sharpe (more stable than daily; same as
    # core.backtest.summarize_strategy convention).
    monthly = daily.resample("ME").sum()
    sharpe_monthly = (monthly.mean() / monthly.std() * np.sqrt(12)
                       if len(monthly) > 1 and monthly.std() > 0 else 0.0)
    # Daily Sharpe is kept in the dict for backward compatibility / for
    # any programmatic callers, but is NO LONGER surfaced in the UI.
    # The realized series is sparse (it only has rows on expiry days,
    # ~180/year, not 252), so mean/std × √252 over-annualises and the
    # number isn't comparable to the MTM daily Sharpe. Use either
    # `sharpe_monthly` (realized) or the MTM curve's diff Sharpe.
    sharpe_daily = (daily.mean() / daily.std() * np.sqrt(252)
                      if len(daily) > 1 and daily.std() > 0 else 0.0)

    win_rate = float((daily > 0).mean() * 100)
    n_days = int((daily != 0).sum())
    # P&L skewness
    if len(daily) > 2 and daily.std() > 0:
        skew = float(((daily - daily.mean()) ** 3).mean()
                       / (daily.std() ** 3))
    else:
        skew = 0.0

    # Annualized return (% of total notional)
    n_years = (daily.index.max() - daily.index.min()).days / 365.25 if len(daily) > 1 else 0
    ann_return_pct = (total_pnl / total_notional / n_years * 100
                       if total_notional > 0 and n_years > 0 else 0.0)

    return {
        "total_pnl_usd": total_pnl,
        "total_notional_usd": total_notional,
        "ann_return_pct": ann_return_pct,
        "sharpe_monthly": sharpe_monthly,
        "sharpe_daily": sharpe_daily,
        "max_drawdown_usd": max_dd_usd,
        "max_drawdown_pct": max_dd_pct,
        "win_rate_pct": win_rate,
        "n_trading_days": n_days,
        "skew": skew,
        "best_day_usd": float(daily.max()),
        "worst_day_usd": float(daily.min()),
        "equity_curve": eq,
        "drawdown_series": dd,
    }


def _mtm_metrics(mtm_df: pd.DataFrame, notional_per_strat: float,
                    n_strategies: int) -> dict:
    """Compute MDD from the MTM equity curve. MTM-MDD is typically
    LARGER than realized MDD because MTM captures intra-trade swings.
    """
    if mtm_df.empty:
        return {"max_drawdown_usd": 0.0, "max_drawdown_pct": 0.0}
    total_notl = notional_per_strat * n_strategies
    dd = mtm_df["drawdown_usd"]
    mdd_usd = float(dd.min()) if not dd.empty else 0.0
    mdd_pct = mdd_usd / total_notl * 100 if total_notl > 0 else 0.0
    return {"max_drawdown_usd": mdd_usd, "max_drawdown_pct": mdd_pct}


# =============================================================================
# Main render
# =============================================================================
def render():
    st.title("Option Portfolio Backtest")
    st.caption(
        "Basket backtest across multiple pairs, strategy types, and "
        "worst-of pair-combinations. Equal-notional, daily-rolling, "
        "uniform parameters across the grid. Results include Sharpe, "
        "MDD, win-rate, skew, and per-pair / per-type drilldowns."
    )

    pairs_avail = _list_pairs(folder)
    if not pairs_avail:
        st.error("No SPOT data found in the data folder.")
        return

    # =========================================================================
    # SECTION 1: Portfolio definition
    # =========================================================================
    st.markdown("### Portfolio definition")
    cc1, cc2, cc3 = st.columns(3)

    with cc1:
        default_pairs = [p for p in ("USDJPY", "USDKRW", "USDTHB")
                         if p in pairs_avail]
        if not default_pairs:
            default_pairs = pairs_avail[:2]
        pairs_sel = st.multiselect(
            "Currency pairs",
            pairs_avail, default=default_pairs, key="opb_pairs",
            help=("Pairs to include. Single-leg strategies (Vanilla, "
                   "EKO, RKO) run once per pair. Worst-of strategies "
                   "(WO-EKO, WO-RKO) run on every C(N,2) pair-"
                   "combination of selected pairs."),
        )
        types_sel = st.multiselect(
            "Strategy types",
            list(STRATEGY_TYPES.keys()),
            default=["EKO Call U&O"],
            key="opb_types",
            help=("Each selected type generates strategies on every "
                   "applicable pair (or pair-combination for WO)."),
        )
        tenor = st.selectbox("Tenor", TENOR_LIST,
                                index=TENOR_LIST.index("2M"),
                                key="opb_tenor")

    with cc2:
        strike_dlabel = st.selectbox("Strike Δ", list(DELTA_CHOICES.keys()),
                                          index=0, key="opb_strike_delta",
                                          help="Strike side: ATMF or delta-Δ.")
        _, strike_dvalue = DELTA_CHOICES[strike_dlabel]

        ko_dlabel = st.selectbox("KO Δ (barrier)",
                                       list(KO_DELTA_CHOICES.keys()),
                                       index=0, key="opb_ko_delta",
                                       help=("Barrier placed at the strike "
                                              "whose vanilla delta = this Δ. "
                                              "Only relevant for KO and WO-KO "
                                              "types."))
        _, ko_dvalue = KO_DELTA_CHOICES[ko_dlabel]

        wo_multiplier = st.slider("WO premium multiplier",
                                        min_value=0.30, max_value=0.60, step=0.01,
                                        value=0.33, key="opb_wo_mult",
                                        help=("Structure premium = mult × min"
                                               "(P_A, P_B). Only used for WO "
                                               "types. 0.33 ≈ legacy default "
                                               "(loose corr), 0.50 = "
                                               "highly-correlated. The Dual CCY "
                                               "Pricer tab uses true joint "
                                               "pricing; this is the "
                                               "legacy approximation."))

    with cc3:
        side_label = st.radio(
            "Side (global)", ["Buy", "Sell"], index=0, horizontal=True,
            key="opb_side",
            help=("Buy = pay premium and receive payoff; Sell = "
                   "receive premium and pay payoff. Applied uniformly "
                   "to every strategy in the basket. Most macro option "
                   "portfolios are Sell-side (vol-harvest)."),
        )
        side_sign = +1.0 if side_label == "Buy" else -1.0

        notional_usd = st.number_input(
            "Notional per strategy (USD)",
            min_value=100_000.0, max_value=500_000_000.0,
            value=10_000_000.0, step=1_000_000.0, format="%.0f",
            key="opb_notional",
        )
        tx_cost_bps = st.number_input(
            "Transaction cost (bps of notl)",
            min_value=0.0, max_value=50.0, value=4.0, step=0.5,
            format="%.1f", key="opb_tx",
        )

    # --- Date range ---
    if pairs_sel:
        date_range = _data_date_range(folder, pairs_sel)
        if date_range is None:
            st.error("No overlapping data across selected pairs.")
            return
        min_d, max_d = date_range
        # Default start = 1-Jan-2023 if data covers it, otherwise the
        # earliest data date.
        default_start = max(min_d, _date(2023, 1, 1))
        if default_start > max_d:
            default_start = min_d
        dc1, dc2 = st.columns(2)
        with dc1:
            start_date = st.date_input(
                "Start date", value=default_start,
                min_value=min_d, max_value=max_d,
                key="opb_start_date",
            )
        with dc2:
            end_date = st.date_input(
                "End date", value=max_d,
                min_value=min_d, max_value=max_d,
                key="opb_end_date",
            )

    # Asia EM prefer
    asia_em_in_basket = [p for p in pairs_sel if p in ASIA_EM]
    prefer = "offshore"
    if asia_em_in_basket:
        prefer = st.radio(
            f"Asia EM variant (applies to: {', '.join(asia_em_in_basket)})",
            ["offshore", "onshore"], index=0, horizontal=True,
            key="opb_prefer",
        )

    if not pairs_sel or not types_sel:
        st.info("Select at least one pair and one strategy type.")
        return

    # --- Worst-of engine settings (only shown when a WO type is in the basket) ---
    has_wo_type = any(STRATEGY_TYPES[t][0] in ("wo_eko", "wo_rko")
                       for t in types_sel)
    wo_engine_ui = "Legacy multiplier"
    wo_corr_source = "manual"
    wo_corr_value = 0.30
    wo_mc_n_paths = 100_000
    if has_wo_type:
        with st.expander("⚙️ Worst-of pricing engine", expanded=False):
            st.caption(
                "How the WORST-OF structure premium is computed at each "
                "trade date. **Legacy multiplier** uses "
                "`multiplier × min(P_A, P_B)` (above slider); fast but "
                "ignores ρ. **Joint CF / MC** use the correlation-aware "
                "joint pricers from `core.worstof_pricer*` — slower but "
                "give properly priced worst-ofs. The same pricer is also "
                "used for daily MTM. WO-EKO uses `closed_form` / "
                "`monte_carlo` engines; WO-RKO auto-routes to the "
                "American-monitored equivalents."
            )
            wec1, wec2 = st.columns([1, 1])
            with wec1:
                wo_engine_ui = st.radio(
                    "Engine", WO_ENGINE_CHOICES, index=0,
                    key="opb_wo_engine",
                    help=("Legacy: ignores ρ. CF: 1-D quadrature, "
                           "deterministic, ~2 ms/trade. MC: stochastic, "
                           "~10-100 ms/trade depending on paths."),
                )
            with wec2:
                if wo_engine_ui != "Legacy multiplier":
                    corr_label = st.radio(
                        "Correlation source",
                        list(WO_CORRELATION_SOURCES.keys()),
                        index=1,    # default 60d rolling
                        key="opb_wo_corr_src",
                    )
                    wo_corr_source = WO_CORRELATION_SOURCES[corr_label]
                    wo_corr_value = st.slider(
                        "ρ (Manual / fallback)", min_value=-0.95,
                        max_value=0.95, value=0.30, step=0.05,
                        key="opb_wo_corr_val",
                    )
                    if wo_engine_ui == "Joint MC":
                        wo_mc_n_paths = st.select_slider(
                            "MC paths per trade",
                            options=[20_000, 50_000, 100_000, 200_000],
                            value=50_000, key="opb_wo_mc_paths",
                            help=("Lower paths = faster but noisier "
                                   "MTM. 50k → ~2bp std error per "
                                   "trade; 100k → ~1bp."),
                        )

    # --- Generate strategies & preview ---
    strategies = _generate_strategies(
        pairs_sel, types_sel, tenor, strike_dlabel, strike_dvalue,
        ko_dlabel, ko_dvalue, wo_multiplier, tx_cost_bps, prefer, side_label,
        wo_engine_ui=wo_engine_ui,
        wo_corr_source=wo_corr_source,
        wo_corr_value=wo_corr_value,
        wo_mc_n_paths=wo_mc_n_paths,
    )

    if not strategies:
        st.warning("Selected combinations produced 0 strategies (e.g. "
                    "WO with only 1 pair selected).")
        return

    # Show strategy preview before run
    with st.expander(f"Strategies to run ({len(strategies)}). Click to preview.",
                       expanded=False):
        prev_rows = []
        for i, s in enumerate(strategies, 1):
            prev_rows.append({
                "#": i, "Type": s["type_label"],
                "Pair(s)": s["pair"], "Display name": s["display_name"],
            })
        st.dataframe(pd.DataFrame(prev_rows), use_container_width=True,
                        hide_index=True)

    # --- MTM option ---
    compute_mtm = st.checkbox(
        "Compute daily MTM (mark-to-market equity curve)",
        value=True, key="opb_compute_mtm",
        help=("Re-prices every open trade on every business day during "
               "its life and builds an equity curve from book MTM + "
               "cumulative cash flows. Adds 1-10× the backtest time but "
               "produces a much smoother P&L curve and proper "
               "intra-trade drawdown. Toggle off for fast realized-only "
               "view (premium paid at entry, payoff at expiry, no "
               "intra-trade MTM)."),
    )

    # --- Run button ---
    if not st.button(f"▶ Run portfolio backtest ({len(strategies)} strategies)",
                       type="primary", use_container_width=True,
                       key="opb_run"):
        st.session_state.setdefault("opb_results", None)
        if st.session_state.get("opb_results") is None:
            st.info("Configure the portfolio above, then click **Run** "
                     "to start the backtest. With small portfolios "
                     "(2–20 strategies) this typically takes 5–60 seconds.")
            return
    else:
        # Run!
        results = _run_portfolio(
            strategies, folder, start_date, end_date,
            notional_usd, side_sign, compute_mtm=compute_mtm,
        )
        st.session_state["opb_results"] = results

    results = st.session_state.get("opb_results")
    if results is None:
        return

    # =========================================================================
    # SECTION 2: Results
    # =========================================================================
    _render_results(results, strategies, notional_usd, side_label,
                       tenor, strike_dlabel, ko_dlabel)


def _run_portfolio(strategies, folder, start_date, end_date,
                       notional_usd, side_sign, compute_mtm: bool = True):
    """Run every strategy and aggregate. Returns a dict with the
    per-strategy and portfolio-level outputs.

    If `compute_mtm` is True, also compute daily mark-to-market equity
    curves per strategy and a portfolio MTM curve.
    """
    panels_cache: dict = {}
    per_strat_ledger: dict[str, pd.DataFrame] = {}
    per_strat_pnl: dict[str, pd.Series] = {}
    per_strat_strat_info: dict[str, dict] = {}

    prog = st.progress(0.0, text="Starting backtests…")
    n = len(strategies)
    for i, strat in enumerate(strategies):
        prog.progress(i / n, text=f"Backtest {i+1}/{n}: {strat['display_name']}")
        try:
            ledger_df, pnl = _run_one_strategy(
                strat, folder, start_date, end_date, notional_usd,
                side_sign, panels_cache,
            )
        except Exception as e:
            st.warning(f"Strategy {strat['display_name']} failed: {e}")
            ledger_df, pnl = pd.DataFrame(), pd.Series(dtype=float)
        per_strat_ledger[strat["display_name"]] = ledger_df
        per_strat_pnl[strat["display_name"]] = pnl
        per_strat_strat_info[strat["display_name"]] = strat
    prog.progress(1.0, text="Backtests done.")

    # Aggregate to portfolio daily P&L: align all series, sum
    if per_strat_pnl:
        all_indices = [s.index for s in per_strat_pnl.values() if not s.empty]
        if all_indices:
            common_idx = sorted(set().union(*[set(idx) for idx in all_indices]))
            common_idx = pd.DatetimeIndex(common_idx)
            aligned = pd.DataFrame(
                {name: s.reindex(common_idx, fill_value=0.0)
                 for name, s in per_strat_pnl.items()},
                index=common_idx,
            )
            portfolio_daily = aligned.sum(axis=1)
        else:
            aligned = pd.DataFrame()
            portfolio_daily = pd.Series(dtype=float)
    else:
        aligned = pd.DataFrame()
        portfolio_daily = pd.Series(dtype=float)

    # ---- MTM computation ----
    per_strat_mtm: dict[str, pd.DataFrame] = {}
    portfolio_mtm_df = pd.DataFrame()
    if compute_mtm:
        from core.portfolio_mtm import (
            compute_strategy_mtm_curve, aggregate_portfolio_mtm,
        )
        mtm_prog = st.progress(0.0, text="Computing MTM…")
        for i, strat in enumerate(strategies):
            name = strat["display_name"]
            mtm_prog.progress(i / n,
                                  text=f"MTM {i+1}/{n}: {name}")
            ledger = per_strat_ledger.get(name, pd.DataFrame())
            if ledger.empty:
                per_strat_mtm[name] = pd.DataFrame()
                continue
            try:
                mtm_df = compute_strategy_mtm_curve(
                    strat, ledger, panels_cache, side_sign,
                )
            except Exception as e:
                st.warning(f"MTM failed for {name}: {e}")
                mtm_df = pd.DataFrame()
            per_strat_mtm[name] = mtm_df
        mtm_prog.progress(1.0, text="MTM done.")

        portfolio_mtm_df = aggregate_portfolio_mtm(per_strat_mtm)

    return {
        "per_strat_ledger": per_strat_ledger,
        "per_strat_pnl": per_strat_pnl,
        "portfolio_daily": portfolio_daily,
        "aligned_per_strat": aligned,
        "n_strategies": len(strategies),
        "per_strat_mtm": per_strat_mtm,
        "portfolio_mtm": portfolio_mtm_df,
        "compute_mtm": compute_mtm,
    }


# =============================================================================
# Basket-strategy summary helpers
# =============================================================================
def _basket_name(strat_info: dict, tenor: str, strike_label: str,
                    ko_label: str) -> str:
    """Build a BASKET name for a given strategy type.

    Mirrors the format used by the EKO Pricer's portfolio tab:
      'BASKET Call-UO 1M ATM/H@10Δ'   (single-leg call up-and-out)
      'BASKET Put-DO 2M ATM/H@10Δ'    (single-leg put down-and-out)
      'BASKET WO-Call-UO 2M ATM/H@10Δ' (worst-of call up-and-out)
      'BASKET Vanilla-Call 1M ATM'    (no barrier for vanillas)
    """
    kind = strat_info["kind"]
    direction = strat_info["direction"]
    type_label = strat_info["type_label"]
    dir_token = direction.capitalize()       # 'Call' / 'Put'

    if kind == "vanilla":
        return f"BASKET Vanilla-{dir_token} {tenor} {strike_label}"

    # KO / WO-KO have a barrier orientation suffix
    bar_suffix = "UO" if "U&O" in type_label else "DO"

    if kind in ("eko", "rko"):
        engine_prefix = "EKO" if kind == "eko" else "RKO"
        return (f"BASKET {engine_prefix}-{dir_token}-{bar_suffix} "
                  f"{tenor} {strike_label}/H@{ko_label}")
    # wo_eko / wo_rko
    engine_prefix = "WO-EKO" if kind == "wo_eko" else "WO-RKO"
    return (f"BASKET {engine_prefix}-{dir_token}-{bar_suffix} "
              f"{tenor} {strike_label}/H@{ko_label}")


def _is_knocked_out_row(row) -> bool:
    """Return True if a trade row reflects a barrier knock-out.

    Handles all three ledger schemas:
      - VanillaTrade (no barrier, always False)
      - Trade (single-leg EKO/RKO; uses 'knocked_out' bool)
      - WorstOfTrade (uses 'leg_a_knocked_out' OR 'leg_b_knocked_out')
    """
    if "knocked_out" in row.index and pd.notna(row.get("knocked_out")):
        return bool(row["knocked_out"])
    a = bool(row.get("leg_a_knocked_out", False)) \
        if "leg_a_knocked_out" in row.index else False
    b = bool(row.get("leg_b_knocked_out", False)) \
        if "leg_b_knocked_out" in row.index else False
    return a or b


def _build_basket_summary_rows(results: dict, strategies: "list[dict]",
                                  tenor: str, strike_label: str,
                                  ko_label: str,
                                  per_strat_mtm: dict | None = None,
                                  ) -> "list[dict]":
    """Build the basket-strategy summary table.

    Each row aggregates trades across all strategies in the same
    `type_label` group — matching the EKO Pricer portfolio view.

    Returns rows with these columns (matching the user's screenshot):
        Basket strategy | n | Pairs | Win % | KO % | Σ Premium |
        Σ Payoff | PnL | Sharpe (m) | Max DD
    """
    # Group strategies by type_label so each "type" → one basket row.
    from collections import defaultdict
    by_type: "dict[str, list[dict]]" = defaultdict(list)
    for s in strategies:
        by_type[s["type_label"]].append(s)

    # Decide the Sharpe column header ONCE based on whether MTM curves
    # are present. Mixed headers across rows would produce two columns
    # in the rendered DataFrame, one mostly NaN. If MTM is enabled with
    # at least one usable curve, every row uses daily-MTM Sharpe.
    mtm_active = bool(per_strat_mtm) and any(
        c is not None and not c.empty for c in per_strat_mtm.values()
    )
    sharpe_col = "Sharpe (d)" if mtm_active else "Sharpe (m)"

    rows = []
    for type_label, strat_group in by_type.items():
        # Pool all trade ledgers in this type
        pooled_dfs = []
        for s in strat_group:
            name = s["display_name"]
            df = results["per_strat_ledger"].get(name, pd.DataFrame())
            if df is not None and not df.empty:
                pooled_dfs.append(df)
        if not pooled_dfs:
            rows.append({
                "Basket strategy": _basket_name(strat_group[0], tenor,
                                                       strike_label, ko_label),
                "n": 0, "Pairs": "—", "Win %": "—", "KO %": "—",
                "Σ Premium": "—", "Σ Payoff": "—", "PnL": "—",
                sharpe_col: "—", "Max DD": "—",
            })
            continue
        pooled = pd.concat(pooled_dfs, ignore_index=True)

        n_trades = int(len(pooled))
        # Pair counting: for WO, 'pair' is "P_a×P_b" — count distinct
        # entries directly (so USDJPY×EURUSD counts as one pair-combo).
        if "pair" in pooled.columns:
            n_pairs = int(pooled["pair"].nunique())
        elif "leg_a_pair" in pooled.columns:
            crosses = pooled["leg_a_pair"] + "×" + pooled["leg_b_pair"]
            n_pairs = int(crosses.nunique())
        else:
            n_pairs = len(strat_group)
        # KO %
        ko_flags = pooled.apply(_is_knocked_out_row, axis=1)
        ko_rate = float(ko_flags.mean() * 100) if n_trades > 0 else 0.0
        # Win % on trade-level pnl_usd
        if "pnl_usd" in pooled.columns:
            win_rate = float((pooled["pnl_usd"] > 0).mean() * 100)
            pnl_sum = float(pooled["pnl_usd"].sum())
        else:
            win_rate = 0.0
            pnl_sum = 0.0
        # Σ Premium — column varies by trade type
        if "premium_usd" in pooled.columns:
            prem_sum = float(pooled["premium_usd"].sum())
        elif "structure_premium_paid_usd" in pooled.columns:
            prem_sum = float(pooled["structure_premium_paid_usd"].sum())
        else:
            prem_sum = 0.0
        # Σ Payoff
        if "actual_payoff_usd" in pooled.columns:
            pay_sum = float(pooled["actual_payoff_usd"].sum())
        elif "worst_of_payoff_usd" in pooled.columns:
            pay_sum = float(pooled["worst_of_payoff_usd"].sum())
        else:
            pay_sum = 0.0

        # Realized Sharpe (monthly, ann.) and MDD on aggregated daily
        # P&L of this basket. This is the FALLBACK shown when MTM is off
        # or when no per-strategy MTM curve is available.
        aligned_pnl = None
        for s in strat_group:
            name = s["display_name"]
            pnl_series = results["per_strat_pnl"].get(name)
            if pnl_series is None or pnl_series.empty:
                continue
            if aligned_pnl is None:
                aligned_pnl = pnl_series.copy()
            else:
                aligned_pnl = aligned_pnl.add(pnl_series, fill_value=0.0)
        if aligned_pnl is not None and not aligned_pnl.empty:
            monthly = aligned_pnl.resample("ME").sum()
            sharpe_val = (float(monthly.mean() / monthly.std() * np.sqrt(12))
                          if len(monthly) > 1 and monthly.std() > 0 else 0.0)
            # Realized MDD on aggregated equity
            eq = aligned_pnl.cumsum()
            mdd = float((eq - eq.cummax()).min())
        else:
            sharpe_val = 0.0
            mdd = 0.0

        # Override Sharpe AND MDD with MTM-based values if available.
        # When MTM is on, the basket's daily book vol (intra-trade vega/
        # gamma/theta) is the right denominator — not the chunky payoff
        # at expiry. We sum per-pair equity curves into a basket curve,
        # diff to daily, then Sharpe = mean/std × √252. The column
        # header (sharpe_col) was decided once above; if MTM is active
        # for the table, every row reports daily-MTM Sharpe so the
        # ordering and units are comparable across rows.
        if per_strat_mtm:
            mtm_curves_in_group = [
                per_strat_mtm.get(s["display_name"]) for s in strat_group
            ]
            mtm_curves_in_group = [c for c in mtm_curves_in_group
                                   if c is not None and not c.empty]
            if mtm_curves_in_group:
                # Align all per-pair equity curves on date index, ffill
                # missing days (pair hasn't started trading yet → 0),
                # then sum to a basket-level equity curve.
                eq_wide = pd.concat(
                    [c["equity_usd"] for c in mtm_curves_in_group], axis=1,
                ).sort_index().ffill().fillna(0.0)
                basket_eq = eq_wide.sum(axis=1)
                daily_mtm = basket_eq.diff().dropna()
                if len(daily_mtm) > 1 and daily_mtm.std() > 0:
                    sharpe_val = float(daily_mtm.mean() / daily_mtm.std()
                                       * np.sqrt(252))
                # MDD from the properly-aggregated MTM curve (replaces
                # the looser "sum of per-strategy MDDs" used previously,
                # which was an upper bound on the basket MDD).
                basket_peak = basket_eq.cummax()
                basket_dd = basket_eq - basket_peak
                mdd = float(basket_dd.min())

        rows.append({
            "Basket strategy": _basket_name(strat_group[0], tenor,
                                                  strike_label, ko_label),
            "n": n_trades,
            "Pairs": n_pairs,
            "Win %": f"{win_rate:.0f}",
            "KO %": f"{ko_rate:.0f}",
            "Σ Premium": _fmt_signed_usd(prem_sum),
            "Σ Payoff": _fmt_signed_usd(pay_sum),
            "PnL": _fmt_signed_usd(pnl_sum),
            sharpe_col: f"{sharpe_val:+.2f}",
            "Max DD": _fmt_signed_usd(mdd),
        })
    return rows


# =============================================================================
# Chart helpers — match EKO Pricer style
# =============================================================================
def _annual_sharpe_per_year(pnl_series: pd.Series) -> "dict[int, float]":
    """Per-calendar-year Sharpe ratio from a daily P&L series.

    Sharpe_y = mean(monthly_pnl) / std(monthly_pnl) × √12 for each
    year. Years with fewer than 2 valid monthly observations return
    NaN so the table can render '—' rather than misleading 0.
    """
    if pnl_series is None or pnl_series.empty:
        return {}
    monthly = pnl_series.resample("ME").sum()
    if monthly.empty:
        return {}
    out: "dict[int, float]" = {}
    for yr, sub in monthly.groupby(monthly.index.year):
        if len(sub) > 1 and sub.std() > 0:
            out[int(yr)] = float(sub.mean() / sub.std() * np.sqrt(12))
        else:
            out[int(yr)] = float("nan")
    return out


def _render_pnl_by_year_chart(yearly: pd.Series, title: str) -> None:
    """Bar chart for yearly PnL (USD). Green for positive, red for negative."""
    if yearly.empty:
        st.caption("(no trades — no yearly PnL)")
        return
    fig = go.Figure()
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in yearly.values]
    fig.add_trace(go.Bar(
        x=[str(y) for y in yearly.index], y=yearly.values,
        marker_color=colors,
        text=[_fmt_signed_usd(v) for v in yearly.values],
        textposition="auto",
        hovertemplate="%{x}<br>PnL: $%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=title, height=340, margin=dict(l=10, r=10, t=40, b=10),
        yaxis_tickprefix="$", yaxis_tickformat=",.0f",
        showlegend=False, xaxis_title="Year",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_pnl_by_pair_chart(by_pair: pd.Series,
                                  pair_label: str = "Pair") -> None:
    """Bar chart for PnL by pair."""
    if by_pair.empty:
        st.caption(f"(no trades by {pair_label.lower()})")
        return
    by_pair_sorted = by_pair.sort_values(ascending=False)
    fig = go.Figure()
    colors = ["#22c55e" if v >= 0 else "#ef4444"
              for v in by_pair_sorted.values]
    fig.add_trace(go.Bar(
        x=by_pair_sorted.index, y=by_pair_sorted.values,
        marker_color=colors,
        text=[_fmt_signed_usd(v) for v in by_pair_sorted.values],
        textposition="auto",
        hovertemplate="%{x}<br>PnL: $%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"P&L by {pair_label.lower()}", height=340,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis_tickprefix="$", yaxis_tickformat=",.0f",
        showlegend=False, xaxis_title=pair_label,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_pnl_heatmap(year_pair: pd.DataFrame,
                            pair_label: str = "Pair") -> None:
    """Year × pair PnL heatmap. Rows = year (descending), cols = pair."""
    if year_pair.empty:
        st.caption("(no trades for the heatmap)")
        return
    col_order = year_pair.sum(axis=0).sort_values(ascending=False).index
    yp = year_pair[col_order].sort_index(ascending=False)
    vmax = float(np.nanmax(np.abs(yp.values))) if yp.size else 1.0
    fig = go.Figure(data=go.Heatmap(
        z=yp.values,
        x=yp.columns.tolist(),
        y=[str(y) for y in yp.index],
        colorscale="RdYlGn", zmid=0, zmin=-vmax, zmax=vmax,
        text=[[_fmt_signed_usd(v) if not pd.isna(v) else "" for v in row]
              for row in yp.values],
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate=(f"{pair_label}: %{{x}}<br>Year: %{{y}}<br>"
                          "PnL: $%{z:,.0f}<extra></extra>"),
        colorbar=dict(title="PnL ($)"),
    ))
    fig.update_layout(
        title=f"P&L heatmap — year × {pair_label.lower()}",
        height=max(280, 38 * len(yp.index) + 100),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_monthly_heatmap(pooled_df: pd.DataFrame) -> None:
    """Year × month PnL heatmap with RdYlGn coloring. Uses
    `core.backtest.monthly_pnl_table` for the pivot."""
    if pooled_df.empty:
        st.caption("(no trades for the monthly heatmap)")
        return
    monthly_df = monthly_pnl_table(pooled_df, value_col="pnl_usd")
    if monthly_df.empty:
        st.caption("(no monthly data)")
        return
    month_labels = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May",
                       6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct",
                       11: "Nov", 12: "Dec"}
    monthly_df = monthly_df.rename(
        columns=lambda c: month_labels.get(c, str(c))
    )
    arr = monthly_df.to_numpy(dtype=float, na_value=np.nan)
    finite = arr[np.isfinite(arr)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 0.0

    def _ryg(v):
        if pd.isna(v) or not np.isfinite(v) or vmax == 0.0:
            return ""
        x = max(-1.0, min(1.0, v / vmax))
        if x >= 0:
            r, g, b = (int(255 + (60 - 255) * x),
                          int(255 + (170 - 255) * x),
                          int(180 + (80 - 180) * x))
        else:
            xm = -x
            r, g, b = (int(255 + (200 - 255) * xm),
                          int(255 + (60 - 255) * xm),
                          int(180 + (60 - 180) * xm))
        return f"background-color: rgb({r},{g},{b}); color: #1a1a1a;"

    st.dataframe(
        monthly_df.style.format("${:,.0f}", na_rep="").map(_ryg),
        use_container_width=True,
    )


def _render_combined_equity_drawdown(eq_series: pd.Series,
                                          dd_series: pd.Series,
                                          title: str) -> None:
    """Two-row plotly chart: equity on top (green), drawdown on bottom (red)."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35], vertical_spacing=0.07,
        subplot_titles=("Cumulative P&L (USD)", "Drawdown (USD)"),
    )
    fig.add_trace(go.Scatter(
        x=eq_series.index, y=eq_series.values,
        mode="lines", line=dict(color="#22c55e", width=2),
        showlegend=False,
        hovertemplate=("%{x|%Y-%m-%d}<br>"
                        "Equity: $%{y:,.0f}<extra></extra>"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dd_series.index, y=dd_series.values,
        mode="lines", line=dict(color="#ef4444", width=1.5),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.25)",
        showlegend=False,
        hovertemplate=("%{x|%Y-%m-%d}<br>"
                        "DD: $%{y:,.0f}<extra></extra>"),
    ), row=2, col=1)
    fig.update_layout(
        height=520, margin=dict(l=10, r=10, t=50, b=10),
        title_text=title,
    )
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    st.plotly_chart(fig, use_container_width=True)


def _pooled_pair_breakdown(pooled_df: pd.DataFrame) -> pd.DataFrame:
    """Per-pair aggregation for the breakdown table.

    Returns columns: Pair | n trades | PnL | Σ Premium | Σ Payoff |
                       KO % | Win %.
    Handles single-leg ('pair') and worst-of (leg_a×leg_b) ledgers.
    """
    df = pooled_df.copy()
    # Resolve a 'pair' column if absent (worst-of trades)
    if "pair" not in df.columns:
        if "leg_a_pair" in df.columns and "leg_b_pair" in df.columns:
            df["pair"] = df["leg_a_pair"] + "×" + df["leg_b_pair"]
        else:
            return pd.DataFrame()
    # Resolve premium / payoff column names
    prem_col = ("premium_usd" if "premium_usd" in df.columns
                  else "structure_premium_paid_usd")
    pay_col = ("actual_payoff_usd" if "actual_payoff_usd" in df.columns
                 else "worst_of_payoff_usd")
    if prem_col not in df.columns:
        df[prem_col] = 0.0
    if pay_col not in df.columns:
        df[pay_col] = 0.0

    df["_ko"] = df.apply(_is_knocked_out_row, axis=1)
    out = df.groupby("pair").apply(lambda g: pd.Series({
        "n trades": int(len(g)),
        "PnL": _fmt_signed_usd(g["pnl_usd"].sum()),
        "Σ Premium": _fmt_signed_usd(g[prem_col].sum()),
        "Σ Payoff": _fmt_signed_usd(g[pay_col].sum()),
        "KO %": f"{(g['_ko'].mean() * 100):.0f}%",
        "Win %": f"{((g['pnl_usd'] > 0).mean() * 100):.0f}%",
        # numeric helper for sorting:
        "_sort_pnl": float(g["pnl_usd"].sum()),
    })).reset_index().sort_values("_sort_pnl", ascending=False).drop(
        columns="_sort_pnl"
    )
    return out


def _render_results(results, strategies, notional_usd, side_label,
                       tenor: str, strike_label: str, ko_label: str):
    """Render the results section: basket summary table + 6-metric strip +
    drilldown tabs."""
    portfolio_daily = results["portfolio_daily"]
    aligned = results["aligned_per_strat"]

    if portfolio_daily.empty:
        st.error("All strategies returned 0 trades. Check date range and data.")
        return

    metrics = _portfolio_metrics(
        portfolio_daily, notional_usd, results["n_strategies"],
    )

    # ---- Build pooled DataFrame across ALL strategies (needed for
    # several portfolio-level views) ----
    pooled_dfs = []
    for s in strategies:
        df = results["per_strat_ledger"].get(s["display_name"],
                                                  pd.DataFrame())
        if df is not None and not df.empty:
            # Normalize: ensure 'pair' column exists for downstream
            if "pair" not in df.columns and "leg_a_pair" in df.columns:
                df = df.copy()
                df["pair"] = df["leg_a_pair"] + "×" + df["leg_b_pair"]
            pooled_dfs.append(df)
    pooled = (pd.concat(pooled_dfs, ignore_index=True)
              if pooled_dfs else pd.DataFrame())

    # =====================================================================
    # Basket-strategy summary table  (top of results — matches screenshot)
    # =====================================================================
    st.markdown("---")
    st.markdown("### Basket-strategy summary")
    st.caption(
        "Each row pools ALL per-pair strategies of a given type into "
        "one **basket strategy** (mirrors the EKO/RKO Pricer's "
        "portfolio view). `n` = total trades across all pairs in the "
        "basket. `Pairs` = number of distinct pairs (or pair-combos "
        "for worst-of). When MTM is enabled, the **Sharpe** column "
        "switches from monthly-realized (×√12) to daily-MTM (×√252) "
        "and **Max DD** is computed from the basket MTM equity curve."
    )
    basket_rows = _build_basket_summary_rows(
        results, strategies, tenor, strike_label, ko_label,
        per_strat_mtm=results.get("per_strat_mtm"),
    )
    if basket_rows:
        st.dataframe(pd.DataFrame(basket_rows), hide_index=True,
                        use_container_width=True)

    # =====================================================================
    # Top-line 6-metric strip (matches EKO Pricer drilldown layout)
    # =====================================================================
    st.markdown("### Portfolio headline")
    # Aggregate KO%, Pairs across the entire portfolio (pooled)
    n_trades_total = int(len(pooled)) if not pooled.empty else 0
    if not pooled.empty:
        ko_flags = pooled.apply(_is_knocked_out_row, axis=1)
        ko_rate_total = float(ko_flags.mean() * 100) if n_trades_total > 0 else 0.0
        # Pairs: count distinct 'pair' values (handles WO crosses)
        n_pairs_total = int(pooled["pair"].nunique()) \
            if "pair" in pooled.columns else 0
        win_rate_total = float((pooled["pnl_usd"] > 0).mean() * 100) \
            if "pnl_usd" in pooled.columns else 0.0
    else:
        ko_rate_total = 0.0
        n_pairs_total = 0
        win_rate_total = 0.0

    cs = st.columns(6)
    cs[0].metric("Trades (pooled)", f"{n_trades_total:,}",
                    f"{n_pairs_total} pairs")
    # Use the COMPACT signed-USD formatter ($+10.81M, $-4.71M) — the
    # full-precision form was being truncated by Streamlit in the
    # 6-column layout. Full precision is still used in the Performance
    # metrics drilldown table below, where the column is wide enough.
    cs[1].metric("Total P&L",
                    _fmt_signed_usd_compact(metrics["total_pnl_usd"]),
                    f"{metrics['ann_return_pct']:+.2f}% annl.")
    # Sharpe and MDD: prefer MTM if available so the toggle is
    # reflected in BOTH cards (previously only MDD branched).
    pmtm = results.get("portfolio_mtm")
    if pmtm is not None and not pmtm.empty:
        # MTM mode → daily Sharpe from the equity curve diff × √252.
        # pmtm["equity_usd"] is the daily basket book value; diff =
        # daily P&L from re-pricing all live options. This is the
        # right Sharpe for the MTM accounting view because it reflects
        # intra-trade vol, not just the chunky payoff at expiry.
        daily_mtm = pmtm["equity_usd"].diff().dropna()
        if len(daily_mtm) > 1 and daily_mtm.std() > 0:
            sharpe_mtm_d = float(daily_mtm.mean() / daily_mtm.std()
                                     * np.sqrt(252))
        else:
            sharpe_mtm_d = 0.0
        cs[2].metric("Sharpe (d)", f"{sharpe_mtm_d:+.2f}",
                        "daily MTM × √252")
        mtm_mdd = float(pmtm["drawdown_usd"].min())
        cs[3].metric("Max DD (MTM)",
                        _fmt_signed_usd_compact(mtm_mdd),
                        "MTM-based")
    else:
        # Subtitle was previously `daily: +X.XX` — but on the sparse
        # expiry-keyed realized series that number (mean/std × √252)
        # is misleading: the series only has rows on expiry days
        # (~180/year), so √252 over-annualises a series that doesn't
        # have 252 obs/year. It is NOT comparable to the daily-MTM
        # Sharpe shown when MTM is on. Subtitle now mirrors the MTM
        # mode's `daily MTM × √252` for visual symmetry.
        cs[2].metric("Sharpe (m)", f"{metrics['sharpe_monthly']:+.2f}",
                        "monthly × √12")
        cs[3].metric("Max DD",
                        _fmt_signed_usd_compact(metrics["max_drawdown_usd"]),
                        "realized")
    cs[4].metric("Win rate", f"{win_rate_total:.0f}%",
                    f"{int(n_trades_total * win_rate_total / 100):,} winners")
    cs[5].metric("KO rate", f"{ko_rate_total:.0f}%",
                    "barrier hit")

    # =====================================================================
    # Drilldown tabs
    # =====================================================================
    tab_port, tab_strat, tab_pair, tab_type, tab_calendar = st.tabs(
        ["📈 Portfolio", "📋 Per-strategy", "💱 Per-pair",
         "🏷️ Per-type", "📅 Calendar"]
    )

    with tab_port:
        _render_portfolio_tab(portfolio_daily, metrics, aligned, notional_usd,
                                  results["n_strategies"], side_label,
                                  results.get("portfolio_mtm", pd.DataFrame()),
                                  results.get("compute_mtm", False),
                                  pooled)

    with tab_strat:
        _render_per_strategy_tab(results, strategies, notional_usd)

    with tab_pair:
        _render_per_pair_tab(results, strategies, notional_usd, pooled)

    with tab_type:
        _render_per_type_tab(results, strategies, notional_usd)

    with tab_calendar:
        _render_calendar_tab(pooled, portfolio_daily)

    # =====================================================================
    # Full pooled trade ledger expander (always at bottom)
    # =====================================================================
    if not pooled.empty:
        with st.expander(
            f"📜 Full pooled trade ledger ({n_trades_total:,} trades)",
            expanded=False,
        ):
            led_cols_default = [
                "pair", "trade_date", "expiry_date", "tenor_label",
                "spot", "strike", "barrier",
                "leg_a_pair", "leg_b_pair",
                "premium_usd", "structure_premium_paid_usd",
                "actual_payoff_usd", "worst_of_payoff_usd",
                "pnl_usd",
                "knocked_out", "leg_a_knocked_out", "leg_b_knocked_out",
            ]
            led_cols = [c for c in led_cols_default if c in pooled.columns]
            sort_by = [c for c in ("pair", "trade_date") if c in pooled.columns]
            st.dataframe(
                pooled[led_cols].sort_values(sort_by) if sort_by
                else pooled[led_cols],
                hide_index=True, use_container_width=True,
            )


# =============================================================================
# Tab renderers
# =============================================================================
def _render_portfolio_tab(portfolio_daily, metrics, aligned, notional_usd,
                              n_strategies, side_label,
                              portfolio_mtm: pd.DataFrame = None,
                              has_mtm: bool = False,
                              pooled: pd.DataFrame = None):
    has_mtm_curve = (has_mtm and portfolio_mtm is not None
                       and not portfolio_mtm.empty)

    st.markdown(f"#### Portfolio P&L curve  ·  side = **{side_label}**  ·  "
                  f"{n_strategies} strategies × ${notional_usd:,.0f} notl/strat")

    # Curve-mode toggle
    if has_mtm_curve:
        curve_mode = st.radio(
            "Curve type",
            ["MTM (mark-to-market)", "Realized only"],
            index=0, horizontal=True, key="opb_curve_mode",
            help=("**MTM**: equity = MTM book value + cumulative cash. "
                   "Smooth daily curve, properly accounts for "
                   "intra-trade drawdowns.  \n"
                   "**Realized**: equity = cumulative realized P&L "
                   "booked at trade entry. Discontinuous; ignores "
                   "mid-life mark moves."),
        )
        use_mtm = curve_mode.startswith("MTM")
    else:
        use_mtm = False

    # Combined equity + drawdown chart (2-row plotly subplot,
    # matching EKO/RKO Pricer drilldown layout)
    if use_mtm:
        eq = portfolio_mtm["equity_usd"]
        dd = portfolio_mtm["drawdown_usd"]
        title = "MTM equity & drawdown (daily, summed across strategies)"
    else:
        eq = metrics["equity_curve"]
        dd = metrics["drawdown_series"]
        title = "Realized equity & drawdown (by expiry, pooled)"
    _render_combined_equity_drawdown(eq, dd, title)

    # Detailed metrics table (realized — always shown)
    st.markdown("##### Performance metrics")
    perf_rows = [
        ("Total P&L (USD)",    _fmt_signed_usd(metrics["total_pnl_usd"])),
        ("Total notional (USD, deployed)", _fmt_usd(metrics["total_notional_usd"])),
        ("Annualized return (% of notl)", f"{metrics['ann_return_pct']:+.3f}%"),
        ("Sharpe (monthly, ann.)", f"{metrics['sharpe_monthly']:+.3f}"),
        # Note: a "Sharpe (daily, ann.)" row was previously shown here,
        # but the realized P&L series is sparse (only expiry days), so
        # mean/std × √252 over-annualises a series that has fewer than
        # 252 obs/year. It's not comparable to the MTM daily Sharpe.
        # The monthly Sharpe is the canonical realized number.
        ("Max drawdown — realized (USD)", _fmt_signed_usd(metrics["max_drawdown_usd"])),
        ("Max drawdown — realized (% of notl)", f"{metrics['max_drawdown_pct']:+.3f}%"),
    ]
    if has_mtm_curve:
        mtm_metrics = _mtm_metrics(portfolio_mtm, notional_usd, n_strategies)
        perf_rows.extend([
            ("Max drawdown — MTM (USD)",
             _fmt_signed_usd(mtm_metrics["max_drawdown_usd"])),
            ("Max drawdown — MTM (% of notl)",
             f"{mtm_metrics['max_drawdown_pct']:+.3f}%"),
        ])
    perf_rows.extend([
        ("Win rate (% of days)", f"{metrics['win_rate_pct']:.2f}%"),
        ("# trading days", f"{metrics['n_trading_days']:,}"),
        ("Best day", _fmt_signed_usd(metrics["best_day_usd"])),
        ("Worst day", _fmt_signed_usd(metrics["worst_day_usd"])),
        ("P&L skewness", f"{metrics['skew']:+.3f}"),
    ])
    st.dataframe(pd.DataFrame(perf_rows, columns=["Metric", "Value"]),
                    use_container_width=True, hide_index=True)

    # =====================================================================
    # P&L by year (matches EKO Pricer drilldown)
    # =====================================================================
    if pooled is not None and not pooled.empty and "expiry_date" in pooled.columns:
        st.divider()
        st.markdown("##### P&L by year")
        pooled_with_year = pooled.copy()
        pooled_with_year["expiry_date"] = pd.to_datetime(
            pooled_with_year["expiry_date"]
        )
        pooled_with_year["_year"] = pooled_with_year["expiry_date"].dt.year
        yearly = pooled_with_year.groupby("_year")["pnl_usd"].sum().sort_index()
        _render_pnl_by_year_chart(yearly, "P&L by expiry year")

        # Per-year Sharpe table from the realized daily P&L stream
        sharpe_by_year = _annual_sharpe_per_year(portfolio_daily)
        yearly_tbl = yearly.reset_index().rename(
            columns={"_year": "Year", "pnl_usd": "PnL (USD)"}
        )
        yearly_tbl["Sharpe (m)"] = yearly_tbl["Year"].map(
            lambda y: sharpe_by_year.get(int(y), float("nan"))
        )
        yearly_tbl = yearly_tbl.assign(**{
            "PnL (USD)": yearly_tbl["PnL (USD)"].apply(_fmt_signed_usd),
            "Sharpe (m)": yearly_tbl["Sharpe (m)"].apply(
                lambda v: f"{v:+.2f}" if pd.notna(v) else "—"
            ),
        })
        st.dataframe(yearly_tbl, hide_index=True, use_container_width=True)
        st.caption(
            "**Sharpe (m)** is the monthly-basis annualized Sharpe ratio "
            "computed *within* each calendar year: "
            "`mean(monthly_pnl) / std(monthly_pnl) × √12`. Years with "
            "fewer than 2 valid monthly observations show '—'."
        )

    # =====================================================================
    # Drawdown attribution
    # =====================================================================
    if not aligned.empty:
        st.divider()
        st.markdown("##### Drawdown attribution — strategies driving the largest DDs")
        eq_for_attr = metrics["equity_curve"]
        dd_for_attr = metrics["drawdown_series"]
        if dd_for_attr.min() < 0:
            trough_idx = dd_for_attr.idxmin()
            pre_trough = eq_for_attr.loc[:trough_idx]
            peak_idx = pre_trough.idxmax()
            window = aligned.loc[peak_idx:trough_idx]
            attribution = window.sum(axis=0).sort_values()
            attribution_df = pd.DataFrame({
                "Strategy": attribution.index,
                "Drawdown contribution (USD)": attribution.values,
            }).head(10)
            attribution_df["Drawdown contribution (USD)"] = (
                attribution_df["Drawdown contribution (USD)"]
                .map(_fmt_signed_usd)
            )
            st.caption(f"Window: {peak_idx.date()} → {trough_idx.date()}  ·  "
                         f"Total DD: "
                         f"{_fmt_signed_usd(eq_for_attr.loc[trough_idx] - eq_for_attr.loc[peak_idx])}")
            st.dataframe(attribution_df, use_container_width=True,
                            hide_index=True)


def _render_calendar_tab(pooled: pd.DataFrame,
                            portfolio_daily: pd.Series) -> None:
    """Calendar drilldown: monthly P&L heatmap (year × month) + year × pair heatmap."""
    if pooled is None or pooled.empty:
        st.info("No trades to display.")
        return

    pooled = pooled.copy()
    pooled["expiry_date"] = pd.to_datetime(pooled["expiry_date"])
    pooled["_year"] = pooled["expiry_date"].dt.year

    # ---- Monthly P&L heatmap (year × month) ----
    st.markdown("#### Monthly P&L heatmap — year × month (USD)")
    st.caption("Each cell = sum of trade P&L expiring in that month. "
                 "Green = profit, red = loss.")
    _render_monthly_heatmap(pooled)

    st.divider()

    # ---- Year × pair heatmap ----
    if "pair" in pooled.columns:
        st.markdown("#### P&L heatmap — year × pair")
        year_pair = pooled.groupby(["_year", "pair"])["pnl_usd"].sum().unstack(
            fill_value=0.0
        )
        _render_pnl_heatmap(year_pair, pair_label="Pair")


def _render_per_strategy_tab(results, strategies, notional_usd):
    st.markdown("#### Per-strategy summary")
    rows = []
    for s in strategies:
        name = s["display_name"]
        ledger = results["per_strat_ledger"].get(name, pd.DataFrame())
        pnl_series = results["per_strat_pnl"].get(name, pd.Series(dtype=float))
        if pnl_series.empty:
            rows.append({
                "Strategy": name, "Type": s["type_label"],
                "Pair(s)": s["pair"],
                "# trades": 0,
                "Total P&L": "—",
                "Win rate": "—",
                "Sharpe (monthly)": "—",
                "MDD": "—",
            })
            continue
        eq = pnl_series.cumsum()
        running_max = eq.cummax()
        dd = eq - running_max
        monthly = pnl_series.resample("ME").sum()
        sharpe_m = (monthly.mean() / monthly.std() * np.sqrt(12)
                       if len(monthly) > 1 and monthly.std() > 0 else 0.0)
        rows.append({
            "Strategy": name, "Type": s["type_label"],
            "Pair(s)": s["pair"],
            "# trades": int(len(ledger)),
            "Total P&L": _fmt_signed_usd(pnl_series.sum()),
            "Win rate": f"{(pnl_series > 0).mean()*100:.1f}%",
            "Sharpe (monthly)": f"{sharpe_m:+.2f}",
            "MDD": _fmt_signed_usd(dd.min()),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Selectable single-strategy drilldown
    st.markdown("##### Strategy detail")
    strat_names = [s["display_name"] for s in strategies]
    if strat_names:
        sel = st.selectbox("Select a strategy to inspect:",
                                 strat_names, index=0, key="opb_strat_sel")
        ledger = results["per_strat_ledger"].get(sel, pd.DataFrame())
        if not ledger.empty:
            cols_to_show = [c for c in (
                "trade_date", "expiry_date", "strike", "barrier",
                "spot", "spot_at_expiry", "sigma_atm", "sigma_smile",
                "premium_pct", "actual_payoff_pct", "pnl_pct",
                "premium_usd", "actual_payoff_usd", "pnl_usd",
                "knocked_out",
            ) if c in ledger.columns]
            st.dataframe(ledger[cols_to_show],
                            use_container_width=True, hide_index=True)
        else:
            st.info("No trades for this strategy in the selected window.")


def _render_per_pair_tab(results, strategies, notional_usd,
                              pooled: pd.DataFrame = None):
    st.markdown("#### Per-pair aggregate")
    st.caption(
        "P&L grouped by pair. For worst-of strategies, the structure's "
        "P&L is attributed to the pair-combo (e.g. 'USDJPY×EURUSD')."
    )

    # ---- Bar chart of total P&L by pair ----
    if pooled is not None and not pooled.empty and "pair" in pooled.columns \
            and "pnl_usd" in pooled.columns:
        by_pair_total = pooled.groupby("pair")["pnl_usd"].sum()
        _render_pnl_by_pair_chart(by_pair_total, pair_label="Pair")

        # ---- Full per-pair breakdown table ----
        st.markdown("##### Per-pair breakdown")
        breakdown_df = _pooled_pair_breakdown(pooled)
        if not breakdown_df.empty:
            breakdown_df.columns = [
                "Pair", "n trades", "PnL",
                "Σ Premium", "Σ Payoff", "KO %", "Win %",
            ]
            st.dataframe(breakdown_df, hide_index=True,
                            use_container_width=True)

    # ---- Cumulative P&L curves overlaid (legacy view, kept) ----
    by_pair: dict[str, pd.Series] = {}
    n_strats_by_pair: dict[str, int] = {}
    for s in strategies:
        pair_key = s["pair"]
        pnl = results["per_strat_pnl"].get(s["display_name"],
                                             pd.Series(dtype=float))
        if pair_key not in by_pair:
            by_pair[pair_key] = pd.Series(dtype=float)
            n_strats_by_pair[pair_key] = 0
        if not pnl.empty:
            by_pair[pair_key] = by_pair[pair_key].add(pnl, fill_value=0.0)
        n_strats_by_pair[pair_key] += 1

    if not by_pair:
        return

    chart_data = {}
    rows = []
    for pair_key, pnl_series in by_pair.items():
        if pnl_series.empty:
            rows.append({"Pair(s)": pair_key,
                          "# strategies": n_strats_by_pair[pair_key],
                          "Total P&L": "—", "Sharpe (monthly)": "—",
                          "Win rate": "—"})
            continue
        monthly = pnl_series.resample("ME").sum()
        sharpe_m = (monthly.mean() / monthly.std() * np.sqrt(12)
                       if len(monthly) > 1 and monthly.std() > 0 else 0.0)
        rows.append({
            "Pair(s)": pair_key,
            "# strategies": n_strats_by_pair[pair_key],
            "Total P&L": _fmt_signed_usd(pnl_series.sum()),
            "Sharpe (monthly)": f"{sharpe_m:+.2f}",
            "Win rate": f"{(pnl_series > 0).mean()*100:.1f}%",
        })
        chart_data[pair_key] = pnl_series.cumsum()

    if chart_data:
        st.markdown("##### Cumulative P&L by pair")
        st.line_chart(pd.DataFrame(chart_data), height=300)


def _render_per_type_tab(results, strategies, notional_usd):
    st.markdown("#### Per-type aggregate")
    st.caption("P&L grouped by strategy type (Vanilla, EKO, RKO, WO-EKO, "
                 "WO-RKO).")
    by_type: dict[str, pd.Series] = {}
    n_strats_by_type: dict[str, int] = {}
    for s in strategies:
        type_key = s["type_label"]
        pnl = results["per_strat_pnl"].get(s["display_name"],
                                             pd.Series(dtype=float))
        if type_key not in by_type:
            by_type[type_key] = pd.Series(dtype=float)
            n_strats_by_type[type_key] = 0
        if not pnl.empty:
            by_type[type_key] = by_type[type_key].add(pnl, fill_value=0.0)
        n_strats_by_type[type_key] += 1

    if not by_type:
        st.info("No type groups to display.")
        return

    rows = []
    chart_data = {}
    for type_key, pnl_series in by_type.items():
        if pnl_series.empty:
            rows.append({"Type": type_key,
                          "# strategies": n_strats_by_type[type_key],
                          "Total P&L": "—",
                          "Sharpe (monthly)": "—",
                          "Win rate": "—"})
            continue
        monthly = pnl_series.resample("ME").sum()
        sharpe_m = (monthly.mean() / monthly.std() * np.sqrt(12)
                       if len(monthly) > 1 and monthly.std() > 0 else 0.0)
        rows.append({
            "Type": type_key,
            "# strategies": n_strats_by_type[type_key],
            "Total P&L": _fmt_signed_usd(pnl_series.sum()),
            "Sharpe (monthly)": f"{sharpe_m:+.2f}",
            "Win rate": f"{(pnl_series > 0).mean()*100:.1f}%",
        })
        chart_data[type_key] = pnl_series.cumsum()

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    if chart_data:
        st.markdown("##### Cumulative P&L by type")
        st.line_chart(pd.DataFrame(chart_data), height=300)


render()

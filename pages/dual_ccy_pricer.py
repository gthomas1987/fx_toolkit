"""Dual CCY Option Pricer — joint pricing of two-pair worst-of structures.

Bloomberg-OVML-style layout: rows are parameters, columns are the two
legs (one per pair). The joint structure premium is computed using
correlation-aware pricers from `core.worstof_pricer*`.

# Supported structures

  Worst-of vanilla            : pays  min(call/put_A, call/put_B)
  Worst-of EKO (European KO)  : both legs European barrier
  Worst-of RKO (American KO)  : both legs continuous-monitored barrier

# Engines per structure

  Vanilla / EKO  → core.worstof_pricer
                   CF (1D quadrature, ~2 ms) + MC (~10 ms)
  RKO            → core.worstof_pricer_american
                   CF-approx (ratio-scaled European, ~2 ms)
                   MC w/ Brownian-bridge monitoring (~100 ms @ 20k paths)

# Correlation sources

  Manual                : user-set ρ slider
  Realized 60d          : 60-business-day rolling realized log-return ρ
  Triangulation         : implied ρ from the cross-pair's ATM vol
                           (e.g. EURJPY for USDJPY × EURUSD)

# Greeks

Via `core.worstof_greeks.worstof_greeks_fd` — pricer-agnostic FD with
common random numbers for MC noise control. Reports:
  - Δ per leg
  - Γ per leg
  - ν per leg
  - ∂V/∂ρ (the distinctive worst-of Greek)
  - Θ per calendar day

# Out of scope (this turn)

- Best-of structures (`max` instead of `min`)
- Dual digitals
- Multi-leg dual-CCY strategies (>2 pairs)
- Buy/Sell — always Buy for now (per user spec)
"""
from __future__ import annotations

import sys
from datetime import date as _date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from core.data_loader import discovery_summary, load_panel, load_by_ticker
from core.calendar_fx import compute_option_dates_for_pair
from core.rates import load_rates_panel, get_rate_at
from core.conventions import get_pip_scale
from core.smile import smile_vol_at_strike
from core.worstof_pricer import (
    WorstOfLeg, worstof_eko_price_cf, worstof_eko_price_mc,
)
from core.worstof_pricer_american import (
    worstof_rko_price_cf_approx, worstof_rko_price_mc,
)
from core.worstof_greeks import worstof_greeks_fd
from core.correlation import (
    realized_correlation_at, implied_correlation_at_T,
)


# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="Dual CCY Option Pricer",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

# Payoff style — what each leg looks like
PAYOFF_STYLES = {
    "Vanilla":             "vanilla",
    "EKO Up & Out":        "eko_uo",
    "EKO Down & Out":      "eko_do",
    "RKO Up & Out":        "rko_uo",
    "RKO Down & Out":      "rko_do",
}

DIRECTIONS = ["Call", "Put"]

CORR_SOURCES = {
    "Manual (slider)":             "manual",
    "Realized 60d (rolling)":      "rolling_60d",
    "Triangulation (cross vol)":   "triangulation",
}

ASIA_EM = {"USDCNH", "USDCNY", "USDINR", "USDIDR", "USDKRW", "USDMYR",
            "USDPHP", "USDTHB", "USDTWD"}


# =============================================================================
# Helpers
# =============================================================================
def _list_pairs(folder: str) -> list[str]:
    try:
        ds = load_panel(folder, "SPOT", None)
        return sorted(ds.columns.tolist())
    except Exception:
        return []


def _resolve_leg_market_data(folder: str, pair: str, prefer: str,
                                tenor: str, val_date: pd.Timestamp,
                                ) -> "dict | None":
    """Pull σ_atm, RR, BF, forward, r_d, r_f, T for one leg.

    Mirrors the same helper in pages/option_pricer.py — kept here so
    the Dual CCY page is self-contained.
    """
    out: dict = {}
    spot_df = load_panel(folder, "SPOT", None, prefer=prefer, pairs=(pair,))
    if spot_df.empty or pair not in spot_df.columns:
        return None
    spot_ts = spot_df[pair].dropna()
    spot = float(spot_ts.asof(val_date))
    if pd.isna(spot):
        return None
    out["spot"] = spot
    out["spot_ts"] = spot_ts   # needed for realized-correlation calc

    vol_df = load_panel(folder, "VOL_ATM", tenor, prefer=prefer, pairs=(pair,))
    if vol_df.empty or pair not in vol_df.columns:
        return None
    sigma_atm_pct = vol_df[pair].asof(val_date)
    if pd.isna(sigma_atm_pct):
        return None
    out["sigma_atm"] = float(sigma_atm_pct) / 100.0

    rr_25 = bf_25 = 0.0
    try:
        rr_df = load_panel(folder, "VOL_RR_25D", tenor, prefer=prefer,
                              pairs=(pair,))
        bf_df = load_panel(folder, "VOL_BF_25D", tenor, prefer=prefer,
                              pairs=(pair,))
        if (not rr_df.empty and pair in rr_df.columns
                and not bf_df.empty and pair in bf_df.columns):
            rr_v = rr_df[pair].asof(val_date)
            bf_v = bf_df[pair].asof(val_date)
            if pd.notna(rr_v) and pd.notna(bf_v):
                rr_25 = float(rr_v) / 100.0
                bf_25 = float(bf_v) / 100.0
    except Exception:
        pass
    out["rr_25"] = rr_25
    out["bf_25"] = bf_25

    F_market = spot
    try:
        fwd_df = load_panel(folder, "FWD_POINTS", tenor, prefer=prefer,
                              pairs=(pair,))
        if not fwd_df.empty and pair in fwd_df.columns:
            fwd_v = fwd_df[pair].asof(val_date)
            if pd.notna(fwd_v):
                pip = get_pip_scale(pair)
                F_market = spot + float(fwd_v) * pip
    except Exception:
        pass
    out["forward"] = F_market

    td = val_date.date() if hasattr(val_date, "date") else val_date
    opt_dates = compute_option_dates_for_pair(td, tenor, pair)
    out["T"] = opt_dates.T_years
    out["expiry"] = opt_dates.option_expiry
    out["delivery"] = opt_dates.option_settlement
    out["spot_settle"] = opt_dates.spot_settlement

    foreign, domestic = pair[:3], pair[3:]
    f_panel = load_rates_panel(folder, foreign, load_by_ticker)
    d_panel = load_rates_panel(folder, domestic, load_by_ticker)
    r_f = get_rate_at(f_panel, out["T"], td)
    r_d = get_rate_at(d_panel, out["T"], td)
    if r_d is None and r_f is None:
        return None
    if r_d is None:
        r_d = r_f + np.log(F_market / spot) / out["T"]
    if r_f is None:
        r_f = r_d - np.log(F_market / spot) / out["T"]
    out["r_d"] = r_d
    out["r_f"] = r_f
    return out


def _resolve_rho(folder: str, pair_a: str, pair_b: str,
                    prefer_a: str, prefer_b: str,
                    spot_a: pd.Series, spot_b: pd.Series,
                    T: float, val_date: pd.Timestamp,
                    source: str, manual_rho: float
                    ) -> "tuple[float, str]":
    """Resolve ρ based on the chosen source. Returns (rho_used, label).

    Falls back to manual_rho if the requested source has no data.
    """
    if source == "manual":
        return float(manual_rho), "manual"
    if source == "rolling_60d":
        r60, n = realized_correlation_at(spot_a, spot_b, val_date, window=60)
        if r60 is not None and not pd.isna(r60):
            return float(r60), f"realized_60d (n={n})"
        return float(manual_rho), "realized_60d_no_data→manual"
    if source == "triangulation":
        try:
            tri = implied_correlation_at_T(folder, pair_a, pair_b, T,
                                              val_date, prefer_a=prefer_a,
                                              prefer_b=prefer_b,
                                              prefer_cross=prefer_a)
            if (tri is not None and tri.rho_implied is not None
                    and not np.isnan(tri.rho_implied)):
                lbl = f"triangulation via {tri.cross_pair}"
                if tri.clipped:
                    lbl += " (CLIPPED)"
                return float(tri.rho_implied), lbl
        except Exception:
            pass
        return float(manual_rho), "triangulation_no_data→manual"
    return float(manual_rho), "manual"


def _build_worstof_leg(opt: str, payoff_style: str,
                          S: float, K: float, H: float | None,
                          sigma: float, r_d: float, r_f: float
                          ) -> WorstOfLeg:
    """Construct a WorstOfLeg with spot normalised to 1.0 (engine
    convention). Internally K and H are rescaled by S so the pricer
    sees a percent-of-notional payoff.
    """
    if payoff_style == "vanilla":
        # Vanilla = KO with barrier far away (use 'none' bar_dir which
        # the WO pricer supports as the vanilla-limit case)
        return WorstOfLeg(
            S=1.0, K=K / S, H=1e9,
            sigma=sigma, r_d=r_d, r_f=r_f,
            opt=opt, bar_dir="none",
        )
    bar_dir_map = {
        "eko_uo": "up_and_out", "eko_do": "down_and_out",
        "rko_uo": "up_and_out", "rko_do": "down_and_out",
    }
    bar_dir = bar_dir_map[payoff_style]
    return WorstOfLeg(
        S=1.0, K=K / S, H=(H / S if H else 1e9),
        sigma=sigma, r_d=r_d, r_f=r_f,
        opt=opt, bar_dir=bar_dir,
    )


# =============================================================================
# Formatters
# =============================================================================
def _fmt_usd(x):
    if x is None or not np.isfinite(x):
        return "—"
    return f"${x:,.0f}"


def _fmt_signed_usd(x):
    if x is None or not np.isfinite(x):
        return "—"
    return f"${x:+,.0f}"


def _fmt_pct(x):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x*100:.4f}%"


# =============================================================================
# Main render
# =============================================================================
def render():
    st.title("Dual CCY Option Pricer")
    st.caption(
        "Joint correlation-aware pricing of two-pair worst-of structures. "
        "Each leg is on a different currency pair; the structure pays "
        "**min(payoff_A, payoff_B)**. Uses `core.worstof_pricer*` for "
        "the joint CF and Monte Carlo engines."
    )

    pairs_avail = _list_pairs(folder)
    if len(pairs_avail) < 2:
        st.error("Need at least 2 currency pairs in the data folder.")
        return

    # ---- Strategy-level controls ----
    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([1, 1, 1])
    with ctrl_c1:
        default_a = ("USDJPY" if "USDJPY" in pairs_avail
                       else pairs_avail[0])
        pair_a = st.selectbox(
            "Leg A — Currency pair", pairs_avail,
            index=pairs_avail.index(default_a), key="dco_pair_a",
        )
        default_b = ("EURUSD" if "EURUSD" in pairs_avail and "EURUSD" != pair_a
                       else (pairs_avail[1] if pairs_avail[1] != pair_a
                              else pairs_avail[0]))
        pair_b_choices = [p for p in pairs_avail if p != pair_a]
        if not pair_b_choices:
            st.error("Need a second distinct pair.")
            return
        default_b_idx = (pair_b_choices.index(default_b)
                          if default_b in pair_b_choices else 0)
        pair_b = st.selectbox(
            "Leg B — Currency pair", pair_b_choices,
            index=default_b_idx, key="dco_pair_b",
        )

    with ctrl_c2:
        # EM variant per leg (offshore/onshore)
        prefer_a = "offshore"
        if pair_a in ASIA_EM:
            prefer_a = st.radio(
                f"{pair_a} variant", ["offshore", "onshore"],
                index=0, horizontal=True, key="dco_prefer_a",
            )
        prefer_b = "offshore"
        if pair_b in ASIA_EM:
            prefer_b = st.radio(
                f"{pair_b} variant", ["offshore", "onshore"],
                index=0, horizontal=True, key="dco_prefer_b",
            )
        tenor = st.selectbox(
            "Tenor (shared)", TENOR_LIST,
            index=TENOR_LIST.index("1M"), key="dco_tenor",
        )

    with ctrl_c3:
        # Default trade date = latest common business day in BOTH spot series
        try:
            spot_a_ts = load_panel(folder, "SPOT", None,
                                     prefer=prefer_a,
                                     pairs=(pair_a,))[pair_a].dropna()
            spot_b_ts = load_panel(folder, "SPOT", None,
                                     prefer=prefer_b,
                                     pairs=(pair_b,))[pair_b].dropna()
            common = spot_a_ts.index.intersection(spot_b_ts.index)
            latest_dt = (common.max().date() if len(common) > 0
                          else _date.today())
        except Exception:
            latest_dt = _date.today()
        td_input = st.date_input(
            "Trade date", value=latest_dt,
            min_value=_date(2010, 1, 1), max_value=latest_dt,
            key="dco_trade_date",
            help="Latest common business day in both pairs' spot series.",
        )
        val_date = pd.Timestamp(td_input)

        notional_usd = st.number_input(
            "Notional (USD, per leg)",
            min_value=100_000.0, max_value=500_000_000.0,
            value=10_000_000.0, step=1_000_000.0, format="%.0f",
            key="dco_notional",
        )

    st.markdown("---")

    # ---- Resolve market data for both legs ----
    md_a = _resolve_leg_market_data(folder, pair_a, prefer_a, tenor,
                                        val_date)
    md_b = _resolve_leg_market_data(folder, pair_b, prefer_b, tenor,
                                        val_date)
    if md_a is None:
        st.error(f"Could not resolve market data for {pair_a}.")
        return
    if md_b is None:
        st.error(f"Could not resolve market data for {pair_b}.")
        return

    # The two legs should have the same T (same trade date + tenor)
    # because we use the same calendar. If they differ (e.g. one pair
    # has a holiday the other doesn't), pick the SMALLER T as the
    # binding expiry to ensure both legs are alive on the same date.
    T = min(md_a["T"], md_b["T"])

    # ---- Per-leg widgets ----
    leg_cols = st.columns(2)
    leg_inputs = {}
    for i, (col, leg_pair, md, prefer) in enumerate(
            [(leg_cols[0], pair_a, md_a, prefer_a),
             (leg_cols[1], pair_b, md_b, prefer_b)]):
        with col:
            st.markdown(f"#### Leg {chr(65+i)} — {leg_pair}")
            direction = st.selectbox(
                "Direction", DIRECTIONS, index=0,
                key=f"dco_dir_{i}",
            )
            payoff_label = st.selectbox(
                "Payoff style", list(PAYOFF_STYLES.keys()),
                index=0, key=f"dco_payoff_{i}",
            )
            payoff = PAYOFF_STYLES[payoff_label]
            is_ko = payoff != "vanilla"

            # Strike input (default ATMF)
            default_K = md["forward"]
            strike_in = st.text_input(
                "Strike",
                value=f"{default_K:.4f}",
                key=f"dco_strike_{i}",
                help="Default = forward. Enter a numeric strike.",
            )
            try:
                K = float(strike_in)
            except ValueError:
                st.warning(f"Invalid strike, using forward {default_K:.4f}")
                K = default_K

            H = None
            if is_ko:
                # Sensible default barrier ~5% OTM in the right direction.
                # NB: payoff codes use _uo / _do suffixes (eko_uo, rko_uo,
                # eko_do, rko_do), so we match those — NOT 'up' / 'down'
                # which DON'T appear in the codes.
                if payoff.endswith("_uo"):
                    default_H = md["spot"] * 1.05
                else:
                    default_H = md["spot"] * 0.95
                barrier_in = st.text_input(
                    "Barrier (H)",
                    value=f"{default_H:.4f}",
                    key=f"dco_barrier_{i}",
                )
                try:
                    H = float(barrier_in)
                except ValueError:
                    st.warning(f"Invalid barrier, using {default_H:.4f}")
                    H = default_H

            leg_inputs[i] = {
                "pair": leg_pair, "md": md, "prefer": prefer,
                "direction": direction, "payoff": payoff,
                "payoff_label": payoff_label,
                "K": K, "H": H,
            }

    # ---- Validate structure compatibility ----
    # For now, both legs must have compatible payoff styles to share an
    # engine: vanilla+vanilla → EKO engine (with H=∞); eko+eko → EKO
    # engine; rko+rko → RKO engine. Mixing eko and rko is supported by
    # the engine technically but is unusual — we allow it but warn.
    payoffs = [leg_inputs[0]["payoff"], leg_inputs[1]["payoff"]]
    has_rko = any(p.startswith("rko") for p in payoffs)
    has_eko = any(p.startswith("eko") for p in payoffs)
    has_vanilla = any(p == "vanilla" for p in payoffs)
    if has_rko and has_eko:
        st.warning(
            "Mixing EKO and RKO legs: pricing uses the **RKO** engine "
            "(continuous monitoring) for both — the EKO leg's barrier "
            "will also be continuously monitored. Use matching payoff "
            "styles for the most accurate pricing."
        )
    engine_family = "rko" if has_rko else "eko"   # vanilla+vanilla → EKO

    # ---- Correlation source ----
    st.markdown("---")
    st.markdown("#### Correlation")
    cc1, cc2 = st.columns([2, 1])
    with cc1:
        corr_src_label = st.selectbox(
            "ρ source",
            list(CORR_SOURCES.keys()),
            index=0,   # default Manual
            key="dco_corr_src",
            help=(
                "**Manual**: ρ from the slider below.\n"
                "**Realized 60d**: rolling 60-business-day log-return "
                "correlation of the two pairs' spots.\n"
                "**Triangulation**: forward-looking implied ρ from the "
                "cross-pair's ATM vol (e.g. EURJPY for USDJPY × EURUSD)."
            ),
        )
        corr_src = CORR_SOURCES[corr_src_label]
    with cc2:
        manual_rho = st.slider(
            "Manual ρ (or fallback)",
            min_value=-0.95, max_value=0.95, value=0.30, step=0.05,
            key="dco_manual_rho",
        )

    # Resolve ρ
    rho_used, rho_label = _resolve_rho(
        folder, pair_a, pair_b, prefer_a, prefer_b,
        md_a["spot_ts"], md_b["spot_ts"], T, val_date,
        corr_src, manual_rho,
    )

    # ---- Engine controls ----
    st.markdown("#### Engine")
    en1, en2, en3 = st.columns([1.2, 1, 1])
    with en1:
        engine_label_options = (
            ["CF (closed-form, ~2 ms)", "MC (Monte Carlo, ~100 ms)"]
        )
        engine_label = st.radio(
            "Pricing engine",
            engine_label_options,
            index=0, key="dco_engine_choice",
            help=(
                "**CF** = 1D quadrature for European (exact), "
                "ratio-scaled CF for American (~2 ms, low-biased "
                "on tight barriers).\n"
                "**MC** = correlated GBM. ~100 ms at 20k paths."
            ),
        )
        use_mc = engine_label.startswith("MC")
    with en2:
        mc_n_paths = st.select_slider(
            "MC paths", options=[10_000, 20_000, 50_000, 100_000, 200_000],
            value=20_000, key="dco_mc_paths",
            disabled=(not use_mc),
        )
    with en3:
        compute_greeks = st.checkbox(
            "Compute Greeks", value=True, key="dco_compute_greeks",
            help=("Greeks via FD on the chosen engine. "
                   "MC Greeks are slower (~10-100x base price) — "
                   "with CF Greeks they're ~40 ms."),
        )

    # ---- Run pricing ----
    st.markdown("---")
    st.markdown("### Results")

    # Per-leg context first (echoed for the user even though it's already
    # in their inputs — for sanity-check at a glance)
    leg_labels = [f"Leg A ({pair_a})", f"Leg B ({pair_b})"]
    cols_for_df = ["Strategy"] + leg_labels

    # ---- Strategy details table ----
    detail_rows = []

    def _add_row(name, vals, total=None):
        d = {"Parameter": name, "Strategy": total if total is not None else "—"}
        for j, v in enumerate(vals):
            d[leg_labels[j]] = v
        detail_rows.append(d)

    _add_row("Direction",
              [leg_inputs[i]["direction"] for i in range(2)])
    _add_row("Payoff style",
              [leg_inputs[i]["payoff_label"] for i in range(2)])
    _add_row("Strike", [f"{leg_inputs[i]['K']:.4f}" for i in range(2)])
    _add_row("Barrier (H)",
              [f"{leg_inputs[i]['H']:.4f}" if leg_inputs[i]["H"] else "—"
               for i in range(2)])
    _add_row("Spot",
              [f"{leg_inputs[i]['md']['spot']:.4f}" for i in range(2)])
    _add_row("Forward",
              [f"{leg_inputs[i]['md']['forward']:.4f}" for i in range(2)])
    _add_row("σ_atm",
              [f"{leg_inputs[i]['md']['sigma_atm']*100:.3f}%"
               for i in range(2)])
    _add_row("RR_25Δ",
              [f"{leg_inputs[i]['md']['rr_25']*100:+.3f}%"
               for i in range(2)])
    _add_row("BF_25Δ",
              [f"{leg_inputs[i]['md']['bf_25']*100:+.3f}%"
               for i in range(2)])
    _add_row("Expiry",
              [str(leg_inputs[i]["md"]["expiry"]) for i in range(2)])
    _add_row("T (years)",
              [f"{leg_inputs[i]['md']['T']:.4f}" for i in range(2)],
              total=f"min={T:.4f}")
    _add_row("Notional (USD)",
              [f"${notional_usd:,.0f}", f"${notional_usd:,.0f}"],
              total="—")

    df_detail = pd.DataFrame(detail_rows).set_index("Parameter")
    st.dataframe(df_detail[cols_for_df], use_container_width=True)

    # ---- ρ resolved + correlation row ----
    st.info(
        f"**ρ = {rho_used:+.4f}**  ·  source: {rho_label}  ·  "
        f"engine family: **{engine_family.upper()}**"
    )

    # ---- Build WorstOfLeg objects ----
    sigma_smile_a = smile_vol_at_strike(
        leg_inputs[0]["md"]["spot"], leg_inputs[0]["K"], T,
        leg_inputs[0]["md"]["sigma_atm"],
        leg_inputs[0]["md"]["rr_25"], leg_inputs[0]["md"]["bf_25"],
        leg_inputs[0]["md"]["r_d"], leg_inputs[0]["md"]["r_f"],
    )
    sigma_smile_b = smile_vol_at_strike(
        leg_inputs[1]["md"]["spot"], leg_inputs[1]["K"], T,
        leg_inputs[1]["md"]["sigma_atm"],
        leg_inputs[1]["md"]["rr_25"], leg_inputs[1]["md"]["bf_25"],
        leg_inputs[1]["md"]["r_d"], leg_inputs[1]["md"]["r_f"],
    )

    opt_a = "call" if leg_inputs[0]["direction"] == "Call" else "put"
    opt_b = "call" if leg_inputs[1]["direction"] == "Call" else "put"

    wol_a = _build_worstof_leg(
        opt_a, leg_inputs[0]["payoff"],
        leg_inputs[0]["md"]["spot"], leg_inputs[0]["K"],
        leg_inputs[0]["H"], sigma_smile_a,
        leg_inputs[0]["md"]["r_d"], leg_inputs[0]["md"]["r_f"],
    )
    wol_b = _build_worstof_leg(
        opt_b, leg_inputs[1]["payoff"],
        leg_inputs[1]["md"]["spot"], leg_inputs[1]["K"],
        leg_inputs[1]["H"], sigma_smile_b,
        leg_inputs[1]["md"]["r_d"], leg_inputs[1]["md"]["r_f"],
    )

    # Discount rate: leg-A DOM (mixed-measure convention; same as the
    # rest of the toolkit's worst-of code uses).
    r_d_struct = leg_inputs[0]["md"]["r_d"]

    # ---- Validate barrier orientation (catch obvious user errors) ----
    err_rows = []
    for i, leg in leg_inputs.items():
        if leg["payoff"] == "vanilla":
            continue
        S = leg["md"]["spot"]
        H = leg["H"]
        # Codes are eko_uo / rko_uo / eko_do / rko_do — match the suffix.
        if leg["payoff"].endswith("_uo") and H <= S:
            err_rows.append(f"Leg {chr(65+i)}: Up-barrier H={H:.4f} ≤ "
                             f"spot {S:.4f} — impossible (would knock "
                             f"out immediately).")
        if leg["payoff"].endswith("_do") and H >= S:
            err_rows.append(f"Leg {chr(65+i)}: Down-barrier H={H:.4f} ≥ "
                             f"spot {S:.4f} — impossible.")
    if err_rows:
        for e in err_rows:
            st.error(e)
        return

    # ---- Select pricer ----
    if engine_family == "rko":
        if use_mc:
            pricer = worstof_rko_price_mc
            pricer_kwargs = {"n_paths": mc_n_paths,
                              "monitoring": "brownian_bridge",
                              "seed": 42}
        else:
            pricer = worstof_rko_price_cf_approx
            pricer_kwargs = {"n_quad": 60}
    else:   # eko (or vanilla which uses the EKO engine)
        if use_mc:
            pricer = worstof_eko_price_mc
            pricer_kwargs = {"n_paths": mc_n_paths, "seed": 42}
        else:
            pricer = worstof_eko_price_cf
            pricer_kwargs = {"n_quad": 80}

    # ---- Price ----
    with st.spinner(f"Pricing with "
                      f"{'MC' if use_mc else 'CF'} engine..."):
        import time
        t0 = time.perf_counter()
        out = pricer(wol_a, wol_b, T, rho_used, r_d_struct,
                       **pricer_kwargs)
        price_ms = (time.perf_counter() - t0) * 1000

    # ---- Display structure premium ----
    price_per_unit = out["price"]
    price_usd = price_per_unit * notional_usd
    se_usd = out.get("std_err", 0.0) * notional_usd

    # Single-leg reference premiums (each leg priced alone for context)
    # to compute the engine-vs-legacy comparison
    from core.ko import ko_price
    from core.american_barrier import ako_closed_form

    def _single_leg_price(pair_idx: int) -> float:
        leg = leg_inputs[pair_idx]
        S = leg["md"]["spot"]
        K = leg["K"]
        sigma = sigma_smile_a if pair_idx == 0 else sigma_smile_b
        r_d = leg["md"]["r_d"]
        r_f = leg["md"]["r_f"]
        if leg["payoff"] == "vanilla":
            from core.vanilla import vanilla_price as vp
            return vp(("call" if leg["direction"] == "Call" else "put"),
                       S, K, T, sigma, r_d, r_f) / S * notional_usd
        H = leg["H"]
        opt = "call" if leg["direction"] == "Call" else "put"
        bar_dir = ("up_and_out" if leg["payoff"].endswith("_uo")
                    else "down_and_out")
        if leg["payoff"].startswith("rko"):
            p = ako_closed_form(opt, bar_dir, S, K, H, T, sigma, r_d, r_f)
        else:
            p = ko_price(opt, bar_dir, S, K, H, T, sigma, r_d, r_f)
        return p / S * notional_usd

    p_a_usd = _single_leg_price(0)
    p_b_usd = _single_leg_price(1)
    legacy_min_usd = min(p_a_usd, p_b_usd)
    ratio_vs_legacy = (price_usd / legacy_min_usd
                        if legacy_min_usd > 0 else float("nan"))

    # Survival probabilities (engine returns these)
    p_alive_joint = out.get("p_alive_joint", out.get("p_survive_joint"))
    if p_alive_joint is None:
        p_alive_joint = out.get("p_both_itm_and_alive", None)

    # ---- Top-line metrics ----
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        ci_str = f" ± ${1.96*se_usd:,.0f}" if use_mc and se_usd > 0 else ""
        st.metric("Structure premium",
                    f"${price_usd:,.0f}{ci_str}",
                    f"{price_per_unit*100:.4f}% of notional")
    with m2:
        st.metric(f"Leg A ({pair_a}) premium (single)",
                    _fmt_usd(p_a_usd))
    with m3:
        st.metric(f"Leg B ({pair_b}) premium (single)",
                    _fmt_usd(p_b_usd))
    with m4:
        st.metric("vs min(leg_A, leg_B)",
                    f"{ratio_vs_legacy:.2f}×"
                    if np.isfinite(ratio_vs_legacy) else "—",
                    f"Joint discount: "
                    f"{(1-ratio_vs_legacy)*100:+.1f}%"
                    if np.isfinite(ratio_vs_legacy) else "")

    # ---- Survival probability breakdown ----
    if p_alive_joint is not None:
        prob_cols = st.columns(4)
        prob_cols[0].metric(
            "P(leg A alive at expiry)",
            f"{out.get('p_alive_leg1', out.get('p_survive_a', 0))*100:.1f}%",
        )
        prob_cols[1].metric(
            "P(leg B alive at expiry)",
            f"{out.get('p_alive_leg2', out.get('p_survive_b', 0))*100:.1f}%",
        )
        prob_cols[2].metric(
            "P(both alive at expiry)",
            f"{p_alive_joint*100:.1f}%",
        )
        if "p_both_itm_and_alive" in out:
            prob_cols[3].metric(
                "P(both alive AND both ITM)",
                f"{out['p_both_itm_and_alive']*100:.1f}%",
            )

    st.caption(
        f"Engine: **{type(pricer).__name__ if not hasattr(pricer, '__name__') else pricer.__name__}**  ·  "
        f"timing: {price_ms:.0f} ms  ·  "
        f"discount r_d (leg A DOM): {r_d_struct*100:.3f}%"
    )

    # =========================================================================
    # Greeks
    # =========================================================================
    if compute_greeks:
        st.markdown("---")
        st.markdown("### Greeks")
        st.caption(
            "Δ, Γ, ν per leg via central FD (bumping one leg at a time).  "
            "**∂V/∂ρ** is the distinctive worst-of Greek — captures the "
            "correlation risk you can only hedge via correlation products.  "
            "Θ via 1-day forward bump on T.  "
            "MC Greeks use common random numbers for noise control."
        )

        with st.spinner("Computing Greeks via FD..."):
            t0 = time.perf_counter()
            g = worstof_greeks_fd(
                wol_a, wol_b, T, rho_used, r_d_struct,
                pricer=pricer, pricer_kwargs=pricer_kwargs,
            )
            greek_ms = (time.perf_counter() - t0) * 1000

        # Display scaling
        # Δ per leg: dollar delta = Δ × N (the FD returns Δ in
        # per-unit-of-rescaled-spot terms; with S_normalized=1.0 in the
        # WoL, Δ × N gives dollar delta on the foreign notional).
        delta_a_usd = g.delta_a * notional_usd
        delta_b_usd = g.delta_b * notional_usd
        gamma_a_usd = g.gamma_a * notional_usd * 0.01   # per 1% spot
        gamma_b_usd = g.gamma_b * notional_usd * 0.01
        vega_a_usd_per_vp = g.vega_a * 0.01 * notional_usd
        vega_b_usd_per_vp = g.vega_b * 0.01 * notional_usd
        rho_sens_per_rp = g.rho_sensitivity * 0.01 * notional_usd
        theta_usd_per_day = g.theta_per_day * notional_usd

        g_rows = [
            {"Greek": "Δ — % of notional ($USD)",
             "Strategy": "—",
             leg_labels[0]: f"{g.delta_a*100:+.2f}%  ({_fmt_signed_usd(delta_a_usd)})",
             leg_labels[1]: f"{g.delta_b*100:+.2f}%  ({_fmt_signed_usd(delta_b_usd)})",
             },
            {"Greek": "Γ — per 1% spot ($USD)",
             "Strategy": "—",
             leg_labels[0]: _fmt_signed_usd(gamma_a_usd),
             leg_labels[1]: _fmt_signed_usd(gamma_b_usd),
             },
            {"Greek": "ν — per 1 vol pt ($USD)",
             "Strategy": "—",
             leg_labels[0]: _fmt_signed_usd(vega_a_usd_per_vp),
             leg_labels[1]: _fmt_signed_usd(vega_b_usd_per_vp),
             },
            {"Greek": "∂V/∂ρ — per 1 rho pt ($USD)",
             "Strategy": _fmt_signed_usd(rho_sens_per_rp),
             leg_labels[0]: "—",
             leg_labels[1]: "—",
             },
            {"Greek": "Θ — per cal day ($USD)",
             "Strategy": _fmt_signed_usd(theta_usd_per_day),
             leg_labels[0]: "—",
             leg_labels[1]: "—",
             },
        ]

        df_g = pd.DataFrame(g_rows).set_index("Greek")
        st.dataframe(df_g[["Strategy"] + leg_labels],
                       use_container_width=True)
        st.caption(
            f"Greeks via {g.method} ({greek_ms:.0f} ms total). "
            f"Bumps: spot ±{g.bump_sizes['spot_frac']*100:.1f}%, "
            f"vol ±{g.bump_sizes['sigma_abs']*100:.0f} vol pt, "
            f"rho ±{g.bump_sizes['rho_abs']:.2f}, "
            f"theta {g.bump_sizes['theta_days']:.0f}d forward."
        )

        # Interpretation hint
        if g.rho_sensitivity > 0:
            rho_dir = "increases"
        elif g.rho_sensitivity < 0:
            rho_dir = "decreases"
        else:
            rho_dir = "unchanged"
        st.caption(
            f"💡 **Correlation interpretation**: A 0.01 increase in ρ "
            f"{rho_dir} the structure value by "
            f"{_fmt_signed_usd(rho_sens_per_rp)}. For UO×UO calls this "
            f"is typically positive (high ρ → both legs more likely "
            f"simultaneously alive AND ITM)."
        )


render()

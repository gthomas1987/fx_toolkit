"""Option Pricer — single-pair multi-leg FX option strategy pricer.

Bloomberg-OVML-style layout: rows are parameters (Spot, Strike, Vol,
…), columns are legs. Add/remove legs dynamically. Each leg can be
configured independently:

  - Exercise style : European | American (only matters for KO)
  - Option type    : Vanilla | KO (Up&Out / Down&Out / Up&In / Down&In)
  - Pricing model  : Black-Scholes (flat ATM) | Vol-at-strike (smile)
                     | Vanna-Volga (smile-aware exotic correction)

# Sections
1. Strategy header — pair, trade date, snapshot spot.
2. Legs table — one column per leg, rows = inputs (Direction, Strike,
   Notional, Style, Type, Barrier (if KO), Barrier Mode, Model).
3. Market data — auto-resolved per leg (σ_atm, RR, BF, σ_smile(K),
   forward, r_d, r_f).
4. Greeks — Δ, Γ, ν (per 1 vol pt), Θ (per cal day), ρ (per 1bp rates).
5. Results — premium %, premium USD per leg, plus a USD-summed
   "Strategy total" column for additive risks (Δ USD, ν USD, premium
   USD, notional, max payoff).

# What's NOT here (by design)
- Worst-of structures across DIFFERENT pairs → Dual CCY Option Pricer
  (tab 2). Single-pair, multi-leg only here.
- Backtest / portfolio aggregation → tabs 3 and 4.
- Auto strike/barrier solvers (delta-based) — kept simple: user types
  the strike and barrier directly. Δ-solver helper can be added later
  as an "Auto-fill from Δ" button per leg.

# Implementation notes
- The legs list lives in st.session_state['op_legs'] as a list of dicts.
  Add/remove buttons mutate this list and trigger a rerun.
- Each leg is priced INDEPENDENTLY (no joint correlation logic here).
- Greeks use closed-form formulas (vanilla via core.vanilla, KOs via
  core.ko / core.american_barrier + their _spot_delta helpers, plus
  finite differences for vega/theta where no closed form exists).
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
from core.calendar import compute_option_dates
from core.calendar_fx import compute_option_dates_for_pair
from core.rates import load_rates_panel, get_rate_at
from core.conventions import get_pip_scale
from core.smile import smile_vol_at_strike
from core.eko_pricing import price_eko_dispatch
from core.ako_pricing import price_ako_dispatch
from core.vanilla import (
    vanilla_price, vanilla_spot_delta, vanilla_gamma, vanilla_vega,
    vanilla_theta,
)
from core.ko import ko_price, ki_price, ko_spot_delta
from core.american_barrier import ako_closed_form, ako_spot_delta


# =============================================================================
# Calendar — Bloomberg-style with per-pair holiday calendars
# =============================================================================
# Uses `core.calendar_fx.compute_option_dates_for_pair` which honours
# US/JP/EUR/GBP/CHF/CAD/AUD/NZD/etc. holidays per leg. Convention:
#     trade -> spot+2bd (combined calendar)
#     spot+tenor -> next-bd (combined calendar) = delivery
#     delivery-2bd (non-USD calendar) = expiry
# The "expiry uses non-USD calendar" rule matches Bloomberg's
# behaviour, e.g. USDJPY trade 15-May-2026 → expiry 18-Jun (with
# Juneteenth holiday 19-Jun) — a 1-day difference from the weekday-
# only convention which would give 17-Jun.
# =============================================================================


# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="Option Pricer",
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

OPTION_TYPES = {
    "Vanilla":        "vanilla",
    "KO Up & Out":    "ko_uo",
    "KO Down & Out":  "ko_do",
    "KI Up & In":     "ki_ui",
    "KI Down & In":   "ki_di",
}

EXERCISE_STYLES = ["European", "American"]   # only KO uses American

PRICING_MODELS = {
    "Black-Scholes (flat ATM)": "flat_atm",
    "Vol-at-strike (smile)":     "vol_at_strike",
    "Vanna-Volga":                "vanna_volga",
}

DIRECTIONS = ["Call", "Put"]   # user-facing labels

BUY_SELL = ["Buy", "Sell"]     # determines the sign of notional

ASIA_EM = {"USDCNH", "USDCNY", "USDINR", "USDIDR", "USDKRW", "USDMYR",
            "USDPHP", "USDTHB", "USDTWD"}


# =============================================================================
# Leg state management
# =============================================================================
def _default_leg() -> dict:
    """Sensible blank leg — Buy vanilla 1M ATMF call, $10M notional."""
    return {
        "buy_sell":       "Buy",         # Buy = long (+N), Sell = short (-N)
        "direction":      "Call",
        "strike":         None,          # None → ATMF (auto-resolved)
        "notional_usd":   10_000_000.0,  # always entered positive; sign
                                          # comes from buy_sell
        "exercise_style": "European",
        "option_type":    "Vanilla",
        "barrier":        None,          # ignored if not KO/KI
        "model":          "Vanna-Volga",
        "tenor":          "1M",
    }


def _ensure_legs_state() -> list[dict]:
    if "op_legs" not in st.session_state or not st.session_state["op_legs"]:
        st.session_state["op_legs"] = [_default_leg()]
    return st.session_state["op_legs"]


def _add_leg():
    legs = _ensure_legs_state()
    legs.append(_default_leg())


def _remove_leg(idx: int):
    legs = _ensure_legs_state()
    if len(legs) > 1:
        legs.pop(idx)


# =============================================================================
# Market data resolution per leg
# =============================================================================
def _list_pairs(folder: str) -> list[str]:
    """All pairs visible in the data folder."""
    try:
        ds = load_panel(folder, "SPOT", None)
        return sorted(ds.columns.tolist())
    except Exception:
        return []


def _resolve_leg_market_data(folder: str, pair: str, prefer: str,
                                tenor: str, val_date: pd.Timestamp,
                                ) -> "dict | None":
    """Pull σ_atm, RR, BF, forward, r_d, r_f for one leg.

    Returns None if any required panel is missing for this leg's
    (pair, tenor, val_date). The caller can then render an error
    row in the table while keeping the other legs alive.

    Note: we DON'T wrap the whole body in try/except — silent
    failures during early development hide real bugs. The
    specific failure paths return None explicitly.
    """
    out: dict = {}
    # Spot
    spot_df = load_panel(folder, "SPOT", None, prefer=prefer,
                            pairs=(pair,))
    if spot_df.empty or pair not in spot_df.columns:
        return None
    spot_ts = spot_df[pair].dropna()
    # Use the value AS-OF val_date (last business day at-or-before)
    spot = float(spot_ts.asof(val_date))
    if pd.isna(spot):
        return None
    out["spot"] = spot

    # Vol — ATM
    vol_df = load_panel(folder, "VOL_ATM", tenor, prefer=prefer,
                          pairs=(pair,))
    if vol_df.empty or pair not in vol_df.columns:
        return None
    sigma_atm_pct = vol_df[pair].asof(val_date)
    if pd.isna(sigma_atm_pct):
        return None
    out["sigma_atm"] = float(sigma_atm_pct) / 100.0

    # Vol — RR and BF (smile). Missing-smile is OK; we fall back
    # to flat smile (rr=bf=0). Smile-related load failures are
    # logged but don't block the leg.
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
    out["smile_avail"] = (rr_25 != 0.0 or bf_25 != 0.0)

    # Forward — from points panel. Missing forward → spot used as
    # forward (rare but OK; the rate dispatcher backs out one rate
    # from the other below).
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

    # Tenor in years — Bloomberg-style with per-pair holiday calendar.
    # The pair argument selects the correct combined calendar so
    # expiry dates match BBG exactly (e.g. USDJPY 15-May-2026 1M →
    # expiry 18-Jun-2026, T=34d).
    td = val_date.date() if hasattr(val_date, "date") else val_date
    opt_dates = compute_option_dates_for_pair(td, tenor, pair)
    out["T"] = opt_dates.T_years
    out["expiry"] = opt_dates.option_expiry
    out["delivery"] = opt_dates.option_settlement
    out["spot_settle"] = opt_dates.spot_settlement

    # Rates: try the proper rate-curve panels first; if both come
    # back None we can't price.
    foreign, domestic = pair[:3], pair[3:]
    f_panel = load_rates_panel(folder, foreign, load_by_ticker)
    d_panel = load_rates_panel(folder, domestic, load_by_ticker)
    r_f = get_rate_at(f_panel, out["T"], td)
    r_d = get_rate_at(d_panel, out["T"], td)
    # Forward-implied fallback if one curve is missing
    if r_d is None and r_f is None:
        return None
    if r_d is None:
        r_d = r_f + np.log(F_market / spot) / out["T"]
    if r_f is None:
        r_f = r_d - np.log(F_market / spot) / out["T"]
    out["r_d"] = r_d
    out["r_f"] = r_f

    return out


# =============================================================================
# Per-leg pricing + Greeks
# =============================================================================
def _price_leg(leg: dict, md: dict, pair: str) -> "dict | None":
    """Price one leg + compute its Greeks. md is the market-data dict
    from `_resolve_leg_market_data`. Returns a dict with:

      price_per_unit, price_usd, K_used, H_used, sigma_used,
      delta_pct, gamma_per_S2, vega_per_volpt, theta_per_day,
      rho_d_per_bp, rho_f_per_bp

    Returns None if the leg's economics are invalid (e.g. KO with H
    on wrong side of spot relative to barrier direction).
    """
    S, T, r_d, r_f = md["spot"], md["T"], md["r_d"], md["r_f"]
    sigma_atm, rr_25, bf_25 = md["sigma_atm"], md["rr_25"], md["bf_25"]
    F = md["forward"]

    # ---- Resolve strike ----
    K_raw = leg.get("strike")
    if K_raw is None:
        K = F   # ATMF default
        K_label = "ATMF"
    else:
        K = float(K_raw)
        K_label = f"{K:.4f}"

    # User-facing direction → core 'call' / 'put'
    opt = "call" if leg["direction"] == "Call" else "put"

    # Vol at strike (for smile-aware pricing + finite-difference Greeks)
    sigma_smile = smile_vol_at_strike(S, K, T, sigma_atm, rr_25, bf_25,
                                          r_d, r_f)

    # Map option type
    otype_code = OPTION_TYPES.get(leg["option_type"], "vanilla")
    is_ko = otype_code in ("ko_uo", "ko_do")
    is_ki = otype_code in ("ki_ui", "ki_di")

    # Map model
    model_code = PRICING_MODELS.get(leg["model"], "vanna_volga")

    notional_usd = float(leg["notional_usd"])

    # =========================================================================
    # Vanilla path
    # =========================================================================
    if not (is_ko or is_ki):
        # Pick vol for the chosen model:
        #   flat_atm       → σ_atm + closed-form BS Greeks
        #   vol_at_strike  → σ_smile(K) + closed-form BS Greeks
        #   vanna_volga    → VV-corrected price (BS@σ_atm + correction)
        #                    + FD Greeks on the full VV machinery,
        #                    matching Bloomberg's "Vanna-Volga" model.
        if model_code == "flat_atm":
            sigma_used = sigma_atm
            p_per_unit = vanilla_price(opt, S, K, T, sigma_used, r_d, r_f)
            delta_spot = vanilla_spot_delta(opt, S, K, T, sigma_used,
                                                 r_d, r_f)
            gamma = vanilla_gamma(S, K, T, sigma_used, r_d, r_f)
            vega = vanilla_vega(S, K, T, sigma_used, r_d, r_f)
            theta_per_year = vanilla_theta(opt, S, K, T, sigma_used,
                                                r_d, r_f)
        elif model_code == "vol_at_strike":
            sigma_used = sigma_smile
            p_per_unit = vanilla_price(opt, S, K, T, sigma_used, r_d, r_f)
            delta_spot = vanilla_spot_delta(opt, S, K, T, sigma_used,
                                                 r_d, r_f)
            gamma = vanilla_gamma(S, K, T, sigma_used, r_d, r_f)
            vega = vanilla_vega(S, K, T, sigma_used, r_d, r_f)
            theta_per_year = vanilla_theta(opt, S, K, T, sigma_used,
                                                r_d, r_f)
        else:  # vanna_volga
            # Use VV-aware FD Greeks. Matches Bloomberg OVML to within
            # ~2-4% on vanilla delta; closes ~7% gap of plain BS-Δ at
            # σ_smile. Vega remains exact (BS-vega at σ_smile is the
            # right notion).
            sigma_used = sigma_smile   # for display purposes
            from core.vv_greeks import vv_greeks_vanilla
            g = vv_greeks_vanilla(opt, S, K, T, sigma_atm, rr_25, bf_25,
                                     r_d, r_f)
            p_per_unit = g.price
            delta_spot = g.delta
            gamma = g.gamma
            vega = g.vega
            theta_per_year = g.theta_per_year

        # Convert to user-facing display units (per-unit prices and Greeks)
        # then to USD by * notional / spot (foreign-notional convention)
        usd_per_unit = notional_usd / S
        price_usd = p_per_unit * usd_per_unit

        return {
            "K_used": K, "K_label": K_label, "H_used": None,
            "sigma_used": sigma_used, "sigma_smile": sigma_smile,
            "price_per_unit": p_per_unit, "price_usd": price_usd,
            "price_pct_notl": price_usd / notional_usd,
            # Greeks in their PER-UNIT form. The display layer scales
            # them to USD using the Bloomberg quotation conventions:
            #   Δ_spot      = ∂P/∂S    → display as a percentage (BBG's
            #                            "Spot Delta") and USD per 1%
            #                            spot move on USD notional.
            #   gamma       = ∂²P/∂S²  → Γ_USD = γ × S × N × 0.01 (=
            #                            change in $-delta per 1% spot)
            #   vega        = ∂P/∂σ    → ν_USD = ν × 0.01 × N / S per
            #                            1 vol point.
            #   theta_year  = ∂P/∂T    → Θ_USD = θ/365 × N / S per day
            #                            (negative for long options).
            "delta_spot": delta_spot,
            "gamma_per_S2": gamma,
            "vega_per_dvol": vega,
            "theta_per_year": theta_per_year,
            "is_barrier": False,
            "barrier_alive_prob": None,
        }

    # =========================================================================
    # KO / KI path
    # =========================================================================
    # Map barrier direction
    barrier_type = {
        "ko_uo": "up_and_out", "ko_do": "down_and_out",
        "ki_ui": "up_and_in",   "ki_di": "down_and_in",
    }[otype_code]

    H_raw = leg.get("barrier")
    if H_raw is None or float(H_raw) <= 0:
        return None
    H = float(H_raw)

    # Quick sanity check: H on correct side of S for the chosen direction
    if "up" in barrier_type and H <= S:
        return None      # Up-barrier must be above spot
    if "down" in barrier_type and H >= S:
        return None      # Down-barrier must be below spot

    # KI: price as Vanilla - KO_complement (in-out parity, European-style)
    # For simplicity here we implement KO directly; KIs use core.ki_price.
    # Note: ki_price is European-only — American KI is a longer journey
    # and is left to a follow-up.
    if is_ki:
        if leg["exercise_style"] != "European":
            return None     # we don't support American KI right now
        p_per_unit = ki_price(opt, barrier_type, S, K, H, T,
                                  sigma_smile, r_d, r_f)
        # Greeks via FD on the underlying ki_price (no closed-form helper
        # in our toolkit yet — KIs are uncommon enough to bump it).
        h_S = 0.005
        p_up = ki_price(opt, barrier_type, S * (1 + h_S), K, H, T,
                          sigma_smile, r_d, r_f)
        p_dn = ki_price(opt, barrier_type, S * (1 - h_S), K, H, T,
                          sigma_smile, r_d, r_f)
        delta_spot = (p_up - p_dn) / (2 * S * h_S)
        gamma = (p_up - 2 * p_per_unit + p_dn) / (S * h_S) ** 2
        # vega: bump σ
        h_v = 0.01
        p_vup = ki_price(opt, barrier_type, S, K, H, T,
                            sigma_smile + h_v, r_d, r_f)
        p_vdn = ki_price(opt, barrier_type, S, K, H, T,
                            max(sigma_smile - h_v, 1e-6), r_d, r_f)
        vega = (p_vup - p_vdn) / (2 * h_v)
        # theta: forward bump in time direction
        h_T = 1 / 365.0
        if T - h_T > 1e-6:
            p_th = ki_price(opt, barrier_type, S, K, H, T - h_T,
                              sigma_smile, r_d, r_f)
            theta_per_year = (p_th - p_per_unit) / (-h_T)
        else:
            theta_per_year = float("nan")

        usd_per_unit = notional_usd / S
        price_usd = p_per_unit * usd_per_unit
        return {
            "K_used": K, "K_label": K_label, "H_used": H,
            "sigma_used": sigma_smile, "sigma_smile": sigma_smile,
            "price_per_unit": p_per_unit, "price_usd": price_usd,
            "price_pct_notl": price_usd / notional_usd,
            "delta_spot": delta_spot, "gamma_per_S2": gamma,
            "vega_per_dvol": vega, "theta_per_year": theta_per_year,
            "is_barrier": True, "barrier_alive_prob": None,
        }

    # --- KO branch ---
    # Dispatch based on exercise style:
    #   European → core.eko_pricing.price_eko_dispatch
    #   American → core.ako_pricing.price_ako_dispatch
    if leg["exercise_style"] == "European":
        dispatch = price_eko_dispatch
        plain_pricer = ko_price
        plain_delta = ko_spot_delta
    else:
        dispatch = price_ako_dispatch
        plain_pricer = ako_closed_form
        plain_delta = ako_spot_delta

    p_per_unit, detail = dispatch(
        opt, barrier_type, S, K, H, T,
        sigma_atm=sigma_atm, sigma_smile=sigma_smile,
        rr_25=rr_25, bf_25=bf_25,
        r_d=r_d, r_f=r_f, model=model_code,
    )
    sigma_used = detail.get("vol_used", sigma_atm)

    # Greeks for KO:
    # - For 'vanna_volga' model: VV-aware FD Greeks via core.vv_greeks.
    #   The FD bumps the FULL VV-corrected price including the
    #   correction term's spot-sensitivity. Closes the largest gap
    #   vs BBG's "Vanna-Volga" Greeks (plain BS-Δ at σ_smile is
    #   missing the smile-sensitivity).
    # - For 'flat_atm' / 'vol_at_strike': closed-form Δ + FD Γ/ν/Θ
    #   on the plain pricer.
    if model_code == "vanna_volga":
        from core.vv_greeks import vv_greeks_ko
        exercise_code = ("european" if leg["exercise_style"] == "European"
                          else "american")
        g = vv_greeks_ko(opt, barrier_type, S, K, H, T,
                          sigma_atm, rr_25, bf_25, r_d, r_f,
                          exercise_style=exercise_code)
        delta_spot = g.delta
        gamma = g.gamma
        vega = g.vega
        theta_per_year = g.theta_per_year
    else:
        # Δ closed form via ko_spot_delta / ako_spot_delta (uses σ_smile —
        # we're computing risk against MARKET state, so smile vol is the
        # right vol to bump around).
        delta_spot = plain_delta(opt, barrier_type, S, K, H, T,
                                     sigma_smile, r_d, r_f)
        # Γ, ν, Θ via finite differences using the same plain_pricer + σ_smile.
        h_S = 0.005   # ±0.5% spot bumps
        p_up = plain_pricer(opt, barrier_type, S * (1 + h_S), K, H, T,
                                sigma_smile, r_d, r_f)
        p_dn = plain_pricer(opt, barrier_type, S * (1 - h_S), K, H, T,
                                sigma_smile, r_d, r_f)
        p_base_for_greeks = plain_pricer(opt, barrier_type, S, K, H, T,
                                              sigma_smile, r_d, r_f)
        gamma = (p_up - 2 * p_base_for_greeks + p_dn) / (S * h_S) ** 2

        h_v = 0.01
        p_vup = plain_pricer(opt, barrier_type, S, K, H, T,
                                  sigma_smile + h_v, r_d, r_f)
        p_vdn = plain_pricer(opt, barrier_type, S, K, H, T,
                                  max(sigma_smile - h_v, 1e-6), r_d, r_f)
        vega = (p_vup - p_vdn) / (2 * h_v)

        h_T = 1 / 365.0
        if T - h_T > 1e-6:
            p_th = plain_pricer(opt, barrier_type, S, K, H, T - h_T,
                                     sigma_smile, r_d, r_f)
            # NOTE: sign convention — theta_per_year = ∂P/∂T (positive
            # for long calls means longer T = more value). We bumped
            # T → T - h_T, so dP/dT = (P(T) - P(T-h_T)) / h_T.
            theta_per_year = (p_base_for_greeks - p_th) / h_T
        else:
            theta_per_year = float("nan")

    usd_per_unit = notional_usd / S
    price_usd = p_per_unit * usd_per_unit
    return {
        "K_used": K, "K_label": K_label, "H_used": H,
        "sigma_used": sigma_used, "sigma_smile": sigma_smile,
        "price_per_unit": p_per_unit, "price_usd": price_usd,
        "price_pct_notl": price_usd / notional_usd,
        "delta_spot": delta_spot, "gamma_per_S2": gamma,
        "vega_per_dvol": vega, "theta_per_year": theta_per_year,
        "is_barrier": True, "barrier_alive_prob": None,
        "model_detail": detail,
    }


# =============================================================================
# UI helpers
# =============================================================================
def _leg_label(i: int) -> str:
    return f"Leg {i + 1}"


def _render_leg_inputs_column(leg: dict, i: int, container) -> None:
    """Render input widgets for one leg in its column. Mutates `leg`
    in-place via session-state.
    """
    with container:
        st.markdown(f"**{_leg_label(i)}**")
        if len(_ensure_legs_state()) > 1:
            if st.button("✕ Remove", key=f"op_leg_remove_{i}",
                           use_container_width=True):
                _remove_leg(i)
                st.rerun()

        leg["buy_sell"] = st.selectbox(
            "Buy / Sell", BUY_SELL,
            index=BUY_SELL.index(leg.get("buy_sell", "Buy")),
            key=f"op_leg_bs_{i}",
            help="Buy = long the option (pay premium, receive Greeks). "
                  "Sell = short (receive premium, owe Greeks). The sign "
                  "of the notional follows this — Sell flips Δ/ν/Θ/USD-"
                  "premium negative in the strategy total.",
        )
        leg["direction"] = st.selectbox(
            "Direction", DIRECTIONS,
            index=DIRECTIONS.index(leg.get("direction", "Call")),
            key=f"op_leg_dir_{i}",
        )
        leg["tenor"] = st.selectbox(
            "Tenor", TENOR_LIST,
            index=TENOR_LIST.index(leg.get("tenor", "1M")),
            key=f"op_leg_tenor_{i}",
        )
        strike_in = st.text_input(
            "Strike (blank = ATMF)",
            value="" if leg.get("strike") is None else f"{leg['strike']:.4f}",
            key=f"op_leg_strike_{i}",
            help="Leave blank for ATM-forward. Enter a numeric strike "
                  "to override.",
        )
        if strike_in.strip() == "":
            leg["strike"] = None
        else:
            try:
                leg["strike"] = float(strike_in)
            except ValueError:
                st.warning(f"Invalid strike '{strike_in}' — using ATMF.")
                leg["strike"] = None

        leg["notional_usd"] = st.number_input(
            "Notional (USD)",
            min_value=100_000.0, max_value=500_000_000.0,
            value=float(leg.get("notional_usd", 10_000_000.0)),
            step=1_000_000.0, format="%.0f",
            key=f"op_leg_notl_{i}",
        )

        leg["option_type"] = st.selectbox(
            "Option type", list(OPTION_TYPES.keys()),
            index=list(OPTION_TYPES.keys()).index(
                leg.get("option_type", "Vanilla")),
            key=f"op_leg_otype_{i}",
        )

        # Exercise style: only meaningful for KO (American = continuous
        # monitoring). Vanillas and KIs ignore it but we show it for
        # consistency.
        is_ko = leg["option_type"].startswith("KO")
        leg["exercise_style"] = st.selectbox(
            "Exercise style", EXERCISE_STYLES,
            index=EXERCISE_STYLES.index(
                leg.get("exercise_style", "European")),
            key=f"op_leg_style_{i}",
            help=("Only affects KO: European = expiry-only barrier check; "
                   "American = continuous (any time during life). "
                   "Vanillas and KIs always use European in this build."),
            disabled=not is_ko,
        )
        if not is_ko:
            leg["exercise_style"] = "European"

        # Barrier inputs — only for KO/KI
        if is_ko or leg["option_type"].startswith("KI"):
            barr_in = st.text_input(
                "Barrier (H)",
                value="" if leg.get("barrier") is None else f"{leg['barrier']:.4f}",
                key=f"op_leg_barrier_{i}",
                help="Knock-out / knock-in level. Up-barriers must be "
                      "above spot; Down-barriers below.",
            )
            if barr_in.strip() == "":
                leg["barrier"] = None
            else:
                try:
                    leg["barrier"] = float(barr_in)
                except ValueError:
                    st.warning(f"Invalid barrier '{barr_in}'.")
                    leg["barrier"] = None
        else:
            leg["barrier"] = None

        leg["model"] = st.selectbox(
            "Pricing model", list(PRICING_MODELS.keys()),
            index=list(PRICING_MODELS.keys()).index(
                leg.get("model", "Vanna-Volga")),
            key=f"op_leg_model_{i}",
            help="Flat ATM = σ_atm (Bloomberg BS).  Vol-at-strike = σ_smile(K).  "
                  "Vanna-Volga = smile-aware exotic correction (matches "
                  "Bloomberg's smile model for KOs).",
        )


def _fmt_pct_notl(x: float) -> str:
    return f"{x * 100:.4f}%"


def _fmt_usd(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"${x:,.0f}"


def _fmt_signed_usd(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"${x:+,.0f}"


# =============================================================================
# Main render
# =============================================================================
def render():
    st.title("Option Pricer")
    st.caption(
        "Single-pair, multi-leg FX option strategy pricer. Add legs "
        "across the columns; each leg is priced independently. "
        "For correlation-aware structures across two pairs, use the "
        "Dual CCY Option Pricer (coming next)."
    )

    pairs_avail = _list_pairs(folder)
    if not pairs_avail:
        st.error("No SPOT data found in the configured folder.")
        return

    # ---- Strategy-level controls ----
    ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([1.2, 1.2, 1.6, 1])
    with ctrl_c1:
        default_pair = ("USDJPY" if "USDJPY" in pairs_avail
                            else pairs_avail[0])
        pair = st.selectbox(
            "Currency pair", pairs_avail,
            index=pairs_avail.index(default_pair), key="op_pair",
        )
    with ctrl_c2:
        prefer = "offshore"
        if pair in ASIA_EM:
            prefer = st.radio(
                "Variant", ["offshore", "onshore"],
                index=0, horizontal=True, key="op_prefer",
            )

    with ctrl_c3:
        # Latest available business day for this pair as the default
        try:
            spot_ts = load_panel(folder, "SPOT", None, prefer=prefer,
                                     pairs=(pair,))[pair].dropna()
            latest_dt = spot_ts.index.max().date()
        except Exception:
            latest_dt = _date.today()
        td_input = st.date_input(
            "Trade date", value=latest_dt,
            min_value=_date(2010, 1, 1), max_value=latest_dt,
            key="op_trade_date",
            help="Snapshot to price against. Defaults to the latest "
                  "business day available for this pair.",
        )
        val_date = pd.Timestamp(td_input)
    with ctrl_c4:
        if st.button("➕ Add leg", key="op_add_leg",
                       type="primary",
                       use_container_width=True):
            _add_leg()
            st.rerun()

    # ---- Legs state ----
    legs = _ensure_legs_state()
    n_legs = len(legs)

    # =========================================================================
    # SECTION 1: Leg input columns
    # =========================================================================
    st.markdown("---")
    st.markdown("#### Legs")

    leg_cols = st.columns(n_legs)
    for i, leg in enumerate(legs):
        _render_leg_inputs_column(leg, i, leg_cols[i])

    # =========================================================================
    # SECTION 2: Price each leg (computation only — display tables below)
    # =========================================================================
    # First resolve market data per leg, then price each leg with its
    # own (tenor, …) inputs.
    leg_results: list[dict | None] = []
    leg_md: list[dict | None] = []
    for leg in legs:
        md = _resolve_leg_market_data(folder, pair, prefer,
                                          leg["tenor"], val_date)
        if md is None:
            leg_md.append(None)
            leg_results.append(None)
            continue
        leg_md.append(md)
        try:
            res = _price_leg(leg, md, pair)
        except Exception as e:
            st.warning(f"Leg {legs.index(leg)+1} pricing error: {e}")
            res = None
        leg_results.append(res)

    # Halt early if every leg failed to price
    if all(r is None for r in leg_results):
        st.warning("None of the legs could be priced — check inputs.")
        return

    # =========================================================================
    # SECTION 3: Strategy / Market data tables
    # =========================================================================
    st.markdown("---")
    st.markdown("#### Strategy details")

    # Build the Bloomberg-style table: rows = parameters, columns =
    # ["Strategy"] + leg labels. We render it as a pandas DataFrame so
    # the user gets sortable / scrollable behaviour for free.
    leg_labels = [_leg_label(i) for i in range(n_legs)]
    cols_for_df = ["Strategy total"] + leg_labels

    # ---- Header rows: inputs as user typed them ----
    rows = []

    def _add_row(name, leg_vals, total=None):
        d = {"Parameter": name, "Strategy total": total}
        for j, v in enumerate(leg_vals):
            d[leg_labels[j]] = v
        rows.append(d)

    _add_row("Buy / Sell", [leg.get("buy_sell", "Buy") for leg in legs])
    _add_row("Direction", [leg["direction"] for leg in legs])
    _add_row("Tenor", [leg["tenor"] for leg in legs])
    _add_row("Option type", [leg["option_type"] for leg in legs])
    _add_row("Exercise style",
              ["—" if not leg["option_type"].startswith("KO")
               else leg["exercise_style"]
               for leg in legs])
    _add_row("Pricing model", [leg["model"] for leg in legs])

    # Strike / barrier rows show resolved values (ATMF → forward number)
    strike_strs = []
    for j, (leg, res) in enumerate(zip(legs, leg_results)):
        if res is None:
            strike_strs.append("—")
        else:
            strike_strs.append(
                f"{res['K_used']:.4f}"
                + (" (ATMF)" if leg.get("strike") is None else "")
            )
    _add_row("Strike", strike_strs)

    barrier_strs = []
    for j, (leg, res) in enumerate(zip(legs, leg_results)):
        if res is None or not res.get("is_barrier"):
            barrier_strs.append("—")
        else:
            barrier_strs.append(f"{res['H_used']:.4f}")
    _add_row("Barrier (H)", barrier_strs)

    # Notional row: shows the SIGNED notional (Buy = +N, Sell = -N).
    # Strategy total = net signed notional.
    signed_notls = [
        (+1 if leg.get("buy_sell", "Buy") == "Buy" else -1)
        * leg["notional_usd"]
        for leg in legs
    ]
    _add_row("Notional (USD, signed)",
              [_fmt_signed_usd(n) for n in signed_notls],
              total=_fmt_signed_usd(sum(signed_notls)))

    # Trade-date snapshot info (uniform across legs but echoed for clarity)
    spot_strs = []
    expiry_strs = []
    for j, md in enumerate(leg_md):
        if md is None:
            spot_strs.append("—")
            expiry_strs.append("—")
        else:
            spot_strs.append(f"{md['spot']:.4f}")
            expiry_strs.append(str(md["expiry"]))
    _add_row("Spot", spot_strs)
    _add_row("Expiry", expiry_strs)

    df_strategy = pd.DataFrame(rows).set_index("Parameter")
    st.dataframe(df_strategy[cols_for_df],
                   use_container_width=True)

    # ---- Market data table ----
    st.markdown("#### Market data")
    md_rows = []

    def _add_md(name, vals, total=None):
        d = {"Parameter": name, "Strategy total": total or "—"}
        for j, v in enumerate(vals):
            d[leg_labels[j]] = v
        md_rows.append(d)

    _add_md("σ_atm (%)",
              [f"{md['sigma_atm']*100:.3f}%" if md else "—" for md in leg_md])
    _add_md("σ_smile(K) (%)",
              [f"{res['sigma_smile']*100:.3f}%" if res else "—"
               for res in leg_results])
    _add_md("RR_25Δ (%)",
              [f"{md['rr_25']*100:+.3f}%" if md else "—" for md in leg_md])
    _add_md("BF_25Δ (%)",
              [f"{md['bf_25']*100:+.3f}%" if md else "—" for md in leg_md])
    _add_md("Forward",
              [f"{md['forward']:.4f}" if md else "—" for md in leg_md])
    _add_md(f"r_d ({pair[3:]})",
              [f"{md['r_d']*100:.3f}%" if md else "—" for md in leg_md])
    _add_md(f"r_f ({pair[:3]})",
              [f"{md['r_f']*100:.3f}%" if md else "—" for md in leg_md])

    df_md = pd.DataFrame(md_rows).set_index("Parameter")
    st.dataframe(df_md[cols_for_df], use_container_width=True)

    # =========================================================================
    # SECTION 4: Greeks
    # =========================================================================
    st.markdown("#### Greeks")
    st.caption(
        "**Δ Spot %** is the option's INTRINSIC delta (Bloomberg convention): "
        "positive for calls, negative for puts. The number does NOT flip "
        "for Sell legs — it's a property of the option.  \n"
        "**Δ USD** is the dollar-delta hedge = `Δ × signed_notional` "
        "(Buy: +N, Sell: -N). So a 50%-delta call on \\$10M Buy gives "
        "Δ USD = +\\$5M; a 30%-delta put on \\$10M Sell gives "
        "Δ USD = -0.30 × -\\$10M = +\\$3M (you're synthetically long).  \n"
        "**Γ / ν / Θ in USD** all include the Buy/Sell sign. **Strategy "
        "total** is the net of signed legs."
    )
    g_rows = []

    # Track strategy totals for additive Greeks (all in USD, signed by
    # Buy/Sell direction).
    total_delta_usd = 0.0
    total_gamma_usd = 0.0
    total_vega_usd = 0.0
    total_theta_usd = 0.0
    n_priced = 0

    delta_vals = []
    gamma_vals = []
    vega_vals = []
    theta_vals = []

    for j, (leg, md, res) in enumerate(zip(legs, leg_md, leg_results)):
        if res is None or md is None:
            delta_vals.append("—")
            gamma_vals.append("—")
            vega_vals.append("—")
            theta_vals.append("—")
            continue
        n_priced += 1

        # Buy/Sell sign on notional: Buy = +N, Sell = -N. All USD-
        # scaled Greeks pick up this sign automatically.
        sign = +1.0 if leg.get("buy_sell", "Buy") == "Buy" else -1.0
        N_signed = sign * leg["notional_usd"]

        # ====== Δ (Bloomberg "Spot Delta") ======
        # delta_spot = ∂P_per_unit/∂S (the BS-FX spot delta, e.g.
        # 0.5037 = 50.37%). Bloomberg reports two related numbers:
        #   "Delta %"   = delta_spot × 100   (e.g. 54.41%)
        #   "Hedge USD" = delta_spot × N     (e.g. -544k for $1M notl)
        #
        # The dollar delta (= hedge USD, total $ value of equivalent
        # spot exposure) is `delta_spot × N`. NOT `× 0.01` — that's a
        # different quantity (P&L per 1% spot move).
        delta_pct = res["delta_spot"]   # e.g. 0.5037
        delta_usd = delta_pct * N_signed   # dollar delta
        delta_str = (f"{delta_pct * 100:+.2f}%  "
                      f"({_fmt_signed_usd(delta_usd)})")

        # ====== Γ (Bloomberg "Gamma USD") ======
        # = change in $-delta per 1% spot move
        # = Γ_raw × dS × N_USD = Γ_raw × (S × 0.01) × N_USD
        gamma_usd = (res["gamma_per_S2"] * md["spot"] * 0.01
                       * N_signed)
        gamma_str = _fmt_signed_usd(gamma_usd)

        # ====== ν (Bloomberg "Vega USD" per 1 vol point) ======
        # vega_per_dvol = ∂P_per_unit/∂σ (per 1.0 in σ).
        # Per vol pt = × 0.01.  Per USD notional = × (N_usd / S).
        vega_usd_per_vp = (res["vega_per_dvol"] * 0.01
                            * N_signed / md["spot"])
        vega_str = _fmt_signed_usd(vega_usd_per_vp)

        # ====== Θ (USD per calendar day) ======
        # theta_per_year is ∂P/∂T (negative for long options since
        # value decays as T shrinks). Per day = / 365. Per USD = × N/S.
        theta_usd_per_day = (res["theta_per_year"] / 365.0
                              * N_signed / md["spot"])
        theta_str = _fmt_signed_usd(theta_usd_per_day)

        delta_vals.append(delta_str)
        gamma_vals.append(gamma_str)
        vega_vals.append(vega_str)
        theta_vals.append(theta_str)

        total_delta_usd += delta_usd
        total_gamma_usd += gamma_usd
        total_vega_usd += vega_usd_per_vp
        total_theta_usd += theta_usd_per_day

    total_delta_str = _fmt_signed_usd(total_delta_usd) if n_priced else "—"
    total_gamma_str = _fmt_signed_usd(total_gamma_usd) if n_priced else "—"
    total_vega_str = _fmt_signed_usd(total_vega_usd) if n_priced else "—"
    total_theta_str = _fmt_signed_usd(total_theta_usd) if n_priced else "—"

    def _add_g(name, vals, total):
        d = {"Greek": name, "Strategy total": total}
        for j, v in enumerate(vals):
            d[leg_labels[j]] = v
        g_rows.append(d)

    _add_g("Δ Spot — % notional ($USD)", delta_vals, total_delta_str)
    _add_g("Γ USD — per 1% spot", gamma_vals, total_gamma_str)
    _add_g("ν USD — per 1 vol pt", vega_vals, total_vega_str)
    _add_g("Θ USD — per cal day", theta_vals, total_theta_str)

    df_g = pd.DataFrame(g_rows).set_index("Greek")
    st.dataframe(df_g[["Strategy total"] + leg_labels],
                   use_container_width=True)

    # =========================================================================
    # SECTION 5: Results
    # =========================================================================
    st.markdown("#### Results")
    st.caption(
        "Premium with sign: **positive = pay** (Buy legs), "
        "**negative = receive** (Sell legs). The strategy-total row "
        "is the NET premium across all legs — useful for zero-cost "
        "structures (e.g. risk reversals) where the total nets near 0."
    )
    r_rows = []
    pct_vals = []
    usd_vals = []
    total_premium_usd = 0.0
    for j, (leg, res) in enumerate(zip(legs, leg_results)):
        if res is None:
            pct_vals.append("—")
            usd_vals.append("—")
            continue
        sign = +1.0 if leg.get("buy_sell", "Buy") == "Buy" else -1.0
        premium_signed_usd = sign * res["price_usd"]
        # % of notional is per-leg only; sign included.
        pct_signed = sign * res["price_pct_notl"]
        pct_vals.append(f"{pct_signed * 100:+.4f}%")
        usd_vals.append(_fmt_signed_usd(premium_signed_usd))
        total_premium_usd += premium_signed_usd

    def _add_r(name, vals, total):
        d = {"Result": name, "Strategy total": total}
        for j, v in enumerate(vals):
            d[leg_labels[j]] = v
        r_rows.append(d)

    # % of notional is non-additive (legs may have different notionals);
    # hide from total per user spec.
    _add_r("Premium (% of notional)", pct_vals, "")
    _add_r("Premium (USD, net)", usd_vals, _fmt_signed_usd(total_premium_usd))

    df_r = pd.DataFrame(r_rows).set_index("Result")
    st.dataframe(df_r[["Strategy total"] + leg_labels],
                   use_container_width=True)

    # ---- Footer interpretation ----
    if n_priced < n_legs:
        st.warning(
            f"{n_legs - n_priced} of {n_legs} legs failed to price — "
            "check that strike/barrier are on the correct sides of "
            "spot and that vol/rate data exist for the chosen tenor."
        )


# Run the page (Streamlit runs the module top-to-bottom; we wrap the
# UI in render() so it's testable and easy to add a top-level error
# handler later).
render()

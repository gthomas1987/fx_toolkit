"""Portfolio definition for app 11 — 10 leveraged FX option trades.

Each Trade is a self-describing dataclass. The pricing engine knows how to
value any trade against a market snapshot and produce Greeks.

Conventions:
    - All notionals in USD.
    - For pair XXX/USD (e.g. EURUSD), notional_usd corresponds to USD face
      and FOR face = notional_usd / spot. We price in DOM per FOR unit then
      multiply.
    - For pair USD/XXX (e.g. USDJPY), notional_usd is the USD face directly
      (foreign in our convention). We price the option struck in DOM (JPY)
      per 1 unit USD, then multiply by USD notional to get JPY P&L, then
      divide by current JPY/USD spot to get USD P&L.

This is a simplified P&L convention sufficient for risk monitoring on a
prototype.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np

from .vanilla import (vanilla_price, vanilla_delta, vanilla_gamma, vanilla_vega,
                       vanilla_vanna, vanilla_volga, vanilla_charm, vanilla_theta)
from .eko import eko_price, survival_prob
from .dual_eko import Leg as DualLeg, dual_eko_price


# =============================================================================
# Trade dataclass
# =============================================================================

@dataclass
class TradeLeg:
    """One European option leg of a structure."""
    opt: str                  # "call" | "put"
    K: float
    H: float | None = None    # barrier (None for vanilla)
    bar_dir: str | None = None  # "up_and_out" | "down_and_out" | None
    qty: float = 1.0          # +1 = long, -1 = short, +/-2 = fly body, etc.


@dataclass
class Trade:
    trade_id: str
    booking_date: pd.Timestamp
    expiry_date: pd.Timestamp
    pair: str                          # "USDJPY", or "USDJPY/AUDUSD" for duals
    structure: str                     # "vanilla" | "call_spread" | "call_fly"
                                       # | "eko" | "dual_eko"
    legs: list[TradeLeg]               # legs for the main pair
    side: str = "buy"                  # "buy" | "sell" (sign of premium and Greeks)
    notional_usd: float = 1e7
    premium_paid_usd: float = 0.0      # signed: + = paid, - = received
    # For dual_eko only
    pair2: str | None = None
    legs2: list[TradeLeg] | None = None
    rho_traded: float | None = None    # correlation assumed at trade
    structure_kind: str | None = None  # "wo_call" / "wo_put" / "bo_call" / "bo_put"
    # Notes for UI
    notes: str = ""

    def days_to_expiry(self, asof: pd.Timestamp) -> int:
        return max(0, (self.expiry_date - pd.Timestamp(asof)).days)

    def life_pct(self, asof: pd.Timestamp) -> float:
        total = (self.expiry_date - self.booking_date).days
        if total <= 0:
            return 1.0
        elapsed = (pd.Timestamp(asof) - self.booking_date).days
        return min(1.0, max(0.0, elapsed / total))


# =============================================================================
# Build the 10-trade portfolio
# =============================================================================

def build_portfolio() -> list[Trade]:
    """The 10-trade book. Booking dates in Mar/Apr 2026; expiries Apr/Aug 2026.

    All trades are sized at a uniform **$100M notional** for ease of
    cross-trade comparison in the risk dashboards. Premiums have been
    scaled proportionally from their originally booked notionals so the
    per-unit-notional cost (the price the desk paid) is preserved.
    """
    trades = [
        # ---- Trade 1: USDJPY 1M ATM call (long, vanilla outright) ----
        Trade(
            trade_id="T1",
            booking_date=pd.Timestamp("2026-04-22"),
            expiry_date=pd.Timestamp("2026-05-22"),
            pair="USDJPY",
            structure="vanilla",
            legs=[TradeLeg(opt="call", K=156.50)],
            side="buy",
            notional_usd=100_000_000,
            premium_paid_usd=1_040_000,
            notes="USDJPY upside via 1M ATM call. Carry-driven bull view.",
        ),
        # ---- Trade 2: USDJPY 3M 153 put (long) ----
        Trade(
            trade_id="T2",
            booking_date=pd.Timestamp("2026-03-20"),
            expiry_date=pd.Timestamp("2026-06-20"),
            pair="USDJPY",
            structure="vanilla",
            legs=[TradeLeg(opt="put", K=153.00)],
            side="buy",
            notional_usd=100_000_000,
            premium_paid_usd=1_266_667,
            notes="3M USDJPY put. Risk-off / MOF intervention hedge against USDJPY rally.",
        ),
        # ---- Trade 3: EURUSD 1M 1.13 call (long) ----
        Trade(
            trade_id="T3",
            booking_date=pd.Timestamp("2026-04-10"),
            expiry_date=pd.Timestamp("2026-05-22"),
            pair="EURUSD",
            structure="vanilla",
            legs=[TradeLeg(opt="call", K=1.1300)],
            side="buy",
            notional_usd=100_000_000,
            premium_paid_usd=725_000,
            notes="EURUSD 6W upside on ECB dovish-pivot expectations.",
        ),
        # ---- Trade 4: EURUSD 2M call spread (long 1.10 / short 1.13) ----
        Trade(
            trade_id="T4",
            booking_date=pd.Timestamp("2026-04-01"),
            expiry_date=pd.Timestamp("2026-06-01"),
            pair="EURUSD",
            structure="call_spread",
            legs=[
                TradeLeg(opt="call", K=1.1000, qty=+1.0),
                TradeLeg(opt="call", K=1.1300, qty=-1.0),
            ],
            side="buy",
            notional_usd=100_000_000,
            premium_paid_usd=775_000,
            notes="2M EURUSD call spread — bullish into Fed June meeting; capped upside.",
        ),
        # ---- Trade 5: AUDUSD 3M call spread (long 0.66 / short 0.69) ----
        Trade(
            trade_id="T5",
            booking_date=pd.Timestamp("2026-03-15"),
            expiry_date=pd.Timestamp("2026-06-15"),
            pair="AUDUSD",
            structure="call_spread",
            legs=[
                TradeLeg(opt="call", K=0.6600, qty=+1.0),
                TradeLeg(opt="call", K=0.6900, qty=-1.0),
            ],
            side="buy",
            notional_usd=100_000_000,
            premium_paid_usd=866_667,
            notes="3M AUDUSD call spread — China reopening + commodity rebound.",
        ),
        # ---- Trade 6: AUDUSD 2M call fly (long 0.665 / -2 × 0.685 / long 0.705) ----
        Trade(
            trade_id="T6",
            booking_date=pd.Timestamp("2026-04-05"),
            expiry_date=pd.Timestamp("2026-06-05"),
            pair="AUDUSD",
            structure="call_fly",
            legs=[
                TradeLeg(opt="call", K=0.6650, qty=+1.0),
                TradeLeg(opt="call", K=0.6850, qty=-2.0),
                TradeLeg(opt="call", K=0.7050, qty=+1.0),
            ],
            side="buy",
            notional_usd=100_000_000,
            premium_paid_usd=210_000,
            notes="2M AUDUSD call fly centred on 0.685 — pinning trade, cheap convexity if AUDUSD stalls there.",
        ),
        # ---- Trade 7: USDJPY 4M UO call EKO (K=152, H=162) ----
        Trade(
            trade_id="T7",
            booking_date=pd.Timestamp("2026-03-25"),
            expiry_date=pd.Timestamp("2026-07-25"),
            pair="USDJPY",
            structure="eko",
            legs=[TradeLeg(opt="call", K=152.00, H=162.00, bar_dir="up_and_out")],
            side="buy",
            notional_usd=100_000_000,
            premium_paid_usd=525_000,
            notes="4M reverse-KO USDJPY call (K=152, H=162). "
                  "Cheaper than vanilla; KO'd if spot fixes ≥162 at expiry.",
        ),
        # ---- Trade 8: EURUSD 3M DO put EKO (K=1.08, H=1.04) ----
        Trade(
            trade_id="T8",
            booking_date=pd.Timestamp("2026-04-20"),
            expiry_date=pd.Timestamp("2026-07-20"),
            pair="EURUSD",
            structure="eko",
            legs=[TradeLeg(opt="put", K=1.0800, H=1.0400, bar_dir="down_and_out")],
            side="buy",
            notional_usd=100_000_000,
            premium_paid_usd=342_857,
            notes="3M reverse-KO EURUSD put (K=1.08, H=1.04). "
                  "Cheap tail hedge; KO'd if EURUSD breaks below 1.04 at expiry.",
        ),
        # ---- Trade 9: Dual EKO worst-of CALL on (USDJPY, USDCNH) ----
        Trade(
            trade_id="T9",
            booking_date=pd.Timestamp("2026-03-10"),
            expiry_date=pd.Timestamp("2026-07-10"),
            pair="USDJPY",
            pair2="USDCNH",
            structure="dual_eko",
            structure_kind="wo_call",
            legs=[TradeLeg(opt="call", K=153.00, H=161.00, bar_dir="up_and_out")],
            legs2=[TradeLeg(opt="call", K=7.2300, H=7.4000, bar_dir="up_and_out")],
            rho_traded=0.45,
            side="buy",
            notional_usd=100_000_000,
            premium_paid_usd=580_000,
            notes="4M worst-of dual EKO call on USDJPY × USDCNH. "
                  "Bullish dollar-bloc bet; pays the weaker performer subject to both UO barriers.",
        ),
        # ---- Trade 10: Dual EKO worst-of PUT on (EURUSD, GBPUSD) ----
        Trade(
            trade_id="T10",
            booking_date=pd.Timestamp("2026-04-08"),
            expiry_date=pd.Timestamp("2026-07-08"),
            pair="EURUSD",
            pair2="GBPUSD",
            structure="dual_eko",
            structure_kind="wo_put",
            legs=[TradeLeg(opt="put", K=1.0900, H=1.0500, bar_dir="down_and_out")],
            legs2=[TradeLeg(opt="put", K=1.2800, H=1.2300, bar_dir="down_and_out")],
            rho_traded=0.65,
            side="buy",
            notional_usd=100_000_000,
            premium_paid_usd=400_000,
            notes="3M worst-of dual EKO put on EURUSD × GBPUSD. "
                  "Cheap pound/euro tail hedge; pays the weaker performer if both stay above DO barriers.",
        ),
    ]
    return trades


# =============================================================================
# Rate inference
# =============================================================================

CCY_RATES = {  # baseline continuous rates (used as fallback)
    "USD": 0.045, "EUR": 0.025, "JPY": 0.005, "GBP": 0.040,
    "AUD": 0.038, "CAD": 0.040, "CHF": 0.010, "CNH": 0.020,
}


def infer_rates(pair: str, snap_pair: dict, tenor_yrs: float
                ) -> tuple[float, float]:
    """Infer (r_d, r_f) for the pair.

    Use forward points to back out (r_d - r_f), and the baseline CCY_RATES
    for the absolute level of one leg.

    F = S * exp((r_d - r_f) * T)
    ⇒ r_d - r_f = ln(F/S) / T
    """
    from .conventions import get_pip_scale
    spot = snap_pair["spot"]
    # closest tenor in fwd_pts
    fpts = snap_pair.get("fwd_pts", {})
    # map to T years
    TT = {"1M": 30/365, "3M": 90/365, "6M": 182/365, "1Y": 365/365}
    if fpts:
        # pick the closest available tenor
        best = min(fpts.keys(), key=lambda t: abs(TT.get(t, 1) - tenor_yrs))
        pts = fpts[best]
        F = spot + pts * get_pip_scale(pair)
        r_diff = np.log(F / spot) / TT.get(best, tenor_yrs)
    else:
        r_diff = 0.0

    # Anchor to USD rate; the other leg is residual.
    r_usd = CCY_RATES["USD"]
    foreign = pair[3:] if pair[:3] == "USD" else pair[:3]
    r_other = CCY_RATES.get(foreign, 0.03)
    if pair[:3] == "USD":
        # F = S * exp((r_other - r_usd) * T) ⇒ r_other - r_usd = r_diff
        # In GK with S=USD/XXX, the convention is r_d = XXX rate (domestic) and r_f = USD (foreign).
        r_d = r_other  # approximate from baseline
        r_f = r_d - r_diff  # consistent with fwd
    else:
        # EUR/USD: S = USD per 1 EUR. DOM=USD, FOR=EUR.
        r_d = r_usd
        r_f = r_d - r_diff
    return float(r_d), float(r_f)

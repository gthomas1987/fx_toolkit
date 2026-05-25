"""Calibrate the Option Pricer against a Bloomberg OVML reference.

Uses the BBG screenshot for USDJPY 4-leg strategy on 15-May-2026 as
the reference. Feeds the same market data directly to our pricers
(bypassing data loading) so we isolate pricing-model differences from
data/calendar issues.

# Reference (BBG screenshot)
- Spot           = 158.75 mid
- Forward (1M)   = 158.30 (points = -44.85)
- ATM vol (1M)   = 7.582%
- 25Δ RR (USD)   = -1.517%
- 25Δ BF         = 0.267%
- USD SOFR       = 3.603%  (USD = foreign for USDJPY)
- JPY (implied)  = 0.601%  (JPY = domestic for USDJPY)
- Trade date     = 15-May-2026 → expiry 18-Jun-2026 → T = 34/365 = 0.0932y
- Notional       = USD 1MM per leg

# Findings (recorded for future debugging)
VEGA:    Essentially exact on vanillas (within 0.2-3%). Confirms our
         pricing core is sound and our vol-smile interpolation is close.
PREMIUM: Within 1.5-4% on vanillas, 7-11% on KOs. Primary driver is
         calendar (BBG uses T=34/365 vs ours T=33/365 — see the
         holiday-calendar TODO in option_pricer.py). KOs additionally
         show small VV-correction differences.
GAMMA:   ~9% LOW on vanillas, scaled correctly via
            Gamma_USD = Γ × S × N_USD × 0.01
         (= USD change in $-delta per 1% spot move). The 9% gap is
         calendar + σ-smile differences.
DELTA:   ~7-8% LOW on vanillas — does NOT close with calendar fix
         alone. Most likely Bloomberg's "Spot Delta" under the VV
         model includes ∂(VV_correction)/∂S, which we omit. Our delta
         is BS-Δ at σ_smile(K). Implementing full VV-Δ requires
         bumping the VV correction numerically. TODO.

# Action items (priority order)
  1. Add US/JP holiday calendar so expiry matches BBG exactly (fixes
     the 1-day T gap → ~1.5% of vanilla premium error)
  2. Implement VV-aware Greeks (fixes ~7-8% delta gap)
  3. Cross-check σ_smile interpolation beyond 25Δ wings against BBG
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np

from core.vanilla import (
    vanilla_price, vanilla_spot_delta, vanilla_gamma, vanilla_vega,
)
from core.ko import ko_price, ko_spot_delta
from core.american_barrier import ako_closed_form, ako_spot_delta
from core.smile import smile_vol_at_strike
from core.eko_pricing import price_eko_dispatch
from core.ako_pricing import price_ako_dispatch


# =============================================================================
# Market data
# =============================================================================
S = 158.75
F = 158.30
SIGMA_ATM = 7.582 / 100.0
RR_25 = -1.517 / 100.0
BF_25 = 0.267 / 100.0
R_D = 0.601 / 100.0     # JPY = domestic for USDJPY
R_F = 3.603 / 100.0     # USD = foreign for USDJPY
T = 34.0 / 365.0        # 15-May -> 18-Jun
NOTIONAL_USD = 1_000_000.0

F_check = S * np.exp((R_D - R_F) * T)
print(f"Forward parity check: BBG F = {F:.4f}, computed F = {F_check:.4f}, "
       f"diff = {(F_check - F) / F * 100:+.3f}%")


# =============================================================================
# Conversion helpers (BBG quotation conventions)
# =============================================================================
def _premium_pct(p: float) -> float:
    """Per-unit price (JPY per USD) -> % of USD notional."""
    return p / S

def _premium_usd(p: float) -> float:
    return _premium_pct(p) * NOTIONAL_USD

def _vega_usd(v: float) -> float:
    """v = dP/dsigma per 1.0 sigma -> per 1 vol pt USD on USD notional."""
    return v * 0.01 * NOTIONAL_USD / S

def _gamma_usd(g: float) -> float:
    """g = d^2P/dS^2 -> change in $-delta per 1% spot move
       = Gamma * dS * N_USD = Gamma * (S * 0.01) * N_USD."""
    return g * S * 0.01 * NOTIONAL_USD


def _report(label, ours, bbg):
    """ours/bbg = (pct, usd, delta, vega_usd, gamma_usd)."""
    def _d(x, y):
        if y == 0 or y is None or not np.isfinite(y):
            return "-"
        return f"{(x - y) / y * 100:+.2f}%"
    print(f"\n{label}")
    print(f"  {'':<14s} {'BBG':>14s} {'Ours':>14s} {'Diff':>10s}")
    pct_b, usd_b, d_b, v_b, g_b = bbg
    pct_o, usd_o, d_o, v_o, g_o = ours
    print(f"  {'Premium %':<14s} {pct_b*100:>13.4f}% {pct_o*100:>13.4f}% "
           f"{_d(pct_o, pct_b):>10s}")
    print(f"  {'Premium USD':<14s} {'$' + format(usd_b, ',.2f'):>14s} "
           f"{'$' + format(usd_o, ',.2f'):>14s} {_d(usd_o, usd_b):>10s}")
    print(f"  {'Delta %':<14s} {d_b*100:>13.4f}% {d_o*100:>13.4f}% "
           f"{_d(d_o, d_b):>10s}")
    print(f"  {'Vega USD':<14s} {v_b:>14,.2f} {v_o:>14,.2f} "
           f"{_d(v_o, v_b):>10s}")
    print(f"  {'Gamma USD':<14s} {g_b:>14,.0f} {g_o:>14,.0f} "
           f"{_d(g_o, g_b):>10s}")


print()
print("=" * 78)
print("CALIBRATION: option_pricer vs Bloomberg OVML, USDJPY 4-leg, 15-May-2026")
print("=" * 78)
print(f"Inputs: S={S}, T={T:.4f}y ({T*365:.0f}d), r_d_JPY={R_D*100:.3f}%, "
       f"r_f_USD={R_F*100:.3f}%")


# =============================================================================
# Leg 1: Vanilla European Call, K = ATMF = 158.30
# =============================================================================
K1 = 158.30
sigma1 = smile_vol_at_strike(S, K1, T, SIGMA_ATM, RR_25, BF_25, R_D, R_F)
p1 = vanilla_price("call", S, K1, T, sigma1, R_D, R_F)
d1 = vanilla_spot_delta("call", S, K1, T, sigma1, R_D, R_F)
v1 = vanilla_vega(S, K1, T, sigma1, R_D, R_F)
g1 = vanilla_gamma(S, K1, T, sigma1, R_D, R_F)
print(f"\n  Leg 1: sigma_smile(K1) = {sigma1*100:.3f}%")
_report(
    "Leg 1 -- Vanilla European ATMF Call",
    (_premium_pct(p1), _premium_usd(p1), d1, _vega_usd(v1), _gamma_usd(g1)),
    (0.009365, 9365.43, 0.5441, 1211.32, 188658.40),
)


# =============================================================================
# Leg 2: Vanilla European Call, K = 160.00
# =============================================================================
K2 = 160.00
sigma2 = smile_vol_at_strike(S, K2, T, SIGMA_ATM, RR_25, BF_25, R_D, R_F)
p2 = vanilla_price("call", S, K2, T, sigma2, R_D, R_F)
d2 = vanilla_spot_delta("call", S, K2, T, sigma2, R_D, R_F)
v2 = vanilla_vega(S, K2, T, sigma2, R_D, R_F)
g2 = vanilla_gamma(S, K2, T, sigma2, R_D, R_F)
print(f"\n  Leg 2: sigma_smile(K2) = {sigma2*100:.3f}%")
_report(
    "Leg 2 -- Vanilla European K=160 Call",
    (_premium_pct(p2), _premium_usd(p2), d2, _vega_usd(v2), _gamma_usd(g2)),
    (0.004673, 4673.07, 0.3447, 1123.83, 187525.25),
)


# =============================================================================
# Leg 3: KO American Up&Out Call, K=160, H=164
# =============================================================================
K3, H3 = 160.00, 164.00
p3, det3 = price_ako_dispatch(
    "call", "up_and_out", S, K3, H3, T,
    sigma_atm=SIGMA_ATM, sigma_smile=sigma2,
    rr_25=RR_25, bf_25=BF_25, r_d=R_D, r_f=R_F,
    model="vanna_volga",
)
d3 = ako_spot_delta("call", "up_and_out", S, K3, H3, T, sigma2, R_D, R_F)
h_v = 0.01
h_S = 0.005
v3 = (ako_closed_form("call", "up_and_out", S, K3, H3, T, sigma2 + h_v, R_D, R_F)
      - ako_closed_form("call", "up_and_out", S, K3, H3, T, max(sigma2 - h_v, 1e-6), R_D, R_F)) / (2 * h_v)
p3_up = ako_closed_form("call", "up_and_out", S * (1 + h_S), K3, H3, T, sigma2, R_D, R_F)
p3_dn = ako_closed_form("call", "up_and_out", S * (1 - h_S), K3, H3, T, sigma2, R_D, R_F)
p3_bs_base = ako_closed_form("call", "up_and_out", S, K3, H3, T, sigma2, R_D, R_F)
g3 = (p3_up - 2 * p3_bs_base + p3_dn) / (S * h_S) ** 2
print(f"\n  Leg 3: VV corr = {det3.get('correction'):.6f}, "
       f"BS = {det3.get('price_bs'):.6f}, VV = {p3:.6f}")
_report(
    "Leg 3 -- KO American Up&Out Call",
    (_premium_pct(p3), _premium_usd(p3), d3, _vega_usd(v3), _gamma_usd(g3)),
    (0.002307, 2306.86, 0.0733, -167.50, -49669.51),
)


# =============================================================================
# Leg 4: KO European Up&Out Call, K=160, H=164
# =============================================================================
K4, H4 = 160.00, 164.00
p4, det4 = price_eko_dispatch(
    "call", "up_and_out", S, K4, H4, T,
    sigma_atm=SIGMA_ATM, sigma_smile=sigma2,
    rr_25=RR_25, bf_25=BF_25, r_d=R_D, r_f=R_F,
    model="vanna_volga",
)
d4 = ko_spot_delta("call", "up_and_out", S, K4, H4, T, sigma2, R_D, R_F)
v4 = (ko_price("call", "up_and_out", S, K4, H4, T, sigma2 + h_v, R_D, R_F)
      - ko_price("call", "up_and_out", S, K4, H4, T, max(sigma2 - h_v, 1e-6), R_D, R_F)) / (2 * h_v)
p4_up = ko_price("call", "up_and_out", S * (1 + h_S), K4, H4, T, sigma2, R_D, R_F)
p4_dn = ko_price("call", "up_and_out", S * (1 - h_S), K4, H4, T, sigma2, R_D, R_F)
p4_bs_base = ko_price("call", "up_and_out", S, K4, H4, T, sigma2, R_D, R_F)
g4 = (p4_up - 2 * p4_bs_base + p4_dn) / (S * h_S) ** 2
print(f"\n  Leg 4: VV corr = {det4.get('correction'):.6f}, "
       f"BS = {det4.get('price_bs'):.6f}, VV = {p4:.6f}")
_report(
    "Leg 4 -- KO European Up&Out Call",
    (_premium_pct(p4), _premium_usd(p4), d4, _vega_usd(v4), _gamma_usd(g4)),
    (0.003170, 3170.24, 0.1701, 220.19, 27165.74),
)


# =============================================================================
# Totals
# =============================================================================
print()
print("=" * 78)
print("Strategy totals (sum of per-leg premiums)")
print("=" * 78)
total_usd_ours = (_premium_usd(p1) + _premium_usd(p2) + _premium_usd(p3)
                    + _premium_usd(p4))
total_usd_bbg = 9365.43 + 4673.07 + 2306.86 + 3170.24
print(f"  BBG  total: ${total_usd_bbg:>10,.2f}")
print(f"  Ours total: ${total_usd_ours:>10,.2f}")
print(f"  Diff: {(total_usd_ours - total_usd_bbg) / total_usd_bbg * 100:+.2f}%")

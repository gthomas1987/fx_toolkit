"""Generate synthetic FX data folder for testing the Worst-of stack.

Schema (per `core/data_loader.py`):
  _index.csv columns: pair, category, tenor, csv_filename,
                       onshore_offshore, bbg_ticker
  Per-CSV: date column + value column.

This generates:
  - SPOT, VOL_ATM, VOL_25R, VOL_25B, FWD_POINTS panels for USDJPY,
    EURUSD, and EURJPY (the implied cross of the first two).
  - USD and JPY OIS rate panels via their Bloomberg tickers.

The vols for EURJPY are CONSISTENT with a chosen ρ=0.30 between
USDJPY and EURUSD via the triangulation identity, so triangulated
correlation reads of this folder will recover that ρ. We add a small
amount of jitter on top so the triangulated ρ varies trade-by-trade
in a realistic way (~±0.02).
"""
from __future__ import annotations
import os, sys, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("/tmp/wop_test_data")
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

end = pd.Timestamp("2025-04-30")
start = end - pd.DateOffset(years=2)
dates = pd.bdate_range(start, end)
n = len(dates)

rng = np.random.default_rng(42)
STANDARD_TENORS = ["1M", "2M", "3M", "6M", "9M", "1Y"]


def _gbm(s0, vol, n, rng):
    dt = 1 / 252
    z = rng.standard_normal(n)
    log_r = -0.5 * vol**2 * dt + vol * np.sqrt(dt) * z
    return s0 * np.exp(np.cumsum(log_r))


def _jitter(val, n, jit, rng):
    return val + rng.normal(0, jit, n)


def _write(path, dates, values, col="value"):
    pd.DataFrame({"date": dates, col: values}).to_csv(path, index=False)


index_rows = []

# -----------------------------------------------------------------------------
# Pair config — note EURJPY's vol is COMPUTED from USDJPY+EURUSD+rho
# at each (date, tenor), not chosen independently.
# -----------------------------------------------------------------------------
TRUE_RHO_FOR_TRIANGULATION = 0.30

pairs_legs = {
    "USDJPY": dict(spot=145.0, vol=8.0, rr=-1.5, bf=0.30, fwd_pip=-2.5),
    "EURUSD": dict(spot=1.08, vol=7.5, rr=-0.3, bf=0.20, fwd_pip=1.5),
}
# Will be computed below from USDJPY + EURUSD per (date, tenor)
pairs_cross = {
    "EURJPY": dict(spot=145.0 * 1.08, rr=0.50, bf=0.40, fwd_pip=-3.0),
}

# Pre-store per-tenor (per-pair) vol time series for the LEG pairs so we
# can derive the cross's vol consistently.
leg_vol_by_pair_tenor: dict[tuple[str, str], np.ndarray] = {}

for pair, cfg in pairs_legs.items():
    spot_vals = _gbm(cfg["spot"], cfg["vol"] / 100.0, n, rng)
    fname = f"{pair}_SPOT.csv"
    _write(OUT / fname, dates, spot_vals)
    index_rows.append(dict(pair=pair, category="SPOT", tenor="",
                            csv_filename=fname,
                            onshore_offshore="OFFSHORE", bbg_ticker=""))
    for tenor in STANDARD_TENORS:
        tenor_yr = {"1M": 1/12, "2M": 2/12, "3M": 3/12, "6M": 6/12,
                     "9M": 9/12, "1Y": 1.0}[tenor]
        # ATM vol (percent)
        atm_pct = _jitter(cfg["vol"], n, 0.2, rng)
        leg_vol_by_pair_tenor[(pair, tenor)] = atm_pct
        for cat, vals in [("VOL_ATM", atm_pct),
                            ("VOL_25R", _jitter(cfg["rr"], n, 0.05, rng)),
                            ("VOL_25B", _jitter(cfg["bf"], n, 0.05, rng))]:
            fname = f"{pair}_{cat}_{tenor}.csv"
            _write(OUT / fname, dates, vals)
            index_rows.append(dict(pair=pair, category=cat, tenor=tenor,
                                    csv_filename=fname,
                                    onshore_offshore="OFFSHORE",
                                    bbg_ticker=""))
        fwd_val = cfg["fwd_pip"] * 12 * tenor_yr
        vals = _jitter(fwd_val, n, 0.1, rng)
        fname = f"{pair}_FWD_POINTS_{tenor}.csv"
        _write(OUT / fname, dates, vals)
        index_rows.append(dict(pair=pair, category="FWD_POINTS", tenor=tenor,
                                csv_filename=fname,
                                onshore_offshore="OFFSHORE", bbg_ticker=""))

# -----------------------------------------------------------------------------
# EURJPY — cross pair. Vol panel is COMPUTED per (date, tenor) from
# USDJPY + EURUSD + chosen ρ so the triangulation formula recovers ρ.
# Pair convention: USDJPY × EURUSD share USD on opposite sides → η = -1
# σ_EURJPY^2 = σ_USDJPY^2 + σ_EURUSD^2 - 2 · η · ρ · σ_USDJPY · σ_EURUSD
#            = σ_USDJPY^2 + σ_EURUSD^2 + 2 · ρ · σ_USDJPY · σ_EURUSD
# -----------------------------------------------------------------------------
eurjpy_cfg = pairs_cross["EURJPY"]

# SPOT for EURJPY — approximate via USDJPY × EURUSD
spot_eurjpy = _gbm(eurjpy_cfg["spot"], 9.0 / 100.0, n, rng)
fname = "EURJPY_SPOT.csv"
_write(OUT / fname, dates, spot_eurjpy)
index_rows.append(dict(pair="EURJPY", category="SPOT", tenor="",
                        csv_filename=fname,
                        onshore_offshore="OFFSHORE", bbg_ticker=""))

# Per-tenor vol panels
for tenor in STANDARD_TENORS:
    sigma_a_pct = leg_vol_by_pair_tenor[("USDJPY", tenor)]
    sigma_b_pct = leg_vol_by_pair_tenor[("EURUSD", tenor)]
    sigma_a = sigma_a_pct / 100.0
    sigma_b = sigma_b_pct / 100.0
    # Add jitter to the implied rho so the cross vol varies trade-by-trade
    # in a realistic way (otherwise triangulation would always return
    # exactly 0.30, which is boring to inspect).
    rho_jitter = rng.normal(0, 0.05, n)
    rho_series = TRUE_RHO_FOR_TRIANGULATION + rho_jitter
    sigma_x = np.sqrt(sigma_a**2 + sigma_b**2 + 2 * rho_series * sigma_a * sigma_b)
    sigma_x_pct = sigma_x * 100.0
    fname = f"EURJPY_VOL_ATM_{tenor}.csv"
    _write(OUT / fname, dates, sigma_x_pct)
    index_rows.append(dict(pair="EURJPY", category="VOL_ATM", tenor=tenor,
                            csv_filename=fname,
                            onshore_offshore="OFFSHORE", bbg_ticker=""))
    # RR and BF for completeness (not used by triangulation but live
    # pricer code may try to read them).
    for cat, base, jit in [("VOL_25R", eurjpy_cfg["rr"], 0.05),
                              ("VOL_25B", eurjpy_cfg["bf"], 0.05)]:
        vals = _jitter(base, n, jit, rng)
        fname = f"EURJPY_{cat}_{tenor}.csv"
        _write(OUT / fname, dates, vals)
        index_rows.append(dict(pair="EURJPY", category=cat, tenor=tenor,
                                csv_filename=fname,
                                onshore_offshore="OFFSHORE", bbg_ticker=""))
    # Forward points
    tenor_yr = {"1M": 1/12, "2M": 2/12, "3M": 3/12, "6M": 6/12,
                 "9M": 9/12, "1Y": 1.0}[tenor]
    fwd_val = eurjpy_cfg["fwd_pip"] * 12 * tenor_yr
    vals = _jitter(fwd_val, n, 0.1, rng)
    fname = f"EURJPY_FWD_POINTS_{tenor}.csv"
    _write(OUT / fname, dates, vals)
    index_rows.append(dict(pair="EURJPY", category="FWD_POINTS", tenor=tenor,
                            csv_filename=fname,
                            onshore_offshore="OFFSHORE", bbg_ticker=""))

# -----------------------------------------------------------------------------
# Extra pairs needed by the Portfolio Analyzer page's `build_portfolio()`
# dummy book (AUDUSD, GBPUSD, USDCNH). Generated as plain GBMs (no
# correlation constraint), with their own VOL_ATM / VOL_RR_25D /
# VOL_BF_25D / FWD_POINTS panels so the page can price every leg of
# every trade.
# -----------------------------------------------------------------------------
PORTFOLIO_EXTRAS = {
    "AUDUSD": dict(spot=0.65, vol=10.0, rr=-0.4, bf=0.25, fwd_pip=0.5),
    "GBPUSD": dict(spot=1.27, vol=8.0,  rr=-0.5, bf=0.20, fwd_pip=0.5),
    "USDCNH": dict(spot=7.20, vol=5.0,  rr=0.3,  bf=0.15, fwd_pip=-2.0),
}
for pair, cfg in PORTFOLIO_EXTRAS.items():
    spot_vals = _gbm(cfg["spot"], cfg["vol"] / 100.0, n, rng)
    fname = f"{pair}_SPOT.csv"
    _write(OUT / fname, dates, spot_vals)
    index_rows.append(dict(pair=pair, category="SPOT", tenor="",
                            csv_filename=fname,
                            onshore_offshore="OFFSHORE", bbg_ticker=""))
    for tenor in STANDARD_TENORS:
        tenor_yr = {"1M": 1/12, "2M": 2/12, "3M": 3/12, "6M": 6/12,
                     "9M": 9/12, "1Y": 1.0}[tenor]
        atm = _jitter(cfg["vol"], n, 0.2, rng)
        for cat, vals in [("VOL_ATM", atm),
                            # Note: the Portfolio Analyzer (via core/data_loader's
                            # `snapshot`) expects VOL_RR_25D / VOL_BF_25D
                            # naming for the 25-delta wings.
                            ("VOL_RR_25D", _jitter(cfg["rr"], n, 0.05, rng)),
                            ("VOL_BF_25D", _jitter(cfg["bf"], n, 0.05, rng))]:
            fname = f"{pair}_{cat}_{tenor}.csv"
            _write(OUT / fname, dates, vals)
            index_rows.append(dict(pair=pair, category=cat, tenor=tenor,
                                    csv_filename=fname,
                                    onshore_offshore="OFFSHORE",
                                    bbg_ticker=""))
        fwd_val = cfg["fwd_pip"] * 12 * tenor_yr
        vals = _jitter(fwd_val, n, 0.1, rng)
        fname = f"{pair}_FWD_POINTS_{tenor}.csv"
        _write(OUT / fname, dates, vals)
        index_rows.append(dict(pair=pair, category="FWD_POINTS", tenor=tenor,
                                csv_filename=fname,
                                onshore_offshore="OFFSHORE", bbg_ticker=""))

# -----------------------------------------------------------------------------
# Rates — USD + JPY + EUR OIS, via Bloomberg tickers.
# -----------------------------------------------------------------------------
from core.rates import RATE_TICKERS
rate_levels = {
    "USD": {"ON": 5.30, "1W": 5.30, "1M": 5.25, "2M": 5.20, "3M": 5.15,
              "6M": 5.00, "9M": 4.85, "1Y": 4.75},
    "JPY": {"ON": 0.10, "1W": 0.10, "1M": 0.15, "2M": 0.20, "3M": 0.25,
              "6M": 0.35, "9M": 0.45, "1Y": 0.55},
    "EUR": {"ON": 3.90, "1W": 3.90, "1M": 3.85, "2M": 3.80, "3M": 3.75,
              "6M": 3.60, "9M": 3.45, "1Y": 3.30},
}
for ccy, levels in rate_levels.items():
    tickers = RATE_TICKERS.get(ccy, {})
    for tenor, ticker in tickers.items():
        if tenor not in levels:
            continue
        vals = _jitter(levels[tenor], n, 0.05, rng)
        fname = f"{ccy}_OIS_{tenor}.csv"
        _write(OUT / fname, dates, vals)
        index_rows.append(dict(pair="", category="RATE", tenor=tenor,
                                csv_filename=fname, onshore_offshore="",
                                bbg_ticker=ticker))

df_idx = pd.DataFrame(index_rows)
df_idx.to_csv(OUT / "_index.csv", index=False)

print(f"Created synthetic data in {OUT}")
print(f"  pairs (legs):     {list(pairs_legs.keys())}")
print(f"  pairs (cross):    {list(pairs_cross.keys())}")
print(f"  pairs (extras):   {list(PORTFOLIO_EXTRAS.keys())}")
print(f"  dates: {dates.min().date()} -> {dates.max().date()} ({n} bdays)")
print(f"  index rows: {len(index_rows)}")
print(f"  ρ baked into cross vol: {TRUE_RHO_FOR_TRIANGULATION:+.2f} "
       f"(with daily jitter ±0.05)")

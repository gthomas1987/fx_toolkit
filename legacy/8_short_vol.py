"""App 8 — Systematic short-vol on selectable FX pair.

Run with:
    streamlit run apps/8_short_vol.py

Spec:
    Sell ATM straddle / 25Δ strangle / 10Δ strangle on a user-selected pair,
    weekly or monthly. One structure active at a time, rolled on expiry.
    Daily delta hedge or threshold-based hedge (0.5 / 1 / 2 % spot move).

Data sources:
    Two market-data folders supported. Folder 1 typically has `_index.csv`
    (the existing project convention); Folder 2 is optional and discovered
    by filename scan — useful for additional pairs without maintaining a
    separate index. Pairs are unioned across both folders.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import pandas as pd
import streamlit as st

from core.multi_folder_loader import (
    discovery_summary, list_pairs_with_full_set, load_panel_multi,
)
from core.strategy_engine_8 import (StrategyConfig8, run_strategy_8,
                                     summary_stats_8)
from core.strategy_dashboard_8 import render_dashboard_8, inject_dashboard_css


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="8 · Short-Vol Selling",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_dashboard_css()


# -----------------------------------------------------------------------------
# Sidebar — data folders
# -----------------------------------------------------------------------------
default_folder = os.environ.get("MARKET_DATA_DIR", "")
default_folder2 = os.environ.get("MARKET_DATA_DIR2", "")

st.sidebar.markdown("**Data sources**")
folder1 = st.sidebar.text_input(
    "Market data folder 1 (with `_index.csv`)",
    value=default_folder,
    help="Primary folder. If `_index.csv` exists, it's used; otherwise "
         "filenames are scanned for pair/category/tenor.",
    key="s8_folder1",
)
folder2 = st.sidebar.text_input(
    "Market data folder 2 (optional, filename scan)",
    value=default_folder2,
    help="Optional second folder. No `_index.csv` required — files are "
         "auto-discovered by filename. Pairs from this folder are added "
         "to the universe; if a pair exists in both, folder 1 wins.",
    key="s8_folder2",
)

folders: list[str] = [f for f in (folder1.strip(), folder2.strip()) if f]
if not folders:
    st.info("Specify at least one market-data folder in the sidebar.")
    st.stop()

# Validate folders
valid_folders: list[str] = []
for f in folders:
    if not Path(f).exists():
        st.sidebar.error(f"Folder does not exist: {f}")
    else:
        valid_folders.append(f)
if not valid_folders:
    st.stop()

# Discovery summary expander
with st.sidebar.expander("Discovered files", expanded=False):
    for f in valid_folders:
        s = discovery_summary(f)
        st.caption(f"**{s['folder']}**  ·  mode: `{s['mode']}`")
        if s["categories"]:
            cat_str = ", ".join(f"{k}={v}" for k, v in
                                sorted(s["categories"].items()))
        else:
            cat_str = "—"
        st.caption(f"  {s['n_pairs']} pairs across {s['n_files']} files. "
                   f"Categories (pair counts): {cat_str}.")
        if s["n_unparseable"] > 0:
            st.caption(f"  {s['n_unparseable']} unparseable files (ignored).")


# -----------------------------------------------------------------------------
# Sidebar — strategy params
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("**Structure**")
structure_label = st.sidebar.radio(
    "Type",
    ["ATM Straddle", "25Δ Strangle", "10Δ Strangle"],
    index=0, key="s8_structure",
)
structure_map = {
    "ATM Straddle": "atm_straddle",
    "25Δ Strangle": "strangle_25d",
    "10Δ Strangle": "strangle_10d",
}
structure = structure_map[structure_label]

cadence = st.sidebar.radio(
    "Cadence", ["weekly", "monthly"], index=1,
    horizontal=True, key="s8_cadence",
    help="Tenor of each option. weekly → 1W expiry, monthly → 1M expiry.",
)
tenor_label = "1W" if cadence == "weekly" else "1M"


# -----------------------------------------------------------------------------
# Sidebar — pair selection (depends on tenor)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _list_pairs_full(folders_tuple: tuple[str, ...], vol_tenor: str) -> list[str]:
    return list_pairs_with_full_set(folders_tuple, vol_tenor)


pairs_available = _list_pairs_full(tuple(valid_folders), tenor_label)
if not pairs_available:
    st.error(f"No pairs found with both SPOT and VOL_ATM at {tenor_label} "
             f"across the specified folders.")
    st.stop()

default_pair = "AUDCAD" if "AUDCAD" in pairs_available else pairs_available[0]
pair = st.sidebar.selectbox(
    "Currency pair",
    pairs_available,
    index=pairs_available.index(default_pair),
    key="s8_pair",
    help=f"{len(pairs_available)} pairs available with SPOT + VOL_ATM "
         f"at {tenor_label} across the specified folders.",
)

# Asia EM onshore/offshore
asia_em = pair in {"USDCNH", "USDCNY", "USDINR", "USDIDR", "USDKRW",
                   "USDMYR", "USDPHP", "USDTHB", "USDTWD"}
prefer = "offshore"
if asia_em:
    prefer = st.sidebar.radio(
        "Variant", ["offshore", "onshore"], index=0,
        horizontal=True, key="s8_prefer",
    )

# Sizing
st.sidebar.markdown("**Sizing**")
notional_usd = st.sidebar.number_input(
    "Notional per leg (USD)", min_value=1_000_000.0, max_value=200_000_000.0,
    value=10_000_000.0, step=1_000_000.0, format="%.0f",
    help="Same USD-equivalent notional applied to both call and put legs.",
    key="s8_notional",
)

# Hedging
st.sidebar.markdown("**Delta hedging**")
hedge_choice = st.sidebar.selectbox(
    "Mode",
    ["daily", "every 0.5% spot move", "every 1% spot move",
     "every 2% spot move"],
    index=0, key="s8_hedge_mode",
)
if hedge_choice == "daily":
    hedge_mode = "daily"
    hedge_threshold_pct = 0.0
else:
    hedge_mode = "threshold"
    hedge_threshold_pct = float(hedge_choice.split()[1].rstrip("%"))

# Costs
st.sidebar.markdown("**Costs**")
vol_bid_ask_pts = st.sidebar.number_input(
    "Vol bid-ask (vol points)", min_value=0.0, max_value=2.0,
    value=0.25, step=0.05, format="%.2f", key="s8_vol_ba",
)
spot_cost_bps = st.sidebar.number_input(
    "Spot cost (bps)", min_value=0.0, max_value=20.0,
    value=1.0, step=0.5, format="%.2f", key="s8_spot_cost",
)
usd_rate_pct = st.sidebar.number_input(
    "USD funding rate (%)", min_value=0.0, max_value=10.0,
    value=3.0, step=0.25, format="%.2f", key="s8_usd_rate",
)


# -----------------------------------------------------------------------------
# Title / caption
# -----------------------------------------------------------------------------
st.title(f"8 · {pair}  ·  Short {structure_label}, {cadence}")
hedge_caption = ("daily delta hedge" if hedge_mode == "daily"
                 else f"hedge every {hedge_threshold_pct:g}% spot move")
st.caption(
    f"Sell {tenor_label} {structure_label.lower()} on {pair} every "
    f"{'week' if cadence == 'weekly' else 'month'}  ·  {hedge_caption}  ·  "
    f"one structure at a time, rolled on expiry"
)


# -----------------------------------------------------------------------------
# Load market data — across both folders
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading market data…")
def load_inputs(folders_tuple: tuple[str, ...], pair: str, prefer: str,
                tenor: str
                ) -> tuple[pd.Series, pd.Series, pd.Series, bool]:
    """Return (spot, atm_vol_decimal, forward_at_tenor, fwd_was_available)."""
    spot_df = load_panel_multi(folders_tuple, "SPOT", None,
                                prefer=prefer, pairs=(pair,))
    vol_df = load_panel_multi(folders_tuple, "VOL_ATM", tenor,
                               prefer=prefer, pairs=(pair,))
    fwd_df = load_panel_multi(folders_tuple, "FWD_POINTS", tenor,
                               prefer=prefer, pairs=(pair,))
    if spot_df.empty or pair not in spot_df.columns:
        return (pd.Series(dtype=float), pd.Series(dtype=float),
                pd.Series(dtype=float), False)
    spot = spot_df[pair].dropna()
    vol = (vol_df[pair].dropna() / 100.0
           if not vol_df.empty and pair in vol_df.columns
           else pd.Series(dtype=float))

    # Forward = spot + fwd_points × pip_scale (conventional FX market quote)
    if not fwd_df.empty and pair in fwd_df.columns:
        try:
            from core.conventions import get_pip_scale
            pip = get_pip_scale(pair)
        except Exception:
            pip = 1e-4
        fwd_pts = fwd_df[pair].dropna()
        fwd = spot.add(fwd_pts * pip, fill_value=0).reindex(spot.index).ffill()
        return spot, vol, fwd, True
    # No forward — fall back to zero-carry (forward = spot)
    fwd = spot.copy()
    return spot, vol, fwd, False


spot, vol_atm, forward, fwd_available = load_inputs(
    tuple(valid_folders), pair, prefer, tenor_label,
)
if spot.empty or vol_atm.empty:
    st.error(f"Insufficient market data for {pair}. Need: SPOT and "
             f"VOL_ATM at {tenor_label} (found across folders).")
    st.stop()

if not fwd_available:
    st.warning(
        f"No FWD_POINTS at {tenor_label} found for {pair} in either folder. "
        f"Falling back to **zero-carry** assumption (forward = spot). "
        f"Strikes will be slightly off if the pair has meaningful carry."
    )

# Date filter
common_idx = spot.index.intersection(vol_atm.index).intersection(forward.index)
if len(common_idx) < 60:
    st.error(f"Less than 60 days of overlapping data for {pair} at "
             f"{tenor_label}.")
    st.stop()

date_min, date_max = common_idx.min(), common_idx.max()
date_range = st.sidebar.date_input(
    "Backtest range",
    value=(date_min.date(), date_max.date()),
    min_value=date_min.date(), max_value=date_max.date(),
    key="s8_date_range",
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    spot = spot.loc[start:end]
    vol_atm = vol_atm.loc[start:end]
    forward = forward.loc[start:end]


# -----------------------------------------------------------------------------
# Run strategy
# -----------------------------------------------------------------------------
config = StrategyConfig8(
    pair=pair, pair_label=pair,
    structure=structure, cadence=cadence,
    notional_usd=notional_usd,
    hedge_mode=hedge_mode, hedge_threshold_pct=hedge_threshold_pct,
    vol_bid_ask_pts=vol_bid_ask_pts, spot_cost_bps=spot_cost_bps,
    usd_rate_pct=usd_rate_pct,
)

with st.spinner(f"Running strategy 8 on {pair}…"):
    result = run_strategy_8(spot, vol_atm, forward, config)
    stats = summary_stats_8(result)


# -----------------------------------------------------------------------------
# Render dashboard
# -----------------------------------------------------------------------------
render_dashboard_8(result, stats)

with st.expander("Methodology notes"):
    st.markdown("""
- **Two-folder data discovery** — the app unions pairs across folder 1
  (with `_index.csv`) and folder 2 (filename-scanned). Folder 2 supports
  these filename conventions:
  `<PAIR>_<CATEGORY>_<TENOR>.csv` · `<PAIR>.csv` (→ SPOT) ·
  `<PAIR>_SPOT.csv` · `<PAIR><TENOR>V.csv` and `<PAIR>V<TENOR>.csv` (→ VOL_ATM).
  When a pair exists in both folders, folder 1 wins. The discovered-files
  expander in the sidebar shows what was picked up.
- **One structure at a time** — when the active structure expires (1W or 1M
  later), a new one is opened the same day. Both legs are short.
- **Strikes** — solved against the matching-tenor ATM vol at trade open.
  This is a flat-smile approximation. For OTM strangles (especially 10Δ),
  the actual market vol on the wing is typically higher than ATM, so the
  premium received here is a **lower bound** vs market.
- **Daily MTM** — uses the matching-tenor ATM vol throughout the trade life.
- **Per-leg delta hedging** — each option leg keeps its own spot sub-position
  so per-leg hedge P&L is fully attributable in the trades and hedge ledgers.
- **Threshold mode** — rebalances a leg's hedge only when spot has moved at
  least the threshold % from the leg's last hedge level. Reduces hedge TC at
  the cost of carrying open delta between rebalances.
- **r_d implied via covered interest parity** from the matching-tenor forward.
  r_f = sidebar USD funding rate. If forward is unavailable, falls back to
  zero-carry (forward = spot).
""")

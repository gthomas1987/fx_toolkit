"""Backtest Viewer — FX Option Strategy Analyzer.

Sub-app of the FX Toolkit. Reached via the landing page or sidebar
nav; not run directly. (Formerly: fx_strategy_analyzer.py.)

Finds the optimal combination of strike, KO strike, tenor and gate
across an FX option backtest CSV (e.g. USDJPY_All.csv produced by
the EKO or RKO pricer's "Backtest" tab).
"""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Backtest Viewer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Make sibling top-level packages (shared/, core/) importable when Streamlit
# executes this page out of pages/.
# ---------------------------------------------------------------------------
import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Global CSS — moved into shared/style.py so all pages stay aligned. The
# rules originated here in the analyzer (tightens padding, typography,
# st.metric values, tab labels) and are reapplied across the toolkit.
from shared.style import inject_base_css
inject_base_css()


# ---------------------------------------------------------------------------
# Password gate REMOVED in the merged FX Toolkit context — page is open by
# default. To re-enable an app-wide gate, wrap app.py instead of each page.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Where to find preloaded result files. Drop CSVs into ./data/ at the
# top of the fx_toolkit repo and they'll show up in the dropdown
# automatically. (The path is one level up because this page lives in
# pages/ — we want the top-level data/ folder, not pages/data/.)
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _is_timeseries_path(p: Path) -> bool:
    """True if the filename indicates a time-series file (vs a summary)."""
    return "timeseries" in p.stem.lower()


def list_preloaded_summaries() -> list[Path]:
    """Summary-style CSVs (legacy *_All.csv or new *_summary.csv).
    Time-series files are excluded — they're paired automatically."""
    if not DATA_DIR.exists():
        return []
    return sorted(p for p in DATA_DIR.glob("*.csv")
                  if not _is_timeseries_path(p))


def find_paired_timeseries(summary_path: Path) -> Optional[Path]:
    """Heuristic: given a summary file like 'backtest_bulk_summary.csv',
    look in the same directory for 'backtest_bulk_timeseries.csv'."""
    if summary_path is None:
        return None
    name = summary_path.stem
    candidates = [
        name.replace("_summary", "_timeseries"),
        name.replace("summary", "timeseries"),
        name.replace("_All", "_timeseries"),
    ]
    for c in candidates:
        if c == name:           # no-op replacement → not a real candidate
            continue
        p = summary_path.with_name(c + ".csv")
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
TENOR_TO_DAYS = {"1W": 7, "2W": 14, "3W": 21, "1M": 30, "6W": 42,
                 "2M": 60, "10W": 70, "3M": 90, "4M": 120, "6M": 180,
                 "9M": 270, "1Y": 365}


def parse_dollar(value) -> float:
    """Parse '$197.23M', '-$10.27M', '$1.2B', '$500K' into a float (USD)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    sign = -1 if s.startswith("-") else 1
    s = s.replace("-", "").replace("$", "").replace(",", "").strip()
    mult = 1.0
    if s.endswith("B"):
        mult, s = 1e9, s[:-1]
    elif s.endswith("M"):
        mult, s = 1e6, s[:-1]
    elif s.endswith("K") or s.endswith("k"):
        mult, s = 1e3, s[:-1]
    try:
        return sign * float(s) * mult
    except ValueError:
        return np.nan


def parse_pct(value) -> float:
    """Parse '41' or '41%' into 41.0."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def strike_to_numeric(strike: str) -> float:
    """ATM -> 50, '25Δ' -> 25, '45Δ' -> 45. Higher number = closer to spot."""
    if not isinstance(strike, str):
        return np.nan
    s = strike.strip()
    if s.upper() == "ATM":
        return 50.0
    m = re.match(r"(\d+(?:\.\d+)?)\s*[Δ∆D]", s)
    return float(m.group(1)) if m else np.nan


def ko_to_numeric(ko: str) -> float:
    """Maps a KO label to a numeric value useful for sorting.
       'H@5Δ' -> 5,  '8×' -> 8."""
    if not isinstance(ko, str):
        return np.nan
    m = re.search(r"@\s*(\d+(?:\.\d+)?)\s*[Δ∆D]", ko)
    if m:
        return float(m.group(1))
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*[x×X]", ko)
    if m:
        return float(m.group(1))
    return np.nan


def tenor_to_days(t: str) -> float:
    return TENOR_TO_DAYS.get(str(t).strip().upper(), np.nan)


# Matches the original delta-KO format AND the new payout-ratio format.
# KO clause is optional (some files put KO info elsewhere) and accepts either
# 'H@10Δ' or '8×'. Gate clause is also optional.
STRATEGY_RE = re.compile(
    r"""^\s*
        (?P<pair>[A-Z]{6})\s+
        (?P<type>\S+)\s+
        (?P<strike>ATM|\d+(?:\.\d+)?[Δ∆D])\s+
        (?P<tenor>\d+[WwMmYy])
        (?:\s+(?P<ko>H@\d+(?:\.\d+)?[Δ∆D]|\d+(?:\.\d+)?[x×X]))?
        (?:\s+\[(?P<gate>[^\]]+)\])?\s*$
    """,
    re.VERBOSE,
)

# Worst-of: "WO[ ...legs... ]  1M".
# `legs` greedily eats up to the OUTERmost closing bracket. Using a negated
# character class for the trailing `]` instead of lazy `.+?` is more robust
# when individual legs contain their own `[gate]` clauses.
WORSTOF_RE = re.compile(
    r"""^\s*WO\[\s*
        (?P<legs>.+)
        \]\s+(?P<tenor>\d+[WwMmYy])\s*$
    """,
    re.VERBOSE,
)

# Individual leg inside a worst-of: "AUDUSD Cuo 35Δ/H@10Δ" with an optional
# trailing "[gate condition]" attached to that leg.
LEG_RE = re.compile(
    r"""^\s*
        (?P<pair>[A-Z]{6})\s+
        (?P<type>[A-Za-z][A-Za-z0-9\-]*)\s+
        (?P<strike>ATM|\d+(?:\.\d+)?[Δ∆D])
        \s*/\s*
        (?P<ko>H@\d+(?:\.\d+)?[Δ∆D]|\d+(?:\.\d+)?[x×X])
        (?:\s+\[(?P<gate>[^\]]+)\])?
        \s*$
    """,
    re.VERBOSE,
)

# Type 3 — EKO_BASKET single-leg structure. The basket is implicit (no per-pair
# breakdown in the name). Example:
#   "BASKET  Call-UO  1M  ATM/H@20Δ"
#   "BASKET  Call-UO  2M  ATM/H@20Δ  [Spot > 50DMA]"
BASKET_RE = re.compile(
    r"""^\s*BASKET\s+
        (?P<type>[A-Za-z][A-Za-z0-9\-]*)\s+
        (?P<tenor>\d+[WwMmYy])\s+
        (?P<strike>ATM|\d+(?:\.\d+)?[Δ∆D])
        \s*/\s*
        (?P<ko>H@\d+(?:\.\d+)?[Δ∆D]|\d+(?:\.\d+)?[x×X])
        (?:\s+\[(?P<gate>[^\]]+)\])?
        \s*$
    """,
    re.VERBOSE,
)

# Type 4 — WO_BASKET. Two basket-legs separated by ∧, each with its own gate.
# Example:
#   "WO-BASKET  Call-UO  1M  A:ATM/H@20Δ [no gate]  ∧  B:ATM/H@20Δ [no gate]"
WO_BASKET_RE = re.compile(
    r"""^\s*WO-BASKET\s+
        (?P<type>[A-Za-z][A-Za-z0-9\-]*)\s+
        (?P<tenor>\d+[WwMmYy])\s+
        A:\s*(?P<strike_a>ATM|\d+(?:\.\d+)?[Δ∆D])
        \s*/\s*
        (?P<ko_a>H@\d+(?:\.\d+)?[Δ∆D]|\d+(?:\.\d+)?[x×X])
        \s+\[(?P<gate_a>[^\]]+)\]\s*
        [∧&]
        \s*B:\s*(?P<strike_b>ATM|\d+(?:\.\d+)?[Δ∆D])
        \s*/\s*
        (?P<ko_b>H@\d+(?:\.\d+)?[Δ∆D]|\d+(?:\.\d+)?[x×X])
        \s+\[(?P<gate_b>[^\]]+)\]\s*$
    """,
    re.VERBOSE,
)


def _gate_text_clean(g):
    """Normalize a parsed gate. 'no gate' (case-insensitive) → None,
    empty/None → None, otherwise return as-is."""
    if not isinstance(g, str):
        return None
    s = g.strip()
    if not s or s.lower() == "no gate":
        return None
    return s


# Canonical empty shape — every parse_strategy() return uses this baseline so
# the resulting columns are stable across rows of different strategy types.
_EMPTY_PARSED = {
    "Pair": None, "OptType": None, "Strike": None,
    "Tenor": None, "KO": None, "Gate": "None",
    "Direction": None,
    "StructureType": None,  # 'single' | 'worst_of' | 'eko_basket' | 'wo_basket'
    "Leg A Pair": None, "Leg A Strike": None, "Leg A KO": None,
    "Leg A Gate": None, "Leg A Direction": None,
    "Leg B Pair": None, "Leg B Strike": None, "Leg B KO": None,
    "Leg B Gate": None, "Leg B Direction": None,
}


def _direction_from_type(type_str) -> Optional[str]:
    """Map a strategy OR leg type code to 'Call' / 'Put'.
    Single-strategy OptType: 'CALL-upout', 'PUT-downin', etc.
    Worst-of leg code: 'Cuo', 'Cdo', 'Puo', 'Pdo'.
    Basket type code: 'Call-UO', 'Put-DO', etc.
    Returns None if the type doesn't look like either."""
    if not isinstance(type_str, str) or not type_str:
        return None
    head = type_str.strip().upper()
    if head.startswith("CALL") or head.startswith("C"):
        return "Call"
    if head.startswith("PUT") or head.startswith("P"):
        return "Put"
    return None


# Regex to extract currency tickers from a basket filename.
# Conventions:
#   EKO_BASKET_<CCY1>_<CCY2>_..._summary.csv
#   WO_BASKET_<CCY1>_<CCY2>_..._summary.csv
# Each <CCYn> is a 3-letter currency code (e.g. KRW). The implicit base
# currency is USD (i.e. KRW → USDKRW); this matches the user's data and is
# the dominant FX-options convention. If a future file uses a different base,
# this assumption can be revisited.
_BASKET_FILENAME_RE = re.compile(
    r"^(?:EKO|WO)_BASKET_(?P<ccys>[A-Z]{3}(?:_[A-Z]{3})+)_summary",
    re.IGNORECASE,
)


def basket_underlying_from_filename(filename: str) -> Optional[str]:
    """Extract the basket currency composition from a filename.

    >>> basket_underlying_from_filename("EKO_BASKET_KRW_JPY_THB_summary.csv")
    'USDKRW × USDJPY × USDTHB'
    >>> basket_underlying_from_filename("WO_BASKET_KRW_JPY_summary.csv")
    'USDKRW × USDJPY'
    >>> basket_underlying_from_filename("not_a_basket.csv") is None
    True
    """
    if not isinstance(filename, str):
        return None
    m = _BASKET_FILENAME_RE.search(filename)
    if not m:
        return None
    ccys = m.group("ccys").upper().split("_")
    if not ccys:
        return None
    return " × ".join(f"USD{c}" for c in ccys)


def parse_strategy(strategy: str) -> dict:
    """Parse a strategy string. Supports four strategy structures:
      1. single (`USDJPY CALL-upout  ATM  1M  H@5Δ  [Spot > 50DMA]`)
      2. worst-of, single-currency legs
         (`WO[AUDUSD Cuo 35Δ/H@10Δ ∧ NZDUSD Cuo 35Δ/H@10Δ] 1M`)
      3. EKO basket — basket of single underlyings as one EKO option
         (`BASKET  Call-UO  1M  ATM/H@20Δ`). The basket composition is
         identified separately via the filename.
      4. WO basket — worst-of structure where each leg is itself a basket
         (`WO-BASKET  Call-UO  1M  A:ATM/H@20Δ [no gate]  ∧  B:ATM/H@20Δ
          [no gate]`)
    """
    if not isinstance(strategy, str):
        return dict(_EMPTY_PARSED)
    s = strategy.strip()

    # ---- WO-BASKET branch (type 4) ----
    # Check this FIRST because "WO-BASKET" would otherwise be partial-matched
    # by checks for "WO[" via prefix logic if we ever loosened that.
    if s.startswith("WO-BASKET"):
        m = WO_BASKET_RE.match(s)
        if m:
            out = dict(_EMPTY_PARSED)
            type_str = m.group("type")
            out["OptType"] = "wo_basket"
            out["StructureType"] = "wo_basket"
            out["Tenor"] = m.group("tenor")
            out["Direction"] = _direction_from_type(type_str)
            # Leg A
            out["Leg A Pair"] = "BASKET"   # placeholder — filled from filename
            out["Leg A Strike"] = m.group("strike_a")
            out["Leg A KO"] = m.group("ko_a")
            out["Leg A Gate"] = _gate_text_clean(m.group("gate_a"))
            out["Leg A Direction"] = out["Direction"]
            # Leg B
            out["Leg B Pair"] = "BASKET"
            out["Leg B Strike"] = m.group("strike_b")
            out["Leg B KO"] = m.group("ko_b")
            out["Leg B Gate"] = _gate_text_clean(m.group("gate_b"))
            out["Leg B Direction"] = out["Direction"]
            # Pair = first leg's pair stand-in (back-compat for grouping)
            out["Pair"] = "BASKET"
            # Roll up gates the same way as single-currency worst-of
            ga, gb = out["Leg A Gate"], out["Leg B Gate"]
            if ga and gb:
                out["Gate"] = ga if ga == gb else f"{ga} ∧ {gb}"
            elif ga:
                out["Gate"] = f"A:{ga}"
            elif gb:
                out["Gate"] = f"B:{gb}"
            else:
                out["Gate"] = "None"
            return out

    # ---- EKO-BASKET branch (type 3) ----
    if s.startswith("BASKET"):
        m = BASKET_RE.match(s)
        if m:
            out = dict(_EMPTY_PARSED)
            type_str = m.group("type")
            out["OptType"] = "eko_basket"
            out["StructureType"] = "eko_basket"
            out["Pair"] = "BASKET"        # placeholder — filled from filename
            out["Tenor"] = m.group("tenor")
            out["Strike"] = m.group("strike")
            out["KO"] = m.group("ko")
            out["Gate"] = _gate_text_clean(m.group("gate")) or "None"
            out["Direction"] = _direction_from_type(type_str)
            return out

    # ---- Single-currency worst-of branch (type 2) ----
    if s.startswith("WO["):
        m = WORSTOF_RE.match(s)
        if m:
            legs_str = m.group("legs")
            # Split on the ∧ (logical-and) symbol with surrounding whitespace
            leg_parts = re.split(r"\s*[∧&]\s*", legs_str)
            parsed_legs = []
            for lp in leg_parts:
                lm = LEG_RE.match(lp.strip())
                if lm:
                    parsed_legs.append(lm.groupdict())

            out = dict(_EMPTY_PARSED)
            out["OptType"] = "worst_of"
            out["StructureType"] = "worst_of"
            out["Tenor"] = m.group("tenor")
            # Pair = first leg's pair (back-compat for any code grouping on Pair)
            if parsed_legs:
                out["Pair"] = parsed_legs[0]["pair"]
                out["Leg A Pair"] = parsed_legs[0]["pair"]
                out["Leg A Strike"] = parsed_legs[0]["strike"]
                out["Leg A KO"] = parsed_legs[0]["ko"]
                out["Leg A Gate"] = parsed_legs[0].get("gate") or None
                out["Leg A Direction"] = _direction_from_type(
                    parsed_legs[0].get("type"))
            if len(parsed_legs) >= 2:
                out["Leg B Pair"] = parsed_legs[1]["pair"]
                out["Leg B Strike"] = parsed_legs[1]["strike"]
                out["Leg B KO"] = parsed_legs[1]["ko"]
                out["Leg B Gate"] = parsed_legs[1].get("gate") or None
                out["Leg B Direction"] = _direction_from_type(
                    parsed_legs[1].get("type"))

            # Roll per-leg directions into the overall Direction field.
            # If both legs match → that direction. If different → 'Mixed'.
            # If only one leg parsed → use that leg's direction.
            da, db = out["Leg A Direction"], out["Leg B Direction"]
            if da and db:
                out["Direction"] = da if da == db else "Mixed"
            elif da or db:
                out["Direction"] = da or db

            # Summarise per-leg gates into the main `Gate` column for back-compat
            # (so the Gate filter and the gate-uplift tab still work).
            ga = out["Leg A Gate"]
            gb = out["Leg B Gate"]
            if ga and gb:
                out["Gate"] = ga if ga == gb else f"{ga} ∧ {gb}"
            elif ga:
                out["Gate"] = f"A:{ga}"
            elif gb:
                out["Gate"] = f"B:{gb}"
            else:
                out["Gate"] = "None"
            return out

    # ---- Single-leg branch (legacy + new format) ----
    m = STRATEGY_RE.match(s)
    if not m:
        # Best-effort fallback: split first 5 tokens on whitespace
        parts = s.split(None, 4)
        gate = "None"
        if len(parts) >= 5 and "[" in parts[4]:
            head, _, rest = parts[4].partition("[")
            parts[4] = head.strip()
            gate = rest.strip(" ]")
        out = dict(_EMPTY_PARSED)
        opt_type = parts[1] if len(parts) > 1 else None
        out.update({
            "Pair": parts[0] if len(parts) > 0 else None,
            "OptType": opt_type,
            "Strike": parts[2] if len(parts) > 2 else None,
            "Tenor": parts[3] if len(parts) > 3 else None,
            "KO": parts[4] if len(parts) > 4 else None,
            "Gate": gate,
            "Direction": _direction_from_type(opt_type),
            "StructureType": "single",
        })
        return out
    d = m.groupdict()
    out = dict(_EMPTY_PARSED)
    out.update({
        "Pair": d["pair"],
        "OptType": d["type"],
        "Strike": d["strike"],
        "Tenor": d["tenor"],
        "KO": d["ko"] if d["ko"] else None,
        "Gate": d["gate"] if d["gate"] else "None",
        "Direction": _direction_from_type(d["type"]),
        "StructureType": "single",
    })
    return out


# ---------------------------------------------------------------------------
# Loader / cleaner
# ---------------------------------------------------------------------------

# Map new-format (raw numeric) column names → existing internal schema.
# Anything not in this map is kept as-is.
NEW_TO_OLD = {
    "strategy_name":           "Strategy",
    "n_trades":                "n trades",
    "total_premium_paid_usd":  "Σ Premium",
    "total_tx_cost_usd":       "Σ TX Cost",
    "total_payout_usd":        "Σ Payout",
    "total_pnl_usd":           "Σ PnL",
    "max_drawdown_usd":        "Max DD",
    "win_rate_pct":            "Win%",
    "premium_recovery_pct":    "Recovery%",
    "sharpe_monthly":          "Sharpe (m)",
    "annual_sharpe_mean":      "Sharpe(y) μ",
    "annual_sharpe_min":       "Sharpe(y) min",
    "n_years":                 "Yrs",
    "pct_positive_years":      "%Pos Yrs",
    "min_annual_pnl_usd":      "Min Ann $",
    "calmar":                  "Calmar",
    "gain_to_pain":            "G2P",
    "ulcer_index":             "Ulcer",
    "feasibility_pct":         "Feas%",
    "ko_rate_pct":             "KO%",
    # Keep these as-is — they're genuinely new metrics
    # "annual_sharpe_std", "annual_sharpe_cv", "annual_sharpe_score",
    # "notional_usd", "strategy_type", "leg_a_ko_rate_pct",
    # "leg_b_ko_rate_pct", "both_survive_rate_pct",
    # "structure_vs_min_leg_pct",
}


def _is_new_format(df: pd.DataFrame) -> bool:
    """Detect whether this is the new-format summary CSV."""
    return "strategy_name" in df.columns and "Strategy" not in df.columns


@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes: bytes) -> pd.DataFrame:
    """Read the raw CSV bytes and return a cleaned, parsed DataFrame.
    Handles both the legacy *_All.csv format and the new *_summary.csv format.
    """
    df = pd.read_csv(BytesIO(file_bytes), encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    # Translate new-format columns to the internal schema
    if _is_new_format(df):
        df = df.rename(columns={k: v for k, v in NEW_TO_OLD.items()
                                if k in df.columns})

    # Parse strategy column into separate columns
    parsed = df["Strategy"].apply(parse_strategy).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)

    # Numeric conversions
    pct_cols = ["Feas%", "KO%", "Win%", "Recovery%", "%Pos Yrs",
                "leg_a_ko_rate_pct", "leg_b_ko_rate_pct",
                "both_survive_rate_pct", "structure_vs_min_leg_pct"]
    dollar_cols = ["Σ Premium", "Σ TX Cost", "Σ Payout", "Σ PnL",
                   "Max DD", "Min Ann $", "notional_usd"]
    float_cols = ["n", "Sharpe (m)", "n trades", "Yrs",
                  "Sharpe(y) μ", "Sharpe(y) min", "Calmar", "G2P", "Ulcer",
                  "annual_sharpe_std", "annual_sharpe_cv",
                  "annual_sharpe_score"]

    for c in pct_cols:
        if c in df.columns:
            df[c] = df[c].apply(parse_pct)
    for c in dollar_cols:
        if c in df.columns:
            df[c] = df[c].apply(parse_dollar)
    for c in float_cols:
        if c in df.columns:
            # to_numeric handles signed floats like "+0.84", "-3.60"
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derived numeric helpers
    df["Strike_num"] = df["Strike"].apply(strike_to_numeric)
    df["KO_num"] = df["KO"].apply(ko_to_numeric)
    df["Tenor_days"] = df["Tenor"].apply(tenor_to_days)

    # Useful derived ratios (guard against div-by-zero)
    df["PnL/Premium"] = np.where(df["Σ Premium"] > 0,
                                 df["Σ PnL"] / df["Σ Premium"], np.nan)
    df["PnL/|DD|"] = np.where(df["Max DD"] < 0,
                              df["Σ PnL"] / df["Max DD"].abs(), np.nan)
    df["TX/Premium %"] = np.where(df["Σ Premium"] > 0,
                                  100 * df["Σ TX Cost"] / df["Σ Premium"],
                                  np.nan)

    # ----- Underlying / pair-group derivation -----
    # For single-currency strategies: just the Pair (e.g. "USDJPY").
    # For worst-of strategies: a sorted "PAIR_A × PAIR_B" string. Sorting
    # keeps "USDJPY × USDKRW" and "USDKRW × USDJPY" collapsed into one group
    # regardless of leg order in the source string.
    def _derive_underlying(row):
        if row.get("OptType") == "worst_of":
            legs = [row.get("Leg A Pair"), row.get("Leg B Pair")]
            legs = [p for p in legs if isinstance(p, str) and p]
            if not legs:
                return None
            return " × ".join(sorted(legs))
        p = row.get("Pair")
        return p if isinstance(p, str) and p else None

    df["Underlying"] = df.apply(_derive_underlying, axis=1)
    return df


@st.cache_data(show_spinner=False)
def load_timeseries(file_bytes: bytes) -> pd.DataFrame:
    """Read a *_timeseries.csv file and return a tidy long DataFrame with
    columns: strategy_name, strategy_type, period_type, period_end (date),
    pnl_usd, equity_usd, drawdown_usd."""
    df = pd.read_csv(BytesIO(file_bytes), encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    for c in ("pnl_usd", "equity_usd", "drawdown_usd"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["strategy_name", "period_type", "period_end"])


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt_usd(x) -> str:
    if pd.isna(x):
        return "—"
    a = abs(x)
    sign = "-" if x < 0 else ""
    if a >= 1e9:
        return f"{sign}${a/1e9:.2f}B"
    if a >= 1e6:
        return f"{sign}${a/1e6:.2f}M"
    if a >= 1e3:
        return f"{sign}${a/1e3:.1f}K"
    return f"{sign}${a:,.0f}"


def fmt_pct(x) -> str:
    return "—" if pd.isna(x) else f"{x:.1f}%"


def fmt_float(x, n=2) -> str:
    return "—" if pd.isna(x) else f"{x:.{n}f}"


# ---------------------------------------------------------------------------
# Direction badge — colored pill for Call / Put / Mixed. Used across multiple
# tabs (summary cards, top-N headline, strategy detail) so the visual cue is
# consistent everywhere. Returns a raw HTML span; the calling site must use
# unsafe_allow_html=True when rendering.
# ---------------------------------------------------------------------------
# Colors are kept calm — Call is a muted green (upward bias), Put is muted
# red (downward bias). Mixed gets a neutral amber. Bg uses translucent
# overlays on whatever the host bg is, so they read well in dark mode.
_DIRECTION_STYLES = {
    "Call":  {"bg": "rgba(86, 197, 137, 0.18)", "fg": "#86d586",
              "arrow": "↗"},
    "Put":   {"bg": "rgba(255, 110, 110, 0.18)", "fg": "#ff9292",
              "arrow": "↘"},
    "Mixed": {"bg": "rgba(255, 196, 86, 0.18)", "fg": "#ffc456",
              "arrow": "↔"},
}


def direction_badge_html(direction, with_arrow: bool = True,
                         compact: bool = False) -> str:
    """Render a 'Call' / 'Put' / 'Mixed' colored badge as an inline HTML span.
    `with_arrow` adds a directional unicode arrow. `compact` shrinks font."""
    if not isinstance(direction, str) or direction not in _DIRECTION_STYLES:
        return ""
    style = _DIRECTION_STYLES[direction]
    arrow = f'<span style="opacity:0.85; margin-right:3px;">{style["arrow"]}</span>' \
        if with_arrow else ""
    fs = "11px" if compact else "12px"
    return (
        f'<span style="display:inline-flex; align-items:center; '
        f'padding:2px 9px; border-radius:12px; font-size:{fs}; '
        f'font-weight:600; background:{style["bg"]}; color:{style["fg"]}; '
        f'line-height:1.3;">{arrow}{direction}</span>'
    )


# ---------------------------------------------------------------------------
# Single source of truth for metric semantics. Every tab pulls from this so
# new columns get picked up consistently.
# Format kind: "usd", "pct", "float", "int"
# ---------------------------------------------------------------------------
METRIC_SPECS: dict[str, dict] = {
    # Return / quality (higher better)
    "Σ PnL":         {"label": "Σ PnL",                  "higher": True,  "fmt": "usd"},
    "Sharpe (m)":    {"label": "Sharpe (monthly)",       "higher": True,  "fmt": "float"},
    "Sharpe(y) μ":   {"label": "Sharpe (yearly, mean)",  "higher": True,  "fmt": "float"},
    "Sharpe(y) min": {"label": "Sharpe (yearly, worst)", "higher": True,  "fmt": "float"},
    "Calmar":        {"label": "Calmar",                 "higher": True,  "fmt": "float"},
    "G2P":           {"label": "Gain-to-Pain",           "higher": True,  "fmt": "float"},
    "Win%":          {"label": "Win %",                  "higher": True,  "fmt": "pct"},
    "%Pos Yrs":      {"label": "% Positive Years",       "higher": True,  "fmt": "pct"},
    "Recovery%":     {"label": "Recovery %",             "higher": True,  "fmt": "pct"},
    "PnL/Premium":   {"label": "PnL ÷ Premium",          "higher": True,  "fmt": "float"},
    "PnL/|DD|":      {"label": "PnL ÷ |MaxDD|",          "higher": True,  "fmt": "float"},
    # Loss-side (less negative is better, so higher=True)
    "Max DD":        {"label": "Max DD (less neg=better)",     "higher": True, "fmt": "usd"},
    "Min Ann $":     {"label": "Worst annual $ (higher=better)", "higher": True, "fmt": "usd"},
    # Risk / cost (lower better)
    "KO%":           {"label": "KO % (lower=better)",        "higher": False, "fmt": "pct"},
    "Ulcer":         {"label": "Ulcer Index (lower=better)", "higher": False, "fmt": "float"},
    "TX/Premium %":  {"label": "TX % of Premium (lower=better)", "higher": False, "fmt": "pct"},
    # New-format-only fields
    "annual_sharpe_std":       {"label": "Annual Sharpe σ (lower=better)",  "higher": False, "fmt": "float"},
    "annual_sharpe_cv":        {"label": "Annual Sharpe CV (lower=better)", "higher": False, "fmt": "float"},
    "annual_sharpe_score":     {"label": "Annual Sharpe score",             "higher": True,  "fmt": "float"},
    "leg_a_ko_rate_pct":       {"label": "Leg A KO % (lower=better)",       "higher": False, "fmt": "pct"},
    "leg_b_ko_rate_pct":       {"label": "Leg B KO % (lower=better)",       "higher": False, "fmt": "pct"},
    "both_survive_rate_pct":   {"label": "Both legs survive %",             "higher": True,  "fmt": "pct"},
    "structure_vs_min_leg_pct":{"label": "Structure vs min leg %",          "higher": True,  "fmt": "pct"},
    "notional_usd":            {"label": "Notional",                        "higher": True,  "fmt": "usd"},
}


def available_metrics(df: pd.DataFrame,
                      only: Optional[list[str]] = None) -> dict[str, dict]:
    """Return a dict of metric_name -> spec for metrics that exist in df
    and have at least one non-NaN value. Optionally restrict to a subset."""
    out = {}
    for k, spec in METRIC_SPECS.items():
        if only is not None and k not in only:
            continue
        if k in df.columns and df[k].notna().any():
            out[k] = spec
    return out


def default_metric_idx(keys, preferred: str = "Sharpe (m)") -> int:
    """Return the index of `preferred` in `keys`, or 0 if not present."""
    keys = list(keys)
    return keys.index(preferred) if preferred in keys else 0


# ---------------------------------------------------------------------------
# Column glossary shown in the help expander on the Optimal & Top-N tab.
# ---------------------------------------------------------------------------
COLUMN_GLOSSARY_MD = """
**Dimensions**
- **Strike** — strike expressed as option delta. `25Δ` = 25-delta strike (further OTM); `ATM` = at-the-money. Higher delta = closer to spot = more expensive.
- **Tenor** — option maturity. `1W`, `1M`, `2M`, `3M`, etc.
- **KO** — knockout barrier. Either as a delta (`H@5Δ` = barrier at the 5-delta level) or as a payout-cap multiplier (`8×` = capped at 8× premium).
- **Gate** — entry filter / regime condition (e.g. `Spot > 50DMA`). `None` = unconditional entry every day.

**Sample size**
- **n** — number of feasible trading days in the backtest sample.
- **n trades** — number of trades actually executed.
- **Yrs** — backtest length in calendar years.

**Frequency / outcome distribution**
- **Feas %** — share of days where the strategy could be entered (passed all gates and feasibility checks).
- **Win %** — share of trades with positive net P&L.
- **KO %** — share of trades that hit the knockout barrier (option became worthless).
- **% Pos Yrs** — share of calendar years with positive P&L.

**Cumulative P&L (USD)**
- **Σ Premium** — total premium paid across all trades.
- **Σ TX Cost** — total transaction costs.
- **Σ Payout** — total option payouts received.
- **Σ PnL** — net = payouts − premium − TX.
- **Min Ann $** — worst single calendar year P&L.
- **Notional** — notional size per trade.

**Risk-adjusted return**
- **Sharpe (m)** — monthly Sharpe ratio of the strategy P&L stream.
- **Sharpe(y) μ** — mean of per-year annualized Sharpe ratios.
- **Sharpe(y) min** — worst single year's annualized Sharpe.
- **Annual Sharpe σ / CV** — dispersion of per-year Sharpe; lower = more consistent.
- **Annual Sharpe Score** — composite stability score combining mean and dispersion of annual Sharpe. Higher = robust *and* high return. This is the recommended optimisation target.
- **Calmar** — annualized return ÷ |Max DD|.
- **G2P** (Gain-to-Pain) — sum of gains ÷ sum of losses. > 1 means more gains than losses.
- **Ulcer** — Ulcer index, measuring drawdown depth and duration; lower is better.

**Drawdown**
- **Max DD** — maximum drawdown in USD (most negative equity peak-to-trough).
- **Recovery %** — share of Max DD that has been recovered.

**Derived ratios**
- **PnL/Premium** — total P&L as a fraction of total premium paid.
- **PnL/|DD|** — total P&L ÷ absolute value of Max DD.
- **TX/Premium %** — transaction costs as % of premium.

**Worst-of structures only**
- **Leg A / B KO %** — knockout rate of each leg individually.
- **Both legs survive %** — share of trades where neither leg knocked out.
- **Structure vs min leg %** — structure-level P&L vs. the worse-performing leg.
"""


def style_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        spec = METRIC_SPECS.get(c)
        if spec is None:
            continue
        kind = spec["fmt"]
        if kind == "usd":
            out[c] = out[c].apply(fmt_usd)
        elif kind == "pct":
            out[c] = out[c].apply(fmt_pct)
        elif kind == "float":
            out[c] = out[c].apply(lambda v: fmt_float(v, 2))
    # Feas% isn't a "metric" per se but format it anyway
    if "Feas%" in out.columns:
        out["Feas%"] = out["Feas%"].apply(fmt_pct)
    # Dollar-valued component columns that aren't optimization metrics —
    # format consistently with Σ PnL / Max DD / etc.
    for c in ("Σ Premium", "Σ TX Cost", "Σ Payout"):
        if c in out.columns:
            out[c] = out[c].apply(fmt_usd)
    return out


# Canonical column order for tables. Dimensions first, then a curated metric
# order, then any remaining METRIC_SPECS metrics (so newly-added metrics still
# appear in the table without needing to edit this list).
TABLE_COLS_ORDER = [
    # Strategy name (auto-included only when worst-of is present, see table_cols)
    "Strategy",
    # Underlying / pair group (auto-included only when multiple groups present)
    "Underlying",
    # Direction (Call/Put) — auto-included when there's >1 direction in the data
    "Direction",
    # Structure type (single/worst_of/eko_basket/wo_basket) — auto-included
    # when the dataset has 2+ distinct structures (mixed file).
    "StructureType",
    # Single-leg dimensions (dropped from view when all-NaN, e.g. worst-of-only)
    "Strike", "KO",
    # Worst-of leg dimensions (auto-included only when worst-of is present)
    "Leg A Pair", "Leg A Direction", "Leg A Strike", "Leg A KO", "Leg A Gate",
    "Leg B Pair", "Leg B Direction", "Leg B Strike", "Leg B KO", "Leg B Gate",
    # Common dimensions
    "Tenor", "Gate",
    # Sample size + summary stats
    "n", "n trades", "Yrs", "Feas%",
    "KO%", "Win%", "%Pos Yrs",
    "Σ PnL", "Σ Premium", "Σ TX Cost", "Σ Payout", "notional_usd",
    "Sharpe (m)", "Sharpe(y) μ", "Sharpe(y) min",
    "annual_sharpe_std", "annual_sharpe_cv", "annual_sharpe_score",
    "Calmar", "G2P", "Max DD", "Min Ann $", "Ulcer", "Recovery%",
    "PnL/Premium", "PnL/|DD|", "TX/Premium %",
    "leg_a_ko_rate_pct", "leg_b_ko_rate_pct",
    "both_survive_rate_pct", "structure_vs_min_leg_pct",
]


def table_cols(df: pd.DataFrame) -> list[str]:
    """Return canonical column order, keeping only columns that exist.
    - 'Strategy' and the Leg A/B columns are included only when the dataset
      contains worst-of strategies (otherwise they're all-NaN clutter).
    - 'Underlying' is included only when there are 2+ underlyings in the
      filtered set (otherwise it's redundant — every row would show the same
      value).
    - 'Direction' is included only when there are 2+ directions in the data
      (e.g. mixed Call/Put file). For single-direction files (most common),
      direction is shown in the title/badges instead.
    - 'Strike' and 'KO' are dropped when entirely NaN (e.g. worst-of-only files).
    - Any METRIC_SPECS column present but not in TABLE_COLS_ORDER is appended,
      so newly-registered metrics show up automatically.
    """
    # 'Has legs' = any structure with leg A/B columns. Covers both single-
    # currency worst-of and WO basket.
    has_legs = (("OptType" in df.columns)
                and df["OptType"].isin(["worst_of", "wo_basket"]).any())
    multi_underlying = (("Underlying" in df.columns)
                        and df["Underlying"].dropna().nunique() >= 2)
    multi_direction = (("Direction" in df.columns)
                       and df["Direction"].dropna().nunique() >= 2)
    multi_structure = (("StructureType" in df.columns)
                       and df["StructureType"].dropna().nunique() >= 2)
    is_leg_col = lambda c: c.startswith("Leg A ") or c.startswith("Leg B ")

    ordered = []
    for c in TABLE_COLS_ORDER:
        if c not in df.columns:
            continue
        if c == "Strategy" and not has_legs:
            continue              # full Strategy name redundant for singles
        if c == "Underlying" and not multi_underlying:
            continue              # all rows same Underlying = no info value
        if c == "Direction" and not multi_direction:
            continue              # all rows same Direction = redundant column
        if c == "StructureType" and not multi_structure:
            continue              # all rows same StructureType = redundant
        if is_leg_col(c) and not has_legs:
            continue              # leg columns irrelevant without worst-of
        if c in ("Strike", "KO") and df[c].dropna().empty:
            continue              # all-NaN dimension column
        ordered.append(c)
    extras = [c for c in METRIC_SPECS
              if c in df.columns and c not in ordered]
    return ordered + extras


# ---------------------------------------------------------------------------
# Sidebar — data source: dropdown of preloaded files OR upload
# ---------------------------------------------------------------------------
st.sidebar.title("📁 Data")


def _friendly_label(stem: str) -> str:
    """Strip the '_summary' suffix so 'USDJPY_summary' renders as 'USDJPY'.
    Leaves '_All' alone — that way a legacy 'USDJPY_All.csv' and a new
    'USDJPY_summary.csv' coexist as separate dropdown entries."""
    if stem.endswith("_summary"):
        return stem[: -len("_summary")]
    return stem


preloaded = list_preloaded_summaries()
options: dict[str, Path | None] = {_friendly_label(p.stem): p
                                    for p in preloaded}
options["⬆️ Upload my own CSV"] = None  # always offer upload as fallback

choice = st.sidebar.selectbox(
    "Dataset",
    list(options.keys()),
    index=0 if preloaded else len(options) - 1,
    help="Pick a preloaded backtest, or upload a CSV with the same schema.",
)

file_bytes: bytes | None = None
file_label: str = ""
ts_bytes: bytes | None = None
ts_label: str = ""

if options[choice] is None:
    uploaded = st.sidebar.file_uploader(
        "Upload summary CSV", type=["csv"], key="summary_upload",
    )
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        file_label = uploaded.name
    uploaded_ts = st.sidebar.file_uploader(
        "Upload timeseries CSV (optional)", type=["csv"], key="ts_upload",
        help="Enables the Strategy detail tab with PnL, drawdown, "
             "monthly/annual bars, and per-year Sharpe.",
    )
    if uploaded_ts is not None:
        ts_bytes = uploaded_ts.getvalue()
        ts_label = uploaded_ts.name
else:
    path = options[choice]
    file_bytes = path.read_bytes()
    file_label = path.name
    # Auto-pair with a timeseries file in the same directory
    ts_path = find_paired_timeseries(path)
    if ts_path is not None:
        ts_bytes = ts_path.read_bytes()
        ts_label = ts_path.name

if file_bytes is None:
    st.title("Backtest Viewer")
    if preloaded:
        st.markdown(
            f"Select a preloaded dataset in the sidebar "
            f"(**{len(preloaded)} available**), or upload your own CSV."
        )
    else:
        st.markdown(
            """
            Upload an FX option backtest CSV in the sidebar to begin.

            Expected schema (e.g. **USDJPY_All.csv**):

            ```
            Strategy, n, Feas%, KO%, Win%, Σ Premium, Σ TX Cost,
            Σ Payout, Σ PnL, Sharpe (m), Max DD, Recovery%, n trades
            ```

            The **Strategy** column is parsed into its components:
            *Pair · OptionType · Strike · Tenor · KO · Gate*.
            """
        )
    st.stop()

data = load_and_clean(file_bytes)

# Defensive: ensure derived columns exist even if a cached DataFrame from a
# previous app version (without these columns) is served by @st.cache_data.
# Without this, accessing data["Direction"] further down would KeyError when
# the cache hit predates the column being added to load_and_clean.
data = data.copy()
for _col in ("Direction", "Leg A Direction", "Leg B Direction",
             "Underlying", "Leg A Gate", "Leg B Gate",
             "StructureType"):
    if _col not in data.columns:
        data[_col] = None

# Basket-underlying fixup. For EKO_BASKET and WO_BASKET strategies the
# strategy_name doesn't carry the basket's currency composition — it has to
# come from the filename (e.g. "EKO_BASKET_KRW_JPY_THB_summary.csv"). The
# parser sets Underlying='BASKET' (or 'BASKET × BASKET' for WO basket) as a
# placeholder; here we replace those rows' Underlying with the real list.
_basket_underlying = basket_underlying_from_filename(file_label)
if _basket_underlying:
    _is_basket = data["StructureType"].isin(["eko_basket", "wo_basket"])
    if _is_basket.any():
        data.loc[_is_basket, "Underlying"] = _basket_underlying
        # Also set Leg A/B Pair on WO basket rows so the leg-display logic
        # later shows the basket composition rather than the BASKET stub.
        _is_wo_basket = data["StructureType"] == "wo_basket"
        if _is_wo_basket.any():
            data.loc[_is_wo_basket, "Leg A Pair"] = _basket_underlying
            data.loc[_is_wo_basket, "Leg B Pair"] = _basket_underlying

ts_data: Optional[pd.DataFrame] = None
if ts_bytes is not None:
    try:
        ts_data = load_timeseries(ts_bytes)
    except Exception as e:
        st.sidebar.warning(f"Couldn't read time-series file: {e}")
        ts_data = None

def _pair_display(df: pd.DataFrame) -> str:
    """Headline string showing underlying currency pairs / worst-of groups.
    Uses the derived `Underlying` column so singles and worst-of legs aren't
    mashed together. Caps the displayed list at 4 items to keep the title
    readable when many groups are present."""
    if "Underlying" not in df.columns:
        return "—"
    groups = sorted(df["Underlying"].dropna().unique().tolist())
    if not groups:
        return "—"
    if len(groups) == 1:
        return groups[0]
    if len(groups) <= 4:
        return " · ".join(groups)
    return f"{len(groups)} underlyings ({', '.join(groups[:3])}, …)"


pair = _pair_display(data)

# Whether any rows in this dataset have leg A/B structure (single-currency
# worst-of OR worst-of basket — both expose leg dimensions).
has_worstof = (("OptType" in data.columns)
               and data["OptType"].isin(["worst_of", "wo_basket"]).any())

st.title(f"Backtest Viewer — {pair}")

# Direction summary line — small colored badges showing the direction(s) of
# strategies in this dataset. Renders directly under the title for visibility.
_dir_counts = data["Direction"].value_counts() if "Direction" in data.columns \
    else pd.Series(dtype=int)
if len(_dir_counts) > 0:
    badge_parts = []
    for dirn in ("Call", "Put", "Mixed"):
        if dirn in _dir_counts.index:
            badge_parts.append(
                direction_badge_html(dirn, with_arrow=True)
                + f' <span style="opacity:0.62; font-size:12px;">'
                f'{_dir_counts[dirn]:,}</span>'
            )
    if badge_parts:
        st.markdown(
            "&nbsp;&nbsp;".join(badge_parts),
            unsafe_allow_html=True,
        )

caption_bits = [
    f"📄 `{file_label}`",
    f"**{len(data):,}** strategies",
]
n_underlyings = data["Underlying"].dropna().nunique() \
    if "Underlying" in data.columns else 0
if n_underlyings >= 2:
    caption_bits.append(f"**{n_underlyings}** underlyings")
if data["Strike"].notna().any():
    caption_bits.append(f"{data['Strike'].nunique()} strikes")
if data["Tenor"].notna().any():
    caption_bits.append(f"{data['Tenor'].nunique()} tenors")
if data["KO"].notna().any():
    caption_bits.append(f"{data['KO'].nunique()} KO levels")
if data["Gate"].nunique() > 1:
    caption_bits.append(f"{data['Gate'].nunique()} gates")
if ts_data is not None:
    caption_bits.append(f"📈 time-series: `{ts_label}`")
st.caption(" · ".join(caption_bits))


# Sidebar filters
st.sidebar.title("🔎 Filters")

with st.sidebar.expander("Dimension filters", expanded=True):
    underlyings = sorted(data["Underlying"].dropna().unique().tolist())
    directions = sorted(
        [d for d in data["Direction"].dropna().unique().tolist()
         if d in ("Call", "Put", "Mixed")],
        key=lambda x: {"Call": 0, "Put": 1, "Mixed": 2}.get(x, 3),
    )
    # Available structures in this file, sorted in a sensible reading order
    # (singles → worst-of → eko_basket → wo_basket).
    _structure_order = {"single": 0, "worst_of": 1,
                        "eko_basket": 2, "wo_basket": 3}
    structures = sorted(
        [s for s in data["StructureType"].dropna().unique().tolist()
         if s in _structure_order],
        key=lambda x: _structure_order.get(x, 99),
    )
    strikes = sorted(data["Strike"].dropna().unique().tolist(),
                     key=strike_to_numeric)
    tenors = sorted(data["Tenor"].dropna().unique().tolist(),
                    key=lambda t: TENOR_TO_DAYS.get(t.upper(), 9999))
    kos = sorted(data["KO"].dropna().unique().tolist(), key=ko_to_numeric)
    gates = sorted(data["Gate"].dropna().unique().tolist())

    # Underlying filter — only shown when the dataset has >1 underlying,
    # since a single-underlying dataset doesn't benefit from the filter
    # and the widget would just take up space.
    if len(underlyings) >= 2:
        sel_underlyings = st.multiselect(
            "Underlying / pair group",
            underlyings, default=underlyings,
            help="Filter by single-currency pair (e.g. USDJPY) or "
                 "worst-of / basket combination (e.g. USDJPY × USDKRW).",
        )
    else:
        sel_underlyings = underlyings

    # Direction filter — only shown when the dataset has >1 direction.
    # For single-direction files (most common — e.g. all CALL-upout), the
    # filter is hidden and direction is communicated via the title badge.
    if len(directions) >= 2:
        sel_directions = st.multiselect(
            "Direction (Call / Put)",
            directions, default=directions,
            help="Filter by option direction. Call = profits on the upside, "
                 "Put = profits on the downside. Mixed = worst-of with one "
                 "Call leg and one Put leg.",
        )
    else:
        sel_directions = directions

    # Structure filter — only shown when the dataset has >1 structure type.
    # Useful when combining files (singles + worst-of + baskets) or when
    # comparing across structures within a single file.
    _structure_labels = {
        "single": "Single",
        "worst_of": "Worst-of (single legs)",
        "eko_basket": "EKO basket",
        "wo_basket": "WO basket",
    }
    if len(structures) >= 2:
        sel_structures = st.multiselect(
            "Structure type",
            structures,
            default=structures,
            format_func=lambda s: _structure_labels.get(s, s),
            help="single = one underlying, one option · "
                 "worst-of = two single-currency legs · "
                 "EKO basket = basket of underlyings as one option · "
                 "WO basket = worst-of structure with basket legs.",
        )
    else:
        sel_structures = structures

    sel_strikes = st.multiselect("Strike", strikes, default=strikes)
    sel_tenors = st.multiselect("Tenor", tenors, default=tenors)
    sel_kos = st.multiselect("KO", kos, default=kos)

    # Gate default: prefer 'None' (ungated universe) if present so the user
    # starts looking at unconditional strategies and adds gates intentionally.
    # If 'None' isn't a value in this dataset, select all gates.
    _gate_none_present = "None" in gates
    _gate_default = ["None"] if _gate_none_present else gates
    sel_gates = st.multiselect(
        "Gate", gates, default=_gate_default,
        help=("Defaults to 'None' (ungated strategies) when available, so "
              "you start with the unconditional universe. Add gates "
              "explicitly to compare regime-filtered variants."),
    )

with st.sidebar.expander("Performance constraints", expanded=False):
    # Use whichever sample-size column is present. Old format has 'n'
    # (feasible days); new format only has 'n trades' (executed trades).
    n_col = "n" if "n" in data.columns and data["n"].notna().any() \
        else ("n trades" if "n trades" in data.columns
              and data["n trades"].notna().any() else None)

    if n_col is not None:
        min_n = int(np.nanmin(data[n_col]))
        max_n = int(np.nanmax(data[n_col]))
        n_min = st.number_input(
            f"Min sample size ({n_col})", min_value=0,
            max_value=max(max_n, 1), value=min_n, step=10,
        )
    else:
        n_min = 0

    has_ko_pct = "KO%" in data.columns and data["KO%"].notna().any()
    max_ko = st.slider("Max KO% (knockout rate)", 0, 100, 100,
                       disabled=not has_ko_pct,
                       help=None if has_ko_pct else "KO% not in this dataset")
    min_sharpe = st.number_input("Min Sharpe", value=-5.0, step=0.1)

# Build the filter mask piece-by-piece so missing dimensions don't crash
mask = pd.Series(True, index=data.index)
if "Underlying" in data.columns and len(underlyings) >= 2:
    mask &= data["Underlying"].isin(sel_underlyings) | data["Underlying"].isna()
if "Direction" in data.columns and len(directions) >= 2:
    mask &= data["Direction"].isin(sel_directions) | data["Direction"].isna()
if "StructureType" in data.columns and len(structures) >= 2:
    mask &= data["StructureType"].isin(sel_structures) | data["StructureType"].isna()
if data["Strike"].notna().any():
    mask &= data["Strike"].isin(sel_strikes) | data["Strike"].isna()
if data["Tenor"].notna().any():
    mask &= data["Tenor"].isin(sel_tenors) | data["Tenor"].isna()
if data["KO"].notna().any():
    mask &= data["KO"].isin(sel_kos) | data["KO"].isna()
if data["Gate"].nunique() > 1:
    mask &= data["Gate"].isin(sel_gates)
if n_col is not None:
    mask &= data[n_col].fillna(0) >= n_min
if has_ko_pct:
    mask &= data["KO%"].fillna(0) <= max_ko
if "Sharpe (m)" in data.columns:
    mask &= data["Sharpe (m)"].fillna(-99) >= min_sharpe
f = data[mask].copy()

if f.empty:
    st.warning("No strategies match the current filters. Loosen them in the sidebar.")
    st.stop()

st.sidebar.markdown(f"**{len(f):,} / {len(data):,}** strategies after filters")

# Banner: surface that the Gate filter is restricted to ungated strategies.
# This helps users notice they're looking at a subset rather than the full
# universe. Only show when Gate filter actually narrowed things — i.e. there
# are 2+ gate values in the data but only 'None' is selected.
_all_gates = data["Gate"].dropna().unique().tolist()
if (len(_all_gates) >= 2 and sel_gates == ["None"]):
    st.info(
        "🚪 **Showing ungated strategies only** — `Gate = None` is the "
        "active filter. Add other gate values in the sidebar to compare "
        "regime-filtered variants.",
        icon="ℹ️",
    )


# ---------------------------------------------------------------------------
# Print-to-PDF toolbar — leverages the browser's native print dialog (which
# offers "Save as PDF" on every modern browser). Renders four buttons:
#   1. Print Summary           → just the Summary tab
#   2. Print Optimal & Top-N   → just that tab
#   3. Print Strategy          → just the Strategy detail tab
#   4. Print All               → all tabs, one per page
#
# Implementation notes:
# - components.html() runs in an iframe; from inside, we access window.parent
#   to manipulate the main Streamlit DOM. This is the reliable way to inject
#   working JS into Streamlit (st.markdown(unsafe_allow_html=True) does not
#   reliably execute <script> tags).
# - The JS finds all tab panels by [data-baseweb="tab-panel"], force-shows
#   the target ones, calls window.parent.print(), then restores visibility.
# - The @media print stylesheet hides Streamlit's chrome (sidebar, header,
#   tab navigation, iframes) and forces a white background + black text so
#   the PDF is readable on paper regardless of in-app theme.
# ---------------------------------------------------------------------------
components.html(
    """
    <style>
      body { margin: 0; padding: 4px 0; background: transparent;
             font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                          sans-serif; }
      .print-toolbar { display: flex; gap: 6px; align-items: center;
                       flex-wrap: wrap; }
      .print-label { font-size: 12px; color: rgba(255,255,255,0.55);
                     margin-right: 4px; }
      .print-btn {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.20);
        color: rgba(255,255,255,0.90);
        padding: 4px 11px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 500;
        cursor: pointer;
        font-family: inherit;
        transition: background 0.1s, border-color 0.1s;
      }
      .print-btn:hover {
        background: rgba(255,255,255,0.10);
        border-color: rgba(255,255,255,0.40);
      }
      .print-btn:active { transform: scale(0.98); }
      .print-btn.all {
        background: rgba(255, 75, 75, 0.20);
        border-color: rgba(255, 75, 75, 0.40);
        color: #ff9292;
      }
      .print-btn.all:hover {
        background: rgba(255, 75, 75, 0.30);
        border-color: rgba(255, 75, 75, 0.60);
      }
    </style>

    <div class="print-toolbar">
      <span class="print-label">🖨️ Print to PDF:</span>
      <button class="print-btn" onclick="printTab(0, 'Summary')">Summary</button>
      <button class="print-btn" onclick="printTab(1, 'Optimal & Top-N')">Optimal &amp; Top-N</button>
      <button class="print-btn" onclick="printTab(2, 'Strategy detail')">Strategy</button>
      <button class="print-btn all" onclick="printTab(-1, 'All tabs')">All</button>
    </div>

    <script>
      // Run in an iframe; manipulate the parent (main Streamlit) document.
      const PDOC = window.parent.document;

      // Inject the print stylesheet into the parent document exactly once.
      // Subsequent reruns of this component skip the injection (idempotent).
      function ensurePrintCSS() {
        if (PDOC.getElementById('fx-print-css')) return;
        const style = PDOC.createElement('style');
        style.id = 'fx-print-css';
        style.textContent = `
          @media print {
            @page { size: A4 landscape; margin: 10mm; }
            /* Force a printable color palette regardless of app theme */
            html, body, .stApp, [data-testid="stAppViewContainer"] {
              background: white !important;
              color: black !important;
            }
            body, p, span, div, label, h1, h2, h3, h4, h5, h6, li, td, th,
            .stMarkdown, .stMarkdown * {
              color: black !important;
            }
            /* Hide all the Streamlit chrome we don't want in the PDF */
            section[data-testid="stSidebar"] { display: none !important; }
            header[data-testid="stHeader"] { display: none !important; }
            [data-testid="stToolbar"] { display: none !important; }
            [data-testid="stDecoration"] { display: none !important; }
            .stTabs [data-baseweb="tab-list"] { display: none !important; }
            iframe { display: none !important; }
            footer { display: none !important; }
            /* Expand main content to full width */
            .block-container, [data-testid="stMain"] .block-container {
              padding: 0.5rem !important;
              max-width: 100% !important;
            }
            /* Each visible tab panel starts a new page (matters for Print All) */
            [data-baseweb="tab-panel"] {
              page-break-after: always !important;
              break-after: page !important;
            }
            [data-baseweb="tab-panel"]:last-of-type {
              page-break-after: avoid !important;
              break-after: auto !important;
            }
            /* Preserve our custom card colors with reasonable contrast */
            .sum-card {
              background: white !important;
              border: 1px solid #ccc !important;
              break-inside: avoid;
              page-break-inside: avoid;
            }
            .sum-best { background: #f6f6f6 !important; }
            .sum-rec-row { background: #f6f6f6 !important; }
            .sum-ul-name { color: black !important; }
            .sum-ul-meta { color: #555 !important; }
            .sum-section-label { color: #777 !important; }
            .sum-best-headlabel { color: #555 !important; }
            .sum-best-headval { color: black !important; }
            .sum-best-dims { color: #555 !important; }
            .sum-best-dims b { color: black !important; }
            .sum-stat-label { color: #555 !important; }
            .sum-stat-val { color: black !important; }
            .sum-rec-label { color: #555 !important; }
            /* Pills: muted printable variants */
            .sum-pill-1 {
              background: #ffe0e0 !important;
              color: #a00 !important;
              border-color: transparent !important;
            }
            .sum-pill-2 {
              background: #eee !important;
              color: #333 !important;
              border-color: #ccc !important;
            }
            /* Streamlit metric cards */
            [data-testid="stMetricValue"],
            [data-testid="stMetricValue"] * {
              color: black !important;
            }
            [data-testid="stMetricLabel"],
            [data-testid="stMetricLabel"] * {
              color: #555 !important;
            }
            /* Tables / dataframes - ensure rows print */
            .stDataFrame { color: black !important; }
            .stDataFrame * { color: black !important; }
            /* Plotly charts - keep their original colors but force white bg */
            .js-plotly-plot .bg, .js-plotly-plot .main-svg {
              background: white !important;
            }
            /* Hide ALL interactive controls — they don't belong in a PDF.
               This covers the Load Drilldown button on summary cards and the
               Optimise on / Aggregate controls at the top of tabs. */
            .stButton, .stDownloadButton, .stFormSubmitButton,
            .stSelectbox, .stRadio, .stMultiSelect, .stNumberInput,
            .stSlider, .stCheckbox, .stTextInput, .stFileUploader {
              display: none !important;
            }
            /* In case any inline button styling slips through, light-ify it */
            button, input[type="button"], input[type="submit"] {
              background: white !important;
              color: black !important;
              border: 1px solid #ccc !important;
            }
            /* st.info / st.warning / st.success banners — make printable. */
            [data-testid="stAlert"], [data-testid="stNotification"] {
              background: #f4f8fc !important;
              color: black !important;
              border: 1px solid #c0d4e8 !important;
            }
            [data-testid="stAlert"] *, [data-testid="stNotification"] * {
              color: black !important;
            }
            /* Expanders — show their header text; hide the expand/collapse
               arrow since it's not interactive on paper. */
            [data-testid="stExpander"] summary {
              background: #f0f0f0 !important;
              color: black !important;
            }
            /* Captions */
            [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {
              color: #555 !important;
            }
            /* Code / inline code in captions */
            code { background: #f0f0f0 !important; color: black !important; }
          }
        `;
        PDOC.head.appendChild(style);
      }

      // Show only the requested tab panel (or all if idx === -1), then print.
      function printTab(idx, label) {
        ensurePrintCSS();
        const panels = PDOC.querySelectorAll('[data-baseweb="tab-panel"]');
        if (!panels.length) {
          alert("No tabs found. Refresh the page and try again.");
          return;
        }
        // Snapshot current visibility state so we can restore after print.
        const orig = Array.from(panels).map(p => ({
          display: p.style.display,
          visibility: p.style.visibility,
          hidden: p.hidden,
        }));
        // Force the requested panel(s) visible; hide the others.
        panels.forEach((p, i) => {
          if (idx === -1 || i === idx) {
            p.style.display = 'block';
            p.style.visibility = 'visible';
            p.hidden = false;
            p.removeAttribute('hidden');
          } else {
            p.style.display = 'none';
          }
        });
        // Give the DOM a tick to apply, then call print on the parent window.
        setTimeout(() => {
          try {
            window.parent.print();
          } catch (e) {
            console.error("Print failed:", e);
            alert("Browser blocked the print dialog. Try Cmd/Ctrl+P instead.");
          }
          // Restore originals after the print dialog closes.
          setTimeout(() => {
            panels.forEach((p, i) => {
              p.style.display = orig[i].display;
              p.style.visibility = orig[i].visibility;
              p.hidden = orig[i].hidden;
            });
          }, 500);
        }, 150);
      }

      // Inject CSS immediately on first load (idempotent).
      ensurePrintCSS();
    </script>
    """,
    height=48,
)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_labels = [
    "🏁 Summary",            # per-underlying recommendations + best strategy
    "🏆 Optimal & Top-N",    # top-N over the full filtered set
    "📈 Strategy detail",    # drill-down on a single strategy (uses timeseries)
    "📊 Per-dimension impact",
    "🔥 Heatmaps",
    "🚪 Gate uplift",
    "⚖️ Pareto frontier",
    "🆚 Side-by-side compare",
    "📄 Data",
]
(tab_summary, tab_top, tab_detail, tab_dim, tab_heat, tab_gate, tab_pareto,
 tab_compare, tab_data) = st.tabs(tab_labels)


# ===========================================================================
# Tab: Summary — per-underlying top-2 recommendations
# ===========================================================================
with tab_summary:
    st.subheader("Per-underlying recommendations")
    st.caption(
        "For each underlying, the best strategy is shown alongside the top 2 "
        "dimension values ranked by the chosen aggregator of the optimisation "
        "metric. Hover any pill to see its score and sample size."
    )

    sum_cA, sum_cB = st.columns([2, 1])
    summary_metric_options = available_metrics(f)
    summary_metric = sum_cA.selectbox(
        "Optimise on",
        list(summary_metric_options.keys()),
        format_func=lambda k: summary_metric_options[k]["label"],
        index=default_metric_idx(summary_metric_options.keys()),
        key="summary_metric",
    )
    summary_agg = sum_cB.radio(
        "Aggregate", ["median", "mean", "max"],
        horizontal=True, key="summary_agg",
        help="median = robust to outliers (recommended). "
             "mean = sensitive to single big wins. "
             "max = best single strategy with that value (upper-bound view).",
    )

    _spec = METRIC_SPECS.get(summary_metric,
                             {"fmt": "float", "higher": True})
    _higher_better = _spec["higher"]
    _metric_label = summary_metric_options[summary_metric]["label"]

    def _fmt_score(x):
        if pd.isna(x):
            return "—"
        if _spec["fmt"] == "usd":
            return fmt_usd(x)
        if _spec["fmt"] == "pct":
            return fmt_pct(x)
        return fmt_float(x, 2)

    def _top2_for_dim(sub: pd.DataFrame, dim: str) -> list[tuple]:
        """Return [(value, score, n), ...] for the top 2 values of `dim` in
        `sub` ranked by `summary_agg` of `summary_metric`. Returns [] if the
        dimension has no usable values."""
        if dim not in sub.columns:
            return []
        if sub[dim].dropna().nunique() < 1:
            return []
        agg = sub.groupby(dim)[summary_metric].agg(summary_agg).dropna()
        sizes = sub.groupby(dim).size()
        ranked = pd.DataFrame({"score": agg, "n": sizes}).dropna(
            subset=["score"])
        if ranked.empty:
            return []
        ranked = ranked.sort_values(
            ["score", "n"], ascending=[not _higher_better, False])
        return [(idx, row["score"], int(row["n"]))
                for idx, row in ranked.head(2).iterrows()]

    # -----------------------------------------------------------------------
    # Single CSS block, injected once at the top of the summary tab. All
    # cards on this tab reference these classes.
    #
    # Theme strategy: use NEUTRAL GRAY with alpha for backgrounds/borders, and
    # `color: inherit` + `opacity` for text. This means:
    #   - On a dark Streamlit theme: gray@8% reads as slightly lighter dark;
    #     opacity dims white body text to muted grays.
    #   - On a light Streamlit theme: gray@8% reads as slightly darker white;
    #     opacity dims black body text to muted grays.
    # The same CSS works correctly in either mode without media queries.
    # Pill colors use a red ramp (rank ①) and neutral gray (rank ②) chosen so
    # both modes have adequate contrast.
    # -----------------------------------------------------------------------
    st.markdown(
        """
        <style>
          .sum-card { background: rgba(128,128,128,0.06); border: 1px solid rgba(128,128,128,0.22); border-radius: 12px; padding: 1.1rem 1.25rem; margin-bottom: 1rem; color: inherit; }
          .sum-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 14px; }
          .sum-ul-name { font-size: 18px; font-weight: 600; margin: 0; color: inherit; }
          .sum-ul-meta { font-size: 13px; opacity: 0.62; margin: 0; }
          .sum-grid-2col { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
          .sum-section-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.55; font-weight: 600; margin: 0 0 8px; }
          .sum-best { background: rgba(128,128,128,0.10); border-radius: 8px; padding: 12px; }
          .sum-best-headrow { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 10px; }
          .sum-best-headlabel { font-size: 11px; opacity: 0.62; }
          .sum-best-headval { font-size: 22px; font-weight: 600; color: inherit; }
          .sum-best-dims { font-size: 12px; line-height: 1.8; opacity: 0.65; }
          .sum-best-dims b { color: inherit; opacity: 1; font-weight: 600; }
          .sum-stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px 8px; margin-top: 10px; }
          .sum-stat-label { font-size: 11px; opacity: 0.62; margin: 0; }
          .sum-stat-val { font-size: 13px; font-weight: 600; margin: 1px 0 0; color: inherit; }
          .sum-rec-list { display: flex; flex-direction: column; gap: 6px; }
          .sum-rec-row { display: flex; justify-content: space-between; align-items: center; padding: 6px 10px; background: rgba(128,128,128,0.10); border-radius: 8px; }
          .sum-rec-label { font-size: 12px; opacity: 0.68; }
          .sum-pill { display: inline-flex; align-items: center; gap: 4px; padding: 3px 9px; border-radius: 14px; font-size: 12px; font-weight: 600; cursor: help; border: 1px solid transparent; margin-left: 4px; }
          /* Rank ① pill — red semantic, contrasts in both light & dark mode. */
          .sum-pill-1 { background: rgba(229, 51, 51, 0.20); color: #c92c2c; }
          /* Rank ② pill — neutral gray, inherits text color from body. */
          .sum-pill-2 { background: rgba(128,128,128,0.18); color: inherit; border-color: rgba(128,128,128,0.28); }
          .sum-rank { font-size: 10px; opacity: 0.70; margin-right: 2px; }
          /* In dark mode, give rank ① a lighter, brighter red so it pops on
             the dark background. prefers-color-scheme aligns with the user's
             OS in most setups; Streamlit's auto theme follows the same. */
          @media (prefers-color-scheme: dark) {
            .sum-pill-1 { color: #ff8888; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    def _pills_html(results: list[tuple]) -> str:
        """Render top-2 results as two pills with tooltips. Tooltips show
        the aggregator's score and the sample size."""
        if not results:
            return '<span style="color: rgba(255,255,255,0.40); font-size: 12px;">—</span>'
        pills = []
        for i, (val, score, n) in enumerate(results):
            cls = "sum-pill-1" if i == 0 else "sum-pill-2"
            rank = "①" if i == 0 else "②"
            tip = f"{summary_agg} {_metric_label}: {_fmt_score(score)} · n={n}"
            # Escape HTML in val for display; html-escape the title too
            import html as _html
            tip_esc = _html.escape(tip)
            val_esc = _html.escape(str(val))
            pills.append(
                f'<span class="sum-pill {cls}" title="{tip_esc}">'
                f'<span class="sum-rank">{rank}</span>{val_esc}</span>'
            )
        return "".join(pills)

    def _rec_row_html(label: str, results: list[tuple]) -> str:
        if not results:
            return ""
        return (
            f'<div class="sum-rec-row">'
            f'<span class="sum-rec-label">{label}</span>'
            f'<span>{_pills_html(results)}</span>'
            f'</div>'
        )

    def _render_card(ul: str, sub: pd.DataFrame, is_worstof: bool):
        """Render one Option-C card for a single underlying group."""
        best_row_df = sub.dropna(subset=[summary_metric]).sort_values(
            summary_metric, ascending=not _higher_better).head(1)
        if best_row_df.empty:
            return
        best = best_row_df.iloc[0]
        best_score_str = _fmt_score(best[summary_metric])

        # Build best-strategy dimension lines
        if is_worstof:
            # For wo_basket, both legs share the same (basket) Pair so don't
            # repeat the long basket name; just say "Basket Leg A/B".
            is_wo_basket_row = best.get("StructureType") == "wo_basket"
            dim_lines = []
            if pd.notna(best.get("Leg A Pair")):
                leg_a_badge = direction_badge_html(
                    best.get("Leg A Direction"), with_arrow=True, compact=True)
                badge_suffix = f"&nbsp;&nbsp;{leg_a_badge}" if leg_a_badge else ""
                if is_wo_basket_row:
                    dim_lines.append(
                        f"Basket Leg A <b>{best.get('Leg A Strike') or '?'} "
                        f"→ {best.get('Leg A KO') or '?'}</b>{badge_suffix}"
                    )
                else:
                    dim_lines.append(
                        f"Leg A <b>{best['Leg A Pair']} {best.get('Leg A Strike') or '?'} "
                        f"→ {best.get('Leg A KO') or '?'}</b>{badge_suffix}"
                    )
            if pd.notna(best.get("Leg B Pair")):
                leg_b_badge = direction_badge_html(
                    best.get("Leg B Direction"), with_arrow=True, compact=True)
                badge_suffix = f"&nbsp;&nbsp;{leg_b_badge}" if leg_b_badge else ""
                if is_wo_basket_row:
                    dim_lines.append(
                        f"Basket Leg B <b>{best.get('Leg B Strike') or '?'} "
                        f"→ {best.get('Leg B KO') or '?'}</b>{badge_suffix}"
                    )
                else:
                    dim_lines.append(
                        f"Leg B <b>{best['Leg B Pair']} {best.get('Leg B Strike') or '?'} "
                        f"→ {best.get('Leg B KO') or '?'}</b>{badge_suffix}"
                    )
            tenor_gate_line = (
                f"Tenor <b>{best.get('Tenor') or '?'}</b> &nbsp;·&nbsp; "
                f"Gate <b>{best['Gate'] if best['Gate'] != 'None' else '— none —'}</b>"
            )
            dim_lines.append(tenor_gate_line)
        else:
            best_dir_badge = direction_badge_html(
                best.get("Direction"), with_arrow=True, compact=True)
            badge_suffix = f"&nbsp;&nbsp;{best_dir_badge}" if best_dir_badge else ""
            dim_lines = [
                f"Strike <b>{best.get('Strike') or '?'}</b> &nbsp;·&nbsp; "
                f"Tenor <b>{best.get('Tenor') or '?'}</b>{badge_suffix}",
                f"KO <b>{best.get('KO') or '?'}</b>",
                f"Gate <b>{best['Gate'] if best['Gate'] != 'None' else '— none —'}</b>",
            ]
        dim_lines_html = "".join(f"<div>{ln}</div>" for ln in dim_lines)

        # Stat grid — 9 small stats below the dims block
        def _stat(label, value):
            return (
                f'<div><p class="sum-stat-label">{label}</p>'
                f'<p class="sum-stat-val">{value}</p></div>'
            )
        stats_html = "".join([
            _stat("Σ PnL", fmt_usd(best.get("Σ PnL", np.nan))),
            _stat("Max DD", fmt_usd(best.get("Max DD", np.nan))),
            _stat("Recovery", fmt_pct(best.get("Recovery%", np.nan))),
            _stat("Win", fmt_pct(best.get("Win%", np.nan))),
            _stat("Calmar", fmt_float(best.get("Calmar", np.nan))),
            _stat("n trades",
                  f"{int(best['n trades']):,}"
                  if pd.notna(best.get("n trades", np.nan)) else "—"),
            _stat("G2P", fmt_float(best.get("G2P", np.nan))),
            _stat("% Pos Yrs", fmt_pct(best.get("%Pos Yrs", np.nan))),
            _stat("Σ Premium", fmt_usd(best.get("Σ Premium", np.nan))),
        ])

        # Recommendation rows — different dimensions for single vs worst-of
        if is_worstof:
            rec_rows = [
                _rec_row_html("Tenor", _top2_for_dim(sub, "Tenor")),
                _rec_row_html("Leg A Strike", _top2_for_dim(sub, "Leg A Strike")),
                _rec_row_html("Leg A KO", _top2_for_dim(sub, "Leg A KO")),
                _rec_row_html("Leg B Strike", _top2_for_dim(sub, "Leg B Strike")),
                _rec_row_html("Leg B KO", _top2_for_dim(sub, "Leg B KO")),
                _rec_row_html("Gate", _top2_for_dim(sub, "Gate")),
            ]
        else:
            rec_rows = [
                _rec_row_html("Tenor", _top2_for_dim(sub, "Tenor")),
                _rec_row_html("Strike", _top2_for_dim(sub, "Strike")),
                _rec_row_html("KO", _top2_for_dim(sub, "KO")),
                _rec_row_html("Gate", _top2_for_dim(sub, "Gate")),
            ]
        rec_rows_html = "".join(rec_rows)
        meta_subtitle = "worst-of" if is_worstof else "single currency"

        # Direction breakdown badges next to the underlying name. For a
        # single-direction group, show one badge; for mixed (rare but
        # possible in worst-of), show both with counts.
        sub_dir_counts = sub["Direction"].value_counts() \
            if "Direction" in sub.columns else pd.Series(dtype=int)
        dir_badges = []
        for dirn in ("Call", "Put", "Mixed"):
            if dirn in sub_dir_counts.index:
                count = sub_dir_counts[dirn]
                if len(sub_dir_counts) == 1:
                    dir_badges.append(direction_badge_html(
                        dirn, with_arrow=True))
                else:
                    dir_badges.append(
                        direction_badge_html(dirn, with_arrow=True)
                        + f' <span style="opacity:0.55; '
                          f'font-size:11px;">{count:,}</span>'
                    )
        dir_badges_html = ("&nbsp;&nbsp;" + "&nbsp;&nbsp;".join(dir_badges)) \
            if dir_badges else ""

        html = f"""
        <div class="sum-card">
          <div class="sum-head">
            <p class="sum-ul-name">{ul}{dir_badges_html}</p>
            <p class="sum-ul-meta">{len(sub):,} strategies · {meta_subtitle}</p>
          </div>
          <div class="sum-grid-2col">
            <div>
              <p class="sum-section-label">Best strategy</p>
              <div class="sum-best">
                <div class="sum-best-headrow">
                  <span class="sum-best-headlabel">{_metric_label}</span>
                  <span class="sum-best-headval">{best_score_str}</span>
                </div>
                <div class="sum-best-dims">{dim_lines_html}</div>
                <div class="sum-stat-grid">{stats_html}</div>
              </div>
            </div>
            <div>
              <p class="sum-section-label">Top recommendations</p>
              <div class="sum-rec-list">{rec_rows_html}</div>
            </div>
          </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

        # "Load drilldown" button — sets the Strategy detail tab's selection
        # to this underlying's best strategy. Streamlit can't programmatically
        # switch tabs, so we toast a reminder for the user to click the
        # "Strategy detail" tab manually.
        best_name = best.get("Strategy")
        # Only show the button when:
        #   1. we have a time-series file loaded (the detail tab uses it)
        #   2. the strategy name exists in the time-series file's coverage
        ts_strategies = set(ts_data["strategy_name"].unique()) \
            if ts_data is not None else set()
        btn_col, hint_col = st.columns([1, 3])
        with btn_col:
            if ts_data is None:
                st.button(
                    "📈 Load drilldown",
                    key=f"sum_btn_{ul}",
                    disabled=True,
                    help="Upload a time-series file to enable drilldown.",
                    use_container_width=True,
                )
            elif best_name not in ts_strategies:
                st.button(
                    "📈 Load drilldown",
                    key=f"sum_btn_{ul}",
                    disabled=True,
                    help=("This strategy isn't in the time-series file "
                          "(may have been filtered out at backtest time)."),
                    use_container_width=True,
                )
            else:
                if st.button(
                    "📈 Load drilldown",
                    key=f"sum_btn_{ul}",
                    help=("Load this strategy into the Strategy detail tab, "
                          "then click that tab to view drill-down charts."),
                    use_container_width=True,
                ):
                    st.session_state["ts_strategy"] = best_name
                    st.toast(
                        f"Loaded **{ul}** best strategy into Strategy detail. "
                        f"Click the **📈 Strategy detail** tab to view it.",
                        icon="📈",
                    )
        with hint_col:
            if (ts_data is not None and best_name in ts_strategies and
                    st.session_state.get("ts_strategy") == best_name):
                st.markdown(
                    '<div style="font-size:12px; opacity:0.62; '
                    'padding-top:8px;">✓ Loaded — switch to the '
                    '<b>📈 Strategy detail</b> tab above.</div>',
                    unsafe_allow_html=True,
                )
        st.markdown("")  # vertical spacing before next card

    # ===== Sections by StructureType =====
    # Each section maps a structure to a (section title, is_legged flag) tuple.
    # `is_legged` controls which card layout to use (leg A/B view or
    # single Strike/KO view).
    _section_specs = [
        ("single",     "### Single-currency strategies",       False),
        ("worst_of",   "### Dual-currency (worst-of) strategies", True),
        ("eko_basket", "### EKO basket strategies",             False),
        ("wo_basket",  "### Worst-of basket strategies",        True),
    ]

    any_rendered = False
    if "StructureType" in f.columns:
        for struct, section_title, is_legged in _section_specs:
            sub_all = f[f["StructureType"] == struct].copy()
            if sub_all.empty or sub_all["Underlying"].dropna().empty:
                continue
            st.markdown(section_title)
            for ul in sorted(sub_all["Underlying"].dropna().unique()):
                sub = sub_all[sub_all["Underlying"] == ul]
                _render_card(ul, sub, is_worstof=is_legged)
                any_rendered = True

    if not any_rendered:
        st.warning("No strategies in the filtered set to summarise.")

    # ----- Methodology note -----
    with st.expander("ℹ️ How the top 2 are picked"):
        st.markdown(
            """
**Method:** for each dimension (Tenor, Strike, KO, Gate), compute the
**median** of the optimisation metric across all strategies sharing that
dimension value within the current underlying group. Rank by median; ties
broken by sample size (larger n wins — more data ⇒ more reliable estimate).

**Why median, not mean.** Mean is pulled up by lucky outliers — one strategy
with a freak Sharpe of 3.5 can make its tenor look universally great.
Median answers "if I pick this tenor, what's the *typical* result I'd see
across the other dimensions?", which is what you actually want from a
recommendation.

**Why not just the single best strategy?** A single-row maximum tells you
*best-case* but not *expected-case*. When the best strategy doesn't match
the recommendations — e.g. the best is a 6W tenor but the top recommended
tenors are 2M and 3M — that's a useful signal. It usually means the best
strategy is an outlier on a tail of the distribution, not a robust
configuration. The recommendations are a safer starting point for a
follow-up backtest.

**Alternatives** (switch via the Aggregate radio):
- `mean` — standard but outlier-sensitive
- `max` — best single strategy per dimension value; useful when you want
  the upper-envelope view rather than expected performance

Hover any pill to see the chosen aggregator's score and the count of
strategies (`n`) backing it.
            """
        )


# ===========================================================================
# Tab: Optimal & Top-N
# ===========================================================================
with tab_top:
    st.subheader("Optimal strategy")

    metric_options = available_metrics(f)
    cA, cB = st.columns([2, 1])
    metric = cA.selectbox(
        "Optimise on",
        list(metric_options.keys()),
        format_func=lambda k: metric_options[k]["label"],
        index=default_metric_idx(metric_options.keys()),
    )
    higher_better = metric_options[metric]["higher"]
    top_n = cB.slider("Show top N", 5, 50, 10)

    ranked = f.dropna(subset=[metric]).sort_values(
        metric, ascending=not higher_better
    )
    if ranked.empty:
        st.warning(f"No rows have a value for {metric}.")
    else:
        best = ranked.iloc[0]

        # Auto-prime the Strategy detail dropdown to the current top strategy
        # on first load (or when the existing selection no longer exists).
        # Once the user picks a different strategy, this respects their choice
        # and does not override it.
        if ts_data is not None:
            _ts_set = set(ts_data["strategy_name"].unique())
            _current = st.session_state.get("ts_strategy")
            if _current is None or _current not in _ts_set:
                _top = best["Strategy"]
                if _top in _ts_set:
                    st.session_state["ts_strategy"] = _top

        # Direction badge for the current best strategy — sits above the
        # headline cards so the reader sees Call/Put context immediately.
        _is_wo = best.get("OptType") in ("worst_of", "wo_basket")
        _best_dir = best.get("Direction")
        if _is_wo:
            # For worst-of, show per-leg direction badges
            la_d = best.get("Leg A Direction")
            lb_d = best.get("Leg B Direction")
            badge_html_parts = []
            if isinstance(la_d, str):
                badge_html_parts.append(
                    f'<span style="opacity:0.62; font-size:12px;">'
                    f'Leg A</span>&nbsp;'
                    f'{direction_badge_html(la_d, with_arrow=True, compact=True)}'
                )
            if isinstance(lb_d, str):
                badge_html_parts.append(
                    f'<span style="opacity:0.62; font-size:12px;">'
                    f'Leg B</span>&nbsp;'
                    f'{direction_badge_html(lb_d, with_arrow=True, compact=True)}'
                )
            if badge_html_parts:
                st.markdown(
                    "&nbsp;&nbsp;&nbsp;".join(badge_html_parts),
                    unsafe_allow_html=True,
                )
        else:
            if isinstance(_best_dir, str):
                st.markdown(
                    f'<span style="opacity:0.62; font-size:12px;">'
                    f'Direction</span>&nbsp;'
                    f'{direction_badge_html(_best_dir, with_arrow=True, compact=True)}',
                    unsafe_allow_html=True,
                )

        # Headline cards - row 1: dimensions (worst-of-aware)
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        if _is_wo:
            # Show leg breakdowns instead of single Strike/KO
            def _leg_value(strike, ko):
                ss = strike if (strike is not None and pd.notna(strike)) else "?"
                kk = ko if (ko is not None and pd.notna(ko)) else "?"
                return f"{ss} → {kk}"
            k1.metric(
                f"Leg A ({best.get('Leg A Pair') or '?'})",
                _leg_value(best.get("Leg A Strike"), best.get("Leg A KO")),
            )
            k2.metric(
                f"Leg B ({best.get('Leg B Pair') or '?'})",
                _leg_value(best.get("Leg B Strike"), best.get("Leg B KO")),
            )
            k3.metric("Tenor", best["Tenor"])
            k4.metric(
                "Gate",
                best["Gate"] if best["Gate"] != "None" else "— none —",
            )
            k5.metric("Σ PnL", fmt_usd(best.get("Σ PnL", np.nan)))
            k6.metric("Sharpe (m)", fmt_float(best.get("Sharpe (m)", np.nan)))
        else:
            k1.metric("Strike", best["Strike"])
            k2.metric("Tenor", best["Tenor"])
            k3.metric("KO", best["KO"])
            k4.metric(
                "Gate",
                best["Gate"] if best["Gate"] != "None" else "— none —",
            )
            k5.metric("Σ PnL", fmt_usd(best.get("Σ PnL", np.nan)))
            k6.metric("Sharpe (m)", fmt_float(best.get("Sharpe (m)", np.nan)))

        # Headline cards - row 2: key risk/quality
        # For worst-of, replace KO% (which is NaN) with the leg KO rates
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Win %", fmt_pct(best.get("Win%", np.nan)))
        if _is_wo and ("leg_a_ko_rate_pct" in best.index
                       or "leg_b_ko_rate_pct" in best.index):
            k2.metric(
                "Leg A KO %",
                fmt_pct(best.get("leg_a_ko_rate_pct", np.nan)),
            )
            # Push Max DD/Recovery one slot right; show Leg B KO % in k3
            k3.metric(
                "Leg B KO %",
                fmt_pct(best.get("leg_b_ko_rate_pct", np.nan)),
            )
            k4.metric("Max DD", fmt_usd(best.get("Max DD", np.nan)))
            k5.metric("Σ Premium", fmt_usd(best.get("Σ Premium", np.nan)))
        else:
            k2.metric("KO %", fmt_pct(best.get("KO%", np.nan)))
            k3.metric("Max DD", fmt_usd(best.get("Max DD", np.nan)))
            k4.metric("Recovery %", fmt_pct(best.get("Recovery%", np.nan)))
            k5.metric("Σ Premium", fmt_usd(best.get("Σ Premium", np.nan)))
        # Show whichever sample-size column the file has
        _n_col = "n" if "n" in best.index and pd.notna(best.get("n")) \
            else ("n trades" if "n trades" in best.index else None)
        if _n_col is not None and pd.notna(best.get(_n_col)):
            k6.metric(_n_col, f"{int(best[_n_col]):,}")
        else:
            k6.metric("n", "—")

        # Headline cards - row 3: annual / risk-adjusted (only if present)
        extra = [c for c in ["Sharpe(y) μ", "Sharpe(y) min",
                             "annual_sharpe_score", "Calmar",
                             "G2P", "%Pos Yrs", "Min Ann $", "Ulcer", "Yrs"]
                 if c in best.index]
        if extra:
            cols = st.columns(min(6, len(extra)))
            for i, c in enumerate(extra[:len(cols)]):
                spec = METRIC_SPECS.get(c, {"fmt": "float"})
                v = best.get(c, np.nan)
                if spec["fmt"] == "usd":
                    s = fmt_usd(v)
                elif spec["fmt"] == "pct":
                    s = fmt_pct(v)
                elif c == "Yrs":
                    s = f"{int(v)}" if pd.notna(v) else "—"
                else:
                    s = fmt_float(v)
                cols[i].metric(c, s)

        st.markdown("---")

        # ---- Column glossary expander ----
        with st.expander("ℹ️ Column glossary — what each metric means"):
            st.markdown(COLUMN_GLOSSARY_MD)

        st.subheader(f"Top {top_n} by {metric_options[metric]['label']}")
        st.caption("👆 Click a row, then click **Strategy detail →** to drill in.")
        cols_to_show = table_cols(ranked)
        top_slice = ranked.head(top_n).copy()
        # Positional alignment: index i in the rendered table == strategy at i
        top_strategies = top_slice["Strategy"].tolist()

        event = st.dataframe(
            style_table(top_slice[cols_to_show]),
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="topn_table",
        )

        selected_rows = (
            event.selection.rows
            if event is not None and hasattr(event, "selection")
            and event.selection is not None else []
        )
        if selected_rows and selected_rows[0] < len(top_strategies):
            chosen = top_strategies[selected_rows[0]]
            ts_strategies = (set(ts_data["strategy_name"].unique())
                             if ts_data is not None else set())
            has_ts = chosen in ts_strategies

            cA, cB = st.columns([4, 1])
            cA.markdown(f"**Selected:** `{chosen}`")
            help_text = ("Pre-loads this strategy in the Strategy detail tab "
                         "— then click the tab to view its charts."
                         if has_ts else
                         "No time-series data available for this strategy.")
            if cB.button(
                "📈 Strategy detail →",
                type="primary",
                use_container_width=True,
                disabled=not has_ts,
                help=help_text,
                key="drill_to_detail_btn",
            ):
                st.session_state["ts_strategy"] = chosen
                st.toast(
                    f"✓ **{chosen}** loaded — click the **📈 Strategy detail** "
                    f"tab to view its charts.",
                    icon="📈",
                )

        # Quick visual — color by monthly Sharpe (consistent red→green scale)
        topdf = ranked.head(top_n).copy()
        topdf["Label"] = topdf["Strategy"].astype(str).str.replace(
            r"\s+", " ", regex=True)
        color_col = "Sharpe (m)" if "Sharpe (m)" in topdf.columns \
            and topdf["Sharpe (m)"].notna().any() else None
        hover = [c for c in ["Σ PnL", "Win%", "KO%", "Max DD", "Recovery%",
                             "Calmar", "G2P", "Ulcer", "Min Ann $",
                             "%Pos Yrs", "annual_sharpe_score"]
                 if c in topdf.columns]
        bar_kwargs = dict(
            x=metric, y="Label", orientation="h",
            hover_data=hover,
            title=f"Top {top_n} strategies by {metric_options[metric]['label']}",
        )
        if color_col is not None:
            bar_kwargs.update(color=color_col,
                              color_continuous_scale="RdYlGn")
        fig = px.bar(topdf[::-1], **bar_kwargs)
        fig.update_layout(height=max(360, 24 * len(topdf)),
                          yaxis_title="", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # ------------------------------------------------------------------
        # Recommendations: for each dimension, the top 2 values by median
        # of the current optimization metric. Useful as a focused grid for
        # follow-up backtests (e.g. a dual-digital structure).
        # ------------------------------------------------------------------
        st.markdown("---")
        st.subheader("🎯 Recommended values for follow-up testing")

        # Candidate dimensions — include leg-level ones when any leg-based
        # structure (single-currency worst-of OR WO basket) is present.
        candidate_dims = ["Tenor", "Strike", "KO", "Gate"]
        if (("OptType" in f.columns)
                and f["OptType"].isin(["worst_of", "wo_basket"]).any()):
            candidate_dims.extend([
                "Leg A Strike", "Leg A KO", "Leg A Gate",
                "Leg B Strike", "Leg B KO", "Leg B Gate",
            ])
        dims_with_variation = [
            d for d in candidate_dims
            if d in f.columns and f[d].dropna().nunique() >= 2
        ]
        if not dims_with_variation:
            st.info(
                "Not enough variation in the filtered set to recommend "
                "dimensional values — only one unique value per dimension. "
                "Loosen filters or load a richer dataset."
            )
        else:
            metric_label = metric_options[metric]["label"]
            st.caption(
                f"For each dimension, the **top 2 values by median "
                f"{metric_label}** across the filtered set. Use these as "
                f"starting points for a focused grid (e.g. dual-digi)."
            )
            spec = METRIC_SPECS.get(metric, {"fmt": "float", "higher": True})
            higher_better_rec = spec["higher"]

            # Lay out in rows of up to 4 cards
            for row_start in range(0, len(dims_with_variation), 4):
                chunk = dims_with_variation[row_start:row_start + 4]
                rec_cols = st.columns(len(chunk))
                for col_idx, dim in enumerate(chunk):
                    agg = f.groupby(dim)[metric].median().dropna()
                    sizes = f.groupby(dim).size()
                    df_rank = pd.DataFrame(
                        {"score": agg, "n": sizes}
                    ).dropna(subset=["score"])
                    df_rank = df_rank.sort_values(
                        ["score", "n"],
                        ascending=[not higher_better_rec, False],
                    )
                    top2 = df_rank.head(2)
                    with rec_cols[col_idx]:
                        st.markdown(f"**{dim}**")
                        if top2.empty:
                            st.write("—")
                        for val, row in top2.iterrows():
                            score = row["score"]
                            nobs = int(row["n"])
                            if spec["fmt"] == "usd":
                                score_str = fmt_usd(score)
                            elif spec["fmt"] == "pct":
                                score_str = fmt_pct(score)
                            else:
                                score_str = fmt_float(score, 2)
                            val_disp = str(val)
                            if dim == "Gate" and len(val_disp) > 30:
                                val_disp = val_disp[:27] + "…"
                            st.markdown(
                                f"- `{val_disp}` — **{score_str}** "
                                f"<span style='color:#888'>(n={nobs})</span>",
                                unsafe_allow_html=True,
                            )


# ===========================================================================
# Tab: Per-dimension impact
# ===========================================================================
with tab_dim:
    st.subheader("Per-dimension impact")
    st.caption(
        "Aggregates over all other dimensions to show how each individual "
        "axis (Strike / Tenor / KO / Gate) affects performance. Use this to "
        "isolate which knob is doing the work."
    )

    _avail = available_metrics(f)
    impact_metric = st.selectbox(
        "Metric",
        list(_avail.keys()),
        format_func=lambda k: _avail[k]["label"],
        index=default_metric_idx(_avail.keys()), key="dim_metric",
    )
    agg_func = st.radio("Aggregate", ["median", "mean"],
                        horizontal=True, key="dim_agg")

    dims = [("Strike", strike_to_numeric),
            ("Tenor", lambda t: TENOR_TO_DAYS.get(str(t).upper(), 9999)),
            ("KO", ko_to_numeric),
            ("Gate", lambda g: g)]
    # Skip the Gate dimension when there are no real gates in the data
    if f["Gate"].nunique() <= 1:
        dims = [d for d in dims if d[0] != "Gate"]

    cols = st.columns(2)
    for i, (dim, key) in enumerate(dims):
        with cols[i % 2]:
            grp = (f.groupby(dim)[impact_metric]
                     .agg(agg_func).reset_index()
                     .dropna(subset=[impact_metric]))
            if grp.empty:
                st.info(f"No data for {dim}.")
                continue
            grp = grp.sort_values(dim, key=lambda s: s.map(key))
            grp["_color"] = grp[impact_metric]
            fig = px.bar(
                grp, x=dim, y=impact_metric,
                color="_color", color_continuous_scale="RdYlGn",
                title=f"{agg_func.title()} {impact_metric} by {dim}",
            )
            fig.update_layout(
                showlegend=False, coloraxis_showscale=False,
                height=320, margin=dict(l=10, r=10, t=50, b=10),
                xaxis_title="", yaxis_title=impact_metric,
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Marginal sensitivity: range vs. mean")
    st.caption(
        "Range (max − min) across each dimension's aggregate — bigger range "
        "= that dimension matters more for the chosen metric."
    )
    ranges = []
    _hb = METRIC_SPECS.get(impact_metric, {"higher": True})["higher"]
    for dim, key in dims:
        grp = (f.groupby(dim)[impact_metric]
                 .agg(agg_func).dropna())
        if len(grp) >= 2:
            ranges.append({"Dimension": dim,
                           "Range": grp.max() - grp.min(),
                           "Min": grp.min(), "Max": grp.max(),
                           "Best level": grp.idxmax() if _hb else grp.idxmin()})
    if ranges:
        rdf = pd.DataFrame(ranges).sort_values("Range", ascending=False)
        rfig = px.bar(rdf, x="Range", y="Dimension", orientation="h",
                      color="Range", color_continuous_scale="Viridis",
                      hover_data=["Min", "Max", "Best level"],
                      title=f"Which dimension moves {impact_metric} the most?")
        rfig.update_layout(height=300, coloraxis_showscale=False,
                           margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(rfig, use_container_width=True)


# ===========================================================================
# Tab: Heatmaps
# ===========================================================================
with tab_heat:
    st.subheader("Heatmaps — two-dimensional interactions")

    axes = ["Strike", "Tenor", "KO"]
    if f["Gate"].nunique() > 1:
        axes.append("Gate")

    c1, c2, c3, c4 = st.columns(4)
    # Default: X = Strike, Y = KO
    x_axis = c1.selectbox(
        "X-axis", axes,
        index=axes.index("Strike") if "Strike" in axes else 0,
        key="hm_x",
    )
    y_options = [a for a in ["KO", "Tenor", "Strike", "Gate"] if a in axes]
    y_axis = c2.selectbox(
        "Y-axis", y_options,
        index=y_options.index("KO") if "KO" in y_options else 0,
        key="hm_y",
    )
    _avail = available_metrics(f)
    heat_metric = c3.selectbox(
        "Metric",
        list(_avail.keys()),
        format_func=lambda k: _avail[k]["label"],
        index=default_metric_idx(_avail.keys()), key="hm_metric",
    )
    heat_agg = c4.radio("Aggregate", ["median", "mean", "max"],
                        horizontal=True, key="hm_agg")

    if x_axis == y_axis:
        st.warning("Choose two different axes.")
    else:
        key_map = {"Strike": strike_to_numeric,
                   "KO": ko_to_numeric,
                   "Tenor": lambda t: TENOR_TO_DAYS.get(str(t).upper(), 9999),
                   "Gate": lambda g: g}
        pivot = (f.pivot_table(values=heat_metric, index=y_axis,
                               columns=x_axis, aggfunc=heat_agg))
        if pivot.empty:
            st.warning("Not enough data for this combination.")
        else:
            pivot = pivot.reindex(
                index=sorted(pivot.index, key=key_map[y_axis]),
                columns=sorted(pivot.columns, key=key_map[x_axis]),
            )
            # Lower-is-better metrics get a reversed color scale
            reverse = not METRIC_SPECS.get(heat_metric, {"higher": True})["higher"]
            scale = "RdYlGn_r" if reverse else "RdYlGn"
            fig = px.imshow(
                pivot, text_auto=".2f", aspect="auto",
                color_continuous_scale=scale,
                labels=dict(color=heat_metric),
                title=f"{heat_agg.title()} {heat_metric}: {y_axis} × {x_axis}",
            )
            fig.update_layout(height=max(360, 32 * len(pivot.index) + 100),
                              margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Show numeric pivot"):
                st.dataframe(pivot.round(2), use_container_width=True)


# ===========================================================================
# Tab: Gate uplift
# ===========================================================================
with tab_gate:
    st.subheader("Gate uplift vs. ungated baseline")
    st.caption(
        "For each (Strike, Tenor, KO), compute the metric without a gate "
        "and with each gate, then show the **average uplift** the gate adds."
    )

    has_baseline = "None" in f["Gate"].unique()
    has_real_gates = (f["Gate"] != "None").any()
    if not has_baseline:
        st.info("No ungated baseline strategies in current filter. "
                "Include Gate = 'None' to enable this view.")
    elif not has_real_gates:
        st.info("This dataset contains only ungated strategies — "
                "no gates to compare. (Load a file with gate variants "
                "to see uplift analysis.)")
    else:
        _avail = available_metrics(
            f, only=["Sharpe (m)", "Σ PnL", "Win%", "%Pos Yrs", "Recovery%",
                     "Sharpe(y) μ", "Sharpe(y) min", "annual_sharpe_score",
                     "Calmar", "G2P",
                     "PnL/Premium", "PnL/|DD|", "Min Ann $",
                     "KO%", "Ulcer", "Max DD"],
        )
        gate_metric = st.selectbox(
            "Metric",
            list(_avail.keys()),
            format_func=lambda k: _avail[k]["label"],
            index=default_metric_idx(_avail.keys()), key="gate_metric",
        )
        key_cols = ["Strike", "Tenor", "KO"]
        baseline = (f[f["Gate"] == "None"]
                    .set_index(key_cols)[gate_metric]
                    .rename("baseline"))
        merged = (f[f["Gate"] != "None"]
                  .merge(baseline, on=key_cols, how="left"))
        merged["uplift"] = merged[gate_metric] - merged["baseline"]
        merged["uplift_pct"] = np.where(
            merged["baseline"].abs() > 1e-12,
            100 * (merged[gate_metric] - merged["baseline"]) / merged["baseline"].abs(),
            np.nan,
        )

        gate_summary = (merged.groupby("Gate")
                              .agg(avg_uplift=("uplift", "mean"),
                                   med_uplift=("uplift", "median"),
                                   avg_uplift_pct=("uplift_pct", "mean"),
                                   pct_better=("uplift",
                                               lambda s: 100 * (s > 0).mean()),
                                   n=("uplift", "count"))
                              .reset_index()
                              .sort_values("avg_uplift", ascending=False))

        reverse = not METRIC_SPECS.get(gate_metric, {"higher": True})["higher"]
        if reverse:
            gate_summary = gate_summary.sort_values("avg_uplift")

        fig = px.bar(
            gate_summary, x="avg_uplift", y="Gate", orientation="h",
            color="avg_uplift",
            color_continuous_scale="RdYlGn_r" if reverse else "RdYlGn",
            hover_data=["med_uplift", "avg_uplift_pct", "pct_better", "n"],
            title=f"Average uplift in {gate_metric} vs. ungated baseline",
        )
        fig.update_layout(height=max(300, 36 * len(gate_summary)),
                          margin=dict(l=10, r=10, t=50, b=10),
                          coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        show = gate_summary.copy()
        _is_dollar = gate_metric in ["Σ PnL", "Max DD", "Min Ann $"]
        show["avg_uplift"] = show["avg_uplift"].apply(
            lambda v: fmt_usd(v) if _is_dollar else fmt_float(v, 3))
        show["med_uplift"] = show["med_uplift"].apply(
            lambda v: fmt_usd(v) if _is_dollar else fmt_float(v, 3))
        show["avg_uplift_pct"] = show["avg_uplift_pct"].apply(fmt_pct)
        show["pct_better"] = show["pct_better"].apply(fmt_pct)
        show = show.rename(columns={
            "avg_uplift": "Avg uplift",
            "med_uplift": "Median uplift",
            "avg_uplift_pct": "Avg uplift %",
            "pct_better": "% of (S,T,KO) where gate helped",
            "n": "# comparisons",
        })
        st.dataframe(show, use_container_width=True, hide_index=True)


# ===========================================================================
# Tab: Pareto frontier
# ===========================================================================
with tab_pareto:
    st.subheader("Risk-return Pareto frontier")
    c1, c2 = st.columns(2)

    # Risk-side metrics: Max DD and Min Ann $ are signed (negative = worse) and
    # need sign-flipping for the "minimize x" Pareto logic.
    risk_candidates = ["Max DD", "Min Ann $", "KO%", "Ulcer", "TX/Premium %"]
    risk_axes = [c for c in risk_candidates
                 if c in f.columns and f[c].notna().any()]
    reward_candidates = ["annual_sharpe_score", "Σ PnL", "Sharpe (m)",
                         "Sharpe(y) μ", "Sharpe(y) min",
                         "Calmar", "G2P", "Win%", "%Pos Yrs", "Recovery%",
                         "PnL/Premium", "PnL/|DD|"]
    reward_axes = [c for c in reward_candidates
                   if c in f.columns and f[c].notna().any()]

    px_x = c1.selectbox("X (cost / risk)", risk_axes, index=0)
    px_y = c2.selectbox(
        "Y (reward)", reward_axes,
        index=default_metric_idx(reward_axes),
    )

    plot_df = f.dropna(subset=[px_x, px_y]).copy()
    if plot_df.empty:
        st.warning("Not enough data.")
    else:
        # Signed-negative risk metrics: less negative = better, so flip sign
        # for the minimize-x Pareto sweep.
        flip_x = px_x in {"Max DD", "Min Ann $", "Sharpe(y) min"}
        x_vals = -plot_df[px_x] if flip_x else plot_df[px_x]
        y_vals = plot_df[px_y]

        # Pareto: minimize x (after flip if needed), maximize y
        # We want non-dominated points: no other has lower x AND higher y.
        order = np.lexsort((-y_vals.values, x_vals.values))  # asc x, desc y
        best_y = -np.inf
        is_pareto = np.zeros(len(plot_df), dtype=bool)
        for idx in order:
            if y_vals.iloc[idx] > best_y:
                best_y = y_vals.iloc[idx]
                is_pareto[idx] = True
        plot_df["Pareto"] = is_pareto

        plot_df["Label"] = plot_df["Strategy"].astype(str)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df.loc[~plot_df["Pareto"], px_x],
            y=plot_df.loc[~plot_df["Pareto"], px_y],
            mode="markers",
            marker=dict(size=6, color="lightgray", opacity=0.55),
            name="Dominated",
            text=plot_df.loc[~plot_df["Pareto"], "Label"],
            hovertemplate="%{text}<br>" + px_x + ": %{x}<br>"
                          + px_y + ": %{y}<extra></extra>",
        ))
        pareto = plot_df[plot_df["Pareto"]].copy()
        # Sort the frontier line: for flipped risk axes, higher x = better/safer
        pareto = pareto.sort_values(px_x, ascending=not flip_x)
        fig.add_trace(go.Scatter(
            x=pareto[px_x], y=pareto[px_y],
            mode="markers+lines",
            marker=dict(size=10, color="crimson",
                        line=dict(width=1, color="black")),
            line=dict(color="crimson", width=2, dash="dot"),
            name="Pareto-efficient",
            text=pareto["Label"],
            hovertemplate="%{text}<br>" + px_x + ": %{x}<br>"
                          + px_y + ": %{y}<extra></extra>",
        ))
        fig.update_layout(
            title=f"{px_y} vs. {px_x} — Pareto frontier in red",
            xaxis_title=px_x, yaxis_title=px_y, height=560,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Pareto-efficient strategies**")
        st.dataframe(style_table(pareto[table_cols(pareto)]),
                     use_container_width=True, hide_index=True)


# ===========================================================================
# Tab: Side-by-side compare
# ===========================================================================
with tab_compare:
    st.subheader("Side-by-side comparison")
    st.caption("Pick any two strategies from the filtered set and diff them.")
    f2 = f.copy()
    # Use the full Strategy string as label — works for all formats, including
    # worst-of where Strike/KO are NaN.
    f2["Label"] = f2["Strategy"].astype(str)
    options = f2["Label"].tolist()
    c1, c2 = st.columns(2)
    a = c1.selectbox("Strategy A", options, index=0, key="cmp_a")
    b_idx = 1 if len(options) > 1 else 0
    b = c2.selectbox("Strategy B", options, index=b_idx, key="cmp_b")

    A = f2[f2["Label"] == a].iloc[0]
    B = f2[f2["Label"] == b].iloc[0]

    # Show every metric that exists in the data, in a sensible order
    metric_order = ["n", "n trades", "Yrs", "Feas%", "KO%", "Win%", "%Pos Yrs",
                    "notional_usd", "Σ Premium", "Σ TX Cost", "Σ Payout",
                    "Σ PnL",
                    "Sharpe (m)", "Sharpe(y) μ", "Sharpe(y) min",
                    "annual_sharpe_std", "annual_sharpe_cv",
                    "annual_sharpe_score",
                    "Calmar", "G2P", "Ulcer",
                    "Max DD", "Min Ann $", "Recovery%",
                    "PnL/Premium", "PnL/|DD|", "TX/Premium %",
                    "leg_a_ko_rate_pct", "leg_b_ko_rate_pct",
                    "both_survive_rate_pct", "structure_vs_min_leg_pct"]
    metrics = [m for m in metric_order if m in f.columns]

    rows = []
    dollar_set = {"Σ Premium", "Σ TX Cost", "Σ Payout", "Σ PnL",
                  "Max DD", "Min Ann $"}
    pct_set = {"Feas%", "KO%", "Win%", "%Pos Yrs", "Recovery%", "TX/Premium %"}
    for m in metrics:
        va, vb = A.get(m, np.nan), B.get(m, np.nan)
        if pd.isna(va) and pd.isna(vb):
            continue
        diff = (vb - va) if (pd.notna(va) and pd.notna(vb)) else np.nan
        if m in dollar_set:
            fa, fb, fd = fmt_usd(va), fmt_usd(vb), fmt_usd(diff)
        elif m in pct_set:
            fa, fb, fd = fmt_pct(va), fmt_pct(vb), (
                "—" if pd.isna(diff) else f"{diff:+.1f} pp")
        elif m in ("n", "n trades", "Yrs"):
            fa = f"{int(va):,}" if pd.notna(va) else "—"
            fb = f"{int(vb):,}" if pd.notna(vb) else "—"
            fd = "—" if pd.isna(diff) else f"{int(diff):+,}"
        else:
            fa, fb = fmt_float(va, 3), fmt_float(vb, 3)
            fd = "—" if pd.isna(diff) else f"{diff:+.3f}"
        rows.append({"Metric": m, "A": fa, "B": fb, "B − A": fd})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ===========================================================================
# Tab: Strategy detail (timeseries-driven)
# ===========================================================================
with tab_detail:
    st.subheader("Strategy detail")

    if ts_data is None:
        st.info(
            "No time-series data loaded. Pair a `*_timeseries.csv` file with "
            "your summary file to unlock this view.\n\n"
            "**Auto-pairing**: in the preloaded `data/` folder, a summary file "
            "named `xxx_summary.csv` will be paired with `xxx_timeseries.csv` "
            "automatically.\n\n"
            "**Manual**: in upload mode, use the second uploader to add the "
            "time-series CSV."
        )
    else:
        # Choose a strategy that exists in BOTH the summary and the timeseries
        ts_strategies = ts_data["strategy_name"].unique().tolist()
        summary_strategies = data["Strategy"].unique().tolist()
        common = [s for s in ts_strategies if s in summary_strategies]

        if not common:
            st.warning(
                "The time-series file doesn't share any strategy names with "
                "the summary file. Make sure they're from the same backtest run."
            )
        else:
            # If the Top-N drill-down primed a strategy that isn't valid for
            # the current timeseries file, clear it so the selectbox doesn't
            # raise.
            if ("ts_strategy" in st.session_state
                    and st.session_state["ts_strategy"] not in common):
                del st.session_state["ts_strategy"]
            sel = st.selectbox("Strategy", common, key="ts_strategy")
            sub = ts_data[ts_data["strategy_name"] == sel]
            summary_row = data[data["Strategy"] == sel].iloc[0]

            # --- Dimensions caption (worst-of aware) ---
            # Show a Direction badge row first so it's the first thing visible
            _det_dir = summary_row.get("Direction")
            if isinstance(_det_dir, str):
                if summary_row.get("OptType") in ("worst_of", "wo_basket"):
                    leg_dir_html = []
                    la_d = summary_row.get("Leg A Direction")
                    lb_d = summary_row.get("Leg B Direction")
                    if isinstance(la_d, str):
                        leg_dir_html.append(
                            f'<span style="opacity:0.62;'
                            f'font-size:12px;">Leg A</span>&nbsp;'
                            f'{direction_badge_html(la_d, with_arrow=True, compact=True)}'
                        )
                    if isinstance(lb_d, str):
                        leg_dir_html.append(
                            f'<span style="opacity:0.62;'
                            f'font-size:12px;">Leg B</span>&nbsp;'
                            f'{direction_badge_html(lb_d, with_arrow=True, compact=True)}'
                        )
                    if leg_dir_html:
                        st.markdown(
                            "&nbsp;&nbsp;&nbsp;".join(leg_dir_html),
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        f'<span style="opacity:0.62;'
                        f'font-size:12px;">Direction</span>&nbsp;'
                        f'{direction_badge_html(_det_dir, with_arrow=True, compact=True)}',
                        unsafe_allow_html=True,
                    )

            if summary_row.get("OptType") in ("worst_of", "wo_basket"):
                def _leg_caption(pair, strike, ko, gate):
                    bits = f"{strike or '?'} → {ko or '?'}"
                    if gate and pd.notna(gate):
                        bits += f"  *[{gate}]*"
                    return bits

                parts = []
                if pd.notna(summary_row.get("Leg A Pair")):
                    parts.append(
                        f"**Leg A** ({summary_row['Leg A Pair']}): "
                        + _leg_caption(
                            summary_row["Leg A Pair"],
                            summary_row.get("Leg A Strike"),
                            summary_row.get("Leg A KO"),
                            summary_row.get("Leg A Gate"),
                        )
                    )
                if pd.notna(summary_row.get("Leg B Pair")):
                    parts.append(
                        f"**Leg B** ({summary_row['Leg B Pair']}): "
                        + _leg_caption(
                            summary_row["Leg B Pair"],
                            summary_row.get("Leg B Strike"),
                            summary_row.get("Leg B KO"),
                            summary_row.get("Leg B Gate"),
                        )
                    )
                if pd.notna(summary_row.get("Tenor")):
                    parts.append(f"**Tenor**: {summary_row['Tenor']}")
                if parts:
                    st.markdown(" · ".join(parts))
            else:
                parts = []
                for label, key in [("Strike", "Strike"), ("Tenor", "Tenor"),
                                    ("KO", "KO"), ("Gate", "Gate")]:
                    v = summary_row.get(key)
                    if pd.notna(v) and v not in ("None",):
                        parts.append(f"**{label}**: {v}")
                if parts:
                    st.markdown(" · ".join(parts))

            # --- Headline cards from the summary row ---
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Σ PnL", fmt_usd(summary_row.get("Σ PnL", np.nan)))
            k2.metric("Max DD", fmt_usd(summary_row.get("Max DD", np.nan)))
            k3.metric("Sharpe (m)",
                      fmt_float(summary_row.get("Sharpe (m)", np.nan)))
            k4.metric("Calmar",
                      fmt_float(summary_row.get("Calmar", np.nan)))
            k5.metric("Win %", fmt_pct(summary_row.get("Win%", np.nan)))
            k6.metric(
                "n trades",
                f"{int(summary_row['n trades']):,}"
                if pd.notna(summary_row.get("n trades", np.nan)) else "—",
            )

            daily = sub[sub["period_type"] == "daily"].copy()
            monthly = sub[sub["period_type"] == "monthly"].copy()
            annual = sub[sub["period_type"] == "annual"].copy()

            # --- Cumulative P&L + drawdown stacked chart ---
            # Prefer daily resolution when present; fall back to monthly.
            # The equity_usd / drawdown_usd columns are point-in-time
            # cumulative values, so monthly snapshots still produce a
            # meaningful curve (just lower resolution).
            curve = daily if not daily.empty else monthly
            curve_label = "daily" if not daily.empty else "monthly"
            if not curve.empty:
                from plotly.subplots import make_subplots
                fig = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.65, 0.35], vertical_spacing=0.06,
                    subplot_titles=(
                        f"Cumulative P&L ({curve_label})",
                        f"Drawdown ({curve_label})",
                    ),
                )
                fig.add_trace(go.Scatter(
                    x=curve["period_end"], y=curve["equity_usd"],
                    mode="lines" if curve_label == "daily" else "lines+markers",
                    name="Equity",
                    line=dict(color="#1f77b4", width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
                ), row=1, col=1)
                fig.add_hline(y=0, line=dict(color="gray", width=1,
                                              dash="dash"), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=curve["period_end"], y=curve["drawdown_usd"],
                    mode="lines" if curve_label == "daily" else "lines+markers",
                    name="Drawdown",
                    line=dict(color="crimson", width=1),
                    fill="tozeroy", fillcolor="rgba(220,20,60,0.25)",
                    hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
                ), row=2, col=1)
                fig.update_yaxes(title_text="USD", row=1, col=1,
                                 tickformat="$,.0f")
                fig.update_yaxes(title_text="USD", row=2, col=1,
                                 tickformat="$,.0f")
                fig.update_layout(
                    height=540, showlegend=False,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(
                    "No daily or monthly time-series data available for "
                    "this strategy."
                )

            # --- Monthly + annual P&L bars side-by-side ---
            c1, c2 = st.columns(2)
            with c1:
                if not monthly.empty:
                    mfig = px.bar(
                        monthly, x="period_end", y="pnl_usd",
                        title="Monthly P&L",
                        color=monthly["pnl_usd"] >= 0,
                        color_discrete_map={True: "#2ca02c", False: "#d62728"},
                    )
                    mfig.update_layout(
                        height=380, showlegend=False,
                        xaxis_title="", yaxis_title="USD",
                        yaxis_tickformat="$,.0f",
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    st.plotly_chart(mfig, use_container_width=True)
                else:
                    st.info("No monthly data.")

            with c2:
                if not annual.empty:
                    annual = annual.copy()
                    annual["year"] = annual["period_end"].dt.year
                    annual = annual.sort_values("year")
                    years_sorted = sorted(annual["year"].unique().tolist())
                    afig = px.bar(
                        annual, x="year", y="pnl_usd",
                        title="Annual P&L",
                        color=annual["pnl_usd"] >= 0,
                        color_discrete_map={True: "#2ca02c", False: "#d62728"},
                        text="pnl_usd",
                        category_orders={"year": years_sorted},
                    )
                    afig.update_traces(
                        texttemplate="$%{text:,.0f}", textposition="outside",
                    )
                    afig.update_layout(
                        height=380, showlegend=False,
                        xaxis_title="", yaxis_title="USD",
                        yaxis_tickformat="$,.0f",
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    afig.update_xaxes(type="category",
                                      categoryorder="array",
                                      categoryarray=years_sorted)
                    st.plotly_chart(afig, use_container_width=True)
                else:
                    st.info("No annual data.")

            # --- Annual Sharpe (computed from monthly P&L per year) ---
            if not monthly.empty:
                m = monthly.copy()
                m["year"] = m["period_end"].dt.year
                stats = (m.groupby("year")["pnl_usd"]
                         .agg(["mean", "std", "count"]).reset_index())
                stats["annual_sharpe"] = np.where(
                    stats["std"] > 0,
                    stats["mean"] / stats["std"] * np.sqrt(12),
                    np.nan,
                )
                stats = stats.sort_values("year")
                sharpe_years_sorted = sorted(stats["year"].unique().tolist())
                if stats["annual_sharpe"].notna().any():
                    sfig = px.bar(
                        stats, x="year", y="annual_sharpe",
                        title="Annual Sharpe (computed from monthly P&L · "
                              "mean/σ × √12)",
                        color=stats["annual_sharpe"] >= 0,
                        color_discrete_map={True: "#2ca02c", False: "#d62728"},
                        text="annual_sharpe",
                        category_orders={"year": sharpe_years_sorted},
                    )
                    sfig.update_traces(
                        texttemplate="%{text:.2f}", textposition="outside",
                    )
                    sfig.add_hline(y=0, line=dict(color="gray", width=1,
                                                   dash="dash"))
                    sfig.update_layout(
                        height=340, showlegend=False,
                        xaxis_title="", yaxis_title="Sharpe",
                        margin=dict(l=10, r=10, t=40, b=10),
                    )
                    sfig.update_xaxes(type="category",
                                      categoryorder="array",
                                      categoryarray=sharpe_years_sorted)
                    st.plotly_chart(sfig, use_container_width=True)

                    # Cross-reference with summary's reported aggregates
                    s_mean = summary_row.get("Sharpe(y) μ", np.nan)
                    s_min = summary_row.get("Sharpe(y) min", np.nan)
                    if pd.notna(s_mean) or pd.notna(s_min):
                        st.caption(
                            f"Summary file reports: mean = "
                            f"**{fmt_float(s_mean)}**, "
                            f"worst = **{fmt_float(s_min)}**. "
                            "Small differences vs. the chart are expected — "
                            "the summary uses the full per-trade timeseries; "
                            "here we estimate from aggregated monthly P&L."
                        )


# ===========================================================================
# Tab: Data
# ===========================================================================
with tab_data:
    st.subheader("Filtered data")
    cols = table_cols(f)
    st.dataframe(style_table(f[cols]), use_container_width=True,
                 hide_index=True, height=600)

    extra = [c for c in ["Pair", "OptType"] if c in f.columns]
    csv_bytes = f[cols + extra].to_csv(index=False).encode()
    st.download_button("⬇️ Download filtered CSV", data=csv_bytes,
                       file_name=f"{pair}_filtered.csv", mime="text/csv")

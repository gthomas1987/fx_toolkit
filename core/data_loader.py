"""Single-folder data loader for the KO pricer.

Strategy 9 uses one market_data folder. Reads `_index.csv` if present,
otherwise scans filenames for SPOT/VOL_ATM/FWD_POINTS using the same
heuristics as strategy 8.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

_PAIR_RE = re.compile(r"^[A-Z]{6}$")
_VALID_TENORS = {"ON", "1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y", "12M"}
_TENOR_ALIASES = {"12M": "1Y"}
_CATEGORY_KEYWORDS = {
    "SPOT": "SPOT",
    "VOL_ATM": "VOL_ATM", "ATM_VOL": "VOL_ATM", "ATM": "VOL_ATM",
    "VOL_25R": "VOL_25R", "25R": "VOL_25R", "RR_25": "VOL_25R",
    "VOL_25B": "VOL_25B", "25B": "VOL_25B", "BF_25": "VOL_25B",
    "FWD_POINTS": "FWD_POINTS", "FWD_PTS": "FWD_POINTS",
    "FWD": "FWD_POINTS", "FORWARD": "FWD_POINTS", "FORWARDS": "FWD_POINTS",
}


def _is_pair(s: str) -> bool:
    return bool(_PAIR_RE.match(s))


def _canon_tenor(t: str) -> str:
    return _TENOR_ALIASES.get(t.upper(), t.upper())


def _parse_filename(stem: str):
    if not stem or stem.startswith("_"):
        return None
    raw = stem.upper()
    onoff = ""
    if "_ONSHORE" in raw:
        onoff = "ONSHORE"; raw = raw.replace("_ONSHORE", "")
    elif "_OFFSHORE" in raw:
        onoff = "OFFSHORE"; raw = raw.replace("_OFFSHORE", "")

    parts = raw.split("_")
    if parts and _is_pair(parts[0]):
        pair = parts[0]
        if len(parts) == 1:
            return {"pair": pair, "category": "SPOT", "tenor": "",
                    "onshore_offshore": onoff}
        last = parts[-1]
        is_tenor_at_end = last in _VALID_TENORS or last == "NA"
        if is_tenor_at_end:
            tenor = "" if last == "NA" else _canon_tenor(last)
            cat_str = "_".join(parts[1:-1])
            cat = _CATEGORY_KEYWORDS.get(cat_str)
            if cat:
                return {"pair": pair, "category": cat, "tenor": tenor,
                        "onshore_offshore": onoff}
        cat_str = "_".join(parts[1:])
        cat = _CATEGORY_KEYWORDS.get(cat_str)
        if cat:
            return {"pair": pair, "category": cat, "tenor": "",
                    "onshore_offshore": onoff}

    no_us = raw.replace("_", "")
    if len(no_us) >= 8 and _is_pair(no_us[:6]):
        pair = no_us[:6]; rest = no_us[6:]
        for t in sorted(_VALID_TENORS, key=lambda x: -len(x)):
            if rest in (f"{t}V", f"V{t}"):
                return {"pair": pair, "category": "VOL_ATM",
                        "tenor": _canon_tenor(t), "onshore_offshore": onoff}
    return None


def _load_index(folder: str) -> pd.DataFrame:
    p = Path(folder) / "_index.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    for col in ("pair", "category", "tenor", "csv_filename", "onshore_offshore"):
        if col not in df.columns:
            df[col] = ""
    df["tenor"] = df["tenor"].fillna("").astype(str).str.upper().apply(
        lambda t: _canon_tenor(t) if t else "")
    df["onshore_offshore"] = (df["onshore_offshore"].fillna("")
                               .astype(str).str.upper())
    # Normalise the category column to the canonical form. This bridges
    # _index.csv files written with either old-style codes (VOL_25R,
    # VOL_25B) or new-style codes (VOL_RR_25D, VOL_BF_25D) into a single
    # form the rest of the loader can match against. canon_category is
    # a passthrough for unknown values, so SPOT / FWD_POINTS / etc. are
    # untouched.
    from core.conventions import canon_category
    df["category"] = (df["category"].fillna("").astype(str)
                       .str.upper().apply(canon_category))
    return df


def _build_index(folder: str) -> pd.DataFrame:
    p = Path(folder)
    if not p.exists():
        return pd.DataFrame()
    rows = []
    for csv in p.glob("*.csv"):
        if csv.name.startswith("_"):
            continue
        parsed = _parse_filename(csv.stem)
        if parsed:
            parsed["csv_filename"] = csv.name
            rows.append(parsed)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def get_index(folder: str) -> pd.DataFrame:
    if not folder:
        return pd.DataFrame()
    p = Path(folder)
    if (p / "_index.csv").exists():
        return _load_index(folder)
    return _build_index(folder)


def discovery_summary(folder: str) -> dict:
    if not folder or not Path(folder).exists():
        return {"folder": folder, "exists": False, "mode": "—",
                "n_files": 0, "n_pairs": 0, "categories": {}}
    has_idx = (Path(folder) / "_index.csv").exists()
    idx = get_index(folder)
    n_pairs = idx["pair"].nunique() if not idx.empty else 0
    cats = (idx.groupby("category")["pair"].nunique().to_dict()
            if not idx.empty else {})
    n_files = len([f for f in Path(folder).glob("*.csv")
                   if not f.name.startswith("_")])
    return {"folder": folder, "exists": True,
            "mode": "_index.csv" if has_idx else "filename scan",
            "n_files": n_files, "n_pairs": int(n_pairs), "categories": cats}


def list_pairs_with_full_set(folder: str, vol_tenor: str) -> list[str]:
    """Pairs with both SPOT and VOL_ATM at the requested tenor."""
    idx = get_index(folder)
    if idx.empty:
        return []
    spot_pairs = set(idx.loc[idx["category"] == "SPOT", "pair"])
    vol_at = set(idx.loc[(idx["category"] == "VOL_ATM")
                          & (idx["tenor"] == _canon_tenor(vol_tenor)), "pair"])
    return sorted(spot_pairs & vol_at)


def load_by_ticker(folder: str, bbg_ticker: str) -> pd.Series:
    """Load a time series by Bloomberg ticker via `_index.csv` lookup.

    Looks for a row in `_index.csv` whose `bbg_ticker` column matches
    `bbg_ticker` (case-sensitive, whitespace-stripped) and reads the
    corresponding `csv_filename`. Returns an empty Series if the ticker
    is not found or the file is missing.
    """
    if not folder:
        return pd.Series(dtype=float)
    p = Path(folder) / "_index.csv"
    if not p.exists():
        return pd.Series(dtype=float)
    try:
        idx = pd.read_csv(p)
    except Exception:
        return pd.Series(dtype=float)
    if "bbg_ticker" not in idx.columns or "csv_filename" not in idx.columns:
        return pd.Series(dtype=float)

    target = str(bbg_ticker).strip()
    rows = idx[idx["bbg_ticker"].astype(str).str.strip() == target]
    if rows.empty:
        return pd.Series(dtype=float)

    fname = str(rows.iloc[0]["csv_filename"]).strip()
    csv_path = Path(folder) / fname
    if not csv_path.exists():
        return pd.Series(dtype=float)
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return pd.Series(dtype=float)

    date_col = next(
        (c for c in df.columns if c.lower() in
         ("date", "dates", "datetime", "timestamp")),
        df.columns[0],
    )
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col)
    val_col = next(
        (c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])),
        df.columns[0],
    )
    return pd.to_numeric(df[val_col], errors="coerce").dropna().sort_index()


def get_pair_value_at_T(folder: str, pair: str, prefer: str, category: str,
                         T_target: float, valuation_date,
                         standard_tenors: tuple[str, ...] = (
                             "1M", "2M", "3M", "6M", "9M", "1Y")
                         ) -> float | None:
    """Linearly interpolate a pair-specific quoted value across tenors.

    Used for ATM vol and forward points when the option's actual T (e.g.
    6W = 42d, 10W = 70d) falls between standard market tenors. Returns
    the value at T_target as of `valuation_date`, or None if no data.

    Linear in T (calendar days). For ATM vol over short horizons (1M-3M)
    this is a fine first-order approximation; market practice for term
    structure interpolation is also linear in T or in variance.
    """
    from core.rates import TENOR_YEARS as _TY

    points = []
    for tenor in standard_tenors:
        df = load_panel(folder, category, tenor, prefer=prefer, pairs=(pair,))
        if df.empty or pair not in df.columns:
            continue
        ser = df[pair].dropna()
        valid = ser.loc[:pd.Timestamp(valuation_date)]
        if valid.empty:
            continue
        T = _TY.get(tenor)
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


def load_panel(folder: str, category: str, tenor: str | None = None,
               prefer: str = "offshore",
               pairs: tuple[str, ...] = ()) -> pd.DataFrame:
    idx = get_index(folder)
    if idx.empty:
        return pd.DataFrame()
    # Canonicalise the query category so callers can pass either old-style
    # ("VOL_25R") or new-style ("VOL_RR_25D") names. _load_index already
    # canonicalises the index side; this closes the loop on the query
    # side. canon_category is a passthrough for non-vol categories
    # (SPOT / FWD_POINTS / RATES).
    from core.conventions import canon_category
    category = canon_category(str(category).upper())
    sel = idx[idx["category"] == category]
    if tenor is not None:
        sel = sel[sel["tenor"] == _canon_tenor(tenor)]
    if pairs:
        sel = sel[sel["pair"].isin(pairs)]
    if sel.empty:
        return pd.DataFrame()
    pref_upper = (prefer or "").upper()
    if pref_upper in ("ONSHORE", "OFFSHORE"):
        sel = sel.assign(_p=(sel["onshore_offshore"] == pref_upper)
                         .astype(int)).sort_values("_p", ascending=False)
        sel = sel.drop(columns="_p")
    out = {}
    for _, row in sel.iterrows():
        if row["pair"] in out:
            continue
        path = Path(folder) / row["csv_filename"]
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        date_col = next((c for c in df.columns
                         if c.lower() in ("date","dates","datetime","timestamp")),
                        df.columns[0])
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col)
        # Column selection: prefer 'close' (case-insensitive) when multiple
        # numeric columns are present (OHLC CSVs are common in spot files).
        # Falls back to first numeric column for single-column CSVs.
        # Without this, a CSV with columns [open, high, low, close] would
        # silently pick `open` — a real bug since prices/vols/barrier-monitoring
        # logic all expect end-of-day closes.
        lower_cols = {c.lower(): c for c in df.columns}
        if "close" in lower_cols:
            val_col = lower_cols["close"]
        else:
            val_col = next((c for c in df.columns
                            if pd.api.types.is_numeric_dtype(df[c])),
                           df.columns[0])
        ser = pd.to_numeric(df[val_col], errors="coerce").dropna()
        if not ser.empty:
            out[row["pair"]] = ser
    return pd.DataFrame(out).sort_index() if out else pd.DataFrame()


# =============================================================================
# OHLC spot loader — for apps that need intraday range data (app 12's
# American-barrier KO uses [Low, High] to determine barrier hits on
# each day, rather than just the daily Close)
# =============================================================================
def load_spot_ohlc(folder: str, pair: str,
                       prefer: str = "offshore") -> "pd.DataFrame":
    """Return SPOT OHLC for a pair as a DataFrame with as many of
    ['open', 'high', 'low', 'close'] columns as the CSV provides.

    The standard `load_panel(folder, "SPOT", ...)` returns only the
    Close column (correctly — that's the canonical end-of-day mark).
    This function preserves the full OHLC structure so callers that
    need intraday range data (e.g. discrete-daily barrier monitoring
    using [Low, High]) can access High/Low alongside Close.

    Returned columns (lowercase, in the order present in the source):
        open, high, low, close
    Missing columns are simply omitted. Index is DatetimeIndex.

    Returns an empty DataFrame if the pair has no SPOT row in
    `_index.csv` or the CSV has no numeric columns.
    """
    idx = get_index(folder)
    if idx.empty:
        return pd.DataFrame()
    sel = idx[(idx["category"] == "SPOT") & (idx["pair"] == pair)]
    if sel.empty:
        return pd.DataFrame()
    pref_upper = (prefer or "").upper()
    if pref_upper in ("ONSHORE", "OFFSHORE"):
        sel = sel.assign(_p=(sel["onshore_offshore"] == pref_upper).astype(int)) \
                 .sort_values("_p", ascending=False) \
                 .drop(columns="_p")
    path = Path(folder) / sel.iloc[0]["csv_filename"]
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    date_col = next((c for c in df.columns
                       if c.lower() in ("date","dates","datetime","timestamp")),
                      df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()

    # Extract OHLC columns (case-insensitive lookup, preserve canonical order)
    out_cols: dict[str, pd.Series] = {}
    for canonical in ("open", "high", "low", "close"):
        for c in df.columns:
            if c.lower() == canonical:
                ser = pd.to_numeric(df[c], errors="coerce")
                out_cols[canonical] = ser
                break

    # Fallback: if no OHLC columns named explicitly but there's a single
    # numeric column, treat it as Close. Lets us tolerate ko_pricer-layout
    # (date,<pair>) CSVs without changing single-column semantics.
    if not out_cols:
        num_col = next((c for c in df.columns
                          if pd.api.types.is_numeric_dtype(df[c])),
                         None)
        if num_col is not None:
            out_cols["close"] = pd.to_numeric(df[num_col], errors="coerce")

    if not out_cols:
        return pd.DataFrame()
    return pd.DataFrame(out_cols).dropna(how="all")


# =============================================================================
# app_11 API (additive — additional convenience functions for the
# portfolio risk monitor). These read the same _index.csv as the
# ko_pricer code path. The functions assume the uploaded data layout
# (column name "close" in each CSV, snake_case category names like
# VOL_RR_25 / VOL_BF_25 alongside the ko_pricer aliases VOL_25R /
# VOL_25B which the underlying _build_index already canonicalises).
# =============================================================================
def _load_app11_ts(folder: Path, fn: str) -> pd.Series:
    """Read a per-(pair, category, tenor) CSV in the app_11 layout
    (`date,close`). Tolerates the ko_pricer layout too (`date,<pair>`):
    picks whichever numeric column is present.
    """
    df = pd.read_csv(folder / fn)
    date_col = next((c for c in df.columns
                       if c.lower() in ("date", "dates", "datetime", "timestamp")),
                      df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    # Prefer "close" if present, else first numeric column.
    if "close" in df.columns:
        return pd.to_numeric(df["close"], errors="coerce").dropna()
    num_col = next((c for c in df.columns
                      if pd.api.types.is_numeric_dtype(df[c])),
                     df.columns[0])
    return pd.to_numeric(df[num_col], errors="coerce").dropna()


def load_index(folder: "str | Path") -> pd.DataFrame:
    """Load `_index.csv` directly without auto-build fallback.

    Used by app_11's `snapshot()` and `available_dates()`. Differs from
    `get_index()` (which falls back to scanning filenames) in that this
    explicitly requires the index file to exist — failing fast is
    preferable for app_11's risk-monitor workflow where the data layout
    is curated.
    """
    folder = Path(folder)
    idx = pd.read_csv(folder / "_index.csv")
    if "tenor" in idx.columns:
        idx["tenor"] = idx["tenor"].fillna("").astype(str)
    return idx


def available_dates(folder: "str | Path") -> list:
    """Sorted list of business-date timestamps available for any pair.

    Uses the first SPOT row in the index as the reference series. All
    panels should be aligned to the same business calendar (or close
    enough — gaps from holidays in one currency don't affect the
    snapshot path).
    """
    folder = Path(folder)
    idx = load_index(folder)
    spot_rows = idx[idx["category"] == "SPOT"]
    if spot_rows.empty:
        return []
    ts = _load_app11_ts(folder, spot_rows.iloc[0]["csv_filename"])
    return list(ts.index)


def snapshot(folder: "str | Path", asof: "pd.Timestamp") -> dict:
    """Return market state at `asof` for every pair in the index.

    Returns a dict keyed by pair string:
        {
          "USDJPY": {
            "spot": 150.12,
            "vols":     {"1W": 8.1, "1M": 8.5, ...},   # ATM
            "rr25":     {"1M": -0.4, ...},             # 25-delta RR
            "bf25":     {"1M": 0.15, ...},             # 25-delta BF
            "fwd_pts":  {"1M": 23.4, ...},
          },
          ...
        }
    All "as-of" values are the most recent observation at or before `asof`.
    Empty dicts are returned for fields the pair doesn't have data for.
    """
    folder = Path(folder)
    idx = load_index(folder)
    asof = pd.Timestamp(asof)
    out: dict = {}
    for pair in sorted(idx["pair"].dropna().unique()):
        sub = idx[idx["pair"] == pair]
        d: dict = {"spot": None, "vols": {}, "rr25": {},
                    "bf25": {}, "fwd_pts": {}}
        for _, row in sub.iterrows():
            try:
                ts = _load_app11_ts(folder, row["csv_filename"])
            except Exception:
                continue
            valid = ts[ts.index <= asof]
            if valid.empty:
                continue
            val = float(valid.iloc[-1])
            cat = row["category"]
            tenor = row.get("tenor", "") or ""
            if cat == "SPOT":
                d["spot"] = val
            elif cat == "VOL_ATM":
                d["vols"][tenor] = val
            elif cat in ("VOL_RR_25", "VOL_25R"):
                d["rr25"][tenor] = val
            elif cat in ("VOL_BF_25", "VOL_25B"):
                d["bf25"][tenor] = val
            elif cat == "FWD_POINTS":
                d["fwd_pts"][tenor] = val

        # Auto-normalize vol units: real FX vols are essentially never
        # > 100% (i.e. > 1.0 as a decimal), so any value > 1 must be a
        # percentage (Bloomberg / industry format, e.g. 7.5 for 7.5%)
        # and we divide by 100 to coerce to decimal. Synthetic data
        # stored as decimals (0.075) passes through unchanged.
        #
        # WHY: the rest of the codebase (vanilla_price, eko_price,
        # dual_eko MC, Greeks, barrier_diagnostics) all assume σ is in
        # decimal form. Without this guard, percentage-format data
        # gets treated as ~750% vol and option prices explode to
        # double-digit-percent-of-notional MTMs.
        #
        # RR can be negative (right-skew vs left-skew), so the magnitude
        # check uses abs(). We make the decision PER-DICT so a
        # potentially mixed-format pair (vols in % but RR/BF already
        # decimal, etc.) is still handled correctly.
        for vol_dict in (d["vols"], d["rr25"], d["bf25"]):
            if vol_dict and any(abs(v) > 1.0 for v in vol_dict.values()):
                for k in list(vol_dict.keys()):
                    vol_dict[k] = vol_dict[k] / 100.0

        out[pair] = d
    return out


def time_series(folder: "str | Path", pair: str, category: str,
                  tenor: str = "") -> pd.Series:
    """Single time series for (pair, category, tenor)."""
    folder = Path(folder)
    idx = load_index(folder)
    from core.conventions import canon_category
    category = canon_category(str(category).upper())
    mask = ((idx["pair"] == pair)
              & (idx["category"] == category)
              & (idx["tenor"].fillna("") == tenor))
    if mask.sum() == 0:
        raise KeyError(f"No data for {pair} {category} {tenor}")
    row = idx[mask].iloc[0]
    return _load_app11_ts(folder, row["csv_filename"])


def interp_vol(snap_pair: dict, tenor_yrs: float,
                  K: float = None, S: float = None) -> float:
    """Interpolate ATM vol at an arbitrary maturity (linear in √T).

    If K and S are provided, applies a rough quadratic smile adjustment
    using the closest available RR/BF tenor (3M preferred).
    """
    TENOR_T = {"1W": 7/365, "1M": 30/365, "2M": 60/365, "3M": 90/365,
                 "6M": 182/365, "1Y": 365/365}
    pairs = [(TENOR_T[t], v) for t, v in snap_pair["vols"].items()
               if t in TENOR_T]
    pairs.sort()
    if not pairs:
        return 0.10
    if tenor_yrs <= pairs[0][0]:
        atm = pairs[0][1]
    elif tenor_yrs >= pairs[-1][0]:
        atm = pairs[-1][1]
    else:
        atm = pairs[0][1]
        for i in range(len(pairs) - 1):
            t1, v1 = pairs[i]
            t2, v2 = pairs[i + 1]
            if t1 <= tenor_yrs <= t2:
                w = ((np.sqrt(tenor_yrs) - np.sqrt(t1))
                       / (np.sqrt(t2) - np.sqrt(t1)))
                atm = v1 + w * (v2 - v1)
                break
    # Optional smile correction
    if K is not None and S is not None and "3M" in snap_pair.get("rr25", {}):
        rr = snap_pair["rr25"].get("3M", 0.0)
        bf = snap_pair["bf25"].get("3M", 0.0)
        m = np.log(K / S) if S > 0 else 0.0
        m_norm = m / max(atm * np.sqrt(max(tenor_yrs, 1/365)), 1e-6)
        atm = atm + 0.5 * rr * m_norm + bf * (m_norm ** 2)
    return float(max(atm, 0.01))

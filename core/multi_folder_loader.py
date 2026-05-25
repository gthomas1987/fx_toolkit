"""Multi-folder data discovery for app 8.

Supports unioning pairs across two folders:
  - Folder 1: standard `_index.csv` convention (handled by core.data_loader)
  - Folder 2: optional, filename-scanned. No _index.csv required.

When a pair exists in both folders, folder 1 wins (preferred-folder
semantic — the user's primary set should take precedence over
auxiliary data).

Public API:
    discovery_summary(folder) → dict
        Summary of what was found in `folder` for the sidebar expander.

    list_pairs_with_full_set(folders, vol_tenor) → list[str]
        Pairs that have BOTH SPOT and VOL_ATM(vol_tenor) across the
        specified folders.

    load_panel_multi(folders, category, tenor, prefer, pairs) → DataFrame
        Like load_panel but searches multiple folders, folder-1 wins.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from core.data_loader import load_panel as _load_panel, get_index
from core.conventions import canon_category


# --- Filename parser for folder-2 (filename-scan) mode -----------------
# Patterns supported:
#   <PAIR>_<CATEGORY>_<TENOR>.csv      e.g. EURUSD_VOL_ATM_1M.csv
#   <PAIR>.csv                          → SPOT
#   <PAIR>_SPOT.csv                     → SPOT
#   <PAIR><TENOR>V.csv                  → VOL_ATM (Bloomberg-style)
#   <PAIR>V<TENOR>.csv                  → VOL_ATM (Bloomberg-style)

_RE_PAIR_CAT_TENOR = re.compile(
    r"^(?P<pair>[A-Z]{6})_(?P<cat>VOL_ATM|VOL_RR_25D|VOL_BF_25D|VOL_RR_10D|"
    r"VOL_BF_10D|FWD_POINTS)_(?P<tenor>[A-Z0-9]+)\.csv$"
)
_RE_PAIR_SPOT = re.compile(r"^(?P<pair>[A-Z]{6})(?:_SPOT)?\.csv$")
_RE_BLOOMBERG_V_SUFFIX = re.compile(
    r"^(?P<pair>[A-Z]{6})(?P<tenor>[A-Z0-9]+)V\.csv$"
)
_RE_BLOOMBERG_V_PREFIX = re.compile(
    r"^(?P<pair>[A-Z]{6})V(?P<tenor>[A-Z0-9]+)\.csv$"
)


def _parse_filename(fn: str) -> "dict | None":
    """Return {pair, category, tenor} or None if unparseable."""
    m = _RE_PAIR_CAT_TENOR.match(fn)
    if m:
        return {"pair": m["pair"], "category": canon_category(m["cat"]),
                  "tenor": m["tenor"]}
    m = _RE_PAIR_SPOT.match(fn)
    if m:
        return {"pair": m["pair"], "category": "SPOT", "tenor": None}
    m = _RE_BLOOMBERG_V_SUFFIX.match(fn)
    if m:
        return {"pair": m["pair"], "category": "VOL_ATM",
                  "tenor": m["tenor"]}
    m = _RE_BLOOMBERG_V_PREFIX.match(fn)
    if m:
        return {"pair": m["pair"], "category": "VOL_ATM",
                  "tenor": m["tenor"]}
    return None


def _has_index_csv(folder: str) -> bool:
    """Does `folder` contain an _index.csv?"""
    return (Path(folder) / "_index.csv").exists()


def _scan_folder(folder: str) -> pd.DataFrame:
    """Scan folder by filename. Returns a DataFrame mirroring the
    _index.csv schema: pair, category, tenor, csv_filename,
    onshore_offshore (empty)."""
    rows = []
    unparseable = 0
    p = Path(folder)
    if not p.exists():
        return pd.DataFrame(columns=["pair", "category", "tenor",
                                        "csv_filename", "onshore_offshore",
                                        "_n_unparseable"])
    for f in sorted(p.glob("*.csv")):
        if f.name.startswith("_"):
            continue
        parsed = _parse_filename(f.name)
        if not parsed:
            unparseable += 1
            continue
        rows.append({
            "pair": parsed["pair"],
            "category": parsed["category"],
            "tenor": parsed["tenor"],
            "csv_filename": f.name,
            "onshore_offshore": "",
        })
    df = pd.DataFrame(rows)
    df.attrs["n_unparseable"] = unparseable
    return df


def _resolve_index(folder: str) -> tuple[pd.DataFrame, str]:
    """Return (index_df, mode_label) where mode is 'index' or 'scan'."""
    if _has_index_csv(folder):
        idx = get_index(folder)
        return idx, "index"
    return _scan_folder(folder), "scan"


def discovery_summary(folder: str) -> dict:
    """Summary dict for the sidebar expander."""
    idx, mode = _resolve_index(folder)
    cats = (idx["category"].value_counts().to_dict()
            if not idx.empty and "category" in idx.columns else {})
    n_unparseable = int(idx.attrs.get("n_unparseable", 0))
    n_pairs = int(idx["pair"].nunique()) if not idx.empty else 0
    return {
        "folder": folder,
        "mode": mode,
        "n_pairs": n_pairs,
        "n_files": len(idx),
        "categories": cats,
        "n_unparseable": n_unparseable,
    }


def list_pairs_with_full_set(folders: tuple[str, ...],
                                  vol_tenor: str) -> list[str]:
    """Pairs that have both SPOT and VOL_ATM(`vol_tenor`) across the
    specified folders. Result is sorted."""
    pairs: set[str] = set()
    for folder in folders:
        idx, _ = _resolve_index(folder)
        if idx.empty:
            continue
        spot_pairs = set(
            idx[idx["category"] == "SPOT"]["pair"].dropna().unique()
        )
        vol_pairs = set(
            idx[(idx["category"] == "VOL_ATM")
                  & (idx["tenor"] == vol_tenor)]["pair"].dropna().unique()
        )
        pairs.update(spot_pairs & vol_pairs)
    return sorted(pairs)


def load_panel_multi(folders: tuple[str, ...],
                          category: str,
                          tenor: str | None = None,
                          prefer: str = "offshore",
                          pairs: tuple[str, ...] = ()) -> pd.DataFrame:
    """Load a panel across multiple folders. Folder-1 wins per pair.

    For folder-1 (index mode) → calls core.data_loader.load_panel.
    For folder-2 (scan mode) → reads matching CSVs directly via the
    same column-detection logic as load_panel.
    """
    out: dict[str, pd.Series] = {}
    canon = canon_category(category)

    for folder in folders:
        idx, mode = _resolve_index(folder)
        if idx.empty:
            continue

        if mode == "index":
            # Use load_panel — same canonicalisation as ts_loader
            from core.ts_loader import load_panel as ts_load_panel
            df = ts_load_panel(folder, category, tenor=tenor,
                                 prefer=prefer, pairs=pairs)
            for col in df.columns:
                if col not in out:  # folder 1 (first) wins
                    out[col] = df[col].dropna()
        else:
            # Scan mode: read matching CSVs directly
            sel = idx[idx["category"] == canon]
            if tenor is not None:
                sel = sel[sel["tenor"] == tenor]
            if pairs:
                sel = sel[sel["pair"].isin(pairs)]
            if sel.empty:
                continue
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
                date_col = next(
                    (c for c in df.columns
                     if c.lower() in ("date", "dates", "datetime", "timestamp")),
                    df.columns[0],
                )
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
                df = df.dropna(subset=[date_col]).set_index(date_col)
                val_col = next(
                    (c for c in df.columns
                     if pd.api.types.is_numeric_dtype(df[c])),
                    df.columns[0],
                )
                ser = pd.to_numeric(df[val_col], errors="coerce").dropna()
                if not ser.empty:
                    out[row["pair"]] = ser

    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_index()

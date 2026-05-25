"""Compatibility shim for apps that import from `core.ts_loader`.

API:
    load_panel(folder, category, tenor=None, prefer="offshore",
                  pairs=()) -> pd.DataFrame
        Wide DataFrame: columns = pair, rows = date.

    list_available_pairs(folder, category="SPOT",
                          prefer="offshore") -> list[str]
        Sorted pairs that have data for the given category.

    list_available_tenors(folder, category, pair=None,
                            prefer="offshore") -> list[str]
        Sorted tenors available for that category (optionally
        restricted to a single pair).

All four are thin wrappers over `core.data_loader.load_panel` (and
its index helper) — single source of truth for the on-disk CSV layout.
"""
from __future__ import annotations

import pandas as pd

from core.data_loader import load_panel as _load_panel, get_index
from core.conventions import canon_category, tenor_sort_key


def load_panel(folder: str,
                  category: str,
                  tenor: str | None = None,
                  prefer: str = "offshore",
                  pairs: tuple[str, ...] = ()) -> pd.DataFrame:
    """Re-export of core.data_loader.load_panel with category aliasing.

    Translates legacy category codes (e.g. 'VOL_25R') to current
    canonical codes (e.g. 'VOL_RR_25D') before lookup so callers can
    use either spelling. See conventions._VOL_CATEGORY_ALIASES.
    """
    canon = canon_category(category)
    df = _load_panel(folder, canon, tenor=tenor, prefer=prefer, pairs=pairs)
    if df.empty and canon != category:
        df = _load_panel(folder, category, tenor=tenor,
                          prefer=prefer, pairs=pairs)
    return df


def list_available_pairs(folder: str,
                            category: str = "SPOT",
                            prefer: str = "offshore") -> list[str]:
    """Pairs with data for the given category in `folder`. Sorted."""
    canon = canon_category(category)
    idx = get_index(folder)
    if idx.empty:
        return []
    sel = idx[idx["category"] == canon]
    if sel.empty and canon != category:
        sel = idx[idx["category"] == category]
    if sel.empty:
        return []
    pref_upper = (prefer or "").upper()
    if pref_upper in ("ONSHORE", "OFFSHORE"):
        sel = (sel.assign(_p=(sel["onshore_offshore"] == pref_upper)
                            .astype(int))
                  .sort_values("_p", ascending=False)
                  .drop(columns="_p"))
    return sorted(sel["pair"].unique().tolist())


def list_available_tenors(folder: str,
                             category: str,
                             pair: str | None = None,
                             prefer: str = "offshore") -> list[str]:
    """Tenors available for the given category (and optional pair).
    Sorted chronologically via conventions.tenor_sort_key."""
    canon = canon_category(category)
    idx = get_index(folder)
    if idx.empty:
        return []
    sel = idx[idx["category"] == canon]
    if sel.empty and canon != category:
        sel = idx[idx["category"] == category]
    if sel.empty:
        return []
    if pair is not None:
        sel = sel[sel["pair"] == pair]
    if sel.empty:
        return []
    pref_upper = (prefer or "").upper()
    if pref_upper in ("ONSHORE", "OFFSHORE"):
        sel = (sel.assign(_p=(sel["onshore_offshore"] == pref_upper)
                            .astype(int))
                  .sort_values("_p", ascending=False)
                  .drop(columns="_p"))
    tenors = [t for t in sel["tenor"].dropna().unique().tolist() if t]
    return sorted(tenors, key=tenor_sort_key)

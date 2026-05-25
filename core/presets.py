"""Preset save/load — bridges app 10 (analytical guidance) to app 9 (bulk runner).

A preset is a small JSON file that captures the strike/KO/tenor
suggestions computed in app 10's "Barrier guidance" tab, in a format
that can populate app 9's multi-selects with one click. The bulk
runner then sweeps a focused grid around the analytical optimum rather
than the user manually transcribing deltas.

# File layout
`<folder>/presets/<pair_a>_<pair_b>_<tenor>_<timestamp>.json`

# Schema (v1)
{
  "preset_version": 1,
  "generated_at": ISO timestamp,
  "label": human-readable string,
  "pair_a": "USDJPY",
  "pair_b": "USDKRW",
  "tenor": "1M",
  "confidence_pct": 95,
  "target_cluster": 0,
  "metadata": {
    "current_spot_a": 156.06,
    "current_spot_b": 1452.91,
    "cluster_mu_a": 156.06,
    "cluster_mu_b": 1452.91,
    "ellipse_half_width_a": 5.9,
    "ellipse_half_width_b": 56.3,
    "expected_sojourn_days": null,
    "sojourn_health": "ok|warn|fail|na",
    "notes": "Free-text description"
  },
  "backtest_grid": {           // for single-leg Backtest tab
    "pairs": ["USDJPY", "USDKRW"],
    "tenors": ["1M"],
    "direction": "Call (up-and-out)",   // matches DIRECTIONS key
    "deltas": ["30Δ", "35Δ", "40Δ"],
    "ko_method": "delta",                // "delta" or "payout"
    "ko_deltas": ["10Δ", "15Δ", "20Δ"],
    "gate_keys": ["hmm_state_0"]
  },
  "worstof_grid": {            // for Worst-of tab
    "pair_combos": [["USDJPY", "USDKRW"]],
    "tenors": ["1M"],
    "direction": "Call (up-and-out)",
    "leg_a_strike_deltas": ["30Δ", "35Δ", "40Δ"],
    "leg_a_ko_deltas": ["10Δ", "15Δ", "20Δ"],
    "leg_b_strike_deltas": ["30Δ", "35Δ", "40Δ"],
    "leg_b_ko_deltas": ["10Δ", "15Δ", "20Δ"],
    "gates_a": ["hmm_state_0"],
    "gates_b": ["hmm_state_0"]
  }
}

# Backward compatibility
Schema includes a `preset_version` field; load_preset checks this and
returns None on unknown versions so a future change can't silently
corrupt the bulk runner.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


CURRENT_VERSION = 1


def _label_to_delta(label: str) -> float:
    """Convert a delta label like '35Δ' to 0.35. 'ATM' → 0.0."""
    if label == "ATM":
        return 0.0
    return float(label.rstrip("Δ").rstrip("Δ")) / 100.0


def nearest_delta_label(target_delta: float,
                            choices: dict[str, float]) -> str:
    """Return the label in `choices` whose numeric value is closest to
    `target_delta`. Used to snap an analytical delta (e.g. 0.347) to
    the nearest discrete grid choice (e.g. '35Δ')."""
    if not choices:
        raise ValueError("Empty choices dict")
    return min(choices.keys(),
                  key=lambda k: abs(choices[k] - target_delta))


def delta_band_around(target_delta: float,
                          choices: dict[str, float],
                          n_each_side: int = 1) -> list[str]:
    """Return a band of labels around the nearest-to-target delta.

    For target=0.35 with choices={50Δ, 45Δ, 40Δ, 35Δ, 30Δ, 25Δ} and
    n_each_side=1, returns ['40Δ', '35Δ', '30Δ'].

    The band always includes the nearest choice. If the nearest is at
    the edge of the grid (e.g. 50Δ for target=0.50), the band can be
    asymmetric. Bands are sorted by numeric delta DESCENDING (matching
    the visual order of the DELTA_CHOICES dict — deep ITM first).
    """
    items_sorted = sorted(choices.items(), key=lambda kv: -kv[1])
    labels_sorted = [k for k, _ in items_sorted]
    nearest = nearest_delta_label(target_delta, choices)
    idx = labels_sorted.index(nearest)
    lo = max(0, idx - n_each_side)
    hi = min(len(labels_sorted), idx + n_each_side + 1)
    return labels_sorted[lo:hi]


def save_preset(folder: str, preset: dict) -> Path:
    """Write preset to `<folder>/presets/<pair_a>_<pair_b>_<tenor>_c<k>_<ts>.json`.

    The `preset_version` field is set automatically. Creates the
    `presets/` subdirectory if missing. Returns the written path.

    Filename includes the target cluster index `c<k>` so that batch
    generations of multiple clusters at the same tenor within the same
    second don't overwrite each other. As a final safety net, a numeric
    suffix is appended if the chosen filename already exists.
    """
    pdir = Path(folder) / "presets"
    pdir.mkdir(parents=True, exist_ok=True)
    out = dict(preset)
    out["preset_version"] = CURRENT_VERSION
    out["generated_at"] = datetime.utcnow().isoformat(
        timespec="microseconds") + "Z"
    pair_a = out.get("pair_a", "PAIR_A")
    pair_b = out.get("pair_b", "PAIR_B")
    tenor = out.get("tenor", "ALL")
    cluster = out.get("target_cluster", 0)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = f"{pair_a}_{pair_b}_{tenor}_c{cluster}_{ts}"
    out_path = pdir / f"{base}.json"
    suffix = 0
    while out_path.exists():
        suffix += 1
        out_path = pdir / f"{base}_{suffix}.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    return out_path


def load_preset(path: Path) -> Optional[dict]:
    """Load a preset from disk, validating the version.

    Returns the parsed dict, or None if the file is malformed or has
    an unrecognised version (older or newer than this module knows
    about). Callers can use the None return to show a friendly
    "preset incompatible with this app version" message.
    """
    try:
        with Path(path).open("r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("preset_version") != CURRENT_VERSION:
        return None
    return data


def list_presets(folder: str) -> list[dict]:
    """Discover all presets in `<folder>/presets/`.

    Returns a list of dicts with keys: 'path', 'label', 'generated_at',
    'pair_a', 'pair_b', 'tenor'. Sorted by generated_at DESCENDING
    (newest first), so the UI can show the most recent at the top.
    Malformed presets are skipped silently.
    """
    pdir = Path(folder) / "presets"
    if not pdir.exists():
        return []
    out = []
    for p in pdir.glob("*.json"):
        data = load_preset(p)
        if data is None:
            continue
        out.append({
            "path": str(p),
            "label": data.get("label",
                                f"{data.get('pair_a','?')} × "
                                f"{data.get('pair_b','?')} · "
                                f"{data.get('tenor','?')}"),
            "generated_at": data.get("generated_at", ""),
            "pair_a": data.get("pair_a"),
            "pair_b": data.get("pair_b"),
            "tenor": data.get("tenor"),
        })
    return sorted(out, key=lambda d: d["generated_at"], reverse=True)


def build_preset(
    pair_a: str, pair_b: str,
    tenor: str,
    confidence_pct: int,
    target_cluster: int,
    direction_label: str,
    target_strike_delta: float,
    target_ko_delta: float,
    delta_choices: dict[str, float],
    ko_delta_choices: dict[str, float],
    metadata: Optional[dict] = None,
    gate_keys: Optional[list[str]] = None,
    band_width: int = 1,
    dynamic_schedule: Optional[list[dict]] = None,
) -> dict:
    """Construct a preset dict from analytical-suggestion inputs.

    Centralises the band-construction logic so the app code in tab 4
    doesn't need to know how the multi-selects are wired. Same
    suggestion is applied to both legs of the worst-of grid (the
    analytical ellipse gives one symmetric optimum per axis); the
    user can edit either leg in app 9 after loading.

    Parameters
    ----------
    direction_label
        Key from the `DIRECTIONS` dict in app 9 (e.g. "Call (up-and-out)").
    target_strike_delta, target_ko_delta
        Analytical optima in [0, 1] — typically from the Mahalanobis
        ellipse + ATM-vol conversion in tab 4. Only used when
        `dynamic_schedule` is None (static mode).
    delta_choices, ko_delta_choices
        The dicts from app 9's constants (DELTA_CHOICES, KO_DELTA_CHOICES).
        Snapping to these guarantees the preset uses values app 9 accepts.
        Only used in static mode.
    band_width
        Half-width of the band, in grid steps. Default 1 = three labels
        (target ± 1 grid step). Increase to widen the swept grid.
        Only used in static mode.
    dynamic_schedule
        WF-C monthly walk-forward schedule (list of dicts from
        `core.wf_schedule.build_monthly_schedule`). When provided, the
        preset is built in DYNAMIC mode: the bulk-runner spec gets the
        schedule attached and looks up (K, H) per trade date. The
        strike/KO band fields are populated with the analytical optima
        as a hint but are NOT used by the engine in dynamic mode.
    """
    is_dynamic = dynamic_schedule is not None

    if is_dynamic:
        # In dynamic mode, no delta band sweep — engine reads levels
        # from the schedule per trade date. But for the loader's sake
        # we still populate strike/KO label slots with the most-recent
        # schedule entry's *implied* deltas so the UI has something to
        # display. The loader ignores these when it sees the schedule.
        strike_band = ["DYN"]
        ko_band = ["DYN"]
    else:
        strike_band = delta_band_around(target_strike_delta, delta_choices,
                                           n_each_side=band_width)
        ko_band = delta_band_around(target_ko_delta, ko_delta_choices,
                                       n_each_side=band_width)
    gates = gate_keys or [None]

    preset = {
        "pair_a": pair_a,
        "pair_b": pair_b,
        "tenor": tenor,
        "confidence_pct": int(confidence_pct),
        "target_cluster": int(target_cluster),
        "look_ahead_mode": ("walk_forward_monthly" if is_dynamic
                                 else "in_sample"),
        "label": (
            f"{pair_a} × {pair_b} · {tenor} · "
            f"{int(confidence_pct)}% cluster {target_cluster}"
            + (" · WF" if is_dynamic else "")
        ),
        "metadata": metadata or {},
        "backtest_grid": {
            "pairs": [pair_a, pair_b],
            "tenors": [tenor],
            "direction": direction_label,
            "deltas": strike_band,
            "ko_method": "delta",
            "ko_deltas": ko_band,
            "gate_keys": gates,
        },
        "worstof_grid": {
            "pair_combos": [[pair_a, pair_b]],
            "tenors": [tenor],
            "direction": direction_label,
            "leg_a_strike_deltas": strike_band,
            "leg_a_ko_deltas": ko_band,
            "leg_b_strike_deltas": strike_band,
            "leg_b_ko_deltas": ko_band,
            "gates_a": gates,
            "gates_b": gates,
        },
    }
    if is_dynamic:
        preset["dynamic_schedule"] = dynamic_schedule
    return preset

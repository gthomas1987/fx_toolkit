# Legacy apps

Apps from the original `fx_levels_monitor` repo that are **not wired into
the FX Toolkit landing page**. Kept here for reference and future
revival — they are NOT discovered by `st.navigation` and won't appear
in the sidebar.

| File                          | What it was               |
|-------------------------------|---------------------------|
| `1b_fwd_points.py`            | Forward points (curve)    |
| `1c_fwd_spreads.py`           | Forward spreads           |
| `1d_fwd_butterflies.py`       | Forward butterflies       |
| `2_india.py`                  | India-specific monitor    |
| `3_audusd_strategy.py`        | AUDUSD strategy           |
| `4_usdnok_strategy.py`        | USDNOK strategy           |
| `5_nzdusd_strategy.py`        | NZDUSD strategy           |
| `6_gbpusd_strategy.py`        | GBPUSD strategy           |
| `7_usdcnh_straddle.py`        | USDCNH 25Δ strangle calendar |
| `8_short_vol.py`              | AUDCAD short-vol (TBD)    |
| `10_joint_distribution.py`    | Adaptive worst-of EKO     |
| `app_11.py`                   | FX exotic risk dashboard  |
| `alerts.py`                   | Vol-surface alerts        |
| `vol_skew.py`                 | Older copy (master lives in `pages/vol_dashboard.py`) |
| `fx_strategy_analyzer.py`     | Older copy (master lives in `pages/backtest_viewer.py`) |
| `portfolio_comparison.py`     | Cross-portfolio comparator |

## Reviving one

To re-add any of these to the landing page:
1. Move the file into `pages/` and rename so the filename is a valid
   Python module (drop digit prefixes — e.g. `10_joint_distribution.py`
   → `joint_distribution.py`).
2. Update its sidebar data-folder block to use `data_dir_input` from
   `core.ui` (so it shares the toolkit's `session_state["data_dir"]`).
3. Add a `PageInfo` entry in `shared/pages.py` and append it to
   `LIVE_PAGES`.
4. The landing card and sidebar entry will appear automatically.

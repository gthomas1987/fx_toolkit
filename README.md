# FX Toolkit

A unified Streamlit super-app that bundles the four user-facing pieces
of the FX desk's research stack behind a single landing page:

| Page             | What it does                                                                |
|------------------|-----------------------------------------------------------------------------|
| Vol Dashboard    | FX vol-surface historical percentiles, normalised smiles, term structure   |
| EKO Pricer       | European-barrier knock-out pricer + backtester + worst-of / basket variants |
| RKO Pricer       | American-barrier knock-out pricer (closed form, trees, PDE) + backtester    |
| Backtest Viewer  | Drilldown UI for the EKO/RKO Pricer CSV outputs                             |

Originally three separate repos (`fx_vol_history`, `fx_levels_monitor`,
`fx-strategy-app`) — merged in 2026.

## Quick start

```bash
pip install -r requirements.txt
./run.sh
# or directly:
streamlit run app.py
```

The landing page opens at `http://localhost:8501`. Click any card to
jump to that sub-app. Use the **Home** entry in the sidebar to come
back.

## Layout

```
fx_toolkit/
├── app.py                  # entrypoint — st.navigation
├── home.py                 # landing page (2×2 card grid)
├── pages/
│   ├── vol_dashboard.py
│   ├── eko_pricer.py
│   ├── rko_pricer.py
│   ├── _rko_pricer_tabs.py  # backtest tab implementations for RKO
│   └── backtest_viewer.py
├── core/                   # 33 modules — pricing, calendar, rates,
│                           # backtest engines, vol surface, charts, ...
├── shared/                 # toolkit-level helpers
│   ├── pages.py            # landing-card / nav metadata registry
│   └── style.py            # shared CSS injectors
├── legacy/                 # earlier apps, NOT in nav — see legacy/README.md
├── market_data/
│   └── _index.csv          # required for every pricing page
├── data/                   # drop EKO/RKO backtest CSVs here for the Viewer
├── .streamlit/config.toml  # dark theme defaults
├── requirements.txt
├── run.sh
└── README.md
```

## Market data folder

Every page that prices something needs a folder containing
`_index.csv` plus per-ticker time-series CSVs (spot, ATM vol, RR, BF,
forward points, OIS rates). The folder picker in the sidebar is
**shared across pages** — set it once on any sub-app and the others
pick it up via `st.session_state["data_dir"]`.

Default: `./market_data/` (the index file is shipped; populate the
per-ticker CSVs yourself, or set `MARKET_DATA_DIR` to point elsewhere).

## Backtest CSVs

The Backtest Viewer reads two flavours of CSV produced by the EKO and
RKO Pricer backtest tabs:

- **Summary** — one row per (pair × delta × tenor × …) strategy
- **Time series** — daily P&L for each strategy

Drop both in `./data/` — the Viewer pairs them automatically by
filename (`*_summary.csv` ↔ `*_timeseries.csv`).

## Decisions made when merging

1. **Modern `st.navigation` API** with a custom landing page (not the
   folder-auto-discovery mode), so the landing isn't forced into the
   sidebar nav.
2. **Shared data-folder** via `session_state["data_dir"]`. All four
   pages call `core.ui.data_dir_input` with the same key.
3. **Password gate removed.** Was protecting the analyzer when
   deployed publicly. Re-add it app-wide by wrapping `app.py` if
   needed.

## Session-state key conventions

`st.navigation` shares `session_state` across pages, so widget keys
and result-cache keys are namespaced by page:

| Page             | Prefix     | Examples                                    |
|------------------|------------|---------------------------------------------|
| EKO Pricer       | `eko_*`    | `eko_backtest_results`, `eko_wo_multiplier_pct`, `ep_*` (pricer widgets) |
| RKO Pricer       | `rko_*`    | `rko_bt_results`, `rko_wo_results`, `rko_rp_results`, `rko_wrp_results`, `rko_wo_multiplier_pct` |
| Vol Dashboard    | `vs_*`     | `vs_pair`, `vs_prefer`                      |
| Backtest Viewer  | own keys   | (separate concern — local to that page)     |
| Shared           | `data_dir` | the only key intentionally shared           |

If you add a new widget to any page, prefix its key with the page
namespace above. Helps prevent surprise carryover when switching pages.

## Download filename convention

CSVs produced by the backtests are named so the Backtest Viewer can
tell them apart when they live side-by-side in `data/`:

| App  | Pattern                                         |
|------|-------------------------------------------------|
| EKO  | `eko_backtest_*.csv`, `eko_worstof_*.csv`, `EKO_*` / `WO-EKO_*` portfolio CSVs |
| RKO  | `rko_backtest_*.csv`, `rko_worstof_*.csv`       |

Summary ↔ time-series pairing still works (the Viewer just swaps
`_summary` → `_timeseries`).

## Adding a sub-app back

See `legacy/README.md` — short version: drop the file in `pages/`,
add a `PageInfo` to `shared/pages.py`, and it shows up on the
landing page automatically.

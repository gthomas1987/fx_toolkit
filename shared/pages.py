"""Page registry — single source of truth for sub-app metadata.

Used by both app.py (to build st.navigation) and home.py (to render
the landing cards). Keeping the metadata here avoids drift between
the two surfaces.

Schema:
    path      : page filepath relative to the project root, the form
                expected by st.Page and st.switch_page
    title     : sidebar label
    icon      : sidebar icon (emoji)
    card_tag  : small uppercase tag on the landing card
    card_body : one-paragraph description on the landing card
    section   : grouping label — drives sidebar section headers via
                st.navigation(dict) and the section blocks on home.py.
                Must be one of the values in SECTIONS below.
"""
from __future__ import annotations

from dataclasses import dataclass


# Section ordering — used by both the sidebar nav and the landing
# page. Order here = display order. Adding a new section is just a
# matter of adding it to this tuple and setting `section=` on the
# PageInfo. Sections with zero pages are automatically hidden.
SECTIONS = (
    "Portfolio",
    "Dashboards",
    "Pricers",
    "Backtest",
    "Workflow",
)


@dataclass(frozen=True)
class PageInfo:
    path: str
    title: str
    icon: str
    card_tag: str
    card_body: str
    section: str = "Tools"


PORTFOLIO_ANALYZER = PageInfo(
    path="pages/portfolio_analyzer.py",
    title="Portfolio Analyzer",
    icon="🎯",
    card_tag="Exotic options risk monitor",
    card_body=(
        "Live mark-to-market, bucketed Greeks, barrier/path risk, "
        "correlation exposure, and multi-axis scenario cubes across a "
        "portfolio of vanillas, call spreads, flys, EKOs, and dual EKOs."
    ),
    section="Portfolio",
)

VOL_DASHBOARD = PageInfo(
    path="pages/vol_dashboard.py",
    title="Vol Dashboard",
    icon="📊",
    card_tag="Vol surface history",
    card_body=(
        "Normalised smile across tenors, ATM percentile bands, term "
        "structure, and an alerts grid for vol-surface extremes."
    ),
    section="Dashboards",
)

EKO_PRICER = PageInfo(
    path="pages/eko_pricer.py",
    title="EKO Pricer",
    icon="🟢",
    card_tag="European-barrier knock-out",
    card_body=(
        "Single-trade pricing, daily-rolling backtest, worst-of EKOs, "
        "and basket portfolios across delta · tenor · pair grids."
    ),
    section="Pricers",
)

RKO_PRICER = PageInfo(
    path="pages/rko_pricer.py",
    title="RKO Pricer",
    icon="🔵",
    card_tag="American-barrier knock-out",
    card_body=(
        "Reiner–Rubinstein closed form, binomial/trinomial trees, "
        "Crank–Nicolson PDE, and a daily-OHLC backtest of continuously"
        "-monitored barriers."
    ),
    section="Pricers",
)

CURRENCY_SCREENER = PageInfo(
    path="pages/currency_screener.py",
    title="Currency Screener",
    icon="🔍",
    card_tag="Worst-of / dual-digital screener",
    card_body=(
        "Screens FX pair combinations for the long-horizon vs daily "
        "correlation wedge that makes good worst-of and dual-digital "
        "candidates: cointegration tests, variance ratios, rolling ρ "
        "term structure, and per-pair drill-downs."
    ),
    section="Dashboards",
)

JOINT_PAIR_ANALYZER = PageInfo(
    path="pages/joint_pair_analyzer.py",
    title="Joint Pair Analyzer",
    icon="🪐",
    card_tag="Joint distribution analysis",
    card_body=(
        "Two-pair joint historical distribution: 2D KDE for density, "
        "Gaussian Mixture clusters with BIC selection for regimes, "
        "HMM dynamics for sojourn times, and Mahalanobis ellipses for "
        "barrier placement guidance."
    ),
    section="Dashboards",
)

BACKTEST_VIEWER = PageInfo(
    path="pages/backtest_viewer.py",
    title="Backtest Viewer",
    icon="📈",
    card_tag="Strategy analyzer",
    card_body=(
        "Ingests EKO/RKO backtest CSVs (single, worst-of, basket "
        "variants) and renders summary cards, drilldowns, comparison "
        "heatmaps, and printable PDF reports."
    ),
    section="Backtest",
)

OPTION_PRICER = PageInfo(
    path="pages/option_pricer.py",
    title="Option Pricer",
    icon="🟢",
    card_tag="Single-pair multi-leg FX options",
    card_body=(
        "Bloomberg-OVML-style strategy pricer: build single or multi-"
        "leg structures on one pair (vanillas, KOs, strategies like "
        "risk reversals and butterflies), pick exercise style and "
        "pricing model per leg, see live Greeks and a USD-summed "
        "strategy total."
    ),
    section="Pricers",
)

DUAL_CCY_PRICER = PageInfo(
    path="pages/dual_ccy_pricer.py",
    title="Dual CCY Option Pricer",
    icon="🟣",
    card_tag="Two-pair correlation-aware worst-of",
    card_body=(
        "Joint pricing of two-pair worst-of structures (vanilla, EKO, "
        "RKO). Correlation source toggles between manual, realized 60d, "
        "and implied triangulation via the cross-pair's vol. CF + MC "
        "engines, per-leg Greeks, ∂V/∂ρ, and survival-probability "
        "breakdowns."
    ),
    section="Pricers",
)

OPTION_PORTFOLIO_BACKTEST = PageInfo(
    path="pages/option_portfolio_backtest.py",
    title="Option Portfolio Backtest",
    icon="📦",
    card_tag="Basket backtest across pairs & strategy types",
    card_body=(
        "Multi-pair, multi-type basket backtester. Pick currency pairs "
        "and strategy types (Vanilla, EKO, RKO, WO-EKO, WO-RKO); the "
        "tool generates the uniform-parameter grid, daily-rolls each, "
        "and aggregates to a portfolio P&L. Drilldown tabs for per-"
        "strategy / per-pair / per-type with Sharpe, MDD, win-rate, "
        "skew, and drawdown attribution."
    ),
    section="Backtest",
)

PROJECT_MANAGEMENT = PageInfo(
    path="pages/project_management.py",
    title="Project Management",
    icon="📋",
    card_tag="PM work tracker",
    card_body=(
        "Kanban board for projects assigned by the PM — status, "
        "priority, due dates, hours estimated vs. spent. CSV-backed "
        "at data/projects.csv so anything with file access can edit."
    ),
    section="Workflow",
)

# Order = section order, then within each section in the sequence
# below. The pages_by_section() helper preserves this ordering when
# grouping for the sidebar nav and the landing-page card sections.
#
# Note: EKO_PRICER and RKO_PRICER are intentionally NOT in this tuple.
# The pages remain on disk for reference and are easy to re-enable by
# adding them back here. They've been superseded by the new layout:
#   - Option Pricer                single-pair, multi-leg (done)
#   - Dual CCY Option Pricer       two-pair, correlation-aware (done)
#   - Option Portfolio Backtest    basket / multi-strategy (done)
#   - Option Backtest              single-strategy drill-down (next)
LIVE_PAGES = (
    # Portfolio
    PORTFOLIO_ANALYZER,
    # Dashboards
    VOL_DASHBOARD,
    CURRENCY_SCREENER,
    JOINT_PAIR_ANALYZER,
    # Pricers
    OPTION_PRICER,
    DUAL_CCY_PRICER,
    # Backtest
    OPTION_PORTFOLIO_BACKTEST,
    BACKTEST_VIEWER,
    # Workflow
    PROJECT_MANAGEMENT,
)


def pages_by_section() -> dict[str, tuple[PageInfo, ...]]:
    """Group LIVE_PAGES by `section`, preserving SECTIONS ordering.

    Sections with zero pages are omitted. Pages whose `section` is not
    listed in SECTIONS get appended at the end under their own bucket,
    so a typo in `section=` is visible rather than silently swallowed.
    """
    buckets: dict[str, list[PageInfo]] = {s: [] for s in SECTIONS}
    extras: dict[str, list[PageInfo]] = {}
    for page in LIVE_PAGES:
        if page.section in buckets:
            buckets[page.section].append(page)
        else:
            extras.setdefault(page.section, []).append(page)
    result: dict[str, tuple[PageInfo, ...]] = {
        s: tuple(pages) for s, pages in buckets.items() if pages
    }
    for s, pages in extras.items():
        result[s] = tuple(pages)
    return result

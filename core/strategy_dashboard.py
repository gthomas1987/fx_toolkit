"""Shared CSS / styling for strategy-app dashboards.

Used by apps 7 and 8 (and any future strategy apps) so the visual
identity stays consistent. Only one public function: inject_dashboard_css().
"""
from __future__ import annotations

import streamlit as st


_DASHBOARD_CSS = """
<style>
/* Tighten Streamlit default padding for dashboard-style layouts */
.block-container { padding-top: 1rem; padding-bottom: 2rem; }

/* Cleaner section dividers */
hr { margin-top: 1.0rem; margin-bottom: 1.0rem; }

/* Metric cards a bit more compact + readable */
[data-testid="stMetricValue"] { font-size: 1.6rem; }
[data-testid="stMetricLabel"] { font-size: 0.78rem; color: #444; }
[data-testid="stMetricDelta"] { font-size: 0.78rem; }

/* Sidebar section headings stand out */
[data-testid="stSidebar"] .stMarkdown h3 {
    margin-top: 0.5rem; margin-bottom: 0.2rem;
    border-bottom: 1px solid rgba(0,0,0,0.08); padding-bottom: 0.2rem;
}

/* Captions a touch larger and softer */
[data-testid="stCaptionContainer"] { color: #555; font-size: 0.82rem; }

/* Dataframes — denser rows */
[data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {
    padding-top: 0.35rem; padding-bottom: 0.35rem;
}
</style>
"""


def inject_dashboard_css() -> None:
    """Inject the shared dashboard CSS into the page. Safe to call
    multiple times (Streamlit dedupes markdown injections by content)."""
    st.markdown(_DASHBOARD_CSS, unsafe_allow_html=True)

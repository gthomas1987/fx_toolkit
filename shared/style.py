"""Shared CSS for the FX Toolkit.

Two functions are exposed:
    - inject_base_css()   : tightens Streamlit's spacing/typography
    - inject_card_css()   : the dark metric-card styling used by the
                            EKO and RKO pricer pages
Both are idempotent within a script run.
"""
from __future__ import annotations

import streamlit as st


def inject_base_css() -> None:
    """Tighten default padding, typography, and metric cards.

    Lifted from the global CSS block in the original
    fx_strategy_analyzer.py — it noticeably improves visual density
    across all pages.
    """
    if st.session_state.get("_fx_toolkit_base_css"):
        return
    st.session_state["_fx_toolkit_base_css"] = True
    st.markdown(
        """
        <style>
          .block-container {
              padding-top: 2.0rem !important;
              padding-bottom: 1.5rem !important;
              padding-left: 2rem !important;
              padding-right: 2rem !important;
              max-width: 1800px;
          }
          h1, h1 span {
              font-size: 1.55rem !important;
              font-weight: 700 !important;
              line-height: 1.25 !important;
              margin-top: 0 !important;
              margin-bottom: 0.25rem !important;
              padding: 0.25rem 0 0 0 !important;
          }
          h2, h2 span {
              font-size: 1.20rem !important;
              font-weight: 600 !important;
              line-height: 1.3 !important;
              margin: 0.6rem 0 0.4rem 0 !important;
              padding: 0 !important;
          }
          h3, h3 span {
              font-size: 1.02rem !important;
              font-weight: 600 !important;
              line-height: 1.3 !important;
              margin: 0.5rem 0 0.35rem 0 !important;
              padding: 0 !important;
          }
          [data-testid="stMetricValue"],
          [data-testid="stMetricValue"] > div,
          [data-testid="stMetricValue"] div {
              font-size: 1.25rem !important;
              line-height: 1.25 !important;
              font-weight: 600 !important;
          }
          [data-testid="stMetricLabel"],
          [data-testid="stMetricLabel"] > div,
          [data-testid="stMetricLabel"] p {
              font-size: 0.75rem !important;
          }
          [data-testid="stCaptionContainer"],
          [data-testid="stCaptionContainer"] p {
              font-size: 0.80rem !important;
          }
          .stTabs [data-baseweb="tab-list"] button {
              padding: 0.4rem 0.7rem !important;
          }
          .stTabs [data-baseweb="tab-list"] button p {
              font-size: 0.9rem !important;
          }
          /* Sidebar — cap width and shrink multiselect pills. */
          section[data-testid="stSidebar"] {
              width: 17rem !important;
              min-width: 17rem !important;
          }
          section[data-testid="stSidebar"] [data-baseweb="tag"] {
              font-size: 0.75rem !important;
          }
          section[data-testid="stSidebar"] label,
          section[data-testid="stSidebar"] label p {
              font-size: 0.85rem !important;
          }
          /* Body markdown text */
          .stMarkdown p,
          .stMarkdown li {
              font-size: 0.92rem;
          }
          /* DataFrames */
          .stDataFrame, .stDataFrame [role="grid"] {
              font-size: 0.85rem !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_card_css() -> None:
    """Dark metric/tag card styling used by EKO and RKO pricer pages.

    Lifted verbatim from the inline <style> blocks at the top of
    9_ko_pricer.py and 12_american_ko_pricer.py so the two pages keep
    looking the same after the move.
    """
    if st.session_state.get("_fx_toolkit_card_css"):
        return
    st.session_state["_fx_toolkit_card_css"] = True
    st.markdown(
        """
        <style>
        .metric-card { background: #161b26; border: 1px solid #2a3243;
                       border-radius: 10px; padding: 12px 14px; margin: 4px 0; }
        .metric-title { font-size: 11px; color: #8b9bb4;
                        text-transform: uppercase; letter-spacing: 0.5px; }
        .metric-value { font-size: 22px; color: #e2e8f0; font-weight: 600; }
        .metric-sub { font-size: 11px; color: #8b9bb4; margin-top: 2px; }
        .tag-call { background: #166534; color: #86efac; padding: 1px 8px;
                    border-radius: 4px; font-size: 11px; font-weight: 600; }
        .tag-put  { background: #991b1b; color: #fca5a5; padding: 1px 8px;
                    border-radius: 4px; font-size: 11px; font-weight: 600; }
        .tag-ko   { background: #92400e; color: #fde68a; padding: 1px 8px;
                    border-radius: 4px; font-size: 11px; font-weight: 600; }
        .tag-amer { background: #1e3a8a; color: #bfdbfe; padding: 1px 8px;
                    border-radius: 4px; font-size: 11px; font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )

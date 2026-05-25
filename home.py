"""FX Toolkit — landing page.

A 2×2 grid of cards, one per live sub-app. Card metadata comes from
shared/pages.py. Clicking "Open" navigates via st.switch_page.
"""
from __future__ import annotations

from pathlib import Path
import sys

# Make sibling top-level packages (shared/, core/) importable when
# Streamlit executes this file out of the project root.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from shared.pages import LIVE_PAGES
from shared.style import inject_base_css


st.set_page_config(
    page_title="FX Toolkit",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_base_css()


# Landing-page-only styling. The card-body class deliberately uses
# flex layout so cards in the same row stay the same height even when
# descriptions vary in length.
st.markdown(
    """
    <style>
      .fx-hero {
          padding: 0.5rem 0 1.0rem 0;
      }
      .fx-hero-title {
          font-size: 2.0rem !important;
          font-weight: 700 !important;
          color: #e2e8f0;
          margin: 0 0 0.35rem 0;
          line-height: 1.15 !important;
      }
      .fx-hero-sub {
          font-size: 1.0rem;
          color: #94a3b8;
          margin: 0;
      }
      .fx-card-icon  { font-size: 2rem; line-height: 1; margin-bottom: 4px; }
      .fx-card-title {
          font-size: 1.15rem;
          font-weight: 600;
          color: #e2e8f0;
          margin: 0;
      }
      .fx-card-tag {
          font-size: 0.70rem;
          color: #94a3b8;
          text-transform: uppercase;
          letter-spacing: 0.5px;
          margin: 2px 0 8px 0;
      }
      .fx-card-body {
          font-size: 0.92rem;
          color: #cbd5e1;
          line-height: 1.5;
          margin-bottom: 10px;
          min-height: 4.2em;   /* keep all cards the same height */
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# Hero
st.markdown(
    """
    <div class="fx-hero">
      <div class="fx-hero-title">FX Toolkit</div>
      <p class="fx-hero-sub">
        FX vol surface, exotic option pricing, and strategy backtest analytics.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Card grid — 2 columns wide, as many rows as needed to fit LIVE_PAGES.
# Pairs Streamlit's bordered container (for the box) with raw HTML inside
# it (for content styling). The "Open →" button is a real Streamlit
# button so st.switch_page can fire.
n_pages = len(LIVE_PAGES)
n_rows = (n_pages + 1) // 2   # ceil(n_pages / 2)
slots: list = []
for _ in range(n_rows):
    slots.extend(st.columns(2, gap="medium"))

for slot, page in zip(slots, LIVE_PAGES):
    with slot:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="fx-card-icon">{page.icon}</div>
                <div class="fx-card-title">{page.title}</div>
                <div class="fx-card-tag">{page.card_tag}</div>
                <div class="fx-card-body">{page.card_body}</div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "Open →",
                key=f"open_{page.title}",
                use_container_width=True,
                type="primary",
            ):
                st.switch_page(page.path)


# Footer
st.markdown("---")
st.caption(
    "Tip: the **Market data folder** picker in any sub-app sidebar is "
    "shared via session state — set it once and every page uses the "
    "same path."
)

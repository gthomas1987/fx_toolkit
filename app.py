"""FX Toolkit — entrypoint.

Runs the multi-page navigation using Streamlit's modern st.navigation
API. The landing page (home.py) is the default; pages/ contains each
sub-app as a standalone script.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path
import sys

# Make `core/` and `shared/` importable from anywhere by adding the
# project root to sys.path BEFORE we import anything internal.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from shared.pages import LIVE_PAGES


def _build_pages() -> list[st.Page]:
    pages: list[st.Page] = [
        st.Page("home.py", title="Home", icon="🏠", default=True),
    ]
    pages.extend(
        st.Page(p.path, title=p.title, icon=p.icon)
        for p in LIVE_PAGES
    )
    return pages


nav = st.navigation(_build_pages(), position="sidebar")
nav.run()

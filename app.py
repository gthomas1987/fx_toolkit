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

from shared.pages import pages_by_section


def _build_pages() -> dict[str, list[st.Page]]:
    """Build the navigation dict for st.navigation.

    Keys become sidebar section headers. The Home page sits at the
    top under a blank-string key, which Streamlit renders without a
    header so it reads as the "go-here-first" page.
    """
    nav: dict[str, list[st.Page]] = {
        "": [st.Page("home.py", title="Home", icon="🏠", default=True)],
    }
    for section_name, pages in pages_by_section().items():
        nav[section_name] = [
            st.Page(p.path, title=p.title, icon=p.icon)
            for p in pages
        ]
    return nav


nav = st.navigation(_build_pages(), position="sidebar")
nav.run()

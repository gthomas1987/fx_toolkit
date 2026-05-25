"""App 6 — GBPUSD VIX-shock systematic strategy.

Run with:
    streamlit run apps/6_gbpusd_strategy.py
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.strategy_app_helper import run_strategy_app

run_strategy_app(
    pair="GBPUSD",
    app_number="6",
    default_sign=-1,
    default_threshold=1.5,
    default_horizon=2,
)

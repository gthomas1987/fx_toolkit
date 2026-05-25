#!/usr/bin/env bash
# FX Toolkit — local launch script.
# Run from the repo root: `./run.sh`
set -euo pipefail

cd "$(dirname "$0")"

# Optional: pre-populate the data folder so the sidebar starts filled.
# Edit this to point at your local market_data folder if you keep one
# outside the repo.
export MARKET_DATA_DIR="${MARKET_DATA_DIR:-$(pwd)/market_data}"

exec streamlit run app.py "$@"

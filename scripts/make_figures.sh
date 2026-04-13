#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

"${PYTHON_BIN}" -m src.plotting.summary_tables
"${PYTHON_BIN}" -m src.plotting.report_figures
"${PYTHON_BIN}" -m src.plotting.concept_diagrams
"${PYTHON_BIN}" -m src.plotting.paper_figure_assets

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REFERENCE_DIR="/tmp/CIRL_ref"
PINNED_COMMIT="c3fd22580a15d1e008570c08e78b38cdd887ef2c"
ARCHIVE_URL="https://codeload.github.com/OptiMaL-PSE-Lab/CIRL/tar.gz/${PINNED_COMMIT}"
VENV_DIR="${ROOT_DIR}/.venv"

echo "Project root: ${ROOT_DIR}"
echo "Ensuring research scaffold directories exist..."
mkdir -p \
  "${ROOT_DIR}/configs/methods" \
  "${ROOT_DIR}/configs/scenarios" \
  "${ROOT_DIR}/configs/execution" \
  "${ROOT_DIR}/results/raw" \
  "${ROOT_DIR}/results/processed" \
  "${ROOT_DIR}/results/figures" \
  "${ROOT_DIR}/results/tables" \
  "${ROOT_DIR}/notebooks" \
  "${ROOT_DIR}/scripts" \
  "${ROOT_DIR}/src/controllers" \
  "${ROOT_DIR}/src/environments" \
  "${ROOT_DIR}/src/evaluation" \
  "${ROOT_DIR}/src/metrics" \
  "${ROOT_DIR}/src/plotting" \
  "${ROOT_DIR}/src/training" \
  "${ROOT_DIR}/src/utils" \
  "${ROOT_DIR}/tests"

if [[ ! -d "${REFERENCE_DIR}" ]]; then
  echo "Downloading pinned official CIRL reference to ${REFERENCE_DIR}..."
  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "${TMP_DIR}"' EXIT
  curl -L "${ARCHIVE_URL}" -o "${TMP_DIR}/cirl.tar.gz"
  tar -xzf "${TMP_DIR}/cirl.tar.gz" -C "${TMP_DIR}"
  EXTRACTED_DIR="$(find "${TMP_DIR}" -maxdepth 1 -type d -name 'CIRL-*' | head -n 1)"
  if [[ -z "${EXTRACTED_DIR}" ]]; then
    echo "Failed to extract the official CIRL archive." >&2
    exit 1
  fi
  rm -rf "${REFERENCE_DIR}"
  mv "${EXTRACTED_DIR}" "${REFERENCE_DIR}"
else
  echo "Reference repo already present at ${REFERENCE_DIR}"
fi

if command -v python3 >/dev/null 2>&1; then
  if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Creating project virtual environment at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
  fi
  echo "Installing Python requirements into ${VENV_DIR}..."
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip
  if [[ "$(uname -s)" == "Linux" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    echo "Installing CUDA-compatible PyTorch wheel (cu124) for Linux + NVIDIA..."
    "${VENV_DIR}/bin/python" -m pip install "torch>=2.5,<2.8" --index-url https://download.pytorch.org/whl/cu124
  else
    echo "Installing default PyTorch wheel for the current platform..."
    "${VENV_DIR}/bin/python" -m pip install "torch>=2.5,<2.8"
  fi
  "${VENV_DIR}/bin/python" -m pip install -r "${ROOT_DIR}/requirements.txt"
else
  echo "python3 not found. Skipping pip installation." >&2
fi

echo
echo "Setup scaffold complete."
echo "Reference repo: ${REFERENCE_DIR}"
echo "Virtual environment: ${VENV_DIR}"
echo "Next step: source ${VENV_DIR}/bin/activate and run pytest or the training/evaluation scripts."

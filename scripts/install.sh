#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but was not found in PATH." >&2
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${ROOT_DIR}"

if [ ! -d "${VENV_DIR}" ]; then
  uv venv "${VENV_DIR}" --python "${PYTHON_BIN}"
fi

uv pip install --python "${VENV_DIR}/bin/python" --upgrade pip setuptools wheel
uv pip install --python "${VENV_DIR}/bin/python" -e .

echo
echo "Environment ready."
echo "Activate with: source .venv/bin/activate"
echo "Download data with: .venv/bin/python scripts/setup_data.py"

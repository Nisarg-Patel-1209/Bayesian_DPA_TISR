#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${PROJECT_DIR}/.conda"
VENV_DIR="${PROJECT_DIR}/.venv"

if [[ -x "${ENV_DIR}/bin/python" ]]; then
  PYTHON_BIN="${ENV_DIR}/bin/python"
elif [[ -x "${VENV_DIR}/bin/python" ]]; then
  PYTHON_BIN="${VENV_DIR}/bin/python"
else
  PYTHON_BIN="$(command -v python3)"
  echo "No .conda/.venv found under ${PROJECT_DIR}; falling back to ${PYTHON_BIN}."
  echo "Create a project env first with: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
fi

ARGS=("$@")
HAS_CONFIG=0
for arg in "$@"; do
  if [[ "${arg}" == "--config" || "${arg}" == --config=* ]]; then
    HAS_CONFIG=1
    break
  fi
done

if [[ "${HAS_CONFIG}" -eq 0 ]]; then
  ARGS=(--config "${PROJECT_DIR}/config/demo.toml" "${ARGS[@]}")
fi

PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  exec "${PYTHON_BIN}" -m typing_task.run "${ARGS[@]}"

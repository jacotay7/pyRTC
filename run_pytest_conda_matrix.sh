#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR" || exit 1

OUT_FILE="${1:-pytest_conda_matrix.log}"
ENV_LIST=(
  "pyrtc-ci-3.9"
  "pyrtc-ci-3.10"
  "pyrtc-ci-3.11"
  "pyrtc-ci-3.12"
  "pyrtc-ci-3.13"
)

CONDA_BASE="$(conda info --base 2>/dev/null)"
if [[ -z "${CONDA_BASE}" ]]; then
  echo "ERROR: Could not determine conda base path." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

{
  echo "============================================================"
  echo "pyRTC pytest conda matrix"
  echo "Started: $(date)"
  echo "Repo: $ROOT_DIR"
  echo "Output file: $OUT_FILE"
  echo "============================================================"
} > "$OUT_FILE"

PASS_COUNT=0
FAIL_COUNT=0

declare -a FAILED_ENVS=()

for ENV_NAME in "${ENV_LIST[@]}"; do
  {
    echo
    echo "------------------------------------------------------------"
    echo "ENV: $ENV_NAME"
    echo "Start: $(date)"
    echo "------------------------------------------------------------"
  } >> "$OUT_FILE"

  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    {
      echo "ERROR: Env '$ENV_NAME' not found."
      echo "RESULT: FAIL"
      echo "End: $(date)"
    } >> "$OUT_FILE"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    FAILED_ENVS+=("$ENV_NAME (missing env)")
    continue
  fi

  set +e
  {
    conda activate "$ENV_NAME"
    echo "Python: $(python --version 2>&1)"
    echo "Conda env: ${CONDA_DEFAULT_ENV:-unknown}"
    echo "Upgrading pip..."
    python -m pip install --upgrade pip
    echo "Installing package..."
    python -m pip install .
    echo "Installing test dependencies..."
    python -m pip install -r requirements-test.txt
    echo "Running pytest..."
    pytest
  } >> "$OUT_FILE" 2>&1
  EXIT_CODE=$?
  set -e

  {
    echo "pytest exit code: $EXIT_CODE"
    if [[ $EXIT_CODE -eq 0 ]]; then
      echo "RESULT: PASS"
    else
      echo "RESULT: FAIL"
    fi
    echo "End: $(date)"
  } >> "$OUT_FILE"

  conda deactivate >> "$OUT_FILE" 2>&1 || true

  if [[ $EXIT_CODE -eq 0 ]]; then
    PASS_COUNT=$((PASS_COUNT + 1))
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    FAILED_ENVS+=("$ENV_NAME")
  fi

done

{
  echo
  echo "============================================================"
  echo "Matrix complete: $(date)"
  echo "PASS: $PASS_COUNT"
  echo "FAIL: $FAIL_COUNT"
  if [[ ${#FAILED_ENVS[@]} -gt 0 ]]; then
    echo "Failed envs: ${FAILED_ENVS[*]}"
  else
    echo "Failed envs: none"
  fi
  echo "============================================================"
} >> "$OUT_FILE"

echo "Wrote matrix log to: $OUT_FILE"

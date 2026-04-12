#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

EPOCHS="${EPOCHS:-10}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-8}"
RUN_ROOT="${RUN_ROOT:-experiments/agent_swin}"
SAVE_DIR="${RUN_ROOT}/checkpoints"
LOG_DIR="${RUN_ROOT}/logs"
REPORT_DIR="${RUN_ROOT}/reports"

mkdir -p "${SAVE_DIR}" "${LOG_DIR}" "${REPORT_DIR}"

echo "[agent] running policy tests"
python scripts/test_agent_workflow.py

echo "[agent] training Swin checkpoint into ${SAVE_DIR}"
python scripts/train.py \
  --model swin_base_patch4_window7_224 \
  --epochs "${EPOCHS}" \
  --save-dir "${SAVE_DIR}" \
  --log-dir "${LOG_DIR}"

echo "[agent] evaluating triage workflow"
python scripts/evaluate_agent.py \
  --checkpoint-path "${SAVE_DIR}/best_model.pth" \
  --image-size "${IMAGE_SIZE}" \
  --batch-size "${BATCH_SIZE}" \
  --output-dir "${REPORT_DIR}"

echo "[agent] complete"

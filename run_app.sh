#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lora-brand-assistant}"
HOST="${BRAND_ASSISTANT_HOST:-127.0.0.1}"
PORT="${BRAND_ASSISTANT_PORT:-8010}"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
elif [[ -d "/opt/anaconda3" ]]; then
  CONDA_BASE="/opt/anaconda3"
else
  echo "Conda was not found. Install Conda or activate your environment manually first."
  exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

cd "${PROJECT_DIR}"
unset KERAS_BACKEND

EXISTING_PID="$(lsof -tiTCP:${PORT} -sTCP:LISTEN || true)"
if [[ -n "${EXISTING_PID}" ]]; then
  echo "Stopping existing app server on port ${PORT} (PID ${EXISTING_PID})..."
  kill "${EXISTING_PID}" || true
  sleep 1
fi

echo "Starting LoRA Brand Assistant on http://${HOST}:${PORT}"
exec python -m uvicorn app:app --host "${HOST}" --port "${PORT}"

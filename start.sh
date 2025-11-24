#!/usr/bin/env bash
set -euo pipefail

# Ensure cache dirs exist on the Runpod network volume
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${VLLM_CACHE_DIR}"

# Toggle your strategy with env:
#  - FAST BOOT (default): VLLM_EAGER=1 (skip CUDA graph capture; fastest startup)
#  - COMPILE & REUSE:     VLLM_EAGER=0 (your app should set CompilationConfig(cache_dir=/runpod-volume/vllm-compile-cache))
: "${VLLM_EAGER:=0}"
export VLLM_EAGER

API_PORT="${PORT:-8080}"
HEALTH_PORT="${PORT_HEALTH:-8081}"

echo "Starting Gunicorn with API:${API_PORT} HEALTH:${HEALTH_PORT}"

# If no args were provided, run our default server command.
if [ "$#" -eq 0 ]; then
  set -- uv run gunicorn -w 1 -k uvicorn_worker.UvicornWorker \
    --preload -b 0.0.0.0:"${API_PORT}" -b 0.0.0.0:"${HEALTH_PORT}" \
    main:app
fi

# Replace shell with the target process so it receives signals properly.
exec "$@"
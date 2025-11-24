# Dockerfile
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS runtime

# Minimal OS deps + Python + certs + curl
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-dev git ca-certificates \
    curl build-essential ninja-build && \
    rm -rf /var/lib/apt/lists/*

ENV CC=/usr/bin/gcc CXX=/usr/bin/g++ \
    TORCHINDUCTOR_CACHE_DIR=/runpod-volume/torch_cache \
    TRITON_CACHE_DIR=/runpod-volume/triton_cache \
    VLLM_CACHE_DIR=/runpod-volume/vllm_cache \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    # VLLM_ATTENTION_BACKEND=FLASHINFER \
    VLLM_EAGER=0 \
    PORT=8080 \
    PORT_HEALTH=8081

# --- Install uv (package manager) — official guidance copies the uv binary
# Pin a uv version; update as you like.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# (uv images and Docker patterns documented here)
# https://docs.astral.sh/uv/guides/integration/docker/

WORKDIR /app

# Copy only project metadata first (better cache): if you have pyproject + lock, use them.
# If you use requirements.txt instead, copy that and change the uv commands below accordingly.
COPY pyproject.toml .python-version uv.lock* ./

# Create project venv and install (deps only first = faster rebuilds)
# If you use requirements.txt: `uv pip install --system -r requirements.txt`
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project
# Copy source code
COPY api/ ./api/
COPY services/ ./services/
COPY models/ ./models/
COPY main.py ./
COPY start.sh /app
RUN chmod +x /app/start.sh

# Finish dependency install (installs your project into the venv)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

# Ensure the venv is on PATH
ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8080 8081

# Start your FastAPI app (change module:app if different)
ENTRYPOINT ["/app/start.sh"]
CMD ["uv", "run", "gunicorn", "-w", "1", "-k", "uvicorn_worker.UvicornWorker", "--preload", "-b", "0.0.0.0:8080", "-b", "0.0.0.0:8081", "main:app"]
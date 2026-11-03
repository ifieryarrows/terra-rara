# =============================================================================
# Stage 1: Build Native Rust PyO3 Wheel (copper_rust_accel)
# =============================================================================
FROM python:3.11-slim@sha256:9a7765b36773a37061455b332f18e265e7f58f6fea9c419a550d2a8b0e9db834 AS rust-builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gcc \
    libc6-dev \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
    && pip install --no-cache-dir maturin \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

COPY ./rust_accel /build/rust_accel
RUN cd /build/rust_accel && maturin build --release -i python3 --out /wheels

# =============================================================================
# Stage 2: Lean Production Runtime Image (Zero Rust Toolchain Bloat)
# =============================================================================
FROM python:3.11-slim@sha256:9a7765b36773a37061455b332f18e265e7f58f6fea9c419a550d2a8b0e9db834

WORKDIR /code

# OS deps: redis-server + supervisor + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    redis-server \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/code \
    REDIS_URL=redis://127.0.0.1:6379/0 \
    HF_HOME=/data/hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TRANSFORMERS_VERBOSITY=error \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    TOKENIZERS_PARALLELISM=false

# Install compiled Rust PyO3 wheel from builder stage (~250 KB) + Rust hf_transfer
COPY --from=rust-builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl hf_transfer && rm -rf /wheels

# Python deps
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Pre-cache ProsusAI/finbert weights in image layer to eliminate cold-start download on HF Space
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoTokenizer.from_pretrained('ProsusAI/finbert'); AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')" || true; \
    mkdir -p /data/hf_cache && chmod -R a+rwX /data/hf_cache

# App code
COPY ./app /code/app
COPY ./config /code/config
COPY ./adapters /code/adapters
COPY ./worker /code/worker
COPY ./pipelines /code/pipelines
COPY ./migrations /code/migrations
COPY ./deep_learning /code/deep_learning
COPY ./backtest /code/backtest

# Copy pre-trained model files (from Kaggle)
COPY ./data/models /data/models

# Supervisor config
COPY ./supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# HF Spaces default port: 7860
EXPOSE 7860

# Run supervisord (manages redis + api + worker)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

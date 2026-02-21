FROM python:3.11-slim

WORKDIR /code

# OS deps: redis-server + supervisor + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    redis-server \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# App code
COPY ./app /code/app
COPY ./config /code/config
COPY ./adapters /code/adapters
COPY ./worker /code/worker
COPY ./pipelines /code/pipelines
COPY ./migrations /code/migrations

# Copy pre-trained model files (from Kaggle)
COPY ./data/models /data/models

# Supervisor config
COPY ./supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# HF Spaces default port: 7860
EXPOSE 7860

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/code \
    REDIS_URL=redis://127.0.0.1:6379/0 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TRANSFORMERS_VERBOSITY=error \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# Run supervisord (manages redis + api + worker)
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

FROM python:3.11-slim

WORKDIR /code

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app
COPY ./config /code/config

# Copy pre-trained model files (from Kaggle)
COPY ./data/models /data/models

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
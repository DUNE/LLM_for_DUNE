# AskDUNE container image
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Build tools needed by some Python packages.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    python -m pip install --prefix=/install -r requirements.txt

# Install the spaCy model into the same package prefix.
RUN PYTHONPATH=/install/lib/python3.11/site-packages \
    python -m spacy download en_core_web_sm \
        --target /install/lib/python3.11/site-packages


# Runtime image
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000 \
    DB_PATH=/data/chroma \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache

# Runtime programs used by health checks and document processing.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgl1 \
        libglib2.0-0 \
        poppler-utils \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /install /usr/local
COPY . .

# Keep application data and downloaded model caches outside the image.
RUN mkdir -p /data/chroma /cache/huggingface /cache/torch && \
    chgrp -R 0 /app /data /cache && \
    chmod -R g=u /app /data /cache

# Compatible with rootless Podman and future OKD non-root deployment.
USER 1001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl --fail http://127.0.0.1:8000/api/health || exit 1

CMD ["python", "cli.py", "serve"]

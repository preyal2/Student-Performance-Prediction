# Base Python runtime
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=1.8.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false

# Install curl and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

# Set working directory
WORKDIR /app

# Copy dependency definition files
COPY pyproject.toml poetry.lock ./

# Install project dependencies without dev packages
RUN poetry install --no-root --without dev --no-interaction --no-ansi

# Copy application files, web UI, and serialized model
COPY predict.py model.bin index.html ./

# Expose API & Web UI port
EXPOSE 9696

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9696/health || exit 1

# Serve using production WSGI server (Waitress)
ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]

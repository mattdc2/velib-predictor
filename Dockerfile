FROM python:3.14-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy project files
COPY pyproject.toml .python-version ./
COPY src/ ./src/
COPY config/ ./config/

# Install ALL dependencies
RUN uv pip install --system -r pyproject.toml

# Create logs directory
RUN mkdir -p /app/logs

# Run collector
CMD ["uv", "run", "python", "-m", "src.data.collector"]
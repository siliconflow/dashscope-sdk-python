FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first (better cache)
COPY pyproject.toml uv.lock* ./

# Install dependencies
RUN uv install --system --no-cache-dir

# Copy app code
COPY . .

EXPOSE 8000

CMD ["python", "tests/mock_server.py"]

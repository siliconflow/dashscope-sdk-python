# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm

WORKDIR /app

# Install uv (better: copy the static binary from the official uv image)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Optional but common: keep uv caches in a known place
ENV UV_CACHE_DIR=/root/.cache/uv

# Copy only dependency inputs first for better Docker layer caching
COPY pyproject.toml uv.lock ./

# Create/sync the project environment from the lockfile
# --frozen: fail if lock and project metadata don't match; don't update the lock
# --no-dev: skip dev dependency groups in the final image
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Now copy the rest of the app
COPY tests/mock_server.py ./main.py

# Make sure the venv's executables are on PATH
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
CMD ["uv", "run", "--frozen", "python", "main.py"]

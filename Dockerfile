# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm

# 建议：将工作目录改为 /src 或 /code，避免和 app 包名混淆
WORKDIR /src

# --- 1. 安装依赖环境 ---
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_CACHE_DIR=/root/.cache/uv

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# --- 2. 复制代码 ---
# 将本地的 app 目录 复制到 容器的 /src/app
COPY app ./app
# 复制配置文件
COPY gunicorn.conf.py .

ENV PATH="/src/.venv/bin:$PATH"

EXPOSE 8000

# --- 3. 启动命令 ---
# app.main:app 代表：包名(app) -> 文件名(main) -> 实例名(app)
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app.main:app"]

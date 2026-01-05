# 使用官方 Python 基础镜像
FROM python:3.11-slim-bookworm AS builder

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY pyproject.toml uv.lock ./

# 导出依赖并安装到系统路径（或使用虚拟环境）
# --system 标志允许直接安装在镜像的 Python 环境中，适合容器场景
RUN uv sync --frozen --no-dev --system

# 最终镜像
FROM python:3.11-slim-bookworm

WORKDIR /app

# 从构建阶段复制安装好的库
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制源代码
COPY . .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "main.py"]

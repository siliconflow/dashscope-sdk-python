FROM python:3.11-slim-bookworm

WORKDIR /app

# 复制源代码
COPY . .
uv install -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "tests/mock_server.py"]

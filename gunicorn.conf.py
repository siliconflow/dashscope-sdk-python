import multiprocessing
import os

# 获取 CPU 核心数
workers_per_core = 2
cores = multiprocessing.cpu_count()

# 核心计算公式：CPU核数 * 2 + 1
# 这是一个经典的 Gunicorn 推荐公式，既能利用多核，又能防止过多的上下文切换
default_workers = (cores * workers_per_core) + 1

# 允许通过环境变量覆盖（可选，方便调试）
workers = int(os.getenv("WORKERS", default_workers))

# 指定 Worker 类型为 Uvicorn（关键！必须支持 ASGI）
worker_class = "uvicorn.workers.UvicornWorker"

# 绑定地址
bind = "0.0.0.0:8000"

# 日志级别
loglevel = "warning"

# 超时设置（根据 DeepSeek 的流式输出特点，建议稍微调大，防止长文本生成断连）
timeout = 120
keepalive = 5

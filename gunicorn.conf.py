import multiprocessing
import os

# --- 基础配置 ---
bind = "0.0.0.0:8000"
workers = int(os.getenv("WORKERS", multiprocessing.cpu_count())) # 推荐 16
worker_class = "uvicorn.workers.UvicornWorker"

# --- 超时配置 (关键修改) ---
# 1. 核心超时时间：2小时 (7200秒)
# 这意味着如果 Worker 在 2 小时内没有任何响应（心跳），才会被 Master 杀掉。
# 在异步模式下，只要 Event Loop 没卡死，Worker 就不会被杀，
# 这能保证长连接即使在等待上游流式输出时也能存活。
timeout = 7200

# 2. 优雅退出超时
# 当你执行 kill -HUP 或重启服务时，Gunicorn 会等待 7200 秒让当前请求跑完。
# 必须设置这个，否则重启时会切断正在生成的长任务。
graceful_timeout = 7200

# 3. Keepalive
# 建议设为 60-120 秒。
# 作用：防止中间的网络设备（防火墙/路由器）因为 TCP 连接长时间没数据包通过而静默掐断连接。
keepalive = 75

# --- 其他 ---
loglevel = "info"
accesslog = "-"
errorlog = "-"

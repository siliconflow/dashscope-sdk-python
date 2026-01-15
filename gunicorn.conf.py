import multiprocessing
import os

# 获取 CPU 核心数 (16)
cores = multiprocessing.cpu_count()

# --- 优化点 1: Worker 数量 ---
# 对于 Uvicorn (异步) 转发服务，Worker 数等于 CPU 核心数通常性能最佳。
# 太多 Worker 会导致 CPU 上下文切换开销，反而降低 QPS。
# 建议设置为 16 (和核数一致) 或者 17 (16+1)
workers = int(os.getenv("WORKERS", cores))

# 指定 Worker 类型
worker_class = "uvicorn.workers.UvicornWorker"

# 绑定地址
bind = "0.0.0.0:8000"

# --- 优化点 2: 积压队列 ---
# 应对 300 并发瞬间涌入的情况，防止连接被拒绝
backlog = 2048

# 日志级别 (压测时 Warning 刚好，减少磁盘 I/O)
loglevel = "warning"

# --- 优化点 3: 超时策略 ---
# 120秒对于 DeepSeek 的长文本 (80k context) 是安全的。
# 但考虑到网络波动，配合 keepalive 防止中间断连。
timeout = 120

# 建议适当调大，配合压测工具的长连接复用，减少 TCP 握手消耗
keepalive = 65

# 预加载应用代码，加快启动速度并节省内存（可选，但推荐）
preload_app = True

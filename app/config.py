import os
import logging
from typing import Dict, Any

# Base URL configuration
SILICON_FLOW_BASE_URL = os.getenv(
    "SILICON_FLOW_BASE_URL", "https://api.siliconflow.cn/v1"
)

# Model mapping configuration
MODEL_MAPPING = {
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "deepseek-v3.1": "deepseek-ai/DeepSeek-V3.1",
    "deepseek-v3.2": "deepseek-ai/DeepSeek-V3.2",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "default": "deepseek-ai/DeepSeek-V3",
    "pre-siliconflow/deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "pre-siliconflow/deepseek-v3.1": "deepseek-ai/DeepSeek-V3.1",
    "pre-siliconflow/deepseek-v3.2": "deepseek-ai/DeepSeek-V3.2",
    "pre-siliconflow/deepseek-r1": "deepseek-ai/DeepSeek-R1",
}

# Application constants
DUMMY_KEY = "dummy-key"
MAX_NUM_MSG_CURL_DUMP = 5

# Timeout configuration (in seconds)
TIMEOUT_CONFIG = {
    "connect": 10.0,
    "read": 7200.0,  # 2 hours
    "write": 600.0,
    "pool": 10.0,
}

# Server configuration
SERVER_CONFIG = {"host": "0.0.0.0", "port": 8000, "access_log": True}


# Request ID filter for logging
class RequestIDFilter(logging.Filter):
    def __init__(self, request_id_ctx=None):
        super().__init__()
        self.request_id_ctx = request_id_ctx

    def filter(self, record):
        if self.request_id_ctx:
            record.request_id = self.request_id_ctx.get()
        else:
            record.request_id = "-"
        return True


# Logging configuration - will be configured after request_id_ctx is available
def get_logging_config(request_id_ctx=None):
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "request_id": {
                "()": RequestIDFilter,
                "request_id_ctx": request_id_ctx,
            }
        },
        "formatters": {
            "standard": {
                "format": "%(asctime)s.%(msecs)03d | %(levelname)s | %(process)d | [%(request_id)s] | %(message)s",
                "datefmt": "%H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "level": "DEBUG",
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "filters": ["request_id"],
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "DeepSeekProxy": {
                "handlers": ["console"],
                "level": "DEBUG",
                "propagate": False,
            },
            "uvicorn": {"handlers": ["console"], "level": "INFO", "propagate": False},
            "uvicorn.access": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False,
            },
        },
    }

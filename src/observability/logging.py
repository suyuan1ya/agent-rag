"""结构化日志配置（structlog）。"""

from __future__ import annotations

import logging
import os

import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """配置 structlog 结构化日志。

    Args:
        log_level: DEBUG | INFO | WARNING | ERROR
    """
    is_dev = os.getenv("ENV", "dev") == "dev"

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
            if is_dev
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper(), logging.INFO),
    )


def get_logger(name: str = __name__):
    return structlog.get_logger(name)

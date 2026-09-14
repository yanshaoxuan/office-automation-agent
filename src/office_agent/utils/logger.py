"""统一日志模块。

为什么不直接 print：
1. 日志有级别（DEBUG/INFO/WARNING/ERROR），生产环境可按级别过滤；
2. 带时间戳、模块名，出问题能定位是哪个节点；
3. 后续接 LangSmith / ELK 时，只需改这一个文件的 handler。

当前输出人类可读的单行格式；未来要接入日志平台时，
可在 setup_logging 中换成 python-json-logger 的 JSONFormatter，业务代码零改动。
"""
from __future__ import annotations

import logging
import sys

# 所有业务日志统一挂在 "office_agent" 命名空间下，便于整体控制级别
_LOGGER_NAMESPACE = "office_agent"
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(level: str = "INFO") -> None:
    """初始化全局日志（幂等：重复调用不会产生重复 handler）。"""
    global _configured

    root_logger = logging.getLogger(_LOGGER_NAMESPACE)
    # 非法级别字符串回退到 INFO，避免因配置笔误导致应用起不来
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(resolved_level)

    # 幂等保护：清理旧 handler，防止 uvicorn reload / 测试多次初始化时日志重复
    if _configured:
        root_logger.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    root_logger.addHandler(handler)

    # 不向 Python root logger 传播，避免被第三方库的默认配置重复打印
    root_logger.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """获取业务 logger。

    约定：各模块传 __name__，自动归入 office_agent 命名空间，
    例如 get_logger(__name__) -> "office_agent.graph.rag_graph"。
    """
    if name == _LOGGER_NAMESPACE or name.startswith(f"{_LOGGER_NAMESPACE}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_LOGGER_NAMESPACE}.{name}")

"""办公工具端口（抽象协议）：日历、邮件、文档系统等。

业务逻辑只依赖本抽象，面试演示用 Mock 适配器，
真实部署只需新增对应适配器并改配置。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class CalendarEvent:
    """创建日历事件的请求参数。"""

    title: str
    date: str  # ISO 格式 YYYY-MM-DD
    owner: str
    description: str = ""


@dataclass
class EmailRequest:
    """发送邮件的请求参数。"""

    to: str
    subject: str
    body: str
    # 可选：HTML 格式正文（优于纯文本时使用）
    html_body: str = ""


class ToolPort(ABC):
    """办公工具统一端口。当前含日历创建事件和邮件发送。

    每个方法失败时返回 False 而非抛异常（不阻断主流程）。
    """

    @abstractmethod
    def create_calendar_event(self, event: CalendarEvent) -> bool:
        """创建日历事件。返回 True 表示成功。

        实现要求：
        - 失败时返回 False 而非抛异常（不阻断主流程）；
        - 真实实现需处理认证、API 限流等。
        """
        raise NotImplementedError

    @abstractmethod
    def send_email(self, request: EmailRequest) -> bool:
        """发送邮件。返回 True 表示成功。

        真实实现需处理 SMTP 认证、附件、TLS 等。
        """
        raise NotImplementedError

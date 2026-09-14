"""Mock 办公工具适配器（日历/邮件）。

面试演示和本地开发用：不调用真实 API，只在日志中记录操作。
换真实实现时新增对应适配器文件即可。
"""
from __future__ import annotations

from office_agent.ports.tools import CalendarEvent, EmailRequest, ToolPort
from office_agent.utils.logger import get_logger

logger = get_logger(__name__)


class MockToolAdapter(ToolPort):
    """Mock 实现：所有操作仅打印日志，不产生真实副作用。"""

    def create_calendar_event(self, event: CalendarEvent) -> bool:
        logger.info(
            "[Mock] 创建日历事件：title='%s' date=%s owner='%s'",
            event.title, event.date, event.owner,
        )
        return True

    def send_email(self, request: EmailRequest) -> bool:
        logger.info(
            "[Mock] 发送邮件：to='%s' subject='%s' body=%d chars",
            request.to, request.subject, len(request.body),
        )
        return True

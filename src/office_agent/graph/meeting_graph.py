"""会议纪要抽取子图：LangGraph StateGraph 组装。

对应阶段三图 B：

    START → normalize → extract → validate → [retry_or_degrade]
                                                  ├─ retry → extract (反馈式重试)
                                                  ├─ degrade → END
                                                  └─ proceed → resolve → sync → END

条件边：
- retry_or_degrade：校验失败时决定重试（带错误反馈回边）还是降级
- resolve 后的 sync 节点内部处理高/低置信分支
"""
from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from office_agent.config import Settings, get_settings
from office_agent.graph.state import MeetingState
from office_agent.nodes.meeting import (
    degrade,
    extract,
    normalize_transcript,
    resolve_dates_and_owners,
    should_retry_or_degrade,
    validate,
)
from office_agent.ports.llm import LLMPort
from office_agent.ports.tools import CalendarEvent, ToolPort
from office_agent.schemas import ActionItem, MeetingResult
from office_agent.utils.logger import get_logger

logger = get_logger(__name__)


def _sync_node_factory(tool: ToolPort, confidence_threshold: float):
    """生成 sync 节点：对高置信待办自动创建日历事件。

    用工厂函数而非 partial，因为需要访问 threshold 和 tool 两个依赖。
    """

    def sync(state: dict[str, Any]) -> dict[str, Any]:
        result: MeetingResult | None = state.get("meeting_result")
        if result is None:
            return {"status": "done"}

        synced = 0
        for item in result.action_items:
            if (
                item.confidence >= confidence_threshold
                and item.owner is not None
                and item.due_date is not None
            ):
                event = CalendarEvent(
                    title=item.task[:50],
                    date=item.due_date,
                    owner=item.owner,
                    description=f"来自会议纪要自动抽取（置信度 {item.confidence:.0%}）",
                )
                try:
                    ok = tool.create_calendar_event(event)
                    if ok:
                        synced += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning("日历同步失败（%s）：%s", item.task[:20], exc)

        if synced:
            logger.info("自动同步 %d 条待办到日历", synced)
        else:
            logger.info("无待办满足自动同步条件（阈值 %.0f%%）", confidence_threshold)

        return {"meeting_result": result, "status": "done"}

    return sync


def build_meeting_graph(
    llm: LLMPort,
    tool: ToolPort,
    attendees: list[str],
    settings: Settings | None = None,
) -> Any:
    """构建可编译的会议纪要抽取状态图。

    attendees 作为参数注入（每次会议的参会人不同）。
    """
    settings = settings or get_settings()
    max_retries = settings.meeting.extraction_max_retries
    sync_threshold = settings.meeting.sync_confidence_threshold

    # 依赖注入：partial 绑定 llm 和 attendees
    extract_node = partial(extract, llm=llm, attendees=attendees)
    validate_node = partial(validate, attendees=attendees)
    sync_node = _sync_node_factory(tool, sync_threshold)

    graph = StateGraph(MeetingState)

    graph.add_node("normalize", normalize_transcript)
    graph.add_node("extract", extract_node)
    graph.add_node("validate", validate_node)
    graph.add_node("resolve", resolve_dates_and_owners)
    graph.add_node("sync", sync_node)
    graph.add_node("degrade", degrade)

    # 固定边
    graph.add_edge(START, "normalize")
    graph.add_edge("normalize", "extract")
    graph.add_edge("extract", "validate")

    # 条件边：校验后决定重试/降级/继续
    graph.add_conditional_edges(
        "validate",
        partial(should_retry_or_degrade, max_retries=max_retries),
        {
            "retry": "extract",
            "degrade": "degrade",
            "proceed": "resolve",
        },
    )

    # resolve → sync → END
    graph.add_edge("resolve", "sync")
    graph.add_edge("sync", END)

    # degrade → END
    graph.add_edge("degrade", END)

    compiled = graph.compile()
    logger.info(
        "Meeting 图编译完成：max_retries=%d, sync_threshold=%.2f",
        max_retries, sync_threshold,
    )
    return compiled


def run_meeting(
    transcript: str,
    attendees: list[str],
    meeting_time: str,
    llm: LLMPort,
    tool: ToolPort,
    settings: Settings | None = None,
) -> MeetingResult:
    """便捷入口：构建图并执行一次会议纪要抽取。"""
    graph = build_meeting_graph(llm, tool, attendees, settings)

    initial_state: MeetingState = {
        "transcript": transcript,
        "attendees": attendees,
        "meeting_time": meeting_time,
        "retry_count": 0,
    }

    logger.info("会议纪要抽取开始（%d 字转写，%d 参会人）", len(transcript), len(attendees))
    final_state = graph.invoke(initial_state)

    result = final_state.get("meeting_result")
    if result is None:
        return MeetingResult(
            summary="（系统错误：流程未产出结果）",
            degraded=True,
            notes="graph returned no meeting_result",
        )

    logger.info(
        "会议纪要抽取结束：degraded=%s, items=%d, decisions=%d",
        result.degraded, len(result.action_items), len(result.decisions),
    )
    return result

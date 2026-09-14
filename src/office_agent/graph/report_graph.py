"""周报生成子图：LangGraph StateGraph 组装（含 interrupt 人审断点）。

对应阶段三图 C：

    START → collect(并发) → aggregate → draft → [interrupt: human_review]
                                                    ├─ approve → send → END
                                                    ├─ revise → draft (回边)
                                                    └─ reject → discard → END

核心亮点：interrupt_before("human_review") + Checkpointer
  图执行到 human_review 节点前暂停，状态存入 Checkpointer；
  用户审阅后传入 review_decision 恢复执行。
  进程退出也不丢失状态——凭 thread_id 可恢复。
"""
from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from office_agent.config import Settings, get_settings
from office_agent.graph.state import ReportState
from office_agent.nodes.report import (
    aggregate,
    collect_calendar,
    collect_emails,
    collect_tasks,
    discard_report,
    draft_report,
    human_review,
    review_decision_router,
    send_report,
)
from office_agent.ports.llm import LLMPort
from office_agent.ports.tools import ToolPort
from office_agent.schemas import Report
from office_agent.utils.logger import get_logger

logger = get_logger(__name__)


def build_report_graph(
    llm: LLMPort,
    tool: ToolPort,
    settings: Settings | None = None,
) -> Any:
    """构建可编译的周报生成状态图（带 Checkpointer）。

    返回 CompiledGraph，调用方用 graph.invoke(state, config={"configurable": {"thread_id": ...}}) 执行。
    """
    settings = settings or get_settings()

    # 依赖注入
    draft_node = partial(draft_report, llm=llm)
    send_node = partial(send_report, tool=tool, recipient="manager@company.com")

    graph = StateGraph(ReportState)

    # 三个采集节点（在图中是三个独立节点，LangGraph 会并行执行同一层的无依赖节点）
    graph.add_node("collect_emails", collect_emails)
    graph.add_node("collect_calendar", collect_calendar)
    graph.add_node("collect_tasks", collect_tasks)
    graph.add_node("aggregate", aggregate)
    graph.add_node("draft", draft_node)
    graph.add_node("human_review", human_review)
    graph.add_node("send", send_node)
    graph.add_node("discard", discard_report)

    # 三个采集节点从 START 并发开始
    graph.add_edge(START, "collect_emails")
    graph.add_edge(START, "collect_calendar")
    graph.add_edge(START, "collect_tasks")

    # 三个采集节点都汇聚到 aggregate
    # 注意：LangGraph 的 reducer 会把三个节点的 raw_items 合并（operator.add）
    graph.add_edge("collect_emails", "aggregate")
    graph.add_edge("collect_calendar", "aggregate")
    graph.add_edge("collect_tasks", "aggregate")

    # aggregate → draft → human_review（interrupt 前）
    graph.add_edge("aggregate", "draft")
    graph.add_edge("draft", "human_review")

    # 条件边：人审决策路由
    graph.add_conditional_edges(
        "human_review",
        review_decision_router,
        {
            "send": "send",
            "revise": "draft",  # 回边：修改后重新生成草稿
            "discard": "discard",
        },
    )

    # send / discard → END
    graph.add_edge("send", END)
    graph.add_edge("discard", END)

    # 编译时配置 Checkpointer + interrupt_before human_review
    checkpointer = MemorySaver()
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],
    )
    logger.info("Report 图编译完成（含 Checkpointer + interrupt）")
    return compiled


def run_report(
    user_id: str,
    time_range_start: str,
    time_range_end: str,
    llm: LLMPort,
    tool: ToolPort,
    settings: Settings | None = None,
    thread_id: str = "default",
    graph: Any | None = None,
) -> tuple[Report, Any]:
    """便捷入口：执行周报生成第一步（到 interrupt 暂停）。

    返回 (草稿Report, graph实例)，graph 供 resume_report 使用以共享 checkpointer。
    """
    if graph is None:
        graph = build_report_graph(llm, tool, settings)

    initial_state: ReportState = {
        "user_id": user_id,
        "time_range_start": time_range_start,
        "time_range_end": time_range_end,
        "raw_items": [],
        "source_status": {},
    }

    config = {"configurable": {"thread_id": thread_id}}

    logger.info("周报生成开始：%s, %s ~ %s", user_id, time_range_start, time_range_end)
    state = graph.invoke(initial_state, config=config)

    report = state.get("report")
    if report is None:
        report = Report(
            user_id=user_id,
            period=f"{time_range_start} ~ {time_range_end}",
            notes="系统错误：流程未产出周报",
        )

    logger.info(
        "周报草稿就绪（待人审）：%d 段落, %d 事项",
        len(report.sections),
        sum(len(s.items) for s in report.sections),
    )
    return report, graph


def resume_report(
    graph: Any,
    thread_id: str = "default",
    decision: str = "approve",
    feedback: str = "",
) -> Report:
    """恢复被 interrupt 暂停的周报流程（需传入同一 graph 实例）。

    用 update_state 注入用户决策，再 invoke(None) 恢复执行。
    """
    config = {"configurable": {"thread_id": thread_id}}

    update: dict[str, Any] = {"review_decision": decision}
    if feedback:
        update["review_feedback"] = feedback

    # 先更新状态（注入用户决策），再恢复执行
    graph.update_state(values=update, config=config)
    state = graph.invoke(None, config=config)

    report = state.get("report")
    if report is None:
        return Report(notes="恢复执行后未获得结果")
    return report

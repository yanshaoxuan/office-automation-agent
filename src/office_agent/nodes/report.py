"""周报生成的图节点函数。

关键设计：
- collect_*：并发模拟拉取邮件/日历/任务系统数据（单源失败不阻断整体）
- aggregate：跨源指纹去重，同一事项只保留一条
- draft：LLM 四段式生成（本周完成/进行中/风险阻塞/下周计划）+ trace 溯源
- human_review：interrupt 节点，暂停图等待用户审阅
- send：用户批准后调用邮件工具发送
"""
from __future__ import annotations

import hashlib
from typing import Any

from office_agent.ports.llm import LLMPort
from office_agent.ports.tools import EmailRequest, ToolPort
from office_agent.schemas import DataSourceItem, Report, ReportSection
from office_agent.utils.logger import get_logger

logger = get_logger(__name__)

# 四段式周报的段落标题
SECTION_TITLES = ["本周完成", "进行中", "风险与阻塞", "下周计划"]

# 风险信号关键词：含这些词的事项自动归入"风险与阻塞"段落
RISK_KEYWORDS = {"延期", "阻塞", "风险", "卡住", "未完成", "受阻", "问题", "延误"}

_DRAFT_SYSTEM = """你是一个周报生成助手。请根据提供的本周工作事项数据，生成一份四段式周报草稿。

格式要求（JSON）：
{
  "sections": [
    {"title": "本周完成", "items": ["完成事项1", "完成事项2"]},
    {"title": "进行中", "items": ["进行中事项1"]},
    {"title": "风险与阻塞", "items": ["风险事项1"]},
    {"title": "下周计划", "items": ["计划1"]}
  ]
}

规则：
1. 只使用提供的数据，不得编造事项。
2. 含"延期""阻塞""风险"等关键词的事项归入"风险与阻塞"段落。
3. 同一事项不得在多个段落重复出现。
4. 无数据的段落填空列表。
5. 合并相似事项，保持简洁。
"""


# ---- 节点 1a/1b/1c：多源数据采集 ----


def collect_emails(state: dict[str, Any]) -> dict[str, Any]:
    """采集邮件数据源。

    在 Mock 模式下返回预设的示例邮件事项。
    真实实现会调用邮件 API（如 Microsoft Graph / Gmail API）。
    """
    user_id = state.get("user_id", "unknown")
    try:
        items = _mock_email_data(user_id)
        logger.info("邮件采集成功：%d 条事项", len(items))
        return {"raw_items": items, "source_status": {"email": "ok"}}
    except Exception as exc:  # noqa: BLE001
        logger.warning("邮件采集失败：%s", exc)
        return {"raw_items": [], "source_status": {"email": f"failed: {exc}"}}


def collect_calendar(state: dict[str, Any]) -> dict[str, Any]:
    """采集日历数据源。"""
    user_id = state.get("user_id", "unknown")
    try:
        items = _mock_calendar_data(user_id)
        logger.info("日历采集成功：%d 条事项", len(items))
        return {"raw_items": items, "source_status": {"calendar": "ok"}}
    except Exception as exc:  # noqa: BLE001
        logger.warning("日历采集失败：%s", exc)
        return {"raw_items": [], "source_status": {"calendar": f"failed: {exc}"}}


def collect_tasks(state: dict[str, Any]) -> dict[str, Any]:
    """采集任务系统数据源。"""
    user_id = state.get("user_id", "unknown")
    try:
        items = _mock_task_data(user_id)
        logger.info("任务系统采集成功：%d 条事项", len(items))
        return {"raw_items": items, "source_status": {"tasks": "ok"}}
    except Exception as exc:  # noqa: BLE001
        logger.warning("任务系统采集失败：%s", exc)
        return {"raw_items": [], "source_status": {"tasks": f"failed: {exc}"}}


# ---- 节点 2：聚合去重 ----


def aggregate(state: dict[str, Any]) -> dict[str, Any]:
    """跨源聚合 + 指纹去重。

    对应阶段一场景三验收标准 1：同一事项在多个源出现时合并为一条。
    """
    raw_items: list[DataSourceItem] = state.get("raw_items", [])
    source_status: dict = state.get("source_status", {})

    if not raw_items:
        logger.warning("无可用数据源事项")
        return {
            "aggregated_items": [],
            "status": "drafting",
            "report": Report(notes="所有数据源均无数据"),
        }

    # 指纹去重：相同 fingerprint 的事项只保留第一个出现的
    seen: set[str] = set()
    unique: list[DataSourceItem] = []
    for item in raw_items:
        fp = item.fingerprint or _make_fingerprint(item)
        if fp not in seen:
            seen.add(fp)
            unique.append(item)

    # 检查数据源可用性
    failed_sources = [
        name for name, status in source_status.items()
        if status != "ok"
    ]
    notes = ""
    if failed_sources:
        notes = f"以下数据源不可用：{', '.join(failed_sources)}，周报基于可用数据生成"

    logger.info(
        "聚合去重：%d 条原始 -> %d 条唯一（去重 %d 条）",
        len(raw_items), len(unique), len(raw_items) - len(unique),
    )
    return {
        "aggregated_items": unique,
        "status": "drafting",
        "report": Report(notes=notes) if notes else state.get("report", Report()),
    }


# ---- 节点 3：草稿生成 ----


def draft_report(
    state: dict[str, Any],
    llm: LLMPort,
) -> dict[str, Any]:
    """LLM 四段式周报草稿生成 + trace 溯源。

    对应阶段一场景三验收标准 2：每条结论必须有数据支撑，无数据不生成。
    """
    items: list[DataSourceItem] = state.get("aggregated_items", [])
    user_id = state.get("user_id", "")
    start = state.get("time_range_start", "")
    end = state.get("time_range_end", "")

    existing_report: Report | None = state.get("report")
    notes = existing_report.notes if existing_report else ""

    if not items:
        report = Report(
            user_id=user_id,
            period=f"{start} ~ {end}",
            sections=[ReportSection(title=t, items=[]) for t in SECTION_TITLES],
            trace=[],
            notes=notes or "本周无可用数据源事项",
        )
        return {"report": report, "status": "reviewing"}

    # 构建数据上下文（含来源标记，供 LLM 溯源）
    context_parts = []
    for i, item in enumerate(items, 1):
        context_parts.append(
            f"[事项{i}] 来源:{item.source} 日期:{item.date}\n"
            f"标题:{item.title}\n内容:{item.content}"
        )
    context = "\n\n".join(context_parts)

    schema = {
        "type": "object",
        "properties": {
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "items": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
        },
    }

    messages = [
        {"role": "system", "content": _DRAFT_SYSTEM},
        {"role": "user", "content": f"本周工作事项数据：\n{context}"},
    ]

    # 如果有修改批注，注入到 prompt
    feedback = state.get("review_feedback", "")
    if feedback:
        messages.append({
            "role": "user",
            "content": f"请根据以下修改意见调整周报：{feedback}",
        })

    try:
        raw = llm.chat_with_structure(messages, schema)
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM 草稿生成失败：%s", exc)
        # 降级：手工分类（用规则代替 LLM）
        raw = _rule_based_draft(items)

    # 组装 Report
    sections: list[ReportSection] = []
    raw_sections = raw.get("sections", [])
    raw_titles = {s.get("title", "") for s in raw_sections}

    # 确保四个段落都有（LLM 可能漏掉空段落）
    for title in SECTION_TITLES:
        matching = next(
            (s for s in raw_sections if s.get("title", "") == title),
            None,
        )
        items_list = matching.get("items", []) if matching else []
        sections.append(ReportSection(title=title, items=items_list))

    report = Report(
        user_id=user_id,
        period=f"{start} ~ {end}",
        sections=sections,
        trace=items,  # 每条原始事项都是溯源记录
        notes=notes,
    )

    logger.info(
        "周报草稿生成：%d 个段落，%d 条事项",
        len(sections),
        sum(len(s.items) for s in sections),
    )
    return {"report": report, "status": "reviewing"}


# ---- 节点 4：人审中断 ----


def human_review(state: dict[str, Any]) -> dict[str, Any]:
    """interrupt 节点：暂停图执行，等待用户审阅。

    在 CLI 模式下由 CLI 代码处理用户输入并恢复图执行。
    在 Web/API 模式下由 checkpointer 持久化状态，用户回来后恢复。

    本函数本身不做决策——它只是一个标记节点，
    让图编译器知道这里需要暂停（配合 interrupt_before）。
    """
    report: Report | None = state.get("report")
    if report:
        logger.info("周报草稿待审阅（%d 段落）", len(report.sections))
    return {"status": "reviewing"}


# ---- 节点 5a：发送邮件 ----


def send_report(
    state: dict[str, Any],
    tool: ToolPort,
    recipient: str = "manager@company.com",
) -> dict[str, Any]:
    """用户批准后发送周报邮件。

    对应阶段一场景三验收标准 4：必须经用户确认后才允许发送。
    """
    report: Report | None = state.get("report")
    if report is None:
        return {"status": "done"}

    # 组装邮件正文
    body = _format_report_email(report)

    try:
        ok = tool.send_email(EmailRequest(
            to=recipient,
            subject=f"周报 {report.period} - {report.user_id}",
            body=body,
        ))
        report.sent = ok
        report.approved = True
        if ok:
            logger.info("周报邮件发送成功 -> %s", recipient)
        else:
            logger.warning("周报邮件发送失败")
    except Exception as exc:  # noqa: BLE001
        logger.error("邮件发送异常：%s", exc)
        report.sent = False

    return {"report": report, "status": "done"}


# ---- 节点 5b：丢弃（用户否决） ----


def discard_report(state: dict[str, Any]) -> dict[str, Any]:
    """用户否决草稿，不发邮件。"""
    report: Report | None = state.get("report")
    if report:
        report.approved = False
        report.sent = False
    logger.info("用户否决周报草稿，未发送")
    return {"report": report, "status": "done"}


# ---- 条件边：人审决策路由 ----


def review_decision_router(state: dict[str, Any]) -> str:
    """根据用户的审阅决策路由到发送/修改/丢弃。"""
    decision = state.get("review_decision", "approve")
    if decision == "approve":
        return "send"
    elif decision == "revise":
        return "revise"
    else:
        return "discard"


# ---- 工具函数 ----


def _make_fingerprint(item: DataSourceItem) -> str:
    """生成内容指纹：标题的 SHA1。

    用于跨源去重：同一事项在不同数据源中表述可能不同，
    但标题通常一致（都叫"支付接口联调完成"）。
    只用 title 而非 title+content，因为内容往往有细节差异。
    """
    return hashlib.sha1(item.title[:80].encode("utf-8")).hexdigest()[:16]


def _rule_based_draft(items: list[DataSourceItem]) -> dict[str, Any]:
    """LLM 失败时的降级分类：用规则代替 LLM 做四段分类。

    规则：
    - 含风险关键词 → "风险与阻塞"
    - source=task 且未含"完成" → "进行中"
    - 含"完成""已" → "本周完成"
    - 其余 → "进行中"
    """
    sections = {title: [] for title in SECTION_TITLES}
    for item in items:
        text = f"{item.title} {item.content}"
        if any(kw in text for kw in RISK_KEYWORDS):
            sections["风险与阻塞"].append(item.title)
        elif "完成" in text or "已" in text:
            sections["本周完成"].append(item.title)
        else:
            sections["进行中"].append(item.title)

    return {
        "sections": [
            {"title": title, "items": items_list}
            for title, items_list in sections.items()
        ]
    }


def _format_report_email(report: Report) -> str:
    """把 Report 格式化为邮件正文文本。"""
    lines = [f"周报周期：{report.period}", f"员工：{report.user_id}", ""]
    if report.notes:
        lines.append(f"备注：{report.notes}")
        lines.append("")
    for section in report.sections:
        lines.append(f"【{section.title}】")
        if section.items:
            for item in section.items:
                lines.append(f"  - {item}")
        else:
            lines.append("  （无）")
        lines.append("")
    lines.append(f"数据溯源：{len(report.trace)} 条原始事项")
    return "\n".join(lines)


# ---- Mock 数据源 ----


def _mock_email_data(user_id: str) -> list[DataSourceItem]:
    """模拟邮件数据。"""
    return [
        DataSourceItem(
            source="email",
            title="支付接口联调完成",
            content="支付接口回调已完成联调测试，通过全部用例",
            date="2026-09-10",
        ),
        DataSourceItem(
            source="email",
            title="客户需求变更通知",
            content="客户提出新增数据导出功能需求，需评估工期",
            date="2026-09-11",
        ),
        DataSourceItem(
            source="email",
            title="设计评审会议纪要",
            content="首页改版设计评审通过，下周开始前端开发",
            date="2026-09-12",
        ),
    ]


def _mock_calendar_data(user_id: str) -> list[DataSourceItem]:
    """模拟日历数据。"""
    return [
        DataSourceItem(
            source="calendar",
            title="产品周会",
            content="参加产品周会，讨论发布计划",
            date="2026-09-09",
        ),
        DataSourceItem(
            source="calendar",
            title="支付接口联调完成",  # 与邮件重复，测试去重
            content="支付接口回调已完成联调测试",
            date="2026-09-10",
        ),
        DataSourceItem(
            source="calendar",
            title="技术分享会",
            content="参加前端技术分享会，主题：React 性能优化",
            date="2026-09-13",
        ),
    ]


def _mock_task_data(user_id: str) -> list[DataSourceItem]:
    """模拟任务系统数据。"""
    return [
        DataSourceItem(
            source="task",
            title="首页改版开发",
            content="首页改版前端开发进行中，预计下周三完成",
            date="2026-09-10",
        ),
        DataSourceItem(
            source="task",
            title="支付接口延期风险",
            content="第三方支付网关响应延迟，存在延期风险，需跟进",
            date="2026-09-11",
        ),
        DataSourceItem(
            source="task",
            title="数据看板原型",
            content="数据看板原型设计下周开始",
            date="2026-09-12",
        ),
    ]

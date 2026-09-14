"""会议纪要抽取的图节点函数。

每个节点是纯函数 (state, ...) -> partial_state。
关键设计：
- extract：让 LLM 输出 JSON（summary + decisions + action_items）
- validate：Pydantic 校验 + 责任人匹配参会人名单
- repair：把校验错误信息回灌给 LLM 重新抽取（比盲目重试有效）
- resolve：相对日期（"下周五"）→ 绝对日期（基于 meeting_time）
- degrade：重试满 2 次仍失败 → 纯文本待办 + 标记"待人工确认"
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any

import pydantic

from office_agent.ports.llm import LLMPort
from office_agent.schemas import ActionItem, MeetingResult
from office_agent.utils.logger import get_logger

logger = get_logger(__name__)

# ---- Prompt 模板 ----

_EXTRACT_SYSTEM = """你是一个会议纪要抽取助手。请从会议转写文本中提取以下信息，以 JSON 格式返回：

{
  "summary": "200字内会议摘要",
  "decisions": ["关键决议1", "关键决议2"],
  "action_items": [
    {
      "task": "任务描述",
      "owner": "责任人姓名（从参会人中匹配，无法确定时填null）",
      "due_date": "截止日期，ISO格式YYYY-MM-DD；含相对日期时原样保留",
      "confidence": 0.0到1.0的置信度,
      "evidence": "抽取该待办的原文句子"
    }
  ]
}

规则：
1. 只从转写文本中提取信息，不得编造。
2. 责任人必须从参会人名单中匹配；无法匹配时填 null。
3. 每条待办必须有 evidence（原文句子）。
4. 没有待办或决议时返回空列表。
"""

_REPAIR_SUFFIX = """

你上次的输出存在以下问题：
{errors}

请修正以上问题并重新输出完整的 JSON。"""


def normalize_transcript(state: dict[str, Any]) -> dict[str, Any]:
    """节点 1：转写文本清洗。

    做基础文本归一化，不改写语义内容：
    - 统一换行符
    - 去除行尾空白
    - 折叠多余空行
    - 在文本开头注入会议时间（供 LLM 解析相对日期时参考）
    """
    transcript = state.get("transcript", "")
    meeting_time = state.get("meeting_time", "")

    text = transcript.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    cleaned = "\n".join(lines)
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    cleaned = cleaned.strip()

    # 内容为空检查：只有会议时间头没有正文，走降级
    if not cleaned or len(cleaned) < 10:
        logger.warning("转写文本内容为空，走降级")
        return {"cleaned_transcript": "", "status": "degraded"}

    if meeting_time:
        # 注入会议时间到文本开头，让 LLM 知道"下周五"是哪天
        cleaned = f"[会议时间：{meeting_time}]\n\n{cleaned}"

    logger.info("转写清洗完成（%d chars）", len(cleaned))
    return {"cleaned_transcript": cleaned, "status": "extracting"}


def extract(
    state: dict[str, Any],
    llm: LLMPort,
    attendees: list[str],
) -> dict[str, Any]:
    """节点 2：LLM 结构化抽取。

    如果 state 中有 validation_errors（反馈式重试），
    把错误信息注入 prompt 让 LLM 修正。
    """
    transcript = state.get("cleaned_transcript", "")
    retry_count = state.get("retry_count", 0)
    validation_errors = state.get("validation_errors", "")

    if not transcript:
        logger.warning("转写文本为空，跳过抽取")
        return {"raw_extraction": {}, "status": "degraded"}

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _EXTRACT_SYSTEM},
        {
            "role": "user",
            "content": (
                f"参会人名单：{', '.join(attendees) if attendees else '无'}\n\n"
                f"转写文本：\n{transcript}"
            ),
        },
    ]

    # 反馈式重试：把上一次校验错误回灌给 LLM
    if validation_errors and retry_count > 0:
        repair_msg = _REPAIR_SUFFIX.format(errors=validation_errors)
        messages.append({"role": "assistant", "content": "我理解，让我修正。"})
        messages.append({"role": "user", "content": repair_msg.strip()})
        logger.info("第 %d 次反馈式重试（错误：%s）", retry_count, validation_errors[:50])

    # 用 JSON schema 要求结构化输出
    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "decisions": {"type": "array", "items": {"type": "string"}},
            "action_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "owner": {"type": ["string", "null"]},
                        "due_date": {"type": ["string", "null"]},
                        "confidence": {"type": "number"},
                        "evidence": {"type": "string"},
                    },
                },
            },
        },
    }

    try:
        raw = llm.chat_with_structure(messages, schema)
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM 抽取失败：%s", exc)
        return {"raw_extraction": {}, "status": "degraded"}

    logger.info(
        "LLM 抽取完成：summary=%d chars, decisions=%d, items=%d",
        len(raw.get("summary", "")),
        len(raw.get("decisions", [])),
        len(raw.get("action_items", [])),
    )
    return {"raw_extraction": raw, "status": "validating"}


def validate(
    state: dict[str, Any],
    attendees: list[str],
) -> dict[str, Any]:
    """节点 3：Pydantic 校验 + 责任人匹配。

    两层校验：
    1. Pydantic 字段类型校验（task 非空、confidence 在 0~1 等）；
    2. 责任人是否在参会人名单中（不在则置 None）。
    """
    raw = state.get("raw_extraction", {})
    retry_count = state.get("retry_count", 0)

    if not raw:
        # LLM 返回空 → 直接降级
        return {"status": "degraded", "validation_errors": "LLM 返回空结果"}

    errors: list[str] = []

    # 校验 action_items
    raw_items = raw.get("action_items", [])
    if not isinstance(raw_items, list):
        raw_items = []
        errors.append("action_items 不是列表")

    action_items: list[ActionItem] = []
    for i, item in enumerate(raw_items):
        try:
            # 责任人校验：如果 owner 不在参会人名单中，置 None（软处理，不触发重试）
            owner = item.get("owner") if isinstance(item, dict) else None
            if owner and attendees and owner not in attendees:
                matched = _fuzzy_match_owner(owner, attendees)
                if matched:
                    owner = matched
                else:
                    # 软处理：置 None + 记录 warning，不作为硬错误触发重试
                    logger.warning(
                        "action_items[%d].owner='%s' 不在参会人名单中，置 None",
                        i, owner,
                    )
                    owner = None

            ai = ActionItem(
                task=item.get("task", ""),
                owner=owner,
                due_date=item.get("due_date"),
                confidence=float(item.get("confidence", 0.5)),
                evidence=item.get("evidence", ""),
            )
            action_items.append(ai)
        except pydantic.ValidationError as exc:
            for err in exc.errors():
                loc = ".".join(str(x) for x in err["loc"])
                errors.append(f"action_items[{i}].{loc}: {err['msg']}")

    # 尝试组装完整结果（即使有部分错误也先组装，降级时仍可用）
    try:
        result = MeetingResult(
            summary=raw.get("summary", ""),
            decisions=raw.get("decisions", []) if isinstance(raw.get("decisions"), list) else [],
            action_items=action_items,
        )
    except pydantic.ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(x) for x in err["loc"])
            errors.append(f"{loc}: {err['msg']}")
        return {
            "status": "degraded" if retry_count >= 2 else "extracting",
            "validation_errors": "; ".join(errors),
            "retry_count": retry_count + 1,
        }

    if errors and retry_count < 2:
        # 有错误但还没到重试上限 → 回边重新抽取
        logger.warning("校验发现 %d 个错误，将反馈式重试", len(errors))
        return {
            "meeting_result": result,
            "validation_errors": "; ".join(errors),
            "retry_count": retry_count + 1,
            "status": "extracting",
        }

    # 校验通过或重试到达上限
    if errors:
        # 重试上限已到但仍有错误 → 降级，但保留已成功抽取的部分
        logger.warning("重试上限已到，降级处理（保留 %d 条待办）", len(action_items))
        result.degraded = True
        result.notes = f"部分字段校验失败，已保留可用结果：{'; '.join(errors[:3])}"
        return {"meeting_result": result, "status": "degraded"}

    logger.info("校验通过：%d 条待办", len(action_items))
    return {"meeting_result": result, "status": "resolving"}


def should_retry_or_degrade(state: dict[str, Any], max_retries: int) -> str:
    """条件边：校验失败后决定重试还是降级。

    返回 "retry" / "degrade" / "proceed"。
    """
    status = state.get("status", "")
    if status == "degraded":
        return "degrade"
    if status == "extracting":
        retry_count = state.get("retry_count", 0)
        if retry_count <= max_retries:
            return "retry"
        return "degrade"
    return "proceed"


def resolve_dates_and_owners(state: dict[str, Any]) -> dict[str, Any]:
    """节点 4：相对日期 → 绝对日期 + 最终责任人确认。

    把"下周五""明天""下周三"等相对日期，基于 meeting_time
    转换为 ISO 格式绝对日期。责任人已在 validate 中匹配过，
    这里只做最终确认。
    """
    result: MeetingResult | None = state.get("meeting_result")
    meeting_time_str = state.get("meeting_time", "")

    if result is None:
        return {"status": "degraded"}

    if not meeting_time_str:
        # 无会议时间无法解析相对日期，但结果仍可用
        logger.warning("无会议时间，跳过日期解析")
        return {"status": "done"}

    try:
        meeting_dt = datetime.fromisoformat(meeting_time_str)
    except (ValueError, TypeError):
        logger.warning("会议时间格式无效：%s", meeting_time_str)
        return {"status": "done"}

    for item in result.action_items:
        if item.due_date:
            resolved = _resolve_relative_date(item.due_date, meeting_dt)
            if resolved:
                item.due_date = resolved

    logger.info("日期解析完成")
    return {"meeting_result": result, "status": "done"}


def degrade(state: dict[str, Any]) -> dict[str, Any]:
    """节点 5a：降级为纯文本待办 + 标记待人工确认。

    重试满 2 次仍失败时走此节点。
    保留已成功抽取的部分（可能有几条是对的），其余标记待确认。
    """
    result: MeetingResult | None = state.get("meeting_result")
    errors = state.get("validation_errors", "")

    if result is None:
        result = MeetingResult(
            summary="（抽取失败，请人工整理）",
            degraded=True,
            notes=f"LLM 结构化抽取失败：{errors}",
        )
    else:
        result.degraded = True
        if not result.notes:
            result.notes = f"重试上限后降级：{errors}"

    # 对 owner 为 None 的待办标注"待确认"
    for item in result.action_items:
        if item.owner is None:
            item.task = f"{item.task} [责任人待确认]"

    logger.warning("降级处理：%d 条待办（其中部分待人工确认）", len(result.action_items))
    return {"meeting_result": result, "status": "done"}


# ---- 工具函数 ----


def _fuzzy_match_owner(owner: str, attendees: list[str]) -> str | None:
    """模糊匹配责任人。

    支持的匹配模式：
    - 完全匹配或一方包含另一方
    - "小X" / "老X" → 姓为 X 的参会人（中文常见称呼）
    """
    for attendee in attendees:
        if owner in attendee or attendee in owner:
            return attendee

    # 中文称呼：小张 → 姓"张"的参会人；老李 → 姓"李"的参会人
    if len(owner) >= 2 and owner[0] in ("小", "老"):
        surname = owner[1]
        for attendee in attendees:
            if attendee and attendee[0] == surname:
                return attendee

    return None


def _resolve_relative_date(date_str: str, meeting_dt: datetime) -> str | None:
    """把相对日期解析为 ISO 绝对日期。

    支持的格式：
    - "下周五" → 下一个周五
    - "明天" → meeting_dt + 1
    - "后天" → meeting_dt + 2
    - "本周五" → 本周的周五
    - "下周三" → 下一个周三
    - 已经是 YYYY-MM-DD → 直接返回
    """
    date_str = date_str.strip()

    # 已经是 ISO 格式
    try:
        datetime.fromisoformat(date_str)
        return date_str
    except (ValueError, TypeError):
        pass

    # 中文相对日期解析
    weekday_map = {
        "一": 0, "二": 1, "三": 2, "四": 3,
        "五": 4, "六": 5, "日": 6, "天": 6,
    }

    # "下周X"：下周的第 X 天（基于日历周，非滚动 7 天）
    # 先算本周一，再加 7 天到下周一，再加 target_wd 天
    match = re.match(r"下周([一二三四五六日天])", date_str)
    if match:
        target_wd = weekday_map[match.group(1)]
        this_monday = meeting_dt - timedelta(days=meeting_dt.weekday())
        next_weeks_x = this_monday + timedelta(days=7 + target_wd)
        return next_weeks_x.date().isoformat()

    # "本周X"：本周的第 X 天（可能已过去）
    match = re.match(r"本周([一二三四五六日天])", date_str)
    if match:
        target_wd = weekday_map[match.group(1)]
        this_monday = meeting_dt - timedelta(days=meeting_dt.weekday())
        this_weeks_x = this_monday + timedelta(days=target_wd)
        return this_weeks_x.date().isoformat()

    # "明天"
    if "明天" in date_str:
        return (meeting_dt + timedelta(days=1)).date().isoformat()

    # "后天"
    if "后天" in date_str:
        return (meeting_dt + timedelta(days=2)).date().isoformat()

    # "今天" / "今日"
    if "今天" in date_str or "今日" in date_str:
        return meeting_dt.date().isoformat()

    logger.warning("无法解析的日期格式：%s", date_str)
    return None

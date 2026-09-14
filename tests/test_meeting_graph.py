"""会议纪要抽取子图单元测试。

策略：用 MockLLMAdapter + MockToolAdapter 跑整张图，
验证反馈式重试、责任人匹配、相对日期解析、降级等关键逻辑。
"""
from __future__ import annotations

from datetime import datetime

import pytest

from office_agent.adapters.mock_llm import MockLLMAdapter
from office_agent.adapters.mock_tools import MockToolAdapter
from office_agent.graph.meeting_graph import build_meeting_graph, run_meeting
from office_agent.nodes.meeting import _fuzzy_match_owner, _resolve_relative_date
from office_agent.config import Settings

SAMPLE_TRANSCRIPT = """
张三：好，那我们今天开个短会。支付接口还差最后一个回调，预计明天能搞定。
王五：没问题，明天之前一定搞完。
张三：李四，你的设计稿需要交付给前端团队，下周一之前完成交接。
另外小赵不在今天这个会，但是之前说过的那个数据看板的需求，需要他下周五之前出一个原型。
产品发布日期定在10月15号，所有准备工作在10月10号之前完成。
"""

ATTENDEES = ["张三", "李四", "王五"]


def test_relative_date_tomorrow():
    """相对日期解析：明天 → 会议日期 + 1 天。"""
    meeting = datetime(2026, 9, 14, 10, 0, 0)
    result = _resolve_relative_date("明天", meeting)
    assert result == "2026-09-15"


def test_relative_date_next_monday():
    """相对日期解析：下周一（9/14 是周一，下周一是 9/21）。"""
    meeting = datetime(2026, 9, 14, 10, 0, 0)  # 周一
    result = _resolve_relative_date("下周一", meeting)
    assert result == "2026-09-21"


def test_relative_date_next_friday():
    """相对日期解析：下周五（9/14 周一，下周五是 9/25）。"""
    meeting = datetime(2026, 9, 14, 10, 0, 0)  # 周一
    result = _resolve_relative_date("下周五", meeting)
    assert result == "2026-09-25"


def test_relative_date_iso_passthrough():
    """已经是 ISO 格式的日期直接返回。"""
    result = _resolve_relative_date("2026-10-15", datetime(2026, 9, 14))
    assert result == "2026-10-15"


def test_fuzzy_match_owner():
    """模糊匹配：小张 → 张三。"""
    assert _fuzzy_match_owner("小张", ["张三", "李四"]) == "张三"
    assert _fuzzy_match_owner("王五", ["张三", "李四", "王五"]) == "王五"
    assert _fuzzy_match_owner("赵六", ["张三", "李四"]) is None


def test_meeting_extraction_with_mock_llm():
    """端到端 Mock：抽取待办 + 责任人匹配 + 日期解析 + 日历同步。"""
    llm = MockLLMAdapter()
    tool = MockToolAdapter()
    settings = Settings()

    result = run_meeting(
        transcript=SAMPLE_TRANSCRIPT,
        attendees=ATTENDEES,
        meeting_time="2026-09-14T10:00:00",
        llm=llm,
        tool=tool,
        settings=settings,
    )

    # 应抽取至少 3 条待办
    assert len(result.action_items) >= 3

    # 责任人校验：王五在名单中 → 保留；小赵不在 → 置 None
    owners = {ai.task: ai.owner for ai in result.action_items}
    assert owners.get("完成支付接口回调") == "王五"
    assert owners.get("设计稿交付前端团队") == "李四"
    # 小赵不在名单中 → None
    assert owners.get("数据看板原型设计") is None

    # 日期解析：明天 → 2026-09-15
    dates = {ai.task: ai.due_date for ai in result.action_items}
    assert dates.get("完成支付接口回调") == "2026-09-15"
    # 下周一 → 2026-09-21
    assert dates.get("设计稿交付前端团队") == "2026-09-21"

    # 有决策
    assert len(result.decisions) >= 1

    # 未降级
    assert result.degraded is False


def test_meeting_extraction_marks_unmatched_owner():
    """责任人不在参会人名单时标记待确认（不崩溃）。"""
    llm = MockLLMAdapter()
    tool = MockToolAdapter()
    # 参会人只有张三，小赵和王五都不在
    result = run_meeting(
        transcript=SAMPLE_TRANSCRIPT,
        attendees=["张三"],
        meeting_time="2026-09-14T10:00:00",
        llm=llm,
        tool=tool,
        settings=Settings(),
    )

    # 王五不在名单 → owner 置 None
    owners = {ai.task: ai.owner for ai in result.action_items}
    assert owners.get("完成支付接口回调") is None
    # 数据看板的 owner 是小赵 → 也不在名单 → None
    assert owners.get("数据看板原型设计") is None


def test_meeting_extraction_empty_transcript():
    """空转写文本应走降级路径，不崩溃。"""
    llm = MockLLMAdapter()
    tool = MockToolAdapter()

    result = run_meeting(
        transcript="",
        attendees=ATTENDEES,
        meeting_time="2026-09-14T10:00:00",
        llm=llm,
        tool=tool,
        settings=Settings(),
    )

    assert result.degraded is True
    assert len(result.action_items) == 0

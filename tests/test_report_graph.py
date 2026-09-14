"""周报生成子图单元测试。

策略：用 MockLLM + MockTool 跑整张图，
验证多源去重、草稿生成、interrupt 人审、发送/丢弃/修改分支。
"""
from __future__ import annotations

import pytest

from office_agent.adapters.mock_llm import MockLLMAdapter
from office_agent.adapters.mock_tools import MockToolAdapter
from office_agent.config import Settings
from office_agent.graph.report_graph import build_report_graph, run_report, resume_report
from office_agent.nodes.report import _make_fingerprint, _rule_based_draft
from office_agent.schemas import DataSourceItem


def _make_item(source="email", title="测试", content="内容", date="2026-09-10"):
    return DataSourceItem(source=source, title=title, content=content, date=date)


# ---- 工具函数 ----


def test_fingerprint_same_content_same_hash():
    """相同标题+内容的不同数据源应生成相同指纹（去重的基础）。"""
    item1 = _make_item(source="email", title="支付接口完成", content="联调通过")
    item2 = _make_item(source="calendar", title="支付接口完成", content="联调通过")
    assert _make_fingerprint(item1) == _make_fingerprint(item2)


def test_fingerprint_different_content_different_hash():
    item1 = _make_item(title="任务A", content="内容A")
    item2 = _make_item(title="任务B", content="内容B")
    assert _make_fingerprint(item1) != _make_fingerprint(item2)


def test_rule_based_draft_risk_keywords():
    """降级分类：含风险关键词的事项归入"风险与阻塞"。"""
    items = [
        _make_item(title="支付接口延期", content="第三方延迟，存在延期风险"),
        _make_item(title="完成设计", content="设计稿已完成"),
    ]
    result = _rule_based_draft(items)
    sections = {s["title"]: s["items"] for s in result["sections"]}
    assert "支付接口延期" in sections["风险与阻塞"]
    assert "完成设计" in sections["本周完成"]


# ---- 端到端 ----


def test_report_generates_draft_with_mock_llm():
    """端到端：Mock 采集 -> 去重 -> 草稿生成（停在 interrupt 前）。"""
    llm = MockLLMAdapter()
    tool = MockToolAdapter()

    report, graph = run_report(
        user_id="张三",
        time_range_start="2026-09-08",
        time_range_end="2026-09-12",
        llm=llm,
        tool=tool,
        settings=Settings(),
        thread_id="test-1",
    )

    # 应有 4 个段落
    assert len(report.sections) == 4
    titles = [s.title for s in report.sections]
    assert "本周完成" in titles
    assert "进行中" in titles
    assert "风险与阻塞" in titles
    assert "下周计划" in titles

    # 应有去重后的溯源记录（原始 9 条去重后应少于 9）
    assert len(report.trace) < 9
    assert len(report.trace) > 0

    # 草稿未发送（需人审）
    assert report.sent is False
    assert report.approved is False


def test_report_resume_approve_sends_email():
    """人审批准后恢复执行 -> 发送邮件。"""
    llm = MockLLMAdapter()
    tool = MockToolAdapter()
    thread_id = "test-approve"

    # 第一步：生成草稿（共享 graph 实例）
    _, graph = run_report(
        user_id="张三",
        time_range_start="2026-09-08",
        time_range_end="2026-09-12",
        llm=llm,
        tool=tool,
        settings=Settings(),
        thread_id=thread_id,
    )

    # 第二步：批准
    final = resume_report(graph=graph, thread_id=thread_id, decision="approve")
    assert final.sent is True
    assert final.approved is True


def test_report_resume_reject_no_email():
    """人审否决 -> 不发送。"""
    llm = MockLLMAdapter()
    tool = MockToolAdapter()
    thread_id = "test-reject"

    _, graph = run_report(
        user_id="张三",
        time_range_start="2026-09-08",
        time_range_end="2026-09-12",
        llm=llm, tool=tool, settings=Settings(),
        thread_id=thread_id,
    )

    final = resume_report(graph=graph, thread_id=thread_id, decision="reject")
    assert final.sent is False
    assert final.approved is False


def test_report_dedup_merges_cross_source():
    """同一事项在邮件和日历中同时出现 -> 去重后只保留一条。"""
    llm = MockLLMAdapter()
    tool = MockToolAdapter()

    report, _ = run_report(
        user_id="张三",
        time_range_start="2026-09-08",
        time_range_end="2026-09-12",
        llm=llm, tool=tool, settings=Settings(),
        thread_id="test-dedup",
    )

    # "支付接口联调完成" 在 email 和 calendar 中都有 -> 去重后只 1 条
    trace_titles = [item.title for item in report.trace]
    assert trace_titles.count("支付接口联调完成") == 1

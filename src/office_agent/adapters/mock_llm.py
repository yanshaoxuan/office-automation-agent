"""Mock LLM 适配器（无 API Key 时的降级方案）。

用途：
- 本地开发/CI 中跑通整张 RAG 图，不需要真实 LLM；
- 面试演示时如果网络不稳定，可切换到此适配器展示流程。

策略：根据输入关键词做简单的模板匹配，返回带"来源"标注的假答案。
显然不是真的智能，但足以验证"图的结构、条件边、回边"是否正确运转。
"""
from __future__ import annotations

import json
import re
from typing import Any

from office_agent.ports.llm import LLMPort
from office_agent.utils.logger import get_logger

logger = get_logger(__name__)


class MockLLMAdapter(LLMPort):
    """基于关键词匹配的假 LLM，用于无网络/无 Key 场景。"""

    # 简单的知识库：关键词 -> 答案模板
    _QA: list[tuple[str, str]] = [
        ("年假|年休假", "根据《员工考勤与假期管理办法》第二章，工作满1年不满10年的年休假为5天，满10年不满20年为10天，满20年为15天。"),
        ("差旅|出差|报销", "根据《差旅费用报销管理制度》，出差须提前在OA系统提交申请单，经审批后出行。"),
        ("密码|信息安全", "根据《信息安全常见问题FAQ》，密码长度不少于12位，须包含大写、小写、数字和特殊符号中的至少三种。"),
    ]

    _REJECT_PHRASES = ("蛋糕", "做饭", "天气", "股票", "游戏")

    def chat(self, messages: list[dict[str, str]]) -> str:
        # 取最后一条 user 消息
        user_msg = ""
        for m in reversed(messages):
            if m["role"] == "user":
                user_msg = m["content"]
                break

        logger.debug("MockLLM 收到查询（%d chars）", len(user_msg))

        # 无关问题模拟拒答
        for phrase in self._REJECT_PHRASES:
            if phrase in user_msg:
                return "未在知识库中找到与该问题相关的依据，无法回答。"

        # 关键词匹配
        for keywords, answer in self._QA:
            if any(k in user_msg for k in keywords.split("|")):
                return answer

        # 默认：返回一个含糊的答案（模拟低置信度场景）
        return "未在知识库中找到与该问题相关的依据，无法回答。"

    def chat_with_structure(
        self, messages: list[dict[str, str]], schema: dict[str, Any]
    ) -> dict[str, Any]:
        """会议抽取场景：根据转写文本关键词返回结构化 JSON。

        Mock 策略：扫描 user 消息中的关键词，返回预设的 JSON 结构。
        """
        # 拼接所有 user 消息内容
        user_content = " ".join(
            m["content"] for m in messages if m["role"] == "user"
        )

        # 判断是否是会议抽取场景（包含"参会人""转写文本"等关键词）
        if "转写文本" in user_content or "参会人" in user_content:
            return self._mock_meeting_extraction(user_content)

        # 判断是否是周报草稿场景（包含"本周工作事项数据"等关键词）
        if "本周工作事项" in user_content or "周报草稿" in user_content:
            return self._mock_report_draft(user_content)

        # 默认降级
        text = self.chat(messages)
        return {"answer": text, "sources": [], "confidence": 0.0}

    def _mock_meeting_extraction(self, content: str) -> dict[str, Any]:
        """模拟会议纪要抽取结果。"""
        # 检测参会人和待办关键词
        action_items: list[dict[str, Any]] = []

        if "支付接口" in content:
            action_items.append({
                "task": "完成支付接口回调",
                "owner": "王五",
                "due_date": "明天",
                "confidence": 0.9,
                "evidence": "支付接口还差最后一个回调，预计明天能搞定",
            })

        if "设计稿" in content and "前端" in content:
            action_items.append({
                "task": "设计稿交付前端团队",
                "owner": "李四",
                "due_date": "下周一",
                "confidence": 0.85,
                "evidence": "你的设计稿需要交付给前端团队，下周一之前完成交接",
            })

        if "数据看板" in content or "小赵" in content:
            action_items.append({
                "task": "数据看板原型设计",
                "owner": "小赵",
                "due_date": "下周五",
                "confidence": 0.75,
                "evidence": "之前说过的那个数据看板的需求，需要他下周五之前出一个原型",
            })

        decisions: list[str] = []
        if "10 月 15" in content or "10月15" in content:
            decisions.append("产品发布日期定在10月15号")
        if "10 月 10" in content or "10月10" in content:
            decisions.append("所有准备工作在10月10号之前完成")

        return {
            "summary": "本次会议对齐了产品发布前的准备工作，包括设计稿交付、支付接口开发和数据看板原型设计。",
            "decisions": decisions,
            "action_items": action_items,
        }

    def _mock_report_draft(self, content: str) -> dict[str, Any]:
        """模拟周报草稿生成：用规则分类代替 LLM。"""
        from office_agent.nodes.report import RISK_KEYWORDS, SECTION_TITLES

        sections = {title: [] for title in SECTION_TITLES}

        # 解析 [事项N] 格式的上下文
        import re
        items = re.findall(r"\[事项\d+\].*?标题:(.+?)(?:\n|$)", content)

        for title in items:
            title = title.strip()
            if any(kw in title for kw in RISK_KEYWORDS):
                sections["风险与阻塞"].append(title)
            elif "完成" in title or "已" in title:
                sections["本周完成"].append(title)
            elif "下周" in title or "计划" in title:
                sections["下周计划"].append(title)
            else:
                sections["进行中"].append(title)

        return {
            "sections": [
                {"title": t, "items": items_list}
                for t, items_list in sections.items()
            ]
        }

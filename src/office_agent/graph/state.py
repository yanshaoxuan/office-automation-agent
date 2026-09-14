"""RAG 图的共享状态定义。

用 TypedDict 而非 Pydantic BaseModel：
- LangGraph 原生支持 TypedDict，节点入参以 dict 形式传入；
- 避免 "BaseModel 实例 vs dict" 的类型漂移坑（见 LangGraph 已知行为）；
- 字段类型用 Annotated + operator.add 声明 reducer，支持并发写入合并。
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from office_agent.schemas import (
    ActionItem,
    DataSourceItem,
    MeetingResult,
    RAGAnswer,
    Report,
    RetrievedChunk,
)

class RAGState(TypedDict, total=False):
    """RAG 问答子图的共享状态。

    total=False：所有字段都是可选的，因为不同节点只写自己负责的字段。
    """

    # ---- 输入 ----
    # 用户原始问题
    question: str
    # 多轮对话历史（[{"role": "user"|"assistant", "content": "..."}]）
    chat_history: list[dict[str, str]]
    # 可选的检索范围过滤
    doc_scope: dict | None

    # ---- 中间产物 ----
    # 改写后的检索查询（指代消解后）
    rewritten_query: str
    # 检索命中的切片
    retrieved_chunks: list[RetrievedChunk]
    # 检索最高分（用于置信度守卫判断）
    max_score: float

    # ---- 输出 ----
    # 最终答案（RAGAnswer 对象）
    answer: RAGAnswer
    # 引用校验重试计数器（防止死循环）
    retry_count: int
    # 流程状态标记（供条件边判断）
    status: str  # "searching" | "rejected" | "generating" | "done"


class MeetingState(TypedDict, total=False):
    """会议纪要抽取子图的共享状态。

    对应阶段三图 B 的流程：
    normalize → extract → validate → [retry|degrade|resolve] → [sync|manual] → END
    """

    # ---- 输入 ----
    # 会议转写文本（ASR 输出或速记）
    transcript: str
    # 参会人名单（用于责任人校验）
    attendees: list[str]
    # 会议时间（ISO 格式，用于解析"下周五"等相对日期）
    meeting_time: str

    # ---- 中间产物 ----
    # 清洗后的转写文本
    cleaned_transcript: str
    # LLM 原始抽取结果（JSON dict）
    raw_extraction: dict
    # Pydantic 校验后的结构化结果
    meeting_result: MeetingResult
    # 结构化抽取重试计数
    retry_count: int
    # 上一次校验的错误信息（用于反馈式重试）
    validation_errors: str
    # 流程状态标记
    status: str  # "extracting" | "validating" | "degraded" | "resolving" | "done"


class ReportState(TypedDict, total=False):
    """周报生成子图的共享状态。

    对应阶段三图 C 的流程：
    collect → aggregate → draft → [interrupt 人审]
                                    ├─ approved → send → END
                                    ├─ revised → draft (回边)
                                    └─ rejected → END
    """

    # ---- 输入 ----
    # 员工标识
    user_id: str
    # 周报周期（起止日期 ISO 格式）
    time_range_start: str
    time_range_end: str

    # ---- 中间产物 ----
    # 各数据源原始事项（并发采集结果合并）
    raw_items: Annotated[list[DataSourceItem], operator.add]
    # 去重聚合后的事项
    aggregated_items: list[DataSourceItem]
    # 数据源可用性标记（三个采集节点并发写入，用 dict 合并 reducer）
    source_status: Annotated[dict, operator.or_]

    # ---- 输出 ----
    # 周报对象
    report: Report
    # 人审决策：approve / revise / reject
    review_decision: str
    # 修改批注（用户要求修改时的反馈）
    review_feedback: str
    # 流程状态
    status: str  # "collecting" | "aggregating" | "drafting" | "reviewing" | "sending" | "done"

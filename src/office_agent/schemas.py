"""全局数据模型（阶段一 I/O 契约的代码化）。

所有跨层传递的数据结构统一定义在这里：
- 索引流水线使用 RawDocument / DocumentChunk
- RAG 问答使用 RetrievedChunk / Source / Answer（后续模块补充）
- 会议、周报场景的模型在对应模块补充

集中定义的好处：节点之间、适配器之间对同一份"契约"编程，
字段变更时类型检查器能立刻指出所有受影响位置。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


# 向量库元数据允许携带的字段（与 DocumentChunk 的非向量字段一一对应）。
# 单独列出是因为 Chroma 的 metadata 只接受标量（str/int/float/bool），
# 不能把 None 写进去——适配器层据此做字段过滤。
CHUNK_METADATA_FIELDS = (
    "source_file",
    "section",
    "department",
    "doc_type",
    "chunk_index",
)


class RawDocument(BaseModel):
    """加载后的原始文档：整篇、未切片。"""

    source_file: str = Field(..., description="文件名或相对路径，用作溯源主键")
    doc_type: str = Field("general", description="文档类别，如 hr_policy/it_faq")
    department: str | None = Field(None, description="归属部门，用于 doc_scope 过滤")
    title: str | None = Field(None, description="文档标题（可来自 front matter）")
    text: str = Field(..., min_length=1, description="文档全文（已做基础清洗）")


class DocumentChunk(BaseModel):
    """切片后的最小检索单元，元数据用于过滤与溯源。"""

    id: str = Field(..., description="确定性 ID：内容哈希派生，保证重建索引幂等")
    content: str = Field(..., min_length=1)
    source_file: str
    # 章节路径，如 "假期制度 > 年假"，让模型/用户知道答案的上下文位置
    section: str = ""
    department: str | None = None
    doc_type: str = "general"
    chunk_index: int = Field(0, ge=0, description="该切片在所属文档内的序号")
    # 索引阶段写入，纯切片流转时可为 None
    embedding: list[float] | None = Field(None, exclude=True)


class RetrievedChunk(BaseModel):
    """检索命中的切片及其相似度分数。"""

    chunk: DocumentChunk
    # 归一化到 [0,1] 的余弦相似度：1 - chroma_distance
    score: float = Field(..., ge=0.0, le=1.0)


class Source(BaseModel):
    """答案引用来源，对应阶段一 I/O 契约中的 sources 字段。"""

    source_file: str = Field(..., description="来源文档名")
    section: str = Field("", description="章节路径")
    content: str = Field(..., description="被引用的原文片段")


class RAGAnswer(BaseModel):
    """RAG 问答的完整输出契约。"""

    answer: str = Field(..., description="自然语言答案")
    sources: list[Source] = Field(default_factory=list, description="引用来源列表")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="检索置信度")
    # 拒答标记：True 表示知识库中无相关内容，answer 是兜底话术
    rejected: bool = Field(False, description="是否因低置信度而拒答")
    reason: str = Field("", description="拒答原因或生成说明")


# ===================== 会议纪要场景 =====================


class ActionItem(BaseModel):
    """从会议中抽取的单条待办事项。

    对应阶段一场景二 I/O 契约的 action_items 字段。
    """

    task: str = Field(..., min_length=1, description="任务描述")
    owner: str | None = Field(
        None,
        description="责任人；无法匹配参会人时为 None，标记'待确认'",
    )
    # ISO 格式字符串（如 "2026-09-19"），解析失败时为 None
    due_date: str | None = Field(
        None,
        description="截止日期（ISO 格式）；含相对日期时由日期解析节点转换为绝对日期",
    )
    # LLM 自评置信度 0~1，低于阈值不自动同步日历
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    # 原文证据：该待办出自转写文本的哪句话（溯源防幻觉）
    evidence: str = Field("", description="抽取该待办的原文句子")


class MeetingResult(BaseModel):
    """会议纪要完整输出。"""

    summary: str = Field("", description="200字内会议摘要")
    decisions: list[str] = Field(default_factory=list, description="关键决议列表")
    action_items: list[ActionItem] = Field(
        default_factory=list, description="待办事项列表"
    )
    # 降级标记：True 表示结构化抽取失败，action_items 是纯文本降级结果
    degraded: bool = Field(False, description="是否降级为纯文本模式")
    notes: str = Field("", description="降级原因或处理说明")


# ===================== 周报场景 =====================


class DataSourceItem(BaseModel):
    """周报数据源中的一条原始事项（来自邮件/日历/任务系统）。"""

    source: str = Field(..., description="数据来源：email/calendar/task/doc")
    title: str = Field("", description="事项标题或摘要")
    content: str = Field("", description="事项内容")
    date: str = Field("", description="事项日期 ISO 格式")
    # 用于去重的指纹：相同 fingerprint 的事项合并为一条
    fingerprint: str = Field("", description="内容指纹，用于跨源去重")


class ReportSection(BaseModel):
    """周报的一个段落。"""

    title: str = Field(..., description="段落标题，如'本周完成'/'进行中'/'风险'/'下周计划'")
    items: list[str] = Field(default_factory=list, description="该段落的事项列表")


class Report(BaseModel):
    """周报完整输出。"""

    user_id: str = Field("", description="员工标识")
    period: str = Field("", description="周报周期，如 2026-09-08 ~ 2026-09-14")
    sections: list[ReportSection] = Field(
        default_factory=list, description="四段式周报内容"
    )
    # 每条结论的溯源：事项内容 -> 来源类型 + 原始条目索引
    trace: list[DataSourceItem] = Field(
        default_factory=list, description="数据溯源记录"
    )
    # 人审状态
    approved: bool = Field(False, description="是否经人工审阅批准")
    sent: bool = Field(False, description="是否已通过邮件发送")
    notes: str = Field("", description="处理说明或缺失数据源标注")

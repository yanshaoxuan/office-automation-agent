"""RAG 图节点函数：查询改写、检索、生成、引用校验。

每个节点是纯函数 (state) -> partial_state：
- 只读写自己负责的 state 字段；
- 无副作用（不修改全局状态）；
- 可独立单测（注入 mock LLM/embedding/store）。
"""
from __future__ import annotations

import json
from typing import Any

from office_agent.nodes.retrieval import retrieve_chunks
from office_agent.ports.embeddings import EmbeddingPort
from office_agent.ports.llm import LLMPort
from office_agent.ports.vectorstore import VectorStorePort
from office_agent.schemas import RAGAnswer, RetrievedChunk, Source
from office_agent.utils.logger import get_logger

logger = get_logger(__name__)

# 生成带引用的答案时的系统 prompt
_ANSWER_SYSTEM_PROMPT = """你是一个企业知识库问答助手。请严格根据提供的上下文片段回答用户问题。

规则：
1. 只使用下方"参考片段"中的信息回答问题，禁止使用外部知识或编造内容。
2. 回答末尾必须标注引用来源，格式为：[来源: 文件名 - 章节]
3. 如果参考片段不足以回答问题，回复"根据现有知识库内容，无法回答该问题"。
4. 回答简洁、准确、有条理。
"""

_REWRITE_SYSTEM_PROMPT = """你是一个查询改写助手。根据对话历史，把用户的最新问题改写为一个独立的、可用于向量检索的完整查询。

要求：
1. 消解指代词（"那个""它""试用期"等需替换为具体所指）。
2. 保留原意，不增加或改变用户意图。
3. 只输出改写后的查询，不要输出其他内容。
"""


def rewrite_query(
    state: dict[str, Any],
    llm: LLMPort,
) -> dict[str, Any]:
    """节点 1：结合对话历史改写查询（指代消解）。

    对应阶段一场景一验收标准 3：多轮对话中"那""它"等指代要被消解。
    """
    question = state.get("question", "")
    history = state.get("chat_history", [])

    if not history:
        # 无历史直接用原问题
        logger.info("无对话历史，跳过改写：%s", question[:50])
        return {"rewritten_query": question, "status": "searching"}

    messages = [
        {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
    ]
    # 注入最近 4 轮对话作为上下文
    for msg in history[-8:]:
        messages.append(msg)
    messages.append({"role": "user", "content": f"请改写这个问题：{question}"})

    try:
        rewritten = llm.chat(messages).strip()
        logger.info("查询改写：%s -> %s", question[:30], rewritten[:30])
    except Exception as exc:  # noqa: BLE001
        logger.warning("查询改写失败，回退到原问题：%s", exc)
        rewritten = question

    return {"rewritten_query": rewritten, "status": "searching"}


def retrieve(
    state: dict[str, Any],
    embedder: EmbeddingPort,
    store: VectorStorePort,
    top_k: int,
) -> dict[str, Any]:
    """节点 2：向量检索。

    延迟到 retrieval.py 的 retrieve_chunks 函数，避免本文件过长。
    """
    query = state.get("rewritten_query") or state.get("question", "")
    doc_scope = state.get("doc_scope")

    chunks = retrieve_chunks(query, embedder, store, top_k, doc_scope)

    if not chunks:
        logger.info("检索结果为空")
        return {
            "retrieved_chunks": [],
            "max_score": 0.0,
            "status": "rejected",
        }

    max_score = chunks[0].score  # 已按降序排列
    logger.info("检索完成：%d 条，最高分 %.3f", len(chunks), max_score)
    return {
        "retrieved_chunks": chunks,
        "max_score": max_score,
        "status": "searching",  # 守卫边会决定下一步
    }


def should_reject(state: dict[str, Any], score_threshold: float) -> str:
    """条件边函数：置信度守卫。

    对应阶段一场景一验收标准 4：相似度全部低于阈值时不生成，直接拒答。
    """
    max_score = state.get("max_score", 0.0)
    if max_score < score_threshold:
        logger.info(
            "置信度守卫触发：max_score=%.3f < 阈值 %.3f，拒答",
            max_score, score_threshold,
        )
        return "reject"
    return "generate"


def generate_answer(
    state: dict[str, Any],
    llm: LLMPort,
) -> dict[str, Any]:
    """节点 3：带引用的答案生成。

    把检索到的片段拼入 prompt，要求 LLM 只用上下文回答并标注来源。
    """
    question = state.get("question", "")
    chunks: list[RetrievedChunk] = state.get("retrieved_chunks", [])

    if not chunks:
        return _build_reject("检索结果为空，无法生成答案")

    # 构建上下文片段（带编号，便于 LLM 引用）
    context_parts = []
    sources: list[Source] = []
    for i, rc in enumerate(chunks, 1):
        chunk = rc.chunk
        context_parts.append(
            f"[片段{i}] 来源: {chunk.source_file} - {chunk.section}\n"
            f"内容: {chunk.content}"
        )
        sources.append(Source(
            source_file=chunk.source_file,
            section=chunk.section,
            content=chunk.content[:200],
        ))
    context = "\n\n".join(context_parts)

    user_prompt = (
        f"参考片段：\n{context}\n\n"
        f"用户问题：{question}\n\n"
        f"请根据以上参考片段回答问题，并在答案末尾标注来源。"
    )

    try:
        answer_text = llm.chat([
            {"role": "system", "content": _ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
    except Exception as exc:  # noqa: BLE001
        logger.error("答案生成失败：%s", exc)
        return _build_reject(f"LLM 生成失败：{exc}")

    rag_answer = RAGAnswer(
        answer=answer_text,
        sources=sources,
        confidence=state.get("max_score", 0.0),
        rejected=False,
        reason="生成成功",
    )
    logger.info("答案生成完成（%d chars，%d 个来源）", len(answer_text), len(sources))
    return {"answer": rag_answer, "status": "generating"}


def check_citation(state: dict[str, Any], max_retries: int = 1) -> str:
    """条件边函数：引用校验。

    对应阶段一场景一验收标准 1：答案必须带出处。
    如果 sources 为空且重试次数未超限，回边重新生成。
    """
    answer = state.get("answer")
    retry_count = state.get("retry_count", 0)

    if answer is None:
        return "reject"

    if answer.rejected:
        return "end"

    # 检查 sources 是否非空
    if not answer.sources and retry_count < max_retries:
        logger.warning("引用校验失败：sources 为空，第 %d 次重试", retry_count + 1)
        return "retry"

    return "end"


def _build_reject(reason: str) -> dict[str, Any]:
    """构造拒答返回值。"""
    return {
        "answer": RAGAnswer(
            answer="未在知识库中找到与该问题相关的依据，无法回答。",
            sources=[],
            confidence=0.0,
            rejected=True,
            reason=reason,
        ),
        "status": "done",
    }


def reject_node(state: dict[str, Any]) -> dict[str, Any]:
    """节点 4a：兜底拒答。

    独立成节点（而非在条件边里直接返回），
    是为了在图上有一个明确可见的"拒答"节点，便于追踪和调试。
    """
    max_score = state.get("max_score", 0.0)
    reason = f"检索最高相似度 {max_score:.3f} 低于阈值，知识库中可能无相关内容"
    return _build_reject(reason)

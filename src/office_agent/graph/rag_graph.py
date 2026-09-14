"""RAG 问答子图：LangGraph StateGraph 组装。

对应阶段三图 A 的结构：

    START → rewrite → retrieve → [guard]
                                   ├─ reject → END
                                   └─ generate → [check]
                                                   ├─ retry → generate (有限次)
                                                   └─ END

条件边：
- guard: max_score < threshold → reject, else → generate
- check: sources 为空且重试未满 → retry, else → END
"""
from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from office_agent.config import Settings, get_settings
from office_agent.graph.state import RAGState
from office_agent.nodes.query import (
    check_citation,
    generate_answer,
    reject_node,
    rewrite_query,
    retrieve,
    should_reject,
)
from office_agent.ports.embeddings import EmbeddingPort
from office_agent.ports.llm import LLMPort
from office_agent.ports.vectorstore import VectorStorePort
from office_agent.schemas import RAGAnswer
from office_agent.utils.logger import get_logger

logger = get_logger(__name__)


def build_rag_graph(
    llm: LLMPort,
    embedder: EmbeddingPort,
    store: VectorStorePort,
    settings: Settings | None = None,
) -> Any:
    """构建可编译的 RAG 状态图。

    返回 CompiledGraph，调用方用 graph.invoke(initial_state) 执行。
    """
    settings = settings or get_settings()
    top_k = settings.rag.top_k
    score_threshold = settings.rag.score_threshold

    # ---- 用 partial 把依赖注入节点函数 ----
    # 节点签名必须是 (state) -> dict，但我们的节点需要 llm/embedder/store，
    # 用 functools.partial 绑定这些依赖，使节点只接收 state。
    rewrite_node = partial(rewrite_query, llm=llm)
    retrieve_node = partial(
        retrieve, embedder=embedder, store=store, top_k=top_k
    )
    generate_node = partial(generate_answer, llm=llm)

    # ---- 构建图 ----
    graph = StateGraph(RAGState)

    # 添加节点
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("reject", reject_node)

    # 添加固定边
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")

    # 条件边：置信度守卫
    graph.add_conditional_edges(
        "retrieve",
        partial(should_reject, score_threshold=score_threshold),
        {
            "reject": "reject",
            "generate": "generate",
        },
    )

    # 条件边：引用校验（带有限次重试回边）
    graph.add_conditional_edges(
        "generate",
        partial(check_citation, max_retries=1),
        {
            "retry": "generate",
            "end": END,
            "reject": "reject",
        },
    )

    # 拒答节点直接结束
    graph.add_edge("reject", END)

    compiled = graph.compile()
    logger.info(
        "RAG 图编译完成：top_k=%d, score_threshold=%.2f",
        top_k, score_threshold,
    )
    return compiled


def run_rag(
    question: str,
    llm: LLMPort,
    embedder: EmbeddingPort,
    store: VectorStorePort,
    chat_history: list[dict[str, str]] | None = None,
    doc_scope: dict | None = None,
    settings: Settings | None = None,
) -> RAGAnswer:
    """便捷入口：构建图并执行一次问答。

    返回 RAGAnswer（含答案、引用来源、置信度、拒答标记）。
    """
    graph = build_rag_graph(llm, embedder, store, settings)

    initial_state: RAGState = {
        "question": question,
        "chat_history": chat_history or [],
        "doc_scope": doc_scope,
        "retry_count": 0,
    }

    logger.info("RAG 问答开始：%s", question[:50])
    final_state = graph.invoke(initial_state)

    answer = final_state.get("answer")
    if answer is None:
        return RAGAnswer(
            answer="系统内部错误：流程未产出答案",
            sources=[],
            confidence=0.0,
            rejected=True,
            reason="graph returned no answer",
        )

    logger.info(
        "RAG 问答结束：rejected=%s, confidence=%.3f, sources=%d",
        answer.rejected, answer.confidence, len(answer.sources),
    )
    return answer

"""检索工具函数：封装 EmbeddingPort + VectorStorePort 的检索逻辑。

独立成函数（而非内联在节点里），便于：
- 单测直接调用（注入假 embedder/store）；
- 后续 Rerank 节点复用本函数的输出。
"""
from __future__ import annotations

from office_agent.ports.embeddings import EmbeddingPort
from office_agent.ports.vectorstore import VectorStorePort
from office_agent.schemas import RetrievedChunk
from office_agent.utils.logger import get_logger

logger = get_logger(__name__)


def retrieve_chunks(
    query: str,
    embedder: EmbeddingPort,
    store: VectorStorePort,
    top_k: int,
    where: dict | None = None,
) -> list[RetrievedChunk]:
    """向量检索：query -> embedding -> similarity_search。

    返回按相似度降序排列的 RetrievedChunk 列表。
    空结果返回空列表（不抛错，由调用方决定如何处理）。
    """
    if not query.strip():
        logger.warning("检索 query 为空")
        return []

    try:
        query_vec = embedder.embed_query(query)
    except Exception as exc:  # noqa: BLE001
        logger.error("查询向量化失败：%s", exc)
        return []

    try:
        results = store.similarity_search(
            query_embedding=query_vec,
            top_k=top_k,
            where=where,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("向量检索失败：%s", exc)
        return []

    logger.info(
        "检索完成：query=%s... -> %d 条结果，分数范围 [%.3f, %.3f]",
        query[:30], len(results),
        results[0].score if results else 0.0,
        results[-1].score if results else 0.0,
    )
    return results

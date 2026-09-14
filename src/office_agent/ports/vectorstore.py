"""向量库端口（抽象协议）。

隔离 Chroma 的具体 API：未来迁 Qdrant 时，nodes/retrieval 无需任何改动，
只新增 qdrant_store.py 适配器并在工厂处切换。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from office_agent.schemas import DocumentChunk, RetrievedChunk


class VectorStorePort(ABC):
    """向量持久化与近似最近邻检索端口。"""

    @abstractmethod
    def upsert_document(self, chunks: list[DocumentChunk]) -> int:
        """按文档粒度幂等写入：先删除同 source_file 的旧切片，再写入新切片。

        语义保证：同一文件反复索引后，库内只保留最新一版的切片，不产生重复。
        返回实际写入的切片数。
        """
        raise NotImplementedError

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        """向量检索 + 可选元数据过滤。

        where 例：{"department": "HR"} 或 {"doc_type": {"$in": [...]}}
        返回按相似度降序排列的结果，分数归一化到 [0,1]。
        """
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        """返回当前集合中的切片总数（供健康检查与测试断言）。"""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """清空集合（仅用于 --reset 全量重建与测试隔离）。"""
        raise NotImplementedError

"""Chroma 向量库适配器。

实现端口 VectorStorePort，对上隐藏 Chroma 全部 API 细节。两个关键坑已处理：
1. Chroma metadata 的值只接受 str/int/float/bool，None 必须剔除；
2. Chroma 返回的是 distance（cosine 空间下越小越相似），
   业务层需要 [0,1] 的相似度，故做 similarity = 1 - distance 并截断。
"""
from __future__ import annotations

from typing import Any

from office_agent.ports.vectorstore import VectorStorePort
from office_agent.schemas import CHUNK_METADATA_FIELDS, DocumentChunk, RetrievedChunk
from office_agent.utils.logger import get_logger

logger = get_logger(__name__)


class ChromaVectorStoreAdapter(VectorStorePort):
    def __init__(self, persist_dir: str, collection_name: str) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError("未安装 chromadb，请先 pip install chromadb") from exc

        # PersistentClient：嵌入式部署，数据落本地目录（Docker 中挂载为卷）
        self._client = chromadb.PersistentClient(path=persist_dir)
        # hnsw:space=cosine：与归一化后的 bge 向量配套
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Chroma 就绪：%s / collection=%s / 现有切片=%d",
            persist_dir, collection_name, self._collection.count(),
        )

    @staticmethod
    def _to_metadata(chunk: DocumentChunk) -> dict[str, Any]:
        """抽取元数据并剔除 None（Chroma 拒绝 None 值，会直接报错）。"""
        data = chunk.model_dump()
        return {
            key: data[key]
            for key in CHUNK_METADATA_FIELDS
            if data.get(key) is not None
        }

    def upsert_document(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0

        # 同一 source_file 先删后写：文档更新后旧切片不会残留，重建幂等
        source_file = chunks[0].source_file
        if any(c.source_file != source_file for c in chunks):
            raise ValueError("upsert_document 只接受同一 source_file 的切片")

        embeddings = [c.embedding for c in chunks]
        if any(vec is None for vec in embeddings):
            raise ValueError("存在未生成 embedding 的切片，拒绝写入")

        try:
            self._collection.delete(where={"source_file": source_file})
            self._collection.upsert(
                ids=[c.id for c in chunks],
                embeddings=embeddings,
                documents=[c.content for c in chunks],
                metadatas=[self._to_metadata(c) for c in chunks],
            )
        except Exception as exc:  # noqa: BLE001 - 统一包装底层库异常
            raise RuntimeError(f"写入向量库失败（{source_file}）：{exc}") from exc

        logger.info("文档 %s 入库 %d 个切片", source_file, len(chunks))
        return len(chunks)

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        try:
            result = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"向量检索失败：{exc}") from exc

        # Chroma 返回嵌套列表（第 0 维对应第 0 条 query，这里只有一条）
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for chunk_id, content, meta, distance in zip(ids, documents, metadatas, distances):
            similarity = max(0.0, min(1.0, 1.0 - float(distance)))
            chunk = DocumentChunk(
                id=chunk_id,
                content=content,
                source_file=meta.get("source_file", ""),
                section=meta.get("section", ""),
                department=meta.get("department"),
                doc_type=meta.get("doc_type", "general"),
                chunk_index=int(meta.get("chunk_index", 0)),
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=similarity))
        return retrieved

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )
        logger.warning("集合 %s 已清空（全量重建/测试隔离）", name)

"""RAG 图单元测试。

策略：用 FakeEmbedder + FakeStore + MockLLM 跑整张图，
验证条件边（置信度守卫、引用校验）的逻辑正确性。
不加载真实模型，毫秒级完成。
"""
from __future__ import annotations

import hashlib
from typing import Any

import pytest

from office_agent.graph.rag_graph import build_rag_graph, run_rag
from office_agent.schemas import DocumentChunk, RAGAnswer, RetrievedChunk, Source
from office_agent.adapters.mock_llm import MockLLMAdapter


# ---- 测试用 Fake 组件（复用 test_indexing 的 FakeEmbedder 思路）----


class FakeEmbedder:
    dim = 32

    def embed_documents(self, texts):
        return [self._one(t) for t in texts]

    def embed_query(self, text):
        return self._one(text)

    @staticmethod
    def _one(text):
        h = hashlib.sha1(text.encode("utf-8")).digest()
        vec = [(h[i % len(h)] / 255.0) for i in range(32)]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]


class FakeStore:
    """可控的假向量库：返回预设的 RetrievedChunk。"""

    def __init__(self, chunks_to_return: list[RetrievedChunk]):
        self._chunks = chunks_to_return

    def upsert_document(self, chunks):
        return len(chunks)

    def similarity_search(self, query_embedding, top_k, where=None):
        return self._chunks[:top_k]

    def count(self):
        return len(self._chunks)

    def reset(self):
        self._chunks.clear()


def _make_chunk(score: float, source_file: str = "test.md",
                section: str = "测试章节", content: str = "测试内容") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=DocumentChunk(
            id="test-001",
            content=content,
            source_file=source_file,
            section=section,
        ),
        score=score,
    )


# ---- 测试用例 ----


def test_rag_rejects_low_confidence():
    """验收标准 4：相似度低于阈值时拒答。"""
    high_score_chunk = _make_chunk(score=0.3)  # 低于阈值 0.5
    store = FakeStore([high_score_chunk])
    llm = MockLLMAdapter()
    emb = FakeEmbedder()

    # 用最小配置（覆盖阈值）
    from office_agent.config import Settings
    settings = Settings(
        rag={"top_k": 4, "score_threshold": 0.5, "rerank_enabled": False},
    )

    answer = run_rag("无关问题", llm=llm, embedder=emb, store=store, settings=settings)
    assert answer.rejected is True
    assert answer.sources == []
    assert "未在知识库中找到" in answer.answer


def test_rag_answers_high_confidence():
    """验收标准 1：高置信度时给出带引用的答案。"""
    good_chunk = _make_chunk(
        score=0.8,
        source_file="假期制度.md",
        section="第二章 年假",
        content="年假5天",
    )
    store = FakeStore([good_chunk])
    llm = MockLLMAdapter()
    emb = FakeEmbedder()

    from office_agent.config import Settings
    settings = Settings(
        rag={"top_k": 4, "score_threshold": 0.5, "rerank_enabled": False},
    )

    answer = run_rag("年假几天", llm=llm, embedder=emb, store=store, settings=settings)
    assert answer.rejected is False
    assert answer.confidence == 0.8
    assert len(answer.sources) > 0
    assert answer.sources[0].source_file == "假期制度.md"


def test_rag_empty_results_rejects():
    """检索结果为空时应该拒答。"""
    store = FakeStore([])
    llm = MockLLMAdapter()
    emb = FakeEmbedder()

    from office_agent.config import Settings
    settings = Settings(
        rag={"top_k": 4, "score_threshold": 0.5, "rerank_enabled": False},
    )

    answer = run_rag("任意问题", llm=llm, embedder=emb, store=store, settings=settings)
    assert answer.rejected is True

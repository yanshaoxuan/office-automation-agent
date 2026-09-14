"""索引流水线单元测试。

分层策略：
- 纯逻辑（loader/splitter）：直接测，毫秒级；
- pipeline 装配与错误隔离：注入 Fake 适配器，不加载真实模型；
- 真实 bge-m3 + Chroma 的端到端验证走手工冒烟脚本，不进 CI 常规用例。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from office_agent.config import get_settings
from office_agent.indexing.loader import DocumentLoadError, load_file
from office_agent.indexing.pipeline import run_indexing
from office_agent.indexing.splitter import split_document
from office_agent.schemas import DocumentChunk, RawDocument


# ------------------------------ splitter ------------------------------


def test_split_by_headings_keeps_breadcrumb():
    doc = RawDocument(
        source_file="hr.md",
        text="# 一级标题\n正文A\n## 二级标题\n正文B\n### 三级标题\n正文C",
    )
    chunks = split_document(doc, chunk_size=500, chunk_overlap=50)

    assert len(chunks) == 3
    assert chunks[0].section == "一级标题"
    assert chunks[1].section == "一级标题 > 二级标题"
    assert chunks[2].section == "一级标题 > 二级标题 > 三级标题"
    # 面包屑注入正文，chunk 脱离原文档也自带上下文
    assert chunks[2].content.startswith("【一级标题 > 二级标题 > 三级标题】")
    # 序号连续且 ID 确定性
    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_long_section_is_windowed_with_overlap():
    long_body = "企业规章制度。" * 100  # 700 字，超过 500
    doc = RawDocument(source_file="long.md", text="# 长章节\n" + long_body)

    chunks = split_document(doc, chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 2

    # 相邻窗口必须存在内容重叠（边界安全网）
    tail = chunks[0].content[-50:]
    assert tail in chunks[1].content


def test_invalid_overlap_rejected():
    doc = RawDocument(source_file="x.md", text="# t\nabc")
    with pytest.raises(ValueError):
        split_document(doc, chunk_size=100, chunk_overlap=100)


def test_chunk_id_deterministic():
    doc = RawDocument(source_file="a.md", text="# t\nhello")
    ids1 = [c.id for c in split_document(doc)]
    ids2 = [c.id for c in split_document(doc)]
    assert ids1 == ids2


# ------------------------------- loader -------------------------------


def test_loader_parses_front_matter(tmp_path: Path):
    f = tmp_path / "policy.md"
    f.write_text(
        "---\ntitle: 测试制度\ndepartment: HR\ndoc_type: hr_policy\n---\n# 正文\n你好",
        encoding="utf-8",
    )
    doc = load_file(f, tmp_path)
    assert doc.department == "HR"
    assert doc.doc_type == "hr_policy"
    assert "你好" in doc.text
    assert doc.source_file == "policy.md"


def test_loader_rejects_empty_body(tmp_path: Path):
    f = tmp_path / "empty.md"
    f.write_text("---\ndepartment: HR\n---\n   \n", encoding="utf-8")
    with pytest.raises(DocumentLoadError):
        load_file(f, tmp_path)


# ------------------------------ pipeline ------------------------------


class FakeEmbedder:
    """用内容哈希生成确定性向量，维度 32，避免加载真实模型。"""

    dim = 32

    def embed_documents(self, texts):
        return [self._one(t) for t in texts]

    def embed_query(self, text):
        return self._one(text)

    @staticmethod
    def _one(text):
        h = hashlib.sha1(text.encode("utf-8")).digest()
        # 重复哈希填满 32 维后归一化
        vec = [(h[i % len(h)] / 255.0) for i in range(32)]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]


class FakeStore:
    """内存版 VectorStorePort，记录调用。"""

    def __init__(self):
        self.data: dict[str, DocumentChunk] = {}

    def upsert_document(self, chunks):
        for c in chunks:
            self.data[c.id] = c
        return len(chunks)

    def similarity_search(self, query_embedding, top_k, where=None):
        return []

    def count(self):
        return len(self.data)

    def reset(self):
        self.data.clear()


def test_pipeline_indexes_valid_file_and_isolates_bad_one(tmp_path: Path):
    (tmp_path / "good.md").write_text("# 制度\n年假 5 天", encoding="utf-8")
    # 空正文文件：加载阶段失败，应被隔离而不是中断整批
    (tmp_path / "bad.txt").write_text("   ", encoding="utf-8")

    settings = get_settings()
    store = FakeStore()
    result = run_indexing(
        tmp_path, settings=settings, embedder=FakeEmbedder(), store=store
    )

    assert result.files_seen == 2
    assert result.files_indexed == 1
    assert result.chunks_indexed == 1
    assert len(result.failures) == 1
    assert result.failures[0][0] == "bad.txt"
    assert store.count() == 1

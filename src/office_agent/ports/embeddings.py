"""Embedding 端口（抽象协议）。

为什么先定义端口：
- 业务节点只依赖本抽象，不知道背后是 bge-m3（sentence-transformers）、
  ONNX 量化模型还是远程 API；
- 换实现时只新增一个适配器文件，nodes/graph/indexing 代码零改动；
- 测试时可注入 FakeEmbedding（不加载模型、毫秒级跑完单测）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingPort(ABC):
    """文本向量化端口。所有实现必须输出与向量库空间一致的向量维度。"""

    #: 向量维度，由具体模型决定（bge-m3=1024）；适配器必须覆盖
    dim: int = 0

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量编码文档切片（索引阶段使用）。

        实现要求：
        - 空列表返回空列表，不允许抛错；
        - 内部做批处理，避免一次性把大量文本送入模型导致内存峰值；
        - 返回 L2 归一化向量，使向量库可用余弦/点积语义比较。
        """
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """编码单条查询（问答阶段使用）。

        查询与文档可能需要不同的编码方式（如部分模型要求给 query 加前缀），
        因此与 embed_documents 分开定义，而不是图省事共用一个方法。
        """
        raise NotImplementedError

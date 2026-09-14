"""离线索引流水线：加载 -> 切片 -> 向量化 -> 入库。

本模块是"组合根"：只有这里知道适配器的具体类，
其余业务代码只依赖 ports 抽象。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from office_agent.adapters.bge_embeddings import BGEEmbeddingAdapter
from office_agent.adapters.chroma_store import ChromaVectorStoreAdapter
from office_agent.config import Settings, get_settings
from office_agent.indexing.loader import DocumentLoadError, discover_documents, load_file
from office_agent.indexing.splitter import split_document
from office_agent.ports.embeddings import EmbeddingPort
from office_agent.ports.vectorstore import VectorStorePort
from office_agent.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IndexingResult:
    """索引结果汇总（供 CLI 输出与测试断言）。"""

    files_seen: int = 0
    files_indexed: int = 0
    chunks_indexed: int = 0
    # (文件名, 失败原因)：单文件失败被隔离记录，不中断整批任务
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.files_indexed > 0 and not self.failures


def run_indexing(
    input_path: Path,
    settings: Settings | None = None,
    reset: bool = False,
    embedder: EmbeddingPort | None = None,
    store: VectorStorePort | None = None,
) -> IndexingResult:
    """执行索引流水线。

    embedder/store 允许测试注入（依赖注入）：
    生产路径按 settings 构造真实适配器，测试可传入假实现，毫秒级跑完。
    """
    settings = settings or get_settings()
    result = IndexingResult()

    # ---- 0. 资源装配（重资源对象只创建一次，全程共享）----
    embedder = embedder or BGEEmbeddingAdapter(
        model_name=settings.embeddings.model_name,
        device=settings.embeddings.device,
        batch_size=settings.indexing.embedding_batch_size,
        model_path=settings.embeddings.local_path,
    )
    store = store or ChromaVectorStoreAdapter(
        persist_dir=settings.vectorstore.persist_dir,
        collection_name=settings.vectorstore.collection_name,
    )

    if reset:
        logger.warning("收到 --reset：先清空向量库集合")
        store.reset()

    # ---- 1. 发现文档 ----
    base_dir = input_path if input_path.is_dir() else input_path.parent
    files = discover_documents(
        input_path, settings.indexing.supported_extensions
    )
    result.files_seen = len(files)

    # ---- 2. 逐文件处理：错误隔离，一个坏文件不拖垮整批 ----
    for path in files:
        try:
            doc = load_file(path, base_dir)
            chunks = split_document(
                doc,
                chunk_size=settings.indexing.chunk_size,
                chunk_overlap=settings.indexing.chunk_overlap,
            )
            if not chunks:
                raise DocumentLoadError("切片结果为空")

            vectors = embedder.embed_documents([c.content for c in chunks])
            for chunk, vector in zip(chunks, vectors):
                chunk.embedding = vector

            written = store.upsert_document(chunks)
            result.files_indexed += 1
            result.chunks_indexed += written
        except (DocumentLoadError, ValueError, RuntimeError) as exc:
            # 预期内的失败：记录并继续；堆栈进 debug，错误摘要进 warning
            logger.warning("文档 %s 索引失败：%s", path.name, exc)
            logger.debug("失败详情：", exc_info=True)
            result.failures.append((path.name, str(exc)))

    logger.info(
        "索引完成：发现 %d / 成功 %d / 失败 %d / 切片总数 %d",
        result.files_seen, result.files_indexed,
        len(result.failures), result.chunks_indexed,
    )
    return result

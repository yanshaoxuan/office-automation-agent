"""bge 系列 Embedding 的 sentence-transformers 适配器。

选型说明（阶段二决策的落地）：
- 本地推理，中文效果好，企业文档不出内网；
- 输出 L2 归一化向量，配合 Chroma 的 cosine 空间使用；
- bge-m3 与 bge 系小模型接口一致，换模型只改配置中的 model_name，
  注意：换模型后向量维度会变，必须 --reset 重建索引。

重型依赖（torch/sentence_transformers）在构造时才导入，
使得"导入本模块"零成本，纯逻辑单测不受拖累。
"""
from __future__ import annotations

from office_agent.ports.embeddings import EmbeddingPort
from office_agent.utils.logger import get_logger

logger = get_logger(__name__)


class BGEEmbeddingAdapter(EmbeddingPort):
    """基于 sentence-transformers 的本地向量适配器。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        batch_size: int = 8,
        model_path: str = "",
    ) -> None:
        try:
            # 延迟导入：torch 加载耗时，只有真正建索引/问答时才付出
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "未安装 sentence-transformers，请先 pip install sentence-transformers"
            ) from exc

        # 本地预下载权重优先：离线/内网环境与 Docker 镜像内均直接加载目录
        from pathlib import Path

        if model_path and Path(model_path).is_dir():
            model_ref: str = model_path
            logger.info("从本地目录加载 Embedding 模型：%s", model_path)
        else:
            if model_path:
                logger.warning(
                    "配置的 local_path 不存在（%s），回退到在线拉取 %s",
                    model_path, model_name,
                )
            model_ref = model_name
            logger.info("在线加载 Embedding 模型：%s（device=%s）", model_name, device)

        self._model = SentenceTransformer(model_ref, device=device)
        self._batch_size = batch_size
        # sentence-transformers 6.x 将方法更名为 get_embedding_dimension，
        # 用 getattr 兼容新旧两个大版本
        get_dim = getattr(
            self._model, "get_embedding_dimension",
            getattr(self._model, "get_sentence_embedding_dimension"),
        )
        self.dim = int(get_dim())
        logger.info("Embedding 模型加载完成，向量维度 dim=%d", self.dim)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        all_vectors: list[list[float]] = []
        # 手动分批：控制长文档场景下的内存峰值，进度可见
        total = len(texts)
        for start in range(0, total, self._batch_size):
            batch = texts[start:start + self._batch_size]
            vectors = self._encode(batch)
            all_vectors.extend(vectors)
            logger.debug(
                "embedding 进度 %d/%d", min(start + self._batch_size, total), total
            )
        return all_vectors

    def embed_query(self, text: str) -> list[float]:
        # bge-m3 对 query 不需要特殊指令前缀；独立成方法是为了保留这个扩展点
        # （未来若换 e5 等要求 "query:" 前缀的模型，只改这里）
        return self._encode([text])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """统一的编码出口：归一化 + numpy 转 list，失败时给出可定位的错误。"""
        try:
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=True,  # 归一化后点积等价余弦相似度
                convert_to_numpy=True,
            )
        except Exception as exc:  # noqa: BLE001 - 模型推理异常统一包装
            raise RuntimeError(f"Embedding 推理失败（{len(texts)} 条文本）：{exc}") from exc
        return embeddings.tolist()

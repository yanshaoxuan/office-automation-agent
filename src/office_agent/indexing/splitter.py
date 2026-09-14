"""标题感知的 Markdown 切片器。

两级切分策略：
1. 先按 Markdown 标题（# ~ ######）切成语义完整的 section，
   并维护"章节面包屑"（如 假期制度 > 年假）；
2. 单个 section 超过 chunk_size 时，再做"带重叠的滑动窗口"二次切分，
   切点优先落在段落/换行边界，避免把一句话硬截断。

为什么不直接按 500 字定长切：
- 定长切片会把同一主题的内容切碎，向量语义被稀释；
- 也无法回答"这段话出自哪一章"，溯源能力丢失。
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator

from office_agent.schemas import DocumentChunk, RawDocument
from office_agent.utils.logger import get_logger

logger = get_logger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def build_chunk_id(source_file: str, chunk_index: int, content: str) -> str:
    """由"来源 + 序号 + 内容"派生确定性 ID。

    - 同一文件、同样内容 -> ID 相同（天然支持去重/增量）；
    - 内容改变 -> ID 改变（配合 upsert_document 的先删后写实现幂等重建）。
    """
    digest = hashlib.sha1(
        f"{source_file}:{chunk_index}:{content}".encode("utf-8")
    ).hexdigest()
    return f"{chunk_index:04d}-{digest[:16]}"


def _iter_sections(text: str) -> Iterator[tuple[str, str]]:
    """把全文按标题切成 (章节面包屑, 正文) 序列。

    用栈维护标题层级，保证看到三级标题时仍能还原出"一级 > 二级 > 三级"。
    """
    stack: list[tuple[int, str]] = []
    body_lines: list[str] = []

    def flush() -> Iterator[tuple[str, str]]:
        body = "\n".join(body_lines).strip()
        if body:
            path = " > ".join(title for _, title in stack)
            yield path, body
        body_lines.clear()

    for line in text.split("\n"):
        match = _HEADING_RE.match(line)
        if match:
            # 遇到新标题：先产出上一个 section
            yield from flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            # 弹出同级及更深层级，再压入当前标题
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            body_lines.append(line)

    yield from flush()


def _window_long_section(
    body: str, chunk_size: int, overlap: int
) -> list[str]:
    """对超长 section 做带重叠的滑动窗口切分。

    切点优先选窗口后一半内的段落边界（空行），其次普通换行；
    都找不到时才在字符边界硬切。overlap 保证跨窗口的句子不丢上下文。
    """
    windows: list[str] = []
    start = 0
    n = len(body)

    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            # 只在窗口后一半寻找边界，避免切出过小的片段
            search_from = start + chunk_size // 2
            boundary = body.rfind("\n\n", search_from, end)
            if boundary == -1:
                boundary = body.rfind("\n", search_from, end)
            if boundary != -1:
                end = boundary

        piece = body[start:end].strip()
        if piece:
            windows.append(piece)

        if end >= n:
            break
        # 回退 overlap 个字符形成重叠；max(..., start+1) 保证起点必然前进
        start = max(end - overlap, start + 1)

    return windows


def split_document(
    doc: RawDocument, chunk_size: int = 500, chunk_overlap: int = 50
) -> list[DocumentChunk]:
    """把一篇 RawDocument 切成 DocumentChunk 列表。"""
    if chunk_overlap >= chunk_size:
        # 与配置层校验呼应：直接调用本函数时也不允许非法参数
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    chunks: list[DocumentChunk] = []
    for section_path, body in _iter_sections(doc.text):
        if len(body) <= chunk_size:
            pieces = [body]
        else:
            pieces = _window_long_section(body, chunk_size, chunk_overlap)

        for piece in pieces:
            index = len(chunks)
            # 把章节面包屑注入正文：让每个 chunk 脱离原文档也自带上下文
            prefix = f"【{section_path}】\n" if section_path else ""
            content = prefix + piece
            chunks.append(
                DocumentChunk(
                    id=build_chunk_id(doc.source_file, index, content),
                    content=content,
                    source_file=doc.source_file,
                    section=section_path,
                    department=doc.department,
                    doc_type=doc.doc_type,
                    chunk_index=index,
                )
            )

    logger.info("文档 %s 切分为 %d 个 chunk", doc.source_file, len(chunks))
    return chunks

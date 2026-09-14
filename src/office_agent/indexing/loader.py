"""文档加载器：把磁盘上的原始文档读成统一的 RawDocument。

当前支持 Markdown(.md) 与纯文本(.txt)。Markdown 支持可选的 YAML front matter
携带业务元数据，示例：

    ---
    title: 员工考勤与假期管理办法
    department: HR
    doc_type: hr_policy
    ---
    # 正文标题 ...

没有 front matter 时，元数据退化为默认值，source_file 始终可用于溯源。
"""
from __future__ import annotations

from pathlib import Path

import yaml

from office_agent.schemas import RawDocument
from office_agent.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentLoadError(Exception):
    """文档加载失败（编码错误 / front matter 非法等），错误信息需带文件名。"""


def discover_documents(input_path: Path, extensions: list[str]) -> list[Path]:
    """递归发现指定后缀的文档，返回排序后的路径列表。

    排序保证多次索引的处理顺序确定（切片序号稳定、日志可读）。
    """
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在：{input_path}")

    allowed = {ext.lower() for ext in extensions}
    if input_path.is_file():
        files = [input_path] if input_path.suffix.lower() in allowed else []
    else:
        files = [
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in allowed
        ]
    files.sort()
    logger.info("在 %s 下发现 %d 个候选文档（%s）", input_path, len(files), allowed)
    return files


def _parse_front_matter(text: str) -> tuple[dict, str]:
    """解析可选的 YAML front matter，返回 (元数据dict, 正文)。

    必须严格以文件首行 '---' 开始，否则视为没有 front matter。
    """
    # startswith("---") 同时兼容带 BOM 的情况（BOM 在调用前已去除）
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    # split 结果形如 ['', '\\nyaml\\n', '\\n正文']
    if len(parts) < 3:
        # 只有起始分隔符没有结束符：属于文档格式问题，明确报错而不是静默吞掉
        raise DocumentLoadError("front matter 缺少结束分隔符 '---'")

    raw_meta, body = parts[1], parts[2]
    try:
        meta = yaml.safe_load(raw_meta) or {}
    except yaml.YAMLError as exc:
        raise DocumentLoadError(f"front matter 不是合法 YAML：{exc}") from exc

    if not isinstance(meta, dict):
        raise DocumentLoadError("front matter 必须是键值对映射")
    return meta, body.lstrip("\r\n")


def _clean_text(text: str) -> str:
    """基础清洗：统一换行、去除行尾空白、折叠 3 个以上连续空行。

    注意：不做任何"理解性"清洗（不改写内容），避免破坏原文语义与可溯源性。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    cleaned = "\n".join(lines)
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned.strip()


def load_file(path: Path, base_dir: Path) -> RawDocument:
    """加载单个文件为 RawDocument。任何失败都包装成带文件名的明确异常。"""
    try:
        # utf-8-sig 可同时吃掉可能存在的 BOM
        raw = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(f"{path} 不是 UTF-8 编码，请转码后重试") from exc
    except OSError as exc:
        raise DocumentLoadError(f"{path} 读取失败：{exc}") from exc

    meta, body = _parse_front_matter(raw)
    text = _clean_text(body)
    if not text:
        raise DocumentLoadError(f"{path} 正文为空")

    # 用相对路径作为稳定主键：无论从哪个目录执行索引，同一文档 ID 一致
    try:
        source = path.relative_to(base_dir).as_posix()
    except ValueError:
        source = path.name

    doc = RawDocument(
        source_file=source,
        doc_type=str(meta.get("doc_type", "general")),
        department=meta.get("department"),
        title=meta.get("title"),
        text=text,
    )
    logger.debug("已加载 %s（%d 字符）", source, len(text))
    return doc

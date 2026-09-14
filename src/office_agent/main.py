"""命令行入口。

模块 0 阶段只提供两个自检命令，验证"骨架 + 配置 + 日志"闭环可用：

    python -m office_agent.main check    # 环境健康检查
    python -m office_agent.main config   # 打印当前生效配置（密钥脱敏）
    python -m office_agent.main index   # 构建文档向量索引
    python -m office_agent.main ask      # RAG 文档问答
    python -m office_agent.main meeting  # 会议纪要抽取
    python -m office_agent.main report   # 自动周报生成

后续模块会在这里挂载：index（建索引）、ask（文档问答）、
meeting（会议纪要）、report（周报）等子命令。
"""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import pydantic

from office_agent import __version__
from office_agent.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, get_settings
from office_agent.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)

# 项目声明的最低 Python 版本（agent.md 约束 Python 3.10+）
MIN_PYTHON = (3, 10)


def _cmd_check(_args: argparse.Namespace) -> int:
    """环境健康检查：逐项验证运行前提，任何一项失败即返回非零退出码。"""
    logger.info("开始环境健康检查，office-agent v%s", __version__)
    ok = True

    # 1) Python 版本
    py_ver = platform.python_version()
    if sys.version_info >= MIN_PYTHON:
        logger.info("Python 版本检查通过：%s", py_ver)
    else:
        logger.error(
            "Python 版本过低：当前 %s，要求 >= %d.%d",
            py_ver, MIN_PYTHON[0], MIN_PYTHON[1],
        )
        ok = False

    # 2) 配置文件
    if DEFAULT_CONFIG_PATH.exists():
        logger.info("配置文件存在：%s", DEFAULT_CONFIG_PATH)
    else:
        logger.error("配置文件缺失：%s", DEFAULT_CONFIG_PATH)
        ok = False

    # 3) settings 可被成功加载（内部会做类型校验与路径解析）
    try:
        settings = get_settings()
        logger.info("配置加载与校验通过，log_level=%s", settings.app.log_level)
    except pydantic.ValidationError as exc:
        # 配置错误是最常见的部署问题，打印人类可读的逐条错误
        logger.error("配置校验失败：")
        for err in exc.errors():
            loc = ".".join(str(x) for x in err["loc"])
            logger.error("  - %s: %s", loc, err["msg"])
        return 1
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    # 4) 数据目录可写（向量库持久化、日志都依赖写权限）
    persist_dir = Path(settings.vectorstore.persist_dir)
    try:
        persist_dir.mkdir(parents=True, exist_ok=True)
        probe = persist_dir / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        logger.info("数据目录可写：%s", persist_dir)
    except OSError as exc:
        logger.error("数据目录不可写：%s（%s）", persist_dir, exc)
        ok = False

    if ok:
        logger.info("健康检查全部通过 ✅")
        return 0
    logger.error("健康检查存在失败项 ❌")
    return 1


def _cmd_config(_args: argparse.Namespace) -> int:
    """打印当前生效的完整配置（密钥脱敏），用于确认环境变量覆盖是否生效。"""
    import json

    try:
        settings = get_settings()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except pydantic.ValidationError as exc:
        logger.error("配置校验失败：")
        for err in exc.errors():
            loc = ".".join(str(x) for x in err["loc"])
            logger.error("  - %s: %s", loc, err["msg"])
        return 1

    # ensure_ascii=False：中文不转义；indent=2：方便肉眼核对
    print(json.dumps(settings.redacted_dict(), ensure_ascii=False, indent=2))
    print(f"\n项目根目录: {PROJECT_ROOT}")
    return 0


def _cmd_index(args: argparse.Namespace) -> int:
    """离线索引：把目录中的文档切片、向量化并写入 Chroma。"""
    # 延迟导入：执行 check/config 等轻量命令时不加载 torch/chromadb
    from office_agent.indexing.pipeline import run_indexing

    input_path = Path(args.input)
    logger.info("开始构建索引，输入路径：%s，全量重置=%s", input_path, args.reset)
    try:
        result = run_indexing(input_path=input_path, reset=args.reset)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except Exception:  # noqa: BLE001 - 装配阶段的未预期错误需要完整堆栈
        logger.error("索引流水线异常终止", exc_info=True)
        return 1

    for name, reason in result.failures:
        logger.warning("  失败文件：%s -> %s", name, reason)
    if result.files_indexed == 0:
        logger.error("没有任何文档索引成功，请检查输入目录与文件格式")
        return 1
    logger.info(
        "索引成功：%d/%d 个文件，共 %d 个切片",
        result.files_indexed, result.files_seen, result.chunks_indexed,
    )
    return 0 if result.success else 2  # 部分失败用退出码 2 区分


def _parse_scope(raw: str) -> dict | None:
    """解析 --scope 过滤参数，兼容两种写法。

    背景：Windows PowerShell 5.1 向原生 exe 传参时会吞掉内层双引号，
    JSON 写法 '{"department":"IT"}' 到达 Python 时已变成 {department:IT}，
    必然解析失败。因此推荐 shell 友好的 key=value 格式：

        --scope department=IT
        --scope department=IT,doc_type=faq

    同时保留 JSON 格式兼容（bash 下仍可用）。
    返回 None 表示解析失败。
    """
    import json as json_mod

    raw = raw.strip()
    if not raw:
        return None

    # JSON 格式：以 { 开头
    if raw.startswith("{"):
        try:
            parsed = json_mod.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except json_mod.JSONDecodeError:
            return None

    # key=value 格式：逗号分隔多个键值对
    scope: dict = {}
    for pair in raw.split(","):
        if "=" not in pair:
            return None
        key, _, value = pair.partition("=")
        key, value = key.strip(), value.strip()
        if not key or not value:
            return None
        scope[key] = value
    return scope


def _cmd_ask(args: argparse.Namespace) -> int:
    """RAG 文档问答：构建图并执行一次问答。"""
    from office_agent.adapters.bge_embeddings import BGEEmbeddingAdapter
    from office_agent.adapters.chroma_store import ChromaVectorStoreAdapter
    from office_agent.adapters.mock_llm import MockLLMAdapter
    from office_agent.adapters.openai_compatible_llm import OpenAICompatibleLLMAdapter
    from office_agent.graph.rag_graph import run_rag
    from office_agent.schemas import RAGAnswer

    settings = get_settings()

    # ---- 装配 LLM ----
    if args.mock:
        llm = MockLLMAdapter()
        logger.info("使用 Mock LLM")
    else:
        api_key = settings.llm.api_key.get_secret_value()
        if not api_key:
            logger.error(
                "未配置 LLM API Key。请在 .env 中设置 OFFICE_AGENT_LLM__API_KEY，"
                "或使用 --mock 演示流程。"
            )
            return 1
        llm = OpenAICompatibleLLMAdapter(
            base_url=settings.llm.base_url,
            model=settings.llm.model,
            api_key=api_key,
            temperature=settings.llm.temperature,
            timeout_seconds=settings.llm.timeout_seconds,
            max_retries=settings.llm.max_retries,
        )

    # ---- 装配 Embedding + VectorStore ----
    embedder = BGEEmbeddingAdapter(
        model_name=settings.embeddings.model_name,
        device=settings.embeddings.device,
        batch_size=settings.indexing.embedding_batch_size,
        model_path=settings.embeddings.local_path,
    )
    store = ChromaVectorStoreAdapter(
        persist_dir=settings.vectorstore.persist_dir,
        collection_name=settings.vectorstore.collection_name,
    )

    # 解析 --scope：支持 key=value（推荐）和 JSON 两种格式
    doc_scope = None
    if args.scope:
        doc_scope = _parse_scope(args.scope)
        if doc_scope is None:
            logger.error(
                "--scope 参数无法解析：%r。"
                "推荐格式 key=value（多个用逗号分隔），如 department=IT",
                args.scope,
            )
            return 1

    # ---- 执行问答 ----
    if args.question:
        # 单次问答模式
        answer = run_rag(
            question=args.question,
            llm=llm,
            embedder=embedder,
            store=store,
            doc_scope=doc_scope,
            settings=settings,
        )
        _print_answer(answer)
        return 0 if not answer.rejected else 0  # 拒答也是正常流程结果

    # ---- 交互式问答模式 ----
    chat_history: list[dict[str, str]] = []
    logger.info("进入交互式问答（输入 quit 退出）")
    while True:
        try:
            question = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        answer = run_rag(
            question=question,
            llm=llm,
            embedder=embedder,
            store=store,
            chat_history=chat_history,
            doc_scope=doc_scope,
            settings=settings,
        )
        _print_answer(answer)

        # 更新对话历史（供下一轮查询改写使用）
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer.answer})

    return 0


def _print_answer(answer) -> None:
    """格式化打印 RAG 答案。"""
    print()
    if answer.rejected:
        print(f"  [拒答] {answer.answer}")
        print(f"  原因: {answer.reason}")
    else:
        print(f"  {answer.answer}")
        if answer.sources:
            print(f"\n  📎 引用来源 (置信度 {answer.confidence:.1%}):")
            for i, src in enumerate(answer.sources, 1):
                print(f"    [{i}] {src.source_file} - {src.section}")
        else:
            print(f"\n  置信度: {answer.confidence:.1%}（无引用来源）")


def _cmd_meeting(args: argparse.Namespace) -> int:
    """会议纪要抽取：转写文本 -> 摘要/决议/待办。"""
    import json as json_mod
    from datetime import datetime

    from office_agent.adapters.mock_llm import MockLLMAdapter
    from office_agent.adapters.mock_tools import MockToolAdapter
    from office_agent.adapters.openai_compatible_llm import OpenAICompatibleLLMAdapter
    from office_agent.graph.meeting_graph import run_meeting

    settings = get_settings()

    # ---- 装配 LLM ----
    if args.mock:
        llm = MockLLMAdapter()
        logger.info("使用 Mock LLM")
    else:
        api_key = settings.llm.api_key.get_secret_value()
        if not api_key:
            logger.error(
                "未配置 LLM API Key。请在 .env 中设置 OFFICE_AGENT_LLM__API_KEY，"
                "或使用 --mock 演示流程。"
            )
            return 1
        llm = OpenAICompatibleLLMAdapter(
            base_url=settings.llm.base_url,
            model=settings.llm.model,
            api_key=api_key,
            temperature=settings.llm.temperature,
            timeout_seconds=settings.llm.timeout_seconds,
            max_retries=settings.llm.max_retries,
        )

    # ---- 装配 Mock 工具 ----
    tool = MockToolAdapter()

    # ---- 读取转写文本 ----
    if args.file:
        transcript_path = Path(args.file)
        if not transcript_path.exists():
            logger.error("转写文件不存在：%s", transcript_path)
            return 1
        transcript = transcript_path.read_text(encoding="utf-8")
        logger.info("从文件加载转写文本：%s（%d chars）", transcript_path, len(transcript))
    else:
        # 使用内置示例
        sample_path = PROJECT_ROOT / "tests" / "fixtures" / "sample_meeting.txt"
        if sample_path.exists():
            transcript = sample_path.read_text(encoding="utf-8")
            logger.info("使用内置示例转写文本（%d chars）", len(transcript))
        else:
            logger.error("未找到示例转写文件：%s", sample_path)
            return 1

    attendees = [a.strip() for a in args.attendees.split(",") if a.strip()]
    meeting_time = args.time or datetime.now().isoformat()

    # ---- 执行抽取 ----
    result = run_meeting(
        transcript=transcript,
        attendees=attendees,
        meeting_time=meeting_time,
        llm=llm,
        tool=tool,
        settings=settings,
    )

    # ---- 打印结果 ----
    print()
    if result.degraded:
        print(f"  [降级模式] {result.notes}")

    print(f"\n  📋 会议摘要：")
    print(f"  {result.summary}")

    if result.decisions:
        print(f"\n  ✅ 关键决议：")
        for i, d in enumerate(result.decisions, 1):
            print(f"    [{i}] {d}")

    if result.action_items:
        print(f"\n  📝 待办事项：")
        for i, item in enumerate(result.action_items, 1):
            owner_str = item.owner or "待确认"
            date_str = item.due_date or "未定"
            conf_str = f"{item.confidence:.0%}"
            print(f"    [{i}] {item.task}")
            print(f"        责任人: {owner_str} | 截止: {date_str} | 置信度: {conf_str}")
            if item.evidence:
                print(f"        原文: \"{item.evidence[:80]}\"")
    else:
        print(f"\n  📝 待办事项：无")

    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """自动周报生成：多源采集 -> 聚合 -> 草稿 -> 人审 -> 发送。"""
    from datetime import date, timedelta

    from office_agent.adapters.mock_llm import MockLLMAdapter
    from office_agent.adapters.mock_tools import MockToolAdapter
    from office_agent.adapters.openai_compatible_llm import OpenAICompatibleLLMAdapter
    from office_agent.graph.report_graph import build_report_graph, run_report, resume_report

    settings = get_settings()

    # ---- 装配 LLM ----
    if args.mock:
        llm = MockLLMAdapter()
        logger.info("使用 Mock LLM")
    else:
        api_key = settings.llm.api_key.get_secret_value()
        if not api_key:
            logger.error(
                "未配置 LLM API Key。请在 .env 中设置 OFFICE_AGENT_LLM__API_KEY，"
                "或使用 --mock 演示流程。"
            )
            return 1
        llm = OpenAICompatibleLLMAdapter(
            base_url=settings.llm.base_url,
            model=settings.llm.model,
            api_key=api_key,
            temperature=settings.llm.temperature,
            timeout_seconds=settings.llm.timeout_seconds,
            max_retries=settings.llm.max_retries,
        )

    tool = MockToolAdapter()

    # ---- 计算本周周期 ----
    today = date.today()
    monday = today - timedelta(days=today.weekday())
    friday = monday + timedelta(days=4)
    start = args.start or monday.isoformat()
    end = args.end or friday.isoformat()

    # ---- 第一步：生成草稿（执行到 interrupt 暂停） ----
    thread_id = f"report-{args.user}-{start}"
    report, graph = run_report(
        user_id=args.user,
        time_range_start=start,
        time_range_end=end,
        llm=llm,
        tool=tool,
        settings=settings,
        thread_id=thread_id,
    )

    # ---- 打印草稿 ----
    print()
    print(f"  📄 周报草稿：{report.period} | 员工：{report.user_id}")
    if report.notes:
        print(f"  ⚠️ {report.notes}")
    print()
    for section in report.sections:
        print(f"  【{section.title}】")
        if section.items:
            for item in section.items:
                print(f"    - {item}")
        else:
            print("    （无）")
        print()
    print(f"  📎 数据溯源：{len(report.trace)} 条原始事项")

    # ---- 第二步：人审交互 ----
    print("\n  请审阅以上周报草稿：")
    print("    [a] 批准并发送邮件")
    print("    [r] 修改（输入批注后重新生成）")
    print("    [d] 丢弃不发")
    try:
        choice = input("\n  选择 (a/r/d): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return 0

    if choice == "a":
        decision = "approve"
        feedback = ""
    elif choice == "r":
        decision = "revise"
        try:
            feedback = input("  请输入修改意见：").strip()
        except (EOFError, KeyboardInterrupt):
            feedback = ""
    else:
        decision = "reject"
        feedback = ""

    # 恢复执行（从 interrupt 继续）
    final_report = resume_report(
        graph=graph,
        thread_id=thread_id,
        decision=decision,
        feedback=feedback,
    )

    print()
    if final_report.sent:
        print(f"  ✅ 周报已发送至主管邮箱（{args.user} {report.period}）")
    elif decision == "reject":
        print(f"  🗑️ 周报草稿已丢弃，未发送")
    elif decision == "revise":
        print(f"  📝 修改后周报（请手动发送）：")
        for section in final_report.sections:
            print(f"    【{section.title}】")
            for item in section.items:
                print(f"      - {item}")
    else:
        print(f"  ⚠️ 周报未发送：{final_report.notes}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器（独立成函数，便于后续为测试传参）。"""
    parser = argparse.ArgumentParser(
        prog="office-agent",
        description="企业办公自动化 Agent：RAG 问答 / 会议纪要 / 自动周报",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check", help="环境健康检查")
    subparsers.add_parser("config", help="打印生效配置（密钥脱敏）")

    index_parser = subparsers.add_parser("index", help="构建/更新文档向量索引")
    index_parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "raw_docs"),
        help="文档文件或目录（默认 data/raw_docs，支持 .md/.txt）",
    )
    index_parser.add_argument(
        "--reset", action="store_true",
        help="索引前清空整个集合（切换 embedding 模型后必须使用）",
    )

    ask_parser = subparsers.add_parser("ask", help="RAG 文档问答")
    ask_parser.add_argument(
        "question", nargs="?", default=None,
        help="要问的问题（省略则进入交互式问答）",
    )
    ask_parser.add_argument(
        "--mock", action="store_true",
        help="使用 Mock LLM（无需 API Key，用于流程演示）",
    )
    ask_parser.add_argument(
        "--scope", default=None,
        help="元数据过滤，推荐 key=value 格式（多个用逗号分隔），如 department=IT",
    )

    meeting_parser = subparsers.add_parser("meeting", help="会议纪要抽取")
    meeting_parser.add_argument(
        "--file", default=None,
        help="会议转写文本文件路径（省略则使用内置示例）",
    )
    meeting_parser.add_argument(
        "--attendees", default="张三,李四,王五",
        help="参会人名单，逗号分隔",
    )
    meeting_parser.add_argument(
        "--time", default=None,
        help="会议时间，ISO 格式如 2026-09-14T10:00:00（省略则用当前时间）",
    )
    meeting_parser.add_argument(
        "--mock", action="store_true",
        help="使用 Mock LLM（无需 API Key）",
    )

    report_parser = subparsers.add_parser("report", help="自动周报生成")
    report_parser.add_argument(
        "--user", default="张三",
        help="员工标识",
    )
    report_parser.add_argument(
        "--start", default=None,
        help="周期起始日期 ISO 格式（默认本周一）",
    )
    report_parser.add_argument(
        "--end", default=None,
        help="周期结束日期 ISO 格式（默认本周五）",
    )
    report_parser.add_argument(
        "--mock", action="store_true",
        help="使用 Mock LLM（无需 API Key）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """程序主入口，返回进程退出码（0 成功 / 非零失败）。"""
    parser = build_parser()
    args = parser.parse_args(argv)

    # 日志必须在任何业务动作之前初始化；先用默认级别，加载配置后再校准
    setup_logging()
    try:
        settings = get_settings()
        setup_logging(settings.app.log_level)
    except (pydantic.ValidationError, FileNotFoundError):
        # 已知的配置错误不在这里打堆栈：交给子命令逐条打印可读信息
        # （如 check 命令会列出具体字段与原因）
        pass
    except Exception:  # noqa: BLE001 - 日志校准失败不应阻断启动
        logger.warning("日志级别校准失败，使用默认级别继续", exc_info=True)

    handlers = {
        "check": _cmd_check,
        "config": _cmd_config,
        "index": _cmd_index,
        "ask": _cmd_ask,
        "meeting": _cmd_meeting,
        "report": _cmd_report,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

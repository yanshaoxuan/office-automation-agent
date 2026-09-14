"""OpenAI 兼容协议 LLM 适配器。

阶段二选型落地的核心：通过统一 base_url + model + api_key 切换供应商。
- DeepSeek: base_url=https://api.deepseek.com/v1, model=deepseek-chat
- 通义千问: base_url=https://dashscope.aliyuncs.com/compatible-mode/v1
- OpenAI:   base_url=https://api.openai.com/v1, model=gpt-4o

用 langchain_openai.ChatOpenAI 做底层（继承重试、超时、类型安全），
但对外只暴露 LLMPort 接口，业务层看不到 langchain 类型。
"""
from __future__ import annotations

import time
from typing import Any

from office_agent.ports.llm import LLMPort
from office_agent.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAICompatibleLLMAdapter(LLMPort):
    """基于 langchain-openai 的 OpenAI 兼容适配器。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float = 0.2,
        timeout_seconds: int = 30,
        max_retries: int = 2,
    ) -> None:
        if not api_key:
            raise ValueError(
                "LLM api_key 为空。请在 .env 中设置 OFFICE_AGENT_LLM__API_KEY"
            )

        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "未安装 langchain-openai，请 pip install langchain-openai"
            ) from exc

        self._llm = ChatOpenAI(
            base_url=base_url,
            model=model,
            api_key=api_key,
            temperature=temperature,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
        self._model_name = model
        logger.info("LLM 就绪：%s @ %s", model, base_url)

    def chat(self, messages: list[dict[str, str]]) -> str:
        """同步对话。对网络/限流错误做指数退避重试。"""
        lc_messages = [
            ("system" if m["role"] == "system"
             else "human" if m["role"] == "user"
             else "ai", m["content"])
            for m in messages
        ]

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = self._llm.invoke(lc_messages)
                content = resp.content if isinstance(resp.content, str) else str(resp.content)
                logger.debug("LLM 回复（%d chars）", len(content))
                return content
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(
                    "LLM 调用第 %d/3 次失败：%s，%ds 后重试",
                    attempt + 1, exc, wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"LLM 调用 3 次全部失败（{self._model_name}）：{last_exc}"
        ) from last_exc

    def chat_with_structure(
        self, messages: list[dict[str, str]], schema: dict[str, Any]
    ) -> dict[str, Any]:
        """用 langchain 的 with_structured_output 做结构化输出。

        比 LLMPort 默认实现更可靠：底层用 OpenAI 的 function calling
        或 response_format 强制 JSON，减少解析失败概率。
        """
        from langchain_core.output_parsers import JsonOutputParser

        try:
            # 用 JSON mode：让 LLM 保证输出合法 JSON
            structured_llm = self._llm.bind(
                response_format={"type": "json_object"}
            )
            lc_messages = [
                ("system" if m["role"] == "system"
                 else "human" if m["role"] == "user"
                 else "ai", m["content"])
                for m in messages
            ]
            schema_hint = (
                "请严格按以下 JSON schema 返回结果，不要输出任何其他内容：\n"
                f"{schema}"
            )
            lc_messages.append(("system", schema_hint))

            resp = structured_llm.invoke(lc_messages)
            import json

            content = resp.content if isinstance(resp.content, str) else str(resp.content)
            return json.loads(content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("结构化输出失败，降级到纯文本模式：%s", exc)
            return super().chat_with_structure(messages, schema)

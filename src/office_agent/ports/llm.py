"""LLM 端口（抽象协议）。

业务节点只依赖本抽象，不知道背后是 langchain-openai、Ollama 还是 Mock。
换供应商只改适配器 + 配置，图和节点代码零改动。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMPort(ABC):
    """对话式 LLM 端口。"""

    @abstractmethod
    def chat(self, messages: list[dict[str, str]]) -> str:
        """发送消息列表，返回纯文本回复。

        messages 格式：[{"role": "system"|"user"|"assistant", "content": "..."}]
        实现需处理超时与重试（由配置驱动），业务层不关心。
        """
        raise NotImplementedError

    def chat_with_structure(
        self, messages: list[dict[str, str]], schema: dict[str, Any]
    ) -> dict[str, Any]:
        """要求 LLM 按 JSON schema 返回结构化结果。

        默认实现：在 system 消息里注入 schema 要求、解析 JSON。
        适配器可覆盖此方法用原生 structured output（如 OpenAI 的
        response_format=json_schema），更可靠。
        """
        # 默认降级：prompt 注入 + JSON 解析
        schema_hint = (
            "请严格按以下 JSON schema 返回，不要输出任何其他内容：\n"
            f"{schema}"
        )
        augmented = [*messages, {"role": "system", "content": schema_hint}]
        raw = self.chat(augmented)
        import json

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # 尝试提取 markdown 代码块中的 JSON
            if "```" in raw:
                block = raw.split("```")[1]
                if block.startswith("json"):
                    block = block[4:]
                return json.loads(block.strip())
            raise

"""LLM 客户端封装（OpenAI 兼容协议）。

设计要点：
- 路由用小模型、导购生成用主模型，成本分层
- 统一的重试与降级：LLM 失败时上层可拿到 None 走兜底话术
- 非流式/流式双接口
"""

from __future__ import annotations

import os
from typing import AsyncIterator

from loguru import logger
from openai import APIError, AsyncOpenAI, RateLimitError
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ["LLM_API_KEY"],
        )
        self.model = os.environ.get("LLM_MODEL", "deepseek-chat")
        self.router_model = os.environ.get("ROUTER_MODEL") or self.model

    async def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float = 0.3,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
    ) -> dict | None:
        """非流式调用。失败返回 None，由上层走降级链。"""
        try:
            kwargs: dict = dict(
                model=model or self.model,
                messages=messages,
                temperature=temperature,
            )
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            if response_format:
                kwargs["response_format"] = response_format
            resp = await self._client.chat.completions.create(**kwargs)
            return resp.choices[0].message.model_dump()
        except (APIError, RateLimitError) as e:
            logger.warning(f"LLM call failed: {e}")
            return None

    async def chat_stream(
        self, messages: list[dict], *, temperature: float = 0.5
    ) -> AsyncIterator[str]:
        """流式输出，逐 token 产出。

        空响应重试：API 偶发返回"流正常结束但零内容"（v2 冒烟测试实测命中），
        零产出时自动重试一次，再退化为非流式调用，避免用户收到兜底话术。
        """
        for attempt in range(2):
            produced = False
            try:
                stream = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    stream=True,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        produced = True
                        yield delta
            except (APIError, RateLimitError) as e:
                logger.warning(f"LLM stream failed (attempt {attempt + 1}): {e}")
            if produced:
                return
            logger.warning(f"LLM stream empty (attempt {attempt + 1}), retrying")
        # 最终降级：非流式补一次
        resp = await self.chat(messages, temperature=temperature)
        if resp and resp.get("content"):
            yield resp["content"]


_llm: LLMClient | None = None


def get_llm() -> LLMClient:
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm

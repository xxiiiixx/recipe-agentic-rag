"""LLM 调用封装。

上游 ``generation_integration.py`` 直接 ``from langchain_moonshot import MoonshotChat`` 并
``os.environ["MOONSHOT_API_KEY"]`` 硬读环境变量，于是：换供应商要改代码，跑测试要真花钱，
无 key 的机器连 import 都过不去。这里退化成 openai 兼容协议（Moonshot / DeepSeek / 通义 /
vLLM / Ollama 都实现了 ``/v1/chat/completions``），配置里换 ``base_url`` 即可。

顺带把 ``usage`` 留下来：tokens per query 是 Agentic RAG 必须报的代价指标之一，
上游完全没有记录，导致"加了反思重试之后贵了多少"无法回答。
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    model: str = "moonshot-v1-8k"
    base_url: str = "https://api.moonshot.cn/v1"
    api_key: str = ""
    api_key_env: str = "LLM_API_KEY"
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout: float = 60.0
    max_retries: int = 2

    def resolve_key(self) -> str:
        key = self.api_key or os.environ.get(self.api_key_env, "")
        if not key:
            raise RuntimeError(
                f"未找到 LLM API key：请设置环境变量 {self.api_key_env}，"
                "或在配置文件的 llm.api_key 中填写。"
            )
        return key


@dataclass
class ChatResult:
    text: str
    model: str
    latency_ms: float
    usage: Dict[str, int] = field(default_factory=dict)


class ChatModel:
    """薄封装。只做三件事：重试、计时、记 usage。"""

    def __init__(self, cfg: LLMConfig) -> None:
        from openai import OpenAI  # 延迟导入，未装 openai 也能 import 本模块

        self.cfg = cfg
        self._client = OpenAI(
            base_url=cfg.base_url, api_key=cfg.resolve_key(), timeout=cfg.timeout
        )

    def chat(self, messages: Sequence[Dict[str, str]], **overrides) -> ChatResult:
        params = {
            "model": overrides.get("model", self.cfg.model),
            "messages": list(messages),
            "temperature": overrides.get("temperature", self.cfg.temperature),
            "max_tokens": overrides.get("max_tokens", self.cfg.max_tokens),
        }
        last_error: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(**params)
                usage = {}
                if getattr(resp, "usage", None) is not None:
                    usage = {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "total_tokens": resp.usage.total_tokens,
                    }
                return ChatResult(
                    text=resp.choices[0].message.content or "",
                    model=params["model"],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    usage=usage,
                )
            except Exception as exc:  # 网络抖动/限流都重试，鉴权失败也重试一次便于暴露真实报错
                last_error = exc
                logger.warning("LLM 调用失败（第 %d 次）：%s", attempt + 1, exc)
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"LLM 调用连续失败：{last_error}")


class MockChatModel:
    """离线占位：不联网，回显上下文首段，用于冒烟与 CI。"""

    cfg = LLMConfig(model="mock")

    def chat(self, messages: Sequence[Dict[str, str]], **overrides) -> ChatResult:
        user = messages[-1]["content"] if messages else ""
        return ChatResult(
            text=f"[mock] 已收到 {len(user)} 字符上下文，未调用真实模型。",
            model="mock",
            latency_ms=0.0,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )


def build_llm(cfg: LLMConfig) -> ChatModel:
    if cfg.model in {"mock", "none"}:
        return MockChatModel()  # type: ignore[return-value]
    return ChatModel(cfg)

from __future__ import annotations

import os
from typing import Any, Optional
from dotenv import load_dotenv

from openai import OpenAI

from .base import BaseLLM

load_dotenv()

class OpenRouterLLM(BaseLLM):
    """
    OpenRouter implementation using the OpenAI-compatible API via the `openai` Python SDK.

    Requires `OPENROUTER_API_KEY` (or `api_key=`).
    """

    def __init__(
        self) -> None:
        super().__init__(key_env_var="OPENROUTER_API_KEY",
        model="openai/gpt-5.2-pro")
        self._client = OpenAI(api_key=self.api_key, base_url="https://openrouter.ai/api/v1", default_headers=self.headers or None)

    def complete(self, prompt: str, *, system: Optional[str] = None, **kwargs: Any) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(model=self.model, messages=messages, **kwargs)
        content = resp.choices[0].message.content
        return (content or "").strip()


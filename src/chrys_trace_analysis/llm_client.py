from __future__ import annotations

import json
import logging

import httpx
from openai import OpenAI

from .config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        if config.no_proxy:
            http_client = httpx.Client(
                transport=httpx.HTTPTransport(retries=0),
                timeout=httpx.Timeout(300.0, connect=10.0),
            )
        else:
            http_client = httpx.Client(
                timeout=httpx.Timeout(300.0, connect=10.0),
            )
        self.client = OpenAI(
            base_url=config.api_endpoint,
            api_key=config.api_key,
            http_client=http_client,
            max_retries=0,
        )

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        logger.info("LLM chat: model=%s, user_prompt_len=%d",
                    self.config.model_name, len(user_prompt))
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        content = response.choices[0].message.content or ""
        logger.info("LLM response length: %d", len(content))
        return content

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict | list:
        raw = self.chat(system_prompt, user_prompt)
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM JSON response: %s", raw[:500])
            raise

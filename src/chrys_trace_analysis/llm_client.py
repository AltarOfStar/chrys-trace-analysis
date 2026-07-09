from __future__ import annotations

import json
import logging
import re

import httpx
from openai import OpenAI

from .config import LLMConfig

logger = logging.getLogger(__name__)


def _fix_common_json_errors(text: str) -> str:
    """Attempt to fix common JSON formatting issues from LLM output."""
    # Remove trailing commas before ] or }
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Remove trailing comma in the last element of a single-line object/array
    text = re.sub(r",\s*$", "", text, flags=re.MULTILINE)
    # Replace single quotes with double quotes, but only outside of already-quoted strings
    # This is a simple approach: replace ' that appear to be JSON keys/values
    text = re.sub(r"(?<!\\)'", '"', text)
    # Remove comments (// and /* */)
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text


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
            raw = "\n".join(lines).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Initial JSON parse failed, attempting fixes...")
            fixed = _fix_common_json_errors(raw)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError as exc:
                logger.error(
                    "Failed to parse LLM JSON response after fixes. "
                    "Error: %s. Position %d. Raw (first 500 chars): %s",
                    exc, exc.pos, raw[:500],
                )
                raise

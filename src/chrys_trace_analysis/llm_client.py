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
    text = re.sub(r"(?<!\\)'", '"', text)
    # Remove comments (// and /* */)
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text


def _salvage_json_fallback(text: str) -> dict | None:
    """Last-resort fallback: extract key fields via regex when JSON is broken.

    Only handles the TurnProblem shape. Returns None if extraction fails.
    """
    session_uuid = None
    turn_index = None
    deviation_action = None
    turn_analysis = None

    m = re.search(r'"session_uuid"\s*:\s*"([^"]+)"', text)
    if m:
        session_uuid = m.group(1)

    m = re.search(r'"turn_index"\s*:\s*(\d+)', text)
    if m:
        turn_index = int(m.group(1))

    m = re.search(r'"deviation_action"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if m:
        deviation_action = m.group(1)

    m = re.search(r'"turn_analysis"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if m:
        turn_analysis = m.group(1)

    if session_uuid and turn_index is not None and turn_analysis:
        result: dict = {
            "session_uuid": session_uuid,
            "turn_index": turn_index,
            "turn_analysis": turn_analysis,
        }
        if deviation_action is not None:
            result["deviation_action"] = deviation_action
        return result

    return None


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
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(raw: str) -> dict | list:
        """Parse LLM response as JSON with progressive repair strategies."""
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        # Strategy 1: Direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Fix common errors and retry
        logger.warning("Initial JSON parse failed, attempting fixes...")
        fixed = _fix_common_json_errors(raw)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # Strategy 3: Try to locate the outermost JSON object and parse just that
        brace_start = raw.find("{")
        if brace_start != -1:
            # Try progressively larger slices from the first { to find valid JSON
            depth = 0
            for i in range(brace_start, len(raw)):
                if raw[i] == "{":
                    depth += 1
                elif raw[i] == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = raw[brace_start : i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            pass
                        try:
                            return json.loads(_fix_common_json_errors(candidate))
                        except json.JSONDecodeError:
                            pass

        # Strategy 4: Regex salvage (extract known fields)
        logger.warning("All JSON repairs failed, attempting regex salvage...")
        salvaged = _salvage_json_fallback(raw)
        if salvaged is not None:
            logger.warning("Regex salvage succeeded, reconstructed partial JSON")
            return salvaged

        logger.error(
            "Failed to parse LLM JSON response after all repair strategies. "
            "Raw (first 500 chars): %s",
            raw[:500],
        )
        raise json.JSONDecodeError("All repair strategies failed", raw, 0)

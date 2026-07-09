from __future__ import annotations

import logging

from .config import PipelineConfig
from .llm_client import LLMClient
from .models import Scenario, Session

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert at analyzing coding agent usage patterns. Your task is to review a batch of conversation trajectories between users and a coding AI assistant, and identify the most common usage scenarios.

For each trajectory, you will see:
- The user's natural language requests
- The assistant's responses
- Which MCP tools were invoked
- Which skills were loaded

Based on these trajectories, identify {min_scenarios} to {max_scenarios} distinct usage scenarios. For each scenario, provide:
1. **name**: A concise label (2-6 words)
2. **description**: A detailed description of what users are trying to accomplish, typical interaction patterns, and distinguishing characteristics

Focus on the user's *intent* and the *nature of the task*, not the technical tools used."""


def _format_sessions(sessions: list[tuple[str, Session]]) -> str:
    parts: list[str] = []
    for i, (user_name, session) in enumerate(sessions, 1):
        header = f"## Session {i} (User: {user_name}, UUID: {session.session_uuid})"
        lines = [header]
        if session.mcp_tools:
            lines.append(f"MCP Tools: {', '.join(session.mcp_tools)}")
        if session.skills:
            lines.append(f"Skills: {', '.join(session.skills)}")
        lines.append("")
        for j, round_ in enumerate(session.session_abstract, 1):
            lines.append(f"**User:** {round_.user_msg}")
            lines.append(f"**Assistant:** {round_.assistant_reply}")
            lines.append("")
        parts.append("\n".join(lines))
    return "\n---\n".join(parts)


def identify_scenarios(
    client: LLMClient,
    sampled: list[tuple[str, Session]],
    config: PipelineConfig,
) -> list[Scenario]:
    system_prompt = SYSTEM_PROMPT.format(
        min_scenarios=config.num_scenarios_min,
        max_scenarios=config.num_scenarios_max,
    )

    user_prompt = f"""Below are {len(sampled)} coding agent conversation trajectories. 

Analyze them and output a JSON array of scenarios. Each scenario object must have "name" and "description" fields.

Output ONLY valid JSON (no markdown, no extra text):

{_format_sessions(sampled)}"""

    logger.info("Identifying scenarios from %d sampled sessions...", len(sampled))
    result = client.chat_json(system_prompt, user_prompt)

    scenarios = [Scenario.model_validate(item) for item in result]
    if not (config.num_scenarios_min <= len(scenarios) <= config.num_scenarios_max):
        logger.warning(
            "Expected %d-%d scenarios, got %d. Proceeding anyway.",
            config.num_scenarios_min, config.num_scenarios_max, len(scenarios),
        )

    logger.info("Identified %d scenarios: %s",
                len(scenarios), [s.name for s in scenarios])
    return scenarios

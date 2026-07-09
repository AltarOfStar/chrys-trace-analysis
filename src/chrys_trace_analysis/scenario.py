from __future__ import annotations

import logging

from .config import PipelineConfig
from .llm_client import LLMClient
from .models import Scenario, ScenarioWithTrace, Session

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert at analyzing coding agent usage patterns. Your task is to review a batch of conversation trajectories between users and a coding AI assistant, and identify the most common usage scenarios.

For each trajectory, you will see:
- The user's natural language requests
- The assistant's responses
- Which MCP tools were invoked
- Which skills were loaded

Identify {min_scenarios} to {max_scenarios} distinct usage scenarios. For each scenario, provide:
1. **name**: A concise label (2-6 words)
2. **description**: A detailed description of what users are trying to accomplish, typical interaction patterns, and distinguishing characteristics
3. **representative_session_uuid**: The UUID of the ONE session from the batch that BEST exemplifies this scenario

Focus on the user's *intent* and the *nature of the task*, not the technical tools used."""

AGGREGATION_SYSTEM_PROMPT = """You are an expert at synthesizing categorization results. You will receive scenario identification results from {num_groups} independent random samples of coding agent conversation trajectories.

Each sample was analyzed independently and produced its own set of scenarios with descriptions and representative traces.

Your task is to synthesize ALL of these results into a single, consolidated set of {min_scenarios} to {max_scenarios} distinct usage scenarios.

Guidelines:
- Merge similar scenarios across groups (e.g. if "Bug Fixing" and "Debugging" appear in multiple groups, unify them).
- Keep scenarios that appear consistently across groups; consider dropping outliers that only appear in one group.
- For each final scenario, write a comprehensive description that captures the consensus across groups.
- The final set should be coherent, non-overlapping, and cover the full spectrum of usage patterns.

Output a JSON array where each element has "name" and "description" fields.
Output ONLY valid JSON (no markdown, no extra text)."""


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


def _identify_scenarios_group(
    client: LLMClient,
    sessions: list[tuple[str, Session]],
    config: PipelineConfig,
    group_index: int,
) -> list[ScenarioWithTrace]:
    """Ask the LLM to identify scenarios from one sampled group, each with a representative trace."""
    system_prompt = SYSTEM_PROMPT.format(
        min_scenarios=config.num_scenarios_min,
        max_scenarios=config.num_scenarios_max,
    )

    user_prompt = f"""Below are {len(sessions)} coding agent conversation trajectories.

Analyze them and output a JSON array of scenarios. Each scenario object must have "name", "description", and "representative_session_uuid" fields.

Output ONLY valid JSON (no markdown, no extra text):

{_format_sessions(sessions)}"""

    logger.info("Group %d: identifying scenarios from %d sessions...",
                group_index, len(sessions))
    result = client.chat_json(system_prompt, user_prompt)

    scenarios = [ScenarioWithTrace.model_validate(item) for item in result]
    logger.info("Group %d: identified %d scenarios: %s",
                group_index, len(scenarios), [s.name for s in scenarios])
    return scenarios


def _format_group_results(
    group_results: list[list[ScenarioWithTrace]],
) -> str:
    parts: list[str] = []
    for i, scenarios in enumerate(group_results, 1):
        lines = [f"## Group {i} results ({len(scenarios)} scenarios)"]
        for s in scenarios:
            lines.append(f"- **{s.name}**: {s.description}")
            lines.append(f"  Representative: {s.representative_session_uuid}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _aggregate_scenarios(
    client: LLMClient,
    group_results: list[list[ScenarioWithTrace]],
    config: PipelineConfig,
) -> list[Scenario]:
    """Ask the LLM to synthesize all group results into a final set of scenarios."""
    num_groups = len(group_results)
    system_prompt = AGGREGATION_SYSTEM_PROMPT.format(
        num_groups=num_groups,
        min_scenarios=config.num_scenarios_min,
        max_scenarios=config.num_scenarios_max,
    )

    user_prompt = f"""Below are the scenario identification results from {num_groups} independent random samples.

Synthesize them into a consolidated set of {config.num_scenarios_min} to {config.num_scenarios_max} scenarios.

Output ONLY valid JSON (no markdown, no extra text):

{_format_group_results(group_results)}"""

    logger.info("Aggregating scenarios from %d groups...", num_groups)
    result = client.chat_json(system_prompt, user_prompt)

    scenarios = [Scenario.model_validate(item) for item in result]
    logger.info("Aggregated to %d scenarios: %s",
                len(scenarios), [s.name for s in scenarios])
    return scenarios


def identify_scenarios(
    client: LLMClient,
    sampled_groups: list[list[tuple[str, Session]]],
    config: PipelineConfig,
) -> list[Scenario]:
    """Multi-group scenario identification with aggregation to reduce randomness.

    1. For each sampled group, ask the LLM to identify scenarios with representative traces.
    2. Aggregate all group results into a final consolidated set of scenarios.
    """
    group_results: list[list[ScenarioWithTrace]] = []
    for i, group_sessions in enumerate(sampled_groups, 1):
        scenarios = _identify_scenarios_group(client, group_sessions, config, group_index=i)
        group_results.append(scenarios)

    return _aggregate_scenarios(client, group_results, config)

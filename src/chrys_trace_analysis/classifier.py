from __future__ import annotations

import logging
import sys

from .llm_client import LLMClient
from .models import ClassifiedSession, Scenario, Session

logger = logging.getLogger(__name__)

_BAR_WIDTH = 40


def _render_progress(processed: int, total: int, batch_num: int, total_batches: int) -> str:
    pct = processed / total if total > 0 else 0
    filled = int(_BAR_WIDTH * pct)
    bar = "█" * filled + "░" * (_BAR_WIDTH - filled)
    return f"\r  [{bar}] {processed}/{total} sessions | batch {batch_num}/{total_batches}"

CLASSIFY_SYSTEM_PROMPT = """You are a classification expert. You will be given a list of coding agent usage scenarios and a batch of conversation trajectories.

For each trajectory, classify it into EXACTLY ONE of the provided scenarios. Output a JSON array where each element has:
- "session_uuid": the session UUID
- "scenario_name": the name of the most fitting scenario

If a trajectory does not clearly fit any scenario, assign it to the closest match. Do NOT create new scenarios.

Output ONLY valid JSON (no markdown, no extra text)."""


def _format_classification_prompt(
    sessions: list[tuple[str, Session]],
    scenarios: list[Scenario],
) -> str:
    scenarios_text = "\n".join(
        f"- **{s.name}**: {s.description}" for s in scenarios
    )

    session_texts: list[str] = []
    for user_name, session in sessions:
        header = f"## Session UUID: {session.session_uuid} (User: {user_name})"
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
        session_texts.append("\n".join(lines))

    return f"""## Scenarios

{scenarios_text}

## Trajectories to classify

{chr(10).join('---' + chr(10) + t for t in session_texts)}"""


def classify_sessions(
    client: LLMClient,
    users_sessions: list[tuple[str, Session]],
    scenarios: list[Scenario],
    batch_size: int,
) -> list[ClassifiedSession]:
    results: list[ClassifiedSession] = []
    total = len(users_sessions)
    scenario_names = {s.name for s in scenarios}
    total_batches = (total + batch_size - 1) // batch_size if total > 0 else 0

    logger.info("Classification queue: %d sessions in %d batches", total, total_batches)

    for batch_num, start in enumerate(range(0, total, batch_size), 1):
        batch = users_sessions[start:start + batch_size]
        processed_so_far = min(start + batch_size, total)
        message = _render_progress(processed_so_far, total, batch_num, total_batches)
        sys.stderr.write(message)
        sys.stderr.flush()

        user_prompt = _format_classification_prompt(batch, scenarios)
        raw = client.chat_json(CLASSIFY_SYSTEM_PROMPT, user_prompt)

        # Build a lookup for session_uuid -> user_name
        uuid_to_user = {s.session_uuid: u for u, s in batch}

        for item in raw:
            item["user_name"] = uuid_to_user.get(item.get("session_uuid", ""), "")
            cs = ClassifiedSession.model_validate(item)
            if cs.scenario_name not in scenario_names:
                logger.warning(
                    "Session %s classified as unknown scenario '%s', "
                    "using first scenario as fallback",
                    cs.session_uuid, cs.scenario_name,
                )
                cs.scenario_name = scenarios[0].name
            results.append(cs)

    sys.stderr.write("\n")
    sys.stderr.flush()

    expected_uuids = {s.session_uuid for _, s in users_sessions}
    got_uuids = {cs.session_uuid for cs in results}
    missing = expected_uuids - got_uuids
    if missing:
        logger.warning("%d sessions not classified: %s", len(missing), missing)

    logger.info("Classified %d sessions into %d scenarios", len(results), len(scenarios))
    return results

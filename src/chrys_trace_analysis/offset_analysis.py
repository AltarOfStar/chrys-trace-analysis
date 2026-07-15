from __future__ import annotations

import json
import logging
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import openai
from httpx import ReadTimeout

from .config import Config
from .loader import load_user_files
from .llm_client import LLMClient
from .models import (
    DeviatedSession,
    DeviationCategory,
    OffsetAnalysisResult,
    RawMessage,
    Session,
    SessionTurn,
    TurnProblem,
)
from .mongo_loader import build_session_turns
from .sampler import flatten_all_sessions

logger = logging.getLogger(__name__)

_BAR_WIDTH = 40
_MAX_RETRIES = 3
_API_URL = "http://lingxi-stats.rnd.huawei.com:8042/api/chrys/query/session"
_FETCH_TIMEOUT = 30

# Keywords that indicate a session was judged as NOT deviated but still output by LLM
_NON_DEVIATION_PATTERNS = [
    "无显著偏离", "无明显偏离", "无明显偏移", "未发生偏离", "未发生偏移",
    "无明显问题", "无偏移", "无偏离", "正常会话", "正常交互", "正常对话",
    "符合预期", "没有偏离", "未偏离", "无问题", "暂无偏离",
    "no deviation", "normal", "no issue",
]


def _is_non_deviation_reason(reason: str) -> bool:
    """Check if a deviation_reason from the LLM actually indicates NO deviation."""
    reason_lower = reason.lower().strip()
    for pattern in _NON_DEVIATION_PATTERNS:
        if pattern.lower() in reason_lower:
            return True
    return False


def _fetch_session_json(uuid: str) -> dict | None:
    """Fetch full session JSON from the HTTP API. Returns parsed dict or None on failure."""
    payload = json.dumps({
        "filter_key": "uuid",
        "filter_value": uuid,
    }).encode("utf-8")

    req = urllib.request.Request(
        _API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to fetch session %s from API: %s", uuid, exc)
        return None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DETECTION_SYSTEM_PROMPT = """You are an expert at analyzing coding agent execution quality. You will be given a batch of conversation trajectories between users and a coding AI assistant.

For each trajectory, determine whether the agent's execution **deviated from the user's intent**. Pay special attention to:
- Whether the user had to **correct** or **re-direct** the agent's behavior
- Whether the agent misunderstood the user's requirements and produced wrong results
- Whether the agent went off on a tangent unrelated to the user's request
- Whether the user expressed frustration or had to repeat themselves

For each session that shows deviation, provide:
1. The session UUID
2. A concise reason (in Chinese) explaining what went wrong and why
3. The **turn number** (1-indexed) where the problem FIRST appeared or became most pronounced. Look at the conversation turns (numbered in each session) and identify which turn shows the deviation starting. If the deviation spans multiple turns, pick the earliest one.

IMPORTANT: Output a JSON array containing ONLY the deviated sessions. Do NOT output entries for sessions that are normal or have no deviation — simply exclude them. Do NOT output entries with reasons like "无显著偏离", "正常会话", "无明显问题", "未发生偏离" etc. If a session is normal, omit it.

Output a JSON array. Each element must have:
- "session_uuid": the session UUID
- "deviation_reason": a concise explanation of the deviation (in Chinese)
- "problematic_turn_index": the 1-indexed turn number where the deviation first appeared (integer)

If no sessions in the batch show deviation, output an empty JSON array: []

Output ONLY valid JSON (no markdown, no extra text)."""

CATEGORIZATION_SYSTEM_PROMPT = """You are an expert at categorizing failure patterns in coding agent interactions. You will be given a list of deviated conversation trajectories, each with a UUID, user name, deviation reason, and the full conversation abstract.

Your task is to group these deviations into **distinct root-cause categories**. For each category, provide:
1. **category_name**: A concise label (2-6 words, in Chinese)
2. **description**: A detailed description of this type of deviation, common patterns, and typical manifestations
3. **session_uuids**: A list of ALL session UUIDs that belong to this category

Guidelines:
- Each session should appear in EXACTLY ONE category
- Categories should be MECE (mutually exclusive, collectively exhaustive)
- Focus on the ROOT CAUSE, not the symptom (e.g. "模型未理解需求意图" rather than just "用户不满意")
- Aim for 3-8 categories depending on the data

Output a JSON array of category objects.
Output ONLY valid JSON (no markdown, no extra text)."""

CONDENSE_SYSTEM_PROMPT = """You are an expert at synthesizing and condensing categorization results. You will receive deviation category data from analysis of coding agent conversation trajectories.

Your task is to review these categories and condense them into a **small, representative set** of root-cause categories. Think of this as a final refinement pass — the goal is to produce the most meaningful, high-level taxonomy of deviation patterns.

Guidelines:
- Merge similar or overlapping categories into broader, more representative ones
- Each final category should be distinct and meaningful — avoid categories that are too narrow or too vague
- For each final category, write a comprehensive, well-crafted description that captures the essence of all merged sub-categories
- Re-assign all session_uuids to the correct merged category — every session must appear in exactly one category
- You MUST produce 4 to 6 categories. If you have more, merge aggressively. If you have fewer, split one broad category.
- The final set must be mutually exclusive and collectively exhaustive

Output a JSON array where each element has "category_name", "description", and "session_uuids" fields.
Output ONLY valid JSON (no markdown, no extra text)."""

CONDENSE_SECOND_PROMPT = """You are an expert at the FINAL refinement of categorization. You will receive a set of deviation categories that have already gone through one round of condensation — but there may still be categories that are essentially the same, just expressed with slightly different wording.

Your task is a rigorous second-pass merge:
- If two categories describe the same root cause with only superficial wording differences, MERGE them into one
- Be aggressive about merging — categories that differ only in nuance or phrasing should be combined
- For each merged category, write a description that captures BOTH the common theme and the specific variations
- Re-assign all session_uuids — every session must appear in exactly one final category
- You MUST produce exactly 4 to 6 categories as the FINAL result. If you have more than 6, merge the least populated ones together. If you have fewer than 4, split the largest one.
- No category should have only 1 session — merge singletons into the nearest broader category

Output a JSON array where each element has "category_name", "description", and "session_uuids" fields.
Output ONLY valid JSON (no markdown, no extra text)."""

TURN_ANALYSIS_SYSTEM_PROMPT = """You are an expert at diagnosing coding agent failures at the individual turn level. You will be given a specific conversational turn from a session where the agent deviated from user intent, along with the deviation reason.

The turn contains the FULL raw messages including:
- User messages (what the user asked)
- Assistant text replies (what the agent said)
- Tool call messages (what tools the agent invoked)
- Tool result messages (what the tools returned)

Your task is to analyze this turn in detail and identify:
1. **Which specific messages within the turn are problematic** — refer to them by their 1-indexed position within the turn (e.g., "message 3: the agent called read_file with the wrong path")
2. **What exactly went wrong** — be specific about the error, misunderstanding, or misstep
3. **Why it happened** — what was the root cause at the message level (e.g., the agent ignored a constraint the user specified, the agent assumed a wrong file path, the agent called a tool with incorrect arguments, etc.)

Focus on actionable, precise observations. Do not restate the overall deviation reason — dig into the specific messages.

Output a JSON object:
- "problematic_message_indices": array of 1-indexed message position numbers within the turn that are problematic
- "turn_analysis": detailed analysis string in Chinese explaining what went wrong and why

Output ONLY valid JSON (no markdown, no extra text)."""

# ---------------------------------------------------------------------------
# Session formatting
# ---------------------------------------------------------------------------


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
            lines.append(f"### Turn {j}")
            lines.append(f"**User:** {round_.user_msg}")
            lines.append(f"**Assistant:** {round_.assistant_reply}")
            lines.append("")
        parts.append("\n".join(lines))
    return "\n---\n".join(parts)


def _format_deviations_for_categorization(
    deviated: list[DeviatedSession],
    sessions_map: dict[str, tuple[str, Session]],
) -> str:
    parts: list[str] = []
    for i, ds in enumerate(deviated, 1):
        entry = sessions_map.get(ds.session_uuid)
        if entry is None:
            lines = [
                f"## Deviation {i}",
                f"Session UUID: {ds.session_uuid}",
                f"User: {ds.user_name}",
                f"Deviation Reason: {ds.deviation_reason}",
                "(Session data not found)",
            ]
        else:
            user_name, session = entry
            lines = [
                f"## Deviation {i}",
                f"Session UUID: {ds.session_uuid}",
                f"User: {user_name}",
                f"Deviation Reason: {ds.deviation_reason}",
                f"Problematic Turn: {ds.problematic_turn_index}",
                "",
                "### Conversation:",
            ]
            for j, round_ in enumerate(session.session_abstract, 1):
                lines.append(f"**Turn {j} - User:** {round_.user_msg}")
                lines.append(f"**Turn {j} - Assistant:** {round_.assistant_reply}")
                lines.append("")
        parts.append("\n".join(lines))
    return "\n---\n".join(parts)


def _format_raw_message(msg: RawMessage, index: int) -> str:
    """Format a single raw message for turn-level analysis."""
    role_label = {"user": "User", "assistant": "Assistant", "system": "System", "tool": "Tool"}
    role_display = role_label.get(msg.role, msg.role.capitalize())

    parts = [f"[{index}] **{role_display}**:"]
    for c in msg.contents:
        if c.type == "text" and c.text:
            parts.append(f"  text: {c.text}")
        elif c.type in ("function_call", "shell_tool_call", "mcp_server_tool_call"):
            parts.append(f"  tool_call → {c.tool_name}({c.arguments})")
        elif c.type in ("function_result", "shell_tool_result", "mcp_server_tool_result"):
            truncated = c.result[:500] + "..." if len(c.result) > 500 else c.result
            parts.append(f"  tool_result: {truncated}")
        elif c.type == "error":
            parts.append(f"  error: {c.text}")
    return "\n".join(parts)


def _format_turn_for_analysis(
    ds: DeviatedSession,
    turn: SessionTurn,
) -> str:
    """Format a single turn's raw messages for LLM analysis."""
    lines = [
        f"Session UUID: {ds.session_uuid}",
        f"User: {ds.user_name}",
        f"Deviation Reason: {ds.deviation_reason}",
        f"Turn {turn.turn_index} ({len(turn.messages)} messages):",
        "",
    ]
    for i, msg in enumerate(turn.messages, 1):
        lines.append(_format_raw_message(msg, i))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------


def _render_progress(processed: int, total: int, batch_num: int, total_batches: int) -> str:
    pct = processed / total if total > 0 else 0
    filled = int(_BAR_WIDTH * pct)
    bar = "█" * filled + "░" * (_BAR_WIDTH - filled)
    return f"\r  [{bar}] {processed}/{total} sessions | batch {batch_num}/{total_batches}"

# ---------------------------------------------------------------------------
# Step 1: Deviation detection
# ---------------------------------------------------------------------------


def _detect_deviations_batch(
    client: LLMClient,
    batch: list[tuple[str, Session]],
    batch_num: int,
    retries: int = _MAX_RETRIES,
) -> list[DeviatedSession] | None:
    uuid_to_user = {s.session_uuid: u for u, s in batch}

    for attempt in range(1, retries + 1):
        try:
            user_prompt = f"""Below are {len(batch)} coding agent conversation trajectories.

Analyze each trajectory and output a JSON array of sessions that show agent deviation from user intent.
For each deviated session, include the turn number where the problem first appears.

Output ONLY valid JSON (no markdown, no extra text):

{_format_sessions(batch)}"""

            raw = client.chat_json(DETECTION_SYSTEM_PROMPT, user_prompt)
        except (ReadTimeout, openai.APITimeoutError) as exc:
            logger.warning(
                "Detection batch %d timed out (attempt %d/%d): %s",
                batch_num, attempt, retries, exc,
            )
            continue
        except Exception as exc:
            logger.warning(
                "Detection batch %d failed (attempt %d/%d): %s",
                batch_num, attempt, retries, exc,
            )
            continue

        try:
            results: list[DeviatedSession] = []
            for item in raw:
                reason = item.get("deviation_reason", "")
                if _is_non_deviation_reason(reason):
                    logger.debug("Filtered non-deviation result: session=%s reason=%s",
                                 item.get("session_uuid", ""), reason)
                    continue
                item["user_name"] = uuid_to_user.get(item.get("session_uuid", ""), "")
                ds = DeviatedSession.model_validate(item)
                results.append(ds)
            return results
        except Exception as exc:
            logger.warning(
                "Detection batch %d parse failed (attempt %d/%d): %s",
                batch_num, attempt, retries, exc,
            )
            continue

    logger.error("Detection batch %d failed after %d retries, skipping %d sessions.",
                 batch_num, retries, len(batch))
    return None


def detect_deviations(
    client: LLMClient,
    all_sessions: list[tuple[str, Session]],
    batch_size: int,
) -> list[DeviatedSession]:
    """Step 1: Detect sessions where agent execution deviated from user intent."""
    results: list[DeviatedSession] = []
    total = len(all_sessions)
    total_batches = (total + batch_size - 1) // batch_size if total > 0 else 0

    logger.info("Deviation detection: %d sessions in %d batches", total, total_batches)

    for batch_num, start in enumerate(range(0, total, batch_size), 1):
        batch = all_sessions[start:start + batch_size]
        processed_so_far = min(start + batch_size, total)
        message = _render_progress(processed_so_far, total, batch_num, total_batches)
        sys.stderr.write(message)
        sys.stderr.flush()

        batch_results = _detect_deviations_batch(client, batch, batch_num)
        if batch_results is not None:
            results.extend(batch_results)

    sys.stderr.write("\n")
    sys.stderr.flush()

    deviation_rate = len(results) / total * 100 if total > 0 else 0
    logger.info("Detected %d deviated sessions out of %d (%.1f%%)",
                len(results), total, deviation_rate)
    return results


# ---------------------------------------------------------------------------
# Step 2: Deviation cause categorization
# ---------------------------------------------------------------------------


def _categorize_deviations_batch(
    client: LLMClient,
    batch: list[DeviatedSession],
    sessions_map: dict[str, tuple[str, Session]],
    batch_num: int,
    retries: int = _MAX_RETRIES,
) -> list[dict] | None:
    """Categorize one batch of deviated sessions. Returns raw parsed JSON."""
    for attempt in range(1, retries + 1):
        try:
            user_prompt = f"""Below are {len(batch)} deviated conversation trajectories.

Group them into distinct root-cause categories. Output a JSON array of category objects, each with
"category_name", "description", and "session_uuids" (list of all session UUIDs in this category).

Output ONLY valid JSON (no markdown, no extra text):

{_format_deviations_for_categorization(batch, sessions_map)}"""

            raw = client.chat_json(CATEGORIZATION_SYSTEM_PROMPT, user_prompt)
            return raw
        except (ReadTimeout, openai.APITimeoutError) as exc:
            logger.warning(
                "Categorization batch %d timed out (attempt %d/%d): %s",
                batch_num, attempt, retries, exc,
            )
            continue
        except Exception as exc:
            logger.warning(
                "Categorization batch %d failed (attempt %d/%d): %s",
                batch_num, attempt, retries, exc,
            )
            continue

    logger.error("Categorization batch %d failed after %d retries.", batch_num, retries)
    return None


def _parse_categories(raw: list[dict]) -> list[DeviationCategory]:
    categories: list[DeviationCategory] = []
    for item in raw:
        # Ensure session_count matches the session_uuids list
        uuids: list[str] = item.get("session_uuids", [])
        item["session_count"] = len(uuids)
        categories.append(DeviationCategory.model_validate(item))
    return categories


def _format_categories_for_condense(
    batch_results: list[list[dict]],
) -> str:
    """Format all batch categorization results for the final condensation pass."""
    parts: list[str] = []
    if len(batch_results) == 1:
        # Single batch: present as a flat list of categories
        lines = [f"## Initial categories ({len(batch_results[0])} total)"]
        for cat in batch_results[0]:
            lines.append(f"- **{cat.get('category_name', '')}**: {cat.get('description', '')}")
            uuids = cat.get("session_uuids", [])
            lines.append(f"  Sessions ({len(uuids)}): {', '.join(uuids)}")
        parts.append("\n".join(lines))
    else:
        for i, categories in enumerate(batch_results, 1):
            lines = [f"## Batch {i} categories ({len(categories)} total)"]
            for cat in categories:
                lines.append(f"- **{cat.get('category_name', '')}**: {cat.get('description', '')}")
                uuids = cat.get("session_uuids", [])
                lines.append(f"  Sessions ({len(uuids)}): {', '.join(uuids)}")
            parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _format_deviation_categories(categories: list[DeviationCategory]) -> str:
    """Format DeviationCategory list for the second condensation pass."""
    lines = [f"## Categories after first condensation ({len(categories)} total)"]
    for cat in categories:
        lines.append(f"- **{cat.category_name}** ({cat.session_count} sessions): {cat.description}")
        lines.append(f"  Sessions: {', '.join(cat.session_uuids)}")
    return "\n".join(lines)


def _condense_categories(
    client: LLMClient,
    batch_results: list[list[dict]],
) -> list[DeviationCategory]:
    """Two-round LLM condensation to merge near-identical categories into representative set."""
    # Round 1: initial condensation
    formatted = _format_categories_for_condense(batch_results)
    user_prompt = f"""Below are deviation categories identified from coding agent conversation analysis.

Review them and condense into a small, representative set of root-cause categories.

Output ONLY valid JSON (no markdown, no extra text):

{formatted}"""

    try:
        raw = client.chat_json(CONDENSE_SYSTEM_PROMPT, user_prompt)
    except Exception as exc:
        logger.error("Round 1 condensation failed: %s. Falling back to raw categories.", exc)
        all_categories: list[DeviationCategory] = []
        for batch_raw in batch_results:
            all_categories.extend(_parse_categories(batch_raw))
        return all_categories

    round1 = _parse_categories(raw)
    logger.info("Round 1 condensed to %d categories: %s",
                len(round1), [c.category_name for c in round1])

    # Round 2: aggressive merge of near-identical categories
    if len(round1) <= 2:
        logger.info("Only %d categories after round 1, skipping round 2.", len(round1))
        return round1

    formatted2 = _format_deviation_categories(round1)
    user_prompt2 = f"""Below are deviation categories after a first round of condensation. Some may still be near-duplicates with different wording.

Merge them aggressively into the FINAL set of root-cause categories.

Output ONLY valid JSON (no markdown, no extra text):

{formatted2}"""

    try:
        raw2 = client.chat_json(CONDENSE_SECOND_PROMPT, user_prompt2)
    except Exception as exc:
        logger.error("Round 2 condensation failed: %s. Returning round 1 results.", exc)
        return round1

    round2 = _parse_categories(raw2)
    logger.info("Round 2 condensed to %d categories: %s",
                len(round2), [c.category_name for c in round2])
    return round2


def categorize_deviations(
    client: LLMClient,
    deviated: list[DeviatedSession],
    all_sessions: list[tuple[str, Session]],
    batch_size: int,
) -> list[DeviationCategory]:
    """Step 2: Categorize deviation causes, then always condense into representative categories via LLM."""
    if not deviated:
        logger.info("No deviated sessions to categorize.")
        return []

    sessions_map: dict[str, tuple[str, Session]] = {
        s.session_uuid: (u, s) for u, s in all_sessions
    }

    total = len(deviated)
    total_batches = (total + batch_size - 1) // batch_size if total > 0 else 0
    logger.info("Deviation categorization: %d sessions in %d batches", total, total_batches)

    # Phase A: Categorize (batched if needed)
    all_batch_results: list[list[dict]] = []
    for batch_num, start in enumerate(range(0, total, batch_size), 1):
        batch = deviated[start:start + batch_size]
        logger.info("Categorizing batch %d/%d (%d sessions)",
                    batch_num, total_batches, len(batch))
        raw = _categorize_deviations_batch(client, batch, sessions_map, batch_num)
        if raw is not None:
            all_batch_results.append(raw)

    if not all_batch_results:
        logger.warning("All categorization batches failed.")
        return []

    # Phase B: Always condense into representative categories
    logger.info("Condensing %d batch results into representative categories...",
                len(all_batch_results))
    return _condense_categories(client, all_batch_results)


# ---------------------------------------------------------------------------
# Step 3: Turn-level detailed analysis
# ---------------------------------------------------------------------------


def _analyze_single_turn(
    client: LLMClient,
    ds: DeviatedSession,
    turn: SessionTurn,
    index: int,
    total: int,
    retries: int = _MAX_RETRIES,
) -> TurnProblem | None:
    """Analyze a single problematic turn, one LLM call per turn."""
    formatted = _format_turn_for_analysis(ds, turn)
    strict_suffix = (
        "\n\nCRITICAL: Your response MUST be a single, valid JSON object. "
        "Do NOT include any markdown formatting, code fences, or extra text. "
        "The 'turn_analysis' string MUST NOT contain unescaped newlines or "
        "characters that would break JSON parsing. Use \\n for line breaks inside strings."
    )
    user_prompt = f"""Below is a problematic conversational turn from a session where the agent deviated from user intent.

Analyze the raw messages in detail and identify:
- Which specific messages (by their 1-indexed position) are problematic
- What exactly went wrong and why

Output a JSON object:
- "session_uuid": "{ds.session_uuid}"
- "turn_index": {turn.turn_index}
- "problematic_message_indices": array of 1-indexed message positions that are problematic
- "turn_analysis": detailed analysis string in Chinese

Output ONLY valid JSON (no markdown, no extra text):

{formatted}"""

    for attempt in range(1, retries + 1):
        try:
            prompt = user_prompt
            if attempt > 1:
                prompt = user_prompt + strict_suffix
            raw = client.chat_json(TURN_ANALYSIS_SYSTEM_PROMPT, prompt)
            tp = TurnProblem.model_validate(raw)
            tp.user_name = ds.user_name
            return tp
        except (ReadTimeout, openai.APITimeoutError) as exc:
            logger.warning(
                "Turn analysis %d/%d timed out (attempt %d/%d): %s",
                index, total, attempt, retries, exc,
            )
            continue
        except Exception as exc:
            logger.warning(
                "Turn analysis %d/%d failed (attempt %d/%d): %s",
                index, total, attempt, retries, exc,
            )
            continue

    logger.error("Turn analysis %d/%d failed after %d retries.", index, total, retries)
    return None


def analyze_turn_deviations(
    client: LLMClient,
    deviated: list[DeviatedSession],
    turn_problems_dir: Path | None = None,
) -> list[tuple[SessionTurn, TurnProblem]]:
    """Step 3: For each deviated session with a problematic turn, fetch full session
    JSON via HTTP API, parse turns, and run LLM analysis on the target turn's messages.

    When *turn_problems_dir* is provided, each turn's per-session JSON file is written
    immediately after analysis, so partial results survive a crash mid-pipeline.

    Returns list of (turn, TurnProblem) tuples for aggregate file saving.
    """
    if not deviated:
        logger.info("No deviated sessions to analyze at turn level.")
        return []

    # Collect (DeviatedSession, SessionTurn) pairs — fetch from HTTP API
    analysis_pairs: list[tuple[DeviatedSession, SessionTurn]] = []
    skipped_no_index = 0
    skipped_fetch_failed = 0
    skipped_no_turn = 0

    for ds in deviated:
        if ds.problematic_turn_index is None:
            skipped_no_index += 1
            continue

        # Fetch full session JSON from HTTP API
        doc = _fetch_session_json(ds.session_uuid)
        if doc is None:
            skipped_fetch_failed += 1
            continue

        messages = doc.get("messages", [])
        if not isinstance(messages, list) or not messages:
            skipped_fetch_failed += 1
            continue

        turns = build_session_turns(messages)

        # Find the matching turn
        target_turn = None
        for turn in turns:
            if turn.turn_index == ds.problematic_turn_index:
                target_turn = turn
                break

        if target_turn is None:
            skipped_no_turn += 1
            logger.debug(
                "Session %s: problematic_turn_index=%d but turn not found in %d turns",
                ds.session_uuid, ds.problematic_turn_index, len(turns),
            )
            continue

        analysis_pairs.append((ds, target_turn))

    logger.info(
        "Turn analysis: %d pairs collected "
        "(%d skipped: no index, %d skipped: fetch failed, %d skipped: no turn)",
        len(analysis_pairs), skipped_no_index, skipped_fetch_failed, skipped_no_turn,
    )

    if not analysis_pairs:
        return []

    total = len(analysis_pairs)
    logger.info("Turn analysis: %d turns to analyze (one turn per LLM call)", total)

    if turn_problems_dir is not None:
        turn_problems_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[SessionTurn, TurnProblem]] = []
    for i, (ds, turn) in enumerate(analysis_pairs, 1):
        message = _render_progress(i, total, i, total)
        sys.stderr.write(message)
        sys.stderr.flush()

        tp = _analyze_single_turn(client, ds, turn, index=i, total=total)
        if tp is not None:
            results.append((turn, tp))
            # Write per-session file immediately so partial results survive a crash
            if turn_problems_dir is not None:
                _write_turn_problem_file(turn_problems_dir, turn, tp)

    sys.stderr.write("\n")
    sys.stderr.flush()

    logger.info("Turn analysis complete: %d turn problems analyzed", len(results))
    return results


def _write_turn_problem_file(
    turn_problems_dir: Path,
    turn: SessionTurn,
    tp: TurnProblem,
) -> None:
    """Write a single turn problem's per-session JSON file."""
    turn_data = {
        "session_uuid": tp.session_uuid,
        "user_name": tp.user_name,
        "turn_index": tp.turn_index,
        "messages": [msg.model_dump() for msg in turn.messages],
        "analysis": tp.model_dump(),
    }
    (turn_problems_dir / f"{tp.session_uuid}.json").write_text(
        json.dumps(turn_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_offset(config: Config, start_step: int = 1) -> OffsetAnalysisResult:
    users = load_user_files(config.paths.data_dir)
    if not users:
        raise RuntimeError(f"No valid user data files found in {config.paths.data_dir}")

    all_sessions = flatten_all_sessions(users)
    logger.info("Loaded %d sessions from %d users", len(all_sessions), len(users))

    client = LLMClient(config.llm)

    offset_dir = config.paths.output_dir / "offset_analysis"
    offset_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Deviation detection (or load from previous run)
    deviated_file = offset_dir / "deviated_sessions.json"
    if start_step <= 1:
        deviated = detect_deviations(
            client, all_sessions, config.offset_pipeline.detection_batch_size,
        )
        deviated_file.write_text(
            json.dumps([ds.model_dump() for ds in deviated], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved %d deviated sessions to %s", len(deviated), deviated_file)
    else:
        if not deviated_file.exists():
            raise FileNotFoundError(
                f"Cannot skip Step 1: intermediate file not found at {deviated_file}"
            )
        deviated_raw = json.loads(deviated_file.read_text(encoding="utf-8"))
        deviated = [DeviatedSession.model_validate(d) for d in deviated_raw]
        logger.info("Loaded %d deviated sessions from %s (step 1 skipped)",
                    len(deviated), deviated_file)

    # Step 2: Categorization (or load from previous run)
    categories_file = offset_dir / "categories.json"
    if start_step <= 2:
        categories = categorize_deviations(
            client, deviated, all_sessions,
            config.offset_pipeline.categorization_batch_size,
        )
        categories_file.write_text(
            json.dumps([c.model_dump() for c in categories], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved %d categories to %s", len(categories), categories_file)
    else:
        if not categories_file.exists():
            raise FileNotFoundError(
                f"Cannot skip Step 2: intermediate file not found at {categories_file}"
            )
        categories_raw = json.loads(categories_file.read_text(encoding="utf-8"))
        categories = [DeviationCategory.model_validate(c) for c in categories_raw]
        logger.info("Loaded %d categories from %s (step 2 skipped)",
                    len(categories), categories_file)

    # Step 3: Turn-level detailed analysis via HTTP API (one turn per LLM call)
    # Per-session JSON files are written immediately inside analyze_turn_deviations.
    turn_problems_dir = offset_dir / "turn_problems"
    turn_results = analyze_turn_deviations(
        client, deviated, turn_problems_dir=turn_problems_dir,
    )

    # Save aggregate turn_problems.json (single write at the end)
    turn_problems = [tp for _, tp in turn_results]
    (offset_dir / "turn_problems.json").write_text(
        json.dumps([tp.model_dump() for tp in turn_problems], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved %d turn problems to %s", len(turn_problems),
                offset_dir / "turn_problems.json")

    # Build summary
    category_counter = Counter()
    for cat in categories:
        category_counter[cat.category_name] = cat.session_count

    sessions_with_turn_index = sum(1 for ds in deviated if ds.problematic_turn_index is not None)

    summary = {
        "total_users": len(users),
        "total_sessions": len(all_sessions),
        "deviated_count": len(deviated),
        "deviation_rate": f"{len(deviated) / len(all_sessions) * 100:.1f}%" if all_sessions else "0%",
        "categories_count": len(categories),
        "per_category_distribution": dict(category_counter),
        "sessions_with_problematic_turn": sessions_with_turn_index,
        "turn_problems_analyzed": len(turn_problems),
    }

    return OffsetAnalysisResult(
        deviated_sessions=deviated,
        categories=categories,
        summary=summary,
        turn_problems=turn_problems,
    )

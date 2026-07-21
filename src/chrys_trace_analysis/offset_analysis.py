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
    OffsetAnalysisResult,
    RawMessage,
    Session,
    SessionRound,
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

For each deviated session, classify it into **ONE** of the following 5 categories:

1. **需求理解偏差与指令遵循失败** — 未能准确理解用户核心意图、业务逻辑或明确约束，导致执行错误任务、提供不符合预期的解决方法或违反规范。

2. **执行幻觉与虚假反馈** — 声称任务已完成、文件已修改或操作已生效，但实际未执行、执行失败或结果与事实不符，导致用户无法验证修改正确性。

3. **代码逻辑缺陷与生成错误** — 生成的内容存在语法错误、逻辑漏洞、编译失败或不符合规范。

4. **系统异常与执行中断** — 因底层问题导致任务无法完成或会话异常中止，包括执行中断、HTTP错误、上下文超出限制、资源限制、网络超时、会话数据缺失等。

5. **其他原因导致的偏差** — 需额外说明具体原因。

Output a JSON array. Each element must have:
- "session_uuid": the session UUID
- "category": EXACTLY one of the 5 category strings above (copy them exactly)

Output ONLY valid JSON (no markdown, no extra text)."""

TURN_ANALYSIS_SYSTEM_PROMPT = """You are an expert at diagnosing coding agent failures at the individual turn level. You will be given a specific conversational turn from a session where the agent deviated from user intent, along with a description of the problem in this turn.

The turn contains the FULL raw messages including:
- User messages (what the user asked)
- Assistant text replies (what the agent said)
- Tool call messages (what tools the agent invoked)
- Tool result messages (what the tools returned)

Your task is to analyze this turn in detail and identify:
1. **deviation_action**: Which specific operation or action within this turn caused the deviation — be specific (e.g., "第3条消息调用了write_file但使用了错误的文件路径", "第2条消息中助手声称已修改文件但实际未执行任何工具调用")
2. **turn_analysis**: A detailed analysis in Chinese explaining what exactly went wrong, why it happened, and how the deviation manifested

Focus on actionable, precise observations. Do not restate the overall problem description — dig into the specific messages and actions.

Output a JSON object:
- "deviation_action": string describing the specific operation/action that caused the deviation
- "turn_analysis": detailed analysis string in Chinese

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
    """Format a single raw message for turn-level analysis. Skips empty fields."""
    role_label = {"user": "User", "assistant": "Assistant", "system": "System", "tool": "Tool"}
    role_display = role_label.get(msg.role, msg.role.capitalize())

    content_lines: list[str] = []
    for c in msg.contents:
        if c.type == "text" and c.text:
            content_lines.append(f"  text: {c.text}")
        elif c.type in ("function_call", "shell_tool_call", "mcp_server_tool_call"):
            if c.tool_name:
                args = f"({c.arguments})" if c.arguments else "()"
                content_lines.append(f"  tool_call → {c.tool_name}{args}")
        elif c.type in ("function_result", "shell_tool_result", "mcp_server_tool_result"):
            if c.result:
                truncated = c.result[:500] + "..." if len(c.result) > 500 else c.result
                content_lines.append(f"  tool_result: {truncated}")
        elif c.type == "error" and c.text:
            content_lines.append(f"  error: {c.text}")

    if not content_lines:
        content_lines.append("  (no content)")

    return "\n".join([f"[{index}] **{role_display}**:"] + content_lines)


def _format_turn_for_analysis(
    session_uuid: str,
    user_name: str,
    deviation_reason: str,
    turn: SessionTurn,
    session_abstract: list[SessionRound],
) -> str:
    """Format a single turn's raw messages for LLM analysis,
    including the full session conversation abstract for context."""
    lines = [
        f"Session UUID: {session_uuid}",
        f"User: {user_name}",
        f"Deviation Reason: {deviation_reason}",
        "",
        "## Full Conversation Summary (all turns, abbreviated)",
    ]
    for j, round_ in enumerate(session_abstract, 1):
        lines.append(f"### Turn {j}")
        lines.append(f"**User:** {round_.user_msg}")
        lines.append(f"**Assistant:** {round_.assistant_reply}")
        lines.append("")

    lines += [
        "## Detailed Turn (problematic turn with raw messages):",
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
# Step 2: Deviation categorization into 5 fixed categories
# ---------------------------------------------------------------------------


def _categorize_deviations_batch(
    client: LLMClient,
    batch: list[DeviatedSession],
    sessions_map: dict[str, tuple[str, Session]],
    batch_num: int,
    retries: int = _MAX_RETRIES,
) -> dict[str, str] | None:
    """Categorize one batch of deviated sessions into 5 fixed categories.
    Returns dict mapping session_uuid -> category string, or None on failure.
    """
    for attempt in range(1, retries + 1):
        try:
            user_prompt = f"""Below are {len(batch)} deviated conversation trajectories.

Classify each into ONE of the 5 categories. Output a JSON array with "session_uuid" and "category" per entry.

Output ONLY valid JSON (no markdown, no extra text):

{_format_deviations_for_categorization(batch, sessions_map)}"""

            raw = client.chat_json(CATEGORIZATION_SYSTEM_PROMPT, user_prompt)
            mapping: dict[str, str] = {}
            for item in raw:
                uuid = item.get("session_uuid", "")
                cat = item.get("category", "")
                if uuid and cat:
                    mapping[uuid] = cat
            return mapping
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


def categorize_deviations(
    client: LLMClient,
    deviated: list[DeviatedSession],
    all_sessions: list[tuple[str, Session]],
    batch_size: int,
) -> list[DeviatedSession]:
    """Step 2: Classify each deviated session into one of 5 fixed categories."""
    if not deviated:
        logger.info("No deviated sessions to categorize.")
        return deviated

    sessions_map: dict[str, tuple[str, Session]] = {
        s.session_uuid: (u, s) for u, s in all_sessions
    }

    total = len(deviated)
    total_batches = (total + batch_size - 1) // batch_size if total > 0 else 0
    logger.info("Deviation categorization: %d sessions in %d batches", total, total_batches)

    all_mappings: dict[str, str] = {}
    for batch_num, start in enumerate(range(0, total, batch_size), 1):
        batch = deviated[start:start + batch_size]
        logger.info("Categorizing batch %d/%d (%d sessions)",
                    batch_num, total_batches, len(batch))
        mapping = _categorize_deviations_batch(client, batch, sessions_map, batch_num)
        if mapping is not None:
            all_mappings.update(mapping)

    # Apply categories back to deviated sessions
    for ds in deviated:
        if ds.session_uuid in all_mappings:
            ds.category = all_mappings[ds.session_uuid]

    categorized_count = sum(1 for ds in deviated if ds.category)
    logger.info("Categorized %d/%d deviated sessions", categorized_count, len(deviated))
    return deviated


# ---------------------------------------------------------------------------
# Step 3: Turn-level detailed analysis
# ---------------------------------------------------------------------------


def _analyze_single_turn(
    client: LLMClient,
    session_uuid: str,
    user_name: str,
    deviation_reason: str,
    turn: SessionTurn,
    session_abstract: list[SessionRound],
    index: int,
    total: int,
    retries: int = _MAX_RETRIES,
) -> TurnProblem | None:
    """Analyze a single problematic turn, one LLM call per turn."""
    formatted = _format_turn_for_analysis(
        session_uuid, user_name, deviation_reason, turn, session_abstract,
    )
    strict_suffix = (
        "\n\nCRITICAL: Your response MUST be a single, valid JSON object. "
        "Do NOT include any markdown formatting, code fences, or extra text. "
        "The 'turn_analysis' string MUST NOT contain unescaped newlines or "
        "characters that would break JSON parsing. Use \\n for line breaks inside strings."
    )
    user_prompt = f"""Below is a problematic conversational turn from a session where the agent deviated from user intent.

Analyze the raw messages in detail and identify:
- Which specific operation/action caused the deviation (deviation_action)
- What exactly went wrong and why (turn_analysis)

Output a JSON object:
- "session_uuid": "{session_uuid}"
- "turn_index": {turn.turn_index}
- "deviation_action": specific operation that caused the deviation
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
            tp.user_name = user_name
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
    sessions_map: dict[str, Session],
    turn_problems_dir: Path | None = None,
) -> list[tuple[SessionTurn, TurnProblem]]:
    """Step 3: For each deviated session with a problematic turn, fetch full session
    JSON via HTTP API, parse turns, and run LLM analysis on the target turn's messages.

    Requires *sessions_map* (uuid → Session) for the session conversation abstract.

    When *turn_problems_dir* is provided, each turn's per-session JSON file is written
    immediately after analysis under a category sub-folder, so partial results survive
    a crash mid-pipeline.

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

        # Look up session abstract from sessions_map
        session = sessions_map.get(ds.session_uuid)
        session_abstract = session.session_abstract if session else []
        category = ds.category or "其他原因导致的偏差"

        tp = _analyze_single_turn(
            client,
            session_uuid=ds.session_uuid,
            user_name=ds.user_name,
            deviation_reason=ds.deviation_reason,
            turn=turn,
            session_abstract=session_abstract,
            index=i,
            total=total,
        )
        if tp is not None:
            results.append((turn, tp))
            if turn_problems_dir is not None:
                _write_turn_problem_file(
                    turn_problems_dir, turn, tp, category, session_abstract,
                )

    sys.stderr.write("\n")
    sys.stderr.flush()

    logger.info("Turn analysis complete: %d turn problems analyzed", len(results))
    return results


def _write_turn_problem_file(
    turn_problems_dir: Path,
    turn: SessionTurn,
    tp: TurnProblem,
    category: str,
    session_abstract: list[SessionRound],
) -> None:
    """Write a single turn problem's per-session JSON file under a category sub-folder."""
    out_dir = turn_problems_dir / category
    out_dir.mkdir(parents=True, exist_ok=True)

    turn_data = {
        "session_uuid": tp.session_uuid,
        "user_name": tp.user_name,
        "turn_index": tp.turn_index,
        "category": category,
        "session_abstract": [
            {"turn": j + 1, "user_msg": r.user_msg, "assistant_reply": r.assistant_reply}
            for j, r in enumerate(session_abstract)
        ],
        "messages": [msg.model_dump() for msg in turn.messages],
        "analysis": tp.model_dump(),
    }
    (out_dir / f"{tp.session_uuid}.json").write_text(
        json.dumps(turn_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_deviation_analysis(config: Config, start_step: int = 1) -> OffsetAnalysisResult:
    users = load_user_files(config.paths.data_dir)
    if not users:
        raise RuntimeError(f"No valid user data files found in {config.paths.data_dir}")

    all_sessions = flatten_all_sessions(users)
    logger.info("Loaded %d sessions from %d users", len(all_sessions), len(users))

    client = LLMClient(config.llm)

    analysis_dir = config.paths.output_dir / "deviation_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Deviation detection (or load from previous run)
    deviated_file = analysis_dir / "deviated_sessions.json"
    if start_step <= 1:
        deviated = detect_deviations(
            client, all_sessions, config.deviation_analysis.detection_batch_size,
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

    # Step 2: Categorization into 5 fixed categories (or load from previous run)
    if start_step <= 2:
        deviated = categorize_deviations(
            client, deviated, all_sessions,
            config.deviation_analysis.categorization_batch_size,
        )
        deviated_file.write_text(
            json.dumps([ds.model_dump() for ds in deviated], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved categorized sessions to %s", deviated_file)
    else:
        # Re-load deviated_sessions.json — it should already have categories from step 2
        if not deviated_file.exists():
            raise FileNotFoundError(
                f"Cannot skip Step 2: intermediate file not found at {deviated_file}"
            )
        deviated_raw = json.loads(deviated_file.read_text(encoding="utf-8"))
        deviated = [DeviatedSession.model_validate(d) for d in deviated_raw]
        logger.info("Loaded %d categorized sessions from %s (steps 1-2 skipped)",
                    len(deviated), deviated_file)

    # Step 3: Turn-level detailed analysis via HTTP API (one turn per LLM call)
    # Per-session JSON files are written immediately inside analyze_turn_deviations
    # under category sub-folders.
    sessions_map: dict[str, Session] = {s.session_uuid: s for _, s in all_sessions}
    turn_problems_dir = analysis_dir / "turn_problems"
    turn_results = analyze_turn_deviations(
        client, deviated, sessions_map, turn_problems_dir=turn_problems_dir,
    )

    # Save aggregate turn_problems.json (single write at the end)
    turn_problems = [tp for _, tp in turn_results]
    (analysis_dir / "turn_problems.json").write_text(
        json.dumps([tp.model_dump() for tp in turn_problems], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved %d turn problems to %s", len(turn_problems),
                analysis_dir / "turn_problems.json")

    # Build summary
    category_counter = Counter(ds.category for ds in deviated)

    sessions_with_turn_index = sum(1 for ds in deviated if ds.problematic_turn_index is not None)

    summary = {
        "total_users": len(users),
        "total_sessions": len(all_sessions),
        "deviated_count": len(deviated),
        "deviation_rate": f"{len(deviated) / len(all_sessions) * 100:.1f}%" if all_sessions else "0%",
        "per_category_distribution": dict(category_counter),
        "sessions_with_problematic_turn": sessions_with_turn_index,
        "turn_problems_analyzed": len(turn_problems),
    }

    return OffsetAnalysisResult(
        deviated_sessions=deviated,
        summary=summary,
        turn_problems=turn_problems,
    )

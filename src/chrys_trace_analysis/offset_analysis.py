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
    ProblematicTurn,
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

For each session that shows deviation, classify it into **ONE** of the following 5 categories:

1. **需求理解偏差与指令遵循失败** — 未能准确理解用户核心意图、业务逻辑或明确约束，导致执行错误任务、提供不符合预期的解决方法或违反规范。

2. **执行幻觉与虚假反馈** — 声称任务已完成、文件已修改或操作已生效，但实际未执行、执行失败或结果与事实不符，导致用户无法验证修改正确性。

3. **代码逻辑缺陷与生成错误** — 生成的内容存在语法错误、逻辑漏洞、编译失败或不符合规范。

4. **系统异常与执行中断** — 因底层问题导致任务无法完成或会话异常中止，包括执行中断、HTTP错误、上下文超出限制、资源限制、网络超时、会话数据缺失等。

5. **其他原因导致的偏差** — 需额外说明具体原因。

For each deviated session, identify which specific turns in the trajectory showed deviation, and describe each deviation in natural language (Chinese).

Output a JSON array. Each element MUST have:
- "session_uuid": the session UUID
- "category": EXACTLY one of the 5 category strings above (copy them exactly)
- "problematic_turns": array of {"turn_index": integer (1-indexed), "description": "detailed Chinese description of what went wrong in this turn"}

If no sessions in the batch show deviation, output an empty JSON array: []

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
    session_uuid: str,
    user_name: str,
    turn_description: str,
    turn: SessionTurn,
) -> str:
    """Format a single turn's raw messages for LLM analysis."""
    lines = [
        f"Session UUID: {session_uuid}",
        f"User: {user_name}",
        f"Turn Problem Description: {turn_description}",
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
# Step 1: Deviation detection + classification
# ---------------------------------------------------------------------------


def _detect_and_classify_batch(
    client: LLMClient,
    batch: list[tuple[str, Session]],
    batch_num: int,
    retries: int = _MAX_RETRIES,
) -> list[DeviatedSession] | None:
    uuid_to_user = {s.session_uuid: u for u, s in batch}

    for attempt in range(1, retries + 1):
        try:
            user_prompt = f"""Below are {len(batch)} coding agent conversation trajectories.

Analyze each trajectory. For sessions that show agent deviation from user intent, classify
into one of the 5 categories and identify which specific turns had problems.

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
                problematic_turns_raw = item.get("problematic_turns", [])
                if not problematic_turns_raw:
                    continue
                problematic_turns = [
                    ProblematicTurn(
                        turn_index=pt.get("turn_index", 0),
                        description=pt.get("description", ""),
                    )
                    for pt in problematic_turns_raw
                ]
                problematic_turns = [pt for pt in problematic_turns if pt.turn_index > 0]
                if not problematic_turns:
                    continue
                ds = DeviatedSession(
                    session_uuid=item.get("session_uuid", ""),
                    user_name=uuid_to_user.get(item.get("session_uuid", ""), ""),
                    category=item.get("category", ""),
                    problematic_turns=problematic_turns,
                )
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


def detect_and_classify_deviations(
    client: LLMClient,
    all_sessions: list[tuple[str, Session]],
    batch_size: int,
) -> list[DeviatedSession]:
    """Step 1: Detect deviated sessions, classify into 5 fixed categories,
    and identify problematic turns with natural-language descriptions."""
    results: list[DeviatedSession] = []
    total = len(all_sessions)
    total_batches = (total + batch_size - 1) // batch_size if total > 0 else 0

    logger.info("Deviation detection + classification: %d sessions in %d batches", total, total_batches)

    for batch_num, start in enumerate(range(0, total, batch_size), 1):
        batch = all_sessions[start:start + batch_size]
        processed_so_far = min(start + batch_size, total)
        message = _render_progress(processed_so_far, total, batch_num, total_batches)
        sys.stderr.write(message)
        sys.stderr.flush()

        batch_results = _detect_and_classify_batch(client, batch, batch_num)
        if batch_results is not None:
            results.extend(batch_results)

    sys.stderr.write("\n")
    sys.stderr.flush()

    deviation_rate = len(results) / total * 100 if total > 0 else 0
    logger.info("Detected %d deviated sessions out of %d (%.1f%%)",
                len(results), total, deviation_rate)
    return results


# ---------------------------------------------------------------------------
# Step 2: Turn-level detailed analysis
# ---------------------------------------------------------------------------


def _analyze_single_turn(
    client: LLMClient,
    session_uuid: str,
    user_name: str,
    turn_description: str,
    turn: SessionTurn,
    index: int,
    total: int,
    retries: int = _MAX_RETRIES,
) -> TurnProblem | None:
    """Analyze a single problematic turn, one LLM call per turn."""
    formatted = _format_turn_for_analysis(session_uuid, user_name, turn_description, turn)
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
    turn_problems_dir: Path | None = None,
) -> list[tuple[SessionTurn, TurnProblem]]:
    """Step 2: For each problematic turn identified in step 1, fetch the full
    session JSON via HTTP API, parse turns, and run LLM analysis on the
    target turn's raw messages.

    When *turn_problems_dir* is provided, each turn's per-session JSON file is written
    immediately after analysis, so partial results survive a crash mid-pipeline.

    Returns list of (turn, TurnProblem) tuples for aggregate file saving.
    """
    if not deviated:
        logger.info("No deviated sessions to analyze at turn level.")
        return []

    # Collect (DeviatedSession, ProblematicTurn, SessionTurn) triples
    analysis_triples: list[tuple[DeviatedSession, ProblematicTurn, SessionTurn]] = []
    skipped_fetch_failed = 0
    skipped_no_turn = 0

    for ds in deviated:
        # Fetch full session JSON once per session
        doc = _fetch_session_json(ds.session_uuid)
        if doc is None:
            skipped_fetch_failed += len(ds.problematic_turns)
            continue

        messages = doc.get("messages", [])
        if not isinstance(messages, list) or not messages:
            skipped_fetch_failed += len(ds.problematic_turns)
            continue

        turns = build_session_turns(messages)
        turns_by_index: dict[int, SessionTurn] = {t.turn_index: t for t in turns}

        for pt in ds.problematic_turns:
            target_turn = turns_by_index.get(pt.turn_index)
            if target_turn is None:
                skipped_no_turn += 1
                logger.debug(
                    "Session %s turn %d: turn not found in %d turns",
                    ds.session_uuid, pt.turn_index, len(turns),
                )
                continue
            analysis_triples.append((ds, pt, target_turn))

    logger.info(
        "Turn analysis: %d triples collected "
        "(%d skipped: fetch failed, %d skipped: no turn)",
        len(analysis_triples), skipped_fetch_failed, skipped_no_turn,
    )

    if not analysis_triples:
        return []

    total = len(analysis_triples)
    logger.info("Turn analysis: %d turns to analyze (one turn per LLM call)", total)

    if turn_problems_dir is not None:
        turn_problems_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[SessionTurn, TurnProblem]] = []
    for i, (ds, pt, turn) in enumerate(analysis_triples, 1):
        message = _render_progress(i, total, i, total)
        sys.stderr.write(message)
        sys.stderr.flush()

        tp = _analyze_single_turn(
            client,
            session_uuid=ds.session_uuid,
            user_name=ds.user_name,
            turn_description=pt.description,
            turn=turn,
            index=i,
            total=total,
        )
        if tp is not None:
            results.append((turn, tp))
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
    (turn_problems_dir / f"{tp.session_uuid}_{tp.turn_index}.json").write_text(
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

    # Step 1: Deviation detection + classification (or load from previous run)
    deviated_file = offset_dir / "deviated_sessions.json"
    if start_step <= 1:
        deviated = detect_and_classify_deviations(
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

    # Step 2: Turn-level detailed analysis via HTTP API
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
    category_counter = Counter(ds.category for ds in deviated)

    sessions_with_problematic_turns = sum(1 for ds in deviated if ds.problematic_turns)

    unique_categories = len({ds.category for ds in deviated if ds.category})

    summary = {
        "total_users": len(users),
        "total_sessions": len(all_sessions),
        "deviated_count": len(deviated),
        "deviation_rate": f"{len(deviated) / len(all_sessions) * 100:.1f}%" if all_sessions else "0%",
        "categories_count": unique_categories,
        "per_category_distribution": dict(category_counter),
        "sessions_with_problematic_turns": sessions_with_problematic_turns,
        "turn_problems_analyzed": len(turn_problems),
    }

    return OffsetAnalysisResult(
        deviated_sessions=deviated,
        summary=summary,
        turn_problems=turn_problems,
    )

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import openai
from httpx import ReadTimeout

from .config import Config
from .loader import load_user_files
from .llm_client import LLMClient
from .models import DeviatedSession, DeviationCategory, OffsetAnalysisResult, Session
from .sampler import flatten_all_sessions

logger = logging.getLogger(__name__)

_BAR_WIDTH = 40
_MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DETECTION_SYSTEM_PROMPT = """You are an expert at analyzing coding agent execution quality. You will be given a batch of conversation trajectories between users and a coding AI assistant.

For each trajectory, determine whether the agent's execution **deviated from the user's intent**. Pay special attention to:
- Whether the user had to **correct** or **re-direct** the agent's behavior
- Whether the agent misunderstood the user's requirements and produced wrong results
- Whether the agent went off on a tangent unrelated to the user's request
- Whether the user expressed frustration or had to repeat themselves

For each session that shows deviation, provide the session UUID and a concise reason (in Chinese) explaining what went wrong and why.

Output a JSON array containing ONLY the deviated sessions. Each element must have:
- "session_uuid": the session UUID
- "deviation_reason": a concise explanation of the deviation (in Chinese)

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

AGGREGATION_SYSTEM_PROMPT = """You are an expert at synthesizing categorization results. You will receive deviation category results from multiple independent batches of coding agent conversation trajectories.

Each batch was analyzed independently and produced its own set of deviation categories.

Your task is to synthesize ALL of these results into a single, consolidated set of deviation categories.

Guidelines:
- Merge similar categories across batches (e.g. "需求理解错误" and "误解用户意图" should be unified)
- Keep categories that appear consistently across batches; consider dropping outliers that only appear in one batch with very few sessions
- For each final category, write a comprehensive description that captures the consensus
- Re-assign all session_uuids to the correct merged category
- The final set should be coherent, non-overlapping, and cover all deviation patterns

Output a JSON array where each element has "category_name", "description", and "session_uuids" fields.
Output ONLY valid JSON (no markdown, no extra text)."""

# ---------------------------------------------------------------------------
# Session formatting (same format as scenario.py)
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
                "",
                "### Conversation:",
            ]
            for j, round_ in enumerate(session.session_abstract, 1):
                lines.append(f"**User:** {round_.user_msg}")
                lines.append(f"**Assistant:** {round_.assistant_reply}")
                lines.append("")
        parts.append("\n".join(lines))
    return "\n---\n".join(parts)


# ---------------------------------------------------------------------------
# Progress bar (same as classifier.py)
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


def _format_batch_categories(batch_results: list[list[dict]], batch_offset: int) -> str:
    """Format all batch categorization results for aggregation."""
    parts: list[str] = []
    for i, categories in enumerate(batch_results, 1):
        lines = [f"## Batch {batch_offset + i} results ({len(categories)} categories)"]
        for cat in categories:
            lines.append(f"- **{cat.get('category_name', '')}**: {cat.get('description', '')}")
            uuids = cat.get("session_uuids", [])
            lines.append(f"  Sessions ({len(uuids)}): {', '.join(uuids)}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def categorize_deviations(
    client: LLMClient,
    deviated: list[DeviatedSession],
    all_sessions: list[tuple[str, Session]],
    batch_size: int,
) -> list[DeviationCategory]:
    """Step 2: Categorize deviation causes, with batching and aggregation if needed."""
    if not deviated:
        logger.info("No deviated sessions to categorize.")
        return []

    sessions_map: dict[str, tuple[str, Session]] = {
        s.session_uuid: (u, s) for u, s in all_sessions
    }

    total = len(deviated)
    total_batches = (total + batch_size - 1) // batch_size if total > 0 else 0
    logger.info("Deviation categorization: %d sessions in %d batches", total, total_batches)

    if total_batches == 1:
        raw = _categorize_deviations_batch(client, deviated, sessions_map, batch_num=1)
        if raw is None:
            logger.warning("Categorization failed, returning empty result.")
            return []
        return _parse_categories(raw)

    # Multi-batch: categorize each batch, then aggregate
    all_batch_results: list[list[dict]] = []
    for batch_num, start in enumerate(range(0, total, batch_size), 1):
        batch = deviated[start:start + batch_size]
        logger.info("Categorizing batch %d/%d (%d sessions)...",
                    batch_num, total_batches, len(batch))
        raw = _categorize_deviations_batch(client, batch, sessions_map, batch_num)
        if raw is not None:
            all_batch_results.append(raw)

    if not all_batch_results:
        logger.warning("All categorization batches failed.")
        return []

    if len(all_batch_results) == 1:
        return _parse_categories(all_batch_results[0])

    # Aggregate
    logger.info("Aggregating categories from %d batches...", len(all_batch_results))
    formatted = _format_batch_categories(all_batch_results, batch_offset=0)

    user_prompt = f"""Below are deviation categorization results from {len(all_batch_results)} independent batches.

Synthesize them into a consolidated set of categories.

Output ONLY valid JSON (no markdown, no extra text):

{formatted}"""

    try:
        raw = client.chat_json(AGGREGATION_SYSTEM_PROMPT, user_prompt)
    except Exception as exc:
        logger.error("Aggregation failed: %s. Returning raw batch results.", exc)
        # Fallback: merge all batch results as-is
        all_categories: list[DeviationCategory] = []
        for batch_raw in all_batch_results:
            all_categories.extend(_parse_categories(batch_raw))
        return all_categories

    return _parse_categories(raw)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_offset(config: Config) -> OffsetAnalysisResult:
    users = load_user_files(config.paths.data_dir)
    if not users:
        raise RuntimeError(f"No valid user data files found in {config.paths.data_dir}")

    all_sessions = flatten_all_sessions(users)
    logger.info("Loaded %d sessions from %d users", len(all_sessions), len(users))

    client = LLMClient(config.llm)

    # Step 1: Deviation detection
    deviated = detect_deviations(
        client, all_sessions, config.offset_pipeline.detection_batch_size,
    )

    # Save intermediate results
    offset_dir = config.paths.output_dir / "offset_analysis"
    offset_dir.mkdir(parents=True, exist_ok=True)
    (offset_dir / "deviated_sessions.json").write_text(
        json.dumps([ds.model_dump() for ds in deviated], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved %d deviated sessions to %s", len(deviated),
                offset_dir / "deviated_sessions.json")

    # Step 2: Categorization
    categories = categorize_deviations(
        client, deviated, all_sessions,
        config.offset_pipeline.categorization_batch_size,
    )

    # Build summary
    category_counter = Counter()
    for cat in categories:
        category_counter[cat.category_name] = cat.session_count

    summary = {
        "total_users": len(users),
        "total_sessions": len(all_sessions),
        "deviated_count": len(deviated),
        "deviation_rate": f"{len(deviated) / len(all_sessions) * 100:.1f}%" if all_sessions else "0%",
        "categories_count": len(categories),
        "per_category_distribution": dict(category_counter),
    }

    return OffsetAnalysisResult(
        deviated_sessions=deviated,
        categories=categories,
        summary=summary,
    )

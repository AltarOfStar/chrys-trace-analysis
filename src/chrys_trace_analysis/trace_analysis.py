"""轨迹分析流水线：展开 compressed_msgs → 按 batch 分析任务与人工介入 → 跨批次聚合标准分类。

处理流程：

1. **展开压缩消息**：从指定目录加载 ``{uuid}.json`` 会话（chrys envelope 格式，
   参考 chrys 本地 ``session.json``，容忍根字段冗余），通过
   :func:`reconstruct.reconstruct_messages` 将 ``compressed_msgs`` 原位展开为
   完整消息，再还原为轨迹（轮次摘要 + 完整轮次）。
2. **分批**：将轨迹按 ``batch_size`` 划分，每个批次数量不低于 ``batch_size``；
   如有剩余，将其归入最后一个批次（默认 batch_size=20）。
3. **批次分析**：对每个批次组装提示词，要求模型逐条轨迹输出两个维度的分析：
   - 任务维度：任务分类（单个词）+ 任务内容概括（一小段话）；
   - 人工介入维度：是否存在人工介入；若存在，逐次给出介入类型（指出错误、
     补充信息、需求变更等）、介入位置与简要描述。
4. **跨批次聚合**：在批次之间将自由形式的任务分类与人工介入类型聚合成几个
   固定种类，并将每个批次的分类替换为聚合后的标准分类，输出最终聚合信息与
   每条轨迹的对应信息。

中间产物（崩溃后可复用）：
- ``output/trace_analysis/batch_analysis/batch_{n}.json`` 每个批次的分析结果；
- ``output/trace_analysis/aggregation.json`` 聚合结果。
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import openai
from httpx import ReadTimeout

from .config import Config
from .llm_client import LLMClient
from .loader import load_sessions
from .models import (
    CategoryDefinition,
    Session,
    TraceAnalysisResult,
    TraceTaskAnalysis,
)

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BAR_WIDTH = 40

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

BATCH_ANALYSIS_SYSTEM_PROMPT = """You are an expert at analyzing coding agent conversation trajectories. You will be given a batch of trajectories between users and a coding AI assistant.

For EACH trajectory in the batch, analyze it and provide two dimensions of analysis:

**Dimension 1 - Task analysis:**
- "task_category": classify the task this trajectory handled with a SINGLE Chinese word (e.g. 编程, 调试, 审查, 部署, 文档, 测试, 运维, 数据处理, ...). The value must be a single word, not a phrase or sentence.
- "task_summary": a short paragraph (2-4 sentences, in Chinese) summarizing what task this trajectory handled: what the user wanted and what was eventually delivered.

**Dimension 2 - Human intervention analysis:**
Determine whether the user (or another human) actively intervened in the agent's workflow. Intervention means the human corrected, supplemented or redirected the agent's behavior, e.g. pointing out an error in the agent's output, providing supplementary information missing from the original request, or changing requirements mid-way. Merely replying "好的" or acknowledging results is NOT an intervention.
- "human_intervention": true or false
- If true, list EACH intervention occurrence under "interventions", one entry per occurrence:
  - "type": the type of this intervention, e.g. 指出错误, 补充信息, 需求变更, 纠正方向, 暂停中止, 其他 (use the closest label)
  - "position": where in the trajectory the intervention occurred (e.g. "第2轮用户消息", "第5轮")
  - "description": a brief description of this intervention (1-2 sentences, in Chinese)

IMPORTANT:
- Output a JSON ARRAY with EXACTLY ONE element per trajectory, in the same order as the trajectories are numbered.
- A trajectory without human intervention must still have an element, with "human_intervention": false and an empty "interventions" array.
- Each element must contain "session_uuid" matching the UUID of that trajectory.

Output ONLY valid JSON (no markdown, no extra text)."""

AGGREGATION_SYSTEM_PROMPT = """You are an expert at synthesizing categorization results. You will be given the task categories and human-intervention types that were independently produced for many conversation trajectories (each trajectory was classified per-batch with a free-form single-word label).

Your task is to aggregate them into a FEW fixed categories:

1. "task_categories": Consolidate all observed task category words into a small set (3-8) of standard task categories. Each element must have:
   - "name": a single Chinese word or short phrase
   - "description": 1-2 sentences in Chinese explaining what tasks this category covers

2. "task_category_mapping": an object mapping EVERY observed task category word (key) to exactly one standard category name (value). Every observed word must appear as a key.

3. "intervention_types": Consolidate all observed human-intervention types into a small set (3-8) of standard intervention types. Each element must have:
   - "name": a single Chinese word or short phrase
   - "description": 1-2 sentences in Chinese explaining what this type covers

4. "intervention_type_mapping": an object mapping EVERY observed intervention type (key) to exactly one standard intervention type (value). Every observed type must appear as a key.

Guidelines:
- Merge similar labels (e.g. "编码" and "写代码" should map to the same standard category).
- Keep categories that cover a meaningful number of trajectories; avoid near-duplicate categories.
- The final sets must be coherent, non-overlapping and cover all observed labels.

Output a JSON object with the four fields above.
Output ONLY valid JSON (no markdown, no extra text)."""


# ---------------------------------------------------------------------------
# 分批
# ---------------------------------------------------------------------------


def chunk_traces(
    sessions: list[tuple[str, Session]],
    batch_size: int,
) -> list[list[tuple[str, Session]]]:
    """按 batch_size 划分批次：每个批次数量不低于 batch_size，剩余不足一个
    批次的轨迹并入最后一个批次。

    例如 65 条轨迹、batch_size=20 → 划分为 [20, 20, 25] 三个批次。
    """
    n = len(sessions)
    if n == 0:
        return []
    if n <= batch_size:
        return [sessions]

    num_batches = n // batch_size
    batches = [
        sessions[i * batch_size:(i + 1) * batch_size]
        for i in range(num_batches - 1)
    ]
    batches.append(sessions[(num_batches - 1) * batch_size:])
    return batches


# ---------------------------------------------------------------------------
# 批次分析
# ---------------------------------------------------------------------------


def _format_batch(batch: list[tuple[str, Session]]) -> str:
    parts: list[str] = []
    for i, (user_name, session) in enumerate(batch, 1):
        header = f"## Trajectory {i} (User: {user_name}, UUID: {session.session_uuid})"
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


def _analyze_batch(
    client: LLMClient,
    batch: list[tuple[str, Session]],
    batch_num: int,
    retries: int = _MAX_RETRIES,
) -> list[TraceTaskAnalysis] | None:
    """让模型分析一个批次的全部轨迹，返回每条轨迹的任务与人介入分析。"""
    uuid_to_user = {s.session_uuid: u for u, s in batch}
    user_prompt = f"""Below are {len(batch)} coding agent conversation trajectories.

Analyze each trajectory and output a JSON array with exactly one element per trajectory, in the same order.

Output ONLY valid JSON (no markdown, no extra text):

{_format_batch(batch)}"""

    for attempt in range(1, retries + 1):
        try:
            raw = client.chat_json(BATCH_ANALYSIS_SYSTEM_PROMPT, user_prompt)
            results: list[TraceTaskAnalysis] = []
            for item in raw:
                session_uuid = item.get("session_uuid", "")
                if not session_uuid:
                    logger.warning(
                        "Batch %d: LLM output item without session_uuid skipped", batch_num,
                    )
                    continue
                item["user_name"] = uuid_to_user.get(session_uuid, "")
                ta = TraceTaskAnalysis.model_validate(item)
                ta.human_intervention = ta.human_intervention or bool(ta.interventions)
                results.append(ta)
            return results
        except (ReadTimeout, openai.APITimeoutError) as exc:
            logger.warning(
                "Batch analysis %d timed out (attempt %d/%d): %s",
                batch_num, attempt, retries, exc,
            )
            continue
        except Exception as exc:
            logger.warning(
                "Batch analysis %d failed (attempt %d/%d): %s",
                batch_num, attempt, retries, exc,
            )
            continue

    logger.error("Batch analysis %d failed after %d retries, skipping %d trajectories.",
                 batch_num, retries, len(batch))
    return None


def run_batch_analysis(
    client: LLMClient,
    batches: list[list[tuple[str, Session]]],
    analysis_dir: Path,
) -> tuple[list[TraceTaskAnalysis], list[int]]:
    """逐批次分析轨迹；已存在的中间文件直接复用，返回 (全部分析, 失败批次号)。"""
    batch_dir = analysis_dir / "batch_analysis"
    batch_dir.mkdir(parents=True, exist_ok=True)

    all_analyses: list[TraceTaskAnalysis] = []
    failed_batches: list[int] = []
    total = len(batches)

    for batch_num, batch in enumerate(batches, 1):
        out_file = batch_dir / f"batch_{batch_num}.json"
        if out_file.exists():
            try:
                data = json.loads(out_file.read_text(encoding="utf-8"))
                batch_analyses = [TraceTaskAnalysis.model_validate(item) for item in data]
                all_analyses.extend(batch_analyses)
                logger.info("Batch %d/%d loaded from intermediate file (%d analyses)",
                            batch_num, total, len(batch_analyses))
                continue
            except Exception:
                logger.warning("Batch %d/%d: reuse of intermediate file failed, "
                               "re-analyzing", batch_num, total)

        sys.stderr.write(_render_progress(batch_num, total, batch_num, total))
        sys.stderr.flush()

        batch_analyses = _analyze_batch(client, batch, batch_num)
        if batch_analyses is None:
            failed_batches.append(batch_num)
            continue

        out_file.write_text(
            json.dumps([a.model_dump() for a in batch_analyses],
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        all_analyses.extend(batch_analyses)

    sys.stderr.write("\n")
    sys.stderr.flush()
    logger.info("Batch analysis complete: %d analyses, %d failed batches",
                len(all_analyses), len(failed_batches))
    return all_analyses, failed_batches


# ---------------------------------------------------------------------------
# 跨批次聚合
# ---------------------------------------------------------------------------


def _format_aggregation_input(analyses: list[TraceTaskAnalysis]) -> str:
    task_counter = Counter(a.task_category for a in analyses if a.task_category)
    type_counter = Counter(
        iv.type for a in analyses for iv in a.interventions if iv.type
    )
    lines = ["Observed task categories (word: count):"]
    for word, count in sorted(task_counter.items(), key=lambda x: -x[1]):
        lines.append(f"- {word}: {count}")
    lines.append("")
    lines.append("Observed human-intervention types (type: count):")
    for t, count in sorted(type_counter.items(), key=lambda x: -x[1]):
        lines.append(f"- {t}: {count}")
    return "\n".join(lines)


def _aggregate_categories(
    client: LLMClient,
    analyses: list[TraceTaskAnalysis],
    retries: int = _MAX_RETRIES,
) -> dict | None:
    """把批次间自由形式的任务分类与人工介入类型聚合为固定种类。"""
    user_prompt = f"""Below are the observed task categories and human-intervention types from {len(analyses)} analyzed trajectories.

Aggregate them into a few fixed categories as described in the system prompt.

{_format_aggregation_input(analyses)}"""

    for attempt in range(1, retries + 1):
        try:
            raw = client.chat_json(AGGREGATION_SYSTEM_PROMPT, user_prompt)
            if not isinstance(raw, dict):
                raise ValueError("aggregation response is not a JSON object")
            if not _extract_mapping(raw, "task_category_mapping") and \
                    not _extract_mapping(raw, "intervention_type_mapping"):
                raise ValueError("aggregation response contains no mappings")
            return raw
        except (ReadTimeout, openai.APITimeoutError) as exc:
            logger.warning(
                "Category aggregation timed out (attempt %d/%d): %s",
                attempt, retries, exc,
            )
            continue
        except Exception as exc:
            logger.warning(
                "Category aggregation failed (attempt %d/%d): %s",
                attempt, retries, exc,
            )
            continue

    logger.error("Category aggregation failed after %d retries.", retries)
    return None


def _extract_mapping(raw: dict | None, key: str) -> dict[str, str]:
    if raw is None:
        return {}
    value = raw.get(key, {})
    if not isinstance(value, dict):
        return {}
    return {str(k): str(v) for k, v in value.items() if v}


def _extract_categories(raw: dict | None, key: str) -> list[dict[str, str]]:
    if raw is None:
        return []
    value = raw.get(key, [])
    if not isinstance(value, list):
        return []
    result: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, dict) and item.get("name"):
            result.append({
                "name": str(item["name"]),
                "description": str(item.get("description", "")),
            })
    return result


def apply_aggregation(
    analyses: list[TraceTaskAnalysis],
    aggregation: dict | None,
) -> tuple[list[CategoryDefinition], list[CategoryDefinition],
           dict[str, str], dict[str, str]]:
    """将批次阶段分类替换为聚合后的标准分类，并返回固定种类与原始→标准映射。

    聚合失败（aggregation 为 None）时退化为恒等映射：原始分类即标准分类。
    """
    task_mapping = _extract_mapping(aggregation, "task_category_mapping")
    type_mapping = _extract_mapping(aggregation, "intervention_type_mapping")
    task_defs = _extract_categories(aggregation, "task_categories")
    type_defs = _extract_categories(aggregation, "intervention_types")

    for a in analyses:
        if a.task_category:
            a.task_category_raw = a.task_category
            a.task_category = task_mapping.get(a.task_category, a.task_category)
        for iv in a.interventions:
            if iv.type:
                iv.type_raw = iv.type
                iv.type = type_mapping.get(iv.type, iv.type)

    task_counter = Counter(a.task_category for a in analyses if a.task_category)
    type_counter = Counter(
        iv.type for a in analyses for iv in a.interventions if iv.type
    )

    if not task_defs:
        task_defs = [{"name": name, "description": ""} for name in task_counter]
    if not type_defs:
        type_defs = [{"name": name, "description": ""} for name in type_counter]

    task_categories = [
        CategoryDefinition(name=d["name"], description=d["description"],
                           count=task_counter.get(d["name"], 0))
        for d in task_defs
    ]
    type_categories = [
        CategoryDefinition(name=d["name"], description=d["description"],
                           count=type_counter.get(d["name"], 0))
        for d in type_defs
    ]

    covered = {c.name for c in task_categories}
    for name, count in task_counter.items():
        if name not in covered:
            task_categories.append(CategoryDefinition(name=name, count=count))
    covered = {c.name for c in type_categories}
    for name, count in type_counter.items():
        if name not in covered:
            type_categories.append(CategoryDefinition(name=name, count=count))

    task_categories = [c for c in task_categories if c.count > 0]
    type_categories = [c for c in type_categories if c.count > 0]
    return task_categories, type_categories, task_mapping, type_mapping


# ---------------------------------------------------------------------------
# 进度条
# ---------------------------------------------------------------------------


def _render_progress(processed: int, total: int, batch_num: int, total_batches: int) -> str:
    pct = processed / total if total > 0 else 0
    filled = int(_BAR_WIDTH * pct)
    bar = "█" * filled + "░" * (_BAR_WIDTH - filled)
    return f"\r  [{bar}] batch {batch_num}/{total_batches}"


# ---------------------------------------------------------------------------
# 流水线入口
# ---------------------------------------------------------------------------


def run_trace_analysis(config: Config) -> TraceAnalysisResult:
    """运行轨迹分析流水线：展开 compressed_msgs → 分批 → 批次分析 → 聚合。"""
    # Step 1: 从指定目录加载 {uuid}.json 会话（自动展开 compressed_msgs）
    traces, reconstruction = load_sessions(config.paths.sessions_dir)
    if not traces:
        raise RuntimeError(
            f"No valid session files found in {config.paths.sessions_dir}"
        )

    # Step 2: 按 batch_size 分批（剩余轨迹并入最后一个批次）
    batch_size = config.trace_analysis.batch_size
    batches = chunk_traces(traces, batch_size)
    logger.info("Chunked %d trajectories into %d batches (batch_size=%d)",
                len(traces), len(batches), batch_size)

    client = LLMClient(config.llm)

    analysis_dir = config.paths.output_dir / "trace_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Step 3: 批次分析（每条轨迹输出任务与人工介入维度分析）
    analyses, failed_batches = run_batch_analysis(client, batches, analysis_dir)

    # Step 4: 跨批次聚合为固定分类（已存在的聚合结果直接复用）
    aggregation_file = analysis_dir / "aggregation.json"
    aggregation: dict | None = None
    if aggregation_file.exists():
        try:
            aggregation = json.loads(aggregation_file.read_text(encoding="utf-8"))
            logger.info("Loaded aggregation result from %s", aggregation_file)
        except Exception:
            logger.warning("Reuse of aggregation file failed, re-aggregating")
    if aggregation is None:
        aggregation = _aggregate_categories(client, analyses)
        if aggregation is not None:
            aggregation_file.write_text(
                json.dumps(aggregation, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            logger.error("Aggregation failed; falling back to raw categories")

    task_categories, type_categories, task_mapping, type_mapping = apply_aggregation(
        analyses, aggregation,
    )

    # Step 5: 汇总并输出
    intervention_count = sum(1 for a in analyses if a.human_intervention)
    task_counter = Counter(c.name for c in task_categories)
    type_counter = Counter(c.name for c in type_categories)

    summary = {
        "total_documents": reconstruction["total_documents"],
        "total_sessions": len(traces),
        "analyzed_sessions": len(analyses),
        "total_batches": len(batches),
        "failed_batches": failed_batches,
        "batch_size": batch_size,
        "intervention_sessions": intervention_count,
        "intervention_rate": f"{intervention_count / len(analyses) * 100:.1f}%"
            if analyses else "0%",
        "per_task_category_distribution": dict(task_counter),
        "per_intervention_type_distribution": dict(type_counter),
        "reconstruction": reconstruction,
    }

    result = TraceAnalysisResult(
        analyses=analyses,
        task_categories=task_categories,
        intervention_types=type_categories,
        task_category_mapping=task_mapping,
        intervention_type_mapping=type_mapping,
        summary=summary,
    )

    result_file = analysis_dir / "trace_analysis_result.json"
    result_file.write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Trace analysis complete: %d analyses, output written to %s",
                len(analyses), result_file)
    return result

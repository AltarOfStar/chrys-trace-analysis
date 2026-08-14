"""轨迹分析流水线：压缩轨迹摘要 → 逐条并行分析 → 跨轨迹聚合固定分类。

处理流程：

1. **加载与摘要压缩**：从指定目录加载会话（扁平 ``{uuid}.json`` 或嵌套
   ``{uuid}/session.json``，后者额外读取 ``approvals/`` 审批记录与
   ``sub_agents/sessions/`` 子代理摘要），通过
   :func:`reconstruct.reconstruct_messages` 将 ``compressed_msgs`` 原位展开为
   完整消息，再本地构建**紧凑轨迹摘要**：每轮保留截断后的用户消息、最终
   助手答复、工具调用（含参数预览）、人工审批检查点与子代理摘要，并限制
   轮次上限（避免单条上万字符的完整长文答复进入提示词）。
2. **逐条并行分析**：对每条轨迹独立调用一次 LLM（小输入、小输出），输出
   任务分类（单个词）、任务概括与每次人工介入（结构化轮次号 + 类型 +
   描述）；通过线程池并发执行，结果按 ``trajectories/{uuid}.json`` 落盘，
   崩溃后可复用续跑。
3. **跨轨迹聚合**：一次 LLM 调用把自由形式的任务分类与人工介入类型聚合为
   固定种类，并将每条轨迹/每次介入的分类替换为聚合后的标准分类。
4. **输出**：分类总览（固定种类定义 + 分布）、每条轨迹的类别与特征、每次
   人工介入的类别。

中间产物（崩溃后可复用）：
- ``output/trace_analysis/trajectories/{uuid}.json`` 每条轨迹的分析结果；
- ``output/trace_analysis/aggregation.json`` 聚合结果。
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import openai
from httpx import ReadTimeout

from .config import Config
from .llm_client import LLMClient
from .loader import load_sessions
from .models import (
    ApprovalRecord,
    CategoryDefinition,
    Intervention,
    Session,
    SessionTurn,
    SubAgentRecord,
    TraceAnalysisResult,
    TraceTaskAnalysis,
)

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BAR_WIDTH = 40
_ANALYSIS_MAX_TOKENS = 4096  # 逐条分析输出上限（单条轨迹，可容纳更详细输出）
_MAX_APPROVALS_SHOWN = 6     # 摘要中每轮最多展示的审批检查点数

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TRACE_ANALYSIS_SYSTEM_PROMPT = """You are an expert at analyzing coding agent conversation trajectories. You will be given a COMPACT DIGEST of ONE trajectory between a user and a coding AI assistant.

Digest format per turn:
- "User": the (possibly truncated) user message
- "Tools": tool calls made by the assistant, each with a short preview of its arguments (possibly truncated)
- "Sub-agents": delegated sub-agent runs (tool name, status, internal message/tool-call counts, task preview)
- "Checkpoints": human approval records for proposed tool actions, each with a verdict (APPROVED/REJECTED) and the human's stated reason
- "Assistant": the (possibly truncated) final assistant answer

Analyze this trajectory and output a JSON object with EXACTLY these fields:

- "task_category": classify the task this trajectory handled with a SINGLE Chinese word (e.g. 编程, 调试, 审查, 部署, 文档, 测试, 运维, 数据处理, ...). The value must be a single word, not a phrase or sentence.
- "task_summary": 1-3 sentences in Chinese summarizing what the task was: what the user wanted and what was eventually delivered.
- "interventions": an array describing EVERY human intervention occurrence. An intervention means the human actively corrected, supplemented or redirected the agent, e.g. pointing out an error in the agent's output, providing supplementary information missing from the original request, changing requirements mid-way, or stopping the agent. Merely replying "好的" / "继续" / acknowledging results is NOT an intervention. Each element must be:
  - "turn_index": the 1-based turn number (as numbered in the digest) of the turn whose USER message constitutes this intervention
  - "type": the closest label from 指出错误, 补充信息, 需求变更, 纠正方向, 暂停中止, 其他
  - "description": 1-2 sentences in Chinese describing this intervention
  If there is no intervention, output an empty array.

Guidance on approval checkpoints:
- An APPROVED checkpoint is routine human verification of a tool action; by itself it is NOT an intervention. But if the human's stated reason redirects or constrains the agent (e.g. "只读即可，不要修改文件", "不要执行这条命令"), that redirection IS an intervention (纠正方向).
- A REJECTED checkpoint is a definitive human intervention (暂停中止 or 纠正方向); report it with the human's stated reason, using the turn where the checkpoint appears.

IMPORTANT:
- Output ONLY valid JSON (no markdown, no extra text).
- Do not invent interventions that are not visible in the digest.
- If a user message was truncated in the digest, judge from what is visible; do not guess. When in doubt, do not mark it as an intervention."""

AGGREGATION_SYSTEM_PROMPT = """You are an expert at synthesizing categorization results. You will be given the task categories and human-intervention types that were independently produced for many conversation trajectories (each trajectory was classified with a free-form single-word label).

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
# 轨迹摘要（本地压缩，不进 LLM 的只有截断后的短文本）
# ---------------------------------------------------------------------------


@dataclass
class ToolCallDigest:
    """一次工具调用：名称 + 参数预览（截断）。"""

    name: str
    args_head: str = ""


@dataclass
class TurnDigest:
    """单轮摘要：截断后的用户消息 / 助手答复 + 工具调用 + 审批检查点 + 子代理。"""

    turn_index: int
    user_text: str = ""
    assistant_text: str = ""
    tool_calls: list[ToolCallDigest] = field(default_factory=list)
    approvals: list[ApprovalRecord] = field(default_factory=list)
    sub_agents: list[SubAgentRecord] = field(default_factory=list)

    @property
    def tool_names(self) -> list[str]:
        return [c.name for c in self.tool_calls]


@dataclass
class TraceDigest:
    """单条轨迹的紧凑摘要与确定性特征。"""

    session_uuid: str
    user_name: str
    turns: list[TurnDigest]
    mcp_tools: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    truncated_turns: bool = False
    features: dict = field(default_factory=dict)


def _truncate(text: str, limit: int) -> str:
    """截断到 limit 字符；超过时保留开头并追加省略号。"""
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _turn_tool_calls(
    turn: SessionTurn | None,
    max_arg_chars: int,
) -> list[ToolCallDigest]:
    """提取一轮内的工具调用（名称 + 参数预览），保留调用顺序与次数。"""
    if turn is None:
        return []
    calls: list[ToolCallDigest] = []
    for msg in turn.messages:
        for content in msg.contents:
            if content.type == "function_call" and content.name:
                calls.append(ToolCallDigest(
                    name=content.name,
                    args_head=_truncate(content.arguments, max_arg_chars),
                ))
    return calls


def _turn_by_index(session: Session, turn_index: int) -> SessionTurn | None:
    return next((t for t in session.turns if t.turn_index == turn_index), None)


def _approvals_for_turn(
    approvals: list[ApprovalRecord],
    turn_index: int,
) -> list[ApprovalRecord]:
    return [a for a in approvals if a.turn_index == turn_index]


def _sub_agents_for_turn(
    sub_agents: list[SubAgentRecord],
    turn_index: int,
) -> list[SubAgentRecord]:
    return [s for s in sub_agents if s.turn_index == turn_index]


def build_trace_digest(
    session: Session,
    user_name: str = "",
    max_turns: int = 60,
    max_user_chars: int = 600,
    max_assistant_chars: int = 1000,
    max_tool_args_chars: int = 200,
) -> TraceDigest:
    """把一条 Session 压缩为紧凑摘要 + 确定性特征。

    摘要中每轮只保留截断后的用户消息、最终助手答复、工具调用（含参数预览）、
    人工审批检查点与子代理摘要；轮次超过 ``max_turns`` 时只保留前
    ``max_turns`` 轮并在摘要中注明。
    """
    rounds = session.session_abstract
    truncated_turns = len(rounds) > max_turns

    turns: list[TurnDigest] = []
    for i, round_ in enumerate(rounds[:max_turns], 1):
        turns.append(TurnDigest(
            turn_index=i,
            user_text=_truncate(round_.user_msg, max_user_chars),
            assistant_text=_truncate(round_.assistant_reply, max_assistant_chars),
            tool_calls=_turn_tool_calls(_turn_by_index(session, i), max_tool_args_chars),
            approvals=_approvals_for_turn(session.approvals, i),
            sub_agents=_sub_agents_for_turn(session.sub_agents, i),
        ))

    tool_counter: Counter[str] = Counter()
    tool_calls = 0
    message_count = 0
    for turn in session.turns:
        message_count += len(turn.messages)
        for msg in turn.messages:
            for content in msg.contents:
                if content.type == "function_call":
                    tool_calls += 1
                    if content.name:
                        tool_counter[content.name] += 1

    features = {
        "turn_count": len(rounds),
        "message_count": message_count,
        "tool_call_count": tool_calls,
        "tool_usage": dict(tool_counter.most_common(10)),
        "approval_count": len(session.approvals),
        "rejection_count": sum(1 for a in session.approvals if not a.approved),
        "sub_agent_count": len(session.sub_agents),
        "sub_agent_tool_calls": sum(s.tool_call_count for s in session.sub_agents),
    }

    return TraceDigest(
        session_uuid=session.session_uuid,
        user_name=user_name,
        turns=turns,
        mcp_tools=session.mcp_tools,
        skills=session.skills,
        truncated_turns=truncated_turns,
        features=features,
    )


def _format_digest(
    digest: TraceDigest,
    max_tool_calls_shown: int = 12,
    max_approvals_shown: int = _MAX_APPROVALS_SHOWN,
) -> str:
    """把轨迹摘要格式化为紧凑的 LLM 提示词文本。"""
    lines = [f"Trajectory {digest.session_uuid}"]
    if digest.user_name:
        lines.append(f"User: {digest.user_name}")
    if digest.mcp_tools:
        lines.append(f"MCP Tools: {', '.join(digest.mcp_tools)}")
    if digest.skills:
        lines.append(f"Skills: {', '.join(digest.skills)}")
    for t in digest.turns:
        lines.append(f"Turn {t.turn_index}")
        if t.user_text:
            lines.append(f"  User: {t.user_text}")
        if t.tool_calls:
            shown = t.tool_calls[:max_tool_calls_shown]
            parts = [
                f"{c.name}({c.args_head})" if c.args_head else c.name
                for c in shown
            ]
            remaining = len(t.tool_calls) - len(shown)
            if remaining > 0:
                parts.append(f"+{remaining} more")
            lines.append(f"  Tools: {', '.join(parts)}")
        for sa in t.sub_agents:
            usage = ", ".join(
                f"{name}×{count}" for name, count in sa.tool_usage.items()
            )
            lines.append(
                f"  Sub-agent {sa.tool_name}: {sa.status or 'unknown'}, "
                f"{sa.message_count} msgs, {sa.tool_call_count} tool calls"
                + (f" ({usage})" if usage else "")
                + (f"; task: {sa.prompt_preview}" if sa.prompt_preview else "")
            )
        if t.approvals:
            rejected = [a for a in t.approvals if not a.approved]
            approved = [a for a in t.approvals if a.approved]
            parts: list[str] = []
            for a in rejected:
                parts.append(f"REJECTED {a.tool_name}"
                             + (f" ({a.reason})" if a.reason else ""))
            for a in approved[:max_approvals_shown]:
                parts.append(f"APPROVED {a.tool_name}"
                             + (f" ({a.reason})" if a.reason else ""))
            remaining = len(approved) - max_approvals_shown
            if remaining > 0:
                parts.append(f"+{remaining} more approved")
            lines.append(f"  Checkpoints: {'; '.join(parts)}")
        if t.assistant_text:
            lines.append(f"  Assistant: {t.assistant_text}")
    if digest.truncated_turns:
        lines.append(f"(trajectory has more turns; only the first {len(digest.turns)} are shown)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 逐条分析
# ---------------------------------------------------------------------------


def _analyze_trace(
    client: LLMClient,
    digest: TraceDigest,
    max_tokens: int = _ANALYSIS_MAX_TOKENS,
    retries: int = _MAX_RETRIES,
) -> TraceTaskAnalysis:
    """让模型分析一条轨迹，返回任务与人介入分析；失败时返回 failed 条目。"""
    user_prompt = _format_digest(digest)

    for attempt in range(1, retries + 1):
        try:
            raw = client.chat_json(
                TRACE_ANALYSIS_SYSTEM_PROMPT, user_prompt, max_tokens=max_tokens,
            )
            if not isinstance(raw, dict):
                raise ValueError("analysis response is not a JSON object")

            interventions: list[Intervention] = []
            raw_interventions = raw.get("interventions") or []
            if isinstance(raw_interventions, list):
                for item in raw_interventions:
                    if not isinstance(item, dict):
                        continue
                    turn_index = item.get("turn_index")
                    if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 1:
                        turn_index = None
                    interventions.append(Intervention(
                        turn_index=turn_index,
                        type=str(item.get("type", "")).strip(),
                        position=str(item.get("position", "")).strip(),
                        description=str(item.get("description", "")).strip(),
                    ))
            # 只保留类型或描述至少有一项为真的介入，过滤空壳
            interventions = [iv for iv in interventions if iv.type or iv.description]

            return TraceTaskAnalysis(
                session_uuid=digest.session_uuid,
                user_name=digest.user_name,
                task_category=str(raw.get("task_category", "")).strip(),
                task_summary=str(raw.get("task_summary", "")).strip(),
                human_intervention=bool(interventions),
                interventions=interventions,
                features=digest.features,
                failed=False,
            )
        except (ReadTimeout, openai.APITimeoutError) as exc:
            logger.warning(
                "Analysis of %s timed out (attempt %d/%d): %s",
                digest.session_uuid, attempt, retries, exc,
            )
            continue
        except Exception as exc:
            logger.warning(
                "Analysis of %s failed (attempt %d/%d): %s",
                digest.session_uuid, attempt, retries, exc,
            )
            continue

    logger.error("Analysis of %s failed after %d retries.", digest.session_uuid, retries)
    return TraceTaskAnalysis(
        session_uuid=digest.session_uuid,
        user_name=digest.user_name,
        features=digest.features,
        failed=True,
    )


def run_trace_analyses(
    client: LLMClient,
    digests: list[TraceDigest],
    trajectories_dir: Path,
    max_workers: int = 8,
    max_tokens: int = _ANALYSIS_MAX_TOKENS,
    retries: int = _MAX_RETRIES,
) -> tuple[list[TraceTaskAnalysis], list[str]]:
    """并发逐条分析全部轨迹；已存在的中间文件直接复用。

    返回 ``(analyses, failed_uuids)``：analyses 按输入顺序排列，每条轨迹
    必有且仅有一条（失败时 ``failed=True``，不会静默缺失）。
    """
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    total = len(digests)
    if total == 0:
        return [], []

    def analyze_one(digest: TraceDigest) -> TraceTaskAnalysis:
        out_file = trajectories_dir / f"{digest.session_uuid}.json"
        if out_file.exists():
            try:
                data = json.loads(out_file.read_text(encoding="utf-8"))
                cached = TraceTaskAnalysis.model_validate(data)
                logger.info("Reused cached analysis for %s", digest.session_uuid)
                return cached
            except Exception:
                logger.warning("Cached analysis for %s is invalid, re-analyzing",
                               digest.session_uuid)
        analysis = _analyze_trace(client, digest, max_tokens, retries)
        if not analysis.failed:
            out_file.write_text(
                json.dumps(analysis.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return analysis

    results: list[TraceTaskAnalysis] = []
    failed_uuids: list[str] = []
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(analyze_one, d): d for d in digests}
        for future in as_completed(future_map):
            digest = future_map[future]
            try:
                analysis = future.result()
            except Exception as exc:  # 防御：单条线程异常不拖垮整体
                logger.error("Unexpected failure analyzing %s: %s", digest.session_uuid, exc)
                analysis = TraceTaskAnalysis(
                    session_uuid=digest.session_uuid,
                    user_name=digest.user_name,
                    features=digest.features,
                    failed=True,
                )
            results.append(analysis)
            if analysis.failed:
                failed_uuids.append(analysis.session_uuid)
            done += 1
            _render_progress(done, total)
            sys.stderr.flush()

    sys.stderr.write("\n")
    sys.stderr.flush()

    order = {d.session_uuid: i for i, d in enumerate(digests)}
    results.sort(key=lambda a: order.get(a.session_uuid, len(order)))
    logger.info("Trace analysis complete: %d/%d analyzed, %d failed",
                len(results) - len(failed_uuids), total, len(failed_uuids))
    return results, failed_uuids


# ---------------------------------------------------------------------------
# 跨轨迹聚合
# ---------------------------------------------------------------------------


def _format_aggregation_input(analyses: list[TraceTaskAnalysis]) -> str:
    task_counter = Counter(a.task_category for a in analyses if a.task_category and not a.failed)
    type_counter = Counter(
        iv.type for a in analyses if not a.failed for iv in a.interventions if iv.type
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
    """把自由形式的任务分类与人工介入类型聚合为固定种类。"""
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
    """将逐条分析阶段的分类替换为聚合后的标准分类，并返回固定种类与原始→标准映射。

    聚合失败（aggregation 为 None）时退化为恒等映射：原始分类即标准分类。
    失败条目（failed=True）不参与映射与计数。
    """
    task_mapping = _extract_mapping(aggregation, "task_category_mapping")
    type_mapping = _extract_mapping(aggregation, "intervention_type_mapping")
    task_defs = _extract_categories(aggregation, "task_categories")
    type_defs = _extract_categories(aggregation, "intervention_types")

    for a in analyses:
        if a.failed:
            continue
        if a.task_category:
            a.task_category_raw = a.task_category
            a.task_category = task_mapping.get(a.task_category, a.task_category)
        for iv in a.interventions:
            if iv.type:
                iv.type_raw = iv.type
                iv.type = type_mapping.get(iv.type, iv.type)

    task_counter = Counter(a.task_category for a in analyses if a.task_category and not a.failed)
    type_counter = Counter(
        iv.type for a in analyses if not a.failed for iv in a.interventions if iv.type
    )

    if not task_defs:
        task_defs = [{"name": name, "description": ""} for name in task_counter]
    if not type_defs:
        type_defs = [{"name": name, "description": ""} for name in type_counter]

    task_members: dict[str, list[str]] = {}
    for a in analyses:
        if a.failed or not a.task_category:
            continue
        task_members.setdefault(a.task_category, []).append(a.session_uuid)
    type_members: dict[str, list[str]] = {}
    for a in analyses:
        if a.failed:
            continue
        for iv in a.interventions:
            if iv.type:
                type_members.setdefault(iv.type, []).append(a.session_uuid)

    task_categories = [
        CategoryDefinition(name=d["name"], description=d["description"],
                           count=task_counter.get(d["name"], 0),
                           session_uuids=task_members.get(d["name"], []))
        for d in task_defs
    ]
    type_categories = [
        CategoryDefinition(name=d["name"], description=d["description"],
                           count=type_counter.get(d["name"], 0),
                           session_uuids=type_members.get(d["name"], []))
        for d in type_defs
    ]

    covered = {c.name for c in task_categories}
    for name, count in task_counter.items():
        if name not in covered:
            task_categories.append(CategoryDefinition(
                name=name, count=count, session_uuids=task_members.get(name, []),
            ))
    covered = {c.name for c in type_categories}
    for name, count in type_counter.items():
        if name not in covered:
            type_categories.append(CategoryDefinition(
                name=name, count=count, session_uuids=type_members.get(name, []),
            ))

    task_categories = [c for c in task_categories if c.count > 0]
    type_categories = [c for c in type_categories if c.count > 0]
    return task_categories, type_categories, task_mapping, type_mapping


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


def build_summary(
    analyses: list[TraceTaskAnalysis],
    failed_uuids: list[str],
    reconstruction: dict,
    total_trajectories: int,
) -> dict:
    """汇总统计：分类分布按真实计数（每条轨迹/每次介入计 1），并附特征总览。"""
    valid = [a for a in analyses if not a.failed]
    task_counter = Counter(a.task_category for a in valid if a.task_category)
    type_counter = Counter(
        iv.type for a in valid for iv in a.interventions if iv.type
    )
    intervention_count = sum(1 for a in valid if a.human_intervention)

    tool_counter: Counter[str] = Counter()
    total_tool_calls = 0
    turn_counts: list[int] = []
    total_approvals = 0
    total_rejections = 0
    sub_agent_count = 0
    sub_agent_tool_calls = 0
    for a in valid:
        features = a.features or {}
        total_tool_calls += int(features.get("tool_call_count", 0) or 0)
        for name, count in (features.get("tool_usage") or {}).items():
            tool_counter[name] += int(count or 0)
        if features.get("turn_count") is not None:
            turn_counts.append(int(features["turn_count"]))
        total_approvals += int(features.get("approval_count", 0) or 0)
        total_rejections += int(features.get("rejection_count", 0) or 0)
        sub_agent_count += int(features.get("sub_agent_count", 0) or 0)
        sub_agent_tool_calls += int(features.get("sub_agent_tool_calls", 0) or 0)

    return {
        "total_documents": reconstruction.get("total_documents", 0),
        "total_trajectories": total_trajectories,
        "analyzed_trajectories": len(valid),
        "failed_trajectories": len(failed_uuids),
        "intervention_sessions": intervention_count,
        "intervention_rate": f"{intervention_count / len(valid) * 100:.1f}%"
            if valid else "0%",
        "per_task_category_distribution": dict(task_counter),
        "per_intervention_type_distribution": dict(type_counter),
        "feature_overview": {
            "total_tool_calls": total_tool_calls,
            "top_tools": dict(tool_counter.most_common(10)),
            "avg_turns": round(sum(turn_counts) / len(turn_counts), 1)
                if turn_counts else 0,
            "max_turns": max(turn_counts) if turn_counts else 0,
            "total_approvals": total_approvals,
            "total_rejections": total_rejections,
            "sub_agent_count": sub_agent_count,
            "sub_agent_tool_calls": sub_agent_tool_calls,
        },
        "reconstruction": reconstruction,
    }


# ---------------------------------------------------------------------------
# 进度条
# ---------------------------------------------------------------------------


def _render_progress(processed: int, total: int) -> str:
    pct = processed / total if total > 0 else 0
    filled = int(_BAR_WIDTH * pct)
    bar = "█" * filled + "░" * (_BAR_WIDTH - filled)
    return f"\r  [{bar}] {processed}/{total}"


# ---------------------------------------------------------------------------
# Markdown 总览输出（人类可读）
# ---------------------------------------------------------------------------


def _short_uuid(uuid: str) -> str:
    return uuid[:8] if uuid else ""


def render_markdown_overview(result: TraceAnalysisResult) -> str:
    """生成人类可读的分类总览 Markdown：分类数量 + 每条轨迹的具体分析。"""
    summary = result.summary
    lines: list[str] = ["# 轨迹分析总览", ""]

    lines.append("## 基本统计")
    lines.append("")
    lines.append(f"- 轨迹总数：{summary.get('total_trajectories', 0)}"
                 f"（分析成功 {summary.get('analyzed_trajectories', 0)}，"
                 f"失败 {summary.get('failed_trajectories', 0)}）")
    lines.append(f"- 含人工介入的轨迹：{summary.get('intervention_sessions', 0)}"
                 f"（{summary.get('intervention_rate', '0%')}）")
    features = summary.get("feature_overview", {})
    if features:
        lines.append(f"- 工具调用总数：{features.get('total_tool_calls', 0)}"
                     f"；常用工具：{', '.join(f'{k}×{v}' for k, v in features.get('top_tools', {}).items())}")
        lines.append(f"- 审批检查点：{features.get('total_approvals', 0)}"
                     f"（拒绝 {features.get('total_rejections', 0)}）"
                     f"；子代理 {features.get('sub_agent_count', 0)} 个"
                     f"（内部 {features.get('sub_agent_tool_calls', 0)} 次工具调用）")
    lines.append("")

    lines.append("## 任务分类总览")
    lines.append("")
    if result.task_categories:
        for i, cat in enumerate(result.task_categories, 1):
            members = ", ".join(_short_uuid(u) for u in cat.session_uuids[:8])
            more = f" 等 {cat.count} 条" if cat.count > 8 else ""
            lines.append(f"{i}. **{cat.name}**（{cat.count} 条）：{cat.description}")
            if members:
                lines.append(f"   - 轨迹：`{members}`{more}")
    else:
        lines.append("（无）")
    lines.append("")

    lines.append("## 人工介入类型总览")
    lines.append("")
    if result.intervention_types:
        for i, cat in enumerate(result.intervention_types, 1):
            lines.append(f"{i}. **{cat.name}**（{cat.count} 次）：{cat.description}")
    else:
        lines.append("（无）")
    lines.append("")

    lines.append("## 逐条轨迹分析")
    lines.append("")
    if not result.analyses:
        lines.append("（无）")
    for a in result.analyses:
        lines.append(f"### {a.session_uuid}（`{_short_uuid(a.session_uuid)}`）")
        lines.append("")
        if a.failed:
            lines.append("**分析失败**（多次重试后仍失败）")
            lines.append("")
            continue
        category = a.task_category
        if a.task_category_raw and a.task_category_raw != a.task_category:
            category = f"{a.task_category}（原始：{a.task_category_raw}）"
        lines.append(f"- 任务分类：**{category}**")
        if a.task_summary:
            lines.append(f"- 任务概括：{a.task_summary}")
        if a.features:
            feat = a.features
            parts = [f"轮次 {feat.get('turn_count', 0)}",
                     f"工具调用 {feat.get('tool_call_count', 0)}"]
            if feat.get("approval_count"):
                parts.append(f"审批 {feat.get('approval_count')}"
                             + (f"（拒绝 {feat.get('rejection_count')}）" if feat.get("rejection_count") else ""))
            if feat.get("sub_agent_count"):
                parts.append(f"子代理 {feat.get('sub_agent_count')}")
            lines.append(f"- 特征：{', '.join(str(p) for p in parts)}")
        if a.human_intervention:
            lines.append(f"- 人工介入（{len(a.interventions)} 次）：")
            for iv in a.interventions:
                iv_type = iv.type
                if iv.type_raw and iv.type_raw != iv.type:
                    iv_type = f"{iv.type}（原始：{iv.type_raw}）"
                where = f"第{iv.turn_index}轮" if iv.turn_index else "位置未知"
                lines.append(f"  - [{where}] **{iv_type}**：{iv.description}")
        else:
            lines.append("- 人工介入：无")
        lines.append("")
    return "\n".join(lines)


def write_markdown_overview(result: TraceAnalysisResult, path: Path) -> Path:
    """把分类总览写入 Markdown 文件并返回路径。"""
    path.write_text(render_markdown_overview(result), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 流水线入口
# ---------------------------------------------------------------------------


def run_trace_analysis(
    config: Config,
    client: LLMClient | None = None,
) -> TraceAnalysisResult:
    """运行轨迹分析流水线：加载 → 摘要压缩 → 逐条并行分析 → 聚合 → 输出。

    分析阶段为**逐条并行**：每次 LLM 调用只包含一条轨迹的紧凑摘要，输出
    上限固定 ``_ANALYSIS_MAX_TOKENS``，不存在 batch 拼接，token 天然不会
    超限；因此每条轨迹可以放宽截断上限（``max_turns`` / ``max_user_chars``
    / ``max_assistant_chars`` / ``max_tool_args_chars``）以保留更多内容，
    并发由 ``max_workers`` 控制。
    """
    # Step 1: 从指定目录加载会话（自动展开 compressed_msgs）
    traces, reconstruction = load_sessions(config.paths.sessions_dir)
    if not traces:
        raise RuntimeError(
            f"No valid session files found in {config.paths.sessions_dir}"
        )

    # Step 2: 本地压缩为紧凑摘要（每轮截断 + 工具调用 + 审批检查点 + 特征）
    ta_config = config.trace_analysis
    digests = [
        build_trace_digest(
            session,
            user_name=user_name,
            max_turns=ta_config.max_turns,
            max_user_chars=ta_config.max_user_chars,
            max_assistant_chars=ta_config.max_assistant_chars,
            max_tool_args_chars=ta_config.max_tool_args_chars,
        )
        for user_name, session in traces
    ]
    prompt_chars = sum(
        len(_format_digest(d, max_tool_calls_shown=ta_config.max_tool_calls_shown))
        for d in digests
    )
    logger.info("Built %d trace digests (total prompt chars ≈ %d)",
                len(digests), prompt_chars)

    if client is None:
        client = LLMClient(config.llm)

    analysis_dir = config.paths.output_dir / "trace_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Step 3: 逐条并行分析（每条轨迹独立调用，中间结果落盘可续跑）
    analyses, failed_uuids = run_trace_analyses(
        client, digests, analysis_dir / "trajectories",
        max_workers=ta_config.max_workers,
    )

    # Step 4: 跨轨迹聚合为固定分类（已存在的聚合结果直接复用）
    aggregation_file = analysis_dir / "aggregation.json"
    aggregation: dict | None = None
    if aggregation_file.exists():
        try:
            aggregation = json.loads(aggregation_file.read_text(encoding="utf-8"))
            logger.info("Loaded aggregation result from %s", aggregation_file)
        except Exception:
            logger.warning("Reuse of aggregation file failed, re-aggregating")
    if aggregation is None and analyses:
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
    summary = build_summary(
        analyses, failed_uuids, reconstruction, total_trajectories=len(traces),
    )

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
    overview_file = write_markdown_overview(
        result, analysis_dir / "overview.md",
    )
    logger.info("Trace analysis complete: %d analyses, output written to %s",
                len(analyses), result_file)
    logger.info("Human-readable overview written to %s", overview_file)
    return result

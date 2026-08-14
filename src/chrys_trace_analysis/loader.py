"""数据加载：读取会话并构建 Session。

支持两种目录布局：

- 扁平布局 ``{sessions_dir}/{uuid}.json``（单个文件即一条会话）；
- 嵌套布局 ``{sessions_dir}/{uuid}/session.json``（chrys 本地 sessions
  目录的真实布局，如 ``AppData/Roaming/chrys/sessions/{uuid}/session.json``）。

嵌套布局下还会读取同目录的伴生信息：

- ``approvals/*.log``：工具调用前的人工审批检查点（人工交互的直接证据）；
- ``sub_agents/sessions/*.json``：子代理完整会话（主会话只保留最终结果）。

文件格式参考 chrys 本地 ``session.json`` envelope（``{"meta": {...},
"state": {...}}``），根字段可能存在冗余字段，一律容忍：

- ``state.messages`` 为 live 消息、``state.compressed_msgs`` 为压缩块时直接
  使用（envelope 形态）；
- 顶层 ``messages`` / ``compressed_msgs``（扁平或 MongoDB 文档形态）自动合并
  进 state。

加载流程：解析 envelope → 通过 :func:`reconstruct.reconstruct_messages` 将
``compressed_msgs`` 原位展开为完整消息 → 构建轮次摘要与完整轮次 → 解析
审批记录与子代理会话并按轮次对齐。
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import (
    ApprovalRecord,
    MessageContent,
    RawMessage,
    Session,
    SessionRound,
    SessionTurn,
    SubAgentRecord,
)
from .reconstruct import reconstruct_messages

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 原始消息解析（消息 dict → 模型）
# ---------------------------------------------------------------------------


def _extract_texts(item: dict) -> list[str]:
    """从一条 message dict 中提取所有 type=text 的 text 内容列表。"""
    texts: list[str] = []
    contents = item.get("contents", [])
    if isinstance(contents, list):
        for c in contents:
            if isinstance(c, dict) and c.get("type") == "text":
                t = c.get("text", "")
                if t:
                    texts.append(t)
    else:
        fallback = item.get("content", "")
        if fallback:
            texts.append(fallback)
    return texts


def _parse_message_content(content_dict: dict) -> MessageContent:
    """Parse a single content block into MessageContent."""
    return MessageContent(
        type=content_dict.get("type", ""),
        text=content_dict.get("text", ""),
        tool_name=content_dict.get("tool_name", ""),
        call_id=content_dict.get("call_id", ""),
        name=content_dict.get("name", ""),
        arguments=json.dumps(content_dict.get("arguments", {}), ensure_ascii=False)
            if isinstance(content_dict.get("arguments"), dict) else str(content_dict.get("arguments", "")),
        result=json.dumps(content_dict.get("result", {}), ensure_ascii=False)
            if isinstance(content_dict.get("result"), dict) else str(content_dict.get("result", "")),
    )


def _parse_raw_message(msg_dict: dict) -> RawMessage:
    """Parse a raw message dict into RawMessage."""
    contents: list[MessageContent] = []
    raw_contents = msg_dict.get("contents", [])
    if isinstance(raw_contents, list):
        for c in raw_contents:
            if isinstance(c, dict):
                contents.append(_parse_message_content(c))

    addl = msg_dict.get("additional_properties")
    additional_properties = dict(addl) if isinstance(addl, dict) else {}

    return RawMessage(
        role=msg_dict.get("role", ""),
        contents=contents,
        additional_properties=additional_properties,
        message_id=msg_dict.get("message_id", ""),
    )


def build_session_abstract(messages: list[dict]) -> list[SessionRound]:
    """遍历 messages，按轮次划分构建 SessionRound 列表。

    轮次分界：role=assistant 且 additional_properties._chrys_kind="turn" 的消息。
    每个轮次块中：
      - user_msg = 块内第一条 user text
      - assistant_reply = 块内最后一条非 turn 分界的 assistant text
    """
    abstract: list[SessionRound] = []
    if not isinstance(messages, list) or not messages:
        return abstract

    items = [m for m in messages if isinstance(m, dict)]
    if not items:
        return abstract

    # 按 turn 分界切割块
    turns: list[list[dict]] = []
    current_turn: list[dict] = []
    for item in items:
        role = item.get("role", "")
        if role == "assistant":
            addl = item.get("additional_properties")
            if isinstance(addl, dict) and addl.get("_chrys_kind") == "turn":
                if current_turn:
                    turns.append(current_turn)
                    current_turn = []
                continue
        current_turn.append(item)

    if current_turn:
        turns.append(current_turn)

    for turn in turns:
        user_msg = ""
        assistant_reply = ""

        # 找块内第一条 user text
        for item in turn:
            if item.get("role") == "user":
                user_texts = _extract_texts(item)
                if user_texts:
                    user_msg = "\n".join(user_texts)
                    break

        # 找块内最后一条非 turn 分界的 assistant text
        for item in reversed(turn):
            if item.get("role") == "assistant":
                asst_texts = _extract_texts(item)
                if asst_texts:
                    assistant_reply = "\n".join(asst_texts)
                    break

        if user_msg or assistant_reply:
            abstract.append(SessionRound(user_msg=user_msg, assistant_reply=assistant_reply))

    return abstract


def build_session_turns(messages: list[dict]) -> list[SessionTurn]:
    """遍历 messages，按轮次划分构建 SessionTurn 列表，保留完整原始消息。

    轮次分界：role=assistant 且 additional_properties._chrys_kind="turn" 的消息
    （该标记消息被丢弃）。
    返回每个轮次内所有原始消息的列表。
    """
    turns: list[SessionTurn] = []
    if not isinstance(messages, list) or not messages:
        return turns

    items = [m for m in messages if isinstance(m, dict)]
    if not items:
        return turns

    # 按 turn 分界切割块
    turn_blocks: list[list[dict]] = []
    current_turn: list[dict] = []
    for item in items:
        role = item.get("role", "")
        if role == "assistant":
            addl = item.get("additional_properties")
            if isinstance(addl, dict) and addl.get("_chrys_kind") == "turn":
                if current_turn:
                    turn_blocks.append(current_turn)
                    current_turn = []
                continue
        current_turn.append(item)

    if current_turn:
        turn_blocks.append(current_turn)

    for i, block in enumerate(turn_blocks, 1):
        raw_messages = [_parse_raw_message(msg) for msg in block]
        turns.append(SessionTurn(turn_index=i, messages=raw_messages))

    return turns


# ---------------------------------------------------------------------------
# envelope 适配与加载
# ---------------------------------------------------------------------------


def to_reconstruct_envelope(doc: Mapping[str, Any]) -> dict[str, Any]:
    """将任意形态的会话文档适配为 reconstruct 期望的 envelope 布局。

    - envelope 形态（``state`` 内含 ``messages``）原样返回；
    - 扁平 / MongoDB 形态（顶层 ``messages`` / ``compressed_msgs``）合并进 state。
    """
    state = doc.get("state")
    if isinstance(state, Mapping) and isinstance(state.get("messages"), list):
        return dict(doc)

    state = dict(state) if isinstance(state, Mapping) else {}
    if isinstance(doc.get("messages"), list):
        state["messages"] = list(doc["messages"])
    if "compressed_msgs" not in state and isinstance(doc.get("compressed_msgs"), list):
        state["compressed_msgs"] = list(doc["compressed_msgs"])

    envelope = {k: v for k, v in doc.items() if k != "state"}
    envelope["state"] = state
    return envelope


def _collect_names(items: Any, name_key: str) -> list[str]:
    """从 MCP 工具 / 技能条目列表提取名称。"""
    names: list[str] = []
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                name = item.get(name_key, "")
                if name:
                    names.append(name)
    return names


def _session_uuid(doc: Mapping[str, Any], filename_stem: str) -> str:
    """解析会话 uuid：优先根字段 uuid，其次 meta.session_id，最后文件名。"""
    uuid = doc.get("uuid", "")
    if isinstance(uuid, str) and uuid:
        return uuid
    meta = doc.get("meta")
    if isinstance(meta, Mapping):
        sid = meta.get("session_id", "")
        if isinstance(sid, str) and sid:
            return sid
    return filename_stem


# ---------------------------------------------------------------------------
# 会话文件发现（扁平 / 嵌套布局）
# ---------------------------------------------------------------------------


def _discover_session_files(
    sessions_dir: Path,
) -> list[tuple[Path, Path | None]]:
    """发现会话文件，返回 ``(session_file, session_dir)`` 列表。

    - 扁平布局：``{sessions_dir}/{uuid}.json``（session_dir 为 None）；
    - 嵌套布局：``{sessions_dir}/{uuid}/session.json``（session_dir 为目录）。

    同一会话同时存在两种布局时，嵌套布局优先（信息更完整）；扁平文件若
    存在同名目录（``{sessions_dir}/{stem}/``），视为嵌套布局处理。
    """
    discovered: list[tuple[Path, Path | None]] = []
    seen_uuids: set[str] = set()

    for fp in sorted(sessions_dir.glob("*.json")):
        # 扁平文件：若存在同名目录则按嵌套处理（uuid 用目录名）
        sibling_dir = sessions_dir / fp.stem
        if fp.name == "session.json":
            # sessions_dir 本身就是某个会话的目录（含 session.json）
            discovered.append((fp, sessions_dir))
            seen_uuids.add(fp.stem)
        elif sibling_dir.is_dir():
            discovered.append((fp, sibling_dir))
            seen_uuids.add(fp.stem)
        else:
            discovered.append((fp, None))

    for session_dir in sorted(
        p for p in sessions_dir.iterdir() if p.is_dir()
    ):
        sf = session_dir / "session.json"
        if not sf.is_file():
            continue
        if session_dir.name in seen_uuids:
            continue  # 已由扁平文件+同名目录登记
        discovered.append((sf, session_dir))

    return discovered


# ---------------------------------------------------------------------------
# 审批记录解析（approvals/*.log）
# ---------------------------------------------------------------------------

_APPROVAL_UTC_RE = re.compile(r"Current time \(UTC\):\s*(\S+)")
_APPROVAL_PROMPT_RE = re.compile(
    r"Latest user prompt:\s*\n?(.+?)\n\s*\n(?:---|Proposed action)",
    re.DOTALL,
)
_APPROVAL_TOOL_RE = re.compile(r"Tool:\s*(\S+)\s*\(kind:\s*([^)]*)\)")
_APPROVAL_ARGS_RE = re.compile(
    r"Arguments:\s*\n(.*?)\n\s*\n--- RESPONSE ---", re.DOTALL,
)
_APPROVAL_RESPONSE_RE = re.compile(
    r"--- RESPONSE ---\s*(.*?)\s*--- VERDICT ---", re.DOTALL,
)
_APPROVAL_VERDICT_RE = re.compile(r"VERDICT:\s*(APPROVED|REJECTED|DENIED)", re.IGNORECASE)


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _parse_approval_response(response_block: str) -> tuple[bool, str]:
    """从 RESPONSE 块解析 ``{"approved": true/false, "reason": "..."}``。"""
    approved = True
    reason = ""
    response_block = response_block.strip()
    try:
        data = json.loads(response_block)
        if isinstance(data, dict):
            approved = bool(data.get("approved", True))
            reason = str(data.get("reason", "")).strip()
            return approved, reason
    except Exception:
        pass
    m = re.search(r'"approved"\s*:\s*(true|false)', response_block)
    if m:
        approved = m.group(1).lower() == "true"
    m = re.search(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)"', response_block)
    if m:
        reason = m.group(1).replace("\\n", " ").strip()
    return approved, reason


def parse_approval_log(text: str) -> ApprovalRecord | None:
    """解析一份审批日志文本为 ApprovalRecord；无法识别时返回 None。"""
    timestamp = ""
    m = _APPROVAL_UTC_RE.search(text)
    if m:
        timestamp = m.group(1)

    prompt = ""
    m = _APPROVAL_PROMPT_RE.search(text)
    if m:
        prompt = m.group(1).strip()

    tool_name = ""
    kind = ""
    m = _APPROVAL_TOOL_RE.search(text)
    if m:
        tool_name = m.group(1).strip()
        kind = m.group(2).strip()

    args_head = ""
    m = _APPROVAL_ARGS_RE.search(text)
    if m:
        args_head = _truncate(m.group(1), 200)

    approved = True
    reason = ""
    m = _APPROVAL_RESPONSE_RE.search(text)
    if m:
        approved, reason = _parse_approval_response(m.group(1))
    else:
        m = _APPROVAL_VERDICT_RE.search(text)
        if m:
            approved = m.group(1).upper() != "REJECTED"

    if not tool_name and not prompt:
        return None
    return ApprovalRecord(
        timestamp=timestamp,
        tool_name=tool_name,
        kind=kind,
        prompt=prompt,
        approved=approved,
        reason=_truncate(reason, 200),
        arguments_head=args_head,
    )


def _load_approvals(session_dir: Path) -> list[ApprovalRecord]:
    """读取会话目录下的全部审批日志（approvals/*.log）。"""
    approvals: list[ApprovalRecord] = []
    approvals_dir = session_dir / "approvals"
    if not approvals_dir.is_dir():
        return approvals
    for fp in sorted(approvals_dir.glob("*.log")):
        try:
            record = parse_approval_log(fp.read_text(encoding="utf-8-sig"))
        except Exception:
            logger.warning("Failed to parse approval log %s", fp, exc_info=True)
            continue
        if record is not None:
            approvals.append(record)
    return approvals


# ---------------------------------------------------------------------------
# 子代理会话摘要（sub_agents/sessions/*.json）
# ---------------------------------------------------------------------------


def _count_sub_agent_calls(state: Mapping[str, Any]) -> tuple[int, Counter[str], int]:
    """统计子代理会话的消息数、function_call 次数与工具分布。"""
    message_count = 0
    tool_counter: Counter[str] = Counter()
    tool_calls = 0

    def count_messages(messages: Any) -> None:
        nonlocal message_count, tool_calls
        if not isinstance(messages, list):
            return
        for msg in messages:
            if not isinstance(msg, Mapping):
                continue
            message_count += 1
            contents = msg.get("contents")
            if not isinstance(contents, list):
                continue
            for content in contents:
                if not isinstance(content, Mapping):
                    continue
                if content.get("type") == "function_call":
                    tool_calls += 1
                    name = content.get("name", "")
                    if isinstance(name, str) and name:
                        tool_counter[name] += 1

    count_messages(state.get("messages"))
    blocks = state.get("compressed_msgs")
    if isinstance(blocks, list):
        for block in blocks:
            if isinstance(block, Mapping):
                count_messages(block.get("messages"))
    return message_count, tool_counter, tool_calls


def parse_sub_agent_session(path: Path) -> SubAgentRecord | None:
    """解析一个子代理会话文件为 SubAgentRecord；失败时返回 None。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        logger.warning("Failed to parse sub-agent session %s", path, exc_info=True)
        return None
    if not isinstance(data, Mapping):
        return None

    meta = data.get("meta")
    meta = dict(meta) if isinstance(meta, Mapping) else {}
    state = data.get("state")
    state = dict(state) if isinstance(state, Mapping) else {}

    message_count, tool_counter, tool_calls = _count_sub_agent_calls(state)

    return SubAgentRecord(
        tool_name=str(meta.get("tool_name", "")),
        status=str(meta.get("status", "")),
        parent_call_id=str(meta.get("parent_provider_call_id", "")),
        prompt_preview=_truncate(str(meta.get("prompt_preview", "")), 200),
        message_count=message_count,
        tool_call_count=tool_calls,
        tool_usage=dict(tool_counter.most_common(10)),
    )


def _load_sub_agents(session_dir: Path) -> list[SubAgentRecord]:
    """读取会话目录下的全部子代理会话摘要。"""
    records: list[SubAgentRecord] = []
    sub_dir = session_dir / "sub_agents" / "sessions"
    if not sub_dir.is_dir():
        return records
    for fp in sorted(sub_dir.glob("*.json")):
        record = parse_sub_agent_session(fp)
        if record is not None:
            records.append(record)
    return records


# ---------------------------------------------------------------------------
# 伴生信息 → 轮次对齐
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _turn_user_times(session: Session) -> list[tuple[int, datetime]]:
    """每轮第一条 user 消息的创建时间（additional_properties._chrys_created_at）。"""
    times: list[tuple[int, datetime]] = []
    for turn in session.turns:
        for msg in turn.messages:
            if msg.role != "user":
                continue
            ts = msg.additional_properties.get("_chrys_created_at")
            if isinstance(ts, str) and ts:
                try:
                    times.append((turn.turn_index, datetime.fromisoformat(ts)))
                except ValueError:
                    pass
            break
    return times


def _align_approvals(
    approvals: list[ApprovalRecord],
    session: Session,
) -> list[ApprovalRecord]:
    """按审批时的用户提示（文本匹配）或时间戳把审批对齐到轮次。"""
    rounds = session.session_abstract
    turn_times = _turn_user_times(session)

    for approval in approvals:
        # 策略 1：Latest user prompt 与轮次 user 消息文本匹配（归一化后相等/包含）
        prompt_norm = _normalize(approval.prompt)
        if prompt_norm:
            for i, round_ in enumerate(rounds, 1):
                user_norm = _normalize(round_.user_msg)
                if user_norm and (prompt_norm == user_norm
                                  or prompt_norm in user_norm
                                  or user_norm in prompt_norm):
                    approval.turn_index = i
                    break
        # 策略 2：按时间戳对齐到最近的已开始轮次
        if approval.turn_index is None and approval.timestamp and turn_times:
            try:
                ts = datetime.fromisoformat(approval.timestamp)
            except ValueError:
                ts = None
            if ts is not None:
                best: tuple[int, datetime] | None = None
                for turn_index, user_time in turn_times:
                    if user_time <= ts and (best is None or user_time > best[1]):
                        best = (turn_index, user_time)
                if best is not None:
                    approval.turn_index = best[0]
    return approvals


def _align_sub_agents(
    records: list[SubAgentRecord],
    session: Session,
) -> list[SubAgentRecord]:
    """按 parent call_id 匹配主会话轮次内的 function_call，对齐子代理到轮次。"""
    for record in records:
        if not record.parent_call_id:
            continue
        for turn in session.turns:
            for msg in turn.messages:
                for content in msg.contents:
                    if (content.type == "function_call"
                            and content.call_id == record.parent_call_id):
                        record.turn_index = turn.turn_index
                        break
    return records


def load_sessions(
    sessions_dir: Path,
) -> tuple[list[tuple[str, Session]], dict[str, int]]:
    """加载 ``{sessions_dir}`` 下的会话（扁平 ``{uuid}.json`` 或嵌套
    ``{uuid}/session.json``）。

    每条会话展开 ``compressed_msgs`` 还原完整轨迹；嵌套布局额外读取
    ``approvals/`` 审批记录与 ``sub_agents/sessions/`` 子代理会话摘要并
    对齐到轮次。跳过畸形文件与还原失败的文档。返回 ``(traces,
    diagnostics)``：``traces`` 为 ``(user_name, Session)`` 列表（user_name
    取自 ``meta.user_id``，缺失时为空串）。
    """
    if not sessions_dir.is_dir():
        raise FileNotFoundError(
            f"Session directory not found: {sessions_dir}. "
            f"Please put {{uuid}}.json or {{uuid}}/session.json files "
            f"under this directory."
        )

    files = [fp for fp, _ in _discover_session_files(sessions_dir)]
    if not files:
        logger.warning("No session files found in %s", sessions_dir)

    diagnostics: dict[str, int] = {
        "total_documents": len(files),
        "restored_message_count": 0,
        "spliced_block_count": 0,
        "orphan_block_count": 0,
        "unresolved_summary_count": 0,
        "reconstruction_failures": 0,
        "approval_records": 0,
        "sub_agent_sessions": 0,
    }

    traces: list[tuple[str, Session]] = []
    for fp, session_dir in _discover_session_files(sessions_dir):
        try:
            data = json.loads(fp.read_text(encoding="utf-8-sig"))
        except Exception:
            logger.warning("Skipping %s: failed to parse", fp.name, exc_info=True)
            continue
        if not isinstance(data, dict):
            logger.warning("Skipping %s: root is not a JSON object", fp.name)
            continue

        session_uuid = _session_uuid(data, fp.stem)
        meta = data.get("meta")
        meta = dict(meta) if isinstance(meta, Mapping) else {}
        user_name = meta.get("user_id", "") if isinstance(meta.get("user_id"), str) else ""

        try:
            result = reconstruct_messages(to_reconstruct_envelope(data))
        except Exception as exc:
            diagnostics["reconstruction_failures"] += 1
            logger.warning("Skipping %s: reconstruction failed: %s",
                           session_uuid or fp.name, exc)
            continue

        messages = result.restored_messages
        diagnostics["restored_message_count"] += result.diagnostics.restored_message_count
        diagnostics["spliced_block_count"] += result.diagnostics.spliced_block_count
        diagnostics["orphan_block_count"] += result.diagnostics.orphan_block_count
        diagnostics["unresolved_summary_count"] += result.diagnostics.unresolved_summary_count

        session = Session(
            session_uuid=session_uuid,
            session_abstract=build_session_abstract(messages),
            mcp_tools=_collect_names(data.get("mcp_tools"), "tool_name"),
            skills=_collect_names(data.get("skills"), "skill_name"),
            turns=build_session_turns(messages),
            meta=meta,
        )

        # 嵌套布局：读取审批记录与子代理会话摘要，并对齐到轮次
        if session_dir is not None:
            approvals = _load_approvals(session_dir)
            sub_agents = _load_sub_agents(session_dir)
            if approvals:
                _align_approvals(approvals, session)
                session.approvals = approvals
                diagnostics["approval_records"] += len(approvals)
            if sub_agents:
                _align_sub_agents(sub_agents, session)
                session.sub_agents = sub_agents
                diagnostics["sub_agent_sessions"] += len(sub_agents)

        traces.append((user_name, session))

    logger.info(
        "Loaded %d sessions from %s "
        "(restored %d messages, spliced %d blocks, orphaned %d blocks, "
        "unresolved summaries %d, failures %d, approvals %d, sub-agents %d)",
        len(traces), sessions_dir,
        diagnostics["restored_message_count"],
        diagnostics["spliced_block_count"],
        diagnostics["orphan_block_count"],
        diagnostics["unresolved_summary_count"],
        diagnostics["reconstruction_failures"],
        diagnostics["approval_records"],
        diagnostics["sub_agent_sessions"],
    )
    return traces, diagnostics

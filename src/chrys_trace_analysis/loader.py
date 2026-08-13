"""数据加载：读取指定目录下的 ``{uuid}.json`` 会话文件并构建 Session。

输入目录中每个 ``.json`` 文件对应一条会话，格式参考 chrys 本地的
``session.json`` envelope（``{"meta": {...}, "state": {...}}``），根字段可能
存在冗余字段，一律容忍：

- ``state.messages`` 为 live 消息、``state.compressed_msgs`` 为压缩块时直接
  使用（envelope 形态）；
- 顶层 ``messages`` / ``compressed_msgs``（扁平或 MongoDB 文档形态）自动合并
  进 state。

加载流程：解析 envelope → 通过 :func:`reconstruct.reconstruct_messages` 将
``compressed_msgs`` 原位展开为完整消息 → 构建轮次摘要与完整轮次。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .models import MessageContent, RawMessage, Session, SessionRound, SessionTurn
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


def load_sessions(
    sessions_dir: Path,
) -> tuple[list[tuple[str, Session]], dict[str, int]]:
    """加载 ``{sessions_dir}/*.json`` 会话（chrys envelope 格式）。

    每条会话展开 ``compressed_msgs`` 还原完整轨迹；跳过畸形文件与还原失败的
    文档。返回 ``(traces, diagnostics)``：``traces`` 为 ``(user_name, Session)``
    列表（user_name 取自 ``meta.user_id``，缺失时为空串）。
    """
    if not sessions_dir.is_dir():
        raise FileNotFoundError(
            f"Session directory not found: {sessions_dir}. "
            f"Please put {{uuid}}.json session files under this directory."
        )

    files = sorted(sessions_dir.glob("*.json"))
    if not files:
        logger.warning("No session files (*.json) found in %s", sessions_dir)

    diagnostics: dict[str, int] = {
        "total_documents": len(files),
        "restored_message_count": 0,
        "spliced_block_count": 0,
        "orphan_block_count": 0,
        "unresolved_summary_count": 0,
        "reconstruction_failures": 0,
    }

    traces: list[tuple[str, Session]] = []
    for fp in files:
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
        traces.append((user_name, session))

    logger.info(
        "Loaded %d sessions from %s "
        "(restored %d messages, spliced %d blocks, orphaned %d blocks, "
        "unresolved summaries %d, failures %d)",
        len(traces), sessions_dir,
        diagnostics["restored_message_count"],
        diagnostics["spliced_block_count"],
        diagnostics["orphan_block_count"],
        diagnostics["unresolved_summary_count"],
        diagnostics["reconstruction_failures"],
    )
    return traces, diagnostics

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .config import MongoConfig
from .models import MessageContent, RawMessage, Session, SessionRound, SessionTurn, UserData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Group config helpers
# ---------------------------------------------------------------------------


def parse_group_members(json_path: str | Path) -> list[dict]:
    """读取 group_members_stat.json，返回 list[dict]：
    {"group_id", "group_name", "members": [{"user_id", "department"}, ...]}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    groups = []
    for g in data.get("groups", []):
        members = [
            {"user_id": m["user_id"], "department": m.get("department", "")}
            for m in g.get("members", [])
            if m.get("user_id")
        ]
        groups.append({
            "group_id": str(g.get("group_id", "")),
            "group_name": g.get("group_name", ""),
            "members": members,
        })
    return groups


def _build_user_to_groups(groups: list[dict]) -> dict[str, list[str]]:
    """构建 user_id -> [group_name, ...] 映射。"""
    mapping: dict[str, list[str]] = {}
    for g in groups:
        for m in g.get("members", []):
            uid = m.get("user_id", "")
            if uid:
                mapping.setdefault(uid, []).append(g["group_name"])
    return mapping


# ---------------------------------------------------------------------------
# Raw message parsing
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
    """Parse a raw MongoDB message dict into RawMessage."""
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


# ---------------------------------------------------------------------------
# Session abstract construction
# ---------------------------------------------------------------------------


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
# MongoDB data fetching
# ---------------------------------------------------------------------------


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def fetch_sessions_by_user(
    user_ids: list[str],
    uri: str,
    database: str = "lingxi",
    collection: str = "sessions",
) -> dict[str, list[dict]]:
    """根据 user_id 列表批量查询 sessions collection。
    返回 dict[user_id, list[session_dict]]。
    """
    if not user_ids:
        return {}

    from pymongo import MongoClient

    client = MongoClient(uri)
    db = client[database]
    coll = db[collection]

    sessions_by_user: dict[str, list[dict]] = {}
    total = len(user_ids)
    for idx, uid in enumerate(user_ids, 1):
        cursor = (
            coll.find({"meta.user_id": uid})
            .sort("created_at", -1)
        )
        sessions = list(cursor)
        sessions_by_user[uid] = sessions if sessions else []

        if idx % 10 == 0 or idx == total:
            logger.info("MongoDB fetch progress: %d/%d", idx, total)

    client.close()
    return sessions_by_user


# ---------------------------------------------------------------------------
# MongoDB documents → UserData models
# ---------------------------------------------------------------------------


def mongo_docs_to_user_data(
    sessions_by_user: dict[str, list[dict]],
    user_to_groups: dict[str, list[str]] | None = None,
) -> list[UserData]:
    """将 MongoDB 拉取的数据转换为 UserData 列表。

    user_to_groups: 可选的 user_id -> [group_name, ...] 映射，会填充到 UserData.groups。
    """
    users: list[UserData] = []
    for uid, session_docs in sessions_by_user.items():
        total_add_lines = 0
        sessions: list[Session] = []
        for doc in session_docs:
            session_uuid = doc.get("uuid", "")

            # 构建 session_abstract（简化轮次视图）
            messages = doc.get("messages", [])
            abstract = build_session_abstract(messages)

            # 构建完整 turns（保留所有原始消息）
            turns = build_session_turns(messages)

            # MCP tools
            mcp_tools: list[str] = []
            raw_tools = doc.get("mcp_tools", [])
            if isinstance(raw_tools, list):
                for t in raw_tools:
                    if isinstance(t, dict):
                        name = t.get("tool_name", "")
                        if name:
                            mcp_tools.append(name)

            # Skills
            skills: list[str] = []
            raw_skills = doc.get("skills", [])
            if isinstance(raw_skills, list):
                for s in raw_skills:
                    if isinstance(s, dict):
                        name = s.get("skill_name", "")
                        if name:
                            skills.append(name)

            # add_lines
            mr = doc.get("mr_relation")
            if isinstance(mr, dict):
                total_add_lines += _safe_int(mr.get("add_lines_hits"))

            # Session metadata
            meta = doc.get("meta", {})
            if isinstance(meta, dict):
                meta = dict(meta)
            else:
                meta = {}

            sessions.append(Session(
                session_uuid=session_uuid,
                session_abstract=abstract,
                mcp_tools=mcp_tools,
                skills=skills,
                turns=turns,
                meta=meta,
            ))

        groups = user_to_groups.get(uid, []) if user_to_groups else []
        users.append(UserData(
            user_name=uid,
            total_add_lines=total_add_lines,
            groups=groups,
            sessions=sessions,
        ))

    return users


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def load_from_mongo(mongo_cfg: MongoConfig) -> list[UserData]:
    """完整的 MongoDB 加载流程：解析群组 → 拉取数据 → 转换为 UserData。"""
    # 1. 解析群组成员
    logger.info("Parsing group members from %s", mongo_cfg.members_file)
    groups = parse_group_members(mongo_cfg.members_file)
    groups = [g for g in groups if g["group_name"] != "Chrys Default"]
    logger.info("Loaded %d groups (excluded Chrys Default)", len(groups))

    user_to_groups = _build_user_to_groups(groups)

    # 收集所有 user_id
    all_user_ids = list(dict.fromkeys(
        m["user_id"] for g in groups for m in g["members"]
    ))
    logger.info("Total unique user IDs: %d", len(all_user_ids))

    # 2. 从 MongoDB 拉取数据
    logger.info("Fetching sessions from MongoDB...")
    sessions_by_user = fetch_sessions_by_user(
        all_user_ids,
        uri=mongo_cfg.uri,
        database=mongo_cfg.database,
        collection=mongo_cfg.collection,
    )

    users_with_data = sum(1 for v in sessions_by_user.values() if v)
    total_sessions = sum(len(v) for v in sessions_by_user.values())
    logger.info("Users with data: %d/%d, total sessions: %d",
                users_with_data, len(all_user_ids), total_sessions)

    # 3. 转换为 UserData
    users = mongo_docs_to_user_data(sessions_by_user, user_to_groups)
    logger.info("Loaded %d UserData objects", len(users))
    return users

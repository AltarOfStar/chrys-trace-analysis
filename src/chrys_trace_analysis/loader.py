from __future__ import annotations

import json
import logging
from pathlib import Path

from .models import Session, SimplifiedTrace, UserData

logger = logging.getLogger(__name__)


def load_simplified_traces(traces_dir: Path) -> list[tuple[str, Session]]:
    """加载 ``{traces_dir}/simplified/{uuid}.json`` 文件，返回 (user_name, Session) 列表。"""
    simplified_dir = traces_dir / "simplified"
    if not simplified_dir.is_dir():
        raise FileNotFoundError(
            f"Simplified traces directory not found: {simplified_dir}. "
            "Run with --mongo first to fetch and simplify traces from MongoDB."
        )

    all_sessions: list[tuple[str, Session]] = []
    json_files = sorted(simplified_dir.glob("*.json"))
    if not json_files:
        logger.warning("No simplified trace files found in %s", simplified_dir)

    for fp in json_files:
        try:
            trace = SimplifiedTrace.model_validate_json(fp.read_text(encoding="utf-8"))
            session = Session(
                session_uuid=trace.uuid,
                session_abstract=trace.session_abstract,
                mcp_tools=trace.mcp_tools,
                skills=trace.skills,
            )
            all_sessions.append((trace.user_name, session))
        except Exception:
            logger.warning("Skipping %s: failed to parse", fp.name, exc_info=True)

    logger.info("Loaded %d simplified traces from %s", len(all_sessions), simplified_dir)
    return all_sessions


def load_user_files(data_dir: Path) -> list[UserData]:
    users: list[UserData] = []
    json_files = sorted(data_dir.glob("*.json"))
    if not json_files:
        logger.warning("No JSON files found in %s", data_dir)

    for fp in json_files:
        try:
            content = fp.read_text(encoding="utf-8")
            user = UserData.model_validate_json(content)
            users.append(user)
            logger.info("Loaded %s (%d sessions)", fp.name, len(user.sessions))
        except Exception:
            logger.warning("Skipping %s: failed to parse", fp.name, exc_info=True)

    logger.info("Loaded %d user(s) with %d total sessions",
                len(users), sum(len(u.sessions) for u in users))
    return users

from __future__ import annotations

import json
import logging
from pathlib import Path

from .models import UserData

logger = logging.getLogger(__name__)


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

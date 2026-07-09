from __future__ import annotations

import random

from .models import Session, UserData


def sample_sessions(
    users: list[UserData],
    batch_size: int,
    seed: int | None = None,
) -> list[tuple[str, Session]]:
    all_sessions: list[tuple[str, Session]] = []
    for user in users:
        for session in user.sessions:
            all_sessions.append((user.user_name, session))

    if len(all_sessions) <= batch_size:
        return all_sessions

    rng = random.Random(seed)
    return rng.sample(all_sessions, batch_size)


def flatten_all_sessions(
    users: list[UserData],
) -> list[tuple[str, Session]]:
    all_sessions: list[tuple[str, Session]] = []
    for user in users:
        for session in user.sessions:
            all_sessions.append((user.user_name, session))
    return all_sessions

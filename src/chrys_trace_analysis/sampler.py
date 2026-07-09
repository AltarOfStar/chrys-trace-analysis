from __future__ import annotations

import random

from .models import Session, UserData


def _collect_all_sessions(users: list[UserData]) -> list[tuple[str, Session]]:
    all_sessions: list[tuple[str, Session]] = []
    for user in users:
        for session in user.sessions:
            all_sessions.append((user.user_name, session))
    return all_sessions


def sample_sessions(
    users: list[UserData],
    batch_size: int,
    seed: int | None = None,
) -> list[tuple[str, Session]]:
    all_sessions = _collect_all_sessions(users)

    if len(all_sessions) <= batch_size:
        return all_sessions

    rng = random.Random(seed)
    return rng.sample(all_sessions, batch_size)


def sample_multiple_groups(
    users: list[UserData],
    batch_size: int,
    num_groups: int,
    base_seed: int | None = None,
) -> list[list[tuple[str, Session]]]:
    """Return *num_groups* independent random samples, each of *batch_size*."""
    all_sessions = _collect_all_sessions(users)

    if len(all_sessions) <= batch_size:
        return [all_sessions] * num_groups

    groups: list[list[tuple[str, Session]]] = []
    for i in range(num_groups):
        seed = (base_seed + i * 10000) if base_seed is not None else None
        rng = random.Random(seed)
        groups.append(rng.sample(all_sessions, batch_size))
    return groups


def flatten_all_sessions(
    users: list[UserData],
) -> list[tuple[str, Session]]:
    all_sessions: list[tuple[str, Session]] = []
    for user in users:
        for session in user.sessions:
            all_sessions.append((user.user_name, session))
    return all_sessions

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from .classifier import classify_sessions
from .config import Config
from .loader import load_user_files
from .llm_client import LLMClient
from .models import AnalysisResult, Scenario, Session
from .sampler import flatten_all_sessions, sample_multiple_groups
from .scenario import identify_scenarios

logger = logging.getLogger(__name__)


def _save_sampled_sessions(
    sampled_groups: list[list[tuple[str, Session]]],
    scenarios: list[Scenario],
    output_dir: Path,
) -> None:
    output_path = output_dir / "sampled_sessions.json"
    data = {
        "scenarios": [s.model_dump() for s in scenarios],
        "groups": [
            {
                "group_index": i + 1,
                "sessions": [
                    {
                        "user_name": user_name,
                        "session_uuid": session.session_uuid,
                        "mcp_tools": session.mcp_tools,
                        "skills": session.skills,
                        "rounds": [
                            {"user_msg": r.user_msg, "assistant_reply": r.assistant_reply}
                            for r in session.session_abstract
                        ],
                    }
                    for user_name, session in group
                ],
            }
            for i, group in enumerate(sampled_groups)
        ],
    }
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    total_sessions = sum(len(g) for g in sampled_groups)
    logger.info("Saved %d groups (%d total sessions) and %d scenarios to %s",
                len(sampled_groups), total_sessions, len(scenarios), output_path)


def run(config: Config) -> AnalysisResult:
    users = load_user_files(config.paths.data_dir)
    if not users:
        raise RuntimeError(f"No valid user data files found in {config.paths.data_dir}")

    sampled_groups = sample_multiple_groups(
        users, config.pipeline.batch_size, config.pipeline.num_sample_groups,
        config.pipeline.random_seed,
    )
    total_sampled = sum(len(g) for g in sampled_groups)
    logger.info("Sampled %d groups × %d sessions = %d total for scenario discovery",
                len(sampled_groups), config.pipeline.batch_size, total_sampled)

    client = LLMClient(config.llm)

    scenarios = identify_scenarios(client, sampled_groups, config.pipeline)

    _save_sampled_sessions(sampled_groups, scenarios, config.paths.output_dir)

    all_sessions = flatten_all_sessions(users)
    classified = classify_sessions(
        client, all_sessions, scenarios, config.pipeline.classification_batch_size,
    )

    scenario_counter = Counter(cs.scenario_name for cs in classified)
    summary = {
        "total_users": len(users),
        "total_sessions": len(all_sessions),
        "scenarios_count": len(scenarios),
        "classified_count": len(classified),
        "per_scenario_distribution": dict(scenario_counter),
        "scenarios": [s.model_dump() for s in scenarios],
    }

    return AnalysisResult(
        scenarios=scenarios,
        classified_sessions=classified,
        summary=summary,
    )

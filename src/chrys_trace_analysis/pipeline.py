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
from .sampler import flatten_all_sessions, sample_sessions
from .scenario import identify_scenarios

logger = logging.getLogger(__name__)


def _save_sampled_sessions(
    sampled: list[tuple[str, Session]],
    scenarios: list[Scenario],
    output_dir: Path,
) -> None:
    output_path = output_dir / "sampled_sessions.json"
    data = {
        "scenarios": [s.model_dump() for s in scenarios],
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
            for user_name, session in sampled
        ],
    }
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved %d sampled sessions and %d scenarios to %s",
                len(sampled), len(scenarios), output_path)


def run(config: Config) -> AnalysisResult:
    users = load_user_files(config.paths.data_dir)
    if not users:
        raise RuntimeError(f"No valid user data files found in {config.paths.data_dir}")

    sampled = sample_sessions(
        users, config.pipeline.batch_size, config.pipeline.random_seed,
    )
    logger.info("Sampled %d sessions for scenario discovery", len(sampled))

    client = LLMClient(config.llm)

    scenarios = identify_scenarios(client, sampled, config.pipeline)

    _save_sampled_sessions(sampled, scenarios, config.paths.output_dir)

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

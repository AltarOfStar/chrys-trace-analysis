from __future__ import annotations

import logging
from collections import Counter

from .classifier import classify_sessions
from .config import Config
from .loader import load_user_files
from .llm_client import LLMClient
from .models import AnalysisResult
from .sampler import flatten_all_sessions, sample_sessions
from .scenario import identify_scenarios

logger = logging.getLogger(__name__)


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

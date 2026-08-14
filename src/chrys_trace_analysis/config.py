from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel


class LLMConfig(BaseModel):
    api_endpoint: str
    api_key: str
    model_name: str
    temperature: float = 0.3
    max_tokens: int = 4096
    no_proxy: bool = False


class PipelineConfig(BaseModel):
    batch_size: int = 200
    num_sample_groups: int = 5
    num_scenarios_min: int = 5
    num_scenarios_max: int = 10
    random_seed: int | None = 42
    classification_batch_size: int = 50


class DeviationAnalysisConfig(BaseModel):
    detection_batch_size: int = 30
    categorization_batch_size: int = 30
    random_seed: int | None = 42


class TraceAnalysisConfig(BaseModel):
    """逐条轨迹分析的并行度与摘要压缩参数。

    分析阶段为逐条并行：每次 LLM 调用只包含一条轨迹，因此可以给每条轨迹
    保留更多内容（放宽截断上限）而无需担心 batch 拼接导致的 token 超限。

    - max_workers: 并发调用 LLM 的线程数；
    - max_turns: 每条轨迹摘要最多保留的轮次数（超出部分截断并在提示词中注明）；
    - max_user_chars / max_assistant_chars: 每轮用户消息 / 助手答复的截断上限；
    - max_tool_args_chars: 每个工具调用参数预览的截断上限；
    - max_tool_calls_shown: 每轮摘要最多展示的工具调用条数。
    """
    max_workers: int = 8
    max_turns: int = 60
    max_user_chars: int = 600
    max_assistant_chars: int = 1000
    max_tool_args_chars: int = 200
    max_tool_calls_shown: int = 12


class PathsConfig(BaseModel):
    output_dir: Path
    sessions_dir: Path


class Config(BaseModel):
    llm: LLMConfig
    pipeline: PipelineConfig
    deviation_analysis: DeviationAnalysisConfig = DeviationAnalysisConfig()
    trace_analysis: TraceAnalysisConfig = TraceAnalysisConfig()
    paths: PathsConfig


_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _resolve_env_vars(value: str) -> str:
    def replacer(m: re.Match) -> str:
        return os.environ.get(m.group(1), "")
    return _ENV_VAR_RE.sub(replacer, value)


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    raw["llm"]["api_key"] = _resolve_env_vars(raw["llm"]["api_key"])

    config_dir = Path(path).resolve().parent
    raw["paths"]["output_dir"] = (config_dir / raw["paths"]["output_dir"]).resolve()
    raw["paths"]["sessions_dir"] = (config_dir / raw["paths"]["sessions_dir"]).resolve()

    return Config.model_validate(raw)

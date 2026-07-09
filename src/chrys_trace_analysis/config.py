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
    num_scenarios_min: int = 5
    num_scenarios_max: int = 10
    random_seed: int | None = 42
    classification_batch_size: int = 50


class PathsConfig(BaseModel):
    data_dir: Path
    output_dir: Path


class Config(BaseModel):
    llm: LLMConfig
    pipeline: PipelineConfig
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
    raw["paths"]["data_dir"] = (config_dir / raw["paths"]["data_dir"]).resolve()
    raw["paths"]["output_dir"] = (config_dir / raw["paths"]["output_dir"]).resolve()

    return Config.model_validate(raw)

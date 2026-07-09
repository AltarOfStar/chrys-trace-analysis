from __future__ import annotations

import json
import logging
import sys

from chrys_trace_analysis.config import load_config
from chrys_trace_analysis.pipeline import run


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    config = load_config("config.yaml")
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)

    result = run(config)

    output_path = config.paths.output_dir / "analysis_result.json"
    output_path.write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Done. Output written to {output_path}")
    print(f"Summary: {json.dumps(result.summary, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()

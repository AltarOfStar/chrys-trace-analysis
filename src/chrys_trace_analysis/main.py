from __future__ import annotations

import argparse
import json
import logging
import sys

from chrys_trace_analysis.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Chrys Trace Analysis")
    parser.add_argument(
        "--pipeline",
        choices=["scenario", "offset"],
        default="scenario",
        help="Which pipeline to run (default: scenario)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Resume offset pipeline from a specific step "
             "(1=detection, 2=categorization, 3=turn analysis). Default: 1.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    config = load_config(args.config)
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)

    if args.pipeline == "offset":
        from chrys_trace_analysis.offset_analysis import run_offset

        result = run_offset(config, start_step=args.start_step)
        output_path = config.paths.output_dir / "offset_analysis" / "offset_analysis_result.json"
    else:
        from chrys_trace_analysis.pipeline import run

        result = run(config)
        output_path = config.paths.output_dir / "analysis_result.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Done. Output written to {output_path}")
    print(f"Summary: {json.dumps(result.summary, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()

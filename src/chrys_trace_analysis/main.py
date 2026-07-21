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
        choices=["deviation_analysis"],
        default="deviation_analysis",
        help="Which pipeline to run (default: deviation_analysis)",
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
        help="Resume deviation analysis from a specific step "
             "(1=detection, 2=categorization, 3=turn analysis). Default: 1.",
    )
    parser.add_argument(
        "--mongo",
        action="store_true",
        help="Load and simplify traces from MongoDB before running the pipeline",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    config = load_config(args.config)
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mongo:
        if config.mongo is None:
            raise ValueError(
                "MongoDB config required when --mongo is specified. "
                "Add a 'mongo' section to your config file."
            )
        from chrys_trace_analysis.mongo_loader import load_and_simplify_from_mongo

        count = load_and_simplify_from_mongo(config.mongo, config.paths.traces_dir)
        logger = logging.getLogger(__name__)
        logger.info("Saved %d simplified trace files to %s",
                    count, config.paths.traces_dir / "simplified")

    from chrys_trace_analysis.offset_analysis import run_deviation_analysis

    result = run_deviation_analysis(config, start_step=args.start_step)
    output_path = config.paths.output_dir / "deviation_analysis" / "deviation_analysis_result.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Done. Output written to {output_path}")
    print(f"Summary: {json.dumps(result.summary, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()

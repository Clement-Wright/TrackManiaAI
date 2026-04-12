from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def log(message: str) -> None:
    print(f"[report-training] {message}", flush=True)


def main() -> int:
    from tm20ai.train.reporting import (
        DEFAULT_PROGRESS_THRESHOLDS,
        write_comparison_report,
        write_training_report,
    )

    parser = argparse.ArgumentParser(
        description="Generate a markdown/JSON report for one training run or compare several runs."
    )
    parser.add_argument("run_dirs", nargs="+", help="One or more training run directories.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory for comparison reports.")
    parser.add_argument(
        "--extra-video",
        action="append",
        default=[],
        help="Optional extra rollout video paths to include in a single-run report.",
    )
    parser.add_argument(
        "--progress-threshold",
        action="append",
        type=float,
        default=None,
        help="Progress threshold for comparison reports. Can be provided multiple times.",
    )
    args = parser.parse_args()

    run_dirs = [Path(run_dir).resolve() for run_dir in args.run_dirs]
    if len(run_dirs) == 1:
        report_paths = write_training_report(run_dirs[0], extra_video_paths=args.extra_video)
        log(f"report_json={report_paths.json_path}")
        log(f"report_markdown={report_paths.markdown_path}")
        return 0

    if args.extra_video:
        parser.error("--extra-video is only supported for single-run reports.")
    thresholds = DEFAULT_PROGRESS_THRESHOLDS if args.progress_threshold is None else tuple(args.progress_threshold)
    report_paths = write_comparison_report(
        run_dirs,
        output_dir=args.output_dir,
        progress_thresholds=thresholds,
    )
    log(f"comparison_json={report_paths.json_path}")
    log(f"comparison_markdown={report_paths.markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

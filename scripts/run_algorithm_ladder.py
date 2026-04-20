from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


ALGORITHM_SPECS = {
    "redq": {
        "config": ROOT / "configs" / "full_redq.yaml",
        "train_script": ROOT / "scripts" / "train_full_redq.py",
    },
    "droq": {
        "config": ROOT / "configs" / "full_droq.yaml",
        "train_script": ROOT / "scripts" / "train_full_droq.py",
    },
    "crossq": {
        "config": ROOT / "configs" / "full_crossq.yaml",
        "train_script": ROOT / "scripts" / "train_full_crossq.py",
    },
}


def log(message: str) -> None:
    print(f"[run-algorithm-ladder] {message}", flush=True)


def _run_command(command: list[str], *, dry_run: bool = False) -> None:
    log("command=" + " ".join(command))
    if dry_run:
        return
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def _write_override_config(*, source_path: Path, target_path: Path, artifact_root: Path) -> None:
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    artifacts = dict(payload.get("artifacts") or {})
    artifacts["root"] = artifact_root.as_posix()
    payload["artifacts"] = artifacts
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _latest_final_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted(run_dir.glob("checkpoints/checkpoint_*_final.pt"))
    if not checkpoints:
        raise RuntimeError(f"No final checkpoint found in {run_dir}")
    return checkpoints[-1]


def main() -> int:
    from tm20ai.data.parquet_writer import ensure_directory, timestamp_tag
    from tm20ai.train.artifact_retention import cleanup_artifact_root
    from tm20ai.train.research import append_results_entry, write_algorithm_comparison_report

    parser = argparse.ArgumentParser(description="Run a fixed-wall-clock REDQ/DroQ/CrossQ comparison ladder.")
    parser.add_argument("--algorithms", default="redq,droq,crossq")
    parser.add_argument("--wall-clock-minutes", type=float, default=90.0)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--results-file", default="algorithm_ladder.md")
    parser.add_argument("--artifact-root", default=str(ROOT / ".tmp" / "artifacts" / "ladder"))
    parser.add_argument("--comparison-output-dir", default=None)
    parser.add_argument("--session-name", default="algorithm_ladder")
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backfill-checkpoint", default=None)
    parser.add_argument("--backfill-config", default=str(ROOT / "configs" / "full_redq.yaml"))
    args = parser.parse_args()

    algorithms = [item.strip().lower() for item in args.algorithms.split(",") if item.strip()]
    invalid = sorted(set(algorithms) - set(ALGORITHM_SPECS))
    if invalid:
        parser.error(f"Unsupported algorithms: {invalid!r}")

    artifact_root = ensure_directory(Path(args.artifact_root).resolve())
    session_tag = f"{args.session_name}_{timestamp_tag()}"
    config_override_dir = ensure_directory(ROOT / ".tmp" / "ladder_configs" / session_tag)
    results_root = ensure_directory(ROOT / "results")
    run_dirs: list[Path] = []

    if args.backfill_checkpoint is not None:
        backfill_run_name = f"{session_tag}_backfill_redq_40000"
        _run_command(
            [
                sys.executable,
                str(ROOT / "scripts" / "force_window_size.py"),
                "--config",
                str(Path(args.backfill_config).resolve()),
            ],
            dry_run=args.dry_run,
        )
        _run_command(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate.py"),
                "--config",
                str(Path(args.backfill_config).resolve()),
                "--policy",
                "checkpoint",
                "--checkpoint",
                str(Path(args.backfill_checkpoint).resolve()),
                "--episodes",
                str(args.eval_episodes),
                "--modes",
                "deterministic,stochastic",
                "--run-name",
                backfill_run_name,
            ],
            dry_run=args.dry_run,
        )

    for algorithm in algorithms:
        spec = ALGORITHM_SPECS[algorithm]
        run_name = f"{session_tag}_{algorithm}"
        override_config = config_override_dir / f"{algorithm}.yaml"
        _write_override_config(
            source_path=Path(spec["config"]).resolve(),
            target_path=override_config,
            artifact_root=artifact_root,
        )
        _run_command(
            [
                sys.executable,
                str(ROOT / "scripts" / "force_window_size.py"),
                "--config",
                str(override_config),
            ],
            dry_run=args.dry_run,
        )
        _run_command(
            [
                sys.executable,
                str(ROOT / "scripts" / "check_environment.py"),
                "--config",
                str(override_config),
                "--require-reward",
            ],
            dry_run=args.dry_run,
        )
        _run_command(
            [
                sys.executable,
                str(spec["train_script"]),
                "--config",
                str(override_config),
                "--run-name",
                run_name,
                "--max-wall-clock-minutes",
                str(args.wall_clock_minutes),
                "--eval-episodes",
                str(args.eval_episodes),
            ],
            dry_run=args.dry_run,
        )
        run_dir = artifact_root / "train" / run_name
        run_dirs.append(run_dir)
        checkpoint_path = _latest_final_checkpoint(run_dir) if not args.dry_run else run_dir / "checkpoints" / "checkpoint_fake_final.pt"
        final_summary = {}
        if not args.dry_run:
            final_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        checkpoint_env_step = int(final_summary.get("env_step", 0)) if final_summary else 0
        post_run_name = f"{run_name}_final_exact_step_{checkpoint_env_step:08d}"
        _run_command(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate.py"),
                "--config",
                str(override_config),
                "--policy",
                "checkpoint",
                "--checkpoint",
                str(checkpoint_path),
                "--episodes",
                str(args.eval_episodes),
                "--modes",
                "deterministic,stochastic",
                "--run-name",
                post_run_name,
            ],
            dry_run=args.dry_run,
        )

    if args.dry_run:
        log("dry_run_complete")
        return 0

    comparison_output_dir = (
        Path(args.comparison_output_dir).resolve()
        if args.comparison_output_dir is not None
        else ensure_directory(results_root / "comparisons" / session_tag)
    )
    report_paths = write_algorithm_comparison_report(
        run_dirs,
        output_dir=comparison_output_dir,
        wall_clock_budget_minutes=args.wall_clock_minutes,
    )
    report = json.loads(report_paths.json_path.read_text(encoding="utf-8"))
    append_results_entry(
        results_root=results_root,
        filename=args.results_file,
        title="Algorithm Ladder",
        summary_lines=[
            f"session={session_tag}",
            f"wall_clock_budget_minutes={args.wall_clock_minutes}",
            f"winner={dict(report.get('winner') or {}).get('algorithm')}",
            f"winner_best_deterministic_progress={dict(report.get('winner') or {}).get('best_deterministic_mean_final_progress_index')}",
        ],
        artifact_links={
            "comparison_json": report_paths.json_path,
            "comparison_markdown": report_paths.markdown_path,
        },
    )
    if not args.keep_artifacts:
        cleanup_result = cleanup_artifact_root(artifact_root, keep_run_dirs=run_dirs, dry_run=False)
        log(
            json.dumps(
                {
                    "kept_paths": cleanup_result.kept_paths,
                    "removed_paths": cleanup_result.removed_paths,
                },
                indent=2,
            )
        )
    log(f"comparison_json={report_paths.json_path}")
    log(f"comparison_markdown={report_paths.markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

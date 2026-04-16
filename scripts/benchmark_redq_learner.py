from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tm20ai.config import load_tm20ai_config
from tm20ai.data.parquet_writer import ensure_directory, resolve_artifact_root, timestamp_tag
from tm20ai.train.diagnostics import benchmark_redq_sweep, default_benchmark_observation_shape


def log(message: str) -> None:
    print(f"[benchmark-redq-learner] {message}", flush=True)


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_bool_list(value: str) -> list[bool]:
    result: list[bool] = []
    for item in value.split(","):
        normalized = item.strip().lower()
        if not normalized:
            continue
        if normalized in {"1", "true", "yes", "on"}:
            result.append(True)
        elif normalized in {"0", "false", "no", "off"}:
            result.append(False)
        else:
            raise ValueError(f"Unsupported boolean value: {item!r}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark REDQ learner-side critic ensembles and shared encoders.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "full_redq_diagnostic.yaml"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument("--measured-updates", type=int, default=6)
    parser.add_argument("--n-critics", default="2,4,10")
    parser.add_argument("--m-subset", default="2")
    parser.add_argument("--share-encoders", default="true,false")
    parser.add_argument("--artifact-dir", default=None)
    args = parser.parse_args()

    config = load_tm20ai_config(args.config)
    default_artifact_dir = resolve_artifact_root(config) / "benchmarks"
    artifact_dir = ensure_directory(Path(args.artifact_dir).resolve()) if args.artifact_dir else ensure_directory(default_artifact_dir)

    n_critics_values = _parse_int_list(args.n_critics)
    m_subset_values = _parse_int_list(args.m_subset)
    share_encoder_values = _parse_bool_list(args.share_encoders)
    sweep = [
        {
            "n_critics": n_critics,
            "m_subset": m_subset,
            "share_encoders": share_encoders,
            "q_updates_per_policy_update": 20,
        }
        for n_critics in n_critics_values
        for m_subset in m_subset_values
        if m_subset <= n_critics
        for share_encoders in share_encoder_values
    ]

    result = benchmark_redq_sweep(
        device=args.device,
        observation_shape=default_benchmark_observation_shape(),
        batch_size=args.batch_size,
        warmup_updates=args.warmup_updates,
        measured_updates=args.measured_updates,
        sweep=sweep,
    )

    artifact_path = artifact_dir / f"redq_learner_benchmark_{timestamp_tag()}.json"
    artifact_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for row in result["results"]:
        log(
            f"n_critics={row['n_critics']} m_subset={row['m_subset']} share_encoders={row['share_encoders']} "
            f"critic_updates_per_second={row['critic_updates_per_second']:.3f} "
            f"critic_update_ms_p50={dict(row['critic_update_ms']).get('p50')}"
        )
    log(f"artifact={artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

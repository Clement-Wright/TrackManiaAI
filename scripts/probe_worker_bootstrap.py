from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tm20ai.config import load_tm20ai_config
from tm20ai.train.worker import worker_bootstrap_probe_entry


def _default_log_path(config_path: Path) -> Path:
    config = load_tm20ai_config(config_path)
    artifacts_root = Path(config.artifacts.root)
    if not artifacts_root.is_absolute():
        artifacts_root = config_path.resolve().parents[1] / artifacts_root
    artifacts_root.mkdir(parents=True, exist_ok=True)
    return artifacts_root / "worker_bootstrap_probe.log"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Trackmania worker bootstrap path up to capture init.")
    parser.add_argument("--config", default="configs/lidar_sac.yaml")
    parser.add_argument("--log-path", default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    log_path = Path(args.log_path).resolve() if args.log_path is not None else _default_log_path(config_path)

    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(
        target=worker_bootstrap_probe_entry,
        args=(str(config_path), str(log_path)),
        name="tm20ai-worker-bootstrap-probe",
    )
    process.start()
    process.join()
    print(f"bootstrap_log={log_path}")
    print(f"exitcode={process.exitcode}")
    return int(process.exitcode or 0)


if __name__ == "__main__":
    raise SystemExit(main())

# TM20 AI

Windows-only Trackmania 2020 research repo for live reinforcement learning, behavior cloning, reward recording, and bridge diagnostics.

The current full-observation baseline is REDQ-SAC with:

- `4` critics
- `m_subset=2`
- shared critic encoders
- a `256x128` Trackmania window resized down to `64x64` grayscale stacks
- scheduled dual-mode evals (`deterministic` and `stochastic`) during REDQ training

SAC and LIDAR training paths are still available, but the main full-observation workflow in this repo now centers on REDQ.

## Recommended Path

1. Follow [docs/setup.md](docs/setup.md) for Windows, Openplanet, plugin, bridge, and window setup.
2. Record a reward trajectory for the target map with `scripts/record_reward.py`.
3. Optionally record demos with `scripts/record_demos.py` and pretrain a BC actor with `scripts/pretrain_bc.py`.
4. Train the full REDQ baseline with `scripts/train_full_redq.py`.
5. Run `scripts/run_redq_diagnostics.py` and `scripts/benchmark_redq_learner.py` when you want walltime and learner-side bottleneck data.
6. Evaluate checkpoints with `scripts/evaluate.py` and generate reports with `scripts/report_training.py`.

## One-Time Setup

Bootstrap the Python environment and the TMRL helper plugin:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_phase1.ps1
```

Then complete the Openplanet and Trackmania steps in [docs/setup.md](docs/setup.md), and verify the bridge:

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py
.\.venv\Scripts\python.exe scripts\check_bridge.py --duration 10 --reset-count 3
```

Use the optional smoke harness only as a prerequisite check, not as the main training path:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_phase1_smoke.ps1 -Mode Both
```

## Core Workflows

### 1. Record reward trajectory

Load the target map, put the car on the start line, make sure the bridge is healthy, and record one clean manual lap:

```powershell
.\.venv\Scripts\python.exe scripts\force_window_size.py --config configs\full_redq.yaml
.\.venv\Scripts\python.exe scripts\record_reward.py --config configs\base.yaml
```

This writes the raw lap and runtime trajectory under `data/reward/<map_uid>/`.

### 2. Record demos

Record full-observation demos with the current REDQ-sized window:

```powershell
.\.venv\Scripts\python.exe scripts\record_demos.py --config configs\full_redq.yaml --episodes 20
```

You can also use a checkpoint or scripted policy instead of human control:

```powershell
.\.venv\Scripts\python.exe scripts\record_demos.py --config configs\full_redq.yaml --policy checkpoint --checkpoint <checkpoint.pt>
```

### 3. Pretrain a BC actor

```powershell
.\.venv\Scripts\python.exe scripts\pretrain_bc.py --config configs\full_bc.yaml --demos-root <demo_run_dir>
```

This produces BC artifacts under `artifacts/bc/<run-name>/`.

### 4. Train the full REDQ baseline

```powershell
.\.venv\Scripts\python.exe scripts\train_full_redq.py --config configs\full_redq.yaml --run-name <run_name>
```

Warm-start from BC if desired:

```powershell
.\.venv\Scripts\python.exe scripts\train_full_redq.py `
  --config configs\full_redq.yaml `
  --run-name <run_name> `
  --init-actor <artifacts\\bc\\...\\actor_checkpoint_best.pt> `
  --init-mode actor_plus_critic_encoders
```

The shipped full REDQ baseline keeps reward shaping, the 2D throttle/steer action path, actor-sync behavior, and dual-mode eval protocol fixed while reducing critic compute with `4` shared-encoder critics.

### 5. Run REDQ diagnostics and benchmarks

Short live diagnostic run:

```powershell
.\.venv\Scripts\python.exe scripts\run_redq_diagnostics.py --config configs\full_redq_diagnostic.yaml --run-name <diag_name>
```

Learner-side synthetic benchmark:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_redq_learner.py --config configs\full_redq_diagnostic.yaml
```

The default learner benchmark sweep starts with the `4 critics / m_subset=2 / shared_encoders=true` baseline and keeps a direct `10 critics / shared_encoders=true` comparison row.

### 6. Evaluate checkpoints and generate reports

Ad hoc checkpoint evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --config configs\full_redq.yaml --policy checkpoint --checkpoint <checkpoint.pt>
```

Generate or regenerate a training report:

```powershell
.\.venv\Scripts\python.exe scripts\report_training.py artifacts\train\<run_name>
```

## Config Matrix

| Config | Purpose |
| --- | --- |
| `configs/base.yaml` | Shared bridge, runtime, capture, and reward defaults; useful for bridge checks and reward recording. |
| `configs/full_bc.yaml` | Full-observation BC pretraining config. |
| `configs/full_redq.yaml` | Main full-observation REDQ training baseline. |
| `configs/full_redq_diagnostic.yaml` | Shorter REDQ diagnostic config with richer profiling and a `.tmp/artifacts` root. |
| `configs/full_sac.yaml` | Full-observation SAC baseline retained for comparison. |
| `configs/lidar_sac.yaml` | LIDAR SAC baseline. |

## Script Matrix

| Script | Purpose |
| --- | --- |
| `scripts/bootstrap_phase1.ps1` | Create `.venv`, install Python dependencies, and install the TMRL helper plugin. |
| `scripts/check_environment.py` | Verify the bridge, plugin layout, Trackmania window size, and optional reward artifact availability. |
| `scripts/check_bridge.py` | Run telemetry soak and reset validation against the custom Openplanet bridge. |
| `scripts/force_window_size.py` | Resize the Trackmania client rect to the dimensions expected by the selected config. |
| `scripts/record_reward.py` | Record a manual lap and build the runtime reward trajectory for the current map. |
| `scripts/record_demos.py` | Record full-observation demo episodes from a human, fixed action, scripted policy, or checkpoint. |
| `scripts/pretrain_bc.py` | Train a BC actor checkpoint from recorded demos. |
| `scripts/train_full_redq.py` | Launch the main full-observation REDQ learner + worker training loop. |
| `scripts/run_redq_diagnostics.py` | Run a shorter REDQ pass with expanded runtime, queue, actor-sync, and resource profiling. |
| `scripts/benchmark_redq_learner.py` | Benchmark critic ensemble sizes and shared-encoder settings without launching Trackmania. |
| `scripts/evaluate.py` | Run ad hoc evaluation episodes against the live env or a checkpoint policy. |
| `scripts/report_training.py` | Generate markdown/JSON reports for one run or compare multiple runs. |
| `scripts/train_full_sac.py` | Full-observation SAC baseline entrypoint. |
| `scripts/train_lidar_sac.py` | LIDAR SAC baseline entrypoint. |

## Artifact Layout

Artifact locations are rooted at `artifacts.root` from the selected config.

- Standard training configs write under `artifacts/`.
- `configs/full_redq_diagnostic.yaml` writes under `.tmp/artifacts/` to keep diagnostic scratch output out of the main artifact tree.

Typical outputs:

- `artifacts/train/<run-name>/summary.json`
- `artifacts/train/<run-name>/report.json`
- `artifacts/train/<run-name>/report.md`
- `artifacts/train/<run-name>/checkpoints/`
- `artifacts/eval/<eval-run-name>/summary.json`
- `artifacts/benchmarks/redq_learner_benchmark_<timestamp>.json`
- `artifacts/bc/<run-name>/summary.json`

During REDQ training, scheduled evals are emitted as separate eval run directories, one per mode, while the training run keeps the latest deterministic eval summary as its compatibility headline.

## Operational Notes

- Full-observation runs expect a `256x128` Trackmania client rect.
- LIDAR runs expect a `958x488` Trackmania client rect.
- `scripts/evaluate.py` and `scripts/record_demos.py` now default to the REDQ full-observation config and the current 2D `throttle,steer` action format. Legacy `gas,brake,steer` fixed-action input is still accepted.
- Reward recording, demos, training, and evaluation all assume the bridge is live and the intended map is already loaded in Trackmania.
- If you need a deeper setup or troubleshooting checklist, use [docs/setup.md](docs/setup.md).

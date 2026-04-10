# TM20 AI

Windows-only Trackmania 2020 research repo with two active layers:

- a known-good Phase 1 `Trackmania 2020 + Openplanet + TMRL` smoke harness
- a repo-owned Phase 2 Openplanet bridge over localhost TCP for the later custom runtime

The rule stays the same: environment stability matters more than the first neural net.

## What this repo does

- creates a local Python 3.11.9 `.venv` with the Phase 1 dependencies
- installs and verifies `tmrl`, `rtgym`, `vgamepad`, `pywin32`, and CUDA-enabled PyTorch
- applies repo-owned TMRL config templates for `TM20LIDAR` and `TM20FULL`
- runs a plain smoke test that launches or reuses Trackmania, focuses the game window, resets the env, steps a fixed analog action, and resets again
- defines the Phase 2 bridge contract:
  - telemetry on `127.0.0.1:9100`
  - commands on `127.0.0.1:9101`
  - newline-delimited JSON
  - typed Python parsing, reconnect logic, soak checks, and reset checks

## What this repo does not do

- no policy/model code
- no trainer/server/worker orchestration
- no custom rewards
- no distributed setup
- no RL experiments

## Quick start

1. Read [docs/setup.md](/C:/Users/clewr/TrackManiaAI/docs/setup.md) and complete the one-time manual Trackmania/Openplanet prep.
2. Run the Phase 1 bootstrap:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_phase1.ps1
```

3. Run the Phase 1 smoke test flow:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_phase1_smoke.ps1 -Mode Both
```

4. Copy the custom Openplanet plugin source from [openplanet/TM20AIBridge](/C:/Users/clewr/TrackManiaAI/openplanet/TM20AIBridge) to `%USERPROFILE%\OpenplanetNext\Plugins\TM20AIBridge` so `info.toml` is at the plugin root, relaunch Trackmania, then check `Openplanet.log`.
5. If Openplanet reports that the plugin is not suitable for the current signature mode, open the Openplanet overlay with `F3` and use `Developer > Reload plugin` against `%USERPROFILE%\OpenplanetNext\Plugins\TM20AIBridge`.
6. Once the bridge is loaded on `127.0.0.1:9100/9101`, run:

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py
.\.venv\Scripts\python.exe scripts\check_bridge.py --duration 10 --reset-count 3
```

## Layout

- [configs/base.yaml](/C:/Users/clewr/TrackManiaAI/configs/base.yaml)
- [config/tmrl.lidar.json](/C:/Users/clewr/TrackManiaAI/config/tmrl.lidar.json)
- [config/tmrl.full.json](/C:/Users/clewr/TrackManiaAI/config/tmrl.full.json)
- [docs/setup.md](/C:/Users/clewr/TrackManiaAI/docs/setup.md)
- [openplanet/TM20AIBridge](/C:/Users/clewr/TrackManiaAI/openplanet/TM20AIBridge)
- [src/tm20ai/bridge](/C:/Users/clewr/TrackManiaAI/src/tm20ai/bridge)
- [scripts/bootstrap_phase1.ps1](/C:/Users/clewr/TrackManiaAI/scripts/bootstrap_phase1.ps1)
- [scripts/apply_tmrl_config.py](/C:/Users/clewr/TrackManiaAI/scripts/apply_tmrl_config.py)
- [scripts/check_bridge.py](/C:/Users/clewr/TrackManiaAI/scripts/check_bridge.py)
- [scripts/check_environment.py](/C:/Users/clewr/TrackManiaAI/scripts/check_environment.py)
- [scripts/smoke_test_env.py](/C:/Users/clewr/TrackManiaAI/scripts/smoke_test_env.py)
- [scripts/run_phase1_smoke.ps1](/C:/Users/clewr/TrackManiaAI/scripts/run_phase1_smoke.ps1)

## Phase 2 bridge contract

The custom bridge is intentionally narrow in this phase.

- Telemetry stream:
  - one persistent TCP client
  - newline-delimited JSON frames
  - exact `TelemetryFrame(session_id, run_id, frame_id, timestamp_ns, map_uid, race_time_ms, cp_count, cp_target, speed_kmh, gear, rpm, pos_xyz, vel_xyz, yaw_pitch_roll, finished, terminal_reason)`
- Command channel:
  - request/response JSON
  - `health`
  - `race_state`
  - `reset_to_start`
  - `set_recording_mode`

## Stability rule

If the custom loop is not stable, stop there and fix the environment. Use:

```powershell
.venv\Scripts\python.exe -m tmrl --check-environment
```

Then correct the Openplanet plugin state, custom bridge installation, window sizing, camera, or focus path before trying any RL code.

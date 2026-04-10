# TM20 Phase 1

Windows-only bootstrap for getting a local `Trackmania 2020 + Openplanet + TMRL` environment loop stable before touching any RL model code.

## What this repo does

- creates a local Python 3.11.9 `.venv` with the Phase 1 dependencies
- installs and verifies `tmrl`, `rtgym`, `vgamepad`, `pywin32`, and CUDA-enabled PyTorch
- applies repo-owned TMRL config templates for `TM20LIDAR` and `TM20FULL`
- runs a plain smoke test that launches or reuses Trackmania, focuses the game window, resets the env, steps a fixed analog action, and resets again

## What this repo does not do

- no policy/model code
- no trainer/server/worker orchestration
- no custom rewards
- no distributed setup
- no RL experiments

## Quick start

1. Read [docs/setup.md](/C:/Users/clewr/TrackManiaAI/docs/setup.md) and complete the one-time manual Trackmania/Openplanet prep.
2. Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_phase1.ps1
```

3. Run the smoke test flow:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_phase1_smoke.ps1 -Mode Both
```

## Layout

- [config/tmrl.lidar.json](/C:/Users/clewr/TrackManiaAI/config/tmrl.lidar.json)
- [config/tmrl.full.json](/C:/Users/clewr/TrackManiaAI/config/tmrl.full.json)
- [docs/setup.md](/C:/Users/clewr/TrackManiaAI/docs/setup.md)
- [scripts/bootstrap_phase1.ps1](/C:/Users/clewr/TrackManiaAI/scripts/bootstrap_phase1.ps1)
- [scripts/apply_tmrl_config.py](/C:/Users/clewr/TrackManiaAI/scripts/apply_tmrl_config.py)
- [scripts/smoke_test_env.py](/C:/Users/clewr/TrackManiaAI/scripts/smoke_test_env.py)
- [scripts/run_phase1_smoke.ps1](/C:/Users/clewr/TrackManiaAI/scripts/run_phase1_smoke.ps1)

## Stability rule

If the custom loop is not stable, stop there and fix the environment. Use:

```powershell
.venv\Scripts\python.exe -m tmrl --check-environment
```

Then correct the Openplanet plugin state, window sizing, camera, or focus path before trying any RL code.

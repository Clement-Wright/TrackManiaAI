# Setup

This repo assumes:

- Windows
- local Trackmania 2020
- Openplanet
- one live environment on the same machine as the learner/worker process

The setup goal is not just "the game launches." The goal is a stable local loop with:

- `TM20AIBridge` loaded in Openplanet
- `TMRL_GrabData.op` installed for the older smoke harness
- the Trackmania window sized correctly for the chosen observation mode
- reward recording and live resets working reliably

## 1. Install prerequisites

1. Install Trackmania 2020.
2. Install Openplanet.
3. Launch Trackmania once so `%USERPROFILE%\OpenplanetNext` and `%USERPROFILE%\Documents\Trackmania` exist.
4. Install any required Microsoft Visual C++ runtime if Openplanet or Python dependencies complain.

## 2. Bootstrap the repo environment

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_phase1.ps1
```

This script:

- creates the local `.venv`
- installs Python dependencies
- installs CUDA-enabled PyTorch
- runs `python -m tmrl --install`
- copies `TMRL_GrabData.op` into `%USERPROFILE%\OpenplanetNext\Plugins` when needed

After bootstrap, copy the TMRL smoke-test map if you want to use the optional smoke harness:

```text
%USERPROFILE%\TmrlData\resources\tmrl-test.Map.Gbx
-> %USERPROFILE%\Documents\Trackmania\Maps\My Maps\
```

## 3. Install the custom Openplanet bridge

The repo-owned bridge source lives at `openplanet/TM20AIBridge`.

The expected developer install shape is:

```text
%USERPROFILE%\OpenplanetNext\Plugins\TM20AIBridge\
  info.toml
  Main.as
  Protocol.as
  Telemetry.as
  Reset.as
  Recorder.as
```

Copy it with:

```powershell
if (Test-Path "$env:USERPROFILE\OpenplanetNext\Plugins\TM20AIBridge") {
    Remove-Item "$env:USERPROFILE\OpenplanetNext\Plugins\TM20AIBridge" -Recurse -Force
}
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\OpenplanetNext\Plugins\TM20AIBridge" | Out-Null
Copy-Item -Path (Join-Path (Resolve-Path openplanet\TM20AIBridge) '*') `
    -Destination "$env:USERPROFILE\OpenplanetNext\Plugins\TM20AIBridge" `
    -Recurse
```

Then:

1. Launch Trackmania.
2. Open the Openplanet overlay with `F3`.
3. If needed, use `Developer > Reload plugin` on `%USERPROFILE%\OpenplanetNext\Plugins\TM20AIBridge`.
4. Check `Openplanet.log` for bridge startup and compile errors.

The custom bridge should expose:

- telemetry on `127.0.0.1:9100`
- command RPC on `127.0.0.1:9101`

## 4. Prepare Trackmania for live use

General expectations:

- Use windowed mode.
- Keep the target map already loaded before running reward recording, demos, eval, or training.
- Make sure the Trackmania window title matches the configured capture target, which is `"Trackmania"` in the shipped configs.

Full-observation mode:

- expected client rect: `256x128`
- camera: default visible-car gameplay camera
- use config: `configs/full_redq.yaml`, `configs/full_redq_diagnostic.yaml`, `configs/full_sac.yaml`, or `configs/full_bc.yaml`

LIDAR mode:

- expected client rect: `958x488`
- camera: cockpit-style view with the car hidden
- use config: `configs/lidar_sac.yaml`

Use the helper whenever the client rect drifts:

```powershell
.\.venv\Scripts\python.exe scripts\force_window_size.py --config configs\full_redq.yaml
```

## 5. Verify the environment and bridge

Basic environment gate:

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py
```

If you already recorded reward for the current map, also verify that the runtime trajectory exists:

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py --require-reward
```

Bridge diagnostics:

```powershell
.\.venv\Scripts\python.exe scripts\check_bridge.py --duration 10 --reset-count 3
```

Longer bridge soak:

```powershell
.\.venv\Scripts\python.exe scripts\check_bridge.py --duration 600 --reset-count 100
```

Optional smoke harness:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_phase1_smoke.ps1 -Mode Both
```

Treat the smoke harness as a prerequisite check, not as the main training interface.

## 6. Reward recording prerequisites

Before running `scripts/record_reward.py`:

- the bridge must already be healthy
- the car must be on the start line
- the intended map must remain loaded for the whole lap
- the Trackmania window must already match the observation mode you will use later

Typical sequence:

```powershell
.\.venv\Scripts\python.exe scripts\force_window_size.py --config configs\full_redq.yaml
.\.venv\Scripts\python.exe scripts\check_environment.py
.\.venv\Scripts\python.exe scripts\record_reward.py --config configs\base.yaml
```

Reward artifacts are stored under `data/reward/<map_uid>/`.

## 7. Training and eval expectations

For the current full-observation REDQ path:

- use `configs/full_redq.yaml`
- keep the Trackmania client rect at `256x128`
- expect scheduled dual-mode evals during training
- expect training outputs under `artifacts/train/<run-name>/`
- expect eval outputs under `artifacts/eval/<eval-run-name>/`

For the diagnostic REDQ config:

- use `configs/full_redq_diagnostic.yaml`
- the artifact root is `.tmp/artifacts/`
- benchmark and diagnostic scratch output intentionally stays out of the main `artifacts/` tree

## 8. Troubleshooting

- `OpenplanetNext` is missing:
  Launch Trackmania once after installing Openplanet.

- `TM20AIBridge` is missing or incomplete:
  Recopy `openplanet/TM20AIBridge` so `info.toml` is at `%USERPROFILE%\OpenplanetNext\Plugins\TM20AIBridge\info.toml`.

- Openplanet says `Plugin is not suitable for the current signature mode`:
  Open `F3`, then use `Developer > Reload plugin` on the installed `TM20AIBridge` folder.

- Bridge ports never come up:
  Open `Openplanet.log` and look for `TM20AIBridge` compile, permission, or socket errors.

- `check_environment.py` fails the window-size gate:
  Run `scripts/force_window_size.py` with the same config you intend to use for demos, eval, or training.

- Reward recording fails immediately:
  Make sure the car is at the start line and the bridge reports a live `map_uid`.

- Resets are flaky or training stalls after a map change:
  Reload the target map manually, confirm the bridge is healthy again with `scripts/check_bridge.py`, then restart the Python side.

- Full observation is unstable:
  Reconfirm the `256x128` client rect, visible-car camera, and that no window is covering the Trackmania client area.

- LIDAR observation is unstable:
  Reconfirm the `958x488` client rect, cockpit-style view, and hidden car view.

- Video export or ffmpeg-dependent workflows fail:
  Re-run `scripts/check_environment.py --check-ffmpeg` and put `ffmpeg` on `PATH`.

- The bridge is healthy but training output goes somewhere unexpected:
  Check `artifacts.root` in the config you launched. The diagnostic REDQ config intentionally uses `.tmp/artifacts`.

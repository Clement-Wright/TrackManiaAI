# Phase 1 Setup

This repo assumes the local machine is the worker machine and the goal is a stable single-process smoke loop, not training.

## One-time manual prep

1. Install Openplanet for Trackmania 2020.
2. If needed, install the Microsoft Visual C++ x64 runtime.
3. Launch Trackmania once after installing Openplanet so `%USERPROFILE%\OpenplanetNext` is created.
4. Run the repo bootstrap:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_phase1.ps1
```

5. Copy `%USERPROFILE%\TmrlData\resources\tmrl-test.Map.Gbx` into `%USERPROFILE%\Documents\Trackmania\Maps\My Maps`.
   The `%USERPROFILE%\Documents\Trackmania` tree is created by the game, so launch Trackmania at least once before this step.

## Manual game prep before each smoke run

1. Launch the `tmrl-test` map in Trackmania.
2. Set the game to windowed mode.
3. Hide the ghost with `g`.
4. For `TM20LIDAR`, press `3` until the car is hidden and the cockpit view is active.
5. For `TM20FULL`, press `1` so the default camera is active and the car is visible.

## Smoke commands

Run both smoke passes in order:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_phase1_smoke.ps1 -Mode Both
```

Run only one variant:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_phase1_smoke.ps1 -Mode Lidar
powershell -ExecutionPolicy Bypass -File scripts\run_phase1_smoke.ps1 -Mode Full
```

## Phase 2 bridge install

The custom bridge source lives in [openplanet/TM20AIBridge](/C:/Users/clewr/TrackManiaAI/openplanet/TM20AIBridge). The developer install shape is a plugin-root folder under `%USERPROFILE%\OpenplanetNext\Plugins` with `info.toml` and the `.as` files directly under that folder:

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
Copy-Item -Path (Join-Path (Resolve-Path openplanet\TM20AIBridge) '*') `
    -Destination "$env:USERPROFILE\OpenplanetNext\Plugins\TM20AIBridge" `
    -Recurse
```

After copying:

1. Launch Trackmania once so Openplanet scans the source plugin and updates `Openplanet.log`.
2. If `Openplanet.log` shows `Plugin is not suitable for the current signature mode`, open the Openplanet overlay with `F3` and use `Developer > Reload plugin` against `%USERPROFILE%\OpenplanetNext\Plugins\TM20AIBridge`.
3. Confirm the bridge log lines show the listeners on `127.0.0.1:9100` and `127.0.0.1:9101`.
4. Load `tmrl-test` or another road-only test map.
5. Run the bridge checks:

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py
.\.venv\Scripts\python.exe scripts\check_bridge.py --duration 10 --reset-count 3
```

For the full Phase 2 acceptance target, use:

```powershell
.\.venv\Scripts\python.exe scripts\check_bridge.py --duration 600 --reset-count 100
```

## Troubleshooting

- If `OpenplanetNext` does not exist, Openplanet is not installed correctly or Trackmania has not been launched since installation.
- If `TMRL_GrabData.op` is missing, rerun the bootstrap script. It copies the plugin from `%USERPROFILE%\TmrlData\resources\Plugins` when needed.
- If `env.reset()` fails with connection errors, open the Openplanet menu with `F3`, reload `TMRL Grab Data`, then retry.
- If the smoke test reports repeated focus failure, click the Trackmania window once, then rerun the test.
- If `TM20FULL` is unstable, double-check the default camera and the small `256x128` window size.
- If `TM20LIDAR` is unstable, double-check cockpit camera and the `958x488` window size.
- If `scripts/check_environment.py` reports that `TM20AIBridge` is missing from `%USERPROFILE%\OpenplanetNext\Plugins`, recopy the folder so `info.toml` is at `%USERPROFILE%\OpenplanetNext\Plugins\TM20AIBridge\info.toml`.
- If Openplanet rejects the source plugin at startup with `Plugin is not suitable for the current signature mode`, use `Developer > Reload plugin` from the Openplanet overlay before running the Python checks.
- If the bridge ports never come up after reload, open `Openplanet.log` and look for `TM20AIBridge` compile or socket errors.

## Acceptance bar

Phase 1 is done only when both `TM20LIDAR` and `TM20FULL`:

- complete the first reset
- step for a few seconds at the TMRL default 20 Hz cadence
- produce observations, reward, and `terminated` or `truncated`
- complete a second reset without restarting Python
- pass twice in a row, once on a fresh game launch and once by reusing the running session

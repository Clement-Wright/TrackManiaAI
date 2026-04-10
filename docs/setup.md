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

## Troubleshooting

- If `OpenplanetNext` does not exist, Openplanet is not installed correctly or Trackmania has not been launched since installation.
- If `TMRL_GrabData.op` is missing, rerun the bootstrap script. It copies the plugin from `%USERPROFILE%\TmrlData\resources\Plugins` when needed.
- If `env.reset()` fails with connection errors, open the Openplanet menu with `F3`, reload `TMRL Grab Data`, then retry.
- If the smoke test reports repeated focus failure, click the Trackmania window once, then rerun the test.
- If `TM20FULL` is unstable, double-check the default camera and the small `256x128` window size.
- If `TM20LIDAR` is unstable, double-check cockpit camera and the `958x488` window size.

## Acceptance bar

Phase 1 is done only when both `TM20LIDAR` and `TM20FULL`:

- complete the first reset
- step for a few seconds at the TMRL default 20 Hz cadence
- produce observations, reward, and `terminated` or `truncated`
- complete a second reset without restarting Python
- pass twice in a row, once on a fresh game launch and once by reusing the running session

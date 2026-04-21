# REDQ Ghost-Driven Offline-To-Online Notes

## 2026-04-20 Implementation And Test Pass

Implemented the first code-bearing version of the REDQ mainline ghost-data roadmap:

- Added checkpoint/eval diagnostics support for determinism conversion score, deterministic-vs-stochastic gaps, final checkpoint eval status, extraction-mode policy evaluation, and progress-threshold metadata.
- Added source-aware replay plus `BalancedReplayBuffer` so online REDQ can sample from separate offline and online buffers with a decaying offline fraction.
- Added ghost config blocks and artifacts for Nadeo top-100 ingestion, Openplanet-export trajectory normalization, diverse ghost bundle selection, ghost-bundle reward projection, offline transition seeding, and elite archive manifests.
- Added `train_full_redq.py --ghost-bundle` and `--offline-init-checkpoint` so ghost-pretrained REDQ weights and offline replay seeding can be used before live fine-tuning.
- Added `pretrain_ghost_redq.py` with BC, REDQ critic warm start, AWAC-style weighted actor updates, and optional CQL-style critic regularization over action-valid offline transitions.
- Added `evaluate_redq_policy_modes.py` for deterministic mean, clipped mean, stochastic temperature sweeps, and best-of-k stochastic diagnostics on one checkpoint.

Important boundary: raw `.gbx` parsing is not faked in Python. Downloaded `.gbx` files are stored reproducibly, and `extract_ghost_trajectories.py` expects Openplanet-exported JSON/JSONL/Parquet trajectory data. Actor/critic pretraining fails closed unless validated action labels and observation sidecars are present.

Validation:

- `python -m compileall -q src scripts`
- `pytest -q tests/test_replay.py tests/test_metrics.py tests/test_redq_stack.py tests/test_ghost_pipeline.py`
- `pytest -q tests/test_evaluator_and_data.py tests/test_reporting.py tests/test_worker_learner.py tests/test_redq_learner.py tests/test_research_and_cleanup.py`
- `pytest -q`

Final result: full suite passed, `107 passed`.

## 2026-04-20 Live Top-100 Run Attempt

Prepared a top-100-target REDQ run on current map UID `oqIJ5rQDRrNwLPTh9H2p_W4tLof`.

- Trackmania window resize succeeded at `256x128`.
- Bridge/environment gate passed with `race_state=start_line`.
- Existing single-trajectory reward artifact was present at `data/reward/oqIJ5rQDRrNwLPTh9H2p_W4tLof/trajectory_0p5m.npz`.
- Long top-100 REDQ training was not launched because the required Nadeo credential environment variables were not visible to this process or Windows User/Machine environment registry:
  `TM20AI_NADEO_DEDI_LOGIN`, `TM20AI_NADEO_DEDI_PASSWORD`, `TM20AI_NADEO_USER_AGENT`.

Decision: do not fall back to the single local trajectory reward for this run, because the requested experiment is specifically top-100-target training.

## 2026-04-20 Credential Retry

Retried top-100 ghost ingestion after locating the credential values in the Windows user registry hive.

- Ghost ingestion tests passed: `pytest -q tests/test_ghost_pipeline.py` -> `3 passed`.
- The Nadeo request reached `prod.trackmania.core.nadeo.online` successfully, so this was not a sandbox/network failure.
- Authentication failed with `401 Invalid credentials` for the `NadeoServices` audience using the stored `TM20AI_NADEO_DEDI_LOGIN` and `TM20AI_NADEO_DEDI_PASSWORD`.
- No `TM20AI_NADEO_CORE_TOKEN` or `TM20AI_NADEO_LIVE_TOKEN` override variables were present.

Decision: top-100-target REDQ training remains blocked until the dedicated-server credentials are corrected or valid Nadeo token overrides are provided. We should not launch the long run on the single local reward trajectory because that would not answer the top-100-target experiment.

## 2026-04-20 Second Credential Retry

Retried top-100 ghost ingestion after the user requested another attempt.

- The fetch command read credential values from `HKEY_USERS\S-1-5-21-1132336749-3538666733-1392677739-1001\Environment`.
- Visible non-secret state: `TM20AI_NADEO_DEDI_LOGIN=clewri163`, password length `14`, user-agent length `45`.
- `TM20AI_NADEO_CORE_TOKEN` and `TM20AI_NADEO_LIVE_TOKEN` were still missing.
- Nadeo again returned `401 Invalid credentials` for dedicated-server authentication.

Decision: the top-100-target run is still blocked on valid Nadeo dedicated-server credentials or token overrides. No long REDQ run was launched.

## 2026-04-20 Top-100 REDQ Run Started

Launched a live top-100 ghost-target REDQ run after the corrected dedicated-server credentials authenticated successfully.

- Current map UID: `oqIJ5rQDRrNwLPTh9H2p_W4tLof`.
- Fetched `100` Nadeo leaderboard entries and downloaded `100` `.gbx` replay files.
- Used a scratch GBX.NET exporter to decode TM2020 `CSceneVehicleVis` ghost samples into trajectory rows with position, velocity, speed, gas, brake, steer, and gear.
- Normalized all `100` ghost trajectories and built `data/ghosts/oqIJ5rQDRrNwLPTh9H2p_W4tLof/ghost_bundle_manifest.json`.
- Bundle selection: `20` representative trajectories selected from `100`; action channel validation passed; offline transition count is `0` because no observation sidecars were available for ghost BC/AWAC pretraining.
- Live gates passed: Trackmania window `256x128`, bridge healthy, environment gate passed.
- Run launched out-of-sandbox as `artifacts/train/redq_top100_20260420_120min` with `--max-wall-clock-minutes 120`, `--eval-episodes 5`, and explicit `--ghost-bundle`.
- Startup check after launch: `env_step=1077`, `learner_step=85`, `actor_step=17`, `replay_size=1077`, `episode_count=13`, movement started rate `1.0`.

Monitoring automation `monitor-top100-redq` is active at 30-minute intervals. Final results should be appended here after completion, including deterministic/stochastic eval summaries and whether the ghost-bundle reward improved line-following behavior.

## 2026-04-20 Top-100 REDQ Run Final Findings

Run `redq_top100_20260420_120min` ended early with a worker fatal error, but it wrote a final checkpoint and training report.

- Final training state: `env_step=57323`, `learner_step=101048`, `actor_step=20209`, `replay_size=57323`, `episodes=1438`.
- Final checkpoint: `artifacts/train/redq_top100_20260420_120min/checkpoints/checkpoint_00057323_final.pt`.
- Report artifacts: `artifacts/train/redq_top100_20260420_120min/report.json` and `report.md`.
- Termination: `fatal_error` from `PermissionError [WinError 5]` replacing `worker_sync/worker_actor_status.tmp` -> `worker_actor_status.json`.
- Final checkpoint eval was not produced because shutdown followed the worker fatal error. Latest verified dual-mode checkpoint eval is step `55008`.
- Best deterministic eval: step `20004` and `25004`, `mean_final_progress_index=91.0`, `completion_rate=0.0`.
- Best stochastic eval: step `20004` and `50015`, `mean_final_progress_index=90.6`, `completion_rate=0.0`.
- Deterministic behavior was healthy from `15k` through `30k` (`89.0-91.0` progress), then collapsed to `0.0` from `35k` onward while stochastic stayed near `89-91`.
- Training episodes were dominated by leaving the ghost corridor/reference: `1265` stray terminations, `173` no-progress terminations, plus `1` unknown.
- Movement itself was not dead: movement-started rate `0.9986`, stall episode rate `0.0090`.
- Diagnostics: bottleneck verdict `worker_env`, `achieved_utd_1k=1.552`, `cumulative_utd=1.763`, current actor staleness `78`, policy-control fraction `0.9993`.

Interpretation: the top-100 ghost bundle successfully created a richer reward target and produced a short useful learning window, but the reward corridor is too brittle as configured. The dominant failure is now stray termination / line-adherence instability, not dead control. The deterministic-vs-stochastic split after `35k` also reappeared: stochastic retained the learned corridor behavior while deterministic mean-action control collapsed.

Next actions:

- Fix the Windows atomic status write path in `SACWorker._write_actor_status()` to tolerate transient file locks before another long run.
- Add a final exact checkpoint dual-mode eval for `checkpoint_00057323_final.pt` when the live environment is available.
- Relax or shape `ghost_bundle_progress` so straying produces a recoverable penalty/window instead of overwhelming training with hard resets.
- Add deterministic extraction sweeps on the `20k`, `25k`, `30k`, and final checkpoints to diagnose why deterministic control collapses while stochastic remains useful.

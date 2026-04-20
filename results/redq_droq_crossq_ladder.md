# REDQ / DroQ / CrossQ Ladder Results

## 2026-04-19 - Ladder Recovery Note

- REDQ completed its 90-minute segment cleanly: `env_step=61082`, `learner_step=91248`, `actor_step=18249`, `termination_reason=max_wall_clock_minutes`.
- REDQ final exact checkpoint eval from `checkpoint_00061082_final.pt`: deterministic `mean_final_progress_index=462.6`, stochastic `290.8`.
- DroQ started and reached `env_step=10329`, `learner_step=15135`, `actor_step=3027`, with completed evals at `5000` and `10000`.
- DroQ then failed on the Chrome/window-compositor issue: Trackmania client rect reported `845x182` while the full-observation capture expected `256x128`.
- Capture was patched to fall back to a fixed-size `256x128` capture region instead of aborting when Windows refuses to resize the client rect.
- Focused validation after the patch: `24 passed` for evaluator/worker/REDQ learner tests.
- DroQ resume is currently blocked from this sandbox by Windows multiprocessing queue creation: `PermissionError: [WinError 5] Access is denied`.
- Resume command to continue the ladder once launched outside the sandbox:

```powershell
.\.venv\Scripts\python.exe scripts\train_full_droq.py --config .tmp\ladder_configs\redq_droq_crossq_chrome_guard_20260419_20260420_012603\droq.yaml --run-name redq_droq_crossq_chrome_guard_20260419_20260420_012603_droq --resume .tmp\live_algorithm_ladder_20260419\train\redq_droq_crossq_chrome_guard_20260419_20260420_012603_droq\checkpoints\checkpoint_00010329_final.pt --max-wall-clock-minutes 75 --eval-episodes 5
```

## 2026-04-19 - DroQ Resume

- DroQ was resumed from `checkpoint_00010329_final.pt` after resizing Trackmania to `256x128` and passing the reward/environment gate.
- Active resume logs:
  - `.tmp/launcher_logs/droq_resume_20260420_000000.stdout.log`
  - `.tmp/launcher_logs/droq_resume_20260420_000000.stderr.log`
- Initial resume heartbeat: `env_step=10329`, `learner_step=15856`, `actor_step=3171`, `episodes=19`.
- The monitor heartbeat now tracks the DroQ resume and will launch CrossQ after DroQ completes.

## 2026-04-20 - DroQ Completion And CrossQ Handoff

- DroQ reached the resumed wall-clock budget and wrote `checkpoint_00050015_final.pt`: `env_step=50015`, `learner_step=90584`, `actor_step=18116`, `episodes=141`, `termination_reason=max_wall_clock_minutes`.
- DroQ shutdown was not fully clean: `clean_shutdown=false`, `eval_in_flight=true`; the final scheduled stochastic eval directory for step `50007` exists but did not write `summary.json`.
- Latest clean dual-mode DroQ eval remains checkpoint-authoritative step `45000`: deterministic `mean_final_progress_index=168.0`, stochastic `673.4`.
- A later deterministic-only DroQ eval completed at step `50007`: deterministic `mean_final_progress_index=169.4`, checkpoint env/learner/actor steps `50007/87912/17582`.
- CrossQ ladder config was created at `.tmp/ladder_configs/redq_droq_crossq_chrome_guard_20260419_20260420_012603/crossq.yaml`, pointing artifacts at `.tmp/live_algorithm_ladder_20260419`.
- Trackmania was verified at the required `256x128` client rect and `scripts/check_environment.py --require-reward` passed for the CrossQ config.
- CrossQ training could not be launched from the current sandbox because Windows denied `multiprocessing.Queue` creation: `PermissionError: [WinError 5] Access is denied`.
- CrossQ handoff command to run from a normal PowerShell session:

```powershell
.\.venv\Scripts\python.exe scripts\train_full_crossq.py --config .tmp\ladder_configs\redq_droq_crossq_chrome_guard_20260419_20260420_012603\crossq.yaml --run-name redq_droq_crossq_chrome_guard_20260419_20260420_012603_crossq --max-wall-clock-minutes 90 --eval-episodes 5
```

## 2026-04-20 - Ladder Conclusion

- CrossQ completed its 90-minute segment cleanly: `env_step=65059`, `learner_step=64274`, `episodes=707`, `termination_reason=max_wall_clock_minutes`, `clean_shutdown=true`.
- CrossQ maintained stable capture at `256x128`, but all `707` training episodes terminated with `no_progress`; latest clean dual-mode checkpoint eval at step `60749` was deterministic `11.8`, stochastic `46.8`.
- A final CrossQ checkpoint was written at `checkpoint_00065059_final.pt`, but no final-exact dual-mode eval artifact was produced for that checkpoint before shutdown. The comparison below uses the highest checkpoint-backed deterministic eval available for each algorithm.
- Official scoreboard by best checkpoint-backed deterministic `mean_final_progress_index`:
  - `redq`: `496.0` at checkpoint env step `50001`; best stochastic `1144.4`; run reached `61082` env steps and `91248` learner steps.
  - `droq`: `234.4` at checkpoint env step `35015`; best stochastic `819.4`; run reached `50015` env steps and `90584` learner steps.
  - `crossq`: `11.8` at checkpoint env step `60749`; best stochastic `46.8`; run reached `65059` env steps and `64274` learner steps.
- Winner for this ladder: `redq`.
- Diagnostics:
  - REDQ and DroQ were both learner-backprop bottlenecked; CrossQ was worker-env bottlenecked but failed to learn useful progress.
  - DroQ achieved the strongest rolling/cumulative UTD diagnostics (`achieved_utd_1k=3.456`, `cumulative_utd=1.811`) but did not convert that into better deterministic eval progress.
  - Stochastic eval outperformed deterministic for all three algorithms, especially REDQ/DroQ, so deterministic-policy collapse or overconfident mean-action behavior remains worth investigating.
  - No algorithm completed the map; all results are still progress-only rather than finish-rate wins.

Artifacts:
- Comparison report: `results/comparisons/redq_droq_crossq_ladder_20260420/algorithm_comparison_report.md`
- Comparison JSON: `results/comparisons/redq_droq_crossq_ladder_20260420/algorithm_comparison_report.json`

# AGENTS.md

Operational guide for Codex and other agents working in this repository.

Keep this file practical. If guidance grows too large, move task-specific detail into `docs/` or `results/` and link it here. Update this file when the same agent mistake happens twice, when routing guidance would prevent excessive file reading, or when a workflow becomes stable enough to encode.

## Project Snapshot

This is a Windows-only Trackmania 2020 research repo for live reinforcement learning, behavior cloning, ghost/replay data, reward design, and eval/reporting.

Current mainline:

- Learner: full-observation REDQ, 4 critics, `m_subset=2`, shared encoders.
- Observation: Trackmania client rect `256x128`, downsampled to grayscale stacks.
- Action path: 2D `throttle, steer`.
- Current `tmrl-test` target: `rank11_100_bundle` at `data/ghosts/oqIJ5rQDRrNwLPTh9H2p_W4tLof/ghost_bundle_rank_011_100.json`.
- Current `tmrl-test` reward profile: Block A winner `A3_buffered_hard_boundary` with 25 m soft corridor, 90 m hard corridor, 60-step patience, 0.10 m recovery progress, 3 km/h recovery speed, and 0.25 m recovery distance delta.
- Near-term roadmap: storage hygiene, exact eval closure, target-family selection, reward stability, deterministic extraction, offline leverage, then algorithm variants.

Key research docs:

- `results/project_progress_research_review_20260428.md`
- `results/future_research_plan_20260428.md`
- `results/phase_0_1_storage_eval_closure_20260429.md`
- `results/phase_2_3_target_reward_20260429.md`
- `results/phase_4_5_extraction_offline_plan_20260501.md`

## Repository Layout

- `src/tm20ai/`: Python package.
- `src/tm20ai/config.py`: config dataclasses and validation.
- `src/tm20ai/env/`: live env, reset, reward trajectory, runtime interface.
- `src/tm20ai/ghosts/`: Nadeo ingestion, ghost bundles, offline data, ghost reward.
- `src/tm20ai/train/`: learner, worker, evaluator, campaign logic, reports, metrics, retention.
- `src/tm20ai/algos/`: SAC, REDQ, DroQ, CrossQ implementations.
- `src/tm20ai/models/`: actor/critic networks.
- `src/tm20ai/capture/`: DXcam/window/preprocess capture path.
- `openplanet/TM20AIBridge/`: custom Openplanet bridge source.
- `scripts/`: entrypoints for setup, gates, training, eval, ghost data, campaigns, cleanup.
- `configs/`: shipped YAML configs.
- `tests/`: unit/integration tests.
- `data/reward/`: recorded reward trajectories. Treat as durable.
- `data/ghosts/`: fetched/extracted ghost data and selected bundles. Treat as durable.
- `artifacts/`: live training/eval outputs. Large and disposable unless selected as keepers.
- `.tmp/`: scratch. Disposable after campaigns.
- `results/`: durable human-readable research findings. Do not delete.

## Environment And Setup

Use the project venv:

```powershell
.\.venv\Scripts\python.exe <script>
```

Bootstrap from a fresh checkout:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_phase1.ps1
```

Read `docs/setup.md` before changing Windows/Openplanet/Trackmania assumptions.

Live full-observation runs require:

- Trackmania 2020 open and visible in the same desktop session as Codex.
- Windowed mode.
- Client rect resized to `256x128`.
- Target map already loaded.
- Openplanet `TM20AIBridge` loaded and healthy.
- Reward/ghost artifacts present for the chosen map.

Live gate sequence:

```powershell
.\.venv\Scripts\python.exe scripts\force_window_size.py --config configs\full_redq_top100_tmrl_test.yaml
.\.venv\Scripts\python.exe scripts\check_environment.py --require-reward
.\.venv\Scripts\python.exe scripts\check_bridge.py --duration 10 --reset-count 3
```

If `force_window_size.py` cannot find a visible Trackmania window, stop. Do not start training.

Run live bridge gates sequentially, not in parallel. The Openplanet command RPC can time out when `check_environment.py` and `check_bridge.py` both issue command requests at the same time; this is command-port contention, not necessarily a broken bridge.

Live GUI/window checks may need to run outside the filesystem sandbox. On 2026-05-01, sandboxed window enumeration failed with `Could not find a visible window containing 'Trackmania'`, but the same `check_environment.py --require-reward` gate passed outside the sandbox after `force_window_size.py` confirmed the client rect. Treat sandboxed window-enumeration failures as an escalation/visibility issue first; verify outside the sandbox before diagnosing Trackmania, Openplanet, or the reward bridge as broken.

## Build, Test, And Lint

Fast syntax/import check:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
```

Full test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Focused checks commonly used in this repo:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_research_and_cleanup.py tests\test_campaign.py tests\test_redq_stack.py tests\test_worker_learner.py
.\.venv\Scripts\python.exe -m pytest -q tests\test_ghost_pipeline.py tests\test_reward.py tests\test_metrics.py tests\test_reporting.py
```

Optional style check when `ruff` is installed:

```powershell
.\.venv\Scripts\python.exe -m ruff check src scripts tests
```

For documentation-only changes, at minimum verify the file exists and links/commands are plausible. Do not run long live tests for docs-only edits.

## Core Workflows

Record reward trajectory:

```powershell
.\.venv\Scripts\python.exe scripts\force_window_size.py --config configs\full_redq.yaml
.\.venv\Scripts\python.exe scripts\record_reward.py --config configs\base.yaml
```

Train current REDQ baseline:

```powershell
.\.venv\Scripts\python.exe scripts\train_full_redq.py --config configs\full_redq.yaml --run-name <run_name>
```

Train top-100/rank11 target config:

```powershell
.\.venv\Scripts\python.exe scripts\train_full_redq.py --config configs\full_redq_top100_tmrl_test.yaml --run-name <run_name>
```

Continue Phase 4 extraction from a validated A3 winner:

```powershell
.\.venv\Scripts\python.exe scripts\run_rank11_100_validation_campaign.py --config configs\full_redq_top100_tmrl_test.yaml --session-name <session> --blocks B --winner-run-dir <validated_A3_run_dir> --winner-config <A3_config_yaml>
```

Run rank11 validation campaign dry run:

```powershell
.\.venv\Scripts\python.exe scripts\run_rank11_100_validation_campaign.py --config configs\full_redq_top100_tmrl_test.yaml --session-name <session> --blocks A --dry-run
```

Run REDQ diagnostics:

```powershell
.\.venv\Scripts\python.exe scripts\run_redq_diagnostics.py --config configs\full_redq_diagnostic.yaml --run-name <diag_name>
```

Evaluate a checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --config configs\full_redq.yaml --policy checkpoint --checkpoint <checkpoint.pt>
```

Generate a report:

```powershell
.\.venv\Scripts\python.exe scripts\report_training.py artifacts\train\<run_name>
```

Clean artifacts with dry-run first:

```powershell
.\.venv\Scripts\python.exe scripts\cleanup_artifacts.py --dry-run
```

## Research Rules

- REDQ is the mainline learner until another branch beats it on exact checkpoint-backed deterministic eval.
- DroQ is a later efficiency branch.
- CrossQ is experimental until eval, reward, and deterministic extraction are stable.
- For `tmrl-test`, use `rank11_100_bundle` as the default target family.
- One ghost bundle equals one route strategy. Do not silently mix intended routes with shortcut/exploit routes.
- Future ambiguous maps must hard-stop unless an official author reference or user-provided baseline runs are available.
- Keep `mean_final_progress_index` for continuity, but also report arc length, progress fraction, ghost-relative timing, corridor diagnostics, and determinism conversion when available.
- Treat stochastic-better-than-deterministic as an extraction/deployment problem, not automatically an algorithm failure.

## Eval And Ranking Rules

A run is not scientifically final unless:

- `summary.json` has `exact_final_eval_complete=true`.
- `incomplete_final_eval=false`.
- `final_eval_state=complete`.
- Exact final deterministic eval artifact exists on disk.
- Exact final stochastic eval artifact exists on disk.

Do not rank a run on the latest scheduled eval if exact final eval is missing. Mark it incomplete and repair/backfill from the final checkpoint when possible.

Suite/campaign legs must not advance until the current leg has exact final deterministic and stochastic eval artifacts.

Official ranking metric remains exact final deterministic `mean_final_progress_index`. Tie-breakers may use ghost-relative time delta, progress fraction, corridor truncation rate, and corridor nonrecovering p95.

## Storage And Artifact Hygiene

This project already hit roughly 1 TB combined under `artifacts/` and `.tmp/`. Do not repeat that.

Before long live runs:

- Require at least `150 GB` free disk unless the user explicitly overrides.
- Use campaign artifact quota defaults around `100-150 GB`.
- Prefer dry-runs for cleanup and campaigns before live execution.
- Keep only winners, near runner-ups, final/best checkpoints, reports, and needed resume artifacts.
- Write durable findings to `results/` before deleting bulky artifacts.

Never delete without explicit user approval:

- `src/`
- `scripts/`
- `tests/`
- `configs/`
- `docs/`
- `results/`
- `data/reward/`
- `data/ghosts/`
- user-supplied replay/baseline data

`.tmp/` is disposable, but still verify resolved paths stay inside the workspace before recursive deletion.

## Engineering Conventions

- Prefer small, testable changes.
- Preserve existing SAC/LIDAR/BC behavior unless the task explicitly targets those paths.
- Keep REDQ config/report compatibility fields when adding new metrics.
- Add new config validation when adding config fields.
- Add or update tests for behavior changes.
- Use `apply_patch` for manual file edits.
- Avoid destructive git commands.
- Do not revert unrelated user changes in a dirty worktree.
- Use `rg`/`rg --files` for search.
- Keep comments rare and useful.
- Keep docs timeless: no raw run metrics in README-style docs unless the file is explicitly a dated result report.

## Done Means

For code changes:

- Relevant tests added or updated.
- `compileall` passes for touched Python code.
- Focused tests pass.
- Full `pytest -q` passes when feasible.
- Reports/results updated when research behavior changes.
- Diff reviewed for regressions and accidental artifact churn.

For live-run or campaign work:

- Window resized successfully before the run.
- Live environment gate passed.
- Exact final dual-mode eval exists.
- Summary/report references exact eval artifacts.
- Dated findings appended under `results/`.
- Non-keeper artifacts cleaned or explicitly left for resume.

For docs-only work:

- Commands and paths are checked against current repo files.
- No stale claims contradict README, setup docs, configs, or results.
- No secrets are included.

## Skills, Plugins, MCPs, And Automations

Use tools when they remove real friction. Do not wire in tools or create skills just because they exist.

### Superpowers

Use Superpowers for disciplined development workflow:

- `superpowers:brainstorming`: ambiguous feature/design work where requirements need shaping.
- `superpowers:writing-plans`: multi-step implementation plans.
- `superpowers:test-driven-development`: behavior changes that should start with tests.
- `superpowers:systematic-debugging`: failing tests, runtime bugs, capture issues, or regressions.
- `superpowers:verification-before-completion`: before claiming a fix is complete.
- `superpowers:requesting-code-review`: major feature or risky refactor before final handoff.
- `superpowers:receiving-code-review`: when applying review feedback.
- `superpowers:using-git-worktrees`: parallel long-running branches or multiple live threads touching the same files.

Use the method, not the theater: for simple edits, keep the process lightweight.

### ATeam

Use ATeam when specialist parallelism is useful:

- `ateam:standup`: quick project status from files and git state.
- `ateam:deepdive`: larger research/design questions that benefit from Researcher/Architect/PM views.
- `ateam:run`: full pipeline execution for a substantial feature.
- `ateam:assign`: hand a focused task to a specific role.
- `ateam:status`: inspect an active team run.
- `ateam:generate`: regenerate `.codex/agents/*.toml` after team config changes.

Do not use ATeam for one-file trivial edits. Do use it for roadmap reviews, large refactors, campaign architecture, or parallel analysis that can finish while the main thread works.

### Codex Reviewer

Use Codex Reviewer for:

- Feature-plan review before committing to a complex plan.
- Implementation review before merge or handoff.
- Risky changes in training/eval/reward/storage code.

### GitNexus MCP

Use GitNexus when repository history or graph context matters:

- Before large refactors.
- To identify hotspots and high-risk files.
- To inspect impact/blast radius of changing a symbol or module.
- To query code flows after the repo has been indexed.

If needed, index the repo first:

```powershell
gitnexus analyze
```

The local Codex MCP config should expose `gitnexus`. If the shell `gitnexus` command is unavailable, use the configured MCP or the user-local GitNexus binary.

### Browser Use

Use Browser Use for local web targets, HTML previews, visual artifacts, and browser inspection. Do not use it as a substitute for Trackmania capture/window tests.

### Documents, Spreadsheets, Presentations

Use these only when the requested output is a `.docx`, spreadsheet, slide deck, or similar office artifact.

### Project/User Skills

Useful installed skills include:

- `diagnose`: disciplined bug/performance diagnosis.
- `tdd`: red-green-refactor workflows.
- `grill-with-docs`: stress-test plans against project terminology/docs.
- `improve-codebase-architecture`: architecture/refactor opportunities.
- `unslop`: rewrite AI-ish prose into human-readable text.
- `find-skills`: discover/install helpful skills.
- `skill-creator`: create or update reusable skills.

When a prompt or workflow gets reused, propose a skill. When a stable manual workflow should run on a cadence, propose an automation. When a tool integration becomes reusable across projects, consider a plugin. Do not create superfluous skills, plugins, or automations.

### Automations

Use automations only after the manual workflow is reliable. Good candidates:

- Monitor a long live run or campaign.
- Daily/weekly storage and artifact drift checks.
- Recurring standup summaries.
- CI/log triage.
- Periodic AGENTS.md friction review.

Skills define the method; automations define the schedule.

## Common Mistakes To Avoid

- Starting live training before Trackmania is visible and resized.
- Ranking a run before exact final deterministic and stochastic eval artifacts exist.
- Training on mixed route families.
- Letting `artifacts/` and `.tmp/` grow without a retention plan.
- Running multiple live threads on the same files without a worktree.
- Skipping tests because a change looks small.
- Treating stochastic policy success as deployable deterministic success.
- Creating automations before the manual workflow is proven.
- Keeping one giant thread for every project task instead of one focused thread per task.

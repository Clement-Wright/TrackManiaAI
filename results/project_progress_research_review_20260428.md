# TrackManiaAI Research Review And Progress Record

Date: 2026-04-28\
Scope: project history, code review, experiment record, and research interpretation through the current repository state\
Primary map discussed: `tmrl-test`, map UID `oqIJ5rQDRrNwLPTh9H2p_W4tLof`\
Primary current target: `rank11_100_bundle`

## Executive Abstract

TrackManiaAI has evolved from a Windows/Openplanet live-environment smoke harness into a full-observation reinforcement-learning research stack for Trackmania 2020. The project now includes reward recording, demo recording, behavior cloning, SAC, REDQ, DroQ, CrossQ, checkpoint-authoritative evaluation, top-100 leaderboard ghost ingestion, route-aware ghost bundle construction, fixed-spacing ghost-bundle rewards, offline-to-online scaffolding, policy-mode evaluation, and campaign orchestration.

The most important research conclusion so far is that the project should keep REDQ as the mainline learner for now. The REDQ/DroQ/CrossQ ladder showed REDQ winning the checkpoint-backed deterministic progress metric, even though DroQ was more attractive computationally on paper and CrossQ offered a lower-UTD promise. DroQ remains a good efficiency branch, but it did not convert better update/freshness diagnostics into better deterministic progress. CrossQ remains experimental because the implemented/configured branch failed to learn useful progress in the ladder.

The second major conclusion is that algorithm choice is no longer the only or even the highest-value bottleneck. The system is also limited by evaluation closure, deterministic policy extraction, reward geometry, data quality, and storage management. Stochastic evaluation repeatedly beat deterministic evaluation, which means the learned stochastic actor often contains useful behavior that the deployed deterministic mean-action policy does not preserve. That makes deterministic extraction and policy-output stabilization a central research track.

The third major conclusion is that top-100 ghosts are not automatically one coherent target. On `tmrl-test`, the top 10 leaderboard ghosts appear to use an exploit or shortcut route involving car flipping, while many later ghosts drive a more intended route. Combining incompatible route families into one reward bundle created a multimodal target: the agent could be rewarded for contradictory strategic behaviors, the corridor could become incoherent, and deterministic mean-action behavior could collapse. This led to the route-aware target decision: one bundle equals one driving strategy. For the current map, the `rank11_100_bundle` is the current mainline target because it won the reference-target comparison.

The fourth major conclusion is operational: uncontrolled artifacts can defeat the research loop. The combined `artifacts/` and `.tmp/` footprint reached about 1 TB during long campaign work and had to be purged. Future long runs need hard storage preflight, retention rules, compact summaries, and result logging to `results/` so evidence survives cleanup.

## Project Goal And Constraints

The project goal is to train a Trackmania 2020 driving policy that can make meaningful track progress and eventually complete laps, then improve toward leaderboard-relevant lap-time performance rather than merely imitating one local human line.

Core constraints:

| Constraint | Practical implication |
|---|---|
| Windows-only live game environment | Training depends on Trackmania, Openplanet, window capture, and Windows process behavior. |
| Laptop RTX 3080 | Compute is meaningful but limited; learner backprop, replay size, and artifact volume must be managed. |
| Single live Trackmania instance | Environment collection is wall-clock limited and vulnerable to window/capture disruptions. |
| `256x128` full-observation window | Full-observation REDQ uses a small client rect resized to `64x64` grayscale frame stacks. |
| 2D action path | Current policy controls `throttle` and `steer`; older `gas/brake/steer` concepts are legacy compatibility. |
| Openplanet bridge | Telemetry, reset, reward recording, and replay extraction all rely on bridge/plugin health. |
| Storage pressure | Long runs can create very large train/eval/checkpoint directories if not aggressively retained. |

The project has therefore prioritized a practical research stack over a purely theoretical algorithm search. Good experiments must be live-runnable, checkpoint-backed, reproducible, and storage-bounded.

## Current System Architecture

At review time, the repository is organized around a full-observation training loop:

| Subsystem | Current role |
|---|---|
| Openplanet bridge | Provides live telemetry, command/reset integration, map UID, position, velocity, inputs, and replay/export support. |
| Capture stack | Uses the Trackmania client rect, primarily `256x128`, then preprocesses to `64x64` grayscale stacks. |
| Action space | Uses a compact 2D `throttle, steer` policy interface for the main full-observation path. |
| Reward | Supports the original single-trajectory progress reward and the newer ghost-bundle progress reward. |
| Learner/worker | Asynchronous learner and live worker exchange replay, actor publications, eval commands, and diagnostics. |
| Algorithms | SAC, REDQ, DroQ, and CrossQ are present; REDQ is the mainline full-observation learner. |
| Evaluation | Scheduled dual-mode evaluation supports deterministic and stochastic modes, action traces, and checkpoint provenance. |
| Reports | Training reports include progress, UTD, actor staleness, eval summaries, route-family provenance, and corridor diagnostics. |
| Ghost pipeline | Fetches top-100 records, stores `.gbx`, consumes Openplanet/GBX.NET trajectory exports, builds target bundles. |
| Offline scaffolding | Adds offline transition loading, BC/REDQ/AWAC/CQL-style pretraining primitives, and balanced replay. |
| Campaign tooling | Runs algorithm ladders, rank11 validation campaigns, exact-final-eval repair, and artifact cleanup. |

Important current configuration facts:

| Setting | Current mainline value |
|---|---|
| Full-observation baseline | REDQ-SAC |
| Critics | `4` |
| REDQ subset | `m_subset=2` |
| Encoder sharing | `share_encoders=true` |
| Policy delay | `q_updates_per_policy_update=5` |
| Target UTD schedule | `max_training_steps_per_environment_step=8.0`, update intervals `8/8` |
| Eval protocol | deterministic and stochastic |
| Eval trace window | `3.0` seconds |
| `tmrl-test` target | `data/ghosts/oqIJ5rQDRrNwLPTh9H2p_W4tLof/ghost_bundle_rank_011_100.json` |
| Ghost progress semantics | fixed-spacing TMRL-style progress, `0.5 m` spacing |
| Metric version | `ghost_bundle_progress_v2_fixed_spacing` |

## Chronology Of Work

### 1. Bridge, Capture, And Phase-1 Foundations

The project began as a live Trackmania environment integration. The early priority was not algorithm sophistication; it was proving that the bridge, reset path, observation capture, and action path could work reliably enough for RL.

Key outcomes:

| Outcome | Why it mattered |
|---|---|
| Openplanet bridge checks | Established telemetry and command reliability as prerequisites. |
| Window sizing scripts | Made the live capture contract explicit for full-observation and lidar modes. |
| Reward recording | Created the first TMRL-style progress reference from a manual lap. |
| Demo recording and BC scripts | Enabled human demonstration collection and actor warm-start experiments. |
| Smoke harness | Provided a diagnostic entrypoint rather than the repo identity. |

This phase created the operational base: the system could observe the game, control the car, reset episodes, and write artifacts.

### 2. SAC And Early Full-Observation RL

SAC was the initial continuous-control baseline because it was simple, standard, and compatible with the actor/critic stack. SAC remained useful as a compatibility and comparison path, but early work indicated that sample efficiency and learner scheduling would be limiting for live Trackmania training.

Key lesson: the main bottleneck was not just whether the control loop worked. It was whether the learner could extract enough progress signal from costly real-time environment interaction.

### 3. REDQ Realignment

The project moved toward REDQ because REDQ's design targets sample efficiency by combining high update-to-data ratios, critic ensembles, and random-subset target minimization. The initial REDQ runs were not actually operating in REDQ's intended regime: learner steps were close to environment steps, giving achieved UTD near 1 rather than high UTD.

The realignment introduced:

| Change | Rationale |
|---|---|
| Smaller update intervals | Avoid large bursty update blocks and sustain a smoother UTD target. |
| `max_training_steps_per_environment_step=8.0` | Make REDQ meaningfully high-UTD. |
| Actor publication after actor updates | Keep live control closer to the latest learned actor. |
| `actor_publish_every=1` | Avoid stranding fresher actors inside critic-heavy update blocks. |
| Actor staleness diagnostics | Make worker/learner policy lag visible. |
| Rolling and cumulative UTD metrics | Separate short-window update health from cumulative learner/env ratio. |
| Dual-mode eval | Diagnose deterministic deployment collapse against stochastic policy behavior. |

This was a major architectural decision: before comparing algorithms, REDQ needed to be made into a real REDQ-like system.

### 4. 10-Critic To 4-Critic REDQ Baseline

After shared critic encoders were added, learner backprop remained the bottleneck. The next compute-reduction decision was to shrink the REDQ ensemble from 10 critics to 4 critics while keeping `m_subset=2` and `share_encoders=true`.

Why this decision was made:

| Observation | Decision |
|---|---|
| 10 critics increased critic-backprop cost. | Reduce to 4 critics. |
| Shared encoders already helped but did not remove the critic bottleneck. | Keep shared encoders as the default. |
| Actor updates were comparatively cheap. | Keep `q_updates_per_policy_update=5`, not the old very delayed actor schedule. |
| Need a realistic laptop baseline. | Make 4-critic REDQ the shipped full-observation baseline. |

This made the current baseline a compute-reduced REDQ, not a full paper-maximal REDQ.

### 5. REDQ / DroQ / CrossQ Ladder

A 90-minute ladder compared REDQ, DroQ, and CrossQ under the then-current live stack.

Durable results from `results/redq_droq_crossq_ladder.md`:

| Algorithm | Best checkpoint-backed deterministic progress | Best stochastic progress | Final/reached env steps | Interpretation |
|---|---:|---:|---:|---|
| REDQ | `496.0` | `1144.4` | `61082` | Winner on official deterministic metric. |
| DroQ | `234.4` | `819.4` | `50015` | Better efficiency diagnostics did not beat REDQ. |
| CrossQ | `11.8` | `46.8` | `65059` | Failed to learn useful progress in this implementation/config. |

Key ladder findings:

| Finding | Research consequence |
|---|---|
| REDQ won deterministic progress. | REDQ remains mainline. |
| DroQ had stronger UTD/freshness but worse progress. | UTD alone is not the limiting factor. |
| CrossQ was cheaper but learned poorly. | CrossQ is experimental, not the next default. |
| Stochastic beat deterministic for all algorithms. | Deterministic extraction is a core problem. |
| No algorithm finished the map. | Progress-only metrics remain preliminary. |

This ladder changed the roadmap. The next work should improve REDQ's reward, data, evaluation, and deployment behavior before another major algorithm swap.

### 6. Checkpoint-Authoritative Eval And Determinism Diagnostics

The project then introduced checkpoint-authoritative evaluation:

| Feature | Purpose |
|---|---|
| Exact checkpoint paths | Eval metadata should name the artifact actually evaluated. |
| Checkpoint SHA-256 | Detect stale or wrong-policy eval provenance. |
| Env/learner/actor step provenance | Tie eval results to specific training state. |
| Deterministic and stochastic mode summaries | Keep deterministic as headline, stochastic as diagnostic. |
| Determinism conversion score | Measure how much stochastic capability survives deterministic extraction. |
| Policy-mode sweeps | Evaluate deterministic mean, clipped mean, stochastic temperatures, and best-of-k diagnostics. |

The reason was concrete: prior eval summaries could lag one checkpoint behind, and stop-step evals could be dropped as in-flight during shutdown. Without fixing this, algorithm comparisons were not trustworthy.

Current status: the code now has validation and repair paths for exact final eval artifacts, but the durable experimental record still contains runs where final exact dual-mode eval was missing or in-flight. This phase should be considered mostly implemented but not fully proven until a long campaign completes with exact-final-eval closure on every leg.

### 7. Top-100 Ghost Ingestion And First Ghost-Target Run

The project added a leaderboard ghost data path:

| Step | Result |
|---|---|
| Nadeo auth and leaderboard fetch | Retrieved 100 top records for `tmrl-test`. |
| Replay download | Stored 100 `.gbx` files. |
| Openplanet/GBX.NET export | Converted ghost samples into positions, velocities, speed, gas/brake/steer, and gear. |
| Trajectory normalization | Produced Parquet plus metadata. |
| Bundle construction | Initially selected 20 representatives across rank bands. |
| Offline transition seeding | Implemented, but offline transition count was 0 without observation sidecars. |

The first 120-minute top-100 REDQ run showed the ghost reward could create a useful early learning window but exposed brittleness:

| Metric | Result |
|---|---|
| Final env steps | `57323` |
| Learner steps | `101048` |
| Actor steps | `20209` |
| Best deterministic eval | `91.0` at steps `20004` and `25004` |
| Best stochastic eval | `90.6` at steps `20004` and `50015` |
| Main termination pattern | `1265` stray, `173` no-progress |
| Movement-started rate | `0.9986` |
| Final exact eval | Missing due worker fatal/shutdown path |

Interpretation: the car was not dead. It moved. The reward corridor and target geometry were too brittle, and deterministic collapse reappeared after initial learning.

### 8. Ghost Reward Semantics Redesign

The first ghost reward used ghost row/sample indices too directly. That made progress values hard to compare to the old TMRL-style reward, which resampled a trajectory at fixed spatial intervals. With multiple ghost lines, row indices are especially fragile because each trajectory can have different sample density and timing.

The reward was redesigned to:

| Change | Rationale |
|---|---|
| Resample each selected ghost line to `reward.spacing_meters` | Make ghost progress comparable to old fixed-distance trajectory reward. |
| Define `progress_index=floor(progress_arc_length_m / spacing)` | Restore TMRL-style progress semantics. |
| Treat source row index as diagnostic only | Avoid training on raw replay sample indices. |
| Use arc-length-compatible line switching | Prevent regressions or false mismatches across differently sampled ghosts. |
| Replace immediate stray reset with corridor penalties | Allow recovery instead of killing the rollout on first corridor miss. |
| Add recovery-gated truncation | End only sustained nonrecovering hard violations, not ordinary drift. |
| Add arc-length and ghost-relative metrics | Make progress, route fraction, and time delta visible. |

This was one of the most important reward-level corrections. It turned the ghost bundle from a row-index reward into a spatial progress reward.

### 9. Route-Aware Target Families

A major live observation changed the project direction: the top 10 `tmrl-test` ghosts used a route/strategy that involved flipping or shortcuting, while many later ghosts drove the route more normally. Mixing those into one top-100 target created a contradictory reward manifold.

The target pipeline was changed to support:

| Feature | Purpose |
|---|---|
| Strategy-family classification | Separate intended routes from shortcut/exploit routes. |
| Canonical reference projection | Compare ghosts against an intended-route reference. |
| Selected ghost override | Use a named/ranked ghost when intended-family selection cannot be trusted. |
| Author/reference fallback | Prefer official or supplied baseline over mixed targets. |
| Hard stop on ambiguity | Avoid silently training on incompatible families. |
| Separate intended/exploit bundles | Preserve exploit data without poisoning the mainline target. |

For `tmrl-test`, the project then created and selected `ghost_bundle_rank_011_100.json`, representing ranks 11 through 100. This was a pragmatic route-family decision: exclude the top-10 exploit family and train on the broader non-top-10 target.

### 10. Reference Target Suite

A 120-minute-per-leg reference suite compared:

| Target | Latest recorded deterministic progress | Meters | Fraction of reference | Ghost-relative delta |
|---|---:|---:|---:|---:|
| `rank11_100_bundle` | `1694.0` | `847.0` | `0.4205` | `-16007.45 ms` |
| `author_account_proxy` | `1490.0` | `745.0` | `0.3757` | `-15563.66 ms` |
| `isfoo_rank11` | `1062.6` | `531.3` | `0.2736` | `-11060.12 ms` |
| `reward_trajectory` | `320.0` | `160.0` | `0.0436` | not available |

Important caveat: all four legs had the same final eval shutdown quirk. These were latest completed scheduled eval values, not confirmed exact final eval values.

Interpretation: `rank11_100_bundle` is the current default target for `tmrl-test`, but the exact-final-eval validation criterion still needs to be enforced before treating future suite winners as final.

### 11. Rank11 Validation Campaign Attempt

The next intended campaign was a structured validation matrix:

| Block | Purpose |
|---|---|
| A0-A4 | Compare five corridor/recovery reward settings for `rank11_100_bundle`. |
| B | Run deterministic extraction sweeps on the winning Block A checkpoint. |
| C | Compare offline warm-start variants after reward and extraction winners are selected. |

Only A0 was run during the last campaign attempt before the storage issue became urgent. The user then requested the full campaign be run, but the combined `artifacts/` and `.tmp/` directories had grown to about 1 TB. The decision was made to delete everything possible in those directories.

This created a necessary operational reset: before more long experiments, artifact retention and disk quota must become part of the run system rather than an afterthought.

## Algorithm Evolution And Rationale

### SAC

SAC remains a baseline and compatibility path. It is useful for verifying the actor/critic stack, action path, and report generation. It is not currently the main full-observation research path because live sample efficiency is too important.

### REDQ

REDQ is the mainline learner because:

| Reason | Evidence |
|---|---|
| Best deterministic ladder result | REDQ beat DroQ and CrossQ on checkpoint-backed deterministic progress. |
| Compatible with current codebase | REDQ reuses the existing actor/replay/eval/control plane. |
| Ensemble targets are stable enough | REDQ provides a practical bias-control mechanism without redesigning the whole learner. |
| Tunable compute | The 4-critic shared-encoder baseline reduced the 10-critic cost. |

Main unresolved REDQ issues:

| Issue | Implication |
|---|---|
| Deterministic extraction collapse | Policy may learn useful stochastic behavior that mean-action driving loses. |
| Reward brittleness | Corridor/route signals can dominate learning before first finish. |
| Learner backprop bottleneck | High UTD remains expensive on a laptop. |
| No completions yet | Progress wins are preliminary. |

### DroQ

DroQ was added as a natural lower-compute branch: fewer critics, shared encoder, dropout, and layer norm. It is still attractive if the REDQ behavior can be retained with less compute. However, in the ladder it did not outperform REDQ despite better UTD/freshness diagnostics.

Current role: secondary efficiency branch to revisit after reward, target, and deterministic extraction are stable.

### CrossQ

CrossQ was added because it promises lower-compute performance at UTD near 1 through careful batch normalization and no target networks. In the actual ladder, the implemented/configured CrossQ run was not viable: it progressed poorly and all training episodes terminated with no progress.

Current role: experimental branch. It should not drive the roadmap until the mainline REDQ stack is cleaner and CrossQ-specific implementation/config issues are investigated.

## Evaluation And Metrics Evolution

The project started with progress as the main scalar. That remains valuable for continuity, but it is not enough for racing. The metric stack now includes or is designed to include:

| Metric | Status | Purpose |
|---|---|---|
| `mean_final_progress_index` | Implemented | Continuity headline metric. |
| `mean_final_progress_meters` | Implemented | Physical progress scale. |
| `final_arc_length_m` | Implemented in reward/eval info path | Arc-length progress under ghost reward. |
| `progress_fraction_of_reference` | Implemented | Fraction of target reference covered. |
| `ghost_relative_time_delta_ms` | Implemented where ghost timing exists | Timing delta against ghost bundle at current arc length. |
| Completion rate | Implemented | Eventually the most basic success metric. |
| Determinism conversion score | Implemented | Deterministic progress divided by stochastic reference progress. |
| Corridor diagnostics | Implemented | Understand reward brittleness and recovery. |
| Sector metrics | Partially implemented/planned | Identify where progress is gained/lost. |
| PLTS-style racing metric | Planned | Combine progress, time potential, stability, and completion. |

The current ranking rule should remain conservative:

| Rule | Reason |
|---|---|
| Official scoreboard uses checkpoint-backed deterministic progress. | Deterministic is the deployable policy mode. |
| Stochastic remains a diagnostic. | It reveals latent policy capability and exploration quality. |
| Exact final eval is preferred over scheduled eval. | Scheduled eval can overstate or understate final model state. |
| Missing exact final eval marks a run incomplete. | Prevents stale scheduled metrics from becoming practical winners. |

## Ghost Data And Offline-To-Online State

Implemented components:

| Component | Current status |
|---|---|
| Nadeo top-100 fetch | Implemented and used successfully after credentials were fixed. |
| `.gbx` replay storage | Implemented. |
| Openplanet/GBX.NET trajectory export path | Implemented as the reliable first extraction route. |
| Trajectory normalization | Implemented. |
| Route-aware bundle construction | Implemented. |
| Selected ghost and author fallback | Implemented. |
| `rank11_100_bundle` config | Implemented for `tmrl-test`. |
| Offline transition loader | Implemented with fail-closed action validation. |
| BC/REDQ/AWAC/CQL-style pretraining script | Implemented. |
| Balanced replay | Implemented. |
| Elite archive scaffolding | Implemented. |

Important boundary: offline pretraining is not yet a proven performance win. The first top-100 run had `offline_transition_count=0` because observation sidecars were not available. Action labels were present from the ghost export path, but actor/critic pretraining requires a validated transition dataset aligned to model observations. This fail-closed behavior is correct: training on guessed actions or mismatched observations would be worse than not pretraining.

## Code Review Summary

### Strengths

| Strength | Evidence in codebase |
|---|---|
| Modular config surface | `src/tm20ai/config.py` has explicit config dataclasses and validation. |
| Mainline REDQ config is explicit | `configs/full_redq_top100_tmrl_test.yaml` points at `rank11_100_bundle` and dual-mode eval. |
| Exact-final eval validation exists | `src/tm20ai/train/campaign.py` validates deterministic/stochastic exact final summaries on disk. |
| Reporting knows exact-final status | `src/tm20ai/train/reporting.py` exposes final eval state and provenance. |
| Ghost reward now uses fixed-spacing progress | `src/tm20ai/ghosts/reward.py` resamples selected ghost lines to `reward.spacing_meters`. |
| Corridor recovery is diagnostic-rich | Reward info includes distance, radii, penalties, recovery, nonrecovering steps, speed, and progress deltas. |
| Route-family provenance propagates | Worker, learner, reporting, and reward paths carry bundle resolution and strategy metadata. |
| Offline data fails closed | `src/tm20ai/ghosts/offline.py` requires action/transition validity before actor imitation. |
| Campaign cleanup exists | `src/tm20ai/train/artifact_retention.py` can keep best/latest runs and referenced evals. |

### Risks And Gaps

| Risk | Current consequence |
|---|---|
| Exact final eval closure not empirically proven after purge | Future campaigns must validate closure before ranking. |
| Storage retention is not hard enough | Existing cleanup can run after campaigns, but preflight quotas and hard-stop thresholds need to be enforced. |
| Deterministic extraction remains unresolved | Stochastic > deterministic means deployable policy quality is limited. |
| No map completions yet | Progress metrics are useful but still pre-finish diagnostics. |
| Offline pretraining not validated live | Sidecar/transition readiness is still gating. |
| Route classification is map-dependent | Future maps must not inherit `tmrl-test` assumptions. |
| Chrome/window interruptions affected runs | Capture/window guard mitigations exist, but desktop focus/compositor stability remains operationally important. |
| `.tmp` is easy to overuse | Long campaigns should not depend on unbounded scratch storage. |

## Key Decisions And Why They Were Made

| Decision | Why it was chosen |
|---|---|
| REDQ remains mainline | It won the algorithm ladder on the official deterministic metric. |
| DroQ becomes an efficiency branch | It is plausible but did not beat REDQ yet. |
| CrossQ becomes experimental | The current implementation/config did not learn useful progress. |
| 4 critics replace 10 critics | Learner backprop was the bottleneck, and 4 critics reduce compute. |
| Shared encoders are default | They reduce critic compute while preserving the REDQ path. |
| Deterministic remains headline | It is the deployable control mode and continuity metric. |
| Stochastic eval stays mandatory | It diagnoses latent policy ability and deterministic collapse. |
| Exact checkpoint eval is required | Eval metadata must name the policy actually evaluated. |
| Ghost progress is fixed-spacing | Raw ghost row indices are not comparable across trajectories. |
| One bundle equals one strategy | Mixed exploit/intended route families create contradictory rewards. |
| `rank11_100_bundle` is default for `tmrl-test` | It won the reference-target suite among current options. |
| No mixed fallback | Ambiguous target selection should ask for author/user baselines, not silently poison training. |
| Storage cleanup is mandatory | Long experiments already filled about 1 TB of artifact/scratch space. |

## Novelty And Research Value

The project is not just a standard SAC/REDQ implementation. It has developed several project-specific research ideas:

| Novel element | Why it matters |
|---|---|
| Route-aware leaderboard ghost bundles | Trackmania top times can contain incompatible route strategies; separating them is essential. |
| Fixed-spacing multi-ghost reward | Combines TMRL-style progress semantics with a diverse ghost target family. |
| Recovery-gated corridor reward | Trains recovery instead of treating every off-corridor drift as fatal. |
| Checkpoint-authoritative dual-mode eval | Makes live RL comparisons more reproducible and provenance-clean. |
| Determinism conversion score | Quantifies the gap between stochastic capability and deterministic deployment. |
| Laptop-scale algorithm ladder | Provides practical evidence for REDQ vs DroQ vs CrossQ under real live-game constraints. |
| Ghost-driven offline-to-online scaffold | Connects leaderboard replays, route selection, offline pretraining, and online REDQ fine-tuning. |

The most original project-specific insight so far is the interaction between Trackmania route diversity and RL reward geometry. A top-100 leaderboard is not a single behavior distribution; it can contain different route families with different strategic prerequisites. Treating those as one target can create a mean-action policy that is wrong for every family.

## Current Completion Status

| Area | Status |
|---|---|
| Live bridge/capture/control | Functional, but operationally sensitive to window/compositor issues. |
| REDQ mainline | Implemented and current default. |
| DroQ branch | Implemented and tested enough for ladder comparison; not mainline. |
| CrossQ branch | Implemented but not currently viable as mainline. |
| Dual-mode eval | Implemented. |
| Exact final eval closure | Mostly implemented, not fully validated in a clean long campaign. |
| Deterministic extraction diagnostics | Implemented; stabilization not solved. |
| Top-100 ingestion | Implemented and used. |
| Route-aware bundles | Implemented. |
| `rank11_100_bundle` target | Implemented and selected for `tmrl-test`. |
| Fixed-spacing ghost reward | Implemented. |
| Corridor recovery reward | Implemented, needs long-run tuning. |
| Offline pretraining scaffold | Implemented; empirical win not yet shown. |
| Artifact retention | Cleanup exists; hard quota/preflight still needed. |

## Evidence Trail

Durable evidence reviewed:

| Source | What it supports |
|---|---|
| `README.md` | Current operator-facing baseline and workflows. |
| `configs/full_redq_top100_tmrl_test.yaml` | Current `tmrl-test` REDQ/top-100 target config. |
| `results/redq_droq_crossq_ladder.md` | Algorithm ladder results and REDQ mainline decision. |
| `results/redq_ghost_offline_training.md` | Ghost ingestion, top-100 run findings, offline scaffold status. |
| `results/reference_target_runs.md` | Reference target suite and `rank11_100_bundle` decision. |
| `results/comparisons/redq_droq_crossq_ladder_20260420/algorithm_comparison_report.md` | Ladder scoreboard artifact. |
| `src/tm20ai/config.py` | Config/API state and validation. |
| `src/tm20ai/ghosts/reward.py` | Fixed-spacing ghost reward and corridor recovery behavior. |
| `src/tm20ai/train/campaign.py` | Exact-final-eval validation and reward/extraction selection logic. |
| `scripts/run_rank11_100_validation_campaign.py` | A/B/C validation campaign design and cleanup path. |
| `src/tm20ai/train/artifact_retention.py` | Current artifact cleanup capability. |
| `tests/test_campaign.py`, `tests/test_ghost_pipeline.py`, `tests/test_reporting.py`, `tests/test_redq_stack.py` | Coverage for current target, eval, route-family, reporting, and config behavior. |

Deleted evidence:

| Deleted area | Reason |
|---|---|
| `artifacts/` | Purged after combined artifact/scratch growth reached about 1 TB. |
| `.tmp/` | Purged for the same storage recovery reason. |

The surviving result documents are therefore the authoritative durable record for prior long-run outcomes. Future experiments should append durable summaries to `results/` before cleanup.

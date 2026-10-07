# TrackManiaAI Future Research Plan

Date: 2026-04-28\
Primary near-term target: `rank11_100_bundle` on `tmrl-test`\
Primary learner: REDQ, 4 critics, `m_subset=2`, shared encoders\
Primary constraint: laptop RTX 3080 with strict storage and wall-clock limits

## Executive Direction

The next phase of the project should not be another broad algorithm search. The evidence so far says the mainline should be REDQ plus better evaluation, target selection, reward stability, deterministic extraction, and only then offline leverage. DroQ and CrossQ should remain in the repository, but they should wait behind the REDQ cleanup stack.

Near-term priority order:

| Priority | Workstream | Why it comes now |
|---|---|---|
| 0 | Storage and run hygiene | Long campaigns already filled about 1 TB; no more long runs should start without retention gates. |
| 1 | Exact eval closure | A run is not scientifically final until exact deterministic and stochastic final checkpoint eval artifacts exist. |
| 2 | Target-family formalization | Route-family mixing is a core failure mode in Trackmania. |
| 3 | Reward stability on `rank11_100_bundle` | The target is selected; now the reward must stop killing recovery too early. |
| 4 | Deterministic extraction stabilization | The learned stochastic policy often outperforms deterministic deployment. |
| 5 | Offline-to-online ghost leverage | Pretraining only makes sense after the target and reward are stable. |
| 6 | Compute-efficient learning | Revisit DroQ/CrossQ/DrQ-v2 after the mainline comparison surface is trustworthy. |
| 7 | Racing metrics and visualization | Progress is useful, but lap-time-aware diagnostics are needed for real racing improvement. |
| 8 | Scaling and world models | Multi-worker and model-based work are higher-upside but should come after data/eval reliability. |

The guiding principle is: make the REDQ mainline trustworthy before trying to replace it.

## Phase 0: Storage And Run Hygiene

### Objective

Prevent another artifact explosion before any more 60-180 minute campaigns. This phase is a prerequisite for all future live runs.

### Implementation Scope

Add or tighten storage controls around campaign scripts, training scripts, and cleanup scripts:

| Feature | Required behavior |
|---|---|
| Disk preflight | Refuse long live runs when free space is below `150 GB` by default. |
| Campaign quota | Default campaign artifact cap should be `100-150 GB`. |
| Scratch policy | `.tmp/` is disposable and must be cleaned after completed campaigns. |
| Keeper policy | Keep only winner, runner-up within threshold, final checkpoint, best deterministic checkpoint, best stochastic diagnostic checkpoint, and selected report artifacts. |
| Dry-run cleanup | Cleanup script must show exactly what it would delete before destructive cleanup. |
| Results durability | Every long run or campaign writes a dated summary to `results/` before cleanup. |
| Eval trace compaction | Store compact JSON summaries by default; keep full traces/videos only for selected diagnostic episodes. |
| Checkpoint cadence review | Avoid excessive checkpoint density during long laptop runs. |

### Acceptance Criteria

| Criterion | Requirement |
|---|---|
| Long-run preflight | Training/campaign scripts refuse unsafe free-space conditions. |
| Cleanup safety | Cleanup cannot delete `results/`, `configs/`, `src/`, `tests/`, `data/ghosts/`, or reward trajectories. |
| Evidence survival | A campaign summary remains in `results/` after artifact cleanup. |
| Quota enforcement | A dry-run report shows projected retained and removed sizes. |

### Tests

| Test | Purpose |
|---|---|
| Artifact retention unit tests | Keep best/latest runs and referenced eval dirs. |
| Dry-run deletion tests | Verify no deletion happens in dry-run mode. |
| Protected path tests | Ensure source, configs, results, and data are never removed. |
| Low-disk preflight tests | Simulate insufficient free space and assert long run refuses to start. |
| Campaign cleanup integration test | Create fake runs/evals, keep selected winners, delete stale dirs. |

### Storage And Compute Budget

| Item | Budget |
|---|---|
| One 90-minute live run | Prefer below `20-40 GB` retained after cleanup. |
| One 5-leg campaign | Prefer below `100-150 GB` retained. |
| Eval traces | Compact summaries by default. |
| Videos | Off by default; only selected diagnostic episodes. |

### Decision Rule

No new long live run should start until Phase 0 protections are in place or the user explicitly overrides the risk.

## Phase 1: Exact Eval Closure

### Objective

Make final evaluation scientifically reliable. A run or suite leg is not valid until exact checkpoint-backed deterministic and stochastic final eval artifacts exist on disk and are referenced by the run summary and report.

### Implementation Scope

| Area | Required behavior |
|---|---|
| Single-run finalization | Write final checkpoint, evaluate exact deterministic and stochastic modes, wait for both summaries, then mark run complete. |
| Campaign progression | Do not advance to the next leg until the previous leg passes exact-final-eval validation. |
| Repair path | If only final eval is missing, backfill from the final checkpoint instead of rerunning the whole leg. |
| Reporting | Prefer exact final eval over scheduled eval in comparisons. |
| Failure state | Mark `exact_final_eval_missing` and exclude from ranking when artifacts are absent. |
| Provenance | Record checkpoint path, SHA-256, env step, learner step, actor step, map UID, seed schedule, and eval mode. |

### Acceptance Criteria

| Criterion | Requirement |
|---|---|
| Valid run | `exact_final_eval_complete=true`, `incomplete_final_eval=false`, `final_eval_state=complete`. |
| Mode artifacts | Final deterministic and stochastic `summary.json` files both exist. |
| Comparison | Incomplete final eval runs are visibly marked and excluded from winner selection. |
| Backfill | Missing final eval can be repaired from final checkpoint without retraining. |

### Tests

| Test | Purpose |
|---|---|
| Single-run finalization test | Learner finalization blocks until both mode summaries exist. |
| Campaign validation test | `validate_campaign_run()` rejects missing deterministic or stochastic final summaries. |
| Backfill test | `backfill_final_checkpoint_eval.py` repairs a summary and updates final eval fields. |
| Report sorting test | Exact-final rows outrank scheduled rows for official comparison. |
| Shutdown regression test | Stop-step eval is not dropped as in-flight at process exit. |

### Live Validation

Run one short REDQ smoke leg with exact final eval enabled and verify:

| Artifact | Expected |
|---|---|
| `summary.json` | exact final complete fields are valid. |
| `report.md` | exact final eval section names checkpoint provenance. |
| eval dirs | deterministic and stochastic final exact directories exist. |

### Decision Rule

Phase 1 is complete only after at least one single run and one campaign leg finish cleanly with exact final dual-mode artifacts.

## Phase 2: Target-Family Formalization

### Objective

Make target selection a required per-map step. Trackmania maps can contain different route families, shortcuts, flips, and exploit lines. Training must select one coherent strategy family at a time.

### Implementation Scope

| Feature | Required behavior |
|---|---|
| Route-family split | Classify ghosts into `intended_route`, `shortcut_or_exploit`, and `unclassified`. |
| Canonical reference | Prefer official author run; otherwise use user-provided baseline runs. |
| Selected ghost override | Allow explicit name/rank selector for map-specific experiments. |
| No mixed fallback | If target family is ambiguous and no author/baselines exist, stop and ask for baselines. |
| Bundle provenance | Store selected family, resolution mode, counts, reference path, and fallback reason. |
| Runtime consistency | Reward, offline pretraining, and eval use the same selected manifest. |

### Current `tmrl-test` Default

| Field | Value |
|---|---|
| Map UID | `oqIJ5rQDRrNwLPTh9H2p_W4tLof` |
| Default target | `rank11_100_bundle` |
| Bundle path | `data/ghosts/oqIJ5rQDRrNwLPTh9H2p_W4tLof/ghost_bundle_rank_011_100.json` |
| Reason | Excludes the top-10 exploit family and won the reference-target suite. |

### Acceptance Criteria

| Criterion | Requirement |
|---|---|
| No silent mixing | Ambiguous maps cannot produce default mixed training bundles. |
| Provenance | Summaries, checkpoints, and reports state selected family and resolution mode. |
| Consistency | Reward and offline data use the same bundle. |
| Map portability | New maps require explicit target-family resolution before long runs. |

### Tests

| Test | Purpose |
|---|---|
| Intended route fixture | Following anchors in order classifies as intended. |
| Exploit fixture | Skipping anchors or early reverse progress classifies as shortcut/exploit. |
| Ambiguous fixture | No author/baseline creates a hard-stop state. |
| Selected ghost test | Name/rank override resolves exactly or fails on ambiguity. |
| Author fallback test | Author/reference manifest becomes single-family fallback. |
| Report propagation test | Bundle resolution appears in summary/checkpoint/report. |

### Decision Rule

For `tmrl-test`, use `rank11_100_bundle` until another coherent intended-family target beats it on exact-final deterministic evaluation. For future maps, never train a long run on mixed route families.

## Phase 3: Reward Stability On `rank11_100_bundle`

### Objective

Tune the reward/corridor design around the selected target family. The goal is less brittle early termination, better recovery learning, and more stable deterministic progress.

### Implementation Scope

Use only `rank11_100_bundle` for this phase. Do not compare target families during reward tuning.

Reward settings to test:

| Variant | Purpose |
|---|---|
| A0 baseline current | Current corridor settings. |
| A1 wider recovery | More patience and lower recovery thresholds. |
| A2 softer wider corridor | Larger soft/hard margins and gentler penalties. |
| A3 buffered hard boundary | Longest patience and easiest recovery gate. |
| A4 hard stray control | Strict legacy-like control to measure brittleness. |

Primary metrics:

| Metric | Use |
|---|---|
| exact final deterministic `mean_final_progress_index` | Official continuity score. |
| `mean_final_progress_meters` | Physical progress. |
| `final_arc_length_m` | Arc-length progress under reward geometry. |
| `progress_fraction_of_reference` | Fraction of route family covered. |
| `ghost_relative_time_delta_ms` | Timing diagnostic where available. |
| `corridor_violation_truncation_rate` | Measures fatal corridor brittleness. |
| `corridor_nonrecovering_steps` p95 | Measures recovery failure. |
| no-progress termination rate | Separates corridor failure from dead driving. |

### Acceptance Criteria

| Criterion | Requirement |
|---|---|
| Valid eval | Every candidate has exact final deterministic and stochastic eval. |
| Lower brittleness | Winner improves progress without raising corridor truncation pathologically. |
| Stable metrics | Progress meters/fraction and corridor diagnostics are present in report. |
| No route confusion | All variants use the same `rank11_100_bundle`. |

### Tests

| Test | Purpose |
|---|---|
| Fixed-spacing progress test | Unequal ghost sampling densities produce comparable progress. |
| Line-switch test | Diverging/reconverging ghosts do not cause raw-index jumps. |
| Recovery test | Hard-corridor drift does not truncate while speed/progress/distance indicate recovery. |
| Nonrecovering test | Sustained hard violation with low speed and no progress truncates after patience. |
| Reporting test | Arc-length, fraction, ghost timing, and corridor fields propagate. |
| Campaign test | Reward winner uses exact final eval, not scheduled spikes. |

### Live Experiment Design

| Run | Budget | Notes |
|---|---:|---|
| A0 | 90 min | Baseline. |
| A1 | 90 min | Recovery threshold test. |
| A2 | 90 min | Softer corridor test. |
| A3 | 90 min | Patience/buffering test. |
| A4 | 90 min | Hard-control comparator. |

Storage gate: do not run the full A0-A4 matrix until Phase 0 is enforced. If storage is tight, first run 15-minute smoke versions of all five, then 90-minute runs for the top two plus A0.

### Decision Rule

Winner is the valid run with best exact final deterministic progress. If top two are within 5 percent, tie-break by lower ghost-relative time delta, higher progress fraction, lower corridor truncation rate, and lower corridor nonrecovering p95.

## Phase 4: Deterministic Extraction Stabilization

### Objective

Make deployed deterministic control preserve as much of the stochastic actor's capability as possible.

### Implementation Scope

Run policy-mode sweeps only on the best `rank11_100_bundle` checkpoints from Phase 3.

Modes to evaluate:

| Mode | Purpose |
|---|---|
| `deterministic_mean` | Current deployment baseline. |
| `clipped_mean` | Safer bounded mean-action extraction. |
| `stochastic_temp_0.5` | Lower-variance stochastic diagnostic. |
| `stochastic_temp_1.0` | Reference stochastic policy. |
| `stochastic_temp_1.5` | Higher-exploration diagnostic. |
| `sample_best_of_k` | Diagnostic upper bound only, never deployment. |

Metrics:

| Metric | Meaning |
|---|---|
| DCS | Deterministic mode progress divided by stochastic temp-1 progress. |
| Deterministic progress | Deployment score. |
| Progress fraction | Route coverage preservation. |
| Ghost-relative time delta | Timing preservation. |
| Action stability | Detect steering/throttle oscillation or mean-action collapse. |

### Acceptance Criteria

| Criterion | Requirement |
|---|---|
| Same checkpoint | All extraction modes evaluate the same checkpoint artifact. |
| DCS target | Deployment mode should target `DCS >= 0.85`. |
| Clipped mean rule | Choose clipped mean only if it improves progress by at least 5 percent and does not reduce progress fraction. |
| Best-of-k rule | Best-of-k remains diagnostic-only. |

### Tests

| Test | Purpose |
|---|---|
| Policy mode path test | `evaluate_redq_policy_modes.py` writes separate mode summaries. |
| Same checkpoint provenance test | All mode summaries share checkpoint path/SHA. |
| DCS calculation test | Determinism conversion is computed per mode. |
| Deployment selection test | Clipped mean chosen only under the rule. |
| Best-of-k exclusion test | Campaign reports never mark best-of-k as deployment. |

### Decision Rule

If neither deterministic mean nor clipped mean reaches `DCS >= 0.85`, queue deterministic-student distillation before additional scale-up.

## Phase 5: Offline-To-Online Ghost Leverage

### Objective

Use selected intended-family ghost data to improve early learning and exploration without trapping the policy in imitation.

### Preconditions

| Precondition | Why |
|---|---|
| Phase 1 complete | Offline gains must be measured by exact final eval. |
| Phase 2 complete | Pretraining on mixed route families is harmful. |
| Phase 3 winner selected | Offline runs need a stable reward target. |
| Phase 4 deployment mode selected | Offline gains must convert to deterministic control. |
| Observation/action sidecars validated | Actor imitation must not train on guessed actions or mismatched observations. |

### Implementation Scope

| Component | Required behavior |
|---|---|
| Sidecar dataset | Build observation/action transition sidecars aligned to ghost trajectory samples. |
| Action validation | Fail closed if throttle/steer/brake/gas labels are missing or inconsistent. |
| BC warm start | Initialize actor from validated ghost actions. |
| REDQ critic warm start | Train critic on ghost transitions under selected reward semantics. |
| AWAC updates | Weight actor updates by estimated advantage. |
| Optional CQL penalty | Regularize critic conservatively when needed. |
| Balanced replay | Start with high offline fraction and decay toward online data. |
| Demonstration release | Decrease imitation pressure so policy can surpass ghosts. |

### Live Comparisons

| Run | Meaning |
|---|---|
| Baseline Phase 3 winner | No offline initialization. |
| C1 weight init only | Load offline-pretrained checkpoint, no replay seeding. |
| C2 full offline-to-online | Load checkpoint, seed ghost replay, enable balanced replay decay. |

### Acceptance Criteria

| Criterion | Requirement |
|---|---|
| Offline provenance | Checkpoints/reports name strategy, dataset hash, transition count, and bundle path. |
| Early lift | C1/C2 improve 25k or 50k checkpoint progress over baseline. |
| Final lift | At least one offline run beats baseline exact final deterministic progress. |
| No mixed data | Offline data comes only from the selected intended family. |

### Tests

| Test | Purpose |
|---|---|
| Dataset hash stability | Same ghost bundle produces stable manifest hash. |
| Action fail-closed | Missing actions prevent actor imitation. |
| Transition schema test | Observations/actions/rewards/next observations align. |
| BC checkpoint test | Pretrained actor loads into worker policy. |
| REDQ critic warm-start test | Critic update runs on offline batch. |
| Balanced replay decay test | Offline fraction decays from `0.75` to `0.10` over configured steps. |
| Provenance test | Summary/report/checkpoint sidecars include offline metadata. |

### Decision Rule

If offline runs do not improve early lift or final exact deterministic progress, do not scale offline training blindly. First inspect action alignment, reward labels, observation sidecars, and deterministic extraction.

## Phase 6: Compute-Efficient Learning

### Objective

Reduce compute cost only after the REDQ comparison surface is stable.

### Candidate Directions

| Direction | Rationale |
|---|---|
| DroQ revisit | Lower critic count plus dropout/layer norm may preserve REDQ behavior with less backprop. |
| REDQ smaller sweep | Test 2, 3, and 4 critics under the fixed reward/eval stack. |
| DrQ-v2 augmentation | Random-shift augmentation is a strong pixel-control improvement. |
| Mixed precision | Reduce memory and maybe improve throughput if stable on laptop GPU. |
| Smaller encoder | Reduce learner backprop cost. |
| CrossQ rehab | Revisit only after understanding why current CrossQ failed. |
| Lower UTD modes | If learner is bottlenecked, test whether lower UTD plus better reward/data wins. |

### Acceptance Criteria

| Criterion | Requirement |
|---|---|
| Same target | Use `rank11_100_bundle` or the selected map family. |
| Same eval | Exact final deterministic/stochastic eval required. |
| Same deployment mode | Use the selected deterministic extraction rule. |
| Compute reporting | Include env steps, learner steps, wall-clock, GPU memory, storage, and UTD. |

### Tests

| Test | Purpose |
|---|---|
| Algorithm smoke tests | Critic/actor updates work for each branch. |
| Checkpoint roundtrip tests | Worker can load each algorithm's actor-compatible checkpoint. |
| Benchmark tests | Learner-side benchmark records throughput and memory. |
| Exact eval tests | All variants produce final exact dual-mode summaries. |

### Decision Rule

An efficiency branch must either beat REDQ on exact final deterministic progress or come close enough with materially lower compute/storage cost to justify adoption.

## Phase 7: Racing Metrics And Visualization

### Objective

Move beyond raw progress toward lap-time-aware racing quality.

### Metric Stack

| Metric | Purpose |
|---|---|
| Deterministic progress | Continuity scoreboard. |
| Completion rate | Basic success once finishes begin. |
| Lap time | Final racing objective after completion. |
| Progress fraction | Pre-finish route coverage. |
| Ghost-relative time delta | Whether progress is fast or merely far. |
| Sector entry speed | Corner-specific quality. |
| Sector reward gain | Where progress is won/lost. |
| Stability metrics | Steering oscillation, braking, low-speed stalls, high slip if available. |
| PLTS-style score | Proposed combined potential lap-time score before full finishes are common. |

### Visualization Work

| Tool | Purpose |
|---|---|
| Ghost bundle viewer | Compare agent trajectory against selected ghost family. |
| Sector dashboard | Show progress/time/stability by segment. |
| Episode trace browser | Inspect action, speed, corridor distance, reward, and termination reason. |
| Optional replay/video overlay | Save only selected diagnostic episodes to control storage. |

### Acceptance Criteria

| Criterion | Requirement |
|---|---|
| Metrics are versioned | Metric definitions include version and inputs. |
| Reports include context | Progress, time, corridor, and deterministic/stochastic comparisons appear together. |
| Storage safe | Visualization artifacts are opt-in and capped. |

### Tests

| Test | Purpose |
|---|---|
| Metric fixture tests | Known trajectories produce expected progress/time deltas. |
| Sector aggregation tests | Sector summaries are stable and monotonic where expected. |
| Report tests | Markdown/JSON include new fields without breaking old reports. |

### Decision Rule

Do not replace `mean_final_progress_index` as continuity headline until the new racing metric is validated on multiple maps and remains interpretable before first finish.

## Phase 8: Scaling And World Models

### Objective

Explore higher-upside architectures only after data, reward, eval, and storage foundations are reliable.

### Multi-Worker Direction

| Component | Purpose |
|---|---|
| Central learner/trainer | Owns GPU updates and replay. |
| Replay/parameter server | Mediates samples and actor weights. |
| Rollout workers | Run Trackmania instances on one or more machines. |
| Dedicated eval worker | Evaluates exact checkpoints without contaminating training workers. |

Start with multi-machine rollout if possible. Same-machine multi-instance Trackmania should be treated carefully because telemetry origin can become ambiguous depending on interface path.

### World-Model Direction

| Candidate | Why consider later |
|---|---|
| TD-MPC2-style control | Strong continuous-control scaling if representation/data are ready. |
| Dreamer-style world model | Potential to reduce live interaction demand with learned dynamics. |
| Video/ghost pretraining | Could improve visual representation before online RL. |

### Acceptance Criteria

| Criterion | Requirement |
|---|---|
| Mainline baseline stable | REDQ reward/eval/extraction stack is already trustworthy. |
| Data availability | Large, clean replay/telemetry datasets exist. |
| Storage controls | Multi-worker/model-based artifacts cannot exceed quota. |
| Evaluation independence | Dedicated eval workers use exact checkpoints. |

### Tests

| Test | Purpose |
|---|---|
| Replay server tests | Samples and priorities are not corrupted. |
| Parameter sync tests | Workers receive correct actor versions. |
| Multi-worker provenance tests | Every sample records worker/source metadata. |
| Eval isolation tests | Eval worker loads checkpoint artifact, not in-memory actor. |

### Decision Rule

Only pursue this phase when single-worker REDQ is producing stable, interpretable improvements and the project has enough data to justify the engineering cost.

## Near-Term Execution Plan

Recommended immediate sequence:

| Step | Action |
|---|---|
| 1 | Implement hard storage preflight and artifact quota enforcement. |
| 2 | Run exact-final-eval smoke validation on a short REDQ run. |
| 3 | Run 15-minute A0-A4 reward smoke variants if storage is constrained. |
| 4 | Run 90-minute reward variants only for the most promising smoke candidates plus A0. |
| 5 | Select reward winner by exact final deterministic progress and tie-breakers. |
| 6 | Run policy-mode extraction sweep on winner checkpoint. |
| 7 | If DCS is poor, implement deterministic student distillation before offline scale-up. |
| 8 | Build validated observation/action sidecars for `rank11_100_bundle`. |
| 9 | Run C1/C2 offline-to-online only after sidecar validation. |
| 10 | Revisit DroQ efficiency only after REDQ mainline has stable reward and extraction. |

## Long-Run Gating Checklist

Before any 60-180 minute live run:

| Gate | Required result |
|---|---|
| Disk free space | At least `150 GB`, preferably more. |
| Artifact quota | Campaign quota configured or explicitly waived. |
| Window size | Trackmania client rect matches config. |
| Live lock | Stale lock cleared. |
| Environment gate | `check_environment.py --require-reward` passes. |
| Target manifest | Bundle exists and matches intended map. |
| Route family | No mixed fallback; selected family is explicit. |
| Eval config | Deterministic and stochastic final eval enabled. |
| Result file | Campaign has a dated `results/` output path. |
| Resume policy | Only resume affected incomplete legs; never duplicate valid completed legs. |

After any long run:

| Gate | Required result |
|---|---|
| Exact final eval | Both deterministic and stochastic final summaries exist. |
| Report generated | `summary.json`, `report.json`, and `report.md` exist. |
| Results entry | Dated findings appended to `results/`. |
| Cleanup | Non-keeper artifacts removed or compressed. |
| Next decision | Winner, blocker, or rerun reason recorded. |

## Open Research Questions

| Question | Why it matters |
|---|---|
| Can deterministic extraction reach `DCS >= 0.85`? | If not, deployment needs distillation or actor redesign. |
| Which corridor variant reduces early resets without hiding bad driving? | Reward stability is prerequisite for meaningful learning. |
| Can ghost-relative timing predict eventual lap-time quality before finishes? | Progress alone may reward slow route coverage. |
| Are rank 11-100 ghosts truly one coherent intended family on every segment? | Target family quality directly shapes reward. |
| Can offline sidecars be built reliably from replay exports? | Pretraining depends on aligned observations and actions. |
| Does DroQ regain competitiveness after reward/extraction fixes? | It may reduce compute once mainline behavior is stable. |
| Can DrQ-v2 augmentation improve visual generalization cheaply? | Pixel-control gains may be more useful than more critics. |
| What is the first reliable path to map completion? | Completion rate should eventually supersede pre-finish progress. |

## Success Criteria For The Next Major Milestone

The next major milestone should be considered achieved when all are true:

| Criterion | Target |
|---|---|
| Storage safety | Long campaign runs without exceeding quota or needing emergency deletion. |
| Eval closure | All valid runs finish with exact deterministic and stochastic final eval artifacts. |
| Reward stability | Best `rank11_100_bundle` reward variant improves exact final deterministic progress and lowers brittle corridor terminations. |
| Deterministic extraction | Deployment mode improves DCS toward or above `0.85`. |
| Evidence durability | Campaign conclusions are recorded in `results/` before cleanup. |
| Offline readiness | Validated ghost transition dataset exists, or fail-closed reason is documented. |

Only after this milestone should the project prioritize large algorithm ladders, multi-worker scaling, or world-model experiments.

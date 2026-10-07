# Training Report: rank11_100_blockA_full_20260501_live_A0_baseline_current

## Summary
- Observation mode: full
- Algorithm: redq
- Init mode: scratch
- Primary metric: mean_final_progress_index
- Env steps: 39022
- Learner steps: 72280
- Achieved UTD (1k window): 0.64
- Cumulative UTD: 1.852288452667726
- Current actor staleness: 470
- Replay size: 39022
- Online replay size: 39022
- Offline replay size: 0
- Training duration (s): 6118.3910000000615
- Exact final eval complete: True
- Final eval state: complete
- Ghost bundle: C:\Users\clewr\TrackManiaAI\data\ghosts\oqIJ5rQDRrNwLPTh9H2p_W4tLof\ghost_bundle_rank_011_100.json
- Canonical reference: source=reward_trajectory_fallback path=C:\Users\clewr\TrackManiaAI\data\reward\oqIJ5rQDRrNwLPTh9H2p_W4tLof\trajectory_0p5m.npz
- Strategy selection: status=manual_rank11_100_default family=rank11_100_bundle mixed_fallback=False
- Bundle resolution: mode=None selector=None resolved_rank=None resolved_name=None author_fallback_used=None

- Strategy family counts: {'intended_route': 0, 'shortcut_or_exploit': 100, 'unclassified': 0}

## Exact Final Eval
- mean_final_progress_index=358.35 mean_final_progress_meters=179.175 mean_final_arc_length_m=179.175 mean_progress_fraction=0.08895218859400639 mean_ghost_delta_ms=-5534.549236701716 completion_rate=0.0
- provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A0_baseline_current\checkpoints\checkpoint_00038997_final.pt checkpoint_env_step=38997 checkpoint_learner_step=72280

## Checkpoints
- env_step=5005 replay_size=5005 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A0_baseline_current\checkpoints\checkpoint_00005005.pt
- env_step=10004 replay_size=10004 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A0_baseline_current\checkpoints\checkpoint_00010004.pt
- env_step=15014 replay_size=15014 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A0_baseline_current\checkpoints\checkpoint_00015014.pt
- env_step=20008 replay_size=20008 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A0_baseline_current\checkpoints\checkpoint_00020008.pt
- env_step=25002 replay_size=25002 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A0_baseline_current\checkpoints\checkpoint_00025002.pt
- env_step=30013 replay_size=30013 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A0_baseline_current\checkpoints\checkpoint_00030013.pt
- env_step=35008 replay_size=35008 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A0_baseline_current\checkpoints\checkpoint_00035008.pt
- env_step=38997 replay_size=38997 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A0_baseline_current\checkpoints\checkpoint_00038997_final.pt

## Eval History
- env_step=5005 mean_progress=0.0 median_progress=0.0 mean_progress_m=0.0 mean_final_arc_length_m=0.0 mean_progress_fraction=0.0 mean_ghost_delta_ms=0.0 completion_rate=0.0 dcs=0.0
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A0_baseline_current\checkpoints\checkpoint_00005005.pt checkpoint_env_step=5005 checkpoint_learner_step=3824
-   stochastic: mean_progress=2.35 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A0_baseline_current_step_00005005_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=2.35 completion_rate_delta=0.0 dcs=0.0
- env_step=10004 mean_progress=0.0 median_progress=0.0 mean_progress_m=0.0 mean_final_arc_length_m=0.0 mean_progress_fraction=0.0 mean_ghost_delta_ms=0.0 completion_rate=0.0 dcs=0.0
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A0_baseline_current\checkpoints\checkpoint_00010004.pt checkpoint_env_step=10004 checkpoint_learner_step=11512
-   stochastic: mean_progress=50.25 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A0_baseline_current_step_00010004_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=50.25 completion_rate_delta=0.0 dcs=0.0
- env_step=15014 mean_progress=0.0 median_progress=0.0 mean_progress_m=0.0 mean_final_arc_length_m=0.0 mean_progress_fraction=0.0 mean_ghost_delta_ms=0.0 completion_rate=0.0 dcs=0.0
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A0_baseline_current\checkpoints\checkpoint_00015014.pt checkpoint_env_step=15014 checkpoint_learner_step=19288
-   stochastic: mean_progress=331.95 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A0_baseline_current_step_00015014_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=331.95 completion_rate_delta=0.0 dcs=0.0
- env_step=20008 mean_progress=209.0 median_progress=209.0 mean_progress_m=104.5 mean_final_arc_length_m=104.5 mean_progress_fraction=0.0518794681628222 mean_ghost_delta_ms=-3949.018055718595 completion_rate=0.0 dcs=0.9012505390254419
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A0_baseline_current\checkpoints\checkpoint_00020008.pt checkpoint_env_step=20008 checkpoint_learner_step=31368
-   stochastic: mean_progress=231.9 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A0_baseline_current_step_00020008_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=22.900000000000006 completion_rate_delta=0.0 dcs=0.9012505390254419
- env_step=25002 mean_progress=559.2 median_progress=205.0 mean_progress_m=279.6 mean_final_arc_length_m=279.6 mean_progress_fraction=0.13880860572559892 mean_ghost_delta_ms=-6747.134673664514 completion_rate=0.0 dcs=2.2435305917753263
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A0_baseline_current\checkpoints\checkpoint_00025002.pt checkpoint_env_step=25002 checkpoint_learner_step=39808
-   stochastic: mean_progress=249.25 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A0_baseline_current_step_00025002_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=False progress_delta=-309.95000000000005 completion_rate_delta=0.0 dcs=2.2435305917753263
- env_step=30013 mean_progress=215.7 median_progress=216.0 mean_progress_m=107.85 mean_final_arc_length_m=107.85 mean_progress_fraction=0.05354258986947726 mean_ghost_delta_ms=-4035.6783300699726 completion_rate=0.0 dcs=0.867484415845566
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A0_baseline_current\checkpoints\checkpoint_00030013.pt checkpoint_env_step=30013 checkpoint_learner_step=51400
-   stochastic: mean_progress=248.65 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A0_baseline_current_step_00030013_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=32.95000000000002 completion_rate_delta=0.0 dcs=0.867484415845566
- env_step=35008 mean_progress=664.6 median_progress=497.0 mean_progress_m=332.3 mean_final_arc_length_m=332.3 mean_progress_fraction=0.16497174421536667 mean_ghost_delta_ms=-7908.981410527367 completion_rate=0.0 dcs=1.623351245725452
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A0_baseline_current\checkpoints\checkpoint_00035008.pt checkpoint_env_step=35008 checkpoint_learner_step=59608
-   stochastic: mean_progress=409.4 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A0_baseline_current_step_00035008_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=False progress_delta=-255.20000000000005 completion_rate_delta=0.0 dcs=1.623351245725452
- env_step=38997 mean_progress=358.35 median_progress=358.0 mean_progress_m=179.175 mean_final_arc_length_m=179.175 mean_progress_fraction=0.08895218859400639 mean_ghost_delta_ms=-5534.549236701716 completion_rate=0.0 dcs=0.7511791216853579
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A0_baseline_current\checkpoints\checkpoint_00038997_final.pt checkpoint_env_step=38997 checkpoint_learner_step=72280
-   stochastic: mean_progress=477.05 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A0_baseline_current_final_exact_step_00038997_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=118.69999999999999 completion_rate_delta=0.0 dcs=0.7511791216853579

## Diagnostics
- Bottleneck verdict: learner_backprop
- learner_backprop_seconds=3660.443103625672
- worker_env_seconds=2336.344426000025
- ipc_backpressure_seconds=0.11654688720591366
- actor_sync_seconds=119.95771449920721
- achieved_utd_1k=0.64 cumulative_utd=1.852288452667726 current_actor_staleness=470
- time_to_first_ready_actor_seconds=88.01600000006147 time_to_first_applied_ready_actor_seconds=90.70299999997951 time_to_first_policy_control_window_seconds=93.31300000008196
- policy_control_fraction=0.9990526315789474 current_versions_behind=0 applied_lag_p50=0.5470000000204891 applied_lag_p95=0.9220000000204891
- positive_progress_mean=0.4552867894621579 nonpositive_progress_mean=0.5447132105378422 max_no_progress_p95=70.0 final_arc_length_mean=79.86497890295358 progress_fraction_mean=0.03964930746717941 ghost_delta_mean_ms=-2757.8955616543944
- corridor_violation_fraction_mean=0.0005896255063476915 corridor_distance_p95=17.871573529092082 max_corridor_distance_p95=20.251503784918476 corridor_nonrecovering_p95=0.0 corridor_truncations=0
- no_movement_episode_count=5 stall_episode_rate=0.04201680672268908 first_stall_delay_p95_ms=0.0
- actor_params=822308 critic_params=3288196 unique_critic_encoder_params=607776

## Event Logs
- learner: events=0 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A0_baseline_current\learner_events.log
- worker: events=1379 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A0_baseline_current\worker_events.log

## Videos
- No rollout videos were discovered for this run.

## Failure Notes
- termination_reason=max_wall_clock_minutes

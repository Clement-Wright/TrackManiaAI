# Training Report: rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary

## Summary
- Observation mode: full
- Algorithm: redq
- Init mode: scratch
- Primary metric: mean_final_progress_index
- Env steps: 39745
- Learner steps: 73512
- Achieved UTD (1k window): 0.654690618762475
- Cumulative UTD: 1.849591143540068
- Current actor staleness: 92
- Replay size: 39745
- Online replay size: 39745
- Offline replay size: 0
- Training duration (s): 7062.094000000041
- Exact final eval complete: True
- Final eval state: complete
- Ghost bundle: C:\Users\clewr\TrackManiaAI\data\ghosts\oqIJ5rQDRrNwLPTh9H2p_W4tLof\ghost_bundle_rank_011_100.json
- Canonical reference: source=reward_trajectory_fallback path=C:\Users\clewr\TrackManiaAI\data\reward\oqIJ5rQDRrNwLPTh9H2p_W4tLof\trajectory_0p5m.npz
- Strategy selection: status=manual_rank11_100_default family=rank11_100_bundle mixed_fallback=False
- Bundle resolution: mode=None selector=None resolved_rank=None resolved_name=None author_fallback_used=None

- Strategy family counts: {'intended_route': 0, 'shortcut_or_exploit': 100, 'unclassified': 0}

## Exact Final Eval
- mean_final_progress_index=1612.8 mean_final_progress_meters=806.4 mean_final_arc_length_m=806.4 mean_progress_fraction=0.40034069977511794 mean_ghost_delta_ms=-14717.18280665465 completion_rate=0.0
- provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary\checkpoints\checkpoint_00039722_final.pt checkpoint_env_step=39722 checkpoint_learner_step=73512

## Checkpoints
- env_step=5004 replay_size=5004 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary\checkpoints\checkpoint_00005004.pt
- env_step=10009 replay_size=10009 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary\checkpoints\checkpoint_00010009.pt
- env_step=15003 replay_size=15003 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary\checkpoints\checkpoint_00015003.pt
- env_step=20005 replay_size=20005 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary\checkpoints\checkpoint_00020005.pt
- env_step=25001 replay_size=25001 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary\checkpoints\checkpoint_00025001.pt
- env_step=30007 replay_size=30007 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary\checkpoints\checkpoint_00030007.pt
- env_step=35001 replay_size=35001 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary\checkpoints\checkpoint_00035001.pt
- env_step=39722 replay_size=39722 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary\checkpoints\checkpoint_00039722_final.pt

## Eval History
- env_step=5004 mean_progress=0.0 median_progress=0.0 mean_progress_m=0.0 mean_final_arc_length_m=0.0 mean_progress_fraction=0.0 mean_ghost_delta_ms=0.0 completion_rate=0.0 dcs=0.0
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary\checkpoints\checkpoint_00005004.pt checkpoint_env_step=5004 checkpoint_learner_step=3936
-   stochastic: mean_progress=3.75 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary_step_00005004_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=3.75 completion_rate_delta=0.0 dcs=0.0
- env_step=10009 mean_progress=0.0 median_progress=0.0 mean_progress_m=0.0 mean_final_arc_length_m=0.0 mean_progress_fraction=0.0 mean_ghost_delta_ms=0.0 completion_rate=0.0 dcs=0.0
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary\checkpoints\checkpoint_00010009.pt checkpoint_env_step=10009 checkpoint_learner_step=11648
-   stochastic: mean_progress=79.55 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary_step_00010009_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=79.55 completion_rate_delta=0.0 dcs=0.0
- env_step=15003 mean_progress=0.0 median_progress=0.0 mean_progress_m=0.0 mean_final_arc_length_m=0.0 mean_progress_fraction=0.0 mean_ghost_delta_ms=0.0 completion_rate=0.0 dcs=0.0
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary\checkpoints\checkpoint_00015003.pt checkpoint_env_step=15003 checkpoint_learner_step=20664
-   stochastic: mean_progress=207.45 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary_step_00015003_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=207.45 completion_rate_delta=0.0 dcs=0.0
- env_step=20005 mean_progress=0.0 median_progress=0.0 mean_progress_m=0.0 mean_final_arc_length_m=0.0 mean_progress_fraction=0.0 mean_ghost_delta_ms=0.0 completion_rate=0.0 dcs=0.0
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary\checkpoints\checkpoint_00020005.pt checkpoint_env_step=20005 checkpoint_learner_step=31600
-   stochastic: mean_progress=204.7 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary_step_00020005_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=204.7 completion_rate_delta=0.0 dcs=0.0
- env_step=25001 mean_progress=221.95 median_progress=199.0 mean_progress_m=110.975 mean_final_arc_length_m=110.975 mean_progress_fraction=0.055094009371954 mean_ghost_delta_ms=-4054.323699671324 completion_rate=0.0 dcs=0.6987250118054462
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary\checkpoints\checkpoint_00025001.pt checkpoint_env_step=25001 checkpoint_learner_step=40296
-   stochastic: mean_progress=317.65 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary_step_00025001_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=95.69999999999999 completion_rate_delta=0.0 dcs=0.6987250118054462
- env_step=30007 mean_progress=201.3 median_progress=201.0 mean_progress_m=100.65 mean_final_arc_length_m=100.65 mean_progress_fraction=0.049968119335770854 mean_ghost_delta_ms=-3852.938452774402 completion_rate=0.0 dcs=0.5949460617703561
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary\checkpoints\checkpoint_00030007.pt checkpoint_env_step=30007 checkpoint_learner_step=50072
-   stochastic: mean_progress=338.35 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary_step_00030007_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=137.05 completion_rate_delta=0.0 dcs=0.5949460617703561
- env_step=35001 mean_progress=743.6 median_progress=461.5 mean_progress_m=371.8 mean_final_arc_length_m=371.8 mean_progress_fraction=0.18458168672667266 mean_ghost_delta_ms=-8560.923843591383 completion_rate=0.0 dcs=1.4739345887016848
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary\checkpoints\checkpoint_00035001.pt checkpoint_env_step=35001 checkpoint_learner_step=59424
-   stochastic: mean_progress=504.5 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary_step_00035001_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=False progress_delta=-239.10000000000002 completion_rate_delta=0.0 dcs=1.4739345887016848
- env_step=39722 mean_progress=1612.8 median_progress=1604.5 mean_progress_m=806.4 mean_final_arc_length_m=806.4 mean_progress_fraction=0.40034069977511794 mean_ghost_delta_ms=-14717.18280665465 completion_rate=0.0 dcs=1.1231197771587744
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary\checkpoints\checkpoint_00039722_final.pt checkpoint_env_step=39722 checkpoint_learner_step=73512
-   stochastic: mean_progress=1436.0 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary_final_exact_step_00039722_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=False progress_delta=-176.79999999999995 completion_rate_delta=0.0 dcs=1.1231197771587744

## Diagnostics
- Bottleneck verdict: learner_backprop
- learner_backprop_seconds=3646.549237774452
- worker_env_seconds=2322.490037594689
- ipc_backpressure_seconds=0.11970820371061563
- actor_sync_seconds=121.07597922044806
- achieved_utd_1k=0.654690618762475 cumulative_utd=1.849591143540068 current_actor_staleness=92
- time_to_first_ready_actor_seconds=86.53100000019185 time_to_first_applied_ready_actor_seconds=89.48500000010245 time_to_first_policy_control_window_seconds=92.5
- policy_control_fraction=0.9990180878552971 current_versions_behind=0 applied_lag_p50=0.5470000000204891 applied_lag_p95=0.8652500000665895
- positive_progress_mean=0.41145876994492936 nonpositive_progress_mean=0.5885412300550706 max_no_progress_p95=70.0 final_arc_length_mean=88.44575471698113 progress_fraction_mean=0.043909270009343575 ghost_delta_mean_ms=-2797.71622624298
- corridor_violation_fraction_mean=0.00016172506738544476 corridor_distance_p95=17.647483719027758 max_corridor_distance_p95=20.59411577637286 corridor_nonrecovering_p95=0.0 corridor_truncations=0
- no_movement_episode_count=4 stall_episode_rate=0.03286384976525822 first_stall_delay_p95_ms=0.0
- actor_params=822308 critic_params=3288196 unique_critic_encoder_params=607776

## Event Logs
- learner: events=0 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary\learner_events.log
- worker: events=1286 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary\worker_events.log

## Videos
- No rollout videos were discovered for this run.

## Failure Notes
- termination_reason=max_wall_clock_minutes

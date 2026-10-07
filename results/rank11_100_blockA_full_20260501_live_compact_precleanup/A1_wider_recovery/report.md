# Training Report: rank11_100_blockA_full_20260501_live_A1_wider_recovery

## Summary
- Observation mode: full
- Algorithm: redq
- Init mode: scratch
- Primary metric: mean_final_progress_index
- Env steps: 35028
- Learner steps: 71344
- Achieved UTD (1k window): 1.7636544190665342
- Cumulative UTD: 2.0367705835331735
- Current actor staleness: 1154
- Replay size: 35028
- Online replay size: 35028
- Offline replay size: 0
- Training duration (s): 6705.515999999829
- Exact final eval complete: True
- Final eval state: complete
- Ghost bundle: C:\Users\clewr\TrackManiaAI\data\ghosts\oqIJ5rQDRrNwLPTh9H2p_W4tLof\ghost_bundle_rank_011_100.json
- Canonical reference: source=reward_trajectory_fallback path=C:\Users\clewr\TrackManiaAI\data\reward\oqIJ5rQDRrNwLPTh9H2p_W4tLof\trajectory_0p5m.npz
- Strategy selection: status=manual_rank11_100_default family=rank11_100_bundle mixed_fallback=False
- Bundle resolution: mode=None selector=None resolved_rank=None resolved_name=None author_fallback_used=None

- Strategy family counts: {'intended_route': 0, 'shortcut_or_exploit': 100, 'unclassified': 0}

## Exact Final Eval
- mean_final_progress_index=197.65 mean_final_progress_meters=98.825 mean_final_arc_length_m=98.825 mean_progress_fraction=0.04906209034632444 mean_ghost_delta_ms=-3808.8144210126948 completion_rate=0.0
- provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A1_wider_recovery\checkpoints\checkpoint_00035028_final.pt checkpoint_env_step=35028 checkpoint_learner_step=71344

## Checkpoints
- env_step=5006 replay_size=5006 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A1_wider_recovery\checkpoints\checkpoint_00005006.pt
- env_step=10004 replay_size=10004 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A1_wider_recovery\checkpoints\checkpoint_00010004.pt
- env_step=15005 replay_size=15005 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A1_wider_recovery\checkpoints\checkpoint_00015005.pt
- env_step=20014 replay_size=20014 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A1_wider_recovery\checkpoints\checkpoint_00020014.pt
- env_step=25000 replay_size=25000 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A1_wider_recovery\checkpoints\checkpoint_00025000.pt
- env_step=30008 replay_size=30008 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A1_wider_recovery\checkpoints\checkpoint_00030008.pt
- env_step=35006 replay_size=35006 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A1_wider_recovery\checkpoints\checkpoint_00035006.pt
- env_step=35028 replay_size=35028 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A1_wider_recovery\checkpoints\checkpoint_00035028_final.pt

## Eval History
- env_step=5006 mean_progress=0.0 median_progress=0.0 mean_progress_m=0.0 mean_final_arc_length_m=0.0 mean_progress_fraction=0.0 mean_ghost_delta_ms=0.0 completion_rate=0.0 dcs=0.0
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A1_wider_recovery\checkpoints\checkpoint_00005006.pt checkpoint_env_step=5006 checkpoint_learner_step=3680
-   stochastic: mean_progress=12.65 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A1_wider_recovery_step_00005006_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=12.65 completion_rate_delta=0.0 dcs=0.0
- env_step=10004 mean_progress=0.0 median_progress=0.0 mean_progress_m=0.0 mean_final_arc_length_m=0.0 mean_progress_fraction=0.0 mean_ghost_delta_ms=0.0 completion_rate=0.0 dcs=0.0
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A1_wider_recovery\checkpoints\checkpoint_00010004.pt checkpoint_env_step=10004 checkpoint_learner_step=10712
-   stochastic: mean_progress=331.95 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A1_wider_recovery_step_00010004_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=331.95 completion_rate_delta=0.0 dcs=0.0
- env_step=15005 mean_progress=1028.9 median_progress=1367.0 mean_progress_m=514.45 mean_final_arc_length_m=514.45 mean_progress_fraction=0.2554008841757309 mean_ghost_delta_ms=-10517.018848836993 completion_rate=0.0 dcs=3.8731413514022215
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A1_wider_recovery\checkpoints\checkpoint_00015005.pt checkpoint_env_step=15005 checkpoint_learner_step=23392
-   stochastic: mean_progress=265.65 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A1_wider_recovery_step_00015005_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=False progress_delta=-763.2500000000001 completion_rate_delta=0.0 dcs=3.8731413514022215
- env_step=20014 mean_progress=219.05 median_progress=219.0 mean_progress_m=109.525 mean_final_arc_length_m=109.525 mean_progress_fraction=0.05437415072280479 mean_ghost_delta_ms=-4079.5284925184415 completion_rate=0.0 dcs=0.852003111629716
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A1_wider_recovery\checkpoints\checkpoint_00020014.pt checkpoint_env_step=20014 checkpoint_learner_step=38744
-   stochastic: mean_progress=257.1 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A1_wider_recovery_step_00020014_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=38.05000000000001 completion_rate_delta=0.0 dcs=0.852003111629716
- env_step=25000 mean_progress=196.0 median_progress=196.0 mean_progress_m=98.0 mean_final_arc_length_m=98.0 mean_progress_fraction=0.04865251559767058 mean_ghost_delta_ms=-3788.8878474486555 completion_rate=0.0 dcs=0.5287294308065822
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A1_wider_recovery\checkpoints\checkpoint_00025000.pt checkpoint_env_step=25000 checkpoint_learner_step=47760
-   stochastic: mean_progress=370.7 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A1_wider_recovery_step_00025000_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=174.7 completion_rate_delta=0.0 dcs=0.5287294308065822
- env_step=30008 mean_progress=197.0 median_progress=197.0 mean_progress_m=98.5 mean_final_arc_length_m=98.5 mean_progress_fraction=0.04890074271806686 mean_ghost_delta_ms=-3800.9675050949236 completion_rate=0.0 dcs=0.5979663074821673
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A1_wider_recovery\checkpoints\checkpoint_00030008.pt checkpoint_env_step=30008 checkpoint_learner_step=59384
-   stochastic: mean_progress=329.45 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A1_wider_recovery_step_00030008_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=132.45 completion_rate_delta=0.0 dcs=0.5979663074821673
- env_step=35006 mean_progress=486.15 median_progress=486.0 mean_progress_m=243.075 mean_final_arc_length_m=243.075 mean_progress_fraction=0.12067561458065076 mean_ghost_delta_ms=-6593.550349916516 completion_rate=0.0 dcs=0.9630546751188589
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A1_wider_recovery\checkpoints\checkpoint_00035006.pt checkpoint_env_step=35006 checkpoint_learner_step=70288
-   stochastic: mean_progress=504.8 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A1_wider_recovery_step_00035006_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=18.650000000000034 completion_rate_delta=0.0 dcs=0.9630546751188589
- env_step=35028 mean_progress=197.65 median_progress=198.0 mean_progress_m=98.825 mean_final_arc_length_m=98.825 mean_progress_fraction=0.04906209034632444 mean_ghost_delta_ms=-3808.8144210126948 completion_rate=0.0 dcs=0.8052556528824608
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A1_wider_recovery\checkpoints\checkpoint_00035028_final.pt checkpoint_env_step=35028 checkpoint_learner_step=71344
-   stochastic: mean_progress=245.45 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A1_wider_recovery_final_exact_step_00035028_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=47.79999999999998 completion_rate_delta=0.0 dcs=0.8052556528824608

## Diagnostics
- Bottleneck verdict: learner_backprop
- learner_backprop_seconds=3702.9702121608425
- worker_env_seconds=2006.7108101190533
- ipc_backpressure_seconds=0.10378809156827629
- actor_sync_seconds=122.07391100097448
- achieved_utd_1k=1.7636544190665342 cumulative_utd=2.0367705835331735 current_actor_staleness=1154
- time_to_first_ready_actor_seconds=86.31299999984913 time_to_first_applied_ready_actor_seconds=89.42199999978766 time_to_first_policy_control_window_seconds=92.51599999982864
- policy_control_fraction=0.9988235294117647 current_versions_behind=0 applied_lag_p50=0.5619999999180436 applied_lag_p95=0.8910000000614673
- positive_progress_mean=0.5275727522805684 nonpositive_progress_mean=0.4724272477194315 max_no_progress_p95=70.0 final_arc_length_mean=101.39759036144578 progress_fraction_mean=0.05033926374108625 ghost_delta_mean_ms=-3322.1406110108237
- corridor_violation_fraction_mean=0.00011446872908622529 corridor_distance_p95=20.676111246589475 max_corridor_distance_p95=20.889843644338107 corridor_nonrecovering_p95=0.0 corridor_truncations=0
- no_movement_episode_count=1 stall_episode_rate=0.03592814371257485 first_stall_delay_p95_ms=0.0
- actor_params=822308 critic_params=3288196 unique_critic_encoder_params=607776

## Event Logs
- learner: events=0 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A1_wider_recovery\learner_events.log
- worker: events=1048 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A1_wider_recovery\worker_events.log

## Videos
- No rollout videos were discovered for this run.

## Failure Notes
- termination_reason=max_wall_clock_minutes

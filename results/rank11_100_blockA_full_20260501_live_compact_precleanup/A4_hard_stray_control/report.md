# Training Report: rank11_100_blockA_full_20260501_live_A4_hard_stray_control

## Summary
- Observation mode: full
- Algorithm: redq
- Init mode: scratch
- Primary metric: mean_final_progress_index
- Env steps: 36742
- Learner steps: 71376
- Achieved UTD (1k window): 0.718562874251497
- Cumulative UTD: 1.9426269664144575
- Current actor staleness: 61
- Replay size: 36742
- Online replay size: 36742
- Offline replay size: 0
- Training duration (s): 5800.469000000041
- Exact final eval complete: True
- Final eval state: complete
- Ghost bundle: C:\Users\clewr\TrackManiaAI\data\ghosts\oqIJ5rQDRrNwLPTh9H2p_W4tLof\ghost_bundle_rank_011_100.json
- Canonical reference: source=reward_trajectory_fallback path=C:\Users\clewr\TrackManiaAI\data\reward\oqIJ5rQDRrNwLPTh9H2p_W4tLof\trajectory_0p5m.npz
- Strategy selection: status=manual_rank11_100_default family=rank11_100_bundle mixed_fallback=False
- Bundle resolution: mode=None selector=None resolved_rank=None resolved_name=None author_fallback_used=None

- Strategy family counts: {'intended_route': 0, 'shortcut_or_exploit': 100, 'unclassified': 0}

## Exact Final Eval
- mean_final_progress_index=229.85 mean_final_progress_meters=114.925 mean_final_arc_length_m=114.925 mean_progress_fraction=0.05705500362308461 mean_ghost_delta_ms=-4171.2506446763955 completion_rate=0.0
- provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A4_hard_stray_control\checkpoints\checkpoint_00036720_final.pt checkpoint_env_step=36720 checkpoint_learner_step=71376

## Checkpoints
- env_step=5003 replay_size=5003 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A4_hard_stray_control\checkpoints\checkpoint_00005003.pt
- env_step=10000 replay_size=10000 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A4_hard_stray_control\checkpoints\checkpoint_00010000.pt
- env_step=15005 replay_size=15005 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A4_hard_stray_control\checkpoints\checkpoint_00015005.pt
- env_step=20012 replay_size=20012 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A4_hard_stray_control\checkpoints\checkpoint_00020012.pt
- env_step=25004 replay_size=25004 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A4_hard_stray_control\checkpoints\checkpoint_00025004.pt
- env_step=30010 replay_size=30010 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A4_hard_stray_control\checkpoints\checkpoint_00030010.pt
- env_step=35002 replay_size=35002 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A4_hard_stray_control\checkpoints\checkpoint_00035002.pt
- env_step=36720 replay_size=36720 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A4_hard_stray_control\checkpoints\checkpoint_00036720_final.pt

## Eval History
- env_step=5003 mean_progress=0.0 median_progress=0.0 mean_progress_m=0.0 mean_final_arc_length_m=0.0 mean_progress_fraction=0.0 mean_ghost_delta_ms=0.0 completion_rate=0.0 dcs=0.0
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A4_hard_stray_control\checkpoints\checkpoint_00005003.pt checkpoint_env_step=5003 checkpoint_learner_step=3720
-   stochastic: mean_progress=3.9 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A4_hard_stray_control_step_00005003_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=3.9 completion_rate_delta=0.0 dcs=0.0
- env_step=10000 mean_progress=0.0 median_progress=0.0 mean_progress_m=0.0 mean_final_arc_length_m=0.0 mean_progress_fraction=0.0 mean_ghost_delta_ms=0.0 completion_rate=0.0 dcs=0.0
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A4_hard_stray_control\checkpoints\checkpoint_00010000.pt checkpoint_env_step=10000 checkpoint_learner_step=10808
-   stochastic: mean_progress=141.65 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A4_hard_stray_control_step_00010000_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=141.65 completion_rate_delta=0.0 dcs=0.0
- env_step=15005 mean_progress=206.0 median_progress=220.0 mean_progress_m=103.0 mean_final_arc_length_m=103.0 mean_progress_fraction=0.05113478680163336 mean_ghost_delta_ms=-3927.666535939437 completion_rate=0.0 dcs=1.0051232007806783
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A4_hard_stray_control\checkpoints\checkpoint_00015005.pt checkpoint_env_step=15005 checkpoint_learner_step=21008
-   stochastic: mean_progress=204.95 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A4_hard_stray_control_step_00015005_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=False progress_delta=-1.0500000000000114 completion_rate_delta=0.0 dcs=1.0051232007806783
- env_step=20012 mean_progress=242.2 median_progress=212.0 mean_progress_m=121.1 mean_final_arc_length_m=121.1 mean_progress_fraction=0.06012060855997864 mean_ghost_delta_ms=-4314.071594729327 completion_rate=0.0 dcs=1.0049792531120332
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A4_hard_stray_control\checkpoints\checkpoint_00020012.pt checkpoint_env_step=20012 checkpoint_learner_step=33384
-   stochastic: mean_progress=241.0 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A4_hard_stray_control_step_00020012_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=False progress_delta=-1.1999999999999886 completion_rate_delta=0.0 dcs=1.0049792531120332
- env_step=25004 mean_progress=336.6 median_progress=336.5 mean_progress_m=168.3 mean_final_arc_length_m=168.3 mean_progress_fraction=0.08355324872538733 mean_ghost_delta_ms=-5339.54293896545 completion_rate=0.0 dcs=1.3092182030338388
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A4_hard_stray_control\checkpoints\checkpoint_00025004.pt checkpoint_env_step=25004 checkpoint_learner_step=42104
-   stochastic: mean_progress=257.1 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A4_hard_stray_control_step_00025004_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=False progress_delta=-79.5 completion_rate_delta=0.0 dcs=1.3092182030338388
- env_step=30010 mean_progress=216.05 median_progress=210.5 mean_progress_m=108.025 mean_final_arc_length_m=108.025 mean_progress_fraction=0.05362946936161596 mean_ghost_delta_ms=-4041.555748399123 completion_rate=0.0 dcs=0.9488361879666228
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A4_hard_stray_control\checkpoints\checkpoint_00030010.pt checkpoint_env_step=30010 checkpoint_learner_step=51648
-   stochastic: mean_progress=227.7 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A4_hard_stray_control_step_00030010_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=11.649999999999977 completion_rate_delta=0.0 dcs=0.9488361879666228
- env_step=35002 mean_progress=658.65 median_progress=497.0 mean_progress_m=329.325 mean_final_arc_length_m=329.325 mean_progress_fraction=0.1634947928490088 mean_ghost_delta_ms=-7854.267591741098 completion_rate=0.0 dcs=1.6563560920407394
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A4_hard_stray_control\checkpoints\checkpoint_00035002.pt checkpoint_env_step=35002 checkpoint_learner_step=59864
-   stochastic: mean_progress=397.65 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A4_hard_stray_control_step_00035002_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=False progress_delta=-261.0 completion_rate_delta=0.0 dcs=1.6563560920407394
- env_step=36720 mean_progress=229.85 median_progress=201.0 mean_progress_m=114.925 mean_final_arc_length_m=114.925 mean_progress_fraction=0.05705500362308461 mean_ghost_delta_ms=-4171.2506446763955 completion_rate=0.0 dcs=0.8969756097560976
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A4_hard_stray_control\checkpoints\checkpoint_00036720_final.pt checkpoint_env_step=36720 checkpoint_learner_step=71376
-   stochastic: mean_progress=256.25 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A4_hard_stray_control_final_exact_step_00036720_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=26.400000000000006 completion_rate_delta=0.0 dcs=0.8969756097560976

## Diagnostics
- Bottleneck verdict: learner_backprop
- learner_backprop_seconds=3676.338793711271
- worker_env_seconds=2228.0718714974355
- ipc_backpressure_seconds=0.11471820157021284
- actor_sync_seconds=120.9853661803063
- achieved_utd_1k=0.718562874251497 cumulative_utd=1.9426269664144575 current_actor_staleness=61
- time_to_first_ready_actor_seconds=89.5 time_to_first_applied_ready_actor_seconds=92.17200000002049 time_to_first_policy_control_window_seconds=95.29700000002049
- policy_control_fraction=0.9990756302521009 current_versions_behind=0 applied_lag_p50=0.5779999999795109 applied_lag_p95=0.9092499999096614
- positive_progress_mean=0.5411165323783557 nonpositive_progress_mean=0.45888346762164445 max_no_progress_p95=70.0 final_arc_length_mean=86.86497890295358 progress_fraction_mean=0.04312448715272729 ghost_delta_mean_ms=-3139.4691492518573
- corridor_violation_fraction_mean=0.00717629735387887 corridor_distance_p95=16.971903243035328 max_corridor_distance_p95=17.172651232813912 corridor_nonrecovering_p95=0.0 corridor_truncations=1
- no_movement_episode_count=1 stall_episode_rate=0.012605042016806723 first_stall_delay_p95_ms=0.0
- actor_params=822308 critic_params=3288196 unique_critic_encoder_params=607776

## Event Logs
- learner: events=0 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A4_hard_stray_control\learner_events.log
- worker: events=1350 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A4_hard_stray_control\worker_events.log

## Videos
- No rollout videos were discovered for this run.

## Failure Notes
- termination_reason=max_wall_clock_minutes

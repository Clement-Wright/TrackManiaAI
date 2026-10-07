# Training Report: rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor

## Summary
- Observation mode: full
- Algorithm: redq
- Init mode: scratch
- Primary metric: mean_final_progress_index
- Env steps: 40025
- Learner steps: 71376
- Achieved UTD (1k window): 3.2111553784860556
- Cumulative UTD: 1.7832854465958776
- Current actor staleness: 2661
- Replay size: 40025
- Online replay size: 40025
- Offline replay size: 0
- Training duration (s): 7093.78199999989
- Exact final eval complete: True
- Final eval state: complete
- Ghost bundle: C:\Users\clewr\TrackManiaAI\data\ghosts\oqIJ5rQDRrNwLPTh9H2p_W4tLof\ghost_bundle_rank_011_100.json
- Canonical reference: source=reward_trajectory_fallback path=C:\Users\clewr\TrackManiaAI\data\reward\oqIJ5rQDRrNwLPTh9H2p_W4tLof\trajectory_0p5m.npz
- Strategy selection: status=manual_rank11_100_default family=rank11_100_bundle mixed_fallback=False
- Bundle resolution: mode=None selector=None resolved_rank=None resolved_name=None author_fallback_used=None

- Strategy family counts: {'intended_route': 0, 'shortcut_or_exploit': 100, 'unclassified': 0}

## Exact Final Eval
- mean_final_progress_index=451.95 mean_final_progress_meters=225.975 mean_final_arc_length_m=225.975 mean_progress_fraction=0.11218624706309806 mean_ghost_delta_ms=-6322.010852419435 completion_rate=0.0
- provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor\checkpoints\checkpoint_00040025_final.pt checkpoint_env_step=40025 checkpoint_learner_step=71376

## Checkpoints
- env_step=5006 replay_size=5006 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor\checkpoints\checkpoint_00005006.pt
- env_step=10003 replay_size=10003 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor\checkpoints\checkpoint_00010003.pt
- env_step=15005 replay_size=15005 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor\checkpoints\checkpoint_00015005.pt
- env_step=20011 replay_size=20011 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor\checkpoints\checkpoint_00020011.pt
- env_step=25005 replay_size=25005 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor\checkpoints\checkpoint_00025005.pt
- env_step=30002 replay_size=30002 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor\checkpoints\checkpoint_00030002.pt
- env_step=35000 replay_size=35000 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor\checkpoints\checkpoint_00035000.pt
- env_step=40000 replay_size=40000 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor\checkpoints\checkpoint_00040000.pt
- env_step=40025 replay_size=40025 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor\checkpoints\checkpoint_00040025_final.pt

## Eval History
- env_step=5006 mean_progress=0.0 median_progress=0.0 mean_progress_m=0.0 mean_final_arc_length_m=0.0 mean_progress_fraction=0.0 mean_ghost_delta_ms=0.0 completion_rate=0.0 dcs=0.0
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor\checkpoints\checkpoint_00005006.pt checkpoint_env_step=5006 checkpoint_learner_step=3776
-   stochastic: mean_progress=3.3 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor_step_00005006_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=3.3 completion_rate_delta=0.0 dcs=0.0
- env_step=10003 mean_progress=0.0 median_progress=0.0 mean_progress_m=0.0 mean_final_arc_length_m=0.0 mean_progress_fraction=0.0 mean_ghost_delta_ms=0.0 completion_rate=0.0 dcs=0.0
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor\checkpoints\checkpoint_00010003.pt checkpoint_env_step=10003 checkpoint_learner_step=11240
-   stochastic: mean_progress=42.7 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor_step_00010003_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=42.7 completion_rate_delta=0.0 dcs=0.0
- env_step=15005 mean_progress=0.0 median_progress=0.0 mean_progress_m=0.0 mean_final_arc_length_m=0.0 mean_progress_fraction=0.0 mean_ghost_delta_ms=0.0 completion_rate=0.0 dcs=0.0
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor\checkpoints\checkpoint_00015005.pt checkpoint_env_step=15005 checkpoint_learner_step=18848
-   stochastic: mean_progress=249.35 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor_step_00015005_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=249.35 completion_rate_delta=0.0 dcs=0.0
- env_step=20011 mean_progress=198.0 median_progress=198.0 mean_progress_m=99.0 mean_final_arc_length_m=99.0 mean_progress_fraction=0.04914896983846313 mean_ghost_delta_ms=-3813.0396834299563 completion_rate=0.0 dcs=0.6615436017373872
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor\checkpoints\checkpoint_00020011.pt checkpoint_env_step=20011 checkpoint_learner_step=29832
-   stochastic: mean_progress=299.3 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor_step_00020011_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=101.30000000000001 completion_rate_delta=0.0 dcs=0.6615436017373872
- env_step=25005 mean_progress=181.95 median_progress=181.0 mean_progress_m=90.975 mean_final_arc_length_m=90.975 mean_progress_fraction=0.04516492455610287 mean_ghost_delta_ms=-3625.6392811298733 completion_rate=0.0 dcs=0.92926455566905
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor\checkpoints\checkpoint_00025005.pt checkpoint_env_step=25005 checkpoint_learner_step=38616
-   stochastic: mean_progress=195.8 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor_step_00025005_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=13.850000000000023 completion_rate_delta=0.0 dcs=0.92926455566905
- env_step=30002 mean_progress=132.0 median_progress=131.5 mean_progress_m=66.0 mean_final_arc_length_m=66.0 mean_progress_fraction=0.032765979892308755 mean_ghost_delta_ms=-3025.7377866391744 completion_rate=0.0 dcs=0.9295774647887324
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor\checkpoints\checkpoint_00030002.pt checkpoint_env_step=30002 checkpoint_learner_step=46712
-   stochastic: mean_progress=142.0 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor_step_00030002_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=10.0 completion_rate_delta=0.0 dcs=0.9295774647887324
- env_step=35000 mean_progress=485.85 median_progress=505.0 mean_progress_m=242.925 mean_final_arc_length_m=242.925 mean_progress_fraction=0.12060114644453188 mean_ghost_delta_ms=-6556.192240254453 completion_rate=0.0 dcs=1.917702782711664
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor\checkpoints\checkpoint_00035000.pt checkpoint_env_step=35000 checkpoint_learner_step=54704
-   stochastic: mean_progress=253.35 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor_step_00035000_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=False progress_delta=-232.50000000000003 completion_rate_delta=0.0 dcs=1.917702782711664
- env_step=40000 mean_progress=506.95 median_progress=512.5 mean_progress_m=253.475 mean_final_arc_length_m=253.475 mean_progress_fraction=0.12583873868489337 mean_ghost_delta_ms=-6753.8494055712235 completion_rate=0.0 dcs=0.6465374314500701
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor\checkpoints\checkpoint_00040000.pt checkpoint_env_step=40000 checkpoint_learner_step=68832
-   stochastic: mean_progress=784.1 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor_step_00040000_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=277.15000000000003 completion_rate_delta=0.0 dcs=0.6465374314500701
- env_step=40025 mean_progress=451.95 median_progress=452.0 mean_progress_m=225.975 mean_final_arc_length_m=225.975 mean_progress_fraction=0.11218624706309806 mean_ghost_delta_ms=-6322.010852419435 completion_rate=0.0 dcs=0.7054553968625614
-   progress_semantics=fixed_spacing_meters spacing_m=0.5 reference_total_arc_m=2014.284334450573
-   provenance: mode=checkpoint_authoritative checkpoint=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor\checkpoints\checkpoint_00040025_final.pt checkpoint_env_step=40025 checkpoint_learner_step=71376
-   stochastic: mean_progress=640.65 completion_rate=0.0 summary_path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\eval\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor_final_exact_step_00040025_stochastic\summary.json
-   deterministic_collapse: meaningfully_outperformed=True progress_delta=188.7 completion_rate_delta=0.0 dcs=0.7054553968625614

## Diagnostics
- Bottleneck verdict: learner_backprop
- learner_backprop_seconds=3653.426911176881
- worker_env_seconds=2395.2703302844893
- ipc_backpressure_seconds=0.12375811068341136
- actor_sync_seconds=120.81204689270817
- achieved_utd_1k=3.2111553784860556 cumulative_utd=1.7832854465958776 current_actor_staleness=2661
- time_to_first_ready_actor_seconds=88.79699999978766 time_to_first_applied_ready_actor_seconds=91.96899999980815 time_to_first_policy_control_window_seconds=94.60999999986961
- policy_control_fraction=0.998948717948718 current_versions_behind=0 applied_lag_p50=0.5779999999795109 applied_lag_p95=0.9339999999385324
- positive_progress_mean=0.44686427975165927 nonpositive_progress_mean=0.5531357202483408 max_no_progress_p95=70.0 final_arc_length_mean=78.56995884773663 progress_fraction_mean=0.039006389268855524 ghost_delta_mean_ms=-2674.1549165017605
- corridor_violation_fraction_mean=1.1447082997075269e-05 corridor_distance_p95=16.775601318885602 max_corridor_distance_p95=20.141792426635806 corridor_nonrecovering_p95=0.0 corridor_truncations=0
- no_movement_episode_count=2 stall_episode_rate=0.02459016393442623 first_stall_delay_p95_ms=0.0
- actor_params=822308 critic_params=3288196 unique_critic_encoder_params=607776

## Event Logs
- learner: events=0 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor\learner_events.log
- worker: events=1415 path=C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor\worker_events.log

## Videos
- No rollout videos were discovered for this run.

## Failure Notes
- termination_reason=max_wall_clock_minutes

# Reference Target Suite

## 2026-04-22 - Reference Target Suite

- suite_name=reference_target_suite_20260422
- map_uid=oqIJ5rQDRrNwLPTh9H2p_W4tLof
- wall_clock_minutes_per_leg=120.0
- isfoo_rank11: best_det_progress=None best_det_m=None best_det_fraction=None best_det_ghost_delta_ms=None bundle_mode=None
- rank11_100_bundle: best_det_progress=None best_det_m=None best_det_fraction=None best_det_ghost_delta_ms=None bundle_mode=None
- author_account_proxy: best_det_progress=None best_det_m=None best_det_fraction=None best_det_ghost_delta_ms=None bundle_mode=None
- reward_trajectory: best_det_progress=None best_det_m=None best_det_fraction=None best_det_ghost_delta_ms=None bundle_mode=None
- assumption=Using the map author's fetched top-100 replay as the author leg proxy.

Artifacts:
- suite_report: C:\Users\clewr\TrackManiaAI\.tmp\live_reference_target_suite_20260422\suite_report.md
- isfoo_rank11_summary: C:\Users\clewr\TrackManiaAI\.tmp\live_reference_target_suite_20260422\artifacts\train\reference_suite_20260422_isfoo_rank11\summary.json
- rank11_100_bundle_summary: C:\Users\clewr\TrackManiaAI\.tmp\live_reference_target_suite_20260422\artifacts\train\reference_suite_20260422_rank11_100\summary.json
- author_account_proxy_summary: C:\Users\clewr\TrackManiaAI\.tmp\live_reference_target_suite_20260422\artifacts\train\reference_suite_20260422_author_account_proxy\summary.json
- reward_trajectory_summary: C:\Users\clewr\TrackManiaAI\.tmp\live_reference_target_suite_20260422\artifacts\train\reference_suite_20260422_reward_trajectory\summary.json

## 2026-04-22 - Findings

- Winner on recorded deterministic `mean_final_progress_index`: `rank11_100_bundle` with `1694.0` at the latest completed eval checkpoint.
- Runner-up: `author_account_proxy` with `1490.0`.
- Third: `isfoo_rank11` with `1062.6`.
- Fourth: `reward_trajectory` with `320.0` in the run summary, although this leg showed a higher intermediate deterministic eval peak of `1186.0` at env step `60013` before regressing.

Recorded latest completed eval metrics from each run summary:

- `isfoo_rank11`
- `mean_final_progress_index=1062.6`
- `mean_final_progress_meters=531.3`
- `progress_fraction_of_reference=0.2736272094575714`
- `ghost_relative_time_delta_ms=-11060.120539661943`

- `rank11_100_bundle`
- `mean_final_progress_index=1694.0`
- `mean_final_progress_meters=847.0`
- `progress_fraction_of_reference=0.42049674195129566`
- `ghost_relative_time_delta_ms=-16007.447715636581`

- `author_account_proxy`
- `mean_final_progress_index=1490.0`
- `mean_final_progress_meters=745.0`
- `progress_fraction_of_reference=0.37569377169157264`
- `ghost_relative_time_delta_ms=-15563.663699385143`

- `reward_trajectory`
- `mean_final_progress_index=320.0`
- `mean_final_progress_meters=160.0`
- `progress_fraction_of_reference=0.043623607605695724`
- `ghost_relative_time_delta_ms` not available for this legacy reward mode

Interpretation:

- The broad `rank11_100_bundle` target was the strongest reference strategy in this suite and beat the single `isfoo` rank-11 target by a large margin.
- The author-leg proxy was also strong, but it remained below the `rank11_100_bundle` result.
- The single selected `isfoo` run was clearly viable, but it underperformed the larger post-rank-10 bundle.
- The legacy reward trajectory baseline was weakest overall on the recorded final metric, even though it produced one notable mid-run spike.

Important caveat:

- All four legs finished with the same known shutdown quirk: the final exact-checkpoint dual-mode eval was still marked in-flight in the summary when the training process finalized. The values above are therefore the latest completed scheduled eval metrics recorded in the run summaries, not confirmed final-exact eval results.
- The author leg used the fetched top-100 replay from the map author's account as a proxy. A distinct author-medal ghost was not materialized locally for this suite.


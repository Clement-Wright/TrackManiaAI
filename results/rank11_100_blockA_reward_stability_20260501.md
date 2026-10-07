# rank11_100 Block A Reward Stability Results


## 2026-05-01 Block A Full Campaign Closeout

- Session: `rank11_100_blockA_full_20260501_live`.
- Target: `rank11_100_bundle` (`data/ghosts/oqIJ5rQDRrNwLPTh9H2p_W4tLof/ghost_bundle_rank_011_100.json`).
- Notes: A4 was manually restarted after the campaign runner correctly refused to start it at `178.59 GB > 150 GB`; A0-A2 bulk artifacts were pruned after compact evidence was copied to `results/rank11_100_blockA_full_20260501_live_compact_precleanup`.
- Validity rule: ranking uses exact final checkpoint-backed deterministic `mean_final_progress_index`; missing exact final dual-mode evals are invalid.

| Rank | Leg | Valid | Det mean progress | Det meters | Det fraction | Det ghost delta ms | Stoch mean progress | DCS | Corridor trunc rate | Final checkpoint |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `A3_buffered_hard_boundary` | yes | 1612.8 | 806.4 | 0.40034069977511794 | -14717.18280665465 | 1436.0 | 1.1231197771587744 | 0.0 | `C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary\checkpoints\checkpoint_00039722_final.pt` |
| 2 | `A2_softer_wider_corridor` | yes | 451.95 | 225.975 | 0.11218624706309806 | -6322.010852419435 | 640.65 | 0.7054553968625614 | 0.0 | `C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A2_softer_wider_corridor\checkpoints\checkpoint_00040025_final.pt` |
| 3 | `A0_baseline_current` | yes | 358.35 | 179.175 | 0.08895218859400639 | -5534.549236701716 | 477.05 | 0.7511791216853579 | 0.0 | `C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A0_baseline_current\checkpoints\checkpoint_00038997_final.pt` |
| 4 | `A4_hard_stray_control` | yes | 229.85 | 114.925 | 0.05705500362308461 | -4171.2506446763955 | 256.25 | 0.8969756097560976 | 0.05 | `C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A4_hard_stray_control\checkpoints\checkpoint_00036720_final.pt` |
| 5 | `A1_wider_recovery` | yes | 197.65 | 98.825 | 0.04906209034632444 | -3808.8144210126948 | 245.45 | 0.8052556528824608 | 0.0 | `C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A1_wider_recovery\checkpoints\checkpoint_00035028_final.pt` |

### Winner
- Winner: `A3_buffered_hard_boundary` (Buffered hard boundary) with exact deterministic mean progress `1612.8` and progress fraction `0.40034069977511794`.
- Winning checkpoint: `C:\Users\clewr\TrackManiaAI\.tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary\checkpoints\checkpoint_00039722_final.pt`.
- No runner-up was within 5% of the winner; by retention policy only the A3 bulk run needs to remain as the Block A keeper.

### Interpretation
- The buffered hard-boundary setting won decisively. The strongest signal is that increasing hard-violation patience to 60 while keeping the 25 m soft / 90 m hard corridor allowed much longer exact deterministic rollouts than both the wider soft corridor and the hard-stray control.
- The hard-stray control is a useful negative control: it recovered a scheduled spike at 35k, but exact final deterministic progress collapsed to 229.85, reinforcing that harsh corridor termination is brittle for this map.
- A3 deterministic outperformed A3 stochastic at exact final eval (`DCS > 1`), which is unusual relative to earlier runs and makes A3 the right checkpoint for Phase 4 extraction sweeps.
- Phase 3 should promote A3 corridor settings as the current `tmrl-test` mainline reward profile. Phase 4 should run deterministic/clipped/stochastic temperature sweeps on the A3 final checkpoint. Phase 5 offline warm-start should wait until that extraction decision is made, and should use the same rank11_100 target family only.

### Retention
- Compact evidence for all legs is stored under `results/rank11_100_blockA_full_20260501_live_compact_precleanup`.
- A0-A2 bulk artifacts were already pruned to unblock A4. A4 bulk artifacts are safe to prune after this entry because A4 is not a winner or near runner-up.

### 2026-05-01 Implementation Follow-Up
- Promoted `A3_buffered_hard_boundary` into `configs/full_redq_top100_tmrl_test.yaml` as the current `tmrl-test` reward default.
- Added a Block B/C continuation path so deterministic extraction and offline leverage can start from the validated A3 winner without rerunning Block A.
- Added Phase 4 target checks so policy-mode sweeps fail if the config does not point at `rank11_100_bundle` with `selected_count=90`.
- Added a configurable deterministic deployment extraction mode; default remains `deterministic_mean` until a Phase 4 sweep proves `clipped_mean` is better.
- Kept Phase 5 fail-closed on missing offline transition sidecars. The rank11 bundle still needs validated observation/action transition data before real offline pretraining can run.

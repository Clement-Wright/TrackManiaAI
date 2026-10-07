# Phase 4-5 Plan: A3 Extraction And Offline Leverage

Date: 2026-05-01

## Block A Result

The `rank11_100_bundle` reward-stability campaign selected `A3_buffered_hard_boundary` as the current `tmrl-test` mainline reward profile. The exact final deterministic score was `1612.8` fixed-spacing progress points, or `806.4 m`, with progress fraction `0.40034069977511794`. This was far ahead of the next best valid run, so no 5 percent tie-break was needed.

The key reward decision is not "wider is always better." `A2` widened the corridor but underperformed. `A3` kept the 25 m soft / 90 m hard geometry and improved recovery by extending hard-violation patience to 60 steps with easier recovery thresholds. That suggests the useful change was buffering recoverable excursions, not weakening the target manifold.

## Current Mainline Changes

- `configs/full_redq_top100_tmrl_test.yaml` now uses the A3 reward profile for future `tmrl-test` work.
- Phase 4 policy-mode sweeps can enforce that the config points at `ghost_bundle_rank_011_100.json`, that the manifest has `selected_training_family=rank11_100_bundle`, and that `selected_count=90`.
- The rank11 campaign runner can now start Block B/C from an existing validated Block A winner via `--winner-run-dir` and `--winner-config`.
- Deterministic eval deployment is configurable through `eval.deployment_extraction_mode`, defaulting to `deterministic_mean`; Phase 4 may later select `clipped_mean`.
- Phase 5 online fine-tuning now loads offline initialization in `weights_only` mode by default, so online runs do not accidentally inherit offline optimizer momentum.

## Phase 4: Deterministic Extraction

### Objective

Run extraction sweeps only on the A3 winning checkpoint and choose a deployable deterministic extraction rule.

### Inputs

- Target bundle: `data/ghosts/oqIJ5rQDRrNwLPTh9H2p_W4tLof/ghost_bundle_rank_011_100.json`
- Winner run: `rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary`
- Winner checkpoint: `checkpoint_00039722_final.pt`
- Base config: `configs/full_redq_top100_tmrl_test.yaml`

### Command Shape

```powershell
.\.venv\Scripts\python.exe scripts\run_rank11_100_validation_campaign.py `
  --config configs\full_redq_top100_tmrl_test.yaml `
  --session-name rank11_100_phase45_20260501 `
  --blocks B `
  --winner-run-dir .tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\artifacts\train\rank11_100_blockA_full_20260501_live_A3_buffered_hard_boundary `
  --winner-config .tmp\live_rank11_100_validation_campaign\rank11_100_blockA_full_20260501_live\configs\A3_buffered_hard_boundary.yaml
```

### Decision Rule

- Compare `deterministic_mean` and `clipped_mean`.
- Choose `clipped_mean` only if it improves deterministic progress by at least 5 percent and does not reduce progress fraction.
- Keep `best_of_k` diagnostic-only.
- If no deployable deterministic mode reaches `DCS >= 0.85`, queue deterministic-student distillation before more scale-up.

## Phase 5: Offline-To-Online Leverage

### Objective

Use validated rank11 ghost transitions as a launch rail, then release the policy into online REDQ. Offline data must remain selected-family only.

### Current Blocker

The rank11 bundle has valid route-family metadata, but real offline warm-start still depends on observation/action transition sidecars. If `offline_transition_npz_path` is missing or `offline_transition_count <= 0`, Block C must hard-stop for live execution. Dry runs may warn, but real pretraining must not proceed.

### Implementation Path

1. Generate or verify observation/action sidecars for `rank11_100_bundle`.
2. Run `scripts/pretrain_ghost_redq.py --strategy bc_redq_awac` with required family/count checks.
3. Run `C1_weight_init_only` with `--offline-init-mode weights_only`, no replay seeding, and balanced replay disabled.
4. Run `C2_full_offline_to_online` with the same weight initialization, ghost replay seeding, and balanced replay decay `0.75 -> 0.10` over `50000` env steps.
5. Compare both against the A3 online baseline using exact final deterministic progress plus 25k/50k early lift.

### Success Criteria

- Offline checkpoints record strategy, dataset hash, transition count, selected family, and bundle path.
- Online summaries record offline init mode and offline pretrain metadata.
- At least one offline run improves early progress at 25k or 50k env steps.
- At least one offline run beats the A3 exact final deterministic baseline before the offline path becomes the default.

## Compute And Storage Notes

Do not launch Phase 4/5 live runs without the Phase 0 storage preflight. Phase 4 eval sweeps are fixed-episode jobs and should be substantially smaller than 90-minute training legs. Phase 5 contains two more 90-minute legs and should be treated as a quota-gated campaign.

Large traces or videos should remain off by default. Keep compact JSON summaries, selected deployment policy JSON, the offline pretrain checkpoint, and the best C1/C2 final checkpoint only.

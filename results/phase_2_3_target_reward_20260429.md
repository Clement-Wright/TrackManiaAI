# Phase 2/3 Implementation Note: Target-Family Hard Stop And Rank11 Reward Campaign

Date: 2026-04-29

## Summary

This pass implemented the Phase 2 target-family hard stop and tightened the Phase 3 `rank11_100_bundle` reward campaign gate. The code now rejects the old `mixed_with_warning` ambiguous-family policy, shipped top-100 REDQ configs use `hard_stop`, and the rank11 campaign preflight verifies that the selected bundle is the 90-trajectory `rank11_100_bundle` and not a mixed fallback.

No training data from a completed live reward leg was produced in this pass. The attempted live campaign stopped correctly at the window gate because no visible Trackmania client window was discoverable by the capture/window API.

## Implemented

- `ghosts.ambiguous_family_policy` now defaults to `hard_stop`.
- `mixed_with_warning` is now rejected as invalid configuration.
- `configs/full_redq_top100.yaml` and `configs/full_redq_top100_tmrl_test.yaml` now ship `ambiguous_family_policy: "hard_stop"`.
- `scripts/run_rank11_100_validation_campaign.py` now preflights that the configured bundle has:
  - `selected_count == 90`
  - `selected_training_family == "rank11_100_bundle"`
  - `mixed_fallback == false`
  - base config `ghosts.ambiguous_family_policy == "hard_stop"`
- The campaign gate now runs both `force_window_size.py` and `check_environment.py --require-reward` before every live leg.
- The A0-A4 reward variant constants are covered by tests so the campaign cannot accidentally change target family or drift from the specified corridor variants.

## Verification

The full test suite passed after implementation:

```text
176 passed in 16.75s
```

The Block A dry-run also passed and showed all five planned 90-minute A0-A4 commands with reward verification before each leg.

## Live Run Attempt

The live command was launched for:

```text
rank11_100_phase23_20260429
```

It completed storage preflight and full pytest, then stopped at the first live gate:

```text
RuntimeError: Could not find a visible window containing 'Trackmania'.
```

Steam/Trackmania launch was requested and a `Trackmania` process existed, but the visible-window enumeration returned zero visible window titles. Because the full-observation capture path requires a visible Trackmania client rect, the campaign was not resumed. This is a safe failure: no invalid live leg was started and no reward comparison row should be ranked from this attempt.

## Next Action

Before retrying the A0-A4 live campaign, Trackmania must be visible in the same desktop/session where the capture scripts can enumerate windows. Once `scripts/force_window_size.py --config configs/full_redq_top100_tmrl_test.yaml` succeeds, resume the same campaign session or relaunch:

```powershell
.\.venv\Scripts\python.exe scripts\run_rank11_100_validation_campaign.py `
  --config configs\full_redq_top100_tmrl_test.yaml `
  --session-name rank11_100_phase23_20260429 `
  --wall-clock-minutes 90 `
  --blocks A `
  --min-free-gb 150 `
  --max-artifact-gb 150 `
  --results-file rank11_100_phase23_reward_stability_20260429.md
```

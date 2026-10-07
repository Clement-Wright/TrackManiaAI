# Phase 0/1 Implementation Note: Storage Hygiene And Exact Eval Closure

Date: 2026-04-29

## Summary

This pass implemented the first two near-term roadmap phases before any further long Trackmania campaigns:

- Phase 0: storage and run hygiene, so long live runs refuse unsafe disk states and cleanup is safer and more informative.
- Phase 1: exact eval closure, so a run or campaign leg is not considered valid unless exact final deterministic and stochastic checkpoint eval summaries exist on disk.

No live training run was launched as part of this pass.

## Phase 0: Storage And Run Hygiene

The artifact retention layer now has explicit disk preflight support. Long training entrypoints and campaign runners can enforce a minimum free-space threshold and a maximum artifact-root quota before launching expensive runs. The default policy is intentionally conservative: refuse long runs below 150 GB free space or when the artifact root exceeds 150 GB, unless explicitly overridden.

Cleanup also gained safer accounting and guardrails. Cleanup dry-runs now report estimated removed bytes, kept bytes, and artifact-root size before and after cleanup. The cleanup root is safety-checked so accidental deletion of protected project directories such as `src`, `scripts`, `tests`, `configs`, `docs`, `results`, or `data` is refused.

The cleanup CLI reports storage preflight state but does not block cleanup, because cleanup may be the mechanism needed to recover from low disk space or an oversized artifact directory.

## Phase 1: Exact Eval Closure

Exact final eval completion is now artifact-backed. The learner no longer treats an in-memory final eval entry as complete unless every configured eval mode has a summary path and each summary file exists on disk.

Campaign validation now requires both deterministic and stochastic exact-final summaries to exist. Missing mode summaries are reported with explicit failure reasons so incomplete runs can be excluded from ranking instead of silently falling back to stale scheduled evals.

Standalone final checkpoint eval backfill now clears missing-final-eval reasons only after successful dual-mode artifact production.

## Validation

Focused validation passed:

```text
34 passed in 15.32s
```

The passing focused suite covered artifact cleanup/preflight, campaign validation, REDQ learner finalization, worker/learner integration, and reporting surfaces. A compile check over `src` and `scripts` also passed.

## Remaining Follow-Up

The next live campaign should begin only after confirming the Trackmania window is resized, the live-env gate passes, and disk preflight is healthy. The first long-run validation target remains the `rank11_100_bundle` reward-stability campaign, but now incomplete exact-final evals should be visible and invalid for ranking rather than silently accepted.

# REDQ Ghost-Driven Offline-To-Online Notes

## 2026-04-20 Implementation And Test Pass

Implemented the first code-bearing version of the REDQ mainline ghost-data roadmap:

- Added checkpoint/eval diagnostics support for determinism conversion score, deterministic-vs-stochastic gaps, final checkpoint eval status, extraction-mode policy evaluation, and progress-threshold metadata.
- Added source-aware replay plus `BalancedReplayBuffer` so online REDQ can sample from separate offline and online buffers with a decaying offline fraction.
- Added ghost config blocks and artifacts for Nadeo top-100 ingestion, Openplanet-export trajectory normalization, diverse ghost bundle selection, ghost-bundle reward projection, offline transition seeding, and elite archive manifests.
- Added `train_full_redq.py --ghost-bundle` and `--offline-init-checkpoint` so ghost-pretrained REDQ weights and offline replay seeding can be used before live fine-tuning.
- Added `pretrain_ghost_redq.py` with BC, REDQ critic warm start, AWAC-style weighted actor updates, and optional CQL-style critic regularization over action-valid offline transitions.
- Added `evaluate_redq_policy_modes.py` for deterministic mean, clipped mean, stochastic temperature sweeps, and best-of-k stochastic diagnostics on one checkpoint.

Important boundary: raw `.gbx` parsing is not faked in Python. Downloaded `.gbx` files are stored reproducibly, and `extract_ghost_trajectories.py` expects Openplanet-exported JSON/JSONL/Parquet trajectory data. Actor/critic pretraining fails closed unless validated action labels and observation sidecars are present.

Validation:

- `python -m compileall -q src scripts`
- `pytest -q tests/test_replay.py tests/test_metrics.py tests/test_redq_stack.py tests/test_ghost_pipeline.py`
- `pytest -q tests/test_evaluator_and_data.py tests/test_reporting.py tests/test_worker_learner.py tests/test_redq_learner.py tests/test_research_and_cleanup.py`
- `pytest -q`

Final result: full suite passed, `107 passed`.

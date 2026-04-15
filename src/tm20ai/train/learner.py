from __future__ import annotations

import queue
import time
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from ..algos.redq import REDQActorUpdateResult, REDQCriticUpdateResult, REDQSACAgent
from ..algos.sac import SACAgent, SACUpdateResult
from ..capture import lidar_feature_dim
from ..config import TM20AIConfig, load_tm20ai_config
from ..data.dataset import seed_replay_from_demo_sidecars
from ..data.parquet_writer import build_run_artifact_paths, ensure_directory, timestamp_tag, write_json
from ..train.features import TELEMETRY_DIM
from .diagnostics import (
    ActorSyncTracker,
    EpisodeDiagnosticsTracker,
    JsonlEventLogger,
    MovementDiagnosticsTracker,
    QueueAccumulator,
    TimingAccumulator,
    build_agent_resource_profile,
    build_bottleneck_verdict,
)
from .metrics import TensorBoardScalarLogger
from .protocol import EvalResult
from .replay import ReplayBuffer


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit(cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:  # noqa: BLE001
        return None
    return completed.stdout.strip() or None


@dataclass(slots=True)
class TrainRunPaths:
    run_dir: Path
    checkpoints_dir: Path
    summary_json: Path
    tensorboard_dir: Path
    worker_sync_dir: Path
    desired_actor_path: Path
    worker_actor_status_path: Path


class SACLearner:
    def __init__(
        self,
        *,
        config_path: str | Path,
        run_name: str | None = None,
        max_env_steps: int | None = None,
        init_actor: str | Path | None = None,
        init_mode: str = "scratch",
        demo_root: str | Path | None = None,
        seed_demos: str | Path | None = None,
        eval_episodes_override: int | None = None,
        diagnostics_enabled: bool = False,
        detailed_cuda_timing: bool = False,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.config: TM20AIConfig = load_tm20ai_config(self.config_path)
        if self.config.train.algorithm != "sac":
            raise ValueError(f"Unsupported training algorithm: {self.config.train.algorithm!r}")
        if init_mode not in {"scratch", "actor_only", "actor_plus_critic_encoders"}:
            raise ValueError(f"Unsupported FULL SAC init_mode: {init_mode!r}")
        if init_mode == "scratch" and init_actor is not None:
            raise ValueError("init_actor requires init_mode to be actor_only or actor_plus_critic_encoders.")
        if init_mode != "scratch" and init_actor is None:
            raise ValueError("BC warm start requires --init-actor with init_mode actor_only or actor_plus_critic_encoders.")
        self.repo_root = self.config_path.parents[1]
        self.config_snapshot = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        run_prefix = f"{self.config.observation.mode}_sac"
        self.run_name = run_name or f"{run_prefix}_{timestamp_tag()}"
        self.max_env_steps = self.config.train.max_env_steps if max_env_steps is None else int(max_env_steps)
        self.eval_episodes = self.config.eval.episodes if eval_episodes_override is None else int(eval_episodes_override)
        self.init_mode = init_mode
        self.bc_checkpoint_path = None if init_actor is None else str(Path(init_actor).resolve())
        self.bc_checkpoint_metadata: dict[str, Any] | None = None
        self.demo_root = None if demo_root is None else str(Path(demo_root).resolve())
        self.replay_seeded = False
        self.run_start_timestamp = _utc_now()
        self.run_end_timestamp: str | None = None
        self._run_start_monotonic = time.monotonic()
        self.diagnostics_enabled = bool(diagnostics_enabled)
        self.detailed_cuda_timing = bool(detailed_cuda_timing)

        run_paths = build_run_artifact_paths(self.config, mode="train", run_name=self.run_name)
        worker_sync_dir = ensure_directory(run_paths.run_dir / "worker_sync")
        self.paths = TrainRunPaths(
            run_dir=run_paths.run_dir,
            checkpoints_dir=ensure_directory(run_paths.run_dir / "checkpoints"),
            summary_json=run_paths.summary_json,
            tensorboard_dir=run_paths.tensorboard_dir,
            worker_sync_dir=worker_sync_dir,
            desired_actor_path=worker_sync_dir / "desired_actor.json",
            worker_actor_status_path=worker_sync_dir / "worker_actor_status.json",
        )
        self.writer = TensorBoardScalarLogger(self.paths.tensorboard_dir)

        requested_device = "cuda" if self.config.train.cuda_training else "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        if self.config.observation.mode == "full":
            observation_shape = (
                self.config.full_observation.frame_stack,
                self.config.full_observation.output_height,
                self.config.full_observation.output_width,
            )
            telemetry_dim = TELEMETRY_DIM
        else:
            observation_shape = (lidar_feature_dim(self.config.lidar_observation),)
            telemetry_dim = 0
        self.agent = SACAgent(
            config=self.config.sac,
            observation_mode=self.config.observation.mode,
            device=self.device,
            observation_shape=observation_shape,
            telemetry_dim=telemetry_dim,
        )
        if init_actor is not None:
            self.bc_checkpoint_metadata = self.agent.load_bc_warm_start(init_actor, init_mode=init_mode)
            if self.demo_root is None and self.bc_checkpoint_metadata.get("demo_root") is not None:
                self.demo_root = str(self.bc_checkpoint_metadata["demo_root"])
        self.replay = ReplayBuffer.from_config(self.config)
        if seed_demos is not None:
            seeded_root = Path(seed_demos).resolve()
            seeded = seed_replay_from_demo_sidecars(self.replay, seeded_root)
            self.replay_seeded = seeded > 0
            if self.demo_root is None:
                self.demo_root = str(seeded_root)
            self.writer.add_scalar("train/demo_seeded_transitions", float(seeded), step=0)

        self.command_queue = None
        self.output_queue = None
        self.eval_result_queue = None
        self.shutdown_event = None
        self.worker_done_event = None
        self.worker_process = None

        self.env_step = 0
        self.learner_step = 0
        self.episode_count = 0
        self.latest_eval_summary: dict[str, Any] | None = None
        self.latest_eval_summary_path: str | None = None
        self.latest_checkpoint_path: Path | None = None
        self.last_worker_heartbeat: dict[str, Any] | None = None
        self.desired_actor_version: int | None = None
        self.desired_actor_ready_for_control = False
        self.desired_actor_control_ready_reason: str | None = None
        self.seen_actor_version: int | None = None
        self.ready_for_control_seen = False
        self.applied_actor_version: int | None = None
        self.actor_ready_for_control = False
        self.applied_source_learner_step: int | None = None
        self.last_actor_apply_env_step: int | None = None
        self.last_actor_apply_episode_index: int | None = None
        self.latest_action_stats: dict[str, Any] | None = None
        self.eval_actor_version: int | None = None
        self.eval_actor_source_learner_step: int | None = None
        self.eval_in_flight = False
        self.pending_eval: dict[str, Any] | None = None
        self.started_eval: dict[str, Any] | None = None
        self.pending_checkpoint_step: int | None = None
        self.latest_eval_step: int = -1
        self.checkpoint_history: list[dict[str, Any]] = []
        self.eval_history: list[dict[str, Any]] = []
        self.termination_reason: str | None = None
        self.clean_shutdown: bool | None = None
        self._actor_sync_version = 0
        self.worker_exit: dict[str, Any] = {
            "done_event_set": False,
            "exitcode": None,
            "terminated": False,
            "timeout": False,
        }

        self.next_update_step = max(1, self.config.train.update_model_interval)
        self.next_eval_step = max(1, self.config.train.eval_interval_steps)
        self.next_checkpoint_step = max(1, self.config.train.checkpoint_interval_steps)
        self._reschedule_from_counters()
        self._initialize_diagnostics_state()
        self._write_run_summary()

    def attach_worker(
        self,
        *,
        command_queue,
        output_queue,
        eval_result_queue,
        shutdown_event,
        worker_done_event,
        worker_process=None,
    ) -> None:  # noqa: ANN001
        self.command_queue = command_queue
        self.output_queue = output_queue
        self.eval_result_queue = eval_result_queue
        self.shutdown_event = shutdown_event
        self.worker_done_event = worker_done_event
        self.worker_process = worker_process

    def _initialize_diagnostics_state(self) -> None:
        self.primary_metric = "mean_final_progress_index"
        self.learner_events_log_path = self.paths.run_dir / "learner_events.log"
        self._learner_event_logger = JsonlEventLogger(self.learner_events_log_path if self.diagnostics_enabled else None)
        self._learner_runtime = {
            "replay_sample": TimingAccumulator(),
            "critic_update": TimingAccumulator(),
            "actor_update": TimingAccumulator(),
            "message_poll_wait": TimingAccumulator(),
            "message_drain": TimingAccumulator(),
            "eval_poll_wait": TimingAccumulator(),
            "eval_drain": TimingAccumulator(),
            "eval_result_handling": TimingAccumulator(),
            "actor_broadcast": TimingAccumulator(),
        }
        self._learner_queue = {
            "command_put": QueueAccumulator(),
        }
        self.worker_runtime_profile: dict[str, Any] = {}
        self.worker_queue_profile: dict[str, Any] = {}
        self.episode_diagnostics_tracker = EpisodeDiagnosticsTracker()
        self.movement_diagnostics_tracker = MovementDiagnosticsTracker()
        self.actor_sync_tracker = ActorSyncTracker(run_start_monotonic=self._run_start_monotonic)
        self.resource_profile = build_agent_resource_profile(self.agent, self.device)
        if self.detailed_cuda_timing and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            self._refresh_resource_profile()

    def _sync_timing_device(self) -> None:
        if self.detailed_cuda_timing and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    def _refresh_resource_profile(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            self.resource_profile.update(build_agent_resource_profile(self.agent, self.device))

    def _write_learner_event(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        self._learner_event_logger.write(
            event,
            {
                "env_step": self.env_step,
                "learner_step": self.learner_step,
                **({} if payload is None else dict(payload)),
            },
        )

    def _ready_for_control(self) -> bool:
        return self.learner_step > 0

    def _current_actor_step(self) -> int | None:
        return None

    def _runtime_profile_snapshot(self) -> dict[str, Any]:
        learner_backprop_seconds = (
            self._learner_runtime["replay_sample"].total_seconds
            + self._learner_runtime["critic_update"].total_seconds
            + self._learner_runtime["actor_update"].total_seconds
        )
        worker_env_seconds = float(self.worker_runtime_profile.get("env_loop_total_seconds", 0.0) or 0.0)
        ipc_backpressure_seconds = (
            self._learner_queue["command_put"].total_wait_seconds
            + float(self.worker_queue_profile.get("output_put", {}).get("total_wait_seconds", 0.0) or 0.0)
            + float(self.worker_queue_profile.get("eval_result_put", {}).get("total_wait_seconds", 0.0) or 0.0)
        )
        actor_sync_seconds = (
            self._learner_runtime["actor_broadcast"].total_seconds
            + float(self.worker_runtime_profile.get("actor_apply", {}).get("total_seconds", 0.0) or 0.0)
        )
        return {
            "learner": {key: value.snapshot() for key, value in self._learner_runtime.items()},
            "worker": dict(self.worker_runtime_profile),
            "bottleneck_verdict": build_bottleneck_verdict(
                learner_backprop_seconds=learner_backprop_seconds,
                worker_env_seconds=worker_env_seconds,
                ipc_backpressure_seconds=ipc_backpressure_seconds,
                actor_sync_seconds=actor_sync_seconds,
            ),
        }

    def _queue_profile_snapshot(self) -> dict[str, Any]:
        return {
            "learner": {key: value.snapshot() for key, value in self._learner_queue.items()},
            "worker": dict(self.worker_queue_profile),
        }

    def _summary_payload(self) -> dict[str, Any]:
        now_timestamp = self.run_end_timestamp or _utc_now()
        self._refresh_resource_profile()
        payload = {
            "run_name": self.run_name,
            "config_path": str(self.config_path),
            "device": str(self.device),
            "run_start_timestamp": self.run_start_timestamp,
            "run_end_timestamp": self.run_end_timestamp,
            "wall_clock_elapsed_seconds": time.monotonic() - self._run_start_monotonic,
            "env_step": self.env_step,
            "learner_step": self.learner_step,
            "episode_count": self.episode_count,
            "replay_size": self.replay.size,
            "init_mode": self.init_mode,
            "bc_checkpoint_path": self.bc_checkpoint_path,
            "bc_checkpoint_metadata": self.bc_checkpoint_metadata,
            "demo_root": self.demo_root,
            "replay_seeded": self.replay_seeded,
            "eval_episodes": self.eval_episodes,
            "latest_checkpoint_path": None if self.latest_checkpoint_path is None else str(self.latest_checkpoint_path),
            "latest_eval_summary": self.latest_eval_summary,
            "latest_eval_summary_path": self.latest_eval_summary_path,
            "eval_in_flight": self.eval_in_flight,
            "pending_eval": self.pending_eval,
            "started_eval": self.started_eval,
            "checkpoint_history": self.checkpoint_history,
            "eval_history": self.eval_history,
            "last_worker_heartbeat": self.last_worker_heartbeat,
            "desired_actor_version": self.desired_actor_version,
            "desired_actor_ready_for_control": self.desired_actor_ready_for_control,
            "desired_actor_control_ready_reason": self.desired_actor_control_ready_reason,
            "seen_actor_version": self.seen_actor_version,
            "ready_for_control_seen": self.ready_for_control_seen,
            "applied_actor_version": self.applied_actor_version,
            "actor_ready_for_control": self.actor_ready_for_control,
            "applied_source_learner_step": self.applied_source_learner_step,
            "last_actor_apply_env_step": self.last_actor_apply_env_step,
            "last_actor_apply_episode_index": self.last_actor_apply_episode_index,
            "latest_action_stats": self.latest_action_stats,
            "eval_actor_version": self.eval_actor_version,
            "eval_actor_source_learner_step": self.eval_actor_source_learner_step,
            "termination_reason": self.termination_reason,
            "clean_shutdown": self.clean_shutdown,
            "worker_exit": self.worker_exit,
            "timestamp": now_timestamp,
            "observation_mode": self.config.observation.mode,
            "primary_metric": self.primary_metric,
            "runtime_profile": self._runtime_profile_snapshot(),
            "queue_profile": self._queue_profile_snapshot(),
            "actor_sync_profile": self.actor_sync_tracker.snapshot(),
            "episode_diagnostics": self.episode_diagnostics_tracker.snapshot(),
            "movement_diagnostics": self.movement_diagnostics_tracker.snapshot(),
            "resource_profile": dict(self.resource_profile),
        }
        actor_step = self._current_actor_step()
        if actor_step is not None:
            payload["actor_step"] = int(actor_step)
        return payload

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        payload = torch.load(Path(checkpoint_path).resolve(), map_location=self.device)
        self.agent.load_state_dict(payload)
        self.learner_step = int(payload.get("learner_step", 0))
        self.env_step = int(payload.get("env_step", 0))
        self._reschedule_from_counters()

    def save_checkpoint(self, *, final: bool = False) -> Path:
        step_tag = f"{self.env_step:08d}"
        suffix = "_final" if final else ""
        checkpoint_path = self.paths.checkpoints_dir / f"checkpoint_{step_tag}{suffix}.pt"
        payload = {
            **self.agent.state_dict(),
            "learner_step": self.learner_step,
            "env_step": self.env_step,
            "config_snapshot": self.config_snapshot,
            "run_name": self.run_name,
        }
        torch.save(payload, checkpoint_path)
        write_json(
            checkpoint_path.with_suffix(".json"),
            {
                "timestamp": _utc_now(),
                "replay_size": self.replay.size,
                "eval_summary": self.latest_eval_summary,
                "git_commit": _git_commit(self.repo_root),
                "learner_step": self.learner_step,
                "env_step": self.env_step,
                "checkpoint_path": str(checkpoint_path),
                "observation_mode": self.config.observation.mode,
                "latest_eval_summary_path": self.latest_eval_summary_path,
                "latest_eval_env_step": None if self.latest_eval_summary is None else self.latest_eval_summary.get("env_step"),
                "desired_actor_version": self.desired_actor_version,
                "desired_actor_ready_for_control": self.desired_actor_ready_for_control,
                "desired_actor_control_ready_reason": self.desired_actor_control_ready_reason,
                "seen_actor_version": self.seen_actor_version,
                "ready_for_control_seen": self.ready_for_control_seen,
                "applied_actor_version": self.applied_actor_version,
                "actor_ready_for_control": self.actor_ready_for_control,
                "applied_source_learner_step": self.applied_source_learner_step,
                "last_actor_apply_env_step": self.last_actor_apply_env_step,
                "last_actor_apply_episode_index": self.last_actor_apply_episode_index,
                "eval_actor_version": self.eval_actor_version,
                "eval_actor_source_learner_step": self.eval_actor_source_learner_step,
                "init_mode": self.init_mode,
                "bc_checkpoint_path": self.bc_checkpoint_path,
                "demo_root": self.demo_root,
                "replay_seeded": self.replay_seeded,
            },
        )
        self.latest_checkpoint_path = checkpoint_path
        checkpoint_entry = {
            "path": str(checkpoint_path),
            "env_step": self.env_step,
            "learner_step": self.learner_step,
            "replay_size": self.replay.size,
            "timestamp": _utc_now(),
            "final": final,
            "latest_eval_summary_path": self.latest_eval_summary_path,
            "latest_eval_env_step": None if self.latest_eval_summary is None else self.latest_eval_summary.get("env_step"),
        }
        if not self.checkpoint_history or self.checkpoint_history[-1] != checkpoint_entry:
            self.checkpoint_history.append(checkpoint_entry)
        self._write_run_summary()
        return checkpoint_path

    def request_shutdown(self, *, force_event: bool = False) -> None:
        if self.command_queue is not None:
            self._put_command({"type": "shutdown"})
        if force_event and self.shutdown_event is not None:
            self.shutdown_event.set()

    def broadcast_actor(self, *, force: bool = False) -> None:
        if not force and self.learner_step <= 0:
            return
        self._write_actor_sync_state()

    def drain_messages(self, *, timeout: float = 0.25) -> int:
        if self.output_queue is None:
            return 0
        transition_count = 0
        wait_start = time.perf_counter()
        try:
            message = self.output_queue.get(timeout=timeout)
        except queue.Empty:
            self._learner_runtime["message_poll_wait"].record(time.perf_counter() - wait_start)
            return 0
        self._learner_runtime["message_poll_wait"].record(time.perf_counter() - wait_start)
        drain_start = time.perf_counter()
        transition_count += self._handle_message(message)
        while True:
            try:
                message = self.output_queue.get_nowait()
            except queue.Empty:
                break
            transition_count += self._handle_message(message)
        self._learner_runtime["message_drain"].record(time.perf_counter() - drain_start)
        return transition_count

    def drain_eval_results(self, *, timeout: float = 0.0) -> int:
        if self.eval_result_queue is None:
            return 0
        handled = 0
        wait_start = time.perf_counter()
        try:
            result = self.eval_result_queue.get(timeout=timeout)
        except queue.Empty:
            self._learner_runtime["eval_poll_wait"].record(time.perf_counter() - wait_start)
            return 0
        self._learner_runtime["eval_poll_wait"].record(time.perf_counter() - wait_start)
        drain_start = time.perf_counter()
        self._handle_eval_result(result)
        handled += 1
        while True:
            try:
                result = self.eval_result_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_eval_result(result)
            handled += 1
        self._learner_runtime["eval_drain"].record(time.perf_counter() - drain_start)
        return handled

    def maybe_train(self) -> int:
        if self.replay.size < self.config.train.environment_steps_before_training:
            return 0
        updates = 0
        updates_per_block = max(
            1,
            int(round(self.config.train.update_model_interval * self.config.train.max_training_steps_per_environment_step)),
        )
        while self.env_step >= self.next_update_step:
            for _ in range(updates_per_block):
                self._sync_timing_device()
                sample_start = time.perf_counter()
                batch = self.replay.sample(self.config.train.batch_size, device=self.device)
                self._sync_timing_device()
                self._learner_runtime["replay_sample"].record(time.perf_counter() - sample_start)
                self._sync_timing_device()
                update_start = time.perf_counter()
                update = self.agent.update(batch)
                self._sync_timing_device()
                update_duration = time.perf_counter() - update_start
                critic_update_seconds = float(getattr(update, "critic_update_seconds", update_duration) or 0.0)
                actor_update_seconds = float(getattr(update, "actor_update_seconds", 0.0) or 0.0)
                self._learner_runtime["critic_update"].record(critic_update_seconds if critic_update_seconds > 0.0 else update_duration)
                if actor_update_seconds > 0.0:
                    self._learner_runtime["actor_update"].record(actor_update_seconds)
                self.learner_step += 1
                self._log_update(update)
            updates += updates_per_block
            self.next_update_step += self.config.train.update_model_interval
        return updates

    def maybe_schedule_eval(self) -> None:
        if self.command_queue is None or self.eval_in_flight or self.eval_episodes <= 0:
            return
        if not self.actor_ready_for_control or self.applied_actor_version is None:
            return
        if self.env_step < self.next_eval_step:
            return
        scheduled_step = self.next_eval_step
        self._put_command(
            {
                "type": "run_eval",
                "episodes": self.eval_episodes,
                "seed_base": self.config.eval.seed_base,
                "record_video": self.config.eval.record_video,
                "run_name": f"{self.run_name}_step_{scheduled_step:08d}",
                "learner_step": self.learner_step,
                "env_step": scheduled_step,
                "eval_actor_version": self.applied_actor_version,
                "eval_actor_source_learner_step": self.applied_source_learner_step,
                "checkpoint_path": str(self.latest_checkpoint_path) if self.latest_checkpoint_path is not None else None,
            }
        )
        self.eval_in_flight = True
        self.pending_eval = {
            "run_name": f"{self.run_name}_step_{scheduled_step:08d}",
            "env_step": scheduled_step,
            "learner_step": self.learner_step,
            "episodes": self.eval_episodes,
            "eval_actor_version": self.applied_actor_version,
            "eval_actor_source_learner_step": self.applied_source_learner_step,
            "scheduled_at": _utc_now(),
        }
        if self.env_step >= self.next_checkpoint_step:
            self.pending_checkpoint_step = scheduled_step
            while self.next_checkpoint_step <= scheduled_step:
                self.next_checkpoint_step += self.config.train.checkpoint_interval_steps
        while self.next_eval_step <= scheduled_step:
            self.next_eval_step += self.config.train.eval_interval_steps
        self._write_learner_event(
            "eval_scheduled",
            {
                "scheduled_env_step": scheduled_step,
                "eval_actor_version": self.applied_actor_version,
                "eval_actor_source_learner_step": self.applied_source_learner_step,
            },
        )
        self._write_run_summary()

    def maybe_checkpoint(self) -> None:
        if self.eval_in_flight:
            return
        if self.env_step < self.next_checkpoint_step:
            return
        self.save_checkpoint()
        while self.next_checkpoint_step <= self.env_step:
            self.next_checkpoint_step += self.config.train.checkpoint_interval_steps

    def should_stop(self) -> bool:
        if self.shutdown_event is not None and self.shutdown_event.is_set():
            return True
        return self.max_env_steps is not None and self.env_step >= self.max_env_steps

    def run(self) -> None:
        if (
            self.command_queue is None
            or self.output_queue is None
            or self.eval_result_queue is None
            or self.shutdown_event is None
            or self.worker_done_event is None
        ):
            raise RuntimeError("attach_worker() must be called before run().")
        self._write_learner_event("run_started", {"run_name": self.run_name})
        self.broadcast_actor(force=True)
        try:
            while not self.should_stop():
                self.drain_messages(timeout=0.25)
                self.drain_eval_results(timeout=0.0)
                updates = self.maybe_train()
                if updates > 0:
                    self.broadcast_actor(force=True)
                self.maybe_schedule_eval()
                self.maybe_checkpoint()
                if self.worker_process is not None and not self.worker_process.is_alive() and not self.shutdown_event.is_set():
                    self.drain_messages(timeout=0.0)
                    self.drain_eval_results(timeout=0.0)
                    worker_done = self.worker_done_event.is_set() if self.worker_done_event is not None else False
                    if self.worker_process.exitcode == 0 and worker_done and self.should_stop():
                        break
                    raise RuntimeError(f"Worker process exited unexpectedly with code {self.worker_process.exitcode}.")
        except KeyboardInterrupt:
            self.termination_reason = "keyboard_interrupt"
            raise
        except Exception:
            self.termination_reason = self.termination_reason or "fatal_error"
            raise
        else:
            if self.termination_reason is None:
                if self.max_env_steps is not None and self.env_step >= self.max_env_steps:
                    self.termination_reason = "max_env_steps"
                else:
                    self.termination_reason = "completed"

    def close(self) -> None:
        self.writer.close()

    def _handle_message(self, message: Mapping[str, Any]) -> int:
        message_type = str(message.get("type", ""))
        if message_type == "transition_batch":
            for transition in message.get("transitions", []):
                self.replay.add(transition)
            self.env_step = max(self.env_step, int(message.get("env_step", self.env_step)))
            return len(message.get("transitions", []))
        if message_type == "transition":
            self.replay.add(message["transition"])
            self.env_step = max(self.env_step, int(message.get("env_step", self.env_step + 1)))
            return 1
        if message_type == "episode_summary":
            self.episode_count += 1
            summary = dict(message.get("summary", {}))
            self.episode_diagnostics_tracker.record(summary)
            self.writer.add_scalars_from_mapping("train/episode", summary, step=self.env_step)
            self._write_run_summary()
            return 0
        if message_type == "movement_episode_summary":
            summary = dict(message.get("summary", {}))
            self.movement_diagnostics_tracker.record(summary)
            self.writer.add_scalars_from_mapping("train/movement", summary, step=self.env_step)
            self._write_run_summary()
            return 0
        if message_type == "eval_result":
            return 0
        if message_type == "actor_sync_desired_seen":
            if message.get("desired_actor_version") is not None:
                self.desired_actor_version = int(message["desired_actor_version"])
            if message.get("desired_actor_ready_for_control") is not None:
                self.desired_actor_ready_for_control = bool(message["desired_actor_ready_for_control"])
            if message.get("seen_actor_version") is not None:
                self.seen_actor_version = int(message["seen_actor_version"])
            if message.get("ready_for_control_seen") is not None:
                self.ready_for_control_seen = bool(message["ready_for_control_seen"])
            if message.get("control_ready_reason") is not None:
                self.desired_actor_control_ready_reason = str(message["control_ready_reason"])
            self.actor_sync_tracker.record_desired_seen(message, received_monotonic=time.monotonic())
            self._write_run_summary()
            return 0
        if message_type == "actor_sync_applied":
            if message.get("desired_actor_version") is not None:
                self.desired_actor_version = int(message["desired_actor_version"])
            if message.get("desired_actor_ready_for_control") is not None:
                self.desired_actor_ready_for_control = bool(message["desired_actor_ready_for_control"])
            if message.get("seen_actor_version") is not None:
                self.seen_actor_version = int(message["seen_actor_version"])
            if message.get("ready_for_control_seen") is not None:
                self.ready_for_control_seen = bool(message["ready_for_control_seen"])
            self.applied_actor_version = (
                int(message.get("applied_actor_version"))
                if message.get("applied_actor_version") is not None
                else None
            )
            if message.get("actor_ready_for_control") is not None:
                self.actor_ready_for_control = bool(message["actor_ready_for_control"])
            self.applied_source_learner_step = (
                int(message["applied_source_learner_step"])
                if message.get("applied_source_learner_step") is not None
                else None
            )
            self.last_actor_apply_env_step = (
                int(message["last_actor_apply_env_step"]) if message.get("last_actor_apply_env_step") is not None else None
            )
            self.last_actor_apply_episode_index = (
                int(message["last_actor_apply_episode_index"])
                if message.get("last_actor_apply_episode_index") is not None
                else None
            )
            if message.get("control_ready_reason") is not None:
                self.desired_actor_control_ready_reason = str(message["control_ready_reason"])
            self.actor_sync_tracker.record_applied(
                message,
                received_monotonic=time.monotonic(),
                current_learner_step=self.learner_step,
                current_env_step=self.env_step,
            )
            self._write_run_summary()
            return 0
        if message_type == "action_stats":
            action_stats = {key: value for key, value in dict(message).items() if key != "type"}
            self.latest_action_stats = action_stats
            if message.get("desired_actor_version") is not None:
                self.desired_actor_version = int(message["desired_actor_version"])
            if message.get("desired_actor_ready_for_control") is not None:
                self.desired_actor_ready_for_control = bool(message["desired_actor_ready_for_control"])
            if message.get("seen_actor_version") is not None:
                self.seen_actor_version = int(message["seen_actor_version"])
            if message.get("ready_for_control_seen") is not None:
                self.ready_for_control_seen = bool(message["ready_for_control_seen"])
            if message.get("applied_actor_version") is not None:
                self.applied_actor_version = int(message["applied_actor_version"])
            if message.get("actor_ready_for_control") is not None:
                self.actor_ready_for_control = bool(message["actor_ready_for_control"])
            if message.get("applied_source_learner_step") is not None:
                self.applied_source_learner_step = int(message["applied_source_learner_step"])
            if message.get("last_actor_apply_env_step") is not None:
                self.last_actor_apply_env_step = int(message["last_actor_apply_env_step"])
            if message.get("last_actor_apply_episode_index") is not None:
                self.last_actor_apply_episode_index = int(message["last_actor_apply_episode_index"])
            if message.get("control_ready_reason") is not None:
                self.desired_actor_control_ready_reason = str(message["control_ready_reason"])
            self.actor_sync_tracker.record_control_window(
                action_stats,
                received_monotonic=time.monotonic(),
                current_learner_step=self.learner_step,
                current_env_step=self.env_step,
            )
            self._write_run_summary()
            return 0
        if message_type == "eval_started":
            self.started_eval = {
                "run_name": message.get("run_name"),
                "env_step": int(message.get("env_step", self.env_step)),
                "learner_step": int(message.get("learner_step", self.learner_step)),
                "episodes": int(message.get("episodes", self.eval_episodes)),
                "timestamp": message.get("timestamp"),
                "eval_actor_version": (
                    int(message["eval_actor_version"]) if message.get("eval_actor_version") is not None else None
                ),
                "eval_actor_source_learner_step": (
                    int(message["eval_actor_source_learner_step"])
                    if message.get("eval_actor_source_learner_step") is not None
                    else None
                ),
            }
            self.eval_actor_version = self.started_eval["eval_actor_version"]
            self.eval_actor_source_learner_step = self.started_eval["eval_actor_source_learner_step"]
            self._write_run_summary()
            return 0
        if message_type == "heartbeat":
            self.last_worker_heartbeat = dict(message)
            if message.get("desired_actor_version") is not None:
                self.desired_actor_version = int(message["desired_actor_version"])
            if message.get("desired_actor_ready_for_control") is not None:
                self.desired_actor_ready_for_control = bool(message["desired_actor_ready_for_control"])
            if message.get("seen_actor_version") is not None:
                self.seen_actor_version = int(message["seen_actor_version"])
            if message.get("ready_for_control_seen") is not None:
                self.ready_for_control_seen = bool(message["ready_for_control_seen"])
            if message.get("applied_actor_version") is not None:
                self.applied_actor_version = int(message["applied_actor_version"])
            if message.get("actor_ready_for_control") is not None:
                self.actor_ready_for_control = bool(message["actor_ready_for_control"])
            if message.get("applied_source_learner_step") is not None:
                self.applied_source_learner_step = int(message["applied_source_learner_step"])
            if message.get("last_actor_apply_env_step") is not None:
                self.last_actor_apply_env_step = int(message["last_actor_apply_env_step"])
            if message.get("last_actor_apply_episode_index") is not None:
                self.last_actor_apply_episode_index = int(message["last_actor_apply_episode_index"])
            if message.get("latest_action_stats") is not None:
                self.latest_action_stats = dict(message["latest_action_stats"])
            if message.get("control_ready_reason") is not None:
                self.desired_actor_control_ready_reason = str(message["control_ready_reason"])
            if message.get("runtime_profile") is not None:
                self.worker_runtime_profile = dict(message["runtime_profile"])
            if message.get("queue_profile") is not None:
                self.worker_queue_profile = dict(message["queue_profile"])
            if isinstance(message.get("latest_action_stats"), Mapping):
                self.actor_sync_tracker.record_control_window(
                    dict(message["latest_action_stats"]),
                    received_monotonic=time.monotonic(),
                    current_learner_step=self.learner_step,
                    current_env_step=self.env_step,
                )
            self._write_run_summary()
            return 0
        if message_type == "fatal_error":
            self.termination_reason = self.termination_reason or "fatal_error"
            raise RuntimeError(str(message.get("error", "Worker reported an unspecified fatal error.")))
        return 0

    def _put_command(self, payload: Mapping[str, Any]) -> None:
        assert self.command_queue is not None
        wait_start = time.perf_counter()
        full_retries = 0
        while True:
            try:
                self.command_queue.put(dict(payload), timeout=0.25)
                wait_seconds = time.perf_counter() - wait_start
                self._learner_queue["command_put"].record(wait_seconds, success=True, full_retries=full_retries)
                if wait_seconds > 0.0 or full_retries > 0:
                    self._write_learner_event(
                        "command_put",
                        {
                            "command_type": payload.get("type"),
                            "wait_seconds": wait_seconds,
                            "full_retries": full_retries,
                        },
                    )
                return
            except queue.Full:
                full_retries += 1
                if self.shutdown_event is not None and self.shutdown_event.is_set():
                    self._learner_queue["command_put"].record(
                        time.perf_counter() - wait_start,
                        success=False,
                        full_retries=full_retries,
                        timed_out=True,
                    )
                    return

    def _write_actor_sync_state(self) -> Path:
        desired_actor_path = self.paths.desired_actor_path
        desired_actor_path.parent.mkdir(parents=True, exist_ok=True)
        temp_manifest_path = desired_actor_path.with_suffix(".tmp")
        next_version = self._actor_sync_version + 1
        actor_state_path = self.paths.worker_sync_dir / f"actor_v{next_version:06d}.pt"
        temp_path = actor_state_path.with_suffix(".tmp")
        self._actor_sync_version = next_version
        ready_for_control = self._ready_for_control()
        control_ready_reason = "trained_updates_available" if ready_for_control else "startup_untrained"
        broadcast_start = time.perf_counter()
        torch.save(self.agent.actor_state_dict_cpu(), temp_path)
        temp_path.replace(actor_state_path)
        actor_step = self._current_actor_step()
        write_json(
            temp_manifest_path,
            {
                "desired_actor_version": self._actor_sync_version,
                "env_step": self.env_step,
                "learner_step": self.learner_step,
                **({} if actor_step is None else {"actor_step": actor_step}),
                "written_at": _utc_now(),
                "actor_state_path": str(actor_state_path),
                "ready_for_control": ready_for_control,
                "control_ready_reason": control_ready_reason,
            },
        )
        temp_manifest_path.replace(desired_actor_path)
        broadcast_duration = time.perf_counter() - broadcast_start
        self._learner_runtime["actor_broadcast"].record(broadcast_duration)
        self.desired_actor_version = self._actor_sync_version
        self.desired_actor_ready_for_control = ready_for_control
        self.desired_actor_control_ready_reason = control_ready_reason
        self.actor_sync_tracker.record_broadcast(
            self._actor_sync_version,
            env_step=self.env_step,
            learner_step=self.learner_step,
            actor_step=actor_step,
            ready_for_control=ready_for_control,
            broadcast_monotonic=time.monotonic(),
        )
        self._write_learner_event(
            "actor_broadcast",
            {
                "desired_actor_version": self._actor_sync_version,
                "ready_for_control": ready_for_control,
                "control_ready_reason": control_ready_reason,
                "broadcast_seconds": broadcast_duration,
                "actor_state_path": str(actor_state_path),
            },
        )
        self._write_run_summary()
        return actor_state_path

    def _log_update(self, update: SACUpdateResult) -> None:
        self.writer.add_scalars_from_mapping("train/update", asdict(update), step=self.learner_step)
        self.writer.add_scalar("train/replay_size", float(self.replay.size), step=self.learner_step)
        self.writer.add_scalar("train/env_step", float(self.env_step), step=self.learner_step)

    def _handle_eval_result(self, result: EvalResult | Mapping[str, Any]) -> None:
        handling_start = time.perf_counter()
        payload = result if isinstance(result, EvalResult) else EvalResult(**dict(result))
        if payload.env_step < self.latest_eval_step:
            self._learner_runtime["eval_result_handling"].record(time.perf_counter() - handling_start)
            return
        self.latest_eval_step = payload.env_step
        self.latest_eval_summary = dict(payload.summary)
        self.latest_eval_summary_path = payload.summary_path
        self.eval_actor_version = (
            int(payload.summary["eval_actor_version"]) if payload.summary.get("eval_actor_version") is not None else None
        )
        self.eval_actor_source_learner_step = (
            int(payload.summary["eval_actor_source_learner_step"])
            if payload.summary.get("eval_actor_source_learner_step") is not None
            else None
        )
        self.eval_in_flight = False
        self.pending_eval = None
        self.started_eval = None
        eval_entry = {
            "checkpoint_step": payload.checkpoint_step,
            "env_step": payload.env_step,
            "learner_step": payload.learner_step,
            "summary_path": payload.summary_path,
            "summary": dict(payload.summary),
            "eval_actor_version": self.eval_actor_version,
            "eval_actor_source_learner_step": self.eval_actor_source_learner_step,
            "timestamp": payload.timestamp,
        }
        self.eval_history.append(eval_entry)
        if self.latest_eval_summary:
            self.writer.add_scalars_from_mapping("eval", self.latest_eval_summary, step=payload.env_step)
        self._write_run_summary()
        if self.pending_checkpoint_step is not None and payload.checkpoint_step >= self.pending_checkpoint_step:
            self.save_checkpoint()
            self.pending_checkpoint_step = None
        self._learner_runtime["eval_result_handling"].record(time.perf_counter() - handling_start)

    def _reschedule_from_counters(self) -> None:
        self.next_update_step = self._next_multiple(self.env_step, self.config.train.update_model_interval)
        self.next_eval_step = self._next_multiple(self.env_step, self.config.train.eval_interval_steps)
        self.next_checkpoint_step = self._next_multiple(self.env_step, self.config.train.checkpoint_interval_steps)

    @staticmethod
    def _next_multiple(current: int, interval: int) -> int:
        if interval <= 0:
            return current + 1
        return ((current // interval) + 1) * interval

    def _write_run_summary(self) -> None:
        write_json(self.paths.summary_json, self._summary_payload())

    def finalize_run(self, *, timeout_seconds: float = 30.0) -> Path:
        if self.termination_reason is None:
            self.termination_reason = "shutdown_requested"
        self.request_shutdown()
        deadline = time.monotonic() + timeout_seconds
        done_event_set = False
        while time.monotonic() < deadline:
            self.drain_messages(timeout=0.25)
            self.drain_eval_results(timeout=0.0)
            done_event_set = self.worker_done_event.is_set() if self.worker_done_event is not None else False
            if done_event_set:
                break
            if self.worker_process is not None and not self.worker_process.is_alive():
                break

        if self.worker_process is not None:
            remaining = max(0.0, deadline - time.monotonic())
            self.worker_process.join(timeout=remaining)
        self.drain_messages(timeout=0.0)
        self.drain_eval_results(timeout=0.0)

        done_event_set = self.worker_done_event.is_set() if self.worker_done_event is not None else False
        worker_alive = self.worker_process.is_alive() if self.worker_process is not None else False
        terminated = False
        timeout_hit = False
        if worker_alive and self.worker_process is not None:
            timeout_hit = True
            if self.shutdown_event is not None:
                self.shutdown_event.set()
            self.worker_process.terminate()
            self.worker_process.join(timeout=5.0)
            terminated = True
        exitcode = None if self.worker_process is None else self.worker_process.exitcode

        self.worker_exit = {
            "done_event_set": done_event_set,
            "exitcode": exitcode,
            "terminated": terminated,
            "timeout": timeout_hit,
        }
        self.clean_shutdown = bool(done_event_set and not terminated)
        self.run_end_timestamp = _utc_now()
        self._write_learner_event(
            "run_finalized",
            {
                "done_event_set": done_event_set,
                "worker_exitcode": exitcode,
                "terminated": terminated,
                "timeout": timeout_hit,
                "clean_shutdown": self.clean_shutdown,
            },
        )
        final_checkpoint = self.save_checkpoint(final=True)
        self._write_run_summary()
        return final_checkpoint


class REDQLearner(SACLearner):
    def __init__(
        self,
        *,
        config_path: str | Path,
        run_name: str | None = None,
        max_env_steps: int | None = None,
        init_actor: str | Path | None = None,
        init_mode: str = "scratch",
        demo_root: str | Path | None = None,
        seed_demos: str | Path | None = None,
        eval_episodes_override: int | None = None,
        diagnostics_enabled: bool = False,
        detailed_cuda_timing: bool = False,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.config: TM20AIConfig = load_tm20ai_config(self.config_path)
        if self.config.train.algorithm != "redq":
            raise ValueError(f"Unsupported training algorithm: {self.config.train.algorithm!r}")
        if init_mode not in {"scratch", "actor_only", "actor_plus_critic_encoders"}:
            raise ValueError(f"Unsupported FULL REDQ init_mode: {init_mode!r}")
        if init_mode == "scratch" and init_actor is not None:
            raise ValueError("init_actor requires init_mode to be actor_only or actor_plus_critic_encoders.")
        if init_mode != "scratch" and init_actor is None:
            raise ValueError("BC warm start requires --init-actor with init_mode actor_only or actor_plus_critic_encoders.")
        self.repo_root = self.config_path.parents[1]
        self.config_snapshot = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        run_prefix = f"{self.config.observation.mode}_redq"
        self.run_name = run_name or f"{run_prefix}_{timestamp_tag()}"
        self.max_env_steps = self.config.train.max_env_steps if max_env_steps is None else int(max_env_steps)
        self.eval_episodes = self.config.eval.episodes if eval_episodes_override is None else int(eval_episodes_override)
        self.init_mode = init_mode
        self.bc_checkpoint_path = None if init_actor is None else str(Path(init_actor).resolve())
        self.bc_checkpoint_metadata: dict[str, Any] | None = None
        self.demo_root = None if demo_root is None else str(Path(demo_root).resolve())
        self.replay_seeded = False
        self.run_start_timestamp = _utc_now()
        self.run_end_timestamp: str | None = None
        self._run_start_monotonic = time.monotonic()
        self.diagnostics_enabled = bool(diagnostics_enabled)
        self.detailed_cuda_timing = bool(detailed_cuda_timing)

        run_paths = build_run_artifact_paths(self.config, mode="train", run_name=self.run_name)
        worker_sync_dir = ensure_directory(run_paths.run_dir / "worker_sync")
        self.paths = TrainRunPaths(
            run_dir=run_paths.run_dir,
            checkpoints_dir=ensure_directory(run_paths.run_dir / "checkpoints"),
            summary_json=run_paths.summary_json,
            tensorboard_dir=run_paths.tensorboard_dir,
            worker_sync_dir=worker_sync_dir,
            desired_actor_path=worker_sync_dir / "desired_actor.json",
            worker_actor_status_path=worker_sync_dir / "worker_actor_status.json",
        )
        self.writer = TensorBoardScalarLogger(self.paths.tensorboard_dir)

        requested_device = "cuda" if self.config.train.cuda_training else "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        if self.config.observation.mode == "full":
            observation_shape = (
                self.config.full_observation.frame_stack,
                self.config.full_observation.output_height,
                self.config.full_observation.output_width,
            )
            telemetry_dim = TELEMETRY_DIM
        else:
            observation_shape = (lidar_feature_dim(self.config.lidar_observation),)
            telemetry_dim = 0
        self.agent = REDQSACAgent(
            sac_config=self.config.sac,
            redq_config=self.config.redq,
            observation_mode=self.config.observation.mode,
            device=self.device,
            observation_shape=observation_shape,
            telemetry_dim=telemetry_dim,
        )
        if init_actor is not None:
            self.bc_checkpoint_metadata = self.agent.load_bc_warm_start(init_actor, init_mode=init_mode)
            if self.demo_root is None and self.bc_checkpoint_metadata.get("demo_root") is not None:
                self.demo_root = str(self.bc_checkpoint_metadata["demo_root"])
        self.replay = ReplayBuffer.from_config(self.config)
        if seed_demos is not None:
            seeded_root = Path(seed_demos).resolve()
            seeded = seed_replay_from_demo_sidecars(self.replay, seeded_root)
            self.replay_seeded = seeded > 0
            if self.demo_root is None:
                self.demo_root = str(seeded_root)
            self.writer.add_scalar("train/demo_seeded_transitions", float(seeded), step=0)

        self.command_queue = None
        self.output_queue = None
        self.eval_result_queue = None
        self.shutdown_event = None
        self.worker_done_event = None
        self.worker_process = None

        self.env_step = 0
        self.learner_step = 0
        self.actor_step = 0
        self.episode_count = 0
        self.latest_eval_summary: dict[str, Any] | None = None
        self.latest_eval_summary_path: str | None = None
        self.latest_checkpoint_path: Path | None = None
        self.last_worker_heartbeat: dict[str, Any] | None = None
        self.desired_actor_version: int | None = None
        self.desired_actor_ready_for_control = False
        self.desired_actor_control_ready_reason: str | None = None
        self.seen_actor_version: int | None = None
        self.ready_for_control_seen = False
        self.applied_actor_version: int | None = None
        self.actor_ready_for_control = False
        self.applied_source_learner_step: int | None = None
        self.last_actor_apply_env_step: int | None = None
        self.last_actor_apply_episode_index: int | None = None
        self.latest_action_stats: dict[str, Any] | None = None
        self.eval_actor_version: int | None = None
        self.eval_actor_source_learner_step: int | None = None
        self.eval_in_flight = False
        self.pending_eval: dict[str, Any] | None = None
        self.started_eval: dict[str, Any] | None = None
        self.pending_checkpoint_step: int | None = None
        self.latest_eval_step: int = -1
        self.checkpoint_history: list[dict[str, Any]] = []
        self.eval_history: list[dict[str, Any]] = []
        self.termination_reason: str | None = None
        self.clean_shutdown: bool | None = None
        self._actor_sync_version = 0
        self.worker_exit: dict[str, Any] = {
            "done_event_set": False,
            "exitcode": None,
            "terminated": False,
            "timeout": False,
        }

        self.next_update_step = max(1, self.config.train.update_model_interval)
        self.next_eval_step = max(1, self.config.train.eval_interval_steps)
        self.next_checkpoint_step = max(1, self.config.train.checkpoint_interval_steps)
        self._reschedule_from_counters()
        self._initialize_diagnostics_state()
        self._write_run_summary()

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        payload = torch.load(Path(checkpoint_path).resolve(), map_location=self.device)
        self.agent.load_state_dict(payload)
        self.learner_step = int(payload.get("learner_step", 0))
        self.actor_step = int(payload.get("actor_step", 0))
        self.env_step = int(payload.get("env_step", 0))
        self._reschedule_from_counters()

    def _ready_for_control(self) -> bool:
        return self.actor_step > 0

    def _current_actor_step(self) -> int | None:
        return self.actor_step

    def save_checkpoint(self, *, final: bool = False) -> Path:
        step_tag = f"{self.env_step:08d}"
        suffix = "_final" if final else ""
        checkpoint_path = self.paths.checkpoints_dir / f"checkpoint_{step_tag}{suffix}.pt"
        payload = {
            **self.agent.state_dict(),
            "learner_step": self.learner_step,
            "actor_step": self.actor_step,
            "env_step": self.env_step,
            "config_snapshot": self.config_snapshot,
            "run_name": self.run_name,
        }
        torch.save(payload, checkpoint_path)
        write_json(
            checkpoint_path.with_suffix(".json"),
            {
                "timestamp": _utc_now(),
                "replay_size": self.replay.size,
                "eval_summary": self.latest_eval_summary,
                "git_commit": _git_commit(self.repo_root),
                "learner_step": self.learner_step,
                "actor_step": self.actor_step,
                "env_step": self.env_step,
                "checkpoint_path": str(checkpoint_path),
                "observation_mode": self.config.observation.mode,
                "latest_eval_summary_path": self.latest_eval_summary_path,
                "latest_eval_env_step": None if self.latest_eval_summary is None else self.latest_eval_summary.get("env_step"),
                "desired_actor_version": self.desired_actor_version,
                "desired_actor_ready_for_control": self.desired_actor_ready_for_control,
                "desired_actor_control_ready_reason": self.desired_actor_control_ready_reason,
                "seen_actor_version": self.seen_actor_version,
                "ready_for_control_seen": self.ready_for_control_seen,
                "applied_actor_version": self.applied_actor_version,
                "actor_ready_for_control": self.actor_ready_for_control,
                "applied_source_learner_step": self.applied_source_learner_step,
                "last_actor_apply_env_step": self.last_actor_apply_env_step,
                "last_actor_apply_episode_index": self.last_actor_apply_episode_index,
                "eval_actor_version": self.eval_actor_version,
                "eval_actor_source_learner_step": self.eval_actor_source_learner_step,
                "init_mode": self.init_mode,
                "bc_checkpoint_path": self.bc_checkpoint_path,
                "demo_root": self.demo_root,
                "replay_seeded": self.replay_seeded,
            },
        )
        self.latest_checkpoint_path = checkpoint_path
        checkpoint_entry = {
            "path": str(checkpoint_path),
            "env_step": self.env_step,
            "learner_step": self.learner_step,
            "actor_step": self.actor_step,
            "replay_size": self.replay.size,
            "timestamp": _utc_now(),
            "final": final,
            "latest_eval_summary_path": self.latest_eval_summary_path,
            "latest_eval_env_step": None if self.latest_eval_summary is None else self.latest_eval_summary.get("env_step"),
        }
        if not self.checkpoint_history or self.checkpoint_history[-1] != checkpoint_entry:
            self.checkpoint_history.append(checkpoint_entry)
        self._write_run_summary()
        return checkpoint_path

    def maybe_train(self) -> int:
        if self.replay.size < self.config.train.environment_steps_before_training:
            return 0
        actor_updates = 0
        updates_per_block = max(
            1,
            int(round(self.config.train.update_model_interval * self.config.train.max_training_steps_per_environment_step)),
        )
        while self.env_step >= self.next_update_step:
            for _ in range(updates_per_block):
                self._sync_timing_device()
                sample_start = time.perf_counter()
                batch = self.replay.sample(self.config.train.batch_size, device=self.device)
                self._sync_timing_device()
                self._learner_runtime["replay_sample"].record(time.perf_counter() - sample_start)
                self._sync_timing_device()
                critic_start = time.perf_counter()
                critic_update = self.agent.update_critics(batch)
                self._sync_timing_device()
                self._learner_runtime["critic_update"].record(time.perf_counter() - critic_start)
                self.learner_step += 1
                self._log_critic_update(critic_update)
                self._sync_timing_device()
                actor_start = time.perf_counter()
                actor_update = self.agent.maybe_update_actor_and_alpha(batch)
                self._sync_timing_device()
                if actor_update is not None:
                    self._learner_runtime["actor_update"].record(time.perf_counter() - actor_start)
                if actor_update is not None:
                    self.actor_step += 1
                    actor_updates += 1
                    self._log_actor_update(actor_update)
            self.next_update_step += self.config.train.update_model_interval
        return actor_updates

    def run(self) -> None:
        if (
            self.command_queue is None
            or self.output_queue is None
            or self.eval_result_queue is None
            or self.shutdown_event is None
            or self.worker_done_event is None
        ):
            raise RuntimeError("attach_worker() must be called before run().")
        self._write_learner_event("run_started", {"run_name": self.run_name})
        self.broadcast_actor(force=True)
        try:
            while not self.should_stop():
                self.drain_messages(timeout=0.25)
                self.drain_eval_results(timeout=0.0)
                actor_updates = self.maybe_train()
                if actor_updates > 0:
                    self.broadcast_actor(force=True)
                self.maybe_schedule_eval()
                self.maybe_checkpoint()
                if self.worker_process is not None and not self.worker_process.is_alive() and not self.shutdown_event.is_set():
                    self.drain_messages(timeout=0.0)
                    self.drain_eval_results(timeout=0.0)
                    worker_done = self.worker_done_event.is_set() if self.worker_done_event is not None else False
                    if self.worker_process.exitcode == 0 and worker_done and self.should_stop():
                        break
                    raise RuntimeError(f"Worker process exited unexpectedly with code {self.worker_process.exitcode}.")
        except KeyboardInterrupt:
            self.termination_reason = "keyboard_interrupt"
            raise
        except Exception:
            self.termination_reason = self.termination_reason or "fatal_error"
            raise
        else:
            if self.termination_reason is None:
                if self.max_env_steps is not None and self.env_step >= self.max_env_steps:
                    self.termination_reason = "max_env_steps"
                else:
                    self.termination_reason = "completed"

    def _write_actor_sync_state(self) -> Path:
        return super()._write_actor_sync_state()

    def _log_critic_update(self, update: REDQCriticUpdateResult) -> None:
        self.writer.add_scalars_from_mapping("train/update_critic", asdict(update), step=self.learner_step)
        self.writer.add_scalar("train/replay_size", float(self.replay.size), step=self.learner_step)
        self.writer.add_scalar("train/env_step", float(self.env_step), step=self.learner_step)
        self.writer.add_scalar("train/actor_step", float(self.actor_step), step=self.learner_step)

    def _log_actor_update(self, update: REDQActorUpdateResult) -> None:
        self.writer.add_scalars_from_mapping("train/update_actor", asdict(update), step=self.actor_step)
        self.writer.add_scalar("train/learner_step", float(self.learner_step), step=self.actor_step)
        self.writer.add_scalar("train/env_step", float(self.env_step), step=self.actor_step)

    def _write_run_summary(self) -> None:
        write_json(self.paths.summary_json, self._summary_payload())

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

from ..algos.sac import SACAgent, SACUpdateResult
from ..capture import lidar_feature_dim
from ..config import TM20AIConfig, load_tm20ai_config
from ..data.dataset import seed_replay_from_demo_sidecars
from ..data.parquet_writer import build_run_artifact_paths, ensure_directory, timestamp_tag, write_json
from ..train.features import TELEMETRY_DIM
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


class SACLearner:
    def __init__(
        self,
        *,
        config_path: str | Path,
        run_name: str | None = None,
        max_env_steps: int | None = None,
        init_actor: str | Path | None = None,
        seed_demos: str | Path | None = None,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.config: TM20AIConfig = load_tm20ai_config(self.config_path)
        if self.config.train.algorithm != "sac":
            raise ValueError(f"Unsupported training algorithm: {self.config.train.algorithm!r}")
        self.repo_root = self.config_path.parents[1]
        self.config_snapshot = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        run_prefix = f"{self.config.observation.mode}_sac"
        self.run_name = run_name or f"{run_prefix}_{timestamp_tag()}"
        self.max_env_steps = self.config.train.max_env_steps if max_env_steps is None else int(max_env_steps)

        run_paths = build_run_artifact_paths(self.config, mode="train", run_name=self.run_name)
        self.paths = TrainRunPaths(
            run_dir=run_paths.run_dir,
            checkpoints_dir=ensure_directory(run_paths.run_dir / "checkpoints"),
            summary_json=run_paths.summary_json,
            tensorboard_dir=run_paths.tensorboard_dir,
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
            self._load_actor_only(init_actor)
        self.replay = ReplayBuffer.from_config(self.config)
        if seed_demos is not None:
            seeded = seed_replay_from_demo_sidecars(self.replay, Path(seed_demos).resolve())
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
        self.eval_in_flight = False
        self.pending_checkpoint_step: int | None = None
        self.latest_eval_step: int = -1
        self.checkpoint_history: list[dict[str, Any]] = []
        self.eval_history: list[dict[str, Any]] = []
        self.termination_reason: str | None = None
        self.clean_shutdown: bool | None = None
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
            },
        )
        self.latest_checkpoint_path = checkpoint_path
        checkpoint_entry = {
            "path": str(checkpoint_path),
            "env_step": self.env_step,
            "learner_step": self.learner_step,
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
        if self.command_queue is None:
            return
        if not force and self.learner_step <= 0:
            return
        self._put_command(
            {
                "type": "set_actor",
                "actor_state_dict": self.agent.actor_state_dict_cpu(),
                "env_step": self.env_step,
                "learner_step": self.learner_step,
                "observation_mode": self.config.observation.mode,
            }
        )

    def drain_messages(self, *, timeout: float = 0.25) -> int:
        if self.output_queue is None:
            return 0
        transition_count = 0
        try:
            message = self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return 0
        transition_count += self._handle_message(message)
        while True:
            try:
                message = self.output_queue.get_nowait()
            except queue.Empty:
                break
            transition_count += self._handle_message(message)
        return transition_count

    def drain_eval_results(self, *, timeout: float = 0.0) -> int:
        if self.eval_result_queue is None:
            return 0
        handled = 0
        try:
            result = self.eval_result_queue.get(timeout=timeout)
        except queue.Empty:
            return 0
        self._handle_eval_result(result)
        handled += 1
        while True:
            try:
                result = self.eval_result_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_eval_result(result)
            handled += 1
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
                batch = self.replay.sample(self.config.train.batch_size, device=self.device)
                update = self.agent.update(batch)
                self.learner_step += 1
                self._log_update(update)
            updates += updates_per_block
            self.next_update_step += self.config.train.update_model_interval
        return updates

    def maybe_schedule_eval(self) -> None:
        if self.command_queue is None or self.eval_in_flight:
            return
        if self.env_step < self.next_eval_step:
            return
        scheduled_step = self.next_eval_step
        self._put_command(
            {
                "type": "run_eval",
                "episodes": self.config.eval.episodes,
                "seed_base": self.config.eval.seed_base,
                "record_video": self.config.eval.record_video,
                "run_name": f"{self.run_name}_step_{scheduled_step:08d}",
                "learner_step": self.learner_step,
                "env_step": scheduled_step,
                "checkpoint_path": str(self.latest_checkpoint_path) if self.latest_checkpoint_path is not None else None,
            }
        )
        self.eval_in_flight = True
        if self.env_step >= self.next_checkpoint_step:
            self.pending_checkpoint_step = scheduled_step
            while self.next_checkpoint_step <= scheduled_step:
                self.next_checkpoint_step += self.config.train.checkpoint_interval_steps
        while self.next_eval_step <= scheduled_step:
            self.next_eval_step += self.config.train.eval_interval_steps

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
            self.writer.add_scalars_from_mapping("train/episode", summary, step=self.env_step)
            self._write_run_summary()
            return 0
        if message_type == "eval_result":
            return 0
        if message_type == "heartbeat":
            self.last_worker_heartbeat = dict(message)
            return 0
        if message_type == "fatal_error":
            self.termination_reason = self.termination_reason or "fatal_error"
            raise RuntimeError(str(message.get("error", "Worker reported an unspecified fatal error.")))
        return 0

    def _put_command(self, payload: Mapping[str, Any]) -> None:
        assert self.command_queue is not None
        while True:
            try:
                self.command_queue.put(dict(payload), timeout=0.25)
                return
            except queue.Full:
                if self.shutdown_event is not None and self.shutdown_event.is_set():
                    return

    def _log_update(self, update: SACUpdateResult) -> None:
        self.writer.add_scalars_from_mapping("train/update", asdict(update), step=self.learner_step)
        self.writer.add_scalar("train/replay_size", float(self.replay.size), step=self.learner_step)
        self.writer.add_scalar("train/env_step", float(self.env_step), step=self.learner_step)

    def _handle_eval_result(self, result: EvalResult | Mapping[str, Any]) -> None:
        payload = result if isinstance(result, EvalResult) else EvalResult(**dict(result))
        if payload.env_step < self.latest_eval_step:
            return
        self.latest_eval_step = payload.env_step
        self.latest_eval_summary = dict(payload.summary)
        self.latest_eval_summary_path = payload.summary_path
        self.eval_in_flight = False
        eval_entry = {
            "checkpoint_step": payload.checkpoint_step,
            "env_step": payload.env_step,
            "learner_step": payload.learner_step,
            "summary_path": payload.summary_path,
            "summary": dict(payload.summary),
            "timestamp": payload.timestamp,
        }
        self.eval_history.append(eval_entry)
        if self.latest_eval_summary:
            self.writer.add_scalars_from_mapping("eval", self.latest_eval_summary, step=payload.env_step)
        self._write_run_summary()
        if self.pending_checkpoint_step is not None and payload.checkpoint_step >= self.pending_checkpoint_step:
            self.save_checkpoint()
            self.pending_checkpoint_step = None

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
        write_json(
            self.paths.summary_json,
            {
                "run_name": self.run_name,
                "config_path": str(self.config_path),
                "device": str(self.device),
                "env_step": self.env_step,
                "learner_step": self.learner_step,
                "episode_count": self.episode_count,
                "replay_size": self.replay.size,
                "latest_checkpoint_path": None if self.latest_checkpoint_path is None else str(self.latest_checkpoint_path),
                "latest_eval_summary": self.latest_eval_summary,
                "latest_eval_summary_path": self.latest_eval_summary_path,
                "checkpoint_history": self.checkpoint_history,
                "eval_history": self.eval_history,
                "last_worker_heartbeat": self.last_worker_heartbeat,
                "termination_reason": self.termination_reason,
                "clean_shutdown": self.clean_shutdown,
                "worker_exit": self.worker_exit,
                "timestamp": _utc_now(),
                "observation_mode": self.config.observation.mode,
            },
        )

    def _load_actor_only(self, checkpoint_path: str | Path) -> None:
        payload = torch.load(Path(checkpoint_path).resolve(), map_location="cpu")
        actor_state = payload.get("actor_state_dict")
        if actor_state is None:
            raise RuntimeError(f"Checkpoint {checkpoint_path} does not contain actor_state_dict.")
        self.agent.actor.load_state_dict(actor_state)

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
        final_checkpoint = self.save_checkpoint(final=True)
        self._write_run_summary()
        return final_checkpoint

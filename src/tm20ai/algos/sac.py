from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import SACConfig
from ..models.full_actor_critic import (
    FullObservationActor,
    FullObservationCritic,
    random_shift_augmentation,
)
from ..models.lidar_actor_critic import LidarActor, LidarCritic
from ..train.features import ACTION_DIM, TELEMETRY_DIM
from ..train.replay import ReplaySample


@dataclass(slots=True)
class SACUpdateResult:
    actor_loss: float
    critic_loss: float
    alpha_loss: float
    alpha: float
    entropy: float
    target_q_mean: float
    q1_mean: float
    q2_mean: float


class SACAgent:
    def __init__(
        self,
        *,
        config: SACConfig,
        observation_mode: str,
        device: torch.device | str,
        observation_shape: tuple[int, ...],
        telemetry_dim: int = TELEMETRY_DIM,
        action_dim: int = ACTION_DIM,
    ) -> None:
        self.config = config
        self.observation_mode = observation_mode
        self.device = torch.device(device)
        self.observation_shape = observation_shape
        self.telemetry_dim = telemetry_dim
        self.action_dim = action_dim

        if self.observation_mode == "full":
            actor: nn.Module = FullObservationActor(
                observation_shape=tuple(int(value) for value in observation_shape),
                telemetry_dim=telemetry_dim,
                action_dim=action_dim,
            )
            critic1: nn.Module = FullObservationCritic(
                observation_shape=tuple(int(value) for value in observation_shape),
                telemetry_dim=telemetry_dim,
                action_dim=action_dim,
            )
            critic2: nn.Module = FullObservationCritic(
                observation_shape=tuple(int(value) for value in observation_shape),
                telemetry_dim=telemetry_dim,
                action_dim=action_dim,
            )
            target_critic1: nn.Module = FullObservationCritic(
                observation_shape=tuple(int(value) for value in observation_shape),
                telemetry_dim=telemetry_dim,
                action_dim=action_dim,
            )
            target_critic2: nn.Module = FullObservationCritic(
                observation_shape=tuple(int(value) for value in observation_shape),
                telemetry_dim=telemetry_dim,
                action_dim=action_dim,
            )
        elif self.observation_mode == "lidar":
            observation_dim = int(observation_shape[0])
            actor = LidarActor(observation_dim=observation_dim, action_dim=action_dim)
            critic1 = LidarCritic(observation_dim=observation_dim, action_dim=action_dim)
            critic2 = LidarCritic(observation_dim=observation_dim, action_dim=action_dim)
            target_critic1 = LidarCritic(observation_dim=observation_dim, action_dim=action_dim)
            target_critic2 = LidarCritic(observation_dim=observation_dim, action_dim=action_dim)
        else:
            raise ValueError(f"Unsupported observation mode: {self.observation_mode!r}")

        self.actor = actor.to(self.device)
        self.critic1 = critic1.to(self.device)
        self.critic2 = critic2.to(self.device)
        self.target_critic1 = target_critic1.to(self.device)
        self.target_critic2 = target_critic2.to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        for parameter in self.target_critic1.parameters():
            parameter.requires_grad_(False)
        for parameter in self.target_critic2.parameters():
            parameter.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=self.config.critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=self.config.critic_lr)

        if self.config.learn_entropy_coef:
            self.log_alpha = nn.Parameter(torch.log(torch.tensor(self.config.alpha, device=self.device, dtype=torch.float32)))
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.alpha_lr)
        else:
            self.log_alpha = torch.log(torch.tensor(self.config.alpha, device=self.device, dtype=torch.float32))
            self.alpha_optimizer = None

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp() if isinstance(self.log_alpha, nn.Parameter) else self.log_alpha.exp()

    def select_action(
        self,
        observation: np.ndarray | Tensor,
        telemetry: np.ndarray | Tensor | None,
        *,
        deterministic: bool,
        device: torch.device | str | None = None,
    ) -> np.ndarray:
        target_device = self.device if device is None else torch.device(device)
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=target_device)
        if observation_tensor.ndim == len(self.observation_shape):
            observation_tensor = observation_tensor.unsqueeze(0)
        telemetry_tensor = None if telemetry is None else torch.as_tensor(telemetry, dtype=torch.float32, device=target_device)
        if telemetry_tensor is not None and telemetry_tensor.ndim == 1:
            telemetry_tensor = telemetry_tensor.unsqueeze(0)
        with torch.no_grad():
            action = self._actor_act(observation_tensor, telemetry_tensor, deterministic=deterministic)
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def update(self, batch: ReplaySample) -> SACUpdateResult:
        obs = self._prepare_observation(batch.obs)
        next_obs = self._prepare_observation(batch.next_obs)
        with torch.no_grad():
            next_action, next_log_prob = self._actor_sample(next_obs, batch.next_telemetry, deterministic=False)
            target_q1 = self._critic_forward(self.target_critic1, next_obs, batch.next_telemetry, next_action)
            target_q2 = self._critic_forward(self.target_critic2, next_obs, batch.next_telemetry, next_action)
            min_target_q = torch.min(target_q1, target_q2)
            target_value = min_target_q - self.alpha.detach() * next_log_prob
            target_q = batch.reward + self.config.gamma * (1.0 - batch.done) * target_value

        q1 = self._critic_forward(self.critic1, obs, batch.telemetry, batch.action)
        q2 = self._critic_forward(self.critic2, obs, batch.telemetry, batch.action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic1_optimizer.zero_grad(set_to_none=True)
        self.critic2_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        pi_action, log_prob = self._actor_sample(obs, batch.telemetry, deterministic=False)
        q1_pi = self._critic_forward(self.critic1, obs, batch.telemetry, pi_action)
        q2_pi = self._critic_forward(self.critic2, obs, batch.telemetry, pi_action)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_prob - min_q_pi).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.config.learn_entropy_coef and self.alpha_optimizer is not None:
            assert isinstance(self.log_alpha, nn.Parameter)
            alpha_loss = -(self.log_alpha * (log_prob + self.config.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.zeros((), device=self.device)

        self.soft_update_targets()
        return SACUpdateResult(
            actor_loss=float(actor_loss.detach().cpu().item()),
            critic_loss=float(critic_loss.detach().cpu().item()),
            alpha_loss=float(alpha_loss.detach().cpu().item()),
            alpha=float(self.alpha.detach().cpu().item()),
            entropy=float((-log_prob).mean().detach().cpu().item()),
            target_q_mean=float(target_q.mean().detach().cpu().item()),
            q1_mean=float(q1.mean().detach().cpu().item()),
            q2_mean=float(q2.mean().detach().cpu().item()),
        )

    def soft_update_targets(self) -> None:
        with torch.no_grad():
            for target_param, source_param in zip(self.target_critic1.parameters(), self.critic1.parameters(), strict=True):
                target_param.mul_(self.config.polyak)
                target_param.add_((1.0 - self.config.polyak) * source_param)
            for target_param, source_param in zip(self.target_critic2.parameters(), self.critic2.parameters(), strict=True):
                target_param.mul_(self.config.polyak)
                target_param.add_((1.0 - self.config.polyak) * source_param)

    def state_dict(self) -> dict[str, Any]:
        return {
            "observation_mode": self.observation_mode,
            "actor_state_dict": self.actor.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "target_critic1_state_dict": self.target_critic1.state_dict(),
            "target_critic2_state_dict": self.target_critic2.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic1_optimizer_state_dict": self.critic1_optimizer.state_dict(),
            "critic2_optimizer_state_dict": self.critic2_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer_state_dict": None if self.alpha_optimizer is None else self.alpha_optimizer.state_dict(),
            "telemetry_dim": self.telemetry_dim,
            "action_dim": self.action_dim,
            "observation_shape": self.observation_shape,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.actor.load_state_dict(payload["actor_state_dict"])
        self.critic1.load_state_dict(payload["critic1_state_dict"])
        self.critic2.load_state_dict(payload["critic2_state_dict"])
        self.target_critic1.load_state_dict(payload["target_critic1_state_dict"])
        self.target_critic2.load_state_dict(payload["target_critic2_state_dict"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
        self.critic1_optimizer.load_state_dict(payload["critic1_optimizer_state_dict"])
        self.critic2_optimizer.load_state_dict(payload["critic2_optimizer_state_dict"])
        if self.config.learn_entropy_coef:
            assert isinstance(self.log_alpha, nn.Parameter)
            with torch.no_grad():
                self.log_alpha.copy_(payload["log_alpha"].detach().to(self.device))
        elif not isinstance(self.log_alpha, nn.Parameter):
            self.log_alpha = torch.log(torch.tensor(float(payload["log_alpha"].exp().item()), device=self.device))
        if self.config.learn_entropy_coef and self.alpha_optimizer is not None and payload.get("alpha_optimizer_state_dict") is not None:
            self.alpha_optimizer.load_state_dict(payload["alpha_optimizer_state_dict"])

    def actor_state_dict_cpu(self) -> dict[str, Tensor]:
        return {key: value.detach().cpu() for key, value in self.actor.state_dict().items()}

    def load_bc_warm_start(self, checkpoint_path: str | Path, *, init_mode: str) -> dict[str, Any]:
        if self.observation_mode != "full":
            raise RuntimeError("BC warm start is only supported for FULL SAC runs.")
        if init_mode not in {"actor_only", "actor_plus_critic_encoders"}:
            raise ValueError(f"Unsupported BC init_mode: {init_mode!r}")

        payload = torch.load(Path(checkpoint_path).resolve(), map_location="cpu")
        observation_mode = str(payload.get("observation_mode", ""))
        if observation_mode != "full":
            raise RuntimeError(
                f"BC checkpoint {checkpoint_path} is not compatible with FULL SAC warm start "
                f"(observation_mode={observation_mode!r})."
            )
        observation_shape = tuple(int(value) for value in payload.get("observation_shape", ()))
        telemetry_dim = int(payload.get("telemetry_dim", -1))
        action_dim = int(payload.get("action_dim", -1))
        if observation_shape and observation_shape != tuple(int(value) for value in self.observation_shape):
            raise RuntimeError(
                f"BC checkpoint {checkpoint_path} observation_shape {observation_shape!r} does not match "
                f"FULL SAC observation_shape {self.observation_shape!r}."
            )
        if telemetry_dim not in {-1, self.telemetry_dim}:
            raise RuntimeError(
                f"BC checkpoint {checkpoint_path} telemetry_dim {telemetry_dim} does not match FULL SAC "
                f"telemetry_dim {self.telemetry_dim}."
            )
        if action_dim not in {-1, self.action_dim}:
            raise RuntimeError(
                f"BC checkpoint {checkpoint_path} action_dim {action_dim} does not match FULL SAC "
                f"action_dim {self.action_dim}."
            )

        actor_state_dict = payload.get("actor_state_dict")
        if actor_state_dict is None:
            raise RuntimeError(f"BC checkpoint {checkpoint_path} does not contain actor_state_dict.")
        self.actor.load_state_dict(actor_state_dict)

        if init_mode == "actor_plus_critic_encoders":
            vision_encoder_state = {
                key.removeprefix("vision_encoder."): value
                for key, value in actor_state_dict.items()
                if key.startswith("vision_encoder.")
            }
            telemetry_encoder_state = {
                key.removeprefix("telemetry_encoder."): value
                for key, value in actor_state_dict.items()
                if key.startswith("telemetry_encoder.")
            }
            for critic in (self.critic1, self.critic2, self.target_critic1, self.target_critic2):
                assert isinstance(critic, FullObservationCritic)
                critic.vision_encoder.load_state_dict(vision_encoder_state)
                critic.telemetry_encoder.load_state_dict(telemetry_encoder_state)

        return {
            "checkpoint_path": str(Path(checkpoint_path).resolve()),
            "checkpoint_kind": payload.get("checkpoint_kind"),
            "map_uid": payload.get("map_uid"),
            "demo_root": payload.get("demo_root"),
            "observation_mode": observation_mode,
            "observation_shape": observation_shape or tuple(int(value) for value in self.observation_shape),
            "telemetry_dim": self.telemetry_dim if telemetry_dim == -1 else telemetry_dim,
            "action_dim": self.action_dim if action_dim == -1 else action_dim,
            "epoch": payload.get("epoch"),
            "train_loss": payload.get("train_loss"),
            "validation_loss": payload.get("validation_loss"),
            "config_snapshot": payload.get("config_snapshot"),
        }

    def _prepare_observation(self, observation: Tensor) -> Tensor:
        if self.observation_mode == "full":
            return random_shift_augmentation(observation)
        return observation.float()

    def _actor_act(self, observation: Tensor, telemetry: Tensor | None, *, deterministic: bool) -> Tensor:
        if self.observation_mode == "full":
            assert telemetry is not None
            return self.actor.act(observation, telemetry, deterministic=deterministic)
        assert isinstance(self.actor, LidarActor)
        return self.actor.act(observation, deterministic=deterministic)

    def _actor_sample(
        self,
        observation: Tensor,
        telemetry: Tensor | None,
        *,
        deterministic: bool,
    ) -> tuple[Tensor, Tensor]:
        if self.observation_mode == "full":
            assert telemetry is not None
            action, log_prob = self.actor.sample(observation, telemetry, deterministic=deterministic)
            assert log_prob is not None
            return action, log_prob
        assert isinstance(self.actor, LidarActor)
        action, log_prob = self.actor.sample(observation, deterministic=deterministic)
        assert log_prob is not None
        return action, log_prob

    def _critic_forward(
        self,
        critic: nn.Module,
        observation: Tensor,
        telemetry: Tensor | None,
        action: Tensor,
    ) -> Tensor:
        if self.observation_mode == "full":
            assert telemetry is not None
            return critic(observation, telemetry, action)
        return critic(observation, action)

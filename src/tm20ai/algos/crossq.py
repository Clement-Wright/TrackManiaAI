from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import CrossQConfig, SACConfig
from ..models.full_actor_critic import (
    FullObservationActor,
    TelemetryEncoder,
    VisionEncoder,
    normalize_image_batch,
    random_shift_augmentation,
)
from ..train.features import ACTION_DIM, TELEMETRY_DIM
from ..train.replay import ReplaySample


class _CrossQCritic(nn.Module):
    def __init__(
        self,
        *,
        observation_shape: tuple[int, int, int],
        telemetry_dim: int = TELEMETRY_DIM,
        action_dim: int = ACTION_DIM,
    ) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoder(observation_shape)
        self.telemetry_encoder = TelemetryEncoder(telemetry_dim)
        fused_dim = 512 + 64 + action_dim
        self.fc1 = nn.Linear(fused_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc_out = nn.Linear(256, 1)

    def encode(self, observation: Tensor, telemetry: Tensor, action: Tensor) -> Tensor:
        observation = normalize_image_batch(observation)
        telemetry = telemetry.float()
        return torch.cat(
            (self.vision_encoder(observation), self.telemetry_encoder(telemetry), action.float()),
            dim=-1,
        )

    def forward_fused(self, fused: Tensor) -> Tensor:
        hidden = self.fc1(fused)
        hidden = self.bn1(hidden)
        hidden = F.relu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.bn2(hidden)
        hidden = F.relu(hidden)
        return self.fc_out(hidden)

    def forward(self, observation: Tensor, telemetry: Tensor, action: Tensor) -> Tensor:
        return self.forward_fused(self.encode(observation, telemetry, action))

    def forward_pair(
        self,
        observation: Tensor,
        telemetry: Tensor,
        action: Tensor,
        next_observation: Tensor,
        next_telemetry: Tensor,
        next_action: Tensor,
    ) -> tuple[Tensor, Tensor]:
        current_fused = self.encode(observation, telemetry, action)
        next_fused = self.encode(next_observation, next_telemetry, next_action)
        pair = torch.cat((current_fused, next_fused), dim=0)
        values = self.forward_fused(pair)
        batch_size = observation.shape[0]
        return values[:batch_size], values[batch_size:]


@dataclass(slots=True)
class CrossQUpdateResult:
    actor_loss: float
    critic_loss: float
    alpha_loss: float
    alpha: float
    entropy: float
    target_q_mean: float
    q1_mean: float
    q2_mean: float


class CrossQAgent:
    algorithm_name = "crossq"

    def __init__(
        self,
        *,
        sac_config: SACConfig,
        crossq_config: CrossQConfig,
        observation_mode: str,
        device: torch.device | str,
        observation_shape: tuple[int, ...],
        telemetry_dim: int = TELEMETRY_DIM,
        action_dim: int = ACTION_DIM,
    ) -> None:
        if observation_mode != "full":
            raise ValueError("CrossQ is only implemented for full-observation runs in this pass.")
        self.config = sac_config
        self.crossq_config = crossq_config
        self.observation_mode = observation_mode
        self.device = torch.device(device)
        self.observation_shape = observation_shape
        self.telemetry_dim = telemetry_dim
        self.action_dim = action_dim
        self.share_encoders = bool(crossq_config.share_encoders)

        self.actor = FullObservationActor(
            observation_shape=tuple(int(value) for value in observation_shape),
            telemetry_dim=telemetry_dim,
            action_dim=action_dim,
        ).to(self.device)
        self.critic1 = _CrossQCritic(
            observation_shape=tuple(int(value) for value in observation_shape),
            telemetry_dim=telemetry_dim,
            action_dim=action_dim,
        ).to(self.device)
        self.critic2 = _CrossQCritic(
            observation_shape=tuple(int(value) for value in observation_shape),
            telemetry_dim=telemetry_dim,
            action_dim=action_dim,
        ).to(self.device)

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
            action = self.actor.act(observation_tensor, telemetry_tensor, deterministic=deterministic)
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def update(self, batch: ReplaySample) -> CrossQUpdateResult:
        obs = random_shift_augmentation(batch.obs)
        next_obs = random_shift_augmentation(batch.next_obs)
        next_action, next_log_prob = self.actor.sample(next_obs, batch.next_telemetry, deterministic=False)

        q1, q1_next = self.critic1.forward_pair(obs, batch.telemetry, batch.action, next_obs, batch.next_telemetry, next_action)
        q2, q2_next = self.critic2.forward_pair(obs, batch.telemetry, batch.action, next_obs, batch.next_telemetry, next_action)
        target_q = batch.reward + self.config.gamma * (1.0 - batch.done) * (
            torch.min(q1_next.detach(), q2_next.detach()) - self.alpha.detach() * next_log_prob.detach()
        )
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic1_optimizer.zero_grad(set_to_none=True)
        self.critic2_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        pi_action, log_prob = self.actor.sample(obs, batch.telemetry, deterministic=False)
        q1_pi = self.critic1(obs, batch.telemetry, pi_action)
        q2_pi = self.critic2(obs, batch.telemetry, pi_action)
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

        return CrossQUpdateResult(
            actor_loss=float(actor_loss.detach().cpu().item()),
            critic_loss=float(critic_loss.detach().cpu().item()),
            alpha_loss=float(alpha_loss.detach().cpu().item()),
            alpha=float(self.alpha.detach().cpu().item()),
            entropy=float((-log_prob).mean().detach().cpu().item()),
            target_q_mean=float(target_q.mean().detach().cpu().item()),
            q1_mean=float(q1.mean().detach().cpu().item()),
            q2_mean=float(q2.mean().detach().cpu().item()),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "algorithm": "crossq",
            "observation_mode": self.observation_mode,
            "actor_state_dict": self.actor.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic1_optimizer_state_dict": self.critic1_optimizer.state_dict(),
            "critic2_optimizer_state_dict": self.critic2_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer_state_dict": None if self.alpha_optimizer is None else self.alpha_optimizer.state_dict(),
            "telemetry_dim": self.telemetry_dim,
            "action_dim": self.action_dim,
            "observation_shape": self.observation_shape,
            "share_encoders": self.share_encoders,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.actor.load_state_dict(payload["actor_state_dict"])
        self.critic1.load_state_dict(payload["critic1_state_dict"])
        self.critic2.load_state_dict(payload["critic2_state_dict"])
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
        if init_mode not in {"actor_only", "actor_plus_critic_encoders"}:
            raise ValueError(f"Unsupported BC init_mode: {init_mode!r}")
        payload = torch.load(Path(checkpoint_path).resolve(), map_location="cpu")
        observation_mode = str(payload.get("observation_mode", ""))
        if observation_mode != "full":
            raise RuntimeError(
                f"BC checkpoint {checkpoint_path} is not compatible with full CrossQ warm start "
                f"(observation_mode={observation_mode!r})."
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
            for critic in (self.critic1, self.critic2):
                critic.vision_encoder.load_state_dict(vision_encoder_state)
                critic.telemetry_encoder.load_state_dict(telemetry_encoder_state)
        return {
            "checkpoint_path": str(Path(checkpoint_path).resolve()),
            "checkpoint_kind": payload.get("checkpoint_kind"),
            "map_uid": payload.get("map_uid"),
            "demo_root": payload.get("demo_root"),
            "observation_mode": observation_mode,
            "observation_shape": tuple(int(value) for value in payload.get("observation_shape", self.observation_shape)),
            "telemetry_dim": int(payload.get("telemetry_dim", self.telemetry_dim)),
            "action_dim": int(payload.get("action_dim", self.action_dim)),
            "epoch": payload.get("epoch"),
            "train_loss": payload.get("train_loss"),
            "validation_loss": payload.get("validation_loss"),
            "config_snapshot": payload.get("config_snapshot"),
        }

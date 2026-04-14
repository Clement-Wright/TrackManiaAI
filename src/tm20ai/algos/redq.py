from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import REDQConfig, SACConfig
from ..models.full_actor_critic import (
    FullObservationActor,
    FullObservationCritic,
    TelemetryEncoder,
    VisionEncoder,
    normalize_image_batch,
    random_shift_augmentation,
)
from ..models.lidar_actor_critic import LidarActor, LidarCritic
from ..train.features import ACTION_DIM, TELEMETRY_DIM
from ..train.replay import ReplaySample


class _SharedEncoderFullObservationCritic(nn.Module):
    def __init__(
        self,
        *,
        vision_encoder: VisionEncoder,
        telemetry_encoder: TelemetryEncoder,
        action_dim: int = ACTION_DIM,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.telemetry_encoder = telemetry_encoder
        fused_dim = 512 + 64 + action_dim
        self.q_network = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward_encoded(self, vision_latent: Tensor, telemetry_latent: Tensor, action: Tensor) -> Tensor:
        fused = torch.cat((vision_latent, telemetry_latent, action.float()), dim=-1)
        return self.q_network(fused)

    def forward(self, observation: Tensor, telemetry: Tensor, action: Tensor) -> Tensor:
        observation = normalize_image_batch(observation)
        telemetry = telemetry.float()
        return self.forward_encoded(
            self.vision_encoder(observation),
            self.telemetry_encoder(telemetry),
            action,
        )


@dataclass(slots=True)
class REDQCriticUpdateResult:
    critic_loss: float
    alpha: float
    target_q_mean: float
    q_mean: float
    subset_target_q_mean: float
    critic_updates_since_actor: int


@dataclass(slots=True)
class REDQActorUpdateResult:
    actor_loss: float
    alpha_loss: float
    alpha: float
    entropy: float
    q_pi_mean: float


class REDQSACAgent:
    def __init__(
        self,
        *,
        sac_config: SACConfig,
        redq_config: REDQConfig,
        observation_mode: str,
        device: torch.device | str,
        observation_shape: tuple[int, ...],
        telemetry_dim: int = TELEMETRY_DIM,
        action_dim: int = ACTION_DIM,
    ) -> None:
        self.sac_config = sac_config
        self.redq_config = redq_config
        self.observation_mode = observation_mode
        self.device = torch.device(device)
        self.observation_shape = observation_shape
        self.telemetry_dim = telemetry_dim
        self.action_dim = action_dim
        self.n_critics = int(redq_config.n_critics)
        self.m_subset = int(redq_config.m_subset)
        self.q_updates_per_policy_update = int(redq_config.q_updates_per_policy_update)
        self.share_encoders_requested = bool(redq_config.share_encoders)
        self.share_encoders = bool(redq_config.share_encoders and observation_mode == "full")
        self.shared_critic_vision_encoder: VisionEncoder | None = None
        self.shared_critic_telemetry_encoder: TelemetryEncoder | None = None
        self.target_shared_critic_vision_encoder: VisionEncoder | None = None
        self.target_shared_critic_telemetry_encoder: TelemetryEncoder | None = None

        if self.observation_mode == "full":
            actor: nn.Module = FullObservationActor(
                observation_shape=tuple(int(value) for value in observation_shape),
                telemetry_dim=telemetry_dim,
                action_dim=action_dim,
            )
            if self.share_encoders:
                shared_vision_encoder = VisionEncoder(tuple(int(value) for value in observation_shape))
                shared_telemetry_encoder = TelemetryEncoder(telemetry_dim)
                target_shared_vision_encoder = VisionEncoder(tuple(int(value) for value in observation_shape))
                target_shared_telemetry_encoder = TelemetryEncoder(telemetry_dim)
                critics = nn.ModuleList(
                    [
                        _SharedEncoderFullObservationCritic(
                            vision_encoder=shared_vision_encoder,
                            telemetry_encoder=shared_telemetry_encoder,
                            action_dim=action_dim,
                        )
                        for _ in range(self.n_critics)
                    ]
                )
                target_critics = nn.ModuleList(
                    [
                        _SharedEncoderFullObservationCritic(
                            vision_encoder=target_shared_vision_encoder,
                            telemetry_encoder=target_shared_telemetry_encoder,
                            action_dim=action_dim,
                        )
                        for _ in range(self.n_critics)
                    ]
                )
                self.shared_critic_vision_encoder = shared_vision_encoder
                self.shared_critic_telemetry_encoder = shared_telemetry_encoder
                self.target_shared_critic_vision_encoder = target_shared_vision_encoder
                self.target_shared_critic_telemetry_encoder = target_shared_telemetry_encoder
            else:
                critics = nn.ModuleList(
                    [
                        FullObservationCritic(
                            observation_shape=tuple(int(value) for value in observation_shape),
                            telemetry_dim=telemetry_dim,
                            action_dim=action_dim,
                        )
                        for _ in range(self.n_critics)
                    ]
                )
                target_critics = nn.ModuleList(
                    [
                        FullObservationCritic(
                            observation_shape=tuple(int(value) for value in observation_shape),
                            telemetry_dim=telemetry_dim,
                            action_dim=action_dim,
                        )
                        for _ in range(self.n_critics)
                    ]
                )
        elif self.observation_mode == "lidar":
            observation_dim = int(observation_shape[0])
            actor = LidarActor(observation_dim=observation_dim, action_dim=action_dim)
            critics = nn.ModuleList(
                [LidarCritic(observation_dim=observation_dim, action_dim=action_dim) for _ in range(self.n_critics)]
            )
            target_critics = nn.ModuleList(
                [LidarCritic(observation_dim=observation_dim, action_dim=action_dim) for _ in range(self.n_critics)]
            )
        else:
            raise ValueError(f"Unsupported observation mode: {self.observation_mode!r}")

        self.actor = actor.to(self.device)
        self.critics = critics.to(self.device)
        self.target_critics = target_critics.to(self.device)
        for target_critic, critic in zip(self.target_critics, self.critics, strict=True):
            target_critic.load_state_dict(critic.state_dict())
        for target_critic in self.target_critics:
            for parameter in target_critic.parameters():
                parameter.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.sac_config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critics.parameters(), lr=self.sac_config.critic_lr)

        if self.sac_config.learn_entropy_coef:
            self.log_alpha = nn.Parameter(
                torch.log(torch.tensor(self.sac_config.alpha, device=self.device, dtype=torch.float32))
            )
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.sac_config.alpha_lr)
        else:
            self.log_alpha = torch.log(torch.tensor(self.sac_config.alpha, device=self.device, dtype=torch.float32))
            self.alpha_optimizer = None

        self.critic_updates_since_actor = 0

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

    def update_critics(self, batch: ReplaySample) -> REDQCriticUpdateResult:
        obs = self._prepare_observation(batch.obs)
        next_obs = self._prepare_observation(batch.next_obs)
        encoded_obs = self._encode_full_critic_inputs(obs, batch.telemetry, target=False)
        encoded_next_obs = self._encode_full_critic_inputs(next_obs, batch.next_telemetry, target=True)

        with torch.no_grad():
            next_action, next_log_prob = self._actor_sample(next_obs, batch.next_telemetry, deterministic=False)
            subset_indices = torch.randperm(self.n_critics)[: self.m_subset].tolist()
            target_qs = torch.stack(
                [
                    self._critic_forward(
                        self.target_critics[index],
                        next_obs,
                        batch.next_telemetry,
                        next_action,
                        encoded_full=encoded_next_obs,
                    )
                    for index in subset_indices
                ],
                dim=0,
            )
            min_target_q = target_qs.min(dim=0).values
            target_value = min_target_q - self.alpha.detach() * next_log_prob
            target_q = batch.reward + self.sac_config.gamma * (1.0 - batch.done) * target_value

        qs = torch.stack(
            [
                self._critic_forward(
                    critic,
                    obs,
                    batch.telemetry,
                    batch.action,
                    encoded_full=encoded_obs,
                )
                for critic in self.critics
            ],
            dim=0,
        )
        critic_loss = F.mse_loss(qs, target_q.expand_as(qs))

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        self.soft_update_targets()
        self.critic_updates_since_actor += 1

        return REDQCriticUpdateResult(
            critic_loss=float(critic_loss.detach().cpu().item()),
            alpha=float(self.alpha.detach().cpu().item()),
            target_q_mean=float(target_q.mean().detach().cpu().item()),
            q_mean=float(qs.mean().detach().cpu().item()),
            subset_target_q_mean=float(target_qs.mean().detach().cpu().item()),
            critic_updates_since_actor=self.critic_updates_since_actor,
        )

    def maybe_update_actor_and_alpha(self, batch: ReplaySample) -> REDQActorUpdateResult | None:
        if self.critic_updates_since_actor < self.q_updates_per_policy_update:
            return None
        self.critic_updates_since_actor = 0

        obs = self._prepare_observation(batch.obs)
        encoded_obs = self._encode_full_critic_inputs(obs, batch.telemetry, target=False)
        pi_action, log_prob = self._actor_sample(obs, batch.telemetry, deterministic=False)
        q_pi = torch.stack(
            [
                self._critic_forward(
                    critic,
                    obs,
                    batch.telemetry,
                    pi_action,
                    encoded_full=encoded_obs,
                )
                for critic in self.critics
            ],
            dim=0,
        ).mean(dim=0)
        actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.sac_config.learn_entropy_coef and self.alpha_optimizer is not None:
            assert isinstance(self.log_alpha, nn.Parameter)
            alpha_loss = -(self.log_alpha * (log_prob + self.sac_config.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.zeros((), device=self.device)

        return REDQActorUpdateResult(
            actor_loss=float(actor_loss.detach().cpu().item()),
            alpha_loss=float(alpha_loss.detach().cpu().item()),
            alpha=float(self.alpha.detach().cpu().item()),
            entropy=float((-log_prob).mean().detach().cpu().item()),
            q_pi_mean=float(q_pi.mean().detach().cpu().item()),
        )

    def soft_update_targets(self) -> None:
        with torch.no_grad():
            if self.observation_mode == "full" and self.share_encoders:
                assert self.shared_critic_vision_encoder is not None
                assert self.shared_critic_telemetry_encoder is not None
                assert self.target_shared_critic_vision_encoder is not None
                assert self.target_shared_critic_telemetry_encoder is not None
                self._soft_update_module(self.target_shared_critic_vision_encoder, self.shared_critic_vision_encoder)
                self._soft_update_module(
                    self.target_shared_critic_telemetry_encoder,
                    self.shared_critic_telemetry_encoder,
                )
                for target_critic, critic in zip(self.target_critics, self.critics, strict=True):
                    assert isinstance(target_critic, _SharedEncoderFullObservationCritic)
                    assert isinstance(critic, _SharedEncoderFullObservationCritic)
                    self._soft_update_module(target_critic.q_network, critic.q_network)
                return

            for target_critic, critic in zip(self.target_critics, self.critics, strict=True):
                self._soft_update_module(target_critic, critic)

    def state_dict(self) -> dict[str, Any]:
        return {
            "algorithm": "redq",
            "observation_mode": self.observation_mode,
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dicts": [critic.state_dict() for critic in self.critics],
            "target_critic_state_dicts": [critic.state_dict() for critic in self.target_critics],
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer_state_dict": None if self.alpha_optimizer is None else self.alpha_optimizer.state_dict(),
            "telemetry_dim": self.telemetry_dim,
            "action_dim": self.action_dim,
            "observation_shape": self.observation_shape,
            "critic_updates_since_actor": self.critic_updates_since_actor,
            "n_critics": self.n_critics,
            "m_subset": self.m_subset,
            "q_updates_per_policy_update": self.q_updates_per_policy_update,
            "share_encoders": self.share_encoders_requested,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        critic_state_dicts = list(payload.get("critic_state_dicts", []))
        target_critic_state_dicts = list(payload.get("target_critic_state_dicts", []))
        if len(critic_state_dicts) != self.n_critics:
            raise RuntimeError(
                f"REDQ checkpoint exposes {len(critic_state_dicts)} critics, expected {self.n_critics}."
            )
        if len(target_critic_state_dicts) != self.n_critics:
            raise RuntimeError(
                f"REDQ checkpoint exposes {len(target_critic_state_dicts)} target critics, expected {self.n_critics}."
            )

        self.actor.load_state_dict(payload["actor_state_dict"])
        for critic, critic_state in zip(self.critics, critic_state_dicts, strict=True):
            critic.load_state_dict(critic_state)
        for target_critic, critic_state in zip(self.target_critics, target_critic_state_dicts, strict=True):
            target_critic.load_state_dict(critic_state)
        self.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
        if self.sac_config.learn_entropy_coef:
            assert isinstance(self.log_alpha, nn.Parameter)
            with torch.no_grad():
                self.log_alpha.copy_(payload["log_alpha"].detach().to(self.device))
        elif not isinstance(self.log_alpha, nn.Parameter):
            self.log_alpha = torch.log(torch.tensor(float(payload["log_alpha"].exp().item()), device=self.device))
        if self.sac_config.learn_entropy_coef and self.alpha_optimizer is not None and payload.get("alpha_optimizer_state_dict") is not None:
            self.alpha_optimizer.load_state_dict(payload["alpha_optimizer_state_dict"])
        self.critic_updates_since_actor = int(payload.get("critic_updates_since_actor", 0))

    def actor_state_dict_cpu(self) -> dict[str, Tensor]:
        return {key: value.detach().cpu() for key, value in self.actor.state_dict().items()}

    def load_bc_warm_start(self, checkpoint_path: str | Path, *, init_mode: str) -> dict[str, Any]:
        if self.observation_mode != "full":
            raise RuntimeError("BC warm start is only supported for FULL REDQ runs.")
        if init_mode not in {"actor_only", "actor_plus_critic_encoders"}:
            raise ValueError(f"Unsupported BC init_mode: {init_mode!r}")

        payload = torch.load(Path(checkpoint_path).resolve(), map_location="cpu")
        observation_mode = str(payload.get("observation_mode", ""))
        if observation_mode != "full":
            raise RuntimeError(
                f"BC checkpoint {checkpoint_path} is not compatible with FULL REDQ warm start "
                f"(observation_mode={observation_mode!r})."
            )
        observation_shape = tuple(int(value) for value in payload.get("observation_shape", ()))
        telemetry_dim = int(payload.get("telemetry_dim", -1))
        action_dim = int(payload.get("action_dim", -1))
        if observation_shape and observation_shape != tuple(int(value) for value in self.observation_shape):
            raise RuntimeError(
                f"BC checkpoint {checkpoint_path} observation_shape {observation_shape!r} does not match "
                f"FULL REDQ observation_shape {self.observation_shape!r}."
            )
        if telemetry_dim not in {-1, self.telemetry_dim}:
            raise RuntimeError(
                f"BC checkpoint {checkpoint_path} telemetry_dim {telemetry_dim} does not match FULL REDQ "
                f"telemetry_dim {self.telemetry_dim}."
            )
        if action_dim not in {-1, self.action_dim}:
            raise RuntimeError(
                f"BC checkpoint {checkpoint_path} action_dim {action_dim} does not match FULL REDQ "
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
            if self.share_encoders:
                assert self.shared_critic_vision_encoder is not None
                assert self.shared_critic_telemetry_encoder is not None
                assert self.target_shared_critic_vision_encoder is not None
                assert self.target_shared_critic_telemetry_encoder is not None
                self.shared_critic_vision_encoder.load_state_dict(vision_encoder_state)
                self.shared_critic_telemetry_encoder.load_state_dict(telemetry_encoder_state)
                self.target_shared_critic_vision_encoder.load_state_dict(vision_encoder_state)
                self.target_shared_critic_telemetry_encoder.load_state_dict(telemetry_encoder_state)
            else:
                for critic in (*self.critics, *self.target_critics):
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

    def _encode_full_critic_inputs(
        self,
        observation: Tensor,
        telemetry: Tensor | None,
        *,
        target: bool,
    ) -> tuple[Tensor, Tensor] | None:
        if self.observation_mode != "full" or not self.share_encoders:
            return None
        assert telemetry is not None
        vision_encoder = self.target_shared_critic_vision_encoder if target else self.shared_critic_vision_encoder
        telemetry_encoder = self.target_shared_critic_telemetry_encoder if target else self.shared_critic_telemetry_encoder
        assert vision_encoder is not None
        assert telemetry_encoder is not None
        normalized_observation = normalize_image_batch(observation)
        return vision_encoder(normalized_observation), telemetry_encoder(telemetry.float())

    def _critic_forward(
        self,
        critic: nn.Module,
        observation: Tensor,
        telemetry: Tensor | None,
        action: Tensor,
        *,
        encoded_full: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        if self.observation_mode == "full":
            assert telemetry is not None
            if self.share_encoders:
                assert encoded_full is not None
                assert isinstance(critic, _SharedEncoderFullObservationCritic)
                vision_latent, telemetry_latent = encoded_full
                return critic.forward_encoded(vision_latent, telemetry_latent, action)
            return critic(observation, telemetry, action)
        return critic(observation, action)

    def _soft_update_module(self, target_module: nn.Module, source_module: nn.Module) -> None:
        for target_param, source_param in zip(target_module.parameters(), source_module.parameters(), strict=True):
            target_param.mul_(self.sac_config.polyak)
            target_param.add_((1.0 - self.sac_config.polyak) * source_param)

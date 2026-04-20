from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import DroQConfig, SACConfig
from ..models.full_actor_critic import (
    FullObservationActor,
    FullObservationCritic,
    TelemetryEncoder,
    VisionEncoder,
    normalize_image_batch,
    random_shift_augmentation,
)
from ..train.features import ACTION_DIM, TELEMETRY_DIM
from ..train.replay import ReplaySample


class _DroQSharedEncoderCritic(nn.Module):
    def __init__(
        self,
        *,
        vision_encoder: VisionEncoder,
        telemetry_encoder: TelemetryEncoder,
        action_dim: int = ACTION_DIM,
        dropout_probability: float = 0.01,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.telemetry_encoder = telemetry_encoder
        fused_dim = 512 + 64 + action_dim
        self.q_network = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
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


class _DroQFullObservationCritic(FullObservationCritic):
    def __init__(
        self,
        *,
        observation_shape: tuple[int, int, int] = (4, 64, 64),
        telemetry_dim: int = TELEMETRY_DIM,
        action_dim: int = ACTION_DIM,
        dropout_probability: float = 0.01,
    ) -> None:
        super().__init__(
            observation_shape=observation_shape,
            telemetry_dim=telemetry_dim,
            action_dim=action_dim,
        )
        fused_dim = 512 + 64 + action_dim
        self.q_network = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(256, 1),
        )


@dataclass(slots=True)
class DroQCriticUpdateResult:
    critic_loss: float
    alpha: float
    target_q_mean: float
    q_mean: float
    subset_target_q_mean: float
    critic_updates_since_actor: int


@dataclass(slots=True)
class DroQActorUpdateResult:
    actor_loss: float
    alpha_loss: float
    alpha: float
    entropy: float
    q_pi_mean: float


class DroQSACAgent:
    algorithm_name = "droq"

    def __init__(
        self,
        *,
        sac_config: SACConfig,
        droq_config: DroQConfig,
        observation_mode: str,
        device: torch.device | str,
        observation_shape: tuple[int, ...],
        telemetry_dim: int = TELEMETRY_DIM,
        action_dim: int = ACTION_DIM,
    ) -> None:
        if observation_mode != "full":
            raise ValueError("DroQ is only implemented for full-observation runs in this pass.")
        self.sac_config = sac_config
        self.droq_config = droq_config
        self.observation_mode = observation_mode
        self.device = torch.device(device)
        self.observation_shape = observation_shape
        self.telemetry_dim = telemetry_dim
        self.action_dim = action_dim
        self.n_critics = int(droq_config.n_critics)
        self.m_subset = int(droq_config.m_subset)
        self.q_updates_per_policy_update = int(droq_config.q_updates_per_policy_update)
        self.share_encoders_requested = bool(droq_config.share_encoders)
        self.share_encoders = bool(droq_config.share_encoders)
        self.dropout_probability = float(droq_config.dropout_probability)
        self.shared_critic_vision_encoder: VisionEncoder | None = None
        self.shared_critic_telemetry_encoder: TelemetryEncoder | None = None
        self.target_shared_critic_vision_encoder: VisionEncoder | None = None
        self.target_shared_critic_telemetry_encoder: TelemetryEncoder | None = None

        self.actor = FullObservationActor(
            observation_shape=tuple(int(value) for value in observation_shape),
            telemetry_dim=telemetry_dim,
            action_dim=action_dim,
        ).to(self.device)

        if self.share_encoders:
            shared_vision_encoder = VisionEncoder(tuple(int(value) for value in observation_shape))
            shared_telemetry_encoder = TelemetryEncoder(telemetry_dim)
            target_shared_vision_encoder = VisionEncoder(tuple(int(value) for value in observation_shape))
            target_shared_telemetry_encoder = TelemetryEncoder(telemetry_dim)
            self.critics = nn.ModuleList(
                [
                    _DroQSharedEncoderCritic(
                        vision_encoder=shared_vision_encoder,
                        telemetry_encoder=shared_telemetry_encoder,
                        action_dim=action_dim,
                        dropout_probability=self.dropout_probability,
                    )
                    for _ in range(self.n_critics)
                ]
            ).to(self.device)
            self.target_critics = nn.ModuleList(
                [
                    _DroQSharedEncoderCritic(
                        vision_encoder=target_shared_vision_encoder,
                        telemetry_encoder=target_shared_telemetry_encoder,
                        action_dim=action_dim,
                        dropout_probability=self.dropout_probability,
                    )
                    for _ in range(self.n_critics)
                ]
            ).to(self.device)
            self.shared_critic_vision_encoder = shared_vision_encoder
            self.shared_critic_telemetry_encoder = shared_telemetry_encoder
            self.target_shared_critic_vision_encoder = target_shared_vision_encoder
            self.target_shared_critic_telemetry_encoder = target_shared_telemetry_encoder
        else:
            self.critics = nn.ModuleList(
                [
                    _DroQFullObservationCritic(
                        observation_shape=tuple(int(value) for value in observation_shape),
                        telemetry_dim=telemetry_dim,
                        action_dim=action_dim,
                        dropout_probability=self.dropout_probability,
                    )
                    for _ in range(self.n_critics)
                ]
            ).to(self.device)
            self.target_critics = nn.ModuleList(
                [
                    _DroQFullObservationCritic(
                        observation_shape=tuple(int(value) for value in observation_shape),
                        telemetry_dim=telemetry_dim,
                        action_dim=action_dim,
                        dropout_probability=self.dropout_probability,
                    )
                    for _ in range(self.n_critics)
                ]
            ).to(self.device)

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
            action = self.actor.act(observation_tensor, telemetry_tensor, deterministic=deterministic)
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32)

    def update_critics(self, batch: ReplaySample) -> DroQCriticUpdateResult:
        obs = self._prepare_observation(batch.obs)
        next_obs = self._prepare_observation(batch.next_obs)
        encoded_obs = self._encode_full_critic_inputs(obs, batch.telemetry, target=False)
        encoded_next_obs = self._encode_full_critic_inputs(next_obs, batch.next_telemetry, target=True)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs, batch.next_telemetry, deterministic=False)
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

        return DroQCriticUpdateResult(
            critic_loss=float(critic_loss.detach().cpu().item()),
            alpha=float(self.alpha.detach().cpu().item()),
            target_q_mean=float(target_q.mean().detach().cpu().item()),
            q_mean=float(qs.mean().detach().cpu().item()),
            subset_target_q_mean=float(target_qs.mean().detach().cpu().item()),
            critic_updates_since_actor=self.critic_updates_since_actor,
        )

    def maybe_update_actor_and_alpha(self, batch: ReplaySample) -> DroQActorUpdateResult | None:
        if self.critic_updates_since_actor < self.q_updates_per_policy_update:
            return None
        self.critic_updates_since_actor = 0

        obs = self._prepare_observation(batch.obs)
        encoded_obs = self._encode_full_critic_inputs(obs, batch.telemetry, target=False)
        pi_action, log_prob = self.actor.sample(obs, batch.telemetry, deterministic=False)
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

        return DroQActorUpdateResult(
            actor_loss=float(actor_loss.detach().cpu().item()),
            alpha_loss=float(alpha_loss.detach().cpu().item()),
            alpha=float(self.alpha.detach().cpu().item()),
            entropy=float((-log_prob).mean().detach().cpu().item()),
            q_pi_mean=float(q_pi.mean().detach().cpu().item()),
        )

    def soft_update_targets(self) -> None:
        with torch.no_grad():
            if self.share_encoders:
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
                    assert isinstance(target_critic, _DroQSharedEncoderCritic)
                    assert isinstance(critic, _DroQSharedEncoderCritic)
                    self._soft_update_module(target_critic.q_network, critic.q_network)
                return

            for target_critic, critic in zip(self.target_critics, self.critics, strict=True):
                self._soft_update_module(target_critic, critic)

    def state_dict(self) -> dict[str, Any]:
        return {
            "algorithm": "droq",
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
            "dropout_probability": self.dropout_probability,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        critic_state_dicts = list(payload.get("critic_state_dicts", []))
        target_critic_state_dicts = list(payload.get("target_critic_state_dicts", []))
        if len(critic_state_dicts) != self.n_critics:
            raise RuntimeError(
                f"DroQ checkpoint exposes {len(critic_state_dicts)} critics, expected {self.n_critics}."
            )
        if len(target_critic_state_dicts) != self.n_critics:
            raise RuntimeError(
                f"DroQ checkpoint exposes {len(target_critic_state_dicts)} target critics, expected {self.n_critics}."
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
        if init_mode not in {"actor_only", "actor_plus_critic_encoders"}:
            raise ValueError(f"Unsupported BC init_mode: {init_mode!r}")
        payload = torch.load(Path(checkpoint_path).resolve(), map_location="cpu")
        observation_mode = str(payload.get("observation_mode", ""))
        if observation_mode != "full":
            raise RuntimeError(
                f"BC checkpoint {checkpoint_path} is not compatible with full DroQ warm start "
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
            "observation_shape": tuple(int(value) for value in payload.get("observation_shape", self.observation_shape)),
            "telemetry_dim": int(payload.get("telemetry_dim", self.telemetry_dim)),
            "action_dim": int(payload.get("action_dim", self.action_dim)),
            "epoch": payload.get("epoch"),
            "train_loss": payload.get("train_loss"),
            "validation_loss": payload.get("validation_loss"),
            "config_snapshot": payload.get("config_snapshot"),
        }

    def _prepare_observation(self, observation: Tensor) -> Tensor:
        return random_shift_augmentation(observation)

    def _encode_full_critic_inputs(
        self,
        observation: Tensor,
        telemetry: Tensor | None,
        *,
        target: bool,
    ) -> tuple[Tensor, Tensor] | None:
        if not self.share_encoders:
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
        assert telemetry is not None
        if self.share_encoders:
            assert encoded_full is not None
            assert isinstance(critic, _DroQSharedEncoderCritic)
            vision_latent, telemetry_latent = encoded_full
            return critic.forward_encoded(vision_latent, telemetry_latent, action)
        return critic(observation, telemetry, action)

    def _soft_update_module(self, target_module: nn.Module, source_module: nn.Module) -> None:
        for target_param, source_param in zip(target_module.parameters(), source_module.parameters(), strict=True):
            target_param.mul_(self.sac_config.polyak)
            target_param.add_((1.0 - self.sac_config.polyak) * source_param)

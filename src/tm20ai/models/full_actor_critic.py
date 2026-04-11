from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn import functional as F

from ..train.features import ACTION_DIM, TELEMETRY_DIM


LOG_STD_MIN: Final[float] = -20.0
LOG_STD_MAX: Final[float] = 2.0


def normalize_image_batch(observation: Tensor) -> Tensor:
    observation = observation.float()
    if observation.max().detach().item() > 1.0:
        observation = observation / 255.0
    return observation


def random_shift_augmentation(observation: Tensor, *, padding: int = 4) -> Tensor:
    if padding <= 0:
        return normalize_image_batch(observation)
    normalized = normalize_image_batch(observation)
    batch, channels, height, width = normalized.shape
    padded = F.pad(normalized, (padding, padding, padding, padding), mode="replicate")
    max_offset = padding * 2 + 1
    offset_x = torch.randint(0, max_offset, (batch,), device=normalized.device)
    offset_y = torch.randint(0, max_offset, (batch,), device=normalized.device)
    shifted = torch.empty_like(normalized)
    for index, (dx, dy) in enumerate(zip(offset_x.tolist(), offset_y.tolist(), strict=True)):
        shifted[index] = padded[index, :, dy : dy + height, dx : dx + width]
    return shifted


class VisionEncoder(nn.Module):
    def __init__(self, observation_shape: tuple[int, int, int]):
        super().__init__()
        channels, height, width = observation_shape
        self.layers = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, channels, height, width, dtype=torch.float32)
            flattened_dim = self.layers(sample).shape[-1]
        self.projection = nn.Sequential(
            nn.Linear(flattened_dim, 512),
            nn.ReLU(),
        )

    def forward(self, observation: Tensor) -> Tensor:
        return self.projection(self.layers(observation))


class TelemetryEncoder(nn.Module):
    def __init__(self, telemetry_dim: int = TELEMETRY_DIM):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(telemetry_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

    def forward(self, telemetry: Tensor) -> Tensor:
        return self.layers(telemetry)


class FullObservationActor(nn.Module):
    def __init__(
        self,
        *,
        observation_shape: tuple[int, int, int] = (4, 64, 64),
        telemetry_dim: int = TELEMETRY_DIM,
        action_dim: int = ACTION_DIM,
    ) -> None:
        super().__init__()
        self.observation_shape = observation_shape
        self.telemetry_dim = telemetry_dim
        self.action_dim = action_dim
        self.vision_encoder = VisionEncoder(observation_shape)
        self.telemetry_encoder = TelemetryEncoder(telemetry_dim)
        fused_dim = 512 + 64
        self.policy_backbone = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)
        self.register_buffer("action_scale", torch.tensor([0.5, 0.5, 1.0], dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor([0.5, 0.5, 0.0], dtype=torch.float32))

    def encode(self, observation: Tensor, telemetry: Tensor) -> Tensor:
        observation = normalize_image_batch(observation)
        telemetry = telemetry.float()
        vision_latent = self.vision_encoder(observation)
        telemetry_latent = self.telemetry_encoder(telemetry)
        return torch.cat((vision_latent, telemetry_latent), dim=-1)

    def forward(self, observation: Tensor, telemetry: Tensor) -> tuple[Tensor, Tensor]:
        fused = self.policy_backbone(self.encode(observation, telemetry))
        mean = self.mean_head(fused)
        log_std = torch.clamp(self.log_std_head(fused), min=LOG_STD_MIN, max=LOG_STD_MAX)
        return mean, log_std

    def sample(
        self,
        observation: Tensor,
        telemetry: Tensor,
        *,
        deterministic: bool = False,
        need_log_prob: bool = True,
    ) -> tuple[Tensor, Tensor | None]:
        mean, log_std = self.forward(observation, telemetry)
        std = log_std.exp()
        distribution = Normal(mean, std)
        pre_tanh = mean if deterministic else distribution.rsample()
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale + self.action_bias
        if not need_log_prob:
            return action, None

        log_prob = distribution.log_prob(pre_tanh)
        log_prob = log_prob - torch.log(1 - squashed.pow(2) + 1e-6)
        log_prob = log_prob - torch.log(self.action_scale).view(1, -1)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def act(self, observation: Tensor, telemetry: Tensor, *, deterministic: bool) -> Tensor:
        action, _ = self.sample(observation, telemetry, deterministic=deterministic, need_log_prob=False)
        return action


class FullObservationCritic(nn.Module):
    def __init__(
        self,
        *,
        observation_shape: tuple[int, int, int] = (4, 64, 64),
        telemetry_dim: int = TELEMETRY_DIM,
        action_dim: int = ACTION_DIM,
    ) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoder(observation_shape)
        self.telemetry_encoder = TelemetryEncoder(telemetry_dim)
        fused_dim = 512 + 64 + action_dim
        self.q_network = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, observation: Tensor, telemetry: Tensor, action: Tensor) -> Tensor:
        observation = normalize_image_batch(observation)
        telemetry = telemetry.float()
        action = action.float()
        fused = torch.cat(
            (self.vision_encoder(observation), self.telemetry_encoder(telemetry), action),
            dim=-1,
        )
        return self.q_network(fused)


@dataclass(frozen=True, slots=True)
class ActorSpec:
    observation_shape: tuple[int, int, int] = (4, 64, 64)
    telemetry_dim: int = TELEMETRY_DIM
    action_dim: int = ACTION_DIM

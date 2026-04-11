from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor, nn
from torch.distributions import Normal


LOG_STD_MIN: Final[float] = -20.0
LOG_STD_MAX: Final[float] = 2.0


class LidarActor(nn.Module):
    def __init__(self, *, observation_dim: int = 83, action_dim: int = 3) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.shared_trunk = nn.Sequential(
            nn.Linear(observation_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)
        self.register_buffer("action_scale", torch.tensor([0.5, 0.5, 1.0], dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor([0.5, 0.5, 0.0], dtype=torch.float32))

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.shared_trunk(observation.float())
        mean = self.mean_head(hidden)
        log_std = torch.clamp(self.log_std_head(hidden), min=LOG_STD_MIN, max=LOG_STD_MAX)
        return mean, log_std

    def sample(
        self,
        observation: Tensor,
        *,
        deterministic: bool = False,
        need_log_prob: bool = True,
    ) -> tuple[Tensor, Tensor | None]:
        mean, log_std = self.forward(observation)
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

    def act(self, observation: Tensor, *, deterministic: bool) -> Tensor:
        action, _ = self.sample(observation, deterministic=deterministic, need_log_prob=False)
        return action


class LidarCritic(nn.Module):
    def __init__(self, *, observation_dim: int = 83, action_dim: int = 3) -> None:
        super().__init__()
        self.q_network = nn.Sequential(
            nn.Linear(observation_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, observation: Tensor, action: Tensor) -> Tensor:
        inputs = torch.cat((observation.float(), action.float()), dim=-1)
        return self.q_network(inputs)


@dataclass(frozen=True, slots=True)
class LidarActorSpec:
    observation_dim: int = 83
    action_dim: int = 3

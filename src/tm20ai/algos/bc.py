from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(slots=True)
class BCUpdateResult:
    loss: float


class BehaviorCloningTrainer:
    def __init__(
        self,
        *,
        actor: nn.Module,
        device: torch.device | str,
        learning_rate: float,
        weight_decay: float = 0.0,
    ) -> None:
        self.actor = actor.to(device)
        self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def train_step(self, observation: Tensor, telemetry: Tensor, action: Tensor) -> BCUpdateResult:
        self.actor.train()
        predicted = self.actor.act(observation, telemetry, deterministic=True)
        loss = F.mse_loss(predicted, action)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return BCUpdateResult(loss=float(loss.detach().cpu().item()))

    def evaluate(self, observation: Tensor, telemetry: Tensor, action: Tensor) -> float:
        self.actor.eval()
        with torch.no_grad():
            predicted = self.actor.act(observation, telemetry, deterministic=True)
            loss = F.mse_loss(predicted, action)
        return float(loss.detach().cpu().item())

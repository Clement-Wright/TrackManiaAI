from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from ..algos.redq import REDQSACAgent
from ..config import OfflinePretrainConfig
from ..train.replay import ReplayBuffer


@dataclass(frozen=True, slots=True)
class OfflinePretrainResult:
    gradient_steps: int
    bc_updates: int
    critic_updates: int
    awac_updates: int
    cql_updates: int
    metrics: dict[str, float]


def _prepare_actor_inputs(agent: REDQSACAgent, batch):
    obs = agent._prepare_observation(batch.obs)
    telemetry = batch.telemetry
    return obs, telemetry


def bc_actor_update(agent: REDQSACAgent, batch, *, weight: float = 1.0) -> dict[str, float]:
    obs, telemetry = _prepare_actor_inputs(agent, batch)
    if agent.observation_mode == "full":
        assert telemetry is not None
        pred = agent.actor.act(obs, telemetry, deterministic=True)
    else:
        pred = agent.actor.act(obs, deterministic=True)
    loss = F.mse_loss(pred, batch.action) * float(weight)
    agent.actor_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    agent.actor_optimizer.step()
    return {"bc_loss": float(loss.detach().cpu().item())}


def awac_actor_update(agent: REDQSACAgent, batch, *, lambda_: float = 1.0) -> dict[str, float]:
    obs, telemetry = _prepare_actor_inputs(agent, batch)
    if agent.observation_mode == "full":
        assert telemetry is not None
        pred = agent.actor.act(obs, telemetry, deterministic=True)
    else:
        pred = agent.actor.act(obs, deterministic=True)
    rewards = batch.reward.detach()
    normalized = rewards - rewards.mean()
    scale = rewards.std().clamp_min(1.0e-6)
    weights = torch.exp((normalized / scale) / max(1.0e-6, float(lambda_))).clamp(max=20.0)
    loss = (weights * (pred - batch.action).pow(2).mean(dim=-1, keepdim=True)).mean()
    agent.actor_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    agent.actor_optimizer.step()
    return {
        "awac_loss": float(loss.detach().cpu().item()),
        "awac_weight_mean": float(weights.mean().detach().cpu().item()),
    }


def cql_critic_regularization_update(agent: REDQSACAgent, batch, *, alpha: float) -> dict[str, float]:
    if alpha <= 0.0:
        return {"cql_loss": 0.0}
    obs = agent._prepare_observation(batch.obs)
    encoded_obs = agent._encode_full_critic_inputs(obs, batch.telemetry, target=False)
    random_actions = torch.empty_like(batch.action).uniform_(-1.0, 1.0)
    data_q = torch.stack(
        [
            agent._critic_forward(critic, obs, batch.telemetry, batch.action, encoded_full=encoded_obs)
            for critic in agent.critics
        ],
        dim=0,
    )
    random_q = torch.stack(
        [
            agent._critic_forward(critic, obs, batch.telemetry, random_actions, encoded_full=encoded_obs)
            for critic in agent.critics
        ],
        dim=0,
    )
    loss = float(alpha) * (torch.logsumexp(random_q, dim=0).mean() - data_q.mean())
    agent.critic_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    agent.critic_optimizer.step()
    return {"cql_loss": float(loss.detach().cpu().item())}


def offline_pretrain_redq(
    *,
    agent: REDQSACAgent,
    replay: ReplayBuffer,
    config: OfflinePretrainConfig,
    device: torch.device,
) -> OfflinePretrainResult:
    if replay.size <= 0:
        raise RuntimeError("Offline REDQ pretraining requires a non-empty offline replay buffer.")
    metrics_accum: dict[str, list[float]] = {}
    bc_updates = 0
    critic_updates = 0
    awac_updates = 0
    cql_updates = 0
    for step in range(int(config.gradient_steps)):
        batch = replay.sample(config.batch_size, device=device)
        if config.strategy in {"bc", "bc_redq_awac"}:
            metrics = bc_actor_update(agent, batch, weight=config.bc_weight)
            bc_updates += 1
            for key, value in metrics.items():
                metrics_accum.setdefault(key, []).append(value)
        if config.strategy in {"redq_critic", "bc_redq_awac", "iql", "cql"}:
            update = agent.update_critics(batch)
            critic_updates += 1
            metrics_accum.setdefault("critic_loss", []).append(update.critic_loss)
        if config.strategy in {"awac", "bc_redq_awac", "iql"}:
            metrics = awac_actor_update(agent, batch, lambda_=config.awac_lambda)
            awac_updates += 1
            for key, value in metrics.items():
                metrics_accum.setdefault(key, []).append(value)
        if config.strategy in {"cql", "bc_redq_awac"} and config.cql_alpha > 0.0:
            metrics = cql_critic_regularization_update(agent, batch, alpha=config.cql_alpha)
            cql_updates += 1
            for key, value in metrics.items():
                metrics_accum.setdefault(key, []).append(value)
        if config.strategy == "bc" and step + 1 >= config.gradient_steps:
            break
    metrics_summary = {
        f"mean_{key}": float(sum(values) / max(1, len(values)))
        for key, values in metrics_accum.items()
    }
    return OfflinePretrainResult(
        gradient_steps=int(config.gradient_steps),
        bc_updates=bc_updates,
        critic_updates=critic_updates,
        awac_updates=awac_updates,
        cql_updates=cql_updates,
        metrics=metrics_summary,
    )

"""Tiny policy/value network."""

from __future__ import annotations

import torch
from torch import nn


class PolicyValueNet(nn.Module):
    def __init__(self, obs_size: int, action_count: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.policy_head = nn.Linear(hidden_size, action_count)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        policy_logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return (policy_logits, value)


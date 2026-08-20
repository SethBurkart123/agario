"""PufferLib 3.0 bridge: the agario arena as a native PufferEnv, plus a
policy that reuses our entity-attention encoder under PufferLib's LSTM
wrapper.

The physics is byte-identical to every other training path — same ArenaEnv,
same Rust core, same reward shaping. The only interface change: the
continuous speed channel is exposed as 8 discrete bins (MultiDiscrete needs
discrete dims); the env maps bin centers back onto the same response curve.
"""

from __future__ import annotations

import numpy as np
import pufferlib
import torch
import torch.nn as nn

from .env import ArenaConfig, ArenaEnv
from .obs import N_OPS, N_TURNS, OBS_DIM

N_SPEED_BINS = 8


def bin_to_speed(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.float64) + 0.5) / N_SPEED_BINS


class AgarioPufferEnv(pufferlib.PufferEnv):
    def __init__(
        self,
        num_learners: int = 10,
        anchors: int = 2,
        anchor_every: int = 4,
        spawn_mass_jitter: float = 1.2,
        seed: int = 0,
        buf=None,
    ):
        import gymnasium.spaces as gym_spaces

        self.single_observation_space = gym_spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.single_action_space = gym_spaces.MultiDiscrete(
            [N_TURNS, N_OPS, N_SPEED_BINS]
        )
        self.num_agents = num_learners
        super().__init__(buf)

        spawn_range = None
        if spawn_mass_jitter > 0:
            spawn_range = (1.0 / (1.0 + spawn_mass_jitter), 1.0 + spawn_mass_jitter)
        cfg = ArenaConfig(
            n_learners=num_learners,
            n_frozen=0,
            n_heuristic=anchors,
            anchor_every=anchor_every,
            spawn_mass_range=spawn_range,
        )
        self.env = ArenaEnv(cfg, seed=int(seed))

    def reset(self, seed=None):
        self.observations[:] = self.env.reset()
        return self.observations, []

    def step(self, actions):
        acts = np.empty((self.num_agents, 3), dtype=np.float64)
        acts[:, 0] = actions[:, 0]
        acts[:, 1] = actions[:, 1]
        acts[:, 2] = bin_to_speed(actions[:, 2])

        obs, rewards, truncated, info = self.env.step(acts)
        episode_stats = []
        if truncated:
            episode_stats = [
                {
                    "episode_mass": float(np.mean(info.get("final_masses", [0.0]))),
                    "episode_kills": float(np.mean(info.get("kills", [0.0]))),
                    "episode_deaths": float(np.mean(info.get("deaths", [0.0]))),
                }
            ]
            obs = self.env.reset()

        self.observations[:] = obs
        self.rewards[:] = rewards
        self.terminals[:] = False  # continuing task: episodes end by truncation only
        self.truncations[:] = truncated
        return self.observations, self.rewards, self.terminals, self.truncations, episode_stats

    def close(self):
        pass


class AgarioPufferPolicy(nn.Module):
    """Our entity-attention encoder + PufferLib-convention heads.

    encode_observations/decode_actions split so pufferlib.models.LSTMWrapper
    can manage the recurrent state. Optionally warm-starts the encoder from
    one of our arch-v4 checkpoints (heads are new: speed is a discrete dim
    here, so only the trunk transfers).
    """

    def __init__(self, env, hidden_size: int = 256, init_from: str | None = None):
        super().__init__()
        from .model import HIDDEN, PolicyNet

        assert hidden_size == HIDDEN, "encoder trunk is built for HIDDEN width"
        self.hidden_size = hidden_size
        self.is_continuous = False
        self.action_nvec = tuple(env.single_action_space.nvec)

        self.net = PolicyNet(rnn_type="gru")  # we use .encode() only
        if init_from:
            ckpt = torch.load(init_from, map_location="cpu", weights_only=True)
            missing, unexpected = self.net.load_state_dict(ckpt["policy"], strict=False)
            print(f"warm-start encoder from {init_from} "
                  f"(missing {len(missing)}, unexpected {len(unexpected)})")

        num_logits = int(sum(self.action_nvec))
        self.action_head = nn.Linear(hidden_size, num_logits)
        nn.init.orthogonal_(self.action_head.weight, 0.01)
        nn.init.zeros_(self.action_head.bias)
        self.value_head = nn.Linear(hidden_size, 1)
        nn.init.orthogonal_(self.value_head.weight, 1.0)
        nn.init.zeros_(self.value_head.bias)

    def encode_observations(self, observations, state=None):
        return self.net.encode(observations.float())

    def decode_actions(self, hidden):
        logits = self.action_head(hidden).split(self.action_nvec, dim=-1)
        values = self.value_head(hidden).squeeze(-1)
        return logits, values

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state)
        return self.decode_actions(hidden)

    def forward(self, observations, state=None):
        return self.forward_eval(observations, state)

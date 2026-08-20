"""PufferEnv over the Rust BatchedArenas: the whole env loop (actions →
physics → rewards → resets → observations) runs in one rayon-parallel FFI
call. Python's per-step work is two numpy frombuffer calls.

Use with PufferLib's Serial backend and num_envs=1 — parallelism lives in
Rust, not in worker processes.
"""

from __future__ import annotations

import numpy as np
import pufferlib

from .obs import N_OPS, N_TURNS, OBS_DIM
from .puffer_env import N_SPEED_BINS, bin_to_speed


class AgarioBatchEnv(pufferlib.PufferEnv):
    def __init__(
        self,
        arenas: int = 24,
        num_learners: int = 10,
        anchors: int = 2,
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
        self.num_agents = arenas * num_learners
        super().__init__(buf)

        from agario_core import BatchedArenas

        assert BatchedArenas.obs_dim() == OBS_DIM
        self.sim = BatchedArenas(arenas, num_learners, anchors, int(seed))
        self._n = self.num_agents

    def reset(self, seed=None):
        buf = self.sim.reset()
        self.observations[:] = np.frombuffer(buf, dtype=np.float32).reshape(
            self._n, OBS_DIM
        )
        return self.observations, []

    def step(self, actions):
        acts = np.empty((self._n, 3), dtype=np.float32)
        acts[:, 0] = actions[:, 0]
        acts[:, 1] = actions[:, 1]
        acts[:, 2] = bin_to_speed(actions[:, 2])

        ob, rb, tb = self.sim.step(acts.ravel().tolist())
        self.observations[:] = np.frombuffer(ob, dtype=np.float32).reshape(
            self._n, OBS_DIM
        )
        self.rewards[:] = np.frombuffer(rb, dtype=np.float32)
        self.terminals[:] = False
        self.truncations[:] = np.frombuffer(tb, dtype=np.uint8).astype(bool)

        infos = []
        stats = self.sim.take_episode_stats()
        if stats:
            infos = [
                {
                    "episode_mass": float(np.mean([s[0] for s in stats])),
                    "episode_kills": float(np.mean([s[1] for s in stats])),
                    "episode_deaths": float(np.mean([s[2] for s in stats])),
                }
            ]
        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def close(self):
        pass

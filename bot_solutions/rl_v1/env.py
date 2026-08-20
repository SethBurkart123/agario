"""Self-play arena environment over the Rust sim core.

ArenaEnv: one world containing three kinds of agents:
  - learners: controlled by the policy being trained
  - frozen:   controlled by a frozen league opponent (past checkpoints,
              AlphaGo-style fictitious self-play) via VecArena.frozen_policy
  - anchors:  solo_smart heuristic bots, grounding the population

Continuing task with fixed horizon: agents that die are auto-respawned by the
world (death penalty in the reward); episodes end by truncation only, so the
agent count is constant and batching stays rectangular.

Reward per decision: sqrt-mass delta (growth) + kill bonus per enemy blob
eaten (aggression) or a death penalty on the step the agent dies.

VecArena: several arenas stepped together; learner observations/rewards are
stacked as (n_arenas * n_learners, ...).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from agario_core import CoreWorld, mechanics

from .heuristic import HeuristicDriver
from .obs import (
    N_DIRECTIONS,
    OBS_DIM,
    action_to_target,
    apply_turn,
)

MECHANICS = mechanics()
TICK_RATE = MECHANICS["tick_rate"]
DT = 1.0 / TICK_RATE


@dataclass(slots=True)
class ArenaConfig:
    world_size: float = 2500.0
    # Per-episode domain randomization: arena side length sampled from this
    # range (food/virus counts scale with area). Covers everything from
    # crowded knife-fights to sparse open space.
    size_range: tuple[float, float] | None = (2000.0, 4200.0)
    food_count: int = 220
    virus_count: int = 6
    n_learners: int = 5
    n_frozen: int = 2
    n_heuristic: int = 1
    frame_skip: int = 2  # decisions at 12.5 Hz
    # Heuristic anchors think every Nth decision (~160 ms at 2) — within their
    # natural reaction times, and the sim keeps steering toward the last
    # target between thinks. Their Python brains are the env's biggest cost.
    anchor_every: int = 2
    episode_decisions: int = 1100  # ~88 sim-seconds
    death_penalty: float = 2.0
    kill_bonus: float = 1.0
    mass_reward_scale: float = 0.1
    # Optional spawn-mass asymmetry creates predator/prey situations.
    # spawns in self-play make splits/hunting structurally worthless (nobody
    # ever has the advantage a split-kill needs), so PPO bleeds those
    # behaviors out. Predator/prey asymmetry keeps them profitable.
    spawn_mass_range: tuple[float, float] | None = None
    def world_overrides(self, size: float) -> dict:
        area_scale = (size * size) / (self.world_size * self.world_size)
        virus_count = max(2, int(self.virus_count * area_scale))
        return {
            "world_width": size,
            "world_height": size,
            "food_target_count": max(40, int(self.food_count * area_scale)),
            "virus_min_count": virus_count,
            "virus_max_count": virus_count * 3,
        }


class ArenaEnv:
    def __init__(self, cfg: ArenaConfig, seed: int) -> None:
        self.cfg = cfg
        self._seed = seed
        self._episode = 0
        self.size = cfg.world_size
        self.world: CoreWorld | None = None
        self.learner_ids: list[str] = []
        self.frozen_ids: list[str] = []
        self.heuristic_ids: list[str] = []
        self._driver: HeuristicDriver | None = None
        self.now = 0.0
        self._decision = 0
        self._prev_sqrt_mass: dict[str, float] = {}
        self._prev_deaths: dict[str, int] = {}
        self._prev_kills: dict[str, int] = {}
        self._headings: dict[str, int] = {}
        # Per-agent (prev_turn, prev_op, prev_speed) — part of the observation.
        self._prev_action: dict[str, tuple[int, int, float]] = {}
        self._obs = np.zeros((cfg.n_learners, OBS_DIM), dtype=np.float32)
        self._frozen_obs = np.zeros((cfg.n_frozen, OBS_DIM), dtype=np.float32)

    def reset(self) -> np.ndarray:
        self._episode += 1
        seed = (self._seed * 1_000_003 + self._episode) % (2**62)
        if self.cfg.size_range is not None:
            lo, hi = self.cfg.size_range
            rng = np.random.default_rng(seed % (2**32))
            self.size = float(rng.uniform(lo, hi))
        else:
            self.size = self.cfg.world_size
        self.world = CoreWorld(seed, config=self.cfg.world_overrides(self.size))
        self.now = 0.0
        self._decision = 0

        self.learner_ids = [
            self.world.add_player(f"learner-{i}", self.now, is_bot=True).id
            for i in range(self.cfg.n_learners)
        ]
        self.frozen_ids = [
            self.world.add_player(f"frozen-{i}", self.now, is_bot=True).id
            for i in range(self.cfg.n_frozen)
        ]
        self.heuristic_ids = [
            self.world.add_player(f"anchor-{i}", self.now, is_bot=True).id
            for i in range(self.cfg.n_heuristic)
        ]
        self._driver = (
            HeuristicDriver(self.heuristic_ids, seed=seed % (2**31))
            if self.heuristic_ids
            else None
        )
        if self.cfg.spawn_mass_range is not None:
            lo, hi = self.cfg.spawn_mass_range
            mass_rng = np.random.default_rng((seed + 7) % (2**32))
            for pid in self.learner_ids + self.frozen_ids + self.heuristic_ids:
                self.world.set_player_mass(
                    pid, float(MECHANICS["player_start_mass"] * mass_rng.uniform(lo, hi))
                )
        heading_rng = np.random.default_rng((seed + 13) % (2**32))
        self._headings = {
            pid: int(heading_rng.integers(N_DIRECTIONS))
            for pid in self.learner_ids + self.frozen_ids
        }
        self._prev_action = {
            pid: (0, 0, 1.0) for pid in self.learner_ids + self.frozen_ids
        }

        self._refresh_baselines()
        self._encode_for(self.learner_ids, self._obs)
        return self._obs

    def _refresh_baselines(self) -> None:
        masses = {pid: m for pid, m, _ in self.world.player_masses()}
        deaths = dict(self.world.death_counts())
        kills = dict(self.world.kill_counts())
        for pid in self.learner_ids:
            self._prev_sqrt_mass[pid] = math.sqrt(max(masses.get(pid, 0.0), 1.0))
            self._prev_deaths[pid] = deaths.get(pid, 0)
            self._prev_kills[pid] = kills.get(pid, 0)

    def frozen_obs(self) -> np.ndarray:
        """Observations for frozen league opponents from the current state."""
        self._encode_for(self.frozen_ids, self._frozen_obs)
        return self._frozen_obs

    def step(
        self, actions: np.ndarray, frozen_actions: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, bool, dict]:
        """actions: (n_learners, 2) int array of [direction, op]."""
        assert self.world is not None, "call reset() first"
        cfg = self.cfg

        centers = self._centers()
        self._apply_actions(self.learner_ids, actions, centers)
        if frozen_actions is not None and self.frozen_ids:
            self._apply_actions(self.frozen_ids, frozen_actions, centers)
        if self._driver is not None and self._decision % cfg.anchor_every == 0:
            self._driver.apply(
                self.world, self.now, DT * cfg.frame_skip * cfg.anchor_every,
                self.size, self.size,
            )

        for _ in range(cfg.frame_skip):
            self.now += DT
            self.world.update(DT, self.now)

        rewards = np.zeros(cfg.n_learners, dtype=np.float32)
        masses = {pid: m for pid, m, _ in self.world.player_masses()}
        deaths = dict(self.world.death_counts())
        kills = dict(self.world.kill_counts())
        for idx, pid in enumerate(self.learner_ids):
            new_sqrt = math.sqrt(max(masses.get(pid, 0.0), 1.0))
            died = deaths.get(pid, 0) > self._prev_deaths[pid]
            new_kills = kills.get(pid, 0) - self._prev_kills[pid]
            if died:
                rewards[idx] = -cfg.death_penalty
            else:
                rewards[idx] = (new_sqrt - self._prev_sqrt_mass[pid]) * cfg.mass_reward_scale
            rewards[idx] += cfg.kill_bonus * new_kills
            self._prev_sqrt_mass[pid] = new_sqrt
            self._prev_deaths[pid] = deaths.get(pid, 0)
            self._prev_kills[pid] = kills.get(pid, 0)

        self._decision += 1
        truncated = self._decision >= cfg.episode_decisions
        self._encode_for(self.learner_ids, self._obs)

        info: dict = {}
        if truncated:
            info["final_masses"] = [masses.get(pid, 0.0) for pid in self.learner_ids]
            info["deaths"] = [deaths.get(pid, 0) for pid in self.learner_ids]
            info["kills"] = [kills.get(pid, 0) for pid in self.learner_ids]
        return self._obs, rewards, truncated, info

    def _apply_actions(
        self, ids: list[str], actions: np.ndarray, centers: dict[str, tuple[float, float]]
    ) -> None:
        """actions rows: [turn, op] or [turn, op, speed] (speed defaults 1.0)."""
        has_speed = actions.shape[1] >= 3
        for idx, pid in enumerate(ids):
            cx, cy = centers.get(pid, (self.size / 2.0, self.size / 2.0))
            turn, op = int(actions[idx][0]), int(actions[idx][1])
            speed = float(actions[idx][2]) if has_speed else 1.0
            heading = apply_turn(self._headings.get(pid, 0), turn)
            self._headings[pid] = heading
            self._prev_action[pid] = (turn, op, speed)
            tx, ty = action_to_target(heading, cx, cy, speed)
            self.world.set_input(
                pid,
                min(max(tx, 0.0), self.size),
                min(max(ty, 0.0), self.size),
                split=op == 1,
                eject=op == 2,
            )

    def _centers(self) -> dict[str, tuple[float, float]]:
        centers: dict[str, tuple[float, float]] = {}
        for pid, _n, _c, _b, _pl, _t, _tx, _ty, blob_rows in self.world.players_compact():
            if not blob_rows:
                continue
            total = sum(b[3] for b in blob_rows)
            if total <= 0.0:
                continue
            cx = sum(b[1] * b[3] for b in blob_rows) / total
            cy = sum(b[2] * b[3] for b in blob_rows) / total
            centers[pid] = (cx, cy)
        return centers

    def _control_for(self, ids: list[str]) -> list[tuple[int, int, int, float]]:
        out = []
        for pid in ids:
            turn, op, speed = self._prev_action.get(pid, (0, 0, 1.0))
            out.append((self._headings.get(pid, 0), turn, op, speed))
        return out

    def _encode_for(self, ids: list[str], out: np.ndarray) -> None:
        buf = self.world.observe(ids, self.now, self._control_for(ids))
        out[:] = np.frombuffer(buf, dtype=np.float32).reshape(len(ids), OBS_DIM)


class VecArena:
    """Steps several arenas as one batched environment.

    Set `frozen_policy` to a callable mapping a stacked observation array
    (n_total_frozen, OBS_DIM) and a done-flag array (n_total_frozen,) to an
    integer action array (n_total_frozen, 2) to drive the league opponent
    slots. The done flags mark frozen agents whose arena just reset, so a
    recurrent opponent can zero its memory there. Left as None the frozen
    slots sit still and just become food (only sensible for smoke tests).
    """

    def __init__(self, cfg: ArenaConfig, n_arenas: int, seed: int) -> None:
        self.cfg = cfg
        self.envs = [ArenaEnv(cfg, seed + 7919 * i) for i in range(n_arenas)]
        self.n_agents = n_arenas * cfg.n_learners
        self.frozen_policy: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
        self._frozen_done = np.ones(len(self.envs) * cfg.n_frozen, dtype=bool)

    def reset(self) -> np.ndarray:
        self._frozen_done[:] = True
        return np.concatenate([env.reset() for env in self.envs], axis=0)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        n = self.cfg.n_learners
        nf = self.cfg.n_frozen

        frozen_actions: np.ndarray | None = None
        if self.frozen_policy is not None and nf > 0:
            frozen_obs = np.concatenate([env.frozen_obs() for env in self.envs], axis=0)
            frozen_actions = self.frozen_policy(frozen_obs, self._frozen_done.copy())
            self._frozen_done[:] = False

        all_obs, all_rewards, all_truncated, infos = [], [], [], []
        for i, env in enumerate(self.envs):
            fa = frozen_actions[i * nf : (i + 1) * nf] if frozen_actions is not None else None
            obs, rewards, truncated, info = env.step(actions[i * n : (i + 1) * n], fa)
            if truncated:
                obs = env.reset()
                self._frozen_done[i * nf : (i + 1) * nf] = True
            all_obs.append(obs.copy())
            all_rewards.append(rewards)
            all_truncated.append(np.full(n, truncated))
            infos.append(info)
        return (
            np.concatenate(all_obs, axis=0),
            np.concatenate(all_rewards, axis=0),
            np.concatenate(all_truncated, axis=0),
            infos,
        )

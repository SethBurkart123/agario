"""PUCT-style rollout search using authoritative world clones."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from math import sqrt

import numpy as np
import torch

from ._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from agario import config
from agario.world import GameWorld

from .action_space import ActionSpace
from .gpu_rollout import GpuRolloutSimulator
from .model import PolicyValueNet
from .observation import build_observation
from .settings import AiBotSettings


@dataclass(slots=True, frozen=True)
class SearchResult:
    action_index: int
    visit_probs: np.ndarray
    root_value: float
    obs_vector: np.ndarray


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    denom = np.sum(exp)
    if denom <= 1e-8:
        return np.full_like(logits, 1.0 / max(1, logits.size))
    return exp / denom


class PUCTPlanner:
    def __init__(
        self,
        *,
        model: PolicyValueNet,
        action_space: ActionSpace,
        device: torch.device,
        settings: AiBotSettings,
    ) -> None:
        self.model = model
        self.action_space = action_space
        self.device = device
        self.settings = settings
        self.gpu_rollout = (
            GpuRolloutSimulator(action_space=action_space, settings=settings, device=device)
            if settings.use_gpu_rollout and device.type == "cuda"
            else None
        )

    def search(
        self,
        *,
        world: GameWorld,
        player_id: str,
        now: float,
        dt: float,
        training: bool,
    ) -> SearchResult:
        built = build_observation(world, player_id, now)
        root_logits, root_value = self._forward_vector(built.vector)
        priors = _softmax(root_logits)
        if training:
            priors = self._apply_dirichlet_noise(priors)

        candidate_mask = np.zeros(self.action_space.size, dtype=bool)
        max_actions = min(self.action_space.size, self.settings.max_considered_actions)
        if max_actions >= self.action_space.size:
            candidate_mask[:] = True
        else:
            top_idx = np.argpartition(priors, -max_actions)[-max_actions:]
            candidate_mask[top_idx] = True
            candidate_mass = float(np.sum(priors[candidate_mask]))
            if candidate_mass > 1e-8:
                priors = priors.copy()
                priors[~candidate_mask] = 0.0
                priors /= candidate_mass

        visits = np.zeros(self.action_space.size, dtype=np.float32)
        q_values = np.zeros(self.action_space.size, dtype=np.float32)
        evaluated = np.zeros(self.action_space.size, dtype=bool)

        if self.gpu_rollout is not None:
            candidate_indices = np.flatnonzero(candidate_mask)
            candidate_values = self.gpu_rollout.evaluate(
                world=world,
                player_id=player_id,
                now=now,
                dt=dt,
                action_indices=candidate_indices.astype(np.int64, copy=False),
            )
            if candidate_indices.size > 0:
                q_values[candidate_indices] = candidate_values
                evaluated[candidate_indices] = True

        for _ in range(self.settings.simulations):
            total_visits = float(np.sum(visits))
            puct_bonus = (
                self.settings.c_puct
                * priors
                * (sqrt(total_visits + 1.0) / (1.0 + visits))
            )
            scores = q_values + puct_bonus
            scores[~candidate_mask] = -1e18
            action_index = int(np.argmax(scores))

            if not evaluated[action_index]:
                estimate = self._evaluate_rollout(
                    world=world,
                    player_id=player_id,
                    action_index=action_index,
                    now=now,
                    dt=dt,
                )
                q_values[action_index] = estimate
                evaluated[action_index] = True
            visits[action_index] += 1.0

        if np.sum(visits) <= 0:
            visit_probs = priors
        else:
            visit_probs = visits / np.sum(visits)

        if training:
            action_index = self._sample_by_temperature(
                visit_probs,
                temperature=self.settings.train_decision_temperature,
            )
        else:
            action_index = int(np.argmax(visit_probs))

        return SearchResult(
            action_index=action_index,
            visit_probs=visit_probs.astype(np.float32, copy=False),
            root_value=float(root_value),
            obs_vector=built.vector,
        )

    def action_to_world_input(
        self,
        *,
        world: GameWorld,
        player_id: str,
        action_index: int,
    ) -> tuple[float, float, bool, bool]:
        player = world.players.get(player_id)
        if player is None:
            center_x = config.WORLD_WIDTH * 0.5
            center_y = config.WORLD_HEIGHT * 0.5
        else:
            center_x, center_y = player.center()
        return self.action_space.to_world_input(
            action_index=action_index,
            center_x=center_x,
            center_y=center_y,
            world_width=config.WORLD_WIDTH,
            world_height=config.WORLD_HEIGHT,
        )

    def _evaluate_rollout(
        self,
        *,
        world: GameWorld,
        player_id: str,
        action_index: int,
        now: float,
        dt: float,
    ) -> float:
        if hasattr(world, "fast_clone"):
            world_copy = world.fast_clone()
        else:
            world_copy = copy.deepcopy(world)
        player_before = world_copy.players.get(player_id)
        if player_before is None:
            return -1.0
        start_mass = max(1.0, player_before.total_mass)
        sim_now = now
        current_action = action_index

        for step in range(self.settings.horizon_steps):
            if step > 0:
                built = build_observation(world_copy, player_id, sim_now)
                logits, _ = self._forward_vector(built.vector)
                current_action = int(np.argmax(logits))

            tx, ty, split, eject = self.action_to_world_input(
                world=world_copy,
                player_id=player_id,
                action_index=current_action,
            )
            world_copy.set_input(
                player_id=player_id,
                target_x=tx,
                target_y=ty,
                split=split,
                eject=eject,
            )
            sim_now += dt
            world_copy.update(dt=dt, now=sim_now)

        player_after = world_copy.players.get(player_id)
        if player_after is None:
            return -1.0

        end_mass = player_after.total_mass
        alive = 1.0 if player_after.blobs else -1.0
        mass_gain = (end_mass - start_mass) / start_mass
        rank = self._rank_score(world_copy, player_id)
        immediate = 0.65 * mass_gain + 0.2 * rank + 0.15 * alive

        built_end = build_observation(world_copy, player_id, sim_now)
        _, end_value = self._forward_vector(built_end.vector)
        total = immediate + (self.settings.discount ** self.settings.horizon_steps) * end_value
        return float(np.clip(total, -1.0, 1.0))

    def _forward_vector(self, vector: np.ndarray) -> tuple[np.ndarray, float]:
        obs = torch.as_tensor(vector, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(obs)
        return (
            logits.squeeze(0).detach().cpu().numpy(),
            float(value.item()),
        )

    def _rank_score(self, world: GameWorld, player_id: str) -> float:
        players = sorted(world.players.values(), key=lambda p: p.total_mass, reverse=True)
        if not players:
            return 0.0
        if len(players) == 1:
            return 1.0
        for idx, player in enumerate(players):
            if player.id == player_id:
                frac = idx / float(len(players) - 1)
                return 1.0 - frac * 2.0
        return 0.0

    def _apply_dirichlet_noise(self, probs: np.ndarray) -> np.ndarray:
        noise = np.random.default_rng().dirichlet(
            np.full(probs.shape[0], self.settings.dirichlet_alpha, dtype=np.float64)
        )
        mixed = (1.0 - self.settings.dirichlet_epsilon) * probs + self.settings.dirichlet_epsilon * noise
        mixed = mixed / np.sum(mixed)
        return mixed.astype(np.float32, copy=False)

    def _sample_by_temperature(self, probs: np.ndarray, temperature: float) -> int:
        if temperature <= 1e-6:
            return int(np.argmax(probs))
        adjusted = np.power(np.maximum(probs, 1e-8), 1.0 / temperature)
        adjusted = adjusted / np.sum(adjusted)
        return int(np.random.default_rng().choice(np.arange(probs.size), p=adjusted))

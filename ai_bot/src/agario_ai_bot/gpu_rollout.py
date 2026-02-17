"""Tensorized rollout simulator for planner search on CUDA."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

import numpy as np
import torch

from ._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from agario import config
from agario.world import GameWorld

from .action_space import ActionSpace
from .settings import AiBotSettings


@dataclass(slots=True, frozen=True)
class _RolloutState:
    player_ids: tuple[str, ...]
    controlled_player_idx: int
    controlled_center_x: float
    controlled_center_y: float
    target_x: np.ndarray
    target_y: np.ndarray
    blob_x: np.ndarray
    blob_y: np.ndarray
    blob_mass: np.ndarray
    blob_vx: np.ndarray
    blob_vy: np.ndarray
    blob_owner: np.ndarray
    food_x: np.ndarray
    food_y: np.ndarray
    food_mass: np.ndarray
    ejected_x: np.ndarray
    ejected_y: np.ndarray
    ejected_mass: np.ndarray
    start_controlled_mass: float


def _nearest_food(world: GameWorld, cx: float, cy: float, limit: int) -> list:
    foods = list(world.foods.values())
    foods.sort(key=lambda item: hypot(item.x - cx, item.y - cy))
    return foods[:limit]


def _nearest_ejected(world: GameWorld, cx: float, cy: float, limit: int) -> list:
    ejected = list(world.ejected.values())
    ejected.sort(key=lambda item: hypot(item.x - cx, item.y - cy))
    return ejected[:limit]


class GpuRolloutSimulator:
    def __init__(
        self,
        *,
        action_space: ActionSpace,
        settings: AiBotSettings,
        device: torch.device,
    ) -> None:
        self.action_space = action_space
        self.settings = settings
        self.device = device

    def evaluate(
        self,
        *,
        world: GameWorld,
        player_id: str,
        now: float,
        dt: float,
        action_indices: np.ndarray,
    ) -> np.ndarray:
        if action_indices.size == 0:
            return np.empty((0,), dtype=np.float32)

        state = self._build_state(world=world, player_id=player_id)
        if state is None:
            return np.full((action_indices.size,), -1.0, dtype=np.float32)
        return self._simulate(state=state, now=now, dt=dt, action_indices=action_indices)

    def _build_state(self, *, world: GameWorld, player_id: str) -> _RolloutState | None:
        players = sorted(world.players.values(), key=lambda p: p.id)
        if not players:
            return None
        player_ids = tuple(p.id for p in players)
        if player_id not in {p.id for p in players}:
            return None
        controlled_idx = player_ids.index(player_id)
        controlled_player = world.players[player_id]
        cx, cy = controlled_player.center()

        target_x = np.asarray([p.target_x for p in players], dtype=np.float32)
        target_y = np.asarray([p.target_y for p in players], dtype=np.float32)

        blobs = []
        for player in players:
            for blob in player.blobs.values():
                blobs.append(blob)
        if not blobs:
            return None

        owner_index = {pid: idx for idx, pid in enumerate(player_ids)}
        blob_x = np.asarray([b.x for b in blobs], dtype=np.float32)
        blob_y = np.asarray([b.y for b in blobs], dtype=np.float32)
        blob_mass = np.asarray([b.mass for b in blobs], dtype=np.float32)
        blob_vx = np.asarray([b.vx for b in blobs], dtype=np.float32)
        blob_vy = np.asarray([b.vy for b in blobs], dtype=np.float32)
        blob_owner = np.asarray([owner_index[b.player_id] for b in blobs], dtype=np.int64)

        foods = _nearest_food(world, cx, cy, self.settings.gpu_rollout_food_limit)
        ejected = _nearest_ejected(world, cx, cy, self.settings.gpu_rollout_ejected_limit)
        food_x = np.asarray([f.x for f in foods], dtype=np.float32)
        food_y = np.asarray([f.y for f in foods], dtype=np.float32)
        food_mass = np.asarray([f.mass for f in foods], dtype=np.float32)
        ejected_x = np.asarray([e.x for e in ejected], dtype=np.float32)
        ejected_y = np.asarray([e.y for e in ejected], dtype=np.float32)
        ejected_mass = np.asarray([e.mass for e in ejected], dtype=np.float32)

        start_mass = float(
            np.sum(blob_mass[blob_owner == controlled_idx]) if np.any(blob_owner == controlled_idx) else 1.0
        )
        start_mass = max(1.0, start_mass)

        return _RolloutState(
            player_ids=player_ids,
            controlled_player_idx=controlled_idx,
            controlled_center_x=cx,
            controlled_center_y=cy,
            target_x=target_x,
            target_y=target_y,
            blob_x=blob_x,
            blob_y=blob_y,
            blob_mass=blob_mass,
            blob_vx=blob_vx,
            blob_vy=blob_vy,
            blob_owner=blob_owner,
            food_x=food_x,
            food_y=food_y,
            food_mass=food_mass,
            ejected_x=ejected_x,
            ejected_y=ejected_y,
            ejected_mass=ejected_mass,
            start_controlled_mass=start_mass,
        )

    def _simulate(self, *, state: _RolloutState, now: float, dt: float, action_indices: np.ndarray) -> np.ndarray:
        batch = int(action_indices.size)
        players = len(state.player_ids)
        blobs = int(state.blob_x.size)
        if blobs == 0:
            return np.full((batch,), -1.0, dtype=np.float32)

        action_targets_x = np.zeros((batch,), dtype=np.float32)
        action_targets_y = np.zeros((batch,), dtype=np.float32)
        action_split = np.zeros((batch,), dtype=bool)
        action_eject = np.zeros((batch,), dtype=bool)
        for i, idx in enumerate(action_indices.tolist()):
            tx, ty, split, eject = self.action_space.to_world_input(
                action_index=int(idx),
                center_x=state.controlled_center_x,
                center_y=state.controlled_center_y,
                world_width=config.WORLD_WIDTH,
                world_height=config.WORLD_HEIGHT,
            )
            action_targets_x[i] = tx
            action_targets_y[i] = ty
            action_split[i] = split
            action_eject[i] = eject

        device = self.device
        x = torch.tensor(state.blob_x, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch, 1)
        y = torch.tensor(state.blob_y, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch, 1)
        mass = torch.tensor(state.blob_mass, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch, 1)
        vx = torch.tensor(state.blob_vx, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch, 1)
        vy = torch.tensor(state.blob_vy, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch, 1)
        active = torch.ones((batch, blobs), dtype=torch.bool, device=device)
        owner = torch.tensor(state.blob_owner, dtype=torch.long, device=device)
        owner_idx = owner.unsqueeze(0).expand(batch, -1)

        target_x = torch.tensor(state.target_x, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch, 1)
        target_y = torch.tensor(state.target_y, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch, 1)
        tx_control = torch.tensor(action_targets_x, dtype=torch.float32, device=device)
        ty_control = torch.tensor(action_targets_y, dtype=torch.float32, device=device)
        split_action = torch.tensor(action_split, dtype=torch.bool, device=device)
        eject_action = torch.tensor(action_eject, dtype=torch.bool, device=device)
        control_idx = int(state.controlled_player_idx)

        control_blob_mask = (owner == control_idx).unsqueeze(0).expand(batch, -1)
        split_boost = split_action.float().unsqueeze(1) * control_blob_mask.float()

        if state.food_x.size > 0:
            food_x = torch.tensor(state.food_x, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch, 1)
            food_y = torch.tensor(state.food_y, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch, 1)
            food_mass = torch.tensor(state.food_mass, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch, 1)
            food_active = torch.ones_like(food_mass, dtype=torch.bool)
        else:
            food_x = food_y = food_mass = food_active = None

        if state.ejected_x.size > 0:
            ej_x = torch.tensor(state.ejected_x, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch, 1)
            ej_y = torch.tensor(state.ejected_y, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch, 1)
            ej_mass = torch.tensor(state.ejected_mass, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch, 1)
            ej_active = torch.ones_like(ej_mass, dtype=torch.bool)
        else:
            ej_x = ej_y = ej_mass = ej_active = None

        if ej_mass is not None:
            can_eject = (
                eject_action.unsqueeze(1)
                & control_blob_mask
                & (mass > config.PLAYER_MIN_EJECT_MASS)
                & ((mass - config.PLAYER_EJECT_MASS) >= config.MIN_BLOB_MASS)
            )
            mass = torch.where(can_eject, mass - config.PLAYER_EJECT_MASS, mass)

        damping = max(0.0, 1.0 - config.BOOST_DAMPING * dt)
        for step in range(self.settings.horizon_steps):
            target_x[:, control_idx] = tx_control
            target_y[:, control_idx] = ty_control

            tx_blob = target_x.gather(1, owner_idx)
            ty_blob = target_y.gather(1, owner_idx)
            dx = tx_blob - x
            dy = ty_blob - y
            dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
            ux = dx / dist
            uy = dy / dist

            input_excess = torch.clamp(dist - config.INPUT_DEADZONE_WORLD, min=0.0)
            ramp = max(1.0, config.INPUT_SPEED_RAMP_WORLD)
            ease_power = max(0.25, config.INPUT_SPEED_EASE_EXPONENT)
            eased_distance = torch.pow(input_excess / ramp, ease_power)
            input_scale = 1.0 - torch.exp(-eased_distance)

            safe_mass = torch.clamp(mass, min=1.0)
            max_speed = torch.clamp(config.PLAYER_BASE_SPEED / torch.pow(safe_mass, config.SPEED_EXPONENT), min=config.PLAYER_MIN_SPEED)
            speed = max_speed * input_scale
            if step <= 1:
                speed = speed * (1.0 + 0.7 * split_boost)

            x = x + (ux * speed + vx) * dt
            y = y + (uy * speed + vy) * dt
            vx = vx * damping
            vy = vy * damping

            radius = torch.sqrt(torch.clamp(mass, min=1.0)) * config.BLOB_RADIUS_FACTOR
            clamp_r = radius * config.BLOB_BOUNDARY_FACTOR
            x = torch.clamp(x, min=clamp_r, max=config.WORLD_WIDTH - clamp_r)
            y = torch.clamp(y, min=clamp_r, max=config.WORLD_HEIGHT - clamp_r)

            if food_mass is not None:
                mass, food_active = self._consume_particles(
                    x=x,
                    y=y,
                    mass=mass,
                    active=active,
                    part_x=food_x,
                    part_y=food_y,
                    part_mass=food_mass,
                    part_active=food_active,
                    eat_factor=config.FOOD_EAT_RANGE_FACTOR,
                )

            if ej_mass is not None:
                mass, ej_active = self._consume_particles(
                    x=x,
                    y=y,
                    mass=mass,
                    active=active,
                    part_x=ej_x,
                    part_y=ej_y,
                    part_mass=ej_mass,
                    part_active=ej_active,
                    eat_factor=config.EJECTED_EAT_RANGE_FACTOR,
                )

            mass, active = self._resolve_blob_eating(
                x=x,
                y=y,
                mass=mass,
                active=active,
                owner=owner,
            )

        control_mass = (mass * active.float() * control_blob_mask.float()).sum(dim=1)
        alive = torch.where((active & control_blob_mask).any(dim=1), 1.0, -1.0)
        start_mass = torch.tensor(state.start_controlled_mass, dtype=torch.float32, device=device)
        mass_gain = (control_mass - start_mass) / start_mass

        player_mass = torch.zeros((batch, players), dtype=torch.float32, device=device)
        player_mass.scatter_add_(1, owner_idx, mass * active.float())
        controlled_player_mass = player_mass[:, control_idx]
        better = (player_mass > controlled_player_mass.unsqueeze(1)).sum(dim=1).float()
        if players > 1:
            rank_score = 1.0 - (better / float(players - 1)) * 2.0
        else:
            rank_score = torch.ones((batch,), dtype=torch.float32, device=device)

        immediate = 0.65 * mass_gain + 0.2 * rank_score + 0.15 * alive
        return torch.clamp(immediate, min=-1.0, max=1.0).detach().cpu().numpy().astype(np.float32)

    def _consume_particles(
        self,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        mass: torch.Tensor,
        active: torch.Tensor,
        part_x: torch.Tensor,
        part_y: torch.Tensor,
        part_mass: torch.Tensor,
        part_active: torch.Tensor,
        eat_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if part_mass is None or part_mass.shape[1] == 0:
            return mass, part_active

        radius_blob = torch.sqrt(torch.clamp(mass, min=1.0)) * config.BLOB_RADIUS_FACTOR
        radius_part = torch.sqrt(torch.clamp(part_mass, min=1e-6)) * config.BLOB_RADIUS_FACTOR
        dx = x.unsqueeze(2) - part_x.unsqueeze(1)
        dy = y.unsqueeze(2) - part_y.unsqueeze(1)
        dist2 = dx * dx + dy * dy
        eat_dist = (radius_blob.unsqueeze(2) + radius_part.unsqueeze(1)) * eat_factor
        can_eat = (
            dist2 <= eat_dist * eat_dist
        ) & active.unsqueeze(2) & part_active.unsqueeze(1)

        neg_inf = torch.tensor(-1e9, dtype=mass.dtype, device=mass.device)
        mass_candidates = torch.where(can_eat, mass.unsqueeze(2), neg_inf)
        best_mass, eater_idx = mass_candidates.max(dim=1)
        has_eater = best_mass > -1e8

        add_values = torch.where(has_eater, part_mass, torch.zeros_like(part_mass))
        safe_idx = torch.where(has_eater, eater_idx, torch.zeros_like(eater_idx))
        gain = torch.zeros_like(mass)
        gain.scatter_add_(1, safe_idx, add_values)
        mass = mass + gain
        part_active = part_active & (~has_eater)
        return mass, part_active

    def _resolve_blob_eating(
        self,
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        mass: torch.Tensor,
        active: torch.Tensor,
        owner: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        blobs = mass.shape[1]
        if blobs <= 1:
            return mass, active

        radius = torch.sqrt(torch.clamp(mass, min=1.0)) * config.BLOB_RADIUS_FACTOR
        dx = x.unsqueeze(2) - x.unsqueeze(1)
        dy = y.unsqueeze(2) - y.unsqueeze(1)
        dist2 = dx * dx + dy * dy

        mass_i = mass.unsqueeze(2)
        mass_j = mass.unsqueeze(1)
        radius_i = radius.unsqueeze(2)
        radius_j = radius.unsqueeze(1)

        can_mass = mass_i >= mass_j * config.BLOB_EAT_RATIO
        eat_distance = radius_i - (radius_j * config.BLOB_EAT_OVERLAP)
        can_dist = (eat_distance > 0.0) & (dist2 <= eat_distance * eat_distance)
        active_mask = active.unsqueeze(2) & active.unsqueeze(1)
        owner_mask = owner.view(1, blobs, 1) != owner.view(1, 1, blobs)
        eye = torch.eye(blobs, dtype=torch.bool, device=mass.device).unsqueeze(0)

        can_eat = can_mass & can_dist & active_mask & owner_mask & (~eye)
        neg_inf = torch.tensor(-1e9, dtype=mass.dtype, device=mass.device)
        eater_candidates = torch.where(can_eat, mass_i, neg_inf)
        best_mass, eater_idx = eater_candidates.max(dim=1)
        victim_eaten = best_mass > -1e8

        victim_mass = mass
        add_values = torch.where(victim_eaten, victim_mass, torch.zeros_like(victim_mass))
        safe_idx = torch.where(victim_eaten, eater_idx, torch.zeros_like(eater_idx))
        gain = torch.zeros_like(mass)
        gain.scatter_add_(1, safe_idx, add_values)
        mass = mass + gain

        active = active & (~victim_eaten)
        mass = torch.where(active, mass, torch.zeros_like(mass))
        return mass, active


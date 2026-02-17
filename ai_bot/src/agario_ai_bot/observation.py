"""Feature extraction from the authoritative GameWorld."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, tanh

import numpy as np

from ._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from agario import config
from agario.models import Blob, EjectedMass, Food, Player, Virus
from agario.world import GameWorld

MAX_MY_BLOBS = 16
MAX_ENEMY_BLOBS = 24
MAX_FOODS = 48
MAX_EJECTED = 24
MAX_VIRUSES = 12

GLOBAL_FEATURES = 13
MY_BLOB_FEATURES = 6
ENEMY_BLOB_FEATURES = 7
FOOD_FEATURES = 3
EJECTED_FEATURES = 4
VIRUS_FEATURES = 4

OBSERVATION_SIZE = (
    GLOBAL_FEATURES
    + MAX_MY_BLOBS * MY_BLOB_FEATURES
    + MAX_ENEMY_BLOBS * ENEMY_BLOB_FEATURES
    + MAX_FOODS * FOOD_FEATURES
    + MAX_EJECTED * EJECTED_FEATURES
    + MAX_VIRUSES * VIRUS_FEATURES
)


def _closest_distance(cx: float, cy: float, points: list[tuple[float, float]]) -> float:
    if not points:
        return 1.0
    dist = min(hypot(px - cx, py - cy) for px, py in points)
    max_dist = hypot(config.WORLD_WIDTH, config.WORLD_HEIGHT)
    return min(1.0, dist / max(1.0, max_dist))


def _rank_score(players: list[Player], player_id: str) -> float:
    ranked = sorted(players, key=lambda p: p.total_mass, reverse=True)
    if not ranked:
        return 0.0
    if len(ranked) == 1:
        return 1.0
    for idx, player in enumerate(ranked):
        if player.id == player_id:
            frac = idx / float(len(ranked) - 1)
            return 1.0 - frac * 2.0
    return 0.0


@dataclass(slots=True, frozen=True)
class BuiltObservation:
    vector: np.ndarray
    center_x: float
    center_y: float
    mass: float
    rank_score: float


def build_observation(world: GameWorld, player_id: str, now: float) -> BuiltObservation:
    player = world.players.get(player_id)
    if player is None:
        return BuiltObservation(
            vector=np.zeros(OBSERVATION_SIZE, dtype=np.float32),
            center_x=config.WORLD_WIDTH * 0.5,
            center_y=config.WORLD_HEIGHT * 0.5,
            mass=0.0,
            rank_score=-1.0,
        )

    cx, cy = player.center()
    total_mass = player.total_mass
    largest_radius = max((blob.radius for blob in player.blobs.values()), default=0.0)

    split_ready = any(blob.mass >= config.PLAYER_MIN_SPLIT_MASS for blob in player.blobs.values())
    eject_ready = any(blob.mass > config.PLAYER_MIN_EJECT_MASS for blob in player.blobs.values())
    split_cd = max(0.0, config.SPLIT_COOLDOWN_SECONDS - (now - player.last_split_at)) / config.SPLIT_COOLDOWN_SECONDS
    eject_cd = max(0.0, config.EJECT_COOLDOWN_SECONDS - (now - player.last_eject_at)) / config.EJECT_COOLDOWN_SECONDS

    enemy_blobs: list[Blob] = []
    for other in world.players.values():
        if other.id == player.id:
            continue
        enemy_blobs.extend(other.blobs.values())

    foods = list(world.foods.values())
    ejected = list(world.ejected.values())
    viruses = list(world.viruses.values())
    all_players = list(world.players.values())

    nearest_food = _closest_distance(cx, cy, [(f.x, f.y) for f in foods])
    nearest_enemy = _closest_distance(cx, cy, [(b.x, b.y) for b in enemy_blobs])

    my_mass = max(1.0, total_mass)
    threat_points = [(b.x, b.y) for b in enemy_blobs if b.mass >= my_mass * config.BLOB_EAT_RATIO * 0.9]
    nearest_threat = _closest_distance(cx, cy, threat_points)
    rank = _rank_score(all_players, player.id)

    feature_rows: list[float] = [
        cx / config.WORLD_WIDTH,
        cy / config.WORLD_HEIGHT,
        tanh(total_mass / 900.0),
        tanh(largest_radius / 160.0),
        min(1.0, len(player.blobs) / float(config.MAX_PLAYER_BLOBS)),
        1.0 if split_ready else 0.0,
        1.0 if eject_ready else 0.0,
        split_cd,
        eject_cd,
        nearest_food,
        nearest_enemy,
        nearest_threat,
        rank,
    ]

    my_blob_items = sorted(player.blobs.values(), key=lambda b: b.mass, reverse=True)
    for blob in my_blob_items[:MAX_MY_BLOBS]:
        feature_rows.extend(
            [
                (blob.x - cx) / config.WORLD_WIDTH,
                (blob.y - cy) / config.WORLD_HEIGHT,
                tanh(blob.mass / 450.0),
                tanh(blob.radius / 100.0),
                blob.vx / 1200.0,
                blob.vy / 1200.0,
            ]
        )
    feature_rows.extend([0.0] * ((MAX_MY_BLOBS - min(MAX_MY_BLOBS, len(my_blob_items))) * MY_BLOB_FEATURES))

    sorted_enemy_blobs = sorted(enemy_blobs, key=lambda b: hypot(b.x - cx, b.y - cy))
    for blob in sorted_enemy_blobs[:MAX_ENEMY_BLOBS]:
        can_eat_me = 1.0 if blob.mass > my_mass * config.BLOB_EAT_RATIO else 0.0
        can_i_eat = 1.0 if my_mass > blob.mass * config.BLOB_EAT_RATIO else 0.0
        feature_rows.extend(
            [
                (blob.x - cx) / config.WORLD_WIDTH,
                (blob.y - cy) / config.WORLD_HEIGHT,
                tanh(blob.mass / my_mass),
                tanh(blob.radius / max(1.0, largest_radius)),
                can_eat_me,
                can_i_eat,
                1.0 if blob.player_id == player.id else 0.0,
            ]
        )
    feature_rows.extend([0.0] * ((MAX_ENEMY_BLOBS - min(MAX_ENEMY_BLOBS, len(sorted_enemy_blobs))) * ENEMY_BLOB_FEATURES))

    sorted_foods = sorted(foods, key=lambda f: hypot(f.x - cx, f.y - cy))
    for food in sorted_foods[:MAX_FOODS]:
        feature_rows.extend(
            [
                (food.x - cx) / config.WORLD_WIDTH,
                (food.y - cy) / config.WORLD_HEIGHT,
                tanh(food.mass / 8.0),
            ]
        )
    feature_rows.extend([0.0] * ((MAX_FOODS - min(MAX_FOODS, len(sorted_foods))) * FOOD_FEATURES))

    sorted_ejected = sorted(ejected, key=lambda e: hypot(e.x - cx, e.y - cy))
    for item in sorted_ejected[:MAX_EJECTED]:
        feature_rows.extend(
            [
                (item.x - cx) / config.WORLD_WIDTH,
                (item.y - cy) / config.WORLD_HEIGHT,
                tanh(item.mass / 16.0),
                min(1.0, item.ttl / max(1e-6, config.EJECTED_MASS_LIFETIME)),
            ]
        )
    feature_rows.extend([0.0] * ((MAX_EJECTED - min(MAX_EJECTED, len(sorted_ejected))) * EJECTED_FEATURES))

    sorted_viruses = sorted(viruses, key=lambda v: hypot(v.x - cx, v.y - cy))
    for virus in sorted_viruses[:MAX_VIRUSES]:
        feature_rows.extend(
            [
                (virus.x - cx) / config.WORLD_WIDTH,
                (virus.y - cy) / config.WORLD_HEIGHT,
                tanh(virus.mass / 260.0),
                tanh(virus.radius / 120.0),
            ]
        )
    feature_rows.extend([0.0] * ((MAX_VIRUSES - min(MAX_VIRUSES, len(sorted_viruses))) * VIRUS_FEATURES))

    vector = np.asarray(feature_rows, dtype=np.float32)
    if vector.size != OBSERVATION_SIZE:
        padded = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
        limit = min(OBSERVATION_SIZE, vector.size)
        padded[:limit] = vector[:limit]
        vector = padded

    return BuiltObservation(
        vector=vector,
        center_x=cx,
        center_y=cy,
        mass=total_mass,
        rank_score=rank,
    )

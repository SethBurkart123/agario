"""RL observation layout and turn-relative action helpers.

Observation values are encoded exclusively by ``agario_core``. This module
keeps only the stable tensor layout used by the model and the mapping from a
policy action to an ordinary Agar.io mouse target.
"""

from __future__ import annotations

import math

VIEW = 1400.0
SPEED_NORM = 400.0
K_OWN = 16
K_ENEMY = 12
K_VIRUS = 3
N_SECTORS = 8
K_FOOD = 6
FOOD_F = 4
SELF_DIM = 28
OWN_F = 9
ENEMY_F = 15
OBS_DIM = (
    SELF_DIM
    + K_OWN * OWN_F
    + K_ENEMY * ENEMY_F
    + N_SECTORS * 2
    + K_FOOD * FOOD_F
    + K_VIRUS * 4
)

N_DIRECTIONS = 16
TURN_OFFSETS = (0, 1, -1, 2, -2, 4, -4, 8)
N_TURNS = len(TURN_OFFSETS)
N_OPS = 3
ACTION_TARGET_DIST = 600.0


def apply_turn(heading: int, turn_action: int) -> int:
    return (heading + TURN_OFFSETS[turn_action]) % N_DIRECTIONS


def speed_from_distance(dist: float) -> float:
    """Normalize mouse-target distance into the policy's 0..1 channel."""
    return min(max(dist / ACTION_TARGET_DIST, 0.0), 1.0)


def distance_from_speed(speed: float) -> float:
    """Map the policy channel to a real mouse-target distance."""
    return min(max(speed, 0.0), 1.0) * ACTION_TARGET_DIST


def action_to_target(
    direction: int, cx: float, cy: float, speed: float = 1.0
) -> tuple[float, float]:
    angle = (direction / N_DIRECTIONS) * 2.0 * math.pi
    distance = distance_from_speed(speed)
    return (
        cx + math.cos(angle) * distance,
        cy + math.sin(angle) * distance,
    )

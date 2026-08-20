"""Observation encoding (v2) shared by the training env and the live neural bot.

The encoder consumes a neutral `AgentPerception` snapshot (plain tuples) so
both the RL env (fed from agario_core compact state) and the live bot plugin
(fed from BotContext views) produce identical observations.

v2 design notes:
- Entities are expressed in polar form around the agent's center of mass:
  distance + bearing as (sin, cos), which is continuous and rotation-readable.
- Enemy cells carry true motion: closing speed (radial component of relative
  velocity — positive means it is approaching) and tangential speed. This is
  what makes charges, flees, and intercepts readable.
- Split-kill geometry is precomputed in both directions: "my biggest half
  post-split could eat this cell" + distance relative to my split reach, and
  the mirror for the opponent. The net learns *when* to use it, not the
  arithmetic.
- Every enemy cell carries its owner's context (total mass ratio, cell
  count), so a small fragment of a big split player reads differently from a
  small lone player.
- Merge lockout is explicit: every cell (own and enemy) carries "seconds
  until it may re-merge", so "I just split and I'm locked for 20s" and "that
  fragment can't consolidate to fight back" are both readable.
- Own cells carry velocity relative to my center, so post-split boost
  scatter is visible while it happens.
- Food stays coarse (8 polar sectors), viruses minimal — deliberately.

v4 additions (imitation-fidelity fixes):
- The agent's own control state is observable: current heading (sin/cos) and
  the previous action (turn one-hot, op one-hot, speed). Turn-relative
  actions are unlearnable without knowing the heading they're relative to.
- Nearest-K food pellets individually (polar), on top of the coarse sectors,
  so pellet-level routing is imitable.
- Movement has a continuous speed channel (0..1): the command target is
  placed at the distance whose physics response yields that speed, mirroring
  how the heuristic expert modulates speed via target distance.

Layout (features roughly in [-1, 1]):
  self:    28  (14 + heading 2 + prev turn 8 + prev op 3 + prev speed 1)
  own:     16 cells x 9   (all of them, sorted by mass desc)
  enemies: 12 cells x 15  (sorted by distance)
  food:    8 sectors x 2  + 6 nearest pellets x 4
  viruses: 3 x 4          (sorted by distance)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

VIEW = 1400.0
SPEED_NORM = 400.0
K_OWN = 16  # MAX_PLAYER_BLOBS — every own cell is always visible
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

EAT_RATIO = 1.12
VIRUS_MASS = 144.0
MIN_SPLIT_MASS = 90.0
MAX_BLOBS = 16
MERGE_DELAY = 25.0
# Split boost travel: 880 u/s decaying at 3.2/s integrates to ~275 world units.
SPLIT_TRAVEL = 275.0

# Action space: turn-relative movement x {none, split, eject}.
# The agent carries a heading (one of 16 compass directions); each decision
# picks a TURN relative to it. Smooth, committed trajectories are the default
# and per-step zigzag is impossible by construction — absolute-direction
# resampling at 12.5 Hz made even good policies look Brownian.
N_DIRECTIONS = 16  # heading resolution (22.5° each)
TURN_OFFSETS = (0, 1, -1, 2, -2, 4, -4, 8)  # in 22.5° units: 0, ±22.5°, ±45°, ±90°, 180°
N_TURNS = len(TURN_OFFSETS)
N_OPS = 3
ACTION_TARGET_DIST = 600.0

# Input response curve constants (mirror config.py / world physics): speed
# scale = 1 - exp(-(((dist - deadzone) / ramp) ** ease)).
INPUT_DEADZONE = 8.0
INPUT_RAMP = 82.0
INPUT_EASE = 0.7
MAX_SPEED_CMD = 0.995  # full speed; finite target distance


def apply_turn(heading: int, turn_action: int) -> int:
    return (heading + TURN_OFFSETS[turn_action]) % N_DIRECTIONS


def speed_from_distance(dist: float) -> float:
    """The speed scale the physics produces for a command at this distance."""
    excess = max(0.0, dist - INPUT_DEADZONE)
    return 1.0 - math.exp(-((excess / INPUT_RAMP) ** INPUT_EASE))


def distance_from_speed(speed: float) -> float:
    """Inverse of the response curve: where to place the target to command
    this speed scale."""
    s = min(max(speed, 0.0), MAX_SPEED_CMD)
    if s <= 0.0:
        return 0.0
    return INPUT_DEADZONE + INPUT_RAMP * (-math.log(1.0 - s)) ** (1.0 / INPUT_EASE)


@dataclass(slots=True)
class AgentPerception:
    """Everything the encoder needs, engine-agnostic.

    own_blobs:   (x, y, mass, radius, vx, vy, merge_in)
    enemy_blobs: (x, y, mass, radius, vx, vy, owner_total_mass, owner_cells, merge_in)
    foods/ejected: (x, y, mass)
    viruses:     (x, y, radius)

    merge_in is seconds until the cell may re-merge (0 = ready now).
    heading / prev_* describe the agent's own control state.
    """

    world_w: float
    world_h: float
    heading: int
    prev_turn: int
    prev_op: int
    prev_speed: float
    own_blobs: list[tuple[float, float, float, float, float, float, float]]
    enemy_blobs: list[
        tuple[float, float, float, float, float, float, float, int, float]
    ]
    foods: list[tuple[float, float, float]]
    ejected: list[tuple[float, float, float]]
    viruses: list[tuple[float, float, float]]


def encode(p: AgentPerception, out: np.ndarray | None = None) -> np.ndarray:
    if out is None:
        out = np.zeros(OBS_DIM, dtype=np.float32)
    else:
        out.fill(0.0)

    if not p.own_blobs:
        return out

    total_mass = sum(b[2] for b in p.own_blobs)
    cx = sum(b[0] * b[2] for b in p.own_blobs) / total_mass
    cy = sum(b[1] * b[2] for b in p.own_blobs) / total_mass
    my_vx = sum(b[4] * b[2] for b in p.own_blobs) / total_mass
    my_vy = sum(b[5] * b[2] for b in p.own_blobs) / total_mass
    own_sorted = sorted(p.own_blobs, key=lambda b: -b[2])
    largest_mass = own_sorted[0][2]
    largest_radius = own_sorted[0][3]
    max_merge_in = max(b[6] for b in p.own_blobs)
    can_split = largest_mass >= MIN_SPLIT_MASS and len(p.own_blobs) < MAX_BLOBS
    # Reach of my biggest cell's split half (its center travels ~SPLIT_TRAVEL
    # past my edge); per-target we add the target radius.
    my_split_reach = largest_radius + SPLIT_TRAVEL
    my_half_mass = largest_mass / 2.0

    i = 0
    out[i] = np.clip(math.log(total_mass / 560.0) / 3.0, -1.5, 1.5); i += 1
    out[i] = len(p.own_blobs) / 16.0; i += 1
    out[i] = min(largest_radius / 250.0, 2.0); i += 1
    out[i] = cx / p.world_w * 2.0 - 1.0; i += 1
    out[i] = cy / p.world_h * 2.0 - 1.0; i += 1
    out[i] = min(min(cx, p.world_w - cx) / VIEW, 1.0); i += 1
    out[i] = min(min(cy, p.world_h - cy) / VIEW, 1.0); i += 1
    out[i] = 1.0 if can_split else 0.0; i += 1
    out[i] = min(1400.0 / (largest_mass**0.45) / 400.0, 1.5); i += 1
    out[i] = 1.0 if largest_mass > VIRUS_MASS * 1.15 else 0.0; i += 1
    out[i] = np.clip(my_vx / SPEED_NORM, -2.0, 2.0); i += 1
    out[i] = np.clip(my_vy / SPEED_NORM, -2.0, 2.0); i += 1
    out[i] = min(my_split_reach / VIEW, 1.0); i += 1
    out[i] = min(max_merge_in / MERGE_DELAY, 1.0); i += 1
    # Own control state: heading + previous action (turn-relative actions are
    # unlearnable without these).
    heading_angle = (p.heading / N_DIRECTIONS) * 2.0 * math.pi
    out[i] = math.sin(heading_angle); i += 1
    out[i] = math.cos(heading_angle); i += 1
    out[i + p.prev_turn] = 1.0; i += N_TURNS
    out[i + p.prev_op] = 1.0; i += N_OPS
    out[i] = min(max(p.prev_speed, 0.0), 1.0); i += 1
    assert i == SELF_DIM

    for bx, by, bm, br, bvx, bvy, merge_in in own_sorted[:K_OWN]:
        dx, dy = bx - cx, by - cy
        dist = math.hypot(dx, dy)
        sin_b, cos_b = (dy / dist, dx / dist) if dist > 1e-6 else (0.0, 1.0)
        out[i] = min(dist / VIEW, 1.5); i += 1
        out[i] = sin_b; i += 1
        out[i] = cos_b; i += 1
        out[i] = bm / total_mass; i += 1
        out[i] = min(br / 250.0, 2.0); i += 1
        out[i] = 1.0 if bm >= MIN_SPLIT_MASS else 0.0; i += 1
        out[i] = np.clip((bvx - my_vx) / SPEED_NORM, -2.5, 2.5); i += 1
        out[i] = np.clip((bvy - my_vy) / SPEED_NORM, -2.5, 2.5); i += 1
        out[i] = min(merge_in / MERGE_DELAY, 1.0); i += 1
    i = SELF_DIM + K_OWN * OWN_F

    enemies = []
    for ex, ey, em, er, evx, evy, o_mass, o_cells, merge_in in p.enemy_blobs:
        dx, dy = ex - cx, ey - cy
        dist = math.hypot(dx, dy)
        if dist <= VIEW * 1.5:
            enemies.append((dist, dx, dy, em, er, evx, evy, o_mass, o_cells, merge_in))
    enemies.sort(key=lambda e: e[0])
    for dist, dx, dy, em, er, evx, evy, o_mass, o_cells, merge_in in enemies[:K_ENEMY]:
        if dist > 1e-6:
            ux, uy = dx / dist, dy / dist
        else:
            ux, uy = 1.0, 0.0
        rel_vx, rel_vy = evx - my_vx, evy - my_vy
        # Positive closing speed = the gap is shrinking.
        closing = -(rel_vx * ux + rel_vy * uy)
        tangential = rel_vx * -uy + rel_vy * ux

        out[i] = min(dist / VIEW, 1.5); i += 1
        out[i] = uy; i += 1  # sin(bearing)
        out[i] = ux; i += 1  # cos(bearing)
        out[i] = np.clip(closing / SPEED_NORM, -2.5, 2.5); i += 1
        out[i] = np.clip(tangential / SPEED_NORM, -2.5, 2.5); i += 1
        out[i] = np.clip(math.log(em / largest_mass) / 2.0, -2.0, 2.0); i += 1
        out[i] = 1.0 if largest_mass >= em * EAT_RATIO else 0.0; i += 1
        out[i] = 1.0 if em >= largest_mass * EAT_RATIO else 0.0; i += 1
        # Split-kill geometry, me -> them.
        out[i] = 1.0 if (can_split and my_half_mass >= em * EAT_RATIO) else 0.0; i += 1
        out[i] = np.clip(dist / (my_split_reach + er), 0.0, 3.0); i += 1
        # Split-kill geometry, them -> me (can their half eat my biggest?).
        their_reach = er + SPLIT_TRAVEL
        out[i] = 1.0 if (em / 2.0 >= largest_mass * EAT_RATIO and em >= MIN_SPLIT_MASS) else 0.0; i += 1
        out[i] = np.clip(dist / (their_reach + largest_radius), 0.0, 3.0); i += 1
        # Owner context: this cell might be a fragment of something bigger.
        out[i] = np.clip(math.log(max(o_mass, 1.0) / total_mass) / 2.0, -2.0, 2.0); i += 1
        out[i] = o_cells / 16.0; i += 1
        out[i] = min(merge_in / MERGE_DELAY, 1.0); i += 1
    i = SELF_DIM + K_OWN * OWN_F + K_ENEMY * ENEMY_F

    sector_mass = [0.0] * N_SECTORS
    sector_near = [VIEW] * N_SECTORS
    for fx, fy, fm in p.foods:
        dx, dy = fx - cx, fy - cy
        dist = math.hypot(dx, dy)
        if dist > VIEW:
            continue
        s = int(((math.atan2(dy, dx) + math.pi) / (2.0 * math.pi)) * N_SECTORS) % N_SECTORS
        sector_mass[s] += fm
        if dist < sector_near[s]:
            sector_near[s] = dist
    for ex, ey, em in p.ejected:
        dx, dy = ex - cx, ey - cy
        dist = math.hypot(dx, dy)
        if dist > VIEW:
            continue
        s = int(((math.atan2(dy, dx) + math.pi) / (2.0 * math.pi)) * N_SECTORS) % N_SECTORS
        sector_mass[s] += em * 2.0  # ejected mass is denser value than pellets
        if dist < sector_near[s]:
            sector_near[s] = dist
    for s in range(N_SECTORS):
        out[i] = min(sector_mass[s] / 40.0, 2.0); i += 1
        out[i] = sector_near[s] / VIEW; i += 1

    # Nearest pellets individually (ejected counts; it's the best food).
    pellets = []
    for fx, fy, fm in p.foods:
        dx, dy = fx - cx, fy - cy
        dist = math.hypot(dx, dy)
        if dist <= VIEW:
            pellets.append((dist, dx, dy, fm))
    for ex, ey, em in p.ejected:
        dx, dy = ex - cx, ey - cy
        dist = math.hypot(dx, dy)
        if dist <= VIEW:
            pellets.append((dist, dx, dy, em))
    pellets.sort(key=lambda f: f[0])
    for dist, dx, dy, fm in pellets[:K_FOOD]:
        ux, uy = (dx / dist, dy / dist) if dist > 1e-6 else (1.0, 0.0)
        out[i] = min(dist / VIEW, 1.0); i += 1
        out[i] = uy; i += 1
        out[i] = ux; i += 1
        out[i] = min(fm / 12.0, 2.0); i += 1
    i = SELF_DIM + K_OWN * OWN_F + K_ENEMY * ENEMY_F + N_SECTORS * 2 + K_FOOD * FOOD_F

    viruses = []
    for vx, vy, vr in p.viruses:
        dx, dy = vx - cx, vy - cy
        dist = math.hypot(dx, dy)
        if dist <= VIEW * 1.5:
            viruses.append((dist, dx, dy, vr))
    viruses.sort(key=lambda v: v[0])
    for dist, dx, dy, _vr in viruses[:K_VIRUS]:
        if dist > 1e-6:
            ux, uy = dx / dist, dy / dist
        else:
            ux, uy = 1.0, 0.0
        out[i] = min(dist / VIEW, 1.5); i += 1
        out[i] = uy; i += 1
        out[i] = ux; i += 1
        out[i] = 1.0 if largest_mass > VIRUS_MASS * 1.15 else 0.0; i += 1

    return out


def action_to_target(
    direction: int, cx: float, cy: float, speed: float = 1.0
) -> tuple[float, float]:
    angle = (direction / N_DIRECTIONS) * 2.0 * math.pi
    dist = distance_from_speed(speed) if speed < MAX_SPEED_CMD else ACTION_TARGET_DIST
    return (cx + math.cos(angle) * dist, cy + math.sin(angle) * dist)

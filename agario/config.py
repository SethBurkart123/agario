"""Runtime tunables for the Agar-like game server."""

from __future__ import annotations

import os

WORLD_WIDTH = 6000.0
WORLD_HEIGHT = 6000.0

TICK_RATE = 75

FOOD_TARGET_COUNT = 1200
FOOD_MIN_MASS = 1.0
FOOD_MAX_MASS = 3.8
FOOD_RADIUS_FACTOR = 4.0
FOOD_EAT_RANGE_FACTOR = 1.06

VIRUS_COUNT = 24
VIRUS_MASS = 144.0
VIRUS_RADIUS_FACTOR = 4.0
VIRUS_BONUS_MASS = 60.0

PLAYER_START_MASS = 560.0
PLAYER_MIN_SPLIT_MASS = 90.0
PLAYER_MIN_EJECT_MASS = 28.0
PLAYER_EJECT_MASS = 12.0
MAX_PLAYER_BLOBS = 16
MIN_BLOB_MASS = 10.0

MASS_DECAY_START = 200.0
MASS_DECAY_END = 1800.0
MASS_DECAY_MIN_RATE = 0.00005
MASS_DECAY_MAX_RATE = 0.006
MASS_DECAY_CURVE = 2.6

BLOB_RADIUS_FACTOR = 4.0
PLAYER_BASE_SPEED = 1400.0
PLAYER_MIN_SPEED = 140.0
SPEED_EXPONENT = 0.45
BLOB_BOUNDARY_FACTOR = 0.84
INPUT_DEADZONE_WORLD = 8.0
INPUT_SPEED_RAMP_WORLD = 82.0
INPUT_SPEED_EASE_EXPONENT = 0.7

SPLIT_BOOST_SPEED = 880.0
EJECT_BOOST_SPEED = 780.0
BOOST_DAMPING = 3.2
SOFTBODY_MIN_DIST_UNMERGED = 0.72
SOFTBODY_MIN_DIST_MERGED = 0.08

EJECTED_MASS_LIFETIME = 12.0
EJECTED_EAT_RANGE_FACTOR = 1.04
MERGE_DELAY_SECONDS = 25.0
MERGE_COVERAGE_FRACTION = 0.5
SPLIT_COOLDOWN_SECONDS = 0.12
EJECT_COOLDOWN_SECONDS = 0.12

BLOB_EAT_RATIO = 1.12
BLOB_EAT_OVERLAP = 0.78

VIRUS_SPLIT_MIN_PARTS = 4
VIRUS_SPLIT_MAX_PARTS = 8

VIEW_WIDTH = 1900.0
VIEW_HEIGHT = 1100.0
VIEW_PADDING = 400.0
SPLIT_ZOOM_MAX_PENALTY = 0.14
SPLIT_ZOOM_MAX_PENALTY_HUGE = 0.26
SPLIT_ZOOM_MASS_SOFT_CAP = 600.0
SPLIT_ZOOM_MASS_HARD_CAP = 2600.0
SPLIT_ZOOM_MASS_CURVE = 1.4
SPLIT_ZOOM_DECAY = 0.25

INPUT_HZ = 90
MAX_PLAYER_NAME_LENGTH = 18


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_csv(name: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(part.strip() for part in raw.split(",") if part.strip())


BOTS_ENABLED = _env_bool("AGARIO_BOTS_ENABLED", True)
BOT_PLUGIN_MODULES = _env_csv("AGARIO_BOT_PLUGIN_MODULES", "agario.bot_plugins.core")
BOT_SPECS = os.getenv(
    "AGARIO_BOT_SPECS",
    "solo_smart:16",
)
BOT_RANDOM_SEED = _env_int("AGARIO_BOT_RANDOM_SEED", 1337)
BOT_SPAWN_ON_EATEN = _env_bool("AGARIO_BOT_SPAWN_ON_EATEN", True)
BOT_SPAWN_PER_ELIMINATION = max(0, _env_int("AGARIO_BOT_SPAWN_PER_ELIMINATION", 1))
BOT_MAX_ACTIVE = max(1, _env_int("AGARIO_BOT_MAX_ACTIVE", 40))

PLAYER_COLORS = [
    "#21B8FF",
    "#33FF3A",
    "#FF364B",
    "#FFBC09",
    "#8E31FF",
    "#FF8A1F",
    "#26E5DF",
    "#FF2CCB",
]

FOOD_COLORS = [
    "#FF2A40",
    "#1D38FF",
    "#22D9F0",
    "#59F12F",
    "#7D2BFF",
    "#FFE625",
    "#FF8D1F",
    "#FF1FCF",
]

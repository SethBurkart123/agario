"""Server and bot-process settings. Simulation mechanics live in Rust."""

from __future__ import annotations

import os

INPUT_HZ = 90


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)).strip())
    except ValueError:
        return default


def _env_csv(name: str, default: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in os.getenv(name, default).split(",") if part.strip())


BOTS_ENABLED = _env_bool("AGARIO_BOTS_ENABLED", True)
BOT_PLUGIN_MODULES = _env_csv("AGARIO_BOT_PLUGIN_MODULES", "bot_solutions.programmatic")
BOT_SPECS = os.getenv("AGARIO_BOT_SPECS", "solo_smart:16")
BOT_RANDOM_SEED = _env_int("AGARIO_BOT_RANDOM_SEED", 1337)
BOT_SPAWN_ON_EATEN = _env_bool("AGARIO_BOT_SPAWN_ON_EATEN", True)
BOT_SPAWN_PER_ELIMINATION = max(0, _env_int("AGARIO_BOT_SPAWN_PER_ELIMINATION", 1))
BOT_MAX_ACTIVE = max(1, _env_int("AGARIO_BOT_MAX_ACTIVE", 40))

"""Core bot plugin contracts and immutable world-view models."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True, frozen=True)
class BlobView:
    id: str
    player_id: str
    x: float
    y: float
    mass: float
    radius: float


@dataclass(slots=True, frozen=True)
class PlayerView:
    id: str
    name: str
    color: str
    is_bot: bool
    plugin_name: str | None
    team_id: str | None
    total_mass: float
    blobs: tuple[BlobView, ...]


@dataclass(slots=True, frozen=True)
class FoodView:
    id: str
    x: float
    y: float
    mass: float
    radius: float
    color: str


@dataclass(slots=True, frozen=True)
class EjectedView:
    id: str
    x: float
    y: float
    mass: float
    radius: float
    owner_id: str
    ttl: float


@dataclass(slots=True, frozen=True)
class VirusView:
    id: str
    x: float
    y: float
    mass: float
    radius: float


@dataclass(slots=True, frozen=True)
class BotAction:
    target_x: float
    target_y: float
    split: bool = False
    eject: bool = False


@dataclass(slots=True, frozen=True)
class BotSpec:
    plugin_name: str
    count: int = 1
    team_id: str | None = None
    name_prefix: str | None = None


@dataclass(slots=True)
class BotContext:
    now: float
    dt: float
    world_width: float
    world_height: float
    me: PlayerView
    players: tuple[PlayerView, ...]
    foods: tuple[FoodView, ...]
    ejected: tuple[EjectedView, ...]
    viruses: tuple[VirusView, ...]
    team_state: dict[str, Any]
    memory: dict[str, Any]


@dataclass(slots=True, frozen=True)
class BotInitContext:
    plugin_name: str
    bot_name: str
    team_id: str | None
    bot_index: int
    rng: random.Random


class BotBrain(Protocol):
    def decide(self, ctx: BotContext) -> BotAction:
        """Return desired input for this bot tick."""


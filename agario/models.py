"""Core dataclasses representing game entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

from . import config


@dataclass(slots=True)
class Blob:
    id: str
    player_id: str
    x: float
    y: float
    mass: float
    vx: float = 0.0
    vy: float = 0.0
    can_merge_at: float = 0.0

    @property
    def radius(self) -> float:
        return sqrt(self.mass) * config.BLOB_RADIUS_FACTOR


@dataclass(slots=True)
class Player:
    id: str
    name: str
    color: str
    is_bot: bool = False
    bot_plugin: str | None = None
    bot_team: str | None = None
    blobs: dict[str, Blob] = field(default_factory=dict)
    target_x: float = 0.0
    target_y: float = 0.0
    split_requested: bool = False
    eject_requested: bool = False
    last_split_at: float = -1e9
    last_eject_at: float = -1e9

    @property
    def total_mass(self) -> float:
        return sum(blob.mass for blob in self.blobs.values())

    @property
    def is_alive(self) -> bool:
        return bool(self.blobs)

    def center(self) -> tuple[float, float]:
        if not self.blobs:
            return (0.0, 0.0)
        total = self.total_mass
        if total <= 0.0:
            blob = next(iter(self.blobs.values()))
            return (blob.x, blob.y)
        cx = sum(blob.x * blob.mass for blob in self.blobs.values()) / total
        cy = sum(blob.y * blob.mass for blob in self.blobs.values()) / total
        return (cx, cy)


@dataclass(slots=True)
class Food:
    id: str
    x: float
    y: float
    mass: float
    color: str

    @property
    def radius(self) -> float:
        return sqrt(self.mass) * config.FOOD_RADIUS_FACTOR


@dataclass(slots=True)
class EjectedMass:
    id: str
    x: float
    y: float
    mass: float
    owner_id: str
    vx: float
    vy: float
    ttl: float

    @property
    def radius(self) -> float:
        return sqrt(self.mass) * config.BLOB_RADIUS_FACTOR


@dataclass(slots=True)
class Virus:
    id: str
    x: float
    y: float
    mass: float

    @property
    def radius(self) -> float:
        return sqrt(self.mass) * config.VIRUS_RADIUS_FACTOR

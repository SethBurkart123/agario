"""Discrete movement + ability action space."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


@dataclass(frozen=True, slots=True)
class DecodedAction:
    direction_index: int
    magnitude_index: int
    ability_index: int  # 0=none, 1=split, 2=eject


class ActionSpace:
    """Maps a compact discrete index to game controls."""

    def __init__(
        self,
        *,
        direction_bins: int,
        magnitude_bins: tuple[float, ...],
        target_distance: float,
    ) -> None:
        self.direction_bins = direction_bins
        self.magnitude_bins = magnitude_bins
        self.target_distance = target_distance
        self.ability_bins = 3
        self.size = self.direction_bins * len(self.magnitude_bins) * self.ability_bins

    def decode(self, action_index: int) -> DecodedAction:
        idx = int(action_index) % self.size
        ability_index = idx % self.ability_bins
        idx //= self.ability_bins
        magnitude_index = idx % len(self.magnitude_bins)
        direction_index = idx // len(self.magnitude_bins)
        return DecodedAction(
            direction_index=direction_index,
            magnitude_index=magnitude_index,
            ability_index=ability_index,
        )

    def to_world_input(
        self,
        *,
        action_index: int,
        center_x: float,
        center_y: float,
        world_width: float,
        world_height: float,
    ) -> tuple[float, float, bool, bool]:
        action = self.decode(action_index)
        angle = (2.0 * pi * action.direction_index) / float(self.direction_bins)
        magnitude = self.magnitude_bins[action.magnitude_index]
        distance = 260.0 + self.target_distance * magnitude

        tx = center_x + cos(angle) * distance
        ty = center_y + sin(angle) * distance
        target_x = _clamp(tx, 0.0, world_width)
        target_y = _clamp(ty, 0.0, world_height)
        split = action.ability_index == 1
        eject = action.ability_index == 2
        return (target_x, target_y, split, eject)


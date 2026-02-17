"""Very small spatial hash for broad-phase entity lookups."""

from __future__ import annotations

from collections import defaultdict
from math import floor
from typing import Generic, TypeVar

T = TypeVar("T")


class SpatialHash(Generic[T]):
    def __init__(self, cell_size: float) -> None:
        self.cell_size = cell_size
        self._buckets: dict[tuple[int, int], list[T]] = defaultdict(list)

    def clear(self) -> None:
        self._buckets.clear()

    def _key(self, x: float, y: float) -> tuple[int, int]:
        return (floor(x / self.cell_size), floor(y / self.cell_size))

    def insert(self, x: float, y: float, item: T) -> None:
        self._buckets[self._key(x, y)].append(item)

    def query_rect(self, min_x: float, min_y: float, max_x: float, max_y: float) -> list[T]:
        min_cx = floor(min_x / self.cell_size)
        max_cx = floor(max_x / self.cell_size)
        min_cy = floor(min_y / self.cell_size)
        max_cy = floor(max_y / self.cell_size)

        hits: list[T] = []
        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                hits.extend(self._buckets.get((cx, cy), ()))
        return hits

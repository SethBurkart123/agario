"""Drives solo_smart heuristic brains inside RL arenas (opponent anchors)."""

from __future__ import annotations

import random

from agario.bots.types import (
    BlobView,
    BotContext,
    BotInitContext,
    EjectedView,
    FoodView,
    PlayerView,
    VirusView,
)
from bot_solutions.programmatic import SoloSmartBrain


class HeuristicDriver:
    """Owns SoloSmartBrain instances for a set of player ids in one arena."""

    def __init__(self, player_ids: list[str], seed: int) -> None:
        rng = random.Random(seed)
        self._brains: dict[str, SoloSmartBrain] = {}
        self._memories: dict[str, dict] = {}
        for idx, pid in enumerate(player_ids):
            init = BotInitContext(
                plugin_name="solo_smart",
                bot_name=f"Anchor-{idx}",
                team_id=None,
                bot_index=idx,
                rng=random.Random(rng.randint(0, 2_000_000_000)),
            )
            self._brains[pid] = SoloSmartBrain(init)
            self._memories[pid] = {}

    @staticmethod
    def build_views(world) -> tuple[dict[str, PlayerView], tuple, tuple, tuple, tuple]:
        players = []
        for pid, name, color, is_bot, plugin, team, _tx, _ty, blob_rows in world.players_compact():
            blobs = tuple(
                BlobView(
                    id=bid, player_id=pid, x=x, y=y, mass=mass, radius=radius,
                    vx=vx, vy=vy, can_merge_at=cma,
                )
                for bid, x, y, mass, radius, vx, vy, cma in blob_rows
            )
            players.append(
                PlayerView(
                    id=pid,
                    name=name,
                    color=color,
                    is_bot=is_bot,
                    plugin_name=plugin,
                    team_id=team,
                    total_mass=sum(b.mass for b in blobs),
                    blobs=blobs,
                )
            )
        players.sort(key=lambda p: p.id)
        players_by_id = {p.id: p for p in players}

        food_rows, ejected_rows, virus_rows = world.entities_compact()
        foods = tuple(
            FoodView(id=fid, x=x, y=y, mass=mass, radius=radius, color=color)
            for fid, x, y, mass, radius, color in food_rows
        )
        ejected = tuple(
            EjectedView(id=eid, x=x, y=y, mass=mass, radius=radius, owner_id=owner, ttl=ttl)
            for eid, x, y, mass, radius, owner, ttl in ejected_rows
        )
        viruses = tuple(
            VirusView(id=vid, x=x, y=y, mass=mass, radius=radius)
            for vid, x, y, mass, radius in virus_rows
        )
        return players_by_id, tuple(players), foods, ejected, viruses

    def apply(self, world, now: float, dt: float, world_w: float, world_h: float) -> None:
        if not self._brains:
            return
        players_by_id, players, foods, ejected, viruses = self.build_views(world)
        for pid, brain in self._brains.items():
            me = players_by_id.get(pid)
            if me is None or not me.blobs:
                continue
            ctx = BotContext(
                now=now,
                dt=dt,
                world_width=world_w,
                world_height=world_h,
                me=me,
                players=players,
                foods=foods,
                ejected=ejected,
                viruses=viruses,
                team_state={},
                memory=self._memories[pid],
            )
            try:
                action = brain.decide(ctx)
            except Exception:
                continue
            world.set_input(
                pid,
                min(max(action.target_x, 0.0), world_w),
                min(max(action.target_y, 0.0), world_h),
                split=bool(action.split),
                eject=bool(action.eject),
            )

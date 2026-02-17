"""Authoritative game-world simulation logic."""

from __future__ import annotations

import random
from itertools import count
from math import cos, exp, sin

from . import config
from .models import Blob, EjectedMass, Food, Player, Virus
from .spatial import SpatialHash


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return min(max(value, min_value), max_value)


def _distance_sq(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


def _unit_vec(dx: float, dy: float) -> tuple[float, float]:
    mag_sq = dx * dx + dy * dy
    if mag_sq <= 1e-9:
        return (1.0, 0.0)
    inv_mag = mag_sq ** -0.5
    return (dx * inv_mag, dy * inv_mag)


class GameWorld:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

        self.players: dict[str, Player] = {}
        self.foods: dict[str, Food] = {}
        self.ejected: dict[str, EjectedMass] = {}
        self.viruses: dict[str, Virus] = {}

        self._player_ids = count(1)
        self._blob_ids = count(1)
        self._food_ids = count(1)
        self._ejected_ids = count(1)
        self._virus_ids = count(1)

        self.food_hash: SpatialHash[Food] = SpatialHash(150.0)
        self.blob_hash: SpatialHash[Blob] = SpatialHash(250.0)
        self.ejected_hash: SpatialHash[EjectedMass] = SpatialHash(180.0)

        self._spawn_initial_food()
        self._spawn_initial_viruses()
        self._rebuild_spatial_indexes()

    def __getstate__(self) -> dict:
        # Spatial hashes are derived data and expensive to deepcopy.
        state = dict(self.__dict__)
        state["food_hash"] = None
        state["blob_hash"] = None
        state["ejected_hash"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.food_hash = SpatialHash(150.0)
        self.blob_hash = SpatialHash(250.0)
        self.ejected_hash = SpatialHash(180.0)

    def _next_player_id(self) -> str:
        return f"p{next(self._player_ids)}"

    def _next_blob_id(self) -> str:
        return f"b{next(self._blob_ids)}"

    def _next_food_id(self) -> str:
        return f"f{next(self._food_ids)}"

    def _next_ejected_id(self) -> str:
        return f"e{next(self._ejected_ids)}"

    def _next_virus_id(self) -> str:
        return f"v{next(self._virus_ids)}"

    def add_player(
        self,
        player_name: str,
        now: float,
        *,
        is_bot: bool = False,
        bot_plugin: str | None = None,
        bot_team: str | None = None,
        color: str | None = None,
    ) -> Player:
        name = (player_name or "Cell").strip()[: config.MAX_PLAYER_NAME_LENGTH]
        if not name:
            name = "Cell"

        player_id = self._next_player_id()
        player_color = color or config.PLAYER_COLORS[(len(self.players)) % len(config.PLAYER_COLORS)]
        player = Player(
            id=player_id,
            name=name,
            color=player_color,
            is_bot=is_bot,
            bot_plugin=bot_plugin,
            bot_team=bot_team,
        )

        spawn_x, spawn_y = self._random_spawn(radius=40.0)
        blob = Blob(
            id=self._next_blob_id(),
            player_id=player.id,
            x=spawn_x,
            y=spawn_y,
            mass=config.PLAYER_START_MASS,
            can_merge_at=now + config.MERGE_DELAY_SECONDS,
        )
        player.blobs[blob.id] = blob
        player.target_x = spawn_x
        player.target_y = spawn_y

        self.players[player.id] = player
        return player

    def remove_player(self, player_id: str) -> None:
        self.players.pop(player_id, None)

    def set_input(
        self,
        player_id: str,
        target_x: float | None,
        target_y: float | None,
        split: bool,
        eject: bool,
    ) -> None:
        player = self.players.get(player_id)
        if player is None:
            return

        if target_x is not None and target_y is not None:
            try:
                player.target_x = float(target_x)
                player.target_y = float(target_y)
            except (TypeError, ValueError):
                pass
        if split:
            player.split_requested = True
        if eject:
            player.eject_requested = True

    def update(self, dt: float, now: float) -> None:
        self._respawn_eliminated_players(now)
        self._apply_actions(now)
        self._move_blobs(dt, now)
        self._move_ejected(dt)

        self._rebuild_spatial_indexes()
        self._resolve_blob_food_collisions()
        self._resolve_blob_ejected_collisions()
        self._resolve_blob_blob_collisions(now)
        self._resolve_virus_blob_collisions(now)

        self._spawn_food_to_target()
        self._rebuild_spatial_indexes()

    def _random_spawn(self, radius: float) -> tuple[float, float]:
        x = self.rng.uniform(radius, config.WORLD_WIDTH - radius)
        y = self.rng.uniform(radius, config.WORLD_HEIGHT - radius)
        return (x, y)

    def _spawn_initial_food(self) -> None:
        while len(self.foods) < config.FOOD_TARGET_COUNT:
            self._spawn_food()

    def _spawn_food(self) -> None:
        fx, fy = self._random_spawn(radius=8.0)
        mass = self.rng.uniform(config.FOOD_MIN_MASS, config.FOOD_MAX_MASS)
        food = Food(
            id=self._next_food_id(),
            x=fx,
            y=fy,
            mass=mass,
            color=self.rng.choice(config.FOOD_COLORS),
        )
        self.foods[food.id] = food

    def _spawn_initial_viruses(self) -> None:
        for _ in range(config.VIRUS_COUNT):
            self._spawn_virus()

    def _spawn_virus(self) -> None:
        vx, vy = self._random_spawn(radius=80.0)
        virus = Virus(id=self._next_virus_id(), x=vx, y=vy, mass=config.VIRUS_MASS)
        self.viruses[virus.id] = virus

    def _respawn_eliminated_players(self, now: float) -> None:
        for player in self.players.values():
            if player.blobs:
                continue
            x, y = self._random_spawn(radius=40.0)
            blob = Blob(
                id=self._next_blob_id(),
                player_id=player.id,
                x=x,
                y=y,
                mass=config.PLAYER_START_MASS,
                can_merge_at=now + config.MERGE_DELAY_SECONDS,
            )
            player.blobs[blob.id] = blob
            player.target_x = x
            player.target_y = y

    def _apply_actions(self, now: float) -> None:
        for player in self.players.values():
            if player.split_requested:
                self._split_player(player, now)
                player.split_requested = False
            if player.eject_requested:
                self._eject_player_mass(player, now)
                player.eject_requested = False

    def _split_player(self, player: Player, now: float) -> None:
        if now - player.last_split_at < config.SPLIT_COOLDOWN_SECONDS:
            return

        if len(player.blobs) >= config.MAX_PLAYER_BLOBS:
            return

        blobs_snapshot = list(player.blobs.values())
        created: list[Blob] = []
        for blob in blobs_snapshot:
            if blob.mass < config.PLAYER_MIN_SPLIT_MASS:
                continue
            if len(player.blobs) + len(created) >= config.MAX_PLAYER_BLOBS:
                break

            dx = player.target_x - blob.x
            dy = player.target_y - blob.y
            ux, uy = _unit_vec(dx, dy)

            split_mass = blob.mass / 2.0
            blob.mass = split_mass
            blob.can_merge_at = now + config.MERGE_DELAY_SECONDS

            offset = blob.radius + (split_mass ** 0.5) * config.BLOB_RADIUS_FACTOR
            new_blob = Blob(
                id=self._next_blob_id(),
                player_id=player.id,
                x=_clamp(blob.x + ux * offset, 0.0, config.WORLD_WIDTH),
                y=_clamp(blob.y + uy * offset, 0.0, config.WORLD_HEIGHT),
                mass=split_mass,
                vx=ux * config.SPLIT_BOOST_SPEED,
                vy=uy * config.SPLIT_BOOST_SPEED,
                can_merge_at=now + config.MERGE_DELAY_SECONDS,
            )
            created.append(new_blob)

        for blob in created:
            player.blobs[blob.id] = blob

        if created:
            player.last_split_at = now

    def _eject_player_mass(self, player: Player, now: float) -> None:
        if now - player.last_eject_at < config.EJECT_COOLDOWN_SECONDS:
            return

        spawned_any = False
        for blob in player.blobs.values():
            if blob.mass <= config.PLAYER_MIN_EJECT_MASS:
                continue

            remaining_mass = blob.mass - config.PLAYER_EJECT_MASS
            if remaining_mass < config.MIN_BLOB_MASS:
                continue

            dx = player.target_x - blob.x
            dy = player.target_y - blob.y
            ux, uy = _unit_vec(dx, dy)

            blob.mass = remaining_mass
            eject_x = blob.x + ux * (blob.radius + 12.0)
            eject_y = blob.y + uy * (blob.radius + 12.0)

            ejected = EjectedMass(
                id=self._next_ejected_id(),
                x=_clamp(eject_x, 0.0, config.WORLD_WIDTH),
                y=_clamp(eject_y, 0.0, config.WORLD_HEIGHT),
                mass=config.PLAYER_EJECT_MASS,
                owner_id=player.id,
                vx=ux * config.EJECT_BOOST_SPEED,
                vy=uy * config.EJECT_BOOST_SPEED,
                ttl=config.EJECTED_MASS_LIFETIME,
            )
            self.ejected[ejected.id] = ejected
            spawned_any = True

        if spawned_any:
            player.last_eject_at = now

    def _move_blobs(self, dt: float, now: float) -> None:
        damping = max(0.0, 1.0 - config.BOOST_DAMPING * dt)

        for player in self.players.values():
            for blob in player.blobs.values():
                dx = player.target_x - blob.x
                dy = player.target_y - blob.y
                ux, uy = _unit_vec(dx, dy)
                input_distance = (dx * dx + dy * dy) ** 0.5
                input_excess = max(0.0, input_distance - config.INPUT_DEADZONE_WORLD)
                ramp = max(1.0, config.INPUT_SPEED_RAMP_WORLD)
                ease_power = max(0.25, config.INPUT_SPEED_EASE_EXPONENT)
                eased_distance = (input_excess / ramp) ** ease_power
                # Saturating response: center stays near-still, nearby mouse keeps high speed, far mouse has diminishing gain.
                input_scale = 1.0 - exp(-eased_distance)

                max_speed = max(
                    config.PLAYER_MIN_SPEED,
                    config.PLAYER_BASE_SPEED / (blob.mass ** config.SPEED_EXPONENT),
                )
                speed = max_speed * input_scale

                blob.x += (ux * speed + blob.vx) * dt
                blob.y += (uy * speed + blob.vy) * dt
                blob.vx *= damping
                blob.vy *= damping

            self._apply_same_player_softbody(player, now)

            for blob in player.blobs.values():
                radius = blob.radius
                clamp_r = radius * config.BLOB_BOUNDARY_FACTOR
                blob.x = _clamp(blob.x, clamp_r, config.WORLD_WIDTH - clamp_r)
                blob.y = _clamp(blob.y, clamp_r, config.WORLD_HEIGHT - clamp_r)

    def _apply_same_player_softbody(self, player: Player, now: float) -> None:
        blobs = list(player.blobs.values())
        if len(blobs) <= 1:
            return

        for i in range(len(blobs)):
            a = blobs[i]
            for j in range(i + 1, len(blobs)):
                b = blobs[j]
                dx = b.x - a.x
                dy = b.y - a.y
                dist_sq = dx * dx + dy * dy
                if dist_sq <= 1e-8:
                    theta = self.rng.random() * 6.283185
                    dx, dy = cos(theta), sin(theta)
                    dist_sq = 1.0

                dist = dist_sq ** 0.5
                ux = dx / dist
                uy = dy / dist

                touch = a.radius + b.radius
                ready_to_merge = now >= a.can_merge_at and now >= b.can_merge_at

                min_dist = touch * (
                    config.SOFTBODY_MIN_DIST_MERGED if ready_to_merge else config.SOFTBODY_MIN_DIST_UNMERGED
                )

                # Keep blobs from collapsing into one center; this preserves the squishy contact feel.
                if dist < min_dist:
                    correction = (min_dist - dist) * 0.5
                    a.x -= ux * correction
                    a.y -= uy * correction
                    b.x += ux * correction
                    b.y += uy * correction

    def _move_ejected(self, dt: float) -> None:
        to_remove: list[str] = []
        damping = max(0.0, 1.0 - config.BOOST_DAMPING * dt)

        for ejected in self.ejected.values():
            ejected.x += ejected.vx * dt
            ejected.y += ejected.vy * dt
            ejected.vx *= damping
            ejected.vy *= damping
            ejected.ttl -= dt

            radius = ejected.radius
            ejected.x = _clamp(ejected.x, radius, config.WORLD_WIDTH - radius)
            ejected.y = _clamp(ejected.y, radius, config.WORLD_HEIGHT - radius)

            if ejected.ttl <= 0.0:
                to_remove.append(ejected.id)

        for eid in to_remove:
            self.ejected.pop(eid, None)

    def _rebuild_spatial_indexes(self) -> None:
        self.food_hash.clear()
        self.blob_hash.clear()
        self.ejected_hash.clear()

        for food in self.foods.values():
            self.food_hash.insert(food.x, food.y, food)

        for player in self.players.values():
            for blob in player.blobs.values():
                self.blob_hash.insert(blob.x, blob.y, blob)

        for ejected in self.ejected.values():
            self.ejected_hash.insert(ejected.x, ejected.y, ejected)

    def _resolve_blob_food_collisions(self) -> None:
        eaten_ids: set[str] = set()

        for player in self.players.values():
            for blob in player.blobs.values():
                nearby = self.food_hash.query_rect(
                    blob.x - blob.radius,
                    blob.y - blob.radius,
                    blob.x + blob.radius,
                    blob.y + blob.radius,
                )
                for food in nearby:
                    if food.id in eaten_ids:
                        continue
                    eat_dist = (blob.radius + food.radius) * config.FOOD_EAT_RANGE_FACTOR
                    if _distance_sq(blob.x, blob.y, food.x, food.y) <= eat_dist * eat_dist:
                        blob.mass += food.mass
                        eaten_ids.add(food.id)

        for food_id in eaten_ids:
            self.foods.pop(food_id, None)

    def _resolve_blob_ejected_collisions(self) -> None:
        consumed: set[str] = set()

        for player in self.players.values():
            for blob in player.blobs.values():
                nearby = self.ejected_hash.query_rect(
                    blob.x - blob.radius,
                    blob.y - blob.radius,
                    blob.x + blob.radius,
                    blob.y + blob.radius,
                )
                for ejected in nearby:
                    if ejected.id in consumed:
                        continue

                    if ejected.owner_id == blob.player_id and ejected.ttl > config.EJECTED_MASS_LIFETIME - 0.35:
                        continue

                    eat_dist = (blob.radius + ejected.radius) * config.EJECTED_EAT_RANGE_FACTOR
                    if _distance_sq(blob.x, blob.y, ejected.x, ejected.y) <= eat_dist * eat_dist:
                        blob.mass += ejected.mass
                        consumed.add(ejected.id)

        for eid in consumed:
            self.ejected.pop(eid, None)

    def _resolve_blob_blob_collisions(self, now: float) -> None:
        eaten_blob_ids: set[str] = set()
        checked_pairs: set[tuple[str, str]] = set()

        all_blobs: list[Blob] = []
        for player in self.players.values():
            all_blobs.extend(player.blobs.values())

        for blob in all_blobs:
            if blob.id in eaten_blob_ids:
                continue

            nearby = self.blob_hash.query_rect(
                blob.x - blob.radius * 2.0,
                blob.y - blob.radius * 2.0,
                blob.x + blob.radius * 2.0,
                blob.y + blob.radius * 2.0,
            )

            for other in nearby:
                if other.id == blob.id or other.id in eaten_blob_ids:
                    continue

                pair = (blob.id, other.id) if blob.id < other.id else (other.id, blob.id)
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)

                # Both blobs still exist at resolution time.
                owner_a = self.players.get(blob.player_id)
                owner_b = self.players.get(other.player_id)
                if owner_a is None or owner_b is None:
                    continue
                if blob.id not in owner_a.blobs or other.id not in owner_b.blobs:
                    continue

                if blob.mass >= other.mass:
                    bigger, smaller = blob, other
                else:
                    bigger, smaller = other, blob

                dist_sq = _distance_sq(bigger.x, bigger.y, smaller.x, smaller.y)

                if bigger.player_id == smaller.player_id:
                    if now < bigger.can_merge_at or now < smaller.can_merge_at:
                        continue
                    if dist_sq <= (max(bigger.radius, smaller.radius) * 0.35) ** 2:
                        bigger.mass += smaller.mass
                        eaten_blob_ids.add(smaller.id)
                    continue

                if bigger.mass < smaller.mass * config.BLOB_EAT_RATIO:
                    continue

                eat_distance = bigger.radius - (smaller.radius * config.BLOB_EAT_OVERLAP)
                if eat_distance <= 0.0:
                    continue
                if dist_sq <= eat_distance * eat_distance:
                    bigger.mass += smaller.mass
                    eaten_blob_ids.add(smaller.id)

        for player in self.players.values():
            for blob_id in list(player.blobs.keys()):
                if blob_id in eaten_blob_ids:
                    del player.blobs[blob_id]

    def _resolve_virus_blob_collisions(self, now: float) -> None:
        consumed_viruses: set[str] = set()

        for player in self.players.values():
            for blob_id in list(player.blobs.keys()):
                blob = player.blobs.get(blob_id)
                if blob is None:
                    continue

                for virus in self.viruses.values():
                    if virus.id in consumed_viruses:
                        continue
                    if blob.mass < virus.mass * 1.15:
                        continue
                    trigger_distance = blob.radius - (virus.radius * 0.18)
                    if trigger_distance <= 0.0:
                        continue

                    if _distance_sq(blob.x, blob.y, virus.x, virus.y) <= trigger_distance * trigger_distance:
                        blob.mass += config.VIRUS_BONUS_MASS
                        self._explode_blob_into_player(blob, player, now)
                        consumed_viruses.add(virus.id)
                        break

        for virus_id in consumed_viruses:
            self.viruses.pop(virus_id, None)

        for _ in range(len(consumed_viruses)):
            self._spawn_virus()

    def _explode_blob_into_player(self, blob: Blob, player: Player, now: float) -> None:
        if blob.id not in player.blobs:
            return

        available_slots = config.MAX_PLAYER_BLOBS - len(player.blobs) + 1
        if available_slots <= 1:
            return

        split_parts = int(blob.mass / 30.0)
        split_parts = max(config.VIRUS_SPLIT_MIN_PARTS, split_parts)
        split_parts = min(config.VIRUS_SPLIT_MAX_PARTS, split_parts, available_slots)
        if split_parts <= 1:
            return

        total_mass = blob.mass
        base_x, base_y = blob.x, blob.y

        del player.blobs[blob.id]

        part_mass = total_mass / split_parts
        for idx in range(split_parts):
            angle = (idx / split_parts) * 6.283185 + self.rng.uniform(-0.15, 0.15)
            ux, uy = cos(angle), sin(angle)
            spawned = Blob(
                id=self._next_blob_id(),
                player_id=player.id,
                x=_clamp(base_x + ux * 14.0, 0.0, config.WORLD_WIDTH),
                y=_clamp(base_y + uy * 14.0, 0.0, config.WORLD_HEIGHT),
                mass=part_mass,
                vx=ux * config.SPLIT_BOOST_SPEED * self.rng.uniform(0.62, 0.92),
                vy=uy * config.SPLIT_BOOST_SPEED * self.rng.uniform(0.62, 0.92),
                can_merge_at=now + config.MERGE_DELAY_SECONDS,
            )
            player.blobs[spawned.id] = spawned

    def _spawn_food_to_target(self) -> None:
        while len(self.foods) < config.FOOD_TARGET_COUNT:
            self._spawn_food()

    def _visible_entities(self, cx: float, cy: float, view_w: float, view_h: float) -> tuple[list[Blob], list[Food], list[EjectedMass], list[Virus]]:
        min_x = _clamp(cx - view_w / 2.0, 0.0, config.WORLD_WIDTH)
        max_x = _clamp(cx + view_w / 2.0, 0.0, config.WORLD_WIDTH)
        min_y = _clamp(cy - view_h / 2.0, 0.0, config.WORLD_HEIGHT)
        max_y = _clamp(cy + view_h / 2.0, 0.0, config.WORLD_HEIGHT)

        blobs = self.blob_hash.query_rect(min_x, min_y, max_x, max_y)
        foods = self.food_hash.query_rect(min_x, min_y, max_x, max_y)
        ejected = self.ejected_hash.query_rect(min_x, min_y, max_x, max_y)
        viruses = [
            virus
            for virus in self.viruses.values()
            if min_x <= virus.x <= max_x and min_y <= virus.y <= max_y
        ]
        return (blobs, foods, ejected, viruses)

    def _leaderboard(self) -> list[dict]:
        return sorted(
            (
                {"name": p.name, "score": round(p.total_mass)}
                for p in self.players.values()
                if p.total_mass > 0
            ),
            key=lambda row: row["score"],
            reverse=True,
        )[:10]

    def _snapshot_payload(
        self,
        *,
        you: str | None,
        player_name: str,
        player_score: float,
        camera_x: float,
        camera_y: float,
        camera_zoom: float,
        blobs: list[Blob],
        foods: list[Food],
        ejected: list[EjectedMass],
        viruses: list[Virus],
    ) -> dict:
        player_lookup = {p.id: p for p in self.players.values()}
        return {
            "type": "state",
            "you": you,
            "world": {"w": config.WORLD_WIDTH, "h": config.WORLD_HEIGHT},
            "camera": {"x": round(camera_x, 2), "y": round(camera_y, 2), "zoom": round(camera_zoom, 3)},
            "player": {"name": player_name, "score": round(player_score)},
            "leaderboard": self._leaderboard(),
            "blobs": [
                {
                    "id": blob.id,
                    "playerId": blob.player_id,
                    "name": player_lookup[blob.player_id].name if blob.player_id in player_lookup else "",
                    "color": player_lookup[blob.player_id].color if blob.player_id in player_lookup else "#ddd",
                    "x": round(blob.x, 2),
                    "y": round(blob.y, 2),
                    "mass": round(blob.mass, 2),
                }
                for blob in blobs
            ],
            "foods": [
                {
                    "id": food.id,
                    "x": round(food.x, 2),
                    "y": round(food.y, 2),
                    "mass": food.mass,
                    "color": food.color,
                }
                for food in foods
            ],
            "ejected": [
                {
                    "id": e.id,
                    "x": round(e.x, 2),
                    "y": round(e.y, 2),
                    "mass": e.mass,
                }
                for e in ejected
            ],
            "viruses": [
                {
                    "id": v.id,
                    "x": round(v.x, 2),
                    "y": round(v.y, 2),
                    "mass": v.mass,
                }
                for v in viruses
            ],
        }

    def snapshot_for(self, player_id: str) -> dict | None:
        player = self.players.get(player_id)
        if player is None:
            return None

        cx, cy = player.center()
        total_mass = max(player.total_mass, config.PLAYER_START_MASS)
        split_count = max(0, len(player.blobs) - 1)
        base_zoom = 1.52 - (total_mass ** 0.4) / 22.0
        split_penalty = split_count * 0.055
        zoom = _clamp(base_zoom - split_penalty, 0.24, 1.35)
        view_w = config.VIEW_WIDTH / zoom + config.VIEW_PADDING
        view_h = config.VIEW_HEIGHT / zoom + config.VIEW_PADDING

        blobs, foods, ejected, viruses = self._visible_entities(cx, cy, view_w, view_h)
        return self._snapshot_payload(
            you=player.id,
            player_name=player.name,
            player_score=player.total_mass,
            camera_x=cx,
            camera_y=cy,
            camera_zoom=zoom,
            blobs=blobs,
            foods=foods,
            ejected=ejected,
            viruses=viruses,
        )

    def snapshot_overview(self) -> dict:
        zoom = min(config.VIEW_WIDTH / config.WORLD_WIDTH, config.VIEW_HEIGHT / config.WORLD_HEIGHT) * 0.92
        return self._snapshot_payload(
            you=None,
            player_name="Spectator",
            player_score=0.0,
            camera_x=config.WORLD_WIDTH * 0.5,
            camera_y=config.WORLD_HEIGHT * 0.5,
            camera_zoom=_clamp(zoom, 0.05, 1.35),
            blobs=[blob for player in self.players.values() for blob in player.blobs.values()],
            foods=list(self.foods.values()),
            ejected=list(self.ejected.values()),
            viruses=list(self.viruses.values()),
        )

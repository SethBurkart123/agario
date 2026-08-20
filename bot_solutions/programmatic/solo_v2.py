"""Stateful solo policy built on the proven solo_smart steering primitives."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from agario.bots.types import BlobView, BotAction, BotContext, BotInitContext, VirusView
from agario_core import mechanics

from . import (
    EAT_MASS_RATIO,
    MAX_BLOBS,
    MIN_BLOB_MASS,
    VIRUS_MASS,
    SoloSmartBrain,
    _best_prey,
    _can_eat,
    _clamp,
    _iter_enemy_blobs,
    _largest_blob,
    _smallest_blob,
    _split_eat_reach,
    _unit,
)

EJECT_LOSS = mechanics()["eject_loss_mass"]
VIRUS_FEEDS = 7


@dataclass(frozen=True, slots=True)
class SoloV2Traits:
    name: str
    aggression: float
    caution: float
    greed: float
    edge_bias: float
    virus_skill: float
    juke_skill: float
    eject_willingness: float


TRAIT_PROFILES = (
    SoloV2Traits("guardian", 0.90, 1.18, 1.02, 1.00, 0.62, 1.12, 0.46),
    SoloV2Traits("hunter", 1.26, 0.88, 0.90, 0.24, 0.72, 0.90, 0.28),
    SoloV2Traits("trickster", 1.10, 1.04, 0.96, 0.52, 1.35, 1.24, 1.20),
    SoloV2Traits("grazer", 0.90, 1.08, 1.28, 0.82, 0.48, 0.86, 0.34),
)


class SoloSmartV2Brain(SoloSmartBrain):
    def __init__(self, init_ctx: BotInitContext) -> None:
        super().__init__(init_ctx)
        self.profile = TRAIT_PROFILES[(init_ctx.bot_index - 1) % len(TRAIT_PROFILES)]
        self.aggression = _clamp(
            self.aggression * self.profile.aggression * self.rng.uniform(0.94, 1.06),
            0.76,
            1.35,
        )
        self.caution = _clamp(
            self.caution * self.profile.caution * self.rng.uniform(0.95, 1.05),
            0.84,
            1.30,
        )
        self.greed = _clamp(
            self.greed * self.profile.greed * self.rng.uniform(1.02, 1.12), 0.86, 1.35
        )
        self.dodge_skill = _clamp(
            self.dodge_skill * self.profile.juke_skill, 0.48, 0.98
        )
        self.view_scale = max(0.92, self.view_scale)

    @staticmethod
    def _stat(ctx: BotContext, name: str) -> None:
        stats = ctx.memory.setdefault("v2_stats", {})
        stats[name] = int(stats.get(name, 0)) + 1

    @staticmethod
    def _closest_threat(ctx: BotContext, me: BlobView):
        best = None
        best_gap = float("inf")
        for player, enemy in _iter_enemy_blobs(ctx):
            if not _can_eat(enemy.mass, me.mass):
                continue
            dist = hypot(enemy.x - me.x, enemy.y - me.y)
            gap = dist - max(0.0, enemy.radius - me.radius / 3.0)
            if gap < best_gap:
                best = (player, enemy, dist, gap)
                best_gap = gap
        return best

    def _pursuit_age(
        self, ctx: BotContext, threat: BlobView | None, me: BlobView
    ) -> float:
        if threat is None:
            ctx.memory.pop("pursuit_id", None)
            return 0.0
        toward_x, toward_y = _unit(me.x - threat.x, me.y - threat.y)
        moving_x, moving_y = _unit(threat.vx, threat.vy)
        direct = toward_x * moving_x + toward_y * moving_y > 0.91
        if not direct:
            ctx.memory.pop("pursuit_id", None)
            return 0.0
        if ctx.memory.get("pursuit_id") != threat.id:
            ctx.memory["pursuit_id"] = threat.id
            ctx.memory["pursuit_since"] = ctx.now
        return ctx.now - float(ctx.memory.get("pursuit_since", ctx.now))

    def _shield_target(
        self,
        ctx: BotContext,
        me: BlobView,
        threat: BlobView,
    ) -> tuple[float, float] | None:
        if me.mass > VIRUS_MASS * EAT_MASS_RATIO:
            return None
        best = None
        best_score = float("inf")
        for virus in ctx.viruses:
            dist = hypot(virus.x - me.x, virus.y - me.y)
            if dist > 760.0:
                continue
            tx, ty = _unit(virus.x - threat.x, virus.y - threat.y)
            score = dist + abs((virus.x - me.x) * ty - (virus.y - me.y) * tx) * 0.45
            if score < best_score:
                stand_off = virus.radius + me.radius + 30.0
                best = (virus.x + tx * stand_off, virus.y + ty * stand_off)
                best_score = score
        return best

    def _virus_weapon_candidate(
        self,
        ctx: BotContext,
        me: BlobView,
    ) -> tuple[VirusView, BlobView] | None:
        if ctx.me.total_mass < EJECT_LOSS * (VIRUS_FEEDS + 2) + MIN_BLOB_MASS:
            return None
        best = None
        best_score = float("inf")
        for virus in ctx.viruses:
            virus_dist = hypot(virus.x - me.x, virus.y - me.y)
            if not (me.radius + virus.radius + 45.0 < virus_dist < 1_200.0):
                continue
            for _, enemy in _iter_enemy_blobs(ctx):
                if enemy.mass < max(180.0, me.mass * 1.16):
                    continue
                enemy_dist = hypot(enemy.x - virus.x, enemy.y - virus.y)
                if enemy_dist > 820.0:
                    continue
                fire_x, fire_y = _unit(virus.x - enemy.x, virus.y - enemy.y)
                stand_off = me.radius + virus.radius + 320.0
                stage_x = virus.x + fire_x * stand_off
                stage_y = virus.y + fire_y * stand_off
                if hypot(stage_x - me.x, stage_y - me.y) > 720.0:
                    continue
                score = virus_dist + enemy_dist * 0.72 - enemy.mass * 0.08
                if score < best_score:
                    best = (virus, enemy)
                    best_score = score
        return best

    def _continue_virus_feed(self, ctx: BotContext, me: BlobView) -> BotAction | None:
        plan = ctx.memory.get("virus_plan")
        if not plan:
            return None
        virus = next((v for v in ctx.viruses if v.id == plan["virus_id"]), None)
        enemy = next(
            (b for _, b in _iter_enemy_blobs(ctx) if b.id == plan["enemy_id"]), None
        )
        if virus is None or enemy is None or ctx.now > float(plan["expires_at"]):
            ctx.memory.pop("virus_plan", None)
            return None
        if (
            ctx.me.total_mass < EJECT_LOSS + MIN_BLOB_MASS
            or hypot(virus.x - me.x, virus.y - me.y) > 1_700.0
        ):
            ctx.memory.pop("virus_plan", None)
            return None

        fire_x, fire_y = _unit(virus.x - enemy.x, virus.y - enemy.y)
        stand_off = me.radius + virus.radius + 320.0
        stage_x = virus.x + fire_x * stand_off
        stage_y = virus.y + fire_y * stand_off
        stage_dist = hypot(stage_x - me.x, stage_y - me.y)
        aim_x, aim_y = _unit(virus.x - me.x, virus.y - me.y)
        shot_x, shot_y = _unit(enemy.x - virus.x, enemy.y - virus.y)
        aligned = aim_x * shot_x + aim_y * shot_y > 0.94
        if plan["feeds"] == 0 and (stage_dist > 140.0 or not aligned):
            return BotAction(
                _clamp(stage_x, 0.0, ctx.world_width),
                _clamp(stage_y, 0.0, ctx.world_height),
            )

        eject = ctx.now >= float(plan.get("next_feed_at", 0.0))
        if eject:
            plan["feeds"] += 1
            plan["next_feed_at"] = ctx.now + 0.18
            self._stat(ctx, "virus_feeds")
        if plan["feeds"] >= VIRUS_FEEDS:
            ctx.memory.pop("virus_plan", None)
            ctx.memory["next_virus_plan_at"] = ctx.now + self.rng.uniform(4.0, 7.0)
            self._stat(ctx, "virus_shots")
        return BotAction(virus.x, virus.y, eject=eject)

    def _safe_virus_farm(self, ctx: BotContext, me: BlobView) -> BotAction | None:
        if len(ctx.me.blobs) < MAX_BLOBS or not _can_eat(me.mass, VIRUS_MASS):
            return None
        virus = min(
            ctx.viruses, key=lambda v: hypot(v.x - me.x, v.y - me.y), default=None
        )
        if virus is None or hypot(virus.x - me.x, virus.y - me.y) > 850.0:
            return None
        self._stat(ctx, "virus_farms")
        return BotAction(virus.x, virus.y)

    def _juke(self, ctx: BotContext, me: BlobView, threat: BlobView) -> BotAction:
        away_x, away_y = _unit(me.x - threat.x, me.y - threat.y)
        side = float(ctx.memory.get("juke_side", self.strafe_dir))
        ctx.memory["juke_side"] = -side
        distance = 760.0
        self._stat(ctx, "jukes")
        return BotAction(
            _clamp(
                me.x + (away_x - away_y * side * 0.92) * distance, 0.0, ctx.world_width
            ),
            _clamp(
                me.y + (away_y + away_x * side * 0.92) * distance, 0.0, ctx.world_height
            ),
        )

    def _refine_split(
        self, ctx: BotContext, me: BlobView, action: BotAction
    ) -> BotAction:
        if not action.split or ctx.memory.get("mode") != "attack":
            return action
        _, prey, _, dist = _best_prey(ctx, me, me.x, me.y)
        if prey is None:
            self._stat(ctx, "unsafe_splits_blocked")
            return BotAction(action.target_x, action.target_y, eject=action.eject)
        predictable = hypot(prey.vx, prey.vy) < 38.0
        near_wall = (
            min(prey.x, prey.y, ctx.world_width - prey.x, ctx.world_height - prey.y)
            < 330.0
        )
        close = dist < me.radius * 2.0 + prey.radius * 1.2
        safe = me.mass * 0.5 > prey.mass * EAT_MASS_RATIO and dist < _split_eat_reach(
            me, prey
        )
        if not safe or (
            not close and not near_wall and not predictable and self.aggression < 1.28
        ):
            self._stat(ctx, "unsafe_splits_blocked")
            return BotAction(action.target_x, action.target_y, eject=action.eject)
        self._stat(ctx, "split_attacks")
        return action

    def _lead_attack(
        self, ctx: BotContext, me: BlobView, action: BotAction
    ) -> BotAction:
        if ctx.memory.get("mode") != "attack":
            return action
        _, prey, _, dist = _best_prey(ctx, me, me.x, me.y)
        if prey is None:
            return action
        lead = _clamp(dist / 1_200.0, 0.12, 0.48)
        target_x = action.target_x + prey.vx * lead
        target_y = action.target_y + prey.vy * lead
        wall = 300.0
        if prey.x < wall:
            target_x -= wall - prey.x
        elif ctx.world_width - prey.x < wall:
            target_x += wall - (ctx.world_width - prey.x)
        if prey.y < wall:
            target_y -= wall - prey.y
        elif ctx.world_height - prey.y < wall:
            target_y += wall - (ctx.world_height - prey.y)
        return BotAction(
            _clamp(target_x, 0.0, ctx.world_width),
            _clamp(target_y, 0.0, ctx.world_height),
            split=action.split,
            eject=action.eject,
        )

    def decide(self, ctx: BotContext) -> BotAction:
        me = _largest_blob(ctx.me)
        small = _smallest_blob(ctx.me)
        if me is None or small is None:
            return super().decide(ctx)

        active_feed = self._continue_virus_feed(ctx, me)
        if active_feed is not None:
            return active_feed

        farm_virus = self._safe_virus_farm(ctx, me)
        if farm_virus is not None:
            return farm_virus

        if ctx.now < float(ctx.memory.get("shield_until", 0.0)):
            shield_target = ctx.memory.get("shield_target")
            if shield_target:
                return BotAction(float(shield_target[0]), float(shield_target[1]))

        threat_row = self._closest_threat(ctx, small)
        threat = threat_row[1] if threat_row else None
        pursuit_age = self._pursuit_age(ctx, threat, small)
        if (
            threat is not None
            and pursuit_age > 1.1 / self.profile.juke_skill
            and ctx.now >= float(ctx.memory.get("next_juke_at", 0.0))
        ):
            ctx.memory["next_juke_at"] = ctx.now + self.rng.uniform(1.2, 2.0)
            return self._juke(ctx, small, threat)

        if (
            threat_row is not None
            and threat_row[3] < 350.0
            and ctx.me.total_mass < 180.0
            and len(ctx.me.blobs) <= 2
            and ctx.now >= float(ctx.memory.get("next_shield_at", 0.0))
        ):
            shield = self._shield_target(ctx, small, threat)
            if shield is not None and self.rng.random() < min(
                0.92, self.caution * 0.62
            ):
                ctx.memory["shield_target"] = shield
                ctx.memory["shield_until"] = ctx.now + self.rng.uniform(0.55, 0.85)
                ctx.memory["next_shield_at"] = ctx.now + self.rng.uniform(1.8, 2.8)
                self._stat(ctx, "virus_shields")
                return BotAction(*shield)

        if (
            threat_row is not None
            and 125.0 < threat_row[3] < 620.0
            and ctx.me.total_mass > max(130.0, EJECT_LOSS * 8.0)
            and ctx.now >= float(ctx.memory.get("next_trail_at", 0.0))
            and self.rng.random() < 0.08 * self.profile.eject_willingness
        ):
            ctx.memory["next_trail_at"] = ctx.now + self.rng.uniform(1.8, 3.4)
            self._stat(ctx, "bait_ejects")
            return BotAction(threat.x, threat.y, eject=True)

        if ctx.now >= float(ctx.memory.get("next_virus_plan_at", 0.0)):
            candidate = self._virus_weapon_candidate(ctx, me)
            if candidate and self.rng.random() < 0.07 * self.profile.virus_skill:
                virus, enemy = candidate
                ctx.memory["virus_plan"] = {
                    "virus_id": virus.id,
                    "enemy_id": enemy.id,
                    "feeds": 0,
                    "next_feed_at": ctx.now,
                    "expires_at": ctx.now + 18.0,
                }
                self._stat(ctx, "virus_plans")
                return self._continue_virus_feed(ctx, me) or BotAction(virus.x, virus.y)

        action = super().decide(ctx)
        action = self._refine_split(ctx, me, action)
        action = self._lead_attack(ctx, me, action)

        if (
            len(ctx.me.blobs) >= 4
            and max(b.can_merge_at for b in ctx.me.blobs) > ctx.now
            and action.split
        ):
            self._stat(ctx, "recovery_splits_blocked")
            action = BotAction(action.target_x, action.target_y, eject=action.eject)

        if ctx.me.total_mass < 150.0 and ctx.memory.get("mode") == "farm":
            margin = min(me.x, me.y, ctx.world_width - me.x, ctx.world_height - me.y)
            if margin < 1_200.0:
                edge_x, edge_y = action.target_x, action.target_y
                nearest = min(
                    (
                        (me.x, 180.0, edge_y),
                        (ctx.world_width - me.x, ctx.world_width - 180.0, edge_y),
                        (me.y, edge_x, 180.0),
                        (ctx.world_height - me.y, edge_x, ctx.world_height - 180.0),
                    ),
                    key=lambda row: row[0],
                )
                blend = 0.08 + self.profile.edge_bias * 0.12
                action = BotAction(
                    action.target_x + (nearest[1] - action.target_x) * blend,
                    action.target_y + (nearest[2] - action.target_y) * blend,
                    split=action.split,
                    eject=action.eject,
                )

        return action

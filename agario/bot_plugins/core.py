"""Built-in advanced bot strategy pack."""

from __future__ import annotations

from math import exp, hypot

from .. import config
from ..bots.registry import BotRegistry
from ..bots.types import BlobView, BotAction, BotContext, BotInitContext, PlayerView


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return min(max(value, min_value), max_value)


def _unit(dx: float, dy: float) -> tuple[float, float]:
    dist = hypot(dx, dy)
    if dist <= 1e-9:
        return (0.0, 0.0)
    return (dx / dist, dy / dist)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _jitter_vector(vx: float, vy: float, rng, amount: float) -> tuple[float, float]:
    if amount <= 1e-6:
        return (vx, vy)
    mag = hypot(vx, vy)
    if mag <= 1e-9:
        return (rng.uniform(-amount, amount), rng.uniform(-amount, amount))
    nx = -vy / mag
    ny = vx / mag
    side = rng.uniform(-amount, amount)
    scale = 1.0 + rng.uniform(-amount * 0.32, amount * 0.22)
    return (vx * scale + nx * mag * side, vy * scale + ny * mag * side)


def _player_center(player: PlayerView) -> tuple[float, float]:
    if not player.blobs:
        return (0.0, 0.0)
    total = max(1.0, player.total_mass)
    x = sum(b.x * b.mass for b in player.blobs) / total
    y = sum(b.y * b.mass for b in player.blobs) / total
    return (x, y)


def _largest_blob(player: PlayerView):
    if not player.blobs:
        return None
    return max(player.blobs, key=lambda b: b.mass)


def _smallest_blob(player: PlayerView):
    if not player.blobs:
        return None
    return min(player.blobs, key=lambda b: b.mass)


def _can_eat(attacker_mass: float, defender_mass: float) -> bool:
    return attacker_mass > defender_mass * config.BLOB_EAT_RATIO


def _iter_enemy_blobs(ctx: BotContext):
    for player in ctx.players:
        if player.id == ctx.me.id or player.total_mass <= 0:
            continue
        for blob in player.blobs:
            yield player, blob


def _closest_food_target(ctx: BotContext, x: float, y: float) -> tuple[float, float]:
    best_dist = float("inf")
    target: tuple[float, float] | None = None

    for item in ctx.ejected:
        d = hypot(item.x - x, item.y - y)
        if d < best_dist:
            best_dist = d
            target = (item.x, item.y)

    for food in ctx.foods:
        d = hypot(food.x - x, food.y - y)
        if d < best_dist:
            best_dist = d
            target = (food.x, food.y)

    if target is not None:
        return target
    return (x, y)


def _threat_field(ctx: BotContext) -> tuple[float, float, float, float]:
    vec_x = 0.0
    vec_y = 0.0
    pressure = 0.0
    imminence = 0.0

    for _, enemy in _iter_enemy_blobs(ctx):
        for me_blob in ctx.me.blobs:
            if not _can_eat(enemy.mass, me_blob.mass):
                continue

            dx = me_blob.x - enemy.x
            dy = me_blob.y - enemy.y
            dist = max(1.0, hypot(dx, dy))
            safe_radius = enemy.radius + me_blob.radius * 2.6 + 95.0
            if dist >= safe_radius:
                continue

            ux, uy = _unit(dx, dy)
            mass_ratio = enemy.mass / max(1.0, me_blob.mass)
            local_pressure = (safe_radius - dist) / safe_radius
            local_pressure *= 0.44 + min(2.8, mass_ratio * 0.56)
            vec_x += ux * local_pressure
            vec_y += uy * local_pressure
            pressure += local_pressure

            eat_reach = max(0.0, enemy.radius - me_blob.radius * config.BLOB_EAT_OVERLAP)
            danger_window = max(20.0, enemy.radius * 0.82)
            gap = dist - eat_reach
            if gap < danger_window:
                imminence = max(imminence, 1.0 - _clamp(gap / danger_window, 0.0, 1.0))

    return (vec_x, vec_y, pressure, imminence)


def _wall_field(ctx: BotContext, x: float, y: float, radius: float) -> tuple[float, float, float]:
    margin = max(170.0, radius * 2.6)
    vx = 0.0
    vy = 0.0
    strength = 0.0

    left = margin - x
    if left > 0:
        p = left / margin
        vx += p
        strength += p

    right = margin - (ctx.world_width - x)
    if right > 0:
        p = right / margin
        vx -= p
        strength += p

    top = margin - y
    if top > 0:
        p = top / margin
        vy += p
        strength += p

    bottom = margin - (ctx.world_height - y)
    if bottom > 0:
        p = bottom / margin
        vy -= p
        strength += p

    return (vx, vy, strength)


def _virus_field(ctx: BotContext, me_blob: BlobView) -> tuple[float, float, float]:
    if me_blob.mass <= config.VIRUS_MASS * 1.08:
        return (0.0, 0.0, 0.0)

    vx = 0.0
    vy = 0.0
    strength = 0.0
    for virus in ctx.viruses:
        dx = me_blob.x - virus.x
        dy = me_blob.y - virus.y
        dist = max(1.0, hypot(dx, dy))
        avoid = me_blob.radius + virus.radius * 1.42 + 54.0
        if dist >= avoid:
            continue
        ux, uy = _unit(dx, dy)
        p = (avoid - dist) / avoid
        vx += ux * p
        vy += uy * p
        strength += p

    return (vx, vy, strength)


def _food_field(ctx: BotContext, x: float, y: float) -> tuple[float, float, tuple[float, float] | None]:
    vx = 0.0
    vy = 0.0
    nearest: tuple[float, tuple[float, float]] | None = None

    for item in ctx.ejected:
        dx = item.x - x
        dy = item.y - y
        dist = max(1.0, hypot(dx, dy))
        if dist > 1700.0:
            continue
        ux, uy = _unit(dx, dy)
        weight = (item.mass * 4.0) / (dist + 32.0)
        vx += ux * weight
        vy += uy * weight
        if nearest is None or dist < nearest[0]:
            nearest = (dist, (item.x, item.y))

    for food in ctx.foods:
        dx = food.x - x
        dy = food.y - y
        dist = max(1.0, hypot(dx, dy))
        if dist > 1400.0:
            continue
        ux, uy = _unit(dx, dy)
        weight = (food.mass * 1.1) / (dist + 18.0)
        vx += ux * weight
        vy += uy * weight
        if nearest is None or dist < nearest[0]:
            nearest = (dist, (food.x, food.y))

    nearest_point = nearest[1] if nearest is not None else None
    return (vx, vy, nearest_point)


def _local_food_density(ctx: BotContext, x: float, y: float, radius: float = 540.0) -> float:
    r2 = radius * radius
    food_hits = 0
    for food in ctx.foods:
        dx = food.x - x
        dy = food.y - y
        if dx * dx + dy * dy <= r2:
            food_hits += 1

    ejected_hits = 0
    for item in ctx.ejected:
        dx = item.x - x
        dy = item.y - y
        if dx * dx + dy * dy <= r2:
            ejected_hits += 1

    return min(1.25, food_hits / 24.0 + ejected_hits / 10.0)


def _best_food_hotspot(ctx: BotContext, x: float, y: float, rng) -> tuple[float, float] | None:
    if not ctx.foods and not ctx.ejected:
        return None

    cell_size = 220.0
    buckets: dict[tuple[int, int], float] = {}

    for food in ctx.foods:
        gx = int(food.x // cell_size)
        gy = int(food.y // cell_size)
        buckets[(gx, gy)] = buckets.get((gx, gy), 0.0) + food.mass * 1.0

    for item in ctx.ejected:
        gx = int(item.x // cell_size)
        gy = int(item.y // cell_size)
        buckets[(gx, gy)] = buckets.get((gx, gy), 0.0) + item.mass * 3.2

    best_score = -1e9
    best: tuple[float, float] | None = None
    for (gx, gy), mass_score in buckets.items():
        cx = (gx + 0.5) * cell_size
        cy = (gy + 0.5) * cell_size
        dist = hypot(cx - x, cy - y)
        score = mass_score - dist * 0.018 + rng.uniform(-0.08, 0.08)
        if score > best_score:
            best_score = score
            best = (
                cx + rng.uniform(-cell_size * 0.16, cell_size * 0.16),
                cy + rng.uniform(-cell_size * 0.16, cell_size * 0.16),
            )

    return best


def _sample_food_route(ctx: BotContext, x: float, y: float, rng, greed: float) -> tuple[float, float] | None:
    if not ctx.foods and not ctx.ejected:
        return None

    best_point: tuple[float, float] | None = None
    best_score = -1e9
    preferred_dist = 240.0 + greed * 320.0

    if ctx.foods:
        samples = min(54, len(ctx.foods))
        for _ in range(samples):
            food = ctx.foods[rng.randrange(len(ctx.foods))]
            dist = max(1.0, hypot(food.x - x, food.y - y))
            value = (food.mass * 2.6) / (dist + 52.0)
            fit = max(0.0, 1.0 - abs(dist - preferred_dist) / max(160.0, preferred_dist * 1.18))
            risk = _target_risk(ctx, food.x, food.y, max(18.0, greed * 22.0))
            score = value * (0.55 + greed * 0.24) + fit * 0.6 - risk * 0.4 + rng.random() * 0.1
            if score > best_score:
                best_score = score
                best_point = (food.x, food.y)

    for item in ctx.ejected:
        dist = max(1.0, hypot(item.x - x, item.y - y))
        value = (item.mass * 7.2) / (dist + 34.0)
        fit = max(0.0, 1.0 - abs(dist - preferred_dist * 0.85) / max(130.0, preferred_dist))
        risk = _target_risk(ctx, item.x, item.y, max(18.0, greed * 22.0))
        score = value + fit * 0.85 - risk * 0.45 + rng.random() * 0.08
        if score > best_score:
            best_score = score
            best_point = (item.x, item.y)

    return best_point


def _clip_context_for_view(ctx: BotContext, me_x: float, me_y: float, view_range: float) -> BotContext:
    range_sq = view_range * view_range

    def in_range(px: float, py: float, extra: float = 0.0) -> bool:
        dx = px - me_x
        dy = py - me_y
        return (dx * dx + dy * dy) <= (view_range + extra) * (view_range + extra)

    foods = tuple(food for food in ctx.foods if in_range(food.x, food.y))
    ejected = tuple(item for item in ctx.ejected if in_range(item.x, item.y))
    viruses = tuple(v for v in ctx.viruses if in_range(v.x, v.y, extra=v.radius))

    players: list[PlayerView] = [ctx.me]
    for player in ctx.players:
        if player.id == ctx.me.id:
            continue
        blobs = tuple(blob for blob in player.blobs if in_range(blob.x, blob.y, extra=blob.radius))
        if not blobs:
            continue
        players.append(
            PlayerView(
                id=player.id,
                name=player.name,
                color=player.color,
                is_bot=player.is_bot,
                plugin_name=player.plugin_name,
                team_id=player.team_id,
                total_mass=sum(blob.mass for blob in blobs),
                blobs=blobs,
            )
        )

    return BotContext(
        now=ctx.now,
        dt=ctx.dt,
        world_width=ctx.world_width,
        world_height=ctx.world_height,
        me=ctx.me,
        players=tuple(players),
        foods=foods,
        ejected=ejected,
        viruses=viruses,
        team_state=ctx.team_state,
        memory=ctx.memory,
    )


def _instant_imminence(ctx: BotContext, me_blob: BlobView) -> float:
    imminence = 0.0
    for _, enemy in _iter_enemy_blobs(ctx):
        if not _can_eat(enemy.mass, me_blob.mass):
            continue
        dist = hypot(enemy.x - me_blob.x, enemy.y - me_blob.y)
        eat_reach = max(0.0, enemy.radius - me_blob.radius * config.BLOB_EAT_OVERLAP)
        danger_window = max(26.0, enemy.radius * 0.95)
        gap = dist - eat_reach
        if gap < danger_window:
            imminence = max(imminence, 1.0 - _clamp(gap / danger_window, 0.0, 1.0))
    return imminence


def _target_risk(ctx: BotContext, x: float, y: float, my_mass: float) -> float:
    risk = 0.0
    for _, enemy in _iter_enemy_blobs(ctx):
        if not _can_eat(enemy.mass, my_mass):
            continue
        dist = hypot(enemy.x - x, enemy.y - y)
        if dist > 900.0:
            continue
        risk += (900.0 - dist) / 900.0 * (enemy.mass / max(1.0, my_mass))

    return risk


def _crowding_penalty(ctx: BotContext, me: PlayerView, target: BlobView) -> float:
    crowd = 0.0
    crowd_radius = target.radius * 3.8 + 180.0
    for player in ctx.players:
        if player.id == me.id:
            continue
        for blob in player.blobs:
            dist = hypot(blob.x - target.x, blob.y - target.y)
            if dist > crowd_radius:
                continue
            factor = 1.0 - _clamp(dist / crowd_radius, 0.0, 1.0)
            if player.is_bot:
                factor *= 1.25
            crowd += factor

    return crowd


def _best_prey(ctx: BotContext, me_blob: BlobView, x: float, y: float):
    best = None
    best_owner = None
    best_score = -1e9
    best_dist = float("inf")

    for player, enemy in _iter_enemy_blobs(ctx):
        if not _can_eat(me_blob.mass, enemy.mass):
            continue
        if player.team_id and ctx.me.team_id and player.team_id == ctx.me.team_id:
            continue

        dist = hypot(enemy.x - x, enemy.y - y)
        close_factor = max(0.0, 1.0 - dist / (me_blob.radius * 9.0 + 520.0))
        mass_adv = (me_blob.mass / max(1.0, enemy.mass)) - config.BLOB_EAT_RATIO
        risk = _target_risk(ctx, enemy.x, enemy.y, me_blob.mass)
        crowd_penalty = _crowding_penalty(ctx, ctx.me, enemy)
        score = close_factor * 1.25 + min(2.2, mass_adv * 0.55) + enemy.mass * 0.004 - risk * 1.08 - crowd_penalty * 0.3

        if score > best_score:
            best_score = score
            best = enemy
            best_owner = player
            best_dist = dist

    return best_owner, best, best_score, best_dist


def _edible_cluster_value(me_blob: BlobView, target_player: PlayerView) -> tuple[int, float]:
    edible_count = 0
    edible_mass = 0.0
    for blob in target_player.blobs:
        if _can_eat(me_blob.mass, blob.mass):
            edible_count += 1
            edible_mass += blob.mass
    return (edible_count, edible_mass)


def _attack_likelihood(
    me_blob: BlobView,
    target_player: PlayerView,
    target_blob: BlobView,
    *,
    dist: float,
    split_ready: bool,
    aggression: float,
    local_food_density: float,
    on_cooldown: bool,
) -> tuple[float, bool, bool, float]:
    mass_ratio = me_blob.mass / max(1.0, target_blob.mass)
    close_kill = dist < (me_blob.radius * 2.0 + target_blob.radius * 1.2 + 72.0) and mass_ratio > 1.9
    chase_range = me_blob.radius * (4.35 + aggression * 1.1) + 360.0

    post_split_mass = me_blob.mass * 0.5
    split_can_eat = post_split_mass > target_blob.mass * (config.BLOB_EAT_RATIO + 0.03)
    split_reach = me_blob.radius * 2.75 + target_blob.radius * 1.45 + 90.0
    split_window = split_ready and split_can_eat and dist < split_reach

    distance_factor = _clamp(1.0 - dist / max(1.0, chase_range), 0.0, 1.0)
    size_factor = _clamp((mass_ratio - config.BLOB_EAT_RATIO) / 1.65, 0.0, 1.0)
    edible_count, edible_mass = _edible_cluster_value(me_blob, target_player)
    cluster_count_factor = _clamp((edible_count - 1) / 4.0, 0.0, 1.0)
    cluster_mass_factor = _clamp(edible_mass / max(1.0, me_blob.mass * 0.95), 0.0, 1.0)
    cluster_factor = cluster_count_factor * 0.6 + cluster_mass_factor * 0.4

    prob = 0.08 + aggression * 0.1
    prob += distance_factor * 0.36
    prob += size_factor * 0.31
    prob += cluster_factor * 0.23
    if split_window:
        prob += 0.24
    if on_cooldown and not close_kill:
        prob -= 0.34
    if dist > chase_range and not split_window:
        prob -= 0.32
    prob -= local_food_density * 0.2

    return (_clamp(prob, 0.02, 0.98), close_kill, split_window, chase_range)


def _crowd_field(ctx: BotContext, me_blob: BlobView) -> tuple[float, float, float]:
    vx = 0.0
    vy = 0.0
    density = 0.0
    comfort = me_blob.radius * 2.45 + 140.0

    for player, blob in _iter_enemy_blobs(ctx):
        dx = me_blob.x - blob.x
        dy = me_blob.y - blob.y
        dist = max(1.0, hypot(dx, dy))
        if dist >= comfort:
            continue
        ux, uy = _unit(dx, dy)
        p = (comfort - dist) / comfort
        if player.is_bot:
            p *= 1.24
        if _can_eat(me_blob.mass, blob.mass):
            p *= 0.55
        vx += ux * p
        vy += uy * p
        density += p

    return (vx, vy, density)


class SoloSmartBrain:
    def __init__(self, init_ctx: BotInitContext) -> None:
        self.rng = init_ctx.rng
        self.aggression = self.rng.uniform(0.84, 1.22)
        self.caution = self.rng.uniform(0.82, 1.2)
        self.greed = self.rng.uniform(0.84, 1.3)
        self.strafe_dir = -1.0 if self.rng.random() < 0.5 else 1.0
        self.reaction_min = self.rng.uniform(0.065, 0.12)
        self.reaction_max = self.rng.uniform(0.14, 0.27)
        self.view_scale = self.rng.uniform(0.8, 1.03)
        self.steer_rate = self.rng.uniform(6.4, 10.8)
        self.dodge_skill = self.rng.uniform(0.52, 0.86)
        self.threat_miss_chance = self.rng.uniform(0.08, 0.26)
        self.threat_jitter = self.rng.uniform(0.08, 0.22)
        self.panic_split_chance = self.rng.uniform(0.35, 0.7)

    def _cached_action(self, ctx: BotContext) -> BotAction | None:
        cached = ctx.memory.get("last_action")
        if not cached:
            return None
        try:
            tx, ty, split, eject = cached
            return BotAction(float(tx), float(ty), bool(split), bool(eject))
        except Exception:
            return None

    def _schedule_next_think(self, ctx: BotContext, mode: str, emergency: bool = False) -> None:
        min_delay = self.reaction_min
        max_delay = self.reaction_max
        if mode == "flee" or emergency:
            min_delay *= 0.74
            max_delay *= 0.95
        elif mode == "attack":
            min_delay *= 0.82
            max_delay *= 0.95
        ctx.memory["next_think_at"] = ctx.now + self.rng.uniform(min_delay, max_delay)

    def _remember_action(self, ctx: BotContext, action: BotAction, mode: str, emergency: bool = False) -> BotAction:
        prev_x = float(ctx.memory.get("aim_x", action.target_x))
        prev_y = float(ctx.memory.get("aim_y", action.target_y))
        dt = max(1 / 120, ctx.dt)
        rate = self.steer_rate
        if mode == "flee" or emergency:
            rate *= 0.95 + self.dodge_skill * 0.55
        elif mode == "attack":
            rate *= 1.02 + self.aggression * 0.22
        alpha = 1.0 - exp(-rate * dt)
        # Slight floor keeps turn response smooth but not sluggish.
        alpha = _clamp(alpha, 0.14, 0.92)

        target_x = _lerp(prev_x, action.target_x, alpha)
        target_y = _lerp(prev_y, action.target_y, alpha)
        ctx.memory["aim_x"] = target_x
        ctx.memory["aim_y"] = target_y

        final_action = BotAction(target_x=target_x, target_y=target_y, split=action.split, eject=action.eject)
        ctx.memory["last_action"] = (
            float(final_action.target_x),
            float(final_action.target_y),
            bool(final_action.split),
            bool(final_action.eject),
        )
        self._schedule_next_think(ctx, mode=mode, emergency=emergency)
        return final_action

    def decide(self, ctx: BotContext) -> BotAction:
        me_blob = _largest_blob(ctx.me)
        me_small = _smallest_blob(ctx.me)
        if me_blob is None or me_small is None:
            return BotAction(ctx.world_width * 0.5, ctx.world_height * 0.5)

        me_x, me_y = _player_center(ctx.me)
        base_view = 860.0 + me_blob.radius * 8.8 + len(ctx.me.blobs) * 52.0
        view_range = base_view * self.view_scale
        perceived_ctx = _clip_context_for_view(ctx, me_x, me_y, view_range)
        instant_danger = _instant_imminence(perceived_ctx, me_small)
        next_think_at = float(ctx.memory.get("next_think_at", 0.0))
        danger_override = 0.94 + (1.0 - self.dodge_skill) * 0.06
        if ctx.now < next_think_at and instant_danger < danger_override:
            cached = self._cached_action(ctx)
            if cached is not None:
                return cached

        max_split_blobs = min(config.MAX_PLAYER_BLOBS, 8)
        split_ready = (
            len(ctx.me.blobs) < max_split_blobs
            and me_blob.mass >= config.PLAYER_MIN_SPLIT_MASS
            and ctx.now >= float(ctx.memory.get("next_split_at", 0.0))
        )

        threat_x, threat_y, threat_pressure, imminence = _threat_field(perceived_ctx)
        if self.rng.random() < self.threat_miss_chance:
            miss_scale = self.rng.uniform(0.22, 0.68)
            threat_x *= miss_scale
            threat_y *= miss_scale
            threat_pressure *= miss_scale
            imminence *= miss_scale
        threat_x, threat_y = _jitter_vector(
            threat_x,
            threat_y,
            self.rng,
            self.threat_jitter * (1.14 - self.dodge_skill),
        )
        wall_x, wall_y, wall_pressure = _wall_field(perceived_ctx, me_x, me_y, me_blob.radius)
        virus_x, virus_y, virus_pressure = _virus_field(perceived_ctx, me_blob)
        crowd_x, crowd_y, crowd_pressure = _crowd_field(perceived_ctx, me_blob)
        prey_owner, prey, attack_score, prey_dist = _best_prey(perceived_ctx, me_blob, me_x, me_y)
        food_x, food_y, nearest_food = _food_field(perceived_ctx, me_x, me_y)
        imminence = max(imminence, instant_danger)
        local_food_density = _local_food_density(perceived_ctx, me_x, me_y)

        effective_threat = threat_pressure + wall_pressure * 0.35 + virus_pressure * 0.65 + crowd_pressure * 0.22
        effective_threat *= 0.74 + self.dodge_skill * 0.4
        flee_threshold = 1.03 / self.caution
        imminent_threshold = 0.76 / self.caution
        attack_threshold = 0.24 / self.aggression
        attack_cooldown_until = float(ctx.memory.get("attack_cooldown_until", 0.0))
        desired_mode = "farm"
        if effective_threat > flee_threshold or imminence > imminent_threshold:
            desired_mode = "flee"
        elif prey is not None and prey_owner is not None:
            on_cooldown = ctx.now < attack_cooldown_until
            engage_prob, close_kill, split_window, chase_range = _attack_likelihood(
                me_blob,
                prey_owner,
                prey,
                dist=prey_dist,
                split_ready=split_ready,
                aggression=self.aggression,
                local_food_density=local_food_density,
                on_cooldown=on_cooldown,
            )

            if ctx.now >= float(ctx.memory.get("next_engage_roll_at", 0.0)):
                ctx.memory["engage_roll"] = self.rng.random()
                ctx.memory["next_engage_roll_at"] = ctx.now + self.rng.uniform(0.28, 0.68)
            engage_roll = float(ctx.memory.get("engage_roll", 1.0))
            willing = close_kill or split_window or engage_roll < engage_prob

            can_attack = close_kill or split_window or prey_dist <= chase_range

            attack_need = attack_threshold + local_food_density * 0.2
            if prey_dist > chase_range * 0.72:
                attack_need += 0.16
            if split_window:
                attack_need -= 0.12

            if can_attack and willing and attack_score > attack_need and effective_threat < 0.94 * self.caution:
                desired_mode = "attack"

        # Small non-deterministic mode nudges to avoid robotic behavior.
        if desired_mode == "farm" and prey is not None and self.rng.random() < 0.03 * self.aggression:
            if attack_score > (attack_threshold + local_food_density * 0.12) * 0.88 and effective_threat < 0.8 * self.caution:
                desired_mode = "attack"
        if desired_mode == "attack" and self.rng.random() < 0.02 * self.caution and effective_threat > 0.75:
            desired_mode = "farm"

        prev_mode = str(ctx.memory.get("mode", "farm"))
        hold_until = float(ctx.memory.get("mode_hold_until", 0.0))
        mode = desired_mode

        if ctx.now < hold_until:
            if prev_mode == "flee":
                if desired_mode != "flee" and effective_threat < 0.3 / self.caution and imminence < 0.2:
                    mode = desired_mode
                else:
                    mode = "flee"
            elif prev_mode == "attack":
                if desired_mode == "flee":
                    mode = "flee"
                elif ctx.now < float(ctx.memory.get("attack_commit_until", 0.0)) and prey is not None:
                    mode = "attack"
                elif prey is not None and effective_threat < 1.05 * self.caution:
                    mode = "attack"
                else:
                    mode = desired_mode
            else:
                mode = "flee" if desired_mode == "flee" else "farm"

        if mode != prev_mode:
            if mode == "flee":
                ctx.memory["mode_hold_until"] = ctx.now + self.rng.uniform(0.62, 1.08)
            elif mode == "attack":
                ctx.memory["mode_hold_until"] = ctx.now + self.rng.uniform(0.55, 0.95)
                ctx.memory["attack_commit_until"] = ctx.now + self.rng.uniform(0.45, 0.92)
            else:
                ctx.memory["mode_hold_until"] = ctx.now + self.rng.uniform(0.38, 0.75)
                if prev_mode == "attack":
                    ctx.memory["attack_cooldown_until"] = ctx.now + self.rng.uniform(0.55, 1.5)
        ctx.memory["mode"] = mode

        if mode == "flee":
            flee_gain = 0.76 + self.dodge_skill * 0.56
            vx = threat_x * (1.05 * flee_gain) + wall_x * 1.05 + virus_x * 1.15 + crowd_x * 0.86
            vy = threat_y * (1.05 * flee_gain) + wall_y * 1.05 + virus_y * 1.15 + crowd_y * 0.86
            vx, vy = _jitter_vector(vx, vy, self.rng, self.threat_jitter * (1.2 - self.dodge_skill))
            ux, uy = _unit(vx, vy)
            if ux == 0.0 and uy == 0.0:
                # Fallback: run toward center if vectors cancel.
                ux, uy = _unit(ctx.world_width * 0.5 - me_x, ctx.world_height * 0.5 - me_y)
            flee_dist = 780.0 + self.rng.uniform(-140.0, 170.0)
            target_x = me_x + ux * flee_dist
            target_y = me_y + uy * flee_dist

            split_escape = (
                split_ready
                and imminence > 0.84 / self.caution
                and effective_threat > 1.06 / self.caution
                and me_small.mass > config.MIN_BLOB_MASS * 1.25
                and self.rng.random() < (0.22 + imminence * 0.28) * self.panic_split_chance
            )
            if split_escape:
                ctx.memory["next_split_at"] = ctx.now + self.rng.uniform(0.78, 1.05)
            emergency_trigger = (
                imminence > (0.95 + (1.0 - self.dodge_skill) * 0.04)
                and self.rng.random() < (0.34 + self.dodge_skill * 0.48)
            )

            return self._remember_action(
                ctx,
                BotAction(
                target_x=_clamp(target_x, 0.0, ctx.world_width),
                target_y=_clamp(target_y, 0.0, ctx.world_height),
                split=split_escape,
                ),
                mode="flee",
                emergency=emergency_trigger,
            )

        if mode == "attack" and prey is not None:
            to_target_x = prey.x - me_x
            to_target_y = prey.y - me_y
            dist = max(1.0, hypot(to_target_x, to_target_y))
            ux, uy = _unit(to_target_x, to_target_y)
            near_finish = dist < (me_blob.radius + prey.radius) * 1.12

            if ctx.now >= float(ctx.memory.get("next_strafe_at", 0.0)):
                self.strafe_dir *= -1.0 if self.rng.random() < 0.78 else 1.0
                ctx.memory["next_strafe_at"] = ctx.now + self.rng.uniform(0.4, 0.95)

            push_through = max(185.0, min(360.0, me_blob.radius * 1.34 + prey.radius * 0.8))
            strafe_amp = 0.0 if near_finish else min(120.0, dist * 0.34) * self.strafe_dir
            px, py = -uy, ux

            # Keep target beyond prey so close-range chases don't stall beside the target.
            target_x = prey.x + ux * push_through + px * strafe_amp
            target_y = prey.y + uy * push_through + py * strafe_amp
            target_x += wall_x * 120.0 + virus_x * 105.0 - threat_x * 38.0 + crowd_x * 84.0
            target_y += wall_y * 120.0 + virus_y * 105.0 - threat_y * 38.0 + crowd_y * 84.0

            post_split_mass = me_blob.mass * 0.5
            split_can_eat = post_split_mass > prey.mass * (config.BLOB_EAT_RATIO + 0.03)
            split_reach = me_blob.radius * 2.75 + prey.radius * 1.45 + 90.0

            split_kill = (
                split_ready
                and split_can_eat
                and prey_dist < split_reach
                and effective_threat < 0.58 * self.caution
                and attack_score > max(0.2, attack_threshold * 0.82)
                and self.rng.random() < (0.52 + min(0.32, attack_score * 0.22))
            )
            if split_kill:
                ctx.memory["next_split_at"] = ctx.now + self.rng.uniform(0.78, 1.02)

            return self._remember_action(
                ctx,
                BotAction(
                target_x=_clamp(target_x, 0.0, ctx.world_width),
                target_y=_clamp(target_y, 0.0, ctx.world_height),
                split=split_kill,
                ),
                mode="attack",
            )

        route_target = ctx.memory.get("route_target")
        route_dist = float("inf")
        if route_target is not None:
            route_dist = hypot(float(route_target[0]) - me_x, float(route_target[1]) - me_y)

        if (
            ctx.now >= float(ctx.memory.get("next_route_at", 0.0))
            or route_target is None
            or route_dist < max(85.0, me_blob.radius * 1.2)
        ):
            hotspot = _best_food_hotspot(perceived_ctx, me_x, me_y, self.rng)
            picked = hotspot or _sample_food_route(perceived_ctx, me_x, me_y, self.rng, self.greed)
            if picked is not None:
                route_target = picked
                ctx.memory["route_target"] = picked
            ctx.memory["next_route_at"] = ctx.now + self.rng.uniform(0.9, 2.1)

        route_x = 0.0
        route_y = 0.0
        if route_target is not None:
            rx = float(route_target[0]) - me_x
            ry = float(route_target[1]) - me_y
            rux, ruy = _unit(rx, ry)
            route_x = rux
            route_y = ruy

        lane_dx = float(ctx.memory.get("lane_dx", 0.0))
        lane_dy = float(ctx.memory.get("lane_dy", 0.0))
        lane_until = float(ctx.memory.get("lane_until", 0.0))
        if ctx.now >= lane_until or (lane_dx == 0.0 and lane_dy == 0.0):
            anchor = route_target
            if anchor is None and nearest_food is not None:
                anchor = nearest_food
            if anchor is not None:
                lane_dx, lane_dy = _unit(float(anchor[0]) - me_x, float(anchor[1]) - me_y)
            if lane_dx == 0.0 and lane_dy == 0.0:
                lane_dx, lane_dy = _unit(
                    self.rng.uniform(-1.0, 1.0),
                    self.rng.uniform(-1.0, 1.0),
                )
            ctx.memory["lane_until"] = ctx.now + self.rng.uniform(0.95, 2.4)

        if route_target is not None:
            steer_x = _lerp(lane_dx, route_x, 0.16)
            steer_y = _lerp(lane_dy, route_y, 0.16)
            lane_dx, lane_dy = _unit(steer_x, steer_y)

        ctx.memory["lane_dx"] = lane_dx
        ctx.memory["lane_dy"] = lane_dy

        food_dir_x, food_dir_y = _unit(food_x, food_y)
        vx = (
            lane_dx * (1.08 + self.greed * 0.28)
            + route_x * 0.58
            + food_dir_x * 0.48
            + wall_x * 0.72
            + virus_x * 0.82
            + crowd_x * 0.95
            - threat_x * 0.2
        )
        vy = (
            lane_dy * (1.08 + self.greed * 0.28)
            + route_y * 0.58
            + food_dir_y * 0.48
            + wall_y * 0.72
            + virus_y * 0.82
            + crowd_y * 0.95
            - threat_y * 0.2
        )
        ux, uy = _unit(vx, vy)
        if ux != 0.0 or uy != 0.0:
            step = 760.0 + self.rng.uniform(-70.0, 110.0)
            target_x = me_x + ux * step
            target_y = me_y + uy * step
            return self._remember_action(
                ctx,
                BotAction(
                target_x=_clamp(target_x, 0.0, ctx.world_width),
                target_y=_clamp(target_y, 0.0, ctx.world_height),
                ),
                mode="farm",
            )

        if nearest_food is not None:
            return self._remember_action(
                ctx,
                BotAction(
                    target_x=nearest_food[0],
                    target_y=nearest_food[1],
                ),
                mode="farm",
            )

        if ctx.now >= float(ctx.memory.get("next_wander_at", 0.0)):
            jitter = 420.0
            ctx.memory["wander_x"] = _clamp(
                me_x + self.rng.uniform(-jitter, jitter),
                0.0,
                ctx.world_width,
            )
            ctx.memory["wander_y"] = _clamp(
                me_y + self.rng.uniform(-jitter, jitter),
                0.0,
                ctx.world_height,
            )
            ctx.memory["next_wander_at"] = ctx.now + self.rng.uniform(0.55, 1.25)

        return self._remember_action(
            ctx,
            BotAction(
                target_x=float(ctx.memory.get("wander_x", me_x)),
                target_y=float(ctx.memory.get("wander_y", me_y)),
            ),
            mode="farm",
        )


class ForagerBrain:
    def __init__(self, init_ctx: BotInitContext) -> None:
        self._inner = SoloSmartBrain(init_ctx)

    def decide(self, ctx: BotContext) -> BotAction:
        # Backward-compat alias: upgraded to the smarter solo policy.
        return self._inner.decide(ctx)


class PredatorBrain:
    def __init__(self, init_ctx: BotInitContext) -> None:
        self._inner = SoloSmartBrain(init_ctx)

    def decide(self, ctx: BotContext) -> BotAction:
        # Backward-compat alias: upgraded to the smarter solo policy.
        return self._inner.decide(ctx)


class TeamSwarmBrain:
    def __init__(self, init_ctx: BotInitContext) -> None:
        self.rng = init_ctx.rng
        self.team_id = init_ctx.team_id or "swarm"

    def decide(self, ctx: BotContext) -> BotAction:
        me_blob = _largest_blob(ctx.me)
        if me_blob is None:
            return BotAction(ctx.world_width * 0.5, ctx.world_height * 0.5)

        teammates = [
            p
            for p in ctx.players
            if p.id != ctx.me.id
            and p.team_id == self.team_id
            and p.plugin_name == ctx.me.plugin_name
            and p.total_mass > 0
        ]
        team_mass = ctx.me.total_mass + sum(p.total_mass for p in teammates)

        focus_id = ctx.team_state.get("focus_id")
        focus_until = float(ctx.team_state.get("focus_until", 0.0))
        focus = None
        if focus_id and ctx.now <= focus_until:
            focus = next((p for p in ctx.players if p.id == focus_id and p.total_mass > 0), None)

        if focus is None:
            candidates = [
                p
                for p in ctx.players
                if p.id != ctx.me.id
                and p.total_mass > 0
                and p.team_id != self.team_id
            ]
            ranked = []
            for enemy in candidates:
                ex, ey = _player_center(enemy)
                d = hypot(ex - me_blob.x, ey - me_blob.y)
                # Prefer close targets that team can overwhelm.
                strength = enemy.total_mass / max(1.0, team_mass)
                score = d * (0.55 + strength)
                ranked.append((score, enemy))
            ranked.sort(key=lambda row: row[0])
            if ranked:
                focus = ranked[0][1]
                ctx.team_state["focus_id"] = focus.id
                ctx.team_state["focus_until"] = ctx.now + 1.1

        if focus is None:
            tx, ty = _closest_food_target(ctx, me_blob.x, me_blob.y)
            return BotAction(tx, ty)

        fx, fy = _player_center(focus)
        members = tuple(ctx.team_state.get("members") or ())
        try:
            rank = members.index(ctx.me.id)
        except ValueError:
            rank = 0

        spacing = 55.0 + (rank // 2) * 26.0
        side = -1.0 if rank % 2 == 0 else 1.0

        to_enemy_x = fx - me_blob.x
        to_enemy_y = fy - me_blob.y
        dist = max(1.0, hypot(to_enemy_x, to_enemy_y))
        nx = to_enemy_x / dist
        ny = to_enemy_y / dist
        px = -ny
        py = nx

        target_x = fx - nx * min(25.0, focus.total_mass * 0.02) + px * side * spacing
        target_y = fy - ny * min(25.0, focus.total_mass * 0.02) + py * side * spacing

        should_split = (
            len(ctx.me.blobs) < 8
            and me_blob.mass > focus.total_mass * 0.62
            and dist < me_blob.radius * 2.2
            and ctx.now >= float(ctx.memory.get("next_split_at", 0.0))
        )
        if should_split:
            ctx.memory["next_split_at"] = ctx.now + 1.0

        should_eject = False
        if teammates and ctx.me.total_mass > 110 and ctx.now >= float(ctx.memory.get("next_eject_at", 0.0)):
            heavier_mate = max(teammates, key=lambda p: p.total_mass, default=None)
            if heavier_mate and heavier_mate.total_mass > ctx.me.total_mass * 1.45:
                mx, my = _player_center(heavier_mate)
                mate_dist = hypot(mx - me_blob.x, my - me_blob.y)
                if mate_dist < me_blob.radius * 6.5:
                    target_x, target_y = mx, my
                    should_eject = True
                    ctx.memory["next_eject_at"] = ctx.now + 0.75

        return BotAction(
            target_x=_clamp(target_x, 0.0, ctx.world_width),
            target_y=_clamp(target_y, 0.0, ctx.world_height),
            split=should_split,
            eject=should_eject,
        )


def register(registry: BotRegistry) -> None:
    registry.register("solo_smart", lambda init_ctx: SoloSmartBrain(init_ctx))
    registry.register("solo", lambda init_ctx: SoloSmartBrain(init_ctx))
    registry.register("forager", lambda init_ctx: ForagerBrain(init_ctx))
    registry.register("predator", lambda init_ctx: PredatorBrain(init_ctx))
    registry.register("team_swarm", lambda init_ctx: TeamSwarmBrain(init_ctx))
    registry.register("swarm", lambda init_ctx: TeamSwarmBrain(init_ctx))

"""Bot plugin manager and runtime orchestration."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .. import config
from .registry import BotRegistry, load_plugin_modules
from .types import (
    BlobView,
    BotAction,
    BotContext,
    BotInitContext,
    BotSpec,
    EjectedView,
    FoodView,
    PlayerView,
    VirusView,
)

if TYPE_CHECKING:
    from ..world import GameWorld

logger = logging.getLogger(__name__)


def parse_bot_specs(raw: str) -> list[BotSpec]:
    specs: list[BotSpec] = []
    cleaned = (raw or "").strip()
    if not cleaned:
        return specs

    for chunk in cleaned.split(","):
        part = chunk.strip()
        if not part:
            continue
        tokens = [t.strip() for t in part.split(":")]
        if not tokens[0]:
            raise ValueError(f"Invalid bot spec '{part}': plugin name is required")

        plugin_name = tokens[0].lower()
        count = 1
        team_id: str | None = None
        name_prefix: str | None = None

        if len(tokens) >= 2 and tokens[1]:
            count = int(tokens[1])
        if len(tokens) >= 3 and tokens[2] and tokens[2] != "-":
            team_id = tokens[2]
        if len(tokens) >= 4 and tokens[3] and tokens[3] != "-":
            name_prefix = tokens[3]

        if count <= 0:
            raise ValueError(f"Invalid bot spec '{part}': count must be > 0")

        specs.append(BotSpec(plugin_name=plugin_name, count=count, team_id=team_id, name_prefix=name_prefix))

    return specs


@dataclass(slots=True)
class _BotAgent:
    player_id: str
    plugin_name: str
    team_id: str | None
    name_prefix: str
    brain: object
    memory: dict


class BotManager:
    def __init__(
        self,
        world: GameWorld,
        *,
        enabled: bool,
        plugin_modules: tuple[str, ...],
        bot_specs: list[BotSpec],
        seed: int,
    ) -> None:
        self.world = world
        self.enabled = enabled
        self.plugin_modules = plugin_modules
        self.bot_specs = bot_specs
        self._started = False
        self._registry = BotRegistry()
        self._rng = random.Random(seed)
        self._agents: dict[str, _BotAgent] = {}
        self._team_state: dict[str, dict] = {}
        self._spawn_index = 0

    @classmethod
    def from_config(cls, world: GameWorld) -> BotManager:
        specs = parse_bot_specs(config.BOT_SPECS)
        return cls(
            world,
            enabled=config.BOTS_ENABLED,
            plugin_modules=config.BOT_PLUGIN_MODULES,
            bot_specs=specs,
            seed=config.BOT_RANDOM_SEED,
        )

    def describe(self) -> dict:
        return {
            "enabled": self.enabled,
            "started": self._started,
            "pluginModules": list(self.plugin_modules),
            "registeredPlugins": list(self._registry.names),
            "botSpecs": [
                {
                    "plugin": spec.plugin_name,
                    "count": spec.count,
                    "team": spec.team_id,
                    "namePrefix": spec.name_prefix,
                }
                for spec in self.bot_specs
            ],
            "activeBots": len(self._agents),
            "spawnOnEaten": config.BOT_SPAWN_ON_EATEN,
            "spawnPerElimination": config.BOT_SPAWN_PER_ELIMINATION,
            "maxActiveBots": config.BOT_MAX_ACTIVE,
        }

    def ensure_started(self, now: float) -> None:
        if self._started or not self.enabled:
            return

        load_plugin_modules(self.plugin_modules, self._registry)
        for spec in self.bot_specs:
            for i in range(spec.count):
                name_prefix = spec.name_prefix or spec.plugin_name.replace("_", " ").title()
                player_name = self._build_name(spec, i)
                self._spawn_bot(
                    plugin_name=spec.plugin_name,
                    team_id=spec.team_id,
                    name_prefix=name_prefix,
                    now=now,
                    explicit_name=player_name,
                )

        self._started = True
        logger.info("BotManager started with %d bots using plugins: %s", len(self._agents), ", ".join(self._registry.names))

    def tick(self, dt: float, now: float) -> None:
        if not self._started or not self._agents:
            return

        players_by_id, players, foods, ejected, viruses = self._build_views()
        if not players_by_id:
            return

        team_members: dict[str, list[str]] = defaultdict(list)
        for agent in self._agents.values():
            team_key = self._team_key(agent.plugin_name, agent.team_id)
            team_members[team_key].append(agent.player_id)
        for key, members in team_members.items():
            state = self._team_state.setdefault(key, {})
            state["members"] = tuple(sorted(pid for pid in members if pid in players_by_id))

        for player_id, agent in list(self._agents.items()):
            me = players_by_id.get(player_id)
            if me is None:
                # If a bot was explicitly removed, drop it from the bot manager.
                self._agents.pop(player_id, None)
                continue

            was_alive = bool(agent.memory.get("_was_alive", bool(me.blobs)))
            is_alive = bool(me.blobs)
            if not is_alive and was_alive:
                self._spawn_extra_on_elimination(agent, now)
            agent.memory["_was_alive"] = is_alive

            team_key = self._team_key(agent.plugin_name, agent.team_id)
            team_state = self._team_state.setdefault(team_key, {})
            ctx = BotContext(
                now=now,
                dt=dt,
                world_width=config.WORLD_WIDTH,
                world_height=config.WORLD_HEIGHT,
                me=me,
                players=players,
                foods=foods,
                ejected=ejected,
                viruses=viruses,
                team_state=team_state,
                memory=agent.memory,
            )

            try:
                action = agent.brain.decide(ctx)
            except Exception:
                logger.exception("Bot plugin '%s' failed for player %s", agent.plugin_name, player_id)
                action = self._fallback_action(me)

            self.world.set_input(
                player_id=player_id,
                target_x=self._clamp(action.target_x, 0.0, config.WORLD_WIDTH),
                target_y=self._clamp(action.target_y, 0.0, config.WORLD_HEIGHT),
                split=bool(action.split),
                eject=bool(action.eject),
            )

    def _build_views(
        self,
    ) -> tuple[
        dict[str, PlayerView],
        tuple[PlayerView, ...],
        tuple[FoodView, ...],
        tuple[EjectedView, ...],
        tuple[VirusView, ...],
    ]:
        players: list[PlayerView] = []
        for player in self.world.players.values():
            blobs = tuple(
                BlobView(
                    id=blob.id,
                    player_id=blob.player_id,
                    x=blob.x,
                    y=blob.y,
                    mass=blob.mass,
                    radius=blob.radius,
                )
                for blob in player.blobs.values()
            )
            players.append(
                PlayerView(
                    id=player.id,
                    name=player.name,
                    color=player.color,
                    is_bot=player.is_bot,
                    plugin_name=player.bot_plugin,
                    team_id=player.bot_team,
                    total_mass=sum(b.mass for b in blobs),
                    blobs=blobs,
                )
            )

        players.sort(key=lambda p: p.id)
        players_by_id = {p.id: p for p in players}

        foods = tuple(
            FoodView(
                id=food.id,
                x=food.x,
                y=food.y,
                mass=food.mass,
                radius=food.radius,
                color=food.color,
            )
            for food in self.world.foods.values()
        )
        ejected = tuple(
            EjectedView(
                id=item.id,
                x=item.x,
                y=item.y,
                mass=item.mass,
                radius=item.radius,
                owner_id=item.owner_id,
                ttl=item.ttl,
            )
            for item in self.world.ejected.values()
        )
        viruses = tuple(
            VirusView(id=v.id, x=v.x, y=v.y, mass=v.mass, radius=v.radius)
            for v in self.world.viruses.values()
        )

        return (players_by_id, tuple(players), foods, ejected, viruses)

    def _fallback_action(self, me: PlayerView) -> BotAction:
        if me.blobs:
            first = me.blobs[0]
            return BotAction(target_x=first.x, target_y=first.y)
        return BotAction(target_x=config.WORLD_WIDTH / 2.0, target_y=config.WORLD_HEIGHT / 2.0)

    def _build_name(self, spec: BotSpec, index: int) -> str:
        prefix = spec.name_prefix or spec.plugin_name.replace("_", " ").title()
        return f"{prefix}-{index + 1}"

    def _spawn_bot(
        self,
        *,
        plugin_name: str,
        team_id: str | None,
        name_prefix: str,
        now: float,
        explicit_name: str | None = None,
    ) -> str:
        self._spawn_index += 1
        sequence = self._spawn_index
        bot_name = explicit_name or f"{name_prefix}-{sequence}"
        player = self.world.add_player(
            player_name=bot_name,
            now=now,
            is_bot=True,
            bot_plugin=plugin_name,
            bot_team=team_id,
        )
        try:
            init_ctx = BotInitContext(
                plugin_name=plugin_name,
                bot_name=player.name,
                team_id=team_id,
                bot_index=sequence,
                rng=random.Random(self._rng.randint(0, 2_000_000_000)),
            )
            brain = self._registry.create(plugin_name, init_ctx)
        except Exception:
            self.world.remove_player(player.id)
            raise

        self._agents[player.id] = _BotAgent(
            player_id=player.id,
            plugin_name=plugin_name,
            team_id=team_id,
            name_prefix=name_prefix,
            brain=brain,
            memory={"_was_alive": True},
        )
        return player.id

    def _spawn_extra_on_elimination(self, eliminated: _BotAgent, now: float) -> None:
        if not config.BOT_SPAWN_ON_EATEN:
            return
        if config.BOT_SPAWN_PER_ELIMINATION <= 0:
            return

        max_active = max(1, config.BOT_MAX_ACTIVE)
        spawned = 0
        for _ in range(config.BOT_SPAWN_PER_ELIMINATION):
            if len(self._agents) >= max_active:
                break
            try:
                self._spawn_bot(
                    plugin_name=eliminated.plugin_name,
                    team_id=eliminated.team_id,
                    name_prefix=eliminated.name_prefix,
                    now=now,
                )
                spawned += 1
            except Exception:
                logger.exception(
                    "Failed to spawn extra bot for eliminated player %s using plugin '%s'",
                    eliminated.player_id,
                    eliminated.plugin_name,
                )
                break

        if spawned > 0:
            logger.info(
                "Bot '%s' was eliminated. Spawned %d extra bot(s). Active bots: %d/%d",
                eliminated.player_id,
                spawned,
                len(self._agents),
                max_active,
            )

    def _team_key(self, plugin_name: str, team_id: str | None) -> str:
        return f"{plugin_name}:{team_id or '-'}"

    @staticmethod
    def _clamp(value: float, min_value: float, max_value: float) -> float:
        return min(max(value, min_value), max_value)

"""Runtime bot plugin that uses a tiny policy/value model + PUCT search."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from agario.bots.registry import BotRegistry
from agario.bots.types import BotAction, BotContext, BotInitContext
from agario.world import GameWorld

from .action_space import ActionSpace
from .io import create_device, load_or_init_model
from .observation import OBSERVATION_SIZE
from .planner import PUCTPlanner
from .settings import AiBotSettings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _RuntimeShared:
    settings: AiBotSettings
    action_space: ActionSpace
    planner: PUCTPlanner


_SHARED: _RuntimeShared | None = None


def _build_shared() -> _RuntimeShared:
    settings = AiBotSettings.from_env()
    action_space = ActionSpace(
        direction_bins=settings.direction_bins,
        magnitude_bins=settings.magnitude_bins,
        target_distance=settings.default_target_distance,
    )
    device = create_device(settings.device_preference)
    model = load_or_init_model(
        model_path=settings.model_path,
        obs_size=OBSERVATION_SIZE,
        action_count=action_space.size,
        device=device,
        hidden_size=256,
    )
    model.eval()
    planner = PUCTPlanner(
        model=model,
        action_space=action_space,
        device=device,
        settings=settings,
    )
    logger.info(
        "Loaded ai_search model from %s on %s with %d simulations",
        settings.model_path,
        device.type,
        settings.simulations,
    )
    return _RuntimeShared(settings=settings, action_space=action_space, planner=planner)


def _get_shared() -> _RuntimeShared:
    global _SHARED
    if _SHARED is None:
        _SHARED = _build_shared()
    return _SHARED


class AISearchBrain:
    def __init__(self, init_ctx: BotInitContext, shared: _RuntimeShared) -> None:
        self.init_ctx = init_ctx
        self.shared = shared

    def decide(self, ctx: BotContext) -> BotAction:
        cached_action = ctx.memory.get("_ai_cached_action")
        next_decision_at = float(ctx.memory.get("_ai_next_decision_at", -1e9))
        if cached_action is not None and ctx.now < next_decision_at:
            return cached_action

        world = ctx.memory.get("_world")
        if not isinstance(world, GameWorld):
            if not ctx.me.blobs:
                action = BotAction(target_x=ctx.world_width * 0.5, target_y=ctx.world_height * 0.5)
                ctx.memory["_ai_cached_action"] = action
                ctx.memory["_ai_next_decision_at"] = ctx.now + self.shared.settings.runtime_decision_interval
                return action
            center = ctx.me.blobs[0]
            action = BotAction(target_x=center.x, target_y=center.y)
            ctx.memory["_ai_cached_action"] = action
            ctx.memory["_ai_next_decision_at"] = ctx.now + self.shared.settings.runtime_decision_interval
            return action

        search = self.shared.planner.search(
            world=world,
            player_id=ctx.me.id,
            now=ctx.now,
            dt=ctx.dt,
            training=False,
        )
        tx, ty, split, eject = self.shared.planner.action_to_world_input(
            world=world,
            player_id=ctx.me.id,
            action_index=search.action_index,
        )
        action = BotAction(target_x=tx, target_y=ty, split=split, eject=eject)
        ctx.memory["_ai_cached_action"] = action
        ctx.memory["_ai_next_decision_at"] = ctx.now + self.shared.settings.runtime_decision_interval
        return action


def register(registry: BotRegistry) -> None:
    shared = _get_shared()
    registry.register("ai_search", lambda init_ctx: AISearchBrain(init_ctx, shared))

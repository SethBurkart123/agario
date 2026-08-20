"""Neural bot plugin: runs a PPO-trained policy in the live game.

Enable with e.g.:
  AGARIO_BOT_PLUGIN_MODULES=bot_solutions.programmatic,bot_solutions.rl_v1.plugin \
  AGARIO_BOT_SPECS=neural:8 uv run python main.py

Checkpoint path comes from AGARIO_NEURAL_CHECKPOINT (default
bot_solutions/rl_v1/checkpoints/rl/latest.pt). Difficulty knobs make the bot fun rather than
maximally sharp:
  AGARIO_NEURAL_TEMPERATURE  (default 1.0; higher = sloppier choices)
  AGARIO_NEURAL_THINK_SECONDS (default 0.08; reaction delay between decisions)
"""

from __future__ import annotations

import os
from functools import lru_cache

from agario.bots.registry import BotRegistry
from agario.bots.types import BotAction, BotContext, BotInitContext

from . import CHECKPOINT_ROOT
from .obs import (
    N_DIRECTIONS,
    AgentPerception,
    action_to_target,
    apply_turn,
    encode,
)

DEFAULT_CHECKPOINT = str(CHECKPOINT_ROOT / "rl/latest.pt")


@lru_cache(maxsize=4)
def _load_policy(path: str):
    import torch

    from .model import PolicyNet

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    policy = PolicyNet(**ckpt.get("model_kwargs", {}))
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    return policy


class NeuralBrain:
    def __init__(self, init_ctx: BotInitContext) -> None:
        self._rng = init_ctx.rng
        self._checkpoint = os.getenv("AGARIO_NEURAL_CHECKPOINT", DEFAULT_CHECKPOINT)
        # Deterministic by default: the policy plays its best judgment.
        # Set AGARIO_NEURAL_SAMPLE=1 (with AGARIO_NEURAL_TEMPERATURE) to get
        # stochastic, sloppier bots for easier difficulty.
        self._sample = os.getenv("AGARIO_NEURAL_SAMPLE", "0").strip() == "1"
        self._temperature = float(os.getenv("AGARIO_NEURAL_TEMPERATURE", "0.7"))
        self._think_seconds = float(os.getenv("AGARIO_NEURAL_THINK_SECONDS", "0.08"))
        # Stagger thinking so a fleet of neural bots doesn't act in lockstep.
        self._next_think = self._rng.random() * self._think_seconds
        self._cached: BotAction | None = None
        self._hidden = None  # GRU memory, reset on death
        self._was_alive = False
        self._heading = self._rng.randrange(N_DIRECTIONS)
        self._prev = (0, 0, 1.0)  # (turn, op, speed) — part of the observation

    def decide(self, ctx: BotContext) -> BotAction:
        if not ctx.me.blobs:
            self._was_alive = False
            return BotAction(target_x=ctx.world_width / 2.0, target_y=ctx.world_height / 2.0)

        if self._cached is not None and ctx.now < self._next_think:
            # Replay movement but never repeat one-shot split/eject actions.
            return BotAction(
                target_x=self._cached.target_x, target_y=self._cached.target_y
            )
        self._next_think = ctx.now + self._think_seconds

        import torch

        now = ctx.now
        own = [
            (b.x, b.y, b.mass, b.radius, b.vx, b.vy, max(0.0, b.can_merge_at - now))
            for b in ctx.me.blobs
        ]
        enemy = [
            (
                b.x, b.y, b.mass, b.radius, b.vx, b.vy,
                p.total_mass, len(p.blobs), max(0.0, b.can_merge_at - now),
            )
            for p in ctx.players
            if p.id != ctx.me.id
            for b in p.blobs
        ]
        perception = AgentPerception(
            world_w=ctx.world_width,
            world_h=ctx.world_height,
            heading=self._heading,
            prev_turn=self._prev[0],
            prev_op=self._prev[1],
            prev_speed=self._prev[2],
            own_blobs=own,
            enemy_blobs=enemy,
            foods=[(f.x, f.y, f.mass) for f in ctx.foods],
            ejected=[(e.x, e.y, e.mass) for e in ctx.ejected],
            viruses=[(v.x, v.y, v.radius) for v in ctx.viruses],
        )
        obs = torch.as_tensor(encode(perception)).unsqueeze(0)

        policy = _load_policy(self._checkpoint)
        if self._hidden is None:
            self._hidden = policy.initial_state(1)
        done = torch.tensor([0.0 if self._was_alive else 1.0])
        self._was_alive = True
        with torch.no_grad():
            turn, op, speed, _, _, self._hidden = policy.act(
                obs, self._hidden, done,
                temperature=self._temperature,
                deterministic=not self._sample,
                op_deterministic=True,
            )
        self._heading = apply_turn(self._heading, int(turn.item()))
        op = int(op.item())
        speed = float(speed.item())
        self._prev = (int(turn.item()), op, speed)

        total = sum(b.mass for b in ctx.me.blobs)
        cx = sum(b.x * b.mass for b in ctx.me.blobs) / total
        cy = sum(b.y * b.mass for b in ctx.me.blobs) / total
        tx, ty = action_to_target(self._heading, cx, cy, speed)

        action = BotAction(
            target_x=min(max(tx, 0.0), ctx.world_width),
            target_y=min(max(ty, 0.0), ctx.world_height),
            split=op == 1,
            eject=op == 2,
        )
        self._cached = action
        return action


def register(registry: BotRegistry) -> None:
    registry.register("neural", lambda init_ctx: NeuralBrain(init_ctx))

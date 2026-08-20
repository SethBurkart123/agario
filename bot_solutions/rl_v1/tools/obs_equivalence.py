"""Guard: the Rust observation encoder (CoreWorld.observe) must match the
Python reference encoder (bot_solutions/rl_v1/obs.py via ArenaEnv._encode_for_py).

Runs arenas with random actions (splits/ejects included) and compares both
encoders on every learner and frozen agent each decision step.

Usage: uv run python -m bot_solutions.rl_v1.tools.obs_equivalence [--steps 400] [--seed 7]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from bot_solutions.rl_v1.env import ArenaConfig, ArenaEnv
from bot_solutions.rl_v1.obs import N_TURNS, OBS_DIM


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()

    from agario_core import CoreWorld

    assert CoreWorld.obs_dim() == OBS_DIM, (
        f"OBS_DIM mismatch: rust={CoreWorld.obs_dim()} python={OBS_DIM} — "
        "agario_core/src/world.rs and bot_solutions/rl_v1/obs.py are out of sync"
    )

    env = ArenaEnv(ArenaConfig(), seed=args.seed)
    env.reset()
    rng = np.random.default_rng(args.seed)
    all_ids = env.learner_ids + env.frozen_ids

    rust_out = np.zeros((len(all_ids), OBS_DIM), dtype=np.float32)
    py_out = np.zeros((len(all_ids), OBS_DIM), dtype=np.float32)
    worst = 0.0

    def rand_actions(n: int) -> np.ndarray:
        acts = np.zeros((n, 3), dtype=np.float64)
        acts[:, 0] = rng.integers(0, N_TURNS, size=n)
        acts[:, 1] = rng.integers(0, 3, size=n)
        acts[:, 2] = rng.uniform(0.0, 1.0, size=n)
        return acts

    for step in range(args.steps):
        _, _, truncated, _ = env.step(
            rand_actions(len(env.learner_ids)), rand_actions(len(env.frozen_ids))
        )
        if truncated:
            env.reset()
            all_ids = env.learner_ids + env.frozen_ids

        buf = env.world.observe(all_ids, env.now, env._control_for(all_ids))
        rust_out[:] = np.frombuffer(buf, dtype=np.float32).reshape(len(all_ids), OBS_DIM)
        env._encode_for_py(all_ids, py_out)

        diff = np.abs(rust_out - py_out).max()
        worst = max(worst, float(diff))
        if diff > args.tol:
            agent, feat = np.unravel_index(np.abs(rust_out - py_out).argmax(), rust_out.shape)
            print(
                f"OBS MISMATCH at step {step}: agent {all_ids[agent]} feature {feat}: "
                f"rust={rust_out[agent, feat]!r} py={py_out[agent, feat]!r} (diff {diff:.3e})"
            )
            return 1

    print(
        f"OBS EQUIVALENCE PASS: {args.steps} steps x {len(all_ids)} agents, "
        f"max |diff| = {worst:.2e} (tol {args.tol})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

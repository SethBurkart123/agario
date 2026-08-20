"""Train the agario policy with PufferLib 3.0's PuffeRL on CUDA.

Same physics, same rewards, same encoder — PufferLib supplies vectorized
multiprocess collection and a GPU-optimized PPO (their Muon + VTrace stack).

Usage (on the GPU box):
  uv run --no-sync python -m bot_solutions.rl_v1.puffer_train --timesteps 20000000
"""

from __future__ import annotations

import argparse
import configparser
from pathlib import Path

import pufferlib
import pufferlib.vector

from .puffer_env import AgarioPufferEnv, AgarioPufferPolicy
from . import CHECKPOINT_ROOT


def base_train_config() -> dict:
    """PufferLib's own default [train] section, coerced to python types."""
    ini = Path(pufferlib.__file__).parent / "config" / "default.ini"
    cp = configparser.ConfigParser()
    cp.read(ini)
    out: dict = {}
    for key, raw in cp["train"].items():
        raw = raw.split("#")[0].strip()
        for cast in (int, float):
            try:
                out[key] = cast(raw.replace("_", ""))
                break
            except ValueError:
                continue
        else:
            if raw.lower() in ("true", "false"):
                out[key] = raw.lower() == "true"
            else:
                out[key] = raw
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=20_000_000)
    parser.add_argument("--num-envs", type=int, default=12)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--learners", type=int, default=10)
    parser.add_argument("--anchors", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=("serial", "mp"), default="mp")
    parser.add_argument(
        "--batched", action="store_true",
        help="use the Rust BatchedArenas env (rayon-parallel, Serial backend)",
    )
    parser.add_argument("--arenas", type=int, default=24)
    parser.add_argument("--bptt", type=int, default=64)
    parser.add_argument("--init-from", default=str(CHECKPOINT_ROOT / "rl/latest.pt"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.batched:
        from .batch_env import AgarioBatchEnv

        vecenv = pufferlib.vector.make(
            AgarioBatchEnv,
            backend=pufferlib.vector.Serial,
            num_envs=1,
            env_kwargs=dict(
                arenas=args.arenas, num_learners=args.learners, anchors=args.anchors
            ),
            seed=args.seed,
        )
        total_agents_override = args.arenas * args.learners
    else:
        backend = (
            pufferlib.vector.Multiprocessing if args.backend == "mp"
            else pufferlib.vector.Serial
        )
        vecenv = pufferlib.vector.make(
            AgarioPufferEnv,
            backend=backend,
            num_envs=args.num_envs,
            num_workers=args.workers if args.backend == "mp" else 1,
            env_kwargs=dict(num_learners=args.learners, anchors=args.anchors),
            seed=args.seed,
        )
        total_agents_override = args.num_envs * args.learners

    base = AgarioPufferPolicy(
        vecenv.driver_env,
        init_from=args.init_from if args.init_from and Path(args.init_from).exists() else None,
    )
    from pufferlib.models import LSTMWrapper

    policy = LSTMWrapper(vecenv.driver_env, base, input_size=256, hidden_size=256)
    policy = policy.to(args.device)

    total_agents = total_agents_override
    batch_size = total_agents * args.bptt
    config = base_train_config()
    config.update(
        env="agario",
        device=args.device,
        seed=args.seed,
        total_timesteps=args.timesteps,
        use_rnn=True,
        bptt_horizon=args.bptt,
        batch_size=batch_size,
        minibatch_size=min(8192, max(256, batch_size // 4)),
        gamma=0.995,
        gae_lambda=0.95,
        compile=False,
        torch_deterministic=False,
        data_dir="experiments",
    )

    from pufferlib import pufferl

    trainer = pufferl.PuffeRL(config, vecenv, policy)

    # Bridge PuffeRL's stats into our web dashboard format.
    import json
    import time

    metrics_path = CHECKPOINT_ROOT / "rl/metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_path.open("w")
    t_start = time.time()
    row_id = 0

    def find(logs: dict, needle: str):
        for k, v in logs.items():
            if needle in k and isinstance(v, (int, float)):
                return round(float(v), 4)
        return None

    while trainer.global_step < args.timesteps:
        trainer.evaluate()
        logs = trainer.train()
        if not logs:
            continue
        row_id += 1
        metrics_file.write(json.dumps({
            "update": row_id,
            "steps": int(trainer.global_step),
            "steps_per_sec": round(trainer.global_step / max(1e-9, time.time() - t_start), 1),
            "ep_mass": find(logs, "episode_mass"),
            "ep_kills": find(logs, "episode_kills"),
            "ep_deaths": find(logs, "episode_deaths"),
            "reward_per_step": find(logs, "reward"),
            "entropy": find(logs, "entropy"),
            "value_loss": find(logs, "value_loss") or find(logs, "v_loss"),
            "wall_seconds": round(time.time() - t_start, 1),
        }) + "\n")
        metrics_file.flush()
    path = trainer.close()
    print(f"puffer training done: {path}")


if __name__ == "__main__":
    main()

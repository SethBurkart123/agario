"""Headless simulation throughput benchmark.

Drives GameWorld + BotManager directly with simulated time (no sockets, no
sleeps) and reports how much faster than realtime the sim runs. This number
determines whether the Python sim is fast enough to serve as an RL training
environment.

Usage: python -m tools.bench_sim [--bots N] [--seconds S] [--engine rust|python]
"""

from __future__ import annotations

import argparse
import time

from agario import config
from agario.bots.manager import BotManager, parse_bot_specs
from agario.world import GameWorld


def run_benchmark(num_bots: int, sim_seconds: float, seed: int = 1337, engine: str = "python") -> None:
    if engine == "rust":
        from agario_core import CoreWorld

        world = CoreWorld(seed)
    else:
        world = GameWorld(seed=seed)
    manager = BotManager(
        world,
        enabled=True,
        plugin_modules=config.BOT_PLUGIN_MODULES,
        bot_specs=parse_bot_specs(f"solo_smart:{num_bots}"),
        seed=seed,
    )

    dt = 1.0 / config.TICK_RATE
    now = 0.0
    manager.ensure_started(now)

    total_ticks = int(sim_seconds * config.TICK_RATE)
    bot_time = 0.0
    world_time = 0.0

    start = time.perf_counter()
    for _ in range(total_ticks):
        now += dt
        t0 = time.perf_counter()
        manager.tick(dt, now)
        t1 = time.perf_counter()
        world.update(dt, now)
        t2 = time.perf_counter()
        bot_time += t1 - t0
        world_time += t2 - t1
    elapsed = time.perf_counter() - start

    ticks_per_sec = total_ticks / elapsed
    print(f"engine={engine} bots={num_bots} sim_seconds={sim_seconds:.0f} ticks={total_ticks}")
    print(f"  wall time:        {elapsed:.2f}s")
    print(f"  ticks/sec:        {ticks_per_sec:,.0f}")
    print(f"  realtime factor:  {ticks_per_sec / config.TICK_RATE:.1f}x")
    print(f"  bot think time:   {bot_time:.2f}s ({100 * bot_time / elapsed:.0f}%)")
    print(f"  world step time:  {world_time:.2f}s ({100 * world_time / elapsed:.0f}%)")
    if hasattr(world, "players_compact"):
        rows = world.players_compact()
        alive = sum(1 for row in rows if row[8])
        total = len(rows)
    else:
        alive = sum(1 for p in world.players.values() if p.blobs)
        total = len(world.players)
    print(f"  players alive:    {alive}/{total}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bots", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--engine", choices=("python", "rust"), default="python")
    args = parser.parse_args()
    run_benchmark(args.bots, args.seconds, args.seed, args.engine)


if __name__ == "__main__":
    main()

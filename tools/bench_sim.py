"""Headless simulation throughput benchmark.

Drives the Rust world and bot manager with simulated time (no sockets or
sleeps) and reports throughput.

Usage: python -m tools.bench_sim [--bots N] [--seconds S]
"""

from __future__ import annotations

import argparse
import time

from agario import config
from agario.bots.manager import BotManager, parse_bot_specs
from agario_core import CoreWorld, mechanics


def run_benchmark(num_bots: int, sim_seconds: float, seed: int = 1337) -> None:
    world = CoreWorld(seed)
    manager = BotManager(
        world,
        enabled=True,
        plugin_modules=config.BOT_PLUGIN_MODULES,
        bot_specs=parse_bot_specs(f"solo_smart:{num_bots}"),
        seed=seed,
    )

    tick_rate = mechanics()["tick_rate"]
    dt = 1.0 / tick_rate
    now = 0.0
    manager.ensure_started(now)

    total_ticks = int(sim_seconds * tick_rate)
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
    print(f"bots={num_bots} sim_seconds={sim_seconds:.0f} ticks={total_ticks}")
    print(f"  wall time:        {elapsed:.2f}s")
    print(f"  ticks/sec:        {ticks_per_sec:,.0f}")
    print(f"  realtime factor:  {ticks_per_sec / tick_rate:.1f}x")
    print(f"  bot think time:   {bot_time:.2f}s ({100 * bot_time / elapsed:.0f}%)")
    print(f"  world step time:  {world_time:.2f}s ({100 * world_time / elapsed:.0f}%)")
    rows = world.players_compact()
    alive = sum(1 for row in rows if row[8])
    total = len(rows)
    print(f"  players alive:    {alive}/{total}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bots", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    run_benchmark(args.bots, args.seconds, args.seed)


if __name__ == "__main__":
    main()

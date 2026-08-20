"""Run repeatable headless bot battles and report combat/tactic outcomes."""

from __future__ import annotations

import argparse
from collections import defaultdict

from agario import config
from agario.bots.manager import BotManager, parse_bot_specs
from agario_core import CoreWorld, mechanics


def run_battle(
    specs: str, seconds: float, seed: int, start_mass: float
) -> dict[str, dict[str, float]]:
    world = CoreWorld(seed)
    manager = BotManager(
        world,
        enabled=True,
        plugin_modules=config.BOT_PLUGIN_MODULES,
        bot_specs=parse_bot_specs(specs),
        seed=seed,
    )
    manager.ensure_started(0.0)
    if start_mass > 0:
        for pid in manager._agents:
            world.set_player_mass(pid, start_mass)
    tick_rate = mechanics()["tick_rate"]
    dt = 1.0 / tick_rate
    now = 0.0
    for _ in range(int(seconds * tick_rate)):
        now += dt
        manager.tick(dt, now)
        world.update(dt, now)

    kills = dict(world.kill_counts())
    deaths = dict(world.death_counts())
    masses = {pid: mass for pid, mass, _ in world.player_masses()}
    rows: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for pid, agent in manager._agents.items():
        brain = agent.brain
        profile = getattr(getattr(brain, "profile", None), "name", "baseline")
        key = f"{agent.plugin_name}/{profile}"
        rows[key]["players"] += 1
        rows[key]["mass"] += masses.get(pid, 0.0)
        rows[key]["kills"] += kills.get(pid, 0)
        rows[key]["deaths"] += deaths.get(pid, 0)
        for name, count in agent.memory.get("v2_stats", {}).items():
            rows[key][name] += count
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", default="solo_smart:8,solo_smart_v2:8")
    parser.add_argument("--seconds", type=float, default=180.0)
    parser.add_argument("--start-mass", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337, 2027, 4099])
    args = parser.parse_args()

    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for seed in args.seeds:
        rows = run_battle(args.specs, args.seconds, seed, args.start_mass)
        print(f"seed {seed}")
        for name, stats in sorted(rows.items()):
            players = max(1.0, stats["players"])
            print(
                f"  {name:28} mass {stats['mass'] / players:7.1f}  "
                f"kills {stats['kills'] / players:5.2f}  deaths {stats['deaths'] / players:5.2f}"
            )
            for key, value in stats.items():
                totals[name][key] += value

    print("aggregate")
    for name, stats in sorted(totals.items()):
        players = max(1.0, stats["players"])
        tactics = ", ".join(
            f"{key}={int(value)}"
            for key, value in sorted(stats.items())
            if key not in {"players", "mass", "kills", "deaths"} and value
        )
        print(
            f"  {name:28} mass {stats['mass'] / players:7.1f}  "
            f"kills {stats['kills'] / players:5.2f}  deaths {stats['deaths'] / players:5.2f}"
            + (f"  [{tactics}]" if tactics else "")
        )


if __name__ == "__main__":
    config.BOT_SPAWN_ON_EATEN = False
    main()

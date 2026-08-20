"""Parity check: Rust core (agario_core.CoreWorld) vs Python GameWorld.

Both worlds are seeded identically and driven with identical scripted inputs
that exercise movement, splitting, ejecting, eating, merging, virus pops, and
mass decay. State is compared tick-by-tick; any drift beyond float tolerance
fails the run.

Usage: uv run python -m tools.parity_check [--ticks 3000] [--players 8] [--seed 1337]
"""

from __future__ import annotations

import argparse
import math
import sys

from agario import config
from agario.world import GameWorld
from agario_core import CoreWorld


def python_state(world: GameWorld) -> dict:
    return {
        "players": [
            {
                "id": p.id,
                "name": p.name,
                "target": (p.target_x, p.target_y),
                "last_split_at": p.last_split_at,
                "last_eject_at": p.last_eject_at,
                "blobs": [
                    (b.id, b.x, b.y, b.mass, b.vx, b.vy, b.can_merge_at)
                    for b in p.blobs.values()
                ],
            }
            for p in world.players.values()
        ],
        "foods": [(f.id, f.x, f.y, f.mass, f.color) for f in world.foods.values()],
        "ejected": [
            (e.id, e.x, e.y, e.mass, e.owner_id, e.vx, e.vy, e.ttl)
            for e in world.ejected.values()
        ],
        "viruses": [(v.id, v.x, v.y, v.mass) for v in world.viruses.values()],
    }


def close(a: float, b: float, tol: float) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def compare_rows(kind: str, py_rows: list, rs_rows: list, tick: int, tol: float) -> list[str]:
    errors: list[str] = []
    if len(py_rows) != len(rs_rows):
        return [f"tick {tick}: {kind} count differs: py={len(py_rows)} rs={len(rs_rows)}"]
    for py_row, rs_row in zip(py_rows, rs_rows):
        for i, (pv, rv) in enumerate(zip(py_row, rs_row)):
            if isinstance(pv, float):
                if not close(pv, rv, tol):
                    errors.append(
                        f"tick {tick}: {kind} {py_row[0]} field {i}: py={pv!r} rs={rv!r}"
                    )
            elif pv != rv:
                errors.append(f"tick {tick}: {kind} {py_row[0]} field {i}: py={pv!r} rs={rv!r}")
            if len(errors) >= 5:
                return errors
    return errors


def compare(py: dict, rs: dict, tick: int, tol: float) -> list[str]:
    errors: list[str] = []
    if len(py["players"]) != len(rs["players"]):
        return [f"tick {tick}: player count differs"]
    for pp, rp in zip(py["players"], rs["players"]):
        if pp["id"] != rp["id"] or pp["name"] != rp["name"]:
            errors.append(f"tick {tick}: player identity {pp['id']} vs {rp['id']}")
        for label in ("last_split_at", "last_eject_at"):
            if not close(pp[label], rp[label], tol):
                errors.append(f"tick {tick}: {pp['id']} {label} py={pp[label]} rs={rp[label]}")
        if not all(close(a, b, tol) for a, b in zip(pp["target"], rp["target"])):
            errors.append(f"tick {tick}: {pp['id']} target differs")
        errors += compare_rows(f"blob[{pp['id']}]", pp["blobs"], rp["blobs"], tick, tol)
        if len(errors) >= 5:
            return errors
    errors += compare_rows("food", py["foods"], rs["foods"], tick, tol)
    errors += compare_rows("ejected", py["ejected"], rs["ejected"], tick, tol)
    errors += compare_rows("virus", py["viruses"], rs["viruses"], tick, tol)
    return errors


def scripted_inputs(tick: int, player_ids: list[str]) -> list[tuple[str, float, float, bool, bool]]:
    """Deterministic pseudo-random-ish inputs that exercise all mechanics."""
    inputs = []
    for j, pid in enumerate(player_ids):
        tx = ((tick * (37 + j * 13)) % 6037) % config.WORLD_WIDTH
        ty = ((tick * (53 + j * 7)) % 5921) % config.WORLD_HEIGHT
        split = tick % 400 == (j * 50) % 400 and tick > 0
        eject = tick % 90 == (j * 11) % 90 and tick > 0
        inputs.append((pid, tx, ty, split, eject))
    return inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=3000)
    parser.add_argument("--players", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--tol", type=float, default=1e-9)
    args = parser.parse_args()

    py_world = GameWorld(seed=args.seed)
    rs_world = CoreWorld(seed=args.seed)

    py_ids = []
    for i in range(args.players):
        p = py_world.add_player(f"P{i}", 0.0, is_bot=True)
        r = rs_world.add_player(f"P{i}", 0.0, is_bot=True)
        assert p.id == r.id, f"player id mismatch: {p.id} vs {r.id}"
        py_ids.append(p.id)

    dt = 1.0 / config.TICK_RATE
    now = 0.0
    first_divergence = None

    for tick in range(args.ticks):
        now += dt
        for pid, tx, ty, split, eject in scripted_inputs(tick, py_ids):
            py_world.set_input(pid, tx, ty, split=split, eject=eject)
            rs_world.set_input(pid, tx, ty, split=split, eject=eject)
        py_world.update(dt, now)
        rs_world.update(dt, now)

        errors = compare(python_state(py_world), rs_world.debug_state(), tick, args.tol)
        if errors:
            first_divergence = tick
            print(f"PARITY FAIL at tick {tick}:")
            for e in errors:
                print(f"  {e}")
            break

    if first_divergence is None:
        n_blobs = sum(len(p.blobs) for p in py_world.players.values())
        print(
            f"PARITY PASS: {args.ticks} ticks, {args.players} players, seed={args.seed} "
            f"(final: {n_blobs} blobs, {len(py_world.foods)} foods, "
            f"{len(py_world.ejected)} ejected, {len(py_world.viruses)} viruses)"
        )
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""Rebuild and follow the RL metrics stream from a trainer's stdout log.

Recovery tool for when the metrics file has been clobbered while a training
run holds the old (deleted) file handle: parses the trainer's printed lines
back into metric rows, backfills the file, then follows the log until the
trainer exits.

Usage: uv run python -m bot_solutions.rl_v1.tools.metrics_bridge /tmp/agario_train7.log
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

from .. import CHECKPOINT_ROOT

OUT = CHECKPOINT_ROOT / "rl/metrics.jsonl"
PAT = re.compile(
    r"update\s+(\d+) \| steps\s+([\d,]+) \|\s+([\d,]+) steps/s \| "
    r"reward/step ([+-][\d.]+) \| ep_mass\s+([\d.]+|nan) \| ep_kills\s+([\d.]+|nan) \| "
    r"ep_deaths\s+([\d.]+|nan) \| pg ([+-][\d.]+) \| v ([\d.]+) \| ent ([\d.]+)"
)


def parse(line: str) -> dict | None:
    m = PAT.search(line)
    if m is None:
        return None
    opt = lambda s: None if s == "nan" else float(s)
    steps = int(m.group(2).replace(",", ""))
    sps = float(m.group(3).replace(",", ""))
    return {
        "update": int(m.group(1)),
        "steps": steps,
        "steps_per_sec": sps,
        "reward_per_step": float(m.group(4)),
        "ep_mass": opt(m.group(5)),
        "ep_deaths": opt(m.group(7)),
        "ep_kills": opt(m.group(6)),
        "pg_loss": float(m.group(8)),
        "value_loss": float(m.group(9)),
        "entropy": float(m.group(10)),
        "wall_seconds": round(steps / sps, 1) if sps > 0 else 0.0,
    }


def trainer_alive() -> bool:
    return subprocess.run(
        ["pgrep", "-f", "bot_solutions.rl_v1.ppo"], capture_output=True
    ).returncode == 0


def main() -> None:
    log_path = Path(sys.argv[1])
    seen: set[int] = set()
    with log_path.open() as log, OUT.open("w") as out:
        # Backfill, then follow.
        while True:
            line = log.readline()
            if line:
                row = parse(line)
                if row is not None and row["update"] not in seen:
                    seen.add(row["update"])
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                continue
            if not trainer_alive():
                break
            time.sleep(2.0)
    print(f"bridge done: {len(seen)} updates written to {OUT}")


if __name__ == "__main__":
    main()

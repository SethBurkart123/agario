"""Marathon training: loop [ExIt generation -> KL-leashed PPO -> eval]
cycles until a time budget runs out.

Resumable by design: progress is recorded in this solution's checkpoints.
after every completed stage, so killing the process (Ctrl-C, reboot, anything)
loses at most the stage in flight — rerun the same command and it continues.
The RL checkpoint directory's latest.pt always holds the newest playable policy, so the live
game can be tested at any point mid-run (restart the game server to reload).

Per-cycle stages:
  exit: search-driven play (Expert Iteration) distilled into the net
  ppo:  KL-leashed PPO burst — sharpens play AND re-fits the value head the
        next cycle's search depends on (distillation alone never trains it)
  eval: vs solo_smart; result appended to the marathon history

Usage:
  uv run python -m bot_solutions.rl_v1.marathon --hours 8
  (interrupt any time; same command resumes)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import CHECKPOINT_ROOT

MARATHON_DIR = CHECKPOINT_ROOT / "marathon"
RL_DIR = CHECKPOINT_ROOT / "rl"
LATEST = RL_DIR / "latest.pt"
STATE = MARATHON_DIR / "state.json"
STATUS = MARATHON_DIR / "status.json"
HISTORY = MARATHON_DIR / "history.jsonl"
# Pure AlphaZero-style cycle: one improvement operator. Search-played games
# provide policy targets AND value targets (returns), so distillation covers
# both heads and no separate RL phase is needed. stage_ppo remains available
# as a manual tool but is out of the loop.
STAGES = ("exit", "eval")


def load_state() -> dict:
    if STATE.exists():
        state = json.loads(STATE.read_text())
        for key in ("current", "best_ckpt"):
            old = Path(state.get(key, ""))
            if old.parts[:1] == ("checkpoints",):
                state[key] = str(CHECKPOINT_ROOT.joinpath(*old.parts[1:]))
        # Migration: older state files used a 3-stage scheme; clamp so a
        # resume can never index past the current STAGES tuple.
        state["stage_idx"] = min(state.get("stage_idx", 0), len(STAGES) - 1)
        return state
    return {"cycle": 1, "stage_idx": 0, "current": str(LATEST)}


def atomic_copy(src, dst: Path) -> None:
    tmp = Path(dst).with_suffix(".tmp")
    shutil.copy(src, tmp)
    os.replace(tmp, dst)


def save_state(state: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, STATE)


def write_status(state: dict, stage: str, stage_started_at: float,
                 hours_remaining: float) -> None:
    """Atomically refresh status.json — a lightweight, dashboard-facing view of
    the run, updated at every stage transition. Best-effort; never fatal."""
    status = {
        "cycle": state["cycle"],
        "stage": stage,
        "stage_started_at": stage_started_at,
        "current": state.get("current"),
        "best_score": state.get("best_score"),
        "best_ckpt": state.get("best_ckpt"),
        "hours_remaining": hours_remaining,
    }
    try:
        tmp = STATUS.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2))
        os.replace(tmp, STATUS)
    except OSError:
        pass


def write_phase_marker(phase: str, detail: str | None = None) -> None:
    """Name the current phase for the dashboard heartbeat. Stages that can't
    update sub-minute (eval) at least get a fresh timestamp at launch so the
    dashboard can show what is running instead of a stale generate beat."""
    try:
        tmp = MARATHON_DIR / "phase_progress.json.tmp"
        tmp.write_text(json.dumps({
            "phase": phase,
            "fraction": None,
            "eta_minutes": None,
            "detail": detail,
            "updated_at": time.time(),
        }))
        os.replace(tmp, MARATHON_DIR / "phase_progress.json")
    except OSError:
        pass


def run(cmd: list[str]) -> str:
    print(f"$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"stage failed (exit {proc.returncode})")
    return proc.stdout


def prune_league_pool(keep: int = 60) -> None:
    pool = sorted(RL_DIR.glob("update_*.pt"))
    anchors = [p for p in pool if p.name == "update_000000.pt"]
    rest = [p for p in pool if p.name != "update_000000.pt"]
    for stale in rest[: max(0, len(rest) - keep)]:
        stale.unlink(missing_ok=True)
    _ = anchors  # the BC/style anchor is always kept


def stage_exit(state: dict, args) -> str:
    """AlphaZero generation: gate -> search-generate -> distill (azero run).
    If the gate fails, azero refuses to distill and we keep the current
    checkpoint — a cycle is skipped, never poisoned."""
    cycle = state["cycle"]
    azero_out = RL_DIR / "azero_gen.pt"
    azero_out.unlink(missing_ok=True)
    run([
        sys.executable, "-m", "bot_solutions.rl_v1.azero", "run",
        "--start", state["current"],
        "--labels", str(args.exit_labels),
        "--out", str(azero_out),
        "--seed", str(100 + cycle),
        "--cycle", str(cycle),
    ])
    if not azero_out.exists():
        print("azero gate failed — skipping distill this cycle", flush=True)
        return state["current"]
    out = MARATHON_DIR / f"cycle{cycle:03d}_azero.pt"
    atomic_copy(azero_out, out)
    atomic_copy(out, LATEST)
    return str(out)


def stage_ppo(state: dict, args) -> str:
    cycle = state["cycle"]
    run([
        sys.executable, "-m", "bot_solutions.rl_v1.ppo",
        "--resume", state["current"],
        "--kl-ref", state["current"],
        "--kl-coef", str(args.kl_coef),
        "--updates", str(args.ppo_updates),
        # Asymmetric world: solo_smart anchors as soft prey + spawn-mass
        # jitter, so hunting/splitting stays profitable. A pure self-twin
        # league makes caution optimal and bleeds the dynamic behaviors out
        # (observed three times before this was added).
        "--arenas", "8", "--minibatches", "2", "--anchors", "2",
        "--spawn-mass-jitter", "1.2", "--kl-op-mult", "4.0",
        "--lr", "1e-4", "--entropy-coef", "0.002", "--entropy-final", "0.001",
        "--seed", str(1000 + cycle),
    ])
    out = MARATHON_DIR / f"cycle{cycle:03d}_ppo.pt"
    shutil.copy(LATEST, out)
    prune_league_pool()
    return str(out)


def stage_eval(state: dict, args) -> str:
    cycle = state["cycle"]
    write_phase_marker("eval", "14 episodes vs solo_smart")
    stdout = run([
        sys.executable, "-m", "bot_solutions.rl_v1.eval",
        "--checkpoint", state["current"],
        # 14 episodes: still noisy, but cycle-over-cycle deltas become
        # meaningful instead of coin flips. Seeds are fixed, so successive
        # cycles are compared on the same arena draws.
        "--episodes", "14", "--temperature", "0.6",
    ])
    # Anchored to the single summary line: a sloppy multi-line match once
    # captured a per-episode mass instead, silently corrupting the ratchet.
    m = re.search(
        r"summary over \d+ episodes: policy mass ([\d.]+) ± [\d.]+ \(sem\) \| "
        r"heuristic mass [\d.]+ \| kills/ep ([\d.]+) \| deaths/ep ([\d.]+) \| "
        r"win rate (\d+)%",
        stdout,
    )
    split_m = re.search(r"split ([\d.]+)%", stdout)
    if m is None:
        # A parse failure must NOT look like a policy collapse (score 0 would
        # trigger a revert every cycle, forever).
        raise RuntimeError("eval output unparseable — eval format changed?")
    row = {
        "cycle": cycle,
        "checkpoint": state["current"],
        "mass": float(m.group(1)) if m else None,
        "kills": float(m.group(2)) if m else None,
        "deaths": float(m.group(3)) if m else None,
        "win_rate": int(m.group(4)) if m else None,
        "split_usage_pct": float(split_m.group(1)) if split_m else None,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with HISTORY.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"cycle {cycle} eval -> {row}", flush=True)

    # Ratchet: track the best checkpoint by eval score; if a cycle regresses
    # far below it (beyond eval noise), restart the next cycle from the best
    # instead of compounding the regression. The style factor keeps a
    # split-less (eroded) policy from being crowned best on raw mass alone.
    split_pct = row["split_usage_pct"] or 0.0
    style = min(1.0, 0.7 + split_pct / 3.0)
    score = (row["mass"] or 0.0) * (0.5 + (row["win_rate"] or 0) / 100.0) * style
    best_pt = MARATHON_DIR / "best.pt"
    if score > state.get("best_score", 0.0):
        state["best_score"] = score
        state["best_ckpt"] = state["current"]
        atomic_copy(state["current"], best_pt)
        print(f"new best (score {score:.0f}): {state['current']}", flush=True)
    elif score < 0.6 * state.get("best_score", 0.0) and best_pt.exists():
        print(
            f"regression (score {score:.0f} < 60% of best "
            f"{state['best_score']:.0f}) — reverting to {best_pt}",
            flush=True,
        )
        atomic_copy(best_pt, LATEST)
        return str(best_pt)
    return state["current"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=8.0)
    # Row budget per generation; scaled with the 30-player lobbies (B=416
    # rows/decision) so a generation is still ~512 searched decisions.
    parser.add_argument("--exit-labels", type=int, default=220_000)
    parser.add_argument("--ppo-updates", type=int, default=800)
    parser.add_argument("--kl-coef", type=float, default=0.15)
    args = parser.parse_args()

    MARATHON_DIR.mkdir(parents=True, exist_ok=True)

    # One marathon at a time: two would interleave on state.json, race on
    # latest.pt, and rmtree each other's live spools.
    pid_path = MARATHON_DIR / "marathon.pid"
    if pid_path.exists():
        try:
            other = int(pid_path.read_text().strip())
            os.kill(other, 0)
            raise SystemExit(f"another marathon (pid {other}) is already running")
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    pid_path.write_text(str(os.getpid()))

    deadline = time.monotonic() + args.hours * 3600
    state = load_state()
    if not Path(state["current"]).exists():
        raise SystemExit(
            f"starting checkpoint {state['current']} does not exist — "
            "train one first (bot_solutions.rl_v1.bc / bot_solutions.rl_v1.ppo) or fix state.json"
        )
    print(
        f"marathon: starting at cycle {state['cycle']} stage "
        f"{STAGES[state['stage_idx']]} from {state['current']} "
        f"({args.hours:.1f}h budget)",
        flush=True,
    )

    runners = {"exit": stage_exit, "ppo": stage_ppo, "eval": stage_eval}
    write_status(state, STAGES[state["stage_idx"]], time.time(),
                 (deadline - time.monotonic()) / 3600)
    while time.monotonic() < deadline:
        stage = STAGES[state["stage_idx"]]
        stage_started_at = time.time()
        write_status(state, stage, stage_started_at,
                     (deadline - time.monotonic()) / 3600)
        print(
            f"\n=== marathon cycle {state['cycle']} | stage {stage} | "
            f"{(deadline - time.monotonic()) / 3600:.1f}h remaining ===",
            flush=True,
        )
        state["current"] = runners[stage](state, args)
        state["stage_idx"] += 1
        if state["stage_idx"] >= len(STAGES):
            state["stage_idx"] = 0
            state["cycle"] += 1
        save_state(state)
        # Reflect the completed transition (new cycle/stage_idx, updated best).
        write_status(state, STAGES[state["stage_idx"]], stage_started_at,
                     (deadline - time.monotonic()) / 3600)

    print(f"marathon: time budget reached at cycle {state['cycle']}", flush=True)


if __name__ == "__main__":
    main()

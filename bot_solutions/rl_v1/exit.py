"""Expert Iteration: the policy plays WITH rollout search; the search's
chosen moves become imitation labels; the existing distillation trainer
absorbs them; the stronger net makes the next generation's search stronger.

AlphaZero's improvement operator, run in alternating phases at laptop scale.
Each generation:
  1. generate: search-driven play in arenas, labels -> spool chunks
  2. distill:  bot_solutions.rl_v1.imitate trainer consumes the spool
  3. eval:     vs solo_smart, printed per generation

Usage:
  uv run python -m bot_solutions.rl_v1.exit --generations 2 --labels-per-gen 40000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .env import ArenaConfig, ArenaEnv
from . import CHECKPOINT_ROOT
from .model import PolicyNet
from .obs import OBS_DIM
from .search import RolloutSearcher

CHECKPOINT_DIR = CHECKPOINT_ROOT / "rl"


def load_policy(path: Path) -> PolicyNet:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    policy = PolicyNet(**ckpt.get("model_kwargs", {}))
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    return policy


def generate(policy: PolicyNet, spool: Path, args) -> None:
    """Search-driven play; every decision's search choice is a label."""
    import shutil

    # Fresh spool: stale chunks from a previous policy (with value targets
    # bootstrapped by an old value head) must not leak into this generation.
    shutil.rmtree(spool, ignore_errors=True)
    spool.mkdir(parents=True, exist_ok=True)
    searcher = RolloutSearcher(policy, horizon=args.horizon)
    cfg = ArenaConfig(n_learners=args.agents, n_frozen=0, n_heuristic=1)
    envs = [ArenaEnv(cfg, seed=args.seed + 31 * i) for i in range(args.arenas)]
    B = args.arenas * args.agents
    T = args.chunk

    obs = np.concatenate([e.reset() for e in envs], axis=0)
    done = np.ones(B, dtype=np.float32)
    hiddens = policy.initial_state(B)
    target_chunks = max(1, args.labels_per_gen // (T * B))
    t_start = time.perf_counter()

    GAMMA = 0.995
    for chunk_idx in range(target_chunks):
        c_obs = np.zeros((T, B, OBS_DIM), dtype=np.float32)
        c_turn = np.zeros((T, B), dtype=np.int64)
        c_op = np.zeros((T, B), dtype=np.int64)
        c_speed = np.zeros((T, B), dtype=np.float32)
        c_rew = np.zeros((T, B), dtype=np.float32)
        c_done = np.zeros((T, B), dtype=np.float32)
        # Boot values for time-limit truncations: episodes end by truncation,
        # not termination, so returns must bootstrap V(final state) rather
        # than cut to zero (else late-episode values are systematically
        # deflated and the search misprices everything near the time limit).
        c_boot = np.zeros((T, B), dtype=np.float32)
        c_h0 = hiddens.numpy().copy()  # chunk-start hidden states for training
        for t in range(T):
            c_obs[t] = obs
            c_done[t] = done

            # Advance root hidden states (h depends on obs only, so this is
            # valid regardless of which action gets executed). The speed
            # output doubles as the searcher's root speed — computing it here
            # avoids consuming the root obs twice.
            with torch.no_grad():
                _, _, root_speed, _, _, hiddens = policy.act(
                    torch.as_tensor(obs), hiddens, torch.as_tensor(done),
                    deterministic=True, op_deterministic=True,
                )
            root_speed = root_speed.numpy()

            done = np.zeros(B, dtype=np.float32)
            parts = []
            for i, env in enumerate(envs):
                lo = i * args.agents
                acts = searcher.plan(
                    env, env.learner_ids,
                    hiddens[lo : lo + args.agents],
                    root_speed[lo : lo + args.agents],
                )
                c_turn[t, lo : lo + args.agents] = acts[:, 0].astype(np.int64)
                c_op[t, lo : lo + args.agents] = acts[:, 1].astype(np.int64)
                c_speed[t, lo : lo + args.agents] = acts[:, 2].astype(np.float32)
                step_obs, rewards, truncated, _ = env.step(acts)
                c_rew[t, lo : lo + args.agents] = rewards
                if truncated:
                    with torch.no_grad():
                        _, _, _, _, boot, _ = policy.act(
                            torch.as_tensor(step_obs),
                            hiddens[lo : lo + args.agents],
                            torch.zeros(args.agents),
                            deterministic=True, op_deterministic=True,
                        )
                    c_boot[t, lo : lo + args.agents] = boot.numpy()
                    step_obs = env.reset()
                    done[lo : lo + args.agents] = 1.0
                parts.append(step_obs.copy())
            obs = np.concatenate(parts, axis=0)

        # Value targets (AlphaZero's z, generalized): discounted return-to-go
        # of the search-played game; chunk boundary and time-limit boundaries
        # both bootstrap with the current value head.
        with torch.no_grad():
            _, _, _, _, boot_v, _ = policy.act(
                torch.as_tensor(obs), hiddens, torch.as_tensor(done),
                deterministic=True, op_deterministic=True,
            )
        c_ret = np.zeros((T, B), dtype=np.float32)
        carry = np.where(done > 0.5, c_boot[T - 1], boot_v.numpy())
        for t in reversed(range(T)):
            carry = c_rew[t] + GAMMA * carry
            c_ret[t] = carry
            if t > 0:
                # done[t] marks obs[t] fresh: the step before it ended at the
                # time limit, so its return continues into V(pre-reset state).
                trunc = c_done[t] > 0.5
                carry = np.where(trunc, c_boot[t - 1], carry)

        import os

        tmp = spool / f"tmp_{chunk_idx:06d}.npz"
        np.savez(tmp, obs=c_obs, turn=c_turn, op=c_op, speed=c_speed, done=c_done,
                 ret=c_ret, h0=c_h0, expert_driven=np.array(True))
        os.replace(tmp, spool / f"chunk_{chunk_idx:06d}.npz")
        done_labels = (chunk_idx + 1) * T * B
        rate = done_labels / (time.perf_counter() - t_start)
        print(
            f"  gen-chunk {chunk_idx + 1}/{target_chunks} | {rate:,.0f} search-labels/s",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--labels-per-gen", type=int, default=40_000)
    parser.add_argument("--distill-labels", type=int, default=400_000,
                        help="gradient labels per distillation (re-passes the gen pool)")
    parser.add_argument("--arenas", type=int, default=2)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--start", default=str(CHECKPOINT_DIR / "latest.pt"))
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()

    current = Path(args.start)
    for gen in range(1, args.generations + 1):
        print(f"=== ExIt generation {gen}: search-driven play from {current} ===", flush=True)
        policy = load_policy(current)
        spool = CHECKPOINT_ROOT / f"exit_gen{gen}"
        generate(policy, spool, args)

        out = CHECKPOINT_DIR / f"exit_gen{gen}.pt"
        print(f"=== ExIt generation {gen}: distilling into {out} ===", flush=True)
        n_chunks = len(list(spool.glob("chunk_*.npz")))
        subprocess.run(
            [
                sys.executable, "-m", "bot_solutions.rl_v1.imitate", "trainer",
                "--spool", str(spool),
                "--labels", str(args.distill_labels),
                "--window", str(n_chunks),
                "--min-chunks", str(min(4, n_chunks)),
                "--resume", str(current),
                "--lr", "3e-4",
                "--out", str(out),
            ],
            check=True,
        )
        current = out

        print(f"=== ExIt generation {gen}: eval ===", flush=True)
        subprocess.run(
            [
                sys.executable, "-m", "bot_solutions.rl_v1.eval",
                "--checkpoint", str(out), "--episodes", "6", "--temperature", "0.6",
            ],
            check=True,
        )
    print("=== ExIt complete ===", flush=True)


if __name__ == "__main__":
    main()

"""Evaluate a trained policy against solo_smart heuristic bots in the arena.

Usage:
  uv run python -m bot_solutions.rl_v1.eval --episodes 8
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from .env import ArenaConfig, ArenaEnv
from . import CHECKPOINT_ROOT
from .model import PolicyNet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_ROOT / "rl/latest.pt"))
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--policy-agents", type=int, default=4)
    parser.add_argument("--heuristic-agents", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    policy = PolicyNet(**ckpt.get("model_kwargs", {}))
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    print(f"loaded {args.checkpoint} (update {ckpt.get('update', '?')})")

    cfg = ArenaConfig(
        n_learners=args.policy_agents, n_frozen=0, n_heuristic=args.heuristic_agents
    )
    env = ArenaEnv(cfg, seed=args.seed)

    policy_masses, heuristic_masses, policy_deaths, policy_kills, wins = [], [], [], [], 0
    op_hist = np.zeros(3, dtype=np.int64)
    for episode in range(args.episodes):
        obs = env.reset()
        truncated = False
        h = policy.initial_state(args.policy_agents)
        done = torch.ones(args.policy_agents)
        while not truncated:
            with torch.no_grad():
                direction, op, speed, _, _, h = policy.act(
                    torch.as_tensor(obs), h, done,
                    temperature=args.temperature, op_deterministic=True,
                )
            done = torch.zeros(args.policy_agents)
            for op_class in range(3):
                op_hist[op_class] += int((op == op_class).sum())
            actions = torch.stack([direction.float(), op.float(), speed], dim=-1).numpy()
            obs, _, truncated, info = env.step(actions)

        masses = {pid: m for pid, m, _ in env.world.player_masses()}
        p_masses = [masses.get(pid, 0.0) for pid in env.learner_ids]
        h_masses = [masses.get(pid, 0.0) for pid in env.heuristic_ids]
        deaths = dict(env.world.death_counts())
        kills = dict(env.world.kill_counts())
        p_deaths = [deaths.get(pid, 0) for pid in env.learner_ids]
        p_kills = [kills.get(pid, 0) for pid in env.learner_ids]

        policy_masses.extend(p_masses)
        heuristic_masses.extend(h_masses)
        policy_deaths.extend(p_deaths)
        policy_kills.extend(p_kills)
        won = max(p_masses) > (max(h_masses) if h_masses else 0.0)
        wins += int(won)
        print(
            f"ep {episode}: policy mass {np.mean(p_masses):7.1f} (max {max(p_masses):7.1f}) | "
            f"heuristic mass {np.mean(h_masses):7.1f} | kills {np.mean(p_kills):.2f} | "
            f"deaths {np.mean(p_deaths):.2f} | {'WIN' if won else 'loss'}"
        )

    mass_mean = np.mean(policy_masses)
    # Standard error of the mean — the honest uncertainty on the headline
    # number. Episode masses are wildly dispersed (snowball runs), so point
    # estimates without this are misleading.
    mass_sem = np.std(policy_masses) / max(1.0, np.sqrt(len(policy_masses)))
    win_p = wins / args.episodes
    win_sem = np.sqrt(max(win_p * (1 - win_p), 1e-9) / args.episodes)
    print(
        f"\nsummary over {args.episodes} episodes: "
        f"policy mass {mass_mean:.1f} ± {mass_sem:.1f} (sem) | "
        f"heuristic mass {np.mean(heuristic_masses):.1f} | "
        f"kills/ep {np.mean(policy_kills):.2f} | deaths/ep {np.mean(policy_deaths):.2f} | "
        f"win rate {win_p:.0%} ± {100 * win_sem:.0f}%"
    )
    total_ops = op_hist.sum()
    print(
        f"action usage: none {op_hist[0]/total_ops:.1%}, "
        f"split {op_hist[1]/total_ops:.2%}, eject {op_hist[2]/total_ops:.2%} "
        f"(solo_smart expert baseline: split ~0.6%)"
    )


if __name__ == "__main__":
    main()

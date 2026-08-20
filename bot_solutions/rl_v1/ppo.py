"""Recurrent PPO self-play training for agario bots.

All learner agents in all arenas share one recurrent policy (parameter
sharing self-play). Each arena also contains:
  - frozen league opponents (AlphaGo-style): a snapshot sampled from past
    checkpoints, refreshed every --opponent-refresh updates, so the policy
    must keep beating its ancestors instead of cycling against its mirror
  - solo_smart heuristic anchors grounding the population

The policy carries a GRU memory; training replays full rollout sequences per
minibatch (hidden states recomputed with stored initial states, reset at
episode boundaries).

Usage:
  uv run python -m bot_solutions.rl_v1.ppo --updates 1000 --arenas 4
  uv run python -m bot_solutions.rl_v1.ppo --resume bot_solutions/rl_v1/checkpoints/rl/latest.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .env import ArenaConfig, VecArena
from . import CHECKPOINT_ROOT
from .model import PolicyNet
from .obs import OBS_DIM

CHECKPOINT_DIR = CHECKPOINT_ROOT / "rl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arenas", type=int, default=4)
    parser.add_argument("--learners", type=int, default=5)
    parser.add_argument("--frozen", type=int, default=2, help="league opponent slots per arena")
    parser.add_argument("--anchors", type=int, default=1)
    parser.add_argument(
        "--opponent-refresh", type=int, default=10,
        help="resample the frozen league opponent every N updates",
    )
    parser.add_argument("--kill-bonus", type=float, default=1.0)
    parser.add_argument("--death-penalty", type=float, default=2.0)
    parser.add_argument(
        "--spawn-mass-jitter", type=float, default=0.0,
        help=">0 spawns agents at 560 x U(1/(1+j), 1+j): predator/prey "
        "asymmetry that keeps splits/hunting profitable in self-play",
    )
    parser.add_argument(
        "--kl-op-mult", type=float, default=3.0,
        help="extra KL weight on the split/eject head — op drift is "
        "numerically tiny averaged over mostly-none states, so style "
        "erosion needs targeted protection",
    )
    parser.add_argument("--rollout", type=int, default=128, help="decisions per rollout")
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument(
        "--entropy-final", type=float, default=0.002,
        help="entropy coef anneals linearly to this by the end of the run",
    )
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument(
        "--minibatches", type=int, default=4,
        help="minibatches per epoch, split across agents (sequences stay whole)",
    )
    parser.add_argument(
        "--kl-ref", default=None,
        help="reference checkpoint for a KL leash (e.g. the BC clone): the "
        "policy is penalized for drifting from its action distribution, so "
        "RL sharpens play without destroying the imitated style",
    )
    parser.add_argument("--kl-coef", type=float, default=0.25)
    parser.add_argument("--rnn", choices=("gru", "mingru"), default="gru")
    parser.add_argument("--optimizer", choices=("adam", "muon"), default="adam")
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument(
        "--prio-replay", action="store_true",
        help="sample minibatch sequences with replacement, weighted by |advantage|",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--checkpoint-dir", default=str(CHECKPOINT_DIR),
        help="checkpoints + metrics destination (also the league opponent pool)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    spawn_range = None
    if args.spawn_mass_jitter > 0:
        spawn_range = (1.0 / (1.0 + args.spawn_mass_jitter), 1.0 + args.spawn_mass_jitter)
    cfg = ArenaConfig(
        n_learners=args.learners,
        n_frozen=args.frozen,
        n_heuristic=args.anchors,
        kill_bonus=args.kill_bonus,
        death_penalty=args.death_penalty,
        spawn_mass_range=spawn_range,
    )
    vec = VecArena(cfg, n_arenas=args.arenas, seed=args.seed)
    n_agents = vec.n_agents

    policy = PolicyNet(rnn_type=args.rnn).to(device)
    if args.optimizer == "muon":
        from .muon import HybridOptimizer

        optimizer = HybridOptimizer(policy, muon_lr=args.muon_lr, adam_lr=args.lr)
    else:
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    start_update = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        policy.load_state_dict(ckpt["policy"])
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                print("optimizer state incompatible (different --optimizer?); starting fresh")
        start_update = ckpt.get("update", 0)
        print(f"resumed from {args.resume} at update {start_update}")

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Refuse to share a checkpoint dir with a live trainer — a second run
    # would clobber metrics.jsonl and pollute the league pool.
    import os
    import signal

    pid_path = ckpt_dir / "trainer.pid"
    if pid_path.exists():
        try:
            other = int(pid_path.read_text().strip())
            os.kill(other, 0)  # signal 0 = existence check
            raise SystemExit(
                f"another trainer (pid {other}) is already using {ckpt_dir}; "
                "pass a different --checkpoint-dir"
            )
        except (ValueError, ProcessLookupError, PermissionError):
            pass  # stale lockfile
    pid_path.write_text(str(os.getpid()))

    metrics_path = ckpt_dir / "metrics.jsonl"
    if start_update == 0 and metrics_path.exists():
        metrics_path.unlink()
    metrics_file = metrics_path.open("a")

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project="agario-rl",
            name=args.run_name,
            config=vars(args),
            resume="allow",
        )

    reference = None
    if args.kl_ref:
        ref_ckpt = torch.load(args.kl_ref, map_location=device, weights_only=True)
        reference = PolicyNet(**ref_ckpt.get("model_kwargs", {})).to(device)
        reference.load_state_dict(ref_ckpt["policy"])
        reference.eval()
        for p in reference.parameters():
            p.requires_grad_(False)
        print(f"KL leash to {args.kl_ref} (coef {args.kl_coef})")

    # League opponent: a frozen recurrent snapshot with its own memory.
    opponent = PolicyNet(rnn_type=args.rnn).to(device)
    opponent.load_state_dict(policy.state_dict())
    opponent.eval()
    league_rng = np.random.default_rng(args.seed)
    n_frozen_total = args.arenas * args.frozen
    opp_h = opponent.initial_state(n_frozen_total, device=device)

    def refresh_opponent() -> str:
        pool = sorted(ckpt_dir.glob("update_*.pt"))
        # Recency-biased league sampling: half the time play the newest
        # quartile (stay sharp vs current strength), half uniform over
        # history (don't forget how to beat older styles). Pure uniform
        # wastes most rollouts farming ancient checkpoints.
        if pool and league_rng.random() < 0.5:
            quartile = max(1, len(pool) // 4)
            candidates = pool[-quartile:]
        else:
            candidates = list(pool)
        league_rng.shuffle(candidates)
        for choice in candidates:
            try:
                ckpt = torch.load(choice, map_location=device, weights_only=True)
                if ckpt.get("obs_dim", -1) != OBS_DIM:
                    continue
                if ckpt.get("arch", 1) != PolicyNet.version:
                    continue
                opponent.load_state_dict(ckpt["policy"])
                return choice.name
            except Exception:
                continue
        opponent.load_state_dict(policy.state_dict())
        return "mirror"

    def frozen_policy(obs_np: np.ndarray, dones_np: np.ndarray) -> np.ndarray:
        nonlocal opp_h
        with torch.no_grad():
            direction, op, speed, _, _, opp_h = opponent.act(
                torch.as_tensor(obs_np, device=device),
                opp_h,
                torch.as_tensor(dones_np, device=device),
            )
        return torch.stack([direction.float(), op.float(), speed], dim=-1).cpu().numpy()

    if args.frozen > 0:
        vec.frozen_policy = frozen_policy

    obs = torch.as_tensor(vec.reset(), device=device)
    T, B = args.rollout, n_agents
    h = policy.initial_state(B, device=device)
    next_done = torch.ones(B, device=device)  # first obs of the run starts episodes

    buf_obs = torch.zeros((T, B, OBS_DIM), device=device)
    buf_dir = torch.zeros((T, B), dtype=torch.long, device=device)
    buf_op = torch.zeros((T, B), dtype=torch.long, device=device)
    buf_logp = torch.zeros((T, B), device=device)
    buf_reward = torch.zeros((T, B), device=device)
    buf_done = torch.zeros((T, B), device=device)
    buf_value = torch.zeros((T, B), device=device)

    episode_masses: list[float] = []
    episode_deaths: list[float] = []
    episode_kills: list[float] = []
    t_start = time.perf_counter()
    steps_done = 0
    opponent_name = "mirror"

    for update in range(start_update + 1, start_update + args.updates + 1):
        if args.frozen > 0 and (update - 1) % args.opponent_refresh == 0:
            opponent_name = refresh_opponent()

        frac = min(1.0, update / max(1, start_update + args.updates))
        ent_coef = args.entropy_coef + (args.entropy_final - args.entropy_coef) * frac

        h0 = h.detach().clone()
        for t in range(T):
            buf_obs[t] = obs
            buf_done[t] = next_done
            with torch.no_grad():
                direction, op, speed, log_prob, value, h = policy.act(obs, h, next_done)
            actions = torch.stack(
                [direction.float(), op.float(), speed], dim=-1
            ).cpu().numpy()
            next_obs, rewards, truncated, infos = vec.step(actions)

            buf_dir[t] = direction
            buf_op[t] = op
            buf_logp[t] = log_prob
            buf_value[t] = value
            buf_reward[t] = torch.as_tensor(rewards, device=device)

            obs = torch.as_tensor(next_obs, device=device)
            next_done = torch.as_tensor(truncated.astype(np.float32), device=device)
            steps_done += B
            for info in infos:
                if "final_masses" in info:
                    episode_masses.extend(info["final_masses"])
                    episode_deaths.extend(info["deaths"])
                    episode_kills.extend(info["kills"])

        with torch.no_grad():
            _, _, _, _, next_value, _ = policy.act(obs, h, next_done)
            advantages = torch.zeros_like(buf_reward)
            last_gae = torch.zeros(B, device=device)
            for t in reversed(range(T)):
                if t == T - 1:
                    not_done = 1.0 - next_done
                    next_v = next_value
                else:
                    not_done = 1.0 - buf_done[t + 1]
                    next_v = buf_value[t + 1]
                delta = buf_reward[t] + args.gamma * next_v * not_done - buf_value[t]
                last_gae = delta + args.gamma * args.gae_lambda * not_done * last_gae
                advantages[t] = last_gae
            returns = advantages + buf_value
            adv_mean, adv_std = advantages.mean(), advantages.std()

        # Minibatch across agents; each agent's sequence is replayed whole so
        # the RNN sees the same temporal context it acted with.
        minibatch_agents = max(1, B // args.minibatches)
        if args.prio_replay:
            # PufferLib-style: sample sequences with replacement, weighted by
            # mean |advantage| — high-surprise trajectories get more epochs.
            priorities = advantages.abs().mean(dim=0) + 1e-6
        pg_losses, v_losses, entropies, kls = [], [], [], []
        for _ in range(args.epochs):
            if args.prio_replay:
                epoch_order = torch.multinomial(priorities, B, replacement=True)
            else:
                epoch_order = torch.as_tensor(np.random.permutation(B), device=device)
            for start in range(0, B, minibatch_agents):
                mb = epoch_order[start : start + minibatch_agents]
                log_prob, entropy, value, dir_logits, op_logits, speed = (
                    policy.evaluate_sequence(
                        buf_obs[:, mb], buf_dir[:, mb], buf_op[:, mb],
                        buf_done[:, mb], h0[mb],
                    )
                )
                mb_adv = (advantages[:, mb] - adv_mean) / (adv_std + 1e-8)
                ratio = (log_prob - buf_logp[:, mb]).exp()
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * ratio.clamp(1.0 - args.clip, 1.0 + args.clip)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = 0.5 * (value - returns[:, mb]).pow(2).mean()
                loss = pg_loss + args.value_coef * v_loss - ent_coef * entropy.mean()

                if reference is not None:
                    with torch.no_grad():
                        ref_feats = reference.sequence_features(
                            buf_obs[:, mb], buf_done[:, mb],
                            reference.initial_state(len(mb), device=device),
                        )
                        ref_dir = reference.dir_head(ref_feats)
                        ref_op = reference.op_head(ref_feats)
                        ref_speed = torch.sigmoid(
                            reference.speed_head(ref_feats)
                        ).squeeze(-1)
                    from torch.distributions import Categorical, kl_divergence

                    kl = (
                        kl_divergence(
                            Categorical(logits=dir_logits), Categorical(logits=ref_dir)
                        )
                        + args.kl_op_mult
                        * kl_divergence(
                            Categorical(logits=op_logits), Categorical(logits=ref_op)
                        )
                    ).mean()
                    speed_drift = (speed - ref_speed).pow(2).mean()
                    loss = loss + args.kl_coef * kl + 2.0 * args.kl_coef * speed_drift
                    kls.append(kl.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(entropy.mean().item())

        sps = steps_done / (time.perf_counter() - t_start)
        mean_mass = np.mean(episode_masses[-200:]) if episode_masses else float("nan")
        mean_deaths = np.mean(episode_deaths[-200:]) if episode_deaths else float("nan")
        mean_kills = np.mean(episode_kills[-200:]) if episode_kills else float("nan")

        metrics = {
            "update": update,
            "steps": steps_done,
            "steps_per_sec": round(sps, 1),
            "reward_per_step": round(buf_reward.mean().item(), 6),
            "ep_mass": None if np.isnan(mean_mass) else round(float(mean_mass), 1),
            "ep_deaths": None if np.isnan(mean_deaths) else round(float(mean_deaths), 3),
            "ep_kills": None if np.isnan(mean_kills) else round(float(mean_kills), 3),
            "opponent": opponent_name,
            "kl": round(float(np.mean(kls)), 5) if kls else None,
            "entropy_coef": round(ent_coef, 5),
            "pg_loss": round(float(np.mean(pg_losses)), 5),
            "value_loss": round(float(np.mean(v_losses)), 5),
            "entropy": round(float(np.mean(entropies)), 4),
            "wall_seconds": round(time.perf_counter() - t_start, 1),
        }
        metrics_file.write(json.dumps(metrics) + "\n")
        metrics_file.flush()
        if wandb_run is not None:
            wandb_run.log(
                {k: v for k, v in metrics.items() if v is not None and k != "opponent"},
                step=update,
            )

        print(
            f"update {update:5d} | steps {steps_done:>10,} | {sps:7,.0f} steps/s | "
            f"reward/step {buf_reward.mean().item():+.4f} | ep_mass {mean_mass:7.1f} | "
            f"ep_kills {mean_kills:5.2f} | ep_deaths {mean_deaths:5.2f} | "
            f"pg {np.mean(pg_losses):+.4f} | v {np.mean(v_losses):.4f} | "
            f"ent {np.mean(entropies):.3f}",
            flush=True,
        )

        if update % args.save_every == 0 or update == start_update + args.updates:
            payload = {
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update,
                "obs_dim": OBS_DIM,
                "arch": PolicyNet.version,
                "model_kwargs": policy.model_kwargs,
                "config": vars(args),
            }
            torch.save(payload, ckpt_dir / f"update_{update:06d}.pt")
            torch.save(payload, ckpt_dir / "latest.pt")


if __name__ == "__main__":
    main()

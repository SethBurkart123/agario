"""Phase 1: behavior cloning from the solo_smart heuristic bots.

Arenas are populated entirely with solo_smart experts, but their chosen
targets are snapped onto the policy's turn-relative action space before being
applied — so the recorded (observation -> action) pairs live in exactly the
dynamics the neural policy will experience. The recurrent policy is then
trained with sequence cross-entropy; split/eject are class-weighted because
they are rare but are precisely the skills RL fails to discover on its own.

The result (bc_init.pt) is a PPO-compatible checkpoint: resume Phase 2 with
  uv run python -m bot_solutions.rl_v1.ppo --resume bot_solutions/rl_v1/checkpoints/rl/bc_init.pt ...
and (with --seed-pool) it is also copied into the league pool as
update_000000.pt so "plays like the handcoded bots" remains a permanent
opponent.

Usage:
  uv run python -m bot_solutions.rl_v1.bc --transitions 600000 --epochs 4
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from agario.bots.types import BotContext, BotInitContext

from bot_solutions.programmatic import SoloSmartBrain

from . import CHECKPOINT_ROOT
from .env import DT, ArenaConfig, ArenaEnv
from .heuristic import HeuristicDriver
from .model import PolicyNet
from .obs import (
    N_DIRECTIONS,
    N_OPS,
    N_TURNS,
    OBS_DIM,
    TURN_OFFSETS,
    speed_from_distance,
)

CHECKPOINT_DIR = CHECKPOINT_ROOT / "rl"


def discretize_turn(heading: int, cx: float, cy: float, tx: float, ty: float) -> int:
    """Snap the expert's desired target direction to the turn action that
    brings the current heading closest to it."""
    desired = math.atan2(ty - cy, tx - cx)
    desired_dir = round(desired / (2.0 * math.pi) * N_DIRECTIONS) % N_DIRECTIONS
    best, best_dist = 0, 99
    for action, offset in enumerate(TURN_OFFSETS):
        new_heading = (heading + offset) % N_DIRECTIONS
        dist = min((new_heading - desired_dir) % N_DIRECTIONS,
                   (desired_dir - new_heading) % N_DIRECTIONS)
        if dist < best_dist or (dist == best_dist and abs(offset) < abs(TURN_OFFSETS[best])):
            best, best_dist = action, dist
    return best


class ExpertArena:
    """One arena whose 'learner' slots are driven by solo_smart experts
    through the discretized action space.

    Personalities are created ONCE per slot and persist across episodes —
    label distributions stay stable instead of re-rolling every reset."""

    def __init__(self, cfg: ArenaConfig, seed: int) -> None:
        self.env = ArenaEnv(cfg, seed)
        self._seed = seed
        self._slot_brains: list[SoloSmartBrain] = [
            SoloSmartBrain(
                BotInitContext(
                    plugin_name="solo_smart",
                    bot_name=f"Expert-{idx}",
                    team_id=None,
                    bot_index=idx,
                    rng=random.Random(seed * 1009 + idx),
                )
            )
            for idx in range(cfg.n_learners)
        ]
        self._brains: dict[str, SoloSmartBrain] = {}
        self._memories: dict[str, dict] = {}

    def reset(self) -> np.ndarray:
        obs = self.env.reset()
        self._brains = {
            pid: self._slot_brains[idx]
            for idx, pid in enumerate(self.env.learner_ids)
        }
        self._memories = {pid: {} for pid in self.env.learner_ids}
        return obs

    def expert_actions(self) -> np.ndarray:
        """Rows: [turn, op, speed] — speed is the physics response the
        expert's chosen target distance would produce."""
        env = self.env
        players_by_id, players, foods, ejected, viruses = HeuristicDriver.build_views(env.world)
        actions = np.zeros((len(env.learner_ids), 3), dtype=np.float64)
        actions[:, 2] = 1.0
        for idx, pid in enumerate(env.learner_ids):
            me = players_by_id.get(pid)
            if me is None or not me.blobs:
                continue
            ctx = BotContext(
                now=env.now,
                dt=DT * env.cfg.frame_skip,
                world_width=env.size,
                world_height=env.size,
                me=me,
                players=players,
                foods=foods,
                ejected=ejected,
                viruses=viruses,
                team_state={},
                memory=self._memories[pid],
            )
            try:
                action = self._brains[pid].decide(ctx)
            except Exception:
                continue
            total = sum(b.mass for b in me.blobs)
            cx = sum(b.x * b.mass for b in me.blobs) / total
            cy = sum(b.y * b.mass for b in me.blobs) / total
            actions[idx][0] = discretize_turn(
                env._headings[pid], cx, cy, action.target_x, action.target_y
            )
            actions[idx][1] = 1 if action.split else (2 if action.eject else 0)
            actions[idx][2] = speed_from_distance(
                math.hypot(action.target_x - cx, action.target_y - cy)
            )
        return actions


def collect(
    args,
    metrics_file=None,
    *,
    transitions: int | None = None,
    student=None,
    row_iter=None,
    seed_offset: int = 0,
) -> tuple[list, np.ndarray]:
    """Collect (obs -> expert label) chunks.

    student=None: the expert drives (vanilla BC data).
    student=policy: DAgger — the STUDENT drives (sampled actions, recurrent
    state maintained), while the expert labels every state the student
    actually visits. This is what closes the distribution-shift gap.
    """
    import torch as _torch

    cfg = ArenaConfig(n_learners=args.experts, n_frozen=0, n_heuristic=0)
    arenas = [
        ExpertArena(cfg, seed=args.seed + seed_offset + 7919 * i) for i in range(args.arenas)
    ]
    B = args.arenas * args.experts
    T = args.chunk
    n_transitions = transitions if transitions is not None else args.transitions

    chunks = []
    op_counts = np.zeros(N_OPS, dtype=np.int64)
    obs = np.concatenate([a.reset() for a in arenas], axis=0)
    done = np.ones(B, dtype=np.float32)
    h = student.initial_state(B) if student is not None else None
    target_chunks = max(1, n_transitions // (T * B))
    t_start = time.perf_counter()

    for chunk_idx in range(target_chunks):
        c_obs = np.zeros((T, B, OBS_DIM), dtype=np.float32)
        c_turn = np.zeros((T, B), dtype=np.int64)
        c_op = np.zeros((T, B), dtype=np.int64)
        c_speed = np.zeros((T, B), dtype=np.float32)
        c_done = np.zeros((T, B), dtype=np.float32)
        for t in range(T):
            c_obs[t] = obs
            c_done[t] = done

            # Expert labels for the current states (always recorded).
            expert_acts = np.zeros((B, 3), dtype=np.float64)
            for i, arena in enumerate(arenas):
                lo = i * args.experts
                expert_acts[lo : lo + args.experts] = arena.expert_actions()
            c_turn[t] = expert_acts[:, 0].astype(np.int64)
            c_op[t] = expert_acts[:, 1].astype(np.int64)
            c_speed[t] = expert_acts[:, 2].astype(np.float32)

            # Who drives: expert (BC) or student (DAgger).
            if student is not None:
                with _torch.no_grad():
                    turn, op, speed, _, _, h = student.act(
                        _torch.as_tensor(obs), h, _torch.as_tensor(done)
                    )
                drive_acts = _torch.stack(
                    [turn.float(), op.float(), speed], dim=-1
                ).numpy()
            else:
                drive_acts = expert_acts

            done = np.zeros(B, dtype=np.float32)
            next_obs_parts = []
            for i, arena in enumerate(arenas):
                lo = i * args.experts
                step_obs, _, truncated, _ = arena.env.step(drive_acts[lo : lo + args.experts])
                if truncated:
                    step_obs = arena.reset()
                    done[lo : lo + args.experts] = 1.0
                next_obs_parts.append(step_obs.copy())
            obs = np.concatenate(next_obs_parts, axis=0)
        c_done[0] = 1.0  # each chunk trains from a fresh hidden state
        for op_class in range(N_OPS):
            op_counts[op_class] += int((c_op == op_class).sum())
        chunks.append((c_obs, c_turn, c_op, c_speed, c_done))
        done_steps = (chunk_idx + 1) * T * B
        rate = done_steps / (time.perf_counter() - t_start)
        driver = "student" if student is not None else "expert"
        print(
            f"collected {done_steps:,}/{target_chunks * T * B:,} transitions "
            f"({driver}-driven, {rate:,.0f}/s)",
            flush=True,
        )
        if metrics_file is not None:
            metrics_file.write(json.dumps({
                "update": next(row_iter) if row_iter is not None else chunk_idx + 1,
                "steps": done_steps,
                "steps_per_sec": round(rate, 1),
                "wall_seconds": round(time.perf_counter() - t_start, 1),
            }) + "\n")
            metrics_file.flush()

    return chunks, op_counts


class Trainer:
    """Sequence cross-entropy training with windowed dashboard logging."""

    def __init__(self, args, policy: PolicyNet, metrics_file, row_iter) -> None:
        self.args = args
        self.policy = policy
        self.metrics_file = metrics_file
        self.row_iter = row_iter
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
        self.labels_seen = 0
        self.t_start = time.perf_counter()
        self._win: list[tuple[float, float, int, int]] = []

    def set_op_weights(self, op_counts: np.ndarray) -> None:
        total = op_counts.sum()
        weights = torch.as_tensor(
            np.sqrt(total / np.maximum(op_counts, 1)), dtype=torch.float32
        )
        weights = (weights / weights[0]).clamp(max=20.0)
        print(f"op class weights: {[round(w, 2) for w in weights.tolist()]}")
        self.ce_op = torch.nn.CrossEntropyLoss(weight=weights)
        self.ce_turn = torch.nn.CrossEntropyLoss()

    def _flush_window(self) -> None:
        losses = [w[0] for w in self._win]
        turns = [w[1] for w in self._win]
        hits = sum(w[2] for w in self._win)
        tot = sum(w[3] for w in self._win)
        self.metrics_file.write(json.dumps({
            "update": next(self.row_iter),
            "steps": self.labels_seen,
            "steps_per_sec": round(self.labels_seen / (time.perf_counter() - self.t_start), 1),
            "bc_loss": round(float(np.mean(losses)), 4),
            "bc_turn_acc": round(float(np.mean(turns)), 4),
            "bc_split_recall": round(hits / tot, 4) if tot else None,
            "wall_seconds": round(time.perf_counter() - self.t_start, 1),
        }) + "\n")
        self.metrics_file.flush()
        self._win.clear()

    def train(self, chunks: list, epochs: int, label: str) -> None:
        B = chunks[0][0].shape[1]
        # Minibatch = (chunk, agent-slice); shuffle ALL of them each epoch so
        # expert-driven and student-driven (DAgger) data interleave — ordered
        # passes make SGD oscillate between dataset sections.
        specs = [
            (ci, start)
            for ci in range(len(chunks))
            for start in range(0, B, self.args.batch_agents)
        ]
        for epoch in range(epochs):
            turn_hits, op_split_hits, op_split_total, seen = 0, 0, 0, 0
            losses = []
            np.random.shuffle(specs)
            for ci, start in specs:
                c_obs, c_turn, c_op, c_speed, c_done = chunks[ci]
                mb = slice(start, min(start + self.args.batch_agents, B))
                obs_seq = torch.as_tensor(c_obs[:, mb])
                turn_seq = torch.as_tensor(c_turn[:, mb])
                op_seq = torch.as_tensor(c_op[:, mb])
                done_seq = torch.as_tensor(c_done[:, mb])
                feats = self.policy.sequence_features(
                    obs_seq, done_seq, self.policy.initial_state(mb.stop - mb.start)
                )
                turn_logits = self.policy.dir_head(feats).reshape(-1, N_TURNS)
                op_logits = self.policy.op_head(feats).reshape(-1, N_OPS)
                speed_pred = torch.sigmoid(self.policy.speed_head(feats)).reshape(-1)
                turn_labels = turn_seq.reshape(-1)
                op_labels = op_seq.reshape(-1)
                speed_labels = torch.as_tensor(c_speed[:, mb]).reshape(-1)
                loss = (
                    self.ce_turn(turn_logits, turn_labels)
                    + self.ce_op(op_logits, op_labels)
                    + 2.0 * torch.nn.functional.mse_loss(speed_pred, speed_labels)
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()

                losses.append(loss.item())
                with torch.no_grad():
                    mb_turn_hits = int((turn_logits.argmax(-1) == turn_labels).sum())
                    turn_hits += mb_turn_hits
                    seen += turn_labels.numel()
                    split_mask = op_labels == 1
                    mb_split_total = int(split_mask.sum())
                    mb_split_hits = int((op_logits.argmax(-1)[split_mask] == 1).sum())
                    op_split_total += mb_split_total
                    op_split_hits += mb_split_hits

                self.labels_seen += turn_labels.numel()
                self._win.append(
                    (loss.item(), mb_turn_hits / turn_labels.numel(),
                     mb_split_hits, mb_split_total)
                )
                if len(self._win) >= 8:
                    self._flush_window()
            split_recall = op_split_hits / max(1, op_split_total)
            print(
                f"[{label}] epoch {epoch + 1}/{epochs}: loss {np.mean(losses):.4f} | "
                f"turn acc {turn_hits / seen:.1%} | split recall {split_recall:.1%}",
                flush=True,
            )
        if self._win:
            self._flush_window()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transitions", type=int, default=500_000)
    parser.add_argument("--dagger-iters", type=int, default=2)
    parser.add_argument("--dagger-transitions", type=int, default=250_000)
    parser.add_argument("--dagger-epochs", type=int, default=2)
    parser.add_argument("--arenas", type=int, default=8)
    parser.add_argument("--experts", type=int, default=6)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-agents", type=int, default=24)
    parser.add_argument("--rnn", choices=("gru", "mingru"), default="gru")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--out", default=str(CHECKPOINT_DIR / "bc_init.pt"))
    parser.add_argument("--seed-pool", action="store_true", default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    metrics_path = Path(args.out).parent / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_file = metrics_path.open("w")

    from itertools import count as _count

    row_iter = _count(1)  # one monotonic update-id stream for all phases
    policy = PolicyNet(rnn_type=args.rnn)
    trainer = Trainer(args, policy, metrics_file, row_iter)

    print("=== round 0: expert-driven demonstrations ===")
    chunks, op_counts = collect(args, metrics_file, row_iter=row_iter)
    total = op_counts.sum()
    print(f"op distribution: none {op_counts[0]/total:.1%}, split {op_counts[1]/total:.2%}, eject {op_counts[2]/total:.2%}")
    trainer.set_op_weights(op_counts)
    trainer.train(chunks, args.epochs, label="bc")

    for it in range(1, args.dagger_iters + 1):
        print(f"=== DAgger round {it}: student drives, expert labels ===")
        policy.eval()
        new_chunks, new_counts = collect(
            args,
            metrics_file,
            transitions=args.dagger_transitions,
            student=policy,
            row_iter=row_iter,
            seed_offset=it * 104729,
        )
        policy.train()
        chunks += new_chunks
        op_counts += new_counts
        trainer.set_op_weights(op_counts)
        trainer.train(chunks, args.dagger_epochs, label=f"dagger-{it}")

    payload = {
        "policy": policy.state_dict(),
        "update": 0,
        "obs_dim": OBS_DIM,
        "arch": PolicyNet.version,
        "model_kwargs": policy.model_kwargs,
        "config": {"bc": vars(args)},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"saved BC policy to {out}")
    from .imitate import atomic_copy

    if args.seed_pool:
        atomic_copy(out, out.parent / "update_000000.pt")
        print(f"seeded league pool: {out.parent / 'update_000000.pt'}")
    atomic_copy(out, out.parent / "latest.pt")
    print(f"updated {out.parent / 'latest.pt'} (live bots load this by default)")


if __name__ == "__main__":
    main()

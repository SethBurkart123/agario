"""Parallel imitation learning: a collector process streams expert-labeled
demonstrations to a spool directory while a trainer process consumes them
continuously. No collect/train alternation — both cores stay busy and the
dashboard shows one uninterrupted training stream.

Collector (DAgger by default): drives arenas with the latest student policy
(refreshed from the spool whenever the trainer publishes a new one; pure
expert-driven until the first student appears, and every --expert-every-th
chunk stays expert-driven so clean demonstrations keep flowing). Every state
is labeled by the solo_smart expert.

Trainer: maintains a rolling window of the newest chunks, trains shuffled
sequence cross-entropy, periodically publishes the student back to the spool
(closing the DAgger loop), and exits after --labels gradient labels.

Usage (run in two terminals or background):
  uv run python -m bot_solutions.rl_v1.imitate collector
  uv run python -m bot_solutions.rl_v1.imitate trainer --labels 5000000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import count
from pathlib import Path

import numpy as np
import torch

from .bc import CHECKPOINT_DIR, ExpertArena
from . import CHECKPOINT_ROOT
from .env import ArenaConfig
from .model import PolicyNet
from .obs import N_OPS, N_TURNS, OBS_DIM


def atomic_save(payload, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def atomic_copy(src: Path, dst: Path) -> None:
    """Copy without ever exposing a half-written file — the live game loads
    latest.pt at arbitrary times."""
    import shutil

    tmp = dst.with_suffix(".tmp")
    shutil.copy(src, tmp)
    os.replace(tmp, dst)


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


def run_collector(args) -> None:
    spool = Path(args.spool)
    spool.mkdir(parents=True, exist_ok=True)
    (spool / "STOP").unlink(missing_ok=True)  # stale STOP from a prior run
    metrics_file = (spool / "collector_metrics.jsonl").open("a")

    cfg = ArenaConfig(n_learners=args.experts, n_frozen=0, n_heuristic=0)
    arenas = [ExpertArena(cfg, seed=args.seed + 7919 * i) for i in range(args.arenas)]
    B = args.arenas * args.experts
    T = args.chunk
    rng = np.random.default_rng(args.seed)

    def seed_scenarios(arena: ExpertArena) -> None:
        """Drop agents into the awkward states organic play rarely visits:
        fragmented-and-scattered, oversized, undersized."""
        if not args.scenarios:
            return
        env = arena.env
        for pid in env.learner_ids:
            roll = rng.random()
            if roll < 0.25:
                env.world.scatter_player(pid, int(rng.integers(3, 11)), env.now)
            elif roll < 0.45:
                env.world.set_player_mass(pid, float(560.0 * rng.uniform(0.3, 4.0)))

    student: PolicyNet | None = None
    student_mtime = 0.0
    student_path = spool / "student.pt"

    existing = sorted(spool.glob("chunk_*.npz"))
    chunk_idx = int(existing[-1].stem.split("_")[1]) + 1 if existing else 0

    obs_parts = []
    for arena in arenas:
        arena.reset()
        seed_scenarios(arena)
        arena.env._encode_for(arena.env.learner_ids, arena.env._obs)
        obs_parts.append(arena.env._obs.copy())
    obs = np.concatenate(obs_parts, axis=0)
    done = np.ones(B, dtype=np.float32)
    h = None
    steps_total = 0
    decision_count = 0
    t_start = time.perf_counter()

    while not (spool / "STOP").exists():
        # Refresh the student policy whenever the trainer publishes one.
        if student_path.exists():
            mtime = student_path.stat().st_mtime
            if mtime > student_mtime:
                try:
                    ckpt = torch.load(student_path, map_location="cpu", weights_only=True)
                    student = PolicyNet(**ckpt.get("model_kwargs", {}))
                    student.load_state_dict(ckpt["policy"])
                    student.eval()
                    student_mtime = mtime
                    h = student.initial_state(B)
                except Exception:
                    pass  # mid-write; retry next chunk

        expert_driven = student is None or (
            args.expert_every > 0 and chunk_idx % args.expert_every == 0
        )

        c_obs = np.zeros((T, B, OBS_DIM), dtype=np.float32)
        c_turn = np.zeros((T, B), dtype=np.int64)
        c_op = np.zeros((T, B), dtype=np.int64)
        c_speed = np.zeros((T, B), dtype=np.float32)
        c_done = np.zeros((T, B), dtype=np.float32)
        for t in range(T):
            c_obs[t] = obs
            c_done[t] = done

            expert_acts = np.zeros((B, 3), dtype=np.float64)
            for i, arena in enumerate(arenas):
                lo = i * args.experts
                expert_acts[lo : lo + args.experts] = arena.expert_actions()
            c_turn[t] = expert_acts[:, 0].astype(np.int64)
            c_op[t] = expert_acts[:, 1].astype(np.int64)
            c_speed[t] = expert_acts[:, 2].astype(np.float32)

            if expert_driven:
                drive = expert_acts
            else:
                with torch.no_grad():
                    turn, op, speed, _, _, h = student.act(
                        torch.as_tensor(obs), h, torch.as_tensor(done)
                    )
                drive = torch.stack(
                    [turn.float(), op.float(), speed], dim=-1
                ).numpy()

            done = np.zeros(B, dtype=np.float32)
            decision_count += 1
            parts = []
            for i, arena in enumerate(arenas):
                lo = i * args.experts
                step_obs, _, truncated, _ = arena.env.step(drive[lo : lo + args.experts])
                if truncated:
                    arena.reset()
                    seed_scenarios(arena)
                    arena.env._encode_for(arena.env.learner_ids, arena.env._obs)
                    step_obs = arena.env._obs
                    done[lo : lo + args.experts] = 1.0
                elif args.scenarios and decision_count % 150 == 0 and rng.random() < 0.3:
                    # Mid-episode disruption: scatter one random agent so the
                    # expert demonstrates recovery from a fresh fragmentation.
                    victim = arena.env.learner_ids[int(rng.integers(args.experts))]
                    arena.env.world.scatter_player(
                        victim, int(rng.integers(3, 11)), arena.env.now
                    )
                parts.append(step_obs.copy())
            obs = np.concatenate(parts, axis=0)
        c_done[0] = 1.0

        tmp = spool / f"tmp_{chunk_idx:06d}.npz"
        np.savez(tmp, obs=c_obs, turn=c_turn, op=c_op, speed=c_speed, done=c_done,
                 expert_driven=np.array(expert_driven))
        os.replace(tmp, spool / f"chunk_{chunk_idx:06d}.npz")
        chunk_idx += 1
        steps_total += T * B

        # Ring buffer: drop chunks far older than the trainer's window.
        all_chunks = sorted(spool.glob("chunk_*.npz"))
        for stale in all_chunks[: max(0, len(all_chunks) - args.keep)]:
            stale.unlink(missing_ok=True)

        rate = steps_total / (time.perf_counter() - t_start)
        driver = "expert" if expert_driven else "student"
        print(f"chunk {chunk_idx:5d} ({driver}-driven) | {rate:,.0f} transitions/s", flush=True)
        metrics_file.write(json.dumps({
            "update": chunk_idx,
            "steps": steps_total,
            "steps_per_sec": round(rate, 1),
            "wall_seconds": round(time.perf_counter() - t_start, 1),
        }) + "\n")
        metrics_file.flush()
    print("collector: STOP file found, exiting")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def run_trainer(args) -> None:
    spool = Path(args.spool)
    spool.mkdir(parents=True, exist_ok=True)
    (spool / "STOP").unlink(missing_ok=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics_file = (out.parent / "metrics.jsonl").open("w")
    row_iter = count(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        policy = PolicyNet(**ckpt.get("model_kwargs", {}))
        policy.load_state_dict(ckpt["policy"])
        print(f"warm-started from {args.resume}")
    else:
        policy = PolicyNet(rnn_type=args.rnn)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    # filename -> (obs, turn, op, speed, done, ret|None, h0|None, expert_driven)
    dataset: dict[str, tuple] = {}
    op_counts = np.zeros(N_OPS, dtype=np.int64)

    def refresh_dataset() -> int:
        nonlocal op_counts
        added = 0
        names = sorted(p.name for p in spool.glob("chunk_*.npz"))
        for name in names:
            if name in dataset:
                continue
            try:
                z = np.load(spool / name)
                if dataset and z["obs"].shape[1] != next(iter(dataset.values()))["obs"].shape[1]:
                    continue  # different arena geometry; refuse to mix
                dataset[name] = {
                    "obs": z["obs"], "turn": z["turn"], "op": z["op"],
                    "speed": z["speed"], "done": z["done"],
                    "ret": z["ret"] if "ret" in z.files else None,
                    "h0": z["h0"] if "h0" in z.files else None,
                    "turn_soft": z["turn_soft"] if "turn_soft" in z.files else None,
                    "op_soft": z["op_soft"] if "op_soft" in z.files else None,
                    "mask": z["label_mask"] if "label_mask" in z.files else None,
                    "expert": bool(z["expert_driven"]),
                }
                added += 1
            except Exception:
                continue  # partial write; next scan gets it
        while len(dataset) > args.window:
            dataset.pop(next(iter(sorted(dataset))))
        if added:
            op_counts = np.zeros(N_OPS, dtype=np.int64)
            for entry in dataset.values():
                op_arr = entry["op"]
                for op_class in range(N_OPS):
                    op_counts[op_class] += int((op_arr == op_class).sum())
        return added

    print("waiting for initial chunks from the collector...")
    refresh_dataset()
    while len(dataset) < args.min_chunks:
        time.sleep(2.0)
        refresh_dataset()
    print(f"starting with {len(dataset)} chunks in the window")

    def make_ce() -> tuple:
        total = op_counts.sum()
        w = torch.as_tensor(np.sqrt(total / np.maximum(op_counts, 1)), dtype=torch.float32)
        w = (w / w[0]).clamp(max=20.0)
        return torch.nn.CrossEntropyLoss(weight=w), torch.nn.CrossEntropyLoss()

    ce_op, ce_turn = make_ce()
    labels_done = 0
    t_start = time.perf_counter()
    win: list[tuple[float, float, int, int]] = []
    mb_count = 0

    def publish_student() -> None:
        atomic_save(
            {"policy": policy.state_dict(), "model_kwargs": policy.model_kwargs},
            spool / "student.pt",
        )

    publish_student()
    while labels_done < args.labels:
        names = list(dataset.keys())
        B = dataset[names[0]]["obs"].shape[1]
        specs = [
            (n, start) for n in names for start in range(0, B, args.batch_agents)
        ]
        np.random.shuffle(specs)
        for name, start in specs:
            if name not in dataset:
                continue
            ch = dataset[name]
            c_obs, c_turn, c_op = ch["obs"], ch["turn"], ch["op"]
            c_speed, c_done, c_ret, c_h0 = ch["speed"], ch["done"], ch["ret"], ch["h0"]
            is_expert = ch["expert"]
            mb = slice(start, min(start + args.batch_agents, B))
            obs_seq = torch.as_tensor(c_obs[:, mb])
            done_seq = torch.as_tensor(c_done[:, mb])
            turn_labels = torch.as_tensor(c_turn[:, mb]).reshape(-1)
            op_labels = torch.as_tensor(c_op[:, mb]).reshape(-1)
            speed_labels = torch.as_tensor(c_speed[:, mb]).reshape(-1)

            # Train with the LIVE chunk-start hidden states when the
            # generator saved them — the value head is used at play/search
            # time with episode-accumulated memory, so training it from
            # amnesiac zeros is a conditioning mismatch.
            h0 = (
                torch.as_tensor(c_h0[mb])
                if c_h0 is not None
                else policy.initial_state(mb.stop - mb.start)
            )
            feats = policy.sequence_features(obs_seq, done_seq, h0)
            turn_logits = policy.dir_head(feats).reshape(-1, N_TURNS)
            op_logits = policy.op_head(feats).reshape(-1, N_OPS)
            speed_pred = torch.sigmoid(policy.speed_head(feats)).reshape(-1)

            if ch["turn_soft"] is not None:
                # AlphaZero-style soft targets: cross-entropy against the
                # search's score-weighted candidate distribution, only on
                # rows the search actually labeled (mask).
                mask = torch.as_tensor(ch["mask"][:, mb]).reshape(-1)
                denom = mask.sum().clamp(min=1.0)
                t_soft = torch.as_tensor(ch["turn_soft"][:, mb]).reshape(-1, N_TURNS)
                o_soft = torch.as_tensor(ch["op_soft"][:, mb]).reshape(-1, N_OPS)
                turn_ce = -(t_soft * torch.log_softmax(turn_logits, -1)).sum(-1)
                op_ce = -(o_soft * torch.log_softmax(op_logits, -1)).sum(-1)
                speed_se = (speed_pred - speed_labels).pow(2)
                loss = (
                    (turn_ce * mask).sum() / denom
                    + (op_ce * mask).sum() / denom
                    + 2.0 * (speed_se * mask).sum() / denom
                )
            else:
                loss = (
                    ce_turn(turn_logits, turn_labels)
                    + ce_op(op_logits, op_labels)
                    + 2.0 * torch.nn.functional.mse_loss(speed_pred, speed_labels)
                )
            if c_ret is not None:
                # AlphaZero-style: game outcomes train the value head during
                # distillation, so the next generation's search prices futures
                # with fresh values — no separate RL phase needed.
                value_pred = policy.value_head(feats).reshape(-1)
                ret_labels = torch.as_tensor(c_ret[:, mb]).reshape(-1)
                loss = loss + 0.5 * torch.nn.functional.mse_loss(value_pred, ret_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            labels_done += turn_labels.numel()
            mb_count += 1
            with torch.no_grad():
                hits = int((turn_logits.argmax(-1) == turn_labels).sum())
                op_pred = op_logits.argmax(-1)
                split_mask = op_labels == 1
                s_tot = int(split_mask.sum())
                s_hit = int((op_pred[split_mask] == 1).sum())
                pred_split = op_pred == 1
                p_tot = int(pred_split.sum())
                p_hit = int((op_labels[pred_split] == 1).sum())
                speed_mae = float((speed_pred - speed_labels).abs().mean())
            win.append(
                (loss.item(), hits / turn_labels.numel(), s_hit, s_tot,
                 is_expert, p_hit, p_tot, speed_mae)
            )

            if len(win) >= 12:
                tot = sum(w_[3] for w_ in win)
                prec_tot = sum(w_[6] for w_ in win)
                exp_accs = [w_[1] for w_ in win if w_[4]]
                dag_accs = [w_[1] for w_ in win if not w_[4]]
                metrics_file.write(json.dumps({
                    "update": next(row_iter),
                    "steps": labels_done,
                    "steps_per_sec": round(labels_done / (time.perf_counter() - t_start), 1),
                    "bc_loss": round(float(np.mean([w_[0] for w_ in win])), 4),
                    # Accuracy split by data source: expert-state accuracy is
                    # the stable signal; DAgger-state accuracy tracks the
                    # ever-harder moving target and reads lower by design.
                    "bc_turn_acc": round(float(np.mean(exp_accs)), 4) if exp_accs else None,
                    "bc_dagger_acc": round(float(np.mean(dag_accs)), 4) if dag_accs else None,
                    "bc_split_recall": round(sum(w_[2] for w_ in win) / tot, 4) if tot else None,
                    "bc_split_prec": round(sum(w_[5] for w_ in win) / prec_tot, 4) if prec_tot else None,
                    "bc_speed_mae": round(float(np.mean([w_[7] for w_ in win])), 4),
                    "wall_seconds": round(time.perf_counter() - t_start, 1),
                }) + "\n")
                metrics_file.flush()
                win.clear()

            if mb_count % args.sync_every == 0:
                publish_student()
                if refresh_dataset():
                    ce_op, ce_turn = make_ce()
            if labels_done >= args.labels:
                break
        progress = labels_done / args.labels
        print(
            f"trainer: {labels_done:,}/{args.labels:,} labels ({progress:.0%}), "
            f"window {len(dataset)} chunks",
            flush=True,
        )

    payload = {
        "policy": policy.state_dict(),
        "update": 0,
        "obs_dim": OBS_DIM,
        "arch": PolicyNet.version,
        "model_kwargs": policy.model_kwargs,
        "config": {"imitate": vars(args)},
    }
    torch.save(payload, out)

    # League anchor: only seed if absent — clobbering it every cycle would
    # silently replace the style anchor the league pool preserves.
    anchor = out.parent / "update_000000.pt"
    if not anchor.exists():
        atomic_copy(out, anchor)
    # The live neural plugin loads latest.pt by default — keep it pointing at
    # the newest playable policy (atomically: the game may load it mid-write).
    atomic_copy(out, out.parent / "latest.pt")
    (spool / "STOP").touch()
    print(f"trainer done: saved {out} (+ latest.pt), STOP signaled")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="role", required=True)

    pc = sub.add_parser("collector")
    pc.add_argument("--spool", default=str(CHECKPOINT_ROOT / "imitate"))
    pc.add_argument("--arenas", type=int, default=8)
    pc.add_argument("--experts", type=int, default=6)
    pc.add_argument("--chunk", type=int, default=128)
    pc.add_argument("--expert-every", type=int, default=4)
    pc.add_argument("--keep", type=int, default=60)
    pc.add_argument("--seed", type=int, default=11)
    pc.add_argument("--scenarios", action=argparse.BooleanOptionalAction, default=True)

    pt = sub.add_parser("trainer")
    pt.add_argument("--spool", default=str(CHECKPOINT_ROOT / "imitate"))
    pt.add_argument("--labels", type=int, default=5_000_000)
    pt.add_argument("--window", type=int, default=48)
    pt.add_argument("--min-chunks", type=int, default=8)
    pt.add_argument("--batch-agents", type=int, default=24)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--sync-every", type=int, default=25)
    pt.add_argument("--rnn", choices=("gru", "mingru"), default="gru")
    pt.add_argument("--seed", type=int, default=5)
    pt.add_argument("--out", default=str(CHECKPOINT_DIR / "bc_init.pt"))
    pt.add_argument("--resume", default=None, help="warm-start from an existing checkpoint")

    args = parser.parse_args()
    if args.role == "collector":
        run_collector(args)
    else:
        run_trainer(args)


if __name__ == "__main__":
    main()

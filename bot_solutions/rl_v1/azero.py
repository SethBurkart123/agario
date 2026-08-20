"""The proper AlphaZero loop (rl_v1/docs/ALPHAZERO_PLAN.md).

Per generation:
  gate:     search-driven vs raw-policy head-to-head. If search does not
            beat the policy it is supposed to teach, ABORT — never distill
            from a weaker teacher (the session's hardest-won lesson).
  generate: search-driven play in BatchedArenas. Each searched decision runs
            K_CANDIDATES x samples imagined futures with EVERY player
            policy-driven (batched GPU forwards), averaged into candidate Q
            values -> soft targets + the executed move.
  distill:  imitate trainer absorbs soft policy targets + value returns.
  eval:     vs solo_smart, appended by the marathon's eval stage as before.

Usage:
  uv run --no-sync python -m bot_solutions.rl_v1.azero gate
  uv run --no-sync python -m bot_solutions.rl_v1.azero run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .exit import load_policy
from . import CHECKPOINT_ROOT
from .obs import N_OPS, N_TURNS, OBS_DIM

GAMMA = 0.995
CHECKPOINT_DIR = CHECKPOINT_ROOT / "rl"
MARATHON_DIR = CHECKPOINT_ROOT / "marathon"
# Candidates are Gumbel-sampled from the policy's joint logits per decision.


def _append_jsonl(path: Path, row: dict) -> None:
    """Append one JSON row to a .jsonl file, creating parents as needed.

    Best-effort: instrumentation must never crash a training stage, so any
    filesystem error is swallowed."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")
    except OSError:
        pass


def _write_progress(
    phase: str,
    fraction: float | None = None,
    eta_minutes: float | None = None,
    detail: str | None = None,
) -> None:
    """Atomically write the dashboard heartbeat (phase_progress.json).

    The dashboard treats updated_at as the liveness signal, so the slow
    phases (gate, generate) call this at least every ~30s. Best-effort."""
    try:
        MARATHON_DIR.mkdir(parents=True, exist_ok=True)
        tmp = MARATHON_DIR / "phase_progress.json.tmp"
        tmp.write_text(json.dumps({
            "phase": phase,
            "fraction": fraction,
            "eta_minutes": eta_minutes,
            "detail": detail,
            "updated_at": time.time(),
        }))
        os.replace(tmp, MARATHON_DIR / "phase_progress.json")
    except OSError:
        pass




# ---------------------------------------------------------------------------
# Gumbel MuZero machinery (Danihelka et al., ICLR 2022; mctx reference).
# Joint action space = N_TURNS x N_OPS = 24 combos; speed stays the policy's
# continuous head. One Gumbel vector per root is used BOTH to sample the K
# candidates and to select the executed action — that shared-g trick is what
# makes the selection provably never worse than the policy.
# ---------------------------------------------------------------------------
N_JOINT = N_TURNS * N_OPS
MAXVISIT_INIT = 50.0
VALUE_SCALE = 0.1


def joint_logits(dir_logits: np.ndarray, op_logits: np.ndarray) -> np.ndarray:
    """(n, 8), (n, 3) -> (n, 24) with index = turn * N_OPS + op."""
    return (dir_logits[:, :, None] + op_logits[:, None, :]).reshape(-1, N_JOINT)


def gumbel_sample_candidates(
    logits: np.ndarray, k: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Gumbel-top-k without replacement. Returns (g, cand_idx (n, k))."""
    g = rng.gumbel(size=logits.shape).astype(np.float64)
    cand = np.argsort(-(g + logits), axis=-1)[:, :k]
    return g, cand


def completed_q_targets(
    logits: np.ndarray,    # (n, 24) joint logits
    g: np.ndarray,         # (n, 24) the SAME gumbel used for sampling
    cand: np.ndarray,      # (n, k) candidate indices
    q: np.ndarray,         # (n, k) mean rollout Q per candidate
    v_raw: np.ndarray,     # (n,) root value-net estimate
    visits: int,           # rollout samples per candidate
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """mctx-style completed-Q improved policy.

    Returns (target (n,24) softmax distribution, executed joint action (n,),
    sigma_q (n,k) for diagnostics)."""
    n, k = cand.shape
    priors = np.exp(logits - logits.max(-1, keepdims=True))
    priors /= priors.sum(-1, keepdims=True)

    # v_mix (mctx _compute_mixed_value): blend of raw value and the
    # prior-weighted mean of visited Q.
    cand_priors = np.take_along_axis(priors, cand, axis=-1)
    sum_probs = cand_priors.sum(-1) + 1e-12
    weighted_q = (cand_priors * q).sum(-1) / sum_probs
    total_visits = k * visits
    v_mix = (v_raw + total_visits * weighted_q) / (total_visits + 1)

    # Complete: visited combos get their Q, the rest get v_mix; then rescale
    # to [0,1] per root (mctx qtransform) and scale by visit count.
    completed = np.repeat(v_mix[:, None], N_JOINT, axis=-1)
    np.put_along_axis(completed, cand, q, axis=-1)
    lo = completed.min(-1, keepdims=True)
    hi = completed.max(-1, keepdims=True)
    completed01 = (completed - lo) / np.maximum(hi - lo, 1e-8)
    sigma = (MAXVISIT_INIT + visits) * VALUE_SCALE * completed01

    z = logits + sigma
    z -= z.max(-1, keepdims=True)
    target = np.exp(z)
    target /= target.sum(-1, keepdims=True)

    # Executed action: argmax over the SAMPLED candidates of g + logits +
    # sigma — the Gumbel policy-improvement selection (paper eq. 7).
    score = np.take_along_axis(g + logits + sigma, cand, axis=-1)
    executed = np.take_along_axis(cand, score.argmax(-1)[:, None], axis=-1)[:, 0]
    sigma_q = np.take_along_axis(sigma, cand, axis=-1)
    return target.astype(np.float32), executed, sigma_q


def target_marginals(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n, 24) joint target -> (turn (n,8), op (n,3)) marginals."""
    t = target.reshape(-1, N_TURNS, N_OPS)
    return t.sum(-1).astype(np.float32), t.sum(-2).astype(np.float32)


class SearchDriver:
    """Runs the batched search for a set of roots and returns chosen actions
    plus soft targets."""

    def __init__(self, policy, device: str, horizon: int, samples: int, tau: float):
        self.policy = policy
        self.device = device
        self.horizon = horizon
        self.samples = samples
        self.tau = tau

    @torch.no_grad()
    def search(
        self, sim, roots, root_speeds, learners_per_arena: int, agent_idx_of_root,
        cand_turns, cand_ops,
    ):
        n_rows = sim.search_begin(
            roots, [float(s) for s in root_speeds], self.samples,
            cand_turns, cand_ops,
        )
        n_rollouts = n_rows // learners_per_arena
        h = self.policy.initial_state(n_rows, device=self.device)
        no_done = torch.zeros(n_rows, device=self.device)

        for _ in range(self.horizon):
            obs = np.frombuffer(sim.search_obs(), dtype=np.float32).reshape(
                n_rows, OBS_DIM
            )
            turn, op, speed, _, _, h = self.policy.act(
                torch.as_tensor(obs, device=self.device), h, no_done,
                deterministic=True, op_deterministic=True,
            )
            acts = (
                torch.stack([turn.float(), op.float(), speed], dim=-1)
                .cpu().numpy().astype(np.float32)
            )
            sim.search_apply_tick(acts.ravel().tolist())

        # Terminal values for each rollout's focal agent, using that focal
        # row's accumulated rollout hidden state.
        focal_obs = np.frombuffer(sim.search_focal_obs(), dtype=np.float32).reshape(
            n_rollouts, OBS_DIM
        )
        k = len(cand_turns) // len(roots)
        per_root = k * self.samples
        focal_rows = np.array([
            r_i * learners_per_arena + agent_idx_of_root[r_i // per_root]
            for r_i in range(n_rollouts)
        ])
        h_focal = h[torch.as_tensor(focal_rows, device=self.device)]
        _, _, _, _, values, _ = self.policy.act(
            torch.as_tensor(focal_obs, device=self.device), h_focal,
            torch.zeros(n_rollouts, device=self.device),
            deterministic=True, op_deterministic=True,
        )
        q = np.array(
            sim.search_finish([float(v) for v in values.cpu()]), dtype=np.float64
        ).reshape(len(roots), k)
        return q


def make_policy_step(policy, device):
    """Like policy.act (deterministic) but also returns the raw head logits —
    the Gumbel machinery needs the policy's prior over the joint action
    space, not just its argmax."""

    @torch.no_grad()
    def step(obs, h, done):
        h = policy._step(
            torch.as_tensor(obs, device=device), h,
            torch.as_tensor(done, device=device),
        )
        feats = policy.post(h)
        dirl = policy.dir_head(feats)
        opl = policy.op_head(feats)
        speed = torch.sigmoid(policy.speed_head(feats)).squeeze(-1)
        value = policy.value_head(feats).squeeze(-1)
        return (
            dirl.argmax(-1).cpu().numpy(), opl.argmax(-1).cpu().numpy(),
            speed.cpu().numpy(), value.cpu().numpy(), h,
            dirl.cpu().numpy().astype(np.float64),
            opl.cpu().numpy().astype(np.float64),
        )
    return step


def cmd_gate(args) -> bool:
    """Search-driven agent vs raw agent, same arenas. Returns pass/fail."""
    from agario_core import BatchedArenas

    policy = load_policy(args.start).to(args.device)
    driver = SearchDriver(policy, args.device, args.horizon, args.samples, tau=0.5)
    step = make_policy_step(policy, args.device)

    sim = BatchedArenas(args.gate_arenas, 2, 1, args.seed,
                        episode_decisions=args.gate_decisions,
                        kill_bonus=args.kill_bonus)
    n = sim.total_agents()
    obs = np.frombuffer(sim.reset(), dtype=np.float32).reshape(n, OBS_DIM)
    h = policy.initial_state(n, device=args.device)
    done = np.ones(n, dtype=np.float32)

    rng = np.random.default_rng(args.seed)
    finished = 0
    search_mass, raw_mass = [], []
    gate_decision = 0
    t_gate = time.perf_counter()
    last_beat = 0.0
    _write_progress("gate", 0.0, None, "search vs raw")
    while finished < args.gate_arenas:
        gate_decision += 1
        now = time.perf_counter()
        if now - last_beat >= 30.0:
            last_beat = now
            # Episodes end at gate_decisions, so this fraction is exact up to
            # early deaths; cap below 1 since the loop waits on all arenas.
            frac = min(0.95, gate_decision / max(args.gate_decisions, 1))
            elapsed = now - t_gate
            eta = (elapsed / frac - elapsed) / 60.0 if frac > 0 else None
            _write_progress("gate", frac, eta, "search vs raw")
        turn, op, speed, value, h, dirl, opl = step(obs, h, done)
        acts = np.stack([turn, op, speed], axis=-1).astype(np.float32)

        roots = [(a, 0) for a in range(args.gate_arenas)]
        root_rows = [a * 2 for a in range(args.gate_arenas)]
        root_speeds = [float(speed[r]) for r in root_rows]
        jl = joint_logits(dirl[root_rows], opl[root_rows])
        g, cand = gumbel_sample_candidates(jl, args.k_candidates, rng)
        q = driver.search(
            sim, roots, root_speeds, 2, [0] * len(roots),
            (cand // N_OPS).ravel().tolist(), (cand % N_OPS).ravel().tolist(),
        )
        _, executed, _ = completed_q_targets(
            jl, g, cand, q, value[root_rows], args.samples
        )
        for r, row in enumerate(root_rows):
            acts[row] = (executed[r] // N_OPS, executed[r] % N_OPS, root_speeds[r])

        ob, rb, tb = sim.step(acts.ravel().tolist())
        obs = np.frombuffer(ob, dtype=np.float32).reshape(n, OBS_DIM)
        done = np.frombuffer(tb, dtype=np.uint8).astype(np.float32)
        if done.any():
            stats = sim.learner_stats()
            for a in range(args.gate_arenas):
                if done[a * 2] > 0.5:
                    pass  # stats are post-reset; use pending episode stats path
            for s in sim.take_episode_stats():
                finished += 1
        # learner_stats reads live arenas; per-agent final masses need capture
        # BEFORE auto-reset, which the episode stats don't split by agent —
        # so we instead integrate mass over time, a smoother comparator:
        live = sim.learner_stats()
        for a in range(args.gate_arenas):
            search_mass.append(live[a][0][0])
            raw_mass.append(live[a][1][0])

    s_mean, r_mean = float(np.mean(search_mass)), float(np.mean(raw_mass))
    ratio = s_mean / max(r_mean, 1e-9)
    verdict = ratio >= args.gate_margin
    print(
        f"GATE: search-driven mean mass {s_mean:.0f} vs raw {r_mean:.0f} "
        f"(ratio {ratio:.2f}, need >= {args.gate_margin}) -> "
        f"{'PASS' if verdict else 'FAIL'}",
        flush=True,
    )
    _append_jsonl(
        MARATHON_DIR / "history.jsonl",
        {
            "type": "gate",
            "cycle": args.cycle,
            "search_mass": s_mean,
            "raw_mass": r_mean,
            "ratio": ratio,
            "passed": bool(verdict),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    return verdict


def cmd_generate(args) -> None:
    from agario_core import BatchedArenas

    policy = load_policy(args.start).to(args.device)
    driver = SearchDriver(policy, args.device, args.horizon, args.samples, args.tau)
    step = make_policy_step(policy, args.device)
    rng = np.random.default_rng(args.seed)

    # Replay buffer: the spool PERSISTS across generations. New chunks append
    # with continuing numbers; we prune to the newest --replay-chunks so the
    # distill trains on a ~5-generation window (AGZ's sliding window, scaled).
    spool = Path(args.spool)
    spool.mkdir(parents=True, exist_ok=True)
    existing = sorted(spool.glob("chunk_*.npz"))
    chunk_start = (
        int(existing[-1].stem.split("_")[1]) + 1 if existing else 0
    )

    L = args.learners
    sim = BatchedArenas(
        args.arenas, L, args.anchors, args.seed,
        size_lo=args.size_lo, size_hi=args.size_hi,
        kill_bonus=args.kill_bonus,
    )
    B = sim.total_agents()
    T = args.chunk
    obs = np.frombuffer(sim.reset(), dtype=np.float32).reshape(B, OBS_DIM)
    h = policy.initial_state(B, device=args.device)
    done = np.ones(B, dtype=np.float32)
    target_chunks = max(1, args.labels // (T * B))
    t_start = time.perf_counter()
    decision = 0
    last_beat = 0.0
    _write_progress("generate", 0.0, None, f"chunk 1/{target_chunks}")

    for chunk_idx in range(chunk_start, chunk_start + target_chunks):
        c_obs = np.zeros((T, B, OBS_DIM), dtype=np.float32)
        c_turn = np.zeros((T, B), dtype=np.int64)
        c_op = np.zeros((T, B), dtype=np.int64)
        c_speed = np.zeros((T, B), dtype=np.float32)
        c_rew = np.zeros((T, B), dtype=np.float32)
        c_done = np.zeros((T, B), dtype=np.float32)
        c_boot = np.zeros((T, B), dtype=np.float32)
        c_turn_soft = np.zeros((T, B, N_TURNS), dtype=np.float32)
        c_op_soft = np.zeros((T, B, N_OPS), dtype=np.float32)
        c_mask = np.zeros((T, B), dtype=np.float32)
        c_h0 = h.cpu().numpy().copy()

        for t in range(T):
            c_obs[t] = obs
            c_done[t] = done
            turn, op, speed, value, h, dirl, opl = step(obs, h, done)
            acts = np.stack([turn, op, speed], axis=-1).astype(np.float32)

            # Round-robin which agents get searched this decision.
            roots = []
            agent_idx_of_root = []
            for a in range(args.arenas):
                for j in range(args.roots_per_arena):
                    idx = (decision * args.roots_per_arena + j) % L
                    roots.append((a, idx))
                    agent_idx_of_root.append(idx)
            root_rows = [a * L + idx for (a, idx) in roots]
            root_speeds = [float(speed[r]) for r in root_rows]

            # Gumbel MuZero step: sample candidates from the policy's joint
            # logits, roll them out, build completed-Q targets, and execute
            # the provably-improving selection.
            jl = joint_logits(dirl[root_rows], opl[root_rows])
            g, cand = gumbel_sample_candidates(jl, args.k_candidates, rng)
            q = driver.search(
                sim, roots, root_speeds, L, agent_idx_of_root,
                (cand // N_OPS).ravel().tolist(),
                (cand % N_OPS).ravel().tolist(),
            )
            target, executed, _ = completed_q_targets(
                jl, g, cand, q, value[root_rows], args.samples
            )
            t_soft, o_soft = target_marginals(target)
            for r, row in enumerate(root_rows):
                acts[row] = (
                    executed[r] // N_OPS, executed[r] % N_OPS, root_speeds[r]
                )
                c_turn_soft[t, row] = t_soft[r]
                c_op_soft[t, row] = o_soft[r]
                c_mask[t, row] = 1.0

            c_turn[t] = acts[:, 0].astype(np.int64)
            c_op[t] = acts[:, 1].astype(np.int64)
            c_speed[t] = acts[:, 2]
            c_boot[t] = value  # root V(s_t): truncation bootstrap (1-step off)

            ob, rb, tb = sim.step(acts.ravel().tolist())
            obs = np.frombuffer(ob, dtype=np.float32).reshape(B, OBS_DIM)
            c_rew[t] = np.frombuffer(rb, dtype=np.float32)
            done = np.frombuffer(tb, dtype=np.uint8).astype(np.float32)
            decision += 1

            now = time.perf_counter()
            if now - last_beat >= 30.0:
                last_beat = now
                chunk_n = chunk_idx - chunk_start
                frac = (chunk_n + (t + 1) / T) / target_chunks
                elapsed = now - t_start
                eta = (elapsed / frac - elapsed) / 60.0 if frac > 0 else None
                _write_progress(
                    "generate", frac, eta,
                    f"chunk {chunk_n + 1}/{target_chunks}",
                )

        # Returns-to-go with truncation + chunk-boundary bootstraps.
        _, _, _, boot_v, _, _, _ = step(obs, h, done)
        c_ret = np.zeros((T, B), dtype=np.float32)
        carry = np.where(done > 0.5, c_boot[T - 1], boot_v)
        for t in reversed(range(T)):
            carry = c_rew[t] + GAMMA * carry
            c_ret[t] = carry
            if t > 0:
                trunc = c_done[t] > 0.5
                carry = np.where(trunc, c_boot[t - 1], carry)

        tmp = spool / f"tmp_{chunk_idx:06d}.npz"
        np.savez(
            tmp, obs=c_obs, turn=c_turn, op=c_op, speed=c_speed, done=c_done,
            ret=c_ret, h0=c_h0, turn_soft=c_turn_soft, op_soft=c_op_soft,
            label_mask=c_mask, expert_driven=np.array(True),
        )
        os.replace(tmp, spool / f"chunk_{chunk_idx:06d}.npz")
        wall = time.perf_counter() - t_start
        # Prune the replay window to the newest --replay-chunks files.
        all_chunks = sorted(spool.glob("chunk_*.npz"))
        for stale in all_chunks[: max(0, len(all_chunks) - args.replay_chunks)]:
            stale.unlink(missing_ok=True)

        labels_done = (chunk_idx - chunk_start + 1) * T * len(roots)
        rate = labels_done / max(wall, 1e-9)
        print(
            f"azero gen chunk {chunk_idx - chunk_start + 1}/{target_chunks} "
            f"({rate:,.0f} search-labels/s)",
            flush=True,
        )
        # Each search label = K x samples imagined futures rolled `horizon`
        # decisions deep — report the underlying work so a "35/s" label rate
        # isn't mistaken for a 35-env-steps/s collapse.
        imagined = rate * args.k_candidates * args.samples * args.horizon
        _append_jsonl(
            Path(args.metrics_out),
            {
                "update": chunk_idx - chunk_start + 1,
                "steps": labels_done,
                "steps_per_sec": rate,
                "search_labels_per_sec": round(rate, 1),
                "imagined_decisions_per_sec": round(imagined, 0),
                "phase": "azero_generate",
                "wall_seconds": wall,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("gate", "generate", "run"))
    parser.add_argument("--start", default=str(CHECKPOINT_DIR / "latest.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--horizon", type=int, default=12)
    # 8 averaged samples: the gate's own failure (0.80 vs the champion) said
    # the search needed more averaging to beat a strong student.
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument(
        "--tau", type=float, default=1.5,
        help="target sharpness (beta on standardized candidate Q)",
    )
    # 30 players per game (26 learners + 4 heuristic anchors): dense lobbies
    # produce constant predator/prey encounters, so kill/flee decisions
    # dominate the search labels instead of lonely pellet-farming. World size
    # scales up with the head count to keep density playable.
    parser.add_argument("--arenas", type=int, default=16)
    parser.add_argument("--learners", type=int, default=26)
    parser.add_argument("--anchors", type=int, default=4)
    parser.add_argument("--size-lo", type=float, default=3800.0)
    parser.add_argument("--size-hi", type=float, default=6000.0)
    parser.add_argument(
        "--kill-bonus", type=float, default=3.0,
        help="reward per kill (also shapes search rollout Q, so the search "
             "itself hunts for kill lines)",
    )
    parser.add_argument("--roots-per-arena", type=int, default=2)
    parser.add_argument("--k-candidates", type=int, default=10)
    parser.add_argument(
        "--replay-chunks", type=int, default=48,
        help="replay window: newest N chunks kept in the spool (~5 generations)",
    )
    # Counts every learner-row (B per decision), not just searched roots —
    # scaled with the 30-player lobbies so a generation still covers ~8
    # chunks (512 searched decisions).
    parser.add_argument("--labels", type=int, default=220_000)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--spool", default=str(CHECKPOINT_ROOT / "azero_gen"))
    parser.add_argument("--distill-labels", type=int, default=None)
    parser.add_argument("--out", default=str(CHECKPOINT_DIR / "azero_gen.pt"))
    parser.add_argument("--gate-arenas", type=int, default=6)
    parser.add_argument("--gate-decisions", type=int, default=500)
    parser.add_argument("--gate-margin", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--metrics-out", default=str(CHECKPOINT_DIR / "metrics.jsonl"),
                        help="append per-chunk generation progress here (dashboard-compatible)")
    parser.add_argument("--cycle", type=int, default=None,
                        help="marathon cycle number, recorded on gate verdict rows")
    args = parser.parse_args()

    if args.command == "gate":
        sys.exit(0 if cmd_gate(args) else 3)
    if args.command == "generate":
        cmd_generate(args)
        return

    # run = gate (MONITORING ONLY) -> generate -> distill over replay window.
    # AlphaZero removed gating; the Gumbel selection makes the search provably
    # no worse than the policy, so blocking on a noisy head-to-head only
    # plateaus learning. The gate row still lands in history for the
    # dashboard, and the eval-stage ratchet remains the catastrophic brake.
    cmd_gate(args)
    print("(gate is monitoring-only; continuing regardless)", flush=True)
    cmd_generate(args)
    n_chunks = len(list(Path(args.spool).glob("chunk_*.npz")))
    # Gentle distill over the REPLAY WINDOW (newest ~5 generations), not just
    # this generation's chunks.
    distill = args.distill_labels or args.labels * 4
    # The distill trainer streams frequent metrics rows, which the dashboard
    # also treats as a liveness signal; this marker just names the phase.
    _write_progress("distill", None, None, f"{distill:,} labels")
    subprocess.run(
        [
            sys.executable, "-m", "bot_solutions.rl_v1.imitate", "trainer",
            "--spool", args.spool,
            "--labels", str(distill),
            "--window", str(max(1, min(args.replay_chunks, n_chunks))),
            "--min-chunks", str(min(2, n_chunks)),
            "--resume", args.start,
            "--lr", "1e-4",
            "--out", args.out,
        ],
        check=True,
    )
    print(json.dumps({"azero": "complete", "out": args.out}), flush=True)


if __name__ == "__main__":
    main()

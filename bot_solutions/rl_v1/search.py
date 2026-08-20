"""Rollout search: imagine candidate futures in cloned Rust worlds, score
them with accumulated reward + the value head, commit to the best first move.

This is the search half of Expert Iteration (AlphaZero's principle at laptop
scale): the policy proposes how futures unfold, the simulator verifies, and
the winning first action becomes both the played move and a training label.

Candidates per decision: each of the 8 turns at policy speed, plus
"split now" and "hover". All candidates roll forward in lockstep — one
fast_clone'd world each, policy forwards batched across candidates — for
`horizon` decisions, then the value head prices the final state.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .env import DT
from .obs import N_TURNS, OBS_DIM, action_to_target, apply_turn

GAMMA = 0.995  # must match ppo.py --gamma; shaping constants come from env.cfg


class RolloutSearcher:
    def __init__(self, policy, horizon: int = 8, include_split: bool = True) -> None:
        self.policy = policy
        self.horizon = horizon
        # (turn, op) first-move candidates; speed comes from the policy.
        self.cands: list[tuple[int, int, float | None]] = [
            (turn, 0, None) for turn in range(N_TURNS)
        ]
        if include_split:
            self.cands.append((0, 1, None))  # split straight ahead
        self.cands.append((0, 0, 0.1))  # hover

    @torch.no_grad()
    def plan(
        self,
        env,
        agent_ids: list[str],
        hiddens: torch.Tensor,  # (n_agents, HIDDEN) — post-root-obs GRU states
        root_speed: np.ndarray,  # (n_agents,) policy speed at the root, from
        # the caller's hidden-state advance (so the root obs is consumed once)
    ) -> np.ndarray:
        """Returns chosen actions (n_agents, 3): [turn, op, speed]."""
        policy = self.policy
        n_a = len(agent_ids)
        n_c = len(self.cands)
        n = n_a * n_c
        frame_skip = env.cfg.frame_skip
        mass_scale = env.cfg.mass_reward_scale
        kill_bonus = env.cfg.kill_bonus
        death_penalty = env.cfg.death_penalty

        worlds = [env.world.fast_clone() for _ in range(n)]
        sim_now = [env.now] * n
        headings = np.empty(n, dtype=np.int64)
        prev = [(0, 0, 1.0)] * n
        speeds = np.empty(n, dtype=np.float64)
        first_actions = np.zeros((n, 3), dtype=np.float64)
        h = hiddens.repeat_interleave(n_c, dim=0).clone()  # (n, HIDDEN)
        score = np.zeros(n, dtype=np.float64)
        no_done = torch.zeros(n)
        base_sqrt = np.zeros(n, dtype=np.float64)
        base_kills = np.zeros(n, dtype=np.int64)
        base_deaths = np.zeros(n, dtype=np.int64)

        def agent_of(k: int) -> str:
            return agent_ids[k // n_c]

        # The nearest opponent in each imagined future is driven by the
        # policy too — otherwise victims never dodge and the search labels
        # "split at them" as a free win (observed as 11%+ split-spam after
        # distillation). One resisting opponent keeps the futures honest.
        opp_of: dict[str, str | None] = {}
        rows = {r[0]: r[8] for r in env.world.players_compact()}
        for pid in agent_ids:
            blobs = rows.get(pid) or []
            if not blobs:
                opp_of[pid] = None
                continue
            total = sum(b[3] for b in blobs)
            cx = sum(b[1] * b[3] for b in blobs) / total
            cy = sum(b[2] * b[3] for b in blobs) / total
            best_pid, best_d = None, float("inf")
            for other, oblobs in rows.items():
                if other == pid or not oblobs:
                    continue
                ot = sum(b[3] for b in oblobs)
                ox = sum(b[1] * b[3] for b in oblobs) / ot
                oy = sum(b[2] * b[3] for b in oblobs) / ot
                d = (ox - cx) ** 2 + (oy - cy) ** 2
                if d < best_d:
                    best_pid, best_d = other, d
            opp_of[pid] = best_pid
        opp_h = self.policy.initial_state(n)
        opp_heading = np.zeros(n, dtype=np.int64)
        opp_prev: list[tuple[int, int, float]] = [(0, 0, 1.0)] * n
        # Seed the opponent's REAL control state where known — a fabricated
        # heading makes its first imagined dodge arbitrary.
        for k in range(n):
            opp = opp_of.get(agent_of(k))
            if opp is not None:
                opp_heading[k] = env._headings.get(opp, 0)
                opp_prev[k] = env._prev_action.get(opp, (0, 0, 1.0))

        # Apply the candidate first moves.
        for k in range(n):
            a, c = divmod(k, n_c)
            pid = agent_ids[a]
            turn, op, speed_override = self.cands[c]
            speed = float(root_speed[a]) if speed_override is None else speed_override
            heading = apply_turn(env._headings.get(pid, 0), turn)
            headings[k] = heading
            speeds[k] = speed
            prev[k] = (turn, op, speed)
            first_actions[k] = (turn, op, speed)
            self._drive(worlds[k], pid, heading, speed, op, env.size)
            masses = {p: m for p, m, _ in worlds[k].player_masses()}
            base_sqrt[k] = math.sqrt(max(masses.get(pid, 0.0), 1.0))
            base_kills[k] = dict(worlds[k].kill_counts()).get(pid, 0)
            base_deaths[k] = dict(worlds[k].death_counts()).get(pid, 0)

        obs = np.zeros((n, OBS_DIM), dtype=np.float32)
        discount = 1.0
        for step in range(self.horizon):
            # Advance every candidate world. Death is CONTINUING (penalty then
            # respawn-and-keep-playing), exactly matching the env the value
            # head is trained on — an absorbing-death convention here would
            # price risk on a different scale than the value head does.
            for k in range(n):
                for _ in range(frame_skip):
                    sim_now[k] += DT
                    worlds[k].update(DT, sim_now[k])
                pid = agent_of(k)
                masses = {p: m for p, m, _ in worlds[k].player_masses()}
                kills = dict(worlds[k].kill_counts()).get(pid, 0)
                deaths = dict(worlds[k].death_counts()).get(pid, 0)
                new_sqrt = math.sqrt(max(masses.get(pid, 0.0), 1.0))
                if deaths > base_deaths[k]:
                    reward = -death_penalty
                else:
                    reward = (new_sqrt - base_sqrt[k]) * mass_scale
                reward += kill_bonus * (kills - base_kills[k])
                score[k] += discount * reward
                base_sqrt[k] = new_sqrt
                base_kills[k] = kills
                base_deaths[k] = deaths

            if step == self.horizon - 1:
                break
            # Policy continues each future (argmax), batched — and drives the
            # nearest opponent so the future fights back.
            opp_obs = np.zeros((n, OBS_DIM), dtype=np.float32)
            for k in range(n):
                pid = agent_of(k)
                ctl = [(int(headings[k]) % 16, prev[k][0], prev[k][1], prev[k][2])]
                buf = worlds[k].observe([pid], sim_now[k], ctl)
                obs[k] = np.frombuffer(buf, dtype=np.float32)
                opp = opp_of.get(pid)
                if opp is not None:
                    octl = [(int(opp_heading[k]) % 16, opp_prev[k][0], opp_prev[k][1], opp_prev[k][2])]
                    obuf = worlds[k].observe([opp], sim_now[k], octl)
                    opp_obs[k] = np.frombuffer(obuf, dtype=np.float32)
            turn_t, op_t, speed_t, _, _, h = policy.act(
                torch.as_tensor(obs), h, no_done,
                deterministic=True, op_deterministic=True,
            )
            o_turn, o_op, o_speed, _, _, opp_h = policy.act(
                torch.as_tensor(opp_obs), opp_h, no_done,
                deterministic=True, op_deterministic=True,
            )
            for k in range(n):
                pid = agent_of(k)
                turn, op = int(turn_t[k]), int(op_t[k])
                speed = float(speed_t[k])
                headings[k] = apply_turn(int(headings[k]), turn)
                prev[k] = (turn, op, speed)
                self._drive(worlds[k], pid, int(headings[k]), speed, op, env.size)
                opp = opp_of.get(pid)
                if opp is not None:
                    ot, oo, osp = int(o_turn[k]), int(o_op[k]), float(o_speed[k])
                    opp_heading[k] = apply_turn(int(opp_heading[k]), ot)
                    opp_prev[k] = (ot, oo, osp)
                    self._drive(worlds[k], opp, int(opp_heading[k]), osp, oo, env.size)
            discount *= GAMMA

        # Price the final states with the value head.
        for k in range(n):
            pid = agent_of(k)
            ctl = [(int(headings[k]), prev[k][0], prev[k][1], prev[k][2])]
            buf = worlds[k].observe([pid], sim_now[k], ctl)
            obs[k] = np.frombuffer(buf, dtype=np.float32)
        _, _, _, _, values, _ = self.policy.act(
            torch.as_tensor(obs), h, no_done, deterministic=True,
            op_deterministic=True,
        )
        score += discount * GAMMA * values.numpy()

        chosen = score.reshape(n_a, n_c).argmax(axis=1)
        return np.stack([first_actions[a * n_c + c] for a, c in enumerate(chosen)])

    @staticmethod
    def _drive(world, pid: str, heading: int, speed: float, op: int, size: float) -> None:
        rows = {r[0]: r[8] for r in world.players_compact()}
        blobs = rows.get(pid)
        if not blobs:
            return
        total = sum(b[3] for b in blobs)
        cx = sum(b[1] * b[3] for b in blobs) / total
        cy = sum(b[2] * b[3] for b in blobs) / total
        tx, ty = action_to_target(heading, cx, cy, speed)
        world.set_input(
            pid,
            min(max(tx, 0.0), size),
            min(max(ty, 0.0), size),
            split=op == 1,
            eject=op == 2,
        )

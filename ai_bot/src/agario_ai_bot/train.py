"""Self-play trainer for the tiny policy/value + search bot."""

from __future__ import annotations

import argparse
import random
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from ._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from agario.world import GameWorld

from .action_space import ActionSpace
from .io import create_device, load_or_init_model, save_model
from .observation import OBSERVATION_SIZE
from .planner import PUCTPlanner
from .settings import AiBotSettings


@dataclass(slots=True)
class ReplayItem:
    obs: np.ndarray
    pi: np.ndarray
    z: float


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._buffer: deque[ReplayItem] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, item: ReplayItem) -> None:
        self._buffer.append(item)

    def sample(self, batch_size: int) -> list[ReplayItem]:
        return random.sample(self._buffer, k=batch_size)


class SelfPlayTrainer:
    def __init__(
        self,
        *,
        settings: AiBotSettings,
        episodes: int,
        episode_steps: int,
        num_players: int,
        batch_size: int,
        gradient_steps: int,
        learning_rate: float,
        replay_capacity: int,
        save_every: int,
        model_path: str,
        seed: int,
        search_workers: int,
    ) -> None:
        self.settings = settings
        self.episodes = episodes
        self.episode_steps = episode_steps
        self.num_players = num_players
        self.batch_size = batch_size
        self.gradient_steps = gradient_steps
        self.save_every = max(1, save_every)
        self.model_path = model_path
        self.seed = seed
        self.search_workers = max(1, search_workers)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.action_space = ActionSpace(
            direction_bins=settings.direction_bins,
            magnitude_bins=settings.magnitude_bins,
            target_distance=settings.default_target_distance,
        )
        self.device = create_device(settings.device_preference)
        self.search_device = create_device(settings.planner_device_preference)
        self._configure_torch_backends()
        self.model = load_or_init_model(
            model_path=model_path,
            obs_size=OBSERVATION_SIZE,
            action_count=self.action_space.size,
            device=self.device,
            hidden_size=256,
        )
        if self.search_device.type == self.device.type:
            self.search_model = self.model
        else:
            self.search_model = load_or_init_model(
                model_path=model_path,
                obs_size=OBSERVATION_SIZE,
                action_count=self.action_space.size,
                device=self.search_device,
                hidden_size=256,
            )
            self.search_model.load_state_dict(self.model.state_dict())
            self.search_model.eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.replay = ReplayBuffer(capacity=replay_capacity)
        self.planner = PUCTPlanner(
            model=self.search_model,
            action_space=self.action_space,
            device=self.search_device,
            settings=settings,
        )
        if self.search_workers > 1 and self.search_device.type == "cpu":
            self._search_pool: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=self.search_workers)
        else:
            self._search_pool = None
            if self.search_workers > 1:
                print(
                    "search-workers>1 requested, but parallel search is only enabled when planner-device=cpu. "
                    "Falling back to single-threaded search.",
                    flush=True,
                )

    def train(self) -> None:
        print(
            f"starting training episodes={self.episodes} players={self.num_players} "
            f"sims={self.settings.simulations} device={self.device.type} "
            f"planner_device={self.search_device.type} search_workers={self.search_workers}",
            flush=True,
        )
        try:
            episode_bar = tqdm(
                range(1, self.episodes + 1),
                desc="training",
                dynamic_ncols=True,
            )
            for episode in episode_bar:
                self._sync_search_model()
                stats = self._run_episode(episode)
                loss = self._run_updates()
                if episode % self.save_every == 0:
                    save_model(self.model, self.model_path)
                    saved = "yes"
                else:
                    saved = "no"
                episode_bar.set_postfix(
                    buffer=len(self.replay),
                    mean_return=f"{stats['mean_return']:.3f}",
                    mean_mass=f"{stats['mean_mass']:.1f}",
                    loss=f"{loss:.4f}",
                    saved=saved,
                )
                print(
                    f"[episode {episode:04d}] "
                    f"buffer={len(self.replay)} "
                    f"mean_return={stats['mean_return']:.3f} "
                    f"mean_mass={stats['mean_mass']:.1f} "
                    f"loss={loss:.4f}"
                )

            episode_bar.close()
            save_model(self.model, self.model_path)
            print(f"saved model to {self.model_path}")
        finally:
            if self._search_pool is not None:
                self._search_pool.shutdown(wait=True)

    def _configure_torch_backends(self) -> None:
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True

    def _sync_search_model(self) -> None:
        if self.search_model is self.model:
            return
        self.search_model.load_state_dict(self.model.state_dict())
        self.search_model.eval()

    def _run_episode(self, episode_index: int) -> dict[str, float]:
        self.model.eval()
        world = GameWorld(seed=self.seed + random.randint(0, 1_000_000))
        now = 0.0
        dt = 1.0 / 20.0

        player_ids: list[str] = []
        for idx in range(self.num_players):
            player = world.add_player(player_name=f"AI-{idx + 1}", now=now, is_bot=True, bot_plugin="ai_search")
            player_ids.append(player.id)

        trajectory: dict[str, list[tuple[np.ndarray, np.ndarray, float]]] = defaultdict(list)

        step_iter = tqdm(
            range(self.episode_steps),
            desc=f"episode {episode_index}/{self.episodes}",
            leave=False,
            dynamic_ncols=True,
        )
        for _ in step_iter:
            actions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            mass_before = {
                pid: world.players[pid].total_mass
                for pid in player_ids
                if pid in world.players
            }
            alive_pids = [pid for pid in player_ids if pid in world.players]
            planned: list[tuple[str, tuple[float, float, bool, bool], np.ndarray, np.ndarray]] = []
            if self._search_pool is not None and len(alive_pids) > 1:
                futures = [
                    self._search_pool.submit(self._plan_action, world, pid, now, dt)
                    for pid in alive_pids
                ]
                for future in futures:
                    planned.append(future.result())
            else:
                for pid in alive_pids:
                    planned.append(self._plan_action(world, pid, now, dt))

            for pid, world_input, obs, pi in planned:
                tx, ty, split, eject = world_input
                world.set_input(
                    player_id=pid,
                    target_x=tx,
                    target_y=ty,
                    split=split,
                    eject=eject,
                )
                actions[pid] = (obs, pi)

            now += dt
            world.update(dt=dt, now=now)

            for pid, (obs, pi) in actions.items():
                player = world.players.get(pid)
                if player is None:
                    reward = -1.0
                    trajectory[pid].append((obs, pi, reward))
                    continue
                before = mass_before.get(pid, 1.0)
                after = player.total_mass
                mass_gain = (after - max(1.0, before)) / max(1.0, before)
                alive = 0.1 if player.blobs else -1.0
                reward = float(np.clip(0.85 * mass_gain + 0.15 * alive, -1.0, 1.0))
                trajectory[pid].append((obs, pi, reward))

        episode_returns: list[float] = []
        final_masses: list[float] = []
        for pid in player_ids:
            steps = trajectory.get(pid, [])
            g = 0.0
            for obs, pi, reward in reversed(steps):
                g = reward + self.settings.discount * g
                self.replay.add(ReplayItem(obs=obs, pi=pi, z=float(np.clip(g, -1.0, 1.0))))
            episode_returns.append(g)
            player = world.players.get(pid)
            final_masses.append(player.total_mass if player else 0.0)

        return {
            "mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
            "mean_mass": float(np.mean(final_masses)) if final_masses else 0.0,
        }

    def _plan_action(
        self,
        world: GameWorld,
        player_id: str,
        now: float,
        dt: float,
    ) -> tuple[str, tuple[float, float, bool, bool], np.ndarray, np.ndarray]:
        result = self.planner.search(
            world=world,
            player_id=player_id,
            now=now,
            dt=dt,
            training=True,
        )
        tx, ty, split, eject = self.planner.action_to_world_input(
            world=world,
            player_id=player_id,
            action_index=result.action_index,
        )
        return (player_id, (tx, ty, split, eject), result.obs_vector, result.visit_probs)

    def _run_updates(self) -> float:
        if len(self.replay) < self.batch_size:
            return 0.0

        self.model.train()
        last_loss = 0.0
        for _ in range(self.gradient_steps):
            batch = self.replay.sample(self.batch_size)
            obs = torch.from_numpy(np.stack([item.obs for item in batch])).to(self.device)
            pi = torch.from_numpy(np.stack([item.pi for item in batch])).to(self.device)
            z = torch.tensor([item.z for item in batch], dtype=torch.float32, device=self.device)

            logits, values = self.model(obs)
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            policy_loss = -(pi * log_probs).sum(dim=-1).mean()
            value_loss = nn.functional.mse_loss(values, z)
            entropy = -(probs * log_probs).sum(dim=-1).mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            last_loss = float(loss.item())

        self.model.eval()
        return last_loss


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train tiny AlphaZero-style Agar bot.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--episode-steps", type=int, default=280)
    parser.add_argument("--players", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gradient-steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--replay-capacity", type=int, default=120_000)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--model-path", type=str, default="ai_bot/models/policy_value.pt")
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--horizon-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "mps", "cuda"])
    parser.add_argument("--planner-device", type=str, default=None, choices=["cpu", "mps", "cuda"])
    parser.add_argument("--max-considered-actions", type=int, default=None)
    parser.add_argument("--search-workers", type=int, default=1)
    parser.add_argument("--gpu-rollout", dest="gpu_rollout", action="store_true")
    parser.add_argument("--no-gpu-rollout", dest="gpu_rollout", action="store_false")
    parser.set_defaults(gpu_rollout=None)
    parser.add_argument("--gpu-rollout-food-limit", type=int, default=None)
    parser.add_argument("--gpu-rollout-ejected-limit", type=int, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    settings = AiBotSettings.from_env()
    if args.simulations is not None:
        settings = replace(settings, simulations=max(4, args.simulations))
    if args.horizon_steps is not None:
        settings = replace(settings, horizon_steps=max(1, args.horizon_steps))
    if args.device is not None:
        settings = replace(settings, device_preference=args.device)
    if args.planner_device is not None:
        settings = replace(settings, planner_device_preference=args.planner_device)
    if args.max_considered_actions is not None:
        settings = replace(settings, max_considered_actions=max(4, args.max_considered_actions))
    if args.gpu_rollout is not None:
        settings = replace(settings, use_gpu_rollout=bool(args.gpu_rollout))
    if args.gpu_rollout_food_limit is not None:
        settings = replace(settings, gpu_rollout_food_limit=max(32, args.gpu_rollout_food_limit))
    if args.gpu_rollout_ejected_limit is not None:
        settings = replace(settings, gpu_rollout_ejected_limit=max(16, args.gpu_rollout_ejected_limit))

    model_path = str(Path(args.model_path))
    trainer = SelfPlayTrainer(
        settings=settings,
        episodes=args.episodes,
        episode_steps=args.episode_steps,
        num_players=args.players,
        batch_size=args.batch_size,
        gradient_steps=args.gradient_steps,
        learning_rate=args.lr,
        replay_capacity=args.replay_capacity,
        save_every=args.save_every,
        model_path=model_path,
        seed=args.seed,
        search_workers=max(1, args.search_workers),
    )
    trainer.train()


if __name__ == "__main__":
    main()

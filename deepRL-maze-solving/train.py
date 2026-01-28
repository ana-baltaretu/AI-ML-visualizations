from __future__ import annotations

import logging
import os
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env import (
    ACTION_DELTAS,
    DOWN,
    LEFT,
    RIGHT,
    UP,
    MazeConfig,
    MazeEnv,
)
from model import DQN, build_dqn
from replay import ReplayBuffer, Transition
from utils import TrainConfig, epsilon_by_step, get_device, size_bucket


NUM_ACTIONS = 4


def _build_env_from_train_cfg(
    cfg: TrainConfig, rng: np.random.RandomState
) -> MazeEnv:
    """Create a MazeEnv with randomized size between min and max."""
    h = rng.randint(cfg.maze_size_min, cfg.maze_size_max + 1)
    w = rng.randint(cfg.maze_size_min, cfg.maze_size_max + 1)
    maze_cfg = MazeConfig(
        height=h,
        width=w,
        max_steps_factor=cfg.max_steps_factor,
        wall_density=float(rng.uniform(0.15, 0.35)),
        seed=int(rng.randint(0, 1_000_000)),
        obs_mode=cfg.obs_mode,
        view_size=cfg.view_size,
    )
    return MazeEnv(maze_cfg)


def _update_history(
    history: List[Tuple[float, float, float, float]],
    mouse_norm: Tuple[float, float],
    goal_dir: Tuple[float, float],
    history_len: int,
) -> np.ndarray:
    """Append new step to history and return fixed-length array [T, F]."""
    my_norm, mx_norm = mouse_norm
    dy, dx = goal_dir
    history.append((my_norm, mx_norm, dy, dx))
    if len(history) > history_len:
        history.pop(0)
    padded = np.zeros((history_len, 4), dtype=np.float32)
    padded[-len(history) :] = np.asarray(history, dtype=np.float32)
    return padded


def _select_action(
    step_idx: int,
    train_cfg: TrainConfig,
    device: torch.device,
    policy_net: DQN,
    obs: np.ndarray,
    goal_dir: Tuple[float, float],
    history_arr: np.ndarray,
    rng: np.random.RandomState,
) -> int:
    """Epsilon-greedy action selection."""
    eps = epsilon_by_step(
        step_idx,
        start=train_cfg.epsilon_start,
        end=train_cfg.epsilon_end,
        decay_steps=train_cfg.epsilon_decay_steps,
    )
    if rng.rand() < eps:
        return int(rng.randint(0, NUM_ACTIONS))

    policy_net.eval()
    with torch.no_grad():
        obs_t = torch.tensor(
            obs[None, ...], dtype=torch.float32, device=device
        )
        goal_t = torch.tensor(
            [goal_dir], dtype=torch.float32, device=device
        )
        hist_t = torch.tensor(
            history_arr[None, ...], dtype=torch.float32, device=device
        )
        q_values = policy_net(obs_t, goal_t, hist_t)
        action = int(torch.argmax(q_values, dim=-1).item())
    policy_net.train()
    return action


def _compute_loss(
    batch,
    policy_net: DQN,
    target_net: DQN,
    device: torch.device,
    train_cfg: TrainConfig,
) -> torch.Tensor:
    obs, history, actions, rewards, next_obs, next_history, dones = batch

    # goal directions must be recomputed from histories (dy, dx are part of history).
    goal_curr = history[:, -1, 2:4]
    goal_next = next_history[:, -1, 2:4]

    q_values = policy_net(obs, goal_curr, history)
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = policy_net(next_obs, goal_next, next_history)
        next_actions = torch.argmax(next_q_values, dim=1)

        next_q_target = target_net(next_obs, goal_next, next_history)
        next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        target = rewards + (1.0 - dones) * train_cfg.gamma * next_q

    loss = nn.SmoothL1Loss()(q_value, target)
    return loss


def run_training(train_cfg: TrainConfig) -> None:
    """Main training loop."""
    logging.info("Starting training with config: %s", asdict(train_cfg))
    device = get_device()
    rng = np.random.RandomState(train_cfg.seed)

    env = _build_env_from_train_cfg(train_cfg, rng)
    policy_net, model_cfg = build_dqn(
        view_size=train_cfg.view_size,
        history_len=train_cfg.history_len,
        num_actions=NUM_ACTIONS,
    )
    target_net, _ = build_dqn(
        view_size=train_cfg.view_size,
        history_len=train_cfg.history_len,
        num_actions=NUM_ACTIONS,
    )
    policy_net.to(device)
    target_net.to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=train_cfg.lr)
    replay = ReplayBuffer(train_cfg.replay_capacity)

    os.makedirs("checkpoints", exist_ok=True)
    global_step = 0
    best_success_rate = 0.0
    success_by_bucket: Dict[str, List[float]] = {
        "small": [],
        "medium": [],
        "large": [],
    }

    for ep in range(1, train_cfg.episodes + 1):
        env = _build_env_from_train_cfg(train_cfg, rng)
        obs = env.reset()
        history_seq: List[Tuple[float, float, float, float]] = []

        mouse_norm, _ = env.get_normalized_positions()
        goal_dir = env.get_goal_direction()
        history_arr = _update_history(
            history_seq,
            mouse_norm=mouse_norm,
            goal_dir=goal_dir,
            history_len=train_cfg.history_len,
        )

        done = False
        ep_reward = 0.0
        steps = 0

        while not done:
            action = _select_action(
                global_step,
                train_cfg,
                device,
                policy_net,
                obs,
                goal_dir,
                history_arr,
                rng,
            )
            next_obs, reward, done, info = env.step(action)
            ep_reward += reward
            steps += 1
            global_step += 1

            mouse_norm_next, _ = env.get_normalized_positions()
            goal_dir_next = env.get_goal_direction()
            next_history_arr = _update_history(
                history_seq,
                mouse_norm=mouse_norm_next,
                goal_dir=goal_dir_next,
                history_len=train_cfg.history_len,
            )

            transition = Transition(
                obs=obs,
                history=history_arr.copy(),
                action=action,
                reward=reward,
                next_obs=next_obs,
                next_history=next_history_arr.copy(),
                done=done,
            )
            replay.push(transition)

            obs = next_obs
            mouse_norm = mouse_norm_next
            goal_dir = goal_dir_next
            history_arr = next_history_arr

            if len(replay) >= train_cfg.batch_size:
                batch = replay.sample(train_cfg.batch_size, device)
                loss = _compute_loss(
                    batch, policy_net, target_net, device, train_cfg
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()

            if global_step % train_cfg.target_update_interval == 0:
                target_net.load_state_dict(policy_net.state_dict())

        bucket = size_bucket(*info["maze_size"])
        success_by_bucket[bucket].append(1.0 if info["success"] else 0.0)

        if ep % 10 == 0:
            stats = {
                b: float(np.mean(v)) if v else 0.0
                for b, v in success_by_bucket.items()
            }
            logging.info(
                "Ep %d | reward=%.2f | steps=%d | maze=%s | bucket_stats=%s",
                ep,
                ep_reward,
                steps,
                info["maze_size"],
                stats,
            )
            avg_success = np.mean(
                [
                    s
                    for v in success_by_bucket.values()
                    for s in (v if v else [0.0])
                ]
            )
            if avg_success > best_success_rate:
                best_success_rate = avg_success
                ckpt_path = train_cfg.checkpoint
                torch.save(
                    {
                        "model_state_dict": policy_net.state_dict(),
                        "model_config": model_cfg.__dict__,
                        "train_config": asdict(train_cfg),
                    },
                    ckpt_path,
                )
                logging.info(
                    "New best checkpoint saved to %s (success=%.3f)",
                    ckpt_path,
                    best_success_rate,
                )


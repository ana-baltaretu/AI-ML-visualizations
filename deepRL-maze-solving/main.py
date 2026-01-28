from __future__ import annotations

import argparse
import logging
from typing import Optional, Tuple

import numpy as np
import torch

from env import MazeConfig, MazeEnv
from model import DQN, ModelConfig, build_dqn
from train import NUM_ACTIONS, run_training
from ui import MazeUI
from utils import TrainConfig, set_global_seeds, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep RL Maze Mouse")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "play", "manual"],
        required=True,
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--history_len", type=int, default=20)
    parser.add_argument(
        "--obs",
        type=str,
        choices=["local"],
        default="local",
        help="Observation mode (currently only 'local' supported).",
    )
    parser.add_argument("--view_size", type=int, default=7)
    parser.add_argument("--maze_w", type=int, default=None)
    parser.add_argument("--maze_h", type=int, default=None)
    parser.add_argument("--maze_size_min", type=int, default=9)
    parser.add_argument("--maze_size_max", type=int, default=25)
    parser.add_argument("--start_x", type=int, default=None)
    parser.add_argument("--start_y", type=int, default=None)
    parser.add_argument("--goal_x", type=int, default=None)
    parser.add_argument("--goal_y", type=int, default=None)
    parser.add_argument(
        "--randomize_start_goal", action="store_true", default=False
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best.pt",
        help="Path for saving/loading checkpoints.",
    )
    return parser.parse_args()


def _build_fixed_env(args: argparse.Namespace) -> MazeEnv:
    if args.maze_w is None or args.maze_h is None:
        raise ValueError(
            "maze_w and maze_h must be provided for play/manual modes."
        )
    cfg = MazeConfig(
        height=args.maze_h,
        width=args.maze_w,
        max_steps_factor=1.2,
        wall_density=0.25,
        seed=args.seed,
        obs_mode=args.obs,
        view_size=args.view_size,
    )
    return MazeEnv(cfg)


def _load_agent(
    checkpoint_path: str, device: torch.device
) -> Tuple[DQN, ModelConfig]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_cfg = ModelConfig(**ckpt["model_config"])
    agent = DQN(model_cfg).to(device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()
    return agent, model_cfg


def main() -> None:
    setup_logging()
    args = parse_args()
    set_global_seeds(args.seed)

    if args.view_size % 2 == 0:
        raise ValueError("--view_size must be an odd number.")

    if args.mode == "train":
        train_cfg = TrainConfig(
            episodes=args.episodes,
            history_len=args.history_len,
            obs_mode=args.obs,
            view_size=args.view_size,
            maze_size_min=args.maze_size_min,
            maze_size_max=args.maze_size_max,
            seed=args.seed,
            render=args.render,
            checkpoint=args.checkpoint,
        )
        run_training(train_cfg)
        return

    # play / manual
    env = _build_fixed_env(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "manual":
        ui = MazeUI(env)
        ui.run_manual(episodes=args.episodes)
    elif args.mode == "play":
        agent, model_cfg = _load_agent(args.checkpoint, device)
        ui = MazeUI(env, agent=agent, device=device, greedy=True)
        ui.run_agent(episodes=args.episodes)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()


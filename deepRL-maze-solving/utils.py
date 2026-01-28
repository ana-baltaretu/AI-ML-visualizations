from __future__ import annotations

import dataclasses
import logging
import random
from typing import Literal, Tuple

import numpy as np
import torch


def setup_logging() -> None:
    """Configure basic logging for the project."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def get_device() -> torch.device:
    """Return the best available torch.device and log it."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using CUDA/GPU for training.")
    else:
        device = torch.device("cpu")
        logging.info("CUDA not available, using CPU.")
    return device


def set_global_seeds(seed: int) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def size_bucket(h: int, w: int) -> Literal["small", "medium", "large"]:
    """Bucket maze sizes for logging."""
    area = h * w
    if area < 15 * 15:
        return "small"
    if area < 30 * 30:
        return "medium"
    return "large"


@dataclasses.dataclass
class TrainConfig:
    """Configuration parameters for training."""

    episodes: int = 500
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 1e-3
    replay_capacity: int = 50_000
    target_update_interval: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 50_000
    history_len: int = 20
    max_steps_factor: float = 1.2
    obs_mode: Literal["local", "full"] = "local"
    view_size: int = 7
    maze_size_min: int = 9
    maze_size_max: int = 25
    seed: int = 1
    render: bool = False
    checkpoint: str = "checkpoints/best.pt"


def epsilon_by_step(
    step: int, start: float, end: float, decay_steps: int
) -> float:
    """Linear epsilon decay function."""
    if step >= decay_steps:
        return end
    frac = step / float(decay_steps)
    return start + frac * (end - start)


def normalize_position(
    y: int, x: int, h: int, w: int
) -> Tuple[float, float]:
    """Normalize grid coordinates to [0, 1]."""
    return y / max(h - 1, 1), x / max(w - 1, 1)


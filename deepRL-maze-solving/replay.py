from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class Transition:
    """Single transition stored in replay buffer."""

    obs: np.ndarray
    history: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    next_history: np.ndarray
    done: bool


class ReplayBuffer:
    """Simple circular replay buffer."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: List[Transition] = []
        self.pos = 0

    def push(self, transition: Transition) -> None:
        """Add a transition to the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of transitions as tensors on the given device."""
        assert len(self.buffer) >= batch_size
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        obs = torch.tensor(
            np.stack([t.obs for t in batch]), dtype=torch.float32, device=device
        )
        history = torch.tensor(
            np.stack([t.history for t in batch]),
            dtype=torch.float32,
            device=device,
        )
        actions = torch.tensor(
            [t.action for t in batch], dtype=torch.long, device=device
        )
        rewards = torch.tensor(
            [t.reward for t in batch], dtype=torch.float32, device=device
        )
        next_obs = torch.tensor(
            np.stack([t.next_obs for t in batch]),
            dtype=torch.float32,
            device=device,
        )
        next_history = torch.tensor(
            np.stack([t.next_history for t in batch]),
            dtype=torch.float32,
            device=device,
        )
        dones = torch.tensor(
            [t.done for t in batch], dtype=torch.float32, device=device
        )
        return obs, history, actions, rewards, next_obs, next_history, dones

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.buffer)


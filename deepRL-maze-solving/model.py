from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Configuration for the DQN model."""

    obs_channels: int = 2
    view_size: int = 7
    history_len: int = 40 # 20, 40, 80, 160
    history_feat_dim: int = 4  # (norm_y, norm_x, d_y, d_x)
    rnn_hidden_dim: int = 64
    cnn_channels: int = 32
    mlp_hidden_dim: int = 128
    num_actions: int = 4


class CNNEncoder(nn.Module):
    """Encode local observation window into an embedding."""

    def __init__(self, in_channels: int, view_size: int, out_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        # Compute flattened size dynamically from view_size.
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, view_size, view_size)
            x = self._forward_convs(dummy)
            flat_dim = x.view(1, -1).shape[1]
        self.fc = nn.Linear(flat_dim, out_dim)

    def _forward_convs(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_convs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


class HistoryEncoder(nn.Module):
    """RNN encoder for past positions and goal deltas."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [B, T, F]
        _, h_n = self.gru(seq)
        return h_n[-1]


class DQN(nn.Module):
    """DQN with CNN observation encoder and GRU-based history encoder."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.cfg = config
        self.obs_encoder = CNNEncoder(
            in_channels=config.obs_channels,
            view_size=config.view_size,
            out_dim=config.cnn_channels,
        )
        # goal features: (dy, dx)
        self.goal_mlp = nn.Sequential(
            nn.Linear(2, config.mlp_hidden_dim // 2),
            nn.ReLU(),
        )
        self.history_encoder = HistoryEncoder(
            input_dim=config.history_feat_dim, hidden_dim=config.rnn_hidden_dim
        )

        combined_dim = (
            config.cnn_channels
            + config.mlp_hidden_dim // 2
            + config.rnn_hidden_dim
        )
        self.head = nn.Sequential(
            nn.Linear(combined_dim, config.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.mlp_hidden_dim, config.num_actions),
        )

    def forward(
        self,
        obs: torch.Tensor,
        goal_dir: torch.Tensor,
        history_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Q-values.

        Args:
            obs: [B, C, H, W] local observation.
            goal_dir: [B, 2] normalized (dy, dx).
            history_seq: [B, T, F] history features.
        """
        obs_emb = self.obs_encoder(obs)
        goal_emb = self.goal_mlp(goal_dir)
        hist_emb = self.history_encoder(history_seq)
        x = torch.cat([obs_emb, goal_emb, hist_emb], dim=-1)
        q = self.head(x)
        return q


def build_dqn(
    view_size: int, history_len: int, num_actions: int
) -> Tuple[DQN, ModelConfig]:
    """Helper to construct model and config."""
    cfg = ModelConfig(
        view_size=view_size,
        history_len=history_len,
        num_actions=num_actions,
    )
    model = DQN(cfg)
    return model, cfg


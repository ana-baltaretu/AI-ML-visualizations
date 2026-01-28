from __future__ import annotations

import logging
from typing import Optional, Tuple

import pygame
import torch

from env import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    MazeEnv,
)
from model import DQN


CELL_SIZE = 24


class MazeUI:
    """Pygame-based UI for manual and agent-controlled play."""

    def __init__(
        self,
        env: MazeEnv,
        agent: Optional[DQN] = None,
        device: Optional[torch.device] = None,
        greedy: bool = True,
    ) -> None:
        pygame.init()
        self.env = env
        self.agent = agent
        self.device = device
        self.greedy = greedy
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("Arial", 18)
        self._recreate_window()

    def _recreate_window(self) -> None:
        h, w = self.env.maze.shape
        self.surface = pygame.display.set_mode(
            (w * CELL_SIZE, h * CELL_SIZE + 60)
        )
        pygame.display.set_caption("Deep RL Maze Mouse")

    def _draw_triangle(
        self, surface: pygame.Surface, color, center: Tuple[int, int], size: int
    ) -> None:
        cx, cy = center
        points = [
            (cx, cy - size),
            (cx - size, cy + size),
            (cx + size, cy + size),
        ]
        pygame.draw.polygon(surface, color, points)

    def _render(self, info) -> None:
        grid = self.env.get_maze()
        h, w = grid.shape
        for y in range(h):
            for x in range(w):
                cell = grid[y, x]
                rect = pygame.Rect(
                    x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE
                )
                if cell == 1:  # wall
                    pygame.draw.rect(self.surface, (40, 40, 40), rect)
                else:
                    pygame.draw.rect(self.surface, (220, 220, 220), rect)

        mouse_pos, cheese_pos = self.env.get_positions()
        my, mx = mouse_pos
        gy, gx = cheese_pos
        # mouse
        pygame.draw.circle(
            self.surface,
            (160, 160, 160),
            (mx * CELL_SIZE + CELL_SIZE // 2, my * CELL_SIZE + CELL_SIZE // 2),
            CELL_SIZE // 3,
        )
        # cheese
        self._draw_triangle(
            self.surface,
            (250, 220, 0),
            (gx * CELL_SIZE + CELL_SIZE // 2, gy * CELL_SIZE + CELL_SIZE // 2),
            CELL_SIZE // 3,
        )

        # HUD
        hud_y = h * CELL_SIZE + 4
        text = (
            f"Step {info.get('step', 0)}/{info.get('max_steps', 0)}  "
            f"Maze {h}x{w}  "
            f"Mouse {mouse_pos}  Cheese {cheese_pos}  "
            f"Reward {info.get('reward', 0):.2f}  "
            f"Success {info.get('success', False)}"
        )
        surface_rect = pygame.Rect(0, h * CELL_SIZE, w * CELL_SIZE, 60)
        pygame.draw.rect(self.surface, (20, 20, 20), surface_rect)
        txt_surf = self.font.render(text, True, (255, 255, 255))
        self.surface.blit(txt_surf, (4, hud_y))

        pygame.display.flip()

    def _agent_action(self, obs, goal_dir, history_arr) -> int:
        assert self.agent is not None and self.device is not None
        self.agent.eval()
        with torch.no_grad():
            obs_t = torch.tensor(
                obs[None, ...], dtype=torch.float32, device=self.device
            )
            goal_t = torch.tensor(
                [goal_dir], dtype=torch.float32, device=self.device
            )
            hist_t = torch.tensor(
                history_arr[None, ...],
                dtype=torch.float32,
                device=self.device,
            )
            q_values = self.agent(obs_t, goal_t, hist_t)
            action = int(torch.argmax(q_values, dim=-1).item())
        return action

    def run_manual(self, episodes: int = 1) -> None:
        """Manual control with arrow keys."""
        for ep in range(episodes):
            obs = self.env.reset(randomize_start_goal=True)
            from train import _update_history  # lazy import to avoid cycle

            history_seq = []
            mouse_norm, _ = self.env.get_normalized_positions()
            goal_dir = self.env.get_goal_direction()
            history_arr = _update_history(
                history_seq,
                mouse_norm=mouse_norm,
                goal_dir=goal_dir,
                history_len=20,
            )
            done = False
            info = {"step": 0, "max_steps": self.env.max_steps}
            while not done:
                action = None
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            return
                        if event.key == pygame.K_UP:
                            action = UP
                        elif event.key == pygame.K_DOWN:
                            action = DOWN
                        elif event.key == pygame.K_LEFT:
                            action = LEFT
                        elif event.key == pygame.K_RIGHT:
                            action = RIGHT
                if action is None:
                    self.clock.tick(30)
                    self._render(info)
                    continue

                obs, reward, done, info = self.env.step(action)
                mouse_norm, _ = self.env.get_normalized_positions()
                goal_dir = self.env.get_goal_direction()
                history_arr = _update_history(
                    history_seq,
                    mouse_norm=mouse_norm,
                    goal_dir=goal_dir,
                    history_len=20,
                )
                self.clock.tick(30)
                self._render(info)

    def run_agent(self, episodes: int = 10) -> None:
        """Run greedy policy agent."""
        if self.agent is None or self.device is None:
            logging.error("Agent and device must be provided for AI play.")
            return
        from train import _update_history  # lazy import

        for ep in range(episodes):
            obs = self.env.reset(randomize_start_goal=True)
            history_seq = []
            mouse_norm, _ = self.env.get_normalized_positions()
            goal_dir = self.env.get_goal_direction()
            history_arr = _update_history(
                history_seq,
                mouse_norm=mouse_norm,
                goal_dir=goal_dir,
                history_len=20,
            )
            done = False
            info = {"step": 0, "max_steps": self.env.max_steps}

            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return

                action = self._agent_action(obs, goal_dir, history_arr)
                obs, reward, done, info = self.env.step(action)
                mouse_norm, _ = self.env.get_normalized_positions()
                goal_dir = self.env.get_goal_direction()
                history_arr = _update_history(
                    history_seq,
                    mouse_norm=mouse_norm,
                    goal_dir=goal_dir,
                    history_len=20,
                )
                self.clock.tick(15)
                self._render(info)


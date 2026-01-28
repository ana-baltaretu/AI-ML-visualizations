from __future__ import annotations

import logging
from typing import Optional, Tuple

import pygame
import torch

from env import DOWN, LEFT, RIGHT, UP, MazeEnv
from model import DQN


WINDOW_W = 960
WINDOW_H = 720
HUD_HEIGHT = 80


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
        self.surface = pygame.display.set_mode((WINDOW_W, WINDOW_H))
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

    def _compute_layout(self, grid) -> Tuple[int, int, int, int]:
        h, w = grid.shape
        cell_size = min(
            max((WINDOW_H - HUD_HEIGHT) // max(h, 1), 1),
            max(WINDOW_W // max(w, 1), 1),
        )
        maze_w_px = cell_size * w
        maze_h_px = cell_size * h
        offset_x = (WINDOW_W - maze_w_px) // 2
        offset_y = (WINDOW_H - HUD_HEIGHT - maze_h_px) // 2
        return cell_size, offset_x, offset_y, h

    def _render_multi(
        self,
        grid,
        mouse_positions,
        cheese_positions,
        best_index: int,
        info,
    ) -> None:
        h, w = grid.shape
        cell_size, offset_x, offset_y, _ = self._compute_layout(grid)

        self.surface.fill((0, 0, 0))
        for y in range(h):
            for x in range(w):
                cell = grid[y, x]
                rect = pygame.Rect(
                    offset_x + x * cell_size,
                    offset_y + y * cell_size,
                    cell_size,
                    cell_size,
                )
                if cell == 1:  # wall
                    pygame.draw.rect(self.surface, (40, 40, 40), rect)
                else:
                    pygame.draw.rect(self.surface, (220, 220, 220), rect)

        # cheeses (support multiple if provided)
        if not cheese_positions:
            cheese_positions = []
        for (gy, gx) in cheese_positions:
            self._draw_triangle(
                self.surface,
                (250, 220, 0),
                (
                    offset_x + gx * cell_size + cell_size // 2,
                    offset_y + gy * cell_size + cell_size // 2,
                ),
                max(cell_size // 3, 2),
            )

        # mice
        for idx, (my, mx) in enumerate(mouse_positions):
            cx = offset_x + mx * cell_size + cell_size // 2
            cy = offset_y + my * cell_size + cell_size // 2
            r = max(cell_size // 3, 2)
            label = str(idx + 1)
            if idx == best_index:
                pygame.draw.circle(self.surface, (160, 160, 160), (cx, cy), r)
                pygame.draw.circle(self.surface, (20, 20, 20), (cx, cy), r + 1, 2)
            else:
                overlay = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
                pygame.draw.circle(
                    overlay,
                    (160, 160, 160, 80),
                    (cell_size // 2, cell_size // 2),
                    r,
                )
                self.surface.blit(
                    overlay,
                    (offset_x + mx * cell_size, offset_y + my * cell_size),
                )
            # Draw label near mouse
            label_surf = self.font.render(label, True, (0, 0, 0))
            self.surface.blit(
                label_surf,
                (cx - label_surf.get_width() // 2, cy - r - label_surf.get_height()),
            )

        # labels for cheeses matching indices when possible
        for idx, (gy, gx) in enumerate(cheese_positions):
            if idx >= len(mouse_positions):
                break
            label = str(idx + 1)
            cx = offset_x + gx * cell_size + cell_size // 2
            cy = offset_y + gy * cell_size + cell_size // 2
            label_surf = self.font.render(label, True, (0, 0, 0))
            self.surface.blit(
                label_surf,
                (cx - label_surf.get_width() // 2, cy + cell_size // 3),
            )

        # HUD
        hud_rect = pygame.Rect(0, WINDOW_H - HUD_HEIGHT, WINDOW_W, HUD_HEIGHT)
        pygame.draw.rect(self.surface, (20, 20, 20), hud_rect)
        hud_y = WINDOW_H - HUD_HEIGHT + 4
        text = (
            f"Step {info.get('step', 0)}/{info.get('max_steps', 0)}  "
            f"Maze {h}x{w}  "
            f"Eps {info.get('epsilon', 0.0):.2f}  "
            f"Reward {info.get('reward', 0):.2f}  "
            f"Success {info.get('success', False)}"
        )
        txt_surf = self.font.render(text, True, (255, 255, 255))
        self.surface.blit(txt_surf, (4, hud_y))

        pygame.display.flip()

    def _render(self, info) -> None:
        grid = self.env.get_maze()
        mouse_pos, cheese_pos = self.env.get_positions()
        self._render_multi(
            grid,
            mouse_positions=[mouse_pos],
            cheese_positions=[cheese_pos],
            best_index=0,
            info=info,
        )

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


class TrainingVisualizer:
    """Helper for training-time visualization with multiple mice."""

    def __init__(self, env: MazeEnv, fps: int = 15) -> None:
        self.ui = MazeUI(env)
        self.clock = pygame.time.Clock()
        self.fps = fps

    def update(
        self,
        maze_grid,
        mouse_positions,
        cheese_positions,
        best_index: int,
        info,
    ) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
        self.ui._render_multi(
            maze_grid,
            mouse_positions=mouse_positions,
            cheese_positions=cheese_positions,
            best_index=best_index,
            info=info,
        )
        self.clock.tick(self.fps)


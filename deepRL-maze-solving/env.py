from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils import normalize_position


Action = int


UP: Action = 0
DOWN: Action = 1
LEFT: Action = 2
RIGHT: Action = 3

ACTION_DELTAS: Dict[Action, Tuple[int, int]] = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}


Cell = int
EMPTY: Cell = 0
WALL: Cell = 1
START: Cell = 2
GOAL: Cell = 3


@dataclass
class MazeConfig:
    """Configuration for maze generation and environment behaviour."""

    height: int
    width: int
    max_steps_factor: float = 1.2
    wall_density: float = 0.2
    seed: Optional[int] = None
    obs_mode: str = "local"
    view_size: int = 7


class MazeEnv:
    """2D grid maze environment with variable size and random start/goal."""

    def __init__(self, config: MazeConfig) -> None:
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.maze: np.ndarray = np.zeros(
            (config.height, config.width), dtype=np.int8
        )

        self.mouse_pos: Tuple[int, int] = (0, 0)
        self.cheese_pos: Tuple[int, int] = (0, 0)
        self.t: int = 0
        self.max_steps: int = int(
            config.max_steps_factor * config.height * config.width
        )

        self._generate_maze()

    # ------------------------------------------------------------------
    # Maze generation
    # ------------------------------------------------------------------
    def _generate_maze(self) -> None:
        """Generate a random, fully connected maze."""
        h, w = self.maze.shape
        self.maze[:, :] = EMPTY

        # Start with random walls.
        mask = self.rng.rand(h, w) < self.config.wall_density
        self.maze[mask] = WALL

        # Ensure border is walls to keep agent inside.
        self.maze[0, :] = WALL
        self.maze[-1, :] = WALL
        self.maze[:, 0] = WALL
        self.maze[:, -1] = WALL

        # Ensure full connectivity of empty cells: all empties mutually reachable.
        self._ensure_full_connectivity()

        # Place start/goal placeholders; real positions are chosen in reset().
        self.mouse_pos = (1, 1)
        self.cheese_pos = (h - 2, w - 2)

    def _neighbors(self, y: int, x: int) -> List[Tuple[int, int]]:
        h, w = self.maze.shape
        neigh: List[Tuple[int, int]] = []
        for dy, dx in ACTION_DELTAS.values():
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                neigh.append((ny, nx))
        return neigh

    def _ensure_full_connectivity(self) -> None:
        """Modify walls so that all EMPTY cells are mutually reachable.

        We repeatedly run BFS from a random empty cell, and if some empties
        are unreachable we carve walls along random frontier locations.
        """
        h, w = self.maze.shape
        empties = [(y, x) for y in range(h) for x in range(w) if self.maze[y, x] == EMPTY]
        if not empties:
            # if everything is wall, carve a simple corridor.
            self.maze[:, :] = WALL
            for y in range(1, h - 1):
                self.maze[y, 1] = EMPTY
            for x in range(1, w - 1):
                self.maze[h - 2, x] = EMPTY
            return

        while True:
            start_y, start_x = empties[0]
            reachable = set()
            frontier = [(start_y, start_x)]
            reachable.add((start_y, start_x))
            while frontier:
                y, x = frontier.pop()
                for ny, nx in self._neighbors(y, x):
                    if self.maze[ny, nx] != WALL and (ny, nx) not in reachable:
                        reachable.add((ny, nx))
                        frontier.append((ny, nx))

            unreachable = [p for p in empties if p not in reachable]
            if not unreachable:
                break

            # For each unreachable empty, carve a path by removing a wall
            # between it and some reachable cell.
            for uy, ux in unreachable:
                # Look for neighbor that is reachable and currently a wall.
                neigh = self._neighbors(uy, ux)
                self.rng.shuffle(neigh)
                carved = False
                for ny, nx in neigh:
                    if (ny, nx) in reachable and self.maze[uy, ux] == EMPTY:
                        # Already empty but not connected; carve one of its walls instead.
                        # Try walls around uy,ux that border reachable.
                        wall_neighbors = self._neighbors(uy, ux)
                        self.rng.shuffle(wall_neighbors)
                        for wy, wx in wall_neighbors:
                            if self.maze[wy, wx] == WALL:
                                self.maze[wy, wx] = EMPTY
                                carved = True
                                break
                    elif self.maze[uy, ux] == EMPTY and self.maze[ny, nx] == WALL:
                        self.maze[ny, nx] = EMPTY
                        carved = True
                    if carved:
                        break

            # Recompute empties and loop again.
            empties = [(y, x) for y in range(h) for x in range(w) if self.maze[y, x] == EMPTY]

    # ------------------------------------------------------------------
    # Reset & step
    # ------------------------------------------------------------------
    def reset(
        self,
        start_pos: Optional[Tuple[int, int]] = None,
        goal_pos: Optional[Tuple[int, int]] = None,
        randomize_start_goal: bool = True,
    ) -> np.ndarray:
        """Reset environment with (possibly) random start and goal.

        Ensures a valid path between start and goal exists.
        """
        h, w = self.maze.shape

        if randomize_start_goal or start_pos is None or goal_pos is None:
            self.mouse_pos, self.cheese_pos = self._sample_valid_start_goal()
        else:
            self.mouse_pos = start_pos
            self.cheese_pos = goal_pos
            if not self._path_exists(self.mouse_pos, self.cheese_pos):
                # fall back to sampling if invalid.
                self.mouse_pos, self.cheese_pos = self._sample_valid_start_goal()

        self.t = 0
        return self._get_observation()

    def _sample_valid_start_goal(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Sample start and goal positions on empty cells with a valid path."""
        h, w = self.maze.shape
        empties = [(y, x) for y in range(h) for x in range(w) if self.maze[y, x] == EMPTY]
        assert empties, "Maze has no empty cells."

        while True:
            s = empties[self.rng.randint(len(empties))]
            g = empties[self.rng.randint(len(empties))]
            if s == g:
                continue
            if self._path_exists(s, g):
                return s, g

    def _path_exists(self, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        """Check if there is a path of empty cells between start and goal."""
        if start == goal:
            return True
        h, w = self.maze.shape
        sy, sx = start
        gy, gx = goal
        if self.maze[sy, sx] == WALL or self.maze[gy, gx] == WALL:
            return False

        visited = np.zeros((h, w), dtype=bool)
        frontier: List[Tuple[int, int]] = [start]
        visited[sy, sx] = True

        while frontier:
            y, x = frontier.pop(0)
            if (y, x) == goal:
                return True
            for ny, nx in self._neighbors(y, x):
                if not visited[ny, nx] and self.maze[ny, nx] != WALL:
                    visited[ny, nx] = True
                    frontier.append((ny, nx))
        return False

    def step(self, action: Action) -> Tuple[np.ndarray, float, bool, Dict]:
        """Apply an action, returning (obs, reward, done, info)."""
        self.t += 1
        y, x = self.mouse_pos
        dy, dx = ACTION_DELTAS[action]
        ny, nx = y + dy, x + dx

        h, w = self.maze.shape
        reward = -0.01  # small step penalty
        done = False

        if 0 <= ny < h and 0 <= nx < w and self.maze[ny, nx] != WALL:
            self.mouse_pos = (ny, nx)
        else:
            # collision penalty
            reward -= 0.05

        if self.mouse_pos == self.cheese_pos:
            reward += 1.0
            done = True
        elif self.t >= self.max_steps:
            done = True

        obs = self._get_observation()
        info = {
            "step": self.t,
            "max_steps": self.max_steps,
            "maze_size": self.maze.shape,
            "mouse_pos": self.mouse_pos,
            "cheese_pos": self.cheese_pos,
            "reward": reward,
            "success": self.mouse_pos == self.cheese_pos,
        }
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        """Return observation; currently only local window is implemented."""
        if self.config.obs_mode == "local":
            return self._get_local_observation()
        raise NotImplementedError("Full-grid observation mode not implemented.")

    def _get_local_observation(self) -> np.ndarray:
        """Return local window observation around the mouse.

        Channels:
            0: walls (1) vs empty (0)
            1: goal-in-window hint (1 where goal is, else 0)
        Global features (dx, dy) are appended separately in the model.
        """
        view = self.config.view_size
        assert view % 2 == 1, "view_size must be odd"
        radius = view // 2

        h, w = self.maze.shape
        my, mx = self.mouse_pos
        gy, gx = self.cheese_pos

        # First channel: walls vs empty in window with padding.
        wall_channel = np.ones((view, view), dtype=np.float32)
        goal_channel = np.zeros((view, view), dtype=np.float32)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                wy, wx = my + dy, mx + dx
                vy, vx = dy + radius, dx + radius
                if 0 <= wy < h and 0 <= wx < w:
                    wall_channel[vy, vx] = 1.0 if self.maze[wy, wx] == WALL else 0.0
                    if (wy, wx) == (gy, gx):
                        goal_channel[vy, vx] = 1.0

        obs = np.stack([wall_channel, goal_channel], axis=0)
        return obs

    # ------------------------------------------------------------------
    # Helpers for UI
    # ------------------------------------------------------------------
    def get_maze(self) -> np.ndarray:
        """Return the maze grid with start/goal markers for rendering."""
        grid = self.maze.copy()
        sy, sx = self.mouse_pos
        gy, gx = self.cheese_pos
        grid[sy, sx] = START
        grid[gy, gx] = GOAL
        return grid

    def get_positions(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return (mouse_pos, cheese_pos)."""
        return self.mouse_pos, self.cheese_pos

    def get_goal_direction(self) -> Tuple[float, float]:
        """Return normalized (dy, dx) from mouse to cheese for global features."""
        (my, mx), (gy, gx) = self.mouse_pos, self.cheese_pos
        h, w = self.maze.shape
        dy = (gy - my) / max(h - 1, 1)
        dx = (gx - mx) / max(w - 1, 1)
        return float(dy), float(dx)

    def get_normalized_positions(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return normalized mouse and cheese positions."""
        h, w = self.maze.shape
        my, mx = self.mouse_pos
        gy, gx = self.cheese_pos
        return normalize_position(my, mx, h, w), normalize_position(gy, gx, h, w)


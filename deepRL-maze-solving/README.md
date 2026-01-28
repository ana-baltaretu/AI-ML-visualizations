## Deep RL Variable-Size Maze Mouse

This project trains a Deep Q-Network (DQN) with RNN memory to solve 2D mazes of **variable size and layout**. The agent (a mouse) must reach the cheese while learning to generalize across different maze sizes, wall layouts, and start/goal positions.

### Features

- Variable-size 2D mazes (rectangular, not hard-coded).
- Random maze generation with guaranteed connectivity.
- Randomized start and goal positions per episode (with solvable path).
- Local observation window around the agent + global goal direction features.
- RNN-based memory over the last `N` steps (positions + goal direction).
- DQN with target network, Double DQN target, Huber loss, gradient clipping.
- Pygame UI for:
  - Manual play via keyboard.
  - Watching a trained agent play via greedy policy.

### Installation

```bash
pip install -r requirements.txt
```

### Training

Train a policy on randomized maze sizes with a local observation window:

```bash
python main.py --mode train --obs local --maze_size_min 5 --maze_size_max 40 --render
```

Key configurable flags (see `main.py`):

- `--episodes`: number of training episodes.
- `--history_len`: length of RNN history.
- `--view_size`: odd integer for local window size (default 7).
- `--maze_size_min`, `--maze_size_max`: min/max maze size for training.
- `--checkpoint`: path to save best model checkpoint.

### Playing with a Trained Agent

After training, run the greedy policy in a fixed-size maze:

```bash
python main.py --mode play --maze_w 30 --maze_h 30 --randomize_start_goal --checkpoint checkpoints/best.pt
```

This opens a Pygame window where you can watch the agent navigate.

### Manual Play

You can manually control the mouse using arrow keys:

```bash
python main.py --mode manual --maze_w 21 --maze_h 15
```

- Arrow keys: move up/down/left/right.
- `Esc`: quit.

### Architecture Overview

- `env.py`: `MazeEnv` implements a variable-size grid maze, random maze generation with full connectivity, random start/goal placement, and local observations.
- `model.py`: CNN encoder for local views + GRU-based history encoder + MLP head for Q-values.
- `replay.py`: Experience replay buffer with transitions including history.
- `train.py`: DQN training loop with Double DQN targets, epsilon-greedy exploration, domain randomization over maze sizes and layouts, and size-bucketed logging.
- `ui.py`: Pygame UI for manual play and visualizing the trained agent.
- `main.py`: CLI entry point that wires everything together.


import numpy as np


import os
os.environ.pop("MPLBACKEND", None)  # ignore PyCharm's override if present

import matplotlib

# Prefer a GUI backend you have installed
try:
	matplotlib.use("Qt5Agg")   # needs: pip install pyqt5
except Exception:
	try:
		matplotlib.use("TkAgg")  # Tk is usually bundled with CPython on Windows
	except Exception:
		matplotlib.use("Agg")    # headless, use savefig instead of show

import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.collections import LineCollection

def mse_gradient(X, errors, m):
	"""
	MSE = sum((predictions - true_labels)^2) / amount_of_samples
	derivative_of_MSE = MSE'
	derivative_of_MSE = 2 * (

	errors = predictions - true_labels
	"""
	#
	return 2 * (X.T @ errors) / m


def samples_gradient_descent(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, record=None):
	"""
	X: shape (m, n) with a column of ones as the first column for the intercept, m samples, n features
	"""
	m, n = X.shape
	y = y.reshape(-1, 1)		# Make sure y is a column vector
	predictions = X @ weights                 	# 2. Compute predictions h(x)
	errors = predictions - y                	# 3. Compute residuals
	gradient = mse_gradient(X, errors, m)  		# 4. Compute gradient
	weights -= learning_rate * gradient         # 5. Update weights for each iteration



	return weights


def linear_regression_gradient_descent(X, y, weights, learning_rate, n_iterations, batch_size=1, method='batch'):
	m, n = X.shape
	weights = weights.reshape(-1, 1)

	# histories
	loss_hist = []
	w_hist = []

	def record_fn(w_new):
		w_hist.append(w_new.ravel().copy())
		# full dataset loss so curves are comparable across methods
		# loss_hist.append(mse_gradient(X, y, X.shape[0]))

	record_fn(weights)
	# 1. Initializing the weights with 0s
	match method:
		case "batch":
			for _ in range(n_iterations):
				weights = samples_gradient_descent(X, y, weights, learning_rate)
				record_fn(weights)
			# return weights.flatten()
		case "stochastic":
			for _ in range(n_iterations):  # epochs
				for i in range(m):  # per-sample update
					weights = samples_gradient_descent(X[i:i + 1, :], y[i:i + 1], weights, learning_rate, record=record_fn)
				record_fn(weights)
			# return weights.flatten()
		case "mini_batch":
			for _ in range(n_iterations):  # epochs
				for i in range(0, m, batch_size):  # per-sample update
					weights = samples_gradient_descent(X[i:i + batch_size, :], y[i:i + batch_size], weights, learning_rate, record=record_fn)
				record_fn(weights)
			# return weights.flatten()
		case _:
			raise ValueError(f"Unknown match method: {method}")

	loss_hist = np.asarray(loss_hist)
	w_hist = np.asarray(w_hist)  # shape: (num_updates, n)
	hist = {"loss": np.asarray(loss_hist), "weights": np.asarray(w_hist)}
	return weights.flatten(), hist


def mse_value(X, y_col, w):
	e = X @ w - y_col
	return float((e.T @ e) / X.shape[0])


def _mse_grid(X, y_col, w_hist, grid_points=200, pad_ratio=0.25):
	"""Build a grid over theta1, theta2 around the path and evaluate MSE."""
	W = np.asarray(w_hist)
	if W.shape[1] != 2:
		raise ValueError("This visualization needs exactly 2 parameters.")

	# grid ranges padded around trajectory
	t1_min, t1_max = W[:, 0].min(), W[:, 0].max()
	t2_min, t2_max = W[:, 1].min(), W[:, 1].max()
	p1 = max(t1_max - t1_min, 1e-6) * pad_ratio
	p2 = max(t2_max - t2_min, 1e-6) * pad_ratio
	t1 = np.linspace(t1_min - p1, t1_max + p1, grid_points)
	t2 = np.linspace(t2_min - p2, t2_max + p2, grid_points)
	T1, T2 = np.meshgrid(t1, t2)

	# evaluate MSE on grid in vectorized form
	Wgrid = np.column_stack([T1.ravel(), T2.ravel()])            # (P^2, 2)
	E = X @ Wgrid.T - y_col                                      # (m, P^2)
	J = (E * E).mean(axis=0).reshape(T1.shape)                   # (P, P)

	return T1, T2, J



def plot_mse_surface_and_contour(
	X, y, w_hist, title_left="3D MSE surface",
	title_right="MSE contour with path",
	grid_points=200, pad_ratio=0.25,
	elev=25, azim=-60, fig_name="fig.png"
):
	"""
	X must have 2 columns so the model has exactly 2 parameters.
	w_hist: array of shape (num_updates, 2).
	"""
	X = np.asarray(X, dtype=float)
	y_col = np.asarray(y, dtype=float).reshape(-1, 1)
	W = np.asarray(w_hist)

	# Grid and surface values
	T1, T2, J = _mse_grid(X, y_col, W, grid_points, pad_ratio)

	# Path z values
	z_path = np.array([mse_value(X, y_col, w.reshape(-1, 1)) for w in W])

	fig = plt.figure(figsize=(12, 5))

	# 3D surface + red path (bright lighting)
	ax1 = fig.add_subplot(1, 2, 1, projection="3d")
	ax1.set_facecolor("white")
	ax1.xaxis.pane.set_facecolor((1, 1, 1, 1))
	ax1.yaxis.pane.set_facecolor((1, 1, 1, 1))
	ax1.zaxis.pane.set_facecolor((1, 1, 1, 1))

	ls = LightSource(azdeg=135, altdeg=65)
	facecolors = ls.shade(J, cmap=plt.cm.viridis, vert_exag=1.0, blend_mode="soft")
	ax1.plot_surface(T1, T2, J, facecolors=facecolors, linewidth=0,
					 antialiased=True, shade=False)

	ax1.plot(W[:, 0], W[:, 1], z_path, color="red", linewidth=2.2, marker="o", ms=4,  zorder=10)
	ax1.scatter(W[0, 0], W[0, 1], z_path[0], color="red", s=45,  zorder=11)      # start
	ax1.scatter(W[-1, 0], W[-1, 1], z_path[-1], color="red", s=60,  zorder=11)   # end

	ax1.set_xlabel(r"$\theta_1$")
	ax1.set_ylabel(r"$\theta_2$")
	ax1.set_zlabel("MSE")
	ax1.view_init(elev=elev, azim=azim)
	ax1.set_title(title_left)

	# Contour + red path with directional markers
	ax2 = fig.add_subplot(1, 2, 2)
	ax2.contourf(T1, T2, J, levels=40)
	ax2.contour(T1, T2, J, levels=40, linewidths=0.5)
	ax2.plot(W[:, 0], W[:, 1], color="red", linewidth=5, label="trajectory")

	# Add directional arrows on contour plot
	for i in range(len(W) - 1):
		dx = W[i + 1, 0] - W[i, 0]
		dy = W[i + 1, 1] - W[i, 1]
		ax2.quiver(W[i, 0], W[i, 1], dx, dy,
				   color="black", angles='xy', scale_units='xy', scale=1,
				   width=0.004, headwidth=6, headlength=5, headaxislength=3.5,
				   zorder=8)

	ax2.scatter(W[0, 0], W[0, 1], color="red", s=35)
	ax2.scatter(W[-1, 0], W[-1, 1], color="red", s=50)

	# Make the filled contour cover the whole axes
	ax2.set_xlim(T1.min(), T1.max())
	ax2.set_ylim(T2.min(), T2.max())
	ax2.set_aspect("auto")

	ax2.set_xlabel(r"$\theta_1$")
	ax2.set_ylabel(r"$\theta_2$")
	ax2.set_title(title_right)
	ax2.legend(loc="upper right")

	fig.suptitle("Gradient Descent trajectory on MSE", y=0.98)
	fig.tight_layout()
	if fig_name:
		plt.savefig(fig_name, dpi=800, bbox_inches="tight")


def plot_fit_evolution(
    X, y, w_hist, *,
    n_points=200, pad_ratio=0.1,
    cmap_name="plasma", alpha_start=0.12, alpha_end=0.95,
    max_lines=300, fig_name="fit_evolution.png", title="Fit evolution"
):
    """
    Visualize how the fitted line changes over updates.

    Assumes a 2-parameter linear model with an intercept column of ones
    and one feature column. Works whether X is [x, 1] or [1, x].
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    W = np.asarray(w_hist, dtype=float)
    if X.shape[1] != 2 or W.shape[1] != 2:
        raise ValueError("This plot expects exactly 2 parameters")

    # Detect which column is the feature and which is the intercept
    stds = X.std(axis=0)
    if np.isclose(stds[0], 0, atol=1e-12) and not np.isclose(stds[1], 0, atol=1e-12):
        idx_bias, idx_x = 0, 1
    elif np.isclose(stds[1], 0, atol=1e-12) and not np.isclose(stds[0], 0, atol=1e-12):
        idx_bias, idx_x = 1, 0
    else:
        # fallback: treat the column closer to ones as bias
        ones_dist = np.abs(X.mean(axis=0) - 1.0)
        idx_bias = int(np.argmin(ones_dist))
        idx_x = 1 - idx_bias

    # Build x-range for drawing lines
    x_min, x_max = X[:, idx_x].min(), X[:, idx_x].max()
    pad = (x_max - x_min) * pad_ratio if x_max > x_min else 1.0
    x_line = np.linspace(x_min - pad, x_max + pad, n_points)

    # Design matrix for the line in the same column order as X
    Xline = np.ones((n_points, 2))
    Xline[:, idx_bias] = 1.0
    Xline[:, idx_x] = x_line

    # Optionally downsample histories so the plot stays readable
    step = max(1, len(W) // max_lines)
    W_draw = W[::step]

    # Build segments and colors
    segments = []
    for w in W_draw:
        y_line = (Xline @ w.reshape(-1, 1)).ravel()
        segments.append(np.column_stack([x_line, y_line]))

    cmap = plt.get_cmap(cmap_name)
    t = np.linspace(0.0, 1.0, len(segments))
    colors = cmap(t)
    alphas = np.linspace(alpha_start, alpha_end, len(segments))
    colors[:, 3] = alphas  # fade older lines

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X[:, idx_x], y, s=40, c="black", label="data", zorder=5)

    lc = LineCollection(segments, colors=colors, linewidths=2.0)
    ax.add_collection(lc)

    # Highlight the most recent fit in red
    y_last = (Xline @ W[-1].reshape(-1, 1)).ravel()
    ax.plot(x_line, y_last, color="red", linewidth=1, label="latest fit")

    # Tidy up axes
    y_all = np.concatenate([y, segments[-1][:, 1]])
    y_pad = (y_all.max() - y_all.min()) * 0.1 if y_all.max() > y_all.min() else 1.0
    ax.set_xlim(x_line.min(), x_line.max())
    ax.set_ylim(y_all.min() - y_pad, y_all.max() + y_pad)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    if fig_name:
        plt.savefig(fig_name, dpi=800, bbox_inches="tight")


def plot_fit_evolution_animated(
		X, y, w_hist, *,
		n_points=200, pad_ratio=0.1,
		interval=50, fig_name="fit_evolution.gif", title="Fit evolution",
		show_trail=True, trail_length=10
):
	"""
    Create an animated visualization of how the fitted line changes over updates.

    Assumes a 2-parameter linear model with an intercept column of ones
    and one feature column. Works whether X is [x, 1] or [1, x].

    Parameters:
    -----------
    show_trail : bool
        If True, show a fading trail of previous fits
    trail_length : int
        Number of previous lines to show in the trail
    interval : int
        Milliseconds between frames
    """
	from matplotlib.animation import FuncAnimation, PillowWriter

	X = np.asarray(X, dtype=float)
	y = np.asarray(y, dtype=float).ravel()
	W = np.asarray(w_hist, dtype=float)
	if X.shape[1] != 2 or W.shape[1] != 2:
		raise ValueError("This plot expects exactly 2 parameters")

	# Detect which column is the feature and which is the intercept
	stds = X.std(axis=0)
	if np.isclose(stds[0], 0, atol=1e-12) and not np.isclose(stds[1], 0, atol=1e-12):
		idx_bias, idx_x = 0, 1
	elif np.isclose(stds[1], 0, atol=1e-12) and not np.isclose(stds[0], 0, atol=1e-12):
		idx_bias, idx_x = 1, 0
	else:
		# fallback: treat the column closer to ones as bias
		ones_dist = np.abs(X.mean(axis=0) - 1.0)
		idx_bias = int(np.argmin(ones_dist))
		idx_x = 1 - idx_bias

	# Build x-range for drawing lines
	x_min, x_max = X[:, idx_x].min(), X[:, idx_x].max()
	pad = (x_max - x_min) * pad_ratio if x_max > x_min else 1.0
	x_line = np.linspace(x_min - pad, x_max + pad, n_points)

	# Design matrix for the line in the same column order as X
	Xline = np.ones((n_points, 2))
	Xline[:, idx_bias] = 1.0
	Xline[:, idx_x] = x_line

	# Pre-compute all y values for all weight updates
	all_y_lines = []
	for w in W:
		y_line = (Xline @ w.reshape(-1, 1)).ravel()
		all_y_lines.append(y_line)

	# Set up the figure
	fig, ax = plt.subplots(figsize=(7, 5))
	ax.scatter(X[:, idx_x], y, s=40, c="black", label="data", zorder=5)

	# Set fixed axis limits
	y_all = np.concatenate([y] + all_y_lines)
	y_pad = (y_all.max() - y_all.min()) * 0.1 if y_all.max() > y_all.min() else 1.0
	ax.set_xlim(x_line.min(), x_line.max())
	ax.set_ylim(y_all.min() - y_pad, y_all.max() + y_pad)

	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_title(title)

	# Initialize plot elements
	line_current, = ax.plot([], [], color="red", linewidth=2.5, label="current fit", zorder=4)
	trail_lines = []
	if show_trail:
		for i in range(trail_length):
			alpha = (i + 1) / trail_length * 0.3
			line, = ax.plot([], [], color="blue", linewidth=1.5, alpha=alpha, zorder=3)
			trail_lines.append(line)

	iteration_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
							 verticalalignment='top', fontsize=10,
							 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

	ax.legend(loc="best")

	def init():
		line_current.set_data([], [])
		for line in trail_lines:
			line.set_data([], [])
		iteration_text.set_text('')
		return [line_current] + trail_lines + [iteration_text]

	def animate(frame):
		# Update current line
		line_current.set_data(x_line, all_y_lines[frame])

		# Update trail
		if show_trail:
			for i, line in enumerate(trail_lines):
				trail_idx = frame - (trail_length - i)
				if trail_idx >= 0:
					line.set_data(x_line, all_y_lines[trail_idx])
				else:
					line.set_data([], [])

		iteration_text.set_text(f'Iteration: {frame + 1}/{len(W)}')
		return [line_current] + trail_lines + [iteration_text]

	anim = FuncAnimation(fig, animate, init_func=init,
						 frames=len(W), interval=interval, blit=True)

	if fig_name:
		# Determine format from filename
		if fig_name.endswith('.gif'):
			writer = PillowWriter(fps=1000 // interval)
			anim.save(fig_name, writer=writer)
		elif fig_name.endswith('.mp4'):
			anim.save(fig_name, writer='ffmpeg', fps=1000 // interval)
		else:
			# Default to gif
			writer = PillowWriter(fps=1000 // interval)
			anim.save(fig_name, writer=writer)

	plt.close()
	return anim

if __name__ == '__main__':
	# X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
	# y = np.array([2, 3, 4, 5])

	# (2, 1), (1.5, 3), (4, 8), (3, 2), (5, 3)
	X = np.array([[2, 1], [1.5, 1], [4, 1], [3, 1], [5, 1]])
	y = np.array([1, 3, 8, 2, 3])

	w0 = np.zeros(X.shape[1])
	lr = 0.05
	# iters = 100
	iters = 49
	batch_size = 2

	# print(linear_regression_gradient_descent(np.array([[1, 1], [1, 2], [1, 3]]), np.array([1, 2, 3]), 0.01, 1000))
	w_b, hist_b = linear_regression_gradient_descent(X, y, w0, lr, iters, method='batch')

	# print(linear_regression_gradient_descent(np.array([[1, 5], [1, 7], [1, 9]]), np.array([10, 14, 18]), 0.01, 5000))
	w_s, hist_s = linear_regression_gradient_descent(X, y, w0, lr, iters, method='stochastic')

	# print(linear_regression_gradient_descent(np.array([[1, 1, 0], [1, 2, 1], [1, 3, 2], [1, 4, 3]]), np.array([5, 6, 7, 8]), 0.05, 2000))
	w_m, hist_m = linear_regression_gradient_descent(X, y, w0, lr, iters, batch_size, method='mini_batch')

	print(w_b, w_s, w_m)
	#
	# Weights over updates for Batch
	plt.figure()
	plt.plot(hist_b["weights"][:, 0], label="w0")
	plt.plot(hist_b["weights"][:, 1], label="w1")
	plt.xlabel("update step")
	plt.ylabel("weight value")
	plt.legend()
	plt.title("Weights over updates (SGD)")
	plt.savefig("plot_b.png", dpi=150)


	# Weights over updates for SGD
	plt.figure()
	plt.plot(hist_s["weights"][:, 0], label="w0")
	plt.plot(hist_s["weights"][:, 1], label="w1")
	plt.xlabel("update step")
	plt.ylabel("weight value")
	plt.legend()
	plt.title("Weights over updates (SGD)")
	plt.savefig("plot_sgd.png", dpi=150)

	# Weights over updates for Minibatch
	plt.figure()
	plt.plot(hist_m["weights"][:, 0], label="w0")
	plt.plot(hist_m["weights"][:, 1], label="w1")
	plt.xlabel("update step")
	plt.ylabel("weight value")
	plt.legend()
	plt.title("Weights over updates (Minibatch)")
	plt.savefig("plot_mb.png", dpi=150)

	# After training
	plot_mse_surface_and_contour(X, y, hist_s["weights"], title_left="SGD 3D", title_right="SGD contour", fig_name="plot_sgd2.png")
	plot_mse_surface_and_contour(X, y, hist_b["weights"], title_left="Batch 3D", title_right="Batch contour", fig_name="plot_b2.png")
	plot_mse_surface_and_contour(X, y, hist_m["weights"], title_left="Mini batch 3D", title_right="Mini batch contour", fig_name="plot_mb2.png")

	# Image approximating line evolution
	plot_fit_evolution(X, y, hist_b["weights"], fig_name="fit_batch.png", title="Batch fit evolution")
	plot_fit_evolution(X, y, hist_s["weights"], fig_name="fit_sgd.png", title="SGD fit evolution")
	plot_fit_evolution(X, y, hist_m["weights"], fig_name="fit_mb.png", title="Mini batch fit evolution")

	# Slower animation
	plot_fit_evolution_animated(X, y, hist_b["weights"], fig_name="fit_batch.gif", interval=100)
	plot_fit_evolution_animated(X, y, hist_s["weights"], fig_name="fit_sgd.gif", interval=100)
	plot_fit_evolution_animated(X, y, hist_m["weights"], fig_name="fit_mb.gif", interval=100)

	print("Done rendering!")

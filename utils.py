"""
utils.py — Utilities for ParetoComm.

Includes:
  - Hypervolume computation (2D, exact)
  - Pareto dominance check and non-dominated set extraction
  - Training metrics logger
  - Plotting functions (Pareto front, training curves)
  - Checkpoint save/load
"""

import os
import json
import math
import numpy as np
import torch
import logging
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from pathlib import Path

# Optional: matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ─── Logging setup ────────────────────────────────────────────────────────────

def setup_logger(name: str, log_file: Optional[str] = None,
                 level: int = logging.INFO) -> logging.Logger:
    """Create a logger with console + optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                            datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ─── Pareto dominance ─────────────────────────────────────────────────────────

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """
    Returns True iff a dominates b (Eq. 5 in paper):
      ∀k: a[k] >= b[k]  AND  ∃k: a[k] > b[k]
    Both a, b are objective vectors (higher is better for all objectives).
    """
    return bool(np.all(a >= b) and np.any(a > b))


def non_dominated_set(points: np.ndarray) -> np.ndarray:
    """
    Extract the non-dominated (Pareto) front from a set of points.

    Args:
        points: (N, m) array of objective vectors (all objectives: higher = better)
    Returns:
        mask: (N,) boolean array, True for non-dominated points
    """
    N = len(points)
    is_dominated = np.zeros(N, dtype=bool)
    for i in range(N):
        for j in range(N):
            if i != j and dominates(points[j], points[i]):
                is_dominated[i] = True
                break
    return ~is_dominated


# ─── Hypervolume (2D exact) ───────────────────────────────────────────────────

def hypervolume_2d(points: np.ndarray, ref_point: np.ndarray) -> float:
    """
    Exact hypervolume computation for 2D objective space.

    Algorithm: sort by first objective descending, sweep second objective.
    Time: O(N log N).

    Args:
        points:    (N, 2) array of objective vectors (higher = better)
        ref_point: (2,) reference point (dominated by all valid points)
    Returns:
        hv: scalar hypervolume value
    """
    if len(points) == 0:
        return 0.0

    # Filter points that actually dominate the reference point
    valid = np.all(points > ref_point, axis=1)
    pts = points[valid]
    if len(pts) == 0:
        return 0.0

    # Extract non-dominated front (for clean HV)
    nd_mask = non_dominated_set(pts)
    pts = pts[nd_mask]

    # Sort by first objective descending
    idx = np.argsort(pts[:, 0])[::-1]
    pts = pts[idx]

    hv = 0.0
    prev_x = ref_point[0]
    max_y  = ref_point[1]

    for p in pts:
        width  = p[0] - prev_x
        height = p[1] - max_y
        if width > 0 and height > 0:
            hv += width * height
        max_y  = max(max_y, p[1])
        prev_x = p[0]

    # Remaining strip to the right of last point
    # (already handled in loop above, width would be negative)
    return float(hv)


def normalize_objectives(
    points:    np.ndarray,
    ref_point: np.ndarray,
    ideal_point: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize objectives to [0,1] for stable HV computation.
    Used when comparing across environments with different reward scales.

    Returns normalized points and normalized reference point.
    """
    if ideal_point is None:
        ideal_point = points.max(axis=0)
    range_ = ideal_point - ref_point
    range_[range_ == 0] = 1.0  # avoid division by zero
    pts_norm = (points - ref_point) / range_
    ref_norm = np.zeros_like(ref_point)
    return pts_norm, ref_norm


# ─── Metrics tracker ──────────────────────────────────────────────────────────

class MetricsTracker:
    """
    Tracks training metrics across episodes and weight vectors.

    Usage:
        tracker = MetricsTracker()
        tracker.log(w_idx=0, episode=1, task_return=0.5, comm_return=-1.2, ...)
        tracker.save("results/metrics.json")
    """
    def __init__(self):
        self.data: Dict[int, List[dict]] = defaultdict(list)

    def log(self, w_idx: int, **kwargs):
        """Log a dictionary of metrics for weight vector index w_idx."""
        self.data[w_idx].append(kwargs)

    def get_final_returns(self) -> np.ndarray:
        """
        Extract final (evaluation) objective vectors for each weight vector.
        Returns (n_weights, 2) array.
        """
        returns = []
        for w_idx in sorted(self.data.keys()):
            episodes = self.data[w_idx]
            if not episodes:
                continue
            last = episodes[-1]
            task = last.get("eval_task_return", 0.0)
            comm = last.get("eval_comm_return", 0.0)
            returns.append([task, comm])
        return np.array(returns, dtype=np.float32)

    def get_hv_curve(self, w_idx: int, ref_point: np.ndarray) -> List[float]:
        """
        Compute hypervolume at each logged episode for weight vector w_idx.
        Treats each episode's vector return as a single-point set.
        """
        hvs = []
        for entry in self.data[w_idx]:
            task = entry.get("task_return", 0.0)
            comm = entry.get("comm_return", 0.0)
            pts  = np.array([[task, comm]])
            hvs.append(hypervolume_2d(pts, ref_point))
        return hvs

    def save(self, path: str):
        """Save metrics to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        serializable = {str(k): v for k, v in self.data.items()}
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Metrics saved to {path}")

    @classmethod
    def load(cls, path: str) -> "MetricsTracker":
        t = cls()
        with open(path) as f:
            raw = json.load(f)
        t.data = {int(k): v for k, v in raw.items()}
        return t


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_pareto_front(
    pf_points:    np.ndarray,
    baselines:    Optional[Dict[str, np.ndarray]] = None,
    ref_point:    Optional[np.ndarray] = None,
    save_path:    Optional[str] = None,
    title:        str = "ParetoComm — Pareto Front Approximation",
):
    """
    Plot the 2D Pareto front and baseline comparison.

    Args:
        pf_points:  (N, 2) ParetoComm objective vectors [task_return, comm_return]
        baselines:  dict of {method_name: (M, 2) points}
        ref_point:  (2,) reference point for HV shading
        save_path:  if given, saves PDF and PNG
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available. Install: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    # Style
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.linewidth": 0.8, "grid.alpha": 0.3,
    })

    # Non-dominated front
    nd_mask = non_dominated_set(pf_points)
    nd_pts  = pf_points[nd_mask]
    dom_pts = pf_points[~nd_mask]

    # Sort non-dominated by first objective for line plot
    sort_idx = np.argsort(nd_pts[:, 0])
    nd_sorted = nd_pts[sort_idx]

    ax.plot(nd_sorted[:, 0], nd_sorted[:, 1],
            "b-o", ms=7, lw=1.5, label="ParetoComm (non-dominated)",
            zorder=5)
    if len(dom_pts) > 0:
        ax.scatter(dom_pts[:, 0], dom_pts[:, 1],
                   c="lightblue", edgecolors="blue", alpha=0.5, s=40,
                   label="ParetoComm (dominated)", zorder=4)

    # Shade HV region
    if ref_point is not None and len(nd_sorted) > 0:
        hv_x = [ref_point[0]] + list(nd_sorted[:, 0]) + [nd_sorted[-1, 0]]
        hv_y = [nd_sorted[0, 1]] + list(nd_sorted[:, 1]) + [ref_point[1]]
        ax.fill_betweenx(
            [ref_point[1], nd_sorted[:, 1].max()],
            ref_point[0], nd_sorted[:, 0].max(),
            alpha=0.08, color="blue", label="Dominated hypervolume"
        )

    # Baselines
    colors  = ["orange", "green", "red", "purple"]
    markers = ["^", "s", "D", "v"]
    if baselines:
        for i, (name, pts) in enumerate(baselines.items()):
            c = colors[i % len(colors)]
            m = markers[i % len(markers)]
            ax.scatter(pts[:, 0], pts[:, 1],
                       c=c, marker=m, s=80, zorder=6,
                       label=name, edgecolors="black", linewidths=0.5)

    ax.set_xlabel("Task return $G^\\mathrm{task}$", fontsize=12)
    ax.set_ylabel("Communication return $G^\\mathrm{comm}$", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path + ".pdf", dpi=300, bbox_inches="tight")
        plt.savefig(save_path + ".png", dpi=200, bbox_inches="tight")
        print(f"Pareto front figure saved to {save_path}.pdf / .png")
    plt.show()
    return fig, ax


def plot_training_curves(
    metrics:      MetricsTracker,
    ref_point:    np.ndarray,
    w_indices:    Optional[List[int]] = None,
    save_path:    Optional[str] = None,
):
    """
    Plot training curves: episode return and hypervolume over training.

    Args:
        metrics:   MetricsTracker with logged data
        ref_point: (2,) reference point for HV computation
        w_indices: which weight vectors to plot (default: all)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    w_keys = w_indices if w_indices else sorted(metrics.data.keys())
    cmap   = plt.cm.viridis(np.linspace(0, 1, len(w_keys)))

    ax1, ax2 = axes

    for i, w_idx in enumerate(w_keys):
        entries = metrics.data[w_idx]
        if not entries:
            continue
        episodes     = [e.get("episode", j) for j, e in enumerate(entries)]
        task_returns = [e.get("task_return", 0.0) for e in entries]
        hv_curve     = metrics.get_hv_curve(w_idx, ref_point)
        w_label      = entries[0].get("weight_str", f"w{w_idx}")

        ax1.plot(episodes, task_returns, color=cmap[i], alpha=0.8,
                 linewidth=1.0, label=w_label if i % 3 == 0 else "_nolegend_")
        ax2.plot(episodes, hv_curve, color=cmap[i], alpha=0.8,
                 linewidth=1.0)

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Task return")
    ax1.set_title("(a) Task return during training")
    ax1.legend(fontsize=7, ncol=3)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Hypervolume")
    ax2.set_title("(b) Hypervolume per weight vector")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path + ".pdf", dpi=300, bbox_inches="tight")
        plt.savefig(save_path + ".png", dpi=200, bbox_inches="tight")
        print(f"Training curves saved to {save_path}")
    plt.show()
    return fig


def plot_communication_tradeoff(
    pf_points:    np.ndarray,
    comm_rates:   np.ndarray,
    save_path:    Optional[str] = None,
):
    """
    Plot communication rate vs task success rate (Figure 3 in paper).

    Args:
        pf_points:  (N, 2) objective vectors
        comm_rates: (N,) communication rates in [0,1]
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    task_returns = pf_points[:, 0]
    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(comm_rates, task_returns,
                    c=np.arange(len(comm_rates)), cmap="viridis",
                    s=80, edgecolors="black", linewidths=0.5, zorder=5)
    plt.colorbar(sc, ax=ax, label="Weight vector index (low w₁ → high w₁)")

    ax.set_xlabel("Communication rate $c_{\\mathrm{rate}}$")
    ax.set_ylabel("Task return $G^{\\mathrm{task}}$")
    ax.set_title("Communication–performance trade-off")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path + ".pdf", dpi=300, bbox_inches="tight")
        plt.savefig(save_path + ".png", dpi=200, bbox_inches="tight")
    plt.show()
    return fig


# ─── Checkpoint utilities ─────────────────────────────────────────────────────

def save_checkpoint(
    agent:      torch.nn.Module,
    optimizer:  torch.optim.Optimizer,
    w_idx:      int,
    episode:    int,
    metrics:    dict,
    path:       str,
):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "w_idx":      w_idx,
        "episode":    episode,
        "model":      agent.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "metrics":    metrics,
    }, path)


def load_checkpoint(path: str, agent: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None):
    """Load model checkpoint."""
    ckpt = torch.load(path, map_location="cpu")
    agent.load_state_dict(ckpt["model"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


# ─── Returns computation ──────────────────────────────────────────────────────

def compute_returns(
    rewards:    List[float],
    gamma:      float = 0.99,
    normalize:  bool  = True,
) -> torch.Tensor:
    """
    Compute Monte Carlo discounted returns from a trajectory.

    G_t = Σ_{t'=t}^T γ^{t'-t} r_{t'}

    Args:
        rewards:   list of scalar rewards (already scalarized)
        gamma:     discount factor
        normalize: if True, normalize returns to zero mean / unit variance

    Returns:
        returns: Tensor of shape (T,)
    """
    R = 0.0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    if normalize and len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def compute_advantages(
    returns:    torch.Tensor,   # (T,)
    values:     torch.Tensor,   # (T,)
) -> torch.Tensor:
    """
    Compute advantages: A_t = G_t - V(s_t).

    Args:
        returns: (T,) discounted returns
        values:  (T,) value estimates from critic
    Returns:
        advantages: (T,)
    """
    advantages = returns - values.detach()
    # Normalize advantages
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages

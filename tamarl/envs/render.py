"""Matplotlib-based renderer for the dynamic traffic assignment environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

# Matplotlib is optional; the renderer degrades gracefully if unavailable.
try:  # pragma: no cover - optional dependency
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.collections import LineCollection
except Exception:  # pragma: no cover - optional dependency
    plt = None
    mcolors = None
    LineCollection = None


@dataclass
class RenderState:
    """Lightweight container for what we need to draw a frame."""

    positions: np.ndarray  # shape [num_nodes, 2]
    edge_index: np.ndarray  # shape [2, num_edges]
    flows: np.ndarray  # shape [num_edges]
    travel_times: np.ndarray  # shape [num_edges]
    agent_nodes: Dict[str, int]
    step: int
    mean_reward: float


class BaseRenderer:
    """Interface used by the environment; Matplotlib implementation follows."""

    available: bool = False

    def render(self, state: RenderState) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class NoOpRenderer(BaseRenderer):
    """Fallback when Matplotlib is not installed or usable."""

    available = False

    def render(self, state: RenderState) -> bool:
        return False

    def close(self) -> None:
        return None


class MatplotlibTrafficRenderer(BaseRenderer):
    """Simple dark-themed renderer using Matplotlib."""

    available = plt is not None

    def __init__(self, figsize: Sequence[float] = (7, 6)) -> None:
        if not self.available:
            raise RuntimeError("Matplotlib is not available.")

        self.figsize = figsize
        self.fig: Optional["plt.Figure"] = None
        self.ax: Optional["plt.Axes"] = None
        self._edge_lc: Optional[LineCollection] = None
        self._node_scatter = None
        self._agent_scatter = None
        self._text_step = None
        self._text_reward = None

        self._bg = "#0b1021"
        self._edge_cmap = "magma"
        self._agent_color = "#56e39f"
        self._node_color = "#e0e0e0"

    def _lazy_setup(self, positions: np.ndarray) -> None:
        if self.fig is not None:
            return

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        if self.fig is None or self.ax is None:
            return

        self.fig.patch.set_facecolor(self._bg)
        self.ax.set_facecolor(self._bg)
        self.ax.axis("off")
        self.ax.set_aspect("equal")

        pad = 0.2
        x_min, y_min = positions.min(axis=0) - pad
        x_max, y_max = positions.max(axis=0) + pad
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)

        self._text_step = self.ax.text(
            0.02,
            0.96,
            "",
            transform=self.ax.transAxes,
            color="#d7f9ff",
            fontsize=10,
            ha="left",
            va="top",
            fontweight="bold",
        )
        self._text_reward = self.ax.text(
            0.98,
            0.96,
            "",
            transform=self.ax.transAxes,
            color="#ffd166",
            fontsize=10,
            ha="right",
            va="top",
            fontweight="bold",
        )

    def _build_edge_segments(self, positions: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        if edge_index.shape[1] == 0:
            return np.zeros((0, 2, 2), dtype=float)
        src = edge_index[0]
        dst = edge_index[1]
        segments = np.stack([positions[src], positions[dst]], axis=1)
        return segments

    def render(self, state: RenderState) -> bool:  # pragma: no cover - visual side-effects
        if not self.available or plt is None or LineCollection is None or mcolors is None:
            return False

        positions = np.asarray(state.positions)
        edge_index = np.asarray(state.edge_index, dtype=int)
        flows = np.asarray(state.flows, dtype=float)
        travel_times = np.asarray(state.travel_times, dtype=float)

        self._lazy_setup(positions)
        if self.fig is None or self.ax is None:
            return False

        segments = self._build_edge_segments(positions, edge_index)

        # Edge widths keyed to flow intensity; colors keyed to travel time.
        if flows.size:
            max_flow = max(float(flows.max()), 1e-6)
            flow_norm = flows / max_flow
        else:
            flow_norm = np.zeros_like(flows)

        if travel_times.size:
            tt_min = float(travel_times.min())
            tt_max = float(travel_times.max())
            if tt_max - tt_min < 1e-6:
                tt_max = tt_min + 1.0
            norm = mcolors.Normalize(vmin=tt_min, vmax=tt_max)
            edge_colors = plt.get_cmap(self._edge_cmap)(norm(travel_times))
        else:
            edge_colors = "#6c5ce7"

        widths = 1.4 + 3.2 * flow_norm

        if self._edge_lc is None:
            self._edge_lc = LineCollection(
                segments,
                linewidths=widths,
                colors=edge_colors,
                alpha=0.95,
                zorder=1,
            )
            self.ax.add_collection(self._edge_lc)
        else:
            self._edge_lc.set_segments(segments)
            self._edge_lc.set_linewidth(widths)
            self._edge_lc.set_color(edge_colors)

        # Nodes
        if self._node_scatter is None:
            self._node_scatter = self.ax.scatter(
                positions[:, 0],
                positions[:, 1],
                s=60,
                c=self._node_color,
                edgecolors="#222222",
                linewidths=0.8,
                zorder=3,
            )
        else:
            self._node_scatter.set_offsets(positions)

        # Agents as lively teal dots with subtle jitter to reduce overlap.
        agent_positions: Iterable[np.ndarray] = []
        if state.agent_nodes:
            jitter = np.random.uniform(-0.02, 0.02, size=(len(state.agent_nodes), 2))
            agent_positions = positions[list(state.agent_nodes.values())] + jitter
        agent_positions = np.array(list(agent_positions))

        if self._agent_scatter is None:
            self._agent_scatter = self.ax.scatter(
                agent_positions[:, 0] if agent_positions.size else [],
                agent_positions[:, 1] if agent_positions.size else [],
                s=90,
                c=self._agent_color,
                edgecolors="#152b2f",
                linewidths=1.2,
                alpha=0.95,
                marker="o",
                zorder=4,
            )
        else:
            if agent_positions.size:
                self._agent_scatter.set_offsets(agent_positions)
            else:
                self._agent_scatter.set_offsets(np.zeros((0, 2)))

        # HUD text
        if self._text_step is not None:
            self._text_step.set_text(f"Step {state.step}")
        if self._text_reward is not None:
            self._text_reward.set_text(f"Mean cumul. r: {state.mean_reward:.2f}")

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.5)
        return True

    def close(self) -> None:  # pragma: no cover - UI side-effects
        if self.fig is not None:
            plt.close(self.fig)
        self.fig = None
        self.ax = None
        self._edge_lc = None
        self._node_scatter = None
        self._agent_scatter = None
        self._text_step = None
        self._text_reward = None


def build_renderer() -> BaseRenderer:
    """Return a usable renderer (Matplotlib if available, otherwise no-op)."""

    if MatplotlibTrafficRenderer.available:
        try:
            return MatplotlibTrafficRenderer()
        except Exception:
            return NoOpRenderer()
    return NoOpRenderer()


__all__ = ["RenderState", "BaseRenderer", "MatplotlibTrafficRenderer", "NoOpRenderer", "build_renderer"]

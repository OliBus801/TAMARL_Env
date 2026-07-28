"""Publication-quality figures for the "Additional Results and Ablations"
appendix section, populated from CSVs produced by
`scripts/run_ablation_sweep.py`.

Style is shared with `tamarl.visualisation.validation_figures` (same
`setup_style()`/`_clean_axes`/`_save` helpers and colorblind-validated
palette) so ablation figures read as part of the same figure set as the
DNL-fidelity validation appendix, rather than a visually distinct add-on.
"""

from __future__ import annotations

import numpy as np

from tamarl.visualisation.validation_figures import (
    BASELINE,
    COLOR_AQUA,
    COLOR_MATSIM,
    COLOR_TORCHDNL,
    COLOR_YELLOW,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    _clean_axes,
    _save,
    setup_style,
)

__all__ = [
    "setup_style",
    "AGENT_COLORS",
    "AGENT_LABELS",
    "plot_metric_vs_param",
    "plot_method_comparison_bars",
    "plot_learning_curves",
    "plot_link_tt_interval_sensitivity",
    "plot_exp3_heatmap",
]

# Fixed per-algorithm color assignment, reusing the validated 4-slot palette
# so agent identity stays consistent with the main paper's figures (Table
# `tab:performance_results`, `fig:learning_graph`) wherever they overlap.
AGENT_COLORS = {
    "ucb": COLOR_TORCHDNL,
    "msa": COLOR_MATSIM,
    "ts": COLOR_AQUA,
    "epsilon_greedy": COLOR_YELLOW,
    "exp3": COLOR_MATSIM,
    "rd": COLOR_TORCHDNL,
}
AGENT_LABELS = {
    "ucb": "UCB",
    "msa": "MSA",
    "ts": "Thompson Sampling",
    "epsilon_greedy": r"$\epsilon$-Greedy",
    "exp3": "Exp3",
    "rd": "Replicator Dynamics",
}

METRIC_LABELS = {
    "mean_travel_time_final": "Mean Travel Time (s)",
    "mean_regret_final": "Mean Regret (s)",
    "epsilon_compliance_rate_final": r"$\epsilon$-Compliance",
    "h_global": r"$H_{global}$ (nats)",
}

# A small qualitative cycle for axes with no natural "algorithm" grouping
# (e.g. MSA step-size schedules), reusing the same validated 4 hues.
QUALITATIVE_CYCLE = [COLOR_MATSIM, COLOR_TORCHDNL, COLOR_AQUA, COLOR_YELLOW]


def _agg(df, group_cols, value_col):
    """Mean +/- std of `value_col` grouped by `group_cols` (pandas-free)."""
    groups: dict[tuple, list[float]] = {}
    for row in df:
        key = tuple(row[c] for c in group_cols)
        if value_col in row and row[value_col] not in (None, ""):
            groups.setdefault(key, []).append(float(row[value_col]))
    out = {}
    for key, vals in groups.items():
        arr = np.asarray(vals, dtype=float)
        out[key] = (float(np.mean(arr)), float(np.std(arr)))
    return out


# --------------------------------------------------------------------------
# A2 / C: metric vs. a scalar hyperparameter, one panel per metric, one line
# per agent (or a single line if the axis has a single agent).
# --------------------------------------------------------------------------
def plot_metric_vs_param(
    rows: list[dict],
    param_col: str,
    metrics: tuple[str, ...],
    output_path: str,
    xlabel: str,
    log_x: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    agents = sorted({r["agent_type"] for r in rows})
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.3))
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        _clean_axes(ax)
        for i, agent in enumerate(agents):
            sub = [r for r in rows if r["agent_type"] == agent]
            agg = _agg(sub, (param_col,), metric)
            if not agg:
                continue
            xs = sorted(agg.keys(), key=lambda k: k[0])
            x_vals = [k[0] for k in xs]
            y_mean = [agg[k][0] for k in xs]
            y_std = [agg[k][1] for k in xs]
            color = AGENT_COLORS.get(agent, QUALITATIVE_CYCLE[i % len(QUALITATIVE_CYCLE)])
            ax.errorbar(
                x_vals, y_mean, yerr=y_std, color=color, marker="o", markersize=3.2,
                linewidth=1.3, capsize=2, zorder=3, label=AGENT_LABELS.get(agent, agent),
            )
        if log_x:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))

    if len(agents) > 1:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.12),
            ncol=len(agents), frameon=False, handlelength=1.6,
        )

    fig.tight_layout()
    _save(fig, output_path)


# --------------------------------------------------------------------------
# A3: candidate-route method comparison (grouped bars).
# --------------------------------------------------------------------------
def plot_method_comparison_bars(
    rows: list[dict],
    metric: str,
    output_path: str,
    methods: tuple[str, ...] = ("penalty", "yen"),
) -> None:
    import matplotlib.pyplot as plt

    agents = sorted({r["agent_type"] for r in rows})
    x = np.arange(len(agents))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(2.6 + 0.6 * len(agents), 2.4))
    _clean_axes(ax, x_grid=False, y_grid=True)

    for i, method in enumerate(methods):
        means, stds = [], []
        for agent in agents:
            sub = [r for r in rows if r["agent_type"] == agent and r.get("path_method") == method]
            agg = _agg(sub, (), metric)
            if agg:
                m, s = next(iter(agg.values()))
            else:
                m, s = 0.0, 0.0
            means.append(m)
            stds.append(s)
        color = QUALITATIVE_CYCLE[i % len(QUALITATIVE_CYCLE)]
        offset = (i - (len(methods) - 1) / 2) * width
        ax.bar(
            x + offset, means, width, yerr=stds, color=color, capsize=2,
            label=method.capitalize(), zorder=3, edgecolor="white", linewidth=0.4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([AGENT_LABELS.get(a, a) for a in agents])
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    legend = ax.legend(loc="upper right", frameon=False, handlelength=1.4)
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)

    fig.tight_layout()
    _save(fig, output_path)


# --------------------------------------------------------------------------
# B1 / B2: learning curves (MTT vs. episode), one line per group.
# --------------------------------------------------------------------------
def plot_learning_curves(
    curve_rows: list[dict],
    group_col: str,
    output_path: str,
    metric: str = "mean_travel_time",
    ylabel: str = "Mean Travel Time (s)",
    group_labels: dict | None = None,
) -> None:
    import matplotlib.pyplot as plt

    groups = sorted({r[group_col] for r in curve_rows})
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    _clean_axes(ax)

    for i, group in enumerate(groups):
        sub = [r for r in curve_rows if r[group_col] == group]
        agg = _agg(sub, ("episode",), metric)
        if not agg:
            continue
        eps = sorted(agg.keys(), key=lambda k: k[0])
        x_vals = [k[0] for k in eps]
        y_mean = [agg[k][0] for k in eps]
        color = QUALITATIVE_CYCLE[i % len(QUALITATIVE_CYCLE)]
        label = (group_labels or {}).get(group, str(group))
        ax.plot(x_vals, y_mean, color=color, linewidth=1.3, zorder=3, label=label)

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    legend = ax.legend(
        loc="upper right", frameon=True, facecolor="white", edgecolor=GRIDLINE,
        framealpha=0.9, fontsize=6.5, handlelength=1.6, borderaxespad=0.3,
    )
    legend.get_frame().set_linewidth(0.7)
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)

    fig.tight_layout()
    _save(fig, output_path)


# --------------------------------------------------------------------------
# B3: TD-oracle time-bin granularity sensitivity (MTT + Regret vs. bin width).
# --------------------------------------------------------------------------
def plot_link_tt_interval_sensitivity(rows: list[dict], output_path: str) -> None:
    plot_metric_vs_param(
        rows,
        param_col="link_tt_interval",
        metrics=("mean_travel_time_final", "mean_regret_final"),
        output_path=output_path,
        xlabel="TD-oracle bin width (s)",
    )


# --------------------------------------------------------------------------
# C: Exp3 (eta, gamma) 2D grid heatmap.
# --------------------------------------------------------------------------
def plot_exp3_heatmap(rows: list[dict], metric: str, output_path: str) -> None:
    import matplotlib.pyplot as plt

    etas = sorted({r["exp3_eta"] for r in rows})
    gammas = sorted({r["exp3_gamma"] for r in rows})
    grid = np.full((len(gammas), len(etas)), np.nan)

    for r in rows:
        if metric not in r or r[metric] in (None, ""):
            continue
        i = gammas.index(r["exp3_gamma"])
        j = etas.index(r["exp3_eta"])
        grid[i, j] = float(r[metric])

    fig, ax = plt.subplots(figsize=(2.8, 2.4))
    im = ax.imshow(grid, cmap="magma_r", aspect="auto", origin="lower")
    ax.set_xticks(range(len(etas)))
    ax.set_xticklabels([str(e) for e in etas])
    ax.set_yticks(range(len(gammas)))
    ax.set_yticklabels([str(g) for g in gammas])
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$\gamma$")

    for i in range(len(gammas)):
        for j in range(len(etas)):
            if not np.isnan(grid[i, j]):
                ax.text(
                    j, i, f"{grid[i, j]:.0f}", ha="center", va="center",
                    fontsize=6, color="white" if grid[i, j] > np.nanmean(grid) else INK_PRIMARY,
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(METRIC_LABELS.get(metric, metric), fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.tight_layout()
    _save(fig, output_path)

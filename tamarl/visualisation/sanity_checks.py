"""Sanity Check Suite — Post-training diagnostic plots and CSV data exports.

Generates a triptych of validation plots to prove that the RL agent
is legitimately improving traffic distribution, not exploiting simulator flaws.
Also saves raw plot data as CSVs for scientific reuse and customization.
"""

from __future__ import annotations

import os
import csv
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ── Shared Dark Theme ─────────────────────────────────────────────────────────

_DARK_BG = '#0d1117'
_DARK_CARD = '#161b22'
_TEXT_COLOR = '#c9d1d9'
_ACCENT_CYAN = '#58a6ff'
_ACCENT_GREEN = '#3fb950'
_ACCENT_RED = '#f85149'
_ACCENT_ORANGE = '#d29922'
_ACCENT_PURPLE = '#bc8cff'
_GRID_COLOR = '#21262d'

def _apply_dark_theme(ax, fig=None):
    """Apply consistent dark theme to axes."""
    ax.set_facecolor(_DARK_CARD)
    ax.tick_params(colors=_TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(_TEXT_COLOR)
    ax.yaxis.label.set_color(_TEXT_COLOR)
    ax.title.set_color(_TEXT_COLOR)
    ax.grid(True, alpha=0.15, color=_GRID_COLOR)
    for spine in ax.spines.values():
        spine.set_color(_GRID_COLOR)
    if fig is not None:
        fig.patch.set_facecolor(_DARK_BG)


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 1: Travel Time vs Free-Flow Lower Bound
# ══════════════════════════════════════════════════════════════════════════════

def plot_tt_vs_fftt(
    realized_tt: torch.Tensor,
    fftt_chosen: torch.Tensor,
    output_path: str,
    original_idx: Optional[np.ndarray] = None,
    planned_dep: Optional[np.ndarray] = None,
    actual_dep: Optional[np.ndarray] = None,
    arrival: Optional[np.ndarray] = None,
    chosen_path_idx: Optional[np.ndarray] = None,
):
    """Hexbin scatter plot of realized travel time vs free-flow travel time.

    The Y=X line represents the physical lower bound. It turns green if no agent
    violates it, and red if violations are found.

    Args:
        realized_tt: [N] tensor of actual travel times (seconds).
        fftt_chosen: [N] tensor of free-flow travel times (seconds).
        output_path: Where to save the plot.
        original_idx: Optional [N] array of the original, unfiltered leg indices.
    """
    tt_np = realized_tt.numpy().astype(np.float64)
    ff_np = fftt_chosen.astype(np.float64) if isinstance(fftt_chosen, np.ndarray) else fftt_chosen.numpy().astype(np.float64)

    # Convert to minutes for readability
    tt_min = tt_np / 60.0
    ff_min = ff_np / 60.0

    # Filter out non-finite values for both validation and plotting
    finite_mask = np.isfinite(tt_np) & np.isfinite(ff_np)

    # ── Validation ──
    violations = (tt_np < (ff_np - 1e-3)) & finite_mask  # 1ms tolerance for float precision
    n_violations = int(violations.sum())
    is_valid = n_violations == 0

    # Save CSV Data
    csv_path = output_path.replace('.png', '.csv')
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['leg_idx', 'fftt_sec', 'realized_tt_sec', 'fftt_min', 'realized_tt_min', 'planned_departure', 'actual_departure', 'arrival', 'chosen_path_idx'])
            for i in range(len(tt_np)):
                idx = int(original_idx[i]) if original_idx is not None else i
                pd_val = int(planned_dep[i]) if planned_dep is not None else -1
                ad_val = int(actual_dep[i]) if actual_dep is not None else -1
                ar_val = int(arrival[i]) if arrival is not None else -1
                cp_val = int(chosen_path_idx[i]) if chosen_path_idx is not None else -1
                writer.writerow([idx, ff_np[i], tt_np[i], ff_min[i], tt_min[i], pd_val, ad_val, ar_val, cp_val])
        print(f"  📄 Saved Plot 1 data to {csv_path}")
    except Exception as e:
        print(f"  ⚠ Failed to save CSV data for Plot 1: {e}")

    fig, ax = plt.subplots(figsize=(8, 7))
    _apply_dark_theme(ax, fig)

    plot_tt = tt_min[finite_mask]
    plot_ff = ff_min[finite_mask]
    
    n_dropped = len(tt_min) - len(plot_tt)
    if n_dropped > 0:
        print(f"  ⚠ Ignored {n_dropped} agents with Inf/NaN travel times in Plot 1.")

    # Hexbin limits
    if len(plot_tt) > 0:
        max_val = max(plot_tt.max(), plot_ff.max()) * 1.05
    else:
        max_val = 1.0
        
    if max_val <= 0 or np.isnan(max_val) or np.isinf(max_val):
        max_val = 1.0

    min_val = 0

    cmap = LinearSegmentedColormap.from_list(
        'dark_heat', ['#0d1117', '#1a3a5c', '#2d7d9a', '#58a6ff', '#79c0ff', '#ffffff']
    )

    if len(plot_tt) > 0:
        hb = ax.hexbin(
            plot_ff, plot_tt,
            gridsize=60, cmap=cmap, mincnt=1,
            extent=[min_val, max_val, min_val, max_val],
        )
        cb = fig.colorbar(hb, ax=ax, pad=0.02)
        cb.set_label('Agent Count', color=_TEXT_COLOR, fontsize=10)
        cb.ax.tick_params(colors=_TEXT_COLOR, labelsize=8)

    # Y=X line (physical lower bound: green if valid, red if violation)
    line_range = np.linspace(min_val, max_val, 100)
    line_color = _ACCENT_GREEN if is_valid else _ACCENT_RED
    ax.plot(line_range, line_range, '--', color=line_color, linewidth=1.5,
            alpha=0.8, label='Y = X (Free-Flow Bound)')

    # Mark violations if any
    if n_violations > 0:
        viol_idx = np.where(violations & finite_mask)[0]
        if len(viol_idx) > 0:
            ax.scatter(ff_min[viol_idx], tt_min[viol_idx], color=_ACCENT_RED,
                       s=15, zorder=5, alpha=0.8, label=f'⚠ {n_violations} violations')

    ax.set_xlabel('Free-Flow Travel Time (minutes)', fontsize=11)
    ax.set_ylabel('Realized Travel Time (minutes)', fontsize=11)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.8,
              facecolor=_DARK_CARD, edgecolor=_GRID_COLOR, labelcolor=_TEXT_COLOR)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 Plot 1 (TT vs FFTT) saved to {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 2: V/C Ratio Distribution (Time-Binned)
# ══════════════════════════════════════════════════════════════════════════════

def plot_vc_ratio(
    link_counts: torch.Tensor,
    flow_capacity_per_step: torch.Tensor,
    link_tt_interval: float,
    dt: float,
    output_path: str,
):
    """Plot V/C ratio histograms: Peak-Hour and Max V/C.

    V/C is computed per 15-minute (link_tt_interval) bin per edge. Two histograms:
      - Peak-Hour: the single worst 15-min interval by total network volume.
      - Max V/C: the worst V/C experienced by each edge across all intervals.

    Args:
        link_counts: [num_intervals, E] vehicle counts per interval per edge.
        flow_capacity_per_step: [E] capacity in veh/step.
        link_tt_interval: Width of each time bin (seconds).
        dt: Simulation timestep (seconds).
        output_path: Where to save the plot.
    """
    counts = link_counts.float()
    num_intervals, num_edges = counts.shape

    # Capacity per interval (convert from veh/step to veh/interval)
    steps_per_interval = link_tt_interval / dt
    capacity_per_interval = flow_capacity_per_step.float() * steps_per_interval  # [E]

    # V/C matrix [num_intervals, E]
    safe_cap = capacity_per_interval.clamp(min=1e-6).unsqueeze(0)  # [1, E]
    vc_matrix = counts / safe_cap  # [num_intervals, E]

    # Only consider edges that saw at least 1 vehicle across the whole simulation
    active_mask = counts.sum(dim=0) > 0  # [E]
    num_active = active_mask.sum().item()
    pct_active = (num_active / num_edges) * 100 if num_edges > 0 else 0.0

    # Find the interval with the highest total network volume (Peak Hour)
    interval_volumes = counts.sum(dim=1)  # [num_intervals]
    peak_interval = int(interval_volumes.argmax().item())
    start_sec = peak_interval * link_tt_interval
    h = int(start_sec // 3600)
    m = int((start_sec % 3600) // 60)
    time_label = f"{h}h{m:02d} ({int(link_tt_interval // 60)}min bin)"

    vc_peak = vc_matrix[peak_interval, active_mask].numpy()
    max_vc = vc_matrix[:, active_mask].max(dim=0).values.numpy()  # [active_edges]

    # Save CSV Data
    csv_peak_path = output_path.replace('.png', '_peak.csv')
    csv_max_path = output_path.replace('.png', '_max.csv')
    try:
        # Save Peak Hour V/C
        with open(csv_peak_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['edge_idx', 'vc_ratio', 'volume', 'capacity_interval'])
            # Get list of edges
            active_indices = torch.nonzero(active_mask, as_tuple=True)[0].cpu().numpy()
            for idx in active_indices:
                writer.writerow([
                    idx,
                    vc_matrix[peak_interval, idx].item(),
                    counts[peak_interval, idx].item(),
                    capacity_per_interval[idx].item()
                ])
        # Save Max V/C
        with open(csv_max_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['edge_idx', 'max_vc_ratio'])
            active_indices = torch.nonzero(active_mask, as_tuple=True)[0].cpu().numpy()
            for i, idx in enumerate(active_indices):
                writer.writerow([idx, max_vc[i]])
        print(f"  📄 Saved Plot 2 data to {csv_peak_path} and {csv_max_path}")
    except Exception as e:
        print(f"  ⚠ Failed to save CSV data for Plot 2: {e}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    _apply_dark_theme(ax1, fig)
    _apply_dark_theme(ax2)

    # ── Peak-Hour Histogram ──
    bins_vc = np.linspace(0, min(vc_peak.max() * 1.1, 5.0) if len(vc_peak) > 0 else 3.0, 50)
    ax1.hist(vc_peak, bins=bins_vc, color=_ACCENT_CYAN, alpha=0.85, edgecolor='none',
             label='Peak interval')
    ax1.axvline(x=1.0, color=_ACCENT_RED, linestyle='--', linewidth=1.5, alpha=0.8,
                label='V/C = 1.0 (capacity)')
    over_capacity = (vc_peak > 1.0).sum()
    pct_over = over_capacity / len(vc_peak) * 100 if len(vc_peak) > 0 else 0

    ax1.text(0.98, 0.98,
             f'Peak interval: {time_label}\n'
             f'edges > 1.0 : {pct_over:.1f}%\n'
             f'Active edges: {pct_active:.1f}%',
             transform=ax1.transAxes, fontsize=9, color=_TEXT_COLOR,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=_DARK_BG, edgecolor=_GRID_COLOR, alpha=0.9))

    ax1.set_xlabel('V/C Ratio', fontsize=11)
    ax1.set_ylabel('Number of Edges', fontsize=11)
    ax1.set_title('Peak-Hour V/C Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, framealpha=0.8, facecolor=_DARK_CARD, edgecolor=_GRID_COLOR, labelcolor=_TEXT_COLOR)

    # ── Max V/C Histogram ──
    bins_max = np.linspace(0, min(max_vc.max() * 1.1, 5.0) if len(max_vc) > 0 else 3.0, 50)
    ax2.hist(max_vc, bins=bins_max, color=_ACCENT_ORANGE, alpha=0.85, edgecolor='none',
             label='Max V/C per edge')
    ax2.axvline(x=1.0, color=_ACCENT_RED, linestyle='--', linewidth=1.5, alpha=0.8,
                label='V/C = 1.0 (capacity)')
    over_max = (max_vc > 1.0).sum()
    pct_over_max = over_max / len(max_vc) * 100 if len(max_vc) > 0 else 0

    ax2.text(0.98, 0.98,
             f'edges > 1.0 : {pct_over_max:.1f}%\n'
             f'Median max V/C: {np.median(max_vc):.2f}',
             transform=ax2.transAxes, fontsize=9, color=_TEXT_COLOR,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=_DARK_BG, edgecolor=_GRID_COLOR, alpha=0.9))

    ax2.set_xlabel('Max V/C Ratio', fontsize=11)
    ax2.set_ylabel('Number of Edges', fontsize=11)
    ax2.set_title('Max V/C Distribution (Worst Interval Per Edge)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, framealpha=0.8, facecolor=_DARK_CARD, edgecolor=_GRID_COLOR, labelcolor=_TEXT_COLOR)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 Plot 2 (V/C Ratio) saved to {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Plot 3: Nash Convergence
# ══════════════════════════════════════════════════════════════════════════════

def plot_nash_convergence(
    regret_history: List[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]],
    output_path: str,
):
    """Plot Nash Convergence lines with standard deviation shading.

    Args:
        regret_history: List of (mean_regret, max_regret, std_regret, epsilon_compliance) per episode.
        output_path: Where to save the plot.
    """
    # Save CSV Data
    csv_history_path = output_path.replace('.png', '_history.csv')
    try:
        with open(csv_history_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'mean_regret_sec', 'max_regret_sec', 'std_regret_sec', 'epsilon_compliance'])
            for ep_idx, vals in enumerate(regret_history):
                writer.writerow([ep_idx, vals[0], vals[1], vals[2], vals[3]])
        print(f"  📄 Saved Plot 3 data to {csv_history_path}")
    except Exception as e:
        print(f"  ⚠ Failed to save CSV data for Plot 3: {e}")

    fig, ax = plt.subplots(figsize=(8, 6))
    _apply_dark_theme(ax, fig)

    mean_regrets = [r[0] for r in regret_history if r[0] is not None]
    max_regrets = [r[1] for r in regret_history if r[1] is not None]
    std_regrets = [r[2] for r in regret_history if r[2] is not None]
    eps_compliance = [r[3] for r in regret_history if r[3] is not None]
    eps_with_regret = [i for i, r in enumerate(regret_history) if r[0] is not None]

    if mean_regrets:
        # Convert to minutes
        mean_min = np.array(mean_regrets) / 60.0
        max_min = np.array(max_regrets) / 60.0
        std_min = np.array(std_regrets) / 60.0

        ax.plot(eps_with_regret, mean_min, color=_ACCENT_CYAN, linewidth=2.0,
                 label='Mean Regret', alpha=0.9)
        
        # Shade standard deviation
        ax.fill_between(eps_with_regret, 
                        np.maximum(0, mean_min - std_min), 
                        mean_min + std_min, 
                        color=_ACCENT_CYAN, alpha=0.2, edgecolor='none')

        ax.plot(eps_with_regret, max_min, color=_ACCENT_RED, linewidth=1.5,
                 label='Max Regret', alpha=0.7)
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel('Regret (minutes)', fontsize=11)

        # Epsilon compliance on twin axis
        if eps_compliance:
            ax_twin = ax.twinx()
            _apply_dark_theme(ax_twin)
            ax_twin.plot(eps_with_regret, np.array(eps_compliance) * 100,
                          color=_ACCENT_GREEN, linewidth=1.5, linestyle='--',
                          label='ε-Compliance', alpha=0.8)
            ax_twin.set_ylabel('ε-Compliance (%)', fontsize=11, color=_ACCENT_GREEN)
            ax_twin.tick_params(axis='y', colors=_ACCENT_GREEN)
            ax_twin.set_ylim(0, 105)
            ax_twin.grid(False)

            # Combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_twin.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9,
                       framealpha=0.8, facecolor=_DARK_CARD, edgecolor=_GRID_COLOR, labelcolor=_TEXT_COLOR)
        else:
            ax.legend(fontsize=9, framealpha=0.8, facecolor=_DARK_CARD,
                       edgecolor=_GRID_COLOR, labelcolor=_TEXT_COLOR)
    else:
        ax.text(0.5, 0.5, 'No regret data available',
                 transform=ax.transAxes, ha='center', va='center', color=_TEXT_COLOR, fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 Plot 3 (Nash Convergence) saved to {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Block 1: Route Choice Distribution
# ══════════════════════════════════════════════════════════════════════════════

def plot_route_allocation_heatmap(
    route_allocation_history: List[np.ndarray],
    major_od_idx: int,
    major_od_count: int,
    output_path: str,
):
    """Plot heatmap of route allocation over episodes for the major OD pair.
    
    Args:
        route_allocation_history: List of arrays (one per episode) containing the 
                                  fraction of agents choosing each of the K routes.
        major_od_idx: The integer ID of the major OD pair.
        major_od_count: The number of agents for this OD pair.
        output_path: Where to save the plot.
    """
    if not route_allocation_history:
        print("  ⚠ No route allocation history available for heatmap.")
        return

    data = np.stack(route_allocation_history).T  # [K, num_episodes]
    num_routes, num_episodes = data.shape

    # Save CSV
    csv_path = output_path.replace('.png', '.csv')
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['episode'] + [f'route_{k}_fraction' for k in range(num_routes)]
            writer.writerow(header)
            for ep in range(num_episodes):
                writer.writerow([ep] + data[:, ep].tolist())
        print(f"  📄 Saved Route Allocation Heatmap data to {csv_path}")
    except Exception as e:
        print(f"  ⚠ Failed to save CSV data for Heatmap: {e}")

    fig, ax = plt.subplots(figsize=(10, 5))
    _apply_dark_theme(ax, fig)

    cmap = plt.cm.get_cmap('magma').copy()
    cmap.set_bad(_DARK_BG)
    im = ax.imshow(data, aspect='auto', cmap=cmap, origin='lower',
                   extent=[0, num_episodes, -0.5, num_routes - 0.5], vmin=0, vmax=1)

    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('Agent Fraction (Modal Share)', color=_TEXT_COLOR, fontsize=10)
    cb.ax.tick_params(colors=_TEXT_COLOR, labelsize=8)

    ax.set_xlabel('Training Episodes', fontsize=11)
    ax.set_ylabel('Candidate Routes (k)', fontsize=11)
    ax.set_yticks(range(num_routes))
    ax.set_yticklabels([f'Route {k}' for k in range(num_routes)])
    ax.set_title(f'Route Allocation Heatmap (OD {major_od_idx}, {major_od_count} agents)', fontsize=12, fontweight='bold')

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 Route Allocation Heatmap saved to {output_path}")


def plot_entropy_scatter(
    chosen_paths: np.ndarray,
    od_indices: np.ndarray,
    original_leg_idx: np.ndarray,
    K: int,
    mean_tt: float,
    output_path: str,
    unique_od: Optional[np.ndarray] = None,
):
    """Calculate and save entropy vs mean travel time for convergence."""
    # Filter to actual valid departed legs
    valid_ods = od_indices[original_leg_idx]
    
    unique_ods = np.unique(valid_ods)
    entropies = []
    
    for od in unique_ods:
        od_mask = (valid_ods == od)
        acts = chosen_paths[od_mask]
        if len(acts) > 0:
            counts = np.bincount(acts, minlength=K)
            p = counts / len(acts)
            p_nz = p[p > 0]
            H = -np.sum(p_nz * np.log(p_nz))
            entropies.append(H)
            
    avg_entropy = np.mean(entropies) if entropies else 0.0

    # Save summary CSV
    csv_path = output_path.replace('.png', '.csv')
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['mean_entropy', 'mean_tt_seconds'])
            writer.writerow([avg_entropy, mean_tt])
        print(f"  📄 Saved Entropy data to {csv_path}")
    except Exception as e:
        print(f"  ⚠ Failed to save CSV data for Entropy: {e}")

    # Save raw choices CSV
    raw_csv_path = output_path.replace('.png', '_raw_choices.csv')
    try:
        with open(raw_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['leg_idx', 'od_idx', 'origin_node', 'destination_node', 'chosen_path_idx'])
            for i, leg_idx in enumerate(original_leg_idx):
                od_idx = valid_ods[i]
                act = chosen_paths[i]
                orig, dest = -1, -1
                if unique_od is not None and 0 <= od_idx < len(unique_od):
                    orig, dest = unique_od[od_idx]
                writer.writerow([leg_idx, od_idx, orig, dest, act])
        print(f"  📄 Saved Raw Choices data to {raw_csv_path}")
    except Exception as e:
        print(f"  ⚠ Failed to save Raw Choices data: {e}")

    fig, ax = plt.subplots(figsize=(6, 5))
    _apply_dark_theme(ax, fig)
    
    # Just plotting one point for the current algorithm
    ax.scatter([avg_entropy], [mean_tt], color=_ACCENT_CYAN, s=150, zorder=5, edgecolor=_TEXT_COLOR, label='Current Run')
    
    ax.set_xlim(0, max(np.log(K), avg_entropy) * 1.1)
    ax.set_ylim(max(0, mean_tt * 0.8), mean_tt * 1.2)
    ax.set_xlabel('Average Entropy (H)', fontsize=11)
    ax.set_ylabel('Mean Travel Time (s)', fontsize=11)
    ax.set_title('Choice Entropy vs Mean Travel Time', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9, framealpha=0.8, facecolor=_DARK_CARD, edgecolor=_GRID_COLOR, labelcolor=_TEXT_COLOR)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 Entropy Scatter Plot saved to {output_path}")


def plot_lorenz_concentration(
    chosen_paths: np.ndarray,
    od_indices: np.ndarray,
    original_leg_idx: np.ndarray,
    major_od_idx: int,
    major_od_count: int,
    K: int,
    output_path: str,
):
    """Plot the Lorenz curve of flows for the major OD pair."""
    valid_ods = od_indices[original_leg_idx]
    major_od_mask = (valid_ods == major_od_idx)
    major_acts = chosen_paths[major_od_mask]
    
    if len(major_acts) == 0:
        print("  ⚠ No actions for major OD to plot Lorenz concentration.")
        return
        
    counts = np.bincount(major_acts, minlength=K)
    # Sort routes implicitly by FFTT (since K routes are usually returned sorted by FFTT)
    # So we just take counts as they are: route 0 is shortest, route K-1 is longest.
    p = counts / len(major_acts)
    cumulative_p = np.cumsum(p) * 100

    csv_path = output_path.replace('.png', '.csv')
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['route_idx', 'fraction', 'cumulative_pct'])
            for k in range(K):
                writer.writerow([k, p[k], cumulative_p[k]])
        print(f"  📄 Saved Lorenz data to {csv_path}")
    except Exception as e:
        print(f"  ⚠ Failed to save CSV data for Lorenz: {e}")

    fig, ax = plt.subplots(figsize=(6, 5))
    _apply_dark_theme(ax, fig)
    
    x_routes = np.arange(K)
    ax.plot(x_routes, cumulative_p, marker='o', color=_ACCENT_GREEN, linewidth=2, label='Current Run')
    
    # Random Uniform Line
    ax.plot(x_routes, np.cumsum(np.ones(K)/K) * 100, linestyle='--', color=_TEXT_COLOR, alpha=0.5, label='Uniform (Random)')

    ax.set_xlabel('Routes (Sorted by FFTT)', fontsize=11)
    ax.set_ylabel('Cumulative % of Population', fontsize=11)
    ax.set_xticks(x_routes)
    ax.set_xticklabels([f'R{k}' for k in range(K)])
    ax.set_ylim(0, 105)
    ax.set_title(f'Route Concentration Profile (OD {major_od_idx}, {major_od_count} agents)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.8, facecolor=_DARK_CARD, edgecolor=_GRID_COLOR, labelcolor=_TEXT_COLOR)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 Lorenz Concentration Plot saved to {output_path}")

# ══════════════════════════════════════════════════════════════════════════════
#  Block 2: Agent Distribution on Network Links
# ══════════════════════════════════════════════════════════════════════════════

def plot_spatiotemporal_violins(
    link_counts: torch.Tensor,
    flow_capacity_per_step: torch.Tensor,
    link_tt_interval: float,
    dt: float,
    output_path: str,
):
    """Plot hourly violins of V/C ratios, dropping 0 volume links."""
    counts = link_counts.float().numpy()
    num_intervals, num_edges = counts.shape

    steps_per_interval = link_tt_interval / dt
    capacity_per_interval = flow_capacity_per_step.float().numpy() * steps_per_interval
    safe_cap = np.maximum(capacity_per_interval, 1e-6)

    # Convert to hourly
    intervals_per_hour = int(3600 / link_tt_interval)
    num_hours = int(np.ceil(num_intervals / intervals_per_hour))
    
    hourly_vc_data = []
    
    csv_path = output_path.replace('.png', '.csv')
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['hour', 'median_vc', 'p90_vc', 'max_vc', 'num_active_links'])
            for h in range(num_hours):
                start_i = h * intervals_per_hour
                end_i = min((h + 1) * intervals_per_hour, num_intervals)
                
                # Sum volumes in this hour
                hr_counts = counts[start_i:end_i].sum(axis=0)
                # Capacity in this hour
                hr_cap = safe_cap * (end_i - start_i)
                
                # Drop 0 volume links
                active_mask = hr_counts > 0
                if active_mask.sum() > 0:
                    hr_vc = hr_counts[active_mask] / hr_cap[active_mask]
                    hourly_vc_data.append((h, hr_vc))
                    writer.writerow([h, np.median(hr_vc), np.percentile(hr_vc, 90), np.max(hr_vc), np.sum(active_mask)])
                else:
                    hourly_vc_data.append((h, np.array([])))
                    writer.writerow([h, 0, 0, 0, 0])
        print(f"  📄 Saved Hourly Violins data to {csv_path}")
    except Exception as e:
        print(f"  ⚠ Failed to save CSV data for Hourly Violins: {e}")

    fig, ax = plt.subplots(figsize=(12, 5))
    _apply_dark_theme(ax, fig)

    valid_hours = [h for h, vc in hourly_vc_data if len(vc) > 0]
    valid_data = [vc for h, vc in hourly_vc_data if len(vc) > 0]

    if valid_data:
        parts = ax.violinplot(valid_data, positions=valid_hours, showmeans=False, showextrema=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(_ACCENT_PURPLE)
            pc.set_edgecolor(_TEXT_COLOR)
            pc.set_alpha(0.7)
        parts['cmedians'].set_color(_ACCENT_GREEN)
        parts['cmins'].set_color(_TEXT_COLOR)
        parts['cmaxes'].set_color(_TEXT_COLOR)
        parts['cbars'].set_color(_TEXT_COLOR)

        ax.axhline(1.0, color=_ACCENT_RED, linestyle='--', alpha=0.8, label='Capacity (V/C=1.0)')

    ax.set_xlabel('Time (Hour)', fontsize=11)
    ax.set_ylabel('V/C Ratio (Active Links Only)', fontsize=11)
    ax.set_title('Spatio-Temporal Distribution of Load (Hourly Violins)', fontsize=12, fontweight='bold')
    if valid_hours:
        ax.set_xticks(valid_hours)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.8, facecolor=_DARK_CARD, edgecolor=_GRID_COLOR, labelcolor=_TEXT_COLOR)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 Spatiotemporal Violins Plot saved to {output_path}")


def plot_critical_corridors_heatmap(
    link_counts: torch.Tensor,
    flow_capacity_per_step: torch.Tensor,
    link_tt_interval: float,
    dt: float,
    routes_flat_csr: torch.Tensor,
    output_path: str,
):
    """Plot V/C heatmap for the Top 100 links by Routing Centrality."""
    if routes_flat_csr is None:
        print("  ⚠ routes_flat_csr not available. Skipping Critical Corridors Heatmap.")
        return

    # Compute routing centrality
    flat_routes_np = routes_flat_csr.numpy()
    valid_edges = flat_routes_np[flat_routes_np >= 0]
    edge_counts = np.bincount(valid_edges)
    
    # Get top 100 edges
    top_100_edges = np.argsort(edge_counts)[::-1][:100]
    top_100_edges = top_100_edges[edge_counts[top_100_edges] > 0] # Filter out edges with 0 frequency

    if len(top_100_edges) == 0:
        print("  ⚠ No active edges found in candidate routes.")
        return

    counts = link_counts.float().numpy()
    steps_per_interval = link_tt_interval / dt
    capacity_per_interval = flow_capacity_per_step.float().numpy() * steps_per_interval
    safe_cap = np.maximum(capacity_per_interval, 1e-6)
    vc_matrix = counts / safe_cap[None, :]

    # Extract V/C for top 100 links [Time, Top100]
    heatmap_data = vc_matrix[:, top_100_edges].T  # [Top100, Time]

    csv_path = output_path.replace('.png', '.csv')
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            time_labels = [f"t_{int((i * link_tt_interval) // 60)}" for i in range(heatmap_data.shape[1])]
            writer.writerow(['link_idx', 'centrality'] + time_labels)
            for i, edge in enumerate(top_100_edges):
                writer.writerow([edge, edge_counts[edge]] + heatmap_data[i].tolist())
        print(f"  📄 Saved Critical Corridors Heatmap data to {csv_path}")
    except Exception as e:
        print(f"  ⚠ Failed to save CSV data for Critical Corridors: {e}")

    fig, ax = plt.subplots(figsize=(12, 6))
    _apply_dark_theme(ax, fig)

    cmap = plt.cm.get_cmap('jet').copy()  # Blue to Red
    cmap.set_bad(_DARK_BG)
    
    im = ax.imshow(heatmap_data, aspect='auto', cmap=cmap, origin='upper', vmin=0, vmax=2.0)
    
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('V/C Ratio', color=_TEXT_COLOR, fontsize=10)
    cb.ax.tick_params(colors=_TEXT_COLOR, labelsize=8)

    ax.set_xlabel(f'Time Bins ({int(link_tt_interval // 60)} min)', fontsize=11)
    ax.set_ylabel('Top Critical Links (Sorted by Centrality)', fontsize=11)
    ax.set_title('Spatiotemporal Corridors Heatmap (Top Critical Links)', fontsize=12, fontweight='bold')

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 Critical Corridors Heatmap saved to {output_path}")

# ══════════════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def generate_sanity_checks(
    best_sanity_data: Dict,
    regret_history: List[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]],
    output_dir: str,
    route_allocation_history: Optional[List[np.ndarray]] = None,
    scenario_id: str = "",
    agent_type: str = "",
):
    """Generate the complete sanity check triptych and raw CSV data files.

    Called at the end of training when --sanity_checks is enabled.

    Args:
        best_sanity_data: Dict with keys: episode, mean_tt, realized_tt, fftt_chosen,
                          link_counts, flow_capacity_per_step, link_tt_interval, dt.
        regret_histograms: List of (episode, bin_edges, bin_counts) from GPU binning.
        regret_history: List of (mean_regret, max_regret, epsilon_compliance) per episode.
        output_dir: Directory to save the plots and CSV files.
        scenario_id: Scenario identifier.
        agent_type: Agent type.
    """
    # ── Plot 1: Travel Time vs FFTT ──
    plot_tt_vs_fftt(
        realized_tt=best_sanity_data['realized_tt'],
        fftt_chosen=best_sanity_data['fftt_chosen'],
        output_path=os.path.join(output_dir, 'sanity_01_tt_vs_fftt.png'),
        original_idx=best_sanity_data.get('original_leg_idx'),
        planned_dep=best_sanity_data.get('planned_dep'),
        actual_dep=best_sanity_data.get('actual_dep'),
        arrival=best_sanity_data.get('arrival'),
        chosen_path_idx=best_sanity_data.get('chosen_path_idx'),
    )

    # ── Plot 2: V/C Ratio ──
    if 'link_counts' in best_sanity_data:
        plot_vc_ratio(
            link_counts=best_sanity_data['link_counts'],
            flow_capacity_per_step=best_sanity_data['flow_capacity_per_step'],
            link_tt_interval=best_sanity_data['link_tt_interval'],
            dt=best_sanity_data['dt'],
            output_path=os.path.join(output_dir, 'sanity_02_vc_ratio.png'),
        )
    else:
        print("  ⚠ Skipping V/C plot — link_counts not available (collect_link_tt was disabled)")

    # ── Plot 3: Nash Convergence ──
    plot_nash_convergence(
        regret_history=regret_history,
        output_path=os.path.join(output_dir, 'sanity_03_regret_convergence.png'),
    )

    # ── New Plots (Block 1 & 2) ──
    if route_allocation_history and best_sanity_data.get('major_od_idx') is not None:
        plot_route_allocation_heatmap(
            route_allocation_history=route_allocation_history,
            major_od_idx=best_sanity_data['major_od_idx'],
            major_od_count=best_sanity_data.get('major_od_count', 0),
            output_path=os.path.join(output_dir, 'sanity_04_route_heatmap.png'),
        )

    if 'chosen_path_idx' in best_sanity_data and 'od_indices' in best_sanity_data and best_sanity_data['od_indices'] is not None:
        plot_entropy_scatter(
            chosen_paths=best_sanity_data['chosen_path_idx'].numpy(),
            od_indices=best_sanity_data['od_indices'].numpy(),
            original_leg_idx=best_sanity_data.get('original_leg_idx', np.arange(len(best_sanity_data['chosen_path_idx']))),
            K=best_sanity_data.get('K', 3),
            mean_tt=best_sanity_data.get('mean_tt', 0.0),
            output_path=os.path.join(output_dir, 'sanity_05_entropy_scatter.png'),
            unique_od=best_sanity_data.get('unique_od'),
        )
        
        if best_sanity_data.get('major_od_idx') is not None:
            plot_lorenz_concentration(
                chosen_paths=best_sanity_data['chosen_path_idx'].numpy(),
                od_indices=best_sanity_data['od_indices'].numpy(),
                original_leg_idx=best_sanity_data.get('original_leg_idx', np.arange(len(best_sanity_data['chosen_path_idx']))),
                major_od_idx=best_sanity_data['major_od_idx'],
                major_od_count=best_sanity_data.get('major_od_count', 0),
                K=best_sanity_data.get('K', 3),
                output_path=os.path.join(output_dir, 'sanity_06_lorenz_concentration.png'),
            )

    if 'link_counts' in best_sanity_data:
        plot_spatiotemporal_violins(
            link_counts=best_sanity_data['link_counts'],
            flow_capacity_per_step=best_sanity_data['flow_capacity_per_step'],
            link_tt_interval=best_sanity_data['link_tt_interval'],
            dt=best_sanity_data['dt'],
            output_path=os.path.join(output_dir, 'sanity_07_spatiotemporal_violins.png'),
        )

        plot_critical_corridors_heatmap(
            link_counts=best_sanity_data['link_counts'],
            flow_capacity_per_step=best_sanity_data['flow_capacity_per_step'],
            link_tt_interval=best_sanity_data['link_tt_interval'],
            dt=best_sanity_data['dt'],
            routes_flat_csr=best_sanity_data.get('routes_flat_csr'),
            output_path=os.path.join(output_dir, 'sanity_08_critical_corridors.png'),
        )

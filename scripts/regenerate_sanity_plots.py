import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Ensure tamarl is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tamarl.visualisation.sanity_checks import (
    _ACCENT_CYAN,
    _ACCENT_GREEN,
    _ACCENT_ORANGE,
    _ACCENT_RED,
    _DARK_BG,
    _DARK_CARD,
    _GRID_COLOR,
    _TEXT_COLOR,
    _apply_dark_theme,
    plot_tt_vs_fftt,
)


def replot_vc_ratio(peak_csv, max_csv, output_path):
    if not os.path.exists(peak_csv) or not os.path.exists(max_csv):
        print("  ⚠ Missing Plot 2 CSV files.")
        return

    df_peak = pd.read_csv(peak_csv)
    df_max = pd.read_csv(max_csv)

    vc_peak = df_peak["vc_ratio"].values
    max_vc = df_max["max_vc_ratio"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    _apply_dark_theme(ax1, fig)
    _apply_dark_theme(ax2)

    # ── Peak-Hour Histogram ──
    bins_vc = np.linspace(0, min(vc_peak.max() * 1.1, 5.0) if len(vc_peak) > 0 else 3.0, 50)
    ax1.hist(
        vc_peak,
        bins=bins_vc,
        color=_ACCENT_CYAN,
        alpha=0.85,
        edgecolor="none",
        label="Peak interval",
    )
    ax1.axvline(
        x=1.0,
        color=_ACCENT_RED,
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="V/C = 1.0 (capacity)",
    )

    over_capacity = (vc_peak > 1.0).sum()
    pct_over = over_capacity / len(vc_peak) * 100 if len(vc_peak) > 0 else 0
    pct_active = 100.0  # As per user specification for large networks

    ax1.text(
        0.98,
        0.98,
        f"Peak interval: Worst Interval\n"
        f"edges > 1.0 : {pct_over:.1f}%\n"
        f"Active edges: ~{pct_active:.1f}%",
        transform=ax1.transAxes,
        fontsize=9,
        color=_TEXT_COLOR,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=_DARK_BG, edgecolor=_GRID_COLOR, alpha=0.9),
    )

    ax1.set_xlabel("V/C Ratio", fontsize=11)
    ax1.set_ylabel("Number of Edges", fontsize=11)
    ax1.set_title("Peak-Hour V/C Distribution", fontsize=12, fontweight="bold")
    ax1.legend(
        fontsize=9,
        framealpha=0.8,
        facecolor=_DARK_CARD,
        edgecolor=_GRID_COLOR,
        labelcolor=_TEXT_COLOR,
    )

    # ── Max V/C Histogram ──
    bins_max = np.linspace(0, min(max_vc.max() * 1.1, 5.0) if len(max_vc) > 0 else 3.0, 50)
    ax2.hist(
        max_vc,
        bins=bins_max,
        color=_ACCENT_ORANGE,
        alpha=0.85,
        edgecolor="none",
        label="Max V/C per edge",
    )
    ax2.axvline(
        x=1.0,
        color=_ACCENT_RED,
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="V/C = 1.0 (capacity)",
    )

    over_max = (max_vc > 1.0).sum()
    pct_over_max = over_max / len(max_vc) * 100 if len(max_vc) > 0 else 0

    ax2.text(
        0.98,
        0.98,
        f"edges > 1.0 : {pct_over_max:.1f}%\n"
        f"Median max V/C: {np.median(max_vc) if len(max_vc) > 0 else 0:.2f}",
        transform=ax2.transAxes,
        fontsize=9,
        color=_TEXT_COLOR,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=_DARK_BG, edgecolor=_GRID_COLOR, alpha=0.9),
    )

    ax2.set_xlabel("Max V/C Ratio", fontsize=11)
    ax2.set_ylabel("Number of Edges", fontsize=11)
    ax2.set_title("Max V/C Distribution (Worst Interval Per Edge)", fontsize=12, fontweight="bold")
    ax2.legend(
        fontsize=9,
        framealpha=0.8,
        facecolor=_DARK_CARD,
        edgecolor=_GRID_COLOR,
        labelcolor=_TEXT_COLOR,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Plot 2 (V/C Ratio) regenerated to {output_path}")


def replot_nash_convergence(history_csv, output_path):
    if not os.path.exists(history_csv):
        print(f"  ⚠ Missing Plot 3 CSV file: {history_csv}")
        return

    df = pd.read_csv(history_csv)
    eps_with_regret = df["episode"].values
    mean_min = df["mean_regret_sec"].values / 60.0
    max_min = df["max_regret_sec"].values / 60.0
    std_min = df["std_regret_sec"].values / 60.0
    eps_compliance = df["epsilon_compliance"].values

    fig, ax = plt.subplots(figsize=(8, 6))
    _apply_dark_theme(ax, fig)

    if len(eps_with_regret) > 0:
        ax.plot(
            eps_with_regret,
            mean_min,
            color=_ACCENT_CYAN,
            linewidth=2.0,
            label="Mean Regret",
            alpha=0.9,
        )
        ax.fill_between(
            eps_with_regret,
            np.maximum(0, mean_min - std_min),
            mean_min + std_min,
            color=_ACCENT_CYAN,
            alpha=0.2,
            edgecolor="none",
        )
        ax.plot(
            eps_with_regret,
            max_min,
            color=_ACCENT_RED,
            linewidth=1.5,
            label="Max Regret",
            alpha=0.7,
        )
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel("Regret (minutes)", fontsize=11)

        if len(eps_compliance) > 0:
            ax_twin = ax.twinx()
            _apply_dark_theme(ax_twin)
            ax_twin.plot(
                eps_with_regret,
                eps_compliance * 100,
                color=_ACCENT_GREEN,
                linewidth=1.5,
                linestyle="--",
                label="ε-Compliance",
                alpha=0.8,
            )
            ax_twin.set_ylabel("ε-Compliance (%)", fontsize=11, color=_ACCENT_GREEN)
            ax_twin.tick_params(axis="y", colors=_ACCENT_GREEN)
            ax_twin.set_ylim(0, 105)
            ax_twin.grid(False)

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax_twin.get_legend_handles_labels()
            ax.legend(
                lines1 + lines2,
                labels1 + labels2,
                loc="upper right",
                fontsize=9,
                framealpha=0.8,
                facecolor=_DARK_CARD,
                edgecolor=_GRID_COLOR,
                labelcolor=_TEXT_COLOR,
            )
        else:
            ax.legend(
                fontsize=9,
                framealpha=0.8,
                facecolor=_DARK_CARD,
                edgecolor=_GRID_COLOR,
                labelcolor=_TEXT_COLOR,
            )
    else:
        ax.text(
            0.5,
            0.5,
            "No regret data available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=_TEXT_COLOR,
            fontsize=11,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"  📊 Plot 3 (Nash Convergence) regenerated to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate Sanity Check Plots from existing CSV files."
    )
    parser.add_argument(
        "dir", type=str, help="Directory containing the graph CSV files (e.g. graphs/)"
    )
    parser.add_argument(
        "--plots",
        type=str,
        default="1,2,3",
        help="Comma separated list of plots to regenerate (e.g., '1,3' or '2')",
    )
    parser.add_argument(
        "--recompute_ff",
        action="store_true",
        help="Recompute Free-Flow Travel Time (ignore first edge) and overwrite CSV",
    )
    parser.add_argument(
        "--population_filter",
        type=str,
        default=None,
        help="Substring to match the correct population file (e.g. '25pct')",
    )
    args = parser.parse_args()

    target_dir = args.dir
    if not os.path.isdir(target_dir):
        print(f"Error: Directory not found at {target_dir}")
        sys.exit(1)

    plots_to_gen = [p.strip() for p in args.plots.split(",")]

    print(f"Regenerating plots {args.plots} in {target_dir}...")

    if "1" in plots_to_gen:
        csv_1 = os.path.join(target_dir, "sanity_01_tt_vs_fftt.csv")
        out_1 = os.path.join(target_dir, "sanity_01_tt_vs_fftt.png")
        if os.path.exists(csv_1):
            df = pd.read_csv(csv_1)

            if args.recompute_ff:
                from tamarl.envs.scenario_loader import load_scenario

                parts = os.path.abspath(target_dir).split(os.sep)
                try:
                    scen_idx = parts.index("scenarios")
                    scenario_path = os.sep.join(parts[: scen_idx + 2])
                except ValueError:
                    scenario_path = None

                if scenario_path and os.path.exists(scenario_path):
                    print(
                        f"  [Recompute] Loading scenario from {scenario_path} (Filter: {args.population_filter})..."
                    )
                    scenario_data = load_scenario(
                        scenario_path, population_filter=args.population_filter
                    )

                    leg_first_edges = []
                    fe_np = scenario_data.first_edges.numpy()
                    nl_np = scenario_data.num_legs.numpy()
                    for i in range(scenario_data.num_agents):
                        for j in range(nl_np[i]):
                            leg_first_edges.append(fe_np[i, j])

                    leg_first_edges = np.array(leg_first_edges)
                    first_edge_fftt = scenario_data.edge_static[leg_first_edges, 4].numpy()

                    print(f"  [Recompute] Subtracting first edge FFTT from {len(df)} agents...")
                    df["fftt_sec"] = (df["fftt_sec"] - first_edge_fftt).clip(lower=0.0)
                    df["fftt_min"] = df["fftt_sec"] / 60.0

                    df.to_csv(csv_1, index=False)
                    print(f"  [Recompute] Overwrote {csv_1} with corrected Free-Flow times.")
                else:
                    print(
                        "  ⚠ Could not infer scenario path to recompute FFTT. Make sure the directory is within scenarios/..."
                    )

            fftt_chosen = torch.tensor(df["fftt_sec"].values, dtype=torch.float32)
            realized_tt = torch.tensor(df["realized_tt_sec"].values, dtype=torch.float32)
            plot_tt_vs_fftt(realized_tt=realized_tt, fftt_chosen=fftt_chosen, output_path=out_1)
        else:
            print(f"  ⚠ Missing Plot 1 CSV file: {csv_1}")

    if "2" in plots_to_gen:
        csv_2_peak = os.path.join(target_dir, "sanity_02_vc_ratio_peak.csv")
        csv_2_max = os.path.join(target_dir, "sanity_02_vc_ratio_max.csv")
        out_2 = os.path.join(target_dir, "sanity_02_vc_ratio.png")
        replot_vc_ratio(csv_2_peak, csv_2_max, out_2)

    if "3" in plots_to_gen:
        csv_3 = os.path.join(target_dir, "sanity_03_regret_convergence_history.csv")
        out_3 = os.path.join(target_dir, "sanity_03_regret_convergence.png")
        replot_nash_convergence(csv_3, out_3)

    print("Done!")


if __name__ == "__main__":
    main()

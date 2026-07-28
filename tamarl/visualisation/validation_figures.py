"""Publication-quality (AAAI supplementary material) figures for the
TorchDNL-vs-MATSim DNL-fidelity validation appendix.

This module is presentation-only: every number plotted here is produced by
`tamarl.validation.geh` / `tamarl.validation.distributional` / the parsers in
`tamarl.validation.matsim_io`, exactly as in `scripts/validate_against_matsim.py`
(which writes the companion `summary.md`). Nothing here recomputes metrics —
it only lays them out.

Design notes (see the repo's dataviz skill for the full method):
  - Two-color categorical palette, held fixed across every figure:
    MATSim = slot-1 blue (#2a78d6), TorchDNL = slot-2 orange (#eb6834).
    Validated colorblind-safe (adjacent CVD dE 24.7, normal-vision dE 33.6,
    both well above the dE>=8 / dE>=15 gates).
  - The four-series GEH figure reuses categorical slots 1-4 (blue, orange,
    aqua, yellow); slots 3-4 sit under the 3:1 contrast floor on a light
    surface, so that figure ships direct end-labels + a legend (the required
    "relief channel") rather than color alone.
  - Vector PDF output only (no raster), serif/mathtext publication font,
    sized for direct \\includegraphics at typical single-column widths.
  - Captions live in the paper's LaTeX source, so figures carry precise axis
    labels/legends but no in-figure titles (consistent across the set).
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Palette (see tamarl/data/scenarios/sioux_falls/dnl_validation/report/ for
# the validator output backing these choices).
# --------------------------------------------------------------------------
COLOR_MATSIM = "#2a78d6"      # categorical slot 1 (blue)
COLOR_TORCHDNL = "#eb6834"    # categorical slot 2 (orange)
COLOR_AQUA = "#1baf7a"        # categorical slot 3 - used only in the 4-series GEH figure
COLOR_YELLOW = "#eda100"      # categorical slot 4 - used only in the 4-series GEH figure

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

GEH_SERIES_COLORS = {
    "en_route": COLOR_MATSIM,
    "departures": COLOR_TORCHDNL,
    "arrivals": COLOR_AQUA,
    "link_volume": COLOR_YELLOW,
}
GEH_SERIES_LABELS = {
    "en_route": "En-route agents",
    "departures": "Departures",
    "arrivals": "Arrivals",
    "link_volume": "Link volume (full-day)",
}
GEH_SERIES_LINESTYLES = {
    "en_route": "-",
    "departures": "--",
    "arrivals": "-.",
    "link_volume": ":",
}

_STYLE_APPLIED = False


def setup_style() -> bool:
    """Configure matplotlib rcParams for AAAI-ready vector figures.

    Tries LaTeX (`text.usetex`) first for true publication typesetting;
    falls back to matplotlib's built-in STIX math/serif fonts (no external
    LaTeX install required) if a quick smoke-test render fails.

    Returns:
        True if `text.usetex` is active, False if the STIX fallback is used
        (callers may want to report this to the user as a caveat).
    """
    global _STYLE_APPLIED
    import matplotlib

    matplotlib.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "lines.linewidth": 1.3,
            "pdf.fonttype": 42,  # embed as proper (non-Type3) outlines
            "ps.fonttype": 42,
            "axes.unicode_minus": True,
        }
    )

    usetex_ok = _try_usetex()
    if usetex_ok:
        matplotlib.rcParams["text.usetex"] = True
        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    else:
        matplotlib.rcParams["text.usetex"] = False
        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["mathtext.fontset"] = "stix"
        matplotlib.rcParams["font.serif"] = ["STIXGeneral", "DejaVu Serif"]

    _STYLE_APPLIED = True
    return usetex_ok


def _try_usetex() -> bool:
    """Smoke-test that a full LaTeX toolchain (latex + dvipng/type1cm etc.)
    actually works with matplotlib, rather than assuming from `which latex`."""
    import io

    import matplotlib

    saved = dict(matplotlib.rcParams)
    try:
        matplotlib.rcParams["text.usetex"] = True
        matplotlib.rcParams["font.family"] = "serif"
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.set_xlabel(r"Test $x^2$ label")
        buf = io.BytesIO()
        fig.savefig(buf, format="pdf")
        plt.close(fig)
        return True
    except Exception:
        return False
    finally:
        matplotlib.rcParams.update(saved)


def _clean_axes(ax, x_grid=False, y_grid=True):
    """Apply the shared spine/gridline/tick treatment to every figure."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelcolor=INK_PRIMARY)
    ax.xaxis.label.set_color(INK_PRIMARY)
    ax.yaxis.label.set_color(INK_PRIMARY)
    if y_grid:
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.7, zorder=0)
    if x_grid:
        ax.grid(axis="x", color=GRIDLINE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, output_path: str) -> None:
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.03)
    import matplotlib.pyplot as plt

    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 1: en-route envelope
# --------------------------------------------------------------------------
def plot_en_route_envelope(
    bins: np.ndarray,
    matsim_en_route: np.ndarray,  # [num_seeds, num_bins]
    torchdnl_en_route: np.ndarray,  # [num_seeds, num_bins]
    output_path: str,
) -> None:
    """Symmetric two-envelope comparison: MATSim's inter-seed IQR band vs
    TorchDNL's own inter-seed IQR band, both over hour-of-day, plus each
    simulator's median. Avoids the clutter of 10 overlapping per-seed lines
    while still showing each simulator's own stochastic spread."""
    import matplotlib.pyplot as plt

    from tamarl.validation import distributional

    m_median, m_lower, m_upper = distributional.build_envelope(matsim_en_route)
    t_median, t_lower, t_upper = distributional.build_envelope(torchdnl_en_route)

    x = bins[:-1] / 3600.0

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    _clean_axes(ax)

    ax.fill_between(
        x, m_lower, m_upper, step="post", color=COLOR_MATSIM, alpha=0.18,
        linewidth=0, zorder=1, label="MATSim (10-seed IQR)",
    )
    ax.fill_between(
        x, t_lower, t_upper, step="post", color=COLOR_TORCHDNL, alpha=0.18,
        linewidth=0, zorder=1, label="TorchDNL (10-seed IQR)",
    )
    ax.step(x, m_median, where="post", color=COLOR_MATSIM, linewidth=1.4, zorder=3, label="MATSim median")
    ax.step(x, t_median, where="post", color=COLOR_TORCHDNL, linewidth=1.4, zorder=3, label="TorchDNL median",
             linestyle=(0, (4, 1.5)))

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Agents en route")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(bottom=0)
    if x.max() - x.min() >= 12:
        ax.xaxis.set_major_locator(plt.MultipleLocator(4))
    else:
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))

    # Both AM and PM peaks reach close to the axis ceiling, so the legend is
    # anchored over the (empty) inter-peak trough rather than a corner.
    legend = ax.legend(
        loc="upper center", bbox_to_anchor=(0.47, 1.0), ncol=2, columnspacing=1.0,
        frameon=True, facecolor="white", edgecolor=GRIDLINE, framealpha=0.9,
        handlelength=1.5, labelspacing=0.3, borderaxespad=0.2,
    )
    legend.get_frame().set_linewidth(0.7)
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)

    _save(fig, output_path)


# --------------------------------------------------------------------------
# Figure 2: pooled trip travel-time ECDF
# --------------------------------------------------------------------------
def plot_travel_time_ecdf(
    matsim_tt: np.ndarray,
    torchdnl_tt: np.ndarray,
    wasserstein: float,
    ks_stat: float,
    ks_pvalue: float,
    output_path: str,
) -> None:
    """ECDF comparison of pooled trip travel times. An ECDF reads more
    rigorously than overlapping histograms here: it needs no bin-width
    choice, makes the KS statistic (the max vertical gap between the two
    curves) directly visible, and two near-coincident step curves are an
    unambiguous "these are the same distribution" signal in a way that two
    translucent, bin-dependent histograms are not."""
    import matplotlib.pyplot as plt

    def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_sorted = np.sort(values)
        y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
        return x_sorted, y

    m_x, m_y = ecdf(matsim_tt)
    t_x, t_y = ecdf(torchdnl_tt)

    # x-axis in minutes reads more naturally than raw seconds for trip times.
    m_x_min = m_x / 60.0
    t_x_min = t_x / 60.0
    x_cap = np.percentile(np.concatenate([m_x_min, t_x_min]), 99.5)

    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    _clean_axes(ax)

    ax.plot(m_x_min, m_y, color=COLOR_MATSIM, linewidth=1.5, drawstyle="steps-post",
            zorder=3, label="MATSim")
    ax.plot(t_x_min, t_y, color=COLOR_TORCHDNL, linewidth=1.5, drawstyle="steps-post",
            zorder=3, label="TorchDNL", linestyle=(0, (4, 1.5)))

    ax.set_xlabel("Trip travel time (min)")
    ax.set_ylabel("Cumulative fraction of trips")
    ax.set_xlim(0, x_cap)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))

    legend = ax.legend(loc="lower right", frameon=False, handlelength=1.6, borderaxespad=0.2)
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)

    stat_text = (
        f"Wasserstein $=$ {wasserstein:.1f}\\,s\n"
        f"KS $=$ {ks_stat:.3f} ($p={ks_pvalue:.2g}$)"
    )
    ax.text(
        0.03, 0.97, stat_text, transform=ax.transAxes, ha="left", va="top",
        fontsize=6.5, color=INK_SECONDARY,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=GRIDLINE, linewidth=0.7),
    )

    _save(fig, output_path)


# --------------------------------------------------------------------------
# Figure 3: per-link full-day volume parity
# --------------------------------------------------------------------------
def plot_link_volume_parity(
    matsim_link_vol: np.ndarray,  # [num_seeds, num_links]
    torchdnl_link_vol: np.ndarray,  # [num_seeds, num_links]
    output_path: str,
) -> None:
    """Per-link full-day volume: TorchDNL vs MATSim, seed-averaged, one
    point per network link, against the y=x parity line. The strongest,
    most visually "irrefutable" result in the validation set (link-volume
    GEH mean/median ~= 0), so it gets its own dedicated scatter."""
    import matplotlib.pyplot as plt

    m_mean = matsim_link_vol.mean(axis=0)
    t_mean = torchdnl_link_vol.mean(axis=0)

    lim_max = max(m_mean.max(), t_mean.max()) * 1.05

    r = np.corrcoef(m_mean, t_mean)[0, 1]

    fig, ax = plt.subplots(figsize=(3.1, 3.0))
    _clean_axes(ax, x_grid=True, y_grid=True)

    ax.plot([0, lim_max], [0, lim_max], color=BASELINE, linewidth=1.0, linestyle="-", zorder=1,
            label="$y = x$")
    ax.scatter(
        m_mean, t_mean, s=11, facecolor=COLOR_TORCHDNL, edgecolor="white", linewidth=0.3,
        alpha=0.75, zorder=3,
    )

    ax.set_xlabel("MATSim link volume (veh/day)")
    ax.set_ylabel("TorchDNL link volume (veh/day)")
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.set_aspect("equal", adjustable="box")

    ax.text(
        0.04, 0.96, f"$r = {r:.4f}$\n$n = {len(m_mean)}$ links",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.5, color=INK_SECONDARY,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=GRIDLINE, linewidth=0.7),
    )
    legend = ax.legend(loc="lower right", frameon=False, handlelength=1.4, borderaxespad=0.2)
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)

    _save(fig, output_path)


# --------------------------------------------------------------------------
# Figure 4: GEH-distribution summary across series
# --------------------------------------------------------------------------
def plot_geh_distribution(
    geh_values_by_series: dict[str, np.ndarray],
    output_path: str,
    order: tuple[str, ...] = ("en_route", "departures", "arrivals", "link_volume"),
) -> None:
    """ECDF of GEH values for each of the four validation series, with the
    standard traffic-engineering acceptance thresholds (GEH<5 good,
    GEH<10 marginal) marked as reference lines. Ships direct end-labels in
    addition to the legend since two of the four series colors sit under
    the 3:1 contrast floor on a light surface (see module docstring)."""
    import matplotlib.pyplot as plt

    def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_sorted = np.sort(values)
        y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
        return x_sorted, y

    fig, ax = plt.subplots(figsize=(3.35, 2.5))
    _clean_axes(ax)

    x_max = 0.0
    for name in order:
        values = np.asarray(geh_values_by_series[name])
        x, y = ecdf(values)
        x_max = max(x_max, np.percentile(x, 99.5))
        ax.plot(
            x, y, color=GEH_SERIES_COLORS[name], linestyle=GEH_SERIES_LINESTYLES[name],
            linewidth=1.4, drawstyle="steps-post", zorder=3, label=GEH_SERIES_LABELS[name],
        )

    x_max = max(x_max, 11.0)
    ax.axvline(5.0, color=INK_MUTED, linewidth=0.8, linestyle=":", zorder=2)
    ax.axvline(10.0, color=INK_MUTED, linewidth=0.8, linestyle=":", zorder=2)
    # Threshold labels sit near the top, clear of the lower-right legend box.
    ax.text(5.0, 0.93, "GEH$<$5", rotation=90, ha="right", va="top", fontsize=6, color=INK_MUTED)
    ax.text(10.0, 0.93, "GEH$<$10", rotation=90, ha="right", va="top", fontsize=6, color=INK_MUTED)

    ax.set_xlabel("GEH statistic")
    ax.set_ylabel("Cumulative fraction of observations")
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1)

    legend = ax.legend(
        loc="lower right", frameon=True, facecolor="white", edgecolor=GRIDLINE,
        framealpha=0.9, handlelength=1.8, labelspacing=0.3, fontsize=6.5,
        borderaxespad=0.3,
    )
    legend.get_frame().set_linewidth(0.7)
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)

    _save(fig, output_path)


# --------------------------------------------------------------------------
# Figure 5: cross-scenario summary grid (3 fidelity signals x N scenarios)
# --------------------------------------------------------------------------
def plot_combined_scenario_grid(scenarios: list[dict], output_path: str) -> None:
    """One figure summarizing DNL fidelity across multiple scenarios: rows
    are the three most legible fidelity signals (temporal en-route envelope,
    per-link volume parity, pooled travel-time distribution), columns are
    scenarios ordered by increasing scale. A single shared legend replaces
    the per-panel legends a naive grid would need; only the leftmost column
    carries y-axis labels, matching standard small-multiples practice.

    Each element of `scenarios` is a dict with keys: `label`, `n_trips`,
    `bins`, `matsim_en_route`, `torchdnl_en_route` (row 1),
    `matsim_link_vol`, `torchdnl_link_vol` (row 2), `matsim_trip_tt`,
    `torchdnl_trip_tt` (row 3). Pass scenarios smallest-to-largest so the
    grid reads left-to-right as a scale progression.
    """
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    from tamarl.validation import distributional

    n_cols = len(scenarios)
    fig, axes = plt.subplots(3, n_cols, figsize=(2.05 * n_cols, 5.3))
    if n_cols == 1:
        axes = axes.reshape(3, 1)

    def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_sorted = np.sort(values)
        y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
        return x_sorted, y

    for col, sc in enumerate(scenarios):
        # ---- Row 0: en-route envelope ----
        ax = axes[0, col]
        _clean_axes(ax)
        m_median, m_lower, m_upper = distributional.build_envelope(sc["matsim_en_route"])
        t_median, t_lower, t_upper = distributional.build_envelope(sc["torchdnl_en_route"])
        x = sc["bins"][:-1] / 3600.0
        ax.fill_between(x, m_lower, m_upper, step="post", color=COLOR_MATSIM, alpha=0.18, linewidth=0, zorder=1)
        ax.fill_between(x, t_lower, t_upper, step="post", color=COLOR_TORCHDNL, alpha=0.18, linewidth=0, zorder=1)
        ax.step(x, m_median, where="post", color=COLOR_MATSIM, linewidth=1.1, zorder=3)
        ax.step(x, t_median, where="post", color=COLOR_TORCHDNL, linewidth=1.1, zorder=3, linestyle=(0, (4, 1.5)))
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(plt.MultipleLocator(8) if x.max() - x.min() >= 12 else plt.MaxNLocator(nbins=4))
        ax.set_title(f"{sc['label']}" + "\n" + rf"($n={sc['n_trips']:,}$ trips)", fontsize=7.5, pad=6)
        ax.set_xlabel("Hour of day", fontsize=6.5)
        if col == 0:
            ax.set_ylabel("Agents en route")

        # ---- Row 1: link-volume parity ----
        ax = axes[1, col]
        _clean_axes(ax, x_grid=True)
        m_mean = sc["matsim_link_vol"].mean(axis=0)
        t_mean = sc["torchdnl_link_vol"].mean(axis=0)
        lim_max = max(m_mean.max(), t_mean.max()) * 1.05
        r = np.corrcoef(m_mean, t_mean)[0, 1]
        ax.plot([0, lim_max], [0, lim_max], color=BASELINE, linewidth=0.9, zorder=1)
        ax.scatter(m_mean, t_mean, s=6, facecolor=COLOR_TORCHDNL, edgecolor="white", linewidth=0.2, alpha=0.7, zorder=3)
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)
        ax.set_aspect("equal", adjustable="box")
        ax.text(
            0.05, 0.95, f"$r={r:.4f}$", transform=ax.transAxes, ha="left", va="top",
            fontsize=6, color=INK_SECONDARY,
        )
        ax.set_xlabel("MATSim volume (veh/day)", fontsize=6.5)
        if col == 0:
            ax.set_ylabel("TorchDNL volume\n(veh/day)")

        # ---- Row 2: pooled travel-time ECDF ----
        ax = axes[2, col]
        _clean_axes(ax)
        m_x, m_y = ecdf(sc["matsim_trip_tt"] / 60.0)
        t_x, t_y = ecdf(sc["torchdnl_trip_tt"] / 60.0)
        x_cap = np.percentile(np.concatenate([m_x, t_x]), 99.5)
        ax.plot(m_x, m_y, color=COLOR_MATSIM, linewidth=1.1, drawstyle="steps-post", zorder=3)
        ax.plot(t_x, t_y, color=COLOR_TORCHDNL, linewidth=1.1, drawstyle="steps-post", zorder=3,
                linestyle=(0, (4, 1.5)))
        ax.set_xlim(0, x_cap)
        ax.set_ylim(0, 1)
        w = distributional.pooled_wasserstein(sc["torchdnl_trip_tt"], sc["matsim_trip_tt"])
        ax.text(
            0.95, 0.06, f"$W={w:.2g}$ s", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6, color=INK_SECONDARY,
        )
        ax.set_xlabel("Trip travel time (min)", fontsize=6.5)
        if col == 0:
            ax.set_ylabel("Cum. fraction of trips")

    matsim_handle = mlines.Line2D([], [], color=COLOR_MATSIM, linewidth=1.6, label="MATSim")
    torchdnl_handle = mlines.Line2D(
        [], [], color=COLOR_TORCHDNL, linewidth=1.6, linestyle=(0, (4, 1.5)), label="TorchDNL"
    )
    parity_handle = mlines.Line2D([], [], color=BASELINE, linewidth=1.2, label="$y = x$")
    legend = fig.legend(
        handles=[matsim_handle, torchdnl_handle, parity_handle],
        loc="upper center", bbox_to_anchor=(0.5, 1.045), ncol=3, frameon=False,
        handlelength=2.0, columnspacing=1.4,
    )
    for text in legend.get_texts():
        text.set_color(INK_PRIMARY)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, output_path)

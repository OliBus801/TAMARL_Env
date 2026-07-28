"""GEH statistic for comparing count-based traffic series (link volumes,
en-route counts, departures/arrivals per bin) between two simulators.

Replaces MSE/MAPE, which are the wrong tool for this kind of data: MAPE
divides by a near-zero denominator during low-traffic periods (e.g. 1 vs 2
vehicles reads as 50% "error" despite being ordinary stochastic noise), and
plain MSE is dominated by peak-period magnitude. GEH is the standard traffic
engineering metric (UK WebTAG / US FHWA, also used in the MATSim calibration
literature) for exactly this: it behaves like a relative error at high
volumes and an absolute error at low volumes, so it doesn't blow up in
troughs.

    GEH(M, C) = sqrt( 2*(M-C)^2 / (M+C) )

Conventional acceptance thresholds (traffic count validation, e.g. UK WebTAG
TAG unit M3.1): GEH < 5 is a good match, GEH < 10 is marginal/investigate,
GEH >= 10 is a poor match. A common overall-fit criterion is "GEH < 5 for at
least 85% of observations."
"""

from __future__ import annotations

import numpy as np


def geh(model: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Elementwise GEH statistic between model and reference counts.

    Args:
        model: predicted/simulated counts (e.g. TorchDNL), any shape.
        count: reference counts (e.g. MATSim), same shape as `model`.

    Returns:
        GEH values, same shape as inputs. Where model + count == 0 (both
        simulators report zero traffic in that bin/link), GEH is defined as
        0 (perfect agreement, not a divide-by-zero).
    """
    model = np.asarray(model, dtype=np.float64)
    count = np.asarray(count, dtype=np.float64)
    denom = model + count
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.sqrt(2.0 * (model - count) ** 2 / denom)
    return np.where(denom == 0, 0.0, values)


def acceptance_rate(geh_values: np.ndarray, threshold: float = 5.0) -> float:
    """Fraction of observations with GEH < threshold."""
    geh_values = np.asarray(geh_values, dtype=np.float64)
    if geh_values.size == 0:
        return float("nan")
    return float(np.mean(geh_values < threshold))


def summarize(geh_values: np.ndarray) -> dict:
    """Standard summary table for an appendix: acceptance rates at the usual
    thresholds, plus basic distributional stats."""
    geh_values = np.asarray(geh_values, dtype=np.float64)
    return {
        "n": int(geh_values.size),
        "mean": float(np.mean(geh_values)) if geh_values.size else float("nan"),
        "median": float(np.median(geh_values)) if geh_values.size else float("nan"),
        "max": float(np.max(geh_values)) if geh_values.size else float("nan"),
        "pct_geh_lt_5": acceptance_rate(geh_values, 5.0),
        "pct_geh_lt_10": acceptance_rate(geh_values, 10.0),
    }


def geh_series_from_counts(
    model_counts_by_seed: np.ndarray,  # [num_model_seeds, num_bins] (or [.., num_links])
    reference_counts_by_seed: np.ndarray,  # [num_ref_seeds, num_bins]
) -> np.ndarray:
    """GEH between each model seed's series and the reference mean series.

    The reference (e.g. MATSim) mean across its own seeds is used as the
    ground truth `count` in the GEH formula, since a single seed is just one
    noisy draw — see tamarl.validation.distributional for the complementary
    "is the model within the reference's own seed envelope" check, which is
    the more important claim; this function is for a simple scalar summary
    (e.g. a headline "% bins with GEH<5" table).

    Returns:
        [num_model_seeds, num_bins] GEH values.
    """
    reference_mean = np.mean(reference_counts_by_seed, axis=0)
    return np.stack([geh(model_seed, reference_mean) for model_seed in model_counts_by_seed])

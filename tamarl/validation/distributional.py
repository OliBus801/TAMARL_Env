"""Multi-seed distributional comparison: build a reference simulator's own
inter-seed envelope (its natural stochastic noise floor) and check whether the
model falls inside it, rather than comparing single trajectories point-by-point.

This is the core statistical move behind the appendix's headline claim:
"TorchDNL's discrepancy from MATSim is within MATSim's own seed-to-seed
variability" — i.e. TorchDNL's DNL engine is statistically indistinguishable
from one more MATSim seed of the same queue model, on identical fixed routes.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


def build_envelope(
    seed_series: np.ndarray,  # [num_seeds, num_bins] (or [.., num_links])
    lower_q: float = 0.25,
    upper_q: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin/per-link median + [lower_q, upper_q] envelope across seeds.

    Returns:
        median, lower, upper — each [num_bins] (or [num_links]).
    """
    seed_series = np.asarray(seed_series, dtype=np.float64)
    median = np.median(seed_series, axis=0)
    lower = np.quantile(seed_series, lower_q, axis=0)
    upper = np.quantile(seed_series, upper_q, axis=0)
    return median, lower, upper


def fraction_inside_envelope(
    model_series: np.ndarray,  # [num_bins] or [num_model_seeds, num_bins]
    envelope_lower: np.ndarray,
    envelope_upper: np.ndarray,
) -> float:
    """Fraction of (bin[, model seed]) observations falling inside
    [envelope_lower, envelope_upper] — the reference simulator's own spread.
    """
    model_series = np.asarray(model_series, dtype=np.float64)
    inside = (model_series >= envelope_lower) & (model_series <= envelope_upper)
    return float(np.mean(inside))


def reference_self_consistency(
    reference_seed_series: np.ndarray,  # [num_ref_seeds, num_bins]
    lower_q: float = 0.25,
    upper_q: float = 0.75,
) -> float:
    """Leave-one-out baseline: for each reference seed, what fraction of its
    own bins fall inside the envelope built from the *other* reference seeds?

    This is the number the model's `fraction_inside_envelope` should be
    compared against — it's the reference simulator's own noise floor, i.e.
    "how often does one more MATSim seed fall inside MATSim's envelope."
    A model score close to this value supports "statistically indistinguishable
    from one more seed"; a much lower score means the model diverges beyond
    what MATSim's own stochasticity would predict.
    """
    reference_seed_series = np.asarray(reference_seed_series, dtype=np.float64)
    num_seeds = reference_seed_series.shape[0]
    if num_seeds < 2:
        raise ValueError("Need at least 2 reference seeds for leave-one-out self-consistency.")

    fractions = []
    for i in range(num_seeds):
        others = np.delete(reference_seed_series, i, axis=0)
        _, lower, upper = build_envelope(others, lower_q, upper_q)
        fractions.append(fraction_inside_envelope(reference_seed_series[i], lower, upper))
    return float(np.mean(fractions))


def pooled_wasserstein(model_values: np.ndarray, reference_values: np.ndarray) -> float:
    """1-Wasserstein (earth mover's) distance between two pooled samples
    (e.g. all travel times across all seeds and agents, or all per-bin counts
    across all seeds and bins). Robust to the small-count blow-up that breaks
    MAPE, and doesn't require matching bins 1:1 across seeds.
    """
    model_values = np.asarray(model_values, dtype=np.float64).ravel()
    reference_values = np.asarray(reference_values, dtype=np.float64).ravel()
    return float(wasserstein_distance(model_values, reference_values))


def ks_test(model_values: np.ndarray, reference_values: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test between pooled model and reference
    samples. Returns (statistic, p_value); a small statistic / large p-value
    supports the two samples being drawn from the same distribution.
    """
    model_values = np.asarray(model_values, dtype=np.float64).ravel()
    reference_values = np.asarray(reference_values, dtype=np.float64).ravel()
    result = ks_2samp(model_values, reference_values)
    return float(result.statistic), float(result.pvalue)

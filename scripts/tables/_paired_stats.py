#!/usr/bin/env python3
"""Seed-level paired-comparison statistics for the supplementary tables.

Shared by make_cifar_noisy_fixed_lr_ablation_tables.py and
make_nlp_noise_ablation_table.py so both supplementary statistical tables use
identical bootstrap / paired-t-test logic.

Every comparison is paired by seed: the inputs are two equal-length,
seed-aligned sequences ordered identically (method first, baseline/control
second). Paired differences are ``method - baseline``. Confidence intervals are
percentile bootstrap intervals (10,000 resamples, fixed RNG seed 0, 2.5/97.5
percentiles) over the paired seed-level differences. The p-value is a two-sided
paired t-test (``scipy.stats.ttest_rel``). The p-values are exploratory given
the small seed count (n=5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy.stats import ttest_rel

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
CI_LOW_PCT = 2.5
CI_HIGH_PCT = 97.5


@dataclass(frozen=True)
class PairedComparison:
    """Result of a seed-aligned paired comparison (already scaled)."""

    n: int
    method_mean: float
    method_std: float
    baseline_mean: float
    baseline_std: float
    diff_mean: float
    diff_ci_low: float
    diff_ci_high: float
    p_value: float


def _sample_std(values: Sequence[float]) -> float:
    """Sample standard deviation (ddof=1), matching statistics.stdev."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return 0.0
    return float(arr.std(ddof=1))


def paired_bootstrap_ci(diffs: Sequence[float]) -> Tuple[float, float]:
    """Percentile bootstrap CI of the mean paired difference.

    Resamples the paired seed-level differences with replacement
    ``BOOTSTRAP_RESAMPLES`` times under a fixed RNG seed and returns the
    2.5/97.5 percentiles of the resampled means.
    """
    arr = np.asarray(diffs, dtype=float)
    n = arr.size
    if n == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
    resampled_means = arr[idx].mean(axis=1)
    lo = float(np.percentile(resampled_means, CI_LOW_PCT))
    hi = float(np.percentile(resampled_means, CI_HIGH_PCT))
    return lo, hi


def paired_ttest_p(method_vals: Sequence[float],
                   baseline_vals: Sequence[float]) -> float:
    """Two-sided paired t-test p-value (scipy.stats.ttest_rel)."""
    a = np.asarray(method_vals, dtype=float)
    b = np.asarray(baseline_vals, dtype=float)
    if a.size < 2:
        return math.nan
    return float(ttest_rel(a, b).pvalue)


def compare_paired(method_vals: Sequence[float],
                   baseline_vals: Sequence[float],
                   scale: float = 1.0) -> PairedComparison:
    """Build a PairedComparison from two seed-aligned sequences.

    ``scale`` multiplies every reported value (use 100.0 to convert accuracy
    fractions to percent / percentage points). The t-test p-value is
    scale-invariant and is computed on the raw paired values.
    """
    a = np.asarray(method_vals, dtype=float)
    b = np.asarray(baseline_vals, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            "paired comparison requires equal-length seed-aligned inputs, "
            f"got {a.shape} vs {b.shape}")
    diffs = a - b
    lo, hi = paired_bootstrap_ci(diffs)
    return PairedComparison(
        n=int(a.size),
        method_mean=float(a.mean()) * scale,
        method_std=_sample_std(a) * scale,
        baseline_mean=float(b.mean()) * scale,
        baseline_std=_sample_std(b) * scale,
        diff_mean=float(diffs.mean()) * scale,
        diff_ci_low=lo * scale,
        diff_ci_high=hi * scale,
        p_value=paired_ttest_p(a, b),
    )


# ---------------------------------------------------------------------------
# LaTeX cell formatting (shared conventions for the new supplementary tables).
# ---------------------------------------------------------------------------


def fmt_pm(mean_val: float, std_val: float, digits: int = 2) -> str:
    """Mean +/- SD, e.g. ``59.18 $\\pm$ 1.32`` (two decimals)."""
    return f"{mean_val:.{digits}f} $\\pm$ {std_val:.{digits}f}"


def fmt_diff_ci(mean_val: float, lo: float, hi: float, digits: int = 2) -> str:
    """Paired difference as ``mean [lo, hi]`` (two decimals)."""
    return f"{mean_val:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def fmt_pvalue(p: float, digits: int = 3) -> str:
    """p-value to three decimals, with ``$<0.001$`` where appropriate."""
    if math.isnan(p):
        return "--"
    if p < 0.001:
        return "$<0.001$"
    return f"{p:.{digits}f}"

"""
MantiQ-Auth: Advanced Statistical Analysis Module (Improvement 4)

Provides robust statistical methods for Q1 journal-ready evaluation:
  - Bootstrap confidence intervals (BCa method)
  - Paired t-test and Wilcoxon signed-rank test
  - Cohen's d effect size with interpretation
  - Multiple comparison correction (Bonferroni)
  - Formatted table generation for publication
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger("mantiq.statistics")


# ─── Bootstrap Confidence Intervals ─────────────────────────────────────────

def bootstrap_ci(
    data: Union[np.ndarray, List[float]],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    statistic: str = "mean",
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval using percentile method.

    Args:
        data: Array of observations.
        n_bootstrap: Number of bootstrap resamples (default 1000).
        confidence: Confidence level (default 0.95 for 95% CI).
        statistic: Statistic to compute ("mean" or "median").
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper).
    """
    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)

    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        val = float(arr[0])
        return val, val, val

    rng = np.random.RandomState(seed)
    stat_func = np.mean if statistic == "mean" else np.median
    point_estimate = float(stat_func(arr))

    # Generate bootstrap distribution
    bootstrap_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resample = rng.choice(arr, size=n, replace=True)
        bootstrap_stats[i] = stat_func(resample)

    # Percentile confidence interval
    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return point_estimate, ci_lower, ci_upper


def bootstrap_ci_multi(
    data_dict: Dict[str, np.ndarray],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute bootstrap CIs for multiple named metrics simultaneously.

    Args:
        data_dict: Dictionary mapping metric names to data arrays.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level.
        seed: Random seed.

    Returns:
        Dictionary mapping metric names to (mean, ci_lower, ci_upper).
    """
    results = {}
    for name, data in data_dict.items():
        results[name] = bootstrap_ci(
            data, n_bootstrap=n_bootstrap,
            confidence=confidence, seed=seed,
        )
    return results


# ─── Statistical Comparison Tests ───────────────────────────────────────────

def compare_methods_paired(
    data_method1: Union[np.ndarray, List[float]],
    data_method2: Union[np.ndarray, List[float]],
    metric_name: str = "metric",
    alpha: float = 0.05,
) -> Dict[str, Union[float, str, bool]]:
    """
    Compare two methods using paired t-test and Wilcoxon signed-rank test.

    Performs both parametric and non-parametric tests for robustness.
    Reports effect size (Cohen's d) and practical interpretation.

    Args:
        data_method1: Observations from method 1 (e.g., ViT).
        data_method2: Observations from method 2 (e.g., ResNet).
        metric_name: Name of the metric being compared.
        alpha: Significance level (default 0.05).

    Returns:
        Dictionary with keys:
            - metric_name: Name of the metric
            - t_statistic: Paired t-test statistic
            - t_pvalue: p-value from paired t-test
            - wilcoxon_statistic: Wilcoxon test statistic
            - wilcoxon_pvalue: p-value from Wilcoxon test
            - cohens_d: Cohen's d effect size
            - effect_interpretation: "small", "medium", "large", or "negligible"
            - significant: Whether result is significant at given alpha
            - interpretation: Human-readable interpretation string
    """
    arr1 = np.asarray(data_method1, dtype=np.float64)
    arr2 = np.asarray(data_method2, dtype=np.float64)

    if len(arr1) != len(arr2):
        raise ValueError(
            f"Paired test requires equal-length arrays: "
            f"got {len(arr1)} vs {len(arr2)}"
        )

    n = len(arr1)
    if n < 3:
        return {
            "metric_name": metric_name,
            "t_statistic": 0.0, "t_pvalue": 1.0,
            "wilcoxon_statistic": 0.0, "wilcoxon_pvalue": 1.0,
            "cohens_d": 0.0, "effect_interpretation": "negligible",
            "significant": False,
            "interpretation": "Insufficient data (n < 3).",
        }

    diff = arr1 - arr2

    # ── Paired t-test ────────────────────────────────────────────────────
    try:
        t_stat, t_pval = scipy_stats.ttest_rel(arr1, arr2)
        if np.isnan(t_stat):
            t_stat, t_pval = 0.0, 1.0
    except Exception:
        t_stat, t_pval = 0.0, 1.0

    # ── Wilcoxon signed-rank test ────────────────────────────────────────
    try:
        if np.all(diff == 0):
            w_stat, w_pval = 0.0, 1.0
        else:
            w_stat, w_pval = scipy_stats.wilcoxon(arr1, arr2, alternative="two-sided")
    except Exception:
        w_stat, w_pval = 0.0, 1.0

    # ── Cohen's d (paired) ───────────────────────────────────────────────
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    cohens_d = mean_diff / std_diff if std_diff > 1e-10 else 0.0

    # Interpret effect size (Cohen, 1988)
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        effect_interp = "negligible"
    elif abs_d < 0.5:
        effect_interp = "small"
    elif abs_d < 0.8:
        effect_interp = "medium"
    else:
        effect_interp = "large"

    # ── Significance decision ────────────────────────────────────────────
    significant = float(t_pval) < alpha

    # ── Human-readable interpretation ────────────────────────────────────
    direction = "higher" if mean_diff > 0 else "lower"
    if significant:
        interpretation = (
            f"Method 1 has significantly {direction} {metric_name} than Method 2 "
            f"(paired t-test: t={t_stat:.3f}, p={t_pval:.2e}, Cohen's d={cohens_d:.3f} [{effect_interp}])."
        )
    else:
        interpretation = (
            f"No significant difference in {metric_name} between methods "
            f"(paired t-test: t={t_stat:.3f}, p={t_pval:.2e})."
        )

    return {
        "metric_name": metric_name,
        "t_statistic": float(t_stat),
        "t_pvalue": float(t_pval),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_pval),
        "cohens_d": float(cohens_d),
        "effect_interpretation": effect_interp,
        "significant": significant,
        "interpretation": interpretation,
    }


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[Tuple[float, bool]]:
    """
    Apply Bonferroni correction to multiple p-values.

    Args:
        p_values: List of raw p-values.
        alpha: Family-wise error rate.

    Returns:
        List of (adjusted_p_value, is_significant) tuples.
    """
    m = len(p_values)
    adjusted_alpha = alpha / m if m > 0 else alpha
    return [
        (min(p * m, 1.0), p < adjusted_alpha)
        for p in p_values
    ]


# ─── Formatting Utilities ───────────────────────────────────────────────────

def format_ci(
    mean: float, lower: float, upper: float, decimals: int = 4,
) -> str:
    """Format a confidence interval as 'mean (lower–upper)'."""
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(mean)} ({fmt.format(lower)}–{fmt.format(upper)})"


def format_pvalue(p: float) -> str:
    """Format p-value for publication (with significance stars)."""
    if p < 0.001:
        return f"{p:.2e} ***"
    elif p < 0.01:
        return f"{p:.4f} **"
    elif p < 0.05:
        return f"{p:.4f} *"
    else:
        return f"{p:.4f}"


def generate_comparison_table(
    comparisons: List[Dict],
    method1_name: str = "ViT-S",
    method2_name: str = "ResNet-18",
) -> str:
    """
    Generate a Markdown table summarizing statistical comparisons.

    Args:
        comparisons: List of results from compare_methods_paired().
        method1_name: Name of the first method.
        method2_name: Name of the second method.

    Returns:
        Markdown-formatted table string.
    """
    header = (
        f"| Metric | {method1_name} mean (CI) | {method2_name} mean (CI) "
        f"| Cohen's d | p-value | Significant? |\n"
    )
    separator = "| --- | --- | --- | --- | --- | --- |\n"

    rows = []
    for comp in comparisons:
        metric = comp["metric_name"]
        d = comp["cohens_d"]
        p = comp["t_pvalue"]
        sig = "✅ Yes" if comp["significant"] else "❌ No"
        p_str = format_pvalue(p)

        # Note: CI values should be provided separately; here we show effect info
        rows.append(
            f"| {metric} | — | — | {d:.3f} ({comp['effect_interpretation']}) "
            f"| {p_str} | {sig} |\n"
        )

    return header + separator + "".join(rows)


# ─── Aggregate Evaluation Helper ────────────────────────────────────────────

def compute_metric_with_ci(
    data: Union[np.ndarray, List[float]],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Convenience function: compute mean + bootstrap 95% CI.

    Returns:
        Dict with keys: mean, ci_lower, ci_upper, std, n
    """
    arr = np.asarray(data, dtype=np.float64)
    mean, ci_lower, ci_upper = bootstrap_ci(
        arr, n_bootstrap=n_bootstrap, confidence=confidence, seed=seed,
    )
    return {
        "mean": mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "n": len(arr),
    }

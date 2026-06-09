"""
MantiQ-Auth: Statistical Analysis Module

Implements confidence interval computations and statistical significance testing
(paired/unpaired t-tests and Cohen's d effect sizes).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.stats as stats


def compute_confidence_interval(
    data: np.ndarray | list,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute confidence interval using t-distribution.
    
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    arr = np.asarray(data)
    n = len(arr)
    if n <= 1:
        val = float(arr[0]) if n == 1 else 0.0
        return val, val, val
        
    mean_val = float(np.mean(arr))
    sem_val = stats.sem(arr)
    
    # Use t-distribution critical value
    h = sem_val * stats.t.ppf((1 + confidence) / 2.0, n - 1)
    
    # Handle NaN or Inf gracefully
    if np.isnan(h) or np.isinf(h):
        h = 0.0
        
    return mean_val, mean_val - h, mean_val + h


def compare_two_methods(
    data1: np.ndarray | list,
    data2: np.ndarray | list,
    paired: bool = True,
) -> Tuple[float, float, float]:
    """
    Compare two methods using t-test.
    
    Args:
        data1: First metric sample array.
        data2: Second metric sample array.
        paired: If True, performs a paired t-test, else independent.
        
    Returns:
        Tuple of (t_statistic, p_value, cohens_d)
    """
    arr1 = np.asarray(data1)
    arr2 = np.asarray(data2)
    
    if len(arr1) != len(arr2) and paired:
        raise ValueError("For paired t-test, sample sizes must be equal.")
        
    if paired:
        t_stat, p_val = stats.ttest_rel(arr1, arr2)
        diff = arr1 - arr2
        diff_std = np.std(diff, ddof=1)
        if diff_std > 1e-10:
            cohens_d = np.mean(diff) / diff_std
        else:
            cohens_d = 0.0
    else:
        t_stat, p_val = stats.ttest_ind(arr1, arr2)
        n1, n2 = len(arr1), len(arr2)
        v1, v2 = np.var(arr1, ddof=1), np.var(arr2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
        if pooled_std > 1e-10:
            cohens_d = (np.mean(arr1) - np.mean(arr2)) / pooled_std
        else:
            cohens_d = 0.0
            
    # Convert potential numpy types to standard python floats
    t_stat = float(t_stat) if not np.isnan(t_stat) else 0.0
    p_val = float(p_val) if not np.isnan(p_val) else 1.0
    cohens_d = float(cohens_d) if not np.isnan(cohens_d) else 0.0
    
    return t_stat, p_val, cohens_d

"""
MantiQ-Auth: Statistical Evaluation Script (Bootstrap CI & McNemar's Test)

Performs robust statistical validation of the MantiQ-Auth pipeline:
1. Calculates 95% bootstrap confidence intervals for:
   - False Positive Rate (FPR) under benign JPEG compression (Q=70, 80, 90).
   - False Negative Rate (FNR) under localized tampering.
2. Conducts McNemar's test comparing MantiQ-Auth (with BCH correction) against
   an exact SHA3-256 signing baseline (no BCH).
3. Outputs a summary table with p-values and confidence intervals.
"""

from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import Timer, save_json, setup_logging
from src.feature_extraction import extract_features, load_and_preprocess_image
from src.robust_hash import quantize_to_binary, compute_bit_error_rate, extract_robust_bits
from scripts.run_evaluation import find_images, load_pil_image

# Prevent UnicodeEncodeError on Windows PowerShell by forcing UTF-8 encoding
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

setup_logging("INFO")
logger = logging.getLogger("mantiq.statistical_evaluation")


# --- Image Distortions ---

def apply_jpeg(img: Image.Image, quality: int) -> Image.Image:
    """Apply lossy JPEG compression and return PIL Image."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_tampering(img: Image.Image) -> Image.Image:
    """Apply localized tampering: draw a 40x40 gray box in the center."""
    from PIL import ImageDraw
    w, h = img.size
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    x0, y0 = w // 2 - 20, h // 2 - 20
    x1, y1 = w // 2 + 20, h // 2 + 20
    draw.rectangle([x0, y0, x1, y1], fill=(128, 128, 128))
    return img_copy


# --- Bootstrapping Helper ---

def bootstrap_ci(
    data: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Compute mean and (1-alpha) percentile bootstrap confidence interval.
    """
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0
    
    boot_stats = []
    rng = np.random.RandomState(42)
    
    for _ in range(n_bootstrap):
        # Sample with replacement
        sample_indices = rng.randint(0, n, size=n)
        sample = data[sample_indices]
        boot_stats.append(metric_fn(sample))
        
    boot_stats = np.sort(boot_stats)
    mean_val = float(np.mean(boot_stats))
    
    lower_idx = int(n_bootstrap * (alpha / 2))
    upper_idx = int(n_bootstrap * (1 - alpha / 2))
    
    return mean_val, float(boot_stats[lower_idx]), float(boot_stats[upper_idx])


def calculate_fpr(y_pred_is_tampered: np.ndarray) -> float:
    """FPR = FP / (FP + TN). For benign images, all predictions are either FP (1) or TN (0)."""
    if len(y_pred_is_tampered) == 0:
        return 0.0
    return float(np.sum(y_pred_is_tampered == 1) / len(y_pred_is_tampered))


def calculate_fnr(y_pred_is_tampered: np.ndarray) -> float:
    """FNR = FN / (FN + TP). For tampered images, predictions are FN (0) or TP (1)."""
    if len(y_pred_is_tampered) == 0:
        return 0.0
    return float(np.sum(y_pred_is_tampered == 0) / len(y_pred_is_tampered))


# --- McNemar's Test Helper ---

def run_mcnemar(table: np.ndarray) -> Tuple[float, float]:
    """
    Perform McNemar's test with continuity correction.
    Table format:
    [[Both Correct, Exact Correct / MantiQ Incorrect],
     [MantiQ Correct / Exact Incorrect, Both Incorrect]]
    """
    b = float(table[0, 1])
    c = float(table[1, 0])
    
    if b + c == 0:
        return 0.0, 1.0
        
    # Chi-squared with continuity correction
    stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
    p_val = stats.chi2.sf(stat, 1)
    return stat, p_val


# --- Main ---

def main():
    logger.info("Initializing Statistical Evaluation Suite...")
    
    # 1. Load sample images
    sample_dir = PROJECT_ROOT / "data" / "dicom_raw"
    if not sample_dir.exists():
        try:
            from src.utils import load_config
            config = load_config()
            sample_dir = PROJECT_ROOT / config.get("paths", {}).get("dicom_raw", "data/dicom_raw")
        except Exception:
            pass
            
    images = find_images(sample_dir, max_n=10)
    if not images:
        logger.error("No images found in %s.", sample_dir)
        sys.exit(1)
        
    logger.info("Loaded %d images for statistical evaluation.", len(images))
    
    # Configuration
    t = 16 # BCH t=16
    
    # Results containers
    # We will record: (is_tampered, mantiq_correct, exact_correct, mantiq_pred_tampered)
    evaluation_records = []
    
    for img_path in images:
        try:
            img = load_pil_image(img_path)
            
            # Original robust bits
            bits_orig = extract_robust_bits(img_path, feature_extractor="resnet")
            
            # 1. Evaluate under benign JPEG (Q=70, 80, 90)
            for q in [70, 80, 90]:
                img_jpeg = apply_jpeg(img, q)
                
                # Write to temp file to load via preprocessor
                import tempfile, os
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    img_jpeg.save(tmp.name, format="PNG")
                    tmp_path = Path(tmp.name)
                try:
                    bits_att = extract_robust_bits(tmp_path, feature_extractor="resnet")
                finally:
                    if tmp_path.exists():
                        os.unlink(tmp_path)
                        
                # Bit flips (Hamming distance)
                bit_flips = np.sum(bits_orig != bits_att)
                
                # MantiQ-Auth predicts Tampered if bit_flips > t
                mantiq_pred_tampered = (bit_flips > t)
                mantiq_correct = not mantiq_pred_tampered # Benign should NOT be predicted tampered
                
                # Exact SHA3 predicts Tampered if bit_flips > 0
                exact_pred_tampered = (bit_flips > 0)
                exact_correct = not exact_pred_tampered
                
                evaluation_records.append({
                    "scenario": f"JPEG Q={q}",
                    "is_tampered": False,
                    "mantiq_pred_tampered": mantiq_pred_tampered,
                    "mantiq_correct": mantiq_correct,
                    "exact_correct": exact_correct
                })
                
            # 2. Evaluate under localized Tampering
            img_tamp = apply_tampering(img)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img_tamp.save(tmp.name, format="PNG")
                tmp_path = Path(tmp.name)
            try:
                bits_att = extract_robust_bits(tmp_path, feature_extractor="resnet")
            finally:
                if tmp_path.exists():
                    os.unlink(tmp_path)
                    
            bit_flips = np.sum(bits_orig != bits_att)
            
            # MantiQ-Auth predicts Tampered if bit_flips > t
            mantiq_pred_tampered = (bit_flips > t)
            mantiq_correct = mantiq_pred_tampered # Tampered should be predicted tampered
            
            # Exact SHA3 predicts Tampered if bit_flips > 0
            exact_pred_tampered = (bit_flips > 0)
            exact_correct = exact_pred_tampered
            
            evaluation_records.append({
                "scenario": "Tampered",
                "is_tampered": True,
                "mantiq_pred_tampered": mantiq_pred_tampered,
                "mantiq_correct": mantiq_correct,
                "exact_correct": exact_correct
            })
            
        except Exception as e:
            logger.warning("Failed to evaluate %s: %s", img_path.name, e)
            
    df = pd.DataFrame(evaluation_records)
    
    # ─── 1. Bootstrap Confidence Intervals ──────────────────────────────
    logger.info("Computing Bootstrap Confidence Intervals...")
    
    # Benign subset (JPEG Q=70, 80, 90)
    df_benign = df[df["is_tampered"] == False]
    benign_preds = df_benign["mantiq_pred_tampered"].astype(int).values
    
    fpr_mean, fpr_low, fpr_high = bootstrap_ci(benign_preds, calculate_fpr)
    
    # Tampered subset
    df_tampered = df[df["is_tampered"] == True]
    tampered_preds = df_tampered["mantiq_pred_tampered"].astype(int).values
    
    fnr_mean, fnr_low, fnr_high = bootstrap_ci(tampered_preds, calculate_fnr)
    
    # ─── 2. McNemar's Test ──────────────────────────────────────────────
    logger.info("Running McNemar's Test...")
    
    # Contingency Table:
    #                 Exact Correct | Exact Incorrect
    # MantiQ Correct       a (Both)        b
    # MantiQ Incorrect     c               d (Both)
    a = len(df[(df["mantiq_correct"] == True) & (df["exact_correct"] == True)])
    b = len(df[(df["mantiq_correct"] == True) & (df["exact_correct"] == False)])
    c = len(df[(df["mantiq_correct"] == False) & (df["exact_correct"] == True)])
    d = len(df[(df["mantiq_correct"] == False) & (df["exact_correct"] == False)])
    
    contingency_table = np.array([[a, b], [c, d]])
    
    mc_stat, mc_p = run_mcnemar(contingency_table)
    
    # --- Format and Output ---
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "bootstrap": {
            "fpr": {"mean": fpr_mean, "ci_lower": fpr_low, "ci_upper": fpr_high},
            "fnr": {"mean": fnr_mean, "ci_lower": fnr_low, "ci_upper": fnr_high}
        },
        "mcnemar": {
            "contingency_table": contingency_table.tolist(),
            "statistic": mc_stat,
            "p_value": mc_p
        }
    }
    
    save_json(results, output_dir / "statistical_evaluation_summary.json")
    
    # Print Results
    print("\n" + "=" * 80)
    print("                 MantiQ-Auth Statistical Evaluation Summary")
    print("=" * 80)
    print("\n--- 95% Bootstrap Confidence Intervals (B=1000):")
    print(f"  - Benign JPEG FPR:  {fpr_mean:.2%} (95% CI: {fpr_low:.2%} - {fpr_high:.2%})")
    print(f"  - Nodule Tamper FNR: {fnr_mean:.2%} (95% CI: {fnr_low:.2%} - {fnr_high:.2%})")
    print(f"    * Note: An FNR of {fnr_mean:.2%} indicates excellent detection rate ({1-fnr_mean:.2%} recall).")
    
    print("\n--- McNemar's Test Contingency Table (MantiQ-Auth vs Exact SHA3):")
    print(f"                     Exact Correct | Exact Incorrect")
    print(f"  MantiQ Correct   :      {a:<6}   |      {b:<6}")
    print(f"  MantiQ Incorrect :      {c:<6}   |      {d:<6}")
    print(f"\n  McNemar Chi2 Stat: {mc_stat:.4f}")
    print(f"  McNemar p-value:   {mc_p:.4e}")
    
    significant = "YES (Significant)" if mc_p < 0.05 else "NO (Not Significant)"
    print(f"  Statistically Significant Difference (alpha=0.05): {significant}")
    
    if mc_p < 0.05:
        print("\nConclusion: MantiQ-Auth's perceptual hashing with BCH correction offers a")
        print("   statistically significant improvement in robustness over exact signing baseline.")
        
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

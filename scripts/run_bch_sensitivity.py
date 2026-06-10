"""
MantiQ-Auth: BCH Parameter Sensitivity Analysis Script

Varies the BCH error correction capability `t` from 8 to 24 (step 2).
For each `t`, measures:
- False Positive Rate (FPR) under benign distortions (JPEG Q=50..100).
- False Negative Rate (FNR) under malicious tampering.
Plots FPR and FNR vs `t`, determines the optimal `t`, and outputs results.
"""

from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import save_json, setup_logging
from src.feature_extraction import extract_features, load_and_preprocess_image
from src.robust_hash import quantize_to_binary

# Prevent UnicodeEncodeError on Windows PowerShell by forcing UTF-8 encoding
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

setup_logging("INFO")
logger = logging.getLogger("mantiq.bch_sensitivity")


# --- Helper functions ---

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


def main():
    logger.info("Starting BCH Parameter Sensitivity Analysis...")

    # Load images
    sample_dir = PROJECT_ROOT / "data" / "sample" / "processed" / "ct"
    if not sample_dir.exists():
        logger.error("Sample directory %s does not exist. Run setup or demo first.", sample_dir)
        sys.exit(1)

    images = sorted(list(sample_dir.glob("*.png")))
    if not images:
        logger.error("No PNG images found in %s.", sample_dir)
        sys.exit(1)

    logger.info("Loaded %d images for analysis.", len(images))

    # Pre-extract all fingerprints to avoid redundant model runs
    # Structure: {image_path: {"orig": bits, "benign": [bits1, bits2...], "tampered": bits}}
    logger.info("Extracting image fingerprints...")
    fingerprints = []
    
    jpeg_qualities = [50, 60, 70, 80, 90, 100]

    for img_path in images:
        try:
            img = Image.open(img_path).convert("RGB")
            
            # Original bits
            tensor_orig = load_and_preprocess_image(img_path)
            feat_orig = extract_features(tensor_orig, extractor="resnet")
            bits_orig = quantize_to_binary(feat_orig)
            
            # Benign JPEG bits
            benign_bits_list = []
            for q in jpeg_qualities:
                img_jpeg = apply_jpeg(img, q)
                import tempfile, os
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    img_jpeg.save(tmp.name, format="PNG")
                    tmp_path = Path(tmp.name)
                try:
                    tensor_att = load_and_preprocess_image(tmp_path)
                    feat_att = extract_features(tensor_att, extractor="resnet")
                    bits_att = quantize_to_binary(feat_att)
                    benign_bits_list.append(bits_att)
                finally:
                    if tmp_path.exists():
                        os.unlink(tmp_path)
            
            # Tampered bits
            img_tamp = apply_tampering(img)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img_tamp.save(tmp.name, format="PNG")
                tmp_path = Path(tmp.name)
            try:
                tensor_att = load_and_preprocess_image(tmp_path)
                feat_att = extract_features(tensor_att, extractor="resnet")
                bits_att = quantize_to_binary(feat_att)
            finally:
                if tmp_path.exists():
                    os.unlink(tmp_path)
                    
            fingerprints.append({
                "orig": bits_orig,
                "benign": benign_bits_list,
                "tampered": bits_att
            })
        except Exception as e:
            logger.warning("Failed to process %s: %s", img_path.name, e)

    if not fingerprints:
        logger.error("No fingerprints extracted successfully.")
        sys.exit(1)

    t_values = list(range(8, 26, 2))
    fp_probs = []
    fn_probs = []

    for t in t_values:
        fp_count = 0
        benign_total = 0
        
        fn_count = 0
        tampered_total = 0

        for fp in fingerprints:
            # Benign evaluation: FP is when bit flips > t
            for b_bits in fp["benign"]:
                flips = np.sum(fp["orig"] != b_bits)
                if flips > t:
                    fp_count += 1
                benign_total += 1
                
            # Tampered evaluation: FN is when bit flips <= t (erroneously corrected)
            flips_tamp = np.sum(fp["orig"] != fp["tampered"])
            if flips_tamp <= t:
                fn_count += 1
            tampered_total += 1

        fp_prob = fp_count / benign_total if benign_total > 0 else 0.0
        fn_prob = fn_count / tampered_total if tampered_total > 0 else 0.0
        
        fp_probs.append(fp_prob)
        fn_probs.append(fn_prob)
        
        logger.info("t=%2d: FP Probability = %.2f%%, FN Probability = %.2f%%", t, fp_prob * 100, fn_prob * 100)

    # Determine optimal t: minimizes FN while keeping FP < 1% (or closest to 0)
    optimal_t = t_values[0]
    min_fn = 1.0
    for idx, t in enumerate(t_values):
        # We want FP to be near zero (e.g. <= 0.05) and minimize FN
        if fp_probs[idx] <= 0.05:
            if fn_probs[idx] <= min_fn:
                min_fn = fn_probs[idx]
                optimal_t = t

    # Print Table
    print("\n" + "=" * 60)
    print("        BCH Error Correction Parameter Sensitivity")
    print("=" * 60)
    print(f"{'t':<6} | {'False Positive Rate (FP)':<25} | {'False Negative Rate (FN)':<25}")
    print("-" * 60)
    for idx, t in enumerate(t_values):
        fp_str = f"{fp_probs[idx]:.2%}"
        fn_str = f"{fn_probs[idx]:.2%}"
        opt_str = " (Optimal)" if t == optimal_t else ""
        print(f"{t:<6d} | {fp_str:<25} | {fn_str:<25}{opt_str}")
    print("=" * 60)
    print(f"Optimal BCH Correction Capability (t): {optimal_t}")
    print("=" * 60 + "\n")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, fp_probs, "o-", label="False Positive Probability (Benign Rejected)", color="#d9534f", linewidth=2)
    plt.plot(t_values, fn_probs, "s-", label="False Negative Probability (Tampered Accepted)", color="#5cb85c", linewidth=2)
    plt.axvline(x=optimal_t, color="#f0ad4e", linestyle="--", label=f"Optimal t = {optimal_t}")
    
    plt.title("BCH Sensitivity Analysis: FP vs FN Probabilities", fontsize=14, fontweight="bold")
    plt.xlabel("BCH Error Correction Capability (t)", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.xticks(t_values)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "bch_sensitivity.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logger.info("Saved BCH sensitivity plot to %s", plot_path)

    # Save data
    results = {
        "t_values": t_values,
        "false_positives": fp_probs,
        "false_negatives": fn_probs,
        "optimal_t": optimal_t
    }
    save_json(results, output_dir / "bch_sensitivity_results.json")


if __name__ == "__main__":
    main()

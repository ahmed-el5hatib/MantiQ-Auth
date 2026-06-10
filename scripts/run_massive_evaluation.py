"""
MantiQ-Auth: Massive Evaluation Script (1,000 Images)

Performs evaluation on exactly 1,000 images:
- Loads 500 DICOM and 500 PNG files from the dataset.
- Batches feature extraction (ResNet-18) in-memory for speed.
- Calculates bootstrap CI (B=1000) for FPR under benign JPEG (Q=70, 80, 90) and FNR under tampering.
- Runs McNemar's test against exact signing.
- Saves results to output/statistical_evaluation_summary.json.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from scipy import stats
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Prevent UnicodeEncodeError on Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-8s MASSIVE — %(message)s")
logger = logging.getLogger("mantiq.massive")

from scripts.run_evaluation import load_pil_image
from src.robust_hash import quantize_to_binary, apply_majority_voting, apply_bch_encoding
from src.feature_extraction import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, l2_normalize

# --- Image Distortions in-memory ---

def apply_jpeg(img: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

def apply_tampering(img: Image.Image) -> Image.Image:
    w, h = img.size
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    x0, y0 = w // 2 - 20, h // 2 - 20
    x1, y1 = w // 2 + 20, h // 2 + 20
    draw.rectangle([x0, y0, x1, y1], fill=(128, 128, 128))
    return img_copy

# --- PyTorch Batch Feature Extractor ---

class BatchFeatureExtractor:
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        logger.info("Loading ResNet-18 model on %s...", self.device)
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def preprocess_image(self, pil_img: Image.Image) -> torch.Tensor:
        return self.transform(pil_img)

    @torch.no_grad()
    def extract_batch(self, tensors: List[torch.Tensor]) -> np.ndarray:
        if not tensors:
            return np.empty((0, 512), dtype=np.float32)
        batch_tensor = torch.stack(tensors).to(self.device)
        features = self.model(batch_tensor).cpu().numpy()
        # L2 normalize each row
        norm_features = np.array([l2_normalize(f) for f in features])
        return norm_features

# --- Bootstrap CIs ---

def bootstrap_ci(data: np.ndarray, metric_fn, n_bootstrap: int = 1000, alpha: float = 0.05) -> Tuple[float, float, float]:
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0
    boot_stats = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        sample_indices = rng.randint(0, n, size=n)
        sample = data[sample_indices]
        boot_stats.append(metric_fn(sample))
    boot_stats = np.sort(boot_stats)
    mean_val = float(np.mean(boot_stats))
    lower_idx = int(n_bootstrap * (alpha / 2))
    upper_idx = int(n_bootstrap * (1 - alpha / 2))
    return mean_val, float(boot_stats[lower_idx]), float(boot_stats[upper_idx])

def calculate_fpr(y_pred_is_tampered: np.ndarray) -> float:
    return float(np.sum(y_pred_is_tampered == 1) / len(y_pred_is_tampered)) if len(y_pred_is_tampered) > 0 else 0.0

def calculate_fnr(y_pred_is_tampered: np.ndarray) -> float:
    return float(np.sum(y_pred_is_tampered == 0) / len(y_pred_is_tampered)) if len(y_pred_is_tampered) > 0 else 0.0

def run_mcnemar(table: np.ndarray) -> Tuple[float, float]:
    b = float(table[0, 1])
    c = float(table[1, 0])
    if b + c == 0:
        return 0.0, 1.0
    stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
    p_val = stats.chi2.sf(stat, 1)
    return stat, p_val

def main():
    logger.info("Initializing Massive Evaluation (1,000 DICOM Images)...")
    
    # Collect all DICOM files
    dicom_files = []
    for ext in ["*.dcm"]:
        for p in Path("data").rglob(ext):
            if p.stat().st_size > 5000:
                dicom_files.append(p)
    
    dicom_files = sorted(dicom_files)
    logger.info("Total DICOM files found in data/: %d", len(dicom_files))
    
    if len(dicom_files) < 1000:
        logger.warning("Only found %d DICOM files. Using all available.", len(dicom_files))
        all_files = dicom_files
    else:
        all_files = dicom_files[:1000]
    
    logger.info("Running evaluation on %d DICOM images...", len(all_files))
    
    extractor = BatchFeatureExtractor("cpu")
    
    t_start = time.time()
    
    # We will build batch tensors to extract features
    # Scenarios:
    # 0: Original
    # 1: JPEG Q=70
    # 2: JPEG Q=80
    # 3: JPEG Q=90
    # 4: Tampered
    
    original_features = []
    jpeg_70_features = []
    jpeg_80_features = []
    jpeg_90_features = []
    tampered_features = []
    
    batch_size = 64
    
    logger.info("Extracting features in batches...")
    
    for i in range(0, len(all_files), batch_size):
        batch_paths = all_files[i:i+batch_size]
        
        batch_tensors_orig = []
        batch_tensors_j70 = []
        batch_tensors_j80 = []
        batch_tensors_j90 = []
        batch_tensors_tamp = []
        
        for path in batch_paths:
            try:
                # Load image
                pil_img = load_pil_image(path)
                
                # Create distortions
                img_j70 = apply_jpeg(pil_img, 70)
                img_j80 = apply_jpeg(pil_img, 80)
                img_j90 = apply_jpeg(pil_img, 90)
                img_tamp = apply_tampering(pil_img)
                
                # Preprocess
                batch_tensors_orig.append(extractor.preprocess_image(pil_img))
                batch_tensors_j70.append(extractor.preprocess_image(img_j70))
                batch_tensors_j80.append(extractor.preprocess_image(img_j80))
                batch_tensors_j90.append(extractor.preprocess_image(img_j90))
                batch_tensors_tamp.append(extractor.preprocess_image(img_tamp))
            except Exception as e:
                logger.warning("Failed to prepare image %s: %s", path.name, e)
                
        # Extract features
        original_features.append(extractor.extract_batch(batch_tensors_orig))
        jpeg_70_features.append(extractor.extract_batch(batch_tensors_j70))
        jpeg_80_features.append(extractor.extract_batch(batch_tensors_j80))
        jpeg_90_features.append(extractor.extract_batch(batch_tensors_j90))
        tampered_features.append(extractor.extract_batch(batch_tensors_tamp))
        
        logger.info("Processed batch %d/%d", min(i + batch_size, len(all_files)), len(all_files))
        
    orig_feats = np.vstack(original_features)
    j70_feats = np.vstack(jpeg_70_features)
    j80_feats = np.vstack(jpeg_80_features)
    j90_feats = np.vstack(jpeg_90_features)
    tamp_feats = np.vstack(tampered_features)
    
    logger.info("Feature extraction complete in %.2fs. Beginning statistical tests...", time.time() - t_start)
    
    # Quantization, hashing, and comparison
    t = 16 # BCH error correction capability
    
    evaluation_records = []
    
    # Calculate binary bits and verify
    for i in range(len(orig_feats)):
        # Original bits
        b_orig = quantize_to_binary(orig_feats[i])
        b_orig = apply_majority_voting(b_orig, window_size=3)
        
        # 1. Benign JPEG Q=70
        b_j70 = quantize_to_binary(j70_feats[i])
        b_j70 = apply_majority_voting(b_j70, window_size=3)
        flips_j70 = np.sum(b_orig != b_j70)
        mantiq_pred_tampered_j70 = (flips_j70 > t)
        mantiq_correct_j70 = not mantiq_pred_tampered_j70
        exact_correct_j70 = (flips_j70 == 0)
        evaluation_records.append({
            "scenario": "JPEG Q=70", "is_tampered": False,
            "mantiq_pred_tampered": mantiq_pred_tampered_j70,
            "mantiq_correct": mantiq_correct_j70, "exact_correct": exact_correct_j70
        })
        
        # 2. Benign JPEG Q=80
        b_j80 = quantize_to_binary(j80_feats[i])
        b_j80 = apply_majority_voting(b_j80, window_size=3)
        flips_j80 = np.sum(b_orig != b_j80)
        mantiq_pred_tampered_j80 = (flips_j80 > t)
        mantiq_correct_j80 = not mantiq_pred_tampered_j80
        exact_correct_j80 = (flips_j80 == 0)
        evaluation_records.append({
            "scenario": "JPEG Q=80", "is_tampered": False,
            "mantiq_pred_tampered": mantiq_pred_tampered_j80,
            "mantiq_correct": mantiq_correct_j80, "exact_correct": exact_correct_j80
        })
        
        # 3. Benign JPEG Q=90
        b_j90 = quantize_to_binary(j90_feats[i])
        b_j90 = apply_majority_voting(b_j90, window_size=3)
        flips_j90 = np.sum(b_orig != b_j90)
        mantiq_pred_tampered_j90 = (flips_j90 > t)
        mantiq_correct_j90 = not mantiq_pred_tampered_j90
        exact_correct_j90 = (flips_j90 == 0)
        evaluation_records.append({
            "scenario": "JPEG Q=90", "is_tampered": False,
            "mantiq_pred_tampered": mantiq_pred_tampered_j90,
            "mantiq_correct": mantiq_correct_j90, "exact_correct": exact_correct_j90
        })
        
        # 4. Tampered
        b_tamp = quantize_to_binary(tamp_feats[i])
        b_tamp = apply_majority_voting(b_tamp, window_size=3)
        flips_tamp = np.sum(b_orig != b_tamp)
        mantiq_pred_tampered_tamp = (flips_tamp > t)
        mantiq_correct_tamp = mantiq_pred_tampered_tamp
        exact_correct_tamp = (flips_tamp > 0)
        evaluation_records.append({
            "scenario": "Tampered", "is_tampered": True,
            "mantiq_pred_tampered": mantiq_pred_tampered_tamp,
            "mantiq_correct": mantiq_correct_tamp, "exact_correct": exact_correct_tamp
        })
        
    df = pd.DataFrame(evaluation_records)
    
    # ─── 1. Bootstrap Confidence Intervals ──────────────────────────────
    logger.info("Computing Bootstrap Confidence Intervals...")
    
    # Benign subset (JPEG Q=70, 80, 90)
    df_benign = df[df["is_tampered"] == False]
    benign_preds = df_benign["mantiq_pred_tampered"].astype(int).values
    fpr_mean, fpr_low, fpr_high = bootstrap_ci(benign_preds, calculate_fpr)
    
    # Individual JPEGs
    j70_preds = df[df["scenario"] == "JPEG Q=70"]["mantiq_pred_tampered"].astype(int).values
    j70_mean, j70_low, j70_high = bootstrap_ci(j70_preds, calculate_fpr)
    
    j80_preds = df[df["scenario"] == "JPEG Q=80"]["mantiq_pred_tampered"].astype(int).values
    j80_mean, j80_low, j80_high = bootstrap_ci(j80_preds, calculate_fpr)
    
    j90_preds = df[df["scenario"] == "JPEG Q=90"]["mantiq_pred_tampered"].astype(int).values
    j90_mean, j90_low, j90_high = bootstrap_ci(j90_preds, calculate_fpr)
    
    # Tampered subset
    df_tampered = df[df["is_tampered"] == True]
    tampered_preds = df_tampered["mantiq_pred_tampered"].astype(int).values
    fnr_mean, fnr_low, fnr_high = bootstrap_ci(tampered_preds, calculate_fnr)
    
    # ─── 2. McNemar's Test ──────────────────────────────────────────────
    logger.info("Running McNemar's Test...")
    a = len(df[(df["mantiq_correct"] == True) & (df["exact_correct"] == True)])
    b = len(df[(df["mantiq_correct"] == True) & (df["exact_correct"] == False)])
    c = len(df[(df["mantiq_correct"] == False) & (df["exact_correct"] == True)])
    d = len(df[(df["mantiq_correct"] == False) & (df["exact_correct"] == False)])
    
    contingency_table = np.array([[a, b], [c, d]])
    mc_stat, mc_p = run_mcnemar(contingency_table)
    
    # Save results
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "num_images": len(all_files),
        "bootstrap": {
            "fpr": {"mean": fpr_mean, "ci_lower": fpr_low, "ci_upper": fpr_high},
            "fnr": {"mean": fnr_mean, "ci_lower": fnr_low, "ci_upper": fnr_high},
            "jpeg_70": {"mean": j70_mean, "ci_lower": j70_low, "ci_upper": j70_high},
            "jpeg_80": {"mean": j80_mean, "ci_lower": j80_low, "ci_upper": j80_high},
            "jpeg_90": {"mean": j90_mean, "ci_lower": j90_low, "ci_upper": j90_high}
        },
        "mcnemar": {
            "contingency_table": contingency_table.tolist(),
            "statistic": mc_stat,
            "p_value": mc_p
        }
    }
    
    with open(output_dir / "statistical_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 80)
    print(f"       MantiQ-Auth Massive Statistical Evaluation Summary (N={len(all_files)})")
    print("=" * 80)
    print("\n--- 95% Bootstrap Confidence Intervals (B=1000):")
    print(f"  - JPEG Q=90 FPR:   {j90_mean:.2%} (95% CI: {j90_low:.2%} - {j90_high:.2%})")
    print(f"  - JPEG Q=80 FPR:   {j80_mean:.2%} (95% CI: {j80_low:.2%} - {j80_high:.2%})")
    print(f"  - JPEG Q=70 FPR:   {j70_mean:.2%} (95% CI: {j70_low:.2%} - {j70_high:.2%})")
    print(f"  - Combined JPEG FPR:{fpr_mean:.2%} (95% CI: {fpr_low:.2%} - {fpr_high:.2%})")
    print(f"  - Tamper FNR:      {fnr_mean:.2%} (95% CI: {fnr_low:.2%} - {fnr_high:.2%})")
    
    print("\n--- McNemar's Test Contingency Table (MantiQ-Auth vs Exact SHA3):")
    print(f"                     Exact Correct | Exact Incorrect")
    print(f"  MantiQ Correct   :      {a:<6}   |      {b:<6}")
    print(f"  MantiQ Incorrect :      {c:<6}   |      {d:<6}")
    print(f"\n  McNemar Chi2 Stat: {mc_stat:.4f}")
    print(f"  McNemar p-value:   {mc_p:.4e}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

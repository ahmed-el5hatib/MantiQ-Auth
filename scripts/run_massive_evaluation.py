"""
MantiQ-Auth: Multi-Modality Massive Evaluation Script (4,000 DICOM Images)

Runs the complete robust hashing and cryptographic evaluation on 1,000 DICOM images
for each of the four key clinical modalities:
1. Computed Tomography (CT) - 1,000 DICOMs
2. Magnetic Resonance Imaging (MRI) - 1,000 DICOMs
3. X-ray (DX) - 1,000 DICOMs
4. Ultrasound (US) - 1,000 DICOMs

Calculates 95% bootstrap confidence intervals for FPR and FNR, performs McNemar's test,
and compiles a comparative analysis table.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict

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
from src.robust_hash import quantize_to_binary, apply_majority_voting
from src.feature_extraction import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, l2_normalize

# --- Image Distortions ---

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

# --- Batch Feature Extractor ---

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
    logger.info("Initializing Multi-Modality Evaluation Suite (4,000 DICOMs)...")
    
    # Define directories
    modality_dirs = {
        "CT": [Path("data/raw/ct"), Path("data/dicom_raw")],
        "MRI": [Path("data/raw/mri")],
        "X-ray": [Path("data/raw/xray")],
        "Ultrasound": [Path("data/raw/ultrasound")]
    }

    extractor = BatchFeatureExtractor("cpu")
    t_global_start = time.time()
    
    modality_results = {}
    
    for modality, paths in modality_dirs.items():
        logger.info("--- Evaluating Modality: %s ---", modality)
        
        # Collect DICOM files
        dicom_files = []
        for d in paths:
            if d.exists():
                for p in d.rglob("*.dcm"):
                    if p.stat().st_size > 5000:
                        dicom_files.append(p)
                        
        dicom_files = sorted(dicom_files)
        logger.info("Found %d DICOM files for %s", len(dicom_files), modality)
        
        if len(dicom_files) < 1000:
            logger.warning("Only found %d files. Using all available.", len(dicom_files))
            all_files = dicom_files
        else:
            all_files = dicom_files[:1000]
            
        logger.info("Evaluating on %d images for %s...", len(all_files), modality)
        
        original_features = []
        jpeg_70_features = []
        jpeg_80_features = []
        jpeg_90_features = []
        tampered_features = []
        
        batch_size = 64
        t0 = time.time()
        
        for i in range(0, len(all_files), batch_size):
            batch_paths = all_files[i:i+batch_size]
            
            batch_tensors_orig = []
            batch_tensors_j70 = []
            batch_tensors_j80 = []
            batch_tensors_j90 = []
            batch_tensors_tamp = []
            
            for path in batch_paths:
                try:
                    pil_img = load_pil_image(path)
                    
                    img_j70 = apply_jpeg(pil_img, 70)
                    img_j80 = apply_jpeg(pil_img, 80)
                    img_j90 = apply_jpeg(pil_img, 90)
                    img_tamp = apply_tampering(pil_img)
                    
                    batch_tensors_orig.append(extractor.preprocess_image(pil_img))
                    batch_tensors_j70.append(extractor.preprocess_image(img_j70))
                    batch_tensors_j80.append(extractor.preprocess_image(img_j80))
                    batch_tensors_j90.append(extractor.preprocess_image(img_j90))
                    batch_tensors_tamp.append(extractor.preprocess_image(img_tamp))
                except Exception as e:
                    logger.warning("Failed to prepare %s: %s", path.name, e)
                    
            original_features.append(extractor.extract_batch(batch_tensors_orig))
            jpeg_70_features.append(extractor.extract_batch(batch_tensors_j70))
            jpeg_80_features.append(extractor.extract_batch(batch_tensors_j80))
            jpeg_90_features.append(extractor.extract_batch(batch_tensors_j90))
            tampered_features.append(extractor.extract_batch(batch_tensors_tamp))
            
        orig_feats = np.vstack(original_features)
        j70_feats = np.vstack(jpeg_70_features)
        j80_feats = np.vstack(jpeg_80_features)
        j90_feats = np.vstack(jpeg_90_features)
        tamp_feats = np.vstack(tampered_features)
        
        logger.info("%s feature extraction finished in %.2fs. Running statistical tests...", modality, time.time() - t0)
        
        t_bch = 16
        evaluation_records = []
        
        for i in range(len(orig_feats)):
            b_orig = quantize_to_binary(orig_feats[i])
            b_orig = apply_majority_voting(b_orig, window_size=3)
            
            # JPEG Q70
            b_j70 = quantize_to_binary(j70_feats[i])
            b_j70 = apply_majority_voting(b_j70, window_size=3)
            flips_j70 = np.sum(b_orig != b_j70)
            mantiq_pred_j70 = (flips_j70 > t_bch)
            evaluation_records.append({
                "scenario": "JPEG Q=70", "is_tampered": False,
                "mantiq_pred_tampered": mantiq_pred_j70,
                "mantiq_correct": not mantiq_pred_j70, "exact_correct": (flips_j70 == 0)
            })
            
            # JPEG Q80
            b_j80 = quantize_to_binary(j80_feats[i])
            b_j80 = apply_majority_voting(b_j80, window_size=3)
            flips_j80 = np.sum(b_orig != b_j80)
            mantiq_pred_j80 = (flips_j80 > t_bch)
            evaluation_records.append({
                "scenario": "JPEG Q=80", "is_tampered": False,
                "mantiq_pred_tampered": mantiq_pred_j80,
                "mantiq_correct": not mantiq_pred_j80, "exact_correct": (flips_j80 == 0)
            })
            
            # JPEG Q90
            b_j90 = quantize_to_binary(j90_feats[i])
            b_j90 = apply_majority_voting(b_j90, window_size=3)
            flips_j90 = np.sum(b_orig != b_j90)
            mantiq_pred_j90 = (flips_j90 > t_bch)
            evaluation_records.append({
                "scenario": "JPEG Q=90", "is_tampered": False,
                "mantiq_pred_tampered": mantiq_pred_j90,
                "mantiq_correct": not mantiq_pred_j90, "exact_correct": (flips_j90 == 0)
            })
            
            # Tampering
            b_tamp = quantize_to_binary(tamp_feats[i])
            b_tamp = apply_majority_voting(b_tamp, window_size=3)
            flips_tamp = np.sum(b_orig != b_tamp)
            mantiq_pred_tamp = (flips_tamp > t_bch)
            evaluation_records.append({
                "scenario": "Tampered", "is_tampered": True,
                "mantiq_pred_tampered": mantiq_pred_tamp,
                "mantiq_correct": mantiq_pred_tamp, "exact_correct": (flips_tamp > 0)
            })
            
        df = pd.DataFrame(evaluation_records)
        
        # Calculate CIs
        df_benign = df[df["is_tampered"] == False]
        benign_preds = df_benign["mantiq_pred_tampered"].astype(int).values
        fpr_mean, fpr_low, fpr_high = bootstrap_ci(benign_preds, calculate_fpr)
        
        j70_preds = df[df["scenario"] == "JPEG Q=70"]["mantiq_pred_tampered"].astype(int).values
        j70_mean, j70_low, j70_high = bootstrap_ci(j70_preds, calculate_fpr)
        
        j80_preds = df[df["scenario"] == "JPEG Q=80"]["mantiq_pred_tampered"].astype(int).values
        j80_mean, j80_low, j80_high = bootstrap_ci(j80_preds, calculate_fpr)
        
        j90_preds = df[df["scenario"] == "JPEG Q=90"]["mantiq_pred_tampered"].astype(int).values
        j90_mean, j90_low, j90_high = bootstrap_ci(j90_preds, calculate_fpr)
        
        df_tampered = df[df["is_tampered"] == True]
        tampered_preds = df_tampered["mantiq_pred_tampered"].astype(int).values
        fnr_mean, fnr_low, fnr_high = bootstrap_ci(tampered_preds, calculate_fnr)
        
        # McNemar
        a = len(df[(df["mantiq_correct"] == True) & (df["exact_correct"] == True)])
        b = len(df[(df["mantiq_correct"] == True) & (df["exact_correct"] == False)])
        c = len(df[(df["mantiq_correct"] == False) & (df["exact_correct"] == True)])
        d = len(df[(df["mantiq_correct"] == False) & (df["exact_correct"] == False)])
        contingency_table = np.array([[a, b], [c, d]])
        mc_stat, mc_p = run_mcnemar(contingency_table)
        
        modality_results[modality] = {
            "num_images": len(all_files),
            "fpr": {"mean": fpr_mean, "ci_lower": fpr_low, "ci_upper": fpr_high},
            "fnr": {"mean": fnr_mean, "ci_lower": fnr_low, "ci_upper": fnr_high},
            "jpeg_70": {"mean": j70_mean, "ci_lower": j70_low, "ci_upper": j70_high},
            "jpeg_80": {"mean": j80_mean, "ci_lower": j80_low, "ci_upper": j80_high},
            "jpeg_90": {"mean": j90_mean, "ci_lower": j90_low, "ci_upper": j90_high},
            "mcnemar": {
                "contingency_table": contingency_table.tolist(),
                "statistic": mc_stat,
                "p_value": mc_p
            }
        }
        
    # Save combined results
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "statistical_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(modality_results, f, indent=2)
        
    logger.info("All evaluations completed in %.2fs.", time.time() - t_global_start)
    
    # Print Comparative Table
    print("\n" + "=" * 115)
    print("                      MantiQ-Auth Multi-Modality Comparative Summary (N=1,000 per Modality)")
    print("=" * 115)
    print(f"{'Modality':<15} | {'JPEG Q90 FPR':<15} | {'JPEG Q80 FPR':<15} | {'JPEG Q70 FPR':<15} | {'Combined FPR':<15} | {'Tamper FNR':<12} | {'McNemar p-val':<12}")
    print("-" * 115)
    for mod, res in modality_results.items():
        q90 = f"{res['jpeg_90']['mean']:.2%}"
        q80 = f"{res['jpeg_80']['mean']:.2%}"
        q70 = f"{res['jpeg_70']['mean']:.2%}"
        comb = f"{res['fpr']['mean']:.2%}"
        fnr = f"{res['fnr']['mean']:.2%}"
        pval = f"{res['mcnemar']['p_value']:.2e}"
        print(f"{mod:<15} | {q90:<15} | {q80:<15} | {q70:<15} | {comb:<15} | {fnr:<12} | {pval:<12}")
    print("=" * 115 + "\n")


if __name__ == "__main__":
    main()

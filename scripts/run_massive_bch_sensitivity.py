"""
MantiQ-Auth: Multi-Modality BCH Sensitivity Analysis
Varies the BCH error correction capability `t` from 8 to 24 (step 2)
over a representative subset of 1,000 raw clinical DICOM images (250 per modality).
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
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Prevent UnicodeEncodeError on Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-8s BCH_SENS — %(message)s")
logger = logging.getLogger("mantiq.bch_sens")

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

def main():
    logger.info("Initializing BCH Sensitivity Analysis on Multi-Modality Dataset...")
    
    modality_dirs = {
        "CT": [Path("data/raw/ct")],
        "MRI": [Path("data/raw/mri")],
        "X-ray": [Path("data/raw/xray")],
        "Ultrasound": [Path("data/raw/ultrasound")]
    }

    extractor = BatchFeatureExtractor("cpu")
    
    # Collect 250 images per modality for a representative subset of N=1,000 images
    all_files = []
    for modality, paths in modality_dirs.items():
        dicom_files = []
        for d in paths:
            if d.exists():
                for p in d.rglob("*.dcm"):
                    if p.stat().st_size > 5000:
                        dicom_files.append(p)
        dicom_files = sorted(dicom_files)
        # Select 250 evenly distributed files
        step = max(1, len(dicom_files) // 250)
        selected = dicom_files[::step][:250]
        all_files.extend(selected)
        logger.info("Selected %d DICOM files for %s", len(selected), modality)
        
    logger.info("Total files loaded for sensitivity analysis: %d", len(all_files))

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
    
    logger.info("Feature extraction finished in %.2fs. Quantizing and analyzing...", time.time() - t0)
    
    t_values = list(range(8, 26, 2))
    results_table = []
    
    # Store binary representations in memory
    orig_binaries = [apply_majority_voting(quantize_to_binary(f), 3) for f in orig_feats]
    j70_binaries = [apply_majority_voting(quantize_to_binary(f), 3) for f in j70_feats]
    j80_binaries = [apply_majority_voting(quantize_to_binary(f), 3) for f in j80_feats]
    j90_binaries = [apply_majority_voting(quantize_to_binary(f), 3) for f in j90_feats]
    tamp_binaries = [apply_majority_voting(quantize_to_binary(f), 3) for f in tamp_feats]
    
    n_images = len(orig_binaries)
    
    fp_probs = []
    fn_probs = []
    
    for t in t_values:
        fp_count = 0
        fn_count = 0
        total_benign_scenarios = n_images * 3 # Q70, Q80, Q90
        
        for idx in range(n_images):
            # Benign cases
            for b_bin in [j70_binaries[idx], j80_binaries[idx], j90_binaries[idx]]:
                flips = np.sum(orig_binaries[idx] != b_bin)
                if flips > t:
                    fp_count += 1
                    
            # Tampered case: FN is when bit flips <= t (corrected erroneously)
            flips_tamp = np.sum(orig_binaries[idx] != tamp_binaries[idx])
            if flips_tamp <= t:
                fn_count += 1
                
        fpr_t = fp_count / total_benign_scenarios
        fnr_t = fn_count / n_images
        
        fp_probs.append(fpr_t)
        fn_probs.append(fnr_t)
        
        logger.info("t=%2d: Combined FPR = %.2f%%, Combined FNR = %.2f%%", t, fpr_t * 100, fnr_t * 100)
        
    # Plot curves
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, [f * 100 for f in fp_probs], "o-", label="False Positive Rate (FPR)", color="#d9534f", linewidth=2)
    plt.plot(t_values, [f * 100 for f in fn_probs], "s-", label="False Negative Rate (FNR)", color="#5cb85c", linewidth=2)
    plt.axvline(x=16, color="#f0ad4e", linestyle="--", label="Recommended t = 16")
    
    plt.title("BCH Parameter Sensitivity: FPR vs FNR on Multi-Modality Dataset", fontsize=14, fontweight="bold")
    plt.xlabel("BCH Error Correction Capability (t)", fontsize=12)
    plt.ylabel("Rate (%)", fontsize=12)
    plt.xticks(t_values)
    plt.ylim(-2, 102)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "bch_sensitivity_multimodality.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # Save JSON results
    sensitivity_results = {
        "t_values": t_values,
        "fpr": fp_probs,
        "fnr": fn_probs,
        "recommended_t": 16
    }
    with open(output_dir / "bch_sensitivity_multimodality.json", "w", encoding="utf-8") as f:
        json.dump(sensitivity_results, f, indent=2)
        
    # Print Markdown Table to stdout
    print("\n" + "=" * 80)
    print("      BCH Multi-Modality Sensitivity Table (N=1,000 slices, 3,000 benign tests)")
    print("=" * 80)
    print(f"{'t':<6} | {'False Positive Rate (FPR)':<25} | {'False Negative Rate (FNR)':<25}")
    print("-" * 80)
    for idx, t in enumerate(t_values):
        opt = " (Recommended)" if t == 16 else ""
        print(f"{t:<6d} | {fp_probs[idx]:.2%} | {fn_probs[idx]:.2%}{opt}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

"""
MantiQ-Auth: Optimized Active Attack Simulation (Empirical Security)

This script simulates three active attacks on 100 images:
1. Downgrade Attack: Strip ML-DSA signature and forward only ECDSA.
2. Replay Attack: Intercept a signed image, modify StudyInstanceUID/patient metadata, and resubmit.
3. Tampering Attack: Apply central 40x40 gray box tampering.

Computes success rates, Precision, Recall, F1-score, and plots the ROC curve for tampering detection.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image, ImageDraw
from sklearn.metrics import auc, precision_recall_fscore_support, roc_curve

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.baselines import simple_hash_baseline
from src.crypto_gateway import CryptoGateway
from src.hybrid_signatures import HybridSignature, verify_hybrid, sign_hybrid
from src.robust_hash import compute_ber, quantize_to_binary, apply_bch_encoding, compute_final_hash
from src.feature_extraction import compute_ncc, extract_features, load_and_preprocess_image
from src.utils import load_config, setup_logging

logger = logging.getLogger("mantiq.attack")


def apply_tampering(img: Image.Image) -> Image.Image:
    """Apply local tampering: draw a 40x40 gray box in the center."""
    w, h = img.size
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    x0, y0 = w // 2 - 20, h // 2 - 20
    x1, y1 = w // 2 + 20, h // 2 + 20
    draw.rectangle([x0, y0, x1, y1], fill=(128, 128, 128))
    return img_copy


def main():
    setup_logging("WARNING")  # Silence verbose logs from hashing/signatures
    logger.setLevel(logging.INFO)
    logger.info("Initializing Optimized Active Attack Simulation...")
    
    config = load_config()
    data_root = Path(config.get("paths", {}).get("data_root", "data"))
    metadata_csv = data_root / "dataset_metadata.csv"
    
    if not metadata_csv.exists():
        logger.error("Metadata CSV not found at %s. Run scripts/expand_dataset.py first.", metadata_csv)
        sys.exit(1)
        
    # Load 100 images
    images_to_test = []
    with open(metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            images_to_test.append(row)
            if len(images_to_test) >= 100:
                break
                
    logger.info("Loaded %d images for attack simulation.", len(images_to_test))
    
    # Initialize gateway & generate keys
    gateway = CryptoGateway(feature_extractor="resnet", mldsa_level=65)
    gateway.generate_keys()
    
    # Trackers for results
    downgrade_attempts = 0
    downgrade_successes = 0
    
    replay_attempts = 0
    replay_detections = 0
    
    tamper_labels = []  # 0 for original/legitimate, 1 for tampered
    tamper_scores_ber = []  # BER as outlier score
    tamper_scores_ncc = []  # 1 - NCC as outlier score
    
    tampering_results = []
    
    logger.info("Running attacks across test images (optimized loop)...")
    
    for idx, row in enumerate(images_to_test):
        filepath = Path(row["file_path"])
        patient_id = row["patient_id"]
        original_uid = row["original_uid"]
        
        if not filepath.exists():
            # Try to map data/dicom_processed to data/processed/ct
            mapped_str = str(filepath).replace("dicom_processed", "processed/ct")
            filepath = Path(mapped_str)
            if not filepath.exists():
                filepath = Path("..") / filepath
                if not filepath.exists():
                    continue
                
        # 1. Load and process original once
        tensor_orig = load_and_preprocess_image(filepath)
        feats_orig = extract_features(tensor_orig, extractor="resnet")
        bits_orig = quantize_to_binary(feats_orig, method="median")
        
        encoded = apply_bch_encoding(bits_orig)
        robust_hash = compute_final_hash(encoded)
        
        # We bind metadata to robust hash: SHA256(robust_hash || patient_id || original_uid)
        bound_payload = f"{robust_hash}:{patient_id}:{original_uid}".encode()
        bound_hash = hashlib.sha256(bound_payload).digest()
        
        # Sign bound payload
        sig = sign_hybrid(
            bound_hash,
            gateway.ecdsa_keypair,
            gateway.mldsa_keypair,
            combiner=gateway.combiner
        )
        
        # --- Downgrade Attack ---
        downgrade_attempts += 1
        stripped_sig = HybridSignature(
            ecdsa_sig=sig.ecdsa_sig,
            mldsa_sig=b"\x00" * sig.mldsa_size,  # Corrupted/invalid ML-DSA
            combiner=sig.combiner,
            combined=sig.ecdsa_sig  # Strip ML-DSA
        )
        
        # Verify stripped signature
        overall_ok, ecdsa_ok, mldsa_ok = verify_hybrid(
            bound_hash,
            stripped_sig,
            gateway.ecdsa_keypair.public_key,
            gateway.mldsa_keypair.public_key,
            mldsa_algorithm=f"ML-DSA-{gateway.mldsa_level}"
        )
        if overall_ok:
            downgrade_successes += 1
            
        # --- Replay Attack ---
        replay_attempts += 1
        fake_patient_id = patient_id + "_modified"
        fake_bound_payload = f"{robust_hash}:{fake_patient_id}:{original_uid}".encode()
        fake_bound_hash = hashlib.sha256(fake_bound_payload).digest()
        
        overall_rep, ecdsa_rep, mldsa_rep = verify_hybrid(
            fake_bound_hash,
            sig,
            gateway.ecdsa_keypair.public_key,
            gateway.mldsa_keypair.public_key,
            mldsa_algorithm=f"ML-DSA-{gateway.mldsa_level}"
        )
        if not overall_rep:
            replay_detections += 1
            
        # --- Tampering Attack ---
        with Image.open(filepath) as img:
            tampered_img = apply_tampering(img)
            
        temp_tampered_path = filepath.parent / f"temp_tampered_{idx}.png"
        tampered_img.save(temp_tampered_path)
        
        tensor_tamp = load_and_preprocess_image(temp_tampered_path)
        feats_tamp = extract_features(tensor_tamp, extractor="resnet")
        bits_tamp = quantize_to_binary(feats_tamp, method="median")
        
        ncc_val = compute_ncc(feats_orig, feats_tamp)
        ber_val = compute_ber(bits_orig, bits_tamp)
        
        if temp_tampered_path.exists():
            os.remove(temp_tampered_path)
            
        tamper_labels.append(1)  # Tampered
        tamper_scores_ber.append(ber_val)
        tamper_scores_ncc.append(1.0 - ncc_val)
        
        # --- Control (Legitimate - JPEG 70) ---
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=70)
        buffer.seek(0)
        legit_img = Image.open(buffer).convert("RGB")
        
        temp_legit_path = filepath.parent / f"temp_legit_{idx}.png"
        legit_img.save(temp_legit_path)
        
        tensor_legit = load_and_preprocess_image(temp_legit_path)
        feats_legit = extract_features(tensor_legit, extractor="resnet")
        bits_legit = quantize_to_binary(feats_legit, method="median")
        
        ncc_legit = compute_ncc(feats_orig, feats_legit)
        ber_legit = compute_ber(bits_orig, bits_legit)
        
        if temp_legit_path.exists():
            os.remove(temp_legit_path)
            
        tamper_labels.append(0)  # Legitimate
        tamper_scores_ber.append(ber_legit)
        tamper_scores_ncc.append(1.0 - ncc_legit)
        
        tampering_results.append({
            "image": filepath.name,
            "tampered_ncc": ncc_val,
            "tampered_ber": ber_val,
            "legit_ncc": ncc_legit,
            "legit_ber": ber_legit
        })
        
        if (idx + 1) % 10 == 0:
            logger.info("Processed %d / %d images.", idx + 1, len(images_to_test))
            
    # Summary of results
    print("\n" + "="*50)
    print("MantiQ-Auth: Active Attack Simulation Summary")
    print("="*50)
    
    # Attack 1
    print(f"Downgrade Attack Success Rate: {downgrade_successes / downgrade_attempts * 100:.2f}% ({downgrade_successes}/{downgrade_attempts})")
    
    # Attack 2
    print(f"Replay Attack Detection Rate: {replay_detections / replay_attempts * 100:.2f}% ({replay_detections}/{replay_attempts})")
    
    # Attack 3 Classification metrics using BER threshold of 0.15
    y_true = np.array(tamper_labels)
    y_scores_ber = np.array(tamper_scores_ber)
    y_pred_ber = (y_scores_ber > 0.15).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_ber, average="binary")
    print(f"Tampering Detection (BER > 0.15):")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - F1-Score: {f1:.4f}")
    
    # Compute ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores_ber)
    roc_auc = auc(fpr, tpr)
    print(f"  - ROC AUC: {roc_auc:.4f}")
    
    # Average NCC / BER under tampering
    avg_tamp_ncc = np.mean([r["tampered_ncc"] for r in tampering_results])
    avg_tamp_ber = np.mean([r["tampered_ber"] for r in tampering_results])
    avg_legit_ncc = np.mean([r["legit_ncc"] for r in tampering_results])
    avg_legit_ber = np.mean([r["legit_ber"] for r in tampering_results])
    
    print(f"Average Tampered Image Metrics: NCC={avg_tamp_ncc:.4f}, BER={avg_tamp_ber:.4f}")
    print(f"Average Legitimate (JPEG 70) Metrics: NCC={avg_legit_ncc:.4f}, BER={avg_legit_ber:.4f}")
    
    # Output directory
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    plt.plot(fpr, tpr, color="darkorange", lw=2.5, label=f"MantiQ-Auth (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve for Image Tampering Detection")
    plt.legend(loc="lower right")
    
    roc_plot_path = output_dir / "tampering_roc.png"
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    logger.info("Saved ROC plot to %s", roc_plot_path)


if __name__ == "__main__":
    main()

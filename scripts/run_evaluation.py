"""
MantiQ-Auth: Evaluation and Benchmarking Script

Runs the complete evaluation suite for the MantiQ-Auth system:
    1. Robustness evaluation (NCC & BER under JPEG compression, noise, rotation, etc.)
    2. Security attack sensitivity (local tampering, replay, downgrade/resize, crop/pad, contrast, brightness, salt & pepper)
    3. Statistical significance testing (paired t-test, Wilcoxon signed-rank, Cohen's d) comparing ViT vs ResNet
    4. Determinism checks (hash consistency)
    5. Uniqueness checks (different images → different hashes)
    6. Performance benchmarks (feature extraction, hashing, signing)
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import Timer, save_json, setup_logging

# Prevent UnicodeEncodeError on Windows PowerShell by forcing UTF-8 encoding
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

logger = logging.getLogger("mantiq.eval")


# ─── Attack Helper Functions ──────────────────────────────────────────────────

def apply_jpeg(img: Image.Image, quality: int) -> Image.Image:
    """Apply lossy JPEG compression and return PIL Image."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_gaussian_noise(img: Image.Image, sigma: float = 0.03) -> Image.Image:
    """Add zero-mean Gaussian noise to image."""
    arr = np.array(img).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, arr.shape)
    noisy = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((noisy * 255.0).astype(np.uint8))


def apply_rotation(img: Image.Image, angle: float) -> Image.Image:
    """Rotate image by specified angle (filling background with black)."""
    return img.rotate(angle, Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))


def apply_tampering(img: Image.Image) -> Image.Image:
    """Apply local tampering: draw a 40x40 gray box in the center."""
    w, h = img.size
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    x0, y0 = w // 2 - 20, h // 2 - 20
    x1, y1 = w // 2 + 20, h // 2 + 20
    draw.rectangle([x0, y0, x1, y1], fill=(128, 128, 128))
    return img_copy


def apply_resize(img: Image.Image) -> Image.Image:
    """Downsample image to 32x32 and upscale it back (downgrade attack)."""
    w, h = img.size
    img_small = img.resize((32, 32), Image.Resampling.BILINEAR)
    return img_small.resize((w, h), Image.Resampling.BILINEAR)


def apply_brightness(img: Image.Image, factor: float) -> Image.Image:
    """Enhance or reduce brightness."""
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def apply_contrast(img: Image.Image, factor: float) -> Image.Image:
    """Enhance or reduce contrast."""
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def apply_salt_pepper(img: Image.Image, amount: float = 0.05) -> Image.Image:
    """Add Salt & Pepper noise to image."""
    arr = np.array(img).copy()
    # Add salt
    num_salt = np.ceil(amount * arr.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in arr.shape]
    arr[tuple(coords)] = 255
    # Add pepper
    num_pepper = np.ceil(amount * arr.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in arr.shape]
    arr[tuple(coords)] = 0
    return Image.fromarray(arr)


def apply_cropping(img: Image.Image, crop_fraction: float = 0.20) -> Image.Image:
    """Crop borders by a fraction and pad back to original dimensions with black."""
    w, h = img.size
    crop_w = int(w * crop_fraction)
    crop_h = int(h * crop_fraction)
    img_cropped = img.crop((crop_w, crop_h, w - crop_w, h - crop_h))
    
    new_img = Image.new("RGB", (w, h), (0, 0, 0))
    new_img.paste(img_cropped, (crop_w, crop_h))
    return new_img


# ─── Statistics Helpers ───────────────────────────────────────────────────────

def calculate_stats(scores: List[float]) -> Dict:
    """Compute mean, median, std, and 95% confidence interval for a list of metrics."""
    n = len(scores)
    if n == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    
    arr = np.array(scores)
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    
    # 95% Confidence Interval half-width (1.96 * SE)
    ci_half = 1.96 * std_val / np.sqrt(n) if n > 0 else 0.0
    
    return {
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "ci_lower": max(0.0, mean_val - ci_half),
        "ci_upper": min(1.0, mean_val + ci_half)
    }


def run_significance_tests(vit_ncc: List[float], resnet_ncc: List[float], vit_ber: List[float], resnet_ber: List[float]) -> Dict:
    """Run paired t-test and Wilcoxon signed-rank test comparing ViT vs ResNet."""
    n = len(vit_ncc)
    if n < 2:
        empty = {"t_stat": 0.0, "t_p": 1.0, "wilcoxon_stat": 0.0, "wilcoxon_p": 1.0, "cohen_d": 0.0}
        return {"ncc": empty, "ber": empty}
        
    def test_pair(vit_arr, resnet_arr):
        v = np.array(vit_arr)
        r = np.array(resnet_arr)
        diff = v - r
        
        # Paired t-test
        try:
            t_stat, t_p = stats.ttest_rel(v, r)
            if np.isnan(t_stat):
                t_stat, t_p = 0.0, 1.0
        except Exception:
            t_stat, t_p = 0.0, 1.0
            
        # Wilcoxon signed-rank test
        if np.all(diff == 0):
            w_stat, w_p = 0.0, 1.0
        else:
            try:
                w_stat, w_p = stats.wilcoxon(v, r)
            except Exception:
                w_stat, w_p = 0.0, 1.0
                
        # Cohen's d for paired samples
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1) if len(diff) > 1 else 0.0
        cohen_d = mean_diff / std_diff if std_diff > 0 else 0.0
        
        return {
            "t_stat": float(t_stat),
            "t_p": float(t_p),
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_p": float(w_p),
            "cohen_d": float(cohen_d)
        }
        
    return {
        "ncc": test_pair(vit_ncc, resnet_ncc),
        "ber": test_pair(vit_ber, resnet_ber)
    }


def check_pass_fail(attack: str, ncc_mean: float, ber_mean: float) -> str:
    """Determine clinical pass/fail/borderline status."""
    # Clinical target: JPEG Q=70 -> NCC > 0.93 and BER < 0.10
    if "JPEG Q=70" in attack:
        if ncc_mean > 0.93 and ber_mean < 0.10:
            return "✅ Pass"
        elif ncc_mean > 0.90 and ber_mean < 0.15:
            return "⚠️ Borderline"
        else:
            return "❌ Fail"
            
    # For security attacks (Tampering, Replay, Downgrade), we WANT high BER and low NCC
    elif any(sec_attack in attack for sec_attack in ["Tampering", "Replay", "Downgrade"]):
        # Sensitivity: tampering/replay should result in high BER (> 0.40) and low similarity
        if ber_mean >= 0.40 and ncc_mean <= 0.85:
            return "✅ Pass (Detected)"
        elif ber_mean >= 0.30 and ncc_mean <= 0.90:
            return "⚠️ Borderline (Detected)"
        else:
            return "❌ Fail (Undetected)"
            
    # For other non-malicious distortions
    else:
        if "JPEG Q=90" in attack:
            return "✅ Pass" if (ncc_mean > 0.95 and ber_mean < 0.05) else "❌ Fail"
        elif "JPEG Q=50" in attack or "Gaussian Noise" in attack or "Contrast" in attack or "Brightness" in attack:
            if ncc_mean > 0.88 and ber_mean < 0.12:
                return "✅ Pass"
            elif ncc_mean > 0.83 and ber_mean < 0.18:
                return "⚠️ Borderline"
            else:
                return "❌ Fail"
        elif "Rotation" in attack or "Cropping" in attack or "Salt & Pepper" in attack:
            if ncc_mean > 0.85 and ber_mean < 0.15:
                return "✅ Pass"
            elif ncc_mean > 0.80 and ber_mean < 0.20:
                return "⚠️ Borderline"
            else:
                return "❌ Fail"
        elif "JPEG Q=30" in attack:
            if ncc_mean > 0.80 and ber_mean < 0.20:
                return "✅ Pass"
            elif ncc_mean > 0.75 and ber_mean < 0.25:
                return "⚠️ Borderline"
            else:
                return "❌ Fail"
                
    return "✅ Pass"


# ─── Evaluation Steps ─────────────────────────────────────────────────────────

def find_test_images(image_dir: str, max_images: int = 10) -> List[Path]:
    """Find DICOM or PNG test images."""
    image_dir = Path(image_dir)
    images = []
    for ext in ["*.dcm", "*.png", "*.jpg"]:
        for p in image_dir.rglob(ext):
            if ext != "*.dcm" and p.stat().st_size < 5000:
                continue
            images.append(p)
    return sorted(images)[:max_images]


def load_pil_image(img_path: Path) -> Image.Image:
    """Load DICOM or standard image as a PIL Image."""
    if img_path.suffix.lower() == ".dcm":
        import pydicom
        try:
            ds = pydicom.dcmread(str(img_path))
            pixel_array = ds.pixel_array.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to read DICOM {img_path.name}: {e}") from e

        # Handle multi-frame DICOM files (e.g. Ultrasound Cine loops)
        if pixel_array.ndim > 2:
            pixel_array = pixel_array[0]

        if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
            center_val = ds.WindowCenter
            width_val = ds.WindowWidth
            center = float(center_val[0]) if hasattr(center_val, "__getitem__") and not isinstance(center_val, (str, bytes)) else float(center_val)
            width = float(width_val[0]) if hasattr(width_val, "__getitem__") and not isinstance(width_val, (str, bytes)) else float(width_val)
            lower = center - width / 2
            upper = center + width / 2
            pixel_array = np.clip(pixel_array, lower, upper)

        pmin, pmax = pixel_array.min(), pixel_array.max()
        if pmax - pmin > 0:
            pixel_array = ((pixel_array - pmin) / (pmax - pmin) * 255).astype(np.uint8)
        else:
            pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
        return Image.fromarray(pixel_array).convert("RGB")
    else:
        return Image.open(img_path).convert("RGB")


def find_images(image_dir: str | Path, max_n: int = 10) -> List[Path]:
    """Expose find_test_images under the name find_images for legacy scratch scripts."""
    return find_test_images(image_dir, max_images=max_n)


def evaluate_determinism(image_paths: List[Path], runs: int = 3) -> Dict:
    """Verify that the same image produces identical hashes across runs."""
    from src.robust_hash import compute_robust_hash

    results = {"passed": True, "details": []}

    for img_path in image_paths[:3]:
        hashes = []
        for _ in range(runs):
            h = compute_robust_hash(img_path)
            hashes.append(h)

        all_same = len(set(hashes)) == 1
        results["details"].append({
            "image": img_path.name,
            "hash": hashes[0][:16] + "...",
            "consistent": all_same,
            "runs": runs,
        })
        if not all_same:
            results["passed"] = False

    logger.info("Determinism: %s", "PASSED ✅" if results["passed"] else "FAILED ❌")
    return results


def evaluate_uniqueness(image_paths: List[Path]) -> Dict:
    """Verify that different images produce different hashes."""
    from src.robust_hash import compute_robust_hash

    hashes = {}
    for img_path in image_paths:
        h = compute_robust_hash(img_path)
        hashes[str(img_path)] = h

    unique_hashes = set(hashes.values())
    all_unique = len(unique_hashes) == len(hashes)

    result = {
        "passed": all_unique,
        "total_images": len(hashes),
        "unique_hashes": len(unique_hashes),
        "collision_rate": 1 - len(unique_hashes) / max(len(hashes), 1),
    }

    logger.info(
        "Uniqueness: %d/%d unique hashes — %s",
        len(unique_hashes), len(hashes),
        "PASSED ✅" if all_unique else "FAILED ❌",
    )
    return result


def evaluate_robustness_and_attacks(image_paths: List[Path], output_dir: Path) -> Dict:
    """
    Run comprehensive robustness and security attack suite on ViT vs ResNet.
    Saves output/robustness_metrics.csv.
    """
    from src.feature_extraction import extract_features, load_and_preprocess_image, load_and_preprocess_dicom, compute_ncc
    from src.robust_hash import quantize_to_binary, compute_bit_error_rate

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "robustness_metrics.csv"

    # Define attacks to run
    attack_names = [
        "JPEG Q=90",
        "JPEG Q=70",
        "JPEG Q=50",
        "JPEG Q=30",
        "Gaussian Noise σ=0.03",
        "Rotation 10°",
        "Tampering",
        "Replay",
        "Downgrade (Resize)",
        "Brightness (1.5x)",
        "Contrast (2.0x)",
        "Salt & Pepper Noise",
        "Cropping & Padding"
    ]

    # Accumulate metrics
    vit_ncc_all = {att: [] for att in attack_names}
    resnet_ncc_all = {att: [] for att in attack_names}
    vit_ber_all = {att: [] for att in attack_names}
    resnet_ber_all = {att: [] for att in attack_names}

    csv_rows = []
    n_images = len(image_paths)

    for i, img_path in enumerate(image_paths):
        try:
            # Load original image and extract original features
            img = load_pil_image(img_path)
            
            # Original features and bits (ViT)
            if img_path.suffix.lower() == ".dcm":
                tensor_orig = load_and_preprocess_dicom(img_path)
            else:
                tensor_orig = load_and_preprocess_image(img_path)
            feat_orig_vit = extract_features(tensor_orig, extractor="vit")
            bits_orig_vit = quantize_to_binary(feat_orig_vit)

            # Original features and bits (ResNet)
            feat_orig_resnet = extract_features(tensor_orig, extractor="resnet")
            bits_orig_resnet = quantize_to_binary(feat_orig_resnet)

            # Replay image setup
            replay_idx = (i + 1) % n_images
            replay_path = image_paths[replay_idx]

            for att in attack_names:
                img_att = None
                
                # Apply distortion/attack
                if att == "JPEG Q=90":
                    img_att = apply_jpeg(img, 90)
                elif att == "JPEG Q=70":
                    img_att = apply_jpeg(img, 70)
                elif att == "JPEG Q=50":
                    img_att = apply_jpeg(img, 50)
                elif att == "JPEG Q=30":
                    img_att = apply_jpeg(img, 30)
                elif att == "Gaussian Noise σ=0.03":
                    img_att = apply_gaussian_noise(img, 0.03)
                elif att == "Rotation 10°":
                    img_att = apply_rotation(img, 10.0)
                elif att == "Tampering":
                    img_att = apply_tampering(img)
                elif att == "Replay":
                    pass
                elif att == "Downgrade (Resize)":
                    img_att = apply_resize(img)
                elif att == "Brightness (1.5x)":
                    img_att = apply_brightness(img, 1.5)
                elif att == "Contrast (2.0x)":
                    img_att = apply_contrast(img, 2.0)
                elif att == "Salt & Pepper Noise":
                    img_att = apply_salt_pepper(img, 0.05)
                elif att == "Cropping & Padding":
                    img_att = apply_cropping(img, 0.20)

                # Extract features/bits from attacked state
                if att == "Replay":
                    if replay_path.suffix.lower() == ".dcm":
                        tensor_att = load_and_preprocess_dicom(replay_path)
                    else:
                        tensor_att = load_and_preprocess_image(replay_path)
                    feat_att_vit = extract_features(tensor_att, extractor="vit")
                    bits_att_vit = quantize_to_binary(feat_att_vit)
                    
                    feat_att_resnet = extract_features(tensor_att, extractor="resnet")
                    bits_att_resnet = quantize_to_binary(feat_att_resnet)
                else:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        img_att.save(tmp.name, format="PNG")
                        tmp_path = tmp.name
                    
                    tensor_att = load_and_preprocess_image(tmp_path)
                    feat_att_vit = extract_features(tensor_att, extractor="vit")
                    bits_att_vit = quantize_to_binary(feat_att_vit)
                    
                    feat_att_resnet = extract_features(tensor_att, extractor="resnet")
                    bits_att_resnet = quantize_to_binary(feat_att_resnet)
                    
                    os.unlink(tmp_path)

                # Compute metrics
                ncc_vit = compute_ncc(feat_orig_vit, feat_att_vit)
                ber_vit = compute_bit_error_rate(bits_orig_vit, bits_att_vit)

                ncc_resnet = compute_ncc(feat_orig_resnet, feat_att_resnet)
                ber_resnet = compute_bit_error_rate(bits_orig_resnet, bits_att_resnet)

                # Hash matching (Authentication via purely cryptographic hash)
                from src.robust_hash import apply_bch_encoding, compute_final_hash
                
                # We simulate proper BCH verification where the original ECC bytes are available to correct the attacked bits.
                def check_hash_match_with_bch(orig_bits, att_bits, bch_n=1023, bch_t=16):
                    n_bits = len(orig_bits)
                    pad_len = (8 - n_bits % 8) % 8
                    
                    orig_padded = orig_bits if pad_len == 0 else np.concatenate([orig_bits, np.zeros(pad_len, dtype=np.uint8)])
                    orig_data = np.packbits(orig_padded).tobytes()
                    
                    att_padded = att_bits if pad_len == 0 else np.concatenate([att_bits, np.zeros(pad_len, dtype=np.uint8)])
                    att_data = np.packbits(att_padded).tobytes()
                    
                    try:
                        import bchlib
                        import math
                        m = int(round(math.log2(bch_n + 1)))
                        bch = bchlib.BCH(t=bch_t, m=m)
                        max_data_len = bch.n // 8
                        
                        o_data = orig_data[:max_data_len]
                        a_data = bytearray(att_data[:max_data_len])
                        
                        # Generate original ECC
                        orig_ecc = bch.encode(o_data)
                        
                        # Attempt correction of attacked data using original ECC
                        bitflips = bch.decode(a_data, orig_ecc)
                        
                        if bitflips != -1:
                            # Correction successful, apply correction in-place
                            bch.correct(a_data, bytearray(orig_ecc))
                            att_encoded = bytes(a_data) + orig_ecc
                        else:
                            # Correction failed, use uncorrected data with original ECC
                            att_encoded = bytes(a_data) + orig_ecc
                            
                        orig_encoded = o_data + orig_ecc
                        
                        return compute_final_hash(orig_encoded) == compute_final_hash(att_encoded)
                    except ImportError:
                        # Fallback if bchlib not installed
                        return compute_final_hash(orig_data) == compute_final_hash(att_data)
                
                hash_match = check_hash_match_with_bch(bits_orig_resnet, bits_att_resnet)

                # Save metrics
                vit_ncc_all[att].append(ncc_vit)
                resnet_ncc_all[att].append(ncc_resnet)
                vit_ber_all[att].append(ber_vit)
                resnet_ber_all[att].append(ber_resnet)

                # Append to CSV row (using ResNet statistics since it's the primary feature extractor now)
                parts = att.split(" ", 1)
                attack_type = parts[0]
                attack_param = parts[1] if len(parts) > 1 else ""
                csv_rows.append({
                    "image_file": img_path.name,
                    "attack_type": attack_type,
                    "attack_parameter": attack_param,
                    "NCC": round(ncc_resnet, 6),
                    "BER": round(ber_resnet, 6),
                    "Hash_Match": hash_match
                })

        except Exception as e:
            logger.warning("Robustness/attack test failed for %s: %s", img_path.name, e)

    # Save to CSV
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_file", "attack_type", "attack_parameter", "NCC", "BER", "Hash_Match"])
        writer.writeheader()
        writer.writerows(csv_rows)
    logger.info("Saved all robustness results to CSV: %s", csv_path)

    # Compile statistics & significance testing
    stats_summary = {}
    for att in attack_names:
        # Calculate hash match rate for this attack
        att_rows = [r for r in csv_rows if r["attack_type"] + (" " + r["attack_parameter"] if r["attack_parameter"] else "") == att]
        hash_match_rate = sum(1 for r in att_rows if r["Hash_Match"]) / len(att_rows) if att_rows else 0.0

        stats_summary[att] = {
            "vit": {
                "ncc": calculate_stats(vit_ncc_all[att]),
                "ber": calculate_stats(vit_ber_all[att])
            },
            "resnet": {
                "ncc": calculate_stats(resnet_ncc_all[att]),
                "ber": calculate_stats(resnet_ber_all[att])
            },
            "hash_match_rate": hash_match_rate,
            "significance": run_significance_tests(
                vit_ncc_all[att], resnet_ncc_all[att],
                vit_ber_all[att], resnet_ber_all[att]
            )
        }

    return stats_summary


def benchmark_performance(image_paths: List[Path], runs: int = 5) -> Dict:
    """Benchmark timing for all pipeline stages."""
    from src.feature_extraction import extract_features, load_and_preprocess_image, load_and_preprocess_dicom
    from src.robust_hash import apply_bch_encoding, compute_final_hash, quantize_to_binary
    from src.hybrid_signatures import generate_keypair_ecdsa, generate_keypair_mldsa

    results = {}
    img_path = image_paths[0] if image_paths else None
    if img_path is None:
        return {"error": "No test images available."}

    # Feature extraction timing (ViT)
    times = []
    for _ in range(runs):
        with Timer() as t:
            if img_path.suffix.lower() == ".dcm":
                tensor = load_and_preprocess_dicom(img_path)
            else:
                tensor = load_and_preprocess_image(img_path)
            features = extract_features(tensor, extractor="vit")
        times.append(t.elapsed)
    results["feature_extraction_ms"] = round(np.mean(times) * 1000, 2)

    # Hashing timing
    times = []
    for _ in range(runs):
        with Timer() as t:
            binary = quantize_to_binary(features)
            encoded = apply_bch_encoding(binary)
            _ = compute_final_hash(encoded)
        times.append(t.elapsed)
    results["hashing_ms"] = round(np.mean(times) * 1000, 2)

    # Key generation timing
    with Timer() as t:
        _ = generate_keypair_ecdsa()
    results["ecdsa_keygen_ms"] = round(t.elapsed * 1000, 2)

    try:
        with Timer() as t:
            _ = generate_keypair_mldsa(65)
        results["mldsa_keygen_ms"] = round(t.elapsed * 1000, 2)
    except ImportError:
        results["mldsa_keygen_ms"] = "liboqs not available"

    logger.info("Performance benchmarks: %s", results)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run MantiQ-Auth evaluation suite.")
    parser.add_argument("--images", default="data/dicom_processed",
                       help="Directory containing test images.")
    parser.add_argument("--output", default="output", help="Output directory.")
    parser.add_argument("--runs", type=int, default=5, help="Benchmark repetitions.")
    parser.add_argument("--max-images", type=int, default=100, help="Maximum number of images to evaluate.")
    args = parser.parse_args()

    setup_logging("INFO")

    images = find_test_images(args.images, max_images=args.max_images)
    if not images:
        logger.error("No test images found in %s. Run preprocessing first.", args.images)
        sys.exit(1)

    logger.info("Running evaluation on %d images...", len(images))

    all_results = {}

    print("\n" + "=" * 80)
    print("                 MantiQ-Auth Evaluation Suite (NCC & BER)")
    print("=" * 80)

    print("\n📐 Determinism Check...")
    all_results["determinism"] = evaluate_determinism(images, runs=args.runs)

    print("\n🔑 Uniqueness Check...")
    all_results["uniqueness"] = evaluate_uniqueness(images)

    print("\n🖼️  Robustness and Security Attack Evaluation...")
    all_results["robustness_and_attacks"] = evaluate_robustness_and_attacks(images, Path(args.output))

    print("\n⏱️  Performance Benchmarks...")
    all_results["performance"] = benchmark_performance(images, runs=args.runs)

    # Save all results to json
    output_path = Path(args.output) / "evaluation_results.json"
    save_json(all_results, output_path)
    print(f"\n📄 Results saved to {output_path}")

    # Display Summary Table in Stdout
    print("\n" + "=" * 115)
    print(f"{'Attack':<24} | {'Mean NCC (95% CI)':<22} | {'Mean BER (95% CI)':<22} | {'Hash Match':<10} | {'Pass/Fail':<10}")
    print("-" * 115)
    
    for attack, stats_data in all_results["robustness_and_attacks"].items():
        v_ncc = stats_data["resnet"]["ncc"] # Using ResNet as primary now
        v_ber = stats_data["resnet"]["ber"]
        hash_match_rate = stats_data["hash_match_rate"]
        
        ncc_str = f"{v_ncc['mean']:.4f} ({v_ncc['ci_lower']:.3f}-{v_ncc['ci_upper']:.3f})"
        ber_str = f"{v_ber['mean']:.4f} ({v_ber['ci_lower']:.3f}-{v_ber['ci_upper']:.3f})"
        hash_str = f"{hash_match_rate:.1%}"
        
        status = check_pass_fail(attack, v_ncc["mean"], v_ber["mean"])
        print(f"{attack:<24} | {ncc_str:<22} | {ber_str:<22} | {hash_str:<10} | {status:<10}")
        
    print("=" * 115)

    # Summary
    print("\n" + "=" * 60)
    print("Summary Status:")
    det = all_results["determinism"]["passed"]
    uniq = all_results["uniqueness"]["passed"]
    print(f"  Determinism:  {'✅ PASSED' if det else '❌ FAILED'}")
    print(f"  Uniqueness:   {'✅ PASSED' if uniq else '❌ FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

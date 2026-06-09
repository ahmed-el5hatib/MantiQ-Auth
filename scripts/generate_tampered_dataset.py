"""
MantiQ-Auth: Tampered Dataset Generator (Improvement 1)

Generates a comprehensive dataset for tampering detection evaluation:
  - Takes original anatomical slices (CT, MRI, X-ray)
  - Produces 5 types of tampering per image.
  - Supports 'standard' and 'hard' difficulties.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, setup_logging

logger = logging.getLogger("mantiq.tamper_gen")

# ─── Tampering Functions ────────────────────────────────────────────────────

def tamper_insertion(img_array: np.ndarray, rng: np.random.RandomState, difficulty: str = "standard") -> np.ndarray:
    h, w = img_array.shape[:2]
    result = img_array.copy()

    cx = rng.randint(int(w * 0.2), int(w * 0.8))
    cy = rng.randint(int(h * 0.2), int(h * 0.8))

    # MODIFICATION: Random between 3-8 mm (hard) vs 5-15 mm (standard)
    if difficulty == "hard":
        radius = rng.randint(3, 9)
    else:
        radius = rng.randint(5, 16)

    local_mean = float(np.mean(img_array[
        max(0, cy - radius):min(h, cy + radius),
        max(0, cx - radius):min(w, cx + radius),
    ]))
    nodule_intensity = min(255, int(local_mean + rng.randint(40, 120)))

    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = dist <= radius

    intensity_map = np.exp(-0.5 * (dist / (max(1, radius * 0.6))) ** 2) * nodule_intensity
    intensity_map = intensity_map.astype(np.float32)

    if result.ndim == 3:
        for c in range(result.shape[2]):
            channel = result[:, :, c].astype(np.float32)
            channel[mask] = intensity_map[mask]
            result[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
    else:
        channel = result.astype(np.float32)
        channel[mask] = intensity_map[mask]
        result = np.clip(channel, 0, 255).astype(np.uint8)

    return result


def tamper_removal(img_array: np.ndarray, rng: np.random.RandomState, difficulty: str = "standard") -> np.ndarray:
    h, w = img_array.shape[:2]
    result = img_array.copy()

    cx = rng.randint(int(w * 0.25), int(w * 0.75))
    cy = rng.randint(int(h * 0.25), int(h * 0.75))
    
    # MODIFICATION: Random between 3-8 mm (hard) vs 8-20 mm (standard)
    if difficulty == "hard":
        radius = rng.randint(3, 9)
    else:
        radius = rng.randint(8, 20)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)

    if result.ndim == 2:
        result = cv2.inpaint(result, mask, inpaintRadius=radius, flags=cv2.INPAINT_NS)
    else:
        result = cv2.inpaint(result, mask, inpaintRadius=radius, flags=cv2.INPAINT_NS)

    return result


def tamper_intensity(img_array: np.ndarray, rng: np.random.RandomState, difficulty: str = "standard") -> np.ndarray:
    h, w = img_array.shape[:2]
    result = img_array.copy().astype(np.float32)

    rw = rng.randint(int(w * 0.10), int(w * 0.25))
    rh = rng.randint(int(h * 0.10), int(h * 0.25))
    rx = rng.randint(int(w * 0.15), int(w * 0.85) - rw)
    ry = rng.randint(int(h * 0.15), int(h * 0.85) - rh)

    # MODIFICATION: Intensity change of +/- 30 to 100 HU (hard) vs 200-500 HU (standard)
    if difficulty == "hard":
        # 30-100 HU is approx 8-25 pixel intensity
        sign = rng.choice([-1, 1])
        intensity_offset = sign * rng.randint(8, 26)
    else:
        intensity_offset = rng.randint(30, 70)

    if result.ndim == 3:
        result[ry:ry + rh, rx:rx + rw, :] += intensity_offset
    else:
        result[ry:ry + rh, rx:rx + rw] += intensity_offset

    return np.clip(result, 0, 255).astype(np.uint8)


def tamper_crop_rescale(img_array: np.ndarray, rng: np.random.RandomState, difficulty: str = "standard") -> np.ndarray:
    h, w = img_array.shape[:2]
    crop_frac = rng.uniform(0.05, 0.10)
    crop_x = int(w * crop_frac)
    crop_y = int(h * crop_frac)
    cropped = img_array[crop_y:h - crop_y, crop_x:w - crop_x]
    
    if img_array.ndim == 3:
        result = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        result = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return result


def tamper_copy_move(img_array: np.ndarray, rng: np.random.RandomState, difficulty: str = "standard") -> np.ndarray:
    h, w = img_array.shape[:2]
    result = img_array.copy()

    # MODIFICATION: Patch size 5-12 (hard) vs 15-30 (standard)
    if difficulty == "hard":
        patch_size = rng.randint(5, 13)
    else:
        patch_size = rng.randint(15, 31)

    src_x = rng.randint(10, max(11, w // 2 - patch_size))
    src_y = rng.randint(10, max(11, h // 2 - patch_size))

    dst_x = rng.randint(max(1, w // 2), max(2, w - patch_size - 5))
    dst_y = rng.randint(max(1, h // 2), max(2, h - patch_size - 5))

    if result.ndim == 3:
        patch = result[src_y:src_y + patch_size, src_x:src_x + patch_size, :].copy()
        result[dst_y:dst_y + patch_size, dst_x:dst_x + patch_size, :] = patch
    else:
        patch = result[src_y:src_y + patch_size, src_x:src_x + patch_size].copy()
        result[dst_y:dst_y + patch_size, dst_x:dst_x + patch_size] = patch

    return result

# NEW SUBTLE TAMPERING TYPE
def tamper_subtle_gradient(img_array: np.ndarray, rng: np.random.RandomState, difficulty: str = "standard") -> np.ndarray:
    """
    Apply a gentle grayscale gradient to a small circular area (radius 3-5 pixels) with opacity 0.1.
    """
    h, w = img_array.shape[:2]
    result = img_array.copy().astype(np.float32)

    cx = rng.randint(int(w * 0.2), int(w * 0.8))
    cy = rng.randint(int(h * 0.2), int(h * 0.8))
    radius = rng.randint(3, 6)

    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = dist <= radius

    # Create gradient: linear falloff based on horizontal position or distance
    gradient_1d = ((xx - (cx - radius)) / (2 * max(1, radius))).clip(0, 1) * 255.0
    gradient = np.broadcast_to(gradient_1d, (h, w))
    
    # Opacity 0.1
    alpha = 0.1
    
    if result.ndim == 3:
        for c in range(result.shape[2]):
            result[mask, c] = result[mask, c] * (1 - alpha) + gradient[mask] * alpha
    else:
        result[mask] = result[mask] * (1 - alpha) + gradient[mask] * alpha

    return np.clip(result, 0, 255).astype(np.uint8)


def get_tampering_functions(difficulty: str):
    if difficulty == "hard":
        # Replace crop_rescale with subtle_gradient to keep 5 types
        return {
            "insertion": tamper_insertion,
            "removal": tamper_removal,
            "intensity_modification": tamper_intensity,
            "subtle_gradient": tamper_subtle_gradient,
            "copy_move": tamper_copy_move,
        }
    else:
        return {
            "insertion": tamper_insertion,
            "removal": tamper_removal,
            "intensity_modification": tamper_intensity,
            "crop_rescale": tamper_crop_rescale,
            "copy_move": tamper_copy_move,
        }


def collect_source_images(input_dir: Path, max_images: int = 200) -> List[Path]:
    processed_dir = input_dir / "dicom_processed"
    images = []

    if processed_dir.exists():
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            for p in sorted(processed_dir.rglob(ext)):
                if p.stat().st_size > 3000:  
                    images.append(p)
                    if len(images) >= max_images:
                        return images

    synthetic_dir = input_dir / "synthetic"
    if synthetic_dir.exists():
        for ext in ["*.png", "*.jpg"]:
            for p in sorted(synthetic_dir.rglob(ext)):
                if p.stat().st_size > 3000:
                    images.append(p)
                    if len(images) >= max_images:
                        return images

    # Also just scan the root if neither exists
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        for p in sorted(input_dir.rglob(ext)):
            if "tampered" not in str(p) and p.stat().st_size > 3000:
                images.append(p)
                if len(images) >= max_images:
                    return images

    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--jpeg-quality", type=int, default=70)
    parser.add_argument("--seed", type=int, default=42)
    # MODIFICATION: Configurable difficulty parameter
    parser.add_argument("--difficulty", type=str, choices=["standard", "hard"], default="standard")
    parser.add_argument("--input-dir", type=str, default=None, help="Input directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    setup_logging("INFO")
    logger.info("=" * 60)
    logger.info(f"MantiQ-Auth: Tampered Dataset Generator (Difficulty: {args.difficulty})")
    logger.info("=" * 60)

    config = load_config()
    data_root = Path(config.get("paths", {}).get("data_root", "data"))

    if args.input_dir:
        input_base = Path(args.input_dir)
    else:
        input_base = data_root

    # MODIFICATION: Output folder based on difficulty
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        folder_name = "tampered_dataset_harder" if args.difficulty == "hard" else "tampered_dataset"
        output_base = data_root / folder_name
        
    original_dir = output_base / "original"
    tampered_dir = output_base / "tampered"
    original_dir.mkdir(parents=True, exist_ok=True)
    tampered_dir.mkdir(parents=True, exist_ok=True)

    source_images = collect_source_images(input_base, max_images=args.max_images)
    if not source_images:
        logger.error(f"No source images found in {input_base}.")
        sys.exit(1)

    rng = np.random.RandomState(args.seed)
    metadata_rows = []

    logger.info("Generating %d untampered (JPEG Q=%d) originals...", len(source_images), args.jpeg_quality)
    for i, src_path in enumerate(source_images):
        try:
            img = Image.open(src_path).convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=args.jpeg_quality)
            buffer.seek(0)
            img_jpeg = Image.open(buffer).convert("RGB")

            out_name = f"original_{i:04d}.png"
            out_path = original_dir / out_name
            img_jpeg.save(out_path)

            rel_path = str(out_path.relative_to(data_root)).replace("\\", "/")
            metadata_rows.append({
                "image_path": rel_path,
                "label": 0,
                "tampering_type": "none",
                "source_image": src_path.name,
                "difficulty": args.difficulty  # MODIFICATION: Added difficulty column
            })
        except Exception as e:
            logger.warning("Failed to process original %s: %s", src_path.name, e)

    funcs = get_tampering_functions(args.difficulty)
    logger.info("Generating tampered images (5 types × %d sources)...", len(source_images))

    for i, src_path in enumerate(source_images):
        try:
            img = Image.open(src_path).convert("RGB")
            img_array = np.array(img)

            for ttype, tfunc in funcs.items():
                local_rng = np.random.RandomState(args.seed + i * 100 + hash(ttype) % 1000)
                tampered_array = tfunc(img_array, local_rng, difficulty=args.difficulty)
                tampered_img = Image.fromarray(tampered_array)

                out_name = f"tampered_{i:04d}_{ttype}.png"
                out_path = tampered_dir / out_name
                tampered_img.save(out_path)

                rel_path = str(out_path.relative_to(data_root)).replace("\\", "/")
                metadata_rows.append({
                    "image_path": rel_path,
                    "label": 1,
                    "tampering_type": ttype,
                    "source_image": src_path.name,
                    "difficulty": args.difficulty  # MODIFICATION: Added difficulty column
                })
        except Exception as e:
            logger.warning("Failed to tamper %s: %s", src_path.name, e)

        if (i + 1) % 50 == 0:
            logger.info("  Processed %d / %d source images...", i + 1, len(source_images))

    csv_path = output_base / "metadata.csv"
    fieldnames = ["image_path", "label", "tampering_type", "source_image", "difficulty"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    logger.info("Dataset generation complete!")
    logger.info("  Metadata:  %s", csv_path)

if __name__ == "__main__":
    main()

"""
MantiQ-Auth: DICOM Preprocessing Script

Loads raw DICOM files, applies windowing and normalization, and saves
processed images for feature extraction and analysis.

Usage:
    python scripts/preprocess_dicom.py [--input data/dicom_raw] [--output data/dicom_processed]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import setup_logging

logger = logging.getLogger("mantiq.preprocess")


def preprocess_dicom_file(
    dicom_path: Path,
    output_dir: Path,
    image_size: int = 224,
    save_format: str = "png",
) -> bool:
    """
    Preprocess a single DICOM file and save as PNG.

    Steps:
        1. Read DICOM pixel data
        2. Apply CT windowing (if available)
        3. Normalize to [0, 255]
        4. Resize to target dimensions
        5. Save as PNG

    Args:
        dicom_path: Path to the DICOM file.
        output_dir: Directory to save processed images.
        image_size: Target image size.
        save_format: Output format (png, jpg).

    Returns:
        True if successful, False otherwise.
    """
    try:
        import pydicom

        ds = pydicom.dcmread(str(dicom_path))
        pixel_array = ds.pixel_array.astype(np.float32)

        # Apply windowing for CT images
        if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
            center_val = ds.WindowCenter
            width_val = ds.WindowWidth
            center = float(center_val[0]) if hasattr(center_val, "__getitem__") and not isinstance(center_val, (str, bytes)) else float(center_val)
            width = float(width_val[0]) if hasattr(width_val, "__getitem__") and not isinstance(width_val, (str, bytes)) else float(width_val)
            lower = center - width / 2
            upper = center + width / 2
            pixel_array = np.clip(pixel_array, lower, upper)

        # Normalize to [0, 255]
        pmin, pmax = pixel_array.min(), pixel_array.max()
        if pmax - pmin > 0:
            pixel_array = ((pixel_array - pmin) / (pmax - pmin) * 255).astype(np.uint8)
        else:
            pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)

        # Convert to PIL and resize
        img = Image.fromarray(pixel_array).convert("RGB")
        img = img.resize((image_size, image_size), Image.LANCZOS)

        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = dicom_path.stem
        output_path = output_dir / f"{stem}.{save_format}"
        img.save(output_path)

        return True

    except Exception as e:
        logger.warning("Failed to preprocess %s: %s", dicom_path.name, e)
        return False


def preprocess_directory(
    input_dir: str = "data/dicom_raw",
    output_dir: str = "data/dicom_processed",
    image_size: int = 224,
) -> int:
    """
    Preprocess all DICOM files in a directory tree.

    Args:
        input_dir: Root directory containing DICOM files.
        output_dir: Output directory for processed images.
        image_size: Target image size.

    Returns:
        Number of successfully processed files.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        logger.error("Input directory does not exist: %s", input_path)
        return 0

    # Find all DICOM files recursively
    dcm_files = list(input_path.rglob("*.dcm"))
    if not dcm_files:
        logger.warning("No DICOM files found in %s.", input_path)
        return 0

    logger.info("Found %d DICOM files to preprocess.", len(dcm_files))

    success_count = 0
    for dcm_path in tqdm(dcm_files, desc="Preprocessing"):
        # Mirror directory structure
        rel_path = dcm_path.relative_to(input_path)
        out_subdir = output_path / rel_path.parent

        if preprocess_dicom_file(dcm_path, out_subdir, image_size):
            success_count += 1

    logger.info("Preprocessed %d/%d DICOM files.", success_count, len(dcm_files))
    return success_count


def main():
    parser = argparse.ArgumentParser(description="Preprocess DICOM images.")
    parser.add_argument("--input", default="data/dicom_raw", help="Input DICOM directory.")
    parser.add_argument("--output", default="data/dicom_processed", help="Output directory.")
    parser.add_argument("--size", type=int, default=224, help="Target image size.")
    args = parser.parse_args()

    setup_logging("INFO")
    total = preprocess_directory(args.input, args.output, args.size)
    print(f"\n✅ Preprocessed {total} images → {args.output}/")


if __name__ == "__main__":
    main()

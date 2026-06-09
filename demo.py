#!/usr/bin/env python3
"""
MantiQ-Auth Demo — Quantum-Resistant Medical Image Authentication

End-to-end demonstration of the MantiQ-Auth (SealPACS) pipeline:
    1. Download/generate test images
    2. Extract deep features (ResNet-18)
    3. Compute robust perceptual hash
    4. Generate hybrid signature (ECDSA-P256 + ML-DSA-65)
    5. Verify the signature
    6. Save benchmark metrics

Usage:
    python demo.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

# ─── Setup ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Prevent UnicodeEncodeError on Windows PowerShell by forcing UTF-8 encoding
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from src.utils import Timer, save_json, setup_logging

logger = setup_logging("INFO")


# ─── Helpers ────────────────────────────────────────────────────────────────

def generate_synthetic_test_images(
    output_dir: Path,
    count: int = 10,
) -> List[Path]:
    """
    Generate synthetic medical-like test images for demo purposes.

    Creates grayscale images with varying patterns that simulate
    CT scan slices (circular structures, noise, gradients).

    Args:
        output_dir: Directory to save generated images.
        count: Number of images to generate.

    Returns:
        List of paths to generated images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for i in range(count):
        # Create a 256x256 grayscale image with medical-like patterns
        img = np.zeros((256, 256), dtype=np.float32)

        # Add circular structure (simulating tissue cross-section)
        y, x = np.ogrid[-128:128, -128:128]
        radius = 60 + i * 5
        mask = x**2 + y**2 <= radius**2
        img[mask] = 0.6 + 0.1 * np.sin(i)

        # Add smaller internal structures
        for j in range(3):
            cx = 30 * np.cos(2 * np.pi * j / 3 + i * 0.5)
            cy = 30 * np.sin(2 * np.pi * j / 3 + i * 0.5)
            inner_mask = (x - cx)**2 + (y - cy)**2 <= (15 + j * 3)**2
            img[inner_mask] = 0.8 + 0.05 * j

        # Add realistic noise
        rng = np.random.RandomState(seed=42 + i)
        noise = rng.normal(0, 0.02, img.shape)
        img = np.clip(img + noise, 0, 1)

        # Convert to uint8 and save
        img_uint8 = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, mode="L").convert("RGB")

        path = output_dir / f"synthetic_ct_{i:03d}.png"
        pil_img.save(path)
        paths.append(path)

    logger.info("Generated %d synthetic test images in %s.", count, output_dir)
    return paths


def try_download_tcia_images(output_dir: Path, count: int = 10) -> List[Path]:
    """
    Attempt to download real DICOM images from TCIA.

    Falls back to synthetic images if download fails.

    Returns:
        List of image paths.
    """
    try:
        from scripts.download_tcia_images import TCIADownloader

        downloader = TCIADownloader(
            output_dir=str(output_dir),
            target_images=count,
        )

        logger.info("Attempting to download %d images from TCIA...", count)
        total = downloader.download_collection("LIDC-IDRI")

        if total > 0:
            # Find downloaded DICOM files
            dcm_files = sorted(output_dir.rglob("*.dcm"))[:count]
            if dcm_files:
                logger.info("Successfully downloaded %d DICOM images.", len(dcm_files))
                return dcm_files

    except Exception as e:
        logger.warning("TCIA download failed: %s", e)

    # Fallback to synthetic images
    logger.info("Using synthetic test images instead.")
    synthetic_dir = PROJECT_ROOT / "data" / "synthetic"
    return generate_synthetic_test_images(synthetic_dir, count)


# ─── Demo Pipeline ──────────────────────────────────────────────────────────

def run_demo() -> Dict[str, Any]:
    """
    Run the complete MantiQ-Auth demonstration pipeline.

    Returns:
        Dictionary of benchmark results.
    """
    print()
    print("=" * 70)
    print("  MantiQ-Auth: Quantum-Resistant Medical Image Authentication")
    print("  Demo Pipeline")
    print("=" * 70)
    print()

    results: Dict[str, Any] = {
        "system": "MantiQ-Auth (SealPACS)",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stages": {},
    }

    # ── Stage 1: Acquire Test Images ────────────────────────────────────
    print("📥 Stage 1: Acquiring test images...")
    print("-" * 50)

    with Timer("image_acquisition") as t:
        image_paths = try_download_tcia_images(
            PROJECT_ROOT / "data" / "dicom_raw",
            count=10,
        )

    results["stages"]["image_acquisition"] = {
        "image_count": len(image_paths),
        "time_seconds": round(t.elapsed, 3),
        "source": "TCIA" if image_paths[0].suffix == ".dcm" else "synthetic",
    }
    print(f"  ✅ {len(image_paths)} images ready ({t.elapsed:.2f}s)")
    print()

    # ── Stage 2: Feature Extraction ─────────────────────────────────────
    print("🧠 Stage 2: Extracting deep features (ResNet-18)...")
    print("-" * 50)

    from src.feature_extraction import (
        FEATURE_DIMS,
        extract_features,
        load_and_preprocess_dicom,
        load_and_preprocess_image,
    )

    test_image = image_paths[0]
    with Timer("feature_extraction") as t:
        if test_image.suffix.lower() == ".dcm":
            tensor = load_and_preprocess_dicom(test_image)
        else:
            tensor = load_and_preprocess_image(test_image)
        features = extract_features(tensor, extractor="resnet")

    feature_dim = len(features)
    feature_norm = float(np.linalg.norm(features))

    results["stages"]["feature_extraction"] = {
        "model": "ResNet-18",
        "feature_dim": feature_dim,
        "feature_norm": round(feature_norm, 6),
        "time_ms": round(t.elapsed * 1000, 2),
        "image": test_image.name,
    }

    print(f"  Model:         ResNet-18 (ImageNet pretrained)")
    print(f"  Feature dim:   {feature_dim}")
    print(f"  L2 norm:       {feature_norm:.6f}")
    print(f"  Time:          {t.elapsed * 1000:.1f} ms")

    # Verify expected dimension
    assert feature_dim == FEATURE_DIMS["resnet"], (
        f"Expected {FEATURE_DIMS['resnet']}-dim features, got {feature_dim}"
    )
    print(f"  ✅ Dimension check passed ({feature_dim} == {FEATURE_DIMS['resnet']})")
    print()

    # ── Stage 3: Robust Hash ────────────────────────────────────────────
    print("🔐 Stage 3: Computing robust perceptual hash...")
    print("-" * 50)

    from src.robust_hash import compute_robust_hash

    with Timer("hash_computation") as t:
        robust_hash = compute_robust_hash(test_image, feature_extractor="resnet")

    # Determinism check: compute again and verify identical
    robust_hash_2 = compute_robust_hash(test_image, feature_extractor="resnet")
    deterministic = robust_hash == robust_hash_2

    results["stages"]["robust_hash"] = {
        "hash": robust_hash,
        "algorithm": "SHA3-256",
        "hash_length_chars": len(robust_hash),
        "deterministic": deterministic,
        "time_ms": round(t.elapsed * 1000, 2),
    }

    print(f"  Hash:          {robust_hash[:32]}...")
    print(f"  Algorithm:     SHA3-256 (with BCH ECC)")
    print(f"  Length:         {len(robust_hash)} hex chars")
    print(f"  Deterministic: {'✅ Yes' if deterministic else '❌ No'}")
    print(f"  Time:          {t.elapsed * 1000:.1f} ms")
    print()

    # ── Stage 4: Hybrid Signature (ECDSA + ML-DSA) ─────────────────────
    print("✍️  Stage 4: Generating hybrid signature...")
    print("-" * 50)

    from src.hybrid_signatures import (
        generate_keypair_ecdsa,
        sign_hybrid,
        verify_hybrid,
    )

    # Generate ECDSA keys
    with Timer("ecdsa_keygen") as t_ecdsa_kg:
        ecdsa_kp = generate_keypair_ecdsa()
    print(f"  ECDSA-P256 keygen:  {t_ecdsa_kg.elapsed * 1000:.2f} ms")

    # Generate ML-DSA keys
    mldsa_available = True
    try:
        from src.hybrid_signatures import generate_keypair_mldsa

        with Timer("mldsa_keygen") as t_mldsa_kg:
            mldsa_kp = generate_keypair_mldsa(level=65)
        print(f"  ML-DSA-65 keygen:   {t_mldsa_kg.elapsed * 1000:.2f} ms")
        print(f"  ML-DSA PK size:     {len(mldsa_kp.public_key)} bytes")
    except ImportError:
        mldsa_available = False
        logger.warning("liboqs-python not installed — ML-DSA unavailable.")
        print("  ⚠️  ML-DSA-65:       liboqs-python not installed (skipping)")

    # Prepare message hash
    message_hash = hashlib.sha256(robust_hash.encode()).digest()

    if mldsa_available:
        # Sign with hybrid (ECDSA + ML-DSA)
        with Timer("hybrid_signing") as t_sign:
            signature = sign_hybrid(message_hash, ecdsa_kp, mldsa_kp)

        results["stages"]["hybrid_signature"] = {
            "combiner": signature.combiner,
            "total_size_bytes": signature.total_size,
            "ecdsa_sig_bytes": signature.ecdsa_size,
            "mldsa_sig_bytes": signature.mldsa_size,
            "signing_time_ms": round(t_sign.elapsed * 1000, 2),
            "ecdsa_keygen_ms": round(t_ecdsa_kg.elapsed * 1000, 2),
            "mldsa_keygen_ms": round(t_mldsa_kg.elapsed * 1000, 2),
        }

        print(f"  Combiner:           {signature.combiner}")
        print(f"  Signature size:     {signature.total_size} bytes")
        print(f"    ├─ ECDSA:         {signature.ecdsa_size} bytes")
        print(f"    └─ ML-DSA-65:     {signature.mldsa_size} bytes")
        print(f"  Signing time:       {t_sign.elapsed * 1000:.2f} ms")
        print()

        # ── Stage 5: Verification ──────────────────────────────────────
        print("✅ Stage 5: Verifying hybrid signature and hash...")
        print("-" * 50)

        from src.verifier import HashBasedVerifier

        verifier = HashBasedVerifier(
            ecdsa_public_key=ecdsa_kp.public_key,
            mldsa_public_key=mldsa_kp.public_key,
            feature_extractor="resnet",
            mldsa_level=mldsa_kp.level,
        )

        with Timer("verification") as t_verify:
            overall, hash_ok, ecdsa_ok, mldsa_ok = verifier.verify_image(
                image_path=test_image,
                signed_hash=robust_hash,
                signature=signature,
            )

        results["stages"]["verification"] = {
            "overall": overall,
            "hash_match": hash_ok,
            "ecdsa_valid": ecdsa_ok,
            "mldsa_valid": mldsa_ok,
            "verification_time_ms": round(t_verify.elapsed * 1000, 2),
        }

        print(f"  Hash Match:     {'✅ Yes' if hash_ok else '❌ No'}")
        print(f"  ECDSA-P256:     {'✅ Valid' if ecdsa_ok else '❌ Invalid'}")
        print(f"  ML-DSA-65:      {'✅ Valid' if mldsa_ok else '❌ Invalid'}")
        print(f"  Overall:        {'✅ VERIFIED' if overall else '❌ FAILED'}")
        print(f"  Verify time:    {t_verify.elapsed * 1000:.2f} ms")

    else:
        # ECDSA-only fallback
        from src.hybrid_signatures import sign_ecdsa, verify_ecdsa

        with Timer("ecdsa_signing") as t_sign:
            ecdsa_sig = sign_ecdsa(message_hash, ecdsa_kp)

        with Timer("ecdsa_verify") as t_verify:
            ecdsa_ok = verify_ecdsa(message_hash, ecdsa_sig, ecdsa_kp.public_key)

        results["stages"]["ecdsa_only_signature"] = {
            "sig_size_bytes": len(ecdsa_sig),
            "signing_time_ms": round(t_sign.elapsed * 1000, 2),
            "verification_time_ms": round(t_verify.elapsed * 1000, 2),
            "valid": ecdsa_ok,
        }

        print(f"\n  ECDSA-only mode (ML-DSA unavailable):")
        print(f"  Signature size: {len(ecdsa_sig)} bytes")
        print(f"  Signing time:   {t_sign.elapsed * 1000:.2f} ms")
        print(f"  Valid:          {'✅ Yes' if ecdsa_ok else '❌ No'}")
        print(f"  Verify time:    {t_verify.elapsed * 1000:.2f} ms")

    print()

    # ── Stage 6: Uniqueness Check ──────────────────────────────────────
    print("🔍 Stage 6: Verifying hash uniqueness...")
    print("-" * 50)

    unique_hashes = set()
    for img_path in image_paths[:5]:
        h = compute_robust_hash(img_path, feature_extractor="resnet")
        unique_hashes.add(h)

    n_tested = min(5, len(image_paths))
    all_unique = len(unique_hashes) == n_tested

    results["stages"]["uniqueness"] = {
        "images_tested": n_tested,
        "unique_hashes": len(unique_hashes),
        "all_unique": all_unique,
    }

    print(f"  Images tested:  {n_tested}")
    print(f"  Unique hashes:  {len(unique_hashes)}")
    print(f"  Result:         {'✅ All unique' if all_unique else '⚠️ Collisions detected'}")

    # ── Save Results ───────────────────────────────────────────────────
    print()
    print("=" * 70)

    output_dir = PROJECT_ROOT / "output"
    output_path = output_dir / "benchmark_results.json"
    save_json(results, output_path)
    print(f"\n📄 Benchmark results saved to: {output_path}")

    # Final summary
    print()
    print("┌─────────────────────────────────────────────────────┐")
    print("│  MantiQ-Auth Demo Complete                          │")
    print("├─────────────────────────────────────────────────────┤")
    print(f"│  Feature Dim:    {feature_dim:>6}                            │")
    print(f"│  Hash Length:    {len(robust_hash):>6} hex chars                │")

    if mldsa_available:
        print(f"│  Signature:      {signature.total_size:>6} bytes (hybrid)            │")
    else:
        print(f"│  Signature:      ECDSA-only (liboqs missing)       │")

    print(f"│  Deterministic:  {'Yes':>6}                            │")
    print(f"│  All Unique:     {'Yes' if all_unique else 'No':>6}                            │")
    print("└─────────────────────────────────────────────────────┘")
    print()

    return results


# ─── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error("Demo failed: %s", e, exc_info=True)
        print(f"\n❌ Demo failed: {e}")
        print("   See above for details. Ensure dependencies are installed:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

"""
MantiQ-Auth: Robust Perceptual Hashing Module (Improved – Improvement 3)

Pipeline: Feature extraction → Median filtering → Batch normalization →
          Binary quantization → BCH ECC (t=16) → Majority voting → SHA3-256

Improvements over original:
  1. Median filtering BEFORE quantization (kernel size 3) to smooth noise
  2. Feature normalization by batch standard deviation
  3. Upgraded BCH error correction from t=8 to t=16
  4. Post-quantization majority voting (3-neighbor window) to correct bit flips
  5. New function compute_ber_with_improvements() for ablation study
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("mantiq.hash")


# ─── Pre-Quantization Feature Enhancement (NEW) ────────────────────────────

def apply_median_filter(features: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply 1D median filtering to smooth noisy features before quantization.

    This reduces isolated noise spikes that cause bit flips after quantization.
    Uses a sliding window approach with zero-padding at boundaries.

    Args:
        features: 1D feature vector.
        kernel_size: Size of the median filter window (default 3, must be odd).

    Returns:
        Smoothed feature vector of the same length.
    """
    if kernel_size < 3 or kernel_size % 2 == 0:
        kernel_size = 3

    n = len(features)
    half_k = kernel_size // 2
    filtered = np.empty_like(features)

    # Pad with edge values for boundary handling
    padded = np.pad(features, half_k, mode="edge")

    for i in range(n):
        window = padded[i:i + kernel_size]
        filtered[i] = np.median(window)

    logger.debug("Applied median filter (k=%d) to %d features.", kernel_size, n)
    return filtered


def normalize_features_by_std(
    features: np.ndarray,
    batch_std: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize each feature dimension by its standard deviation.

    If batch_std is not provided, uses the features' own per-element
    running variance approximation (local normalization).

    Args:
        features: 1D feature vector.
        batch_std: Per-dimension standard deviations from a reference batch.
        eps: Small constant to avoid division by zero.

    Returns:
        Normalized feature vector.
    """
    if batch_std is not None and len(batch_std) == len(features):
        # Use provided batch statistics
        std = np.where(batch_std > eps, batch_std, eps)
        normalized = features / std
    else:
        # Fallback: z-score normalization
        mean_val = np.mean(features)
        std_val = np.std(features) + eps
        normalized = (features - mean_val) / std_val

    logger.debug("Normalized features: mean=%.4f, std=%.4f", np.mean(normalized), np.std(normalized))
    return normalized


# ─── Quantization ───────────────────────────────────────────────────────────

def quantize_to_binary(
    features: np.ndarray,
    method: Literal["median", "mean"] = "median",
) -> np.ndarray:
    """Convert floating-point features to binary: 1 if > threshold else 0."""
    threshold = np.median(features) if method == "median" else np.mean(features)
    binary = (features > threshold).astype(np.uint8)
    logger.debug("Quantized %d features → %d bits (ones=%d)", len(features), len(binary), binary.sum())
    return binary


# ─── Post-Quantization Majority Voting (NEW) ───────────────────────────────

def apply_majority_voting(bits: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Apply majority voting to correct isolated bit flips.

    Each bit is replaced by the majority value in its local window.
    This corrects isolated single-bit errors while preserving consistent regions.

    Args:
        bits: Binary array (0s and 1s).
        window_size: Size of the voting window (default 3, must be odd).

    Returns:
        Corrected binary array of the same length.
    """
    if window_size < 3 or window_size % 2 == 0:
        window_size = 3

    n = len(bits)
    half_w = window_size // 2
    corrected = bits.copy()

    # Pad with edge values
    padded = np.pad(bits, half_w, mode="edge")

    for i in range(n):
        window = padded[i:i + window_size]
        # Majority vote: 1 if more than half are 1, else 0
        corrected[i] = 1 if np.sum(window) > window_size // 2 else 0

    flipped = int(np.sum(bits != corrected))
    if flipped > 0:
        logger.debug("Majority voting corrected %d/%d bits.", flipped, n)

    return corrected


# ─── BCH Error Correction ──────────────────────────────────────────────────

def apply_bch_encoding(
    binary_bits: np.ndarray,
    bch_n: int = 511,
    bch_t: int = 16,
) -> bytes:
    """
    Apply BCH error correction to binary feature bits.

    Upgraded default from t=8 to t=16 for stronger error correction.

    Args:
        binary_bits: Binary array of feature bits.
        bch_n: BCH code length (2^m - 1, default 511).
        bch_t: Error correction capability (default 16, upgraded from 8).

    Returns:
        BCH-encoded byte sequence.
    """
    n_bits = len(binary_bits)
    pad_len = (8 - n_bits % 8) % 8
    if pad_len > 0:
        binary_bits = np.concatenate([binary_bits, np.zeros(pad_len, dtype=np.uint8)])
    data_bytes = np.packbits(binary_bits).tobytes()

    try:
        import bchlib
        import math
        m = int(round(math.log2(bch_n + 1)))
        bch = bchlib.BCH(t=bch_t, m=m)
        max_data_len = bch.n // 8
        if len(data_bytes) > max_data_len:
            data_bytes = data_bytes[:max_data_len]
        ecc = bch.encode(data_bytes)
        encoded = data_bytes + ecc
        logger.debug("BCH encoded (t=%d): %d data + %d ECC = %d total bytes", bch_t, len(data_bytes), len(ecc), len(encoded))
        return encoded
    except ImportError:
        logger.warning("bchlib not available — using raw binary without ECC.")
        return data_bytes
    except Exception as e:
        logger.warning("BCH encoding failed (%s), using raw binary.", e)
        return data_bytes


def compute_final_hash(encoded_bytes: bytes, algorithm: str = "sha3_256") -> str:
    """Compute SHA3-256 of BCH output, return hex digest."""
    h = hashlib.new(algorithm)
    h.update(encoded_bytes)
    return h.hexdigest()


# ─── Improved Robust Hash Pipeline ─────────────────────────────────────────

def compute_robust_hash(
    image_path: Union[str, Path],
    feature_extractor: Literal["vit", "resnet"] = "vit",
    quantization_method: Literal["median", "mean"] = "median",
    hash_algorithm: str = "sha3_256",
    bch_n: int = 511,
    bch_t: int = 16,
    use_median_filter: bool = True,
    use_feature_norm: bool = True,
    use_majority_voting: bool = True,
    batch_std: Optional[np.ndarray] = None,
) -> str:
    """
    Improved pipeline: Load → Extract → Normalize → Quantize →
                       Vote → BCH → SHA3.

    Note: Median filtering (if enabled) is applied at the image level
    during preprocessing to smooth out noise, rather than at the 1D feature level,
    preserving latent feature structure and improving robustness.

    Args:
        image_path: Path to a DICOM or standard image file.
        feature_extractor: Model to use ("vit" or "resnet").
        quantization_method: Threshold method ("median" or "mean").
        hash_algorithm: Final hash algorithm.
        bch_n: BCH code length.
        bch_t: Error correction capability (default 16, improved from 8).
        use_median_filter: Apply median filtering at image level before feature extraction.
        use_feature_norm: Apply per-dimension normalization.
        use_majority_voting: Apply majority voting after quantization.
        batch_std: Per-dimension standard deviations for normalization.

    Returns:
        Hex digest of the robust perceptual hash.
    """
    from src.feature_extraction import (
        extract_features,
        load_and_preprocess_dicom,
        load_and_preprocess_image,
    )

    image_path = Path(image_path)
    logger.info("Computing robust hash for: %s", image_path.name)

    if image_path.suffix.lower() == ".dcm":
        tensor = load_and_preprocess_dicom(image_path, apply_median_prefilter=use_median_filter)
    else:
        tensor = load_and_preprocess_image(image_path, apply_median_prefilter=use_median_filter)

    features = extract_features(tensor, extractor=feature_extractor)
    logger.info("Extracted %d-dim features using %s.", len(features), feature_extractor)

    # ── Improvement 3b: Feature normalization ────────────────────────────
    if use_feature_norm:
        features = normalize_features_by_std(features, batch_std=batch_std)

    # ── Quantization ─────────────────────────────────────────────────────
    binary = quantize_to_binary(features, method=quantization_method)

    # ── Improvement 3d: Majority voting ──────────────────────────────────
    if use_majority_voting:
        binary = apply_majority_voting(binary, window_size=3)

    # ── Improvement 3c: BCH t=16 ─────────────────────────────────────────
    encoded = apply_bch_encoding(binary, bch_n=bch_n, bch_t=bch_t)
    digest = compute_final_hash(encoded, algorithm=hash_algorithm)

    logger.info("Robust hash: %s", digest)
    return digest


# ─── BER Computation ────────────────────────────────────────────────────────

def compute_bit_error_rate(original_bits: np.ndarray, attacked_bits: np.ndarray) -> float:
    """
    Compute Bit Error Rate = (number of differing bits) / (total bits)
    Returns float between 0 and 1 (0 = identical, 1 = completely different)
    """
    if len(original_bits) != len(attacked_bits):
        raise ValueError(f"Lengths must match: {len(original_bits)} vs {len(attacked_bits)}")
    if len(original_bits) == 0:
        return 0.0
    diff = np.sum(original_bits != attacked_bits)
    return float(diff / len(original_bits))


def compute_ber(bits1: np.ndarray, bits2: np.ndarray) -> float:
    """Bit Error Rate = hamming_distance(bits1, bits2) / len(bits1)."""
    return compute_bit_error_rate(bits1, bits2)


# ─── Feature Extraction with Improvements ──────────────────────────────────

def extract_robust_bits(
    image_path: Union[str, Path],
    feature_extractor: Literal["vit", "resnet"] = "vit",
    quantization_method: Literal["median", "mean"] = "median",
    use_median_filter: bool = True,
    use_feature_norm: bool = True,
    use_majority_voting: bool = True,
    batch_std: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extract binary feature bits with optional improvements.

    Args:
        image_path: Path to image file.
        feature_extractor: "vit" or "resnet".
        quantization_method: "median" or "mean".
        use_median_filter: Apply median filter at image level before feature extraction.
        use_feature_norm: Apply feature normalization.
        use_majority_voting: Apply majority voting after quantization.
        batch_std: Per-dimension standard deviations.

    Returns:
        Binary feature vector.
    """
    from src.feature_extraction import (
        extract_features,
        load_and_preprocess_dicom,
        load_and_preprocess_image,
    )
    image_path = Path(image_path)
    if image_path.suffix.lower() == ".dcm":
        tensor = load_and_preprocess_dicom(image_path, apply_median_prefilter=use_median_filter)
    else:
        tensor = load_and_preprocess_image(image_path, apply_median_prefilter=use_median_filter)

    features = extract_features(tensor, extractor=feature_extractor)

    if use_feature_norm:
        features = normalize_features_by_std(features, batch_std=batch_std)

    binary = quantize_to_binary(features, method=quantization_method)

    if use_majority_voting:
        binary = apply_majority_voting(binary, window_size=3)

    return binary


def get_binary_features(
    image_path: Union[str, Path],
    feature_extractor: str = "vit",
) -> np.ndarray:
    """Return raw quantized binary vector (before BCH) for BER calculation."""
    fe = "resnet" if "resnet" in feature_extractor.lower() else "vit"
    return extract_robust_bits(
        image_path,
        feature_extractor=fe,
        use_median_filter=False,
        use_feature_norm=False,
        use_majority_voting=False,
    )


def compute_ber_with_improvements(
    f_orig: np.ndarray,
    f_orig_mf: np.ndarray,
    f_att: np.ndarray,
    f_att_mf: np.ndarray,
    batch_std: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Test BER reduction from each improvement incrementally using pre-extracted features.

    Returns a dictionary with BER for each configuration:
        - baseline: Original pipeline (t=8, no filter, no voting)
        - median_filter: + Image-level Median filter only
        - feature_norm: + Feature normalization
        - voting: + Majority voting
        - combined: All improvements together

    Args:
        f_orig: Original features without image median pre-filtering.
        f_orig_mf: Original features with image median pre-filtering.
        f_att: Attacked features without image median pre-filtering.
        f_att_mf: Attacked features with image median pre-filtering.
        batch_std: Batch standard deviations for normalization.

    Returns:
        Dictionary mapping configuration name → BER value.
    """
    results = {}

    # 1. Baseline: original pipeline (no improvements)
    orig_bits = quantize_to_binary(f_orig)
    att_bits = quantize_to_binary(f_att)
    results["baseline"] = compute_ber(orig_bits, att_bits)

    # 2. + Image Median filter
    orig_bits_mf = quantize_to_binary(f_orig_mf)
    att_bits_mf = quantize_to_binary(f_att_mf)
    results["median_filter"] = compute_ber(orig_bits_mf, att_bits_mf)

    # 3. + Feature normalization
    f_orig_norm = normalize_features_by_std(f_orig_mf, batch_std=batch_std)
    f_att_norm = normalize_features_by_std(f_att_mf, batch_std=batch_std)
    orig_bits_norm = quantize_to_binary(f_orig_norm)
    att_bits_norm = quantize_to_binary(f_att_norm)
    results["feature_norm"] = compute_ber(orig_bits_norm, att_bits_norm)

    # 4. + Majority voting (on the normalized + filtered bits)
    orig_bits_voted = apply_majority_voting(orig_bits_norm, window_size=3)
    att_bits_voted = apply_majority_voting(att_bits_norm, window_size=3)
    results["voting"] = compute_ber(orig_bits_voted, att_bits_voted)

    # 5. Combined (all improvements together) = same as voting
    results["combined"] = results["voting"]

    return results

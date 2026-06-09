"""
MantiQ-Auth: Robust Perceptual Hashing Tests

Validates:
    1. Binary quantization (median and mean thresholds)
    2. BCH error-correcting code encoding (graceful fallback)
    3. Final SHA3-256 hash generation
    4. Complete robust hash pipeline determinism
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.robust_hash import (
    quantize_to_binary,
    apply_bch_encoding,
    compute_final_hash,
    compute_robust_hash,
)


@pytest.fixture
def sample_features() -> np.ndarray:
    """Create a sample feature vector."""
    # 10 values, median is 4.5
    return np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])


def test_quantize_to_binary_median(sample_features):
    """Verify median-based binary quantization."""
    binary = quantize_to_binary(sample_features, method="median")
    # Features > median (4.5) should be 1, rest 0
    # Expected: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    expected = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.uint8)
    np.testing.assert_array_equal(binary, expected)


def test_quantize_to_binary_mean(sample_features):
    """Verify mean-based binary quantization."""
    binary = quantize_to_binary(sample_features, method="mean")
    # Mean is 4.5. Expected is same as median in this case.
    expected = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.uint8)
    np.testing.assert_array_equal(binary, expected)


def test_apply_bch_encoding():
    """Verify BCH encoding completes without exception."""
    binary_bits = np.random.randint(0, 2, 384, dtype=np.uint8)
    encoded = apply_bch_encoding(binary_bits)
    assert isinstance(encoded, bytes)
    assert len(encoded) > 0


def test_compute_final_hash():
    """Verify final SHA3-256 hash formatting."""
    data = b"mantiq-auth-test-data"
    digest = compute_final_hash(data, algorithm="sha3_256")
    assert isinstance(digest, str)
    assert len(digest) == 64  # SHA3-256 hex length
    # Check that it matches expected SHA3-256 digest
    import hashlib
    h = hashlib.new("sha3_256")
    h.update(data)
    assert digest == h.hexdigest()


def test_compute_robust_hash_determinism(tmp_path):
    """Verify that robust hash pipeline is deterministic."""
    # Create mock image
    from PIL import Image
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img_path = tmp_path / "mock_ct.png"
    img.save(img_path)

    # Compute hash twice
    hash1 = compute_robust_hash(img_path, feature_extractor="resnet")
    hash2 = compute_robust_hash(img_path, feature_extractor="resnet")

    assert hash1 == hash2
    assert len(hash1) == 64


def test_compute_bit_error_rate():
    """Verify compute_bit_error_rate logic."""
    from src.robust_hash import compute_bit_error_rate
    a = np.array([0, 1, 0, 1, 0, 1])
    b = np.array([0, 1, 0, 1, 0, 1])
    c = np.array([1, 0, 1, 0, 1, 0])
    d = np.array([0, 1, 0, 0, 0, 1])

    assert compute_bit_error_rate(a, b) == 0.0
    assert compute_bit_error_rate(a, c) == 1.0
    assert compute_bit_error_rate(a, d) == pytest.approx(1.0 / 6.0)

    with pytest.raises(ValueError):
        compute_bit_error_rate(a, np.array([0, 1]))


def test_extract_robust_bits(tmp_path):
    """Verify extract_robust_bits returns expected shapes and values."""
    from src.robust_hash import extract_robust_bits
    from PIL import Image
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img_path = tmp_path / "mock_ct_bits.png"
    img.save(img_path)

    bits = extract_robust_bits(img_path, feature_extractor="resnet")
    assert isinstance(bits, np.ndarray)
    assert bits.ndim == 1
    assert len(bits) == 512
    assert set(bits).issubset({0, 1})


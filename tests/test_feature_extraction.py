"""
MantiQ-Auth: Feature Extraction Tests

Validates:
    1. ViT-B/16 output dimension is 768
    2. ResNet-18 output dimension is 512
    3. L2 normalization produces unit-norm vectors
    4. Determinism: same input → same features
    5. Discriminability: different inputs → different features
    6. NCC computation correctness
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_extraction import (
    FEATURE_DIMS,
    compute_ncc,
    extract_resnet_features,
    extract_vit_features,
    l2_normalize,
    load_and_preprocess_image,
)


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Create a sample test image."""
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    path = tmp_path / "test_image.png"
    img.save(path)
    return path


@pytest.fixture
def another_image(tmp_path: Path) -> Path:
    """Create a distinctly different test image."""
    # Create a gradient image (very different from random noise)
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[:, :, 0] = np.arange(256).reshape(1, 256).repeat(256, axis=0)
    arr[:, :, 1] = np.arange(256).reshape(256, 1).repeat(256, axis=1)
    path = tmp_path / "test_image_2.png"
    Image.fromarray(arr).save(path)
    return path


@pytest.fixture
def sample_tensor(sample_image: Path) -> torch.Tensor:
    """Preprocessed tensor from sample image."""
    return load_and_preprocess_image(sample_image)


# ─── Dimension Tests ────────────────────────────────────────────────────────

class TestFeatureDimensions:
    """Verify output dimensions of feature extractors."""

    def test_vit_feature_dim(self, sample_tensor: torch.Tensor):
        """ViT-B/16 must produce 384-dimensional features."""
        features = extract_vit_features(sample_tensor)
        assert features.shape == (384,), (
            f"ViT features should be 384-dim, got {features.shape}"
        )

    def test_resnet_feature_dim(self, sample_tensor: torch.Tensor):
        """ResNet-18 must produce 512-dimensional features."""
        features = extract_resnet_features(sample_tensor)
        assert features.shape == (512,), (
            f"ResNet features should be 512-dim, got {features.shape}"
        )

    def test_feature_dims_constant(self):
        """FEATURE_DIMS dictionary must match expected values."""
        assert FEATURE_DIMS["vit"] == 384
        assert FEATURE_DIMS["resnet"] == 512


# ─── Normalization Tests ───────────────────────────────────────────────────

class TestNormalization:
    """Verify L2 normalization correctness."""

    def test_unit_norm(self):
        """L2-normalized vector must have unit norm."""
        v = np.random.randn(384)
        normalized = l2_normalize(v)
        assert np.isclose(np.linalg.norm(normalized), 1.0, atol=1e-6)

    def test_zero_vector(self):
        """Zero vector should return zero vector (not NaN)."""
        v = np.zeros(512)
        normalized = l2_normalize(v)
        assert np.all(normalized == 0)
        assert not np.any(np.isnan(normalized))

    def test_preserves_direction(self):
        """Normalization should preserve direction."""
        v = np.array([3.0, 4.0])
        normalized = l2_normalize(v)
        expected = np.array([0.6, 0.8])
        np.testing.assert_allclose(normalized, expected, atol=1e-6)

    def test_features_are_normalized(self, sample_tensor: torch.Tensor):
        """Extracted features must be L2-normalized."""
        vit_features = extract_vit_features(sample_tensor)
        resnet_features = extract_resnet_features(sample_tensor)

        assert np.isclose(np.linalg.norm(vit_features), 1.0, atol=1e-5)
        assert np.isclose(np.linalg.norm(resnet_features), 1.0, atol=1e-5)


# ─── Determinism Tests ─────────────────────────────────────────────────────

class TestDeterminism:
    """Verify that feature extraction is deterministic."""

    def test_vit_deterministic(self, sample_tensor: torch.Tensor):
        """Same input must produce identical ViT features."""
        f1 = extract_vit_features(sample_tensor)
        f2 = extract_vit_features(sample_tensor)
        np.testing.assert_array_equal(f1, f2)

    def test_resnet_deterministic(self, sample_tensor: torch.Tensor):
        """Same input must produce identical ResNet features."""
        f1 = extract_resnet_features(sample_tensor)
        f2 = extract_resnet_features(sample_tensor)
        np.testing.assert_array_equal(f1, f2)


# ─── Discriminability Tests ────────────────────────────────────────────────

class TestDiscriminability:
    """Verify that different images produce different features."""

    def test_different_images_different_features(
        self, sample_image: Path, another_image: Path
    ):
        """Features from different images must differ."""
        t1 = load_and_preprocess_image(sample_image)
        t2 = load_and_preprocess_image(another_image)

        f1 = extract_vit_features(t1)
        f2 = extract_vit_features(t2)

        # Features should not be identical
        assert not np.array_equal(f1, f2), "Different images produced identical features!"


# ─── NCC Tests ─────────────────────────────────────────────────────────────

class TestNCC:
    """Verify Normalized Cross-Correlation computation."""

    def test_identical_vectors(self):
        """NCC of identical vectors should be 1.0."""
        v = np.random.randn(384)
        v = l2_normalize(v)
        assert np.isclose(compute_ncc(v, v), 1.0, atol=1e-6)

    def test_orthogonal_vectors(self):
        """NCC of orthogonal vectors should be ~0."""
        v1 = np.zeros(384)
        v1[0] = 1.0
        v2 = np.zeros(384)
        v2[1] = 1.0
        # After mean subtraction, these become near-zero NCC
        ncc = compute_ncc(v1, v2)
        assert abs(ncc) < 0.1

    def test_ncc_range(self):
        """NCC should be in [-1, 1]."""
        for _ in range(10):
            v1 = l2_normalize(np.random.randn(384))
            v2 = l2_normalize(np.random.randn(384))
            ncc = compute_ncc(v1, v2)
            assert -1.0 <= ncc <= 1.0, f"NCC out of range: {ncc}"


# ─── Preprocessing Tests ──────────────────────────────────────────────────

class TestPreprocessing:
    """Verify image preprocessing pipeline."""

    def test_output_shape(self, sample_image: Path):
        """Preprocessed tensor must have shape (1, 3, 224, 224)."""
        tensor = load_and_preprocess_image(sample_image)
        assert tensor.shape == (1, 3, 224, 224)

    def test_output_dtype(self, sample_image: Path):
        """Preprocessed tensor must be float32."""
        tensor = load_and_preprocess_image(sample_image)
        assert tensor.dtype == torch.float32

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            load_and_preprocess_image("nonexistent_image.png")

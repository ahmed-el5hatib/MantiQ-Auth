"""
MantiQ-Auth: Crypto Gateway Tests

Validates:
    1. Gateway initialization and configuration
    2. Key generation state management
    3. Authentication of images (producing robust hash & hybrid signatures)
    4. Verification of images
    5. Serialization summaries
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.crypto_gateway import CryptoGateway, AuthenticationResult


@pytest.fixture
def mock_image_path(tmp_path: Path) -> Path:
    """Create a mock image to test the gateway."""
    from PIL import Image
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    path = tmp_path / "gateway_test_img.png"
    img.save(path)
    return path


def test_gateway_initialization():
    """Verify that parameters are correctly set on initialization."""
    gateway = CryptoGateway(
        feature_extractor="resnet",
        mldsa_level=65,
        ecdsa_curve="SECP256R1",
        combiner="concatenation"
    )
    assert gateway.feature_extractor == "resnet"
    assert gateway.mldsa_level == 65
    assert gateway.ecdsa_curve == "SECP256R1"
    assert gateway.combiner == "concatenation"

    # Accessing keys before generation should raise RuntimeError
    with pytest.raises(RuntimeError):
        _ = gateway.ecdsa_keypair

    with pytest.raises(RuntimeError):
        _ = gateway.mldsa_keypair


def test_gateway_keypair_generation():
    """Verify key pair generation works and updates gateway state."""
    try:
        # noinspection PyPackageRequirements
        import oqs
    except ImportError:
        pytest.skip("liboqs-python not available, cannot run keypair generation test.")

    gateway = CryptoGateway()
    gateway.generate_keys()

    assert gateway.ecdsa_keypair is not None
    assert gateway.mldsa_keypair is not None


def test_gateway_authenticate_and_verify(mock_image_path):
    """Test full authenticate-and-verify flow via the gateway."""
    try:
        # noinspection PyPackageRequirements
        import oqs
    except ImportError:
        pytest.skip("liboqs-python not available, cannot run full authentication test.")

    gateway = CryptoGateway(feature_extractor="resnet")
    gateway.generate_keys()

    # Authenticate
    result = gateway.authenticate_image(mock_image_path)
    assert isinstance(result, AuthenticationResult)
    assert result.robust_hash is not None
    assert len(result.robust_hash) == 64
    assert result.feature_dim == 512

    # Verify summary serialization
    summary = result.summary()
    assert summary["image"] == str(mock_image_path)
    assert summary["hash"] == result.robust_hash
    assert "timing_ms" in summary

    # Verify
    overall, hash_match, ecdsa_ok, mldsa_ok = gateway.verify_image(
        mock_image_path,
        expected_hash=result.robust_hash,
        signature=result.signature
    )
    assert overall is True
    assert hash_match is True
    assert ecdsa_ok is True
    assert mldsa_ok is True

    # Tampered verification (using modified hash)
    bad_hash = "f" * 64
    overall_bad, hash_match_bad, _, _ = gateway.verify_image(
        mock_image_path,
        expected_hash=bad_hash,
        signature=result.signature
    )
    assert overall_bad is False
    assert hash_match_bad is False


def test_gateway_authenticate_and_verify_silithium(mock_image_path):
    """Test full authenticate-and-verify flow with silithium combiner."""
    try:
        # noinspection PyPackageRequirements
        import oqs
    except ImportError:
        pytest.skip("liboqs-python not available, cannot run full authentication test.")

    gateway = CryptoGateway(feature_extractor="resnet", combiner="silithium")
    gateway.generate_keys()

    # Authenticate
    result = gateway.authenticate_image(mock_image_path)
    assert isinstance(result, AuthenticationResult)
    assert result.robust_hash is not None
    assert len(result.robust_hash) == 64
    assert result.signature.combiner == "silithium"
    # Expected size: 8 + ecdsa_size + mldsa_size + 32
    assert result.signature.total_size == 8 + result.signature.ecdsa_size + result.signature.mldsa_size + 32

    # Verify summary serialization
    summary = result.summary()
    assert summary["image"] == str(mock_image_path)
    assert summary["hash"] == result.robust_hash
    assert summary["signature_size_bytes"] == result.signature.total_size

    # Verify
    overall, hash_match, ecdsa_ok, mldsa_ok = gateway.verify_image(
        mock_image_path,
        expected_hash=result.robust_hash,
        signature=result.signature
    )
    assert overall is True
    assert hash_match is True
    assert ecdsa_ok is True
    assert mldsa_ok is True


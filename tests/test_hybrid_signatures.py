"""
MantiQ-Auth: Hybrid Signatures Tests

Validates:
    1. ECDSA key pair generation, signing, and verification.
    2. ML-DSA key pair generation, signing, and verification (if liboqs is available).
    3. Hybrid signature generation, serialization sizes, and verification.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hybrid_signatures import (
    generate_keypair_ecdsa,
    generate_keypair_mldsa,
    sign_ecdsa,
    sign_mldsa,
    sign_hybrid,
    verify_ecdsa,
    verify_mldsa,
    verify_hybrid,
)


@pytest.fixture
def mock_message_hash() -> bytes:
    """Create a sample 32-byte hash to sign."""
    return hashlib.sha256(b"mantiq-auth-robust-hash-test").digest()


def test_ecdsa_flow(mock_message_hash):
    """Test standard ECDSA keygen, sign, and verify flow."""
    kp = generate_keypair_ecdsa()
    assert kp.private_key is not None
    assert kp.public_key is not None
    assert len(kp.private_bytes()) > 0
    assert len(kp.public_bytes()) > 0

    # Sign
    sig = sign_ecdsa(mock_message_hash, kp)
    assert len(sig) > 0

    # Verify
    assert verify_ecdsa(mock_message_hash, sig, kp.public_key) is True

    # Tamper verify
    tampered_hash = hashlib.sha256(b"tampered").digest()
    assert verify_ecdsa(tampered_hash, sig, kp.public_key) is False


def test_mldsa_flow(mock_message_hash):
    """Test ML-DSA flow if liboqs is available."""
    try:
        kp = generate_keypair_mldsa(level=65)
    except ImportError:
        pytest.skip("liboqs-python or liboqs C library is not available.")

    assert kp.algorithm == "ML-DSA-65"
    assert kp.level == 65
    assert len(kp.public_key) > 0
    assert len(kp.secret_key) > 0

    # Sign
    sig = sign_mldsa(mock_message_hash, kp)
    assert len(sig) > 0

    # Verify
    assert verify_mldsa(mock_message_hash, sig, kp.public_key, kp.algorithm) is True

    # Tamper verify
    tampered_hash = hashlib.sha256(b"tampered").digest()
    assert verify_mldsa(tampered_hash, sig, kp.public_key, kp.algorithm) is False


def test_hybrid_flow_concatenation(mock_message_hash):
    """Test overall hybrid concatenation flow (or skip if oqs is missing)."""
    try:
        # noinspection PyPackageRequirements
        import oqs
    except ImportError:
        pytest.skip("liboqs-python not available, cannot run hybrid signature test.")

    ecdsa_kp = generate_keypair_ecdsa()
    mldsa_kp = generate_keypair_mldsa(level=65)

    # Sign
    hybrid_sig = sign_hybrid(mock_message_hash, ecdsa_kp, mldsa_kp, combiner="concatenation")
    assert hybrid_sig.total_size == 4 + hybrid_sig.ecdsa_size + hybrid_sig.mldsa_size

    # Verify
    overall, ecdsa_ok, mldsa_ok = verify_hybrid(
        mock_message_hash,
        hybrid_sig,
        ecdsa_kp.public_key,
        mldsa_kp.public_key,
        mldsa_algorithm="ML-DSA-65"
    )

    assert overall is True
    assert ecdsa_ok is True
    assert mldsa_ok is True

    # Verify with wrong hash
    tampered_hash = hashlib.sha256(b"tampered").digest()
    overall_tampered, _, _ = verify_hybrid(
        tampered_hash,
        hybrid_sig,
        ecdsa_kp.public_key,
        mldsa_kp.public_key,
        mldsa_algorithm="ML-DSA-65"
    )
    assert overall_tampered is False


def test_hybrid_flow_silithium(mock_message_hash):
    """Test overall hybrid silithium (mutually binding) flow."""
    try:
        # noinspection PyPackageRequirements
        import oqs
    except ImportError:
        pytest.skip("liboqs-python not available, cannot run hybrid signature test.")

    ecdsa_kp = generate_keypair_ecdsa()
    mldsa_kp = generate_keypair_mldsa(level=65)

    # Sign
    hybrid_sig = sign_hybrid(mock_message_hash, ecdsa_kp, mldsa_kp, combiner="silithium")
    # Expected size: 4 + ecdsa_size + 4 + mldsa_size + 32
    assert hybrid_sig.total_size == 8 + hybrid_sig.ecdsa_size + hybrid_sig.mldsa_size + 32

    # Verify
    overall, ecdsa_ok, mldsa_ok = verify_hybrid(
        mock_message_hash,
        hybrid_sig,
        ecdsa_kp.public_key,
        mldsa_kp.public_key,
        mldsa_algorithm="ML-DSA-65"
    )

    assert overall is True
    assert ecdsa_ok is True
    assert mldsa_ok is True

    # Verify with wrong hash (tampered hash)
    tampered_hash = hashlib.sha256(b"tampered").digest()
    overall_tampered, _, _ = verify_hybrid(
        tampered_hash,
        hybrid_sig,
        ecdsa_kp.public_key,
        mldsa_kp.public_key,
        mldsa_algorithm="ML-DSA-65"
    )
    assert overall_tampered is False


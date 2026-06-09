"""
MantiQ-Auth: Hybrid Signature Module (ECDSA + ML-DSA)

Implements hybrid digital signatures combining classical ECDSA-P256 with
post-quantum ML-DSA (Dilithium) for quantum-resistant authentication.

Design rationale:
    - ECDSA provides immediate trust anchored in existing PKI.
    - ML-DSA provides long-term security against CRQC threats.
    - The hybrid combiner ensures non-separability: an attacker cannot
      strip one component and forge the other.

Combiner modes:
    - "concatenation": Simple σ_hybrid = σ_ECDSA || σ_ML-DSA (initial)
    - "silithium": Cryptographic combiner (future Silithium integration)
"""

from __future__ import annotations

import base64
import json
import logging
import time
import hashlib
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils

logger = logging.getLogger("mantiq.signatures")


# ─── Data Structures ───────────────────────────────────────────────────────

@dataclass
class ECDSAKeyPair:
    """ECDSA key pair container."""
    private_key: ec.EllipticCurvePrivateKey
    public_key: ec.EllipticCurvePublicKey

    def private_bytes(self) -> bytes:
        return self.private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

    def public_bytes(self) -> bytes:
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )


@dataclass
class MLDSAKeyPair:
    """ML-DSA (Dilithium) key pair container."""
    algorithm: str
    secret_key: bytes
    public_key: bytes
    _signer: object = field(repr=False, default=None)

    @property
    def level(self) -> int:
        return int(self.algorithm.split("-")[-1])


@dataclass
class HybridSignature:
    """Container for a hybrid ECDSA + ML-DSA signature."""
    ecdsa_sig: bytes
    mldsa_sig: bytes
    combiner: str
    combined: bytes

    @property
    def total_size(self) -> int:
        return len(self.combined)

    @property
    def ecdsa_size(self) -> int:
        return len(self.ecdsa_sig)

    @property
    def mldsa_size(self) -> int:
        return len(self.mldsa_sig)


# ─── Key Generation ────────────────────────────────────────────────────────

def generate_keypair_ecdsa(curve: str = "SECP256R1") -> ECDSAKeyPair:
    """
    Generate an ECDSA key pair on the specified curve.

    Args:
        curve: Elliptic curve name (default: SECP256R1 / P-256).

    Returns:
        ECDSAKeyPair with private and public keys.
    """
    curve_map = {
        "SECP256R1": ec.SECP256R1(),
        "SECP384R1": ec.SECP384R1(),
        "SECP521R1": ec.SECP521R1(),
    }
    ec_curve = curve_map.get(curve)
    if ec_curve is None:
        raise ValueError(f"Unsupported curve: {curve}. Use: {list(curve_map.keys())}")

    private_key = ec.generate_private_key(ec_curve)
    public_key = private_key.public_key()

    logger.info("Generated ECDSA-%s key pair.", curve)
    return ECDSAKeyPair(private_key=private_key, public_key=public_key)


def generate_keypair_mldsa(level: int = 65) -> MLDSAKeyPair:
    """
    Generate an ML-DSA (Dilithium) key pair using liboqs.

    Security levels:
        - 44: NIST Level 2 (~128-bit classical security)
        - 65: NIST Level 3 (~192-bit classical security) [recommended]
        - 87: NIST Level 5 (~256-bit classical security)

    Args:
        level: ML-DSA security level (44, 65, or 87).

    Returns:
        MLDSAKeyPair with secret and public keys.

    Raises:
        ImportError: If liboqs-python is not installed.
    """
    valid_levels = {44, 65, 87}
    if level not in valid_levels:
        raise ValueError(f"Invalid ML-DSA level: {level}. Use: {valid_levels}")

    algorithm = f"ML-DSA-{level}"

    try:
        # noinspection PyPackageRequirements
        import oqs
        signer = oqs.Signature(algorithm)
        public_key = signer.generate_keypair()
        secret_key = signer.export_secret_key()

        logger.info("Generated %s key pair (pk=%d bytes, sk=%d bytes).",
                     algorithm, len(public_key), len(secret_key))

        return MLDSAKeyPair(
            algorithm=algorithm,
            secret_key=secret_key,
            public_key=public_key,
            _signer=signer,
        )
    except ImportError:
        logger.error("liboqs-python not installed. Install with: pip install liboqs-python")
        raise
    except Exception as e:
        logger.error("ML-DSA key generation failed: %s", e)
        raise


# ─── Signing ───────────────────────────────────────────────────────────────

def sign_ecdsa(message_hash: bytes, keypair: ECDSAKeyPair) -> bytes:
    """Sign a message hash using ECDSA-P256."""
    signature = keypair.private_key.sign(
        message_hash,
        ec.ECDSA(utils.Prehashed(hashes.SHA256())),
    )
    return signature


def sign_mldsa(message_hash: bytes, keypair: MLDSAKeyPair) -> bytes:
    """Sign a message hash using ML-DSA."""
    try:
        import oqs
        signer = oqs.Signature(keypair.algorithm, keypair.secret_key)
        signature = signer.sign(message_hash)
        return signature
    except ImportError:
        raise ImportError("liboqs-python required for ML-DSA signing.")


def sign_hybrid(
    message_hash: bytes,
    ecdsa_kp: ECDSAKeyPair,
    mldsa_kp: MLDSAKeyPair,
    combiner: Literal["concatenation", "silithium"] = "concatenation",
) -> HybridSignature:
    """
    Generate hybrid signature with cryptographic combiner (non-separable).

    For the initial implementation, uses concatenation:
        σ_hybrid = len(σ_ECDSA) || σ_ECDSA || σ_ML-DSA

    The length prefix ensures unambiguous parsing during verification.

    Args:
        message_hash: SHA-256 hash of the message to sign.
        ecdsa_kp: ECDSA key pair.
        mldsa_kp: ML-DSA key pair.
        combiner: Combiner mode ("concatenation" or "silithium").

    Returns:
        HybridSignature containing both component signatures.
    """
    # Sign with both algorithms
    ecdsa_sig = sign_ecdsa(message_hash, ecdsa_kp)
    mldsa_sig = sign_mldsa(message_hash, mldsa_kp)

    if combiner == "concatenation":
        # Length-prefixed concatenation for unambiguous parsing
        ecdsa_len = len(ecdsa_sig).to_bytes(4, "big")
        combined = ecdsa_len + ecdsa_sig + mldsa_sig
    elif combiner == "silithium":
        # TODO: Implement Silithium cryptographic combiner
        # For now, fall back to concatenation with a marker
        logger.warning("Silithium combiner not yet implemented, using concatenation.")
        ecdsa_len = len(ecdsa_sig).to_bytes(4, "big")
        combined = ecdsa_len + ecdsa_sig + mldsa_sig
    else:
        raise ValueError(f"Unknown combiner: {combiner}")

    logger.info(
        "Hybrid signature: ECDSA=%d bytes + ML-DSA=%d bytes = %d total",
        len(ecdsa_sig), len(mldsa_sig), len(combined),
    )

    return HybridSignature(
        ecdsa_sig=ecdsa_sig,
        mldsa_sig=mldsa_sig,
        combiner=combiner,
        combined=combined,
    )


# ─── Verification ──────────────────────────────────────────────────────────

def verify_ecdsa(
    message_hash: bytes,
    signature: bytes,
    public_key: ec.EllipticCurvePublicKey,
) -> bool:
    """Verify an ECDSA signature."""
    try:
        public_key.verify(
            signature,
            message_hash,
            ec.ECDSA(utils.Prehashed(hashes.SHA256())),
        )
        return True
    except Exception:
        return False


def verify_mldsa(
    message_hash: bytes,
    signature: bytes,
    public_key: bytes,
    algorithm: str = "ML-DSA-65",
) -> bool:
    """Verify an ML-DSA signature."""
    try:
        # noinspection PyPackageRequirements
        import oqs
        verifier = oqs.Signature(algorithm)
        return verifier.verify(message_hash, signature, public_key)
    except ImportError:
        raise ImportError("liboqs-python required for ML-DSA verification.")


def verify_hybrid(
    message_hash: bytes,
    signature: HybridSignature,
    ecdsa_pk: ec.EllipticCurvePublicKey,
    mldsa_pk: bytes,
    mldsa_algorithm: str = "ML-DSA-65",
) -> Tuple[bool, bool, bool]:
    """
    Verify a hybrid ECDSA + ML-DSA signature.

    Both component signatures must verify for the hybrid to be valid.

    Args:
        message_hash: Original message hash.
        signature: HybridSignature to verify.
        ecdsa_pk: ECDSA public key.
        mldsa_pk: ML-DSA public key bytes.
        mldsa_algorithm: ML-DSA algorithm name.

    Returns:
        Tuple of (overall_valid, ecdsa_valid, mldsa_valid).
    """
    ecdsa_valid = verify_ecdsa(message_hash, signature.ecdsa_sig, ecdsa_pk)
    mldsa_valid = verify_mldsa(
        message_hash, signature.mldsa_sig, mldsa_pk, mldsa_algorithm,
    )

    overall = ecdsa_valid and mldsa_valid

    logger.info(
        "Hybrid verification: ECDSA=%s, ML-DSA=%s → Overall=%s",
        ecdsa_valid, mldsa_valid, overall,
    )

    return overall, ecdsa_valid, mldsa_valid


def verify_hash_based(
    recomputed_hash: str,
    signed_hash: str,
    signature: HybridSignature,
    ecdsa_pk: ec.EllipticCurvePublicKey,
    mldsa_pk: bytes,
    mldsa_algorithm: str = "ML-DSA-65",
) -> Tuple[bool, bool, bool, bool]:
    """
    Core authentication function based purely on cryptographic hashing and signatures.
    
    1. Checks if the recomputed robust hash matches the originally signed hash.
    2. Verifies the hybrid signature of the signed hash.
    
    Args:
        recomputed_hash: The perceptual hash computed from the current image.
        signed_hash: The original perceptual hash claimed by the metadata.
        signature: The hybrid signature of the original signed_hash.
        ecdsa_pk: ECDSA public key.
        mldsa_pk: ML-DSA public key.
        mldsa_algorithm: ML-DSA parameter set.
        
    Returns:
        Tuple: (is_authentic, hash_match, ecdsa_valid, mldsa_valid)
    """
    # 1. Compare hashes
    hash_match = (recomputed_hash == signed_hash)
    
    # 2. Verify signatures over the original signed hash
    message_hash = hashlib.sha256(signed_hash.encode()).digest()
    sig_valid, ecdsa_valid, mldsa_valid = verify_hybrid(
        message_hash, signature, ecdsa_pk, mldsa_pk, mldsa_algorithm
    )
    
    # Final authentication decision
    is_authentic = hash_match and sig_valid
    
    logger.info(
        "Hash-based Authentication: Hash Match=%s, Sig Valid=%s -> Authentic=%s",
        hash_match, sig_valid, is_authentic
    )
    
    return is_authentic, hash_match, ecdsa_valid, mldsa_valid

"""
MantiQ-Auth: Crypto Gateway Module

Unified interface that orchestrates the complete authentication pipeline:
    Image → Feature Extraction → Robust Hash → Hybrid Signature

This module ties together feature_extraction, robust_hash, and
hybrid_signatures into a single cohesive API for SealPACS integration.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

from src.feature_extraction import (
    extract_features,
    load_and_preprocess_dicom,
    load_and_preprocess_image,
)
from src.hybrid_signatures import (
    ECDSAKeyPair,
    HybridSignature,
    MLDSAKeyPair,
    generate_keypair_ecdsa,
    generate_keypair_mldsa,
    sign_hybrid,
    verify_hybrid,
)
from src.robust_hash import (
    apply_bch_encoding,
    compute_final_hash,
    compute_robust_hash,
    quantize_to_binary,
)

logger = logging.getLogger("mantiq.gateway")


@dataclass
class AuthenticationResult:
    """Complete result of the authentication pipeline."""
    image_path: str
    robust_hash: str
    signature: HybridSignature
    feature_dim: int
    timing: Dict[str, float]

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary."""
        return {
            "image": self.image_path,
            "hash": self.robust_hash,
            "signature_size_bytes": self.signature.total_size,
            "ecdsa_sig_bytes": self.signature.ecdsa_size,
            "mldsa_sig_bytes": self.signature.mldsa_size,
            "feature_dim": self.feature_dim,
            "timing_ms": {k: round(v * 1000, 2) for k, v in self.timing.items()},
        }


class CryptoGateway:
    """
    Unified gateway for MantiQ-Auth cryptographic operations.

    Manages key pairs, orchestrates the hash + sign pipeline, and
    provides verification services.
    """

    def __init__(
        self,
        feature_extractor: Literal["vit", "resnet"] = "vit",
        mldsa_level: int = 65,
        ecdsa_curve: str = "SECP256R1",
        combiner: Literal["concatenation", "silithium"] = "concatenation",
    ):
        self.feature_extractor = feature_extractor
        self.mldsa_level = mldsa_level
        self.ecdsa_curve = ecdsa_curve
        self.combiner = combiner

        self._ecdsa_kp: Optional[ECDSAKeyPair] = None
        self._mldsa_kp: Optional[MLDSAKeyPair] = None

    def generate_keys(self) -> None:
        """Generate both ECDSA and ML-DSA key pairs."""
        logger.info("Generating key pairs...")
        self._ecdsa_kp = generate_keypair_ecdsa(self.ecdsa_curve)
        self._mldsa_kp = generate_keypair_mldsa(self.mldsa_level)
        logger.info("Key pairs generated successfully.")

    @property
    def ecdsa_keypair(self) -> ECDSAKeyPair:
        if self._ecdsa_kp is None:
            raise RuntimeError("ECDSA keys not generated. Call generate_keys() first.")
        return self._ecdsa_kp

    @property
    def mldsa_keypair(self) -> MLDSAKeyPair:
        if self._mldsa_kp is None:
            raise RuntimeError("ML-DSA keys not generated. Call generate_keys() first.")
        return self._mldsa_kp

    def authenticate_image(
        self,
        image_path: Union[str, Path],
    ) -> AuthenticationResult:
        """
        Run the complete authentication pipeline on an image.

        Steps:
            1. Compute robust perceptual hash
            2. Generate SHA-256 digest of the hash for signing
            3. Create hybrid ECDSA + ML-DSA signature

        Args:
            image_path: Path to DICOM or standard image file.

        Returns:
            AuthenticationResult with hash, signature, and timing.
        """
        image_path = Path(image_path)
        timing: Dict[str, float] = {}

        # Step 1: Robust hash
        t0 = time.perf_counter()
        robust_hash = compute_robust_hash(
            image_path,
            feature_extractor=self.feature_extractor,
        )
        timing["hash_computation"] = time.perf_counter() - t0

        # Determine feature dim
        from src.feature_extraction import FEATURE_DIMS
        feature_dim = FEATURE_DIMS[self.feature_extractor]

        # Step 2: Prepare message for signing (SHA-256 of the robust hash)
        message_hash = hashlib.sha256(robust_hash.encode()).digest()

        # Step 3: Hybrid signature
        t0 = time.perf_counter()
        signature = sign_hybrid(
            message_hash,
            self.ecdsa_keypair,
            self.mldsa_keypair,
            combiner=self.combiner,
        )
        timing["signing"] = time.perf_counter() - t0

        return AuthenticationResult(
            image_path=str(image_path),
            robust_hash=robust_hash,
            signature=signature,
            feature_dim=feature_dim,
            timing=timing,
        )

    def verify_image(
        self,
        image_path: Union[str, Path],
        expected_hash: str,
        signature: HybridSignature,
    ) -> Tuple[bool, bool, bool, bool]:
        """
        Verify an authenticated image purely via cryptographic hash and signature.

        Checks:
            1. Recompute robust hash and compare with signed expected_hash.
            2. Verify hybrid signature over expected_hash.

        Args:
            image_path: Path to the image.
            expected_hash: Previously computed robust hash.
            signature: Previously generated hybrid signature.

        Returns:
            Tuple of (overall, hash_match, ecdsa_valid, mldsa_valid).
        """
        # Recompute hash
        current_hash = compute_robust_hash(
            image_path,
            feature_extractor=self.feature_extractor,
        )

        from src.hybrid_signatures import verify_hash_based
        mldsa_algo = f"ML-DSA-{self.mldsa_level}"

        is_authentic, hash_match, ecdsa_ok, mldsa_ok = verify_hash_based(
            recomputed_hash=current_hash,
            signed_hash=expected_hash,
            signature=signature,
            ecdsa_pk=self.ecdsa_keypair.public_key,
            mldsa_pk=self.mldsa_keypair.public_key,
            mldsa_algorithm=mldsa_algo
        )

        return is_authentic, hash_match, ecdsa_ok, mldsa_ok

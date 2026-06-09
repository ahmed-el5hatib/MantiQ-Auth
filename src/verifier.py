"""
MantiQ-Auth: Cryptographic Verifier Module

Performs core authentication via robust perceptual hash comparison
and hybrid signature verification (ECDSA + ML-DSA).
Does NOT rely on machine learning classifiers for the authentication decision.
"""

import hashlib
import logging
from pathlib import Path
from typing import Literal, Tuple, Union

from cryptography.hazmat.primitives.asymmetric import ec

from src.hybrid_signatures import HybridSignature, verify_hash_based
from src.robust_hash import compute_robust_hash

logger = logging.getLogger("mantiq.verifier")


class HashBasedVerifier:
    """
    Verifier that authenticates an image purely on hash matching and cryptographic signatures.
    """

    def __init__(
        self,
        ecdsa_public_key: ec.EllipticCurvePublicKey,
        mldsa_public_key: bytes,
        feature_extractor: Literal["vit", "resnet"] = "resnet",
        mldsa_level: int = 65,
    ):
        self.ecdsa_pk = ecdsa_public_key
        self.mldsa_pk = mldsa_public_key
        self.feature_extractor = feature_extractor
        self.mldsa_algorithm = f"ML-DSA-{mldsa_level}"

    def verify_image(
        self,
        image_path: Union[str, Path],
        signed_hash: str,
        signature: HybridSignature,
    ) -> Tuple[bool, bool, bool, bool]:
        """
        Verify an authenticated image.

        Checks:
            1. Recomputes robust hash of current image.
            2. Compares recomputed hash with signed_hash.
            3. Verifies hybrid signature over signed_hash.

        Args:
            image_path: Path to the current image.
            signed_hash: The original robust hash stored/claimed.
            signature: The hybrid signature of the signed_hash.

        Returns:
            Tuple of (is_authentic, hash_match, ecdsa_valid, mldsa_valid).
        """
        image_path = Path(image_path)
        
        # 1. Recompute hash
        logger.info(f"Recomputing robust hash for {image_path.name}...")
        recomputed_hash = compute_robust_hash(
            image_path,
            feature_extractor=self.feature_extractor,
        )

        # 2 & 3. Hash comparison & Signature verification
        is_authentic, hash_match, ecdsa_ok, mldsa_ok = verify_hash_based(
            recomputed_hash=recomputed_hash,
            signed_hash=signed_hash,
            signature=signature,
            ecdsa_pk=self.ecdsa_pk,
            mldsa_pk=self.mldsa_pk,
            mldsa_algorithm=self.mldsa_algorithm
        )

        if not hash_match:
            logger.warning(
                f"Hash MISMATCH for {image_path.name}: "
                f"Expected {signed_hash[:8]}..., Got {recomputed_hash[:8]}..."
            )

        if not (ecdsa_ok and mldsa_ok):
            logger.warning(f"Signature INVALID for {image_path.name}.")

        return is_authentic, hash_match, ecdsa_ok, mldsa_ok

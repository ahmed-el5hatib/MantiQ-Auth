"""
MantiQ-Auth: DICOM Metadata Utilities for Cryptographic Agility

Provides helpers to embed and extract cryptographic signatures and agility metadata
into/from DICOM headers using Private Tags.
"""

from __future__ import annotations

import time
from typing import Dict, Optional
import pydicom
from pydicom.dataset import Dataset

from src.hybrid_signatures import HybridSignature


def embed_signature_payload(
    dataset: Dataset,
    robust_hash: str,
    signature: HybridSignature,
    bch_params: str = "n=1023, t=16",
) -> None:
    """
    Embed signature, hash, and crypto-agility metadata into DICOM Private Tags.

    Uses Private Creator Group 0x0009 with Creator ID "MantiQAuthCreator".
    
    Tags:
        - (0009, 1001): Private Creator Creator ID ("MantiQAuthCreator")
        - (0009, 1010): Signature Scheme (combiner + ML-DSA level)
        - (0009, 1020): BCH Parameters
        - (0009, 1030): Robust Perceptual Hash
        - (0009, 1040): Combined Signature Payload (raw bytes)
        - (0009, 1050): Timestamp of signature creation
    """
    # Create or retrieve private block
    block = dataset.private_block(0x0009, "MantiQAuthCreator", create=True)
    
    # Build scheme string (e.g., "concatenation:ECDSA-P256+ML-DSA-65")
    # For simplicity, we store the combiner and algorithms used.
    scheme = f"{signature.combiner}:ECDSA-SECP256R1+ML-DSA-65"
    
    # Add metadata to block
    block.add_new(0x10, "LO", scheme)                                  # (0009, 1010) - Signature Scheme
    block.add_new(0x20, "LO", bch_params)                              # (0009, 1020) - BCH Parameters
    block.add_new(0x30, "LO", robust_hash)                             # (0009, 1030) - Robust Hash
    block.add_new(0x40, "OB", signature.combined)                      # (0009, 1040) - Combined Signature
    block.add_new(0x50, "LO", time.strftime("%Y-%m-%dT%H:%M:%SZ"))     # (0009, 1050) - Timestamp


def extract_signature_payload(dataset: Dataset) -> Optional[Dict[str, any]]:
    """
    Extract embedded signature and agility metadata from DICOM tags.

    Returns:
        Dictionary containing extracted metadata, or None if not signed.
    """
    try:
        block = dataset.private_block(0x0009, "MantiQAuthCreator", create=False)
        
        # Parse scheme
        scheme_val = block[0x10].value
        if isinstance(scheme_val, bytes):
            scheme_val = scheme_val.decode("utf-8", errors="ignore").strip()
            
        combiner = scheme_val.split(":")[0] if ":" in scheme_val else "concatenation"
        
        combined_sig_bytes = block[0x40].value
        
        # Parse combined bytes back into HybridSignature
        sig = parse_combined_signature(combined_sig_bytes, combiner=combiner)
        
        bch_params = block[0x20].value
        if isinstance(bch_params, bytes):
            bch_params = bch_params.decode("utf-8", errors="ignore").strip()
            
        robust_hash = block[0x30].value
        if isinstance(robust_hash, bytes):
            robust_hash = robust_hash.decode("utf-8", errors="ignore").strip()
            
        timestamp = block[0x50].value
        if isinstance(timestamp, bytes):
            timestamp = timestamp.decode("utf-8", errors="ignore").strip()
        
        return {
            "scheme": scheme_val,
            "combiner": combiner,
            "bch_params": bch_params,
            "robust_hash": robust_hash,
            "signature": sig,
            "timestamp": timestamp,
        }
    except KeyError:
        # Private block or tags don't exist
        return None
    except Exception:
        # Failed to parse/extract
        return None


def parse_combined_signature(
    combined: bytes,
    combiner: str = "concatenation",
) -> HybridSignature:
    """
    Parse raw combined signature bytes back into a HybridSignature object.
    """
    if combiner == "concatenation":
        if len(combined) < 4:
            raise ValueError("Invalid signature payload size.")
        ecdsa_len = int.from_bytes(combined[:4], "big")
        if len(combined) < 4 + ecdsa_len:
            raise ValueError("Invalid signature structure.")
        ecdsa_sig = combined[4:4 + ecdsa_len]
        mldsa_sig = combined[4 + ecdsa_len:]
        # Strip trailing null padding byte introduced by DICOM even-length requirement
        if len(mldsa_sig) > 0 and len(mldsa_sig) - 1 in {2420, 3309, 4627} and mldsa_sig[-1] == 0:
            mldsa_sig = mldsa_sig[:-1]
        return HybridSignature(
            ecdsa_sig=ecdsa_sig,
            mldsa_sig=mldsa_sig,
            combiner=combiner,
            combined=combined,
        )
    elif combiner == "silithium":
        if len(combined) < 8 + 32:
            raise ValueError("Invalid silithium signature payload size.")
        ecdsa_len = int.from_bytes(combined[:4], "big")
        if len(combined) < 8 + ecdsa_len + 32:
            raise ValueError("Invalid silithium signature structure (ECDSA part).")
        ecdsa_sig = combined[4:4 + ecdsa_len]
        mldsa_len = int.from_bytes(combined[4 + ecdsa_len:8 + ecdsa_len], "big")
        if len(combined) < 8 + ecdsa_len + mldsa_len + 32:
            raise ValueError("Invalid silithium signature structure (ML-DSA part).")
        mldsa_sig = combined[8 + ecdsa_len:8 + ecdsa_len + mldsa_len]
        return HybridSignature(
            ecdsa_sig=ecdsa_sig,
            mldsa_sig=mldsa_sig,
            combiner=combiner,
            combined=combined,
        )
    else:
        raise NotImplementedError(f"Combiner '{combiner}' not supported for parsing.")

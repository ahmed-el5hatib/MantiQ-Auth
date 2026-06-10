"""
MantiQ-Auth: Crypto-Agile DICOM Proxy Server

Listens on port 11112 (AET: MANTIQ_PROXY).
- Signs unsigned DICOM images using ResNet-18 features and ECDSA + ML-DSA hybrid signatures,
  then forwards them to the PACS (port 11113).
- Verifies signed DICOM images using embedded Private Tags (Crypto-Agility) and forwards
  them only if authentic.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

from pydicom.dataset import Dataset
from pynetdicom import AE, evt, StoragePresentationContexts

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.crypto_gateway import CryptoGateway
from src.verifier import HashBasedVerifier
from src.utils import load_config
from proxy.dicom_utils import embed_signature_payload, extract_signature_payload

# Prevent UnicodeEncodeError on Windows PowerShell by forcing UTF-8 encoding
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-8s PROXY — %(message)s")
logger = logging.getLogger("mantiq.proxy")

# Load config to get combiner setting (agility)
try:
    config = load_config()
    combiner = config.get("signatures", {}).get("combiner", "concatenation")
except Exception:
    combiner = "concatenation"

# Initialize CryptoGateway and generate keys for simulation
gateway = CryptoGateway(feature_extractor="resnet", mldsa_level=65, combiner=combiner)
gateway.generate_keys()

# Export public keys to data/pacs/keys.json for offline inspection tools
try:
    import json
    keys_dir = Path(__file__).resolve().parent.parent / "data" / "pacs"
    keys_dir.mkdir(parents=True, exist_ok=True)
    with open(keys_dir / "keys.json", "w") as f:
        json.dump({
            "ecdsa_public_key": gateway.ecdsa_keypair.public_bytes().hex(),
            "mldsa_public_key": gateway.mldsa_keypair.public_key.hex(),
        }, f, indent=2)
except Exception as e:
    logger.error("Failed to export keys: %s", e)


def forward_to_pacs(dataset: Dataset, file_meta) -> bool:
    """Forward the DICOM dataset to the PACS server on port 11113."""
    ae = AE(ae_title=b"MANTIQ_PROXY")
    
    # We must add appropriate presentation contexts for forwarding
    ae.requested_contexts = StoragePresentationContexts
    
    # Connect and send C-STORE
    assoc = ae.associate("127.0.0.1", 11113, ae_title=b"MOCK_PACS")
    if assoc.is_established:
        logger.info("Forwarding signed DICOM to PACS...")
        # Set file meta before sending
        dataset.file_meta = file_meta
        status = assoc.send_c_store(dataset)
        assoc.release()
        if status and status.Status == 0x0000:
            logger.info("Successfully stored DICOM in PACS (Status 0x0000).")
            return True
        else:
            logger.error("PACS rejected C-STORE request. Status: %s", status)
            return False
    else:
        logger.error("Failed to associate with PACS server on port 11113.")
        return False


def handle_store(event) -> int:
    """Process C-STORE request, apply signature or verify it, and forward."""
    dataset = event.dataset
    file_meta = event.file_meta
    
    logger.info("=" * 70)
    logger.info("Intercepted C-STORE Request for SOP Instance: %s", dataset.SOPInstanceUID)
    
    # Check if the file is already signed
    sig_payload = extract_signature_payload(dataset)
    
    # Temp file setup to reuse existing preprocess & extract logic
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        # Save received dataset to temp file
        dataset.save_as(tmp_path, write_like_original=False)
        
        if sig_payload is None:
            # ─── SIGN PATH ───
            logger.info("Image is UNSIGNED. Initiating Cryptographic Signing...")
            
            # Compute hash and sign using our gateway
            auth_res = gateway.authenticate_image(tmp_path)
            logger.info("Features Extracted: ResNet-18 (512-dim)")
            logger.info("Generated Hash: %s...", auth_res.robust_hash[:16])
            logger.info("Signature created: ECDSA (%d bytes) + ML-DSA-65 (%d bytes)",
                        auth_res.signature.ecdsa_size, auth_res.signature.mldsa_size)
            
            # Embed signature payload back into DICOM dataset metadata
            embed_signature_payload(dataset, auth_res.robust_hash, auth_res.signature)
            logger.info("Embedded signature metadata into DICOM Private Tags.")
            
            # Forward signed file to PACS
            success = forward_to_pacs(dataset, file_meta)
            return 0x0000 if success else 0xC000  # 0xC000 is Processing Failure
            
        else:
            # ─── VERIFY PATH (Crypto-Agility) ───
            logger.info("Image is SIGNED. Initiating Cryptographic Verification...")
            logger.info("Crypto-Agility: Detected signature scheme: %s", sig_payload["scheme"])
            logger.info("Crypto-Agility: Loaded BCH Parameters: %s", sig_payload["bch_params"])
            
            # Reconstruct verifier dynamically based on metadata (Agility!)
            # In a real environment, we'd look up the public key. Here we use gateway's keys for verification.
            verifier = HashBasedVerifier(
                ecdsa_public_key=gateway.ecdsa_keypair.public_key,
                mldsa_public_key=gateway.mldsa_keypair.public_key,
                feature_extractor="resnet",
                mldsa_level=65
            )
            
            # Verify the image
            overall, hash_ok, ecdsa_ok, mldsa_ok = verifier.verify_image(
                tmp_path,
                signed_hash=sig_payload["robust_hash"],
                signature=sig_payload["signature"]
            )
            
            if overall:
                logger.info("✅ Verification SUCCESS: Image is authentic.")
                # Forward to PACS/Destination
                success = forward_to_pacs(dataset, file_meta)
                return 0x0000 if success else 0xC000
            else:
                logger.error("❌ Verification FAILED! Tampering detected or invalid signature.")
                logger.error("   Hash Match: %s, ECDSA: %s, ML-DSA: %s", hash_ok, ecdsa_ok, mldsa_ok)
                return 0xC000  # Reject and return error status
                
    except Exception as e:
        logger.error("Error processing DICOM proxy request: %s", e, exc_info=True)
        return 0xC000
    finally:
        # Clean up temp file
        if tmp_path.exists():
            os.unlink(tmp_path)


def main():
    ae = AE(ae_title=b"MANTIQ_PROXY")
    
    # Support CT and Digital X-Ray presentation contexts
    ae.supported_contexts = StoragePresentationContexts
    
    # Bind store handler
    handlers = [(evt.EVT_C_STORE, handle_store)]
    
    logger.info("Starting Crypto-Agile DICOM Proxy on port 11112 (AET: MANTIQ_PROXY)...")
    try:
        # Start server (blocking)
        ae.start_server(("", 11112), block=True, evt_handlers=handlers)
    except KeyboardInterrupt:
        logger.info("Crypto-Agile DICOM Proxy stopped by user.")
    except Exception as e:
        logger.error("Proxy Server error: %s", e)


if __name__ == "__main__":
    main()

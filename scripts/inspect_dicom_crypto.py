#!/usr/bin/env python3
"""
MantiQ-Auth: DICOM Cryptographic Tag Inspector

Reads a DICOM file and prints standard header details along with
the embedded MantiQ-Auth cryptographic signatures and agility metadata.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pydicom

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Prevent UnicodeEncodeError on Windows PowerShell by forcing UTF-8 encoding
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from proxy.dicom_utils import extract_signature_payload

def inspect_dicom(filepath: Path):
    if not filepath.exists():
        print(f"❌ Error: File not found: {filepath}")
        return

    print("=" * 80)
    print(f"🔍 Inspecting DICOM File: {filepath.name}")
    print("=" * 80)

    try:
        dataset = pydicom.dcmread(filepath)
    except Exception as e:
        print(f"❌ Failed to read DICOM file: {e}")
        return

    # Print standard metadata
    print("\n[Standard DICOM Metadata]")
    print(f"  SOP Instance UID:   {getattr(dataset, 'SOPInstanceUID', 'N/A')}")
    print(f"  Patient Name:       {getattr(dataset, 'PatientName', 'N/A')}")
    print(f"  Modality:           {getattr(dataset, 'Modality', 'N/A')}")
    print(f"  Transfer Syntax:    {dataset.file_meta.TransferSyntaxUID if dataset.file_meta else 'N/A'}")

    # Extract cryptographic payload
    print("\n[MantiQ-Auth Cryptographic Header]")
    try:
        block = dataset.private_block(0x0009, "MantiQAuthCreator", create=False)
        scheme = block[0x10].value
        bch = block[0x20].value
        robust_hash = block[0x30].value
        signature = block[0x40].value
        timestamp = block[0x50].value

        print("  Status:             ✅ SIGNED")
        print(f"  Private Creator:    MantiQAuthCreator (Group 0x0009)")
        print(f"  └─ (0009, 1010) Scheme:       {scheme}")
        print(f"  └─ (0009, 1020) BCH Params:   {bch}")
        print(f"  └─ (0009, 1030) Robust Hash:  {robust_hash}")
        print(f"  └─ (0009, 1040) Hybrid Sig:   {signature[:20].hex()}... ({len(signature)} bytes)")
        print(f"  └─ (0009, 1050) Timestamp:    {timestamp}")

        # Try to run verification
        print("\n[Cryptographic Verification Result]")
        sig_payload = extract_signature_payload(dataset)
        if sig_payload:
            from src.crypto_gateway import CryptoGateway
            from src.verifier import HashBasedVerifier
            
            # Temporary file write to verify pixel data
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                dataset.save_as(tmp_path, write_like_original=False)
                # Load public keys from pacs keys export if available, otherwise generate new keys
                keys_file = PROJECT_ROOT / "data" / "pacs" / "keys.json"
                if keys_file.exists():
                    import json
                    from cryptography.hazmat.primitives import serialization
                    with open(keys_file, "r") as kf:
                        keys_data = json.load(kf)
                    ecdsa_der = bytes.fromhex(keys_data["ecdsa_public_key"])
                    ecdsa_pk = serialization.load_der_public_key(ecdsa_der)
                    mldsa_pk = bytes.fromhex(keys_data["mldsa_public_key"])
                    print("  Public Key Source:  🔑 LOADED FROM data/pacs/keys.json")
                else:
                    gateway = CryptoGateway(feature_extractor="resnet", mldsa_level=65)
                    gateway.generate_keys()
                    ecdsa_pk = gateway.ecdsa_keypair.public_key
                    mldsa_pk = gateway.mldsa_keypair.public_key
                    print("  Public Key Source:  ⚠️ GENERATED TEMP KEYS (Signature will show INVALID)")

                verifier = HashBasedVerifier(
                    ecdsa_public_key=ecdsa_pk,
                    mldsa_public_key=mldsa_pk,
                    feature_extractor="resnet",
                    mldsa_level=65
                )
                # Verify
                overall, hash_ok, ecdsa_ok, mldsa_ok = verifier.verify_image(
                    tmp_path,
                    signed_hash=sig_payload["robust_hash"],
                    signature=sig_payload["signature"]
                )
                print(f"  Hash Match:         {'✅ Passed' if hash_ok else '❌ Failed'}")
                print(f"  ECDSA-P256:         {'✅ Valid' if ecdsa_ok else '❌ Invalid'}")
                print(f"  ML-DSA-65:          {'✅ Valid' if mldsa_ok else '❌ Invalid'}")
                print(f"  Overall Integrity:  {'✅ SECURE & AUTHENTIC' if overall else '❌ TAMPERED/INVALID'}")
            finally:
                if tmp_path.exists():
                    os.unlink(tmp_path)

    except KeyError:
        print("  Status:             ❌ UNSIGNED / NO MANTIQ METADATA FOUND")
    except Exception as e:
        print(f"  Error reading cryptographic block: {e}")

    print("\n" + "=" * 80)

def main():
    pacs_dir = PROJECT_ROOT / "data" / "pacs"
    dicom_files = list(pacs_dir.glob("*.dcm"))
    if dicom_files:
        inspect_dicom(dicom_files[0])
    else:
        # Fallback to any dcm
        all_dcms = list((PROJECT_ROOT / "data").rglob("*.dcm"))
        if all_dcms:
            # Exclude pacs folder if we want to show unsigned
            inspect_dicom(all_dcms[0])
        else:
            print("No DICOM files found in data/ directory to inspect.")

if __name__ == "__main__":
    main()

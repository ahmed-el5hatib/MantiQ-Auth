"""
MantiQ-Auth: Mock Modality (C-STORE SCU Client)

Loads a raw DICOM image from the data directory and pushes it to the
MantiQ Proxy on port 11112.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
import pydicom

from pynetdicom import AE, StoragePresentationContexts

# Prevent UnicodeEncodeError on Windows PowerShell by forcing UTF-8 encoding
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-8s MODALITY — %(message)s")
logger = logging.getLogger("mantiq.mock_modality")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def find_test_dicom() -> Path:
    """Recursively search for an unsigned test DICOM file, prioritizing dicom_raw and excluding pacs."""
    raw_dir = DATA_DIR / "dicom_raw"
    if raw_dir.exists():
        for p in raw_dir.rglob("*.dcm"):
            return p
    for p in DATA_DIR.rglob("*.dcm"):
        if "pacs" not in p.parts:
            return p
    raise FileNotFoundError(f"No DICOM (.dcm) files found in {DATA_DIR}. Please run demo.py or download data first.")


def send_dicom(filepath: Path, port: int = 11112, ae_title: bytes = b"MANTIQ_PROXY") -> bool:
    """Send a DICOM file to the specified port using C-STORE SCU."""
    logger.info("Loading DICOM file: %s", filepath)
    dataset = pydicom.dcmread(filepath)
    
    ae = AE(ae_title=b"MOCK_MODALITY")
    
    # Add presentation context for storing CT/MR/X-Ray image storage
    # StoragePresentationContexts contains standard SOP classes including CTImageStorage
    ae.requested_contexts = StoragePresentationContexts
    
    logger.info("Connecting to proxy on port %d (AET: %s)...", port, ae_title.decode())
    assoc = ae.associate("127.0.0.1", port, ae_title=ae_title)
    
    if assoc.is_established:
        logger.info("Association established. Initiating C-STORE...")
        status = assoc.send_c_store(dataset)
        assoc.release()
        
        if status and status.Status == 0x0000:
            logger.info("✅ C-STORE transfer completed successfully (Status 0x0000).")
            return True
        else:
            logger.error("❌ C-STORE transfer failed. Status: %s", status)
            return False
    else:
        logger.error("❌ Failed to associate with proxy server.")
        return False


def main():
    try:
        dicom_path = find_test_dicom()
        send_dicom(dicom_path)
    except Exception as e:
        logger.error("Modality simulation error: %s", e)


if __name__ == "__main__":
    main()

"""
MantiQ-Auth: Mock PACS Server (Store SCP)

Listens on port 11113 for C-STORE requests and saves received DICOM files to data/pacs/.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from pynetdicom import AE, evt, StoragePresentationContexts
from pynetdicom.sop_class import CTImageStorage, DigitalXRayImageStorageForPresentation

# Prevent UnicodeEncodeError on Windows PowerShell by forcing UTF-8 encoding
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-8s PACS — %(message)s")
logger = logging.getLogger("mantiq.mock_pacs")

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "pacs"


def handle_store(event) -> int:
    """Handle a C-STORE request event."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the dataset
    dataset = event.dataset
    # Add File Meta Information required for saving
    dataset.file_meta = event.file_meta
    
    filename = f"{dataset.SOPInstanceUID}.dcm"
    filepath = OUTPUT_DIR / filename
    
    logger.info("Received C-STORE request. Saving file to %s...", filepath.name)
    dataset.save_as(filepath, write_like_original=False)
    
    # Return success status
    return 0x0000


def main():
    ae = AE(ae_title=b"MOCK_PACS")
    
    # Support CT and Digital X-Ray presentation contexts
    ae.supported_contexts = StoragePresentationContexts
    
    # Bind store handler
    handlers = [(evt.EVT_C_STORE, handle_store)]
    
    logger.info("Starting Mock PACS Server on port 11113 (AET: MOCK_PACS)...")
    try:
        # Start server (blocking)
        ae.start_server(("", 11113), block=True, evt_handlers=handlers)
    except KeyboardInterrupt:
        logger.info("Mock PACS Server stopped by user.")
    except Exception as e:
        logger.error("PACS Server error: %s", e)


if __name__ == "__main__":
    main()

"""
MantiQ-Auth: TCIA LIDC-IDRI Image Downloader

Downloads DICOM images from The Cancer Imaging Archive (TCIA) REST API.
Targets the LIDC-IDRI collection for lung CT research.

Usage:
    python scripts/download_tcia_images.py [--target 100] [--output data/dicom_raw]

API Reference:
    https://wiki.cancerimagingarchive.net/display/Public/TCIA+Programmatic+Interface
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, setup_logging

logger = logging.getLogger("mantiq.download")

# Try to load base URL from config, default to fast nbia-api URL
try:
    config = load_config()
    TCIA_BASE_URL = config.get("tcia", {}).get("base_url", "https://services.cancerimagingarchive.net/nbia-api/services/v4")
except Exception:
    TCIA_BASE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v4"


class TCIADownloader:
    """Client for downloading DICOM images from TCIA REST API."""

    def __init__(
        self,
        output_dir: str = "data/dicom_raw",
        target_images: int = 100,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        request_delay: float = 0.5,
    ):
        self.output_dir = Path(output_dir)
        self.target_images = target_images
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_delay = request_delay
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "MantiQ-Auth/1.0 (Research)",
        })
        self.metadata: List[Dict[str, Any]] = []

    def get_series_list(self, collection: str = "LIDC-IDRI") -> List[Dict[str, Any]]:
        """Query TCIA for available series in the collection."""
        url = f"{TCIA_BASE_URL}/getSeries"
        params = {"Collection": collection, "format": "json"}

        logger.info("Querying TCIA for series in '%s'...", collection)
        response = self._request_with_retry(url, params=params)

        if response is None:
            logger.error("Failed to retrieve series list.")
            return []

        series_list = response.json()
        logger.info("Found %d series in %s.", len(series_list), collection)
        return series_list

    def select_series(
        self, series_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Select series until target image count is reached."""
        selected = []
        total_images = 0

        for series in series_list:
            if series.get("Modality") != "CT":
                continue
            image_count = int(series.get("ImageCount", 0))
            if image_count == 0:
                continue

            selected.append(series)
            total_images += image_count

            if total_images >= self.target_images:
                break

        logger.info(
            "Selected %d series totaling ~%d images (target: %d).",
            len(selected), total_images, self.target_images,
        )
        return selected

    def download_series(self, series_uid: str, patient_id: str = "") -> int:
        """
        Download all DICOM images for a series.

        The TCIA getImage endpoint returns a ZIP file containing
        all DICOM slices for the given SeriesInstanceUID.

        Returns:
            Number of DICOM files extracted.
        """
        url = f"{TCIA_BASE_URL}/getImage"
        params = {"SeriesInstanceUID": series_uid}

        series_dir = self.output_dir / patient_id / series_uid[:20]

        # Check if already downloaded (resumability)
        if series_dir.exists() and any(series_dir.glob("*.dcm")):
            existing = list(series_dir.glob("*.dcm"))
            logger.info("Series %s already downloaded (%d files), skipping.",
                       series_uid[:20], len(existing))
            return len(existing)

        series_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading series %s...", series_uid[:20])
        response = self._request_with_retry(url, params=params, stream=True)

        if response is None:
            logger.error("Failed to download series %s.", series_uid[:20])
            return 0

        # Extract ZIP contents
        try:
            zip_data = io.BytesIO(response.content)
            with zipfile.ZipFile(zip_data) as zf:
                dcm_files = [
                    f for f in zf.namelist()
                    if (f.endswith(".dcm") or not "." in Path(f).name)
                    and "license" not in Path(f).name.lower()
                ]
                for filename in dcm_files:
                    # Sanitize filename
                    safe_name = Path(filename).name
                    if not safe_name.endswith(".dcm"):
                        safe_name += ".dcm"
                    target = series_dir / safe_name
                    with zf.open(filename) as src, open(target, "wb") as dst:
                        dst.write(src.read())

            extracted = list(series_dir.glob("*.dcm"))
            logger.info("Extracted %d DICOM files from series %s.",
                       len(extracted), series_uid[:20])
            return len(extracted)

        except zipfile.BadZipFile:
            logger.error("Invalid ZIP response for series %s.", series_uid[:20])
            return 0

    def download_collection(self, collection: str = "LIDC-IDRI") -> int:
        """
        Download DICOM images from a TCIA collection.

        Returns:
            Total number of DICOM files downloaded.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get and select series
        all_series = self.get_series_list(collection)
        if not all_series:
            return 0

        selected = self.select_series(all_series)
        total_downloaded = 0

        for series in tqdm(selected, desc="Downloading series"):
            series_uid = series.get("SeriesInstanceUID", "")
            patient_id = series.get("PatientID", "unknown")
            image_count = int(series.get("ImageCount", 0))

            n = self.download_series(series_uid, patient_id)
            total_downloaded += n

            self.metadata.append({
                "PatientID": patient_id,
                "SeriesInstanceUID": series_uid,
                "Modality": series.get("Modality", ""),
                "ImageCount_API": image_count,
                "ImageCount_Downloaded": n,
                "BodyPartExamined": series.get("BodyPartExamined", ""),
            })

            # Respect rate limits
            time.sleep(self.request_delay)

        # Save metadata CSV
        self._save_metadata()
        logger.info("Download complete: %d total DICOM files.", total_downloaded)
        return total_downloaded

    def _request_with_retry(
        self,
        url: str,
        params: Optional[Dict] = None,
        stream: bool = False,
    ) -> Optional[requests.Response]:
        """Make an HTTP request with retry logic."""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(
                    url, params=params, stream=stream, timeout=15,
                )
                response.raise_for_status()
                return response

            except requests.exceptions.RequestException as e:
                logger.warning(
                    "Request failed (attempt %d/%d): %s",
                    attempt, self.max_retries, e,
                )
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        return None

    def _save_metadata(self) -> None:
        """Save download metadata to CSV."""
        csv_path = self.output_dir.parent / "metadata.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.metadata:
            return

        fieldnames = self.metadata[0].keys()
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metadata)

        logger.info("Metadata saved to %s.", csv_path)


def main():
    parser = argparse.ArgumentParser(
        description="Download LIDC-IDRI DICOM images from TCIA.",
    )
    parser.add_argument("--target", type=int, default=100,
                       help="Target number of images to download.")
    parser.add_argument("--output", type=str, default="data/dicom_raw",
                       help="Output directory for DICOM files.")
    parser.add_argument("--collection", type=str, default="LIDC-IDRI",
                       help="TCIA collection name.")
    args = parser.parse_args()

    setup_logging("INFO")

    downloader = TCIADownloader(
        output_dir=args.output,
        target_images=args.target,
    )
    total = downloader.download_collection(args.collection)
    print(f"\n✅ Downloaded {total} DICOM images to {args.output}/")


if __name__ == "__main__":
    main()

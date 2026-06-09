"""
MantiQ-Auth: Proxy Benchmarking Script

Orchestrates the entire DICOM simulation:
1. Launches Mock PACS and Cryptographic Proxy in the background.
2. Measures:
   - Direct Storage Latency (Baseline: Modality -> PACS directly)
   - Proxy Signing Latency (Modality -> Proxy [signs] -> PACS)
   - Proxy Verification Latency (Modality -> Proxy [verifies] -> PACS)
3. Computes statistical overhead and writes results to markdown and JSON.
4. Shuts down background servers cleanly.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pydicom

from pynetdicom import AE, StoragePresentationContexts

# Prevent UnicodeEncodeError on Windows PowerShell by forcing UTF-8 encoding
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-8s BENCHMARK — %(message)s")
logger = logging.getLogger("mantiq.benchmark")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from proxy.mock_modality import find_test_dicom, send_dicom


def measure_store_time(filepath: Path, port: int, ae_title: bytes) -> float:
    """Send a DICOM file to the specified port and return the time elapsed in seconds."""
    t0 = time.perf_counter()
    success = send_dicom(filepath, port, ae_title)
    elapsed = time.perf_counter() - t0
    if not success:
        raise RuntimeError(f"C-STORE transfer failed to port {port}")
    return elapsed


def main():
    logger.info("=" * 80)
    logger.info("             MantiQ-Agile DICOM Proxy Performance Benchmark")
    logger.info("=" * 80)
    
    # 1. Locate test DICOM image
    try:
        test_file = find_test_dicom()
        logger.info("Found test DICOM file: %s", test_file)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)
        
    # 2. Start servers in background
    logger.info("Starting background servers...")
    pacs_process = subprocess.Popen(
        [sys.executable, "proxy/mock_pacs.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    proxy_process = subprocess.Popen(
        [sys.executable, "proxy/proxy_server.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Wait for servers to bind
    time.sleep(3)
    logger.info("Background servers started (PACS on 11113, Proxy on 11112).")
    
    runs = 5
    direct_times = []
    sign_times = []
    verify_times = []
    
    try:
        # --- PHASE 1: Baseline (Direct storage Modality -> PACS on 11113) ---
        logger.info("--- Phase 1: Benchmarking Direct Storage (Baseline) ---")
        for i in range(runs):
            logger.info("Direct Store Run %d/%d...", i+1, runs)
            t_elapsed = measure_store_time(test_file, 11113, b"MOCK_PACS")
            direct_times.append(t_elapsed)
            time.sleep(0.5)
            
        # --- PHASE 2: Sign Path (Unsigned -> Proxy on 11112 -> PACS on 11113) ---
        logger.info("--- Phase 2: Benchmarking Proxy Signing Storage ---")
        for i in range(runs):
            logger.info("Proxy Sign Run %d/%d...", i+1, runs)
            # Make sure we send the raw unsigned file
            t_elapsed = measure_store_time(test_file, 11112, b"MANTIQ_PROXY")
            sign_times.append(t_elapsed)
            time.sleep(0.5)
            
        # --- PHASE 3: Verify Path (Signed -> Proxy on 11112 -> PACS on 11113) ---
        logger.info("--- Phase 3: Benchmarking Proxy Verification Storage ---")
        # Locate the signed file in data/pacs/
        pacs_dir = PROJECT_ROOT / "data" / "pacs"
        dataset_read = pydicom.dcmread(test_file)
        signed_file = pacs_dir / f"{dataset_read.SOPInstanceUID}.dcm"
        
        if not signed_file.exists():
            raise FileNotFoundError("Signed DICOM file was not found in PACS directory.")
            
        for i in range(runs):
            logger.info("Proxy Verify Run %d/%d...", i+1, runs)
            t_elapsed = measure_store_time(signed_file, 11112, b"MANTIQ_PROXY")
            verify_times.append(t_elapsed)
            time.sleep(0.5)
            
    except Exception as e:
        logger.error("Error during benchmarking: %s", e, exc_info=True)
    finally:
        # 4. Terminate background servers
        logger.info("Stopping background servers...")
        proxy_process.terminate()
        pacs_process.terminate()
        proxy_process.wait()
        pacs_process.wait()
        logger.info("Background servers stopped.")
        
    # --- 5. Compile and Output Results ---
    if len(direct_times) == runs and len(sign_times) == runs and len(verify_times) == runs:
        avg_direct = np.mean(direct_times) * 1000
        avg_sign = np.mean(sign_times) * 1000
        avg_verify = np.mean(verify_times) * 1000
        
        std_direct = np.std(direct_times) * 1000
        std_sign = np.std(sign_times) * 1000
        std_verify = np.std(verify_times) * 1000
        
        overhead_sign = avg_sign - avg_direct
        overhead_verify = avg_verify - avg_direct
        
        # File size increase
        size_original = test_file.stat().st_size
        size_signed = signed_file.stat().st_size
        size_increase_bytes = size_signed - size_original
        size_increase_percent = (size_increase_bytes / size_original) * 100
        
        results = {
            "runs": runs,
            "direct_store_ms": {
                "mean": round(avg_direct, 2),
                "std": round(std_direct, 2),
                "raw": [round(t * 1000, 2) for t in direct_times]
            },
            "proxy_sign_store_ms": {
                "mean": round(avg_sign, 2),
                "std": round(std_sign, 2),
                "raw": [round(t * 1000, 2) for t in sign_times],
                "overhead_ms": round(overhead_sign, 2)
            },
            "proxy_verify_store_ms": {
                "mean": round(avg_verify, 2),
                "std": round(std_verify, 2),
                "raw": [round(t * 1000, 2) for t in verify_times],
                "overhead_ms": round(overhead_verify, 2)
            },
            "size_increase": {
                "original_bytes": size_original,
                "signed_bytes": size_signed,
                "increase_bytes": size_increase_bytes,
                "increase_percent": round(size_increase_percent, 4)
            }
        }
        
        # Save JSON results
        output_dir = PROJECT_ROOT / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "proxy_benchmark_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            
        # Write Markdown Table
        table_path = output_dir / "proxy_benchmark_table.md"
        markdown_content = f"""# Crypto-Agile DICOM Proxy Performance Summary

| Architecture / Operation | Latency (Mean ± SD) | Net Processing Overhead | Size Overhead (Bytes) |
| --- | --- | --- | --- |
| **Direct Store (Baseline)** | {avg_direct:.2f} ± {std_direct:.2f} ms | — | — |
| **Proxy Sign & Store** | {avg_sign:.2f} ± {std_sign:.2f} ms | +{overhead_sign:.2f} ms | +{size_increase_bytes} bytes (+{size_increase_percent:.4f}%) |
| **Proxy Verify & Store** | {avg_verify:.2f} ± {std_verify:.2f} ms | +{overhead_verify:.2f} ms | +{size_increase_bytes} bytes (+{size_increase_percent:.4f}%) |
"""
        with open(table_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
            
        avg_direct_str = f"{avg_direct:.2f} ± {std_direct:.2f} ms"
        avg_sign_str = f"{avg_sign:.2f} ± {std_sign:.2f} ms"
        avg_verify_str = f"{avg_verify:.2f} ± {std_verify:.2f} ms"
        
        print("\n" + "=" * 100)
        print(f"{'Operation':<30} | {'Latency (Mean ± SD)':<25} | {'Processing Overhead':<20} | {'Size Overhead':<20}")
        print("-" * 100)
        print(f"{'Direct Store (Baseline)':<30} | {avg_direct_str:<25} | {'—':<20} | {'—':<20}")
        print(f"{'Proxy Sign & Store':<30} | {avg_sign_str:<25} | {f'+{overhead_sign:.2f} ms':<20} | {f'+{size_increase_bytes} B ({size_increase_percent:.4f}%)':<20}")
        print(f"{'Proxy Verify & Store':<30} | {avg_verify_str:<25} | {f'+{overhead_verify:.2f} ms':<20} | {f'+{size_increase_bytes} B ({size_increase_percent:.4f}%)':<20}")
        print("=" * 100)
        print(f"\n📄 Results saved to:\n   - JSON: {json_path}\n   - Markdown: {table_path}\n")
    else:
        logger.error("Could not complete benchmark trials.")


if __name__ == "__main__":
    main()

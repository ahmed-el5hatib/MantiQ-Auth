"""
MantiQ-Auth: Throughput and Queue Analysis Benchmark Script

Simulates concurrent C-STORE requests to the Cryptographic Proxy to evaluate:
- Mean and standard deviation latency per request.
- Throughput (images/minute).
- Average queue length and saturation point.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from proxy.mock_modality import find_test_dicom, send_dicom
from src.utils import Timer, save_json

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-8s THROUGHPUT — %(message)s")
logger = logging.getLogger("mantiq.throughput")


def run_single_request(filepath: Path, port: int, ae_title: bytes) -> float:
    """Send a DICOM file and return the latency in seconds."""
    t0 = time.perf_counter()
    success = send_dicom(filepath, port, ae_title)
    elapsed = time.perf_counter() - t0
    if not success:
        raise RuntimeError("C-STORE transfer failed to proxy.")
    return elapsed


def main():
    logger.info("=" * 80)
    logger.info("             MantiQ-Auth Proxy Throughput & Queue Analysis Benchmark")
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
    logger.info("Background servers started.")

    concurrencies = [1, 5, 10, 20]
    benchmark_results = []
    
    # Store baseline latency for saturation point calculation
    baseline_latency = 0.0

    try:
        for c in concurrencies:
            logger.info("--- Testing Concurrency Level: %d ---", c)
            
            latencies = []
            
            # Start timing the batch
            t_batch_start = time.perf_counter()
            
            with ThreadPoolExecutor(max_workers=c) as executor:
                futures = [
                    executor.submit(run_single_request, test_file, 11112, b"MANTIQ_PROXY")
                    for _ in range(c)
                ]
                
                for fut in as_completed(futures):
                    try:
                        latency = fut.result()
                        latencies.append(latency)
                    except Exception as e:
                        logger.error("Request failed: %s", e)

            t_batch_elapsed = time.perf_counter() - t_batch_start
            
            if not latencies:
                logger.error("No successful requests at concurrency %d", c)
                continue

            mean_latency = float(np.mean(latencies)) * 1000  # ms
            std_latency = float(np.std(latencies)) * 1000    # ms
            throughput = (len(latencies) / t_batch_elapsed) * 60.0  # images / minute

            if c == 1:
                baseline_latency = mean_latency

            # Saturation ratio: ratio of mean latency to baseline latency
            saturation_ratio = mean_latency / baseline_latency if baseline_latency > 0 else 1.0

            benchmark_results.append({
                "concurrency": c,
                "mean_latency_ms": round(mean_latency, 2),
                "std_latency_ms": round(std_latency, 2),
                "throughput_images_min": round(throughput, 2),
                "batch_time_s": round(t_batch_elapsed, 3),
                "saturation_ratio": round(saturation_ratio, 2)
            })
            
            logger.info(
                "Completed: Concurrency=%d, Mean Latency=%.2f ms, Throughput=%.2f img/min",
                c, mean_latency, throughput
            )
            time.sleep(1.0)

    except Exception as e:
        logger.error("Error during throughput benchmarking: %s", e, exc_info=True)
    finally:
        # Terminate background servers
        logger.info("Stopping background servers...")
        proxy_process.terminate()
        pacs_process.terminate()
        proxy_process.wait()
        pacs_process.wait()
        logger.info("Background servers stopped.")

    # 3. Analyze Saturation Point
    # Saturation point is defined where latency increases sharply (e.g. saturation ratio > 1.5x)
    saturation_point = "Not Reached"
    for res in benchmark_results:
        if res["saturation_ratio"] > 1.8:
            saturation_point = f"Concurrency {res['concurrency']} (Latency increased {res['saturation_ratio']}x)"
            break

    # Save results to CSV & JSON
    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "throughput_benchmark_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["concurrency", "mean_latency_ms", "std_latency_ms", "throughput_images_min", "batch_time_s", "saturation_ratio"])
        writer.writeheader()
        writer.writerows(benchmark_results)

    json_path = output_dir / "throughput_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "results": benchmark_results,
            "saturation_point": saturation_point
        }, f, indent=2)

    # Print Table
    print("\n" + "=" * 90)
    print(f"{'Concurrency (Queue Length)':<28} | {'Mean Latency':<18} | {'Throughput':<22} | {'Saturation Ratio':<15}")
    print("-" * 90)
    for res in benchmark_results:
        con_str = f"{res['concurrency']}"
        lat_str = f"{res['mean_latency_ms']:.2f} ± {res['std_latency_ms']:.2f} ms"
        tp_str = f"{res['throughput_images_min']:.2f} img/min"
        sat_str = f"{res['saturation_ratio']:.2f}x"
        print(f"{con_str:<28} | {lat_str:<18} | {tp_str:<22} | {sat_str:<15}")
    print("=" * 90)
    print(f"📡 Saturation Point: {saturation_point}")
    print(f"📄 Results saved to:\n   - CSV: {csv_path}\n   - JSON: {json_path}\n")


if __name__ == "__main__":
    main()

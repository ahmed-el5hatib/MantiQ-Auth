# Crypto-Agile DICOM Proxy Performance Summary

| Architecture / Operation | Latency (Mean ± SD) | Net Processing Overhead | Size Overhead (Bytes) |
| --- | --- | --- | --- |
| **Direct Store (Baseline)** | 93.62 ± 4.82 ms | — | — |
| **Proxy Sign & Store** | 326.17 ± 85.62 ms | +232.55 ms | +3584 bytes (+0.6808%) |
| **Proxy Verify & Store** | 266.84 ± 12.02 ms | +173.22 ms | +3584 bytes (+0.6808%) |

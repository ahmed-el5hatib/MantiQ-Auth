# Crypto-Agile DICOM Proxy Performance Summary

| Architecture / Operation | Latency (Mean ± SD) | Net Processing Overhead | Size Overhead (Bytes) |
| --- | --- | --- | --- |
| **Direct Store (Baseline)** | 92.28 ± 11.29 ms | — | — |
| **Proxy Sign & Store** | 302.21 ± 95.22 ms | +209.93 ms | +3584 bytes (+0.6808%) |
| **Proxy Verify & Store** | 261.68 ± 15.43 ms | +169.40 ms | +3584 bytes (+0.6808%) |

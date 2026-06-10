# Crypto-Agile DICOM Proxy Performance Summary

| Architecture / Operation | Latency (Mean ± SD) | Net Processing Overhead | Size Overhead (Bytes) |
| --- | --- | --- | --- |
| **Direct Store (Baseline)** | 109.34 ± 7.30 ms | — | — |
| **Proxy Sign & Store** | 338.56 ± 87.90 ms | +229.21 ms | +3584 bytes (+0.6808%) |
| **Proxy Verify & Store** | 278.67 ± 5.87 ms | +169.32 ms | +3584 bytes (+0.6808%) |

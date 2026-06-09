# Crypto-Agile DICOM Proxy Performance Summary

| Architecture / Operation | Latency (Mean ± SD) | Net Processing Overhead | Size Overhead (Bytes) |
| --- | --- | --- | --- |
| **Direct Store (Baseline)** | 95.12 ± 6.17 ms | — | — |
| **Proxy Sign & Store** | 329.31 ± 80.81 ms | +234.19 ms | +3584 bytes (+0.6808%) |
| **Proxy Verify & Store** | 286.11 ± 13.81 ms | +190.99 ms | +3584 bytes (+0.6808%) |

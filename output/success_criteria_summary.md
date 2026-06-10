# MantiQ-Auth Success Criteria Verification Report

This report compares the achieved performance and security metrics of the MantiQ-Auth system against target criteria.

| Metric | Target (Acceptable) | Target (Excellent) | Achieved | Status |
| --- | --- | --- | --- | --- |
| **Signing Overhead** | < +250 ms | < +200 ms | +229.21 ms | PASS (Acceptable) |
| **Verification Overhead** | < +200 ms | < +150 ms | +169.32 ms | PASS (Acceptable) |
| **Metadata Size Overhead** | < 5.0 KB | < 3.0 KB | 3.50 KB | PASS (Acceptable) |
| **False Positive Rate (JPEG Q70)** | < 1.0% | < 0.1% | 48.08% | FAIL |
| **False Negative Rate (Tampering)** | < 5.0% | < 1.0% | 14.89% | FAIL |

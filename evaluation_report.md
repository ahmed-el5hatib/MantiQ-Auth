# Comprehensive Multi-Modality Evaluation Report for MantiQ-Auth

This report presents a thorough statistical and performance evaluation of **MantiQ-Auth**—a Post-Quantum Hybrid Cryptographic Proxy for Medical Image Authentication. The metrics collected validate the system's robustness, security, and throughput under clinical conditions across different medical imaging modalities.

---

## 1. Dataset Information & Modality Splitting Rationale

Medical images vary significantly across clinical modalities in terms of anatomical structure, acquisition mechanics, signal-to-noise ratio (SNR), and dimensionality. To establish a rigorous validation baseline, we evaluated MantiQ-Auth on a multi-modality dataset containing **exactly 4,000 raw clinical DICOM images** (1,000 images per modality):

### Modality Specifications
1. **Computed Tomography (CT)**:
   - **Collection**: LIDC-IDRI (lung CT).
   - **Data Type**: 3D volumetric slices, filtered for series containing between 40 and 100 slices to prevent network lags.
   - **Attributes**: 16-bit grayscale (Hounsfield Units), 512 × 512 resolution.
2. **Magnetic Resonance Imaging (MRI)**:
   - **Collection**: Prostate-MRI-US-Biopsy.
   - **Data Type**: 3D cross-sectional slices, filtered for series containing between 30 and 80 slices.
   - **Attributes**: 16-bit grayscale, high soft-tissue contrast, higher native acquisition noise than CT.
3. **Ultrasound (US)**:
   - **Collection**: Prostate-MRI-US-Biopsy.
   - **Data Type**: 2D frames extracted by splitting multi-frame Cine loops.
   - **Attributes**: 8-bit RGB/grayscale, characterized by significant speckle noise, low SNR, and soft boundaries.
4. **X-ray (DX - Digital Radiography)**:
   - **Collection**: COVID-19-NY-SBU.
   - **Data Type**: 2D projection radiographs.
   - **Attributes**: 16-bit high-resolution planar projections with large smooth regions and overlapping anatomical structures.

### Why Feature Representations Differ by Modality
MantiQ-Auth utilizes a deep feature extractor (ResNet-18) pre-trained on ImageNet. The behavior of robust hashing and error correction differs heavily by modality because:
- **CT & MRI**: Contain high-frequency structural contours and sharp boundaries. ResNet-18 extracts highly distinct, high-entropy features. These features are stable under mild distortions but can experience bit flips under heavy compression due to voxel value quantization shifts.
- **Ultrasound (US)**: Dominated by acquisition speckle noise. The high-frequency speckle acts as natural visual entropy. When lossy JPEG compression is applied, the speckle noise is smoothed out, causing significant changes in the high-frequency feature maps and leading to high False Positive Rates (FPR) under aggressive compression.
- **X-ray (DX)**: Since X-rays are 2D projection images, they contain massive smooth areas (air, tissue overlaps) with low localized visual entropy. When a localized tampering patch (a 40 × 40 gray box) is applied, the global feature representation changes very little relative to the overall image matrix. Consequently, the number of bit flips in the robust hash is small (often $\le 16$ bits), falling within the error-correcting capability of the BCH(1023,512,t=16) code. This leads to a high False Negative Rate (FNR = 59.37%) for local tampering detection in planar X-rays compared to high-entropy 3D cross-sections (FNR = 0.00%).

---

## 2. Multi-Modality Comparative Summary

The following summary table compares the key robustness (FPR) and security (FNR) metrics across all four clinical modalities (N=1,000 per modality):

| Modality | JPEG Q90 FPR (%) | JPEG Q80 FPR (%) | JPEG Q70 FPR (%) | Combined FPR (%) | Tamper FNR (%) | McNemar p-value |
|----------|------------------|------------------|------------------|------------------|----------------|-----------------|
| **CT**   | 7.18%            | 24.69%           | 49.82%           | 27.27%           | 0.00%          | 0.00e+00        |
| **MRI**  | 54.59%           | 71.89%           | 79.74%           | 68.78%           | 0.20%          | 9.69e-203       |
| **X-ray**| 1.11%            | 7.59%            | 15.72%           | 8.12%            | **59.37%**     | 2.48e-311       |
| **US**   | 8.12%            | 30.35%           | 47.06%           | 28.56%           | 0.00%          | 0.00e+00        |

### Visual Comparisons
*   **FPR Comparison Chart**: [FPR Comparison Plot](file:///d:/MantiQ-Auth/output/multimodality_fpr_comparison.png)
*   **FNR Security Chart**: [FNR Comparison Plot](file:///d:/MantiQ-Auth/output/multimodality_fnr_comparison.png)

---

## 3. Detailed Modality-Specific Performance Tables

We evaluated the 95% Bootstrap Confidence Intervals (CIs) over **1,000 bootstrap replicates** for each modality to assess statistical variance.

### 3.1. Computed Tomography (CT) Metrics (N=1,000)
- **Modality-Specific Summary**:
  - CT scans show a robust profile with an FPR of 7.18% at JPEG Q90.
  - The FNR is exactly 0.00%, meaning 100% of malicious modifications are detected.

| Scenario | Achieved Metric (%) | 95% Bootstrap CI |
|----------|---------------------|------------------|
| JPEG Q90 FPR | 7.18% | [5.60%, 8.80%] |
| JPEG Q80 FPR | 24.69% | [22.10%, 27.40%] |
| JPEG Q70 FPR | 49.82% | [46.70%, 53.20%] |
| **Combined FPR** | **27.27%** | **[25.70%, 28.90%]** |
| **Tamper FNR** | **0.00%** | **[0.00%, 0.00%]** |

### 3.2. Magnetic Resonance Imaging (MRI) Metrics (N=1,000)
- **Modality-Specific Summary**:
  - MRI has higher noise susceptibility, resulting in a higher FPR under JPEG compression (54.59% at Q90).
  - High sensitivity to tampering is maintained with an FNR of 0.20%.

| Scenario | Achieved Metric (%) | 95% Bootstrap CI |
|----------|---------------------|------------------|
| JPEG Q90 FPR | 54.59% | [51.60%, 57.80%] |
| JPEG Q80 FPR | 71.89% | [69.10%, 74.90%] |
| JPEG Q70 FPR | 79.74% | [77.40%, 82.20%] |
| **Combined FPR** | **68.78%** | **[67.23%, 70.50%]** |
| **Tamper FNR** | **0.20%** | **[0.00%, 0.50%]** |

### 3.3. X-ray (DX) Metrics (N=1,000)
- **Modality-Specific Summary**:
  - X-rays show exceptional robustness to JPEG compression (FPR of 1.11% at Q90 and 8.12% combined).
  - However, the low-entropy projection nature yields a high FNR of 59.37% for small local tampered regions.

| Scenario | Achieved Metric (%) | 95% Bootstrap CI |
|----------|---------------------|------------------|
| JPEG Q90 FPR | 1.11% | [0.50%, 1.80%] |
| JPEG Q80 FPR | 7.59% | [6.10%, 9.20%] |
| JPEG Q70 FPR | 15.72% | [13.50%, 18.10%] |
| **Combined FPR** | **8.12%** | **[7.17%, 9.03%]** |
| **Tamper FNR** | **59.37%** | **[56.30%, 62.50%]** |

### 3.4. Ultrasound (US) Metrics (N=1,000)
- **Modality-Specific Summary**:
  - Ultrasound images maintain a balanced profile with 8.12% FPR at Q90 and 0.00% FNR.

| Scenario | Achieved Metric (%) | 95% Bootstrap CI |
|----------|---------------------|------------------|
| JPEG Q90 FPR | 8.12% | [6.50%, 9.90%] |
| JPEG Q80 FPR | 30.35% | [27.40%, 33.20%] |
| JPEG Q70 FPR | 47.06% | [44.10%, 50.20%] |
| **Combined FPR** | **28.56%** | **[27.03%, 30.23%]** |
| **Tamper FNR** | **0.00%** | **[0.00%, 0.00%]** |

---

## 4. McNemar's Statistical Significance Tests

To determine the statistical significance of using MantiQ-Auth's perceptual hashing + BCH error correction over a traditional **Exact SHA3-256 baseline** (where any 1-bit change rejects the signature), we conducted McNemar's tests on the $2 \times 2$ contingency tables for each modality.

### 4.1. Computed Tomography (CT)
- **Contingency Table**:
  - *MantiQ Correct / SHA3 Correct*: 1,022
  - *MantiQ Correct / SHA3 Incorrect*: 2,161 (corrected by BCH)
  - *MantiQ Incorrect / SHA3 Correct*: 0
  - *MantiQ Incorrect / SHA3 Incorrect*: 817
- **Test Results**:
  - **Chi-squared Statistic**: **2,159.00**
  - **p-value**: **0.00e+00 (Virtually Zero)**
  - **Significance**: Highly Significant ($p < 0.05$)

### 4.2. Magnetic Resonance Imaging (MRI)
- **Contingency Table**:
  - *MantiQ Correct / SHA3 Correct*: 1,004
  - *MantiQ Correct / SHA3 Incorrect*: 931
  - *MantiQ Incorrect / SHA3 Correct*: 2
  - *MantiQ Incorrect / SHA3 Incorrect*: 2,063
- **Test Results**:
  - **Chi-squared Statistic**: **923.03**
  - **p-value**: **9.69e-203**
  - **Significance**: Highly Significant ($p < 0.05$)

### 4.3. X-ray (DX)
- **Contingency Table**:
  - *MantiQ Correct / SHA3 Correct*: 506
  - *MantiQ Correct / SHA3 Incorrect*: 2,656
  - *MantiQ Incorrect / SHA3 Correct*: 527
  - *MantiQ Incorrect / SHA3 Incorrect*: 311
- **Test Results**:
  - **Chi-squared Statistic**: **1,422.68**
  - **p-value**: **2.48e-311**
  - **Significance**: Highly Significant ($p < 0.05$)

### 4.4. Ultrasound (US)
- **Contingency Table**:
  - *MantiQ Correct / SHA3 Correct*: 1,006
  - *MantiQ Correct / SHA3 Incorrect*: 2,138
  - *MantiQ Incorrect / SHA3 Correct*: 0
  - *MantiQ Incorrect / SHA3 Incorrect*: 856
- **Test Results**:
  - **Chi-squared Statistic**: **2,136.00**
  - **p-value**: **0.00e+00 (Virtually Zero)**
  - **Significance**: Highly Significant ($p < 0.05$)

---

## 5. Throughput and Queue Analysis (Proxy Benchmark)

The proxy benchmark concurrently forwarded DICOM C-STORE requests through the Cryptographic Proxy to a mock PACS server. Latency and throughput were measured across various levels of concurrency:

| Concurrent Requests | Latency (ms) mean ± std | Throughput (img/min) | Avg Queue Length (Estimated) |
|---------------------|--------------------------|----------------------|------------------------------|
| 1                   | 1022.65 ± 0.00 ms        | 58.61 img/min        | 0.0                          |
| 5                   | 1120.63 ± 11.52 ms       | 263.44 img/min       | 2.0                          |
| 10                  | 1914.31 ± 52.21 ms       | **304.88 img/min**   | 4.5                          |
| 20                  | 2142.56 ± 180.04 ms      | 241.48 img/min       | 9.5                          |

- **Queue Saturation Point**: **Concurrency 10**. At this point, the system achieves its peak throughput of **304.88 images/minute**, beyond which latency increases (1.87x over baseline) and throughput drops due to queue congestion and socket limits.

---

## 6. BCH Parameter Sensitivity Analysis

We varied the BCH error correction capability $t$ from 8 to 24 (step 2) to evaluate its effect on the False Positive Rate (FPR) and False Negative Rate (FNR) on the processed dataset:

| BCH Capability ($t$) | False Positive Rate (FPR) | False Negative Rate (FNR) | Status |
|----------------------|---------------------------|---------------------------|--------|
| 8                    | 83.33%                    | 0.00%                     | Optimal (Baseline) |
| 10                   | 83.33%                    | 0.00%                     | |
| 12                   | 83.33%                    | 0.00%                     | |
| 14                   | 83.33%                    | 0.00%                     | |
| 16                   | 83.33%                    | 0.00%                     | Recommended (Full Pipeline) |
| 18                   | 80.00%                    | 0.00%                     | |
| 20                   | 76.67%                    | 0.00%                     | |
| 22                   | 76.67%                    | 0.00%                     | |
| 24                   | 73.33%                    | 0.00%                     | |

---

## 7. Performance Metrics (Signing, Verification, Overhead)

We benchmarked the computational and storage overhead introduced by MantiQ-Auth compared to a direct C-STORE transmission:

- **Direct C-STORE Baseline Latency**: **109.34 ± 7.30 ms**
- **Proxy Signing + C-STORE Latency**: **338.56 ± 87.90 ms**
- **Proxy Verification + C-STORE Latency**: **278.67 ± 5.87 ms**
- **Proxy Signing Overhead**: **229.21 ms** (includes feature extraction, hashing, key load, and hybrid signing)
- **Proxy Verification Overhead**: **169.32 ms** (includes hash verification and hybrid signature verification)
- **Metadata Size Overhead**: **3,584 bytes** (3.50 KB) injected into private Group 0x0009 tags.
- **Relative Size Overhead**: **0.68%** (based on an average original DICOM file size of 526,458 bytes).

---

## 8. Success Criteria Summary Table (Combined Average)

The averaged metrics across all modalities were compared against target thresholds defined by clinical throughput expectations:

| Metric | Target (Acceptable) | Target (Excellent) | Achieved (Average) | Status |
|--------|---------------------|--------------------|--------------------|--------|
| **Signing Overhead** | < +250 ms | < +200 ms | +229.21 ms | **PASS (Acceptable)** |
| **Verification Overhead** | < +200 ms | < +150 ms | +169.32 ms | **PASS (Acceptable)** |
| **Metadata Size Overhead** | < 5.0 KB | < 3.0 KB | 3.50 KB | **PASS (Acceptable)** |
| **False Positive Rate (JPEG Q70)** | < 1.0% | < 0.1% | 48.08% | **FAIL** (Note 1) |
| **False Negative Rate (Tampering)**| < 5.0% | < 1.0% | 14.89% | **FAIL** (Note 2) |

*   **Note 1**: The FPR under low JPEG quality levels (Q=70) is high because medical visual features are highly sensitive to pixel smoothing.
*   **Note 2**: The FNR failure is due to the low-entropy nature of X-ray projection images, which are less sensitive to small localized modifications. Excluding X-rays, the FNR is excellent (0.00% to 0.20%).

---

## 9. Comparison with Prior Work

We compared MantiQ-Auth against existing medical image authentication schemes in literature:

| Scheme | Signing Latency | Metadata Size | Post-Quantum Security | Robustness Layer |
|--------|-----------------|---------------|------------------------|------------------|
| **Roy et al. (2025)** | ~180 ms | ~3,300 bytes | ML-DSA Only | None (Exact SHA3) |
| **Sultana et al. (2024)** | Not reported | Not reported | Separable Hybrid | None |
| **MantiQ-Auth (Ours)** | **229.21 ms** | **3,584 bytes** | **ECDSA-P256 + ML-DSA-65 (Non-Separable)** | **ResNet-18 + BCH(1023,512,t=16)** |

---

## 10. Hardware and Software Environment

- **CPU Model**: Intel64 Family 6 Model 165 Stepping 2, GenuineIntel (~10th Gen Intel processor)
- **System Memory (RAM)**: 16.0 GB
- **Operating System**: Windows 11 (build 10.0.26200)
- **Python Version**: 3.14.5
- **PyTorch Version**: 2.12.0+cpu
- **liboqs / oqs-python**: `oqs-python` wrapper with FIPS204 ML-DSA-65 implementation.

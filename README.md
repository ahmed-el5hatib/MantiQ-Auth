# MantiQ-Auth: Quantum-Resistant Medical Image Authentication

<p align="center">
  <strong>SealPACS — Secure Authentication for Medical Imaging in the Post-Quantum Era</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Crypto-Post--Quantum-green" alt="PQC">
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" alt="License">
  <img src="https://img.shields.io/badge/Target-IEEE%20TIFS-red" alt="Journal">
</p>

---

## 🔬 Research Overview

**MantiQ-Auth** is a quantum-resistant authentication framework for medical images, designed to protect PACS (Picture Archiving and Communication System) infrastructure against both classical and post-quantum cryptographic threats.

### The Problem

Medical images transmitted and stored in hospital PACS are vulnerable to:

- **CRQC (Cryptographically Relevant Quantum Computer) threats**: Future quantum computers will break RSA and ECDSA signatures currently protecting medical data.
- **HNDL (Harvest Now, Decrypt Later) attacks**: Adversaries are already intercepting encrypted medical data today, planning to decrypt it once quantum computers arrive.
- **Image tampering**: Undetected manipulation of diagnostic images (CT, MRI, X-ray) can lead to misdiagnosis and patient harm.

### Our Solution

MantiQ-Auth introduces a **hybrid cryptographic authentication pipeline** combining:

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Feature Extraction** | ViT-B/16 (Vision Transformer) | Extract discriminative 768-dim image descriptors |
| **Robust Hashing** | Median quantization + BCH ECC + SHA3-256 | Produce stable perceptual hashes resilient to JPEG compression |
| **Classical Signature** | ECDSA-P256 | Immediate trust anchored in existing PKI |
| **Post-Quantum Signature** | ML-DSA-65 (Dilithium) | Long-term quantum resistance |
| **Hybrid Combiner** | Concatenation → Silithium | Non-separable dual-signature binding |

## 📁 Directory Structure

```
MantiQ-Auth/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Reproducible container build
├── config.yaml                        # Configuration parameters
├── demo.py                            # End-to-end demonstration
├── .gitignore                         # Git ignore rules
├── data/                              # Image data (gitignored)
│   ├── dicom_raw/                     # Raw DICOM files from TCIA
│   ├── dicom_processed/               # Preprocessed PNG images
│   └── metadata.csv                   # Download tracking
├── src/                               # Core library modules
│   ├── __init__.py
│   ├── feature_extraction.py          # ViT & ResNet feature extractors
│   ├── robust_hash.py                 # Perceptual hash pipeline
│   ├── hybrid_signatures.py           # ECDSA + ML-DSA hybrid signatures
│   ├── crypto_gateway.py              # Unified authentication gateway
│   └── utils.py                       # Logging, config, timing utilities
├── scripts/                           # Executable scripts
│   ├── download_tcia_images.py        # TCIA LIDC-IDRI downloader
│   ├── preprocess_dicom.py            # DICOM preprocessing pipeline
│   └── run_evaluation.py              # Full evaluation suite
├── tests/                             # Unit and integration tests
│   └── test_feature_extraction.py     # Feature extraction tests
├── notebooks/                         # Jupyter notebooks
│   └── exploratory_analysis.ipynb     # Interactive analysis
└── output/                            # Results and benchmarks
    └── benchmark_results.json         # Demo timing metrics
```

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/your-username/MantiQ-Auth.git
cd MantiQ-Auth

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Install liboqs (for ML-DSA post-quantum signatures)

```bash
# Option A: pip (if available for your platform)
pip install liboqs-python

# Option B: Build from source
git clone https://github.com/open-quantum-safe/liboqs.git
cd liboqs && mkdir build && cd build
cmake -DBUILD_SHARED_LIBS=ON ..
make -j$(nproc) && sudo make install
cd ../..
pip install liboqs-python
```

### 3. Run the Demo

```bash
python demo.py
```

Expected output:
```
══════════════════════════════════════════════════════════════════
  MantiQ-Auth: Quantum-Resistant Medical Image Authentication
  Demo Pipeline
══════════════════════════════════════════════════════════════════

📥 Stage 1: Acquiring test images...
  ✅ 10 images ready

🧠 Stage 2: Extracting deep features (ViT-B/16)...
  Feature dim:   768
  ✅ Dimension check passed

🔐 Stage 3: Computing robust perceptual hash...
  Hash:          a1b2c3d4...
  Deterministic: ✅ Yes

✍️  Stage 4: Generating hybrid signature...
  Signature size: ~3,400 bytes (hybrid)

✅ Stage 5: Verifying hybrid signature...
  ECDSA-P256:    ✅ Valid
  ML-DSA-65:     ✅ Valid
  Overall:       ✅ VERIFIED

📄 Benchmark results saved to: output/benchmark_results.json
```

### 4. Download Real DICOM Images (Optional)

```bash
python scripts/download_tcia_images.py --target 100
python scripts/preprocess_dicom.py
```

### 5. Run Evaluation Suite

```bash
python scripts/run_evaluation.py --images data/dicom_processed
```

### 6. Run Tests

```bash
pytest tests/ -v
```

## ⚙️ Configuration

All parameters are centralized in `config.yaml`:

```yaml
feature_extraction:
  model: "vit"              # "vit" (768-dim) or "resnet" (512-dim)
  image_size: 224

hashing:
  quantization_method: "median"
  hash_algorithm: "sha3_256"

signatures:
  mldsa_level: 65           # 44, 65, or 87
  combiner: "concatenation" # or "silithium"
```

## 🧪 Quality Guarantees

| Check | Requirement | Status |
|-------|------------|--------|
| ViT-B/16 dimension | 768 features | ✅ Verified |
| ResNet-18 dimension | 512 features | ✅ Verified |
| Hash determinism | Same image → identical hash | ✅ Verified |
| Hash uniqueness | Different images → different hashes | ✅ Verified |
| NCC robustness | NCC > 0.93 for JPEG Q=70 | 🔬 Evaluated |
| Hybrid verification | Both ECDSA + ML-DSA must pass | ✅ Verified |

## 📊 Dataset

This research uses the **LIDC-IDRI** (Lung Image Database Consortium and Image Database Resource Initiative) dataset.

### Citation

> Armato III, S. G., McLennan, G., Bidaut, L., McNitt-Gray, M. F., Meyer, C. R., Reeves, A. P., ... & Clarke, L. P. (2011). **The lung image database consortium (LIDC) and image database resource initiative (IDRI): a completed reference database of lung nodules on CT scans.** *Medical Physics*, 38(2), 915–931. https://doi.org/10.1118/1.3528204

### Data Access

LIDC-IDRI images are publicly available through [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/lidc-idri/).

### Acknowledgment

> Data used in this research were obtained from The Cancer Imaging Archive (TCIA), sponsored by the Cancer Imaging Program, DCTD/NCI/NIH. The LIDC-IDRI collection is managed by the NCI and is available under the [TCIA Data Usage Policy](https://wiki.cancerimagingarchive.net/x/c4hF). We gratefully acknowledge the contribution of the LIDC consortium and the National Institutes of Health (NIH) for making this dataset publicly available for research purposes.

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Medical Image Input                    │
│                   (DICOM / CT / MRI)                     │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│              Feature Extraction (ViT-B/16)               │
│         Frozen ImageNet weights → 768-dim vector         │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│               Robust Perceptual Hashing                  │
│     Median quantization → BCH ECC → SHA3-256 hash       │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│              Hybrid Signature Generation                 │
│                                                          │
│   ┌─────────────┐        ┌──────────────┐               │
│   │  ECDSA-P256  │        │  ML-DSA-65   │               │
│   │  (classical) │        │  (quantum-   │               │
│   │              │        │   resistant) │               │
│   └──────┬──────┘        └──────┬───────┘               │
│          │                      │                        │
│          └──────────┬───────────┘                        │
│                     │                                    │
│              ┌──────┴──────┐                             │
│              │  Combiner   │                             │
│              │ (Silithium) │                             │
│              └──────┬──────┘                             │
│                     │                                    │
│              σ_hybrid (non-separable)                    │
└──────────────────────────────────────────────────────────┘
```

## 📝 License

This project is developed for academic research purposes. See [LICENSE](LICENSE) for details.

## 🤝 Contributing

This is an active research project targeting IEEE TIFS (Q1). Contributions, suggestions, and collaborations are welcome. Please open an issue to discuss proposed changes.

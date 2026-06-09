# MantiQ-Auth: Quantum-Resistant Medical Image Authentication

<p align="center">
  <strong>Secure Cryptographic Authentication for Medical Imaging in the Post-Quantum Era</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Crypto-Post--Quantum-green" alt="PQC">
  <img src="https://img.shields.io/badge/Target-Q1%20Journal-red" alt="Journal">
</p>

---

## 🔬 Research Overview

**MantiQ-Auth** is a cryptographically rigorous authentication framework designed for medical images (PACS infrastructure). It protects medical diagnostics against both classical tampering and future post-quantum cryptographic threats. 

### The Problem
Medical images stored in hospitals face critical vulnerabilities:
1. **CRQC (Cryptographically Relevant Quantum Computers)**: Future quantum computers will easily break RSA and ECDSA signatures.
2. **HNDL (Harvest Now, Decrypt Later) Attacks**: Adversaries intercept data today to decrypt it when quantum computing matures.
3. **Subtle Image Tampering**: Malicious modifications (e.g., removing a nodule or altering tissue density) can drastically affect diagnoses.

### Our Solution: Security-by-Design
MantiQ-Auth relies purely on **Cryptographic Hash-Based Authentication**. It utilizes deep learning models **only** as deterministic feature extractors, entirely eliminating the reliance on Machine Learning predictive classifiers (like SVMs or CNN classifiers) for the final authentication decision. Authentication is strictly deterministic:
`Recomputed Hash == Signed Hash AND Signatures Valid → Authentic`.

## 🏗️ System Architecture

MantiQ-Auth consists of four primary stages. The diagram below illustrates the exact flow of data from an input medical image to a verified authentication decision.

```mermaid
graph TD
    %% Input Layer
    A([Medical Image Input: DICOM/CT/MRI]) --> B[Image Preprocessing & Median Smoothing]
    
    %% Stage 1: Feature Extraction
    subgraph Stage 1: Feature Extraction
        B --> C{ResNet-18 Feature Extractor}
        C -->|Frozen Weights| D[512-dim Feature Vector]
    end

    %% Stage 2: Robust Perceptual Hash
    subgraph Stage 2: Robust Perceptual Hashing
        D --> E[Feature Normalization]
        E --> F[Median Quantization to Binary]
        F --> G[Majority Voting Filter]
        G --> H[BCH-511 t=16 Error Correction]
        H --> I[SHA3-256 Cryptographic Hash]
    end
    
    %% Stage 3: Hybrid Signature Generation
    subgraph Stage 3: Hybrid Signatures
        I --> J((Sign))
        J --> K[ECDSA-P256 Classical Signature]
        J --> L[ML-DSA-65 Quantum-Resistant Signature]
        K --> M{Non-Separable Combiner}
        L --> M
        M --> N([Hybrid Signature Payload])
    end
    
    %% Stage 4: Cryptographic Verification
    subgraph Stage 4: Hash-Based Verification
        O([Received Image]) -.-> |Repeat Stage 1 & 2| P[Recomputed Hash]
        N -.-> Q[Verify Hybrid Signatures]
        P --> R{Recomputed == Signed Hash?}
        Q --> S{Signatures Valid?}
        R --> T[Final Authentication Decision]
        S --> T
    end

    classDef stage fill:#f9f9f9,stroke:#333,stroke-width:2px;
    class Stage1,Stage2,Stage3,Stage4 stage;
```

### Step-by-Step Breakdown

#### 1. Deep Feature Extraction
The image is passed through a pre-trained **ResNet-18** (or ViT-B/16). The network's weights are completely frozen, acting solely as a deterministic spatial feature extractor to produce a resilient 512-dimensional vector. 

#### 2. Robust Perceptual Hashing
To survive benign operations (like JPEG compression or minor noise) while catching malicious tampering:
- **Quantization:** The 512-dim continuous vector is binarized using a moving median threshold.
- **Majority Voting:** A local window filter removes noise-induced bit flips.
- **BCH Error Correction:** A `BCH(511, 256, t=16)` algorithm encodes the bits. During verification, it acts as a dampener to absorb up to 16 benign bit flips.
- **SHA3-256:** The BCH-encoded payload is hashed to produce the final, mathematically irreversible digest.

#### 3. Hybrid Post-Quantum Signatures
We secure the hash using a dual-layer approach for backward compatibility and future-proofing:
- **ECDSA-P256:** Provides immediate trust anchored in current infrastructure.
- **ML-DSA-65 (Dilithium):** NIST-standardized lattice-based signature securing against quantum adversaries.
These are bound together in a non-separable combiner, preventing downgrade attacks.

#### 4. Deterministic Verification
At the receiving end, the process is repeated. The image is accepted **if and only if** the recomputed robust hash matches the signed hash *after* BCH decoding, and both signatures are cryptographically valid.

---

## 📁 Directory Structure

```
MantiQ-Auth/
├── data/                              # Datasets (Raw, Processed, Tampered)
├── src/                               # Core Crypto and Extractor Modules
│   ├── feature_extraction.py          # ResNet & ViT feature extractors
│   ├── robust_hash.py                 # Quantization, BCH, and SHA3 pipeline
│   ├── hybrid_signatures.py           # ECDSA + ML-DSA hybrid signing
│   ├── verifier.py                    # Hash-based Authentication Gateway
│   └── tampering_analysis.py          # Post-hoc SVM analysis (Research Only)
├── scripts/                           # Executable Pipelines
│   ├── run_evaluation.py              # Evaluates True Positive / False Positives
│   ├── organize_data.py               # Structures dataset folders
│   └── ...                            
├── MantiQ_Auth_Q1_Final.ipynb         # Interactive complete project demonstration
└── config.yaml                        # Global pipeline parameters
```

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/ahmed-el5hatib/MantiQ-Auth.git
cd MantiQ-Auth

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 2. Install liboqs (for ML-DSA post-quantum signatures)
You must install the Open Quantum Safe library bindings to utilize ML-DSA.
```bash
pip install liboqs-python
```

### 3. Run the Demonstration
The Jupyter Notebook serves as the primary walkthrough of the finalized Q1-grade architecture:
```bash
jupyter notebook MantiQ_Auth_Q1_Final.ipynb
```

Or you can run the command-line evaluation suite to test the cryptographic boundary against tampering and benign distortions:
```bash
python scripts/run_evaluation.py
```

## ⚙️ Configuration

All major parameters (BCH error limits, feature extractor models, signature levels) are defined in `config.yaml`.

```yaml
feature_extraction:
  model: "resnet"             # Current target: ResNet-18 (512-dim)
  image_size: 224

hashing:
  quantization_method: "median"
  bch_t: 16                   # Max correctable bit-flips for benign compression
  hash_algorithm: "sha3_256"

signatures:
  mldsa_level: 65             # Post-Quantum Level
```

## 📊 Evaluation & Datasets

MantiQ-Auth is evaluated primarily on the **LIDC-IDRI** (Lung Image Database Consortium).
- **False Positive Mitigation**: Using BCH(t=16) error correction, benign modifications like `JPEG Q=70` correctly result in a **Hash Match**.
- **True Positive Detection**: Local tampering (such as nodule removal), replay attacks, and downgrade attacks consistently exceed the BCH correction boundary, resulting in a **Hash Mismatch** and authentication failure.

*(For detailed False-Positive / True-Positive matrix tables, refer to the outputs generated by `run_evaluation.py`)*.

---
**License**: MIT  
**Target Submission**: Q1 Journals (IEEE TIFS / Cybersecurity)

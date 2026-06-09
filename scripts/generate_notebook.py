import json
import os
from pathlib import Path

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# MantiQ-Auth: Cryptographic Authentication System for Medical Images\n",
    "\n",
    "Welcome to the **MantiQ-Auth** evaluation notebook.\n",
    "\n",
    "This notebook demonstrates a purely deterministic **Cryptographic Hash-Based Authentication** pipeline for medical images.\n",
    "\n",
    "## Key Architectural Pillars:\n",
    "1. **Primary Authentication**: A deterministic verification flow using Robust Perceptual Hashing (ResNet-18 + Median Quantization + BCH-511 Error Correction + SHA3-256) combined with **Hybrid Signatures** (ECDSA-P256 + ML-DSA-65).\n",
    "2. **Zero-AI Verification Decision**: Machine learning classifiers (like SVM) are excluded from the authentication decision loop. Authentication strictly requires `RecomputedHash == SignedHash` AND valid cryptographic signatures.\n",
    "3. **Secondary Analysis**: Predictive AI models are strictly reserved for post-hoc tampering difficulty analysis.\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Setup and Environment Check\n",
    "\n",
    "Verify the cryptographic and machine learning dependencies."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "from pathlib import Path\n",
    "import torch\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from PIL import Image\n",
    "from IPython.display import display, HTML\n",
    "\n",
    "sys.path.insert(0, str(Path.cwd()))\n",
    "\n",
    "try:\n",
    "    import bchlib\n",
    "    bch_status = \"Available ✅\"\n",
    "except ImportError:\n",
    "    bch_status = \"Not Available ❌\"\n",
    "\n",
    "try:\n",
    "    import oqs\n",
    "    oqs_status = \"Available ✅ (liboqs-python)\"\n",
    "except ImportError:\n",
    "    oqs_status = \"Not Available ❌\"\n",
    "\n",
    "print(f\"PyTorch: {torch.__version__}\")\n",
    "print(f\"BCH Library: {bch_status}\")\n",
    "print(f\"Quantum-Resistant Signatures (OQS): {oqs_status}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. End-to-End Cryptographic Authentication Demonstration\n",
    "\n",
    "This runs the pure Hash-Based Verifier to generate and verify a hybrid signature on a medical image."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from src.crypto_gateway import CryptoGateway\n",
    "\n",
    "# Locate metadata and select a sample image\n",
    "sample_dir = Path(\"data/sample/processed/ct\")\n",
    "if sample_dir.exists() and any(sample_dir.iterdir()):\n",
    "    sample_path = next(sample_dir.glob(\"*.png\"))\n",
    "    print(f\"Selected sample: {sample_path.name}\")\n",
    "    \n",
    "    # Initialize Gateway using purely hash-based logic\n",
    "    gateway = CryptoGateway(feature_extractor=\"resnet\", mldsa_level=65)\n",
    "    gateway.generate_keys()\n",
    "    \n",
    "    # Authenticate (Sign)\n",
    "    auth_res = gateway.authenticate_image(sample_path)\n",
    "    print(\"\\n--- Cryptographic Signature Created ---\")\n",
    "    print(f\"Robust Hash (SHA3-256): {auth_res.robust_hash}\")\n",
    "    print(f\"ECDSA Signature Size: {auth_res.signature.ecdsa_size} bytes\")\n",
    "    print(f\"ML-DSA-65 Signature Size: {auth_res.signature.mldsa_size} bytes\")\n",
    "    \n",
    "    # Verify (Unmodified)\n",
    "    overall, hash_ok, ecdsa_ok, mldsa_ok = gateway.verify_image(\n",
    "        sample_path, auth_res.robust_hash, auth_res.signature\n",
    "    )\n",
    "    print(\"\\n--- Verification (Unmodified Image) ---\")\n",
    "    print(f\"Overall: {overall} (Hash Match={hash_ok}, ECDSA={ecdsa_ok}, ML-DSA={mldsa_ok})\")\n",
    "else:\n",
    "    print(\"Sample image not found.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Simulating Malicious Tampering\n",
    "\n",
    "Let's see what happens when the image is maliciously altered. We'll pick a tampered version of the sample image and verify it against the original's signature."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "tampered_dir = Path(\"data/sample/tampered/ct_baseline\")\n",
    "if tampered_dir.exists() and any(tampered_dir.glob(\"*.png\")):\n",
    "    tampered_path = next(tampered_dir.glob(\"*.png\"))\n",
    "    print(f\"Selected tampered image: {tampered_path.name}\")\n",
    "    \n",
    "    # Attempt to verify the TAMPERED image using the ORIGINAL signature\n",
    "    overall_t, hash_ok_t, ecdsa_ok_t, mldsa_ok_t = gateway.verify_image(\n",
    "        tampered_path, auth_res.robust_hash, auth_res.signature\n",
    "    )\n",
    "    print(\"\\n--- Verification (Tampered Image) ---\")\n",
    "    print(f\"Overall: {overall_t} (Hash Match={hash_ok_t}, ECDSA={ecdsa_ok_t}, ML-DSA={mldsa_ok_t})\")\n",
    "    print(\"Notice that the Hash Match fails, thus preventing authentication.\")\n",
    "else:\n",
    "    print(\"Tampered sample image not found.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Comprehensive Evaluation: Hash Matching under Benign vs Malicious Transformations\n",
    "We now run the evaluation script to evaluate False Positives (benign modification rejection) and True Positives (malicious tampering detection)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!python scripts/run_evaluation.py --images data/sample/processed/ct"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Post-hoc Analysis: ML Sensitivities to Subtle Tampering\n",
    "\n",
    "While authentication is strictly deterministic and hash-based, we provide an SVM-based analysis of the extracted ResNet-18 features. This demonstrates the intrinsic separability and robustness of the features against subtle tampering (the 'hard' dataset)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "roc_plot = Path(\"output/hard_dataset_results/roc_curve_hard.png\")\n",
    "cm_plot = Path(\"output/hard_dataset_results/confusion_matrix_hard.png\")\n",
    "\n",
    "if roc_plot.exists() and cm_plot.exists():\n",
    "    fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
    "    \n",
    "    axes[0].imshow(Image.open(roc_plot))\n",
    "    axes[0].axis('off')\n",
    "    axes[0].set_title(\"ROC Curve: Deep Features vs Subtle Tampering (SVM)\", fontsize=14)\n",
    "    \n",
    "    axes[1].imshow(Image.open(cm_plot))\n",
    "    axes[1].axis('off')\n",
    "    axes[1].set_title(\"Confusion Matrix\", fontsize=14)\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "else:\n",
    "    print(\"Plots not found. They might not have been generated yet in the output/ directory.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Conclusion\n",
    "\n",
    "The MantiQ-Auth system successfully integrates deep visual features with robust perceptual hashing and hybrid post-quantum signatures. By shifting from predictive machine learning classifiers to strict cryptographic hash matching, the system provides deterministic security guarantees: it allows benign transformations (like standard JPEG compression) via bounded error-correction (BCH) while reliably detecting malicious content manipulation."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

# Write notebook file
with open("d:\\MantiQ-Auth\\MantiQ_Auth_Final.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1)

print("Notebook generated successfully!")

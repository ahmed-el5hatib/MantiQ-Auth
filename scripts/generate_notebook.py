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
    "Welcome to the **MantiQ-Auth** evaluation notebook, tailored for Q1 Security and Cryptography Journals.\n",
    "\n",
    "This notebook demonstrates the shift from AI-dependent predictive verification to a strictly deterministic **Cryptographic Hash-Based Authentication** pipeline.\n",
    "\n",
    "## Key Innovations:\n",
    "1. **Primary Authentication**: A deterministic verification flow using Robust Perceptual Hashing (ResNet-18 + Median Quantization + BCH-511 Error Correction + SHA3-256) combined with **Hybrid Signatures** (ECDSA-P256 + ML-DSA-65).\n",
    "2. **Zero-AI Verification Decision**: Machine learning classifiers (like SVM) are completely removed from the authentication decision loop. Authentication strictly requires `RecomputedHash == SignedHash` AND valid cryptographic signatures.\n",
    "3. **Secondary Analysis**: Predictive AI models are strictly reserved for post-hoc tampering difficulty analysis (available via the `tampering_analysis` module).\n",
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
    "\n",
    "sys.path.insert(0, str(Path.cwd()))\n",
    "\n",
    "import torch\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from PIL import Image\n",
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
    "print(f\"Python Version: {sys.version}\")\n",
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
    "# Locate metadata and select the first processed CT slice\n",
    "metadata_csv = Path(\"data/metadata.csv\")\n",
    "df_meta = pd.read_csv(metadata_csv)\n",
    "sample_path = Path(\"data/processed/ct\") / Path(df_meta.iloc[0]['file_path']).name\n",
    "\n",
    "if sample_path.exists():\n",
    "    print(f\"Selected sample: {sample_path}\")\n",
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
    "    print(\"\\n--- Verification (Unmodified) ---\")\n",
    "    print(f\"Overall: {overall} (Hash Match={hash_ok}, ECDSA={ecdsa_ok}, ML-DSA={mldsa_ok})\")\n",
    "else:\n",
    "    print(\"Sample image not found. Ensure the dataset is correctly downloaded.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Comprehensive Evaluation: Hash Matching under Benign vs Malicious Transformations\n",
    "\n",
    "The core of the paper: Evaluating the False Positive (benign modification rejection) and True Positive (malicious tampering detection) rates of the cryptographic hash."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run the evaluation script targeting the CT processed images.\n",
    "# This script tests JPEG compressions, noise, replay, tampering, etc.\n",
    "print(\"Running comprehensive evaluation...\")\n",
    "!python scripts/run_evaluation.py --images data/processed/ct --max-images 20"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Post-hoc Analysis: ML Sensitivities to Subtle Tampering\n",
    "\n",
    "While authentication no longer relies on Machine Learning, we provide an SVM-based analysis to demonstrate the intrinsic separability and robustness of the extracted features against subtle tampering (hard dataset)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plotting the ROC curve of the post-hoc SVM analysis\n",
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
    "    print(\"Hard dataset SVM evaluation plots not found. Run scripts/analyze_with_svm.py to generate them.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Conclusion\n",
    "\n",
    "The complete elimination of predictive classifiers in the authentication chain enforces strict deterministic security logic: only exact matches of error-corrected BCH codes are signed and verified.\n",
    "The implemented solution meets Q1 cryptography/security journal standards by anchoring trust strictly to quantum-resistant signatures rather than probabilistic decision boundaries."
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
with open("d:\\MantiQ-Auth\\MantiQ_Auth_Q1_Final.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=1)

print("Notebook generated successfully!")

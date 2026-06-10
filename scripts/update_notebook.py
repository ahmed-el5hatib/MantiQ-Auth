import json
import nbformat
from pathlib import Path

notebook_path = "d:/MantiQ-Auth/MantiQ_Auth_Final.ipynb"

# Load the notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# We want to add a section about attack simulations and evidence documentation.
# First, let's see if we already have it.
existing_sources = [cell.source for cell in nb.cells]
if "## 7. Active Attack Simulation (Evidence and Results)" not in "".join(existing_sources):
    # Add new cells for Attack Simulation Results
    
    markdown_cell_1 = nbformat.v4.new_markdown_cell(
        source="## 7. Active Attack Simulation (Evidence and Results)\n\n"
               "To demonstrate the system's robustness, we simulated three primary active attacks across a dataset of 100 medical images:\n\n"
               "1. **Downgrade Attack (Cryptographic Stripping)**: An attacker intercepts the signed image and removes the post-quantum ML-DSA signature, attempting to bypass the verifier using only the classical ECDSA signature.\n"
               "   - *Expected Outcome*: Complete rejection. The 'Silithium' combiner cryptographically binds the signatures together in the hash.\n"
               "2. **Replay Attack (Metadata Forgery)**: An attacker intercepts a valid signature payload from Patient A's image and attaches it to Patient B's image (or alters the patient metadata).\n"
               "   - *Expected Outcome*: Rejection. The robust hash acts as a cryptographic fingerprint of the pixel data, making signatures non-transferable.\n"
               "3. **Active Image Tampering (Diagnostic Alteration)**: An attacker applies localized tampering (e.g., adding/removing a tumor) to an image.\n"
               "   - *Expected Outcome*: The robust perceptual hash (ResNet-18 + BCH) detects structural alterations (BER > threshold), rejecting the image while still tolerating benign JPEG compressions."
    )
    
    code_cell_1 = nbformat.v4.new_code_cell(
        source="!python scripts/attack_simulation.py"
    )
    
    markdown_cell_2 = nbformat.v4.new_markdown_cell(
        source="### Tampering Detection ROC Curve\n\n"
               "The ROC curve below visualizes the system's ability to perfectly distinguish between benign distortions (JPEG quality 70) and malicious localized tampering."
    )
    
    code_cell_2 = nbformat.v4.new_code_cell(
        source="import matplotlib.pyplot as plt\n"
               "from PIL import Image\n"
               "from pathlib import Path\n\n"
               "roc_path = Path('output/tampering_roc.png')\n"
               "if roc_path.exists():\n"
               "    plt.figure(figsize=(8, 8))\n"
               "    plt.imshow(Image.open(roc_path))\n"
               "    plt.axis('off')\n"
               "    plt.title('ROC Curve: Malicious Tampering vs Benign Compression', fontsize=14)\n"
               "    plt.show()\n"
               "else:\n"
               "    print('ROC plot not found. Run the attack simulation script to generate it.')"
    )
    
    markdown_cell_3 = nbformat.v4.new_markdown_cell(
        source="### Visual Evidence of Feature Corruption (Tampering vs Benign)\n\n"
               "When an image is maliciously tampered with, the resulting feature vector experiences a massive amount of bit flips (Bit Error Rate). Below, we display an example of the visual tampering and the corresponding hash failure."
    )
    
    code_cell_3 = nbformat.v4.new_code_cell(
        source="showcase_path = Path('output/tampering_showcase.png')\n"
               "if showcase_path.exists():\n"
               "    plt.figure(figsize=(16, 6))\n"
               "    plt.imshow(Image.open(showcase_path))\n"
               "    plt.axis('off')\n"
               "    plt.show()\n"
               "else:\n"
               "    print('Showcase plot not found.')"
    )

    nb.cells.extend([markdown_cell_1, code_cell_1, markdown_cell_2, code_cell_2, markdown_cell_3, code_cell_3])

with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Notebook updated successfully with Attack Simulation documentation and analytical plots.")

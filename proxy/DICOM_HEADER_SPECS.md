# DICOM Header Specifications for Cryptographic Agility

This document specifies the custom DICOM Private Tags layout used by the **MantiQ-Auth** gateway to secure medical images while ensuring cryptographic agility.

---

## 🏷️ Cryptographic Tag Layout

All cryptographic metadata is stored inside **Group `0x0009`** under the Private Creator Identifier **`MantiQAuthCreator`**. 

Standard DICOM viewers ignore these tags as they reside in a private group, allowing the image to remain fully compatible with existing legacy PACS archives.

| Tag Offset | pydicom Tag | Type (VR) | Name / Keyword | Description / Example Value |
| :--- | :--- | :--- | :--- | :--- |
| `0x10` | `(0009, 1010)` | **LO** (Long String) | `Signature Scheme` | The combiner format and algorithms used (e.g., `concatenation:ECDSA-SECP256R1+ML-DSA-65`). |
| `0x20` | `(0009, 1020)` | **LO** (Long String) | `BCH Parameters` | Error correction coding configuration (e.g., `n=1023, t=16`). |
| `0x30` | `(0009, 1030)` | **LO** (Long String) | `Robust Perceptual Hash` | The SHA3-256 digest of the BCH-corrected perceptual features. |
| `0x40` | `(0009, 1040)` | **OB** (Other Byte string) | `Hybrid Signature` | The raw combined signature bytes (ECDSA length prefix + ECDSA signature + ML-DSA signature). |
| `0x50` | `(0009, 1050)` | **LO** (Long String) | `Signature Timestamp` | ISO-8601 UTC timestamp of signature creation (e.g., `2026-06-10T00:21:30Z`). |

---

## 🔍 How to Inspect the Header

We have created an automated utility to dump and analyze these tags:
```powershell
python scripts/inspect_dicom_crypto.py
```

### Manual Inspection via Python
You can also read and print these tags in a raw Python session:

```python
import pydicom

# 1. Load the signed dataset
dataset = pydicom.dcmread("data/pacs/signed_file.dcm")

# 2. Access the private block
block = dataset.private_block(0x0009, "MantiQAuthCreator", create=False)

# 3. Read values
print("Scheme:   ", block[0x10].value.decode('utf-8'))
print("BCH ECC:  ", block[0x20].value.decode('utf-8'))
print("Hash:     ", block[0x30].value.decode('utf-8'))
print("Timestamp:", block[0x50].value.decode('utf-8'))
print("Sig Size: ", len(block[0x40].value), "bytes")
```

---

## 🔐 Verification Workflow

The verification process follows these cryptographic steps:
1. **Extract private tags:** Parse the private block `MantiQAuthCreator` from the dataset.
2. **Recompute Robust Hash:**
   - Preprocess image pixel data (normalization + resize).
   - Run feature extraction (ResNet-18) to get a 512-dimensional vector.
   - Extract raw bits, apply BCH error correction decoding to repair transmission noise, and compute the SHA3-256 digest.
3. **Verify Signatures:**
   - Extract the signature scheme dynamically from tag `(0009, 1010)`.
   - Separate the ECDSA and ML-DSA signature segments from the byte string in tag `(0009, 1040)`.
   - Verify both signatures over the recomputed hash using the respective public keys.

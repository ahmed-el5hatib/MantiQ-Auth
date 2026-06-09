# Crypto-Agile DICOM Proxy (Middleware)

This directory contains the implementation of the **Crypto-Agile DICOM Proxy** for securing medical images in PACS infrastructures.

## Architecture & Communication Flow

The proxy operates as an intermediary between the diagnostic modality (e.g., CT/X-Ray machines) and the PACS storage server, intercepting DICOM network communications:

```text
[Modality (mock_modality.py)] 
       │ 
       ▼ (C-STORE Request to Port 11112)
[DICOM Cryptographic Proxy (proxy_server.py)] 
       │ ──► Extracts Features & Signs using configured cryptographic suite in config.yaml
       ▼ (C-STORE Request to Port 11113)
[PACS Server (mock_pacs.py)] ──► Saves signed files to data/pacs/
```

## Cryptographic Agility & DICOM Metadata

The proxy embeds the cryptographic signature and metadata directly into the DICOM file headers using **DICOM Private Tags**. This ensures that the cryptographic scheme remains **agile**:
- The signature payload specifies the signing parameters (e.g., `"ML-DSA-65 + ECDSA-P256"`).
- The verification engine reads this metadata dynamically to choose the correct verification algorithms, enabling backward compatibility and seamless upgrades of cryptographic algorithms.

Private Tags used under Creator ID `"MantiQAuthCreator"`:
- `(0009, 1001)`: Creator ID (String: `"MantiQAuthCreator"`)
- `(0009, 1010)`: Signature Scheme (e.g. `"ML-DSA-65 + ECDSA-P256"`)
- `(0009, 1020)`: BCH Parameters (e.g. `"n=1023, t=16"`)
- `(0009, 1030)`: Robust Perceptual Hash (Hex string)
- `(0009, 1040)`: Hybrid Signature Payload (Base64 string)
- `(0009, 1050)`: Signing Timestamp

## Setup & Running the Simulation

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### 1. Start the Simulated PACS Server
The PACS server listens for incoming C-STORE requests on port `11113` and saves files to `data/pacs/`:
```bash
python proxy/mock_pacs.py
```

### 2. Start the Cryptographic Proxy
The proxy listens on port `11112`. It signs incoming files and forwards them to port `11113`:
```bash
python proxy/proxy_server.py
```

### 3. Run the Modality Simulation
The modality loads a test DICOM file and pushes it to the proxy at port `11112`:
```bash
python proxy/mock_modality.py
```

### 4. Run the Performance Benchmarks
To evaluate the network and processing latency introduced by the proxy:
```bash
python scripts/run_proxy_benchmark.py
```

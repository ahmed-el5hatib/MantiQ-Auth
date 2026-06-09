# MantiQ-Auth: Quantum-Resistant Medical Image Authentication
# Multi-stage build for reproducibility

FROM python:3.11-slim AS base

# System dependencies for OpenCV, liboqs, and medical imaging
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Build liboqs from source ────────────────────────────────────────────────
FROM base AS liboqs-builder

RUN git clone --depth 1 --branch main https://github.com/open-quantum-safe/liboqs.git /tmp/liboqs \
    && cd /tmp/liboqs \
    && mkdir build && cd build \
    && cmake -GNinja \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        .. \
    && ninja \
    && ninja install

# ── Final image ─────────────────────────────────────────────────────────────
FROM base AS runtime

# Copy liboqs shared libraries
COPY --from=liboqs-builder /usr/local/lib/liboqs* /usr/local/lib/
COPY --from=liboqs-builder /usr/local/include/oqs /usr/local/include/oqs
RUN ldconfig

WORKDIR /app

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Create necessary directories
RUN mkdir -p data/dicom_raw data/dicom_processed output models_cache

# Default command: run the demo
CMD ["python", "demo.py"]

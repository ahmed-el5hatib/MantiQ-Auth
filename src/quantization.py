"""
MantiQ-Auth: Fixed-Point Quantization Module

Implements deterministic integer inference (int8 fixed-point) for ResNet-18 features:
- Calibrates scale factors across a set of calibration images.
- Quantizes float features to int8: x_int8 = clamp(round(x_float / scale), -128, 127)
- Ensures identical binary fingerprints across platforms (CPU vs GPU).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Union

import numpy as np
import torch

from src.utils import get_project_root

logger = logging.getLogger("mantiq.quantization")


class FixedPointQuantizer:
    """Manages calibration, fixed-point quantization, and deterministic mapping."""

    def __init__(self, calibration_file: Optional[Union[str, Path]] = None):
        if calibration_file is None:
            root = get_project_root()
            self.calibration_file = root / "output" / "quantization_calibration.json"
        else:
            self.calibration_file = Path(calibration_file)

        self.scales: Optional[np.ndarray] = None
        self._load_calibration()

    def _load_calibration(self) -> None:
        """Load calibration scales if the file exists."""
        if self.calibration_file.exists():
            try:
                with open(self.calibration_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.scales = np.array(data["scales"], dtype=np.float32)
                logger.info("Loaded quantization scales from %s", self.calibration_file.name)
            except Exception as e:
                logger.warning("Failed to load calibration file: %s", e)

    def calibrate(self, image_paths: List[Path], feature_extractor_fn) -> None:
        """
        Calibrate quantization scaling factors by running float inference.
        Finds the maximum absolute value per feature dimension and maps to 127.
        """
        logger.info("Calibrating fixed-point quantization on %d images...", len(image_paths))
        features_list = []

        for p in image_paths:
            try:
                # Extract float features
                feat = feature_extractor_fn(p)
                features_list.append(feat)
            except Exception as e:
                logger.warning("Failed to extract features during calibration for %s: %s", p.name, e)

        if not features_list:
            raise ValueError("No features extracted successfully during calibration.")

        features_matrix = np.vstack(features_list) # Shape: (N, D)
        
        # Find max absolute value per dimension
        max_vals = np.max(np.abs(features_matrix), axis=0)
        
        # Prevent division by zero
        eps = 1e-8
        scales = np.where(max_vals > eps, max_vals / 127.0, eps / 127.0).astype(np.float32)
        
        self.scales = scales

        # Save to file
        self.calibration_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.calibration_file, "w", encoding="utf-8") as f:
            json.dump({"scales": scales.tolist(), "num_images": len(image_paths)}, f, indent=2)

        logger.info("Calibration complete. Saved scales to %s", self.calibration_file.name)

    def quantize(self, float_features: np.ndarray) -> np.ndarray:
        """
        Quantize float features to int8 fixed-point, then reconstruct them
        to keep pipeline compatibility while ensuring deterministic quantization.
        """
        if self.scales is None:
            logger.warning("Quantizer not calibrated. Using default global scale 1/127.")
            # Fallback to global scale based on L2 norm (which is 1.0 for normalized features)
            self.scales = np.ones_like(float_features, dtype=np.float32) * (1.0 / 127.0)

        # Scale and round
        scaled = float_features / self.scales
        quantized = np.clip(np.round(scaled), -128, 127).astype(np.int8)

        # Reconstruct (dequantize) for the rest of the float pipeline
        reconstructed = quantized.astype(np.float32) * self.scales
        return reconstructed

    def verify_determinism(self, float_features: np.ndarray, device: Union[str, torch.device]) -> bool:
        """
        Simulate/verify bit-identity of binary fingerprint after fixed-point quantization.
        """
        # Run quantization on numpy (representing CPU)
        quantized_cpu = np.clip(np.round(float_features / self.scales), -128, 127).astype(np.int8)
        
        # Run quantization in PyTorch on target device (e.g. GPU)
        device = torch.device(device)
        tensor_feat = torch.from_numpy(float_features).to(device)
        tensor_scales = torch.from_numpy(self.scales).to(device)
        
        tensor_quantized = torch.clamp(torch.round(tensor_feat / tensor_scales), -128, 127).to(torch.int8)
        quantized_gpu = tensor_quantized.cpu().numpy()

        # Check bit-identity
        identical = np.array_equal(quantized_cpu, quantized_gpu)
        logger.info("Quantization determinism check on %s: %s", device, "SUCCESS" if identical else "FAILED")
        return identical

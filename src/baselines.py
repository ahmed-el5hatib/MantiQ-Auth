"""
MantiQ-Auth: Baseline Methods for Comparison

This module implements the baseline authentication and feature extraction methods
used to evaluate the performance of MantiQ-Auth against classical approaches.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image

from src.feature_extraction import extract_resnet_features


def simple_hash_baseline(image_path: Union[str, Path]) -> str:
    """
    Baseline A: Simple Hash + ECDSA
    SHA256 of raw pixel bytes (no robustness, completely fragile).
    """
    image_path = Path(image_path)
    with Image.open(image_path) as img:
        # Load raw pixel data in grayscale to be consistent
        gray_img = img.convert("L")
        pixel_bytes = gray_img.tobytes()
    return hashlib.sha256(pixel_bytes).hexdigest()


def simple_hash_bits(image_path: Union[str, Path]) -> np.ndarray:
    """
    Unpacks the fragile SHA256 of the image into a 256-bit binary array.
    """
    hex_digest = simple_hash_baseline(image_path)
    h_bytes = bytes.fromhex(hex_digest)
    return np.unpackbits(np.frombuffer(h_bytes, dtype=np.uint8))


def surf_svd_features(image_path: Union[str, Path]) -> np.ndarray:
    """
    Baseline B: SURF + SVD (Taj et al. PLOS ONE 2024)
    Extract keypoints, apply SVD, and return a binary hash/descriptor.
    
    If SURF is not available in OpenCV, SIFT/ORB is used as a fallback.
    If keypoints are insufficient, SVD of the resized image itself is computed.
    """
    image_path = Path(image_path)
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.array(Image.open(image_path).convert("L"))

    descriptors = None
    try:
        # Attempt SURF (using xfeatures2d)
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
        _, descriptors = surf.detectAndCompute(img, None)
    except (AttributeError, cv2.error):
        # Fallback to SIFT
        try:
            sift = cv2.SIFT_create()
            _, descriptors = sift.detectAndCompute(img, None)
        except (AttributeError, cv2.error):
            # Fallback to ORB
            try:
                orb = cv2.ORB_create()
                _, descriptors = orb.detectAndCompute(img, None)
            except Exception:
                pass
    except Exception:
        pass

    # Process descriptors or fallback to direct SVD of image structure
    if descriptors is not None and len(descriptors) > 1:
        # Compute SVD of the descriptors matrix
        try:
            _, S, Vt = np.linalg.svd(descriptors.astype(np.float32), full_matrices=False)
            feats = Vt.flatten()
            # Standardize length to 384 bits
            if len(feats) < 384:
                feats = np.pad(feats, (0, 384 - len(feats)), 'constant')
            else:
                feats = feats[:384]
        except Exception:
            descriptors = None

    if descriptors is None or len(descriptors) <= 1:
        # Image-based SVD fallback
        img_resized = cv2.resize(img, (64, 64))
        try:
            U, S, Vt = np.linalg.svd(img_resized.astype(np.float32), full_matrices=False)
            # Combine singular values and primary components
            feats = np.concatenate([S, U[:, :3].flatten(), Vt[:3, :].flatten()])
            if len(feats) < 384:
                feats = np.pad(feats, (0, 384 - len(feats)), 'constant')
            else:
                feats = feats[:384]
        except Exception:
            # Fallback of fallbacks: simple pixel array
            feats = np.linspace(0.0, 1.0, 384)

    # Quantize using median thresholding to get a robust binary perceptual hash
    threshold = np.median(feats)
    binary_bits = (feats > threshold).astype(np.uint8)
    return binary_bits


def extract_resnet18_features(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Baseline C: ResNet-18 (frozen, same as ViT but different backbone)
    512-dim features from torchvision.models.resnet18 (pretrained, frozen).
    """
    return extract_resnet_features(image_tensor)

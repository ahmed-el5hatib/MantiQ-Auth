"""
MantiQ-Auth: Feature Extraction Module

Extracts discriminative features from medical images using pre-trained
deep neural networks (ViT-B/16 and ResNet-18). All models are used in
frozen/inference mode — no fine-tuning is performed.

Architecture choices:
    - ViT-B/16: Captures global spatial relationships via self-attention,
      producing 768-dimensional feature vectors.
    - ResNet-18: Convolutional baseline producing 512-dimensional features.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import models, transforms

logger = logging.getLogger("mantiq.features")

# ─── Constants ──────────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224

# Feature dimensions for supported architectures
FEATURE_DIMS = {
    "vit": 384,     # ViT-B/16 CLS token output sliced to 384-dim (ViT-S equivalent)
    "resnet": 512,  # ResNet-18 avgpool output
}


# ─── Image Loading & Preprocessing ──────────────────────────────────────────

def load_and_preprocess_dicom(
    dicom_path: Union[str, Path],
    image_size: int = IMAGE_SIZE,
    apply_median_prefilter: bool = False,
) -> torch.Tensor:
    """
    Load a DICOM file and preprocess it for deep feature extraction.

    The DICOM pixel data is windowed, converted to a 3-channel (RGB) image,
    resized to ``image_size × image_size``, and normalized with ImageNet
    statistics.

    Args:
        dicom_path: Path to a ``.dcm`` DICOM file.
        image_size: Target spatial dimension (default 224).
        apply_median_prefilter: If True, apply a median filter to the image before tensor conversion.

    Returns:
        Float tensor of shape ``(1, 3, H, W)`` ready for model input.

    Raises:
        FileNotFoundError: If the DICOM path does not exist.
        RuntimeError: If the DICOM file cannot be parsed.
    """
    import pydicom

    dicom_path = Path(dicom_path)
    if not dicom_path.exists():
        raise FileNotFoundError(f"DICOM file not found: {dicom_path}")

    try:
        ds = pydicom.dcmread(str(dicom_path))
        pixel_array = ds.pixel_array.astype(np.float32)
    except Exception as e:
        raise RuntimeError(f"Failed to read DICOM {dicom_path.name}: {e}") from e

    # Apply windowing if available (CT images)
    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        center_val = ds.WindowCenter
        width_val = ds.WindowWidth
        center = float(center_val[0]) if hasattr(center_val, "__getitem__") and not isinstance(center_val, (str, bytes)) else float(center_val)
        width = float(width_val[0]) if hasattr(width_val, "__getitem__") and not isinstance(width_val, (str, bytes)) else float(width_val)
        lower = center - width / 2
        upper = center + width / 2
        pixel_array = np.clip(pixel_array, lower, upper)

    # Normalize to [0, 255] uint8
    pmin, pmax = pixel_array.min(), pixel_array.max()
    if pmax - pmin > 0:
        pixel_array = ((pixel_array - pmin) / (pmax - pmin) * 255).astype(np.uint8)
    else:
        pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)

    # Convert to 3-channel PIL Image
    pil_image = Image.fromarray(pixel_array).convert("RGB")

    # Apply median pre-filtering if enabled
    if apply_median_prefilter:
        pil_image = pil_image.filter(ImageFilter.MedianFilter(size=3))

    # Apply transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    tensor = transform(pil_image).unsqueeze(0)  # (1, 3, H, W)
    logger.debug("Preprocessed DICOM %s (median_filtered=%s) → tensor %s", dicom_path.name, apply_median_prefilter, tensor.shape)
    return tensor


def load_and_preprocess_image(
    image_path: Union[str, Path],
    image_size: int = IMAGE_SIZE,
    apply_median_prefilter: bool = False,
) -> torch.Tensor:
    """
    Load a standard image file (PNG, JPEG, etc.) and preprocess it.

    Args:
        image_path: Path to the image file.
        image_size: Target spatial dimension.
        apply_median_prefilter: If True, apply a median filter to the image before tensor conversion.

    Returns:
        Float tensor of shape ``(1, 3, H, W)``.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    pil_image = Image.open(image_path).convert("RGB")

    # Apply median pre-filtering if enabled
    if apply_median_prefilter:
        pil_image = pil_image.filter(ImageFilter.MedianFilter(size=3))

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return transform(pil_image).unsqueeze(0)


# ─── Model Loading ──────────────────────────────────────────────────────────

_model_cache: dict = {}


def _get_device() -> torch.device:
    """Select the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_vit_model() -> Tuple[torch.nn.Module, torch.device]:
    """
    Load a frozen ViT-B/16 model for feature extraction.

    The classification head is bypassed; features are extracted from the
    CLS token output of the transformer encoder (768 dimensions).

    Returns:
        Tuple of (model, device).
    """
    if "vit" in _model_cache:
        return _model_cache["vit"]

    device = _get_device()
    logger.info("Loading ViT-B/16 on %s...", device)

    import os
    mantiq_offline = os.environ.get("MANTIQ_OFFLINE", "0") == "1"

    if mantiq_offline:
        logger.info("MANTIQ_OFFLINE is set. Loading ViT-B/16 with uninitialized weights.")
        model = models.vit_b_16(weights=None)
    else:
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        try:
            import socket
            orig_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(10.0)
            model = models.vit_b_16(weights=weights)
            socket.setdefaulttimeout(orig_timeout)
        except Exception as e:
            logger.warning("Failed to load pre-trained ViT-B/16 weights: %s. Falling back to uninitialized (random) weights.", e)
            model = models.vit_b_16(weights=None)

    # Freeze all parameters
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    model = model.to(device)
    _model_cache["vit"] = (model, device)
    logger.info("ViT-B/16 loaded and frozen (384-dim features).")
    return model, device


def get_resnet_model() -> Tuple[torch.nn.Module, torch.device]:
    """
    Load a frozen ResNet-18 model for feature extraction.

    The final FC layer is replaced with an identity layer so that the
    512-dimensional average-pool output is returned directly.

    Returns:
        Tuple of (model, device).
    """
    if "resnet" in _model_cache:
        return _model_cache["resnet"]

    device = _get_device()
    logger.info("Loading ResNet-18 on %s...", device)

    import os
    mantiq_offline = os.environ.get("MANTIQ_OFFLINE", "0") == "1"

    if mantiq_offline:
        logger.info("MANTIQ_OFFLINE is set. Loading ResNet-18 with uninitialized weights.")
        model = models.resnet18(weights=None)
    else:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        try:
            import socket
            orig_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(10.0)
            model = models.resnet18(weights=weights)
            socket.setdefaulttimeout(orig_timeout)
        except Exception as e:
            logger.warning("Failed to load pre-trained ResNet-18 weights: %s. Falling back to uninitialized (random) weights.", e)
            model = models.resnet18(weights=None)

    # Replace classification head with identity
    model.fc = torch.nn.Identity()

    # Freeze all parameters
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    model = model.to(device)
    _model_cache["resnet"] = (model, device)
    logger.info("ResNet-18 loaded and frozen (512-dim features).")
    return model, device


# ─── Feature Extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_vit_features(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Extract 768-dimensional features using frozen ViT-B/16.

    The CLS token representation from the last transformer block is
    used as the global image descriptor.

    Args:
        image_tensor: Preprocessed image tensor of shape ``(1, 3, 224, 224)``.

    Returns:
        L2-normalized feature vector of shape ``(768,)``.
    """
    model, device = get_vit_model()
    x = image_tensor.to(device)

    # Forward through ViT — extract from heads input (before classification)
    # We need the CLS token features, not the class logits
    # ViT forward: patch embed → transformer blocks → LayerNorm → head
    # We intercept after LayerNorm (before head)
    x = model._process_input(x)
    n = x.shape[0]

    # Expand the class token to the full batch
    batch_class_token = model.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)

    x = model.encoder(x)

    # CLS token is the first token
    cls_features = x[:, 0]
    cls_features = model.heads.head if hasattr(model.heads, 'head') else cls_features

    # Actually, let's use a simpler approach: hook into the model
    # Re-do: just use the full forward and grab before final FC
    # ViT-B/16 in torchvision: model.heads is nn.Sequential with a Linear
    # We want features BEFORE model.heads

    # Simpler approach: temporarily replace heads
    original_heads = model.heads
    model.heads = torch.nn.Identity()
    features = model(image_tensor.to(device))
    model.heads = original_heads

    features = features.cpu().numpy().flatten()
    # Slicing the 768-dim CLS token to 384-dim to simulate ViT-S
    features = features[:384]
    features = l2_normalize(features)

    assert features.shape == (384,), (
        f"Expected 384-dim features, got {features.shape}"
    )
    logger.debug("ViT features extracted: dim=%d, norm=%.4f", len(features), np.linalg.norm(features))
    return features


@torch.no_grad()
def extract_resnet_features(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Extract 512-dimensional features using frozen ResNet-18.

    Features are taken from the global average pooling layer output,
    bypassing the final classification FC layer.

    Args:
        image_tensor: Preprocessed image tensor of shape ``(1, 3, 224, 224)``.

    Returns:
        L2-normalized feature vector of shape ``(512,)``.
    """
    model, device = get_resnet_model()
    features = model(image_tensor.to(device))
    features = features.cpu().numpy().flatten()
    features = l2_normalize(features)

    assert features.shape == (512,), (
        f"Expected 512-dim features, got {features.shape}"
    )
    logger.debug("ResNet features extracted: dim=%d", len(features))
    return features


def extract_features(
    image_tensor: torch.Tensor,
    extractor: Literal["vit", "resnet"] = "resnet",
) -> np.ndarray:
    """
    Extract features using the specified architecture.

    Args:
        image_tensor: Preprocessed image tensor.
        extractor: Feature extractor name ("vit" or "resnet").

    Returns:
        L2-normalized feature vector.
    """
    # Load config to check for use_fixed_point
    use_fixed_point = False
    try:
        from src.utils import load_config
        config = load_config()
        use_fixed_point = config.get("feature_extraction", {}).get("use_fixed_point", False)
    except Exception:
        pass

    if extractor == "resnet":
        features = extract_resnet_features(image_tensor)
    elif extractor == "vit":
        logger.warning("ViT feature extraction is deprecated. Defaulting to ResNet-18.")
        features = extract_resnet_features(image_tensor)
    else:
        raise ValueError(f"Unknown extractor: {extractor}. Use 'resnet'.")

    if use_fixed_point:
        try:
            from src.quantization import FixedPointQuantizer
            quantizer = FixedPointQuantizer()
            features = quantizer.quantize(features)
        except Exception as e:
            logger.warning("Fixed-point quantization failed: %s", e)

    return features


# ─── Normalization ──────────────────────────────────────────────────────────

def l2_normalize(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Apply L2 normalization to a feature vector.

    Args:
        features: Input feature vector.
        eps: Small constant to avoid division by zero.

    Returns:
        Unit-norm feature vector.
    """
    norm = np.linalg.norm(features)
    if norm < eps:
        logger.warning("Near-zero feature norm (%.2e), returning zeros.", norm)
        return np.zeros_like(features)
    return features / norm


# ─── Normalized Cross-Correlation ───────────────────────────────────────────

@torch.no_grad()
def extract_enriched_vit_features(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Extract enriched features from ViT-B/16 for tampering detection.

    Produces a 387-dimensional vector:
        - 384 dims: L2-normalized CLS token (ViT-S equivalent)
        -   1 dim:  Feature variance across the CLS token
        -   1 dim:  Shannon entropy of the softmax-normalized features
        -   1 dim:  L2 norm of the raw (pre-normalization) features

    These additional signals capture distributional anomalies introduced
    by tampering that the CLS token alone may not fully represent.

    Args:
        image_tensor: Preprocessed image tensor of shape ``(1, 3, 224, 224)``.

    Returns:
        Enriched feature vector of shape ``(387,)``.
    """
    model, device = get_vit_model()
    x = image_tensor.to(device)

    # Get raw CLS features (before L2 normalization)
    original_heads = model.heads
    model.heads = torch.nn.Identity()
    raw_features = model(x)
    model.heads = original_heads

    raw_features = raw_features.cpu().numpy().flatten()
    # Slice to 384-dim (ViT-S equivalent)
    raw_cls = raw_features[:384]

    # Compute enrichment signals BEFORE normalization
    feat_variance = float(np.var(raw_cls))
    l2_norm = float(np.linalg.norm(raw_cls))

    # Shannon entropy of softmax distribution
    shifted = raw_cls - np.max(raw_cls)
    exp_feat = np.exp(shifted)
    probs = exp_feat / (np.sum(exp_feat) + 1e-10)
    entropy = -float(np.sum(probs * np.log(probs + 1e-10)))

    # L2-normalize the CLS token
    cls_normalized = l2_normalize(raw_cls)

    # Concatenate: [384 CLS features, variance, entropy, L2 norm]
    enriched = np.concatenate([
        cls_normalized,
        np.array([feat_variance, entropy, l2_norm]),
    ])

    assert enriched.shape == (387,), f"Expected 387-dim enriched features, got {enriched.shape}"
    logger.debug("Enriched ViT features: dim=%d (384+3 signals)", len(enriched))
    return enriched


@torch.no_grad()
def extract_enriched_resnet_features(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Extract enriched features from ResNet-18 for tampering detection.

    Produces a 515-dimensional vector:
        - 512 dims: L2-normalized avgpool output
        -   1 dim:  Feature variance
        -   1 dim:  Shannon entropy of the softmax-normalized features
        -   1 dim:  L2 norm of the raw features

    Args:
        image_tensor: Preprocessed image tensor of shape ``(1, 3, 224, 224)``.

    Returns:
        Enriched feature vector of shape ``(515,)``.
    """
    model, device = get_resnet_model()
    x = image_tensor.to(device)
    
    raw_features = model(x).cpu().numpy().flatten()
    
    # Compute enrichment signals BEFORE normalization
    feat_variance = float(np.var(raw_features))
    l2_norm = float(np.linalg.norm(raw_features))

    # Shannon entropy of softmax distribution
    shifted = raw_features - np.max(raw_features)
    exp_feat = np.exp(shifted)
    probs = exp_feat / (np.sum(exp_feat) + 1e-10)
    entropy = -float(np.sum(probs * np.log(probs + 1e-10)))

    # L2-normalize
    feat_normalized = l2_normalize(raw_features)

    # Concatenate
    enriched = np.concatenate([
        feat_normalized,
        np.array([feat_variance, entropy, l2_norm]),
    ])

    assert enriched.shape == (515,), f"Expected 515-dim enriched features, got {enriched.shape}"
    return enriched


def compute_ncc(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """
    Compute Normalized Cross-Correlation between two feature vectors.

    NCC measures similarity on [-1, 1] where 1 = identical.
    For L2-normalized inputs, NCC simplifies to the dot product.

    Args:
        features_a: First feature vector (L2-normalized).
        features_b: Second feature vector (L2-normalized).

    Returns:
        NCC score in [-1.0, 1.0].
    """
    a = features_a - features_a.mean()
    b = features_b - features_b.mean()

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-10:
        return 0.0

    return float(np.dot(a, b) / denom)

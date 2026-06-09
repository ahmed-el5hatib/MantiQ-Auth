"""
MantiQ-Auth: Utility functions for configuration, logging, and I/O.

Provides shared helpers used across all modules of the MantiQ-Auth
quantum-resistant medical image authentication system.
"""

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


# ─── Logging ────────────────────────────────────────────────────────────────

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Configure project-wide logging with console and optional file output.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to a log file.

    Returns:
        Configured logger instance.
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    logger = logging.getLogger("mantiq")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler (optional)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─── Configuration ──────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary of configuration values.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_project_root() -> Path:
    """Return the project root directory (where config.yaml lives)."""
    # Walk up from this file's location to find config.yaml
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "config.yaml").exists():
            return current
        current = current.parent

    # Fallback: use CWD
    return Path.cwd()


def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Create all required data directories from config.

    Args:
        config: Loaded configuration dictionary.
    """
    root = get_project_root()
    paths = config.get("paths", {})
    for key, rel_path in paths.items():
        if key.endswith("_csv"):
            # Ensure parent directory of CSV files exists
            full_path = root / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            full_path = root / rel_path
            full_path.mkdir(parents=True, exist_ok=True)


# ─── Timing ─────────────────────────────────────────────────────────────────

class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, label: str = ""):
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self.elapsed = time.perf_counter() - self._start

    def __repr__(self) -> str:
        return f"Timer({self.label}: {self.elapsed:.4f}s)"


# ─── File I/O ───────────────────────────────────────────────────────────────

def save_json(data: Any, filepath: Union[str, Path]) -> None:
    """
    Save data as a formatted JSON file.

    Args:
        data: JSON-serializable object.
        filepath: Output file path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: Union[str, Path]) -> Any:
    """
    Load data from a JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Parsed JSON data.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_file_hash(filepath: Union[str, Path], algorithm: str = "sha256") -> str:
    """
    Compute cryptographic hash of a file's contents.

    Args:
        filepath: Path to the file.
        algorithm: Hash algorithm name (sha256, sha3_256, md5, etc.).

    Returns:
        Hex digest string.
    """
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── Byte Formatting ────────────────────────────────────────────────────────

def format_bytes(size_bytes: int) -> str:
    """
    Format byte count into human-readable string.

    Args:
        size_bytes: Number of bytes.

    Returns:
        Formatted string (e.g., "1.23 MB").
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

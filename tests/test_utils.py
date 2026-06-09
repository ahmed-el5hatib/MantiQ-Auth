"""
MantiQ-Auth: Utility Module Tests

Validates:
    1. Logging setup (prevention of duplicate handlers, file logging)
    2. YAML configuration loader
    3. Directory generation helper
    4. Execution timer context manager
    5. JSON I/O helpers
    6. File cryptographic hashing
    7. Human-readable byte size formatter
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (
    setup_logging,
    load_config,
    get_project_root,
    ensure_directories,
    Timer,
    save_json,
    load_json,
    compute_file_hash,
    format_bytes,
)


def test_setup_logging(tmp_path):
    """Test logger configuration."""
    log_file = tmp_path / "test.log"
    logger = setup_logging(level="DEBUG", log_file=str(log_file))

    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.DEBUG

    # Logging to file should create the log file
    logger.info("Test logging message")
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Test logging message" in content


def test_load_config(tmp_path):
    """Test YAML config loading."""
    config_data = {
        "mantiq": {
            "feature_extractor": "vit",
            "mldsa_level": 65,
        },
        "paths": {
            "raw_dir": "data/dicom_raw",
        }
    }
    cfg_file = tmp_path / "test_config.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f)

    loaded = load_config(str(cfg_file))
    assert loaded["mantiq"]["feature_extractor"] == "vit"
    assert loaded["paths"]["raw_dir"] == "data/dicom_raw"

    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_config_file_xyz.yaml")


def test_get_project_root():
    """Verify project root retrieval returns a Path."""
    root = get_project_root()
    assert isinstance(root, Path)
    assert root.exists()


def test_ensure_directories(tmp_path, monkeypatch):
    """Verify that ensure_directories creates paths relative to project root."""
    config = {
        "paths": {
            "test_dir": "data/test_temp_subdir_1",
            "test_file_csv": "data/test_temp_subdir_2/test.csv"
        }
    }

    # Mock get_project_root to use tmp_path
    monkeypatch.setattr("src.utils.get_project_root", lambda: tmp_path)

    ensure_directories(config)

    assert (tmp_path / "data/test_temp_subdir_1").exists()
    assert (tmp_path / "data/test_temp_subdir_2").exists()


def test_timer_context_manager():
    """Verify Timer measures elapsed time correctly."""
    import time
    with Timer("Sleep Test") as timer:
        time.sleep(0.05)

    assert timer.elapsed >= 0.04
    assert "Sleep Test" in repr(timer)


def test_json_io(tmp_path):
    """Verify save_json and load_json operate correctly."""
    data = {"test_key": "test_value", "nested": [1, 2, 3]}
    filepath = tmp_path / "test_data.json"

    save_json(data, filepath)
    assert filepath.exists()

    loaded = load_json(filepath)
    assert loaded["test_key"] == "test_value"
    assert loaded["nested"] == [1, 2, 3]


def test_compute_file_hash(tmp_path):
    """Verify cryptographic hash of file contents."""
    filepath = tmp_path / "hash_test.txt"
    filepath.write_bytes(b"hello world")

    # SHA256 of "hello world"
    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    digest = compute_file_hash(filepath, "sha256")
    assert digest == expected


def test_format_bytes():
    """Verify human-readable byte count formatting."""
    assert format_bytes(500) == "500.00 B"
    assert format_bytes(2048) == "2.00 KB"
    assert format_bytes(1024 * 1024 * 3) == "3.00 MB"
    assert format_bytes(1024 * 1024 * 1024 * 5) == "5.00 GB"

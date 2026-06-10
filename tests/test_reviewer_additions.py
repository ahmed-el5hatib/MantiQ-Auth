"""
Unit tests for reviewer-requested features: SoftwareHSM and FixedPointQuantizer.
"""

from __future__ import annotations

import os
import tempfile
import json
from pathlib import Path
import numpy as np
import pytest

from src.key_management import SoftwareHSM
from src.quantization import FixedPointQuantizer


def test_software_hsm():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        store_path = tmpdir_path / "key_store.enc"
        rev_path = tmpdir_path / "revocation_list.json"
        
        password = "test_secure_password_123"
        hsm = SoftwareHSM(store_path, rev_path)
        
        # Test Initial Key Generation
        active_id = hsm.generate_and_store_initial_keys(password)
        assert store_path.exists()
        assert active_id is not None
        assert active_id.startswith("key_")
        
        # Test Load Active Key
        loaded_id, ecdsa_kp, mldsa_kp = hsm.load_active_keys(password)
        assert loaded_id == active_id
        assert ecdsa_kp is not None
        assert mldsa_kp is not None
        
        # Test Load with Incorrect Password
        with pytest.raises(Exception):
            hsm.load_active_keys("wrong_password")
            
        # Test Rotation
        rotated_id = hsm.rotate_keys(password)
        assert rotated_id != active_id
        
        # Verify rotated key is now active
        loaded_id_rotated, _, _ = hsm.load_active_keys(password)
        assert loaded_id_rotated == rotated_id
        
        # Verify historical key can still be loaded
        hist_ecdsa, hist_mldsa = hsm.load_historical_key(password, active_id)
        assert hist_ecdsa is not None
        assert hist_mldsa is not None
        
        # Test Revocation
        assert not hsm.is_key_revoked(active_id)
        hsm.revoke_key(active_id)
        assert hsm.is_key_revoked(active_id)
        
        # Loading a revoked key should raise ValueError
        with pytest.raises(ValueError, match="revoked"):
            hsm.load_historical_key(password, active_id)


def test_fixed_point_quantizer():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        cal_path = tmpdir_path / "calibration.json"
        
        quantizer = FixedPointQuantizer(cal_path)
        assert quantizer.scales is None
        
        # Dummy feature extractor function
        dummy_feat = np.array([0.5, -0.2, 0.8, -0.9, 0.0], dtype=np.float32)
        def dummy_extractor(path):
            return dummy_feat
            
        # Calibrate
        dummy_paths = [Path("dummy1.png"), Path("dummy2.png")]
        quantizer.calibrate(dummy_paths, dummy_extractor)
        
        assert cal_path.exists()
        assert quantizer.scales is not None
        assert len(quantizer.scales) == 5
        
        # Check scales are calibrated properly (max absolute value / 127)
        expected_scales = np.array([0.5, 0.2, 0.8, 0.9, 1e-8], dtype=np.float32) / 127.0
        np.testing.assert_allclose(quantizer.scales, expected_scales, rtol=1e-5)
        
        # Test Quantization and Reconstruction
        test_feat = np.array([0.25, -0.1, 0.4, -0.45, 0.0], dtype=np.float32)
        reconstructed = quantizer.quantize(test_feat)
        
        # Expected: test_feat / expected_scales -> round -> clamp to int8 -> scale back
        scaled = test_feat / expected_scales
        q_int8 = np.clip(np.round(scaled), -128, 127).astype(np.int8)
        expected_reconstructed = q_int8.astype(np.float32) * expected_scales
        
        np.testing.assert_allclose(reconstructed, expected_reconstructed, rtol=1e-5)
        
        # Verify Determinism on CPU
        assert quantizer.verify_determinism(test_feat, "cpu")

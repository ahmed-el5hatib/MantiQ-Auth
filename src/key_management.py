"""
MantiQ-Auth: Key Management Module (Software HSM)

Provides secure software-based HSM capabilities:
- AES-256-GCM encryption of keys using master passwords (PBKDF2 KDF).
- ECDSA-P256 and ML-DSA-65 key generation, storage, and retrieval.
- Key rotation (maintaining historical keys for verification).
- Key revocation tracking.
"""

from __future__ import annotations

import os
import json
import time
import uuid
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from src.hybrid_signatures import (
    ECDSAKeyPair,
    MLDSAKeyPair,
    generate_keypair_ecdsa,
    generate_keypair_mldsa,
)

logger = logging.getLogger("mantiq.hsm")


def derive_key(password: str, salt: bytes, iterations: int = 100000) -> bytes:
    """Derive a 256-bit AES key from a password using PBKDF2HMAC."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=iterations,
    )
    return kdf.derive(password.encode())


def encrypt_data(plaintext: bytes, password: str) -> bytes:
    """Encrypt plaintext bytes using AES-256-GCM with a PBKDF2 key derived from password."""
    salt = os.urandom(16)
    nonce = os.urandom(12)
    key = derive_key(password, salt)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)
    # Return concatenation of salt, nonce, and ciphertext
    return salt + nonce + ciphertext


def decrypt_data(encrypted_data: bytes, password: str) -> bytes:
    """Decrypt ciphertext bytes using AES-256-GCM with a PBKDF2 key derived from password."""
    if len(encrypted_data) < 28:
        raise ValueError("Encrypted data too short.")
    salt = encrypted_data[:16]
    nonce = encrypted_data[16:28]
    ciphertext = encrypted_data[28:]
    key = derive_key(password, salt)
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None)


class SoftwareHSM:
    """Software-based Key Management and HSM simulating secure hardware."""

    def __init__(self, key_store_path: Union[str, Path], revocation_list_path: Union[str, Path]):
        self.key_store_path = Path(key_store_path)
        self.revocation_list_path = Path(revocation_list_path)

    def generate_and_store_initial_keys(self, password: str, mldsa_level: int = 65, ecdsa_curve: str = "SECP256R1") -> str:
        """Generate a fresh hybrid key pair and store it encrypted in a new store."""
        ecdsa_kp = generate_keypair_ecdsa(ecdsa_curve)
        mldsa_kp = generate_keypair_mldsa(mldsa_level)

        key_id = f"key_{uuid.uuid4().hex[:12]}_{int(time.time())}"
        
        store_data = {
            "active_key_id": key_id,
            "keys": {
                key_id: {
                    "id": key_id,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "ecdsa_private_key_der": ecdsa_kp.private_bytes().hex(),
                    "ecdsa_public_key_der": ecdsa_kp.public_bytes().hex(),
                    "mldsa_secret_key": mldsa_kp.secret_key.hex(),
                    "mldsa_public_key": mldsa_kp.public_key.hex(),
                    "mldsa_algorithm": mldsa_kp.algorithm,
                    "ecdsa_curve": ecdsa_curve
                }
            }
        }

        plaintext = json.dumps(store_data).encode("utf-8")
        encrypted = encrypt_data(plaintext, password)

        self.key_store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.key_store_path, "wb") as f:
            f.write(encrypted)
            
        logger.info("Initialized Software HSM and saved active key: %s", key_id)
        return key_id

    def load_active_keys(self, password: str) -> Tuple[str, ECDSAKeyPair, MLDSAKeyPair]:
        """Load and decrypt the active key pair from the key store."""
        if not self.key_store_path.exists():
            raise FileNotFoundError(f"Key store file not found: {self.key_store_path}")

        with open(self.key_store_path, "rb") as f:
            encrypted = f.read()

        try:
            plaintext = decrypt_data(encrypted, password)
            store_data = json.loads(plaintext.decode("utf-8"))
        except Exception as e:
            raise RuntimeError("Failed to decrypt key store. Check password or file integrity.") from e

        active_id = store_data.get("active_key_id")
        if not active_id or active_id not in store_data["keys"]:
            raise ValueError("No active key found in key store.")

        # Check if the active key has been revoked
        if self.is_key_revoked(active_id):
            raise ValueError(f"The active key {active_id} has been revoked!")

        key_entry = store_data["keys"][active_id]
        return active_id, self._reconstruct_ecdsa(key_entry), self._reconstruct_mldsa(key_entry)

    def load_historical_key(self, password: str, key_id: str) -> Tuple[ECDSAKeyPair, MLDSAKeyPair]:
        """Load any key from the store (even older/rotated ones) for signature verification."""
        if not self.key_store_path.exists():
            raise FileNotFoundError(f"Key store file not found: {self.key_store_path}")

        with open(self.key_store_path, "rb") as f:
            encrypted = f.read()

        plaintext = decrypt_data(encrypted, password)
        store_data = json.loads(plaintext.decode("utf-8"))

        if key_id not in store_data.get("keys", {}):
            raise KeyError(f"Key ID {key_id} not found in store.")

        if self.is_key_revoked(key_id):
            raise ValueError(f"The key {key_id} has been revoked!")

        key_entry = store_data["keys"][key_id]
        return self._reconstruct_ecdsa(key_entry), self._reconstruct_mldsa(key_entry)

    def rotate_keys(self, password: str, mldsa_level: int = 65, ecdsa_curve: str = "SECP256R1") -> str:
        """Rotate the active keys by generating a new pair, making it active, but preserving historical keys."""
        if not self.key_store_path.exists():
            return self.generate_and_store_initial_keys(password, mldsa_level, ecdsa_curve)

        with open(self.key_store_path, "rb") as f:
            encrypted = f.read()

        plaintext = decrypt_data(encrypted, password)
        store_data = json.loads(plaintext.decode("utf-8"))

        ecdsa_kp = generate_keypair_ecdsa(ecdsa_curve)
        mldsa_kp = generate_keypair_mldsa(mldsa_level)

        new_key_id = f"key_{uuid.uuid4().hex[:12]}_{int(time.time())}"

        store_data["keys"][new_key_id] = {
            "id": new_key_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ecdsa_private_key_der": ecdsa_kp.private_bytes().hex(),
            "ecdsa_public_key_der": ecdsa_kp.public_bytes().hex(),
            "mldsa_secret_key": mldsa_kp.secret_key.hex(),
            "mldsa_public_key": mldsa_kp.public_key.hex(),
            "mldsa_algorithm": mldsa_kp.algorithm,
            "ecdsa_curve": ecdsa_curve
        }
        
        store_data["active_key_id"] = new_key_id

        # Re-encrypt and save
        plaintext = json.dumps(store_data).encode("utf-8")
        encrypted = encrypt_data(plaintext, password)

        with open(self.key_store_path, "wb") as f:
            f.write(encrypted)

        logger.info("Rotated keys. New active key: %s", new_key_id)
        return new_key_id

    def revoke_key(self, key_id: str) -> None:
        """Revoke a specific key by adding its ID to the local revocation list."""
        revocations = []
        if self.revocation_list_path.exists():
            try:
                with open(self.revocation_list_path, "r", encoding="utf-8") as f:
                    revocations = json.load(f)
            except Exception:
                revocations = []

        if key_id not in revocations:
            revocations.append(key_id)

        self.revocation_list_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.revocation_list_path, "w", encoding="utf-8") as f:
            json.dump(revocations, f, indent=2)

        logger.warning("Revoked key: %s (added to revocation list %s)", key_id, self.revocation_list_path.name)

    def is_key_revoked(self, key_id: str) -> bool:
        """Check if a key is revoked."""
        if not self.revocation_list_path.exists():
            return False
        try:
            with open(self.revocation_list_path, "r", encoding="utf-8") as f:
                revocations = json.load(f)
            return key_id in revocations
        except Exception:
            return False

    def _reconstruct_ecdsa(self, key_entry: Dict[str, Any]) -> ECDSAKeyPair:
        """Reconstruct ECDSAKeyPair from hex serialized DER."""
        priv_der = bytes.fromhex(key_entry["ecdsa_private_key_der"])
        pub_der = bytes.fromhex(key_entry["ecdsa_public_key_der"])

        private_key = serialization.load_der_private_key(priv_der, password=None)
        public_key = serialization.load_der_public_key(pub_der)

        assert isinstance(private_key, ec.EllipticCurvePrivateKey)
        assert isinstance(public_key, ec.EllipticCurvePublicKey)
        return ECDSAKeyPair(private_key=private_key, public_key=public_key)

    def _reconstruct_mldsa(self, key_entry: Dict[str, Any]) -> MLDSAKeyPair:
        """Reconstruct MLDSAKeyPair from hex serialized bytes."""
        sk = bytes.fromhex(key_entry["mldsa_secret_key"])
        pk = bytes.fromhex(key_entry["mldsa_public_key"])
        algo = key_entry["mldsa_algorithm"]

        return MLDSAKeyPair(
            algorithm=algo,
            secret_key=sk,
            public_key=pk,
            _signer=None,
        )

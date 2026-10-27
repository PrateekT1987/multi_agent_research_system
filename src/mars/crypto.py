"""Optional at-rest encryption for state, dead letters and approval payloads.

Encryption is deliberately opt-in: a single-tenant developer run should not
need a key, while a production deployment can set ``MARS_ENCRYPTION_KEY`` and
have every payload that contains topics, prompts or findings written as
Fernet ciphertext.  Keys may be a valid Fernet key or any high-entropy string
(the latter is stretched with SHA-256).
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

__all__ = ["ENCRYPTED_PREFIX", "PayloadCipher"]

ENCRYPTED_PREFIX = "MARSENC1:"


def _normalise_key(key: str | bytes) -> bytes:
    raw = key.encode("utf-8") if isinstance(key, str) else key
    try:
        base64.urlsafe_b64decode(raw)
        if len(base64.urlsafe_b64decode(raw)) == 32:
            return raw
    except Exception:
        pass
    return base64.urlsafe_b64encode(hashlib.sha256(raw).digest())


class PayloadCipher:
    """Encrypt/decrypt JSON-ish payloads when a key is configured."""

    def __init__(self, key: str | bytes | None = None) -> None:
        self._fernet = None
        if key:
            from cryptography.fernet import Fernet

            self._fernet = Fernet(_normalise_key(key))

    @property
    def enabled(self) -> bool:
        return self._fernet is not None

    @classmethod
    def from_key(cls, key: str | bytes | None) -> PayloadCipher:
        return cls(key)

    def encrypt_bytes(self, data: bytes) -> bytes:
        if self._fernet is None:
            return data
        return ENCRYPTED_PREFIX.encode("utf-8") + self._fernet.encrypt(data)

    def decrypt_bytes(self, data: bytes) -> bytes:
        if self._fernet is None or not data.startswith(ENCRYPTED_PREFIX.encode("utf-8")):
            return data
        return self._fernet.decrypt(data[len(ENCRYPTED_PREFIX) :])

    def encrypt_text(self, text: str) -> str:
        if self._fernet is None:
            return text
        return ENCRYPTED_PREFIX + self._fernet.encrypt(text.encode("utf-8")).decode("utf-8")

    def decrypt_text(self, text: str) -> str:
        if self._fernet is None or not text.startswith(ENCRYPTED_PREFIX):
            return text
        return self._fernet.decrypt(text[len(ENCRYPTED_PREFIX) :].encode("utf-8")).decode("utf-8")

    def encrypt_json(self, value: Any) -> str:
        payload = json.dumps(value, default=str, sort_keys=True)
        return self.encrypt_text(payload)

    def decrypt_json(self, text: str) -> Any:
        return json.loads(self.decrypt_text(text))

    def is_encrypted(self, text: str) -> bool:
        return bool(text) and text.startswith(ENCRYPTED_PREFIX)

"""P2: optional encryption at rest for state, dead letters and approvals."""

from __future__ import annotations

import pytest

from mars.crypto import ENCRYPTED_PREFIX, PayloadCipher
from mars.reliability import DeadLetterQueue, RunStateStore


def test_disabled_cipher_is_a_pass_through() -> None:
    cipher = PayloadCipher(None)

    assert cipher.enabled is False
    assert cipher.encrypt_text("plain") == "plain"
    assert cipher.decrypt_text("plain") == "plain"


def test_text_json_and_bytes_round_trip() -> None:
    cipher = PayloadCipher("a-long-high-entropy-development-secret")

    assert cipher.enabled is True
    text = cipher.encrypt_text("secret topic")
    assert text.startswith(ENCRYPTED_PREFIX)
    assert "secret topic" not in text
    assert cipher.decrypt_text(text) == "secret topic"

    payload = {"topic": "agentic AI", "n": 3}
    encoded = cipher.encrypt_json(payload)
    assert cipher.decrypt_json(encoded) == payload

    blob = cipher.encrypt_bytes(b"bytes")
    assert cipher.decrypt_bytes(blob) == b"bytes"


def test_a_valid_fernet_key_is_accepted() -> None:
    from cryptography.fernet import Fernet

    cipher = PayloadCipher(Fernet.generate_key())

    assert cipher.decrypt_text(cipher.encrypt_text("hello")) == "hello"


def test_wrong_key_cannot_decrypt() -> None:
    from cryptography.fernet import InvalidToken

    first = PayloadCipher("key-one-high-entropy")
    second = PayloadCipher("key-two-high-entropy")

    with pytest.raises(InvalidToken):
        second.decrypt_text(first.encrypt_text("hello"))


def test_run_state_is_encrypted_on_disk(tmp_path) -> None:
    cipher = PayloadCipher("state-encryption-key")
    store = RunStateStore(tmp_path, "run-x", cipher=cipher)
    store.transition("plan", "completed", summary={"topic": "secret topic"})
    raw = store.path.read_text(encoding="utf-8")

    assert raw.startswith(ENCRYPTED_PREFIX)
    assert "secret topic" not in raw

    loaded = RunStateStore.load(tmp_path, "run-x", cipher=cipher)
    assert loaded is not None
    assert loaded.stage_status("plan") == "completed"
    assert loaded.stage_summary("plan")["topic"] == "secret topic"


def test_dead_letters_are_encrypted_on_disk(tmp_path) -> None:
    cipher = PayloadCipher("dead-letter-key")
    queue = DeadLetterQueue(tmp_path, "run-x", cipher=cipher)
    path = queue.enqueue("research", RuntimeError("boom"), {"topic": "secret topic"})
    raw = path.read_text(encoding="utf-8")

    assert raw.startswith(ENCRYPTED_PREFIX)
    assert "secret topic" not in raw
    items = queue.read_items()
    assert items[0]["stage"] == "research"
    assert items[0]["payload"]["topic"] == "secret topic"

"""P2: secret and personal-data redaction before prompts are disclosed."""

from __future__ import annotations

from mars.redaction import (
    REDACTED_SECRET,
    redact_mapping,
    redact_messages,
    redact_text,
    redact_value,
)


def test_credentials_are_redacted() -> None:
    text = (
        "key sk-or-v1-abcdefghijklmnopqrstuvwxyz and token ghp_abcdefghijklmnopqrstuvwxyz0123"
    )

    cleaned = redact_text(text)

    assert "sk-or-v1-" not in cleaned
    assert "ghp_" not in cleaned
    assert REDACTED_SECRET in cleaned


def test_bearer_and_assignment_shapes_are_redacted() -> None:
    cleaned = redact_text(
        "Authorization: Bearer abcdefghijklmnop1234\npassword=supersecretvalue"
    )

    assert "abcdefghijklmnop1234" not in cleaned
    assert "supersecretvalue" not in cleaned


def test_known_secret_values_are_redacted_even_without_a_pattern() -> None:
    cleaned = redact_text("the key is hunter2hunter2", secrets=["hunter2hunter2"])

    assert "hunter2hunter2" not in cleaned
    assert REDACTED_SECRET in cleaned


def test_personal_data_is_redacted_by_default() -> None:
    cleaned = redact_text(
        "Contact jane.doe@example.com, SSN 123-45-6789, card 4111 1111 1111 1111."
    )

    assert "jane.doe@example.com" not in cleaned
    assert "123-45-6789" not in cleaned
    assert "4111 1111 1111 1111" not in cleaned
    assert "[REDACTED_EMAIL]" in cleaned
    assert "[REDACTED_SSN]" in cleaned
    assert "[REDACTED_CARD]" in cleaned


def test_invalid_card_number_is_left_alone() -> None:
    cleaned = redact_text("reference 1234 5678 9012 3456 is not a card")

    assert "1234 5678 9012 3456" in cleaned


def test_pii_redaction_can_be_disabled() -> None:
    cleaned = redact_text("Contact jane.doe@example.com", redact_pii=False)

    assert "jane.doe@example.com" in cleaned


def test_nested_values_are_redacted() -> None:
    value = {
        "token": "sk-or-v1-abcdefghijklmnopqrstuvwxyz",
        "items": ["jane.doe@example.com", {"nested": "password=supersecretvalue"}],
    }

    cleaned = redact_value(value)

    assert "sk-or-v1-" not in str(cleaned)
    assert "jane.doe@example.com" not in str(cleaned)
    assert "supersecretvalue" not in str(cleaned)


def test_mapping_keys_are_redacted() -> None:
    cleaned = redact_mapping({"jane.doe@example.com": "value"})

    assert "[REDACTED_EMAIL]" in cleaned


def test_chat_messages_are_redacted() -> None:
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "email jane.doe@example.com and key sk-or-v1-abcdefghijklmnopqrstuvwxyz"},
    ]

    cleaned = redact_messages(messages)

    assert cleaned[0]["content"] == "you are helpful"
    assert "jane.doe@example.com" not in cleaned[1]["content"]
    assert "sk-or-v1-" not in cleaned[1]["content"]

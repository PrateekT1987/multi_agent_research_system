"""Data-governance redaction for prompts, logs and persisted payloads.

Mars sends topics, prompts and fetched content to a third-party model provider.
Before any of that leaves the process, obvious credentials and (optionally)
personal data are replaced with stable placeholders.  The same helper is used
for dead-letter payloads and approval artefacts so a support bundle does not
become a credential leak.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "REDACTED_SECRET",
    "REDACTED_PII",
    "redact_text",
    "redact_messages",
    "redact_value",
    "redact_mapping",
]

REDACTED_SECRET = "[REDACTED_SECRET]"
REDACTED_PII = "[REDACTED_PII]"

# Ordered: the most specific credential shapes first.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"\bsk-or-v1-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"\btvly-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\b(?:gsk|gpt|hf)_[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(
        r"(?i)\b(api[_-]?key|secret|token|password|credential)\b\s*[:=]\s*['\"]?[^\s'\"]{8,}"
    ),
)

_PII_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[REDACTED_CARD]"),
    (re.compile(r"(?<!\w)\+?\d{1,3}[\s.\-]?\(?\d{2,4}\)?[\s.\-]?\d{3}[\s.\-]?\d{3,4}(?!\w)"), "[REDACTED_PHONE]"),
)

_CREDIT_CARD_RE = re.compile(r"^(?:\d[ -]?){13,19}$")


def _looks_like_card(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        digit = int(char)
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def redact_text(
    text: str,
    *,
    redact_pii: bool = True,
    secrets: Iterable[str] = (),
) -> str:
    """Replace credentials and, optionally, personal data in ``text``."""
    if not text:
        return text
    cleaned = str(text)
    for secret in secrets:
        if secret and len(secret) >= 8:
            cleaned = cleaned.replace(secret, REDACTED_SECRET)
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub(REDACTED_SECRET, cleaned)
    if redact_pii:
        for pattern, replacement in _PII_PATTERNS:
            if replacement == "[REDACTED_CARD]":
                cleaned = pattern.sub(
                    lambda match, replacement=replacement: replacement
                    if _looks_like_card(match.group(0))
                    else match.group(0),
                    cleaned,
                )
            else:
                cleaned = pattern.sub(replacement, cleaned)
    return cleaned


def redact_value(
    value: Any,
    *,
    redact_pii: bool = True,
    secrets: Iterable[str] = (),
) -> Any:
    """Recursively redact strings inside JSON-like values."""
    if isinstance(value, str):
        return redact_text(value, redact_pii=redact_pii, secrets=secrets)
    if isinstance(value, Mapping):
        return {
            key: redact_value(item, redact_pii=redact_pii, secrets=secrets)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item, redact_pii=redact_pii, secrets=secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item, redact_pii=redact_pii, secrets=secrets) for item in value)
    return value


def redact_mapping(
    values: Mapping[str, Any],
    *,
    redact_pii: bool = True,
    secrets: Iterable[str] = (),
) -> dict[str, Any]:
    """Redact values (and string keys) of a mapping."""
    result: dict[str, Any] = {}
    for key, value in values.items():
        safe_key = redact_text(str(key), redact_pii=redact_pii, secrets=secrets)
        result[safe_key] = redact_value(value, redact_pii=redact_pii, secrets=secrets)
    return result


def redact_messages(
    messages: Iterable[Any],
    *,
    redact_pii: bool = True,
    secrets: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Redact the ``content`` of OpenAI-style chat messages."""
    out: list[dict[str, Any]] = []
    for message in messages:
        item = dict(message) if isinstance(message, Mapping) else {"content": message}
        content = item.get("content")
        if isinstance(content, str):
            item["content"] = redact_text(content, redact_pii=redact_pii, secrets=secrets)
        elif isinstance(content, list):
            item["content"] = redact_value(content, redact_pii=redact_pii, secrets=secrets)
        out.append(item)
    return out

"""Authentication and tenant helpers for the Streamlit UI.

The UI is an operator surface: it can spend money, run arbitrary research and
read every report written under its tenant directory.  It therefore fails
closed unless a password is configured, with an explicit opt-out for local
demos (``MARS_UI_ALLOW_ANONYMOUS=1``).
"""

from __future__ import annotations

import hmac
import os
import re
from collections.abc import Mapping
from pathlib import Path

__all__ = [
    "allow_anonymous",
    "configured_credentials",
    "tenant_slug",
    "verify_credentials",
]

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in _TRUTHY


def allow_anonymous(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the operator explicitly opted out of the login gate."""
    env = os.environ if environ is None else environ
    return _truthy(env.get("MARS_UI_ALLOW_ANONYMOUS"))


def configured_credentials(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, str] | None:
    """Return ``(username, password)`` from the environment, if configured."""
    env = os.environ if environ is None else environ
    password = env.get("MARS_UI_PASSWORD")
    secret_file = env.get("MARS_UI_PASSWORD_FILE")
    if not password and secret_file:
        try:
            password = Path(secret_file).read_text(encoding="utf-8").strip()
        except OSError:
            password = None
    if not password:
        return None
    return env.get("MARS_UI_USERNAME", "admin"), password


def verify_credentials(
    username: str,
    password: str,
    expected_username: str,
    expected_password: str,
) -> bool:
    """Constant-time credential comparison."""
    return hmac.compare_digest(username or "", expected_username) and hmac.compare_digest(
        password or "", expected_password
    )


def tenant_slug(user: str) -> str:
    """Map an authenticated username to a safe, single path component."""
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", (user or "").strip())
    slug = slug.strip(".-")[:64]
    if not slug or slug in {".", ".."}:
        return "anonymous"
    return slug

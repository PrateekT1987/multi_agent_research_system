"""Tests for the Streamlit UI authentication and tenant helpers."""

from __future__ import annotations

from mars.auth import (
    allow_anonymous,
    configured_credentials,
    tenant_slug,
    verify_credentials,
)


def test_configured_credentials_requires_a_password() -> None:
    assert configured_credentials({}) is None
    assert configured_credentials({"MARS_UI_PASSWORD": ""}) is None


def test_configured_credentials_uses_default_username() -> None:
    assert configured_credentials({"MARS_UI_PASSWORD": "s3cret"}) == ("admin", "s3cret")
    assert configured_credentials(
        {"MARS_UI_PASSWORD": "s3cret", "MARS_UI_USERNAME": "alice"}
    ) == ("alice", "s3cret")


def test_anonymous_access_is_explicit_opt_in() -> None:
    assert allow_anonymous({}) is False
    assert allow_anonymous({"MARS_UI_ALLOW_ANONYMOUS": "0"}) is False
    assert allow_anonymous({"MARS_UI_ALLOW_ANONYMOUS": "1"}) is True
    assert allow_anonymous({"MARS_UI_ALLOW_ANONYMOUS": "true"}) is True


def test_verify_credentials_checks_both_fields() -> None:
    assert verify_credentials("alice", "s3cret", "alice", "s3cret") is True
    assert verify_credentials("alice", "wrong", "alice", "s3cret") is False
    assert verify_credentials("bob", "s3cret", "alice", "s3cret") is False


def test_tenant_slug_is_a_safe_single_path_component() -> None:
    assert tenant_slug("alice") == "alice"
    assert tenant_slug("../../etc/passwd") == "etc-passwd"
    assert tenant_slug("a/b c") == "a-b-c"
    assert tenant_slug("") == "anonymous"
    assert tenant_slug("..") == "anonymous"
    assert "/" not in tenant_slug("evil/../../path")

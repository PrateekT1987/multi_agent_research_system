"""Human-in-the-loop approval gates for the Mars pipeline.

Checkpoints are written as JSON files and can be approved/rejected via the
CLI (``mars approve`` / ``mars reject``) or by writing an approval file.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .crypto import PayloadCipher

__all__ = ["ApprovalGate", "ApprovalRequest", "sign_payload", "verify_payload"]


def _canonical(payload: dict[str, Any]) -> bytes:
    """Deterministic bytes for an approval/rejection payload."""
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    return json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def sign_payload(secret: str, payload: dict[str, Any]) -> str:
    """Return an HMAC-SHA256 signature over a decision payload."""
    return hmac.new(secret.encode("utf-8"), _canonical(payload), hashlib.sha256).hexdigest()


def verify_payload(secret: str, payload: dict[str, Any]) -> bool:
    """Constant-time verification of a decision payload's signature."""
    supplied = str(payload.get("signature") or "")
    if not supplied:
        return False
    return hmac.compare_digest(sign_payload(secret, payload), supplied)


@dataclass(slots=True)
class ApprovalRequest:
    checkpoint: str
    run_id: str
    topic: str
    status: str
    requested_at: float
    summary: dict[str, Any]
    payload_path: str | None = None
    approved_at: float | None = None
    approver: str | None = None
    reason: str | None = None


class ApprovalGate:
    def __init__(self, settings: Any, reporter: Any = None) -> None:
        self.settings = settings
        self.reporter = reporter or (lambda _: None)
        self.approval_dir = Path(getattr(settings, "approval_dir", Path.cwd() / ".mars_approvals"))
        self.approval_dir.mkdir(parents=True, exist_ok=True)
        self.secret = getattr(settings, "approval_secret", None) or None
        raw_timeout = getattr(settings, "approval_timeout_seconds", 1800)
        self.timeout = None if raw_timeout is None else max(0, int(raw_timeout))
        key = getattr(settings, "encryption_key", None) if getattr(settings, "encrypt_state", False) else None
        self.cipher = PayloadCipher.from_key(key)

    def _request_path(self, run_id: str, checkpoint: str) -> Path:
        return self.approval_dir / f"{run_id}-{checkpoint}.request.json"

    def _approval_path(self, run_id: str, checkpoint: str) -> Path:
        return self.approval_dir / f"{run_id}-{checkpoint}.approved.json"

    def _rejection_path(self, run_id: str, checkpoint: str) -> Path:
        return self.approval_dir / f"{run_id}-{checkpoint}.rejected.json"

    def _write_payload(self, run_id: str, checkpoint: str, payload: Any) -> str | None:
        try:
            path = self.approval_dir / f"{run_id}-{checkpoint}.payload.json"
            if isinstance(payload, (dict, list)):
                text = (
                    self.cipher.encrypt_json(payload)
                    if self.cipher.enabled
                    else json.dumps(payload, indent=2, default=str)
                )
            elif isinstance(payload, str):
                text = self.cipher.encrypt_text(payload) if self.cipher.enabled else payload
            else:
                text = str(payload)
            path.write_text(text, encoding="utf-8")
            return str(path)
        except OSError:
            return None

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        """Read a decision file, tolerating a partial write."""
        for _ in range(10):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else None
            except (OSError, json.JSONDecodeError):
                time.sleep(0.1)
        return None

    def _verified_decision(self, path: Path, kind: str) -> dict[str, Any] | None:
        """Return a decision only if its HMAC is valid (when configured)."""
        data = self._read_json(path)
        if data is None:
            return None
        if self.secret and not verify_payload(self.secret, data):
            raise RuntimeError(
                f"{kind} file {path.name} has an invalid or missing signature"
            )
        return data

    def _read_rejection_reason(self, run_id: str, checkpoint: str) -> str:
        """Read and verify the reason out of a rejection file."""
        path = self._rejection_path(run_id, checkpoint)
        data = self._verified_decision(path, "rejection")
        if data is None:
            return "No reason given"
        reason = data.get("reason")
        return str(reason) if reason else "No reason given"

    def request_approval(
        self,
        checkpoint: str,
        run_id: str,
        topic: str,
        summary: dict[str, Any],
        payload: Any = None,
        *,
        force: bool = False,
    ) -> bool:
        if getattr(self.settings, "approval_auto_approve", False):
            return True
        if not force and not getattr(self.settings, "require_human_approval", False):
            return True
        if not force and checkpoint not in getattr(self.settings, "approval_checkpoints", ()):
            return True
        if not self.secret:
            raise RuntimeError(
                "MARS_APPROVAL_SECRET is required when human approval is enabled; "
                "approval files must be signed"
            )

        payload_path = self._write_payload(run_id, checkpoint, payload)
        req = ApprovalRequest(
            checkpoint=checkpoint,
            run_id=run_id,
            topic=topic,
            status="pending",
            requested_at=time.time(),
            summary=summary,
            payload_path=payload_path,
        )
        req_path = self._request_path(run_id, checkpoint)
        req_path.write_text(
            json.dumps(asdict(req), indent=2, default=str),
            encoding="utf-8",
        )

        self.reporter(
            f"⏸  HUMAN APPROVAL REQUIRED at checkpoint '{checkpoint}'. "
            f"Review: {req_path} (payload: {payload_path or 'n/a'}). "
            f"Approve with: mars approve --run-id {run_id} --checkpoint {checkpoint} "
            f"or write {self._approval_path(run_id, checkpoint).name}. "
            f"Reject with: mars reject --run-id {run_id} --checkpoint {checkpoint} ..."
        )

        start = time.time()
        while True:
            approval_path = self._approval_path(run_id, checkpoint)
            if approval_path.exists():
                if self._verified_decision(approval_path, "approval") is None:
                    raise RuntimeError(
                        f"approval file {approval_path.name} could not be read"
                    )
                self.reporter(f"✓ Approved at checkpoint '{checkpoint}'")
                return True
            if self._rejection_path(run_id, checkpoint).exists():
                reason = self._read_rejection_reason(run_id, checkpoint)
                raise RuntimeError(f"Human rejected checkpoint '{checkpoint}': {reason}")
            elapsed = time.time() - start
            if self.timeout is not None and elapsed >= self.timeout:
                raise RuntimeError(f"Approval timed out at checkpoint '{checkpoint}'")
            if self.timeout is None:
                time.sleep(2.0)
            else:
                time.sleep(min(2.0, max(0.05, self.timeout - elapsed)))

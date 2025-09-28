"""
Toy account tools used by the Agent.

They are deterministic and offline-only, suitable for demos/tests.

APIs:
- lookup_user(email) -> {exists: bool, status: str, plan: str | None, verified: bool | None}
- reset_password(email) -> {ok: bool, token: str | None, reason: str | None}

Behavior:
- A small in-memory mock DB seeded with a few users.
- lookup_user returns existence and simple account status.
- reset_password succeeds only if user exists and is verified; returns a deterministic token.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Any

# Small deterministic mock DB
# You can extend this in demos/tests as needed.
_MOCK_USERS: Dict[str, Dict[str, Any]] = {
    "ana@example.com": {"verified": True, "plan": "annual", "status": "active"},
    "bob@example.com": {"verified": False, "plan": "monthly", "status": "pending_email_verification"},
    "carlos@demo.io": {"verified": True, "plan": "free", "status": "active"},
}


def _token_for(email: str) -> str:
    """Deterministic short token for a given email."""
    h = hashlib.sha1(email.encode("utf-8")).hexdigest()[:10]
    return f"reset_{h}"


def lookup_user(email: str) -> Dict[str, Any]:
    """Return a mock account status for the given email.

    Response shape:
    {"exists": bool, "status": str, "plan": str | None, "verified": bool | None}
    """
    rec = _MOCK_USERS.get(email)
    if not rec:
        return {"exists": False, "status": "not_found", "plan": None, "verified": None}
    return {
        "exists": True,
        "status": rec.get("status", "active"),
        "plan": rec.get("plan"),
        "verified": bool(rec.get("verified", False)),
    }


def reset_password(email: str) -> Dict[str, Any]:
    """Attempt a password reset for the given email.

    Rules:
    - If user doesn't exist -> ok=False, reason="not_found".
    - If exists but not verified -> ok=False, reason="email_unverified".
    - If exists and verified -> ok=True, token=<deterministic>, reason=None.
    """
    info = lookup_user(email)
    if not info.get("exists"):
        return {"ok": False, "token": None, "reason": "not_found"}
    if not info.get("verified"):
        return {"ok": False, "token": None, "reason": "email_unverified"}
    return {"ok": True, "token": _token_for(email), "reason": None}


# Helpers for demos/tests (optional; no side-effects unless called)

def get_mock_db() -> Dict[str, Dict[str, Any]]:
    """Access the in-memory mock DB (read-only by convention)."""
    return _MOCK_USERS


def set_mock_user(email: str, *, verified: bool, plan: str = "free", status: str = "active") -> None:
    """Create or update a user in the mock DB. Useful for tests/demos."""
    _MOCK_USERS[email] = {"verified": bool(verified), "plan": plan, "status": status}

import hashlib

from tools.account_tools import (
    lookup_user,
    reset_password,
    get_mock_db,
    set_mock_user,
)


def test_lookup_user_known_and_unknown():
    # Known user (seeded): ana@example.com
    ana = lookup_user("ana@example.com")
    assert ana["exists"] is True
    assert ana["status"] in {"active", "pending_email_verification"}
    assert ana["plan"] in {"annual", "monthly", "free"}
    assert isinstance(ana["verified"], bool)

    # Unknown user
    ghost = lookup_user("ghost@nowhere.tld")
    assert ghost == {"exists": False, "status": "not_found", "plan": None, "verified": None}


def test_reset_password_rules_nonexistent_and_unverified():
    # Non-existent user
    res_none = reset_password("nobody@example.com")
    assert res_none == {"ok": False, "token": None, "reason": "not_found"}

    # Unverified user (seeded): bob@example.com
    res_unverified = reset_password("bob@example.com")
    assert res_unverified == {"ok": False, "token": None, "reason": "email_unverified"}


def test_reset_password_verified_and_token_deterministic():
    email = "ana@example.com"  # seeded as verified
    # Should succeed and return deterministic token
    res_ok = reset_password(email)
    assert res_ok["ok"] is True and res_ok["token"] and res_ok["reason"] is None

    # Token is deterministic and derived from sha1(email)[:10]
    expected = f"reset_{hashlib.sha1(email.encode('utf-8')).hexdigest()[:10]}"
    assert res_ok["token"] == expected

    # Repeating the call yields same token
    res_ok_2 = reset_password(email)
    assert res_ok_2["token"] == expected


def test_set_mock_user_updates_db_and_enables_reset():
    email = "new.user@example.org"
    # Ensure doesn't exist yet
    info0 = lookup_user(email)
    assert info0["exists"] is False

    # Create verified user via helper
    set_mock_user(email, verified=True, plan="pro", status="active")
    db = get_mock_db()
    assert email in db and db[email]["verified"] is True

    # Now reset should succeed
    res = reset_password(email)
    assert res["ok"] is True and res["token"] is not None and res["reason"] is None

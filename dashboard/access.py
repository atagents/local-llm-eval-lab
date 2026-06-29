"""Time-limited demo access codes for the public eval dashboard.

The master admin password (env EVALS_DASHBOARD_PASSWORD) is permanent and lives outside
this store. Demo codes are minted here with a TTL (e.g. 1h / 2d), handed out, and stop
working once expired - the password itself is the credential (no username needed by the
viewer); `label` is only the admin's note for tracking / revoking.

Passwords are never stored: each row keeps a per-row salt + pbkdf2-sha256 hash. Shares the
dashboard SQLite (dashboard/evals.db, gitignored).

Run:  python dashboard/access.py        # self-test (uses a temp db)
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import time
from pathlib import Path

_DB = Path(__file__).parent / "evals.db"          # module global so the self-test can redirect it
_ITER = 200_000
TTL_UNITS = {"m": 60, "h": 3600, "d": 86400}


def _conn():
    c = sqlite3.connect(_DB)
    c.execute("""CREATE TABLE IF NOT EXISTS demo_access(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        label TEXT, salt TEXT, pw_hash TEXT,
        created_at REAL, expires_at REAL)""")
    return c


def _hash(pw: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), _ITER).hex()


def parse_ttl(s: str) -> int:
    """'30m' / '2h' / '7d' -> seconds."""
    s = str(s).strip().lower()
    if len(s) < 2 or s[-1] not in TTL_UNITS or not s[:-1].isdigit():
        raise ValueError("ttl must look like 30m / 2h / 7d")
    return int(s[:-1]) * TTL_UNITS[s[-1]]


def add(label: str, ttl, now: float | None = None) -> tuple[str, float]:
    """Mint a demo code. Returns (password_shown_once, expires_at_epoch)."""
    now = time.time() if now is None else now
    secs = ttl if isinstance(ttl, (int, float)) else parse_ttl(ttl)
    pw = secrets.token_urlsafe(9)                 # ~12 chars, easy to paste
    salt = secrets.token_hex(16)
    exp = now + secs
    c = _conn()
    try:
        c.execute("INSERT INTO demo_access(label,salt,pw_hash,created_at,expires_at) VALUES(?,?,?,?,?)",
                  (label or "demo", salt, _hash(pw, salt), now, exp))
        c.commit()
    finally:
        c.close()
    return pw, exp


def verify(pw: str, now: float | None = None) -> float | None:
    """Return the code's expires_at if pw matches a NON-expired demo code, else None."""
    if not pw:
        return None
    now = time.time() if now is None else now
    c = _conn()
    try:
        rows = c.execute("SELECT salt,pw_hash,expires_at FROM demo_access WHERE expires_at>?",
                         (now,)).fetchall()
    finally:
        c.close()
    for salt, pw_hash, exp in rows:
        if hmac.compare_digest(_hash(pw, salt), pw_hash):
            return exp
    return None


def list_codes(now: float | None = None) -> list[dict]:
    now = time.time() if now is None else now
    c = _conn()
    try:
        rows = c.execute("SELECT id,label,created_at,expires_at FROM demo_access ORDER BY expires_at DESC").fetchall()
    finally:
        c.close()
    return [{"id": r[0], "label": r[1], "created_at": r[2], "expires_at": r[3],
             "active": r[3] > now, "hours_left": round((r[3] - now) / 3600, 1)} for r in rows]


def revoke(code_id: int) -> None:
    c = _conn()
    try:
        c.execute("DELETE FROM demo_access WHERE id=?", (code_id,))
        c.commit()
    finally:
        c.close()


def purge_expired(now: float | None = None) -> int:
    now = time.time() if now is None else now
    c = _conn()
    try:
        n = c.execute("DELETE FROM demo_access WHERE expires_at<=?", (now,)).rowcount
        c.commit()
    finally:
        c.close()
    return n


# ----------------------------------------------------------------- self-test
def _selftest():
    import tempfile
    global _DB
    _DB = Path(tempfile.mkdtemp()) / "access_test.db"
    t0 = 1_000_000.0

    assert parse_ttl("30m") == 1800 and parse_ttl("2h") == 7200 and parse_ttl("7d") == 604800
    try:
        parse_ttl("5x"); assert False, "bad ttl accepted"
    except ValueError:
        pass

    pw, exp = add("recruiter-acme", "1h", now=t0)
    assert exp == t0 + 3600
    # valid within window
    assert verify(pw, now=t0 + 1800) == exp, "valid code rejected"
    # wrong password
    assert verify(pw + "x", now=t0 + 1800) is None, "wrong pw accepted"
    # expired
    assert verify(pw, now=t0 + 3601) is None, "expired code accepted"

    # second code, listing + revoke + purge
    pw2, _ = add("demo-2d", "2d", now=t0)
    codes = list_codes(now=t0)
    assert len(codes) == 2 and all(c["active"] for c in codes), codes
    revoke(codes[0]["id"])
    assert len(list_codes(now=t0)) == 1, "revoke failed"
    # purge after everything expired
    assert purge_expired(now=t0 + 10 * 86400) == 1, "purge count wrong"
    assert list_codes(now=t0 + 10 * 86400) == []

    print("access self-test: PASS")
    print("  parse_ttl 30m/2h/7d, mint+verify, wrong-pw reject, expiry reject, list/revoke/purge")


if __name__ == "__main__":
    _selftest()

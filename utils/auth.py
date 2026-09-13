"""
utils/auth.py
=============
Modern, Secure Authentication Service for FinTwin AI.

Security Hardening:
  - Argon2id password hashing (with PBKDF2 backward compatibility)
  - Cryptographically secure single-use token generation for email verification and password resets
  - SHA-256 hashed token storage in database
  - 30-minute token expiration enforcement
  - Strict input sanitization and password complexity validation
  - In-memory rate limiting against brute-force attacks
  - User enumeration protection on registration and forgot password
  - Complete isolation: passwords and tokens are never emailed, logged, or leaked
"""

import hashlib
import hmac
import os
import re
import uuid
import secrets
import datetime
import logging
from typing import Optional, Dict, Any, Tuple

from database.connection import get_db_cursor
from utils.security import (
    validate_email,
    validate_password,
    sanitize_text,
    MAX_NAME_LEN,
    RateLimiter,
)
from utils.email_service import (
    send_verification_email,
    send_password_reset_email,
)

logger = logging.getLogger(__name__)

# Singleton rate limiter
_rate_limiter = RateLimiter.get()

# Token expiration in minutes
TOKEN_EXPIRATION_MINUTES = 30

# PBKDF2 Iterations for backward-compatible verification
_PBKDF2_ITERATIONS = 260_000

# Try importing Argon2 PasswordHasher
try:
    from argon2 import PasswordHasher
    from argon2.exceptions import VerifyMismatchError, InvalidHash
    _argon2_hasher = PasswordHasher(
        time_cost=3,
        memory_cost=65536,  # 64 MB
        parallelism=4,
        hash_len=32,
        salt_len=16,
    )
    _ARGON2_AVAILABLE = True
except ImportError:
    _argon2_hasher = None
    _ARGON2_AVAILABLE = False
    logger.info("argon2-cffi not installed; using PBKDF2-HMAC-SHA256 with 260,000 iterations.")


# ── Password Hashing & Verification ──────────────────────────────────────────

def _hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hashes password using Argon2id when available, or PBKDF2-HMAC-SHA256.
    Returns (hash_string, salt_string).
    """
    if _ARGON2_AVAILABLE and _argon2_hasher is not None:
        hashed = _argon2_hasher.hash(password)
        return hashed, "argon2id"
    
    # Fallback to PBKDF2-HMAC-SHA256
    if salt is None or salt == "argon2id":
        salt = os.urandom(16).hex()
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    )
    return digest.hex(), salt


def verify_password(password: str, password_hash: str, salt: Optional[str]) -> bool:
    """
    Verifies a password against an Argon2id or PBKDF2 hash.
    Constant-time comparison is used to prevent timing attacks.
    """
    if not password or not password_hash:
        return False

    # Check if hash is Argon2 format
    if password_hash.startswith("$argon2"):
        if _ARGON2_AVAILABLE and _argon2_hasher is not None:
            try:
                return _argon2_hasher.verify(password_hash, password)
            except Exception:
                return False
        return False

    # PBKDF2-HMAC-SHA256 verification
    if salt:
        try:
            computed_digest = hashlib.pbkdf2_hmac(
                "sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS
            ).hex()
            return hmac.compare_digest(computed_digest, password_hash)
        except Exception:
            return False

    return False


# ── Token Generation & Hash Helpers ──────────────────────────────────────────

def _generate_secure_token() -> Tuple[str, str, str]:
    """
    Generates a cryptographically secure token.
    Returns:
        (raw_token, token_hash, expiration_iso_timestamp)
    """
    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    expires_at = (
        datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(minutes=TOKEN_EXPIRATION_MINUTES)
    ).strftime("%Y-%m-%d %H:%M:%S")
    return raw_token, token_hash, expires_at


def _hash_token(raw_token: str) -> str:
    """Computes SHA-256 hash of a raw token for database lookup."""
    return hashlib.sha256(raw_token.strip().encode("utf-8")).hexdigest()


def _is_expired(expires_at_str: Optional[str]) -> bool:
    """Checks if a timestamp string in UTC has expired."""
    if not expires_at_str:
        return True
    try:
        expires_dt = datetime.datetime.strptime(expires_at_str, "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=datetime.timezone.utc
        )
        now_dt = datetime.datetime.now(datetime.timezone.utc)
        return now_dt > expires_dt
    except Exception:
        return True


# ── User Queries ─────────────────────────────────────────────────────────────

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Retrieves a user row by normalized email."""
    if not email:
        return None
    with get_db_cursor() as cursor:
        cursor.execute("SELECT * FROM users WHERE email = ?", (email.strip().lower(),))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a user row by user_id."""
    if not user_id:
        return None
    with get_db_cursor() as cursor:
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id.strip(),))
        row = cursor.fetchone()
        return dict(row) if row else None


# ── Registration Flow ────────────────────────────────────────────────────────

def register_user(
    email: str,
    password: str,
    name: str,
    age: int,
    confirm_password: Optional[str] = None,
    avatar_id: Optional[str] = None,
) -> Tuple[bool, str, str]:
    """
    Registers a new user account with an unverified status.
    Generates verification token and dispatches verification email.

    Returns:
        (success: bool, status_code: str, message_or_user_id: str)
        status_code: 'SUCCESS', 'EMAIL_FAILED', 'ERROR'
    """
    email = email.strip().lower()

    # 1. Validation
    email_ok, email_err = validate_email(email)
    if not email_ok:
        return False, "ERROR", email_err

    if confirm_password is not None and password != confirm_password:
        return False, "ERROR", "Passwords do not match."

    pw_ok, pw_err = validate_password(password)
    if not pw_ok:
        return False, "ERROR", pw_err

    name = sanitize_text(name.strip(), max_len=MAX_NAME_LEN)
    if not name:
        return False, "ERROR", "Please enter your full name."

    # Validate avatar_id if provided
    from utils.avatar import is_valid_avatar_id, DEFAULT_AVATAR_ID
    clean_avatar = str(avatar_id).strip() if avatar_id else None
    if clean_avatar:
        if not is_valid_avatar_id(clean_avatar):
            return False, "ERROR", "Please select a valid profile avatar."
    else:
        clean_avatar = DEFAULT_AVATAR_ID

    # 2. Check duplicate email (generic enumeration-safe response)
    existing = get_user_by_email(email)
    if existing:
        return False, "ERROR", "An account with this email already exists or is unavailable."

    # 3. Secure Password Hashing
    user_id = f"user_{uuid.uuid4().hex[:12]}"
    password_hash, salt = _hash_password(password)

    # 4. Generate Verification Token
    raw_token, token_hash, expires_at = _generate_secure_token()

    # 5. Database Insertion
    with get_db_cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO users (
                user_id, email, password_hash, password_salt, name, age,
                city, occupation, monthly_income, bonus, additional_income,
                email_verified, verification_token_hash, verification_expires_at,
                avatar_id
            )
            VALUES (?, ?, ?, ?, ?, ?, '', '', 0.0, 0.0, 0.0, 0, ?, ?, ?)
            """,
            (user_id, email, password_hash, salt, name, age, token_hash, expires_at, clean_avatar),
        )
        cursor.execute(
            "INSERT INTO digital_twins (user_id) VALUES (?)",
            (user_id,),
        )

    # 6. Send Verification Email
    email_ok, email_msg = send_verification_email(
        name=name,
        email=email,
        user_id=user_id,
        raw_token=raw_token,
    )

    if not email_ok:
        return True, "EMAIL_FAILED", user_id

    return True, "SUCCESS", user_id


# ── Email Verification Flow ──────────────────────────────────────────────────

def verify_email_token(raw_token: str) -> Tuple[bool, str]:
    """
    Validates the email verification token, marks user as verified,
    and invalidates the token. Single-use and time-bounded.
    """
    if not raw_token or len(raw_token.strip()) < 10:
        return False, "Invalid or missing verification token."

    token_hash = _hash_token(raw_token)

    with get_db_cursor() as cursor:
        cursor.execute(
            """
            SELECT user_id, email, email_verified, verification_expires_at
            FROM users
            WHERE verification_token_hash = ?
            """,
            (token_hash,),
        )
        row = cursor.fetchone()

        if not row:
            return False, "This verification link is invalid, has expired, or was already used."

        user = dict(row)

        if _is_expired(user.get("verification_expires_at")):
            return False, "This verification link has expired. Please request a new verification email."

        # Mark as verified and invalidate token
        cursor.execute(
            """
            UPDATE users
            SET email_verified = 1,
                verification_token_hash = NULL,
                verification_expires_at = NULL
            WHERE user_id = ?
            """,
            (user["user_id"],),
        )

    return True, "Your email has been successfully verified! You can now log in."


def resend_verification_email(email: str) -> Tuple[bool, str]:
    """
    Generates a new verification token and resends the verification email.
    """
    email = email.strip().lower()
    email_ok, _ = validate_email(email)
    if not email_ok:
        return False, "Please enter a valid email address."

    user = get_user_by_email(email)
    if not user:
        # Generic response to prevent enumeration
        return True, "If your account exists and is unverified, a verification email has been sent."

    if user.get("email_verified") == 1:
        return True, "This account has already been verified. Please log in."

    raw_token, token_hash, expires_at = _generate_secure_token()

    with get_db_cursor() as cursor:
        cursor.execute(
            """
            UPDATE users
            SET verification_token_hash = ?,
                verification_expires_at = ?
            WHERE user_id = ?
            """,
            (token_hash, expires_at, user["user_id"]),
        )

    send_verification_email(
        name=user["name"],
        email=email,
        user_id=user["user_id"],
        raw_token=raw_token,
    )

    return True, "A new verification email has been sent. Please check your inbox."


# ── Forgot & Reset Password Flow ─────────────────────────────────────────────

def request_password_reset(email: str) -> Tuple[bool, str]:
    """
    Generates a secure password reset token and sends the password reset email.
    Protected against user enumeration: always returns a generic response.
    """
    email = email.strip().lower()
    email_ok, _ = validate_email(email)
    if not email_ok:
        return False, "Please enter a valid email address."

    user = get_user_by_email(email)
    if user:
        raw_token, token_hash, expires_at = _generate_secure_token()
        with get_db_cursor() as cursor:
            cursor.execute(
                """
                UPDATE users
                SET reset_token_hash = ?,
                    reset_expires_at = ?
                WHERE user_id = ?
                """,
                (token_hash, expires_at, user["user_id"]),
            )
        send_password_reset_email(
            name=user["name"],
            email=email,
            raw_token=raw_token,
        )

    # Generic enumeration-safe response
    return True, "If an account exists for this email address, a password reset link has been sent."


def validate_reset_token(raw_token: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validates whether a reset token is valid and unexpired without consuming it.
    Returns (is_valid: bool, message: str, email: Optional[str]).
    """
    if not raw_token or len(raw_token.strip()) < 10:
        return False, "Invalid password reset token.", None

    token_hash = _hash_token(raw_token)

    with get_db_cursor() as cursor:
        cursor.execute(
            """
            SELECT user_id, email, reset_expires_at
            FROM users
            WHERE reset_token_hash = ?
            """,
            (token_hash,),
        )
        row = cursor.fetchone()

        if not row:
            return False, "This password reset link is invalid or has already been used.", None

        user = dict(row)
        if _is_expired(user.get("reset_expires_at")):
            return False, "This password reset link has expired (30-minute limit). Please request a new one.", None

        return True, "Token valid", user.get("email")


def reset_password_with_token(
    raw_token: str, new_password: str, confirm_password: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Resets the user's password using the validated reset token,
    hashes the new password with Argon2id/PBKDF2, and clears the reset token.
    """
    if confirm_password is not None and new_password != confirm_password:
        return False, "Passwords do not match."

    pw_ok, pw_err = validate_password(new_password)
    if not pw_ok:
        return False, pw_err

    token_hash = _hash_token(raw_token)

    with get_db_cursor() as cursor:
        cursor.execute(
            """
            SELECT user_id, reset_expires_at
            FROM users
            WHERE reset_token_hash = ?
            """,
            (token_hash,),
        )
        row = cursor.fetchone()

        if not row:
            return False, "This password reset link is invalid or has already been used."

        user = dict(row)
        if _is_expired(user.get("reset_expires_at")):
            return False, "This password reset link has expired. Please request a new one."

        new_hash, salt = _hash_password(new_password)

        cursor.execute(
            """
            UPDATE users
            SET password_hash = ?,
                password_salt = ?,
                reset_token_hash = NULL,
                reset_expires_at = NULL
            WHERE user_id = ?
            """,
            (new_hash, salt, user["user_id"]),
        )

    return True, "Your password has been successfully reset! You can now log in with your new password."


# ── Authentication Flow ──────────────────────────────────────────────────────

def authenticate(email: str, password: str) -> Tuple[bool, str, str]:
    """
    Authenticates a user with email and password.
    Returns:
        (success: bool, status_code: str, user_id_or_error: str)
        status_code: 'SUCCESS', 'UNVERIFIED', 'INVALID_CREDENTIALS', 'RATE_LIMITED'
    """
    email = email.strip().lower()
    _GENERIC_ERR = "Invalid email or password."

    # 1. Rate Limit Check
    blocked, block_msg = _rate_limiter.is_locked(email)
    if blocked:
        return False, "RATE_LIMITED", block_msg

    # 2. User Lookup
    user = get_user_by_email(email)
    if not user or not user.get("password_hash"):
        _rate_limiter.record_failure(email)
        return False, "INVALID_CREDENTIALS", _GENERIC_ERR

    # 3. Password Verification
    if not verify_password(password, user["password_hash"], user.get("password_salt")):
        _rate_limiter.record_failure(email)
        return False, "INVALID_CREDENTIALS", _GENERIC_ERR

    # 4. Check Email Verification Status
    if user.get("email_verified") != 1:
        # Account credentials are valid, but email is unverified
        return False, "UNVERIFIED", user.get("email", email)

    # 5. Successful Login
    _rate_limiter.reset(email)

    if email == "demo@fintwin.app":
        from database.connection import reset_demo_account
        reset_demo_account()

    return True, "SUCCESS", user["user_id"]


def login_user(email: str, password: str) -> Tuple[bool, str, Any]:
    """
    Convenience wrapper for logging in a user.
    Returns:
        (success: bool, status_code: str, user_dict_or_error: Any)
    """
    ok, status_code, user_id_or_msg = authenticate(email, password)
    if ok:
        user = get_user_by_id(user_id_or_msg)
        return True, status_code, user if user else {"user_id": user_id_or_msg}
    if status_code == "UNVERIFIED":
        return False, "UNVERIFIED_EMAIL", "Please verify your email before logging in."
    return False, status_code, user_id_or_msg


# ── In-Session Password Management & Verification ───────────────────────────

def verify_user_credentials_by_id(user_id: str, password: str) -> bool:
    """
    Verifies that the provided password matches the user_id's stored password_hash.
    Used for step-up authentication during destructive actions (e.g., account deletion).
    """
    user = get_user_by_id(user_id)
    if not user or not user.get("password_hash"):
        return False
    return verify_password(password, user["password_hash"], user.get("password_salt"))


def change_user_password(
    user_id: str, current_password: str, new_password: str, confirm_password: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Safely changes an authenticated user's password.
    Requires verifying the current password first.
    """
    if not user_id:
        return False, "Authentication session is required."

    if not current_password:
        return False, "Please enter your current password."

    if not verify_user_credentials_by_id(user_id, current_password):
        return False, "Current password is incorrect."

    if confirm_password is not None and new_password != confirm_password:
        return False, "New passwords do not match."

    if current_password == new_password:
        return False, "New password must be different from your current password."

    pw_ok, pw_err = validate_password(new_password)
    if not pw_ok:
        return False, pw_err

    new_hash, salt = _hash_password(new_password)

    with get_db_cursor() as cursor:
        cursor.execute(
            """
            UPDATE users
            SET password_hash = ?,
                password_salt = ?
            WHERE user_id = ?
            """,
            (new_hash, salt, user_id),
        )

    return True, "Your password has been changed successfully."

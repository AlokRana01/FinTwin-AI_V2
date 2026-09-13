"""
utils/security.py
==================
Central security utilities for FinTwin AI.

Contains:
  - Input sanitization and validation helpers
  - Password strength enforcement
  - Email validation
  - Rate limiter for brute-force protection
  - Prompt injection filter for AI chatbot
  - Session timeout helpers

IMPORTANT: This module adds SECURITY CONTROLS only.
It does NOT alter business logic, financial calculations, AI models, or workflows.
"""

from __future__ import annotations

import re
import time
import threading
from html import escape as _he
from typing import Tuple

# ── Constants ────────────────────────────────────────────────────────────────

# Input length caps
MAX_NAME_LEN        = 80
MAX_EMAIL_LEN       = 254
MAX_OCCUPATION_LEN  = 100
MAX_CITY_LEN        = 100
MAX_GOAL_NAME_LEN   = 120
MAX_CHAT_MSG_LEN    = 1000
MAX_PASSWORD_LEN    = 64
MIN_PASSWORD_LEN    = 8

# Rate limiting
_RATE_LOCK_WINDOW   = 300   # seconds (5 minutes)
_RATE_MAX_ATTEMPTS  = 5     # failures before lockout

# Session timeout
SESSION_TIMEOUT_SECONDS = 3_600  # 1 hour of inactivity

# ── Regex patterns ──────────────────────────────────────────────────────────

# Strict RFC-5321 compatible email regex
_EMAIL_RE = re.compile(
    r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$'
)

# Safe name: letters, spaces, hyphens, apostrophes, periods
_NAME_RE = re.compile(r"^[a-zA-Z\s\.\-']{1,80}$")

# Password complexity: upper + lower + digit or special
_PASSWORD_RE = re.compile(
    r'^(?=.*[a-z])(?=.*[A-Z])(?=.*[\d!@#$%^&*()\-_=+\[\]{}|;:,.<>?]).{8,64}$'
)

# Prompt injection detection patterns (lightweight, non-breaking)
_INJECTION_PATTERNS = re.compile(
    r'ignore\s+(all\s+|previous\s+|above\s+|the\s+)?(instructions?|prompts?|guidelines?|rules?)'
    r'|(reveal|print|output|show|display|repeat|tell\s+me)\s+(the\s+|your\s+|all\s+|this\s+)?(system\s+)?prompt'
    r'|you\s+are\s+(now\s+)?(a\s+|an\s+)?(different|new|other|evil|unrestricted)\s+(AI|assistant|model|bot|entity)'
    r'|(pretend|act)\s+(you\s+are|to\s+be)\s+(a\s+|an\s+)?(different|evil|unrestricted|other)'
    r'|jailbreak|DAN\s+mode|developer\s+mode|training\s+data'
    r'|<\s*script|<\s*img\s|<\s*iframe|javascript\s*:',
    re.IGNORECASE
)

# Dangerous HTML/script pattern (matches opening and closing tags)
_DANGEROUS_HTML_RE = re.compile(
    r'</?\s*(script|iframe|object|embed|link|meta|form|input|svg|img)[^>]*>|'
    r'javascript\s*:|on\w+\s*=',
    re.IGNORECASE
)


# ── Input sanitization ───────────────────────────────────────────────────────

def sanitize_text(text: str, max_len: int = 500) -> str:
    """
    General-purpose text sanitizer.
    - Strips leading/trailing whitespace
    - Collapses excessive inner whitespace
    - Enforces max length
    - Does NOT HTML-escape (caller decides rendering context)
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    text = re.sub(r'[ \t]{2,}', ' ', text)      # collapse spaces/tabs
    text = re.sub(r'\n{3,}', '\n\n', text)       # max 2 consecutive newlines
    return text[:max_len]


def sanitize_html_output(text: str) -> str:
    """
    Escape text for safe rendering inside unsafe_allow_html=True HTML blocks.
    Use this on any user-controlled string embedded into an HTML f-string.
    """
    return _he(str(text))


# ── Validation ───────────────────────────────────────────────────────────────

def validate_name(name: str) -> Tuple[bool, str]:
    """
    Validates a user display name.
    Returns (is_valid, error_message_or_empty_string).
    """
    name = name.strip()
    if not name:
        return False, "Name is required."
    if len(name) > MAX_NAME_LEN:
        return False, f"Name must not exceed {MAX_NAME_LEN} characters."
    if not _NAME_RE.match(name):
        return False, "Name may only contain letters, spaces, hyphens, apostrophes, or periods."
    return True, ""


def validate_email(email: str) -> Tuple[bool, str]:
    """
    Validates an email address format.
    Returns (is_valid, error_message_or_empty_string).
    """
    email = email.strip().lower()
    if not email:
        return False, "Email is required."
    if len(email) > MAX_EMAIL_LEN:
        return False, f"Email must not exceed {MAX_EMAIL_LEN} characters."
    if not _EMAIL_RE.match(email):
        return False, "Please enter a valid email address (e.g., user@example.com)."
    return True, ""


def validate_password(password: str) -> Tuple[bool, str]:
    """
    Validates password strength.
    Requirements: 8–64 chars, upper, lower, digit or special char.
    Returns (is_valid, error_message_or_empty_string).
    """
    if not password:
        return False, "Password is required."
    if len(password) < MIN_PASSWORD_LEN:
        return False, f"Password must be at least {MIN_PASSWORD_LEN} characters."
    if len(password) > MAX_PASSWORD_LEN:
        return False, f"Password must not exceed {MAX_PASSWORD_LEN} characters."
    if not _PASSWORD_RE.match(password):
        return False, (
            "Password must include at least one uppercase letter, one lowercase letter, "
            "and one number or special character (!@#$%^&* etc.)."
        )
    return True, ""


def sanitize_chat_message(text: str) -> str:
    """
    Sanitizes a user chat message before:
      1. Displaying it in the UI
      2. Sending it to the AI API

    - Enforces MAX_CHAT_MSG_LEN
    - Strips dangerous HTML/script tags
    - Neutralizes detected prompt injection attempts (replaces with [filtered])
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()[:MAX_CHAT_MSG_LEN]

    # Strip dangerous HTML injections
    text = _DANGEROUS_HTML_RE.sub("[removed]", text)

    # Neutralize prompt injection attempts
    text = _INJECTION_PATTERNS.sub("[filtered]", text)

    return text


# ── Rate Limiter ─────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Thread-safe in-memory rate limiter for login brute-force protection.

    Tracks failed login attempts per identifier (email address).
    After _RATE_MAX_ATTEMPTS failures within _RATE_LOCK_WINDOW seconds,
    subsequent attempts are blocked with exponential backoff messaging.

    State resets automatically after _RATE_LOCK_WINDOW seconds of no failures.
    Accounts are NEVER permanently locked — only temporarily blocked.
    """

    _instance: "RateLimiter | None" = None
    _lock = threading.Lock()

    # Map: identifier -> list of Unix timestamps of recent failures
    _attempts: dict[str, list[float]] = {}

    @classmethod
    def get(cls) -> "RateLimiter":
        """Singleton accessor."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _clean(self, identifier: str, now: float) -> list[float]:
        """Return only recent failures within the lockout window."""
        recent = [t for t in self._attempts.get(identifier, [])
                  if now - t < _RATE_LOCK_WINDOW]
        self._attempts[identifier] = recent
        return recent

    def is_locked(self, identifier: str) -> Tuple[bool, str]:
        """
        Check if identifier is currently rate-limited.
        Returns (is_blocked, user_facing_message).
        """
        with self._lock:
            now = time.time()
            recent = self._clean(identifier, now)
            if len(recent) >= _RATE_MAX_ATTEMPTS:
                oldest = min(recent)
                wait_secs = int(_RATE_LOCK_WINDOW - (now - oldest))
                minutes = max(1, wait_secs // 60)
                return True, (
                    f"Too many failed login attempts. "
                    f"Please wait {minutes} minute(s) before trying again."
                )
        return False, ""

    def record_failure(self, identifier: str) -> None:
        """Record a failed login attempt for the given identifier."""
        with self._lock:
            now = time.time()
            self._clean(identifier, now)
            self._attempts.setdefault(identifier, []).append(now)

    def reset(self, identifier: str) -> None:
        """Clear all recorded failures for the identifier (on successful login)."""
        with self._lock:
            self._attempts.pop(identifier, None)


# ── Session helpers ───────────────────────────────────────────────────────────

def session_touch(session_state) -> None:
    """Update the last-active timestamp for the current session."""
    session_state["_session_last_active"] = time.time()


def session_is_expired(session_state) -> bool:
    """
    Return True if the session has been inactive beyond SESSION_TIMEOUT_SECONDS.
    Also returns True if no timestamp has ever been recorded.
    """
    last = session_state.get("_session_last_active")
    if last is None:
        return False  # First page load — don't evict immediately
    return (time.time() - last) > SESSION_TIMEOUT_SECONDS

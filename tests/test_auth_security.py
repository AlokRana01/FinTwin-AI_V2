"""
tests/test_auth_security.py
===========================
Comprehensive unit and integration test suite for FinTwin AI authentication,
password hashing, email verification, password reset, and security controls.
"""

import os
import sys
import unittest
import sqlite3
import datetime
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from database.connection import init_db, get_db_cursor
from utils import auth
from utils.security import validate_password, validate_email, validate_name
from utils import email_service
from utils.email_service import is_smtp_configured, get_smtp_config, send_verification_email, send_password_reset_email

# Mock SMTP delivery during automated test runs so real emails are never dispatched
email_service._send_smtp_email = lambda *args, **kwargs: (True, "Test environment mocked email delivery")


class TestFinTwinAuthentication(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize database schema and migrations
        init_db()

    def setUp(self):
        self.test_email = "test.security.user@example.com"
        self.test_pw = "FinTwinSecure#2026"
        self.test_name = "Alex Test"
        self.test_age = 28

        # Clean up test user if exists
        with get_db_cursor() as cursor:
            cursor.execute("DELETE FROM users WHERE email = ?", (self.test_email,))

    def tearDown(self):
        # Clean up test user
        with get_db_cursor() as cursor:
            cursor.execute("DELETE FROM users WHERE email = ?", (self.test_email,))

    def test_01_password_validation(self):
        """Tests password strength rules."""
        # Too short (< 8 chars)
        ok, err = validate_password("Short1!")
        self.assertFalse(ok)
        self.assertIn("at least 8 characters", err)

        # Missing number/symbol
        ok, err = validate_password("NoNumbersOrSpecial")
        self.assertFalse(ok)

        # Missing uppercase
        ok, err = validate_password("lowercase123!")
        self.assertFalse(ok)

        # Valid strong password
        ok, err = validate_password(self.test_pw)
        self.assertTrue(ok)
        self.assertEqual(err, "")

    def test_02_registration_flow(self):
        """Tests user registration creates unverified account with token."""
        ok, status_code, user_id = auth.register_user(
            email=self.test_email,
            password=self.test_pw,
            name=self.test_name,
            age=self.test_age,
            confirm_password=self.test_pw,
        )
        self.assertTrue(ok)
        self.assertIn(status_code, ["SUCCESS", "EMAIL_FAILED"])
        self.assertTrue(user_id.startswith("user_"))

        # Verify DB entry
        user = auth.get_user_by_email(self.test_email)
        self.assertIsNotNone(user)
        self.assertEqual(user["email_verified"], 0)
        self.assertIsNotNone(user["verification_token_hash"])
        self.assertIsNotNone(user["verification_expires_at"])
        # Verify password is NOT in plaintext
        self.assertNotEqual(user["password_hash"], self.test_pw)
        self.assertTrue(user["password_hash"].startswith("$argon2") or len(user["password_hash"]) == 64)

    def test_03_login_unverified_account_blocked(self):
        """Tests unverified users cannot log in and receive UNVERIFIED status."""
        auth.register_user(
            email=self.test_email,
            password=self.test_pw,
            name=self.test_name,
            age=self.test_age,
            confirm_password=self.test_pw,
        )

        ok, status_code, result = auth.authenticate(self.test_email, self.test_pw)
        self.assertFalse(ok)
        self.assertEqual(status_code, "UNVERIFIED")
        self.assertEqual(result, self.test_email)

    def test_04_email_verification_flow(self):
        """Tests valid token verifies account and invalidates token."""
        auth.register_user(
            email=self.test_email,
            password=self.test_pw,
            name=self.test_name,
            age=self.test_age,
            confirm_password=self.test_pw,
        )

        # Get raw token from resend or simulation
        user = auth.get_user_by_email(self.test_email)
        raw_token, token_hash, expires_at = auth._generate_secure_token()
        with get_db_cursor() as cursor:
            cursor.execute(
                "UPDATE users SET verification_token_hash = ?, verification_expires_at = ? WHERE email = ?",
                (token_hash, expires_at, self.test_email),
            )

        # 1. Invalid token fails
        ok, msg = auth.verify_email_token("invalid-token-1234567890")
        self.assertFalse(ok)

        # 2. Valid token succeeds
        ok, msg = auth.verify_email_token(raw_token)
        self.assertTrue(ok)

        # User is now verified
        user_after = auth.get_user_by_email(self.test_email)
        self.assertEqual(user_after["email_verified"], 1)
        self.assertIsNone(user_after["verification_token_hash"])

        # 3. Single-use: Replay attempt fails
        ok_replay, msg_replay = auth.verify_email_token(raw_token)
        self.assertFalse(ok_replay)

        # 4. Now user can log in successfully
        ok_login, status_code, uid = auth.authenticate(self.test_email, self.test_pw)
        self.assertTrue(ok_login)
        self.assertEqual(status_code, "SUCCESS")

    def test_05_forgot_and_reset_password_flow(self):
        """Tests forgot password, enumeration safety, reset token validation, and password update."""
        # 1. Non-existent email returns generic safe message
        ok, msg = auth.request_password_reset("nonexistent@example.com")
        self.assertTrue(ok)
        self.assertIn("If an account exists", msg)

        # 2. Register and verify user
        auth.register_user(
            email=self.test_email,
            password=self.test_pw,
            name=self.test_name,
            age=self.test_age,
            confirm_password=self.test_pw,
        )

        # Trigger reset
        ok, msg = auth.request_password_reset(self.test_email)
        self.assertTrue(ok)

        # Seed known reset token for testing
        raw_token, token_hash, expires_at = auth._generate_secure_token()
        with get_db_cursor() as cursor:
            cursor.execute(
                "UPDATE users SET reset_token_hash = ?, reset_expires_at = ?, email_verified = 1 WHERE email = ?",
                (token_hash, expires_at, self.test_email),
            )

        # 3. Validate token
        is_valid, v_msg, email = auth.validate_reset_token(raw_token)
        self.assertTrue(is_valid)
        self.assertEqual(email, self.test_email)

        # 4. Perform password reset with new password
        new_pw = "NewFinTwinPass#2027"
        ok_reset, reset_msg = auth.reset_password_with_token(raw_token, new_pw, new_pw)
        self.assertTrue(ok_reset)

        # 5. Old password no longer works
        ok_old, status_old, _ = auth.authenticate(self.test_email, self.test_pw)
        self.assertFalse(ok_old)

        # 6. New password works
        ok_new, status_new, _ = auth.authenticate(self.test_email, new_pw)
        self.assertTrue(ok_new)
        self.assertEqual(status_new, "SUCCESS")

        # 7. Token was invalidated and cannot be reused
        is_valid_again, _, _ = auth.validate_reset_token(raw_token)
        self.assertFalse(is_valid_again)

    def test_06_token_expiration(self):
        """Tests that expired tokens are rejected."""
        auth.register_user(
            email=self.test_email,
            password=self.test_pw,
            name=self.test_name,
            age=self.test_age,
        )

        # Set expired timestamp (1 hour in the past)
        raw_token, token_hash, _ = auth._generate_secure_token()
        expired_time = (
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)
        ).strftime("%Y-%m-%d %H:%M:%S")

        with get_db_cursor() as cursor:
            cursor.execute(
                "UPDATE users SET verification_token_hash = ?, verification_expires_at = ? WHERE email = ?",
                (token_hash, expired_time, self.test_email),
            )

        ok, msg = auth.verify_email_token(raw_token)
        self.assertFalse(ok)
        self.assertIn("expired", msg.lower())

    def test_07_demo_user_authentication(self):
        """Tests that demo user is always seeded as email_verified = 1 and logs in properly."""
        demo_email = os.environ.get("DEMO_EMAIL", "demo@fintwin.app")
        demo_password = os.environ.get("DEMO_PASSWORD", "Demo@123")
        ok, status_code, user_id = auth.authenticate(demo_email, demo_password)
        self.assertTrue(ok)
        self.assertEqual(status_code, "SUCCESS")
        self.assertEqual(user_id, "demo_user")


if __name__ == "__main__":
    unittest.main(verbosity=2)

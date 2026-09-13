"""
tests/test_privacy_security.py
==============================
Automated test suite for Privacy & Security, Download My Data, Change Password,
and Atomic Account Deletion in FinTwin AI.
"""

import os
import sys
import unittest
import json
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from database.connection import init_db, get_db_cursor
from database.db_manager import DBManager
from utils import auth
from utils.security import validate_password
from utils import email_service

# Mock SMTP delivery during automated test runs so real emails are never dispatched
email_service._send_smtp_email = lambda *args, **kwargs: (True, "Test environment mocked email delivery")


class TestPrivacyAndSecurity(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()

    def setUp(self):
        self.user_a_email = "test.privacy.a@example.com"
        self.user_a_pw = "PrivacyPass#A2026"
        self.user_b_email = "test.privacy.b@example.com"
        self.user_b_pw = "PrivacyPass#B2026"

        # Clean up any existing test users
        with get_db_cursor() as cursor:
            cursor.execute("DELETE FROM users WHERE email IN (?, ?)", (self.user_a_email, self.user_b_email))

        # Register User A
        ok_a, _, self.user_a_id = auth.register_user(
            email=self.user_a_email,
            password=self.user_a_pw,
            name="Alice Privacy",
            age=29,
            confirm_password=self.user_a_pw,
        )
        # Register User B
        ok_b, _, self.user_b_id = auth.register_user(
            email=self.user_b_email,
            password=self.user_b_pw,
            name="Bob Privacy",
            age=34,
            confirm_password=self.user_b_pw,
        )

        # Mark both as email verified
        with get_db_cursor() as cursor:
            cursor.execute("UPDATE users SET email_verified = 1 WHERE user_id IN (?, ?)", (self.user_a_id, self.user_b_id))

            # Add test transaction & goal for User A
            cursor.execute(
                "INSERT INTO transactions (transaction_id, user_id, date, category, amount, type) VALUES (?, ?, ?, ?, ?, ?)",
                ("TX_A_01", self.user_a_id, "2026-09-01", "Groceries", 4500.0, "Expense"),
            )
            cursor.execute(
                "INSERT INTO goals (goal_id, user_id, goal_name, target_amount, current_amount, target_date, goal_type) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("G_A_01", self.user_a_id, "Emergency Savings", 500000.0, 150000.0, "2027-12-31", "Safety Net"),
            )

            # Add test transaction & goal for User B
            cursor.execute(
                "INSERT INTO transactions (transaction_id, user_id, date, category, amount, type) VALUES (?, ?, ?, ?, ?, ?)",
                ("TX_B_01", self.user_b_id, "2026-09-02", "Electronics", 35000.0, "Expense"),
            )
            cursor.execute(
                "INSERT INTO goals (goal_id, user_id, goal_name, target_amount, current_amount, target_date, goal_type) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("G_B_01", self.user_b_id, "Car Purchase", 1200000.0, 300000.0, "2028-06-30", "Vehicle"),
            )

    def tearDown(self):
        # Clean up test users
        with get_db_cursor() as cursor:
            cursor.execute("DELETE FROM transactions WHERE user_id IN (?, ?)", (self.user_a_id, self.user_b_id))
            cursor.execute("DELETE FROM goals WHERE user_id IN (?, ?)", (self.user_a_id, self.user_b_id))
            cursor.execute("DELETE FROM digital_twins WHERE user_id IN (?, ?)", (self.user_a_id, self.user_b_id))
            cursor.execute("DELETE FROM users WHERE user_id IN (?, ?)", (self.user_a_id, self.user_b_id))

    def test_01_download_my_data_and_isolation(self):
        """Tests that exported data contains ONLY the authenticated user's records and no secrets."""
        export_a = DBManager.export_user_data(self.user_a_id)
        self.assertIsNotNone(export_a)

        # 1. Structure Verification
        self.assertIn("metadata", export_a)
        self.assertIn("account", export_a)
        self.assertIn("digital_twin", export_a)
        self.assertIn("transactions", export_a)
        self.assertIn("goals", export_a)

        # 2. Sensitive Data Exclusion (No password_hash, salt, or tokens)
        account_data = export_a["account"]
        self.assertNotIn("password_hash", account_data)
        self.assertNotIn("password_salt", account_data)
        self.assertNotIn("verification_token_hash", account_data)
        self.assertNotIn("reset_token_hash", account_data)

        # 3. Correct User A data
        self.assertEqual(account_data["user_id"], self.user_a_id)
        self.assertEqual(account_data["email"], self.user_a_email)
        self.assertEqual(len(export_a["transactions"]), 1)
        self.assertEqual(export_a["transactions"][0]["transaction_id"], "TX_A_01")
        self.assertEqual(export_a["goals"][0]["goal_id"], "G_A_01")

        # 4. Strict Isolation: User A's export must NOT contain User B's records
        tx_ids = [tx["transaction_id"] for tx in export_a["transactions"]]
        self.assertNotIn("TX_B_01", tx_ids)
        goal_ids = [g["goal_id"] for g in export_a["goals"]]
        self.assertNotIn("G_B_01", goal_ids)

    def test_02_change_password_flow(self):
        """Tests in-session password change with current password re-verification."""
        # 1. Incorrect current password fails
        ok, err = auth.change_user_password(
            user_id=self.user_a_id,
            current_password="WrongCurrentPassword123!",
            new_password="NewValidPassword#2026",
            confirm_password="NewValidPassword#2026",
        )
        self.assertFalse(ok)
        self.assertIn("incorrect", err.lower())

        # 2. Mismatched confirmation fails
        ok, err = auth.change_user_password(
            user_id=self.user_a_id,
            current_password=self.user_a_pw,
            new_password="NewValidPassword#2026",
            confirm_password="MismatchPassword#2026",
        )
        self.assertFalse(ok)
        self.assertIn("not match", err.lower())

        # 3. Same password as current fails
        ok, err = auth.change_user_password(
            user_id=self.user_a_id,
            current_password=self.user_a_pw,
            new_password=self.user_a_pw,
            confirm_password=self.user_a_pw,
        )
        self.assertFalse(ok)
        self.assertIn("different", err.lower())

        # 4. Valid password change succeeds
        new_pw = "NewUpdatedSecure#2027"
        ok, msg = auth.change_user_password(
            user_id=self.user_a_id,
            current_password=self.user_a_pw,
            new_password=new_pw,
            confirm_password=new_pw,
        )
        self.assertTrue(ok)

        # 5. Old password no longer authenticates
        ok_old, _, _ = auth.authenticate(self.user_a_email, self.user_a_pw)
        self.assertFalse(ok_old)

        # 6. New password authenticates successfully
        ok_new, status, uid = auth.authenticate(self.user_a_email, new_pw)
        self.assertTrue(ok_new)
        self.assertEqual(status, "SUCCESS")
        self.assertEqual(uid, self.user_a_id)

    def test_03_atomic_account_deletion(self):
        """Tests that permanent account deletion deletes all user records atomically and preserves other users."""
        # 1. Password verification helper check
        self.assertFalse(auth.verify_user_credentials_by_id(self.user_a_id, "WrongPassword#123"))
        self.assertTrue(auth.verify_user_credentials_by_id(self.user_a_id, self.user_a_pw))

        # 2. Execute atomic deletion for User A
        ok, msg = DBManager.delete_user_account_atomic(self.user_a_id)
        self.assertTrue(ok)

        # 3. Verify User A records are completely gone
        with get_db_cursor() as cursor:
            cursor.execute("SELECT count(*) FROM users WHERE user_id = ?", (self.user_a_id,))
            self.assertEqual(cursor.fetchone()[0], 0)
            cursor.execute("SELECT count(*) FROM digital_twins WHERE user_id = ?", (self.user_a_id,))
            self.assertEqual(cursor.fetchone()[0], 0)
            cursor.execute("SELECT count(*) FROM transactions WHERE user_id = ?", (self.user_a_id,))
            self.assertEqual(cursor.fetchone()[0], 0)
            cursor.execute("SELECT count(*) FROM goals WHERE user_id = ?", (self.user_a_id,))
            self.assertEqual(cursor.fetchone()[0], 0)

        # 4. Deleted user cannot log in
        ok_login, _, _ = auth.authenticate(self.user_a_email, self.user_a_pw)
        self.assertFalse(ok_login)

        # 5. Strict Isolation: User B's records are completely intact!
        user_b = DBManager.get_user_profile(self.user_b_id)
        self.assertIsNotNone(user_b)
        self.assertEqual(user_b["email"], self.user_b_email)
        tx_b_df = DBManager.get_transactions(self.user_b_id)
        self.assertEqual(len(tx_b_df), 1)
        self.assertEqual(tx_b_df.iloc[0]["transaction_id"], "TX_B_01")

        # 6. ML models, raw datasets, and demo account are preserved
        demo_user = DBManager.get_user_profile("demo_user")
        self.assertIsNotNone(demo_user)


if __name__ == "__main__":
    unittest.main(verbosity=2)

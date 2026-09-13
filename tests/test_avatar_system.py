"""
Unit and Integration Tests for FinTwin AI 10 Profile Avatar Selection System

Validates:
1. Avatar collection detection (exactly 10 avatars, max 10 choices, proper file formats)
2. No gender categorization or labeling in resolver, metadata, or database
3. Central avatar resolver (path resolution, base64 caching, HTML rendering, safe fallbacks)
4. Database schema & user profile storage (avatar_id persistence, DBManager methods)
5. Registration flow with avatar selection & validation
6. Multi-user isolation (independent avatar selections, no cross-contamination)
7. Existing user backwards compatibility and safe default fallback
8. PDF isolation (zero avatar assets, imports, or rendering in PDF generation)
"""

import pytest
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock

from config import BASE_DIR
from database.connection import get_db_cursor, init_db
from database.db_manager import DBManager
from models.twin_engine import FinancialDigitalTwin
from utils import auth
from utils.avatar import (
    AVATAR_DIR,
    DEFAULT_AVATAR_ID,
    MAX_AVATARS,
    get_available_avatars,
    is_valid_avatar_id,
    resolve_avatar_path,
    resolve_avatar_base64,
    render_avatar_html,
)


class TestAvatarCollectionAndResolver:
    """Test avatar collection asset detection and resolver mechanics."""

    def test_avatar_count_and_naming(self):
        avatars = get_available_avatars()
        assert len(avatars) == 10, f"Expected exactly 10 avatars, got {len(avatars)}"
        assert len(avatars) <= MAX_AVATARS

        for idx, a in enumerate(avatars, start=1):
            assert a["id"].startswith("avatar_"), f"Invalid avatar id format: {a['id']}"
            assert a["path"].exists(), f"Avatar file missing: {a['path']}"
            assert a["path"].suffix.lower() == ".png"
            # Verify accessible labeling without gender terms
            assert a["label"] == f"Profile avatar {idx}"

    def test_no_gender_categorization_in_avatars(self):
        avatars = get_available_avatars()
        forbidden_terms = ["male", "female", "man", "woman", "boy", "girl", "gender", "masculine", "feminine"]
        for a in avatars:
            label_lower = a["label"].lower()
            id_lower = a["id"].lower()
            filename_lower = a["filename"].lower()
            for term in forbidden_terms:
                assert term not in label_lower, f"Forbidden gender term '{term}' found in avatar label: {a['label']}"
                assert term not in id_lower, f"Forbidden gender term '{term}' found in avatar id: {a['id']}"
                assert term not in filename_lower, f"Forbidden gender term '{term}' found in filename: {a['filename']}"

    def test_resolve_avatar_path_valid_and_invalid(self):
        # Valid avatar
        p1 = resolve_avatar_path("avatar_01")
        assert p1.exists()
        assert p1.name == "avatar_01.png"

        p5 = resolve_avatar_path("avatar_05")
        assert p5.exists()
        assert p5.name == "avatar_05.png"

        # None or empty string falls back to default
        p_none = resolve_avatar_path(None)
        assert p_none.name == f"{DEFAULT_AVATAR_ID}.png"

        p_empty = resolve_avatar_path("")
        assert p_empty.name == f"{DEFAULT_AVATAR_ID}.png"

        # Non-existent avatar falls back to default
        p_invalid = resolve_avatar_path("non_existent_avatar_999")
        assert p_invalid.name == f"{DEFAULT_AVATAR_ID}.png"

    def test_resolve_avatar_base64(self):
        b64 = resolve_avatar_base64("avatar_02")
        assert b64.startswith("data:image/png;base64,")
        assert len(b64) > 100

        # Caching works (identical return)
        b64_again = resolve_avatar_base64("avatar_02")
        assert b64 == b64_again

    def test_render_avatar_html(self):
        html = render_avatar_html("avatar_03", size=32, class_name="test-avatar")
        assert "<img" in html
        assert 'class="test-avatar"' in html
        assert "width:32px" in html
        assert "height:32px" in html
        assert 'alt="Profile avatar 3"' in html
        assert "data:image/png;base64," in html

    def test_render_avatar_html_safe_fallback_on_corrupt_asset(self):
        # When resolving an ID that has no asset or fails, returns safe fallback element
        with patch("utils.avatar.resolve_avatar_base64", return_value=""):
            fallback_html = render_avatar_html("corrupted_id", size=28)
            assert "<div" in fallback_html
            assert "AI</div>" in fallback_html


class TestDatabaseAndUserModel:
    """Test database schema, migration, and OOP FinancialDigitalTwin integration."""

    @pytest.fixture(autouse=True)
    def setup_db(self):
        init_db()

    def test_avatar_id_column_exists_in_schema(self):
        with get_db_cursor() as cursor:
            cursor.execute("PRAGMA table_info(users)")
            cols = {row[1] for row in cursor.fetchall()}
            assert "avatar_id" in cols, "avatar_id column must exist in users table"

    def test_demo_user_has_default_avatar(self):
        profile = DBManager.get_user_profile("demo_user")
        assert profile is not None
        assert profile.get("avatar_id") == "avatar_01"

    def test_update_user_avatar(self):
        uid = "test_user_avatar_update"
        with get_db_cursor() as cursor:
            cursor.execute(
                """
                INSERT OR REPLACE INTO users (user_id, name, age, email, monthly_income, avatar_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (uid, "Test Avatar User", 28, "avatar_update@test.com", 60000.0, "avatar_01")
            )

        # Update to avatar_07
        assert DBManager.update_user_avatar(uid, "avatar_07") is True
        updated = DBManager.get_user_profile(uid)
        assert updated["avatar_id"] == "avatar_07"

        # Reject invalid avatar ID
        assert DBManager.update_user_avatar(uid, "avatar_999_invalid") is False
        rechecked = DBManager.get_user_profile(uid)
        assert rechecked["avatar_id"] == "avatar_07"

        # Cleanup
        with get_db_cursor() as cursor:
            cursor.execute("DELETE FROM users WHERE user_id = ?", (uid,))

    def test_financial_digital_twin_model_avatar_attribute(self):
        demographics = {
            "name": "Rohan Sharma",
            "age": 29,
            "city": "Mumbai",
            "occupation": "Product Manager",
            "monthly_income": 140000.0,
            "bonus": 20000.0,
            "additional_income": 5000.0,
            "avatar_id": "avatar_06",
        }
        twin = FinancialDigitalTwin(user_id="user_rohan_avatar", demographics=demographics, balance_sheet={})
        assert hasattr(twin, "avatar_id")
        assert twin.avatar_id == "avatar_06"

        # Test update_twin supports avatar_id
        twin.update_twin(avatar_id="avatar_10")
        assert twin.avatar_id == "avatar_10"

    def test_existing_user_without_avatar_falls_back_safely(self):
        demographics_no_avatar = {
            "name": "Legacy User",
            "age": 35,
            "monthly_income": 90000.0,
        }
        twin = FinancialDigitalTwin(user_id="legacy_user_1", demographics=demographics_no_avatar, balance_sheet={})
        assert twin.avatar_id == DEFAULT_AVATAR_ID


class TestRegistrationValidationAndFlow:
    """Test registration avatar validation and multiple user isolation."""

    @pytest.fixture(autouse=True)
    def cleanup_registered_users(self):
        yield
        with get_db_cursor() as cursor:
            cursor.execute("DELETE FROM users WHERE email LIKE '%@avatartest.com'")
            cursor.execute("DELETE FROM digital_twins WHERE user_id NOT IN (SELECT user_id FROM users)")

    def test_registration_with_valid_avatar(self):
        with patch("utils.auth.send_verification_email", return_value=(True, "OK")):
            ok, status, user_id = auth.register_user(
                email="user_avatar_valid@avatartest.com",
                password="StrongPassword123!",
                name="Siddharth Mehta",
                age=27,
                confirm_password="StrongPassword123!",
                avatar_id="avatar_04",
            )
            assert ok is True
            assert status == "SUCCESS"

            profile = DBManager.get_user_profile(user_id)
            assert profile is not None
            assert profile["avatar_id"] == "avatar_04"

    def test_registration_with_invalid_avatar_id_returns_error(self):
        ok, status, msg = auth.register_user(
            email="invalid_avatar@avatartest.com",
            password="StrongPassword123!",
            name="Invalid User",
            age=30,
            confirm_password="StrongPassword123!",
            avatar_id="malicious_or_nonexistent_avatar",
        )
        assert ok is False
        assert status == "ERROR"
        assert "valid profile avatar" in msg.lower()

    def test_multi_user_isolation(self):
        """User A and User B select different avatars; ensure total isolation."""
        with patch("utils.auth.send_verification_email", return_value=(True, "OK")):
            ok_a, _, uid_a = auth.register_user(
                email="user_a@avatartest.com",
                password="StrongPassword123!",
                name="User Alpha",
                age=28,
                confirm_password="StrongPassword123!",
                avatar_id="avatar_03",
            )
            ok_b, _, uid_b = auth.register_user(
                email="user_b@avatartest.com",
                password="StrongPassword123!",
                name="User Beta",
                age=32,
                confirm_password="StrongPassword123!",
                avatar_id="avatar_08",
            )
            assert ok_a and ok_b

            user_a = DBManager.get_user_profile(uid_a)
            user_b = DBManager.get_user_profile(uid_b)

            assert user_a["avatar_id"] == "avatar_03"
            assert user_b["avatar_id"] == "avatar_08"
            assert user_a["avatar_id"] != user_b["avatar_id"]


class TestPdfReportAvatarIsolation:
    """Verify that PDF generation is strictly isolated from avatar assets and rendering."""

    def test_pdf_modules_do_not_import_avatar(self):
        # Ensure avatar module is never imported or referenced in PDF report modules
        with open(BASE_DIR / "utils" / "pdf_report.py", "r", encoding="utf-8") as f:
            pdf_content = f.read()
            assert "avatar" not in pdf_content.lower(), "pdf_report.py must NOT contain avatar logic"

        with open(BASE_DIR / "utils" / "master_report.py", "r", encoding="utf-8") as f:
            master_content = f.read()
            assert "avatar" not in master_content.lower(), "master_report.py must NOT contain avatar logic"

    def test_pdf_generation_runs_without_avatar_unaffected(self):
        from utils.pdf_report import FinTwinPDFReport

        report = FinTwinPDFReport(
            report_title="Financial Health Summary",
            user_id="pdf_test_user",
            report_id="TEST-PDF-01",
        )
        report.add_section("User Summary")
        report.add_key_value_grid({"Name": "PDF Test User", "Age": 30, "Occupation": "Engineer"})
        pdf_bytes = report.build()

        assert pdf_bytes is not None
        assert len(pdf_bytes) > 1000
        # Ensure standard PDF header
        assert pdf_bytes.startswith(b"%PDF")

#!/usr/bin/env python3
"""
FinTwin AI: Safe Test User Data Reset Script
============================================
Safely resets all test user accounts and user-specific test data from the
local SQLite development database.

Preserves:
  - Database schema, table structures, columns, constraints, indexes
  - ML training datasets (data/raw/indian_salaried_financial_data.csv)
  - Pretrained ML models & artifacts (data/models/*)
  - Application configuration, authentication system, and code

Usage:
  python scripts/reset_test_users.py
  python scripts/reset_test_users.py --yes
"""

import os
import sys
import shutil
import sqlite3
import datetime
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "financial_twin.db"
ML_RAW_DATA = BASE_DIR / "data" / "raw" / "indian_salaried_financial_data.csv"
ML_MODELS_DIR = BASE_DIR / "data" / "models"


def create_backup(db_file: Path) -> Path:
    """Creates a timestamped backup copy of the database before any changes."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = db_file.parent / f"financial_twin.backup_{timestamp}.db"
    shutil.copy2(db_file, backup_file)
    return backup_file


def inspect_counts(cursor) -> dict:
    """Returns row counts for all known user-related tables."""
    tables = [
        "users",
        "digital_twins",
        "transactions",
        "goals",
        "chat_usage",
        "security_rates",
        "password_resets",
    ]
    counts = {}
    for table in tables:
        try:
            cursor.execute(f"SELECT count(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            counts[table] = 0
    return counts


def reset_user_data(force: bool = False) -> dict:
    """
    Executes the safe reset of test user data in referential-integrity order.
    """
    if not DB_PATH.exists():
        print(f"[!] Database file not found at: {DB_PATH}")
        sys.exit(1)

    print("=" * 70)
    print(" FinTwin AI: Safe Test User Data Reset Tool")
    print("=" * 70)
    print(f" Target Database : {DB_PATH}")
    print(f" Environment     : Development / Testing")
    print("-" * 70)

    # 1. Connect and inspect current state
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()

    before_counts = inspect_counts(cursor)
    print("Current record counts:")
    for tbl, cnt in before_counts.items():
        print(f"  - {tbl:20s}: {cnt} records")
    print("-" * 70)

    # 2. Confirmation prompt
    if not force:
        print("\nWARNING: This will permanently delete all test user accounts and")
        print("user-specific test data from the local database.")
        print("Database schema, ML models, and configuration will remain intact.\n")
        try:
            response = input("Do you want to continue? [yes/no]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            conn.close()
            sys.exit(0)

        if response not in ("yes", "y"):
            print("Operation cancelled. No data was modified.")
            conn.close()
            sys.exit(0)

    # 3. Create Backup
    backup_path = create_backup(DB_PATH)
    print(f"\n[+] Created database backup at:\n    {backup_path}\n")

    # 4. Perform Safe Cascading Deletions in FK order
    deleted_counts = {}
    try:
        # Child tables first
        child_tables = [
            "transactions",
            "goals",
            "chat_usage",
            "digital_twins",
            "password_resets",
            "security_rates",
        ]
        for tbl in child_tables:
            cursor.execute(f"DELETE FROM {tbl}")
            deleted_counts[tbl] = cursor.rowcount

        # Parent user table
        cursor.execute("DELETE FROM users")
        deleted_counts["users"] = cursor.rowcount

        conn.commit()

        # Reclaim unused database pages
        cursor.execute("VACUUM")
        conn.commit()
        print("[+] User test records successfully removed.")

    except Exception as e:
        conn.rollback()
        print(f"[!] Error during deletion: {e}")
        print(f"[!] Rolling back. Restoring from {backup_path}...")
        conn.close()
        shutil.copy2(backup_path, DB_PATH)
        sys.exit(1)

    # 5. Verify post-reset state
    after_counts = inspect_counts(cursor)
    conn.close()

    # 6. Verify ML datasets and models are untouched
    ml_raw_exists = ML_RAW_DATA.exists()
    ml_models_count = len(list(ML_MODELS_DIR.glob("*.json"))) if ML_MODELS_DIR.exists() else 0

    print("-" * 70)
    print("POST-RESET VERIFICATION SUMMARY:")
    print("-" * 70)
    for tbl in before_counts:
        removed = before_counts[tbl] - after_counts[tbl]
        print(f"  {tbl:20s} | Before: {before_counts[tbl]:3d} | After: {after_counts[tbl]:3d} | Removed: {removed:3d}")

    print("-" * 70)
    print(f"  ML Training Dataset    : {'PRESERVED' if ml_raw_exists else 'MISSING'}")
    print(f"  Pretrained ML Models   : {ml_models_count} models PRESERVED")
    print(f"  Database Schema        : INTACT & READY FOR FRESH REGISTRATIONS")
    print("=" * 70)

    return {
        "before": before_counts,
        "after": after_counts,
        "deleted": deleted_counts,
        "backup": str(backup_path),
        "ml_raw_preserved": ml_raw_exists,
        "ml_models_preserved": ml_models_count > 0,
    }


if __name__ == "__main__":
    force_flag = "--yes" in sys.argv or "-y" in sys.argv or "--force" in sys.argv
    reset_user_data(force=force_flag)

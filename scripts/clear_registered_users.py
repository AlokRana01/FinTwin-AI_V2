"""
scripts/clear_registered_users.py
=================================
Removes all registered user accounts and their associated digital twins,
transactions, and goals from the database, while preserving the demo user.
"""

import sys
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from database.connection import get_db_cursor, ensure_demo_user_seeded, init_db

def clear_registered_users():
    init_db()
    with get_db_cursor() as cursor:
        # Find registered user IDs
        cursor.execute("SELECT user_id, email FROM users WHERE user_id != 'demo_user'")
        users = cursor.fetchall()
        print(f"Found {len(users)} registered user account(s) to remove:")
        for uid, email in users:
            print(f"  - {uid} ({email})")

        # Delete registered users and cascaded children
        for tbl in ["transactions", "goals", "digital_twins", "chat_usage"]:
            cursor.execute(f"DELETE FROM {tbl} WHERE user_id != 'demo_user'")
        
        cursor.execute("DELETE FROM users WHERE user_id != 'demo_user'")

        # Ensure demo user is intact and verified
        ensure_demo_user_seeded(cursor)

    print("\n[OK] All registered accounts have been removed successfully.")
    print("[OK] Database is clean and ready for fresh registrations.")

if __name__ == "__main__":
    clear_registered_users()

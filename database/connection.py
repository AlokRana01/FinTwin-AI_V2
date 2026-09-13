"""
SQLite Connection Manager
Provides helper functions and context managers to manage SQLite connections in Streamlit.
"""

import sqlite3
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from config import DB_PATH

logger = logging.getLogger(__name__)

def get_connection():
    """
    Establishes and returns a connection to the SQLite database.
    PRAGMA foreign_keys = ON is set on every connection to enforce
    referential integrity (SQLite disables it by default).
    WAL mode, NORMAL synchronous, and 5-second busy timeout are enabled
    for low-latency non-blocking concurrent reads and safe writes.
    """
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10.0)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        # Log only the filename to avoid exposing the full server-side filesystem path
        logger.error(f"Error connecting to database ({Path(DB_PATH).name}): {e}")
        raise

@contextmanager
def get_db_cursor():
    """
    Context manager that yields a cursor and handles commits/rollbacks automatically.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        yield cursor
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database transaction error: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def _load_dotenv() -> None:
    """Reads the project-root .env file and injects variables into os.environ.
    Also synchronizes from st.secrets if running inside Streamlit Cloud / deployment.
    Only sets keys that are not already present, so real environment variables
    always take precedence over the .env file.
    """
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val
        except Exception:
            pass  # Non-fatal; app still starts with the environment defaults

    # Synchronize secrets from Streamlit Cloud / local secrets.toml into os.environ
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for k, v in st.secrets.items():
                if k not in os.environ and isinstance(v, (str, int, float, bool)):
                    os.environ[k] = str(v)
    except Exception:
        pass


_load_dotenv()

# Demo account credentials — loaded from environment so they are never hardcoded.
# Set DEMO_EMAIL and DEMO_PASSWORD in your .env or .streamlit/secrets.toml.
# The fallback password shown below is intentionally non-functional; it will not
# match any hash unless you explicitly set the environment variable to this value.
DEMO_EMAIL    = os.environ.get("DEMO_EMAIL",    "demo@fintwin.app")
DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD", "ChangeMe_SetInEnv!")


def ensure_demo_user_seeded(cursor):
    """
    Inserts or replaces ONE demo account ('Demo User') that a person logs
    into with real credentials (see DEMO_EMAIL / DEMO_PASSWORD above), so
    reviewers can see a populated Digital Twin without registering first.

    This is intentionally the ONLY hard-coded profile in the application.
    It is a demo account behind a login, not a stand-in for the training
    dataset, and it is never inserted alongside thousands of other rows.
    """
    from utils.auth import _hash_password

    password_hash, salt = _hash_password(DEMO_PASSWORD)

    cursor.execute("SELECT avatar_id FROM users WHERE user_id = 'demo_user'")
    existing_demo = cursor.fetchone()
    demo_avatar = existing_demo[0] if (existing_demo and existing_demo[0]) else 'avatar_01'

    cursor.execute(
        """
        INSERT OR REPLACE INTO users (
            user_id, email, password_hash, password_salt,
            name, age, city, occupation, monthly_income, bonus, additional_income,
            email_verified, avatar_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
        """,
        ('demo_user', DEMO_EMAIL, password_hash, salt,
         'Demo User', 30, 'New Delhi', 'Software Engineer', 120000.0, 150000.0, 10000.0, demo_avatar)
    )
    
    cursor.execute(
        """
        INSERT OR REPLACE INTO digital_twins (
            user_id, net_worth, bank_savings, fd_amount, emergency_fund,
            sip_amount, mutual_funds, stocks, ppf_investment, nps_investment,
            loan_amount, car_loan, home_loan, credit_card_debt, monthly_emi,
            health_insurance, life_insurance, rent, groceries, utilities,
            transport, food_delivery, entertainment, shopping, goal_type, goal_amount
        ) VALUES (
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?
        )
        """,
        (
            'demo_user', 1500000.0, 200000.0, 300000.0, 250000.0,
            20000.0, 800000.0, 200000.0, 100000.0, 5000.0,
            400000.0, 400000.0, 0.0, 15000.0, 12000.0,
            300000.0, 15000000.0, 25000.0, 12000.0, 6000.0,
            5000.0, 8000.0, 10000.0, 12000.0, 'House', 8000000.0
        )
    )
    
    cursor.execute(
        """
        INSERT OR REPLACE INTO goals (goal_id, user_id, goal_name, target_amount, current_amount, target_date, goal_type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ('G_demo_user', 'demo_user', 'House Fund', 8000000.0, 40000.0, '2031-12-31', 'Home Purchase')
    )
    
    # Mock transactions for charts
    import datetime
    base_date = datetime.date(2026, 6, 25)
    cats = ["Rent", "Groceries", "Utilities", "Transport", "Food_Delivery", "Entertainment", "Shopping"]
    amounts = [25000.0, 12000.0, 6000.0, 5000.0, 8000.0, 10000.0, 12000.0]
    
    for i, (cat, amount) in enumerate(zip(cats, amounts)):
        cursor.execute(
            """
            INSERT OR REPLACE INTO transactions (transaction_id, user_id, date, category, amount, type)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (f"T_demo_user_{cat}", 'demo_user', (base_date - datetime.timedelta(days=i)).isoformat(), cat, amount, "Expense")
        )
        
    cursor.execute(
        """
        INSERT OR REPLACE INTO transactions (transaction_id, user_id, date, category, amount, type)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("T_demo_user_Income", 'demo_user', (base_date - datetime.timedelta(days=25)).isoformat(), "Salary", 120000.0, "Income")
    )


def reset_demo_account():
    """Clears and re-seeds the demo user account to its default state."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # Delete custom entries
        cursor.execute("DELETE FROM goals WHERE user_id = 'demo_user'")
        cursor.execute("DELETE FROM transactions WHERE user_id = 'demo_user'")
        cursor.execute("DELETE FROM chat_usage WHERE user_id = 'demo_user'")
        
        # Re-seed default demo account
        ensure_demo_user_seeded(cursor)
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Error resetting demo account: {e}")
    finally:
        cursor.close()
        conn.close()


def init_db():
    """
    Initializes the database schema using schema.sql.
    """
    from config import BASE_DIR

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        table_exists = cursor.fetchone()

        if not table_exists:
            logger.info("Initializing database schema...")
            schema_path = BASE_DIR / "database" / "schema.sql"
            if not schema_path.exists():
                logger.error(f"Schema file not found at {schema_path}")
                return
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            cursor.executescript(schema_sql)
            conn.commit()
            logger.info("Database schema applied successfully.")

        # Ensure migrations apply to pre-existing DBs
        _ensure_auth_columns(cursor)
        _ensure_chat_usage_table(cursor)

        _purge_legacy_training_rows(cursor)

        ensure_demo_user_seeded(cursor)
        conn.commit()
        logger.info("Database ready.")

    except Exception as e:
        conn.rollback()
        logger.error(f"Error initializing database: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def _ensure_auth_columns(cursor):
    """Adds email/password_hash/password_salt and verification/reset token columns if migrating an older DB file."""
    cursor.execute("PRAGMA table_info(users)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    for col, decl in [
        ("email", "TEXT"),
        ("password_hash", "TEXT"),
        ("password_salt", "TEXT"),
        ("email_verified", "INTEGER DEFAULT 0"),
        ("verification_token_hash", "TEXT"),
        ("verification_expires_at", "TIMESTAMP"),
        ("reset_token_hash", "TEXT"),
        ("reset_expires_at", "TIMESTAMP"),
        ("avatar_id", "TEXT DEFAULT 'avatar_01'"),
    ]:
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE users ADD COLUMN {col} {decl}")
    cursor.execute("UPDATE users SET avatar_id = 'avatar_01' WHERE avatar_id IS NULL OR avatar_id = ''")


def _ensure_chat_usage_table(cursor):
    """Creates the FinBot daily-usage tracking table if it doesn't exist yet."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_usage (
            user_id TEXT NOT NULL,
            usage_date DATE NOT NULL,
            message_count INTEGER DEFAULT 0,
            PRIMARY KEY (user_id, usage_date)
        )
    """)


def _purge_legacy_training_rows(cursor):
    """
    Deletes any user rows that were bulk-inserted from the training CSV.
    """
    cursor.execute(
        "SELECT user_id FROM users WHERE email IS NULL AND user_id != 'demo_user'"
    )
    legacy_ids = [row[0] for row in cursor.fetchall()]
    if not legacy_ids:
        return

    logger.warning(f"Purging {len(legacy_ids)} legacy training-data rows...")
    CHUNK = 500
    for i in range(0, len(legacy_ids), CHUNK):
        chunk_ids = legacy_ids[i:i + CHUNK]
        placeholders = ",".join("?" for _ in chunk_ids)
        for table in ("transactions", "goals", "digital_twins", "users"):
            cursor.execute(f"DELETE FROM {table} WHERE user_id IN ({placeholders})", chunk_ids)

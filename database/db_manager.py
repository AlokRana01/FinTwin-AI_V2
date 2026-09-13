"""
Database CRUD Manager
Encapsulates CRUD operations on database tables using Python SQLite connections.
"""

import json
import os
import sqlite3
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from database.connection import get_db_cursor, get_connection

from utils.validators import (
    is_finite_number,
    validate_finite_number,
    validate_age,
    validate_text,
    sanitize_text,
    MAX_FINANCIAL_AMOUNT,
    MAX_MONTHLY_INCOME,
)

_COHORT_BENCHMARK_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "models", "cohort_benchmarks.json"
)
MIN_REAL_COHORT_SIZE = 5  # below this many real users of an occupation, blend in the offline benchmark

class DBManager:
    """
    Handles all read and write queries to the SQLite Database.
    """
    
    @staticmethod
    def get_all_users() -> List[Dict[str, Any]]:
        """
        Retrieves all registered users. NOT used by any Streamlit page — the
        live app only ever loads the single authenticated user's own record
        (see utils/session.py). Kept only for admin/maintenance tooling.
        """
        with get_db_cursor() as cursor:
            cursor.execute("SELECT user_id, name, occupation, monthly_income FROM users ORDER BY name LIMIT 200")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    @staticmethod
    def get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves user demographics and profile details by user ID.
        """
        with get_db_cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (str(user_id).strip(),))
            row = cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    def save_user_profile(user_data: Dict[str, Any]) -> bool:
        """
        Saves or updates user demographics and profile details with strict validation.
        """
        uid = str(user_data.get("user_id", "")).strip()
        if not uid:
            raise ValueError("user_id is required")

        name = sanitize_text(user_data.get("name", ""), max_len=80)
        if not name:
            raise ValueError("name is required and cannot be empty")

        age_ok, age_err, age_val = validate_age(user_data.get("age", 0))
        if not age_ok:
            raise ValueError(age_err)

        for inc_key, inc_label in [("monthly_income", "Monthly income"), ("bonus", "Bonus"), ("additional_income", "Additional income")]:
            val = user_data.get(inc_key, 0.0)
            ok, err, _ = validate_finite_number(val, inc_label, min_val=0.0, max_val=MAX_MONTHLY_INCOME)
            if not ok:
                raise ValueError(err)

        with get_db_cursor() as cursor:
            avatar_id = user_data.get("avatar_id")
            if avatar_id:
                cursor.execute(
                    """
                    INSERT INTO users (user_id, name, age, city, occupation, monthly_income, bonus, additional_income, avatar_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        name = excluded.name,
                        age = excluded.age,
                        city = excluded.city,
                        occupation = excluded.occupation,
                        monthly_income = excluded.monthly_income,
                        bonus = excluded.bonus,
                        additional_income = excluded.additional_income,
                        avatar_id = excluded.avatar_id
                    """,
                    (
                        uid,
                        name,
                        age_val,
                        sanitize_text(user_data.get("city", ""), max_len=100),
                        sanitize_text(user_data.get("occupation", ""), max_len=100),
                        float(user_data.get("monthly_income", 0.0)),
                        float(user_data.get("bonus", 0.0)),
                        float(user_data.get("additional_income", 0.0)),
                        sanitize_text(str(avatar_id), max_len=50)
                    )
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO users (user_id, name, age, city, occupation, monthly_income, bonus, additional_income)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        name = excluded.name,
                        age = excluded.age,
                        city = excluded.city,
                        occupation = excluded.occupation,
                        monthly_income = excluded.monthly_income,
                        bonus = excluded.bonus,
                        additional_income = excluded.additional_income
                    """,
                    (
                        uid,
                        name,
                        age_val,
                        sanitize_text(user_data.get("city", ""), max_len=100),
                        sanitize_text(user_data.get("occupation", ""), max_len=100),
                        float(user_data.get("monthly_income", 0.0)),
                        float(user_data.get("bonus", 0.0)),
                        float(user_data.get("additional_income", 0.0))
                    )
                )
            return True

    @staticmethod
    def update_user_avatar(user_id: str, avatar_id: str) -> bool:
        """
        Updates the profile avatar for a user with validation.
        """
        uid = str(user_id).strip()
        if not uid:
            return False
        aid = str(avatar_id).strip()
        if not aid:
            return False
        from utils.avatar import is_valid_avatar_id
        if not is_valid_avatar_id(aid):
            return False

        with get_db_cursor() as cursor:
            cursor.execute(
                "UPDATE users SET avatar_id = ? WHERE user_id = ?",
                (aid, uid)
            )
            return cursor.rowcount > 0

    @staticmethod
    def get_digital_twin(user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the financial digital twin state metrics for a user.
        """
        with get_db_cursor() as cursor:
            cursor.execute("SELECT * FROM digital_twins WHERE user_id = ?", (str(user_id).strip(),))
            row = cursor.fetchone()
            return dict(row) if row else None

    @staticmethod
    def save_digital_twin(twin_data: Dict[str, Any]) -> bool:
        """
        Saves or updates digital twin state metrics with strict finite number checks.
        """
        uid = str(twin_data.get("user_id", "")).strip()
        if not uid:
            raise ValueError("user_id is required")
        
        numeric_fields = [
            "net_worth", "bank_savings", "fd_amount", "emergency_fund",
            "sip_amount", "mutual_funds", "stocks", "ppf_investment", "nps_investment",
            "loan_amount", "car_loan", "home_loan", "credit_card_debt", "monthly_emi",
            "health_insurance", "life_insurance", "rent", "groceries", "utilities",
            "transport", "food_delivery", "entertainment", "shopping", "goal_amount"
        ]
        for field in numeric_fields:
            if field in twin_data:
                val = twin_data[field]
                # Net worth can be negative if liabilities exceed assets, other balance items must be >= 0
                min_v = -MAX_FINANCIAL_AMOUNT if field == "net_worth" else 0.0
                ok, err, _ = validate_finite_number(val, field, min_val=min_v, max_val=MAX_FINANCIAL_AMOUNT)
                if not ok:
                    raise ValueError(err)

        with get_db_cursor() as cursor:
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
                    twin_data["user_id"],
                    twin_data.get("net_worth", 0.0),
                    twin_data.get("bank_savings", 0.0),
                    twin_data.get("fd_amount", 0.0),
                    twin_data.get("emergency_fund", 0.0),
                    twin_data.get("sip_amount", 0.0),
                    twin_data.get("mutual_funds", 0.0),
                    twin_data.get("stocks", 0.0),
                    twin_data.get("ppf_investment", 0.0),
                    twin_data.get("nps_investment", 0.0),
                    twin_data.get("loan_amount", 0.0),
                    twin_data.get("car_loan", 0.0),
                    twin_data.get("home_loan", 0.0),
                    twin_data.get("credit_card_debt", 0.0),
                    twin_data.get("monthly_emi", 0.0),
                    twin_data.get("health_insurance", 0.0),
                    twin_data.get("life_insurance", 0.0),
                    twin_data.get("rent", 0.0),
                    twin_data.get("groceries", 0.0),
                    twin_data.get("utilities", 0.0),
                    twin_data.get("transport", 0.0),
                    twin_data.get("food_delivery", 0.0),
                    twin_data.get("entertainment", 0.0),
                    twin_data.get("shopping", 0.0),
                    twin_data.get("goal_type", ""),
                    twin_data.get("goal_amount", 0.0)
                )
            )
            return True

    @staticmethod
    def get_transactions(user_id: str) -> pd.DataFrame:
        """
        Fetches historical transactions for a user as a Pandas DataFrame.
        """
        conn = get_connection()
        try:
            df = pd.read_sql_query(
                "SELECT * FROM transactions WHERE user_id = ? ORDER BY date DESC",
                conn,
                params=(user_id,)
            )
            return df
        finally:
            conn.close()

    @staticmethod
    def save_transaction(transaction_data: Dict[str, Any]) -> bool:
        """
        Logs a single transaction in the database ledger.
        """
        with get_db_cursor() as cursor:
            cursor.execute(
                """
                INSERT OR REPLACE INTO transactions (transaction_id, user_id, date, category, amount, type)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    transaction_data["transaction_id"],
                    transaction_data["user_id"],
                    transaction_data["date"],
                    transaction_data["category"],
                    transaction_data["amount"],
                    transaction_data["type"]
                )
            )
            return True

    @staticmethod
    def get_goals(user_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all active financial goals for a user.
        """
        with get_db_cursor() as cursor:
            cursor.execute("SELECT * FROM goals WHERE user_id = ?", (user_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    @staticmethod
    def save_goal(goal_data: Dict[str, Any]) -> bool:
        """
        Saves or updates a financial goal.
        """
        with get_db_cursor() as cursor:
            cursor.execute(
                """
                INSERT OR REPLACE INTO goals (goal_id, user_id, goal_name, target_amount, current_amount, target_date, goal_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    goal_data["goal_id"],
                    goal_data["user_id"],
                    goal_data["goal_name"],
                    goal_data["target_amount"],
                    goal_data.get("current_amount", 0.0),
                    goal_data["target_date"],
                    goal_data.get("goal_type", "")
                )
            )
            return True

    @staticmethod
    def _load_offline_cohort_benchmark(occupation: str) -> Dict[str, float]:
        """
        Reads the anonymous, aggregate-only cohort benchmark produced once by
        training/train_offline.py from the training dataset. Contains no
        individual records — only mean spend/savings figures per occupation.
        Used as a cold-start fallback until enough real users have registered.
        """
        if not os.path.exists(_COHORT_BENCHMARK_PATH):
            return {}
        with open(_COHORT_BENCHMARK_PATH, "r") as f:
            benchmarks = json.load(f)
        return benchmarks.get(occupation) or benchmarks.get("_overall", {})

    @staticmethod
    def get_cohort_averages(occupation: str) -> Dict[str, float]:
        """
        Retrieves average financial metrics for REAL registered users sharing
        the given occupation. Only ever averages actual account data — never
        the raw training dataset. If fewer than MIN_REAL_COHORT_SIZE real
        users of that occupation exist yet, falls back to a precomputed,
        aggregate-only benchmark from the offline training pipeline so the
        feature still works for a brand-new deployment.
        """
        conn = get_connection()
        try:
            count_df = pd.read_sql_query(
                "SELECT COUNT(*) as n FROM users WHERE occupation = ? AND email IS NOT NULL",
                conn, params=(occupation,)
            )
            real_count = int(count_df.iloc[0]["n"]) if not count_df.empty else 0

            if real_count < MIN_REAL_COHORT_SIZE:
                return DBManager._load_offline_cohort_benchmark(occupation)

            query = """
            SELECT 
                AVG(rent) as rent,
                AVG(groceries) as groceries,
                AVG(utilities) as utilities,
                AVG(transport) as transport,
                AVG(food_delivery) as food_delivery,
                AVG(entertainment) as entertainment,
                AVG(shopping) as shopping,
                AVG(sip_amount) as sip_amount,
                AVG(monthly_emi) as monthly_emi,
                AVG(emergency_fund) as emergency_fund
            FROM digital_twins dt
            JOIN users u ON dt.user_id = u.user_id
            WHERE u.occupation = ? AND u.email IS NOT NULL
            """
            df = pd.read_sql_query(query, conn, params=(occupation,))
            if not df.empty:
                return df.iloc[0].to_dict()
            return DBManager._load_offline_cohort_benchmark(occupation)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error fetching cohort averages: {e}")
            return DBManager._load_offline_cohort_benchmark(occupation)
        finally:
            conn.close()

    @staticmethod
    def get_population_features() -> pd.DataFrame:
        """
        Returns a DataFrame of REGISTERED users with the 4 clustering
        features: savings_ratio, expense_ratio, investment_ratio,
        debt_to_income_ratio.

        NOT used by any live Streamlit page anymore. The clustering model
        and its cohort statistics are trained once offline
        (training/train_offline.py) against the training dataset and loaded
        from data/models/ at runtime (see models/clustering.py,
        FinancialPersonalityClusterer.load()). This method is retained only
        for future retraining tooling that wants to incorporate real user
        data, and it is scoped to registered accounts (email IS NOT NULL) —
        it never included the raw training CSV rows in the first place.

        Uses a single JOIN query for efficiency — avoids N individual twin fetches.
        Rows with zero or null income are excluded to prevent division errors.
        """
        conn = get_connection()
        try:
            query = """
            SELECT
                u.user_id,
                u.name,
                u.occupation,
                u.monthly_income,
                u.additional_income,
                (u.monthly_income + u.additional_income)                         AS total_income,
                (dt.rent + dt.groceries + dt.utilities + dt.transport +
                 dt.food_delivery + dt.entertainment + dt.shopping)               AS total_expenses,
                dt.sip_amount,
                dt.monthly_emi,
                dt.loan_amount,
                dt.bank_savings,
                dt.fd_amount,
                dt.emergency_fund,
                dt.mutual_funds,
                dt.stocks,
                dt.ppf_investment,
                dt.nps_investment
            FROM users u
            JOIN digital_twins dt ON u.user_id = dt.user_id
            WHERE (u.monthly_income + u.additional_income) > 0
              AND u.email IS NOT NULL
            """
            df = pd.read_sql_query(query, conn)

            if df.empty:
                return df

            # Compute ratio features
            df["savings_ratio"]        = (df["total_income"] - df["total_expenses"] - df["monthly_emi"]) / df["total_income"]
            df["expense_ratio"]        = df["total_expenses"] / df["total_income"]
            df["investment_ratio"]     = df["sip_amount"]    / df["total_income"]
            df["debt_to_income_ratio"] = df["monthly_emi"]   / df["total_income"]

            # Clip to [0, 1] — guard against negative or > 1 edge cases
            for col in ["savings_ratio", "expense_ratio", "investment_ratio", "debt_to_income_ratio"]:
                df[col] = df[col].clip(0.0, 1.0)

            return df
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error fetching population features: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    # ── FinBot Daily Usage Tracking ─────────────────────────────────────────

    @staticmethod
    def get_chat_usage_today(user_id: str) -> int:
        """Returns how many FinBot messages this user has sent today."""
        import datetime
        today = datetime.date.today().isoformat()
        with get_db_cursor() as cursor:
            cursor.execute(
                "SELECT message_count FROM chat_usage WHERE user_id = ? AND usage_date = ?",
                (user_id, today),
            )
            row = cursor.fetchone()
            return int(row["message_count"]) if row else 0

    @staticmethod
    def check_and_consume_quota(user_id: str, limit: int) -> Tuple[bool, int]:
        """
        Atomically checks whether today's message count for this user is below `limit`
        and, if so, increments the message count by 1 in an immediate write transaction.

        Returns:
            (allowed: bool, new_count: int)
            allowed=True: quota was available and exactly one message was consumed.
            allowed=False: quota was exhausted (current count >= limit); no increment occurred.
        """
        import datetime
        today = datetime.date.today().isoformat()
        conn = get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            cursor.execute(
                "SELECT message_count FROM chat_usage WHERE user_id = ? AND usage_date = ?",
                (user_id, today),
            )
            row = cursor.fetchone()
            current = int(row["message_count"]) if row else 0

            if current >= limit:
                conn.commit()
                return False, current

            try:
                cursor.execute(
                    """
                    INSERT INTO chat_usage (user_id, usage_date, message_count)
                    VALUES (?, ?, 1)
                    ON CONFLICT(user_id, usage_date)
                    DO UPDATE SET message_count = message_count + 1
                    RETURNING message_count
                    """,
                    (user_id, today),
                )
                new_row = cursor.fetchone()
                if new_row is not None:
                    new_count = int(new_row["message_count"])
                else:
                    new_count = current + 1
            except Exception:
                cursor.execute(
                    """
                    INSERT INTO chat_usage (user_id, usage_date, message_count)
                    VALUES (?, ?, 1)
                    ON CONFLICT(user_id, usage_date)
                    DO UPDATE SET message_count = message_count + 1
                    """,
                    (user_id, today),
                )
                cursor.execute(
                    "SELECT message_count FROM chat_usage WHERE user_id = ? AND usage_date = ?",
                    (user_id, today),
                )
                fallback_row = cursor.fetchone()
                new_count = int(fallback_row["message_count"]) if fallback_row else current + 1

            conn.commit()
            return True, new_count
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()

    @staticmethod
    def increment_chat_usage(user_id: str) -> int:
        """Increments today's FinBot message count for this user and returns the new total atomically."""
        allowed, count = DBManager.check_and_consume_quota(user_id, limit=999999)
        return count

    # ── Privacy & Data Management (Export & Deletion) ───────────────────────

    @staticmethod
    def export_user_data(user_id: str) -> Optional[Dict[str, Any]]:
        """
        Exports all personal and financial data owned by the authenticated user
        in a structured, portable JSON format.
        Strictly scrubs sensitive authentication secrets (password_hash, salt, tokens).
        """
        import datetime
        user = DBManager.get_user_profile(user_id)
        if not user:
            return None

        # Clean sensitive auth fields from user profile
        scrubbed_account = {
            "user_id": user.get("user_id"),
            "name": user.get("name"),
            "email": user.get("email"),
            "age": user.get("age"),
            "city": user.get("city", ""),
            "occupation": user.get("occupation", ""),
            "monthly_income": user.get("monthly_income", 0.0),
            "bonus": user.get("bonus", 0.0),
            "additional_income": user.get("additional_income", 0.0),
            "email_verified": bool(user.get("email_verified", 0)),
            "created_at": str(user.get("created_at", "")),
        }

        # Digital Twin
        twin_data = DBManager.get_digital_twin(user_id) or {}
        scrubbed_twin = {k: v for k, v in twin_data.items() if k != "user_id"}

        # Transactions
        tx_df = DBManager.get_transactions(user_id)
        transactions = tx_df.to_dict(orient="records") if not tx_df.empty else []

        # Goals
        goals = DBManager.get_goals(user_id)

        # Chat Usage
        with get_db_cursor() as cursor:
            cursor.execute(
                "SELECT usage_date, message_count FROM chat_usage WHERE user_id = ? ORDER BY usage_date DESC",
                (user_id,),
            )
            chat_usage = [dict(row) for row in cursor.fetchall()]

        export_payload = {
            "metadata": {
                "application": "FinTwin AI",
                "version": "2.0.0",
                "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "description": "Personal financial health and digital twin data export.",
            },
            "account": scrubbed_account,
            "digital_twin": scrubbed_twin,
            "transactions": transactions,
            "goals": goals,
            "ai_coach_usage": chat_usage,
        }

        return export_payload

    @staticmethod
    def delete_user_account_atomic(user_id: str) -> Tuple[bool, str]:
        """
        Permanently and atomically deletes an authenticated user account and
        all associated personal records from the SQLite database.
        Uses a strict single database transaction with automatic rollback.
        Preserves all ML models, datasets, and static application assets.
        """
        import logging
        logger = logging.getLogger(__name__)

        if not user_id or not str(user_id).strip():
            return False, "Invalid user identifier."

        conn = get_connection()
        cursor = conn.cursor()
        try:
            # Enforce foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON;")

            # 1. Delete dependent child records in referential integrity order
            cursor.execute("DELETE FROM transactions WHERE user_id = ?", (user_id,))
            cursor.execute("DELETE FROM goals WHERE user_id = ?", (user_id,))
            cursor.execute("DELETE FROM chat_usage WHERE user_id = ?", (user_id,))
            cursor.execute("DELETE FROM digital_twins WHERE user_id = ?", (user_id,))

            # 2. Delete parent user record
            cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))

            conn.commit()
            logger.info(f"Account for user_id '{user_id}' was atomically deleted.")
            return True, "Your account and all associated personal data have been permanently deleted."

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed atomic account deletion for '{user_id}': {e}", exc_info=True)
            return False, "We couldn't complete the account deletion. No changes were made to your account. Please try again."

        finally:
            cursor.close()
            conn.close()

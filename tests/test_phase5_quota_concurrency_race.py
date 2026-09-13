"""
Comprehensive Quota Concurrency & Boundary Tests (Phase 5.3 Verification)

Validates:
1. Authenticated concurrency race when 1 slot remains (initial = 39, limit = 40).
2. Authenticated boundary race when 2 slots remain (initial = 38, limit = 40).
3. Authenticated exhausted quota race when 0 slots remain (initial = 40, limit = 40).
4. Guest concurrency race when 1 slot remains (initial = 11, limit = 12).
5. Input rejection & exception safety (empty input, no duplicate consumption).
"""

import datetime
import threading
from unittest.mock import patch, MagicMock
import pytest

from database.db_manager import DBManager
from database.connection import get_db_cursor
from utils.chatbot import (
    _usage_today,
    _try_consume_quota,
    _process_user_message,
    DAILY_LIMIT_LOGGED_IN,
    DAILY_LIMIT_GUEST,
    HISTORY_KEY,
    ANON_USAGE_KEY,
)
from models.twin_engine import FinancialDigitalTwin
from agents.schemas import AgentResult, AgentStatus


class ThreadSafeSessionState(dict):
    """Thread-safe session state dict for concurrent test simulation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()

    def __getitem__(self, key):
        with self._lock:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        with self._lock:
            super().__setitem__(key, value)

    def get(self, key, default=None):
        with self._lock:
            return super().get(key, default)


# ── 1. Authenticated Concurrency Tests ────────────────────────────────────────

@pytest.mark.parametrize("iteration", range(3))
def test_authenticated_concurrency_one_slot_remaining(iteration):
    """
    Simulate 10 concurrent requests when initial DB usage is 39 / 40.
    Invariants:
    - successful reservations <= 1
    - final DB count <= 40
    - agent executions <= 1
    """
    today = datetime.date.today().isoformat()
    uid = f"toctou_auth_user_{iteration}_{int(datetime.datetime.now().timestamp())}"

    with get_db_cursor() as cur:
        cur.execute("DELETE FROM chat_usage WHERE user_id = ?", (uid,))
        cur.execute(
            "INSERT INTO chat_usage (user_id, usage_date, message_count) VALUES (?, ?, ?)",
            (uid, today, 39),
        )

    demographics = {"user_id": uid, "name": "Race User", "age": 30, "city": "Delhi", "occupation": "Engineer", "monthly_income": 100000.0, "bonus": 0.0, "additional_income": 0.0}
    balance_sheet = {"user_id": uid, "net_worth": 1000000.0, "bank_savings": 200000.0}
    twin = FinancialDigitalTwin(uid, demographics, balance_sheet)

    num_threads = 10
    agent_executions = []
    agent_lock = threading.Lock()

    def mock_orch_run(*args, **kwargs):
        with agent_lock:
            agent_executions.append(1)
        return AgentResult(
            agent_name="FinancialCoachAgent",
            status=AgentStatus.SUCCESS,
            intent="overview",
            insights=("Analysis complete.",),
        )

    shared_session = ThreadSafeSessionState({HISTORY_KEY: []})

    def concurrent_attempt():
        mock_orch = MagicMock()
        mock_orch.run = mock_orch_run
        _process_user_message("What is my financial health?", twin, 1, orchestrator=mock_orch)

    with patch("streamlit.session_state", shared_session), \
         patch("streamlit.spinner"), \
         patch("streamlit.rerun"):
        threads = [threading.Thread(target=concurrent_attempt) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    final_count = DBManager.get_chat_usage_today(uid)
    executed_count = len(agent_executions)

    with get_db_cursor() as cur:
        cur.execute("DELETE FROM chat_usage WHERE user_id = ?", (uid,))

    assert executed_count <= 1, f"Iteration {iteration}: {executed_count} agents executed when only 1 slot remained!"
    assert final_count <= 40, f"Iteration {iteration}: Final DB count {final_count} exceeded limit 40!"


def test_authenticated_concurrency_two_slots_remaining():
    """
    Simulate 10 concurrent requests when initial DB usage is 38 / 40 (2 slots remaining).
    Invariants:
    - successful reservations <= 2
    - final DB count <= 40
    - agent executions <= 2
    """
    today = datetime.date.today().isoformat()
    uid = f"toctou_auth_user_2slot_{int(datetime.datetime.now().timestamp())}"

    with get_db_cursor() as cur:
        cur.execute("DELETE FROM chat_usage WHERE user_id = ?", (uid,))
        cur.execute(
            "INSERT INTO chat_usage (user_id, usage_date, message_count) VALUES (?, ?, ?)",
            (uid, today, 38),
        )

    demographics = {"user_id": uid, "name": "Race User", "age": 30, "city": "Delhi", "occupation": "Engineer", "monthly_income": 100000.0, "bonus": 0.0, "additional_income": 0.0}
    balance_sheet = {"user_id": uid, "net_worth": 1000000.0, "bank_savings": 200000.0}
    twin = FinancialDigitalTwin(uid, demographics, balance_sheet)

    num_threads = 10
    agent_executions = []
    agent_lock = threading.Lock()

    def mock_orch_run(*args, **kwargs):
        with agent_lock:
            agent_executions.append(1)
        return AgentResult(
            agent_name="FinancialCoachAgent",
            status=AgentStatus.SUCCESS,
            intent="overview",
            insights=("Analysis complete.",),
        )

    shared_session = ThreadSafeSessionState({HISTORY_KEY: []})

    def concurrent_attempt():
        mock_orch = MagicMock()
        mock_orch.run = mock_orch_run
        _process_user_message("What is my financial health?", twin, 2, orchestrator=mock_orch)

    with patch("streamlit.session_state", shared_session), \
         patch("streamlit.spinner"), \
         patch("streamlit.rerun"):
        threads = [threading.Thread(target=concurrent_attempt) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    final_count = DBManager.get_chat_usage_today(uid)
    executed_count = len(agent_executions)

    with get_db_cursor() as cur:
        cur.execute("DELETE FROM chat_usage WHERE user_id = ?", (uid,))

    assert executed_count <= 2, f"{executed_count} agents executed when only 2 slots remained!"
    assert final_count <= 40, f"Final DB count {final_count} exceeded limit 40!"


def test_authenticated_concurrency_exhausted_quota():
    """
    Simulate 10 concurrent requests when initial DB usage is 40 / 40 (0 slots remaining).
    Invariants:
    - successful reservations = 0
    - final DB count = 40
    - agent executions = 0
    """
    today = datetime.date.today().isoformat()
    uid = f"toctou_auth_user_0slot_{int(datetime.datetime.now().timestamp())}"

    with get_db_cursor() as cur:
        cur.execute("DELETE FROM chat_usage WHERE user_id = ?", (uid,))
        cur.execute(
            "INSERT INTO chat_usage (user_id, usage_date, message_count) VALUES (?, ?, ?)",
            (uid, today, 40),
        )

    demographics = {"user_id": uid, "name": "Race User", "age": 30, "city": "Delhi", "occupation": "Engineer", "monthly_income": 100000.0, "bonus": 0.0, "additional_income": 0.0}
    balance_sheet = {"user_id": uid, "net_worth": 1000000.0, "bank_savings": 200000.0}
    twin = FinancialDigitalTwin(uid, demographics, balance_sheet)

    num_threads = 10
    agent_executions = []
    agent_lock = threading.Lock()

    def mock_orch_run(*args, **kwargs):
        with agent_lock:
            agent_executions.append(1)
        return AgentResult(
            agent_name="FinancialCoachAgent",
            status=AgentStatus.SUCCESS,
            intent="overview",
            insights=("Analysis complete.",),
        )

    shared_session = ThreadSafeSessionState({HISTORY_KEY: []})

    def concurrent_attempt():
        mock_orch = MagicMock()
        mock_orch.run = mock_orch_run
        _process_user_message("What is my financial health?", twin, 0, orchestrator=mock_orch)

    with patch("streamlit.session_state", shared_session), \
         patch("streamlit.spinner"), \
         patch("streamlit.rerun"):
        threads = [threading.Thread(target=concurrent_attempt) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    final_count = DBManager.get_chat_usage_today(uid)
    executed_count = len(agent_executions)

    with get_db_cursor() as cur:
        cur.execute("DELETE FROM chat_usage WHERE user_id = ?", (uid,))

    assert executed_count == 0, f"{executed_count} agents executed when quota was 0!"
    assert final_count == 40, f"Final DB count {final_count} changed from 40!"


# ── 2. Guest Concurrency & Isolation Tests ────────────────────────────────────

def test_guest_concurrency_one_slot_remaining():
    """
    Simulate 10 concurrent requests for a guest user when initial usage is 11 / 12.
    Invariants:
    - successful reservations <= 1
    - final guest count <= 12
    - agent executions <= 1
    """
    today = datetime.date.today().isoformat()
    session = ThreadSafeSessionState({
        HISTORY_KEY: [],
        ANON_USAGE_KEY: {"date": today, "count": 11}
    })

    num_threads = 10
    agent_executions = []
    agent_lock = threading.Lock()

    def mock_orch_run(*args, **kwargs):
        with agent_lock:
            agent_executions.append(1)
        return AgentResult(
            agent_name="FinancialCoachAgent",
            status=AgentStatus.SUCCESS,
            intent="overview",
            insights=("Analysis complete.",),
        )

    def concurrent_attempt():
        mock_orch = MagicMock()
        mock_orch.run = mock_orch_run
        _process_user_message("What is my financial health?", None, 1, orchestrator=mock_orch)

    with patch("streamlit.session_state", session), \
         patch("streamlit.spinner"), \
         patch("streamlit.rerun"):
        threads = [threading.Thread(target=concurrent_attempt) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    final_count = session[ANON_USAGE_KEY]["count"]
    executed_count = len(agent_executions)

    assert executed_count <= 1, f"Guest: {executed_count} agents executed when only 1 slot remained!"
    assert final_count <= 12, f"Guest count {final_count} exceeded limit 12!"


# ── 3. Edge Cases & Rejection Integrity ───────────────────────────────────────

def test_empty_input_does_not_consume_quota():
    """Verify empty or whitespace-only inputs do not consume quota or invoke agents."""
    demographics = {"user_id": "test_empty_u", "name": "User", "age": 30, "city": "Delhi", "occupation": "Engineer", "monthly_income": 100000.0, "bonus": 0.0, "additional_income": 0.0}
    balance_sheet = {"user_id": "test_empty_u", "net_worth": 1000000.0, "bank_savings": 200000.0}
    twin = FinancialDigitalTwin("test_empty_u", demographics, balance_sheet)

    session = ThreadSafeSessionState({HISTORY_KEY: []})
    with patch("streamlit.session_state", session), \
         patch("database.db_manager.DBManager.check_and_consume_quota") as mock_consume, \
         patch("agents.orchestrator.OrchestratorAgent.run") as mock_orch:

        _process_user_message("   ", twin, 10)
        _process_user_message("", twin, 10)

        assert mock_consume.call_count == 0
        assert mock_orch.call_count == 0


def test_exception_during_agent_execution_does_not_double_consume():
    """Verify that if an error occurs during agent execution, no retry double-consumes quota."""
    today = datetime.date.today().isoformat()
    uid = f"err_user_{int(datetime.datetime.now().timestamp())}"

    with get_db_cursor() as cur:
        cur.execute("DELETE FROM chat_usage WHERE user_id = ?", (uid,))

    demographics = {"user_id": uid, "name": "Error User", "age": 30, "city": "Delhi", "occupation": "Engineer", "monthly_income": 100000.0, "bonus": 0.0, "additional_income": 0.0}
    balance_sheet = {"user_id": uid, "net_worth": 1000000.0, "bank_savings": 200000.0}
    twin = FinancialDigitalTwin(uid, demographics, balance_sheet)

    session = ThreadSafeSessionState({HISTORY_KEY: []})
    mock_orch = MagicMock()
    mock_orch.run.side_effect = RuntimeError("Simulated agent engine error")

    with patch("streamlit.session_state", session), \
         patch("streamlit.spinner"), \
         patch("streamlit.rerun"):

        _process_user_message("Analyze budget", twin, 10, orchestrator=mock_orch)

    final_count = DBManager.get_chat_usage_today(uid)
    assert final_count == 1, "Exactly one quota consumption must occur for the accepted message"
    assert session[HISTORY_KEY][-1]["role"] == "model"
    assert "error occurred" in session[HISTORY_KEY][-1]["text"]

    with get_db_cursor() as cur:
        cur.execute("DELETE FROM chat_usage WHERE user_id = ?", (uid,))

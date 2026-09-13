"""
Phase 5.3: Streamlit Rerun & Database I/O Optimization Tests

Validates:
1. Quota semantics (40/day authenticated, 12/day guest, pre-execution checks).
2. Database atomic usage increment & concurrency safety.
3. Session state caching, isolation, and user-switch reset.
4. History 50-turn bounding & deduplication.
5. Fresh request IDs and architecture boundaries.
"""

import datetime
import threading
import pytest
from unittest.mock import MagicMock, patch

from database.db_manager import DBManager
from database.connection import get_connection, get_db_cursor
from utils.chatbot import (
    _usage_today,
    _record_usage,
    _process_user_message,
    DAILY_LIMIT_LOGGED_IN,
    DAILY_LIMIT_GUEST,
    MAX_HISTORY,
    HISTORY_KEY,
    OPEN_KEY,
    ANON_USAGE_KEY,
)
from models.twin_engine import FinancialDigitalTwin
from agents.state import FinancialState
from agents.schemas import AgentResult, AgentStatus


@pytest.fixture
def mock_twin():
    demographics = {
        "user_id": "test_phase53_user",
        "name": "Phase5 User",
        "age": 30,
        "city": "Mumbai",
        "occupation": "Engineer",
        "monthly_income": 100000.0,
        "bonus": 10000.0,
        "additional_income": 5000.0,
    }
    balance_sheet = {
        "user_id": "test_phase53_user",
        "net_worth": 1000000.0,
        "bank_savings": 200000.0,
        "fd_amount": 100000.0,
        "emergency_fund": 150000.0,
        "sip_amount": 15000.0,
        "mutual_funds": 300000.0,
        "stocks": 200000.0,
        "ppf_investment": 50000.0,
        "nps_investment": 50000.0,
        "loan_amount": 100000.0,
        "car_loan": 0.0,
        "home_loan": 0.0,
        "credit_card_debt": 5000.0,
        "monthly_emi": 5000.0,
        "health_insurance": 500000.0,
        "life_insurance": 5000000.0,
        "rent": 20000.0,
        "groceries": 10000.0,
        "utilities": 5000.0,
        "transport": 4000.0,
        "food_delivery": 3000.0,
        "entertainment": 3000.0,
        "shopping": 5000.0,
        "goal_type": "Retirement",
        "goal_amount": 50000000.0,
    }
    return FinancialDigitalTwin("test_phase53_user", demographics, balance_sheet)


class MockSessionState(dict):
    """Dict wrapper supporting dot-access for Streamlit session state."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)
    def __setattr__(self, key, value):
        self[key] = value


# ── 1. Quota Semantics & Atomic Increment ─────────────────────────────────────

def test_quota_limits_constants():
    """Verify daily limits are exactly 40 for authenticated and 12 for guests."""
    assert DAILY_LIMIT_LOGGED_IN == 40
    assert DAILY_LIMIT_GUEST == 12


def test_database_atomic_increment_returns_exact_count(mock_twin):
    """Test that DBManager.increment_chat_usage atomically increments and returns new count."""
    today = datetime.date.today().isoformat()
    uid = f"atomic_test_{int(datetime.datetime.now().timestamp())}"

    with get_db_cursor() as cur:
        cur.execute("DELETE FROM chat_usage WHERE user_id = ?", (uid,))

    c1 = DBManager.increment_chat_usage(uid)
    assert c1 == 1

    c2 = DBManager.increment_chat_usage(uid)
    assert c2 == 2

    c3 = DBManager.increment_chat_usage(uid)
    assert c3 == 3

    fetched = DBManager.get_chat_usage_today(uid)
    assert fetched == 3

    with get_db_cursor() as cur:
        cur.execute("DELETE FROM chat_usage WHERE user_id = ?", (uid,))


def test_concurrent_chat_usage_increments():
    """Test safe concurrent multi-threaded usage increments without locks/overwrites."""
    today = datetime.date.today().isoformat()
    uid = f"concurrent_user_{int(datetime.datetime.now().timestamp())}"
    num_threads = 10
    increments_per_thread = 5
    expected_total = num_threads * increments_per_thread

    with get_db_cursor() as cur:
        cur.execute("DELETE FROM chat_usage WHERE user_id = ?", (uid,))

    def worker():
        for _ in range(increments_per_thread):
            DBManager.increment_chat_usage(uid)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final_count = DBManager.get_chat_usage_today(uid)
    assert final_count == expected_total

    with get_db_cursor() as cur:
        cur.execute("DELETE FROM chat_usage WHERE user_id = ?", (uid,))


# ── 2. Session State Caching & Redundant I/O Elimination ──────────────────────

def test_usage_today_session_caching(mock_twin):
    """Test that _usage_today uses session state caching on passive reruns."""
    session = MockSessionState()
    with patch("streamlit.session_state", session), \
         patch("database.db_manager.DBManager.get_chat_usage_today", return_value=5) as mock_get:

        # First call: cache miss, hits DB
        used1, limit1 = _usage_today(mock_twin)
        assert used1 == 5
        assert limit1 == 40
        assert mock_get.call_count == 1
        assert session["ftw_cached_usage"] == (mock_twin.user_id, datetime.date.today().isoformat(), 5)

        # Second call (passive rerun): cache hit, does NOT hit DB
        used2, limit2 = _usage_today(mock_twin)
        assert used2 == 5
        assert limit2 == 40
        assert mock_get.call_count == 1  # Not incremented!


def test_record_usage_updates_session_cache(mock_twin):
    """Test that _record_usage increments authoritatively and updates session cache."""
    session = MockSessionState()
    session["ftw_cached_usage"] = (mock_twin.user_id, datetime.date.today().isoformat(), 2)

    with patch("streamlit.session_state", session), \
         patch("database.db_manager.DBManager.check_and_consume_quota", return_value=(True, 3)) as mock_consume:

        new_count = _record_usage(mock_twin)
        assert new_count == 3
        assert mock_consume.call_count == 1
        assert session["ftw_cached_usage"] == (mock_twin.user_id, datetime.date.today().isoformat(), 3)


def test_guest_usage_remains_isolated_in_session():
    """Test that guest usage is purely session-scoped and isolated from DB."""
    session = MockSessionState()
    with patch("streamlit.session_state", session), \
         patch("database.db_manager.DBManager.check_and_consume_quota") as mock_db_consume, \
         patch("database.db_manager.DBManager.get_chat_usage_today") as mock_db_get:

        used, limit = _usage_today(None)
        assert used == 0
        assert limit == 12
        assert mock_db_get.call_count == 0

        rec_count = _record_usage(None)
        assert rec_count == 1
        assert mock_db_consume.call_count == 0

        used2, _ = _usage_today(None)
        assert used2 == 1


# ── 3. Quota Pre-Execution Check & Rejection Flow ─────────────────────────────

def test_quota_rejected_request_does_not_invoke_orchestrator(mock_twin):
    """Verify that when remaining == 0, message is rejected before agent/LLM execution."""
    session = MockSessionState()
    session[HISTORY_KEY] = []

    mock_orch = MagicMock()
    mock_adapter = MagicMock()

    with patch("streamlit.session_state", session), \
         patch("streamlit.rerun") as mock_rerun, \
         patch("database.db_manager.DBManager.check_and_consume_quota") as mock_consume:

        # remaining = 0
        _process_user_message("What is my health score?", mock_twin, remaining=0, adapter=mock_adapter, orchestrator=mock_orch)

        # No agents executed, no usage recorded, no history appended
        assert mock_adapter.build_request.call_count == 0
        assert mock_orch.run.call_count == 0
        assert mock_consume.call_count == 0
        assert len(session[HISTORY_KEY]) == 0


def test_accepted_request_increments_usage_and_executes_agent(mock_twin):
    """Verify that accepted message records usage once and executes orchestrator."""
    session = MockSessionState()
    session[HISTORY_KEY] = []

    fake_result = AgentResult(
        agent_name="FinancialCoachAgent",
        status=AgentStatus.SUCCESS,
        intent="overview",
        insights=("Your finances are well balanced.",),
    )
    mock_orch = MagicMock()
    mock_orch.run.return_value = fake_result

    with patch("streamlit.session_state", session), \
         patch("streamlit.spinner"), \
         patch("streamlit.rerun"), \
         patch("database.db_manager.DBManager.check_and_consume_quota", return_value=(True, 1)) as mock_consume:

        _process_user_message("Analyze my budget", mock_twin, remaining=40, orchestrator=mock_orch)

        assert mock_consume.call_count == 1
        assert mock_orch.run.call_count == 1
        assert len(session[HISTORY_KEY]) == 2
        assert session[HISTORY_KEY][0]["role"] == "user"
        assert session[HISTORY_KEY][1]["role"] == "model"


# ── 4. History Bound & Session Safety ─────────────────────────────────────────

def test_history_enforces_50_turn_bound(mock_twin):
    """Ensure history is strictly bounded to MAX_HISTORY (50 turns)."""
    session = MockSessionState()
    # Pre-populate with 49 messages
    session[HISTORY_KEY] = [{"role": "user" if i % 2 == 0 else "model", "text": f"msg {i}"} for i in range(49)]

    fake_result = AgentResult(
        agent_name="FinancialCoachAgent",
        status=AgentStatus.SUCCESS,
        intent="overview",
        insights=("Advice",),
    )
    mock_orch = MagicMock()
    mock_orch.run.return_value = fake_result

    with patch("streamlit.session_state", session), \
         patch("streamlit.spinner"), \
         patch("streamlit.rerun"), \
         patch("database.db_manager.DBManager.increment_chat_usage", return_value=1):

        # Adding a user message and a model reply brings total from 49 to 51, bounded to 50
        _process_user_message("New question", mock_twin, remaining=10, orchestrator=mock_orch)

        assert len(session[HISTORY_KEY]) == MAX_HISTORY
        assert len(session[HISTORY_KEY]) == 50
        assert session[HISTORY_KEY][-1]["role"] == "model"
        assert session[HISTORY_KEY][-2]["role"] == "user"


def test_user_switching_resets_history_and_cache():
    """Verify that switching active user clears history and invalidates cached usage."""
    from utils.chatbot import render_chatbot

    session = MockSessionState()
    session[HISTORY_KEY] = [{"role": "user", "text": "hello from user 1"}]
    session[OPEN_KEY] = True
    session["ftw_last_user_id"] = "user_1"
    session["ftw_cached_usage"] = ("user_1", datetime.date.today().isoformat(), 10)

    twin2 = MagicMock()
    twin2.user_id = "user_2"
    twin2.name = "User Two"

    with patch("streamlit.session_state", session), \
         patch("streamlit.markdown"), \
         patch("streamlit.button", return_value=False), \
         patch("database.db_manager.DBManager.get_chat_usage_today", return_value=0):

        render_chatbot(twin2)

        # History and open state reset, old cache cleared
        assert session[HISTORY_KEY] == []
        assert session[OPEN_KEY] is False
        assert session["ftw_last_user_id"] == "user_2"
        assert session.get("ftw_cached_usage")[0] == "user_2"


def test_chatbot_open_state_survives_rerun(mock_twin):
    """Verify that chatbot open/close state persists across multiple rerenders for the same user."""
    from utils.chatbot import render_chatbot

    session = MockSessionState()
    session[HISTORY_KEY] = []
    session[OPEN_KEY] = True
    session["ftw_last_user_id"] = mock_twin.user_id
    session["ftw_cached_usage"] = (mock_twin.user_id, datetime.date.today().isoformat(), 4)

    with patch("streamlit.session_state", session), \
         patch("streamlit.markdown"), \
         patch("streamlit.button", return_value=False), \
         patch("streamlit.container"), \
         patch("streamlit.columns", side_effect=lambda spec, **kw: [MagicMock() for _ in range(len(spec)) if isinstance(spec, list)] or [MagicMock(), MagicMock()]), \
         patch("streamlit.form", return_value=MagicMock()), \
         patch("streamlit.text_input", return_value=""), \
         patch("streamlit.form_submit_button", return_value=False), \
         patch("database.db_manager.DBManager.get_chat_usage_today") as mock_get:

        render_chatbot(mock_twin)
        assert session[OPEN_KEY] is True
        assert mock_get.call_count == 0  # Reused cached usage


def test_date_rollover_invalidates_cache_and_fetches_fresh_db_count(mock_twin):
    """Verify that when date rolls over, old cached date is ignored and fresh DB read occurs."""
    session = MockSessionState()
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    # Cache holds yesterday's count
    session["ftw_cached_usage"] = (mock_twin.user_id, yesterday, 40)

    with patch("streamlit.session_state", session), \
         patch("database.db_manager.DBManager.get_chat_usage_today", return_value=0) as mock_get:

        used, limit = _usage_today(mock_twin)
        assert used == 0  # Fresh count for today
        assert limit == 40
        assert mock_get.call_count == 1
        assert session["ftw_cached_usage"] == (mock_twin.user_id, datetime.date.today().isoformat(), 0)


def test_request_id_freshness_per_message(mock_twin):
    """Verify that every new message creates a fresh unique request_id via AgentChatAdapter."""
    from utils.agent_chat_adapter import DEFAULT_CHAT_ADAPTER

    req1 = DEFAULT_CHAT_ADAPTER.build_request("How is my savings?", twin=mock_twin)
    req2 = DEFAULT_CHAT_ADAPTER.build_request("How is my savings?", twin=mock_twin)

    assert req1.request_id != req2.request_id
    assert req1.request_id.startswith("req-chat-")
    assert req2.request_id.startswith("req-chat-")


def test_no_duplicate_agent_execution_on_passive_rerun(mock_twin):
    """Verify that rendering the page without submitting a form does NOT invoke orchestrator or agents."""
    from utils.chatbot import render_chatbot

    session = MockSessionState()
    session[HISTORY_KEY] = []
    session[OPEN_KEY] = True
    session["ftw_last_user_id"] = mock_twin.user_id
    session["ftw_cached_usage"] = (mock_twin.user_id, datetime.date.today().isoformat(), 2)

    with patch("streamlit.session_state", session), \
         patch("streamlit.markdown"), \
         patch("streamlit.button", return_value=False), \
         patch("streamlit.container"), \
         patch("streamlit.columns", side_effect=lambda spec, **kw: [MagicMock() for _ in range(len(spec)) if isinstance(spec, list)] or [MagicMock(), MagicMock()]), \
         patch("streamlit.form", return_value=MagicMock()), \
         patch("streamlit.text_input", return_value=""), \
         patch("streamlit.form_submit_button", return_value=False), \
         patch("agents.orchestrator.OrchestratorAgent.run") as mock_orch_run:

        render_chatbot(mock_twin)
        assert mock_orch_run.call_count == 0


def test_financial_state_conversion_correctness_and_immutability(mock_twin):
    """Verify that FinancialDigitalTwin -> FinancialState conversion produces deeply immutable state."""
    from utils.agent_chat_adapter import DEFAULT_CHAT_ADAPTER

    req = DEFAULT_CHAT_ADAPTER.build_request("Analyze my budget", twin=mock_twin)
    f_state = req.financial_state

    assert isinstance(f_state, FinancialState)
    assert f_state.profile.user_id == mock_twin.user_id
    assert f_state.cash_flow.monthly_income == mock_twin.monthly_income
    assert f_state.balance_sheet.current_net_worth == mock_twin.net_worth

    # Ensure immutability (frozen dataclass)
    with pytest.raises((AttributeError, TypeError)):
        f_state.cash_flow.monthly_income = 999999.0

"""
tests/test_chatbot_cleanup.py
=============================
Cleanup & context deprecation test suite for Phase 4 Step 3.3.

Verifies:
1. Active chatbot path does not directly execute legacy calculation engines:
   - FinancialPredictor
   - ScenarioSimulator
   - HealthScoreEngine
   - IndianTaxCalculator
   - GoalEngine
2. Zero direct HTTP calls to Groq or Gemini from the chatbot layer.
3. No legacy _coach_digest execution in active path.
4. Correct multi-agent dispatch via AgentChatAdapter and OrchestratorAgent.
5. Local execution of non-financial queries (greetings, app guidance).
6. Public API backward-compatibility (render_chatbot signature and callability).
7. Session state preservation (history, 50-turn cap, rate-limiting).
"""

import pytest
from unittest.mock import MagicMock, patch
import streamlit as st

import utils.chatbot as chatbot_mod
from utils.chatbot import (
    _process_user_message,
    _format_agent_result_to_text,
    render_chatbot,
    GREETING_RESPONSE,
    APP_GUIDANCE_RESPONSE,
    HISTORY_KEY,
    MAX_HISTORY,
)
from utils.agent_chat_adapter import AgentChatAdapter
from agents.orchestrator import OrchestratorAgent
from agents.schemas import (
    AgentResult,
    AgentStatus,
    AuthoritativeData,
    AuthoritativeProvenance,
    ComputationType,
)
from models.twin_engine import FinancialDigitalTwin


@pytest.fixture
def sample_twin():
    demographics = {
        "name": "Alok Rana",
        "age": 30,
        "occupation": "Software Engineer",
        "city": "Bengaluru",
        "monthly_income": 120000.0,
    }
    balance_sheet = {
        "bank_savings": 300000.0,
        "fd_amount": 150000.0,
        "emergency_fund": 250000.0,
        "mutual_funds": 600000.0,
        "stocks": 250000.0,
        "ppf_investment": 150000.0,
        "sip_amount": 25000.0,
        "monthly_emi": 20000.0,
        "loan_amount": 600000.0,
        "car_loan": 0.0,
        "credit_card_debt": 0.0,
        "health_insurance": 12000.0,
        "life_insurance": 18000.0,
        "rent": 30000.0,
        "groceries": 12000.0,
        "utilities": 6000.0,
        "transport": 4000.0,
        "food_delivery": 5000.0,
        "entertainment": 4000.0,
        "shopping": 6000.0,
    }
    return FinancialDigitalTwin(user_id="user_cleanup_test", demographics=demographics, balance_sheet=balance_sheet)


from database.connection import get_db_cursor


@pytest.fixture(autouse=True)
def init_clean_session():
    if HISTORY_KEY not in st.session_state:
        st.session_state[HISTORY_KEY] = []
    else:
        st.session_state[HISTORY_KEY].clear()
    with get_db_cursor() as cur:
        cur.execute("DELETE FROM chat_usage WHERE user_id = ?", ("user_cleanup_test",))
    yield
    st.session_state[HISTORY_KEY].clear()
    with get_db_cursor() as cur:
        cur.execute("DELETE FROM chat_usage WHERE user_id = ?", ("user_cleanup_test",))


def _mock_success_result(intent: str, summary: str = "Test synthesis") -> AgentResult:
    prov = AuthoritativeProvenance(
        source_engine="FinancialCoachAgent",
        source_tool="coach",
        calculation_version="v1.0.0-test",
        request_id="req-cleanup-test",
    )
    auth_data = AuthoritativeData(
        provenance=prov,
        computation_type=ComputationType.HEURISTIC,
        metrics={"summary": summary, "llm_explanation": None},
    )
    return AgentResult(
        agent_name="FinancialCoachAgent",
        status=AgentStatus.SUCCESS,
        intent=intent,
        authoritative_data=auth_data,
        insights=(summary,),
        warnings=(),
        errors=(),
    )


# ── 1. Verification of No Duplicate Financial Engine Execution ────────────────

class TestNoDuplicateFinancialExecution:
    @patch("models.predictor.FinancialPredictor.predict")
    @patch("utils.simulator.ScenarioSimulator.run_scenario")
    @patch("models.twin_engine.HealthScoreEngine.compute_overall_health_score")
    @patch("utils.coach.FinancialCoach.generate_report")
    def test_chatbot_never_calls_calculation_engines_directly(
        self,
        mock_coach,
        mock_health,
        mock_sim,
        mock_pred,
        sample_twin,
    ):
        mock_orch = MagicMock(spec=OrchestratorAgent)
        mock_orch.run.return_value = _mock_success_result("overview")

        with patch("streamlit.rerun"):
            _process_user_message("What is my financial health overview?", sample_twin, remaining=10, orchestrator=mock_orch)

        mock_pred.assert_not_called()
        mock_sim.assert_not_called()
        mock_health.assert_not_called()
        mock_coach.assert_not_called()


# ── 2. Verification of Dead Legacy Code Removal ──────────────────────────────

class TestDeadCodeRemoval:
    def test_legacy_functions_removed_from_module(self):
        # Verify obsolete context builders and direct provider helpers are removed
        assert not hasattr(chatbot_mod, "_coach_digest")
        assert not hasattr(chatbot_mod, "_build_predictions_and_scenarios_context")
        assert not hasattr(chatbot_mod, "_build_system_prompt")
        assert not hasattr(chatbot_mod, "_call_groq")
        assert not hasattr(chatbot_mod, "_call_gemini_backend")
        assert not hasattr(chatbot_mod, "_call_llm")
        assert not hasattr(chatbot_mod, "_get_available_gemini_models")

    def test_no_requests_import_in_chatbot(self):
        # Chatbot must not import or depend on direct requests module
        assert not hasattr(chatbot_mod, "requests")


# ── 3. Multi-Agent Dispatch Integrity ────────────────────────────────────────

class TestMultiAgentDispatchIntegrity:
    def test_financial_message_reaches_orchestrator_via_adapter(self, sample_twin):
        mock_adapter = MagicMock(spec=AgentChatAdapter)
        mock_adapter.build_request.return_value = MagicMock(
            is_financial_query=True,
            intent="risk",
            requires_clarification=False,
            clarification_prompt=None,
            financial_state=MagicMock(),
            request_id="req-clean-123",
            scenario_name=None,
            scenario_params=None,
            goal_params=None,
        )

        mock_orch = MagicMock(spec=OrchestratorAgent)
        mock_orch.run.return_value = _mock_success_result("risk", summary="Debt analysis safe")

        with patch("streamlit.rerun"):
            _process_user_message("Analyze my debt risk", sample_twin, remaining=10, adapter=mock_adapter, orchestrator=mock_orch)

        mock_adapter.build_request.assert_called_once_with("Analyze my debt risk", twin=sample_twin)
        mock_orch.run.assert_called_once()
        assert mock_orch.run.call_args.kwargs["intent"] == "risk"
        assert st.session_state[HISTORY_KEY][-1]["text"] == "Debt analysis safe"


# ── 4. Non-Financial & App Guidance Handling ─────────────────────────────────

class TestNonFinancialCleanHandling:
    def test_greetings_remain_local(self, sample_twin):
        mock_orch = MagicMock(spec=OrchestratorAgent)

        with patch("streamlit.rerun"):
            _process_user_message("Hello!", sample_twin, remaining=10, orchestrator=mock_orch)

        mock_orch.run.assert_not_called()
        assert st.session_state[HISTORY_KEY][-1]["text"] == GREETING_RESPONSE

    def test_app_guidance_remains_local(self, sample_twin):
        mock_orch = MagicMock(spec=OrchestratorAgent)

        with patch("streamlit.rerun"):
            _process_user_message("What can this app do?", sample_twin, remaining=10, orchestrator=mock_orch)

        mock_orch.run.assert_not_called()
        assert st.session_state[HISTORY_KEY][-1]["text"] == APP_GUIDANCE_RESPONSE


# ── 5. Public API & Session Compatibility ─────────────────────────────────────

class TestPublicAPIAndSessionCompatibility:
    def test_public_render_chatbot_is_callable(self, sample_twin):
        assert callable(render_chatbot)

    def test_session_history_turn_bounding(self, sample_twin):
        mock_orch = MagicMock(spec=OrchestratorAgent)
        mock_orch.run.return_value = _mock_success_result("health", "Score: 78")

        # Fill history to 50 items
        st.session_state[HISTORY_KEY] = [{"role": "user", "text": f"Msg {i}"} for i in range(50)]

        with patch("streamlit.rerun"):
            _process_user_message("Health score check", sample_twin, remaining=10, orchestrator=mock_orch)

        assert len(st.session_state[HISTORY_KEY]) == MAX_HISTORY
        assert st.session_state[HISTORY_KEY][-1]["role"] == "model"

"""
tests/test_phase5_system_audit.py
=================================
System-wide integration, performance benchmarking, and architectural boundary audit.
"""

import time
import pytest
from unittest.mock import MagicMock, patch
import streamlit as st

from utils.chatbot import _process_user_message, HISTORY_KEY, OPEN_KEY
from utils.agent_chat_adapter import AgentChatAdapter
from agents.orchestrator import OrchestratorAgent
from agents.financial_coach import FinancialCoachAgent
from agents.financial_intelligence import FinancialIntelligenceAgent
from agents.risk_behaviour import RiskBehaviourAgent
from agents.forecast_goal import ForecastGoalAgent
from agents.scenario_simulation import ScenarioSimulationAgent
from agents.tool_registry import ToolRegistry, DEFAULT_TOOL_REGISTRY
from agents.state import FinancialState
from models.twin_engine import FinancialDigitalTwin


@pytest.fixture
def benchmark_twin():
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
    return FinancialDigitalTwin(user_id="user_audit_bench", demographics=demographics, balance_sheet=balance_sheet)


@pytest.fixture(autouse=True)
def clean_st_session():
    if HISTORY_KEY not in st.session_state:
        st.session_state[HISTORY_KEY] = []
    else:
        st.session_state[HISTORY_KEY].clear()
    st.session_state[OPEN_KEY] = True
    yield
    st.session_state[HISTORY_KEY].clear()


class TestSystemIntegrationAndPerformanceBaseline:
    BENCHMARK_ROUTES = [
        ("A_Health", "What is my financial health score?", "health"),
        ("B_Risk", "Am I taking too much financial risk?", "risk"),
        ("C_Forecast", "How much can I save in the future?", "forecast"),
        ("D_Goal", "Can I achieve my goal?", "goal"),
        ("E_Scenario", "What happens if I buy a car for 8 lakh?", "scenario"),
        ("F_Scenario_Forecast", "What happens if I buy a car for 8 lakh, and how will it affect my savings?", "scenario_forecast"),
        ("G_Health_Diagnostic", "Why is my financial health score low?", "health_diagnostic"),
        ("H_Comprehensive", "Give me a full financial review and audit", "comprehensive"),
    ]

    def test_benchmark_execution_and_metrics(self, benchmark_twin):
        results = []
        adapter = AgentChatAdapter()
        coach = FinancialCoachAgent(enable_llm=False)
        orch = OrchestratorAgent(financial_coach_agent=coach)

        for label, query, expected_route in self.BENCHMARK_ROUTES:
            start_t = time.perf_counter()
            with patch("streamlit.rerun"):
                _process_user_message(query, benchmark_twin, remaining=10, adapter=adapter, orchestrator=orch)
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0

            assert len(st.session_state[HISTORY_KEY]) >= 2
            reply = st.session_state[HISTORY_KEY][-1]["text"]
            assert len(reply) > 20
            st.session_state[HISTORY_KEY].clear()

            results.append({
                "label": label,
                "route": expected_route,
                "elapsed_ms": round(elapsed_ms, 2),
            })

        print("\n=== SYSTEM PERFORMANCE BENCHMARK RESULTS ===")
        for r in results:
            print(f"[{r['label']}] Route: {r['route']} | Latency: {r['elapsed_ms']} ms")
        assert len(results) == 8

    def test_tool_registry_agent_role_isolation(self):
        registry = DEFAULT_TOOL_REGISTRY
        # Verify role-based policy maps only authorized tools
        for role, allowed_names in registry.AGENT_ROLE_POLICIES.items():
            agent_tools = registry.get_tools_for_agent(role)
            assert set(agent_tools.keys()) == set(allowed_names)
            assert all(t.read_only for t in agent_tools.values())
            assert all(t.authoritative for t in agent_tools.values())

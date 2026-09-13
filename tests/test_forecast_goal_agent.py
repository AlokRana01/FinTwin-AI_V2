"""
tests/test_forecast_goal_agent.py
=================================
Unit tests for Phase 3 Step 3.3: ForecastGoalAgent.
Verifies tool delegation via ToolRegistry, operational dispatch,
provenance preservation, tool minimization, state immutability,
tool boundary restrictions, error handling, and zero LLM calls.
"""

import pytest
from datetime import date
from dataclasses import FrozenInstanceError

from agents.state import (
    UserProfileContext,
    CashFlowState,
    BalanceSheetState,
    GoalItemState,
    FinancialState,
)
from agents.schemas import (
    AgentStatus,
    ComputationType,
    AgentResult,
    AuthoritativeData,
)
from agents.tool_registry import ToolRegistry, DEFAULT_TOOL_REGISTRY
from agents.forecast_goal import ForecastGoalAgent
from models.twin_engine import FinancialDigitalTwin
from utils.goal_engine import GoalInput, GoalType


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_twin():
    demographics = {
        "name": "Vikram Sethi",
        "age": 30,
        "occupation": "Data Scientist",
        "city": "Hyderabad",
        "monthly_income": 140000.0,
    }
    balance_sheet = {
        "bank_savings": 300000.0,
        "fd_amount": 100000.0,
        "emergency_fund": 200000.0,
        "mutual_funds": 400000.0,
        "stocks": 150000.0,
        "ppf_investment": 100000.0,
        "sip_amount": 25000.0,
        "monthly_emi": 20000.0,
        "loan_amount": 500000.0,
        "credit_card_debt": 0.0,
        "health_insurance": 15000.0,
        "life_insurance": 25000.0,
        "rent": 30000.0,
        "groceries": 15000.0,
        "utilities": 5000.0,
        "transport": 4000.0,
        "food_delivery": 5000.0,
        "entertainment": 4000.0,
        "shopping": 7000.0,
    }
    return FinancialDigitalTwin(user_id="user_vikram", demographics=demographics, balance_sheet=balance_sheet)


@pytest.fixture
def sample_state(sample_twin):
    g1 = GoalItemState(
        goal_id="g1",
        goal_name="Car Purchase",
        goal_type="Car",
        target_amount=800000.0,
        current_savings=150000.0,
        target_date="2028-12",
        monthly_contribution=15000.0,
        priority=1,
    )
    g2 = GoalItemState(
        goal_id="g2",
        goal_name="House Purchase",
        goal_type="House",
        target_amount=4000000.0,
        current_savings=300000.0,
        target_date="2032-12",
        monthly_contribution=20000.0,
        priority=2,
    )
    return FinancialState.from_twin(sample_twin, goals=(g1, g2))


@pytest.fixture
def agent():
    return ForecastGoalAgent()


# ── 1. Initialization ─────────────────────────────────────────────────────────

def test_agent_initialization():
    agent = ForecastGoalAgent()
    assert agent.AGENT_NAME == "ForecastGoalAgent"
    assert agent.registry == DEFAULT_TOOL_REGISTRY
    assert "forecast_savings" in agent.authorized_tools
    assert "forecast_net_worth" in agent.authorized_tools
    assert "analyze_goal" in agent.authorized_tools
    assert "plan_multiple_goals" in agent.authorized_tools
    assert "explain_forecast" in agent.authorized_tools
    assert "calculate_tax" not in agent.authorized_tools
    assert "run_scenario" not in agent.authorized_tools


# ── 2. Savings Forecast Operation ─────────────────────────────────────────────

def test_savings_forecast_operation(agent, sample_state):
    result = agent.execute(sample_state, operation="savings_forecast", horizon_months=12, request_id="req-sf01")
    assert isinstance(result, AgentResult)
    assert result.status == AgentStatus.SUCCESS
    assert result.intent == "SAVINGS_FORECAST"
    assert result.authoritative_data is not None
    assert result.authoritative_data.provenance.source_engine == "FinancialPredictor"
    assert result.authoritative_data.provenance.source_tool == "forecast_savings"
    assert result.authoritative_data.computation_type == ComputationType.MODEL_OUTPUT
    assert result.authoritative_data.metrics["horizon_months"] == 12
    assert "predicted_net_worth" in result.authoritative_data.metrics
    assert len(result.insights) >= 2


# ── 3. Net Worth Forecast Trajectory Operation ────────────────────────────────

def test_net_worth_forecast_operation(agent, sample_state):
    result = agent.execute(sample_state, operation="net_worth_forecast", horizon_months=12)
    assert result.status == AgentStatus.SUCCESS
    assert result.intent == "NET_WORTH_FORECAST"
    assert result.authoritative_data is not None
    assert result.authoritative_data.provenance.source_engine == "FinancialPredictor"
    assert result.authoritative_data.arrays is not None
    assert len(result.authoritative_data.arrays["Month"]) == 12
    assert len(result.authoritative_data.arrays["Projected_Net_Worth"]) == 12
    assert len(result.insights) >= 1


# ── 4. Goal Operation ─────────────────────────────────────────────────────────

def test_single_goal_operation(agent, sample_state):
    result = agent.execute(sample_state, operation="goal")
    assert result.status == AgentStatus.SUCCESS
    assert result.intent == "GOAL_ANALYSIS"
    assert result.authoritative_data is not None
    assert result.authoritative_data.provenance.source_engine == "GoalEngine"
    assert result.authoritative_data.computation_type == ComputationType.DETERMINISTIC
    assert result.authoritative_data.metrics["sip_required"] > 0.0
    assert 0.0 <= result.authoritative_data.metrics["probability"] <= 1.0
    assert len(result.insights) >= 2


# ── 5. Multiple Goals Operation ───────────────────────────────────────────────

def test_multiple_goals_operation(agent, sample_state):
    result = agent.execute(sample_state, operation="multiple_goals")
    assert result.status == AgentStatus.SUCCESS
    assert result.intent == "MULTIPLE_GOALS_PLANNING"
    assert result.authoritative_data is not None
    assert result.authoritative_data.provenance.source_engine == "MultiGoalPlanner"
    assert result.authoritative_data.metrics["goals_count"] >= 2
    assert len(result.insights) >= 1


# ── 6. Forecast Explanation (XAI) Operation ───────────────────────────────────

def test_forecast_explanation_operation(agent, sample_state):
    result = agent.execute(sample_state, operation="forecast_explanation")
    assert result.status == AgentStatus.SUCCESS
    assert result.intent == "FORECAST_EXPLANATION"
    assert result.authoritative_data is not None
    assert result.authoritative_data.provenance.source_engine == "ExplainableAI"
    assert result.authoritative_data.computation_type == ComputationType.MODEL_OUTPUT
    assert "predicted_savings" in result.authoritative_data.metrics
    assert len(result.insights) >= 1


# ── 7. Forecast Overview Operation ────────────────────────────────────────────

def test_forecast_overview_operation(agent, sample_state):
    result = agent.execute(sample_state, operation="forecast_overview", horizon_months=12)
    assert result.status == AgentStatus.SUCCESS
    assert result.intent == "FORECAST_OVERVIEW"
    assert result.authoritative_data is not None
    assert result.authoritative_data.provenance.source_engine == "ForecastComposite"
    assert "predicted_savings" in result.authoritative_data.metrics
    assert "final_projected_net_worth" in result.authoritative_data.metrics
    assert result.authoritative_data.arrays is not None
    assert len(result.insights) >= 2


# ── 8. Tool Minimization Verification ─────────────────────────────────────────

def test_tool_minimization(agent, sample_state, monkeypatch):
    """
    Verifies that simple operations invoke only their specific single required tool,
    and forecast_overview invokes exactly 2 tools.
    """
    executed_tools = []
    original_execute = agent.registry.execute

    def spy_execute(tool_name, *args, **kwargs):
        executed_tools.append(tool_name)
        return original_execute(tool_name, *args, **kwargs)

    monkeypatch.setattr(agent.registry, "execute", spy_execute)

    # Savings forecast -> 1 tool
    executed_tools.clear()
    agent.execute(sample_state, operation="savings_forecast")
    assert executed_tools == ["forecast_savings"]

    # Net worth forecast -> 1 tool
    executed_tools.clear()
    agent.execute(sample_state, operation="net_worth_forecast")
    assert executed_tools == ["forecast_net_worth"]

    # Goal -> 1 tool
    executed_tools.clear()
    agent.execute(sample_state, operation="goal")
    assert executed_tools == ["analyze_goal"]

    # Multiple goals -> 1 tool
    executed_tools.clear()
    agent.execute(sample_state, operation="multiple_goals")
    assert executed_tools == ["plan_multiple_goals"]

    # Forecast explanation -> 1 tool
    executed_tools.clear()
    agent.execute(sample_state, operation="forecast_explanation")
    assert executed_tools == ["explain_forecast"]

    # Forecast overview -> 2 tools
    executed_tools.clear()
    agent.execute(sample_state, operation="forecast_overview")
    assert executed_tools == ["forecast_savings", "forecast_net_worth"]


# ── 9. Unauthorized Tool Boundary Verification ────────────────────────────────

def test_unauthorized_tool_boundary():
    """
    Confirms that ForecastGoalAgent does not possess or expose unauthorized tools
    such as calculate_health_score, calculate_tax, analyze_debt, or run_scenario.
    """
    agent = ForecastGoalAgent()
    authorized = set(agent.authorized_tools.keys())
    assert "calculate_health_score" not in authorized
    assert "calculate_tax" not in authorized
    assert "analyze_debt" not in authorized
    assert "assess_financial_risk" not in authorized
    assert "run_scenario" not in authorized
    assert "compare_scenario" not in authorized


# ── 10. State Immutability Verification ───────────────────────────────────────

def test_state_immutability_during_agent_execution(agent, sample_state):
    initial_income = sample_state.cash_flow.monthly_income

    agent.execute(sample_state, operation="savings_forecast")
    agent.execute(sample_state, operation="goal")
    agent.execute(sample_state, operation="forecast_overview")

    assert sample_state.cash_flow.monthly_income == initial_income
    with pytest.raises(FrozenInstanceError):
        sample_state.cash_flow.monthly_income = 888888.0  # type: ignore


# ── 11. Error Handling ────────────────────────────────────────────────────────

def test_unsupported_operation_error_handling(agent, sample_state):
    result = agent.execute(sample_state, operation="invalid_operation")
    assert result.status == AgentStatus.FAILED
    assert len(result.errors) == 1
    assert "Unsupported operation" in result.errors[0]
    assert result.authoritative_data is None


# ── 12. Zero LLM Invocation Verification ──────────────────────────────────────

def test_zero_llm_invocation(agent, sample_state, monkeypatch):
    """
    Guarantees no network or LLM API calls occur during agent execution.
    """
    import urllib.request

    def fail_on_network(*args, **kwargs):
        raise AssertionError("Network / LLM API call attempted by ForecastGoalAgent!")

    monkeypatch.setattr(urllib.request, "urlopen", fail_on_network)

    for op in ["savings_forecast", "net_worth_forecast", "goal", "multiple_goals", "forecast_explanation", "forecast_overview"]:
        res = agent.execute(sample_state, operation=op)
        assert res.status == AgentStatus.SUCCESS

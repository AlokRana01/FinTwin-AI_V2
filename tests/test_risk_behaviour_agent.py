"""
tests/test_risk_behaviour_agent.py
==================================
Unit tests for Phase 3 Step 3.2: RiskBehaviourAgent.
Verifies tool delegation via ToolRegistry, operational dispatch,
provenance preservation, tool minimization, state immutability,
tool boundary restrictions, error handling, and zero LLM calls.
"""

import pytest
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
from agents.risk_behaviour import RiskBehaviourAgent
from models.twin_engine import FinancialDigitalTwin


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_twin():
    demographics = {
        "name": "Karan Malhotra",
        "age": 34,
        "occupation": "Banker",
        "city": "Delhi",
        "monthly_income": 120000.0,
    }
    balance_sheet = {
        "bank_savings": 150000.0,
        "fd_amount": 50000.0,
        "emergency_fund": 100000.0,
        "mutual_funds": 200000.0,
        "stocks": 100000.0,
        "ppf_investment": 50000.0,
        "sip_amount": 10000.0,
        "monthly_emi": 55000.0,
        "loan_amount": 1800000.0,
        "car_loan": 400000.0,
        "home_loan": 1400000.0,
        "credit_card_debt": 45000.0,
        "health_insurance": 10000.0,
        "life_insurance": 15000.0,
        "rent": 20000.0,
        "groceries": 12000.0,
        "utilities": 4000.0,
        "transport": 4000.0,
        "food_delivery": 8000.0,
        "entertainment": 6000.0,
        "shopping": 10000.0,
    }
    return FinancialDigitalTwin(user_id="user_karan", demographics=demographics, balance_sheet=balance_sheet)


@pytest.fixture
def sample_state(sample_twin):
    return FinancialState.from_twin(sample_twin)


@pytest.fixture
def agent():
    return RiskBehaviourAgent()


# ── 1. Initialization ─────────────────────────────────────────────────────────

def test_agent_initialization():
    agent = RiskBehaviourAgent()
    assert agent.AGENT_NAME == "RiskBehaviourAgent"
    assert agent.registry == DEFAULT_TOOL_REGISTRY
    assert "analyze_debt" in agent.authorized_tools
    assert "analyze_emergency_fund" in agent.authorized_tools
    assert "analyze_spending_behavior" in agent.authorized_tools
    assert "assess_financial_risk" in agent.authorized_tools
    assert "calculate_tax" not in agent.authorized_tools


# ── 2. Debt Operation ─────────────────────────────────────────────────────────

def test_debt_operation_delegation(agent, sample_state):
    result = agent.execute(sample_state, operation="debt", request_id="req-d01")
    assert isinstance(result, AgentResult)
    assert result.status == AgentStatus.SUCCESS
    assert result.intent == "DEBT_ANALYSIS"
    assert result.authoritative_data is not None
    assert result.authoritative_data.provenance.source_engine == "FinancialDigitalTwin"
    assert result.authoritative_data.provenance.source_tool == "analyze_debt"
    assert result.authoritative_data.metrics["total_loan_amount"] == 1800000.0
    assert result.authoritative_data.metrics["monthly_emi"] == 55000.0
    assert result.authoritative_data.metrics["credit_card_debt"] == 45000.0
    assert len(result.insights) >= 2
    assert len(result.warnings) >= 1  # High EMI ratio + credit card debt flagged


# ── 3. Emergency Fund Operation ───────────────────────────────────────────────

def test_emergency_fund_operation_delegation(agent, sample_state):
    result = agent.execute(sample_state, operation="emergency_fund")
    assert result.status == AgentStatus.SUCCESS
    assert result.intent == "EMERGENCY_FUND_ANALYSIS"
    assert result.authoritative_data is not None
    assert result.authoritative_data.provenance.source_engine == "HealthScoreEngine"
    assert result.authoritative_data.metrics["emergency_fund_amount"] == 100000.0
    assert result.authoritative_data.metrics["months_covered"] > 0.0
    assert len(result.insights) >= 2


# ── 4. Spending Behavior Operation ────────────────────────────────────────────

def test_spending_behavior_operation_delegation(agent, sample_state):
    result = agent.execute(sample_state, operation="spending_behavior")
    assert result.status == AgentStatus.SUCCESS
    assert result.intent == "SPENDING_BEHAVIOR_ANALYSIS"
    assert result.authoritative_data is not None
    assert result.authoritative_data.provenance.source_engine == "FinancialCoach"
    assert result.authoritative_data.computation_type == ComputationType.HEURISTIC
    assert "active_alerts_count" in result.authoritative_data.metrics


# ── 5. Risk Assessment Operation ──────────────────────────────────────────────

def test_risk_operation_delegation(agent, sample_state):
    result = agent.execute(sample_state, operation="risk")
    assert result.status == AgentStatus.SUCCESS
    assert result.intent == "RISK_ASSESSMENT"
    assert result.authoritative_data is not None
    assert result.authoritative_data.provenance.source_engine == "FinancialDigitalTwin"
    assert result.authoritative_data.metrics["risk_level"] in ["Low", "Moderate", "High", "Critical"]
    assert len(result.insights) >= 2


# ── 6. Risk Overview Operation ────────────────────────────────────────────────

def test_risk_overview_operation(agent, sample_state):
    result = agent.execute(sample_state, operation="risk_overview")
    assert result.status == AgentStatus.SUCCESS
    assert result.intent == "RISK_OVERVIEW"
    assert result.authoritative_data is not None
    metrics = result.authoritative_data.metrics
    assert "risk_level" in metrics
    assert "total_loan_amount" in metrics
    assert "emergency_fund_amount" in metrics
    assert "active_alerts_count" in metrics
    assert len(result.insights) >= 3


# ── 7. Tool Minimization Verification ─────────────────────────────────────────

def test_tool_minimization(agent, sample_state, monkeypatch):
    """
    Verifies that simple operations invoke only the single required tool,
    and risk_overview invokes exactly all 4 authorized tools.
    """
    executed_tools = []
    original_execute = agent.registry.execute

    def spy_execute(tool_name, *args, **kwargs):
        executed_tools.append(tool_name)
        return original_execute(tool_name, *args, **kwargs)

    monkeypatch.setattr(agent.registry, "execute", spy_execute)

    # Debt -> 1 tool
    executed_tools.clear()
    agent.execute(sample_state, operation="debt")
    assert executed_tools == ["analyze_debt"]

    # Emergency fund -> 1 tool
    executed_tools.clear()
    agent.execute(sample_state, operation="emergency_fund")
    assert executed_tools == ["analyze_emergency_fund"]

    # Spending behavior -> 1 tool
    executed_tools.clear()
    agent.execute(sample_state, operation="spending_behavior")
    assert executed_tools == ["analyze_spending_behavior"]

    # Risk -> 1 tool
    executed_tools.clear()
    agent.execute(sample_state, operation="risk")
    assert executed_tools == ["assess_financial_risk"]

    # Risk overview -> 4 tools
    executed_tools.clear()
    agent.execute(sample_state, operation="risk_overview")
    assert executed_tools == [
        "analyze_debt",
        "analyze_emergency_fund",
        "analyze_spending_behavior",
        "assess_financial_risk",
    ]


# ── 8. State Immutability Verification ────────────────────────────────────────

def test_state_immutability_during_agent_execution(agent, sample_state):
    initial_income = sample_state.cash_flow.monthly_income

    agent.execute(sample_state, operation="debt")
    agent.execute(sample_state, operation="emergency_fund")
    agent.execute(sample_state, operation="risk_overview")

    assert sample_state.cash_flow.monthly_income == initial_income
    with pytest.raises(FrozenInstanceError):
        sample_state.cash_flow.monthly_income = 999999.0  # type: ignore


# ── 9. Error Handling ─────────────────────────────────────────────────────────

def test_unsupported_operation_error_handling(agent, sample_state):
    result = agent.execute(sample_state, operation="unknown_op")
    assert result.status == AgentStatus.FAILED
    assert len(result.errors) == 1
    assert "Unsupported operation" in result.errors[0]
    assert result.authoritative_data is None


# ── 10. Tool Boundary Policy Verification ─────────────────────────────────────

def test_tool_boundary_policy():
    """
    Confirms that RiskBehaviourAgent does not possess or expose unauthorized tools
    such as calculate_tax, forecast_savings, or run_scenario.
    """
    agent = RiskBehaviourAgent()
    authorized = set(agent.authorized_tools.keys())
    assert "calculate_tax" not in authorized
    assert "forecast_savings" not in authorized
    assert "run_scenario" not in authorized
    assert "classify_financial_personality" not in authorized


# ── 11. Zero LLM Invocation Verification ──────────────────────────────────────

def test_zero_llm_invocation(agent, sample_state, monkeypatch):
    """
    Guarantees no network or LLM API calls occur during agent execution.
    """
    import urllib.request

    def fail_on_network(*args, **kwargs):
        raise AssertionError("Network / LLM API call attempted by RiskBehaviourAgent!")

    monkeypatch.setattr(urllib.request, "urlopen", fail_on_network)

    for op in ["debt", "emergency_fund", "spending_behavior", "risk", "risk_overview"]:
        res = agent.execute(sample_state, operation=op)
        assert res.status == AgentStatus.SUCCESS

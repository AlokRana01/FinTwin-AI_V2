"""
tests/test_financial_intelligence_agent.py
==========================================
Unit tests for Phase 3 Step 3.1: Financial Intelligence Agent.
Verifies tool delegation via ToolRegistry, operational dispatch,
provenance preservation, tool minimization, state immutability,
tax encapsulation, XAI integration, error handling, and zero LLM calls.
"""

import pytest
from dataclasses import FrozenInstanceError

from agents.state import (
    UserProfileContext,
    CashFlowState,
    BalanceSheetState,
    GoalItemState,
    FinancialState,
    SessionContext,
)
from agents.schemas import (
    AgentStatus,
    ComputationType,
    AgentResult,
    AuthoritativeData,
)
from agents.tool_registry import ToolRegistry, DEFAULT_TOOL_REGISTRY
from agents.financial_intelligence import FinancialIntelligenceAgent
from models.twin_engine import FinancialDigitalTwin
from utils.tax_calculator import DeductionProfile


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_twin():
    demographics = {
        "name": "Ananya Roy",
        "age": 31,
        "occupation": "Product Manager",
        "city": "Bengaluru",
        "monthly_income": 150000.0,
    }
    balance_sheet = {
        "bank_savings": 400000.0,
        "fd_amount": 200000.0,
        "emergency_fund": 300000.0,
        "mutual_funds": 500000.0,
        "stocks": 200000.0,
        "ppf_investment": 100000.0,
        "sip_amount": 30000.0,
        "monthly_emi": 25000.0,
        "loan_amount": 600000.0,
        "credit_card_debt": 0.0,
        "health_insurance": 25000.0,
        "life_insurance": 30000.0,
        "rent": 35000.0,
        "groceries": 15000.0,
        "utilities": 6000.0,
        "transport": 5000.0,
        "food_delivery": 6000.0,
        "entertainment": 5000.0,
        "shopping": 8000.0,
    }
    return FinancialDigitalTwin(user_id="user_ananya", demographics=demographics, balance_sheet=balance_sheet)


@pytest.fixture
def sample_state(sample_twin):
    return FinancialState.from_twin(sample_twin)


@pytest.fixture
def agent():
    return FinancialIntelligenceAgent()


# ── 1. Initialization ─────────────────────────────────────────────────────────

def test_agent_initialization():
    agent = FinancialIntelligenceAgent()
    assert agent.AGENT_NAME == "FinancialIntelligenceAgent"
    assert agent.registry == DEFAULT_TOOL_REGISTRY
    assert "calculate_health_score" in agent.authorized_tools
    assert "calculate_tax" in agent.authorized_tools


# ── 2. Health Operation ───────────────────────────────────────────────────────

def test_health_operation_delegation(agent, sample_state):
    result = agent.execute(sample_state, operation="health", request_id="req-h01")
    assert isinstance(result, AgentResult)
    assert result.status == AgentStatus.SUCCESS
    assert result.intent == "HEALTH_ANALYSIS"
    assert result.authoritative_data is not None
    assert result.authoritative_data.provenance.source_engine == "HealthScoreEngine"
    assert result.authoritative_data.provenance.source_tool == "calculate_health_score"
    assert 0 <= result.authoritative_data.metrics["health_score"] <= 100
    assert result.authoritative_data.metrics["grade"] in ["A", "B", "C", "D"]
    assert len(result.insights) >= 2


# ── 3. Ratios Operation ───────────────────────────────────────────────────────

def test_ratios_operation_delegation(agent, sample_state):
    result = agent.execute(sample_state, operation="ratios")
    assert result.status == AgentStatus.SUCCESS
    assert result.intent == "RATIOS_ANALYSIS"
    assert result.authoritative_data is not None
    assert result.authoritative_data.provenance.source_engine == "FinancialDigitalTwin"
    assert result.authoritative_data.metrics["savings_rate"] > 0.0
    assert result.authoritative_data.metrics["emergency_fund_months"] > 0.0
    assert len(result.insights) >= 3


# ── 4. Personality Operation ──────────────────────────────────────────────────

def test_personality_operation_delegation(agent, sample_state):
    result = agent.execute(sample_state, operation="personality")
    assert result.status == AgentStatus.SUCCESS
    assert result.intent == "PERSONALITY_CLASSIFICATION"
    assert result.authoritative_data is not None
    assert result.authoritative_data.provenance.source_engine == "FinancialPersonalityClusterer"
    assert result.authoritative_data.computation_type == ComputationType.MODEL_OUTPUT
    assert result.authoritative_data.metrics["personality"] in [
        "Saver", "Investor", "Spender", "Debt Heavy", "Balanced Planner"
    ]
    assert len(result.insights) >= 2


# ── 5. Tax Operation (Encapsulated in Financial Intelligence) ──────────────────

def test_tax_operation_delegation(agent, sample_state):
    ded = DeductionProfile(investment_80c=150000.0, health_insurance_self=25000.0)
    result = agent.execute(sample_state, operation="tax", deductions=ded)
    assert result.status == AgentStatus.SUCCESS
    assert result.intent == "TAX_ANALYSIS"
    assert result.authoritative_data is not None
    assert result.authoritative_data.provenance.source_engine == "IndianTaxCalculator"
    assert result.authoritative_data.provenance.source_tool == "calculate_tax"
    assert "recommended_regime" in result.authoritative_data.metrics
    assert result.authoritative_data.metrics["gross_income"] == 150000.0 * 12.0
    assert len(result.insights) >= 2


# ── 6. Health Explanation (XAI Integration) ───────────────────────────────────

def test_health_explanation_operation(agent, sample_state):
    result = agent.execute(sample_state, operation="health_explanation")
    assert result.status == AgentStatus.SUCCESS
    assert result.intent == "HEALTH_EXPLANATION"
    assert result.authoritative_data is not None
    assert result.authoritative_data.xai_attributions is not None
    assert len(result.authoritative_data.xai_attributions) == 6
    assert "baseline_score" in result.authoritative_data.metrics
    assert len(result.insights) >= 1


# ── 7. Financial Overview Operation ───────────────────────────────────────────

def test_financial_overview_operation(agent, sample_state):
    result = agent.execute(sample_state, operation="financial_overview")
    assert result.status == AgentStatus.SUCCESS
    assert result.intent == "FINANCIAL_OVERVIEW"
    assert result.authoritative_data is not None
    metrics = result.authoritative_data.metrics
    assert "health_score" in metrics
    assert "savings_rate" in metrics
    assert "personality" in metrics
    assert "emergency_fund_months" in metrics


# ── 8. Tool Minimization Verification ─────────────────────────────────────────

def test_tool_minimization(agent, sample_state, monkeypatch):
    """
    Verifies that calling 'personality' executes ONLY classify_financial_personality
    and does NOT call health, tax, or XAI tools.
    """
    executed_tools = []
    original_execute = agent.registry.execute

    def spy_execute(tool_name, *args, **kwargs):
        executed_tools.append(tool_name)
        return original_execute(tool_name, *args, **kwargs)

    monkeypatch.setattr(agent.registry, "execute", spy_execute)

    agent.execute(sample_state, operation="personality")
    assert executed_tools == ["classify_financial_personality"]

    executed_tools.clear()
    agent.execute(sample_state, operation="tax")
    assert executed_tools == ["calculate_tax"]


# ── 9. State Immutability Verification ────────────────────────────────────────

def test_state_immutability_during_agent_execution(agent, sample_state):
    initial_income = sample_state.cash_flow.monthly_income

    agent.execute(sample_state, operation="financial_overview")
    agent.execute(sample_state, operation="tax")
    agent.execute(sample_state, operation="health_explanation")

    assert sample_state.cash_flow.monthly_income == initial_income
    with pytest.raises(FrozenInstanceError):
        sample_state.cash_flow.monthly_income = 500000.0  # type: ignore


# ── 10. Error Handling ────────────────────────────────────────────────────────

def test_unsupported_operation_error_handling(agent, sample_state):
    result = agent.execute(sample_state, operation="non_existent_op")
    assert result.status == AgentStatus.FAILED
    assert len(result.errors) == 1
    assert "Unsupported operation" in result.errors[0]
    assert result.authoritative_data is None


# ── 11. Zero LLM Invocation Verification ──────────────────────────────────────

def test_zero_llm_invocation(agent, sample_state, monkeypatch):
    """
    Guarantees no network or LLM API calls occur during agent execution.
    """
    import urllib.request

    def fail_on_network(*args, **kwargs):
        raise AssertionError("Network / LLM API call attempted by FinancialIntelligenceAgent!")

    monkeypatch.setattr(urllib.request, "urlopen", fail_on_network)

    for op in ["health", "ratios", "personality", "tax", "health_explanation", "financial_overview"]:
        res = agent.execute(sample_state, operation=op)
        assert res.status == AgentStatus.SUCCESS

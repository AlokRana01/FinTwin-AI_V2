"""
tests/test_scenario_simulation_agent.py
======================================
Comprehensive test suite for ScenarioSimulationAgent verifying:
1. Agent initialization and dependency wiring.
2. Correct delegation to Authoritative Tool Registry for scenario simulation and comparison.
3. Tool minimization (exactly 1 tool executed per single operation).
4. Unauthorized tool boundary enforcement (cannot access health, tax, risk, forecast, or goal tools).
5. State immutability (FinancialState remains read-only and frozen).
6. Preserved computational provenance and computation type.
7. Health score deduplication regression test (no direct call to HealthScoreEngine or calculate_health_score).
8. No duplicated scenario calculations (agent consumes ToolRegistry output).
9. Error handling for unsupported operations and tool failures.
10. Zero-LLM and zero-network execution guarantee.
"""

import pytest
from unittest.mock import MagicMock, patch

from agents.state import (
    FinancialState,
    UserProfileContext,
    CashFlowState,
    BalanceSheetState,
    GoalItemState,
    DerivedIntelligenceState,
)
from agents.schemas import (
    AgentResult,
    AgentStatus,
    AuthoritativeData,
    AuthoritativeProvenance,
    ComputationType,
)
from agents.tool_registry import ToolRegistry, DEFAULT_TOOL_REGISTRY
from agents.scenario_simulation import ScenarioSimulationAgent
from models.twin_engine import FinancialDigitalTwin


@pytest.fixture
def sample_twin():
    demographics = {
        "name": "Priya Sharma",
        "age": 30,
        "occupation": "Software Engineer",
        "city": "Bengaluru",
        "monthly_income": 120000.0,
    }
    balance_sheet = {
        "bank_savings": 250000.0,
        "fd_amount": 150000.0,
        "emergency_fund": 200000.0,
        "mutual_funds": 450000.0,
        "stocks": 250000.0,
        "ppf_investment": 100000.0,
        "sip_amount": 20000.0,
        "monthly_emi": 20000.0,
        "loan_amount": 600000.0,
        "car_loan": 300000.0,
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
    return FinancialDigitalTwin(user_id="user_priya", demographics=demographics, balance_sheet=balance_sheet)


@pytest.fixture
def sample_state(sample_twin) -> FinancialState:
    return FinancialState.from_twin(sample_twin)


# ── 1. Initialization ─────────────────────────────────────────────────────────

def test_agent_initialization():
    agent = ScenarioSimulationAgent()
    assert agent.AGENT_NAME == "ScenarioSimulationAgent"
    assert "scenario" in agent.SUPPORTED_OPERATIONS
    assert "compare" in agent.SUPPORTED_OPERATIONS
    assert "run_scenario" in agent.authorized_tools
    assert "compare_scenario" in agent.authorized_tools
    assert "calculate_health_score" not in agent.authorized_tools
    assert "forecast_savings" not in agent.authorized_tools


# ── 2. Scenario Simulation Operation ──────────────────────────────────────────

def test_scenario_operation_delegation(sample_state):
    agent = ScenarioSimulationAgent()
    result = agent.execute(
        sample_state,
        operation="scenario",
        scenario_name="Salary Hike",
        params={"hike_percent": 25.0},
    )

    assert isinstance(result, AgentResult)
    assert result.status == AgentStatus.SUCCESS
    assert result.agent_name == "ScenarioSimulationAgent"
    assert result.intent == "SCENARIO_SIMULATION"
    assert result.authoritative_data is not None

    prov = result.authoritative_data.provenance
    assert prov.source_engine == "ScenarioSimulator"
    assert prov.source_tool == "run_scenario"
    assert prov.calculation_version == "v1.0.0-simulator"
    assert prov.execution_time_ms > 0

    metrics = result.authoritative_data.metrics
    assert metrics["scenario_name"] == "Salary Hike"
    assert metrics["before_health_score"] > 0
    assert metrics["after_health_score"] >= metrics["before_health_score"]
    assert metrics["health_score_delta"] >= 0
    assert metrics["after_monthly_surplus"] > metrics["before_monthly_surplus"]
    assert metrics["surplus_delta"] > 0
    assert len(result.insights) >= 4


# ── 3. Scenario Comparison Operation ──────────────────────────────────────────

def test_compare_operation_delegation(sample_state):
    agent = ScenarioSimulationAgent()
    result = agent.execute(
        sample_state,
        operation="compare",
        scenario_name="Car Purchase",
        params={"car_price": 700000.0, "downpayment_pct": 0.20},
    )

    assert isinstance(result, AgentResult)
    assert result.status == AgentStatus.SUCCESS
    assert result.agent_name == "ScenarioSimulationAgent"
    assert result.intent == "SCENARIO_COMPARISON"
    assert result.authoritative_data is not None

    prov = result.authoritative_data.provenance
    assert prov.source_engine == "ScenarioSimulator"
    assert prov.source_tool == "compare_scenario"
    assert result.authoritative_data.computation_type == ComputationType.DETERMINISTIC

    metrics = result.authoritative_data.metrics
    assert metrics["scenario_name"] == "Car Purchase"
    assert "before_health_score" in metrics
    assert "after_health_score" in metrics
    assert "health_score_delta" in metrics
    assert "before_monthly_surplus" in metrics
    assert "after_monthly_surplus" in metrics
    assert "surplus_delta" in metrics
    assert "net_worth_delta_5y" in metrics


# ── 4. Standardized .run() Interface ──────────────────────────────────────────

def test_run_interface_alias(sample_state):
    agent = ScenarioSimulationAgent()
    res_exec = agent.execute(sample_state, operation="scenario", scenario_name="Job Loss", params={"duration_months": 3})
    res_run = agent.run(sample_state, operation="scenario", scenario_name="Job Loss", params={"duration_months": 3})

    assert res_exec.status == res_run.status == AgentStatus.SUCCESS
    assert res_exec.intent == res_run.intent
    assert res_exec.authoritative_data.metrics["scenario_name"] == res_run.authoritative_data.metrics["scenario_name"]


# ── 5. Tool Minimization ──────────────────────────────────────────────────────

def test_tool_minimization(sample_state):
    mock_registry = MagicMock(spec=ToolRegistry)
    mock_auth = AuthoritativeData(
        provenance=AuthoritativeProvenance(
            source_engine="ScenarioSimulator",
            source_tool="run_scenario",
            calculation_version="v1.0.0-simulator",
            request_id="req-min-test",
            execution_time_ms=5.0,
        ),
        computation_type=ComputationType.DETERMINISTIC,
        metrics={
            "scenario_name": "Salary Hike",
            "before_health_score": 75.0,
            "after_health_score": 82.0,
            "health_score_delta": 7.0,
            "before_monthly_surplus": 50000.0,
            "after_monthly_surplus": 65000.0,
            "surplus_delta": 15000.0,
            "net_worth_delta_5y": 900000.0,
            "before_risk_level": "Low",
            "after_risk_level": "Low",
        },
    )
    mock_registry.execute.return_value = mock_auth
    mock_registry.get_tools_for_agent.return_value = {"run_scenario": None, "compare_scenario": None}

    agent = ScenarioSimulationAgent(registry=mock_registry)

    # Test scenario minimization: exactly 1 call
    agent.execute(sample_state, operation="scenario", scenario_name="Salary Hike")
    assert mock_registry.execute.call_count == 1
    assert mock_registry.execute.call_args[0][0] == "run_scenario"

    # Reset and test compare minimization: exactly 1 call
    mock_registry.execute.reset_mock()
    agent.execute(sample_state, operation="compare", scenario_name="Salary Hike")
    assert mock_registry.execute.call_count == 1
    assert mock_registry.execute.call_args[0][0] == "compare_scenario"


# ── 6. Unauthorized Tool Boundary Enforcement ─────────────────────────────────

def test_unauthorized_tool_boundary(sample_state):
    agent = ScenarioSimulationAgent()
    unauthorized_tools = [
        "calculate_health_score",
        "calculate_financial_ratios",
        "classify_financial_personality",
        "calculate_tax",
        "analyze_debt",
        "analyze_emergency_fund",
        "analyze_spending_behavior",
        "assess_financial_risk",
        "forecast_savings",
        "forecast_net_worth",
        "analyze_goal",
        "plan_multiple_goals",
        "explain_health_score",
        "explain_forecast",
    ]

    for tool in unauthorized_tools:
        assert tool not in agent.authorized_tools, f"Security violation: Unauthorized tool '{tool}' accessible by ScenarioSimulationAgent"


# ── 7. Health-Score Duplication Regression Test ────────────────────────────────

def test_no_direct_health_score_engine_or_tool_call(sample_state):
    """
    Ensures ScenarioSimulationAgent never directly instantiates HealthScoreEngine
    or calls calculate_health_score tool directly.
    """
    agent = ScenarioSimulationAgent()
    with patch("agents.scenario_simulation.DEFAULT_TOOL_REGISTRY.execute", wraps=DEFAULT_TOOL_REGISTRY.execute) as spy_execute:
        res = agent.execute(sample_state, operation="scenario", scenario_name="Salary Hike", params={"hike_percent": 15.0})
        assert res.status == AgentStatus.SUCCESS
        # Check that calculate_health_score was not invoked directly by the agent
        called_tools = [call[0][0] for call in spy_execute.call_args_list]
        assert "calculate_health_score" not in called_tools
        assert called_tools == ["run_scenario"]


# ── 8. No Duplicated Scenario Math ────────────────────────────────────────────

def test_no_duplicated_scenario_math_consumption(sample_state):
    """
    Verifies the agent passes through and formats authoritative output without
    recalculating deltas or overriding engine output.
    """
    mock_registry = MagicMock(spec=ToolRegistry)
    custom_auth = AuthoritativeData(
        provenance=AuthoritativeProvenance(
            source_engine="ScenarioSimulator",
            source_tool="run_scenario",
            calculation_version="v1.0.0-simulator",
            request_id="req-custom-test",
            execution_time_ms=10.0,
        ),
        computation_type=ComputationType.DETERMINISTIC,
        metrics={
            "scenario_name": "Custom Engine Output",
            "before_health_score": 60.0,
            "after_health_score": 45.0,
            "health_score_delta": -15.0,
            "before_monthly_surplus": 30000.0,
            "after_monthly_surplus": 0.0,
            "surplus_delta": -30000.0,
            "net_worth_delta_5y": -500000.0,
            "before_risk_level": "Low",
            "after_risk_level": "High",
        },
    )
    mock_registry.execute.return_value = custom_auth
    mock_registry.get_tools_for_agent.return_value = {"run_scenario": None, "compare_scenario": None}

    agent = ScenarioSimulationAgent(registry=mock_registry)
    result = agent.execute(sample_state, operation="scenario", scenario_name="Custom Engine Output")

    assert result.status == AgentStatus.SUCCESS
    assert result.authoritative_data.metrics["health_score_delta"] == -15.0
    assert result.authoritative_data.metrics["surplus_delta"] == -30000.0
    # Warnings should be triggered from the negative deltas
    assert any("health score reduction" in w.lower() for w in result.warnings)
    assert any("cash flow deficit" in w.lower() for w in result.warnings)
    assert any("risk escalation" in w.lower() for w in result.warnings)


# ── 9. State Immutability ─────────────────────────────────────────────────────

def test_state_immutability_during_agent_execution(sample_state):
    agent = ScenarioSimulationAgent()
    orig_income = sample_state.cash_flow.monthly_income
    orig_nw = sample_state.balance_sheet.current_net_worth
    orig_score = sample_state.derived.health_score

    res = agent.execute(sample_state, operation="scenario", scenario_name="Job Loss", params={"duration_months": 6})
    assert res.status == AgentStatus.SUCCESS

    assert sample_state.cash_flow.monthly_income == orig_income
    assert sample_state.balance_sheet.current_net_worth == orig_nw
    assert sample_state.derived.health_score == orig_score

    # Confirm frozen dataclass mutation rejection
    with pytest.raises((TypeError, Exception)):
        sample_state.cash_flow.monthly_income = 0.0


# ── 10. Error Handling ────────────────────────────────────────────────────────

def test_unsupported_operation_error_handling(sample_state):
    agent = ScenarioSimulationAgent()
    result = agent.execute(sample_state, operation="unsupported_op")

    assert isinstance(result, AgentResult)
    assert result.status == AgentStatus.FAILED
    assert len(result.errors) > 0
    assert "Unsupported operation" in result.errors[0]


def test_tool_failure_error_handling(sample_state):
    mock_registry = MagicMock(spec=ToolRegistry)
    mock_registry.execute.side_effect = RuntimeError("Simulator engine crash")
    mock_registry.get_tools_for_agent.return_value = {"run_scenario": None, "compare_scenario": None}

    agent = ScenarioSimulationAgent(registry=mock_registry)
    result = agent.execute(sample_state, operation="scenario", scenario_name="Salary Hike")

    assert isinstance(result, AgentResult)
    assert result.status == AgentStatus.FAILED
    assert len(result.errors) > 0
    assert "Tool execution failed" in result.errors[0]


# ── 11. Zero LLM / Zero Network Guarantee ─────────────────────────────────────

def test_zero_llm_invocation(sample_state):
    agent = ScenarioSimulationAgent()
    with patch("urllib.request.urlopen") as mock_url, \
         patch("http.client.HTTPConnection") as mock_http:
        res = agent.execute(sample_state, operation="scenario", scenario_name="Salary Hike")
        assert res.status == AgentStatus.SUCCESS
        mock_url.assert_not_called()
        mock_http.assert_not_called()

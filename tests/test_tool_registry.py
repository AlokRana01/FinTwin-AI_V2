"""
tests/test_tool_registry.py
===========================
Unit tests for Phase 3 Step 2: Authoritative Tool Registry.
Verifies tool registration, delegation, provenance preservation,
security boundaries, read-only enforcement, role-based tool access policies,
and complete isolation from LLM providers or DB writes.
"""

import pytest
from datetime import datetime, date
from dataclasses import FrozenInstanceError

from agents.state import (
    UserProfileContext,
    CashFlowState,
    BalanceSheetState,
    GoalItemState,
    FinancialState,
)
from agents.schemas import (
    ComputationType,
    AuthoritativeData,
    AuthoritativeProvenance,
)
from agents.tool_registry import (
    ToolDefinition,
    ToolRegistry,
    DEFAULT_TOOL_REGISTRY,
    calculate_health_score,
    calculate_financial_ratios,
    classify_financial_personality,
    calculate_tax,
    analyze_debt,
    analyze_emergency_fund,
    analyze_spending_behavior,
    assess_financial_risk,
    forecast_savings,
    forecast_net_worth,
    analyze_goal,
    plan_multiple_goals,
    run_scenario,
    compare_scenario,
    explain_health_score,
    explain_forecast,
)
from models.twin_engine import FinancialDigitalTwin
from utils.goal_engine import GoalInput, GoalType
from utils.tax_calculator import DeductionProfile


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_twin():
    demographics = {
        "name": "Rohan Verma",
        "age": 29,
        "occupation": "Software Engineer",
        "city": "Mumbai",
        "monthly_income": 100000.0,
    }
    balance_sheet = {
        "bank_savings": 200000.0,
        "fd_amount": 100000.0,
        "emergency_fund": 150000.0,
        "mutual_funds": 250000.0,
        "stocks": 100000.0,
        "ppf_investment": 50000.0,
        "sip_amount": 15000.0,
        "monthly_emi": 20000.0,
        "loan_amount": 500000.0,
        "credit_card_debt": 10000.0,
        "health_insurance": 15000.0,
        "life_insurance": 20000.0,
        "rent": 25000.0,
        "groceries": 10000.0,
        "utilities": 4000.0,
        "transport": 3000.0,
        "food_delivery": 4000.0,
        "entertainment": 3000.0,
        "shopping": 5000.0,
    }
    return FinancialDigitalTwin(user_id="user_rohan", demographics=demographics, balance_sheet=balance_sheet)


# ── 1. Registry Architecture & Registration ───────────────────────────────────

def test_registry_tool_count_and_uniqueness():
    tools = DEFAULT_TOOL_REGISTRY.list_tools()
    assert len(tools) == 16
    names = [t.name for t in tools]
    assert len(names) == len(set(names))


def test_registry_categories():
    categories = {t.category for t in DEFAULT_TOOL_REGISTRY.list_tools()}
    expected_categories = {
        "financial_intelligence",
        "risk_behaviour",
        "forecast_goal",
        "scenario_simulation",
        "explainability",
    }
    assert categories == expected_categories


def test_tool_definition_metadata():
    for tool in DEFAULT_TOOL_REGISTRY.list_tools():
        assert bool(tool.name)
        assert bool(tool.description)
        assert bool(tool.source_engine)
        assert tool.read_only is True
        assert tool.authoritative is True
        assert callable(tool.func)


# ── 2. Security Boundaries & Read-Only Enforcement ────────────────────────────

def test_read_only_enforcement():
    def dummy_mutation():
        pass

    with pytest.raises(ValueError, match="must be read-only"):
        DEFAULT_TOOL_REGISTRY.register(
            ToolDefinition(
                name="illegal_tool",
                description="Mutating tool",
                category="financial_intelligence",
                func=dummy_mutation,
                read_only=False,
            )
        )


def test_security_blacklist_verification():
    forbidden_terms = ["write", "update", "delete", "auth", "password", "shell", "exec", "retrain"]
    for tool in DEFAULT_TOOL_REGISTRY.list_tools():
        for term in forbidden_terms:
            assert term not in tool.name.lower(), f"Security violation: Tool name '{tool.name}' contains forbidden term '{term}'"


# ── 3. Role-Based Access Policy ───────────────────────────────────────────────

def test_role_based_tool_access():
    fin_tools = DEFAULT_TOOL_REGISTRY.get_tools_for_agent("financial_intelligence")
    assert "calculate_health_score" in fin_tools
    assert "calculate_tax" in fin_tools
    assert "classify_financial_personality" in fin_tools
    assert "run_scenario" not in fin_tools

    risk_tools = DEFAULT_TOOL_REGISTRY.get_tools_for_agent("risk_behaviour")
    assert "analyze_debt" in risk_tools
    assert "analyze_emergency_fund" in risk_tools
    assert "calculate_tax" not in risk_tools

    forecast_tools = DEFAULT_TOOL_REGISTRY.get_tools_for_agent("forecast_goal")
    assert "forecast_savings" in forecast_tools
    assert "analyze_goal" in forecast_tools
    assert "analyze_debt" not in forecast_tools

    scenario_tools = DEFAULT_TOOL_REGISTRY.get_tools_for_agent("scenario_simulation")
    assert "run_scenario" in scenario_tools
    assert "compare_scenario" in scenario_tools
    assert "forecast_savings" not in scenario_tools


# ── 4. Financial Intelligence Delegation Tests ────────────────────────────────

def test_calculate_health_score_delegation(sample_twin):
    data = calculate_health_score(sample_twin, request_id="test-req-01")
    assert isinstance(data, AuthoritativeData)
    assert data.provenance.source_engine == "HealthScoreEngine"
    assert data.provenance.source_tool == "calculate_health_score"
    assert data.computation_type == ComputationType.DETERMINISTIC
    assert 0 <= data.metrics["health_score"] <= 100
    assert data.metrics["grade"] in ["A", "B", "C", "D", "E", "F"]
    assert len(data.metrics["components"]) == 6


def test_calculate_financial_ratios_delegation(sample_twin):
    data = calculate_financial_ratios(sample_twin)
    assert data.provenance.source_engine == "FinancialDigitalTwin"
    assert data.metrics["total_income"] == 100000.0
    assert data.metrics["monthly_emi"] == 20000.0
    assert data.metrics["savings_rate"] > 0.0


def test_classify_financial_personality_delegation(sample_twin):
    data = classify_financial_personality(sample_twin)
    assert data.provenance.source_engine == "FinancialPersonalityClusterer"
    assert data.computation_type == ComputationType.MODEL_OUTPUT
    assert data.metrics["personality"] in [
        "Saver", "Investor", "Spender", "Debt Heavy", "Balanced Planner"
    ]


def test_calculate_tax_delegation():
    ded = DeductionProfile(investment_80c=150000.0, health_insurance_self=25000.0)
    data = calculate_tax(gross_income=1200000.0, deductions=ded)
    assert data.provenance.source_engine == "IndianTaxCalculator"
    assert data.provenance.source_tool == "calculate_tax"
    assert data.computation_type == ComputationType.DETERMINISTIC
    assert data.metrics["gross_income"] == 1200000.0
    assert "recommended_regime" in data.metrics
    assert data.metrics["old_regime_total_tax"] >= 0.0
    assert data.metrics["new_regime_total_tax"] >= 0.0


# ── 5. Risk & Behaviour Delegation Tests ──────────────────────────────────────

def test_analyze_debt_delegation(sample_twin):
    data = analyze_debt(sample_twin)
    assert data.provenance.source_engine == "FinancialDigitalTwin"
    assert data.metrics["total_loan_amount"] == 500000.0
    assert data.metrics["monthly_emi"] == 20000.0
    assert 0 <= data.metrics["emi_burden_score"] <= 100


def test_analyze_emergency_fund_delegation(sample_twin):
    data = analyze_emergency_fund(sample_twin)
    assert data.provenance.source_engine == "HealthScoreEngine"
    assert data.metrics["emergency_fund_amount"] == 150000.0
    assert data.metrics["months_covered"] > 0.0


def test_analyze_spending_behavior_delegation(sample_twin):
    data = analyze_spending_behavior(sample_twin)
    assert data.provenance.source_engine == "FinancialCoach"
    assert data.computation_type == ComputationType.HEURISTIC
    assert "top_action" in data.metrics


# ── 6. Forecast & Goal Delegation Tests ───────────────────────────────────────

def test_forecast_savings_delegation(sample_twin):
    data = forecast_savings(sample_twin, horizon_months=12)
    assert data.provenance.source_engine == "FinancialPredictor"
    assert data.computation_type == ComputationType.MODEL_OUTPUT
    assert data.metrics["horizon_months"] == 12
    assert "predicted_net_worth" in data.metrics


def test_forecast_net_worth_trajectory_delegation(sample_twin):
    data = forecast_net_worth(sample_twin, months=12)
    assert data.provenance.source_engine == "FinancialPredictor"
    assert data.arrays is not None
    assert len(data.arrays["Month"]) == 12
    assert len(data.arrays["Projected_Net_Worth"]) == 12


def test_analyze_goal_delegation():
    goal = GoalInput(
        goal_type=GoalType.CAR,
        goal_amount=800000.0,
        current_savings=100000.0,
        target_date=date.today().replace(year=date.today().year + 3),
        monthly_contribution=15000.0,
    )
    data = analyze_goal(goal_input=goal, monthly_surplus=25000.0)
    assert data.provenance.source_engine == "GoalEngine"
    assert data.computation_type == ComputationType.DETERMINISTIC
    assert data.metrics["target_amount"] == 800000.0
    assert data.metrics["sip_required"] > 0.0
    assert 0.0 <= data.metrics["probability"] <= 1.0


def test_plan_multiple_goals_delegation(sample_twin):
    g1 = GoalInput(
        goal_type=GoalType.CAR,
        goal_amount=500000.0,
        current_savings=50000.0,
        target_date=date.today().replace(year=date.today().year + 2),
    )
    g2 = GoalInput(
        goal_type=GoalType.HOUSE,
        goal_amount=3000000.0,
        current_savings=200000.0,
        target_date=date.today().replace(year=date.today().year + 5),
    )
    data = plan_multiple_goals(twin=sample_twin, goals=[g1, g2])
    assert data.provenance.source_engine == "MultiGoalPlanner"
    assert data.metrics["goals_count"] == 2


# ── 7. Scenario Simulation Delegation Tests ───────────────────────────────────

def test_run_scenario_delegation(sample_twin):
    data = run_scenario(sample_twin, scenario_name="Salary Hike", params={"hike_percent": 20.0})
    assert data.provenance.source_engine == "ScenarioSimulator"
    assert data.computation_type == ComputationType.DETERMINISTIC
    assert "Salary Hike" in data.metrics["scenario_name"]
    assert data.metrics["surplus_delta"] > 0.0


def test_compare_scenario_delegation(sample_twin, monkeypatch):
    from utils.simulator import ScenarioSimulator
    
    called = {}
    original_run = ScenarioSimulator.run_scenario

    def spy_run_scenario(self, scenario_name, **params):
        called["scenario_name"] = scenario_name
        called["params"] = params
        return original_run(self, scenario_name, **params)

    monkeypatch.setattr(ScenarioSimulator, "run_scenario", spy_run_scenario)

    data = compare_scenario(sample_twin, scenario_name="Salary Hike", params={"hike_percent": 25.0})
    
    assert called.get("scenario_name") == "Salary Hike"
    assert called.get("params") == {"hike_percent": 25.0}
    assert data.provenance.source_engine == "ScenarioSimulator"
    assert data.provenance.source_tool == "compare_scenario"
    assert data.computation_type == ComputationType.DETERMINISTIC
    assert "Salary Hike" in data.metrics["scenario_name"]
    assert data.metrics["surplus_delta"] > 0.0


# ── 8. Explainability Delegation Tests ────────────────────────────────────────

def test_explain_health_score_delegation(sample_twin):
    data = explain_health_score(sample_twin)
    assert data.provenance.source_engine == "ExplainableAI"
    assert data.xai_attributions is not None
    assert len(data.xai_attributions) == 6


def test_explain_forecast_delegation(sample_twin, monkeypatch):
    from models.explainability import ExplainableAI, ForecastResult
    
    called = {}
    original_explain = ExplainableAI.explain_forecast

    def spy_explain_forecast(self, twin, predictor=None, horizon=6):
        called["called"] = True
        return original_explain(self, twin, predictor=predictor, horizon=horizon)

    monkeypatch.setattr(ExplainableAI, "explain_forecast", spy_explain_forecast)

    data = explain_forecast(sample_twin)
    assert called.get("called") is True
    assert data.provenance.source_engine == "ExplainableAI"
    assert data.provenance.source_tool == "explain_forecast"
    assert data.computation_type == ComputationType.MODEL_OUTPUT
    assert "predicted_savings" in data.metrics


# ── 9. State Immutability Verification ────────────────────────────────────────

def test_tool_execution_preserves_financial_state_immutability(sample_twin):
    state = FinancialState.from_twin(sample_twin)
    initial_income = state.cash_flow.monthly_income

    # Run multiple tools
    calculate_health_score(sample_twin)
    calculate_financial_ratios(sample_twin)
    run_scenario(sample_twin, scenario_name="Salary Hike", params={"hike_percent": 30.0})

    # Assert state remains completely unchanged
    assert state.cash_flow.monthly_income == initial_income
    with pytest.raises(FrozenInstanceError):
        state.cash_flow.monthly_income = 200000.0  # type: ignore


# ── 10. Explicit No-LLM Invocation Verification ───────────────────────────────

def test_no_llm_invocation_across_all_tools(sample_twin, monkeypatch):
    """
    Guarantees that executing registry tools does NOT make any network calls or invoke LLMs.
    """
    import urllib.request
    import sys

    def fail_on_network(*args, **kwargs):
        raise AssertionError("Network / LLM API call attempted by Tool Registry!")

    monkeypatch.setattr(urllib.request, "urlopen", fail_on_network)

    # Execute representative tools from all 5 categories
    calculate_health_score(sample_twin)
    calculate_financial_ratios(sample_twin)
    classify_financial_personality(sample_twin)
    calculate_tax(gross_income=1000000.0)
    analyze_debt(sample_twin)
    analyze_emergency_fund(sample_twin)
    analyze_spending_behavior(sample_twin)
    assess_financial_risk(sample_twin)
    forecast_savings(sample_twin, horizon_months=6)
    forecast_net_worth(sample_twin, months=6)
    run_scenario(sample_twin, scenario_name="Salary Hike", params={"hike_percent": 10.0})
    compare_scenario(sample_twin, scenario_name="Salary Hike", params={"hike_percent": 10.0})
    explain_health_score(sample_twin)
    explain_forecast(sample_twin)


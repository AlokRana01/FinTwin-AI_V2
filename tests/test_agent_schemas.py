"""
tests/test_agent_schemas.py
===========================
Unit tests for Phase 3 Step 1: Core State & Schemas.
Verifies deep immutability, data contracts, validation, provenance,
and clean separation of FinancialState and SessionContext.
"""

import pytest
from dataclasses import FrozenInstanceError
from types import MappingProxyType

from agents.state import (
    UserProfileContext,
    CashFlowState,
    BalanceSheetState,
    GoalItemState,
    DerivedIntelligenceState,
    FinancialState,
    SessionContext,
)
from agents.schemas import (
    AgentStatus,
    ComputationType,
    AuthoritativeProvenance,
    AuthoritativeData,
    AgentResult,
)


# ── 1. FinancialState Instantiation & Immutability ─────────────────────────────

def test_financial_state_instantiation():
    profile = UserProfileContext(user_id="user_123", username="Alok", age=28)
    cash_flow = CashFlowState(monthly_income=100000.0, monthly_expenses=40000.0)
    balance_sheet = BalanceSheetState(bank_savings=200000.0, current_net_worth=500000.0)
    goal = GoalItemState(goal_id="g1", goal_name="House", goal_type="House", target_amount=2500000.0)

    state = FinancialState(
        snapshot_id="snap-001",
        timestamp="2026-09-06T12:00:00",
        profile=profile,
        cash_flow=cash_flow,
        balance_sheet=balance_sheet,
        goals=(goal,),
        derived=DerivedIntelligenceState(health_score=82.0, health_grade="A"),
    )

    assert state.snapshot_id == "snap-001"
    assert state.profile.username == "Alok"
    assert state.cash_flow.monthly_income == 100000.0
    assert len(state.goals) == 1
    assert state.derived.health_score == 82.0


def test_financial_state_deep_immutability():
    profile = UserProfileContext(user_id="user_123", username="Alok", age=28)
    cash_flow = CashFlowState(monthly_income=100000.0)
    balance_sheet = BalanceSheetState(bank_savings=200000.0)
    goal = GoalItemState(goal_id="g1", goal_name="House", goal_type="House", target_amount=2500000.0)

    state = FinancialState(
        snapshot_id="snap-001",
        timestamp="2026-09-06T12:00:00",
        profile=profile,
        cash_flow=cash_flow,
        balance_sheet=balance_sheet,
        goals=(goal,),
    )

    # Reassignment must raise FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        state.snapshot_id = "snap-999"  # type: ignore

    with pytest.raises(FrozenInstanceError):
        state.profile.age = 30  # type: ignore

    with pytest.raises(FrozenInstanceError):
        state.cash_flow.monthly_income = 120000.0  # type: ignore

    # Goals collection is an immutable tuple
    assert isinstance(state.goals, tuple)
    with pytest.raises(AttributeError):
        state.goals.append(goal)  # type: ignore


# ── 2. Structural Validation ──────────────────────────────────────────────────

def test_schema_structural_validation():
    # Invalid age
    with pytest.raises(ValueError, match="Age must be between 18 and 100"):
        UserProfileContext(user_id="u1", age=15)

    with pytest.raises(ValueError, match="Age must be between 18 and 100"):
        UserProfileContext(user_id="u1", age=105)

    # Negative cash flow
    with pytest.raises(ValueError, match="monthly_income cannot be negative"):
        CashFlowState(monthly_income=-500.0)

    # Negative goal amount
    with pytest.raises(ValueError, match="target_amount cannot be negative"):
        GoalItemState(goal_id="g1", goal_name="Car", goal_type="Car", target_amount=-1000.0)


# ── 3. SessionContext Mutability Separation ────────────────────────────────────

def test_session_context_mutability():
    session = SessionContext(session_id="sess_abc", active_page="01_Dashboard")
    assert session.turn_count == 0
    assert len(session.history) == 0

    turn = session.add_turn(
        user_query="What is my health score?",
        orchestrator_intent="HEALTH_OVERVIEW",
        final_response="Your health score is 78/100 (Grade B).",
        agents_invoked=["FinancialIntelligenceAgent", "FinancialCoachAgent"],
    )

    assert session.turn_count == 1
    assert len(session.history) == 1
    assert turn.turn_id == "turn-001"
    assert "health score is 78" in turn.final_response


# ── 4. AgentResult Success, Partial, and Failure ──────────────────────────────

def test_agent_result_success_and_provenance():
    prov = AuthoritativeProvenance(
        source_engine="HealthScoreEngine",
        source_tool="calculate_health_score",
        calculation_version="v1.2.0-deterministic",
        request_id="req-001",
        execution_time_ms=15.4,
    )

    data = AuthoritativeData(
        provenance=prov,
        computation_type=ComputationType.DETERMINISTIC,
        metrics={"health_score": 85.0, "grade": "A"},
        xai_attributions={"savings_rate": 14.2, "emergency_fund": -5.0},
    )

    result = AgentResult.create_success(
        agent_name="FinancialIntelligenceAgent",
        intent="HEALTH_OVERVIEW",
        authoritative_data=data,
        insights=["Your savings rate is above average."],
        warnings=[],
    )

    assert result.status == AgentStatus.SUCCESS
    assert result.authoritative_data is not None
    assert result.authoritative_data.provenance.source_engine == "HealthScoreEngine"
    assert result.authoritative_data.computation_type == ComputationType.DETERMINISTIC
    assert result.authoritative_data.metrics["health_score"] == 85.0
    assert len(result.insights) == 1
    assert len(result.errors) == 0

    # Immutability check
    with pytest.raises(FrozenInstanceError):
        result.status = AgentStatus.FAILED  # type: ignore


def test_agent_result_partial_and_failure():
    # Partial result
    prov = AuthoritativeProvenance(
        source_engine="FinancialPredictor",
        source_tool="predict_financial_forecast",
        calculation_version="v1.0.0-fallback",
        request_id="req-002",
        execution_time_ms=2.1,
    )
    fallback_data = AuthoritativeData(
        provenance=prov,
        computation_type=ComputationType.FALLBACK,
        metrics={"projected_net_worth": 600000.0},
    )
    partial_result = AgentResult.create_partial(
        agent_name="ForecastGoalAgent",
        intent="FORECAST_12M",
        authoritative_data=fallback_data,
        warnings=["ML model unavailable, used compound growth fallback."],
        errors=["Model file missing."],
    )

    assert partial_result.status == AgentStatus.PARTIAL
    assert len(partial_result.warnings) == 1
    assert len(partial_result.errors) == 1

    # Failure result
    fail_result = AgentResult.create_failure(
        agent_name="ScenarioSimulationAgent",
        intent="SCENARIO_SIMULATION",
        errors=["Invalid loan tenure provided."],
    )
    assert fail_result.status == AgentStatus.FAILED
    assert fail_result.authoritative_data is None
    assert len(fail_result.errors) == 1


# ── 5. Serialization Check ───────────────────────────────────────────────────

def test_serialization_to_dict():
    profile = UserProfileContext(user_id="u1", username="Alok")
    state = FinancialState(
        snapshot_id="s1",
        timestamp="2026-09-06T12:00:00",
        profile=profile,
        cash_flow=CashFlowState(monthly_income=50000.0),
        balance_sheet=BalanceSheetState(current_net_worth=100000.0),
    )

    d = state.to_dict()
    assert isinstance(d, dict)
    assert d["snapshot_id"] == "s1"
    assert d["profile"]["username"] == "Alok"
    assert d["cash_flow"]["monthly_income"] == 50000.0

    prov = AuthoritativeProvenance(
        source_engine="HealthScoreEngine",
        source_tool="calculate_health_score",
        calculation_version="v1.2",
        request_id="r1",
    )
    auth_data = AuthoritativeData(
        provenance=prov,
        computation_type=ComputationType.DETERMINISTIC,
        metrics={"score": 80},
    )
    res = AgentResult.create_success("FinAgent", "INTENT", auth_data)
    res_dict = res.to_dict()
    assert res_dict["status"] == "success"
    assert res_dict["authoritative_data"]["metrics"]["score"] == 80


# ── 6. Adapter from FinancialDigitalTwin ───────────────────────────────────────

def test_financial_state_from_twin():
    from models.twin_engine import FinancialDigitalTwin

    demographics = {
        "name": "Priya Sharma",
        "age": 30,
        "occupation": "Software Engineer",
        "city": "Bengaluru",
        "monthly_income": 120000.0,
    }
    balance_sheet = {
        "bank_savings": 300000.0,
        "emergency_fund": 200000.0,
        "monthly_emi": 25000.0,
        "rent": 30000.0,
        "groceries": 15000.0,
        "utilities": 5000.0,
        "transport": 5000.0,
        "food_delivery": 5000.0,
        "entertainment": 5000.0,
        "shopping": 5000.0,
    }
    twin = FinancialDigitalTwin(user_id="user_99", demographics=demographics, balance_sheet=balance_sheet)
    
    state = FinancialState.from_twin(twin)
    assert state.profile.user_id == "user_99"
    assert state.profile.username == "Priya Sharma"
    assert state.profile.age == 30
    assert state.cash_flow.monthly_income == 120000.0
    assert state.cash_flow.monthly_expenses == 70000.0
    assert state.cash_flow.monthly_emi == 25000.0
    assert state.cash_flow.monthly_savings == 25000.0
    assert state.balance_sheet.emergency_fund == 200000.0
    assert isinstance(state.goals, tuple)


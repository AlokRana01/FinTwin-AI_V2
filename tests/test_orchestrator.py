"""
tests/test_orchestrator.py
==========================
Comprehensive test suite for OrchestratorAgent (agents/orchestrator.py) verifying:
1. Intent routing across all supported routes (health, risk, forecast, goal, scenario, overview, health_diagnostic, scenario_forecast, comprehensive).
2. Minimum-agent principle (only invokes required specialists per route).
3. Seamless handoff of specialist AgentResults to FinancialCoachAgent.
4. Conditional DAG ordering (Scenario -> Forecast in scenario_forecast).
5. Partial execution and graceful recovery when one specialist fails.
6. Dependency failure handling (skips dependent forecast if scenario simulation fails).
7. Safe rejection and error handling for unknown/unsupported intents.
8. Request ID propagation and upstream provenance retention.
9. State immutability (FinancialState remains read-only).
10. Coach failure tolerance (returns specialist results without crashing).
11. Zero-LLM, zero-network, and zero-tool-registry-bypass guarantees.
"""

import pytest
from unittest.mock import MagicMock, patch

from agents.state import FinancialState
from agents.schemas import (
    AgentResult,
    AgentStatus,
    AuthoritativeData,
    AuthoritativeProvenance,
    ComputationType,
)
from agents.orchestrator import OrchestratorAgent, AGENT_REGISTRY, DEFAULT_ORCHESTRATOR
from agents.financial_intelligence import FinancialIntelligenceAgent
from agents.risk_behaviour import RiskBehaviourAgent
from agents.forecast_goal import ForecastGoalAgent
from agents.scenario_simulation import ScenarioSimulationAgent
from agents.financial_coach import FinancialCoachAgent
from models.twin_engine import FinancialDigitalTwin


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_twin():
    demographics = {
        "name": "Arjun Nair",
        "age": 31,
        "occupation": "Tech Lead",
        "city": "Bengaluru",
        "monthly_income": 160000.0,
    }
    balance_sheet = {
        "bank_savings": 350000.0,
        "fd_amount": 150000.0,
        "emergency_fund": 300000.0,
        "mutual_funds": 500000.0,
        "stocks": 250000.0,
        "ppf_investment": 120000.0,
        "sip_amount": 30000.0,
        "monthly_emi": 35000.0,
        "loan_amount": 1200000.0,
        "car_loan": 400000.0,
        "credit_card_debt": 0.0,
        "health_insurance": 18000.0,
        "life_insurance": 30000.0,
        "rent": 35000.0,
        "groceries": 15000.0,
        "utilities": 5000.0,
        "transport": 5000.0,
        "food_delivery": 4000.0,
        "entertainment": 5000.0,
        "shopping": 6000.0,
    }
    return FinancialDigitalTwin(user_id="user_arjun", demographics=demographics, balance_sheet=balance_sheet)


@pytest.fixture
def sample_state(sample_twin) -> FinancialState:
    return FinancialState.from_twin(sample_twin)


@pytest.fixture
def orchestrator() -> OrchestratorAgent:
    return OrchestratorAgent()


# ── 1. Registry & Supported Routes ────────────────────────────────────────────

def test_orchestrator_initialization_and_registry(orchestrator):
    assert orchestrator.AGENT_NAME == "OrchestratorAgent"
    assert len(orchestrator.SUPPORTED_ROUTES) == 9
    assert "health" in orchestrator.SUPPORTED_ROUTES
    assert "risk" in orchestrator.SUPPORTED_ROUTES
    assert "forecast" in orchestrator.SUPPORTED_ROUTES
    assert "goal" in orchestrator.SUPPORTED_ROUTES
    assert "scenario" in orchestrator.SUPPORTED_ROUTES
    assert "overview" in orchestrator.SUPPORTED_ROUTES
    assert "health_diagnostic" in orchestrator.SUPPORTED_ROUTES
    assert "scenario_forecast" in orchestrator.SUPPORTED_ROUTES
    assert "comprehensive" in orchestrator.SUPPORTED_ROUTES

    assert AGENT_REGISTRY["financial_intelligence"] is FinancialIntelligenceAgent
    assert AGENT_REGISTRY["risk_behaviour"] is RiskBehaviourAgent
    assert AGENT_REGISTRY["forecast_goal"] is ForecastGoalAgent
    assert AGENT_REGISTRY["scenario_simulation"] is ScenarioSimulationAgent
    assert AGENT_REGISTRY["financial_coach"] is FinancialCoachAgent


# ── 2. Route Routing & Minimum Agent Principle ────────────────────────────────

def test_route_health_minimum_agents(orchestrator, sample_state):
    res = orchestrator.run(intent="health", state=sample_state)

    assert res.status == AgentStatus.SUCCESS
    assert res.agent_name == "OrchestratorAgent"
    assert res.intent == "HEALTH"
    assert res.authoritative_data is not None

    metrics = res.authoritative_data.metrics
    assert metrics["route"] == "health"
    assert metrics["agents_requested"] == ("FinancialIntelligenceAgent", "FinancialCoachAgent")
    assert metrics["agents_completed"] == ("FinancialIntelligenceAgent", "FinancialCoachAgent")
    assert metrics["agents_failed"] == ()
    assert metrics["specialist_count"] == 1


def test_route_risk_minimum_agents(orchestrator, sample_state):
    res = orchestrator.run(intent="risk", state=sample_state)

    assert res.status == AgentStatus.SUCCESS
    metrics = res.authoritative_data.metrics
    assert metrics["route"] == "risk"
    assert metrics["agents_requested"] == ("RiskBehaviourAgent", "FinancialCoachAgent")
    assert metrics["agents_completed"] == ("RiskBehaviourAgent", "FinancialCoachAgent")
    assert metrics["specialist_count"] == 1


def test_route_forecast_minimum_agents(orchestrator, sample_state):
    res = orchestrator.run(intent="forecast", state=sample_state, horizon_months=12)

    assert res.status == AgentStatus.SUCCESS
    metrics = res.authoritative_data.metrics
    assert metrics["route"] == "forecast"
    assert metrics["agents_requested"] == ("ForecastGoalAgent", "FinancialCoachAgent")
    assert metrics["specialist_count"] == 1


def test_route_goal_minimum_agents(orchestrator, sample_state):
    res = orchestrator.run(intent="goal", state=sample_state)

    assert res.status == AgentStatus.SUCCESS
    metrics = res.authoritative_data.metrics
    assert metrics["route"] == "goal"
    assert metrics["agents_requested"] == ("ForecastGoalAgent", "FinancialCoachAgent")
    assert metrics["specialist_count"] == 1


def test_route_scenario_minimum_agents(orchestrator, sample_state):
    res = orchestrator.run(intent="scenario", state=sample_state, scenario_name="Salary Hike", scenario_params={"hike_percent": 20.0})

    assert res.status == AgentStatus.SUCCESS
    metrics = res.authoritative_data.metrics
    assert metrics["route"] == "scenario"
    assert metrics["agents_requested"] == ("ScenarioSimulationAgent", "FinancialCoachAgent")
    assert metrics["specialist_count"] == 1


def test_route_overview_agents(orchestrator, sample_state):
    res = orchestrator.run(intent="overview", state=sample_state)

    assert res.status == AgentStatus.SUCCESS
    metrics = res.authoritative_data.metrics
    assert metrics["route"] == "overview"
    assert "FinancialIntelligenceAgent" in metrics["agents_completed"]
    assert "RiskBehaviourAgent" in metrics["agents_completed"]
    assert "FinancialCoachAgent" in metrics["agents_completed"]
    assert metrics["specialist_count"] == 2


def test_route_health_diagnostic_agents(orchestrator, sample_state):
    res = orchestrator.run(intent="health_diagnostic", state=sample_state)

    assert res.status == AgentStatus.SUCCESS
    metrics = res.authoritative_data.metrics
    assert metrics["route"] == "health_diagnostic"
    assert metrics["specialist_count"] == 3  # FI (health) + Risk (debt) + Risk (emergency_fund)


def test_route_comprehensive_agents(orchestrator, sample_state):
    res = orchestrator.run(intent="comprehensive", state=sample_state, horizon_months=12)

    assert res.status == AgentStatus.SUCCESS
    metrics = res.authoritative_data.metrics
    assert metrics["route"] == "comprehensive"
    assert metrics["specialist_count"] >= 3
    assert "FinancialIntelligenceAgent" in metrics["agents_completed"]
    assert "RiskBehaviourAgent" in metrics["agents_completed"]
    assert "ForecastGoalAgent" in metrics["agents_completed"]
    assert "FinancialCoachAgent" in metrics["agents_completed"]


# ── 3. Conditional DAG Execution & Ordering (Scenario + Forecast) ─────────────

def test_route_scenario_forecast_dag_ordering(sample_state):
    call_order = []

    mock_sim = MagicMock(spec=ScenarioSimulationAgent)
    mock_sim.AGENT_NAME = "ScenarioSimulationAgent"
    def sim_side_effect(*args, **kwargs):
        call_order.append("ScenarioSimulationAgent")
        return AgentResult.create_success(
            agent_name="ScenarioSimulationAgent",
            intent="SCENARIO",
            authoritative_data=AuthoritativeData(
                provenance=AuthoritativeProvenance("ScenarioSimulator", "run_scenario", "v1", "r1"),
                computation_type=ComputationType.DETERMINISTIC,
                metrics={"scenario_name": "Car Purchase", "health_score_delta": -5.0},
            ),
        )
    mock_sim.execute.side_effect = sim_side_effect

    mock_fg = MagicMock(spec=ForecastGoalAgent)
    mock_fg.AGENT_NAME = "ForecastGoalAgent"
    def fg_side_effect(*args, **kwargs):
        call_order.append("ForecastGoalAgent")
        return AgentResult.create_success(
            agent_name="ForecastGoalAgent",
            intent="FORECAST",
            authoritative_data=AuthoritativeData(
                provenance=AuthoritativeProvenance("FinancialPredictor", "forecast_savings", "v1", "r1"),
                computation_type=ComputationType.MODEL_OUTPUT,
                metrics={"predicted_savings": 150000.0},
            ),
        )
    mock_fg.execute.side_effect = fg_side_effect

    mock_coach = MagicMock(spec=FinancialCoachAgent)
    mock_coach.AGENT_NAME = "FinancialCoachAgent"
    def coach_side_effect(*args, **kwargs):
        call_order.append("FinancialCoachAgent")
        return AgentResult.create_success(
            agent_name="FinancialCoachAgent",
            intent="COACH",
            authoritative_data=AuthoritativeData(
                provenance=AuthoritativeProvenance("FinancialCoachAgent", "coach", "v1", "r1"),
                computation_type=ComputationType.HEURISTIC,
                metrics={"summary": "Grounded synthesis"},
            ),
            insights=("Grounded synthesis insight",),
        )
    mock_coach.run.side_effect = coach_side_effect

    orch = OrchestratorAgent(
        scenario_simulation_agent=mock_sim,
        forecast_goal_agent=mock_fg,
        financial_coach_agent=mock_coach,
    )

    res = orch.run(intent="scenario_forecast", state=sample_state, scenario_name="Car Purchase")

    assert res.status == AgentStatus.SUCCESS
    # Verify strict DAG execution order: Scenario -> Forecast -> Coach
    assert call_order == ["ScenarioSimulationAgent", "ForecastGoalAgent", "FinancialCoachAgent"]

    # Verify ForecastGoalAgent received the scenario-adjusted context
    fg_call_kwargs = mock_fg.execute.call_args[1]
    assert "monthly_surplus_override" in fg_call_kwargs

    # Verify Coach received the actual specialist results (both Scenario and Forecast)
    coach_call_kwargs = mock_coach.run.call_args[1]
    passed_specialists = coach_call_kwargs["specialist_results"]
    assert len(passed_specialists) == 2
    assert passed_specialists[0].agent_name == "ScenarioSimulationAgent"
    assert passed_specialists[1].agent_name == "ForecastGoalAgent"


# ── 4. Dependency Failure Handling ────────────────────────────────────────────

def test_scenario_forecast_dependency_failure(sample_state):
    mock_sim = MagicMock(spec=ScenarioSimulationAgent)
    mock_sim.AGENT_NAME = "ScenarioSimulationAgent"
    mock_sim.execute.return_value = AgentResult.create_failure(
        agent_name="ScenarioSimulationAgent",
        intent="SCENARIO",
        errors=["Scenario simulation engine crashed"],
    )

    mock_fg = MagicMock(spec=ForecastGoalAgent)
    mock_fg.AGENT_NAME = "ForecastGoalAgent"

    mock_coach = MagicMock(spec=FinancialCoachAgent)
    mock_coach.AGENT_NAME = "FinancialCoachAgent"
    mock_coach.run.return_value = AgentResult.create_partial(
        agent_name="FinancialCoachAgent",
        intent="COACH",
        authoritative_data=None,
        insights=("Partial coaching summary",),
    )

    orch = OrchestratorAgent(
        scenario_simulation_agent=mock_sim,
        forecast_goal_agent=mock_fg,
        financial_coach_agent=mock_coach,
    )

    res = orch.run(intent="scenario_forecast", state=sample_state, scenario_name="Job Loss")

    assert res.status in (AgentStatus.PARTIAL, AgentStatus.FAILED)
    # Forecast must NOT be called when scenario simulation fails
    mock_fg.execute.assert_not_called()
    assert any("Dependent forecast skipped" in w for w in res.warnings)


# ── 5. Partial Specialist Failure Handling ────────────────────────────────────

def test_partial_specialist_failure_recovery(sample_state):
    mock_fi = MagicMock(spec=FinancialIntelligenceAgent)
    mock_fi.AGENT_NAME = "FinancialIntelligenceAgent"
    mock_fi.execute.return_value = AgentResult.create_success(
        agent_name="FinancialIntelligenceAgent",
        intent="HEALTH",
        authoritative_data=AuthoritativeData(
            provenance=AuthoritativeProvenance("HealthScoreEngine", "calculate_health_score", "v1", "r1"),
            computation_type=ComputationType.DETERMINISTIC,
            metrics={"health_score": 80.0},
        ),
    )

    mock_risk = MagicMock(spec=RiskBehaviourAgent)
    mock_risk.AGENT_NAME = "RiskBehaviourAgent"
    mock_risk.execute.return_value = AgentResult.create_failure(
        agent_name="RiskBehaviourAgent",
        intent="RISK",
        errors=["Risk calculation module timeout"],
    )

    mock_coach = MagicMock(spec=FinancialCoachAgent)
    mock_coach.AGENT_NAME = "FinancialCoachAgent"
    mock_coach.run.return_value = AgentResult.create_success(
        agent_name="FinancialCoachAgent",
        intent="COACH",
        authoritative_data=AuthoritativeData(
            provenance=AuthoritativeProvenance("FinancialCoachAgent", "coach", "v1", "r1"),
            computation_type=ComputationType.HEURISTIC,
            metrics={"summary": "Coached based on FinancialIntelligence"},
        ),
        insights=("Coached based on FinancialIntelligence",),
    )

    orch = OrchestratorAgent(
        financial_intelligence_agent=mock_fi,
        risk_behaviour_agent=mock_risk,
        financial_coach_agent=mock_coach,
    )

    res = orch.run(intent="overview", state=sample_state)

    assert res.status == AgentStatus.PARTIAL
    metrics = res.authoritative_data.metrics
    assert "FinancialIntelligenceAgent" in metrics["agents_completed"]
    assert "RiskBehaviourAgent" in metrics["agents_failed"]
    # Coach must receive the successful FI result
    assert mock_coach.run.call_count == 1
    passed_results = mock_coach.run.call_args[1]["specialist_results"]
    assert len(passed_results) == 2  # FI success + Risk failure passed for transparency


# ── 6. Coach Failure Handling ─────────────────────────────────────────────────

def test_coach_failure_fallback_to_specialist_insights(sample_state):
    mock_fi = MagicMock(spec=FinancialIntelligenceAgent)
    mock_fi.AGENT_NAME = "FinancialIntelligenceAgent"
    mock_fi.execute.return_value = AgentResult.create_success(
        agent_name="FinancialIntelligenceAgent",
        intent="HEALTH",
        authoritative_data=AuthoritativeData(
            provenance=AuthoritativeProvenance("HealthScoreEngine", "calculate_health_score", "v1", "r1"),
            computation_type=ComputationType.DETERMINISTIC,
            metrics={"health_score": 75.0},
        ),
        insights=("Specialist Health Insight: Score is 75.0.",),
    )

    mock_coach = MagicMock(spec=FinancialCoachAgent)
    mock_coach.AGENT_NAME = "FinancialCoachAgent"
    mock_coach.run.side_effect = RuntimeError("Coach rule engine syntax error")

    orch = OrchestratorAgent(
        financial_intelligence_agent=mock_fi,
        financial_coach_agent=mock_coach,
    )

    res = orch.run(intent="health", state=sample_state)

    assert res.status == AgentStatus.PARTIAL
    assert "FinancialCoachAgent" in res.authoritative_data.metrics["agents_failed"]
    # Should fall back to specialist raw insights
    assert any("Specialist Health Insight" in ins for ins in res.insights)


# ── 7. Unknown Intent Handling ────────────────────────────────────────────────

def test_unknown_intent_rejection(orchestrator, sample_state):
    res = orchestrator.run(intent="arbitrary_unknown_intent", state=sample_state)

    assert res.status == AgentStatus.FAILED
    assert len(res.errors) > 0
    assert "Unsupported orchestration intent" in res.errors[0]


# ── 8. Request ID Propagation & Provenance Retention ──────────────────────────

def test_request_id_propagation_and_provenance(orchestrator, sample_state):
    custom_req_id = "req-custom-trace-999"
    res = orchestrator.run(intent="overview", state=sample_state, request_id=custom_req_id)

    assert res.status == AgentStatus.SUCCESS
    auth = res.authoritative_data
    assert auth.provenance.request_id == custom_req_id
    assert auth.provenance.source_engine == "OrchestratorAgent"
    assert auth.provenance.source_tool == "orchestrate"

    upstream_prov = auth.metrics["upstream_provenance"]
    assert len(upstream_prov) == 2
    for p in upstream_prov:
        assert p["request_id"] == custom_req_id


# ── 9. State Immutability ─────────────────────────────────────────────────────

def test_state_immutability(orchestrator, sample_state):
    orig_income = sample_state.cash_flow.monthly_income
    orig_nw = sample_state.balance_sheet.current_net_worth
    orig_score = sample_state.derived.health_score

    res = orchestrator.run(intent="comprehensive", state=sample_state)
    assert res.status == AgentStatus.SUCCESS

    assert sample_state.cash_flow.monthly_income == orig_income
    assert sample_state.balance_sheet.current_net_worth == orig_nw
    assert sample_state.derived.health_score == orig_score


# ── 10. Security & Boundary Verification ──────────────────────────────────────

def test_zero_llm_or_direct_tool_registry_invocation(sample_state):
    """
    Verifies the Orchestrator coordinates only agents and never directly
    invokes ToolRegistry or external network/LLM calls.
    """
    orch = OrchestratorAgent(financial_coach_agent=FinancialCoachAgent(enable_llm=False))
    with patch("agents.llm.LLMService.generate") as mock_llm_gen, \
         patch("urllib.request.urlopen") as mock_url:

        res = orch.run(intent="overview", state=sample_state)
        assert res.status == AgentStatus.SUCCESS

        mock_llm_gen.assert_not_called()
        mock_url.assert_not_called()


def test_standardized_execute_alias(orchestrator, sample_state):
    res_run = orchestrator.run(intent="health", state=sample_state)
    res_exec = orchestrator.execute(intent="health", state=sample_state)

    assert res_run.status == res_exec.status == AgentStatus.SUCCESS
    assert res_run.intent == res_exec.intent

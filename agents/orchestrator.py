"""
agents/orchestrator.py
======================
Central Orchestrator Agent for FinTwin AI's Multi-Agent Architecture.

Architectural Guarantees:
- Pure coordination layer: zero financial math, formula execution, or ML inference.
- Conditional DAG execution: invokes ONLY the minimum required specialist agents per intent.
- Preserves computational provenance across all participating specialist engines.
- Feeds authoritative specialist AgentResults directly into FinancialCoachAgent for grounded synthesis.
- Zero LLM calls in orchestration: routing and execution order are strictly deterministic.
- Deep state immutability: treats FinancialState as a read-only snapshot.
"""

from typing import Optional, Any, Dict, List, Union, Mapping, Sequence, Tuple
import uuid
import time
from datetime import datetime, timezone

from agents.state import FinancialState, SessionContext
from agents.schemas import (
    AgentResult,
    AgentStatus,
    AuthoritativeData,
    AuthoritativeProvenance,
    ComputationType,
)

# Specialist Agent Imports
from agents.financial_intelligence import FinancialIntelligenceAgent
from agents.risk_behaviour import RiskBehaviourAgent
from agents.forecast_goal import ForecastGoalAgent
from agents.scenario_simulation import ScenarioSimulationAgent
from agents.financial_coach import FinancialCoachAgent


# ── Explicit Specialist Agent Registry ────────────────────────────────────────

AGENT_REGISTRY = {
    "financial_intelligence": FinancialIntelligenceAgent,
    "risk_behaviour": RiskBehaviourAgent,
    "forecast_goal": ForecastGoalAgent,
    "scenario_simulation": ScenarioSimulationAgent,
    "financial_coach": FinancialCoachAgent,
}


# ── Orchestrator Agent Implementation ─────────────────────────────────────────

class OrchestratorAgent:
    """
    Central coordinator managing intent routing, sequential conditional DAG execution,
    specialist result aggregation, and handoff to FinancialCoachAgent.
    """
    AGENT_NAME = "OrchestratorAgent"

    SUPPORTED_ROUTES = {
        "health",
        "risk",
        "forecast",
        "goal",
        "scenario",
        "overview",
        "health_diagnostic",
        "scenario_forecast",
        "comprehensive",
    }

    def __init__(
        self,
        financial_intelligence_agent: Optional[FinancialIntelligenceAgent] = None,
        risk_behaviour_agent: Optional[RiskBehaviourAgent] = None,
        forecast_goal_agent: Optional[ForecastGoalAgent] = None,
        scenario_simulation_agent: Optional[ScenarioSimulationAgent] = None,
        financial_coach_agent: Optional[FinancialCoachAgent] = None,
    ):
        self.fi_agent = financial_intelligence_agent or FinancialIntelligenceAgent()
        self.risk_agent = risk_behaviour_agent or RiskBehaviourAgent()
        self.forecast_agent = forecast_goal_agent or ForecastGoalAgent()
        self.scenario_agent = scenario_simulation_agent or ScenarioSimulationAgent()
        self.coach_agent = financial_coach_agent or FinancialCoachAgent()

    def _call_specialist(self, agent: Any, state: FinancialState, operation: str, **kwargs) -> AgentResult:
        """Invokes a specialist agent using execute() or run()."""
        if hasattr(agent, "execute"):
            return agent.execute(state, operation=operation, **kwargs)
        elif hasattr(agent, "run"):
            return agent.run(state, operation=operation, **kwargs)
        else:
            raise AttributeError(f"Agent '{getattr(agent, 'AGENT_NAME', type(agent).__name__)}' has neither execute() nor run() method.")

    def _call_coach(self, specialist_results: Sequence[AgentResult], state: FinancialState, request_id: str) -> AgentResult:
        """Invokes FinancialCoachAgent using run() or execute()."""
        if hasattr(self.coach_agent, "run"):
            return self.coach_agent.run(operation="coach", specialist_results=specialist_results, state=state, request_id=request_id)
        elif hasattr(self.coach_agent, "execute"):
            return self.coach_agent.execute(state_or_twin=state, operation="coach", specialist_results=specialist_results, request_id=request_id)
        else:
            raise AttributeError("FinancialCoachAgent has neither run() nor execute() method.")

    def run(
        self,
        intent: str,
        state: FinancialState,
        request_id: str = "",
        session_context: Optional[SessionContext] = None,
        scenario_name: Optional[str] = None,
        scenario_params: Optional[Dict[str, Any]] = None,
        horizon_months: int = 12,
        goal_input: Optional[Any] = None,
        goals: Optional[Sequence[Any]] = None,
    ) -> AgentResult:
        """
        Primary execution entry point for multi-agent workflow orchestration.
        """
        start_t = time.perf_counter()
        req_id = request_id or f"req-{uuid.uuid4().hex[:8]}"
        route = intent.lower().strip()

        if route not in self.SUPPORTED_ROUTES:
            return AgentResult.create_failure(
                agent_name=self.AGENT_NAME,
                intent=intent.upper(),
                errors=[f"Unsupported orchestration intent: '{intent}'. Supported routes: {sorted(self.SUPPORTED_ROUTES)}"],
            )

        agents_requested: List[str] = []
        agents_completed: List[str] = []
        agents_failed: List[str] = []
        specialist_results: List[AgentResult] = []
        warnings: List[str] = []
        errors: List[str] = []

        try:
            # ── 1. Execute Conditional Specialist Route ───────────────────────

            if route == "health":
                agents_requested.append(self.fi_agent.AGENT_NAME)
                res = self._call_specialist(self.fi_agent, state, operation="health", request_id=req_id)
                self._record_specialist_result(res, self.fi_agent.AGENT_NAME, specialist_results, agents_completed, agents_failed, errors)

            elif route == "risk":
                agents_requested.append(self.risk_agent.AGENT_NAME)
                res = self._call_specialist(self.risk_agent, state, operation="risk", request_id=req_id)
                self._record_specialist_result(res, self.risk_agent.AGENT_NAME, specialist_results, agents_completed, agents_failed, errors)

            elif route == "forecast":
                agents_requested.append(self.forecast_agent.AGENT_NAME)
                res = self._call_specialist(self.forecast_agent, state, operation="savings_forecast", horizon_months=horizon_months, request_id=req_id)
                self._record_specialist_result(res, self.forecast_agent.AGENT_NAME, specialist_results, agents_completed, agents_failed, errors)

            elif route == "goal":
                agents_requested.append(self.forecast_agent.AGENT_NAME)
                op = "multiple_goals" if goals else "goal"
                res = self._call_specialist(self.forecast_agent, state, operation=op, goal_input=goal_input, goals=goals, request_id=req_id)
                self._record_specialist_result(res, self.forecast_agent.AGENT_NAME, specialist_results, agents_completed, agents_failed, errors)

            elif route == "scenario":
                agents_requested.append(self.scenario_agent.AGENT_NAME)
                s_name = scenario_name or "Salary Hike"
                res = self._call_specialist(self.scenario_agent, state, operation="scenario", scenario_name=s_name, params=scenario_params, request_id=req_id)
                self._record_specialist_result(res, self.scenario_agent.AGENT_NAME, specialist_results, agents_completed, agents_failed, errors)

            elif route == "overview":
                agents_requested.extend([self.fi_agent.AGENT_NAME, self.risk_agent.AGENT_NAME])
                res_fi = self._call_specialist(self.fi_agent, state, operation="financial_overview", request_id=req_id)
                self._record_specialist_result(res_fi, self.fi_agent.AGENT_NAME, specialist_results, agents_completed, agents_failed, errors)

                res_risk = self._call_specialist(self.risk_agent, state, operation="risk_overview", request_id=req_id)
                self._record_specialist_result(res_risk, self.risk_agent.AGENT_NAME, specialist_results, agents_completed, agents_failed, errors)

            elif route == "health_diagnostic":
                agents_requested.extend([self.fi_agent.AGENT_NAME, self.risk_agent.AGENT_NAME])
                res_fi = self._call_specialist(self.fi_agent, state, operation="health", request_id=req_id)
                self._record_specialist_result(res_fi, self.fi_agent.AGENT_NAME, specialist_results, agents_completed, agents_failed, errors)

                res_debt = self._call_specialist(self.risk_agent, state, operation="debt", request_id=req_id)
                self._record_specialist_result(res_debt, self.risk_agent.AGENT_NAME, specialist_results, agents_completed, agents_failed, errors)

                res_ef = self._call_specialist(self.risk_agent, state, operation="emergency_fund", request_id=req_id)
                self._record_specialist_result(res_ef, self.risk_agent.AGENT_NAME, specialist_results, agents_completed, agents_failed, errors)

            elif route == "scenario_forecast":
                agents_requested.extend([self.scenario_agent.AGENT_NAME, self.forecast_agent.AGENT_NAME])
                # Step A: Execute Scenario Simulation
                s_name = scenario_name or "Salary Hike"
                res_sim = self._call_specialist(self.scenario_agent, state, operation="scenario", scenario_name=s_name, params=scenario_params, request_id=req_id)
                self._record_specialist_result(res_sim, self.scenario_agent.AGENT_NAME, specialist_results, agents_completed, agents_failed, errors)

                # Step B: Dependent Forecast execution
                if res_sim.status == AgentStatus.FAILED:
                    warnings.append("Dependent forecast skipped because scenario simulation failed.")
                    agents_failed.append(self.forecast_agent.AGENT_NAME)
                else:
                    scenario_metrics = res_sim.authoritative_data.metrics if res_sim.authoritative_data else {}
                    after_surplus = scenario_metrics.get("after_monthly_surplus")
                    res_fg = self._call_specialist(
                        self.forecast_agent,
                        state,
                        operation="savings_forecast",
                        horizon_months=horizon_months,
                        monthly_surplus_override=after_surplus,
                        request_id=req_id,
                    )
                    self._record_specialist_result(res_fg, self.forecast_agent.AGENT_NAME, specialist_results, agents_completed, agents_failed, errors)

            elif route == "comprehensive":
                agents_requested.extend([self.fi_agent.AGENT_NAME, self.risk_agent.AGENT_NAME, self.forecast_agent.AGENT_NAME])
                res_fi = self._call_specialist(self.fi_agent, state, operation="financial_overview", request_id=req_id)
                self._record_specialist_result(res_fi, self.fi_agent.AGENT_NAME, specialist_results, agents_completed, agents_failed, errors)

                res_risk = self._call_specialist(self.risk_agent, state, operation="risk_overview", request_id=req_id)
                self._record_specialist_result(res_risk, self.risk_agent.AGENT_NAME, specialist_results, agents_completed, agents_failed, errors)

                res_fg = self._call_specialist(self.forecast_agent, state, operation="forecast_overview", horizon_months=horizon_months, request_id=req_id)
                self._record_specialist_result(res_fg, self.forecast_agent.AGENT_NAME, specialist_results, agents_completed, agents_failed, errors)

                if scenario_name or scenario_params:
                    agents_requested.append(self.scenario_agent.AGENT_NAME)
                    res_sim = self._call_specialist(self.scenario_agent, state, operation="scenario", scenario_name=scenario_name, params=scenario_params, request_id=req_id)
                    self._record_specialist_result(res_sim, self.scenario_agent.AGENT_NAME, specialist_results, agents_completed, agents_failed, errors)

            # ── 2. Handoff Specialist Findings to FinancialCoachAgent ───────────

            agents_requested.append(self.coach_agent.AGENT_NAME)
            coach_result: Optional[AgentResult] = None
            try:
                coach_result = self._call_coach(
                    specialist_results=specialist_results,
                    state=state,
                    request_id=req_id,
                )
                if coach_result.status == AgentStatus.FAILED:
                    agents_failed.append(self.coach_agent.AGENT_NAME)
                    warnings.append(f"FinancialCoachAgent failed: {'; '.join(coach_result.errors)}")
                else:
                    agents_completed.append(self.coach_agent.AGENT_NAME)
            except Exception as e:
                agents_failed.append(self.coach_agent.AGENT_NAME)
                warnings.append(f"FinancialCoachAgent execution error: {str(e)}")

            # ── 3. Assemble Final Orchestrator Result ──────────────────────────

            exec_ms = (time.perf_counter() - start_t) * 1000
            return self._build_orchestrator_result(
                route=route,
                specialist_results=specialist_results,
                coach_result=coach_result,
                agents_requested=agents_requested,
                agents_completed=agents_completed,
                agents_failed=agents_failed,
                warnings=warnings,
                errors=errors,
                exec_ms=exec_ms,
                req_id=req_id,
            )

        except Exception as e:
            exec_ms = (time.perf_counter() - start_t) * 1000
            return AgentResult.create_failure(
                agent_name=self.AGENT_NAME,
                intent=route.upper(),
                errors=[f"Orchestration failure in route '{route}': {str(e)}"] + errors,
                warnings=warnings,
            )

    def execute(
        self,
        intent: str,
        state: FinancialState,
        request_id: str = "",
        session_context: Optional[SessionContext] = None,
        scenario_name: Optional[str] = None,
        scenario_params: Optional[Dict[str, Any]] = None,
        horizon_months: int = 12,
        goal_input: Optional[Any] = None,
        goals: Optional[Sequence[Any]] = None,
    ) -> AgentResult:
        """Standard standardized `.execute()` alias for run()."""
        return self.run(
            intent=intent,
            state=state,
            request_id=request_id,
            session_context=session_context,
            scenario_name=scenario_name,
            scenario_params=scenario_params,
            horizon_months=horizon_months,
            goal_input=goal_input,
            goals=goals,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _record_specialist_result(
        self,
        result: AgentResult,
        agent_name: str,
        results_list: List[AgentResult],
        completed_list: List[str],
        failed_list: List[str],
        errors_list: List[str],
    ) -> None:
        """Records the outcome of an individual specialist execution."""
        results_list.append(result)
        if result.status == AgentStatus.FAILED:
            failed_list.append(agent_name)
            errors_list.extend(result.errors)
        else:
            completed_list.append(agent_name)

    def _build_orchestrator_result(
        self,
        route: str,
        specialist_results: List[AgentResult],
        coach_result: Optional[AgentResult],
        agents_requested: List[str],
        agents_completed: List[str],
        agents_failed: List[str],
        warnings: List[str],
        errors: List[str],
        exec_ms: float,
        req_id: str,
    ) -> AgentResult:
        """Builds the canonical AgentResult container representing the orchestration workflow."""
        # Aggregate upstream provenance
        upstream_provs: List[Dict[str, Any]] = []
        for r in specialist_results:
            if r.authoritative_data:
                upstream_provs.append(r.authoritative_data.provenance.to_dict())

        # Collect combined warnings
        combined_warnings = list(warnings)
        for r in specialist_results:
            combined_warnings.extend(r.warnings)
        if coach_result:
            combined_warnings.extend(coach_result.warnings)

        # Determine overall status
        if agents_failed:
            if agents_completed:
                status = AgentStatus.PARTIAL
            else:
                status = AgentStatus.FAILED
        else:
            status = AgentStatus.SUCCESS

        # Metrics summary
        metrics = {
            "route": route,
            "request_id": req_id,
            "agents_requested": tuple(agents_requested),
            "agents_completed": tuple(agents_completed),
            "agents_failed": tuple(agents_failed),
            "specialist_count": len(specialist_results),
            "total_execution_time_ms": round(exec_ms, 2),
            "coach_status": coach_result.status.value if coach_result else "skipped",
            "upstream_provenance": tuple(upstream_provs),
        }

        prov = AuthoritativeProvenance(
            source_engine="OrchestratorAgent",
            source_tool="orchestrate",
            calculation_version="v1.0.0-orchestrator",
            request_id=req_id,
            execution_time_ms=exec_ms,
        )

        auth_data = AuthoritativeData(
            provenance=prov,
            computation_type=ComputationType.HEURISTIC,
            metrics=metrics,
        )

        # Primary insights come from CoachAgent, or fallback to specialist insights
        if coach_result and coach_result.insights:
            insights = coach_result.insights
        else:
            raw_insights = []
            for r in specialist_results:
                raw_insights.extend(r.insights)
            insights = tuple(raw_insights) or (f"Orchestration completed route '{route}'.",)

        return AgentResult(
            agent_name=self.AGENT_NAME,
            status=status,
            intent=route.upper(),
            authoritative_data=auth_data,
            insights=tuple(insights),
            warnings=tuple(combined_warnings),
            errors=tuple(errors),
        )


# Global default orchestrator instance
DEFAULT_ORCHESTRATOR = OrchestratorAgent()

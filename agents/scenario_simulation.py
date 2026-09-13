"""
agents/scenario_simulation.py
=============================
Specialist agent responsible for life-event financial scenario simulations,
before-vs-after delta impact analysis, and grounded scenario comparisons.

Architectural Guarantees:
- Zero duplicated math: all operations delegate exclusively to the Authoritative Tool Registry.
- Strictly read-only execution: no mutations to FinancialState, database, or configuration.
- Strictly deterministic reasoning: zero LLM calls, prompts, or external network requests.
- Returns standardized AgentResult with preserved AuthoritativeProvenance.
- Health score results are preserved directly from ScenarioSimulator without secondary calculation.
"""

from typing import Optional, Any, Dict, List, Union, Mapping
import uuid
import time

from agents.state import FinancialState, SessionContext
from agents.schemas import (
    AgentResult,
    AgentStatus,
    AuthoritativeData,
    AuthoritativeProvenance,
    ComputationType,
)
from agents.tool_registry import ToolRegistry, DEFAULT_TOOL_REGISTRY
from models.twin_engine import FinancialDigitalTwin


class ScenarioSimulationAgent:
    """
    Authoritative Specialist Agent for Scenario Simulation & Impact Comparison.
    """
    AGENT_NAME = "ScenarioSimulationAgent"

    SUPPORTED_OPERATIONS = {
        "scenario",
        "compare",
    }

    def __init__(self, registry: Optional[ToolRegistry] = None):
        self.registry = registry or DEFAULT_TOOL_REGISTRY
        # Cache authorized tools for this agent role
        self.authorized_tools = self.registry.get_tools_for_agent("scenario_simulation")

    def _ensure_twin(self, state_or_twin: Union[FinancialState, Any]) -> Any:
        """
        Converts an immutable FinancialState snapshot to a read-only FinancialDigitalTwin instance
        for tool execution, or passes through an existing FinancialDigitalTwin unchanged.
        """
        if isinstance(state_or_twin, FinancialState):
            state = state_or_twin
            demographics = {
                "name": state.profile.username,
                "age": state.profile.age,
                "occupation": state.profile.occupation,
                "city": state.profile.city,
                "monthly_income": state.cash_flow.monthly_income,
                "bonus": state.cash_flow.bonus,
                "additional_income": state.cash_flow.additional_income,
            }
            balance_sheet = {
                "bank_savings": state.balance_sheet.bank_savings,
                "fd_amount": state.balance_sheet.fixed_deposits,
                "emergency_fund": state.balance_sheet.emergency_fund,
                "mutual_funds": state.balance_sheet.mutual_funds,
                "stocks": state.balance_sheet.stocks,
                "ppf_investment": state.balance_sheet.ppf_investment,
                "nps_investment": state.balance_sheet.nps_investment,
                "sip_amount": state.balance_sheet.sip_amount,
                "loan_amount": state.balance_sheet.loan_amount,
                "car_loan": state.balance_sheet.car_loan,
                "home_loan": state.balance_sheet.home_loan,
                "credit_card_debt": state.balance_sheet.credit_card_debt,
                "monthly_emi": state.cash_flow.monthly_emi,
                "health_insurance": state.balance_sheet.health_insurance,
                "life_insurance": state.balance_sheet.life_insurance,
                "rent": state.cash_flow.non_discretionary_expenses,
                "net_worth": state.balance_sheet.current_net_worth,
            }
            twin = FinancialDigitalTwin(
                user_id=state.profile.user_id,
                demographics=demographics,
                balance_sheet=balance_sheet,
            )
            twin.basic_expenses = state.cash_flow.monthly_expenses
            twin.total_income = state.cash_flow.monthly_income + state.cash_flow.additional_income
            return twin
        return state_or_twin

    def execute(
        self,
        state_or_twin: Union[FinancialState, Any],
        operation: str = "scenario",
        scenario_name: str = "Salary Hike",
        params: Optional[Dict[str, Any]] = None,
        request_id: str = "",
        session_context: Optional[SessionContext] = None,
    ) -> AgentResult:
        """
        Executes a targeted scenario simulation or comparison operation.
        """
        req_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

        if operation not in self.SUPPORTED_OPERATIONS:
            return AgentResult.create_failure(
                agent_name=self.AGENT_NAME,
                intent=operation,
                errors=[f"Unsupported operation: '{operation}'. Supported operations: {sorted(self.SUPPORTED_OPERATIONS)}"],
            )

        twin = self._ensure_twin(state_or_twin)

        try:
            if operation == "scenario":
                return self._handle_scenario(twin, scenario_name, params, req_id)
            elif operation == "compare":
                return self._handle_compare(twin, scenario_name, params, req_id)
            else:
                return AgentResult.create_failure(
                    agent_name=self.AGENT_NAME,
                    intent=operation,
                    errors=[f"Unhandled operation '{operation}'."],
                )
        except Exception as e:
            return AgentResult.create_failure(
                agent_name=self.AGENT_NAME,
                intent=operation,
                errors=[f"Tool execution failed during '{operation}': {str(e)}"],
            )

    def run(
        self,
        state_or_twin: Union[FinancialState, Any],
        operation: str = "scenario",
        scenario_name: str = "Salary Hike",
        params: Optional[Dict[str, Any]] = None,
        request_id: str = "",
        session_context: Optional[SessionContext] = None,
    ) -> AgentResult:
        """Standardized `.run()` alias for execute()."""
        return self.execute(
            state_or_twin=state_or_twin,
            operation=operation,
            scenario_name=scenario_name,
            params=params,
            request_id=request_id,
            session_context=session_context,
        )

    # ── Operation Handlers ────────────────────────────────────────────────────

    def _handle_scenario(
        self,
        twin: Any,
        scenario_name: str,
        params: Optional[Dict[str, Any]],
        req_id: str,
    ) -> AgentResult:
        """Executes scenario simulation via run_scenario tool."""
        auth_data = self.registry.execute(
            "run_scenario",
            twin=twin,
            scenario_name=scenario_name,
            params=params,
            request_id=req_id,
        )
        return self._format_scenario_result(auth_data, intent="SCENARIO_SIMULATION")

    def _handle_compare(
        self,
        twin: Any,
        scenario_name: str,
        params: Optional[Dict[str, Any]],
        req_id: str,
    ) -> AgentResult:
        """Executes scenario comparison via compare_scenario tool."""
        auth_data = self.registry.execute(
            "compare_scenario",
            twin=twin,
            scenario_name=scenario_name,
            params=params,
            request_id=req_id,
        )
        return self._format_scenario_result(auth_data, intent="SCENARIO_COMPARISON")

    def _format_scenario_result(self, auth_data: AuthoritativeData, intent: str) -> AgentResult:
        """Extracts metrics and formats grounded insights/warnings without duplicating calculations."""
        metrics = auth_data.metrics

        s_name = metrics.get("scenario_name", "Scenario")
        b_score = metrics.get("before_health_score", 0.0)
        a_score = metrics.get("after_health_score", 0.0)
        d_score = metrics.get("health_score_delta", 0.0)
        b_surplus = metrics.get("before_monthly_surplus", 0.0)
        a_surplus = metrics.get("after_monthly_surplus", 0.0)
        d_surplus = metrics.get("surplus_delta", 0.0)
        nw_delta_5y = metrics.get("net_worth_delta_5y", 0.0)
        b_risk = metrics.get("before_risk_level", "Moderate")
        a_risk = metrics.get("after_risk_level", "Moderate")

        score_sign = "+" if d_score >= 0 else ""
        surplus_sign = "+" if d_surplus >= 0 else ""
        nw_sign = "+" if nw_delta_5y >= 0 else ""

        insights = [
            f"Scenario '{s_name}' Simulation:",
            f"• Health Score: {b_score:.1f} → {a_score:.1f} ({score_sign}{d_score:.1f} pts).",
            f"• Monthly Surplus: ₹{b_surplus:,.0f} → ₹{a_surplus:,.0f} ({surplus_sign}₹{d_surplus:,.0f}/mo).",
            f"• 5-Year Projected Net Worth Impact: {nw_sign}₹{nw_delta_5y:,.0f}.",
            f"• Risk Tier: {b_risk} → {a_risk}.",
        ]

        warnings = []
        if d_score < -5.0:
            warnings.append(f"Significant health score reduction of {d_score:.1f} points under this scenario.")
        if a_surplus <= 0 and b_surplus > 0:
            warnings.append(f"Cash flow deficit: Monthly surplus drops to zero under '{s_name}'.")
        if a_risk in ("High", "Critical") and b_risk not in ("High", "Critical"):
            warnings.append(f"Risk escalation: Risk level shifts from '{b_risk}' to '{a_risk}'.")
        if nw_delta_5y < 0:
            warnings.append(f"Negative 5-year wealth impact: Net worth reduced by ₹{abs(nw_delta_5y):,.0f}.")

        return AgentResult.create_success(
            agent_name=self.AGENT_NAME,
            intent=intent,
            authoritative_data=auth_data,
            insights=insights,
            warnings=warnings,
        )

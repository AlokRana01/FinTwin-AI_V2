"""
agents/forecast_goal.py
=======================
Specialist agent responsible for future savings projections, net-worth trajectories,
single-goal analysis, multi-goal prioritization, and forecast XAI explanations.

Architectural Guarantees:
- Zero duplicated math: all operations delegate exclusively to the Authoritative Tool Registry.
- Strictly read-only execution: no mutations to FinancialState, database, or configuration.
- Strictly deterministic reasoning: zero LLM calls or prompt templates.
- Returns standardized AgentResult with preserved AuthoritativeProvenance.
"""

from typing import Optional, Any, Dict, List, Union, Mapping, Sequence
import uuid
import time
from datetime import date

from agents.state import FinancialState, GoalItemState, SessionContext
from agents.schemas import (
    AgentResult,
    AgentStatus,
    AuthoritativeData,
    AuthoritativeProvenance,
    ComputationType,
)
from agents.tool_registry import ToolRegistry, DEFAULT_TOOL_REGISTRY
from models.twin_engine import FinancialDigitalTwin
from utils.goal_engine import GoalInput, GoalType


class ForecastGoalAgent:
    """
    Authoritative Specialist Agent for Forecast & Goal Planning.
    """
    AGENT_NAME = "ForecastGoalAgent"

    SUPPORTED_OPERATIONS = {
        "savings_forecast",
        "net_worth_forecast",
        "goal",
        "multiple_goals",
        "forecast_explanation",
        "forecast_overview",
    }

    def __init__(self, registry: Optional[ToolRegistry] = None):
        self.registry = registry or DEFAULT_TOOL_REGISTRY
        # Cache authorized tools for this agent role
        self.authorized_tools = self.registry.get_tools_for_agent("forecast_goal")

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

    def _convert_goal_item_state(self, item: GoalItemState) -> GoalInput:
        """Helper converting GoalItemState to GoalInput for GoalEngine execution."""
        goal_type = GoalType.HOUSE
        for gt in GoalType:
            if gt.value.lower() in item.goal_type.lower() or item.goal_type.lower() in gt.value.lower():
                goal_type = gt
                break

        # Approximate target date if string
        today = date.today()
        target_d = today.replace(year=today.year + 3)
        if item.target_date:
            try:
                parts = [int(p) for p in item.target_date.split("-")]
                if len(parts) >= 2:
                    target_d = date(parts[0], parts[1], 1)
                elif len(parts) == 1:
                    target_d = date(parts[0], 1, 1)
            except Exception:
                pass

        return GoalInput(
            goal_type=goal_type,
            goal_amount=max(item.target_amount, 1000.0),
            current_savings=max(item.current_savings, 0.0),
            target_date=target_d if target_d > today else today.replace(year=today.year + 1),
            monthly_contribution=max(item.monthly_contribution, 0.0),
        )

    def execute(
        self,
        state_or_twin: Union[FinancialState, Any],
        operation: str = "savings_forecast",
        horizon_months: int = 12,
        goal_input: Optional[GoalInput] = None,
        goals: Optional[Sequence[GoalInput]] = None,
        monthly_surplus_override: Optional[float] = None,
        priority_order: Optional[Sequence[Any]] = None,
        request_id: str = "",
        session_context: Optional[SessionContext] = None,
    ) -> AgentResult:
        """
        Executes a targeted forecast or goal planning operation.
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
            if operation == "savings_forecast":
                return self._handle_savings_forecast(twin, horizon_months, req_id)
            elif operation == "net_worth_forecast":
                return self._handle_net_worth_forecast(twin, horizon_months, req_id)
            elif operation == "goal":
                return self._handle_goal(state_or_twin, twin, goal_input, monthly_surplus_override, req_id)
            elif operation == "multiple_goals":
                return self._handle_multiple_goals(state_or_twin, twin, goals, priority_order, req_id)
            elif operation == "forecast_explanation":
                return self._handle_forecast_explanation(twin, req_id)
            elif operation == "forecast_overview":
                return self._handle_forecast_overview(twin, horizon_months, req_id)
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

    # ── Operation Handlers ────────────────────────────────────────────────────

    def _handle_savings_forecast(self, twin: Any, horizon_months: int, req_id: str) -> AgentResult:
        """Executes savings prediction via FinancialPredictor.predict."""
        auth_data = self.registry.execute(
            "forecast_savings",
            twin=twin,
            horizon_months=horizon_months,
            request_id=req_id,
        )
        metrics = auth_data.metrics

        pred_sav = metrics.get("predicted_savings", 0.0)
        pred_nw = metrics.get("predicted_net_worth", 0.0)
        h = metrics.get("horizon_months", horizon_months)

        insights = [
            f"At a {h}-month horizon, projected cumulative savings are ₹{pred_sav:,.0f}.",
            f"Projected net worth at {h} months reaches ₹{pred_nw:,.0f}.",
        ]

        warnings = []
        if pred_sav <= 0:
            warnings.append("Projected savings are stagnant or zero due to tight monthly cash flow.")

        return AgentResult.create_success(
            agent_name=self.AGENT_NAME,
            intent="SAVINGS_FORECAST",
            authoritative_data=auth_data,
            insights=insights,
            warnings=warnings,
        )

    def _handle_net_worth_forecast(self, twin: Any, months: int, req_id: str) -> AgentResult:
        """Executes month-by-month trajectory forecasting."""
        auth_data = self.registry.execute(
            "forecast_net_worth",
            twin=twin,
            months=months,
            request_id=req_id,
        )
        metrics = auth_data.metrics
        arrays = auth_data.arrays or {}

        final_nw = metrics.get("final_projected_net_worth", 0.0)
        final_sav = metrics.get("final_projected_savings", 0.0)
        m_count = metrics.get("horizon_months", months)

        insights = [
            f"Forward {m_count}-month wealth trajectory projects net worth reaching ₹{final_nw:,.0f}.",
            f"Projected monthly savings rate accumulation reaches ₹{final_sav:,.0f}/month.",
        ]

        return AgentResult.create_success(
            agent_name=self.AGENT_NAME,
            intent="NET_WORTH_FORECAST",
            authoritative_data=auth_data,
            insights=insights,
        )

    def _handle_goal(
        self,
        state_or_twin: Union[FinancialState, Any],
        twin: Any,
        goal_input: Optional[GoalInput],
        monthly_surplus_override: Optional[float],
        req_id: str,
    ) -> AgentResult:
        """Executes single-goal feasibility and required SIP analysis."""
        gi = goal_input
        if gi is None:
            if isinstance(state_or_twin, FinancialState) and state_or_twin.goals:
                gi = self._convert_goal_item_state(state_or_twin.goals[0])
            else:
                today = date.today()
                gi = GoalInput(
                    goal_type=GoalType.CAR,
                    goal_amount=800000.0,
                    current_savings=100000.0,
                    target_date=today.replace(year=today.year + 3),
                    monthly_contribution=15000.0,
                )

        if monthly_surplus_override is not None:
            surplus = float(monthly_surplus_override)
        else:
            surplus = max(0.0, float(twin.total_income - twin.basic_expenses - twin.monthly_emi))

        auth_data = self.registry.execute(
            "analyze_goal",
            goal_input=gi,
            monthly_surplus=surplus,
            request_id=req_id,
        )
        metrics = auth_data.metrics

        name = metrics.get("goal_name", "Goal")
        target = metrics.get("target_amount", 0.0)
        sip_req = metrics.get("sip_required", 0.0)
        prob = metrics.get("probability", 0.0)
        prob_lbl = metrics.get("probability_label", "Moderate")
        is_funded = metrics.get("is_already_funded", False)
        shortfall = metrics.get("shortfall_months", 0)
        proj_comp = metrics.get("projected_completion", "")

        insights = [
            f"Goal '{name}' (Target: ₹{target:,.0f}): Required monthly SIP is ₹{sip_req:,.0f}.",
            f"Achievement Probability is {prob:.0%} ({prob_lbl}).",
        ]
        if proj_comp:
            insights.append(f"Estimated completion date: {proj_comp}.")

        warnings = []
        if is_funded:
            insights.append("Goal is already fully funded by existing earmarked savings.")
        elif shortfall > 0:
            warnings.append(f"Timeline shortfall: At current savings rate, goal is projected to be delayed by {shortfall} months.")
        if prob < 0.50 and not is_funded:
            warnings.append(f"Low achievement probability ({prob:.0%}): Available monthly surplus (₹{surplus:,.0f}) is below required SIP (₹{sip_req:,.0f}).")

        return AgentResult.create_success(
            agent_name=self.AGENT_NAME,
            intent="GOAL_ANALYSIS",
            authoritative_data=auth_data,
            insights=insights,
            warnings=warnings,
        )

    def _handle_multiple_goals(
        self,
        state_or_twin: Union[FinancialState, Any],
        twin: Any,
        goals: Optional[Sequence[GoalInput]],
        priority_order: Optional[Sequence[Any]],
        req_id: str,
    ) -> AgentResult:
        """Executes multi-goal simultaneous surplus planning."""
        goal_list = list(goals) if goals else []
        if not goal_list:
            if isinstance(state_or_twin, FinancialState) and state_or_twin.goals:
                goal_list = [self._convert_goal_item_state(g) for g in state_or_twin.goals]
            else:
                today = date.today()
                goal_list = [
                    GoalInput(
                        goal_type=GoalType.CAR,
                        goal_amount=600000.0,
                        current_savings=50000.0,
                        target_date=today.replace(year=today.year + 2),
                    ),
                    GoalInput(
                        goal_type=GoalType.HOUSE,
                        goal_amount=3000000.0,
                        current_savings=200000.0,
                        target_date=today.replace(year=today.year + 5),
                    ),
                ]

        auth_data = self.registry.execute(
            "plan_multiple_goals",
            twin=twin,
            goals=goal_list,
            priority_order=priority_order,
            request_id=req_id,
        )
        metrics = auth_data.metrics

        surplus = metrics.get("monthly_surplus", 0.0)
        g_count = metrics.get("goals_count", len(goal_list))
        g_summaries = metrics.get("goals", ())

        insights = [
            f"Multi-Goal Portfolio: Planning across {g_count} active goals with monthly surplus of ₹{surplus:,.0f}.",
        ]

        warnings = []
        for g in g_summaries:
            if isinstance(g, dict):
                g_name = g.get("goal_name", "")
                g_sip = g.get("sip_required", 0.0)
                g_prob = g.get("probability", 0.0)
                g_prob_lbl = g.get("probability_label", "")
                insights.append(f"• {g_name}: Required SIP ₹{g_sip:,.0f}/mo | Probability: {g_prob:.0%} ({g_prob_lbl})")
                if g_prob < 0.50:
                    warnings.append(f"Goal '{g_name}' has low funding feasibility ({g_prob:.0%}) under current surplus distribution.")

        return AgentResult.create_success(
            agent_name=self.AGENT_NAME,
            intent="MULTIPLE_GOALS_PLANNING",
            authoritative_data=auth_data,
            insights=insights,
            warnings=warnings,
        )

    def _handle_forecast_explanation(self, twin: Any, req_id: str) -> AgentResult:
        """Executes TreeSHAP forecast feature attribution explainability."""
        auth_data = self.registry.execute(
            "explain_forecast",
            twin=twin,
            request_id=req_id,
        )
        metrics = auth_data.metrics
        attributions = auth_data.xai_attributions or {}

        pred_sav = metrics.get("predicted_savings", 0.0)
        base_val = metrics.get("base_value_savings", 0.0)
        method = metrics.get("method", "treeshap")

        insights = [
            f"Forecast Explanation ({method}): Predicted savings ₹{pred_sav:,.0f} vs base value ₹{base_val:,.0f}.",
        ]

        if attributions:
            sorted_attribs = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)
            top_feature = sorted_attribs[0]
            insights.append(f"Top forecast driver: '{top_feature[0]}' with impact of {top_feature[1]:+.2f} on model output.")

        return AgentResult.create_success(
            agent_name=self.AGENT_NAME,
            intent="FORECAST_EXPLANATION",
            authoritative_data=auth_data,
            insights=insights,
        )

    def _handle_forecast_overview(self, twin: Any, months: int, req_id: str) -> AgentResult:
        """Executes savings and net worth forecasts into a concise forecast overview."""
        sav_data = self.registry.execute("forecast_savings", twin=twin, horizon_months=months, request_id=req_id)
        nw_data = self.registry.execute("forecast_net_worth", twin=twin, months=months, request_id=req_id)

        merged_metrics = {
            "horizon_months": months,
            "predicted_savings": sav_data.metrics.get("predicted_savings", 0.0),
            "predicted_net_worth": sav_data.metrics.get("predicted_net_worth", 0.0),
            "final_projected_net_worth": nw_data.metrics.get("final_projected_net_worth", 0.0),
            "final_projected_savings": nw_data.metrics.get("final_projected_savings", 0.0),
        }

        combined_auth_data = AuthoritativeData(
            provenance=AuthoritativeProvenance(
                source_engine="ForecastComposite",
                source_tool="forecast_overview",
                calculation_version="v1.0.0-composite",
                request_id=req_id,
            ),
            computation_type=ComputationType.MODEL_OUTPUT,
            metrics=merged_metrics,
            arrays=nw_data.arrays,
        )

        insights = [
            f"{months}-Month Outlook: Projected cumulative savings ₹{merged_metrics['predicted_savings']:,.0f}.",
            f"Final Projected Net Worth reaches ₹{merged_metrics['final_projected_net_worth']:,.0f}.",
        ]

        return AgentResult.create_success(
            agent_name=self.AGENT_NAME,
            intent="FORECAST_OVERVIEW",
            authoritative_data=combined_auth_data,
            insights=insights,
        )

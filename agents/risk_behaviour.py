"""
agents/risk_behaviour.py
========================
Specialist agent responsible for debt liability analysis, emergency fund adequacy,
spending behavior diagnostics, and multi-factor financial risk assessment.

Architectural Guarantees:
- Zero duplicated math: all operations delegate exclusively to the Authoritative Tool Registry.
- Strictly read-only execution: no mutations to FinancialState, database, or configuration.
- Strictly deterministic reasoning: zero LLM calls or prompt templates.
- Returns standardized AgentResult with preserved AuthoritativeProvenance.
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


class RiskBehaviourAgent:
    """
    Authoritative Specialist Agent for Risk & Behaviour Analysis.
    """
    AGENT_NAME = "RiskBehaviourAgent"

    SUPPORTED_OPERATIONS = {
        "debt",
        "emergency_fund",
        "spending_behavior",
        "risk",
        "risk_overview",
    }

    def __init__(self, registry: Optional[ToolRegistry] = None):
        self.registry = registry or DEFAULT_TOOL_REGISTRY
        # Cache authorized tools for this agent role
        self.authorized_tools = self.registry.get_tools_for_agent("risk_behaviour")

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
        operation: str = "risk",
        request_id: str = "",
        session_context: Optional[SessionContext] = None,
    ) -> AgentResult:
        """
        Executes a targeted risk or behavioral analysis operation.
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
            if operation == "debt":
                return self._handle_debt(twin, req_id)
            elif operation == "emergency_fund":
                return self._handle_emergency_fund(twin, req_id)
            elif operation == "spending_behavior":
                return self._handle_spending_behavior(twin, req_id)
            elif operation == "risk":
                return self._handle_risk(twin, req_id)
            elif operation == "risk_overview":
                return self._handle_risk_overview(twin, req_id)
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

    def _handle_debt(self, twin: Any, req_id: str) -> AgentResult:
        """Executes debt and EMI burden analysis."""
        auth_data = self.registry.execute("analyze_debt", twin, request_id=req_id)
        metrics = auth_data.metrics

        loan_total = metrics.get("total_loan_amount", 0.0)
        emi = metrics.get("monthly_emi", 0.0)
        emi_ratio = metrics.get("emi_to_income_ratio", 0.0)
        emi_score = metrics.get("emi_burden_score", 0.0)
        cc_debt = metrics.get("credit_card_debt", 0.0)

        insights = [
            f"Total outstanding debt liability is ₹{loan_total:,.0f} with monthly EMI of ₹{emi:,.0f}.",
            f"EMI-to-Income ratio is {emi_ratio:.1%} (EMI Burden Score: {emi_score:.1f}/100).",
        ]

        warnings = []
        if emi_ratio > 0.40:
            warnings.append(f"High debt exposure: Monthly EMI consumes {emi_ratio:.1%} of income (safe threshold <= 35%).")
        if cc_debt > 0:
            warnings.append(f"High-interest revolving credit card debt of ₹{cc_debt:,.0f} detected.")

        return AgentResult.create_success(
            agent_name=self.AGENT_NAME,
            intent="DEBT_ANALYSIS",
            authoritative_data=auth_data,
            insights=insights,
            warnings=warnings,
        )

    def _handle_emergency_fund(self, twin: Any, req_id: str) -> AgentResult:
        """Executes emergency fund liquidity assessment."""
        auth_data = self.registry.execute("analyze_emergency_fund", twin, request_id=req_id)
        metrics = auth_data.metrics

        ef_amount = metrics.get("emergency_fund_amount", 0.0)
        months_covered = metrics.get("months_covered", 0.0)
        ef_score = metrics.get("emergency_fund_score", 0.0)
        is_adequate = metrics.get("is_adequate", False)

        insights = [
            f"Emergency reserves stand at ₹{ef_amount:,.0f}, covering {months_covered:.1f} months of non-discretionary expenses.",
            f"Emergency Fund Score: {ef_score:.1f}/100.",
        ]

        warnings = []
        if not is_adequate:
            if months_covered < 3.0:
                warnings.append(f"Critical liquidity risk: Emergency fund covers only {months_covered:.1f} months (recommended: 6 months).")
            else:
                warnings.append(f"Moderate liquidity risk: Emergency fund covers {months_covered:.1f} months (target: 6+ months).")

        return AgentResult.create_success(
            agent_name=self.AGENT_NAME,
            intent="EMERGENCY_FUND_ANALYSIS",
            authoritative_data=auth_data,
            insights=insights,
            warnings=warnings,
        )

    def _handle_spending_behavior(self, twin: Any, req_id: str) -> AgentResult:
        """Executes spending behavior diagnosis and active alert retrieval."""
        auth_data = self.registry.execute("analyze_spending_behavior", twin, request_id=req_id)
        metrics = auth_data.metrics

        top_action = metrics.get("top_action", "")
        alerts = metrics.get("alerts", ())
        crit_count = metrics.get("critical_alerts_count", 0)

        insights = []
        if top_action:
            insights.append(f"Primary actionable coaching focus: {top_action}")

        warnings = []
        for a in alerts:
            if isinstance(a, dict):
                sev = a.get("severity", "")
                hl = a.get("headline", "")
                msg = a.get("message", "")
                if sev in ("Critical", "High"):
                    warnings.append(f"[{sev.upper()}] {hl}: {msg}")
                else:
                    insights.append(f"[{sev}] {hl}: {msg}")

        return AgentResult.create_success(
            agent_name=self.AGENT_NAME,
            intent="SPENDING_BEHAVIOR_ANALYSIS",
            authoritative_data=auth_data,
            insights=insights,
            warnings=warnings,
        )

    def _handle_risk(self, twin: Any, req_id: str) -> AgentResult:
        """Executes overall multi-factor financial risk assessment."""
        auth_data = self.registry.execute("assess_financial_risk", twin, request_id=req_id)
        metrics = auth_data.metrics

        risk_level = metrics.get("risk_level", "Moderate")
        sr = metrics.get("savings_rate", 0.0)
        ef_m = metrics.get("emergency_fund_months", 0.0)
        loan = metrics.get("loan_amount", 0.0)

        insights = [
            f"Overall financial risk tier is evaluated as '{risk_level}'.",
            f"Risk profile drivers: Savings Rate = {sr:.1%}, Liquidity Runway = {ef_m:.1f} months, Outstanding Debt = ₹{loan:,.0f}.",
        ]

        warnings = []
        if risk_level in ("High", "Critical"):
            warnings.append(f"Elevated financial vulnerability: User is in the '{risk_level}' risk tier due to debt or low liquidity.")

        return AgentResult.create_success(
            agent_name=self.AGENT_NAME,
            intent="RISK_ASSESSMENT",
            authoritative_data=auth_data,
            insights=insights,
            warnings=warnings,
        )

    def _handle_risk_overview(self, twin: Any, req_id: str) -> AgentResult:
        """Combines all four risk and behavioral tools into a consolidated risk profile."""
        debt_data = self.registry.execute("analyze_debt", twin, request_id=req_id)
        ef_data = self.registry.execute("analyze_emergency_fund", twin, request_id=req_id)
        spend_data = self.registry.execute("analyze_spending_behavior", twin, request_id=req_id)
        risk_data = self.registry.execute("assess_financial_risk", twin, request_id=req_id)

        merged_metrics = {
            "risk_level": risk_data.metrics.get("risk_level", "Moderate"),
            "total_loan_amount": debt_data.metrics.get("total_loan_amount", 0.0),
            "monthly_emi": debt_data.metrics.get("monthly_emi", 0.0),
            "emi_to_income_ratio": debt_data.metrics.get("emi_to_income_ratio", 0.0),
            "emergency_fund_amount": ef_data.metrics.get("emergency_fund_amount", 0.0),
            "emergency_fund_months": ef_data.metrics.get("months_covered", 0.0),
            "active_alerts_count": spend_data.metrics.get("active_alerts_count", 0),
            "top_action": spend_data.metrics.get("top_action", ""),
        }

        combined_auth_data = AuthoritativeData(
            provenance=AuthoritativeProvenance(
                source_engine="RiskBehaviourComposite",
                source_tool="risk_overview",
                calculation_version="v1.0.0-composite",
                request_id=req_id,
            ),
            computation_type=ComputationType.DETERMINISTIC,
            metrics=merged_metrics,
            confidence_score=1.0,
        )

        insights = [
            f"Consolidated Risk Status: '{merged_metrics['risk_level']}'.",
            f"Debt Exposure: Monthly EMI of ₹{merged_metrics['monthly_emi']:,.0f} ({merged_metrics['emi_to_income_ratio']:.1%} of income).",
            f"Liquidity Buffer: ₹{merged_metrics['emergency_fund_amount']:,.0f} ({merged_metrics['emergency_fund_months']:.1f} months covered).",
        ]
        if merged_metrics["top_action"]:
            insights.append(f"Recommended Risk Mitigation: {merged_metrics['top_action']}")

        warnings = []
        if merged_metrics["emi_to_income_ratio"] > 0.40:
            warnings.append("High debt burden detected (> 40% EMI ratio).")
        if merged_metrics["emergency_fund_months"] < 3.0:
            warnings.append("Inadequate emergency liquidity (< 3 months reserve).")

        return AgentResult.create_success(
            agent_name=self.AGENT_NAME,
            intent="RISK_OVERVIEW",
            authoritative_data=combined_auth_data,
            insights=insights,
            warnings=warnings,
        )

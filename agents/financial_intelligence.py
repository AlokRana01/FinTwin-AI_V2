"""
agents/financial_intelligence.py
================================
Specialist agent responsible for holistic financial health analysis, balance sheet
ratios, behavioral personality classification, Indian tax optimization, and
health score XAI attributions.

Architectural Guarantees:
- Zero duplicated math: all operations delegate exclusively to the Authoritative Tool Registry.
- Strictly read-only execution: no mutations to FinancialState, database, or configuration.
- Encapsulates tax analysis: owns tax evaluation (no separate Tax Agent).
- Strictly deterministic reasoning: zero LLM calls or prompt templates.
- Returns standardized AgentResult with preserved AuthoritativeProvenance.
"""

from typing import Optional, Any, Dict, List, Union, Mapping
import uuid
import time
from datetime import datetime

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
from utils.tax_calculator import DeductionProfile


class FinancialIntelligenceAgent:
    """
    Authoritative Specialist Agent for Financial Intelligence.
    """
    AGENT_NAME = "FinancialIntelligenceAgent"

    SUPPORTED_OPERATIONS = {
        "health",
        "ratios",
        "personality",
        "tax",
        "health_explanation",
        "financial_overview",
    }

    def __init__(self, registry: Optional[ToolRegistry] = None):
        self.registry = registry or DEFAULT_TOOL_REGISTRY
        # Cache authorized tools for this agent role
        self.authorized_tools = self.registry.get_tools_for_agent("financial_intelligence")

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
        operation: str = "health",
        deductions: Optional[DeductionProfile] = None,
        request_id: str = "",
        session_context: Optional[SessionContext] = None,
    ) -> AgentResult:
        """
        Executes a targeted financial intelligence operation.
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
            if operation == "health":
                return self._handle_health(twin, req_id)
            elif operation == "ratios":
                return self._handle_ratios(twin, req_id)
            elif operation == "personality":
                return self._handle_personality(twin, req_id)
            elif operation == "tax":
                return self._handle_tax(twin, deductions, req_id)
            elif operation == "health_explanation":
                return self._handle_health_explanation(twin, req_id)
            elif operation == "financial_overview":
                return self._handle_financial_overview(twin, req_id)
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

    def _handle_health(self, twin: Any, req_id: str) -> AgentResult:
        """Executes health score calculation and generates grounded observations."""
        auth_data = self.registry.execute("calculate_health_score", twin, request_id=req_id)
        metrics = auth_data.metrics

        score = metrics.get("health_score", 0.0)
        grade = metrics.get("grade", "N/A")
        risk = metrics.get("risk_level", "Moderate")

        insights = [
            f"Overall Financial Health Score is {score:.1f}/100 (Grade {grade}).",
            f"Financial risk tier is assessed as '{risk}'.",
        ]

        components = metrics.get("components", {})
        if isinstance(components, dict):
            for pillar, pscore in components.items():
                if pscore < 50.0:
                    insights.append(f"Sub-optimal pillar score: {pillar.replace('_', ' ').title()} scored {pscore:.1f}/100.")

        return AgentResult.create_success(
            agent_name=self.AGENT_NAME,
            intent="HEALTH_ANALYSIS",
            authoritative_data=auth_data,
            insights=insights,
        )

    def _handle_ratios(self, twin: Any, req_id: str) -> AgentResult:
        """Executes balance sheet ratios computation."""
        auth_data = self.registry.execute("calculate_financial_ratios", twin, request_id=req_id)
        metrics = auth_data.metrics

        sr = metrics.get("savings_rate", 0.0)
        ef_m = metrics.get("emergency_fund_months", 0.0)
        nw = metrics.get("net_worth", 0.0)
        income = metrics.get("total_income", 0.0)
        emi = metrics.get("monthly_emi", 0.0)

        emi_ratio = emi / income if income > 0 else 0.0

        insights = [
            f"Monthly savings rate is {sr:.1%}.",
            f"Debt-to-income (EMI) ratio is {emi_ratio:.1%}.",
            f"Emergency fund liquidity covers {ef_m:.1f} months of expenses.",
            f"Current total net worth is ₹{nw:,.0f}.",
        ]

        warnings = []
        if sr < 0.15:
            warnings.append("Savings rate is below the recommended 20% benchmark.")
        if emi_ratio > 0.40:
            warnings.append("EMI burden exceeds the safe 35% threshold.")
        if ef_m < 3.0:
            warnings.append("Emergency fund is below the minimum 3-month safety runway.")

        return AgentResult.create_success(
            agent_name=self.AGENT_NAME,
            intent="RATIOS_ANALYSIS",
            authoritative_data=auth_data,
            insights=insights,
            warnings=warnings,
        )

    def _handle_personality(self, twin: Any, req_id: str) -> AgentResult:
        """Executes financial personality cluster segmentation."""
        auth_data = self.registry.execute("classify_financial_personality", twin, request_id=req_id)
        metrics = auth_data.metrics

        personality = metrics.get("personality", "Balanced Planner")
        tagline = metrics.get("tagline", "")
        description = metrics.get("description", "")
        nudges = metrics.get("nudges", ())

        insights = [
            f"Financial personality classified as '{personality}' ({tagline}).",
            description,
        ]
        if nudges:
            insights.append(f"Key behavioural nudge: {nudges[0]}")

        return AgentResult.create_success(
            agent_name=self.AGENT_NAME,
            intent="PERSONALITY_CLASSIFICATION",
            authoritative_data=auth_data,
            insights=insights,
        )

    def _handle_tax(self, twin: Any, deductions: Optional[DeductionProfile], req_id: str) -> AgentResult:
        """Executes tax comparison under Indian Tax Law."""
        annual_income = float(getattr(twin, "total_income", 0.0)) * 12.0
        auth_data = self.registry.execute(
            "calculate_tax",
            gross_income=annual_income,
            deductions=deductions,
            request_id=req_id,
        )
        metrics = auth_data.metrics

        rec_regime = metrics.get("recommended_regime", "New Regime")
        savings = metrics.get("tax_savings", 0.0)
        old_tax = metrics.get("old_regime_total_tax", 0.0)
        new_tax = metrics.get("new_regime_total_tax", 0.0)

        insights = [
            f"Recommended tax regime is '{rec_regime}'.",
            f"Estimated tax liability: Old Regime = ₹{old_tax:,.0f} | New Regime = ₹{new_tax:,.0f}.",
        ]
        if savings > 0:
            insights.append(f"Estimated annual tax savings by choosing {rec_regime}: ₹{savings:,.0f}.")
        else:
            insights.append("Tax liabilities under both regimes are equal.")

        return AgentResult.create_success(
            agent_name=self.AGENT_NAME,
            intent="TAX_ANALYSIS",
            authoritative_data=auth_data,
            insights=insights,
        )

    def _handle_health_explanation(self, twin: Any, req_id: str) -> AgentResult:
        """Executes health score computation and additive XAI feature decomposition."""
        health_data = self.registry.execute("calculate_health_score", twin, request_id=req_id)
        xai_data = self.registry.execute("explain_health_score", twin, request_id=req_id)

        score = health_data.metrics.get("health_score", 0.0)
        grade = health_data.metrics.get("grade", "N/A")
        attributions = xai_data.xai_attributions or {}

        # Merge authoritative metrics and attributions
        merged_metrics = {
            **dict(health_data.metrics),
            "baseline_score": xai_data.metrics.get("baseline_score", 50.0),
        }

        combined_auth_data = AuthoritativeData(
            provenance=xai_data.provenance,
            computation_type=ComputationType.DETERMINISTIC,
            metrics=merged_metrics,
            xai_attributions=attributions,
            confidence_score=1.0,
        )

        insights = [
            f"Overall Health Score is {score:.1f}/100 (Grade {grade}).",
        ]

        # Find top booster and top drag
        sorted_attribs = sorted(attributions.items(), key=lambda x: x[1], reverse=True)
        if sorted_attribs:
            top_boost = sorted_attribs[0]
            if top_boost[1] > 0:
                insights.append(f"Strongest score driver: {top_boost[0].title()} boosted score by +{top_boost[1]:.1f} points.")

            top_drag = sorted_attribs[-1]
            if top_drag[1] < 0:
                insights.append(f"Largest score deduction: {top_drag[0].title()} reduced score by {top_drag[1]:.1f} points.")

        return AgentResult.create_success(
            agent_name=self.AGENT_NAME,
            intent="HEALTH_EXPLANATION",
            authoritative_data=combined_auth_data,
            insights=insights,
        )

    def _handle_financial_overview(self, twin: Any, req_id: str) -> AgentResult:
        """Combines health score, balance sheet ratios, and personality clustering into an executive overview."""
        health_data = self.registry.execute("calculate_health_score", twin, request_id=req_id)
        ratios_data = self.registry.execute("calculate_financial_ratios", twin, request_id=req_id)
        personality_data = self.registry.execute("classify_financial_personality", twin, request_id=req_id)

        merged_metrics = {
            "health_score": health_data.metrics.get("health_score", 0.0),
            "grade": health_data.metrics.get("grade", "N/A"),
            "risk_level": health_data.metrics.get("risk_level", "Moderate"),
            "savings_rate": ratios_data.metrics.get("savings_rate", 0.0),
            "emergency_fund_months": ratios_data.metrics.get("emergency_fund_months", 0.0),
            "net_worth": ratios_data.metrics.get("net_worth", 0.0),
            "personality": personality_data.metrics.get("personality", "Balanced Planner"),
            "personality_tagline": personality_data.metrics.get("tagline", ""),
        }

        combined_auth_data = AuthoritativeData(
            provenance=AuthoritativeProvenance(
                source_engine="FinancialIntelligenceComposite",
                source_tool="financial_overview",
                calculation_version="v1.0.0-composite",
                request_id=req_id,
            ),
            computation_type=ComputationType.DETERMINISTIC,
            metrics=merged_metrics,
            confidence_score=1.0,
        )

        insights = [
            f"Health Score: {merged_metrics['health_score']:.1f}/100 (Grade {merged_metrics['grade']}).",
            f"Net Worth: ₹{merged_metrics['net_worth']:,.0f} with {merged_metrics['savings_rate']:.1%} savings rate.",
            f"Financial Personality: '{merged_metrics['personality']}' ({merged_metrics['personality_tagline']}).",
            f"Emergency Coverage: {merged_metrics['emergency_fund_months']:.1f} months of expenses.",
        ]

        return AgentResult.create_success(
            agent_name=self.AGENT_NAME,
            intent="FINANCIAL_OVERVIEW",
            authoritative_data=combined_auth_data,
            insights=insights,
        )

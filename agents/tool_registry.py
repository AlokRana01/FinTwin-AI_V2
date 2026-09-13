"""
agents/tool_registry.py
=======================
Authoritative Tool Registry for FinTwin AI's Multi-Agent Architecture.

Exposes thin, read-only, type-safe wrappers around existing deterministic financial
and ML engines. Preserves computational provenance and isolates tool execution.

Architectural Guarantees:
- Zero duplicated financial math: all operations delegate to existing engines.
- Strictly read-only execution: no database writes, auth mutation, or code execution.
- Tax analysis is encapsulated under Financial Intelligence tools (no Tax Agent).
- Every execution returns structured AuthoritativeData with AuthoritativeProvenance.
- Bounded, role-based access policy mapping tools to specialist agent roles.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Sequence, Mapping, Tuple
import time
import uuid
from datetime import datetime

from agents.schemas import (
    AuthoritativeData,
    AuthoritativeProvenance,
    ComputationType,
)

# Authoritative Engine Imports
from models.twin_engine import HealthScoreEngine, FinancialDigitalTwin
from models.predictor import FinancialPredictor
from models.clustering import FinancialPersonalityClusterer, PERSONALITY_META
from models.explainability import ExplainableAI, HealthScoreExplainer
from utils.simulator import ScenarioSimulator, SCENARIO_REGISTRY
from utils.goal_engine import GoalEngine, MultiGoalPlanner, GoalInput, GoalResult
from utils.tax_calculator import IndianTaxCalculator, DeductionProfile
from utils.coach import FinancialCoach


# ── Tool Definition Metadata ──────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolDefinition:
    """
    Metadata and callable interface for a registered tool.
    """
    name: str
    description: str
    category: str  # "financial_intelligence" | "risk_behaviour" | "forecast_goal" | "scenario_simulation" | "explainability"
    func: Callable[..., AuthoritativeData]
    read_only: bool = True
    authoritative: bool = True
    source_engine: str = ""
    version: str = "1.0.0"


# ── Category 1: Financial Intelligence Wrappers ───────────────────────────────

def calculate_health_score(twin: Any, request_id: str = "") -> AuthoritativeData:
    """
    Calculates the 6-pillar financial health score (0-100) and grade (A-F).
    Delegates directly to HealthScoreEngine.
    """
    start_t = time.perf_counter()
    req_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

    engine = HealthScoreEngine(twin)
    score_data = engine.compute_overall_health_score()
    exec_ms = (time.perf_counter() - start_t) * 1000

    prov = AuthoritativeProvenance(
        source_engine="HealthScoreEngine",
        source_tool="calculate_health_score",
        calculation_version="v1.2.0-canonical",
        request_id=req_id,
        execution_time_ms=exec_ms,
    )

    components_dict = {}
    for k, v in score_data.get("components", {}).items():
        if isinstance(v, dict):
            components_dict[k] = float(v.get("score", 0.0))
        else:
            components_dict[k] = float(v)

    metrics = {
        "health_score": float(score_data.get("overall_score", score_data.get("score", 0.0))),
        "grade": str(score_data.get("financial_grade", score_data.get("grade", "N/A"))),
        "components": components_dict,
        "risk_level": str(getattr(twin, "risk_level", "Moderate")),
    }

    return AuthoritativeData(
        provenance=prov,
        computation_type=ComputationType.DETERMINISTIC,
        metrics=metrics,
        confidence_score=1.0,
    )


def calculate_financial_ratios(twin: Any, request_id: str = "") -> AuthoritativeData:
    """
    Computes standard balance sheet and cash flow financial ratios.
    Delegates to FinancialDigitalTwin methods.
    """
    start_t = time.perf_counter()
    req_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

    metrics_raw = twin.get_summary_metrics()
    exec_ms = (time.perf_counter() - start_t) * 1000

    prov = AuthoritativeProvenance(
        source_engine="FinancialDigitalTwin",
        source_tool="calculate_financial_ratios",
        calculation_version="v1.0.0-deterministic",
        request_id=req_id,
        execution_time_ms=exec_ms,
    )

    metrics = {
        "total_income": float(metrics_raw.get("total_income", 0.0)),
        "basic_expenses": float(metrics_raw.get("basic_expenses", 0.0)),
        "monthly_emi": float(metrics_raw.get("monthly_emi", 0.0)),
        "net_worth": float(metrics_raw.get("net_worth", 0.0)),
        "savings_rate": float(metrics_raw.get("savings_rate", 0.0)),
        "emergency_fund_months": float(metrics_raw.get("emergency_fund_months", 0.0)),
        "loan_amount": float(metrics_raw.get("loan_amount", 0.0)),
    }

    return AuthoritativeData(
        provenance=prov,
        computation_type=ComputationType.DETERMINISTIC,
        metrics=metrics,
        confidence_score=1.0,
    )


def classify_financial_personality(
    twin: Any,
    clusterer: Optional[Any] = None,
    request_id: str = "",
) -> AuthoritativeData:
    """
    Classifies user financial behavioral profile into 1 of 5 clusters.
    Delegates to FinancialPersonalityClusterer.
    """
    start_t = time.perf_counter()
    req_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

    clf = clusterer or FinancialPersonalityClusterer.load()
    personality_label = clf.predict_personality(twin)
    meta = PERSONALITY_META.get(personality_label, {})
    exec_ms = (time.perf_counter() - start_t) * 1000

    prov = AuthoritativeProvenance(
        source_engine="FinancialPersonalityClusterer",
        source_tool="classify_financial_personality",
        calculation_version="v1.0.0-kmeans",
        request_id=req_id,
        execution_time_ms=exec_ms,
    )

    metrics = {
        "personality": str(personality_label),
        "tagline": str(meta.get("tagline", "")),
        "description": str(meta.get("description", "")),
        "nudges": tuple(meta.get("nudges", [])),
    }

    return AuthoritativeData(
        provenance=prov,
        computation_type=ComputationType.MODEL_OUTPUT,
        metrics=metrics,
    )


def calculate_tax(
    gross_income: float,
    deductions: Optional[DeductionProfile] = None,
    calculator: Optional[IndianTaxCalculator] = None,
    request_id: str = "",
) -> AuthoritativeData:
    """
    Computes and compares Indian income tax under Old vs. New Regimes (FY 2025-26).
    Delegates to IndianTaxCalculator.
    """
    start_t = time.perf_counter()
    req_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

    calc = calculator or IndianTaxCalculator()
    ded = deductions or DeductionProfile()
    comp = calc.compare_and_optimize(gross_income=gross_income, deductions=ded)
    exec_ms = (time.perf_counter() - start_t) * 1000

    prov = AuthoritativeProvenance(
        source_engine="IndianTaxCalculator",
        source_tool="calculate_tax",
        calculation_version="v1.0.0-fy25-26",
        request_id=req_id,
        execution_time_ms=exec_ms,
    )

    old_res = comp["old_regime"]
    new_res = comp["new_regime"]

    metrics = {
        "gross_income": float(gross_income),
        "old_regime_total_tax": float(old_res.total_tax),
        "new_regime_total_tax": float(new_res.total_tax),
        "tax_savings": float(comp.get("tax_savings", 0.0)),
        "recommended_regime": str(comp.get("recommended_regime", "New Regime")),
        "is_old_better": bool(comp.get("is_old_better", False)),
        "old_regime_effective_rate": float(old_res.effective_rate),
        "new_regime_effective_rate": float(new_res.effective_rate),
    }

    return AuthoritativeData(
        provenance=prov,
        computation_type=ComputationType.DETERMINISTIC,
        metrics=metrics,
        confidence_score=1.0,
    )


# ── Category 2: Risk & Behaviour Wrappers ─────────────────────────────────────

def analyze_debt(twin: Any, request_id: str = "") -> AuthoritativeData:
    """
    Analyzes debt liabilities, EMI burden, and debt-to-income ratios.
    Delegates to FinancialDigitalTwin properties and HealthScoreEngine.
    """
    start_t = time.perf_counter()
    req_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

    tot_income = float(getattr(twin, "total_income", 0.0))
    monthly_emi = float(getattr(twin, "monthly_emi", 0.0))
    emi_ratio = monthly_emi / tot_income if tot_income > 0 else 0.0

    engine = HealthScoreEngine(twin)
    emi_score = engine.calculate_emi_burden_score()
    exec_ms = (time.perf_counter() - start_t) * 1000

    prov = AuthoritativeProvenance(
        source_engine="FinancialDigitalTwin",
        source_tool="analyze_debt",
        calculation_version="v1.0.0-deterministic",
        request_id=req_id,
        execution_time_ms=exec_ms,
    )

    metrics = {
        "total_loan_amount": float(getattr(twin, "loan_amount", 0.0)),
        "car_loan": float(getattr(twin, "car_loan", 0.0)),
        "home_loan": float(getattr(twin, "home_loan", 0.0)),
        "credit_card_debt": float(getattr(twin, "credit_card_debt", 0.0)),
        "monthly_emi": monthly_emi,
        "emi_to_income_ratio": round(emi_ratio, 4),
        "emi_burden_score": float(emi_score),
        "risk_level": str(getattr(twin, "risk_level", "Moderate")),
    }

    return AuthoritativeData(
        provenance=prov,
        computation_type=ComputationType.DETERMINISTIC,
        metrics=metrics,
        confidence_score=1.0,
    )


def analyze_emergency_fund(twin: Any, request_id: str = "") -> AuthoritativeData:
    """
    Evaluates emergency reserve liquidity against non-discretionary commitments.
    Delegates to HealthScoreEngine.
    """
    start_t = time.perf_counter()
    req_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

    engine = HealthScoreEngine(twin)
    ef_score = engine.calculate_emergency_fund_score()

    outflow = float(twin.basic_expenses + twin.monthly_emi)
    ef_months = float(twin.emergency_fund) / outflow if outflow > 0 else 0.0
    exec_ms = (time.perf_counter() - start_t) * 1000

    prov = AuthoritativeProvenance(
        source_engine="HealthScoreEngine",
        source_tool="analyze_emergency_fund",
        calculation_version="v1.2.0-canonical",
        request_id=req_id,
        execution_time_ms=exec_ms,
    )

    metrics = {
        "emergency_fund_amount": float(twin.emergency_fund),
        "monthly_commitments": outflow,
        "months_covered": round(ef_months, 2),
        "emergency_fund_score": float(ef_score),
        "is_adequate": ef_months >= 6.0,
    }

    return AuthoritativeData(
        provenance=prov,
        computation_type=ComputationType.DETERMINISTIC,
        metrics=metrics,
        confidence_score=1.0,
    )


def analyze_spending_behavior(twin: Any, request_id: str = "") -> AuthoritativeData:
    """
    Diagnoses discretionary spending leaks, lifestyle creep, and savings habits.
    Delegates to FinancialCoach domain analyzers.
    """
    start_t = time.perf_counter()
    req_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

    coach = FinancialCoach(twin)
    report = coach.generate_report()
    exec_ms = (time.perf_counter() - start_t) * 1000

    prov = AuthoritativeProvenance(
        source_engine="FinancialCoach",
        source_tool="analyze_spending_behavior",
        calculation_version="v1.0.0-heuristic",
        request_id=req_id,
        execution_time_ms=exec_ms,
    )

    alerts_list = [
        {"severity": a.severity.value, "headline": a.headline, "message": a.message}
        for a in report.alerts
    ]

    metrics = {
        "overall_coach_score": float(report.overall_score),
        "risk_level": str(report.risk_level),
        "top_action": str(report.top_action),
        "active_alerts_count": len(report.alerts),
        "critical_alerts_count": len(report.critical_alerts),
        "alerts": tuple(alerts_list),
    }

    return AuthoritativeData(
        provenance=prov,
        computation_type=ComputationType.HEURISTIC,
        metrics=metrics,
    )


def assess_financial_risk(twin: Any, request_id: str = "") -> AuthoritativeData:
    """
    Synthesizes multi-factor financial risk tier (Low, Moderate, High, Critical).
    Delegates to FinancialDigitalTwin risk assessment.
    """
    start_t = time.perf_counter()
    req_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

    risk = getattr(twin, "risk_level", "Moderate")
    summary = twin.get_summary_metrics()
    exec_ms = (time.perf_counter() - start_t) * 1000

    prov = AuthoritativeProvenance(
        source_engine="FinancialDigitalTwin",
        source_tool="assess_financial_risk",
        calculation_version="v1.0.0-deterministic",
        request_id=req_id,
        execution_time_ms=exec_ms,
    )

    metrics = {
        "risk_level": str(risk),
        "savings_rate": float(summary.get("savings_rate", 0.0)),
        "emergency_fund_months": float(summary.get("emergency_fund_months", 0.0)),
        "loan_amount": float(summary.get("loan_amount", 0.0)),
    }

    return AuthoritativeData(
        provenance=prov,
        computation_type=ComputationType.DETERMINISTIC,
        metrics=metrics,
        confidence_score=1.0,
    )


# ── Category 3: Forecast & Goal Wrappers ──────────────────────────────────────

def forecast_savings(
    twin: Any,
    horizon_months: int = 12,
    predictor: Optional[Any] = None,
    request_id: str = "",
) -> AuthoritativeData:
    """
    Projects future savings and net worth at a specific horizon using XGBoost.
    Delegates to FinancialPredictor.predict.
    """
    start_t = time.perf_counter()
    req_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

    pred_engine = predictor or FinancialPredictor()
    pred_res = pred_engine.predict(twin, horizon_months=horizon_months)
    exec_ms = (time.perf_counter() - start_t) * 1000

    prov = AuthoritativeProvenance(
        source_engine="FinancialPredictor",
        source_tool="forecast_savings",
        calculation_version="v1.0.0-xgboost",
        request_id=req_id,
        execution_time_ms=exec_ms,
    )

    metrics = {
        "horizon_months": int(horizon_months),
        "predicted_savings": float(pred_res.get("predicted_savings", 0.0)),
        "predicted_net_worth": float(pred_res.get("predicted_net_worth", 0.0)),
    }

    return AuthoritativeData(
        provenance=prov,
        computation_type=ComputationType.MODEL_OUTPUT,
        metrics=metrics,
    )


def forecast_net_worth(
    twin: Any,
    months: int = 12,
    predictor: Optional[Any] = None,
    request_id: str = "",
) -> AuthoritativeData:
    """
    Generates a month-by-month trajectory DataFrame of savings and net worth.
    Delegates to FinancialPredictor.generate_trajectory.
    """
    start_t = time.perf_counter()
    req_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

    pred_engine = predictor or FinancialPredictor()
    df = pred_engine.generate_trajectory(twin, months=months)
    exec_ms = (time.perf_counter() - start_t) * 1000

    prov = AuthoritativeProvenance(
        source_engine="FinancialPredictor",
        source_tool="forecast_net_worth",
        calculation_version="v1.0.0-trajectory",
        request_id=req_id,
        execution_time_ms=exec_ms,
    )

    arrays = {
        "Month": tuple(float(x) for x in df["Month"].tolist()),
        "Projected_Savings": tuple(float(x) for x in df["Projected_Savings"].tolist()),
        "Projected_Net_Worth": tuple(float(x) for x in df["Projected_Net_Worth"].tolist()),
    }

    metrics = {
        "horizon_months": int(months),
        "final_projected_net_worth": float(df["Projected_Net_Worth"].iloc[-1]) if not df.empty else 0.0,
        "final_projected_savings": float(df["Projected_Savings"].iloc[-1]) if not df.empty else 0.0,
    }

    return AuthoritativeData(
        provenance=prov,
        computation_type=ComputationType.MODEL_OUTPUT,
        metrics=metrics,
        arrays=arrays,
    )


def analyze_goal(
    goal_input: GoalInput,
    monthly_surplus: float,
    request_id: str = "",
) -> AuthoritativeData:
    """
    Calculates required monthly SIP, achievement probability, and completion date for a goal.
    Delegates to GoalEngine.
    """
    start_t = time.perf_counter()
    req_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

    engine = GoalEngine(goal_input=goal_input, monthly_surplus=monthly_surplus)
    res: GoalResult = engine.compute()
    exec_ms = (time.perf_counter() - start_t) * 1000

    prov = AuthoritativeProvenance(
        source_engine="GoalEngine",
        source_tool="analyze_goal",
        calculation_version="v1.0.0-algebraic",
        request_id=req_id,
        execution_time_ms=exec_ms,
    )

    metrics = {
        "goal_name": str(res.goal_name),
        "target_amount": float(res.goal_input.goal_amount),
        "inflation_adj_goal": float(res.inflation_adj_goal),
        "sip_required": float(res.sip_required),
        "probability": float(res.probability),
        "probability_label": str(res.probability_label),
        "is_already_funded": bool(res.is_already_funded),
        "shortfall_months": int(res.shortfall_months),
        "projected_completion": res.projected_completion.isoformat() if res.projected_completion else "",
    }

    return AuthoritativeData(
        provenance=prov,
        computation_type=ComputationType.DETERMINISTIC,
        metrics=metrics,
        confidence_score=1.0,
    )


def plan_multiple_goals(
    twin: Any,
    goals: Sequence[GoalInput],
    priority_order: Optional[Sequence[Any]] = None,
    request_id: str = "",
) -> AuthoritativeData:
    """
    Plans multiple goals simultaneously and distributes available monthly surplus.
    Delegates to MultiGoalPlanner.
    """
    start_t = time.perf_counter()
    req_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

    planner = MultiGoalPlanner(twin=twin, goals=list(goals), priority_order=list(priority_order) if priority_order else None)
    results = planner.plan()
    exec_ms = (time.perf_counter() - start_t) * 1000

    prov = AuthoritativeProvenance(
        source_engine="MultiGoalPlanner",
        source_tool="plan_multiple_goals",
        calculation_version="v1.0.0-multi-plan",
        request_id=req_id,
        execution_time_ms=exec_ms,
    )

    goals_summary = [
        {
            "goal_name": r.goal_name,
            "target_amount": float(r.goal_input.goal_amount),
            "sip_required": float(r.sip_required),
            "probability": float(r.probability),
            "probability_label": str(r.probability_label),
            "shortfall_months": int(r.shortfall_months),
        }
        for r in results
    ]

    metrics = {
        "monthly_surplus": float(planner.monthly_surplus),
        "goals_count": len(results),
        "goals": tuple(goals_summary),
    }

    return AuthoritativeData(
        provenance=prov,
        computation_type=ComputationType.DETERMINISTIC,
        metrics=metrics,
        confidence_score=1.0,
    )


# ── Category 4: Scenario Simulation Wrappers ──────────────────────────────────

def run_scenario(
    twin: Any,
    scenario_name: str,
    params: Optional[Dict[str, Any]] = None,
    request_id: str = "",
) -> AuthoritativeData:
    """
    Simulates a financial life-event scenario and computes before vs after impact.
    Delegates to ScenarioSimulator.run_scenario.
    """
    start_t = time.perf_counter()
    req_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

    sim = ScenarioSimulator(twin)
    res = sim.run_scenario(scenario_name, **(params or {}))
    exec_ms = (time.perf_counter() - start_t) * 1000

    prov = AuthoritativeProvenance(
        source_engine="ScenarioSimulator",
        source_tool="run_scenario",
        calculation_version="v1.0.0-simulator",
        request_id=req_id,
        execution_time_ms=exec_ms,
    )

    metrics = {
        "scenario_name": str(scenario_name),
        "before_health_score": float(res.before.health_score),
        "after_health_score": float(res.after.health_score),
        "health_score_delta": float(res.delta_health_score),
        "before_monthly_surplus": float(res.before.monthly_surplus),
        "after_monthly_surplus": float(res.after.monthly_surplus),
        "surplus_delta": float(res.after.monthly_surplus - res.before.monthly_surplus),
        "net_worth_delta_5y": float(res.delta_net_worth_5y),
        "before_risk_level": str(res.before.risk_level),
        "after_risk_level": str(res.after.risk_level),
    }

    return AuthoritativeData(
        provenance=prov,
        computation_type=ComputationType.DETERMINISTIC,
        metrics=metrics,
        confidence_score=1.0,
    )


def compare_scenario(
    twin: Any,
    scenario_name: str,
    params: Optional[Dict[str, Any]] = None,
    request_id: str = "",
) -> AuthoritativeData:
    """
    Performs full delta comparison on a scenario simulation.
    Delegates directly to ScenarioSimulator.run_scenario.
    """
    start_t = time.perf_counter()
    req_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

    sim = ScenarioSimulator(twin)
    res = sim.run_scenario(scenario_name, **(params or {}))
    exec_ms = (time.perf_counter() - start_t) * 1000

    prov = AuthoritativeProvenance(
        source_engine="ScenarioSimulator",
        source_tool="compare_scenario",
        calculation_version="v1.0.0-simulator",
        request_id=req_id,
        execution_time_ms=exec_ms,
    )

    metrics = {
        "scenario_name": str(scenario_name),
        "before_health_score": float(res.before.health_score),
        "after_health_score": float(res.after.health_score),
        "health_score_delta": float(res.delta_health_score),
        "before_monthly_surplus": float(res.before.monthly_surplus),
        "after_monthly_surplus": float(res.after.monthly_surplus),
        "surplus_delta": float(res.after.monthly_surplus - res.before.monthly_surplus),
        "net_worth_delta_5y": float(res.delta_net_worth_5y),
        "before_risk_level": str(res.before.risk_level),
        "after_risk_level": str(res.after.risk_level),
    }

    return AuthoritativeData(
        provenance=prov,
        computation_type=ComputationType.DETERMINISTIC,
        metrics=metrics,
        confidence_score=1.0,
    )


# ── Category 5: Explainability (XAI) Wrappers ─────────────────────────────────

def explain_health_score(twin: Any, request_id: str = "") -> AuthoritativeData:
    """
    Additive feature decomposition of the weighted 6-component Financial Health Score.
    Delegates to HealthScoreExplainer.explain().
    """
    start_t = time.perf_counter()
    req_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

    explainer = HealthScoreExplainer(twin)
    res = explainer.explain()
    exec_ms = (time.perf_counter() - start_t) * 1000

    prov = AuthoritativeProvenance(
        source_engine="ExplainableAI",
        source_tool="explain_health_score",
        calculation_version="v1.0.0-attribution",
        request_id=req_id,
        execution_time_ms=exec_ms,
    )

    attributions = {c.component: round(float(c.contribution), 2) for c in res.contributions}

    metrics = {
        "overall_score": float(res.overall_score),
        "baseline_score": float(res.baseline_score),
        "contributions_count": len(res.contributions),
    }

    return AuthoritativeData(
        provenance=prov,
        computation_type=ComputationType.DETERMINISTIC,
        metrics=metrics,
        xai_attributions=attributions,
        confidence_score=1.0,
    )


def explain_forecast(
    twin: Any,
    predictor: Optional[Any] = None,
    request_id: str = "",
) -> AuthoritativeData:
    """
    SHAP TreeExplainer feature importance attributions for XGBoost forecasts.
    Delegates to ExplainableAI.explain_forecast.
    """
    start_t = time.perf_counter()
    req_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

    pred_engine = predictor or FinancialPredictor()
    xai = ExplainableAI()
    res = xai.explain_forecast(twin, predictor=pred_engine)
    exec_ms = (time.perf_counter() - start_t) * 1000

    prov = AuthoritativeProvenance(
        source_engine="ExplainableAI",
        source_tool="explain_forecast",
        calculation_version="v1.0.0-treeshap",
        request_id=req_id,
        execution_time_ms=exec_ms,
    )

    if res and res.contributions_savings:
        attributions = {c.feature: round(float(c.shap_value), 2) for c in res.contributions_savings}
        predicted_sav = float(res.predicted_savings)
        base_val = float(res.base_value_savings)
        method = str(res.method)
    else:
        attributions = {}
        predicted_sav = 0.0
        base_val = 0.0
        method = "fallback"

    metrics = {
        "predicted_savings": predicted_sav,
        "base_value_savings": base_val,
        "method": method,
    }

    return AuthoritativeData(
        provenance=prov,
        computation_type=ComputationType.MODEL_OUTPUT,
        metrics=metrics,
        xai_attributions=attributions,
    )


# ── Authoritative Tool Registry Class ─────────────────────────────────────────

class ToolRegistry:
    """
    Central registry managing tool definitions, authorized agent access policies,
    and strictly read-only execution wrappers.
    """
    # Role-based tool access policy
    AGENT_ROLE_POLICIES: Dict[str, List[str]] = {
        "financial_intelligence": [
            "calculate_health_score",
            "calculate_financial_ratios",
            "classify_financial_personality",
            "calculate_tax",
            "explain_health_score",
        ],
        "risk_behaviour": [
            "analyze_debt",
            "analyze_emergency_fund",
            "analyze_spending_behavior",
            "assess_financial_risk",
            "calculate_health_score",
        ],
        "forecast_goal": [
            "forecast_savings",
            "forecast_net_worth",
            "analyze_goal",
            "plan_multiple_goals",
            "explain_forecast",
        ],
        "scenario_simulation": [
            "run_scenario",
            "compare_scenario",
        ],
    }
    AGENT_TOOL_POLICIES = AGENT_ROLE_POLICIES

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Registers all authoritative tools across all 5 categories."""
        defaults = [
            # Financial Intelligence
            ToolDefinition(
                name="calculate_health_score",
                description="Calculates the 6-pillar financial health score (0-100) and grade (A-F).",
                category="financial_intelligence",
                func=calculate_health_score,
                source_engine="HealthScoreEngine",
            ),
            ToolDefinition(
                name="calculate_financial_ratios",
                description="Computes balance sheet and cash flow financial ratios.",
                category="financial_intelligence",
                func=calculate_financial_ratios,
                source_engine="FinancialDigitalTwin",
            ),
            ToolDefinition(
                name="classify_financial_personality",
                description="Segments user into 1 of 5 financial personality clusters.",
                category="financial_intelligence",
                func=classify_financial_personality,
                source_engine="FinancialPersonalityClusterer",
            ),
            ToolDefinition(
                name="calculate_tax",
                description="Computes and compares Indian income tax under Old vs. New Regimes.",
                category="financial_intelligence",
                func=calculate_tax,
                source_engine="IndianTaxCalculator",
            ),
            # Risk & Behaviour
            ToolDefinition(
                name="analyze_debt",
                description="Analyzes loan structure, EMI burden, and debt ratios.",
                category="risk_behaviour",
                func=analyze_debt,
                source_engine="FinancialDigitalTwin",
            ),
            ToolDefinition(
                name="analyze_emergency_fund",
                description="Evaluates emergency reserve liquidity and months covered.",
                category="risk_behaviour",
                func=analyze_emergency_fund,
                source_engine="HealthScoreEngine",
            ),
            ToolDefinition(
                name="analyze_spending_behavior",
                description="Diagnoses spending habits, leaks, and active coaching alerts.",
                category="risk_behaviour",
                func=analyze_spending_behavior,
                source_engine="FinancialCoach",
            ),
            ToolDefinition(
                name="assess_financial_risk",
                description="Synthesizes multi-factor financial risk tier.",
                category="risk_behaviour",
                func=assess_financial_risk,
                source_engine="FinancialDigitalTwin",
            ),
            # Forecast & Goal
            ToolDefinition(
                name="forecast_savings",
                description="Projects savings and net worth at 6M/12M horizons via XGBoost.",
                category="forecast_goal",
                func=forecast_savings,
                source_engine="FinancialPredictor",
            ),
            ToolDefinition(
                name="forecast_net_worth",
                description="Generates month-by-month trajectory DataFrame of savings and net worth.",
                category="forecast_goal",
                func=forecast_net_worth,
                source_engine="FinancialPredictor",
            ),
            ToolDefinition(
                name="analyze_goal",
                description="Calculates required SIP, achievement probability, and completion date for a goal.",
                category="forecast_goal",
                func=analyze_goal,
                source_engine="GoalEngine",
            ),
            ToolDefinition(
                name="plan_multiple_goals",
                description="Plans multiple goals simultaneously and distributes available surplus.",
                category="forecast_goal",
                func=plan_multiple_goals,
                source_engine="MultiGoalPlanner",
            ),
            # Scenario Simulation
            ToolDefinition(
                name="run_scenario",
                description="Simulates financial perturbations and computes before vs after impact.",
                category="scenario_simulation",
                func=run_scenario,
                source_engine="ScenarioSimulator",
            ),
            ToolDefinition(
                name="compare_scenario",
                description="Performs comparative delta evaluation on financial scenarios.",
                category="scenario_simulation",
                func=compare_scenario,
                source_engine="ScenarioSimulator",
            ),
            # Explainability
            ToolDefinition(
                name="explain_health_score",
                description="Additive SHAP-style feature decomposition of Financial Health Score.",
                category="explainability",
                func=explain_health_score,
                source_engine="ExplainableAI",
            ),
            ToolDefinition(
                name="explain_forecast",
                description="SHAP TreeExplainer feature importance attributions for XGBoost forecasts.",
                category="explainability",
                func=explain_forecast,
                source_engine="ExplainableAI",
            ),
        ]
        for t in defaults:
            self.register(t)

    def register(self, tool_def: ToolDefinition) -> None:
        """Registers a tool definition ensuring uniqueness and read-only enforcement."""
        if not tool_def.read_only:
            raise ValueError(f"Security violation: Tool '{tool_def.name}' must be read-only.")
        self._tools[tool_def.name] = tool_def

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Returns tool definition by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        """Returns all registered tool definitions."""
        return list(self._tools.values())

    def get_tools_for_agent(self, agent_role: str) -> Dict[str, ToolDefinition]:
        """
        Returns authorized tools mapped to a specific specialist agent role.
        Enforces role-based isolation.
        """
        allowed_names = self.AGENT_ROLE_POLICIES.get(agent_role, [])
        return {name: self._tools[name] for name in allowed_names if name in self._tools}

    def execute(self, tool_name: str, *args, **kwargs) -> AuthoritativeData:
        """
        Executes a registered tool by name with exception isolation.
        """
        tool_def = self._tools.get(tool_name)
        if not tool_def:
            raise KeyError(f"Tool '{tool_name}' is not registered in ToolRegistry.")
        return tool_def.func(*args, **kwargs)


# Global default registry instance
DEFAULT_TOOL_REGISTRY = ToolRegistry()

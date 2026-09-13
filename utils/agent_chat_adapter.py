"""
utils/agent_chat_adapter.py
===========================
Chatbot adapter layer bridging the Streamlit conversation interface with
the FinTwin AI OrchestratorAgent and multi-agent system.

Responsibilities:
- Input sanitization & security validation.
- Deterministic intent classification strictly matching the 9 Orchestrator routes.
- Identification of non-financial queries (greetings, app guidance) without invoking agents.
- Parameter extraction without inventing unstated assumptions or financial values.
- Safe immutable conversion of FinancialDigitalTwin to FinancialState.
- Request container construction with zero financial calculations or direct LLM execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
import re
import uuid

from utils.security import sanitize_chat_message
from agents.state import FinancialState
from models.twin_engine import FinancialDigitalTwin


# ── Canonical Orchestrator Routes (9 Exact Routes) ────────────────────────────

SUPPORTED_ORCHESTRATOR_ROUTES = {
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


# ── Request Container ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChatOrchestrationRequest:
    """
    Standardized, immutable request structure for multi-agent chatbot orchestration.
    """
    request_id: str
    user_message: str
    sanitized_message: str
    is_financial_query: bool
    intent: Optional[str]  # One of the 9 Orchestrator routes, or None for non-financial
    query_category: str  # "FINANCIAL" | "GREETING" | "APP_GUIDANCE" | "INVALID"
    financial_state: Optional[FinancialState]
    scenario_name: Optional[str] = None
    scenario_params: Optional[Dict[str, Any]] = None
    missing_scenario_params: Tuple[str, ...] = ()
    goal_params: Optional[Dict[str, Any]] = None
    missing_goal_params: Tuple[str, ...] = ()
    requires_clarification: bool = False
    clarification_prompt: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "user_message": self.user_message,
            "sanitized_message": self.sanitized_message,
            "is_financial_query": self.is_financial_query,
            "intent": self.intent,
            "query_category": self.query_category,
            "has_financial_state": self.financial_state is not None,
            "scenario_name": self.scenario_name,
            "scenario_params": self.scenario_params,
            "missing_scenario_params": self.missing_scenario_params,
            "goal_params": self.goal_params,
            "missing_goal_params": self.missing_goal_params,
            "requires_clarification": self.requires_clarification,
            "clarification_prompt": self.clarification_prompt,
        }


# ── Agent Chat Adapter ────────────────────────────────────────────────────────

class AgentChatAdapter:
    """
    Lightweight, deterministic adapter between Chatbot UI and OrchestratorAgent.
    """

    # Non-financial keyword patterns
    _GREETING_PATTERNS = [
        r"^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening)|howdy)\b",
        r"^(thanks|thank\s+you|thankyou|thx)\b",
        r"^(bye|goodbye|see\s+you|cya)\b",
    ]

    _APP_GUIDANCE_PATTERNS = [
        r"\b(what\s+can\s+this\s+app\s+do|how\s+to\s+use|app\s+features|what\s+is\s+fintwin|help\s+me\s+use|app\s+guide)\b",
    ]

    # Scenario trigger keywords
    _SCENARIO_TRIGGERS = [
        "what if", "suppose", "simulate", "scenario", "if i buy", "if my salary",
        "if i lose", "if i take", "if i increase", "hypothetical", "what happens if"
    ]

    # Forecast keywords
    _FORECAST_TRIGGERS = [
        "forecast", "future savings", "net worth in", "projection",
        "savings projection", "wealth trajectory", "predict", "grow over",
        "how much will i save", "trajectory",
    ]

    # Health diagnostic triggers
    _HEALTH_DIAGNOSTIC_TRIGGERS = [
        "why is my financial health", "why is my score", "diagnose", "what is causing my low score",
        "what is dragging my score", "why is my grade", "health diagnostic", "why is score low"
    ]

    # Comprehensive triggers
    _COMPREHENSIVE_TRIGGERS = [
        "comprehensive", "full review", "full financial review", "audit my finances",
        "complete financial analysis", "complete review", "360 review", "full audit"
    ]

    def __init__(self, orchestrator: Optional[Any] = None):
        self._orchestrator = orchestrator

    def classify_intent(self, text: str) -> Tuple[Optional[str], str]:
        """
        Classifies user query into one of the 9 Orchestrator routes or a non-financial category.
        Enforces strict precedence:
        1. Non-financial (Greeting / App Guidance)
        2. Scenario + Forecast
        3. Scenario
        4. Health Diagnostic
        5. Comprehensive
        6. Forecast
        7. Goal
        8. Health
        9. Risk
        10. Overview (Safe Fallback)
        """
        if not text or not text.strip():
            return None, "INVALID"

        norm = text.lower().strip()

        # 1. Non-financial greetings
        for pattern in self._GREETING_PATTERNS:
            if re.search(pattern, norm):
                # Ensure it's not a financial question appended to greeting
                if not any(w in norm for w in ["health", "score", "loan", "emi", "tax", "save", "savings", "invest", "debt"]):
                    return None, "GREETING"

        # 2. Non-financial app guidance
        for pattern in self._APP_GUIDANCE_PATTERNS:
            if re.search(pattern, norm):
                return None, "APP_GUIDANCE"

        has_scenario = any(t in norm for t in self._SCENARIO_TRIGGERS)
        has_forecast = any(t in norm for t in self._FORECAST_TRIGGERS)

        # 3. Scenario + Forecast (Combined dependency route)
        if has_scenario and has_forecast:
            return "scenario_forecast", "FINANCIAL"

        # 4. Scenario Simulation
        if has_scenario:
            return "scenario", "FINANCIAL"

        # 5. Health Diagnostic
        if any(t in norm for t in self._HEALTH_DIAGNOSTIC_TRIGGERS):
            return "health_diagnostic", "FINANCIAL"

        # 6. Comprehensive Review
        if any(t in norm for t in self._COMPREHENSIVE_TRIGGERS):
            return "comprehensive", "FINANCIAL"

        # 7. Forecast Projections
        if has_forecast:
            return "forecast", "FINANCIAL"

        # 8. Goals Planning
        if any(w in norm for w in ["goal", "target", "save for", "saving for", "reach my", "sip target", "sip required", "retirement target"]) or re.search(r"\bsave\s+\d+", norm):
            return "goal", "FINANCIAL"

        # 9. Health & Ratios
        if any(w in norm for w in ["health", "score", "grade", "financial health", "ratio", "personality", "pillar"]):
            return "health", "FINANCIAL"

        # 10. Risk & Behavior
        if any(w in norm for w in ["debt", "emi", "loan", "emergency fund", "credit card", "spending", "risk", "leaks", "vulnerability"]):
            return "risk", "FINANCIAL"

        # 11. Overview / Ambiguous Fallback
        return "overview", "FINANCIAL"

    def _parse_amount(self, text: str) -> Optional[float]:
        """Helper to parse currency amounts in Indian financial notation (e.g., 8 lakh, 800000, 50k, 1.5 cr)."""
        norm = text.lower().replace(",", "").replace("₹", "").replace("rs.", "").replace("rs", "").strip()

        # Match "X lakh" / "X lakhs" / "X l"
        m_lakh = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakhs?|lacs?|lac|l)\b", norm)
        if m_lakh:
            try:
                return float(m_lakh.group(1)) * 100000.0
            except ValueError:
                pass

        # Match "X crore" / "X cr"
        m_cr = re.search(r"(\d+(?:\.\d+)?)\s*(?:crores?|crs?|cr)\b", norm)
        if m_cr:
            try:
                return float(m_cr.group(1)) * 10000000.0
            except ValueError:
                pass

        # Match "X k" / "X thousand"
        m_k = re.search(r"(\d+(?:\.\d+)?)\s*(?:k|thousand)\b", norm)
        if m_k:
            try:
                return float(m_k.group(1)) * 1000.0
            except ValueError:
                pass

        # Match plain number with optional decimals (e.g., 800000)
        m_num = re.search(r"\b(\d{4,10}(?:\.\d+)?)\b", norm)
        if m_num:
            try:
                return float(m_num.group(1))
            except ValueError:
                pass

        return None

    def extract_scenario_parameters(self, text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]], Tuple[str, ...]]:
        """
        Deterministically extracts scenario type and explicit parameters from user message.
        NEVER invents missing numerical values.
        """
        norm = text.lower()
        scenario_name: Optional[str] = None
        params: Dict[str, Any] = {}
        missing: List[str] = []

        if "car" in norm:
            scenario_name = "Car Purchase"
            amt = self._parse_amount(norm)
            if amt is not None:
                params["car_price"] = amt
            else:
                missing.append("car_price")

        elif "salary" in norm and ("hike" in norm or "increase" in norm or "raise" in norm):
            scenario_name = "Salary Hike"
            m_pct = re.search(r"(\d+(?:\.\d+)?)\s*%", norm)
            if m_pct:
                params["hike_percent"] = float(m_pct.group(1))
            else:
                amt = self._parse_amount(norm)
                if amt is not None:
                    params["hike_amount"] = amt
                else:
                    missing.append("hike_percent")

        elif "job loss" in norm or "lose my job" in norm or "lost my job" in norm or "unemployed" in norm:
            scenario_name = "Job Loss"
            m_m = re.search(r"(\d+)\s*(?:months?|mo)\b", norm)
            if m_m:
                params["duration_months"] = int(m_m.group(1))
            else:
                missing.append("duration_months")

        elif "home" in norm and ("loan" in norm or "house" in norm or "flat" in norm or "property" in norm):
            scenario_name = "Home Loan"
            amt = self._parse_amount(norm)
            if amt is not None:
                params["purchase_price"] = amt
            else:
                missing.append("purchase_price")

        elif "marriage" in norm or "wedding" in norm:
            scenario_name = "Marriage Expense"
            amt = self._parse_amount(norm)
            if amt is not None:
                params["one_time_cost"] = amt
            else:
                missing.append("one_time_cost")

        elif "sip" in norm and ("increase" in norm or "boost" in norm or "raise" in norm):
            scenario_name = "Increase SIP"
            m_pct = re.search(r"(\d+(?:\.\d+)?)\s*%", norm)
            if m_pct:
                params["sip_increase_pct"] = float(m_pct.group(1))
            else:
                amt = self._parse_amount(norm)
                if amt is not None:
                    params["sip_increase_amount"] = amt
                else:
                    missing.append("sip_increase_pct")

        return scenario_name, (params if params else None), tuple(missing)

    def extract_goal_parameters(self, text: str) -> Tuple[Optional[Dict[str, Any]], Tuple[str, ...]]:
        """
        Deterministically extracts explicit goal parameters. Never invents missing values.
        """
        norm = text.lower()
        params: Dict[str, Any] = {}
        missing: List[str] = []

        amt = self._parse_amount(norm)
        if amt is not None:
            params["target_amount"] = amt

        m_years = re.search(r"(\d+)\s*(?:years?|yrs?|yr)\b", norm)
        if m_years:
            params["horizon_years"] = int(m_years.group(1))

        return (params if params else None), tuple(missing)

    def build_request(
        self,
        user_message: str,
        twin: Optional[Union[FinancialDigitalTwin, Any]] = None,
        request_id: str = "",
        financial_state: Optional[FinancialState] = None,
    ) -> ChatOrchestrationRequest:
        """
        Main entry point for constructing an immutable ChatOrchestrationRequest.
        """
        req_id = request_id or f"req-chat-{uuid.uuid4().hex[:8]}"
        sanitized = sanitize_chat_message(user_message or "")

        # 1. Classify intent
        intent, category = self.classify_intent(sanitized)

        # 2. Immutable FinancialState resolution
        f_state = financial_state
        if f_state is None:
            if isinstance(twin, FinancialDigitalTwin):
                f_state = FinancialState.from_twin(twin)
            elif isinstance(twin, FinancialState):
                f_state = twin
            elif twin is None:
                f_state = FinancialState.from_twin(None)

        # 3. Parameter extraction and clarification handling
        scenario_name: Optional[str] = None
        scenario_params: Optional[Dict[str, Any]] = None
        missing_scenario: Tuple[str, ...] = ()
        goal_params: Optional[Dict[str, Any]] = None
        missing_goal: Tuple[str, ...] = ()
        requires_clarification = False
        clarification_prompt: Optional[str] = None

        if intent in ("scenario", "scenario_forecast"):
            scenario_name, scenario_params, missing_scenario = self.extract_scenario_parameters(sanitized)
            if missing_scenario:
                requires_clarification = True
                missing_str = ", ".join(missing_scenario)
                s_name = scenario_name or "scenario"
                clarification_prompt = (
                    f"To accurately simulate the {s_name}, please specify the required value(s): {missing_str}."
                )

        elif intent == "goal":
            goal_params, missing_goal = self.extract_goal_parameters(sanitized)

        return ChatOrchestrationRequest(
            request_id=req_id,
            user_message=user_message,
            sanitized_message=sanitized,
            is_financial_query=(category == "FINANCIAL"),
            intent=intent,
            query_category=category,
            financial_state=f_state,
            scenario_name=scenario_name,
            scenario_params=scenario_params,
            missing_scenario_params=missing_scenario,
            goal_params=goal_params,
            missing_goal_params=missing_goal,
            requires_clarification=requires_clarification,
            clarification_prompt=clarification_prompt,
        )


# Global default adapter instance
DEFAULT_CHAT_ADAPTER = AgentChatAdapter()

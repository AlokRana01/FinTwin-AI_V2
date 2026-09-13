"""
agents/state.py
===============
Defines the canonical, deeply immutable FinancialState snapshot and the
mutable SessionContext container for FinTwin AI's multi-agent architecture.

Architectural Rule:
- FinancialState represents an immutable point-in-time financial profile snapshot.
- SessionContext represents mutable conversational session history and UI interaction state.
- These two concepts are strictly decoupled.
"""

from dataclasses import dataclass, field
from typing import Mapping, Tuple, Optional, Any, Sequence, Dict, List
from types import MappingProxyType
from datetime import datetime, timezone
import uuid


@dataclass(frozen=True)
class UserProfileContext:
    """
    Immutable user profile and demographic context.
    """
    user_id: str
    username: str = "User"
    age: int = 25
    occupation: str = "Other"
    city: str = ""
    city_tier: int = 1
    risk_tolerance: str = "Moderate"  # "Conservative" | "Moderate" | "Aggressive"
    currency: str = "INR"

    def __post_init__(self):
        if not (18 <= self.age <= 100):
            raise ValueError(f"Age must be between 18 and 100, got {self.age}")
        if self.city_tier not in (1, 2, 3):
            object.__setattr__(self, "city_tier", 1)


@dataclass(frozen=True)
class CashFlowState:
    """
    Immutable monthly cash flow metrics.
    """
    monthly_income: float = 0.0
    bonus: float = 0.0
    additional_income: float = 0.0
    monthly_expenses: float = 0.0
    discretionary_expenses: float = 0.0
    non_discretionary_expenses: float = 0.0
    monthly_emi: float = 0.0
    monthly_savings: float = 0.0
    savings_rate: float = 0.0
    emi_ratio: float = 0.0

    def __post_init__(self):
        # Validate non-negative cashflow metrics
        for field_name in (
            "monthly_income", "bonus", "additional_income", "monthly_expenses",
            "discretionary_expenses", "non_discretionary_expenses", "monthly_emi",
        ):
            val = getattr(self, field_name)
            if val < 0.0:
                raise ValueError(f"{field_name} cannot be negative: {val}")


@dataclass(frozen=True)
class BalanceSheetState:
    """
    Immutable balance sheet asset and liability snapshot.
    """
    bank_savings: float = 0.0
    fixed_deposits: float = 0.0
    emergency_fund: float = 0.0
    mutual_funds: float = 0.0
    stocks: float = 0.0
    ppf_investment: float = 0.0
    nps_investment: float = 0.0
    sip_amount: float = 0.0
    total_assets: float = 0.0
    loan_amount: float = 0.0
    car_loan: float = 0.0
    home_loan: float = 0.0
    credit_card_debt: float = 0.0
    total_liabilities: float = 0.0
    current_net_worth: float = 0.0
    emergency_fund_months: float = 0.0
    health_insurance: float = 0.0
    life_insurance: float = 0.0

    def __post_init__(self):
        for field_name in (
            "bank_savings", "fixed_deposits", "emergency_fund", "mutual_funds",
            "stocks", "ppf_investment", "nps_investment", "sip_amount",
            "total_assets", "loan_amount", "car_loan", "home_loan",
            "credit_card_debt", "total_liabilities", "health_insurance", "life_insurance",
        ):
            val = getattr(self, field_name)
            if val < 0.0:
                raise ValueError(f"{field_name} cannot be negative: {val}")


@dataclass(frozen=True)
class GoalItemState:
    """
    Immutable individual financial goal item.
    """
    goal_id: str
    goal_name: str
    goal_type: str              # e.g., "House", "Car", "Retirement", "Emergency Fund"
    target_amount: float
    current_savings: float = 0.0
    target_date: str = ""       # "YYYY-MM" or year string
    monthly_contribution: float = 0.0
    priority: int = 1           # 1 (highest) to 5 (lowest)

    def __post_init__(self):
        if self.target_amount < 0.0:
            raise ValueError(f"target_amount cannot be negative: {self.target_amount}")
        if self.current_savings < 0.0:
            raise ValueError(f"current_savings cannot be negative: {self.current_savings}")
        if self.monthly_contribution < 0.0:
            raise ValueError(f"monthly_contribution cannot be negative: {self.monthly_contribution}")
        if not (1 <= self.priority <= 10):
            object.__setattr__(self, "priority", 1)


@dataclass(frozen=True)
class DerivedIntelligenceState:
    """
    Immutable cached derived intelligence metrics computed by authoritative engines.
    """
    health_score: Optional[float] = None
    health_grade: Optional[str] = None
    personality_type: Optional[str] = None
    key_risk_level: Optional[str] = None  # "Low" | "Moderate" | "High" | "Critical"
    pillar_scores: Mapping[str, float] = MappingProxyType({})

    def __post_init__(self):
        if isinstance(self.pillar_scores, dict):
            object.__setattr__(self, "pillar_scores", MappingProxyType(dict(self.pillar_scores)))


@dataclass(frozen=True)
class FinancialState:
    """
    Canonical, deeply immutable snapshot of user financial state for a single execution lifecycle.
    Guaranteed thread-safe with zero shared mutable references.
    """
    snapshot_id: str
    timestamp: str
    profile: UserProfileContext
    cash_flow: CashFlowState
    balance_sheet: BalanceSheetState
    goals: Tuple[GoalItemState, ...] = ()
    derived: DerivedIntelligenceState = DerivedIntelligenceState()
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self):
        # Enforce tuple for goals collection
        if not isinstance(self.goals, tuple):
            object.__setattr__(self, "goals", tuple(self.goals))
        # Enforce read-only mapping for metadata
        if isinstance(self.metadata, dict):
            object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_twin(
        cls,
        twin: Any,
        goals: Optional[Sequence[GoalItemState]] = None,
        snapshot_id: Optional[str] = None,
        derived: Optional[DerivedIntelligenceState] = None,
    ) -> "FinancialState":
        """
        Factory method constructing an immutable FinancialState snapshot from a FinancialDigitalTwin instance.
        """
        sid = snapshot_id or f"snap-{uuid.uuid4().hex[:8]}"
        ts = datetime.now(timezone.utc).isoformat()

        # Build Profile Context
        profile = UserProfileContext(
            user_id=str(getattr(twin, "user_id", "guest")),
            username=str(getattr(twin, "name", "User")),
            age=int(getattr(twin, "age", 25)),
            occupation=str(getattr(twin, "occupation", "Other")),
            city=str(getattr(twin, "city", "")),
            city_tier=1,
            risk_tolerance=str(getattr(twin, "risk_tolerance", "Moderate")),
        )

        # Build Cash Flow
        total_income = float(getattr(twin, "total_income", getattr(twin, "monthly_income", 0.0)))
        basic_expenses = float(getattr(twin, "basic_expenses", 0.0))
        monthly_emi = float(getattr(twin, "monthly_emi", 0.0))
        monthly_savings = max(0.0, total_income - basic_expenses - monthly_emi)
        savings_rate = monthly_savings / total_income if total_income > 0 else 0.0
        emi_ratio = monthly_emi / total_income if total_income > 0 else 0.0

        # Compute discretionary vs non-discretionary expenses
        rent = float(getattr(twin, "rent", 0.0))
        groceries = float(getattr(twin, "groceries", 0.0))
        utilities = float(getattr(twin, "utilities", 0.0))
        transport = float(getattr(twin, "transport", 0.0))
        food_delivery = float(getattr(twin, "food_delivery", 0.0))
        entertainment = float(getattr(twin, "entertainment", 0.0))
        shopping = float(getattr(twin, "shopping", 0.0))

        non_disc = rent + groceries + utilities + transport
        disc = food_delivery + entertainment + shopping

        cash_flow = CashFlowState(
            monthly_income=float(getattr(twin, "monthly_income", 0.0)),
            bonus=float(getattr(twin, "bonus", 0.0)),
            additional_income=float(getattr(twin, "additional_income", 0.0)),
            monthly_expenses=basic_expenses,
            discretionary_expenses=disc,
            non_discretionary_expenses=non_disc,
            monthly_emi=monthly_emi,
            monthly_savings=monthly_savings,
            savings_rate=savings_rate,
            emi_ratio=emi_ratio,
        )

        # Build Balance Sheet
        emergency_fund = float(getattr(twin, "emergency_fund", 0.0))
        monthly_commitments = basic_expenses + monthly_emi
        ef_months = emergency_fund / monthly_commitments if monthly_commitments > 0 else 0.0

        bank_savings = float(getattr(twin, "bank_savings", 0.0))
        fd_amount = float(getattr(twin, "fd_amount", 0.0))
        mutual_funds = float(getattr(twin, "mutual_funds", 0.0))
        stocks = float(getattr(twin, "stocks", 0.0))
        ppf = float(getattr(twin, "ppf_investment", 0.0))
        nps = float(getattr(twin, "nps_investment", 0.0))
        sip = float(getattr(twin, "sip_amount", 0.0))

        total_assets = bank_savings + fd_amount + emergency_fund + mutual_funds + stocks + ppf + nps
        
        loan_amount = float(getattr(twin, "loan_amount", 0.0))
        car_loan = float(getattr(twin, "car_loan", 0.0))
        home_loan = float(getattr(twin, "home_loan", 0.0))
        cc_debt = float(getattr(twin, "credit_card_debt", 0.0))
        total_liab = loan_amount + car_loan + home_loan + cc_debt

        net_worth = float(getattr(twin, "net_worth", total_assets - total_liab))

        balance_sheet = BalanceSheetState(
            bank_savings=bank_savings,
            fixed_deposits=fd_amount,
            emergency_fund=emergency_fund,
            mutual_funds=mutual_funds,
            stocks=stocks,
            ppf_investment=ppf,
            nps_investment=nps,
            sip_amount=sip,
            total_assets=total_assets,
            loan_amount=loan_amount,
            car_loan=car_loan,
            home_loan=home_loan,
            credit_card_debt=cc_debt,
            total_liabilities=total_liab,
            current_net_worth=net_worth,
            emergency_fund_months=ef_months,
            health_insurance=float(getattr(twin, "health_insurance", 0.0)),
            life_insurance=float(getattr(twin, "life_insurance", 0.0)),
        )

        # Build Goals tuple
        goals_tuple = tuple(goals) if goals else ()

        # Default derived state if not provided
        derived_state = derived or DerivedIntelligenceState(
            key_risk_level=getattr(twin, "risk_level", None)
        )

        return cls(
            snapshot_id=sid,
            timestamp=ts,
            profile=profile,
            cash_flow=cash_flow,
            balance_sheet=balance_sheet,
            goals=goals_tuple,
            derived=derived_state,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the financial snapshot to a sanitized, plain Python dictionary.
        """
        return {
            "snapshot_id": self.snapshot_id,
            "timestamp": self.timestamp,
            "profile": {
                "user_id": self.profile.user_id,
                "username": self.profile.username,
                "age": self.profile.age,
                "occupation": self.profile.occupation,
                "city": self.profile.city,
                "city_tier": self.profile.city_tier,
                "risk_tolerance": self.profile.risk_tolerance,
                "currency": self.profile.currency,
            },
            "cash_flow": {
                "monthly_income": self.cash_flow.monthly_income,
                "monthly_expenses": self.cash_flow.monthly_expenses,
                "monthly_emi": self.cash_flow.monthly_emi,
                "monthly_savings": self.cash_flow.monthly_savings,
                "savings_rate": round(self.cash_flow.savings_rate, 4),
                "emi_ratio": round(self.cash_flow.emi_ratio, 4),
            },
            "balance_sheet": {
                "total_assets": self.balance_sheet.total_assets,
                "total_liabilities": self.balance_sheet.total_liabilities,
                "current_net_worth": self.balance_sheet.current_net_worth,
                "emergency_fund_months": round(self.balance_sheet.emergency_fund_months, 2),
            },
            "goals": [
                {
                    "goal_id": g.goal_id,
                    "goal_name": g.goal_name,
                    "goal_type": g.goal_type,
                    "target_amount": g.target_amount,
                    "current_savings": g.current_savings,
                    "target_date": g.target_date,
                    "monthly_contribution": g.monthly_contribution,
                    "priority": g.priority,
                }
                for g in self.goals
            ],
            "derived": {
                "health_score": self.derived.health_score,
                "health_grade": self.derived.health_grade,
                "personality_type": self.derived.personality_type,
                "key_risk_level": self.derived.key_risk_level,
            },
        }


# ==============================================================================
# MUTABLE SESSION CONTEXT (Separated from FinancialState)
# ==============================================================================

@dataclass
class ConversationTurn:
    """
    Represents a single conversational turn in session history.
    """
    turn_id: str
    timestamp: str
    user_query: str
    orchestrator_intent: str
    final_response: str
    agents_invoked: Tuple[str, ...] = ()


@dataclass
class SessionContext:
    """
    Mutable session state container managed in Streamlit st.session_state.
    Kept strictly decoupled from the immutable FinancialState snapshot.
    """
    session_id: str
    active_page: str = "01_Dashboard"
    turn_count: int = 0
    history: List[ConversationTurn] = field(default_factory=list)
    ui_flags: Dict[str, Any] = field(default_factory=dict)

    def add_turn(
        self,
        user_query: str,
        orchestrator_intent: str,
        final_response: str,
        agents_invoked: Sequence[str] = (),
    ) -> ConversationTurn:
        """
        Appends a conversation turn to session history.
        """
        self.turn_count += 1
        turn = ConversationTurn(
            turn_id=f"turn-{self.turn_count:03d}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_query=user_query,
            orchestrator_intent=orchestrator_intent,
            final_response=final_response,
            agents_invoked=tuple(agents_invoked),
        )
        self.history.append(turn)
        # Keep sliding window of last 10 turns
        if len(self.history) > 10:
            self.history = self.history[-10:]
        return turn

    def get_recent_history_summary(self, max_turns: int = 3) -> str:
        """
        Returns a concise plain-text summary of recent conversation turns.
        """
        if not self.history:
            return "No previous conversation turns."
        recent = self.history[-max_turns:]
        lines = []
        for t in recent:
            lines.append(f"User: {t.user_query}")
            lines.append(f"Assistant: {t.final_response[:120]}...")
        return "\n".join(lines)

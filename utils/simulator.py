"""
Scenario Simulation Engine  (Module 7)
OOP-based engine for simulating 6 life-event scenarios and computing
their impact on Financial Health Score, Future Savings, Future Net Worth,
and Risk Level — with full before-vs-after comparison.

Classes:
    BaseScenario            - Abstract base; defines the comparison contract
    SalaryHikeScenario      - Models a salary increment
    JobLossScenario         - Models sudden income loss
    CarPurchaseScenario     - Models a car loan + upfront cost
    HomeLoanScenario        - Models a home purchase with mortgage
    MarriageExpenseScenario - Models a one-time wedding expense + lifestyle bump
    IncreaseSIPScenario     - Models an increase in monthly SIP
    ScenarioSimulator       - Legacy wrapper (preserved for Module 6 compat) + new run()

Forecasting note — two intentionally different projection methods coexist:

    FinancialSnapshot.future_net_worth_5y
        Purpose: relative what-if delta inside the Scenario Simulator.
        Horizon: 5 years, mathematical SIP+savings accumulation.
        Used by: pages/06_Scenario_Simulator.py (before/after comparison).

    FinancialPredictor.generate_trajectory()  [models/predictor.py]
        Purpose: absolute month-by-month forward forecast anchored by XGBoost.
        Horizon: 12 months compound growth trajectory.
        Used by: pages/05_Forecasting.py and utils/master_report.py.

    These serve different purposes and are intentionally kept separate.
    Do NOT unify them — the scenario projection needs to be self-contained
    and runnable without loading the XGBoost models.

Health Score note:
    FinancialSnapshot._compute_health_score() delegates to HealthScoreEngine
    (models/twin_engine.py), which is the SINGLE canonical implementation.
    This ensures the Scenario Simulator produces scores identical to those
    shown on the Financial Health page.
"""

from __future__ import annotations

import copy
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------------
# Health-score delegation helper
# ---------------------------------------------------------------------------
# FinancialSnapshot holds only the 7 fields that scenarios mutate. To compute
# a health score that is IDENTICAL to the main Financial Health page we create
# a minimal FinancialDigitalTwin proxy and pass it to HealthScoreEngine.
# Fields not tracked by the snapshot (insurance, loan totals) are set to 0.
# Since both the before and after snapshots use the same approach, the delta
# (after.health_score - before.health_score) accurately reflects the scenario
# impact. Insurance/debt components will score 0 for both sides equally.

def _compute_snapshot_health_score(snapshot: "FinancialSnapshot") -> float:
    """
    Computes the financial health score for a FinancialSnapshot by delegating
    to the canonical HealthScoreEngine.

    A minimal FinancialDigitalTwin proxy is constructed from the snapshot’s
    known fields. Fields unavailable in a scenario snapshot (life_insurance,
    health_insurance, loan_amount) default to 0.0 so that both the before and
    after snapshots are evaluated under identical assumptions, preserving
    accurate before/after deltas.

    Args:
        snapshot: A FinancialSnapshot instance (before or after a scenario).

    Returns:
        Overall health score as a float in [0.0, 100.0].
    """
    from models.twin_engine import FinancialDigitalTwin, HealthScoreEngine

    # Build a minimal demographics dict (income is the relevant demographic here)
    demographics = {
        "name":              "scenario_proxy",
        "age":               30,
        "city":              "",
        "occupation":        "",
        "monthly_income":    snapshot.monthly_income,
        "bonus":             0.0,
        "additional_income": 0.0,
    }
    # Build a minimal balance_sheet from the 7 snapshot fields.
    # Fields the snapshot does not track (insurance, individual loans) default
    # to 0.0 and affect both before and after equally, so deltas are correct.
    balance_sheet = {
        "net_worth":       snapshot.net_worth,
        "bank_savings":    snapshot.bank_savings,
        "fd_amount":       0.0,
        "emergency_fund":  snapshot.emergency_fund,
        "sip_amount":      snapshot.sip_amount,
        "mutual_funds":    0.0,
        "stocks":          0.0,
        "ppf_investment":  0.0,
        "nps_investment":  0.0,
        # Liabilities
        "loan_amount":     0.0,   # not tracked per-scenario (affects debt_level equally)
        "car_loan":        0.0,
        "home_loan":       0.0,
        "credit_card_debt":0.0,
        "monthly_emi":     snapshot.monthly_emi,
        # Insurance (not tracked per-scenario; both sides score 0 equally)
        "health_insurance":0.0,
        "life_insurance":  0.0,
        # Expenses (reconstructed from basic_expenses total; individual splits
        # not tracked — assign to rent as a single bucket)
        "rent":            snapshot.basic_expenses,
        "groceries":       0.0,
        "utilities":       0.0,
        "transport":       0.0,
        "food_delivery":   0.0,
        "entertainment":   0.0,
        "shopping":        0.0,
        # Goal
        "goal_type":       "",
        "goal_amount":     0.0,
    }

    proxy_twin = FinancialDigitalTwin(
        user_id      = "_scenario_proxy_",
        demographics = demographics,
        balance_sheet= balance_sheet,
    )
    result = HealthScoreEngine(proxy_twin).compute_overall_health_score()
    return result["overall_score"]



# ══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FinancialSnapshot:
    """
    Immutable snapshot of key financial metrics at a point in time.
    Computed from a FinancialDigitalTwin or a mutated copy thereof.
    """
    monthly_income:   float
    basic_expenses:   float
    monthly_emi:      float
    sip_amount:       float
    bank_savings:     float
    net_worth:        float
    emergency_fund:   float

    # Computed on __post_init__
    monthly_surplus:  float = field(init=False)
    savings_rate:     float = field(init=False)
    emi_ratio:        float = field(init=False)
    health_score:     float = field(init=False)
    risk_level:       str   = field(init=False)
    future_savings_12m: float = field(init=False)
    future_net_worth_5y: float = field(init=False)

    def __post_init__(self):
        from utils.validators import sanitize_numeric
        self.monthly_income = sanitize_numeric(self.monthly_income, min_val=0.0)
        self.basic_expenses = sanitize_numeric(self.basic_expenses, min_val=0.0)
        self.monthly_emi    = sanitize_numeric(self.monthly_emi, min_val=0.0)
        self.sip_amount     = sanitize_numeric(self.sip_amount, min_val=0.0)
        self.bank_savings   = sanitize_numeric(self.bank_savings, min_val=0.0)
        self.net_worth      = sanitize_numeric(self.net_worth, min_val=-1_000_000_000.0)
        self.emergency_fund = sanitize_numeric(self.emergency_fund, min_val=0.0)

        self.monthly_surplus  = max(self.monthly_income - self.basic_expenses - self.monthly_emi, 0.0)
        self.savings_rate     = (self.monthly_surplus / self.monthly_income) if self.monthly_income > 0 else 0.0
        self.emi_ratio        = (self.monthly_emi / self.monthly_income) if self.monthly_income > 0 else 0.0
        self.health_score     = self._compute_health_score()
        self.risk_level       = self._compute_risk_level()
        self.future_savings_12m  = self._project_savings(months=12)
        self.future_net_worth_5y = self._project_net_worth(years=5)

    # ------------------------------------------------------------------
    # Internal computation helpers
    # ------------------------------------------------------------------

    def _compute_health_score(self) -> float:
        """
        Delegates to the canonical HealthScoreEngine via a minimal proxy twin.

        This ensures the Scenario Simulator produces health scores that are
        IDENTICAL to those shown on the Financial Health page — there is only
        ONE health-score implementation in this codebase.

        See _compute_snapshot_health_score() for the delegation mechanics and
        a full explanation of how missing snapshot fields are handled.
        """
        return _compute_snapshot_health_score(self)

    def _compute_risk_level(self) -> str:
        if self.monthly_income <= 0:
            return "Critical"
        if self.emi_ratio > 0.50 or self.savings_rate < 0:
            return "Critical"
        elif self.emi_ratio > 0.35 or self.savings_rate < 0.05:
            return "High"
        elif self.emi_ratio > 0.20 or self.savings_rate < 0.15:
            return "Moderate"
        else:
            return "Low"

    def _project_savings(self, months: int = 12, annual_savings_interest: float = 0.035) -> float:
        """Projects accumulated savings over `months` at savings account rate."""
        monthly_rate = annual_savings_interest / 12
        fv = self.bank_savings
        for _ in range(months):
            fv = fv * (1 + monthly_rate) + self.monthly_surplus
        return round(fv, 2)

    def _project_net_worth(self, years: int = 5, investment_return: float = 0.12, inflation: float = 0.06) -> float:
        """Projects net worth over `years` using SIP future value + savings accumulation."""
        months = years * 12
        monthly_inv = investment_return / 12
        monthly_inf = inflation / 12

        # SIP corpus FV
        if monthly_inv > 0:
            sip_fv = self.sip_amount * (((1 + monthly_inv) ** months - 1) / monthly_inv)
        else:
            sip_fv = self.sip_amount * months

        # Real surplus eroded by inflation
        real_surplus = self.monthly_surplus * max(1 - monthly_inf * months, 0)
        savings_acc  = real_surplus * months

        return round(max(self.net_worth + savings_acc + sip_fv, 0.0), 2)

    @classmethod
    def from_twin(cls, twin) -> "FinancialSnapshot":
        """Creates a FinancialSnapshot from a FinancialDigitalTwin instance."""
        return cls(
            monthly_income  = float(twin.total_income),
            basic_expenses  = float(twin.basic_expenses),
            monthly_emi     = float(twin.monthly_emi),
            sip_amount      = float(twin.sip_amount),
            bank_savings    = float(twin.bank_savings),
            net_worth       = float(twin.net_worth),
            emergency_fund  = float(twin.emergency_fund),
        )


@dataclass
class ScenarioResult:
    """
    Holds the before + after comparison for a given life-event scenario.
    """
    scenario_name:  str
    before:         FinancialSnapshot
    after:          FinancialSnapshot
    extra_details:  Dict[str, Any] = field(default_factory=dict)

    # Computed deltas
    @property
    def delta_health_score(self) -> float:
        return round(self.after.health_score - self.before.health_score, 2)

    @property
    def delta_future_savings(self) -> float:
        return round(self.after.future_savings_12m - self.before.future_savings_12m, 2)

    @property
    def delta_net_worth_5y(self) -> float:
        return round(self.after.future_net_worth_5y - self.before.future_net_worth_5y, 2)

    @property
    def risk_changed(self) -> bool:
        return self.before.risk_level != self.after.risk_level

    def comparison_df(self) -> pd.DataFrame:
        """Returns a DataFrame comparing key metrics before and after."""
        metrics = {
            "Metric": [
                "Monthly Income (₹)",
                "Monthly Expenses (₹)",
                "Monthly EMI (₹)",
                "Monthly SIP (₹)",
                "Monthly Surplus (₹)",
                "Savings Rate (%)",
                "Bank Savings (₹)",
                "Emergency Fund (₹)",
                "Net Worth (₹)",
                "Health Score",
                "Future Savings 12M (₹)",
                "Future Net Worth 5Y (₹)",
                "Risk Level",
            ],
            "Before": [
                f"₹{self.before.monthly_income:,.0f}",
                f"₹{self.before.basic_expenses:,.0f}",
                f"₹{self.before.monthly_emi:,.0f}",
                f"₹{self.before.sip_amount:,.0f}",
                f"₹{self.before.monthly_surplus:,.0f}",
                f"{self.before.savings_rate * 100:.1f}%",
                f"₹{self.before.bank_savings:,.0f}",
                f"₹{self.before.emergency_fund:,.0f}",
                f"₹{self.before.net_worth:,.0f}",
                f"{self.before.health_score:.1f} / 100",
                f"₹{self.before.future_savings_12m:,.0f}",
                f"₹{self.before.future_net_worth_5y:,.0f}",
                self.before.risk_level,
            ],
            "After": [
                f"₹{self.after.monthly_income:,.0f}",
                f"₹{self.after.basic_expenses:,.0f}",
                f"₹{self.after.monthly_emi:,.0f}",
                f"₹{self.after.sip_amount:,.0f}",
                f"₹{self.after.monthly_surplus:,.0f}",
                f"{self.after.savings_rate * 100:.1f}%",
                f"₹{self.after.bank_savings:,.0f}",
                f"₹{self.after.emergency_fund:,.0f}",
                f"₹{self.after.net_worth:,.0f}",
                f"{self.after.health_score:.1f} / 100",
                f"₹{self.after.future_savings_12m:,.0f}",
                f"₹{self.after.future_net_worth_5y:,.0f}",
                self.after.risk_level,
            ],
        }
        return pd.DataFrame(metrics)


# ══════════════════════════════════════════════════════════════════════════════
# Base Scenario
# ══════════════════════════════════════════════════════════════════════════════

class BaseScenario(ABC):
    """
    Abstract base class for all life-event scenarios.
    Subclasses implement `apply()` which mutates the snapshot fields
    and returns a ScenarioResult with before/after comparison.
    """

    def __init__(self, twin):
        self.twin = twin
        self._before = FinancialSnapshot.from_twin(twin)

    def _make_after(self, **overrides) -> FinancialSnapshot:
        """Creates an 'after' snapshot by applying field overrides to the before state."""
        return FinancialSnapshot(
            monthly_income = overrides.get("monthly_income", self._before.monthly_income),
            basic_expenses = overrides.get("basic_expenses", self._before.basic_expenses),
            monthly_emi    = overrides.get("monthly_emi",    self._before.monthly_emi),
            sip_amount     = overrides.get("sip_amount",     self._before.sip_amount),
            bank_savings   = overrides.get("bank_savings",   self._before.bank_savings),
            net_worth      = overrides.get("net_worth",      self._before.net_worth),
            emergency_fund = overrides.get("emergency_fund", self._before.emergency_fund),
        )

    @abstractmethod
    def apply(self, **params) -> ScenarioResult:
        """Apply the scenario with given parameters and return a ScenarioResult."""
        ...


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 1: Salary Hike
# ══════════════════════════════════════════════════════════════════════════════

class SalaryHikeScenario(BaseScenario):
    """
    Models a salary increment as a percentage increase.
    Income rises → surplus increases → health score improves → risk drops.
    """

    def apply(self, hike_percent: float = 20.0) -> ScenarioResult:
        """
        Args:
            hike_percent: Salary increase as a percentage (e.g. 20 = 20% hike).
        """
        multiplier     = 1 + (hike_percent / 100)
        new_income     = self._before.monthly_income * multiplier
        income_delta   = new_income - self._before.monthly_income
        new_net_worth  = self._before.net_worth + income_delta  # Immediate impact

        after = self._make_after(
            monthly_income = new_income,
            net_worth      = new_net_worth,
        )

        return ScenarioResult(
            scenario_name = f"Salary Hike (+{hike_percent:.0f}%)",
            before        = self._before,
            after         = after,
            extra_details = {
                "hike_percent":   hike_percent,
                "monthly_gain":   round(income_delta, 2),
                "annual_gain":    round(income_delta * 12, 2),
            }
        )


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 2: Job Loss
# ══════════════════════════════════════════════════════════════════════════════

class JobLossScenario(BaseScenario):
    """
    Models sudden complete income loss.
    Income drops to 0. Emergency fund and bank savings drain over time.
    After re-employment (duration_months), income is restored (optionally reduced).
    """

    def apply(self, duration_months: int = 6, recovery_salary_pct: float = 100.0) -> ScenarioResult:
        """
        Args:
            duration_months:     How long the person is unemployed.
            recovery_salary_pct: Income level after re-employment (% of original).
        """
        monthly_obligation = self._before.basic_expenses + self._before.monthly_emi
        ef  = self._before.emergency_fund
        bs  = self._before.bank_savings

        timeline      = []
        total_deficit = 0.0

        for m in range(1, duration_months + 1):
            if ef >= monthly_obligation:
                ef -= monthly_obligation
                status = "EF covering expenses"
            elif ef > 0:
                shortfall = monthly_obligation - ef
                ef = 0.0
                if bs >= shortfall:
                    bs -= shortfall
                    status = "EF depleted — drawing from savings"
                else:
                    total_deficit += shortfall - bs
                    bs = 0.0
                    status = "Cash deficit"
            elif bs >= monthly_obligation:
                bs -= monthly_obligation
                status = "Drawing from bank savings"
            elif bs > 0:
                total_deficit += monthly_obligation - bs
                bs = 0.0
                status = "Cash deficit"
            else:
                total_deficit += monthly_obligation
                status = "Cash deficit — all reserves exhausted"

            timeline.append({
                "Month": m,
                "Emergency Fund": round(ef, 2),
                "Bank Savings":   round(bs, 2),
                "Monthly Burn":   round(monthly_obligation, 2),
                "Status":         status,
            })

        # Post job-loss: income at recovery level
        new_income = self._before.monthly_income * (recovery_salary_pct / 100.0)

        after = self._make_after(
            monthly_income = new_income,
            bank_savings   = bs,
            emergency_fund = ef,
            net_worth      = self._before.net_worth - (self._before.bank_savings - bs) - (self._before.emergency_fund - ef),
        )

        return ScenarioResult(
            scenario_name = f"Job Loss ({duration_months}M)",
            before        = self._before,
            after         = after,
            extra_details = {
                "duration_months":     duration_months,
                "total_deficit":       round(total_deficit, 2),
                "ef_remaining":        round(ef, 2),
                "bs_remaining":        round(bs, 2),
                "recovery_salary_pct": recovery_salary_pct,
                "timeline_df":         pd.DataFrame(timeline),
            }
        )


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 3: Car Purchase
# ══════════════════════════════════════════════════════════════════════════════

class CarPurchaseScenario(BaseScenario):
    """
    Models purchasing a car with an upfront payment + car loan.
    Increases EMI, reduces liquid savings, and adds a depreciating asset.
    """

    def apply(
        self,
        car_price:         float = 800000,
        downpayment_pct:   float = 0.20,
        loan_rate:         float = 0.085,
        tenure_years:      int   = 5,
    ) -> ScenarioResult:
        """
        Args:
            car_price:        Total on-road price of the car (₹).
            downpayment_pct:  Fraction paid upfront (default 20%).
            loan_rate:        Annual car loan interest rate (default 8.5%).
            tenure_years:     Loan tenure in years (default 5).
        """
        downpayment = car_price * downpayment_pct
        loan_amount = car_price - downpayment

        # EMI using reducing balance formula
        monthly_rate = loan_rate / 12
        n = tenure_years * 12
        if monthly_rate > 0 and n > 0:
            new_car_emi = loan_amount * monthly_rate * (1 + monthly_rate) ** n / ((1 + monthly_rate) ** n - 1)
        else:
            new_car_emi = loan_amount / max(n, 1)

        new_emi         = self._before.monthly_emi + new_car_emi
        new_bank_savings = max(self._before.bank_savings - downpayment, 0.0)

        # Car depreciates ~15% per year; net worth impact = car value - loan
        car_current_value = car_price * 0.85  # After 1 year
        net_worth_impact  = car_current_value - loan_amount
        new_net_worth     = self._before.net_worth + net_worth_impact - downpayment

        after = self._make_after(
            monthly_emi  = new_emi,
            bank_savings = new_bank_savings,
            net_worth    = new_net_worth,
        )

        return ScenarioResult(
            scenario_name = "Car Purchase",
            before        = self._before,
            after         = after,
            extra_details = {
                "car_price":         car_price,
                "downpayment":       round(downpayment, 2),
                "loan_amount":       round(loan_amount, 2),
                "new_car_emi":       round(new_car_emi, 2),
                "total_emi_after":   round(new_emi, 2),
                "bank_savings_left": round(new_bank_savings, 2),
                "can_afford":        downpayment <= self._before.bank_savings,
            }
        )


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 4: Home Loan
# ══════════════════════════════════════════════════════════════════════════════

class HomeLoanScenario(BaseScenario):
    """
    Models buying a home with a mortgage.
    Large EMI commitment + stamp duty depletion + property asset addition.
    """

    def apply(
        self,
        purchase_price:   float = 5000000,
        downpayment_pct:  float = 0.20,
        home_loan_rate:   float = 0.085,
        tenure_years:     int   = 20,
    ) -> ScenarioResult:
        """
        Args:
            purchase_price:   Property price (₹).
            downpayment_pct:  Fraction paid upfront (default 20%).
            home_loan_rate:   Annual loan interest rate (default 8.5%).
            tenure_years:     Loan tenure in years (default 20).
        """
        stamp_duty    = purchase_price * 0.07   # ~7% stamp + registration
        downpayment   = purchase_price * downpayment_pct
        total_upfront = downpayment + stamp_duty
        loan_amount   = purchase_price - downpayment

        monthly_rate = home_loan_rate / 12
        n = tenure_years * 12
        if monthly_rate > 0 and n > 0:
            new_emi = loan_amount * monthly_rate * (1 + monthly_rate) ** n / ((1 + monthly_rate) ** n - 1)
        else:
            new_emi = loan_amount / max(n, 1)

        new_total_emi    = self._before.monthly_emi + new_emi
        new_bank_savings = max(self._before.bank_savings - total_upfront, 0.0)

        # Property appreciates ~5% per year; net worth = property + assets - liabilities
        new_net_worth = (
            self._before.net_worth
            - total_upfront              # Cash out
            + purchase_price             # Property added at cost
            - loan_amount                # Loan liability added
        )

        after = self._make_after(
            monthly_emi  = new_total_emi,
            bank_savings = new_bank_savings,
            net_worth    = new_net_worth,
        )

        return ScenarioResult(
            scenario_name = "Home Loan",
            before        = self._before,
            after         = after,
            extra_details = {
                "purchase_price":     purchase_price,
                "downpayment":        round(downpayment, 2),
                "stamp_duty":         round(stamp_duty, 2),
                "total_upfront":      round(total_upfront, 2),
                "home_loan_amount":   round(loan_amount, 2),
                "new_home_emi":       round(new_emi, 2),
                "total_emi_after":    round(new_total_emi, 2),
                "bank_savings_left":  round(new_bank_savings, 2),
                "can_afford":         total_upfront <= (self._before.bank_savings + self._before.fd_amount if hasattr(self._before, "fd_amount") else self._before.bank_savings),
            }
        )

    @property
    def _before_fd(self) -> float:
        return float(getattr(self.twin, "fd_amount", 0.0))


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 5: Marriage Expense
# ══════════════════════════════════════════════════════════════════════════════

class MarriageExpenseScenario(BaseScenario):
    """
    Models a one-time marriage expense + an ongoing lifestyle expense increase.
    Immediate savings shock + higher monthly expenses going forward.
    """

    def apply(
        self,
        one_time_cost:      float = 500000,
        monthly_expense_hike: float = 10000,
    ) -> ScenarioResult:
        """
        Args:
            one_time_cost:         Total wedding + honeymoon expense (₹).
            monthly_expense_hike:  Increase in monthly household expenses post-marriage (₹).
        """
        new_bank_savings = max(self._before.bank_savings - one_time_cost, 0.0)
        ef_dip = max(0, one_time_cost - self._before.bank_savings)
        new_emergency_fund = max(self._before.emergency_fund - ef_dip, 0.0)

        new_expenses  = self._before.basic_expenses + monthly_expense_hike
        new_net_worth = self._before.net_worth - one_time_cost  # One-time spend

        after = self._make_after(
            basic_expenses = new_expenses,
            bank_savings   = new_bank_savings,
            emergency_fund = new_emergency_fund,
            net_worth      = new_net_worth,
        )

        return ScenarioResult(
            scenario_name = "Marriage Expense",
            before        = self._before,
            after         = after,
            extra_details = {
                "one_time_cost":        one_time_cost,
                "monthly_expense_hike": monthly_expense_hike,
                "annual_extra_expense": monthly_expense_hike * 12,
                "savings_after_wedding": round(new_bank_savings, 2),
            }
        )


# ══════════════════════════════════════════════════════════════════════════════
# Scenario 6: Increase SIP
# ══════════════════════════════════════════════════════════════════════════════

class IncreaseSIPScenario(BaseScenario):
    """
    Models increasing the monthly SIP investment.
    Reduces monthly surplus / liquid cash but improves long-term net worth.
    """

    def apply(self, new_sip_amount: float = None, sip_increase_pct: float = 50.0) -> ScenarioResult:
        """
        Args:
            new_sip_amount:    Absolute new SIP amount (₹). If None, uses sip_increase_pct.
            sip_increase_pct:  Percentage increase in SIP (used if new_sip_amount is None).
        """
        if new_sip_amount is None:
            new_sip_amount = self._before.sip_amount * (1 + sip_increase_pct / 100)

        sip_delta = new_sip_amount - self._before.sip_amount

        after = self._make_after(
            sip_amount = new_sip_amount,
            # SIP increase comes from surplus; bank_savings stays same (it's a flow)
        )

        return ScenarioResult(
            scenario_name = f"Increase SIP (+{sip_increase_pct:.0f}%)" if new_sip_amount is None else "Increase SIP",
            before        = self._before,
            after         = after,
            extra_details = {
                "old_sip":          self._before.sip_amount,
                "new_sip":          round(new_sip_amount, 2),
                "monthly_increase": round(sip_delta, 2),
                "annual_increase":  round(sip_delta * 12, 2),
            }
        )


# ══════════════════════════════════════════════════════════════════════════════
# Scenario Registry + Runner
# ══════════════════════════════════════════════════════════════════════════════

SCENARIO_REGISTRY: Dict[str, type] = {
    "Salary Hike":          SalaryHikeScenario,
    "Job Loss":             JobLossScenario,
    "Car Purchase":         CarPurchaseScenario,
    "Home Loan":            HomeLoanScenario,
    "Marriage Expense":     MarriageExpenseScenario,
    "Increase SIP":         IncreaseSIPScenario,
}


# ══════════════════════════════════════════════════════════════════════════════
# Legacy Wrapper (Module 6 compatibility — used by old tab2 / tab3 code)
# ══════════════════════════════════════════════════════════════════════════════

class ScenarioSimulator:
    """
    Backward-compatible wrapper used by the legacy Forecast page tabs.
    Also exposes `run_scenario()` for the new Module 7 tab.
    """

    def __init__(self, twin):
        self.twin = twin

    def run_scenario(self, scenario_name: str, **params) -> ScenarioResult:
        """
        Runs a named scenario and returns a ScenarioResult.

        Args:
            scenario_name: Key in SCENARIO_REGISTRY.
            **params:      Keyword arguments forwarded to the scenario's apply().
        """
        cls = SCENARIO_REGISTRY[scenario_name]
        scenario = cls(self.twin)
        return scenario.apply(**params)

    # ── Legacy methods (preserved for Module 6 page compatibility) ──────────

    def simulate_job_loss(self, duration_months: int = 6) -> Dict[str, Any]:
        result = JobLossScenario(self.twin).apply(duration_months=duration_months)
        return {
            "monthly_timeline":    result.extra_details["timeline_df"].to_dict("records"),
            "exhaustion_month_ef": next(
                (r["Month"] for r in result.extra_details["timeline_df"].to_dict("records")
                 if r["Emergency Fund"] == 0), -1),
            "exhaustion_month_bs": next(
                (r["Month"] for r in result.extra_details["timeline_df"].to_dict("records")
                 if r["Bank Savings"] == 0), -1),
            "total_deficit":       result.extra_details["total_deficit"],
            "df":                  result.extra_details["timeline_df"],
        }

    def simulate_home_purchase(
        self,
        purchase_price: float,
        downpayment_pct: float = 0.20,
        home_loan_rate: float = 0.085,
        tenure_years: int = 20,
    ) -> Dict[str, Any]:
        result = HomeLoanScenario(self.twin).apply(
            purchase_price  = purchase_price,
            downpayment_pct = downpayment_pct,
            home_loan_rate  = home_loan_rate,
            tenure_years    = tenure_years,
        )
        d = result.extra_details
        new_monthly_surplus = result.after.monthly_surplus
        old_monthly_surplus = result.before.monthly_surplus
        rows = []
        bs = d["bank_savings_left"]
        for m in range(1, 13):
            rows.append({
                "Month":           m,
                "Monthly Surplus": round(max(new_monthly_surplus, 0) * m, 2),
                "Bank Savings":    round(bs + max(new_monthly_surplus, 0) * m, 2),
            })
        return {
            "purchase_price":      d["purchase_price"],
            "downpayment":         d["downpayment"],
            "stamp_duty":          d["stamp_duty"],
            "total_upfront_cash":  d["total_upfront"],
            "home_loan_amount":    d["home_loan_amount"],
            "new_emi":             d["new_home_emi"],
            "total_emi_after":     d["total_emi_after"],
            "new_monthly_surplus": round(new_monthly_surplus, 2),
            "old_monthly_surplus": round(old_monthly_surplus, 2),
            "surplus_change":      round(new_monthly_surplus - old_monthly_surplus, 2),
            "remaining_savings":   d["bank_savings_left"],
            "can_afford":          d["can_afford"],
            "df":                  pd.DataFrame(rows),
        }

    def simulate_market_returns_and_inflation(
        self,
        years: int = 5,
        inflation_rate: float = 0.06,
        investment_growth: float = 0.12,
    ) -> pd.DataFrame:
        scenarios = {
            "Base Case":            (investment_growth,        inflation_rate),
            "Optimistic Scenario":  (investment_growth + 0.04, inflation_rate - 0.015),
            "Pessimistic Scenario": (investment_growth - 0.05, inflation_rate + 0.02),
        }
        monthly_surplus = max(
            self.twin.total_income - self.twin.basic_expenses - self.twin.monthly_emi, 0
        )
        initial_corpus = self.twin.sip_amount
        base_nw        = float(self.twin.net_worth)

        rows = []
        for y in range(1, years + 1):
            row = {"Year": y}
            for name, (inv_rate, inf_rate) in scenarios.items():
                monthly_inv = inv_rate / 12
                monthly_inf = inf_rate / 12
                months      = y * 12
                if monthly_inv > 0:
                    sip_fv = initial_corpus * (((1 + monthly_inv) ** months - 1) / monthly_inv)
                else:
                    sip_fv = initial_corpus * months
                real_surplus = monthly_surplus * max(1 - monthly_inf * months, 0)
                savings_acc  = real_surplus * months
                nw = base_nw + savings_acc + sip_fv
                row[name] = round(max(nw, 0), 2)
            rows.append(row)
        return pd.DataFrame(rows)

    def run_monte_carlo(
        self,
        years: int = 5,
        num_simulations: int = 1000,
        annual_return_mean: float = 0.12,
        annual_return_std: float = 0.18,
    ) -> pd.DataFrame:
        if not (1 <= years <= 30):
            raise ValueError("Simulation horizon must be between 1 and 30 years.")
        if not (100 <= num_simulations <= 5000):
            raise ValueError("Simulations count must be between 100 and 5000.")
        if annual_return_mean < -0.9 or annual_return_mean > 2.0:
            raise ValueError("Mean return must be between -90% and 200%.")
        if annual_return_std < 0 or annual_return_std > 1.0:
            raise ValueError("Return volatility standard deviation must be between 0% and 100%.")

        months        = years * 12
        monthly_mu    = annual_return_mean / 12
        monthly_sigma = annual_return_std / (12 ** 0.5)
        monthly_surplus = max(
            self.twin.total_income - self.twin.basic_expenses - self.twin.monthly_emi, 0
        )
        sip    = float(self.twin.sip_amount)
        base_nw = float(self.twin.net_worth)

        rng = np.random.default_rng(seed=42)
        monthly_returns = rng.normal(monthly_mu, monthly_sigma, size=(num_simulations, months))

        all_nw = np.zeros((num_simulations, months))
        for sim in range(num_simulations):
            corpus = 0.0
            nw_t   = base_nw
            for m in range(months):
                r      = monthly_returns[sim, m]
                corpus = (corpus + sip) * (1 + r)
                nw_t  += monthly_surplus
                all_nw[sim, m] = nw_t + corpus

        rows = []
        for m in range(months):
            col = all_nw[:, m]
            rows.append({
                "Month": m + 1,
                "p10":   round(float(np.percentile(col, 10)), 2),
                "p25":   round(float(np.percentile(col, 25)), 2),
                "p50":   round(float(np.percentile(col, 50)), 2),
                "p75":   round(float(np.percentile(col, 75)), 2),
                "p90":   round(float(np.percentile(col, 90)), 2),
            })
        return pd.DataFrame(rows)

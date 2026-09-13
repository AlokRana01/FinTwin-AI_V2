"""
Financial Digital Twin & Health Score Engine
Includes the FinancialDigitalTwin class representing the OOP model of the user's financial state
and calculates the Indian-contextualized Financial Health Score (0-100).

Module 4 additions:
  - create_twin()  : classmethod factory to instantiate a twin from profile dicts
  - update_twin()  : in-place field update with automatic recomputation of derived values
  - export_twin()  : full JSON serialization with 6 structured sub-profiles
  - risk_level     : property returning Low / Moderate / High / Critical
"""

import json
import functools
import copy
from datetime import datetime
from typing import Dict, Any

from config import HEALTH_SCORE_WEIGHTS
from utils.validators import sanitize_numeric, sanitize_text, is_finite_number

class FinancialDigitalTwin:
    """
    OOP Representation of a user's financial twin state.
    """
    def __init__(self, user_id: str, demographics: Dict[str, Any], balance_sheet: Dict[str, Any]):
        self.user_id = str(user_id)
        
        # Demographics
        self.name = sanitize_text(demographics.get("name", "Unknown"), max_len=80)
        try:
            self.age = int(sanitize_numeric(demographics.get("age", 25), default=25, min_val=18, max_val=100))
        except (ValueError, TypeError):
            self.age = 25
        self.city = sanitize_text(demographics.get("city", ""), max_len=100)
        self.occupation = sanitize_text(demographics.get("occupation", ""), max_len=100)
        self.monthly_income = sanitize_numeric(demographics.get("monthly_income", 0.0), min_val=0.0)
        self.bonus = sanitize_numeric(demographics.get("bonus", 0.0), min_val=0.0)
        self.additional_income = sanitize_numeric(demographics.get("additional_income", 0.0), min_val=0.0)
        self.avatar_id = sanitize_text(str(demographics.get("avatar_id") or "avatar_01"), max_len=50) or "avatar_01"
        
        # Balance Sheet Assets
        self.net_worth = sanitize_numeric(balance_sheet.get("net_worth", 0.0), min_val=-1_000_000_000.0)
        self.bank_savings = sanitize_numeric(balance_sheet.get("bank_savings", 0.0), min_val=0.0)
        self.fd_amount = sanitize_numeric(balance_sheet.get("fd_amount", 0.0), min_val=0.0)
        self.emergency_fund = sanitize_numeric(balance_sheet.get("emergency_fund", 0.0), min_val=0.0)
        self.sip_amount = sanitize_numeric(balance_sheet.get("sip_amount", 0.0), min_val=0.0)
        self.mutual_funds = sanitize_numeric(balance_sheet.get("mutual_funds", 0.0), min_val=0.0)
        self.stocks = sanitize_numeric(balance_sheet.get("stocks", 0.0), min_val=0.0)
        self.ppf_investment = sanitize_numeric(balance_sheet.get("ppf_investment", 0.0), min_val=0.0)
        self.nps_investment = sanitize_numeric(balance_sheet.get("nps_investment", 0.0), min_val=0.0)
        
        # Liabilities
        self.loan_amount = sanitize_numeric(balance_sheet.get("loan_amount", 0.0), min_val=0.0)
        self.car_loan = sanitize_numeric(balance_sheet.get("car_loan", 0.0), min_val=0.0)
        self.home_loan = sanitize_numeric(balance_sheet.get("home_loan", 0.0), min_val=0.0)
        self.credit_card_debt = sanitize_numeric(balance_sheet.get("credit_card_debt", 0.0), min_val=0.0)
        self.monthly_emi = sanitize_numeric(balance_sheet.get("monthly_emi", 0.0), min_val=0.0)
        
        # Insurance
        self.health_insurance = sanitize_numeric(balance_sheet.get("health_insurance", 0.0), min_val=0.0)
        self.life_insurance = sanitize_numeric(balance_sheet.get("life_insurance", 0.0), min_val=0.0)
        
        # Expenses
        self.rent = sanitize_numeric(balance_sheet.get("rent", 0.0), min_val=0.0)
        self.groceries = sanitize_numeric(balance_sheet.get("groceries", 0.0), min_val=0.0)
        self.utilities = sanitize_numeric(balance_sheet.get("utilities", 0.0), min_val=0.0)
        self.transport = sanitize_numeric(balance_sheet.get("transport", 0.0), min_val=0.0)
        self.food_delivery = sanitize_numeric(balance_sheet.get("food_delivery", 0.0), min_val=0.0)
        self.entertainment = sanitize_numeric(balance_sheet.get("entertainment", 0.0), min_val=0.0)
        self.shopping = sanitize_numeric(balance_sheet.get("shopping", 0.0), min_val=0.0)
        
        # Goal
        self.goal_type = sanitize_text(balance_sheet.get("goal_type", ""), max_len=50)
        self.goal_amount = sanitize_numeric(balance_sheet.get("goal_amount", 0.0), min_val=0.0)
        
        # Computed Properties
        self.total_income = self.monthly_income + self.additional_income
        self.basic_expenses = (
            self.rent + self.groceries + self.utilities + self.transport +
            self.food_delivery + self.entertainment + self.shopping
        )

    def get_summary_metrics(self) -> Dict[str, Any]:
        """
        Returns structured metrics summarizing the current state.
        """
        return {
            "total_income": self.total_income,
            "basic_expenses": self.basic_expenses,
            "monthly_emi": self.monthly_emi,
            "net_worth": self.net_worth,
            "savings_rate": (self.total_income - self.basic_expenses - self.monthly_emi) / self.total_income if self.total_income > 0 else 0.0,
            "emergency_fund_months": self.emergency_fund / (self.basic_expenses + self.monthly_emi) if (self.basic_expenses + self.monthly_emi) > 0 else 0.0,
            "loan_amount": self.loan_amount
        }

    def _recompute_derived(self):
        """
        Recomputes all derived / aggregated attributes from raw fields.
        Called internally after any field mutation.
        """
        self.total_income = self.monthly_income + self.additional_income
        self.basic_expenses = (
            self.rent + self.groceries + self.utilities + self.transport +
            self.food_delivery + self.entertainment + self.shopping
        )

    # ------------------------------------------------------------------
    # Module 4 Methods
    # ------------------------------------------------------------------

    @classmethod
    def create_twin(cls, user_id: str, demographics: Dict[str, Any], balance_sheet: Dict[str, Any]) -> "FinancialDigitalTwin":
        """
        Factory classmethod to create a new FinancialDigitalTwin from profile dicts.

        Args:
            user_id      : Unique identifier for the user.
            demographics : Dict containing name, age, city, occupation, income fields.
            balance_sheet: Dict containing all asset, liability, and expense fields.

        Returns:
            A fully initialised FinancialDigitalTwin instance.
        """
        return cls(user_id, demographics, balance_sheet)

    def update_twin(self, **kwargs) -> None:
        """
        Updates one or more fields on the twin in-place.
        Automatically recomputes derived properties (total_income, basic_expenses)
        after applying all updates.

        Example:
            twin.update_twin(monthly_income=90000, sip_amount=12000)

        Args:
            **kwargs: Field name -> new value pairs. Unknown keys are silently ignored.
        """
        _allowed_fields = {
            # Demographics
            "name", "age", "city", "occupation",
            "monthly_income", "bonus", "additional_income", "avatar_id",
            # Assets
            "net_worth", "bank_savings", "fd_amount", "emergency_fund",
            "sip_amount", "mutual_funds", "stocks", "ppf_investment", "nps_investment",
            # Liabilities
            "loan_amount", "car_loan", "home_loan", "credit_card_debt", "monthly_emi",
            # Insurance
            "health_insurance", "life_insurance",
            # Expenses
            "rent", "groceries", "utilities", "transport",
            "food_delivery", "entertainment", "shopping",
            # Goal
            "goal_type", "goal_amount"
        }
        for field, value in kwargs.items():
            if field in _allowed_fields:
                setattr(self, field, value)
        self._recompute_derived()

    @property
    def risk_level(self) -> str:
        """
        Derives a risk label from the user's debt-to-income ratio and savings rate.

        Risk Rules:
          - Critical  : EMI > 50% income  OR  savings_rate < 0
          - High      : EMI > 35% income  OR  savings_rate < 5%
          - Moderate  : EMI > 20% income  OR  savings_rate < 15%
          - Low       : Otherwise

        Returns:
            One of: 'Low', 'Moderate', 'High', 'Critical'
        """
        if self.total_income <= 0:
            return "Critical"
        emi_ratio = self.monthly_emi / self.total_income
        savings_rate = (self.total_income - self.basic_expenses - self.monthly_emi) / self.total_income

        if emi_ratio > 0.50 or savings_rate < 0:
            return "Critical"
        elif emi_ratio > 0.35 or savings_rate < 0.05:
            return "High"
        elif emi_ratio > 0.20 or savings_rate < 0.15:
            return "Moderate"
        else:
            return "Low"

    def export_twin(self) -> Dict[str, Any]:
        """
        Serializes the full Digital Twin to a structured, JSON-compatible dictionary
        with 6 sub-profiles as specified in Module 4.

        Sub-profiles:
            income_profile     : All income streams and totals
            expense_profile    : Itemised monthly expenses and totals
            debt_profile       : All loan/liability fields + ratios
            investment_profile : All asset/savings buckets + totals
            risk_level         : Derived risk label (Low/Moderate/High/Critical)
            health_score       : Full HealthScoreEngine output (score + grade + breakdown)

        Returns:
            A JSON-serializable Dict ready for json.dumps() or st.json().
        """
        # Compute health score inline (import here to avoid circular at module level)
        score_data = HealthScoreEngine(self).compute_overall_health_score()

        # ── Income Profile ─────────────────────────────────────────────
        income_profile = {
            "monthly_salary": round(self.monthly_income, 2),
            "annual_bonus": round(self.bonus, 2),
            "additional_monthly_income": round(self.additional_income, 2),
            "total_monthly_income": round(self.total_income, 2),
            "annual_income": round(self.total_income * 12, 2)
        }

        # ── Expense Profile ────────────────────────────────────────────
        expense_profile = {
            "rent": round(self.rent, 2),
            "groceries": round(self.groceries, 2),
            "utilities": round(self.utilities, 2),
            "transport": round(self.transport, 2),
            "food_delivery": round(self.food_delivery, 2),
            "entertainment": round(self.entertainment, 2),
            "shopping": round(self.shopping, 2),
            "total_monthly_expenses": round(self.basic_expenses, 2),
            "expense_to_income_ratio": round(
                self.basic_expenses / self.total_income, 4
            ) if self.total_income > 0 else 0.0
        }

        # ── Debt Profile ───────────────────────────────────────────────
        annual_income = self.total_income * 12
        debt_profile = {
            "total_outstanding_loans": round(self.loan_amount, 2),
            "home_loan": round(self.home_loan, 2),
            "car_loan": round(self.car_loan, 2),
            "credit_card_debt": round(self.credit_card_debt, 2),
            "monthly_emi": round(self.monthly_emi, 2),
            "emi_to_income_ratio": round(
                self.monthly_emi / self.total_income, 4
            ) if self.total_income > 0 else 0.0,
            "debt_to_annual_income_ratio": round(
                self.loan_amount / annual_income, 4
            ) if annual_income > 0 else 0.0
        }

        # ── Investment Profile ─────────────────────────────────────────
        total_investments = (
            self.bank_savings + self.fd_amount + self.emergency_fund +
            self.mutual_funds + self.stocks + self.ppf_investment + self.nps_investment
        )
        investment_profile = {
            "bank_savings": round(self.bank_savings, 2),
            "fixed_deposits": round(self.fd_amount, 2),
            "emergency_fund": round(self.emergency_fund, 2),
            "monthly_sip": round(self.sip_amount, 2),
            "mutual_funds_corpus": round(self.mutual_funds, 2),
            "stocks": round(self.stocks, 2),
            "ppf": round(self.ppf_investment, 2),
            "nps": round(self.nps_investment, 2),
            "total_invested_corpus": round(total_investments, 2),
            "investment_to_income_ratio": round(
                self.sip_amount / self.total_income, 4
            ) if self.total_income > 0 else 0.0
        }

        # ── Assemble Full Export ───────────────────────────────────────
        return {
            "meta": {
                "user_id": self.user_id,
                "name": self.name,
                "age": self.age,
                "city": self.city,
                "occupation": self.occupation,
                "exported_at": datetime.now().isoformat()
            },
            "income_profile": income_profile,
            "expense_profile": expense_profile,
            "debt_profile": debt_profile,
            "investment_profile": investment_profile,
            "risk_level": self.risk_level,
            "health_score": score_data
        }


@functools.lru_cache(maxsize=2048)
def _compute_health_score_from_features(
    total_income: float,
    basic_expenses: float,
    monthly_emi: float,
    emergency_fund: float,
    sip_amount: float,
    life_insurance: float,
    monthly_income: float,
    health_insurance: float,
    loan_amount: float,
    weights_tuple: tuple = (),
) -> Dict[str, Any]:
    """Pure, deterministic, memoized calculation of overall health score from numeric features."""
    # 1. Savings rate score
    if total_income <= 0:
        savings_rate_score = 0.0
        raw_savings_pct = 0.0
    else:
        surplus = total_income - basic_expenses - monthly_emi
        savings_rate = surplus / total_income
        savings_rate_score = float(min(max(savings_rate / 0.30, 0.0), 1.0) * 100.0)
        raw_savings_pct = round(savings_rate * 100, 2)

    # 2. EMI burden score
    if total_income <= 0:
        emi_burden_score = 100.0 if monthly_emi == 0 else 0.0
        raw_emi_pct = 0.0
    else:
        emi_ratio = monthly_emi / total_income
        raw_emi_pct = round(emi_ratio * 100, 2)
        if emi_ratio <= 0.35:
            emi_burden_score = 100.0
        elif emi_ratio >= 0.70:
            emi_burden_score = 0.0
        else:
            emi_burden_score = float(100.0 - ((emi_ratio - 0.35) / 0.35) * 100.0)

    # 3. Emergency fund score
    outflow = basic_expenses + monthly_emi
    if outflow <= 0:
        emergency_fund_score = 100.0 if emergency_fund > 0 else 0.0
        raw_ef_months = 0.0
    else:
        months_covered = emergency_fund / outflow
        raw_ef_months = round(months_covered, 1)
        emergency_fund_score = float(min(months_covered / 6.0, 1.0) * 100.0)

    # 4. Investment score
    if total_income <= 0:
        investment_score = 0.0
        raw_sip_pct = 0.0
    else:
        invest_ratio = sip_amount / total_income
        raw_sip_pct = round(invest_ratio * 100, 2)
        investment_score = float(min(invest_ratio / 0.15, 1.0) * 100.0)

    # 5. Insurance score
    if monthly_income <= 0:
        life_adequacy = 1.0 if life_insurance > 0 else 0.0
    else:
        life_adequacy = min(life_insurance / (monthly_income * 120.0), 1.0)
    health_adequacy = min(health_insurance / 500000.0, 1.0)
    raw_ins_pct = round((0.5 * life_adequacy + 0.5 * health_adequacy) * 100, 2) if (monthly_income > 0 or life_insurance > 0 or health_insurance > 0) else 0.0
    insurance_score = float((0.5 * life_adequacy + 0.5 * health_adequacy) * 100.0)

    # 6. Debt score
    if loan_amount == 0:
        debt_score = 100.0
        raw_debt_mult = 0.0
    elif total_income <= 0:
        debt_score = 0.0
        raw_debt_mult = 0.0
    else:
        annual_income = total_income * 12.0
        debt_to_income = loan_amount / annual_income
        raw_debt_mult = round(debt_to_income, 2)
        if debt_to_income <= 1.5:
            debt_score = 100.0
        elif debt_to_income >= 4.0:
            debt_score = 0.0
        else:
            debt_score = float(100.0 - ((debt_to_income - 1.5) / 2.5) * 100.0)

    # Pull weights from tuple or canonical config dict
    w = dict(weights_tuple) if weights_tuple else HEALTH_SCORE_WEIGHTS
    overall_score = (
        w["savings_rate"]       * savings_rate_score +
        w["emi_burden"]         * emi_burden_score +
        w["emergency_fund"]     * emergency_fund_score +
        w["investment_ratio"]   * investment_score +
        w["insurance_adequacy"] * insurance_score +
        w["debt_level"]         * debt_score
    )

    # Clamp strictly between 0 and 100
    overall_score = min(max(overall_score, 0.0), 100.0)

    # Grade mapping: A=80-100, B=60-79, C=40-59, D=<40
    if overall_score >= 80.0:
        grade = "A"
    elif overall_score >= 60.0:
        grade = "B"
    elif overall_score >= 40.0:
        grade = "C"
    else:
        grade = "D"

    return {
        "overall_score": round(overall_score, 2),
        "financial_grade": grade,
        "components": {
            "savings_rate": {
                "score": round(savings_rate_score, 2),
                "raw_value": raw_savings_pct,
                "benchmark": ">= 30%",
                "unit": "%",
                "weight": int(w["savings_rate"] * 100),
            },
            "emi_burden": {
                "score": round(emi_burden_score, 2),
                "raw_value": raw_emi_pct,
                "benchmark": "<= 35%",
                "unit": "%",
                "weight": int(w["emi_burden"] * 100),
            },
            "emergency_fund": {
                "score": round(emergency_fund_score, 2),
                "raw_value": raw_ef_months,
                "benchmark": ">= 6 months",
                "unit": " months",
                "weight": int(w["emergency_fund"] * 100),
            },
            "investment_ratio": {
                "score": round(investment_score, 2),
                "raw_value": raw_sip_pct,
                "benchmark": ">= 15%",
                "unit": "%",
                "weight": int(w["investment_ratio"] * 100),
            },
            "insurance_adequacy": {
                "score": round(insurance_score, 2),
                "raw_value": raw_ins_pct,
                "benchmark": "Term >= 10x annual; Health >= 5L",
                "unit": "% adequacy",
                "weight": int(w["insurance_adequacy"] * 100),
            },
            "debt_level": {
                "score": round(debt_score, 2),
                "raw_value": raw_debt_mult,
                "benchmark": "<= 1.5x annual income",
                "unit": "x annual",
                "weight": int(w["debt_level"] * 100),
            },
        },
    }


class HealthScoreEngine:
    """
    Computes a weighted, multi-dimensional health score based on financial ratios.
    """
    def __init__(self, twin: FinancialDigitalTwin):
        self.twin = twin

    def calculate_savings_rate_score(self) -> float:
        """
        Calculates savings rate score (optimal rate is 30%+ of income).
        """
        if self.twin.total_income <= 0:
            return 0.0
        surplus = self.twin.total_income - self.twin.basic_expenses - self.twin.monthly_emi
        savings_rate = surplus / self.twin.total_income
        score = min(max(savings_rate / 0.30, 0.0), 1.0) * 100.0
        return float(score)

    def calculate_emi_burden_score(self) -> float:
        """
        Calculates EMI burden score (optimal EMI-to-income is <= 35%).
        Decreases linearly to 0 at 70%+.
        """
        if self.twin.total_income <= 0:
            return 100.0 if self.twin.monthly_emi == 0 else 0.0
        emi_ratio = self.twin.monthly_emi / self.twin.total_income
        if emi_ratio <= 0.35:
            score = 100.0
        elif emi_ratio >= 0.70:
            score = 0.0
        else:
            score = 100.0 - ((emi_ratio - 0.35) / 0.35) * 100.0
        return float(score)

    def calculate_emergency_fund_score(self) -> float:
        """
        Calculates emergency fund score (optimal is 6+ months of expenses).
        """
        outflow = self.twin.basic_expenses + self.twin.monthly_emi
        if outflow <= 0:
            return 100.0 if self.twin.emergency_fund > 0 else 0.0
        months_covered = self.twin.emergency_fund / outflow
        score = min(months_covered / 6.0, 1.0) * 100.0
        return float(score)

    def calculate_investment_score(self) -> float:
        """
        Calculates SIP investment score (optimal investment-to-income is 15%+).
        """
        if self.twin.total_income <= 0:
            return 0.0
        invest_ratio = self.twin.sip_amount / self.twin.total_income
        score = min(invest_ratio / 0.15, 1.0) * 100.0
        return float(score)

    def calculate_insurance_score(self) -> float:
        """
        Calculates insurance adequacy (Life term cover >= 120x monthly salary; Health cover >= 5L).
        Each contributes 50% to the insurance score.
        """
        if self.twin.monthly_income <= 0:
            life_adequacy = 1.0 if self.twin.life_insurance > 0 else 0.0
        else:
            life_adequacy = min(self.twin.life_insurance / (self.twin.monthly_income * 120.0), 1.0)
            
        health_adequacy = min(self.twin.health_insurance / 500000.0, 1.0)
        score = (0.5 * life_adequacy + 0.5 * health_adequacy) * 100.0
        return float(score)

    def calculate_debt_score(self) -> float:
        """
        Calculates debt level score (optimal outstanding debt-to-annual income is <= 1.5x).
        Decreases linearly to 0 at 4.0x+.
        """
        if self.twin.loan_amount == 0:
            return 100.0
        if self.twin.total_income <= 0:
            return 0.0
            
        annual_income = self.twin.total_income * 12.0
        debt_to_income = self.twin.loan_amount / annual_income
        
        if debt_to_income <= 1.5:
            score = 100.0
        elif debt_to_income >= 4.0:
            score = 0.0
        else:
            score = 100.0 - ((debt_to_income - 1.5) / 2.5) * 100.0
        return float(score)

    def compute_overall_health_score(self) -> Dict[str, Any]:
        """
        Computes the final weighted financial health score (0-100) and returns sub-components.
        Utilizes thread-safe LRU memoization for identical input features.
        """
        weights_tuple = tuple(sorted(HEALTH_SCORE_WEIGHTS.items()))
        res = _compute_health_score_from_features(
            round(float(self.twin.total_income), 2),
            round(float(self.twin.basic_expenses), 2),
            round(float(self.twin.monthly_emi), 2),
            round(float(self.twin.emergency_fund), 2),
            round(float(self.twin.sip_amount), 2),
            round(float(self.twin.life_insurance), 2),
            round(float(self.twin.monthly_income), 2),
            round(float(self.twin.health_insurance), 2),
            round(float(self.twin.loan_amount), 2),
            weights_tuple,
        )
        return copy.deepcopy(res)

    @classmethod
    def clear_cache(cls) -> None:
        """Clears the memoized health score cache."""
        _compute_health_score_from_features.cache_clear()

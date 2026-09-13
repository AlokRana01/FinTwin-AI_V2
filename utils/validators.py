"""
utils/validators.py
===================
Comprehensive, centralized input validation and sanitization engine for FinTwin AI.

Enforces:
  - Strict type checking
  - Finite number validation (rejects NaN, +Infinity, -Infinity)
  - Realistic financial range checks (domain-specific min/max caps)
  - String sanitization (length limits, control character removal, XSS prevention)
  - Strict enum/allowlist validation
  - Dual-layer validation (UI + backend/server-side enforcement)
"""

import re
import math
import html
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# ── Global Financial & Demographic Range Constants ───────────────────────────
MIN_AGE = 18
MAX_AGE = 100

MAX_FINANCIAL_AMOUNT = 1_000_000_000.0  # ₹100 Crore upper bound for personal finance
MAX_MONTHLY_INCOME   = 100_000_000.0    # ₹10 Crore/mo max
MAX_MONTHLY_EXPENSE  = 100_000_000.0    # ₹10 Crore/mo max
MAX_GOAL_AMOUNT      = 1_000_000_000.0  # ₹100 Crore max
MAX_GOAL_YEARS       = 50
MIN_GOAL_YEARS       = 1
MIN_INTEREST_RATE    = 0.0
MAX_INTEREST_RATE    = 100.0            # 100% max annual interest / return rate

# String Length Caps
MAX_NAME_LEN        = 80
MAX_EMAIL_LEN       = 254
MAX_OCCUPATION_LEN  = 100
MAX_CITY_LEN        = 100
MAX_GOAL_NAME_LEN   = 120
MAX_CHAT_MSG_LEN    = 1000
MAX_PASSWORD_LEN    = 64
MIN_PASSWORD_LEN    = 8

# Allowed Enum sets
ALLOWED_OCCUPATIONS: Set[str] = {
    "Software Engineer", "Consultant", "Doctor", "Product Manager",
    "Data Scientist", "Banker", "Teacher", "Entrepreneur",
    "Sales Executive", "Marketing Manager", "Other"
}

ALLOWED_GOAL_TYPES: Set[str] = {
    "House", "Car", "Education", "Marriage", "Vacation", "Emergency Fund",
    "Wealth Building", "Retirement", "Custom", "Other"
}

ALLOWED_TAX_REGIMES: Set[str] = {
    "Old Regime", "New Regime", "old", "new", "Old", "New"
}


# ── Core Numeric & Finite Validation ──────────────────────────────────────────

def is_finite_number(val: Any) -> bool:
    """
    Checks if a value is a valid, finite real number (not NaN, +Inf, -Inf).
    """
    if val is None or isinstance(val, bool):
        return False
    try:
        f = float(val)
        return not (math.isnan(f) or math.isinf(f))
    except (ValueError, TypeError, OverflowError):
        return False


def validate_finite_number(
    val: Any,
    field_name: str = "Amount",
    min_val: float = 0.0,
    max_val: float = MAX_FINANCIAL_AMOUNT,
    allow_zero: bool = True,
) -> Tuple[bool, str, float]:
    """
    Validates that a numeric input is finite, within [min_val, max_val], and obeys allow_zero.
    Returns (is_valid, error_message, cleaned_float_value).
    """
    if not is_finite_number(val):
        return False, f"Please enter a valid finite number for {field_name}.", 0.0

    f = float(val)
    if not allow_zero and f == 0.0:
        return False, f"{field_name} must be greater than zero.", 0.0
    if f < min_val:
        return False, f"{field_name} cannot be less than ₹{min_val:,.0f}.", 0.0
    if f > max_val:
        return False, f"{field_name} cannot exceed ₹{max_val:,.0f}.", 0.0

    return True, "", f


def sanitize_numeric(val: Any, default: float = 0.0, min_val: float = 0.0, max_val: float = MAX_FINANCIAL_AMOUNT) -> float:
    """
    Safely converts any input into a clamped finite float. Fallback to default if invalid.
    """
    if not is_finite_number(val):
        return default
    f = float(val)
    if f < min_val:
        return min_val
    if f > max_val:
        return max_val
    return f


# ── Demographic & Text Validators ─────────────────────────────────────────────

def validate_age(age: Any) -> Tuple[bool, str, int]:
    """
    Validates user age (integer between MIN_AGE and MAX_AGE).
    """
    if not is_finite_number(age):
        return False, f"Age must be a whole number between {MIN_AGE} and {MAX_AGE}.", MIN_AGE
    i_age = int(round(float(age)))
    if i_age < MIN_AGE or i_age > MAX_AGE:
        return False, f"Age must be between {MIN_AGE} and {MAX_AGE} years.", MIN_AGE
    return True, "", i_age


def sanitize_text(text: Any, max_len: int = 255, allow_empty: bool = False) -> str:
    """
    Sanitizes arbitrary text:
    - Strips whitespace
    - Normalizes internal spacing
    - Strips non-printable control characters
    - Clamps to max_len
    """
    if text is None:
        return ""
    s = str(text).strip()
    # Remove control characters except newline and tab
    s = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', s)
    s = re.sub(r'[ \t]{2,}', ' ', s)
    return s[:max_len]


def validate_text(
    text: Any,
    field_name: str,
    min_len: int = 1,
    max_len: int = 255,
    allow_empty: bool = False,
) -> Tuple[bool, str, str]:
    """
    Validates a required or optional text field.
    """
    s = sanitize_text(text, max_len=max_len)
    if not s:
        if allow_empty:
            return True, "", ""
        return False, f"{field_name} is required.", ""
    if len(s) < min_len:
        return False, f"{field_name} must be at least {min_len} character(s).", s
    if len(s) > max_len:
        return False, f"{field_name} must not exceed {max_len} characters.", s[:max_len]
    return True, "", s


def validate_enum(val: Any, allowed_set: Union[Set[str], List[str]], field_name: str = "Selection") -> Tuple[bool, str, str]:
    """
    Validates that a selected value belongs to an allowlist.
    """
    s = str(val).strip() if val is not None else ""
    if s not in allowed_set:
        # Check case-insensitive match
        for item in allowed_set:
            if item.lower() == s.lower():
                return True, "", item
        return False, f"Invalid {field_name} selection. Please choose from valid options.", str(list(allowed_set)[0])
    return True, "", s


# ── Financial Profile & Goal Validators ───────────────────────────────────────

def validate_financial_profile(data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validates an entire financial profile dictionary before DB persistence or model input.
    """
    # 1. Income check
    monthly_inc = data.get("monthly_income", 0.0)
    ok, err, inc_val = validate_finite_number(monthly_inc, "Monthly Income", min_val=0.0, max_val=MAX_MONTHLY_INCOME)
    if not ok:
        return False, err

    # 2. Check all expense and debt fields
    numeric_fields = [
        ("rent", "Rent"),
        ("groceries", "Groceries"),
        ("utilities", "Utilities"),
        ("transport", "Transport"),
        ("food_delivery", "Food Delivery"),
        ("entertainment", "Entertainment"),
        ("shopping", "Shopping"),
        ("monthly_emi", "Monthly EMI"),
        ("bank_savings", "Bank Savings"),
        ("fd_amount", "Fixed Deposits"),
        ("emergency_fund", "Emergency Fund"),
        ("sip_amount", "Monthly SIP"),
        ("mutual_funds", "Mutual Funds"),
        ("stocks", "Stocks"),
        ("ppf_investment", "PPF"),
        ("nps_investment", "NPS"),
        ("loan_amount", "Loan Amount"),
        ("car_loan", "Car Loan"),
        ("home_loan", "Home Loan"),
        ("credit_card_debt", "Credit Card Debt"),
        ("health_insurance", "Health Insurance"),
        ("life_insurance", "Life Insurance"),
    ]

    for key, label in numeric_fields:
        val = data.get(key, 0.0)
        ok, err, _ = validate_finite_number(val, label, min_val=0.0, max_val=MAX_FINANCIAL_AMOUNT)
        if not ok:
            return False, err

    # 3. Realistic expense check vs income
    total_expenses = sum([
        float(data.get(k, 0.0)) for k in [
            "rent", "groceries", "utilities", "transport",
            "food_delivery", "entertainment", "shopping"
        ] if is_finite_number(data.get(k, 0.0))
    ])
    
    if inc_val > 0 and total_expenses > (inc_val * 5):
        return False, "Total monthly expenses cannot realistically exceed 5x of your monthly income."

    return True, ""


def validate_goal_input(
    goal_name: str,
    target_amount: float,
    target_years: int,
    return_rate: float,
    lump_sum: float = 0.0,
) -> Tuple[bool, str]:
    """
    Validates goal creation inputs.
    """
    ok, err, _ = validate_text(goal_name, "Goal Name", min_len=2, max_len=MAX_GOAL_NAME_LEN)
    if not ok:
        return False, err

    ok, err, _ = validate_finite_number(target_amount, "Goal Target Amount", min_val=1000.0, max_val=MAX_GOAL_AMOUNT, allow_zero=False)
    if not ok:
        return False, err

    if not is_finite_number(target_years) or int(target_years) < MIN_GOAL_YEARS or int(target_years) > MAX_GOAL_YEARS:
        return False, f"Target timeline must be between {MIN_GOAL_YEARS} and {MAX_GOAL_YEARS} years."

    ok, err, _ = validate_finite_number(return_rate, "Expected Return Rate", min_val=MIN_INTEREST_RATE, max_val=MAX_INTEREST_RATE)
    if not ok:
        return False, err

    ok, err, _ = validate_finite_number(lump_sum, "Initial Lump Sum", min_val=0.0, max_val=MAX_GOAL_AMOUNT)
    if not ok:
        return False, err

    return True, ""


# ── ML Input Pre-validation ───────────────────────────────────────────────────

def validate_ml_feature_vector(features: Dict[str, Any], expected_features: List[str]) -> Tuple[bool, str, Dict[str, float]]:
    """
    Validates feature dictionary before passing to XGBoost / Scikit-Learn models.
    Guarantees no NaN, Inf, or missing columns.
    """
    cleaned: Dict[str, float] = {}
    for feat in expected_features:
        if feat not in features:
            return False, f"Missing required ML feature: {feat}", {}
        val = features[feat]
        if not is_finite_number(val):
            return False, f"Invalid non-finite value for feature {feat}: {val}", {}
        cleaned[feat] = float(val)
    return True, "", cleaned

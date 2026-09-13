"""
Configuration File
Contains central configurations, constants, and settings for the
AI Financial Health Digital Twin application.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "financial_twin.db"

# Target Demographics
MIN_AGE = 22
MAX_AGE = 40

# Economic Assumptions
DEFAULT_INFLATION_RATE = 0.06  # 6% average annual inflation in India
DEFAULT_INVESTMENT_RETURN = 0.12  # 12% average equity return (mutual funds/stocks)
DEFAULT_SAVINGS_INTEREST = 0.035  # 3.5% savings bank interest rate
DEFAULT_FD_RETURN = 0.065  # 6.5% Fixed Deposit return

# ── Indian Tax Regime Constants (FY 2025-26 / 2026-27) ──────────────────────
# Surcharge tiers (applied on base tax, income thresholds)
SURCHARGE_TIERS = [
    (5_000_000,   0.00),   # Below ₹50L: no surcharge
    (10_000_000,  0.10),   # ₹50L–₹1Cr: 10%
    (20_000_000,  0.15),   # ₹1Cr–₹2Cr: 15%
    (50_000_000,  0.25),   # ₹2Cr–₹5Cr: 25%
    (float("inf"), 0.37),  # Above ₹5Cr: 37% (marginal relief applies)
]

# Health & Education Cess (4% on tax + surcharge)
CESS_RATE = 0.04

# Section 87A Rebate (FY 2025-26)
# Old Regime: Full tax rebate if taxable income <= ₹5L (max rebate ₹12,500)
# New Regime: Full tax rebate if taxable income <= ₹12L (max rebate ₹60,000)
REBATE_87A = {
    "OLD": {"limit": 500000,   "max_rebate": 12500},
    "NEW": {"limit": 1200000,  "max_rebate": 60000},
}

TAX_REGIMES = {
    "NEW": {
        "slabs": [
            (400000,       0.00),
            (800000,       0.05),
            (1200000,      0.10),
            (1600000,      0.15),
            (2000000,      0.20),
            (2400000,      0.25),
            (float("inf"), 0.30)
        ],
        "standard_deduction": 75000,
        # No deductions allowed under new regime (except std deduction)
        "allows_deductions": False,
    },
    "OLD": {
        "slabs": [
            (250000,       0.00),
            (500000,       0.05),
            (1000000,      0.20),
            (float("inf"), 0.30)
        ],
        "standard_deduction": 50000,
        "max_80c":          150000,   # PPF, ELSS, EPF, LIC premium, etc.
        "max_80d_self":      25000,   # Health insurance (self + family)
        "max_80d_parents":   50000,   # Health insurance (parents, senior citizen)
        "max_80ccd1b":       50000,   # NPS additional contribution
        "max_section_24b":  200000,   # Home loan interest deduction
        "allows_deductions": True,
    }
}

# Spending Profile Settings
SPEND_CATEGORIES = [
    "Rent/Home Loan EMI",
    "Groceries & Food",
    "Utilities & Bills",
    "Transport & Fuel",
    "Entertainment & Lifestyle",
    "Healthcare & Insurance",
    "Education & Self-Improvement",
    "Miscellaneous"
]

# Personality Types — canonical 5-label list matching the KMeans clusterer
# (models/clustering.py uses these exact string labels; this list is the
#  single source of truth for validation and display throughout the app)
PERSONALITY_LABELS = [
    "Saver",
    "Investor",
    "Spender",
    "Debt Heavy",
    "Balanced Planner",
]

# Health Score Weights — canonical 6-component definition
# This is the SINGLE SOURCE OF TRUTH consumed by HealthScoreEngine
# (models/twin_engine.py) and all downstream consumers (Scenario Simulator,
#  Financial Coach, Reports, future agents).
# DO NOT hard-code these weights anywhere else in the codebase.
HEALTH_SCORE_WEIGHTS = {
    "savings_rate":       0.25,   # Savings rate vs 30% target
    "emi_burden":         0.20,   # EMI-to-income ratio vs 35% ceiling
    "emergency_fund":     0.20,   # Emergency fund months coverage vs 6M target
    "investment_ratio":   0.15,   # SIP-to-income ratio vs 15% target
    "insurance_adequacy": 0.10,   # Life + health insurance adequacy
    "debt_level":         0.10,   # Outstanding debt vs annual income
}

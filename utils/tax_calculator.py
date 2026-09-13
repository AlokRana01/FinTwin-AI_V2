"""
Indian Tax Intelligence Engine  (Module 8)
Full implementation of Indian Income Tax computation for FY 2025-26 / 2026-27.

Supports:
  - Old Tax Regime (with deductions: 80C, 80D, 80CCD(1B), Standard Deduction, HRA)
  - New Tax Regime (only Standard Deduction ₹75,000; new slabs)
  - Section 87A Rebate
  - Surcharge tiers
  - Health & Education Cess (4%)
  - Regime comparison and recommendation
  - Estimated tax savings from switching regimes

Classes:
    DeductionProfile         - Holds all deduction inputs; enforces statutory limits
    TaxComputation           - Full tax computation result for one regime
    IndianTaxCalculator      - Core engine; reads slab config from config.py
"""

from __future__ import annotations

import functools
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

from config import TAX_REGIMES, SURCHARGE_TIERS, CESS_RATE, REBATE_87A


# ══════════════════════════════════════════════════════════════════════════════
# Deduction Profile
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DeductionProfile:
    """
    Represents all eligible deductions under the Old Tax Regime.
    Each field is capped at the statutory maximum before being used in computation.
    """
    # 80C: PPF, ELSS, EPF, LIC premium, NSC, ULIP, home loan principal, etc.
    investment_80c:    float = 0.0   # Max ₹1,50,000

    # 80D: Health insurance premiums
    health_insurance_self:    float = 0.0  # Self + family; max ₹25,000
    health_insurance_parents: float = 0.0  # Parents (senior citizen max ₹50,000)

    # 80CCD(1B): Additional NPS contribution over and above 80C
    nps_80ccd1b:       float = 0.0   # Max ₹50,000

    # Section 24(b): Home loan interest deduction (self-occupied property)
    home_loan_interest: float = 0.0  # Max ₹2,00,000

    # Other deductions (80E, 80G, 80TTA, etc.)
    other_deductions:  float = 0.0   # No statutory cap applied here

    def __post_init__(self):
        from utils.validators import is_finite_number
        for attr in ["investment_80c", "health_insurance_self", "health_insurance_parents", "nps_80ccd1b", "home_loan_interest", "other_deductions"]:
            val = getattr(self, attr)
            if not is_finite_number(val) or float(val) < 0.0:
                raise ValueError(f"Deduction value for {attr} must be a valid, non-negative finite number.")

    @property
    def effective_80c(self) -> float:
        """80C capped at ₹1,50,000."""
        return min(self.investment_80c, TAX_REGIMES["OLD"]["max_80c"])

    @property
    def effective_80d(self) -> float:
        """80D: Self portion capped at ₹25,000; parents capped at ₹50,000."""
        return (
            min(self.health_insurance_self,    TAX_REGIMES["OLD"]["max_80d_self"]) +
            min(self.health_insurance_parents, TAX_REGIMES["OLD"]["max_80d_parents"])
        )

    @property
    def effective_80ccd1b(self) -> float:
        """80CCD(1B) capped at ₹50,000."""
        return min(self.nps_80ccd1b, TAX_REGIMES["OLD"]["max_80ccd1b"])

    @property
    def effective_24b(self) -> float:
        """Section 24(b) home loan interest capped at ₹2,00,000."""
        return min(self.home_loan_interest, TAX_REGIMES["OLD"]["max_section_24b"])

    @property
    def total_deductions(self) -> float:
        """Sum of all capped deductions (excludes standard deduction which is applied separately)."""
        return (
            self.effective_80c +
            self.effective_80d +
            self.effective_80ccd1b +
            self.effective_24b +
            self.other_deductions
        )

    def breakdown(self) -> Dict[str, float]:
        """Returns a dict of effective deduction components for display."""
        return {
            "Standard Deduction (Sec 16)":  TAX_REGIMES["OLD"]["standard_deduction"],
            "80C (PPF/ELSS/EPF)":           self.effective_80c,
            "80D (Health Insurance Self)":  min(self.health_insurance_self, TAX_REGIMES["OLD"]["max_80d_self"]),
            "80D (Health Insurance Parents)": min(self.health_insurance_parents, TAX_REGIMES["OLD"]["max_80d_parents"]),
            "80CCD(1B) (NPS Extra)":        self.effective_80ccd1b,
            "Sec 24(b) (Home Loan Interest)": self.effective_24b,
            "Other Deductions":             self.other_deductions,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Tax Computation Result
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TaxComputation:
    """
    Complete tax computation result for one regime.
    """
    regime:                str
    gross_income:          float
    standard_deduction:    float
    total_deductions:      float    # Excluding standard deduction
    taxable_income:        float
    base_tax:              float
    rebate_87a:            float
    tax_after_rebate:      float
    surcharge:             float
    cess:                  float
    total_tax:             float
    effective_rate:        float    # total_tax / gross_income
    slab_breakdown:        List[Dict[str, Any]] = field(default_factory=list)
    deduction_breakdown:   Dict[str, float]     = field(default_factory=dict)

    @property
    def take_home_monthly(self) -> float:
        """Estimated monthly take-home after tax."""
        return round((self.gross_income - self.total_tax) / 12, 2)


# ══════════════════════════════════════════════════════════════════════════════
# Core Tax Calculator Engine
# ══════════════════════════════════════════════════════════════════════════════

class IndianTaxCalculator:
    """
    Core Indian Income Tax calculator for FY 2025-26 / 2026-27.
    Reads slab and deduction limits from config.TAX_REGIMES.

    Usage:
        calc = IndianTaxCalculator()
        old  = calc.calculate_old_regime_tax(gross_income=1200000, deductions=DeductionProfile(...))
        new  = calc.calculate_new_regime_tax(gross_income=1200000)
        rec  = calc.compare_and_optimize(gross_income=1200000, deductions=DeductionProfile(...))
    """

    # ── Private: Slab Computation ───────────────────────────────────────────

    @staticmethod
    def _apply_tax_slabs(taxable_income: float, slabs: List[Tuple[float, float]]) -> Tuple[float, List[Dict]]:
        """
        Applies progressive tax slabs to taxable income.

        Args:
            taxable_income: Net taxable income after all deductions.
            slabs:          List of (upper_limit, rate) tuples from config.

        Returns:
            (total_base_tax, slab_breakdown_list)
        """
        total_tax = 0.0
        breakdown = []
        prev_limit = 0.0

        for upper_limit, rate in slabs:
            if taxable_income <= prev_limit:
                break
            taxable_in_slab = min(taxable_income, upper_limit) - prev_limit
            tax_in_slab     = taxable_in_slab * rate
            total_tax      += tax_in_slab

            if taxable_in_slab > 0:
                breakdown.append({
                    "Slab":             f"₹{prev_limit/100000:.1f}L – {'∞' if upper_limit == float('inf') else f'₹{upper_limit/100000:.1f}L'}",
                    "Rate":             f"{rate * 100:.0f}%",
                    "Taxable Amount":   round(taxable_in_slab, 2),
                    "Tax":              round(tax_in_slab, 2),
                })
            prev_limit = upper_limit

        return round(total_tax, 2), breakdown

    @staticmethod
    def _compute_surcharge(gross_income: float, base_tax: float) -> float:
        """
        Computes surcharge based on gross income tier.

        Args:
            gross_income: Gross annual income (pre-deduction) for threshold check.
            base_tax:     Tax computed after rebate.

        Returns:
            Surcharge amount (₹).
        """
        rate = 0.0
        for limit, surcharge_rate in SURCHARGE_TIERS:
            if gross_income <= limit:
                rate = surcharge_rate
                break
        return round(base_tax * rate, 2)

    @staticmethod
    def _compute_87a_rebate(taxable_income: float, base_tax: float, regime: str) -> float:
        """
        Computes Section 87A rebate.
        Full rebate if taxable income is within the limit; else no rebate.

        Args:
            taxable_income: Net taxable income.
            base_tax:       Computed tax before rebate.
            regime:         'OLD' or 'NEW'.

        Returns:
            Rebate amount (₹).
        """
        rules = REBATE_87A[regime]
        if taxable_income <= rules["limit"]:
            return min(base_tax, rules["max_rebate"])
        return 0.0

    # ── Public: Old Regime ──────────────────────────────────────────────────

    def calculate_old_regime_tax(
        self,
        gross_income: float,
        deductions:   DeductionProfile,
    ) -> TaxComputation:
        """
        Computes income tax under the Old Tax Regime.

        Deduction order:
            1. Standard Deduction (₹50,000)
            2. 80C (PPF, ELSS, EPF, etc.) — max ₹1,50,000
            3. 80D (Health Insurance)      — max ₹75,000 combined
            4. 80CCD(1B) (NPS extra)       — max ₹50,000
            5. Section 24(b) home loan interest — max ₹2,00,000
            6. Other deductions

        Args:
            gross_income: Gross annual income (₹).
            deductions:   A populated DeductionProfile instance.

        Returns:
            TaxComputation with full breakdown.
        """
        if gross_income < 0:
            raise ValueError("Gross income cannot be negative")
        cfg = TAX_REGIMES["OLD"]
        std_ded     = cfg["standard_deduction"]
        total_ded   = deductions.total_deductions
        taxable_inc = max(gross_income - std_ded - total_ded, 0.0)

        base_tax, slab_breakdown = self._apply_tax_slabs(taxable_inc, cfg["slabs"])
        rebate      = self._compute_87a_rebate(taxable_inc, base_tax, "OLD")
        tax_after_rebate = max(base_tax - rebate, 0.0)
        surcharge   = self._compute_surcharge(gross_income, tax_after_rebate)
        cess        = round((tax_after_rebate + surcharge) * CESS_RATE, 2)
        total_tax   = round(tax_after_rebate + surcharge + cess, 2)
        eff_rate    = round(total_tax / gross_income * 100, 2) if gross_income > 0 else 0.0

        return TaxComputation(
            regime               = "Old Regime",
            gross_income         = gross_income,
            standard_deduction   = std_ded,
            total_deductions     = total_ded,
            taxable_income       = taxable_inc,
            base_tax             = base_tax,
            rebate_87a           = rebate,
            tax_after_rebate     = tax_after_rebate,
            surcharge            = surcharge,
            cess                 = cess,
            total_tax            = total_tax,
            effective_rate       = eff_rate,
            slab_breakdown       = slab_breakdown,
            deduction_breakdown  = deductions.breakdown(),
        )

    # ── Public: New Regime ──────────────────────────────────────────────────

    def calculate_new_regime_tax(self, gross_income: float) -> TaxComputation:
        """
        Computes income tax under the New Tax Regime (FY 2025-26 slabs).
        Only Standard Deduction of ₹75,000 is available; no other deductions.

        New Regime Slabs (FY 2025-26):
            ₹0       – ₹4L:   0%
            ₹4L      – ₹8L:   5%
            ₹8L      – ₹12L:  10%
            ₹12L     – ₹16L:  15%
            ₹16L     – ₹20L:  20%
            ₹20L     – ₹24L:  25%
            Above ₹24L:       30%

        Section 87A Rebate: Full rebate if taxable income ≤ ₹12L (max ₹60,000).
        Effective zero-tax threshold: ₹12,75,000 (₹12L + ₹75K std deduction).

        Args:
            gross_income: Gross annual income (₹).

        Returns:
            TaxComputation with full breakdown.
        """
        if gross_income < 0:
            raise ValueError("Gross income cannot be negative")
        cfg         = TAX_REGIMES["NEW"]
        std_ded     = cfg["standard_deduction"]
        taxable_inc = max(gross_income - std_ded, 0.0)

        base_tax, slab_breakdown = self._apply_tax_slabs(taxable_inc, cfg["slabs"])
        rebate           = self._compute_87a_rebate(taxable_inc, base_tax, "NEW")
        tax_after_rebate = max(base_tax - rebate, 0.0)
        surcharge        = self._compute_surcharge(gross_income, tax_after_rebate)
        cess             = round((tax_after_rebate + surcharge) * CESS_RATE, 2)
        total_tax        = round(tax_after_rebate + surcharge + cess, 2)
        eff_rate         = round(total_tax / gross_income * 100, 2) if gross_income > 0 else 0.0

        return TaxComputation(
            regime               = "New Regime",
            gross_income         = gross_income,
            standard_deduction   = std_ded,
            total_deductions     = 0.0,
            taxable_income       = taxable_inc,
            base_tax             = base_tax,
            rebate_87a           = rebate,
            tax_after_rebate     = tax_after_rebate,
            surcharge            = surcharge,
            cess                 = cess,
            total_tax            = total_tax,
            effective_rate       = eff_rate,
            slab_breakdown       = slab_breakdown,
            deduction_breakdown  = {"Standard Deduction (Sec 16)": std_ded},
        )

    # ── Public: Compare & Optimize ──────────────────────────────────────────

    def compare_and_optimize(
        self,
        gross_income: float,
        deductions:   DeductionProfile,
    ) -> Dict[str, Any]:
        """
        Computes tax under both regimes and recommends the optimal one.
        Utilizes thread-safe LRU memoization on hashable input parameters.

        Args:
            gross_income: Gross annual income (₹).
            deductions:   DeductionProfile with all applicable deductions.

        Returns:
            Dict with keys:
              recommended_regime   : 'Old Regime' or 'New Regime'
              old_regime           : TaxComputation for old regime
              new_regime           : TaxComputation for new regime
              annual_savings       : ₹ saved by choosing the recommended regime
              monthly_savings      : annual_savings / 12
              recommendation_text  : Human-readable explanation
              tips                 : List[str] of actionable tax-saving tips
        """
        ded_tuple = (
            round(float(deductions.investment_80c), 2),
            round(float(deductions.health_insurance_self), 2),
            round(float(deductions.health_insurance_parents), 2),
            round(float(deductions.nps_80ccd1b), 2),
            round(float(deductions.home_loan_interest), 2),
            round(float(deductions.other_deductions), 2),
        )
        res = self._compare_and_optimize_memoized(round(float(gross_income), 2), ded_tuple)
        return copy.deepcopy(res)

    @classmethod
    @functools.lru_cache(maxsize=2048)
    def _compare_and_optimize_memoized(cls, gross_income: float, ded_tuple: Tuple[float, ...]) -> Dict[str, Any]:
        """Pure, memoized tax optimization calculation."""
        deductions = DeductionProfile(
            investment_80c=ded_tuple[0],
            health_insurance_self=ded_tuple[1],
            health_insurance_parents=ded_tuple[2],
            nps_80ccd1b=ded_tuple[3],
            home_loan_interest=ded_tuple[4],
            other_deductions=ded_tuple[5],
        )
        calc = cls()
        old = calc.calculate_old_regime_tax(gross_income, deductions)
        new = calc.calculate_new_regime_tax(gross_income)

        if old.total_tax <= new.total_tax:
            recommended = "Old Regime"
            annual_savings = new.total_tax - old.total_tax
            rec_text = (
                f"The <strong>Old Regime saves you ₹{annual_savings:,.0f}/year</strong> "
                f"because your deductions (₹{old.standard_deduction + old.total_deductions:,.0f} total) "
                f"significantly reduce your taxable income."
            )
        else:
            recommended = "New Regime"
            annual_savings = old.total_tax - new.total_tax
            rec_text = (
                f"The <strong>New Regime saves you ₹{annual_savings:,.0f}/year</strong>. "
                f"Your current deductions (₹{old.total_deductions:,.0f}) are insufficient "
                f"to overcome the lower slab rates of the new regime."
            )

        tips = cls._generate_tips(gross_income, deductions, old, new, recommended)

        return {
            "recommended_regime":  recommended,
            "old_regime":          old,
            "new_regime":          new,
            "annual_savings":      round(annual_savings, 2),
            "monthly_savings":     round(annual_savings / 12, 2),
            "recommendation_text": rec_text,
            "tips":                tips,
        }

    @classmethod
    def clear_cache(cls) -> None:
        """Clears the memoized tax calculation cache."""
        cls._compare_and_optimize_memoized.cache_clear()

    # ── Private: Tips Engine ────────────────────────────────────────────────

    @staticmethod
    def _generate_tips(
        gross_income: float,
        deductions:   DeductionProfile,
        old:          TaxComputation,
        new:          TaxComputation,
        recommended:  str,
    ) -> List[str]:
        """Generates personalized tax-saving tips based on the user's deduction profile."""
        tips = []
        cfg = TAX_REGIMES["OLD"]

        # 80C utilization
        max_80c = cfg["max_80c"]
        used_80c = deductions.effective_80c
        if used_80c < max_80c:
            gap = max_80c - used_80c
            tips.append(
                f"You can invest ₹{gap:,.0f} more under 80C (PPF, ELSS, EPF) "
                f"to maximise your ₹1.5L exemption."
            )

        # NPS 80CCD(1B)
        max_nps = cfg["max_80ccd1b"]
        if deductions.effective_80ccd1b < max_nps:
            gap = max_nps - deductions.effective_80ccd1b
            tips.append(
                f"Contribute ₹{gap:,.0f} more to NPS under Sec 80CCD(1B) "
                f"for an additional ₹50,000 deduction (exclusive of 80C limit)."
            )

        # Health Insurance
        if deductions.health_insurance_self < cfg["max_80d_self"]:
            tips.append(
                "Increase health insurance coverage for yourself/family "
                f"(can claim up to ₹{cfg['max_80d_self']:,} under 80D)."
            )

        if deductions.health_insurance_parents == 0:
            tips.append(
                "Buy health insurance for your parents to claim "
                f"additional ₹{cfg['max_80d_parents']:,} deduction under 80D."
            )

        # Home loan interest
        if deductions.home_loan_interest == 0 and gross_income > 500000:
            tips.append(
                "If you have a home loan, claim Section 24(b) interest deduction "
                "(up to ₹2,00,000/year) under the Old Regime."
            )

        # New regime tip
        if recommended == "New Regime":
            tips.append(
                "Under the New Regime, focus on growing your SIP/MF corpus — "
                "LTCG up to ₹1.25L on equity MF is exempt (Sec 10(38))."
            )

        if not tips:
            tips.append("Your tax profile is well-optimised. Continue maximising all eligible deductions.")

        return tips

    # ── Convenience: HRA Calculation ────────────────────────────────────────

    @staticmethod
    def calculate_hra_exemption(
        basic_salary:      float,
        hra_received:      float,
        actual_rent_paid:  float,
        is_metro:          bool = True,
    ) -> float:
        """
        Calculates the Section 10(13A) House Rent Allowance tax exemption.
        The exemption is the MINIMUM of:
          1. HRA actually received
          2. Actual rent paid − 10% of Basic Salary
          3. 50% of Basic Salary (Metro) or 40% of Basic Salary (Non-metro)

        Args:
            basic_salary:     Annual basic salary (₹).
            hra_received:     Annual HRA received (₹).
            actual_rent_paid: Annual rent paid (₹).
            is_metro:         True for metro cities (Mumbai, Delhi, Kolkata, Chennai).

        Returns:
            HRA exemption amount (₹).
        """
        limit_1 = hra_received
        limit_2 = max(actual_rent_paid - 0.10 * basic_salary, 0.0)
        limit_3 = 0.50 * basic_salary if is_metro else 0.40 * basic_salary
        return round(min(limit_1, limit_2, limit_3), 2)

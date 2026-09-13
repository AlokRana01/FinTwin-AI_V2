"""
AI Financial Coach Engine  (Module 10)
Rule-based coaching engine that analyses a user's FinancialDigitalTwin across
5 financial domains and generates structured, prioritised advice.

Domains:
  1. Savings Habits      — surplus rate, discretionary spend leaks
  2. Investment Habits   — SIP adequacy, diversification gaps
  3. Debt Burden         — EMI-to-income ratio, loan structure
  4. Emergency Fund      — months coverage vs 6-month target
  5. Tax Efficiency      — 80C utilisation, NPS, insurance deductions

Output structures:
  Recommendation        — a single actionable suggestion with severity + domain
  Alert                 — a risk warning requiring urgent attention
  CoachingReport        — full output of the engine for one twin

Entry point:
  FinancialCoach(twin).generate_report() → CoachingReport
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


# ══════════════════════════════════════════════════════════════════════════════
# Enums & Constants
# ══════════════════════════════════════════════════════════════════════════════

class Domain(str, Enum):
    SAVINGS    = "Savings"
    INVESTMENT = "Investment"
    DEBT       = "Debt"
    EMERGENCY  = "Emergency Fund"
    TAX        = "Tax Efficiency"

class Severity(str, Enum):
    CRITICAL = "Critical"   # Immediate action required
    HIGH     = "High"       # Should act within 30 days
    MODERATE = "Moderate"   # Plan for next quarter
    LOW      = "Low"        # Good-to-have improvement
    POSITIVE = "Positive"   # Strength / compliment

# Severity display metadata
SEVERITY_META = {
    Severity.CRITICAL: {"icon": '<i class="fa-solid fa-triangle-exclamation"></i>', "color": "#EF4444", "bg": "rgba(239,68,68,0.10)"},
    Severity.HIGH:     {"icon": '<i class="fa-solid fa-circle-exclamation"></i>', "color": "#F97316", "bg": "rgba(249,115,22,0.10)"},
    Severity.MODERATE: {"icon": '<i class="fa-solid fa-circle-info"></i>', "color": "#F59E0B", "bg": "rgba(245,158,11,0.10)"},
    Severity.LOW:      {"icon": '<i class="fa-solid fa-info"></i>', "color": "#3B82F6", "bg": "rgba(59,130,246,0.10)"},
    Severity.POSITIVE: {"icon": '<i class="fa-solid fa-circle-check"></i>', "color": "#10B981", "bg": "rgba(16,185,129,0.10)"},
}

# 80C limit (FY 2025-26)
_80C_LIMIT       = 150_000
_NPS_80CCD_LIMIT =  50_000
_80D_SELF_LIMIT  =  25_000


# ══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Recommendation:
    """
    A single prioritised coaching recommendation.
    """
    domain:      Domain
    severity:    Severity
    title:       str          # Short headline (≤ 60 chars)
    detail:      str          # Full natural-language explanation
    action:      str          # Concrete next step (1-2 sentences)
    impact:      str          # Quantified impact estimate if applicable
    tags:        List[str] = field(default_factory=list)

    @property
    def icon(self) -> str:
        return SEVERITY_META[self.severity]["icon"]

    @property
    def color(self) -> str:
        return SEVERITY_META[self.severity]["color"]

    @property
    def bg(self) -> str:
        return SEVERITY_META[self.severity]["bg"]

    def to_natural_language(self) -> str:
        """Returns a complete paragraph of natural-language advice."""
        return (
            f"**{self.title}** — {self.detail} "
            f"**What to do:** {self.action} "
            f"**Expected Impact:** {self.impact}"
        )


@dataclass
class Alert:
    """
    A risk alert requiring urgent user attention.
    """
    domain:   Domain
    severity: Severity
    headline: str    # Bold one-liner shown in the alert banner
    message:  str    # Expanded explanation

    @property
    def icon(self) -> str:
        return SEVERITY_META[self.severity]["icon"]

    @property
    def color(self) -> str:
        return SEVERITY_META[self.severity]["color"]


@dataclass
class DomainSummary:
    """
    Aggregated score + key metrics for one coaching domain.
    """
    domain:      Domain
    score:       float          # 0-100
    grade:       str            # A / B / C / D
    headline:    str            # One-line assessment
    metrics:     Dict[str, Any] # Key numbers for display
    suggestions: List[str]      # Short bullet nudges (3-5)


@dataclass
class CoachingReport:
    """
    Full output of the FinancialCoach engine for one twin.
    """
    twin_name:       str
    overall_score:   float                  # Weighted average domain scores
    risk_level:      str                    # Low / Moderate / High / Critical
    alerts:          List[Alert]            # Sorted: Critical first
    recommendations: List[Recommendation]  # Sorted: highest severity first
    domain_summaries: Dict[str, DomainSummary]  # One per Domain
    top_action:      str                    # The single most important thing to do now
    narrative:       str                    # Full paragraph summary in natural language

    @property
    def critical_alerts(self) -> List[Alert]:
        return [a for a in self.alerts if a.severity == Severity.CRITICAL]

    @property
    def by_domain(self) -> Dict[str, List[Recommendation]]:
        result: Dict[str, List[Recommendation]] = {}
        for r in self.recommendations:
            result.setdefault(r.domain.value, []).append(r)
        return result


# ══════════════════════════════════════════════════════════════════════════════
# Domain Analysers
# ══════════════════════════════════════════════════════════════════════════════

class _SavingsAnalyser:
    """Analyses savings habits: surplus rate, discretionary spending leaks."""

    def __init__(self, twin):
        self.twin = twin
        t = twin
        self.income      = t.total_income
        self.expenses    = t.basic_expenses
        self.emi         = t.monthly_emi
        self.surplus     = max(self.income - self.expenses - self.emi, 0)
        self.rate        = self.surplus / self.income if self.income > 0 else 0.0
        # Discretionary components
        self.discretionary = t.food_delivery + t.entertainment + t.shopping
        self.disc_pct      = self.discretionary / self.income if self.income > 0 else 0.0

    def analyse(self) -> tuple[DomainSummary, List[Recommendation], List[Alert]]:
        recs: List[Recommendation] = []
        alerts: List[Alert]        = []
        t = self.twin

        # Score: normalise savings rate to 0-100 (100% at 30%+ savings rate)
        score = min(self.rate / 0.30, 1.0) * 100

        # ── Alerts ──────────────────────────────────────────────────────────
        if self.rate < 0:
            alerts.append(Alert(
                domain   = Domain.SAVINGS,
                severity = Severity.CRITICAL,
                headline = "Negative Monthly Surplus — You are spending more than you earn!",
                message  = (
                    f"Your total outflow (₹{self.expenses + self.emi:,.0f}/mo expenses + EMI) "
                    f"exceeds your income (₹{self.income:,.0f}/mo). "
                    f"Deficit: ₹{abs(self.surplus):,.0f}/month. "
                    "This will erode savings and create new debt each month."
                ),
            ))
        elif self.rate < 0.05:
            alerts.append(Alert(
                domain   = Domain.SAVINGS,
                severity = Severity.HIGH,
                headline = "Critically Low Savings Rate — Under 5%",
                message  = (
                    f"You are saving only {self.rate:.1%} of income "
                    f"(₹{self.surplus:,.0f}/mo). Target at least 25%. "
                    "This level provides almost no financial buffer."
                ),
            ))

        # ── Recommendations ──────────────────────────────────────────────────
        if self.rate < 0.10:
            recs.append(Recommendation(
                domain   = Domain.SAVINGS,
                severity = Severity.CRITICAL if self.rate < 0 else Severity.HIGH,
                title    = "Increase Monthly Savings Rate Urgently",
                detail   = (
                    f"Your current savings rate is {self.rate:.1%} "
                    f"(₹{self.surplus:,.0f}/month). "
                    "A healthy household target is 25–30% of net income."
                ),
                action   = (
                    "Implement the 50/30/20 rule: 50% needs, 30% wants, 20% savings. "
                    "Set up an auto-transfer of ₹"
                    f"{max(int(self.income * 0.20 - self.surplus), 0):,} "
                    "on salary day to a separate savings account."
                ),
                impact   = (
                    f"Saving 20% of ₹{self.income:,.0f} = ₹{self.income * 0.20:,.0f}/mo "
                    f"→ ₹{self.income * 0.20 * 12:,.0f}/year extra corpus."
                ),
                tags     = ["savings-rate", "50-30-20", "auto-save"],
            ))
        elif self.rate < 0.25:
            recs.append(Recommendation(
                domain   = Domain.SAVINGS,
                severity = Severity.MODERATE,
                title    = "Savings Rate Below 25% Target",
                detail   = (
                    f"Your savings rate is {self.rate:.1%}. "
                    "You're on the right track, but 25–30% is the Indian household benchmark "
                    "for wealth accumulation."
                ),
                action   = (
                    f"Increase your monthly SIP by ₹{max(int(self.income * 0.05), 500):,} "
                    "and reduce one discretionary category by 10%."
                ),
                impact   = (
                    f"Bridging to 25% saves ₹{(0.25 - self.rate) * self.income:,.0f}/month "
                    f"extra — ₹{(0.25 - self.rate) * self.income * 12:,.0f}/year."
                ),
                tags     = ["savings-rate"],
            ))
        else:
            recs.append(Recommendation(
                domain   = Domain.SAVINGS,
                severity = Severity.POSITIVE,
                title    = "Excellent Savings Rate — Keep It Up!",
                detail   = (
                    f"Your savings rate of {self.rate:.1%} "
                    f"(₹{self.surplus:,.0f}/month) is above the recommended 25% benchmark."
                ),
                action   = (
                    "Continue this discipline. Consider step-up SIPs (10% annual increase) "
                    "to grow wealth faster with every salary hike."
                ),
                impact   = (
                    f"At this rate, you accumulate ₹{self.surplus * 12:,.0f}/year "
                    "before investment returns."
                ),
                tags     = ["savings-rate", "strength"],
            ))

        # Discretionary overspend
        if self.disc_pct > 0.20:
            recs.append(Recommendation(
                domain   = Domain.SAVINGS,
                severity = Severity.HIGH,
                title    = "Discretionary Spending Draining Surplus",
                detail   = (
                    f"You spend ₹{self.discretionary:,.0f}/month "
                    f"({self.disc_pct:.1%} of income) on food delivery, entertainment & shopping — "
                    "above the recommended 15% limit."
                ),
                action   = (
                    f"Target a 20% reduction in discretionary spend (save "
                    f"₹{int(self.discretionary * 0.20):,}/month). "
                    "Use the 48-hour rule: wait 2 days before non-essential purchases."
                ),
                impact   = (
                    f"Saving ₹{int(self.discretionary * 0.20):,}/month = "
                    f"₹{int(self.discretionary * 0.20 * 12):,}/year redirectable to SIP."
                ),
                tags     = ["discretionary", "lifestyle"],
            ))
        elif self.disc_pct > 0.12:
            recs.append(Recommendation(
                domain   = Domain.SAVINGS,
                severity = Severity.LOW,
                title    = "Moderate Discretionary Spending — Room to Trim",
                detail   = (
                    f"Discretionary spend (food delivery + entertainment + shopping) is "
                    f"₹{self.discretionary:,.0f}/month ({self.disc_pct:.1%} of income)."
                ),
                action   = (
                    "Track weekly discretionary spend with a budgeting app. "
                    "Aim to keep it under 12% of monthly income."
                ),
                impact   = f"Cutting to 12% frees ₹{max(int((self.disc_pct - 0.12) * self.income), 0):,}/month.",
                tags     = ["discretionary"],
            ))

        # Rent burden
        rent_pct = t.rent / self.income if self.income > 0 else 0
        if rent_pct > 0.35:
            recs.append(Recommendation(
                domain   = Domain.SAVINGS,
                severity = Severity.HIGH,
                title    = "High Rent-to-Income Ratio",
                detail   = (
                    f"Rent consumes {rent_pct:.1%} of your income "
                    f"(₹{t.rent:,.0f}/month). Ideal is under 30%."
                ),
                action   = (
                    "Consider relocating 10–15 km further from the city centre to save 15–25% on rent, "
                    "or negotiate a longer lease for a lower monthly rate."
                ),
                impact   = (
                    f"Reducing rent by 15% saves ₹{int(t.rent * 0.15):,}/month "
                    f"= ₹{int(t.rent * 0.15 * 12):,}/year."
                ),
                tags     = ["rent", "housing"],
            ))

        grade = "A" if score >= 80 else ("B" if score >= 60 else ("C" if score >= 40 else "D"))
        summary = DomainSummary(
            domain   = Domain.SAVINGS,
            score    = round(score, 1),
            grade    = grade,
            headline = (
                f"Saving {self.rate:.1%} of income "
                f"(₹{self.surplus:,.0f}/mo). "
                f"{'Well above target.' if self.rate >= 0.25 else 'Below 25% target.'}"
            ),
            metrics  = {
                "Monthly Surplus":      f"₹{self.surplus:,.0f}",
                "Savings Rate":         f"{self.rate:.1%}",
                "Discretionary Spend":  f"₹{self.discretionary:,.0f} ({self.disc_pct:.1%})",
                "Rent":                 f"₹{t.rent:,.0f} ({rent_pct:.1%})",
                "Target Savings Rate":  "≥ 25%",
            },
            suggestions = [
                "Auto-transfer 20% of salary on payday",
                "Use zero-based budgeting — assign every rupee a job",
                f"Cap food delivery at ₹{min(int(t.food_delivery), int(self.income * 0.05)):,}/month",
                "Review subscriptions quarterly — cancel unused ones",
                "Step-up SIP by 10% every January",
            ],
        )
        return summary, recs, alerts


class _InvestmentAnalyser:
    """Analyses investment habits: SIP rate, diversification, NPS, PPF usage."""

    def __init__(self, twin):
        self.twin         = twin
        t = twin
        self.income       = t.total_income
        self.sip          = t.sip_amount
        self.mutual_funds = t.mutual_funds
        self.stocks       = t.stocks
        self.ppf          = t.ppf_investment
        self.nps          = t.nps_investment
        self.fd           = t.fd_amount
        self.total_corpus = self.mutual_funds + self.stocks + self.ppf + self.nps + self.fd
        self.sip_pct      = self.sip / self.income if self.income > 0 else 0.0
        self.equity_pct   = (self.mutual_funds + self.stocks) / max(self.total_corpus, 1)

    def analyse(self) -> tuple[DomainSummary, List[Recommendation], List[Alert]]:
        recs: List[Recommendation] = []
        alerts: List[Alert]        = []
        t = self.twin

        score = min(self.sip_pct / 0.15, 1.0) * 100

        # ── Alerts ──────────────────────────────────────────────────────────
        if self.sip_pct == 0 and self.total_corpus == 0:
            alerts.append(Alert(
                domain   = Domain.INVESTMENT,
                severity = Severity.CRITICAL,
                headline = "No Active Investments Detected — Wealth Is Not Growing!",
                message  = (
                    "You have zero SIP and no investment corpus. Inflation at 6% is silently "
                    "eroding your purchasing power. Start investing immediately."
                ),
            ))
        elif self.sip_pct < 0.05:
            alerts.append(Alert(
                domain   = Domain.INVESTMENT,
                severity = Severity.HIGH,
                headline = f"SIP Rate Critically Low — Only {self.sip_pct:.1%} of Income",
                message  = (
                    f"Monthly SIP of ₹{self.sip:,.0f} covers only {self.sip_pct:.1%} of income. "
                    "Target at least 15% for long-term wealth building."
                ),
            ))

        # ── Recommendations ──────────────────────────────────────────────────
        # SIP adequacy
        if self.sip_pct < 0.10:
            sip_target = int(self.income * 0.15)
            recs.append(Recommendation(
                domain   = Domain.INVESTMENT,
                severity = Severity.HIGH,
                title    = "SIP Investment Below 15% Benchmark",
                detail   = (
                    f"You invest ₹{self.sip:,.0f}/month ({self.sip_pct:.1%} of income) in SIPs. "
                    "Financial planners recommend 15% of income in equity-oriented funds."
                ),
                action   = (
                    f"Start or increase a Flexi-Cap/Multi-Cap mutual fund SIP to "
                    f"₹{sip_target:,}/month. Use ELSS funds to simultaneously save tax under 80C."
                ),
                impact   = (
                    f"₹{sip_target:,}/month in equity at 12% CAGR for 10 years = "
                    f"₹{int(sip_target * ((1.01**120 - 1) / 0.01)):,} corpus."
                ),
                tags     = ["sip", "mutual-funds", "wealth-building"],
            ))
        elif self.sip_pct < 0.15:
            recs.append(Recommendation(
                domain   = Domain.INVESTMENT,
                severity = Severity.MODERATE,
                title    = "Step Up SIP to 15% of Income",
                detail   = (
                    f"SIP rate of {self.sip_pct:.1%} is solid but below the 15% target. "
                    f"Gap: ₹{int((0.15 - self.sip_pct) * self.income):,}/month."
                ),
                action   = (
                    "Activate a Step-Up SIP: increase current SIP by 10% each year. "
                    "This aligns your investment growth with salary growth."
                ),
                impact   = (
                    f"Adding ₹{int((0.15 - self.sip_pct) * self.income):,}/month at 12% for 10 years "
                    f"= ₹{int((0.15 - self.sip_pct) * self.income * ((1.01**120 - 1) / 0.01)):,} extra."
                ),
                tags     = ["sip", "step-up"],
            ))
        else:
            recs.append(Recommendation(
                domain   = Domain.INVESTMENT,
                severity = Severity.POSITIVE,
                title    = f"Strong SIP Discipline — {self.sip_pct:.1%} of Income Invested",
                detail   = (
                    f"Monthly SIP of ₹{self.sip:,.0f} ({self.sip_pct:.1%} of income) "
                    "exceeds the 15% benchmark. Excellent wealth-building behaviour."
                ),
                action   = (
                    "Activate annual 10% step-up in SIP to outpace inflation. "
                    "Consider diversifying into international funds (Nasdaq/US equity) for geographic spread."
                ),
                impact   = (
                    f"Current SIP at 12% for 15 years = "
                    f"₹{int(self.sip * ((1.01**180 - 1) / 0.01)):,} corpus."
                ),
                tags     = ["sip", "strength", "step-up"],
            ))

        # NPS — tax-advantaged retirement
        if t.nps_investment == 0:
            recs.append(Recommendation(
                domain   = Domain.INVESTMENT,
                severity = Severity.MODERATE,
                title    = "Open NPS Account for Tax-Free Retirement Corpus",
                detail   = (
                    "NPS (National Pension System) allows an additional ₹50,000/year deduction "
                    "under 80CCD(1B) — over and above the ₹1.5L 80C limit."
                ),
                action   = (
                    "Open an NPS Tier-I account (eNPS or via your bank). "
                    "Invest ₹4,167/month to claim the full ₹50,000/year deduction."
                ),
                impact   = (
                    "₹50,000 deduction at 30% tax bracket = ₹15,000/year tax saved. "
                    "Invested at 10% CAGR for 20 years = ₹5.7L+ from tax savings alone."
                ),
                tags     = ["nps", "tax", "retirement"],
            ))

        # PPF
        if t.ppf_investment < 500_000:
            recs.append(Recommendation(
                domain   = Domain.INVESTMENT,
                severity = Severity.LOW,
                title    = "Increase PPF Corpus for Risk-Free Long-Term Returns",
                detail   = (
                    f"Current PPF corpus: ₹{t.ppf_investment:,.0f}. "
                    "PPF offers 7.1% tax-free returns, qualifies under 80C, and is EEE — "
                    "Exempt on contribution, interest, and maturity."
                ),
                action   = (
                    "Invest ₹12,500/month (₹1.5L/year) in PPF to max the 80C-qualified bucket "
                    "and build a guaranteed debt allocation."
                ),
                impact   = (
                    "₹1.5L/year in PPF for 15 years at 7.1% = ₹40.7L tax-free corpus."
                ),
                tags     = ["ppf", "debt", "tax", "80c"],
            ))

        # Diversification
        if self.equity_pct > 0.85 and self.total_corpus > 100_000:
            recs.append(Recommendation(
                domain   = Domain.INVESTMENT,
                severity = Severity.MODERATE,
                title    = "Over-Concentrated in Equity — Add Debt Allocation",
                detail   = (
                    f"{self.equity_pct:.0%} of your investment corpus is in equity (MFs + stocks). "
                    "For your age, a 70:30 equity-debt split is commonly recommended."
                ),
                action   = (
                    "Rebalance by directing next 6 months' SIP increase to a debt fund or PPF. "
                    "Target 20–30% in debt/fixed-income instruments."
                ),
                impact   = "Reduces portfolio volatility by ~30–40% during market downturns.",
                tags     = ["diversification", "asset-allocation"],
            ))
        elif self.equity_pct < 0.30 and t.age < 40:
            recs.append(Recommendation(
                domain   = Domain.INVESTMENT,
                severity = Severity.MODERATE,
                title    = "Under-Invested in Equity at This Age",
                detail   = (
                    f"Only {self.equity_pct:.0%} of corpus is in equity. "
                    f"At age {t.age}, you can afford higher equity exposure for long-term growth."
                ),
                action   = (
                    "Shift at least 60% of future SIPs to diversified equity mutual funds "
                    "(Flexi Cap or Index funds). Reduce over-reliance on FDs."
                ),
                impact   = (
                    "Equity at 12% CAGR vs FD at 6.5% = "
                    "2.2x more wealth over 15 years for the same investment."
                ),
                tags     = ["equity", "asset-allocation", "growth"],
            ))

        grade = "A" if score >= 80 else ("B" if score >= 60 else ("C" if score >= 40 else "D"))
        summary = DomainSummary(
            domain   = Domain.INVESTMENT,
            score    = round(score, 1),
            grade    = grade,
            headline = (
                f"SIP: ₹{self.sip:,.0f}/mo ({self.sip_pct:.1%} of income). "
                f"Total corpus: ₹{self.total_corpus:,.0f}."
            ),
            metrics  = {
                "Monthly SIP":       f"₹{self.sip:,.0f}",
                "SIP-to-Income":     f"{self.sip_pct:.1%}",
                "Mutual Funds":      f"₹{self.mutual_funds:,.0f}",
                "Stocks":            f"₹{self.stocks:,.0f}",
                "PPF":               f"₹{t.ppf_investment:,.0f}",
                "NPS":               f"₹{t.nps_investment:,.0f}",
                "Equity %":          f"{self.equity_pct:.0%}",
                "Target SIP Rate":   "≥ 15%",
            },
            suggestions = [
                "Start a Step-Up SIP — increase 10% every year",
                "Open NPS Tier-I for extra ₹50,000 80CCD deduction",
                "Add index funds (Nifty 50) for low-cost broad exposure",
                "Review fund performance annually — exit laggards",
                "Maintain 70:30 equity-debt ratio for your age",
            ],
        )
        return summary, recs, alerts


class _DebtAnalyser:
    """Analyses debt burden: EMI-to-income, total debt leverage, CC debt, prepayment strategy."""

    def __init__(self, twin):
        self.twin      = twin
        t = twin
        self.income    = t.total_income
        self.emi       = t.monthly_emi
        self.total_debt = t.loan_amount
        self.home_loan = t.home_loan
        self.car_loan  = t.car_loan
        self.cc_debt   = t.credit_card_debt
        self.emi_pct   = self.emi / self.income if self.income > 0 else 0.0
        self.debt_to_income = self.total_debt / (self.income * 12) if self.income > 0 else 0.0

    def analyse(self) -> tuple[DomainSummary, List[Recommendation], List[Alert]]:
        recs: List[Recommendation] = []
        alerts: List[Alert]        = []
        t = self.twin

        # Score: 100 at EMI ≤ 20%, 0 at EMI ≥ 70%
        if self.emi_pct <= 0.20:
            score = 100.0
        elif self.emi_pct >= 0.70:
            score = 0.0
        else:
            score = 100.0 - ((self.emi_pct - 0.20) / 0.50) * 100.0

        # ── Alerts ──────────────────────────────────────────────────────────
        if self.cc_debt > 0:
            cc_interest_pa = self.cc_debt * 0.36  # ~36% PA on CC
            alerts.append(Alert(
                domain   = Domain.DEBT,
                severity = Severity.CRITICAL,
                headline = f"Credit Card Debt — ₹{self.cc_debt:,.0f} at ~36% Interest!",
                message  = (
                    f"Credit card debt of ₹{self.cc_debt:,.0f} is costing you approximately "
                    f"₹{int(cc_interest_pa / 12):,}/month in interest (36% PA). "
                    "Clear this before any investment — no market return beats 36%."
                ),
            ))

        if self.emi_pct > 0.50:
            alerts.append(Alert(
                domain   = Domain.DEBT,
                severity = Severity.CRITICAL,
                headline = f"EMI Exceeds 50% of Income — Debt Trap Risk!",
                message  = (
                    f"EMI payments consume {self.emi_pct:.1%} of income "
                    f"(₹{self.emi:,.0f}/month). This severely limits your financial flexibility "
                    "and leaves almost no room for savings or emergencies."
                ),
            ))
        elif self.emi_pct > 0.35:
            alerts.append(Alert(
                domain   = Domain.DEBT,
                severity = Severity.HIGH,
                headline = f"EMI Burden at {self.emi_pct:.1%} — Above Safe 35% Threshold",
                message  = (
                    f"Your EMI-to-income ratio of {self.emi_pct:.1%} exceeds the recommended 35% limit. "
                    "Prioritise debt prepayment before taking on new loans or credit cards."
                ),
            ))

        # ── Recommendations ──────────────────────────────────────────────────
        # CC debt — highest priority
        if self.cc_debt > 0:
            recs.append(Recommendation(
                domain   = Domain.DEBT,
                severity = Severity.CRITICAL,
                title    = "Eliminate Credit Card Debt Immediately",
                detail   = (
                    f"Credit card debt of ₹{self.cc_debt:,.0f} accrues ~36% interest PA — "
                    "the most expensive debt you can carry. "
                    f"It costs you ~₹{int(self.cc_debt * 0.36 / 12):,}/month in interest."
                ),
                action   = (
                    "Pay the full outstanding balance this month if possible. "
                    "If not, convert to a 0% balance transfer offer or personal loan at 12–14%. "
                    "Never pay only the minimum — it extends debt to years."
                ),
                impact   = (
                    f"Clearing ₹{self.cc_debt:,.0f} CC debt saves "
                    f"₹{int(self.cc_debt * 0.36):,}/year in interest."
                ),
                tags     = ["credit-card", "high-interest", "debt-emergency"],
            ))

        # EMI burden
        if self.emi_pct > 0.35:
            recs.append(Recommendation(
                domain   = Domain.DEBT,
                severity = Severity.HIGH,
                title    = "Reduce EMI Burden — Prepay Smaller Loans First",
                detail   = (
                    f"EMI of ₹{self.emi:,.0f}/month ({self.emi_pct:.1%}) exceeds 35% safe limit. "
                    "Use the 'avalanche method': prepay the highest-interest loan first."
                ),
                action   = (
                    "Redirect ₹5,000–₹10,000/month as lump-sum prepayment to the smallest "
                    f"loan. {'Prioritise car loan over home loan (higher rate).' if t.car_loan > 0 else 'Target personal loan prepayment first.'}"
                ),
                impact   = (
                    f"Prepaying ₹10,000/month extra on a 10% loan reduces tenure by ~3–4 years "
                    "and saves ₹2–3L in interest."
                ),
                tags     = ["emi", "prepayment", "avalanche"],
            ))
        elif self.emi_pct > 0.20:
            recs.append(Recommendation(
                domain   = Domain.DEBT,
                severity = Severity.MODERATE,
                title    = "Consider Prepaying Loans to Free Cash Flow",
                detail   = (
                    f"EMI is {self.emi_pct:.1%} of income. While within safe range, "
                    "prepaying loans reduces interest outflow and improves monthly surplus."
                ),
                action   = (
                    "Use annual bonus to make partial prepayment on the highest-rate loan. "
                    "Even ₹50,000 lump sum significantly reduces total interest paid."
                ),
                impact   = (
                    "₹50,000 prepayment on a ₹10L loan at 10% saves ~₹25,000 in interest."
                ),
                tags     = ["prepayment", "bonus"],
            ))
        else:
            recs.append(Recommendation(
                domain   = Domain.DEBT,
                severity = Severity.POSITIVE,
                title    = "Healthy EMI Burden — Well Within Safe Limits",
                detail   = (
                    f"EMI of ₹{self.emi:,.0f}/month = {self.emi_pct:.1%} of income. "
                    "This is well below the 35% safe threshold."
                ),
                action   = (
                    "Maintain discipline. Avoid adding new consumer debt (personal loans, "
                    "car loans) until this drops further."
                ),
                impact   = f"Low EMI frees ₹{int(self.income * 0.35 - self.emi):,}/month for savings.",
                tags     = ["emi", "strength"],
            ))

        # Total debt leverage
        if self.debt_to_income > 3.0:
            recs.append(Recommendation(
                domain   = Domain.DEBT,
                severity = Severity.HIGH,
                title    = "Total Debt Exceeds 3× Annual Income",
                detail   = (
                    f"Outstanding loans of ₹{self.total_debt:,.0f} = "
                    f"{self.debt_to_income:.1f}× annual income. "
                    "Safe leverage is under 1.5×."
                ),
                action   = (
                    "Do not take any new loans. Channel every annual bonus into debt reduction. "
                    "Consult a debt counsellor for a structured repayment plan."
                ),
                impact   = (
                    "Reducing debt to 1.5× income "
                    f"frees ₹{int((self.total_debt - self.income * 12 * 1.5) * 0.10 / 12):,}/month "
                    "in interest savings."
                ),
                tags     = ["leverage", "total-debt"],
            ))

        grade = "A" if score >= 80 else ("B" if score >= 60 else ("C" if score >= 40 else "D"))
        summary = DomainSummary(
            domain   = Domain.DEBT,
            score    = round(score, 1),
            grade    = grade,
            headline = (
                f"EMI: ₹{self.emi:,.0f}/mo ({self.emi_pct:.1%} of income). "
                f"Total debt: ₹{self.total_debt:,.0f}."
            ),
            metrics  = {
                "Monthly EMI":          f"₹{self.emi:,.0f}",
                "EMI-to-Income":        f"{self.emi_pct:.1%}",
                "Total Loans":          f"₹{self.total_debt:,.0f}",
                "Home Loan":            f"₹{self.home_loan:,.0f}",
                "Car Loan":             f"₹{self.car_loan:,.0f}",
                "Credit Card Debt":     f"₹{self.cc_debt:,.0f}",
                "Debt × Annual Income": f"{self.debt_to_income:.2f}×",
                "Safe Threshold":       "EMI ≤ 35% | Debt ≤ 1.5×",
            },
            suggestions = [
                "Pay full credit card outstanding every month — never carry balance",
                "Prepay 1 EMI extra per year using any windfall",
                "Avoid personal loans for lifestyle expenses",
                "Refinance home loan if a better rate is available (switch lenders)",
                "Set a 'no new loan' rule until EMI < 20% of income",
            ],
        )
        return summary, recs, alerts


class _EmergencyFundAnalyser:
    """Analyses emergency fund adequacy vs 6-month expense target."""

    def __init__(self, twin):
        self.twin       = twin
        t = twin
        self.ef         = t.emergency_fund
        self.monthly_out = t.basic_expenses + t.monthly_emi
        self.months     = self.ef / self.monthly_out if self.monthly_out > 0 else 0.0
        self.target_ef  = self.monthly_out * 6
        self.gap        = max(self.target_ef - self.ef, 0)
        self.income     = t.total_income

    def analyse(self) -> tuple[DomainSummary, List[Recommendation], List[Alert]]:
        recs: List[Recommendation] = []
        alerts: List[Alert]        = []

        score = min(self.months / 6.0, 1.0) * 100

        # ── Alerts ──────────────────────────────────────────────────────────
        if self.months < 1:
            alerts.append(Alert(
                domain   = Domain.EMERGENCY,
                severity = Severity.CRITICAL,
                headline = "No Emergency Fund — One Crisis Away from Debt!",
                message  = (
                    f"Emergency fund covers only {self.months:.1f} months of expenses. "
                    f"Target: 6 months (₹{self.target_ef:,.0f}). "
                    "Without a buffer, any job loss or medical emergency forces you into high-interest debt."
                ),
            ))
        elif self.months < 3:
            alerts.append(Alert(
                domain   = Domain.EMERGENCY,
                severity = Severity.HIGH,
                headline = f"Thin Emergency Fund — Only {self.months:.1f} Months Coverage",
                message  = (
                    f"Emergency fund of ₹{self.ef:,.0f} covers {self.months:.1f} months. "
                    f"Build to 6 months (₹{self.target_ef:,.0f}) as a priority."
                ),
            ))

        # ── Recommendations ──────────────────────────────────────────────────
        if self.months < 3:
            monthly_save = max(int(self.gap / 12), 5000)
            recs.append(Recommendation(
                domain   = Domain.EMERGENCY,
                severity = Severity.CRITICAL if self.months < 1 else Severity.HIGH,
                title    = "Build Emergency Fund to 3 Months as First Priority",
                detail   = (
                    f"Current emergency fund: ₹{self.ef:,.0f} ({self.months:.1f} months). "
                    f"Minimum safe: 3 months = ₹{self.monthly_out * 3:,.0f}. "
                    f"Target: 6 months = ₹{self.target_ef:,.0f}."
                ),
                action   = (
                    f"Park ₹{monthly_save:,}/month in a high-yield savings account or liquid mutual fund. "
                    "Do NOT invest this in equity — it must be instantly accessible."
                ),
                impact   = (
                    f"Reaching 6 months in {int(self.gap / max(monthly_save, 1))} months. "
                    "Protects against job loss without touching equity investments."
                ),
                tags     = ["emergency-fund", "liquid"],
            ))
        elif self.months < 6:
            monthly_save = max(int(self.gap / 6), 3000)
            recs.append(Recommendation(
                domain   = Domain.EMERGENCY,
                severity = Severity.MODERATE,
                title    = f"Top Up Emergency Fund to 6 Months (Gap: ₹{self.gap:,.0f})",
                detail   = (
                    f"Emergency fund covers {self.months:.1f} months. "
                    f"Need ₹{self.gap:,.0f} more to reach the 6-month target."
                ),
                action   = (
                    f"Save ₹{monthly_save:,}/month in a liquid fund or sweep FD. "
                    "Top up with next salary hike increment for 6 months."
                ),
                impact   = (
                    f"Completing the fund in ~{max(int(self.gap / monthly_save), 1)} months "
                    "gives full financial security buffer."
                ),
                tags     = ["emergency-fund", "liquid"],
            ))
        else:
            recs.append(Recommendation(
                domain   = Domain.EMERGENCY,
                severity = Severity.POSITIVE,
                title    = f"Strong Emergency Fund — {self.months:.1f} Months Coverage",
                detail   = (
                    f"Emergency fund of ₹{self.ef:,.0f} covers {self.months:.1f} months of expenses. "
                    "This is above the recommended 6-month benchmark."
                ),
                action   = (
                    "Keep the fund in a liquid fund or sweep FD (earning 5–7%). "
                    "Any amount beyond 9 months should be invested in equity."
                ),
                impact   = (
                    f"₹{max(int(self.ef - self.target_ef * 1.5), 0):,} above 9-month threshold "
                    "can be deployed into SIPs for higher returns."
                ),
                tags     = ["emergency-fund", "strength"],
            ))

        # Suggest liquid fund parking
        if self.ef > 50_000 and self.months < 9:
            recs.append(Recommendation(
                domain   = Domain.EMERGENCY,
                severity = Severity.LOW,
                title    = "Park Emergency Fund in Liquid Mutual Fund — Not Savings Account",
                detail   = (
                    "Savings accounts earn 3–3.5% interest. Liquid mutual funds earn 5.5–7% "
                    "with same-day redemption — better return with no liquidity sacrifice."
                ),
                action   = (
                    "Move emergency fund to a liquid mutual fund (e.g., SBI Liquid Fund, "
                    "Nippon Liquid Fund) via any mutual fund platform."
                ),
                impact   = (
                    f"₹{self.ef:,.0f} at 6% vs 3.5% = "
                    f"₹{int(self.ef * 0.025):,}/year extra interest."
                ),
                tags     = ["liquid-fund", "idle-cash"],
            ))

        grade = "A" if score >= 80 else ("B" if score >= 60 else ("C" if score >= 40 else "D"))
        summary = DomainSummary(
            domain   = Domain.EMERGENCY,
            score    = round(score, 1),
            grade    = grade,
            headline = (
                f"Emergency fund: ₹{self.ef:,.0f} — covers {self.months:.1f} months. "
                f"{'Adequate.' if self.months >= 6 else f'Need ₹{self.gap:,.0f} more.'}"
            ),
            metrics  = {
                "Emergency Fund":    f"₹{self.ef:,.0f}",
                "Months Covered":    f"{self.months:.1f} months",
                "Monthly Burn":      f"₹{self.monthly_out:,.0f}",
                "Target (6 months)": f"₹{self.target_ef:,.0f}",
                "Gap":               f"₹{self.gap:,.0f}",
            },
            suggestions = [
                "Park fund in liquid MF — earns more than savings account",
                "Set auto-replenishment if emergency fund is ever used",
                "Never mix emergency fund with investment corpus",
                "Review target every year as expenses grow",
                "Consider 9-month fund if in a volatile job/sector",
            ],
        )
        return summary, recs, alerts


class _TaxAnalyser:
    """Analyses tax efficiency: 80C utilisation, NPS, insurance deductions, regime choice."""

    def __init__(self, twin):
        self.twin         = twin
        t = twin
        self.annual_income = t.total_income * 12
        # Proxy deduction utilisations from available fields
        self.ppf_pa       = t.ppf_investment       # assume annual contribution stored
        self.nps_pa       = t.nps_investment        # assume annual NPS
        self.life_ins_prem = min(t.life_insurance * 0.01, 50_000)  # ~1% of cover as premium proxy
        self.health_prem  = min(t.health_insurance * 0.02, 25_000) # ~2% of cover as premium proxy
        self.sip_elss     = min(t.sip_amount * 12 * 0.30, 150_000) # assume 30% SIPs in ELSS
        # 80C utilised
        self.deduction_80c = min(self.ppf_pa + self.sip_elss + self.life_ins_prem, _80C_LIMIT)
        self.unused_80c    = max(_80C_LIMIT - self.deduction_80c, 0)
        self.nps_deduction = min(self.nps_pa, _NPS_80CCD_LIMIT)
        self.unused_nps    = max(_NPS_80CCD_LIMIT - self.nps_deduction, 0)
        self.income        = t.total_income

    def analyse(self) -> tuple[DomainSummary, List[Recommendation], List[Alert]]:
        recs: List[Recommendation] = []
        alerts: List[Alert]        = []
        t = self.twin

        # Score: based on how much of available deductions are being used
        max_deductions = _80C_LIMIT + _NPS_80CCD_LIMIT + _80D_SELF_LIMIT
        used_deductions = self.deduction_80c + self.nps_deduction + self.health_prem
        score = min(used_deductions / max_deductions, 1.0) * 100

        # Estimate tax bracket
        tax_bracket = self._estimate_bracket()

        # ── Recommendations ──────────────────────────────────────────────────
        # Unused 80C headroom
        if self.unused_80c > 10_000:
            tax_saved = int(self.unused_80c * tax_bracket)
            recs.append(Recommendation(
                domain   = Domain.TAX,
                severity = Severity.HIGH if self.unused_80c > 75_000 else Severity.MODERATE,
                title    = f"₹{self.unused_80c:,} Unused 80C Deduction — Leaving Tax on the Table",
                detail   = (
                    f"Only ₹{self.deduction_80c:,.0f} of the ₹1,50,000 80C limit is used. "
                    f"You have ₹{self.unused_80c:,} of headroom left — this costs "
                    f"₹{tax_saved:,}/year in unnecessary tax."
                ),
                action   = (
                    f"Invest ₹{self.unused_80c:,}/year in ELSS mutual funds (lock-in 3 years, "
                    "highest return among 80C instruments). Alternatively, max out PPF contributions."
                ),
                impact   = f"₹{tax_saved:,}/year tax saving at {tax_bracket:.0%} bracket.",
                tags     = ["80c", "elss", "ppf", "tax-saving"],
            ))

        # NPS headroom
        if self.unused_nps > 10_000:
            nps_tax_saved = int(self.unused_nps * tax_bracket)
            recs.append(Recommendation(
                domain   = Domain.TAX,
                severity = Severity.MODERATE,
                title    = f"₹{self.unused_nps:,} 80CCD(1B) NPS Deduction Unused",
                detail   = (
                    "NPS allows an additional ₹50,000 deduction under 80CCD(1B) "
                    "— separate from and in addition to the ₹1.5L 80C limit."
                ),
                action   = (
                    f"Contribute ₹{int(self.unused_nps / 12):,}/month to NPS Tier-I "
                    "to claim the full ₹50,000 additional deduction."
                ),
                impact   = (
                    f"₹{nps_tax_saved:,}/year tax saving + retirement corpus building."
                ),
                tags     = ["nps", "80ccd", "retirement"],
            ))

        # Health insurance deduction
        if self.health_prem < _80D_SELF_LIMIT * 0.5:
            recs.append(Recommendation(
                domain   = Domain.TAX,
                severity = Severity.MODERATE,
                title    = "Improve Health Insurance — Claim Full 80D Deduction",
                detail   = (
                    f"Estimated health insurance premium: ~₹{int(self.health_prem):,}/year. "
                    f"80D allows ₹25,000 for self/family + ₹50,000 for senior citizen parents. "
                    f"You may be leaving ₹{int((_80D_SELF_LIMIT - self.health_prem) * tax_bracket):,}/year "
                    "in tax savings unused."
                ),
                action   = (
                    "Upgrade health cover to ₹10L+ family floater. "
                    "If parents are senior citizens, buy separate ₹5L+ policy for ₹50,000 80D deduction."
                ),
                impact   = (
                    f"Full 80D utilisation saves ₹{int(_80D_SELF_LIMIT * tax_bracket):,}/year in taxes "
                    "while providing critical health protection."
                ),
                tags     = ["health-insurance", "80d", "tax"],
            ))

        # Regime suggestion
        if self.annual_income > 700_000:
            recs.append(Recommendation(
                domain   = Domain.TAX,
                severity = Severity.LOW,
                title    = "Compare Old vs New Tax Regime Annually",
                detail   = (
                    f"At annual income ₹{self.annual_income:,.0f}, the optimal regime depends on "
                    "your total deductions. Old regime wins if deductions exceed ~₹3.75L."
                ),
                action   = (
                    "Use the Tax Intelligence module (Page 5) to compute exact savings "
                    "under both regimes. File ITR under the optimal regime each year."
                ),
                impact   = (
                    "Choosing the correct regime can save ₹10,000–₹50,000+ annually "
                    "depending on deductions."
                ),
                tags     = ["tax-regime", "old-vs-new", "itr"],
            ))

        # HRA if renting
        if t.rent > 0:
            recs.append(Recommendation(
                domain   = Domain.TAX,
                severity = Severity.LOW,
                title    = "Claim HRA Exemption If in Old Tax Regime",
                detail   = (
                    f"You pay ₹{t.rent:,.0f}/month rent. Under the old regime, "
                    "HRA exemption can be significant — minimum of actual HRA, "
                    "actual rent − 10% of basic, or 50% of basic (metro cities)."
                ),
                action   = (
                    "Ensure rent receipts are maintained. If your employer doesn't provide HRA, "
                    "claim deduction under Section 80GG (₹60,000/year max)."
                ),
                impact   = (
                    f"HRA exemption can offset ₹{min(int(t.rent * 12 * 0.5), 200_000):,}+ "
                    "of taxable income in metro cities."
                ),
                tags     = ["hra", "rent", "old-regime"],
            ))

        grade = "A" if score >= 80 else ("B" if score >= 60 else ("C" if score >= 40 else "D"))
        summary = DomainSummary(
            domain   = Domain.TAX,
            score    = round(score, 1),
            grade    = grade,
            headline = (
                f"80C used: ₹{self.deduction_80c:,.0f}/₹1,50,000. "
                f"NPS: ₹{self.nps_deduction:,.0f}/₹50,000. "
                f"Estimated bracket: {tax_bracket:.0%}."
            ),
            metrics  = {
                "Annual Income":    f"₹{self.annual_income:,.0f}",
                "80C Used":         f"₹{self.deduction_80c:,.0f}",
                "80C Headroom":     f"₹{self.unused_80c:,.0f}",
                "NPS Used":         f"₹{self.nps_deduction:,.0f}",
                "NPS Headroom":     f"₹{self.unused_nps:,.0f}",
                "Health Premium":   f"₹{int(self.health_prem):,}",
                "Est. Tax Bracket": f"{tax_bracket:.0%}",
            },
            suggestions = [
                "Max 80C: ₹1.5L in ELSS/PPF/EPF each financial year",
                "Add ₹50,000 NPS for extra 80CCD(1B) deduction",
                "Buy ₹10L+ health cover — deduct up to ₹25,000 under 80D",
                "Compare old vs new regime every March before filing",
                "Claim HRA if you pay rent (collect rent receipts)",
            ],
        )
        return summary, recs, alerts

    def _estimate_bracket(self) -> float:
        """Approximate marginal tax bracket under old regime."""
        ti = max(self.annual_income - 50_000, 0)  # std deduction
        if ti <= 250_000:   return 0.0
        if ti <= 500_000:   return 0.05
        if ti <= 1_000_000: return 0.20
        return 0.30


# ══════════════════════════════════════════════════════════════════════════════
# FinancialCoach — Main Entry Point
# ══════════════════════════════════════════════════════════════════════════════

class FinancialCoach:
    """
    Rule-based AI Financial Coach.

    Analyses a FinancialDigitalTwin across 5 domains and generates a structured,
    prioritised CoachingReport with alerts, recommendations, domain summaries,
    and natural-language narrative.

    Usage:
        report = FinancialCoach(twin).generate_report()
    """

    # Severity sort order
    _SEVERITY_ORDER = {
        Severity.CRITICAL: 0,
        Severity.HIGH:     1,
        Severity.MODERATE: 2,
        Severity.LOW:      3,
        Severity.POSITIVE: 4,
    }

    def __init__(self, twin):
        self.twin = twin

    def generate_report(self) -> CoachingReport:
        """
        Runs all 5 domain analysers, merges results, and returns CoachingReport.
        """
        analysers = [
            _SavingsAnalyser(self.twin),
            _InvestmentAnalyser(self.twin),
            _DebtAnalyser(self.twin),
            _EmergencyFundAnalyser(self.twin),
            _TaxAnalyser(self.twin),
        ]

        all_recs:    List[Recommendation]        = []
        all_alerts:  List[Alert]                 = []
        summaries:   Dict[str, DomainSummary]    = {}

        for analyser in analysers:
            summary, recs, alerts = analyser.analyse()
            summaries[summary.domain.value] = summary
            all_recs.extend(recs)
            all_alerts.extend(alerts)

        # Sort alerts: Critical → High
        all_alerts.sort(key=lambda a: self._SEVERITY_ORDER[a.severity])

        # Sort recommendations: Critical first, then Positive last
        all_recs.sort(key=lambda r: self._SEVERITY_ORDER[r.severity])

        # Overall score: average domain scores (weighted slightly toward savings + debt)
        weights = {
            Domain.SAVINGS.value:    0.25,
            Domain.INVESTMENT.value: 0.20,
            Domain.DEBT.value:       0.25,
            Domain.EMERGENCY.value:  0.20,
            Domain.TAX.value:        0.10,
        }
        overall = sum(
            summaries[d].score * w
            for d, w in weights.items()
            if d in summaries
        )

        # Top action = first critical/high recommendation
        top = next(
            (r for r in all_recs if r.severity in (Severity.CRITICAL, Severity.HIGH)),
            all_recs[0] if all_recs else None,
        )
        top_action = top.action if top else "Your finances are in great shape — keep it up!"

        narrative = self._build_narrative(summaries, all_recs, all_alerts, overall)

        return CoachingReport(
            twin_name        = self.twin.name,
            overall_score    = round(overall, 1),
            risk_level       = self.twin.risk_level,
            alerts           = all_alerts,
            recommendations  = all_recs,
            domain_summaries = summaries,
            top_action       = top_action,
            narrative        = narrative,
        )

    def _build_narrative(
        self,
        summaries: Dict[str, DomainSummary],
        recs: List[Recommendation],
        alerts: List[Alert],
        overall: float,
    ) -> str:
        """Generates a full natural-language coaching paragraph."""
        t = self.twin
        sav_sum  = summaries.get(Domain.SAVINGS.value)
        inv_sum  = summaries.get(Domain.INVESTMENT.value)
        debt_sum = summaries.get(Domain.DEBT.value)
        ef_sum   = summaries.get(Domain.EMERGENCY.value)

        grade = "A" if overall >= 80 else ("B" if overall >= 60 else ("C" if overall >= 40 else "D"))
        opening = (
            f"Hello {t.name}! Your overall financial coaching score is "
            f"**{overall:.0f}/100 (Grade {grade})**.\n\n"
        )

        if len(alerts) == 0:
            alert_text = "- **Risk Status:** You have no critical risk alerts — your financial foundations are solid.\n"
        elif len([a for a in alerts if a.severity == Severity.CRITICAL]) > 0:
            n = len([a for a in alerts if a.severity == Severity.CRITICAL])
            alert_text = (
                f"- **Risk Status:** You have **{n} critical alert{'s' if n > 1 else ''}** requiring immediate attention. "
                "Address these before optimising other areas.\n"
            )
        else:
            alert_text = f"- **Risk Status:** You have {len(alerts)} advisory alerts to review.\n"

        savings_text = (
            f"- **Savings:** Your savings rate is **{sav_sum.score:.0f}/100** — "
            f"{sav_sum.headline}\n" if sav_sum else ""
        )
        invest_text = (
            f"- **Investment:** Investment discipline scores **{inv_sum.score:.0f}/100** — "
            f"{inv_sum.headline}\n" if inv_sum else ""
        )
        debt_text = (
            f"- **Debt:** Debt management is **{debt_sum.score:.0f}/100** — "
            f"{debt_sum.headline}\n" if debt_sum else ""
        )
        ef_text = (
            f"- **Emergency Fund:** Emergency preparedness: **{ef_sum.score:.0f}/100** — "
            f"{ef_sum.headline}\n" if ef_sum else ""
        )

        closing = (
            "\nReview each domain tab below for detailed recommendations. "
            "Start with the highest-severity actions and work your way down. "
            "Small, consistent steps compounded over years create extraordinary wealth."
        )

        return opening + alert_text + savings_text + invest_text + debt_text + ef_text + closing

    # ── Legacy / backward-compatible methods ────────────────────────────────────

    def get_personality_coaching_nudges(self) -> List[str]:
        """Legacy stub — use generate_report().recommendations instead."""
        report = self.generate_report()
        return [r.title for r in report.recommendations[:5]]

    def get_actionable_milestones(self) -> List[Dict[str, Any]]:
        """Legacy stub — returns top recommendations as milestone dicts."""
        report = self.generate_report()
        return [
            {"title": r.title, "action": r.action, "domain": r.domain.value, "severity": r.severity.value}
            for r in report.recommendations
            if r.severity in (Severity.CRITICAL, Severity.HIGH)
        ]

    def generate_llm_coaching_prompt(self, user_query: str) -> str:
        """Builds a structured context prompt for LLM advisory (legacy method)."""
        t = self.twin
        return (
            f"You are a certified Indian financial advisor. "
            f"User profile: {t.name}, age {t.age}, {t.occupation} in {t.city}. "
            f"Monthly income: ₹{t.total_income:,.0f}. "
            f"Expenses: ₹{t.basic_expenses:,.0f}. EMI: ₹{t.monthly_emi:,.0f}. "
            f"SIP: ₹{t.sip_amount:,.0f}. Net worth: ₹{t.net_worth:,.0f}. "
            f"Emergency fund: {t.emergency_fund / max(t.basic_expenses + t.monthly_emi, 1):.1f} months. "
            f"Risk level: {t.risk_level}. "
            f"User question: {user_query}"
        )


# ── Backward-compatible alias ────────────────────────────────────────────────
class AIBehavioralCoach:
    """
    Backward-compatible wrapper kept so existing pages don't break.
    Wraps FinancialCoach with the old interface.
    """
    def __init__(self, user_profile: Dict[str, Any], health_metrics: Dict[str, Any], cluster_label: int):
        self.user_profile   = user_profile
        self.health_metrics = health_metrics
        self.cluster_label  = cluster_label

    def get_personality_coaching_nudges(self) -> List[str]:
        return []

    def get_actionable_milestones(self) -> List[Dict[str, Any]]:
        return []

    def generate_llm_coaching_prompt(self, user_query: str) -> str:
        return ""

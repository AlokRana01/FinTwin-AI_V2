"""
Goal Planning Engine  (Module 11)
Rule-based engine that computes required monthly savings, probability of achievement,
and projected completion for 5 life-goal types.

Goals supported:
  Car Purchase | House Purchase | Marriage | Education | Vacation

Maths:
  Future Value of existing savings (compounded monthly at return_rate):
      FV_current = current_savings × (1 + r)^n

  Future Value of a monthly SIP annuity:
      FV_sip = monthly_sip × ((1+r)^n - 1) / r

  Required monthly SIP to fill the gap:
      SIP_required = (goal_amount - FV_current) × r / ((1+r)^n - 1)
      clamped to 0 if FV_current already >= goal_amount

  Probability of achievement (P):
      Based on how much of the required monthly SIP the user can commit
      relative to their available monthly surplus.
      P = min(surplus / SIP_required, 1.0) — with sigmoid smoothing.

  Projected completion date (when user commits a fixed monthly amount):
      n* = log(1 + (goal_amount - FV0) × r / monthly_contribution) / log(1 + r)

Classes:
    GoalType           — Enum of supported goals
    GoalConfig         — Static metadata per goal (icon, color, default amounts)
    GoalInput          — User-supplied parameters for one goal
    GoalResult         — Full computation output
    GoalEngine         — Core computation engine
    MultiGoalPlanner   — Plans multiple goals simultaneously with priority ranking
"""

from __future__ import annotations

import math
import datetime
import functools
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# Enums & Config
# ══════════════════════════════════════════════════════════════════════════════

class GoalType(str, Enum):
    CAR        = "Car Purchase"
    HOUSE      = "House Purchase"
    MARRIAGE   = "Marriage"
    EDUCATION  = "Education"
    VACATION   = "Vacation"


# Static metadata per goal type
GOAL_CONFIG: Dict[GoalType, Dict] = {
    GoalType.CAR: {
        "icon":           '<i class="fa-solid fa-car"></i>',
        "icon_class":     "fa-car",
        "color":          "#3B82F6",
        "bg":             "rgba(59,130,246,0.10)",
        "default_amount": 800_000,
        "default_years":  3,
        "description":    "On-road price of your dream car including registration & insurance.",
        "tips": [
            "Consider a downpayment of 30–40% to keep EMI below 15% of income",
            "Buying 2 years old used car saves 25–35% vs new",
            "Total cost of ownership (insurance + fuel + maintenance) is ~15% of car value annually",
        ],
        "return_rate": 0.07,   # Debt funds / liquid FDs (short horizon)
    },
    GoalType.HOUSE: {
        "icon":           '<i class="fa-solid fa-house"></i>',
        "icon_class":     "fa-house",
        "color":          "#10B981",
        "bg":             "rgba(16,185,129,0.10)",
        "default_amount": 2_000_000,
        "default_years":  7,
        "description":    "Down-payment (20%) + registration for your first home.",
        "tips": [
            "Target 20% downpayment to avoid PMI and get best home loan rates",
            "Add 6–8% of property value for registration, stamp duty & interiors",
            "Home loan EMI should stay under 30% of monthly income",
        ],
        "return_rate": 0.10,   # Balanced / hybrid funds (medium horizon)
    },
    GoalType.MARRIAGE: {
        "icon":           '<i class="fa-solid fa-gem"></i>',
        "icon_class":     "fa-gem",
        "color":          "#F59E0B",
        "bg":             "rgba(245,158,11,0.10)",
        "default_amount": 1_500_000,
        "default_years":  4,
        "description":    "Total wedding expenses including venue, catering, trousseau & honeymoon.",
        "tips": [
            "Budget 60% venue + catering, 20% attire, 10% decor, 10% honeymoon",
            "Book venue 12–18 months in advance for 20–30% discounts",
            "A destination wedding can often cost less than a city hotel wedding",
        ],
        "return_rate": 0.09,   # Conservative hybrid
    },
    GoalType.EDUCATION: {
        "icon":           '<i class="fa-solid fa-graduation-cap"></i>',
        "icon_class":     "fa-graduation-cap",
        "color":          "#8B5CF6",
        "bg":             "rgba(139,92,246,0.10)",
        "default_amount": 2_500_000,
        "default_years":  10,
        "description":    "Higher education corpus — MBA / abroad degree / professional course.",
        "tips": [
            "Indian MBA costs ₹20–35L, US/UK costs ₹60–100L total",
            "Education loans have 0.5% lower rate if you start repaying during course",
            "Education savings schemes for child education",
        ],
        "return_rate": 0.12,   # Equity mutual funds (long horizon)
    },
    GoalType.VACATION: {
        "icon":           '<i class="fa-solid fa-plane"></i>',
        "icon_class":     "fa-plane",
        "color":          "#EC4899",
        "bg":             "rgba(236,72,153,0.10)",
        "default_amount": 300_000,
        "default_years":  2,
        "description":    "International vacation fund — flights, hotels, activities & travel insurance.",
        "tips": [
            "Book international flights 3–4 months early to save 20–40%",
            "Travel in shoulder season (Apr–May, Sep–Oct) for 30% lower prices",
            "Multi-city passes and rail passes can reduce internal travel cost by 40%",
        ],
        "return_rate": 0.065,  # Liquid / ultra-short debt (very short horizon)
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GoalInput:
    """
    User-supplied parameters for a single goal planning computation.
    """
    goal_type:        GoalType
    goal_amount:      float          # Target corpus in ₹
    current_savings:  float          # Already saved/earmarked for this goal
    target_date:      datetime.date  # Desired completion date
    monthly_contribution: float = 0.0  # User's intended monthly savings for this goal (0 = compute)
    return_rate:      Optional[float] = None  # Annual return; defaults to GOAL_CONFIG value
    inflation_rate:   float = 0.06   # 6% annual inflation applied to goal amount

    def __post_init__(self):
        if self.goal_amount <= 0:
            raise ValueError("Goal amount must be greater than zero.")
        if self.current_savings < 0:
            raise ValueError("Current savings cannot be negative.")
        if self.monthly_contribution < 0:
            raise ValueError("Monthly contribution cannot be negative.")
        if self.target_date <= datetime.date.today():
            raise ValueError("Goal target date must be set in the future.")

    @property
    def months_remaining(self) -> int:
        today = datetime.date.today()
        delta = (self.target_date.year - today.year) * 12 + (self.target_date.month - today.month)
        return max(delta, 1)

    @property
    def effective_return_rate(self) -> float:
        return self.return_rate or GOAL_CONFIG[self.goal_type]["return_rate"]

    @property
    def monthly_rate(self) -> float:
        return self.effective_return_rate / 12

    @property
    def inflation_adjusted_goal(self) -> float:
        """Goal amount adjusted for inflation over the horizon."""
        months = self.months_remaining
        return self.goal_amount * (1 + self.inflation_rate) ** (months / 12)


@dataclass
class MonthlyProjection:
    """One month in the savings trajectory."""
    month:         int
    date:          datetime.date
    corpus:        float   # Total accumulated corpus at month end
    sip_cumulative: float  # Cumulative SIP contributions
    returns_earned: float  # Cumulative investment returns


@dataclass
class GoalResult:
    """
    Full computation output for one goal.
    """
    goal_input:           GoalInput
    inflation_adj_goal:   float          # Inflation-adjusted target
    fv_current_savings:   float          # Future value of current savings at target date
    sip_required:         float          # Monthly SIP needed to hit goal on time
    gap:                  float          # Amount still to fill (goal - FV_current)
    is_already_funded:    bool           # True if current savings alone are sufficient
    probability:          float          # 0.0–1.0 probability of achievement
    probability_label:    str            # "High" / "Moderate" / "Low" / "Very Low"
    projected_completion: Optional[datetime.date]  # When goal is hit with chosen contribution
    shortfall_months:     int            # 0 if on time, else months late
    trajectory:           pd.DataFrame   # Month-by-month projection table
    milestone_months:     List[int]      # Months at which 25/50/75/100% is reached
    tips:                 List[str]      # Goal-type specific tips

    @property
    def goal_name(self) -> str:
        return self.goal_input.goal_type.value

    @property
    def icon(self) -> str:
        return GOAL_CONFIG[self.goal_input.goal_type]["icon"]

    @property
    def color(self) -> str:
        return GOAL_CONFIG[self.goal_input.goal_type]["color"]

    @property
    def probability_color(self) -> str:
        if self.probability >= 0.80: return "#10B981"
        if self.probability >= 0.55: return "#F59E0B"
        if self.probability >= 0.30: return "#EF4444"
        return "#7C3AED"


# ══════════════════════════════════════════════════════════════════════════════
# Goal Engine
# ══════════════════════════════════════════════════════════════════════════════

class GoalEngine:
    """
    Core computation engine for a single financial goal.

    Usage:
        gi     = GoalInput(goal_type=GoalType.HOUSE, goal_amount=2_000_000,
                           current_savings=200_000, target_date=date(2032, 1, 1))
        result = GoalEngine(gi, monthly_surplus=25_000).compute()
    """

    def __init__(self, goal_input: GoalInput, monthly_surplus: float = 0.0):
        """
        Args:
            goal_input:      GoalInput instance with user parameters.
            monthly_surplus: User's available monthly surplus for all goals.
        """
        self.gi       = goal_input
        self.surplus  = max(monthly_surplus, 0.0)

    def compute(self) -> GoalResult:
        """Runs the full goal planning computation and returns GoalResult."""
        gi     = self.gi
        n      = gi.months_remaining
        r      = gi.monthly_rate
        target = gi.inflation_adjusted_goal

        # Future value of existing savings
        fv_current = gi.current_savings * (1 + r) ** n

        # Gap: how much more corpus is needed
        gap = max(target - fv_current, 0.0)
        is_funded = fv_current >= target

        # Required monthly SIP to fund the gap
        if is_funded or gap == 0:
            sip_required = 0.0
        elif r == 0:
            sip_required = gap / max(n, 1)
        else:
            denom = (1 + r) ** n - 1
            sip_required = (gap * r / denom) if denom > 0 else (gap / max(n, 1))

        # User's monthly contribution for this goal
        contribution = gi.monthly_contribution if gi.monthly_contribution > 0 else self.surplus

        # Probability of achievement
        prob = self._compute_probability(sip_required, contribution, is_funded)
        prob_label = self._probability_label(prob)

        # Projected completion date with user's contribution
        projected_date, shortfall_months = self._projected_completion(
            target, gi.current_savings, contribution, r, n
        )

        # Month-by-month trajectory
        trajectory = self._build_trajectory(
            target         = target,
            current_savings = gi.current_savings,
            monthly_sip    = max(contribution, sip_required * 0.5) if not is_funded else 0,
            r              = r,
            months         = max(n + shortfall_months + 6, n + 12),
            target_months  = n,
        )

        # Milestone months (25/50/75/100%)
        milestones = self._find_milestones(trajectory, target)

        return GoalResult(
            goal_input           = gi,
            inflation_adj_goal   = round(target, 0),
            fv_current_savings   = round(fv_current, 0),
            sip_required         = round(sip_required, 0),
            gap                  = round(gap, 0),
            is_already_funded    = is_funded,
            probability          = round(prob, 4),
            probability_label    = prob_label,
            projected_completion = projected_date,
            shortfall_months     = shortfall_months,
            trajectory           = trajectory,
            milestone_months     = milestones,
            tips                 = GOAL_CONFIG[gi.goal_type]["tips"],
        )

    @classmethod
    def clear_cache(cls) -> None:
        """Clears any cached goal calculation state."""
        pass

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_probability(sip_required: float, contribution: float, is_funded: bool) -> float:
        """
        Sigmoid-smoothed probability based on contribution-to-required ratio.
        P = 1.0 if already funded.
        P → sigmoid(6 × (ratio - 0.5)) mapped to [0.02, 0.99]
        """
        if is_funded or sip_required <= 0:
            return 1.0
        if contribution <= 0:
            return 0.02

        ratio = contribution / sip_required
        # Sigmoid: 0.5 at ratio=0.5, 0.98 at ratio=1.0+
        raw = 1 / (1 + math.exp(-6 * (ratio - 0.5)))
        return round(min(max(raw, 0.02), 0.99), 4)

    @staticmethod
    def _probability_label(prob: float) -> str:
        if prob >= 0.80: return "High"
        if prob >= 0.55: return "Moderate"
        if prob >= 0.30: return "Low"
        return "Very Low"

    @staticmethod
    def _projected_completion(
        target: float, current: float, contribution: float, r: float, n_target: int
    ) -> Tuple[Optional[datetime.date], int]:
        """
        Computes the month when corpus first crosses `target` with given monthly contribution.
        Returns (projected_date, months_late) where months_late = max(projected_month - n_target, 0).
        """
        if contribution <= 0:
            return None, 99

        corpus = current
        for m in range(1, 601):   # cap at 50 years
            corpus = corpus * (1 + r) + contribution
            if corpus >= target:
                shortfall = max(m - n_target, 0)
                proj_date = datetime.date.today() + datetime.timedelta(days=m * 30.44)
                return proj_date, shortfall
        return None, 99

    @staticmethod
    def _build_trajectory(
        target: float,
        current_savings: float,
        monthly_sip: float,
        r: float,
        months: int,
        target_months: int,
    ) -> pd.DataFrame:
        """
        Builds a month-by-month savings projection DataFrame.
        Columns: Month, Date, Corpus, SIP_Cumulative, Returns_Earned, Target, On_Track
        """
        rows = []
        corpus   = current_savings
        sip_cum  = 0.0
        prev_corpus = current_savings
        today = datetime.date.today()

        for m in range(1, months + 1):
            corpus = corpus * (1 + r) + monthly_sip
            sip_cum += monthly_sip
            returns  = corpus - current_savings - sip_cum

            proj_date = today + datetime.timedelta(days=m * 30.44)
            on_track  = corpus >= (target * m / target_months) if m <= target_months else (corpus >= target)

            rows.append({
                "Month":          m,
                "Date":           proj_date,
                "Corpus":         round(corpus, 0),
                "SIP_Cumulative": round(sip_cum, 0),
                "Returns_Earned": round(returns, 0),
                "Target":         round(target, 0),
                "On_Track":       on_track,
            })

        return pd.DataFrame(rows)

    @staticmethod
    def _find_milestones(trajectory: pd.DataFrame, target: float) -> List[int]:
        """Returns the month indices when 25%, 50%, 75%, 100% of target is first crossed."""
        milestones = []
        thresholds = [0.25, 0.50, 0.75, 1.00]
        for t in thresholds:
            crossed = trajectory[trajectory["Corpus"] >= target * t]
            milestones.append(int(crossed["Month"].iloc[0]) if not crossed.empty else -1)
        return milestones


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Goal Planner
# ══════════════════════════════════════════════════════════════════════════════

class MultiGoalPlanner:
    """
    Plans multiple goals simultaneously and distributes available surplus.

    Priority ordering:
        Emergency Fund → Education → House → Marriage → Car → Vacation
        (configurable via priority_order)

    Usage:
        planner = MultiGoalPlanner(twin, goals=[gi1, gi2, gi3])
        results = planner.plan()
    """

    DEFAULT_PRIORITY = [
        GoalType.EDUCATION,
        GoalType.HOUSE,
        GoalType.MARRIAGE,
        GoalType.CAR,
        GoalType.VACATION,
    ]

    def __init__(self, twin, goals: List[GoalInput], priority_order: Optional[List[GoalType]] = None):
        self.twin     = twin
        self.goals    = goals
        self.priority = priority_order or self.DEFAULT_PRIORITY
        self.monthly_surplus = max(
            twin.total_income - twin.basic_expenses - twin.monthly_emi, 0.0
        )

    def plan(self) -> List[GoalResult]:
        """
        Distributes available monthly surplus across goals by priority.
        Returns a list of GoalResult, one per goal input.

        Strategy:
          - Sort goals by (priority rank, target_date).
          - Allocate SIP_required from surplus sequentially.
          - Remaining surplus after all required SIPs → bonus to top-priority goal.
        """
        # Sort by priority rank then target date
        sorted_goals = sorted(
            self.goals,
            key=lambda g: (
                self.priority.index(g.goal_type) if g.goal_type in self.priority else 99,
                g.months_remaining,
            ),
        )

        remaining_surplus = self.monthly_surplus
        contributions: Dict[int, float] = {}  # index → allocated contribution

        for idx, gi in enumerate(sorted_goals):
            engine = GoalEngine(gi, monthly_surplus=0)
            temp   = engine.compute()
            alloc  = min(temp.sip_required, remaining_surplus)
            contributions[idx] = max(alloc, gi.monthly_contribution)
            remaining_surplus  = max(remaining_surplus - alloc, 0.0)

        # Final results with allocated contributions
        results: List[GoalResult] = []
        for idx, gi in enumerate(sorted_goals):
            contrib = contributions.get(idx, 0.0)
            result  = GoalEngine(gi, monthly_surplus=contrib).compute()
            results.append(result)

        return results

    def summary_table(self, results: List[GoalResult]) -> pd.DataFrame:
        """Returns a summary DataFrame of all goals for display."""
        rows = []
        for r in results:
            rows.append({
                "Goal":              r.goal_name,
                "Target Amount":     f"₹{r.goal_input.goal_amount:,.0f}",
                "Inflation Adj.":    f"₹{r.inflation_adj_goal:,.0f}",
                "Target Date":       r.goal_input.target_date.strftime("%b %Y"),
                "SIP Required":      f"₹{r.sip_required:,.0f}/mo",
                "Probability":       f"{r.probability:.0%} ({r.probability_label})",
                "Projected Done":    r.projected_completion.strftime("%b %Y") if r.projected_completion else "N/A",
                "Months Late":       r.shortfall_months if r.shortfall_months > 0 else "On Time",
            })
        return pd.DataFrame(rows)

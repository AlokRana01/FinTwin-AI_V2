"""
Explainable AI (XAI) Module  (Module 9)
Full SHAP-powered explainability for two prediction surfaces:

  1. HealthScoreExplainer
     Additive decomposition of the weighted 6-component Financial Health Score.
     Each component's signed contribution vs. a 50/100 neutral baseline is computed,
     producing statements like:
       "High EMI reduced your score by 14.2 points."
       "Strong savings increased your score by 8.1 points."

  2. ForecastExplainer
     True SHAP TreeExplainer on the XGBoost savings + net_worth models.
     Returns shapley values, base values, and feature contribution rankings for both
     prediction targets (monthly savings at horizon H, net worth at horizon H).
     Falls back to XGBoost native feature_importances_ if SHAP is unavailable.

  3. ExplainableAI  (orchestrator facade)
     Unified entry-point used by the Streamlit page.
     explain_health_score(twin)         → HealthScoreResult
     explain_forecast(twin, predictor)  → ForecastResult
     generate_plain_english(twin, predictor) → List[str] narrative sentences

Classes:
    ComponentContribution   - Dataclass for one factor's signed SHAP-style contribution
    HealthScoreResult       - Full explainability output for health score
    ForecastContribution    - Dataclass for one feature's SHAP value in forecasting
    ForecastResult          - Full explainability output for forecasting
    HealthScoreExplainer    - Computes health score decomposition
    ForecastExplainer       - Wraps XGBoost with SHAP TreeExplainer
    ExplainableAI           - Facade / orchestrator
"""

from __future__ import annotations

import threading
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from models.twin_engine import HealthScoreEngine

# ── In-Memory TreeExplainer Cache & Lock ────────────────────────────────────────
_TREE_EXPLAINER_CACHE: Dict[int, Any] = {}
_TREE_EXPLAINER_LOCK = threading.Lock()


def get_tree_explainer(model: Any) -> Any:
    """
    Returns a cached shap.TreeExplainer instance for the provided fitted tree model.
    Process-local, thread-safe, and keyed by model instance id.
    Contains zero user-specific state, request IDs, or provenance.
    """
    import shap
    model_key = id(model)
    if model_key in _TREE_EXPLAINER_CACHE:
        return _TREE_EXPLAINER_CACHE[model_key]

    with _TREE_EXPLAINER_LOCK:
        if model_key in _TREE_EXPLAINER_CACHE:
            return _TREE_EXPLAINER_CACHE[model_key]
        explainer = shap.TreeExplainer(model)
        _TREE_EXPLAINER_CACHE[model_key] = explainer
        return explainer


def clear_tree_explainer_cache() -> None:
    """Clears the in-memory TreeExplainer cache."""
    with _TREE_EXPLAINER_LOCK:
        _TREE_EXPLAINER_CACHE.clear()


# ══════════════════════════════════════════════════════════════════════════════
# Data Structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ComponentContribution:
    """
    A single health-score component's signed SHAP-style contribution.
    contribution > 0  →  this factor boosted the score above baseline
    contribution < 0  →  this factor dragged the score below baseline
    """
    component:      str     # e.g. "EMI Burden"
    raw_value:      float   # actual ratio/value, e.g. 35.2 (%)
    unit:           str     # e.g. "%" or " months"
    score:          float   # 0-100 sub-score
    weight:         float   # 0-1 weight in overall score
    baseline_score: float   # neutral baseline (50/100)
    contribution:   float   # signed impact on overall score
    benchmark:      str     # human-readable benchmark string

    @property
    def direction(self) -> str:
        return "positive" if self.contribution >= 0 else "negative"

    @property
    def abs_contribution(self) -> float:
        return abs(self.contribution)

    def to_sentence(self) -> str:
        """Generates a plain-English explanation sentence for this component."""
        sign_word = "boosted" if self.contribution >= 0 else "reduced"
        icon      = '<i class="fa-solid fa-arrow-trend-up" style="color:#10B981; margin-right:4px;"></i>' if self.contribution >= 0 else '<i class="fa-solid fa-arrow-trend-down" style="color:#EF4444; margin-right:4px;"></i>'
        pts       = abs(self.contribution)

        component_descriptions = {
            "Savings Rate":        f"your savings rate of {self.raw_value:.1f}%",
            "EMI Burden":          f"your EMI burden of {self.raw_value:.1f}% of income",
            "Emergency Fund":      f"your emergency fund covering {self.raw_value:.1f} months",
            "Investment Ratio":    f"your SIP investment ratio of {self.raw_value:.1f}%",
            "Insurance Adequacy":  f"your insurance adequacy of {self.raw_value:.1f}%",
            "Debt Level":          f"your debt level of {self.raw_value:.2f}× annual income",
        }
        desc = component_descriptions.get(self.component, f"your {self.component.lower()}")
        return f"{icon} <strong>{self.component}</strong> — {desc} {sign_word} your score by <strong>{pts:+.1f} pts</strong>."


@dataclass
class HealthScoreResult:
    """
    Complete explainability output for the Financial Health Score.
    """
    overall_score:      float
    baseline_score:     float                       # Neutral baseline (all at 50)
    contributions:      List[ComponentContribution] # Sorted by abs impact
    narrative:          List[str]                   # Plain-English sentences
    force_plot_data:    Dict[str, Any]              # For horizontal waterfall chart
    summary_data:       pd.DataFrame                # For bar chart (feature importance)

    @property
    def top_positive(self) -> List[ComponentContribution]:
        return [c for c in self.contributions if c.contribution > 0]

    @property
    def top_negative(self) -> List[ComponentContribution]:
        return [c for c in self.contributions if c.contribution < 0]


@dataclass
class ForecastContribution:
    """
    A single input feature's SHAP value for a forecasting prediction.
    """
    feature:        str    # e.g. "monthly_emi"
    display_name:   str    # e.g. "Monthly EMI"
    value:          float  # User's actual feature value (₹)
    shap_value:     float  # Signed contribution to prediction
    direction:      str    # "positive" or "negative"

    def to_sentence(self, target: str = "savings") -> str:
        """Generates a plain-English sentence for this feature's SHAP contribution."""
        sign_word = "increased" if self.shap_value >= 0 else "reduced"
        icon      = '<i class="fa-solid fa-arrow-trend-up" style="color:#10B981; margin-right:6px;"></i>' if self.shap_value >= 0 else '<i class="fa-solid fa-arrow-trend-down" style="color:#EF4444; margin-right:6px;"></i>'
        amt       = abs(self.shap_value)
        return (
            f"{icon} <strong>{self.display_name}</strong> (₹{self.value:,.0f}/mo) "
            f"{sign_word} predicted {target} by <strong>₹{amt:,.0f}</strong>."
        )


@dataclass
class ForecastResult:
    """
    Complete explainability output for the XGBoost forecast models.
    """
    # Savings model
    savings_prediction:    float
    savings_base_value:    float
    savings_contributions: List[ForecastContribution]
    savings_narrative:     List[str]

    # Net worth model
    net_worth_prediction:    float
    net_worth_base_value:    float
    net_worth_contributions: List[ForecastContribution]
    net_worth_narrative:     List[str]

    # Combined feature importance (average of both models)
    combined_importance:   pd.DataFrame

    horizon_months:        int


# ══════════════════════════════════════════════════════════════════════════════
# Health Score Explainer
# ══════════════════════════════════════════════════════════════════════════════

# Neutral baseline: if every component scored exactly 50/100
_BASELINE_COMPONENT_SCORE = 50.0

# Weights (must match HealthScoreEngine)
_COMPONENT_WEIGHTS = {
    "Savings Rate":       0.25,
    "EMI Burden":         0.20,
    "Emergency Fund":     0.20,
    "Investment Ratio":   0.15,
    "Insurance Adequacy": 0.10,
    "Debt Level":         0.10,
}

# Map engine component key → display name
_COMPONENT_KEY_MAP = {
    "savings_rate":       "Savings Rate",
    "emi_burden":         "EMI Burden",
    "emergency_fund":     "Emergency Fund",
    "investment_ratio":   "Investment Ratio",
    "insurance_adequacy": "Insurance Adequacy",
    "debt_level":         "Debt Level",
}

_BASELINE_OVERALL = sum(
    _BASELINE_COMPONENT_SCORE * w for w in _COMPONENT_WEIGHTS.values()
)  # = 50.0


class HealthScoreExplainer:
    """
    Computes SHAP-style additive decomposition of the Financial Health Score.

    The score is a weighted sum of 6 components (each 0-100).
    Baseline: all components at 50/100 → baseline overall = 50.0

    Contribution of component i:
        contribution_i = (score_i - 50) × weight_i

    Sum of all contributions = overall_score - baseline = overall_score - 50
    This guarantees additivity (SHAP's efficiency axiom).
    """

    def __init__(self, twin):
        self.twin   = twin
        self.engine = HealthScoreEngine(twin)

    def explain(self) -> HealthScoreResult:
        """
        Runs the full health score explanation pipeline.

        Returns:
            HealthScoreResult with all explanation artefacts.
        """
        score_data  = self.engine.compute_overall_health_score()
        overall     = score_data["overall_score"]
        components  = score_data["components"]

        contributions: List[ComponentContribution] = []

        for key, display_name in _COMPONENT_KEY_MAP.items():
            comp     = components.get(key, {})
            score    = comp.get("score",     0.0)
            raw_val  = comp.get("raw_value", 0.0)
            unit     = comp.get("unit",      "")
            bench    = comp.get("benchmark", "")
            weight   = _COMPONENT_WEIGHTS[display_name]

            # Signed contribution vs. neutral baseline
            contribution = (score - _BASELINE_COMPONENT_SCORE) * weight

            contributions.append(ComponentContribution(
                component      = display_name,
                raw_value      = raw_val,
                unit           = unit,
                score          = score,
                weight         = weight,
                baseline_score = _BASELINE_COMPONENT_SCORE,
                contribution   = round(contribution, 2),
                benchmark      = bench,
            ))

        # Sort by absolute contribution (largest impact first)
        contributions.sort(key=lambda c: abs(c.contribution), reverse=True)

        # Generate narrative sentences
        narrative = [c.to_sentence() for c in contributions]

        # Force plot data
        force_plot_data = self._build_force_plot_data(contributions, overall)

        # Summary DataFrame (for bar chart)
        summary_data = pd.DataFrame([{
            "Feature":             c.component,
            "SHAP Value":          c.contribution,
            "Absolute SHAP":       c.abs_contribution,
            "Direction":           c.direction,
            "Sub-Score":           c.score,
            "Weight (%)":          int(c.weight * 100),
            f"Value ({c.unit})":   c.raw_value,
            "Benchmark":           c.benchmark,
        } for c in contributions])

        return HealthScoreResult(
            overall_score   = overall,
            baseline_score  = _BASELINE_OVERALL,
            contributions   = contributions,
            narrative       = narrative,
            force_plot_data = force_plot_data,
            summary_data    = summary_data,
        )

    @staticmethod
    def _build_force_plot_data(
        contributions: List[ComponentContribution],
        overall_score: float,
    ) -> Dict[str, Any]:
        """
        Builds data for a horizontal waterfall / force plot.
        Starts at baseline (50), each component pushes left or right.
        """
        positives = sorted(
            [c for c in contributions if c.contribution > 0],
            key=lambda c: c.contribution, reverse=True,
        )
        negatives = sorted(
            [c for c in contributions if c.contribution < 0],
            key=lambda c: c.contribution,
        )

        return {
            "baseline":    _BASELINE_OVERALL,
            "prediction":  overall_score,
            "positives": [{"label": c.component, "value": c.contribution} for c in positives],
            "negatives": [{"label": c.component, "value": c.contribution} for c in negatives],
            "all_sorted":  contributions,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Forecast Explainer
# ══════════════════════════════════════════════════════════════════════════════

_FEATURE_DISPLAY_NAMES = {
    "monthly_income":      "Monthly Income",
    "monthly_expenses":    "Monthly Expenses",
    "monthly_savings":     "Monthly Savings",
    "monthly_investments": "Monthly SIP",
    "monthly_emi":         "Monthly EMI",
}


class ForecastExplainer:
    """
    SHAP TreeExplainer wrapper for the XGBoost savings + net_worth models.

    Strategy:
    1. Try shap.TreeExplainer (exact Shapley values for tree ensembles).
    2. If SHAP is unavailable or fails, fall back to signed feature importance
       approximation using model.feature_importances_ weighted by feature
       deviation from population mean.
    """

    def __init__(self, predictor, twin):
        """
        Args:
            predictor: A fitted FinancialPredictor instance.
            twin:      The user's FinancialDigitalTwin.
        """
        self.predictor = predictor
        self.twin      = twin
        self._features = self._extract_features()

    def _extract_features(self) -> Dict[str, float]:
        """Extracts the 5 XGBoost input features from the twin."""
        return {
            "monthly_income":      float(self.twin.total_income),
            "monthly_expenses":    float(self.twin.basic_expenses),
            "monthly_savings":     float(max(
                self.twin.total_income - self.twin.basic_expenses - self.twin.monthly_emi, 0
            )),
            "monthly_investments": float(self.twin.sip_amount),
            "monthly_emi":         float(self.twin.monthly_emi),
        }

    def explain(self, horizon_months: int = 6) -> ForecastResult:
        """
        Runs SHAP explanation on both XGBoost models.

        Args:
            horizon_months: Forecast horizon (6 or 12).

        Returns:
            ForecastResult with SHAP contributions for savings + net worth.
        """
        from models.predictor import FEATURE_COLS

        X_user = pd.DataFrame([self._features])[FEATURE_COLS]

        predictions = self.predictor.predict(self.twin, horizon_months=horizon_months)
        sav_pred    = predictions["predicted_savings"]
        nw_pred     = predictions["predicted_net_worth"]

        # Try real SHAP values first
        sav_contribs, sav_base = self._compute_shap(
            self.predictor.model_savings, X_user, target="savings"
        )
        nw_contribs, nw_base = self._compute_shap(
            self.predictor.model_net_worth, X_user, target="net_worth"
        )

        # Build narrative
        sav_narrative = [c.to_sentence("savings") for c in sav_contribs]
        nw_narrative  = [c.to_sentence("net worth") for c in nw_contribs]

        # Combined feature importance (average |SHAP| across both models, normalised)
        combined = self._build_combined_importance(sav_contribs, nw_contribs)

        return ForecastResult(
            savings_prediction      = sav_pred,
            savings_base_value      = sav_base,
            savings_contributions   = sav_contribs,
            savings_narrative       = sav_narrative,
            net_worth_prediction    = nw_pred,
            net_worth_base_value    = nw_base,
            net_worth_contributions = nw_contribs,
            net_worth_narrative     = nw_narrative,
            combined_importance     = combined,
            horizon_months          = horizon_months,
        )

    # ── Private helpers ─────────────────────────────────────────────────────

    def _compute_shap(
        self,
        model,
        X_user: pd.DataFrame,
        target: str,
    ) -> Tuple[List[ForecastContribution], float]:
        """
        Tries shap.TreeExplainer; falls back to signed importance approximation.

        Returns:
            (contributions sorted by |shap|, base_value)
        """
        try:
            explainer   = get_tree_explainer(model)
            shap_vals   = explainer.shap_values(X_user)
            base_value  = float(explainer.expected_value)

            feature_names = list(X_user.columns)
            sv = shap_vals[0] if hasattr(shap_vals, "__len__") else shap_vals

            contribs = []
            for fname, sval in zip(feature_names, sv):
                contribs.append(ForecastContribution(
                    feature      = fname,
                    display_name = _FEATURE_DISPLAY_NAMES.get(fname, fname),
                    value        = float(X_user[fname].iloc[0]),
                    shap_value   = round(float(sval), 2),
                    direction    = "positive" if sval >= 0 else "negative",
                ))

        except Exception:
            # Fallback: signed importance approximation
            contribs, base_value = self._fallback_importance(model, X_user)

        contribs.sort(key=lambda c: abs(c.shap_value), reverse=True)
        return contribs, base_value

    def _fallback_importance(
        self,
        model,
        X_user: pd.DataFrame,
    ) -> Tuple[List[ForecastContribution], float]:
        """
        Approximates feature contributions without SHAP.
        Uses XGBoost feature_importances_ (gain) weighted by the feature's
        z-score deviation from its own value — positive if above mean,
        negative if below mean — scaled to prediction magnitude.

        This gives directional, human-readable approximate contributions.
        """
        try:
            importances = model.feature_importances_    # shape (n_features,)
            feature_names = list(X_user.columns)
            user_vals     = X_user.iloc[0].to_dict()

            # Approximate base value: mean prediction from training
            # We use model.predict(X_user) as the anchor and distribute contributions
            pred = float(model.predict(X_user)[0])

            # Rough population centre values (typical urban Indian salaried profile)
            _pop_means = {
                "monthly_income":      75000.0,
                "monthly_expenses":    30000.0,
                "monthly_savings":     20000.0,
                "monthly_investments":  8000.0,
                "monthly_emi":         12000.0,
            }
            # Features where higher is good (income, savings, investments) vs bad (expenses, emi)
            _positive_direction = {
                "monthly_income":      True,
                "monthly_expenses":    False,
                "monthly_savings":     True,
                "monthly_investments": True,
                "monthly_emi":         False,
            }

            total_importance = importances.sum() or 1.0
            base_val = pred * 0.5   # Rough baseline at 50% of prediction

            contribs = []
            for fname, imp in zip(feature_names, importances):
                uval   = user_vals.get(fname, 0.0)
                pmean  = _pop_means.get(fname, uval or 1.0)
                is_pos = _positive_direction.get(fname, True)

                # Fractional share of prediction this feature "explains"
                share  = (imp / total_importance) * (pred - base_val)

                # Direction: if "higher is good" and user > mean, contribution is positive
                dev    = (uval - pmean) / (pmean or 1.0)
                signed = share if (is_pos == (dev >= 0)) else -share

                contribs.append(ForecastContribution(
                    feature      = fname,
                    display_name = _FEATURE_DISPLAY_NAMES.get(fname, fname),
                    value        = uval,
                    shap_value   = round(signed, 2),
                    direction    = "positive" if signed >= 0 else "negative",
                ))

            return contribs, base_val

        except Exception:
            # Last resort: equal contribution split
            feature_names = list(X_user.columns)
            pred          = 0.0
            base_val      = 0.0
            contribs      = [
                ForecastContribution(f, _FEATURE_DISPLAY_NAMES.get(f, f),
                                     float(X_user[f].iloc[0]), 0.0, "positive")
                for f in feature_names
            ]
            return contribs, base_val

    @staticmethod
    def _build_combined_importance(
        sav: List[ForecastContribution],
        nw:  List[ForecastContribution],
    ) -> pd.DataFrame:
        """Averages |SHAP| values across both models for a combined importance view."""
        sav_map = {c.feature: abs(c.shap_value) for c in sav}
        nw_map  = {c.feature: abs(c.shap_value) for c in nw}
        all_feats = list(sav_map.keys())

        rows = []
        for f in all_feats:
            s = sav_map.get(f, 0.0)
            n = nw_map.get(f, 0.0)
            rows.append({
                "Feature":                _FEATURE_DISPLAY_NAMES.get(f, f),
                "Savings |SHAP|":         s,
                "Net Worth |SHAP|":       n,
                "Mean |SHAP|":            round((s + n) / 2, 2),
            })

        df = pd.DataFrame(rows).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
        return df


# ══════════════════════════════════════════════════════════════════════════════
# ExplainableAI — Orchestrator Facade
# ══════════════════════════════════════════════════════════════════════════════

class ExplainableAI:
    """
    Unified entry-point for all XAI operations.

    Usage:
        xai = ExplainableAI()
        hs_result  = xai.explain_health_score(twin)
        fc_result  = xai.explain_forecast(twin, predictor, horizon=6)
        sentences  = xai.generate_plain_english(twin, predictor)
    """

    def __init__(self, model: Any = None, background_data: pd.DataFrame = None):
        """
        Args (legacy, kept for backward compat):
            model:           Ignored; models are passed per-call.
            background_data: Ignored; replaced by per-call approach.
        """
        self.model           = model
        self.background_data = background_data
        self.explainer       = None   # legacy placeholder

    # ── Public API ──────────────────────────────────────────────────────────

    def explain_health_score(self, twin) -> HealthScoreResult:
        """
        Explains the Financial Health Score for the given twin.

        Args:
            twin: FinancialDigitalTwin instance.

        Returns:
            HealthScoreResult with contributions, force_plot_data, narrative.
        """
        return HealthScoreExplainer(twin).explain()

    def explain_forecast(self, twin, predictor, horizon: int = 6) -> Optional[ForecastResult]:
        """
        Explains the XGBoost forecast for the given twin.

        Args:
            twin:      FinancialDigitalTwin instance.
            predictor: A fitted FinancialPredictor instance.
            horizon:   Forecast horizon in months (6 or 12).

        Returns:
            ForecastResult, or None if predictor is not fitted.
        """
        if not (predictor and predictor._is_fitted):
            return None
        return ForecastExplainer(predictor, twin).explain(horizon_months=horizon)

    def generate_plain_english(
        self,
        twin,
        predictor = None,
        horizon: int = 6,
    ) -> Dict[str, List[str]]:
        """
        Generates a structured dict of plain-English explanation sentences
        for both health score and forecast models.

        Returns:
            {
              "health_score": [sentence, ...],  # sorted by abs impact
              "forecast_savings":   [sentence, ...],
              "forecast_net_worth": [sentence, ...],
            }
        """
        hs_result = self.explain_health_score(twin)
        result    = {"health_score": hs_result.narrative}

        if predictor and predictor._is_fitted:
            fc_result = self.explain_forecast(twin, predictor, horizon=horizon)
            if fc_result:
                result["forecast_savings"]   = fc_result.savings_narrative
                result["forecast_net_worth"] = fc_result.net_worth_narrative
            else:
                result["forecast_savings"]   = []
                result["forecast_net_worth"] = []
        else:
            result["forecast_savings"]   = []
            result["forecast_net_worth"] = []

        return result

    # ── Legacy stubs (preserved for backward compatibility) ─────────────────

    def initialize_explainer(self) -> None:
        """Legacy stub — explainers are now initialised per-call."""
        pass

    def get_shap_values(self, user_instance: pd.DataFrame) -> Dict[str, Any]:
        """Legacy stub — use explain_forecast() instead."""
        return {"base_value": 0.0, "shap_values": [], "feature_names": []}

    def generate_explanation_plot_data(self, user_instance: pd.DataFrame) -> Dict[str, float]:
        """Legacy stub — use explain_health_score() or explain_forecast() instead."""
        return {}

"""
Financial Personality Clusterer  (Module 5)
Uses K-Means clustering to segment individuals into 5 behavioral personality types:
  - Saver          : High savings_ratio, low expense / debt
  - Investor       : High investment_ratio, moderate savings
  - Spender        : High expense_ratio, low savings
  - Debt Heavy     : High debt_to_income_ratio
  - Balanced Planner: All ratios moderate and healthy

Input Features:
  savings_ratio, expense_ratio, investment_ratio, debt_to_income_ratio

Public API:
  fit_on_population(df)           - Train KMeans on population DataFrame
  predict_personality(twin)       - Predict label for a FinancialDigitalTwin instance
  get_feature_vector(twin)        - Extract raw ratio dict from twin
  get_cluster_stats(df)           - Per-cluster mean stats DataFrame
  get_cluster_definitions()       - Static descriptions for each personality label
"""

import os
import json
import threading
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional

# Persisted artifact produced ONLY by the offline training pipeline
# (training/train_offline.py). The live application loads this file at
# runtime — it never re-fits KMeans on live user data.
_MODEL_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "models")
_CLUSTER_PATH = os.path.join(_MODEL_DIR, "personality_cluster_model.json")

# ── In-Memory Clusterer Cache & Lock ───────────────────────────────────────────
_CLUSTERER_CACHE: Dict[str, Dict[str, Any]] = {}
_CLUSTERER_LOCK = threading.Lock()

# ── Constants ──────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "savings_ratio",
    "expense_ratio",
    "investment_ratio",
    "debt_to_income_ratio",
]

# Personality metadata: label, icon, color, description, nudges
PERSONALITY_META: Dict[str, Dict[str, Any]] = {
    "Saver": {
        "icon": '<i class="fa-solid fa-piggy-bank"></i>',
        "icon_class": "fa-piggy-bank",
        "color": "#10B981",
        "tagline": "The Cautious Accumulator",
        "description": (
            "You prioritize building cash reserves above all else. "
            "Your savings rate is well above average, but you may be "
            "under-investing your surplus — money sitting idle loses "
            "real value to inflation over time."
        ),
        "nudges": [
            "Consider redirecting 10–15% of surplus into equity SIPs.",
            "Ladder your FDs to reduce liquidity drag.",
            "Review if your emergency fund exceeds 9 months — excess can be invested.",
        ],
    },
    "Investor": {
        "icon": '<i class="fa-solid fa-chart-line"></i>',
        "icon_class": "fa-chart-line",
        "color": "#6366F1",
        "tagline": "The Wealth Builder",
        "description": (
            "You actively channel income into markets and mutual funds. "
            "Your investment ratio is strong. Make sure insurance and "
            "emergency reserves are not being sacrificed for aggressive investing."
        ),
        "nudges": [
            "Ensure your emergency fund covers at least 6 months before SIP step-ups.",
            "Diversify across equity, debt, and gold to reduce concentration risk.",
            "Review tax efficiency — ELSS funds can save Section 80C taxes.",
        ],
    },
    "Spender": {
        "icon": '<i class="fa-solid fa-bag-shopping"></i>',
        "icon_class": "fa-bag-shopping",
        "color": "#F59E0B",
        "tagline": "The Lifestyle Maximizer",
        "description": (
            "A large share of your income flows into lifestyle expenses. "
            "Your savings and investment ratios are below par. "
            "Small, consistent cuts in discretionary spending compound "
            "dramatically over time."
        ),
        "nudges": [
            "Apply the 50/30/20 rule — 50% needs, 30% wants, 20% savings.",
            "Automate a SIP on salary day before lifestyle spending begins.",
            "Track food delivery & shopping — these are the easiest cuts.",
        ],
    },
    "Debt Heavy": {
        "icon": '<i class="fa-solid fa-credit-card"></i>',
        "icon_class": "fa-credit-card",
        "color": "#EF4444",
        "tagline": "The Leveraged Striver",
        "description": (
            "EMI obligations consume a significant portion of your monthly income, "
            "limiting your ability to save or invest. Reducing debt burden is "
            "your highest-priority financial action."
        ),
        "nudges": [
            "Use the Avalanche method — prepay highest-interest debt first.",
            "Avoid new EMIs until current debt-to-income drops below 30%.",
            "Consider balance transfers to lower-interest lenders.",
        ],
    },
    "Balanced Planner": {
        "icon": '<i class="fa-solid fa-scale-balanced"></i>',
        "icon_class": "fa-scale-balanced",
        "color": "#3B82F6",
        "tagline": "The Disciplined Optimizer",
        "description": (
            "Your financial ratios are well-calibrated. You save meaningfully, "
            "invest regularly, and keep debt under control. "
            "Focus on optimizing — tax efficiency, goal alignment, and SIP step-ups."
        ),
        "nudges": [
            "Step up your SIP by 10% each year to match salary increments.",
            "Review insurance covers — life cover should be 10x annual income.",
            "Start goal-based investing if not already doing so.",
        ],
    },
}


class FinancialPersonalityClusterer:
    """
    K-Means clustering manager for financial personality segmentation.
    n_clusters is fixed at 5 — one per personality archetype.
    """

    N_CLUSTERS = 5

    def __init__(self):
        self.kmeans: Optional[KMeans] = None
        self.scaler: StandardScaler = StandardScaler()
        self._label_map: Dict[int, str] = {}   # cluster_index -> personality label
        self._is_fitted: bool = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit_on_population(self, df: pd.DataFrame) -> None:
        """
        Trains KMeans on the population feature DataFrame.
        DataFrame must contain columns: savings_ratio, expense_ratio,
        investment_ratio, debt_to_income_ratio.

        Centroid-to-label mapping is derived automatically by inspecting
        which feature has the highest value in each centroid (dominant signal).

        Args:
            df: Population DataFrame from DBManager.get_population_features()
        """
        if df.empty or not all(c in df.columns for c in FEATURE_COLS):
            return

        X = df[FEATURE_COLS].fillna(0.0).values
        X_scaled = self.scaler.fit_transform(X)

        self.kmeans = KMeans(
            n_clusters=self.N_CLUSTERS,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        self.kmeans.fit(X_scaled)

        # Map each cluster index to a personality label via centroid inspection
        self._label_map = self._assign_labels_from_centroids()
        self._is_fitted = True

    # ── Persistence (training-time write / runtime read) ────────────────────

    def save(self, path: str = _CLUSTER_PATH) -> None:
        """
        Persists the fitted scaler + KMeans centroids + label map as JSON.
        Called ONLY by the offline training pipeline, never by the live app.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted clusterer. Call fit_on_population() first.")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "feature_cols": FEATURE_COLS,
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "centroids": self.kmeans.cluster_centers_.tolist(),
            "label_map": {str(k): v for k, v in self._label_map.items()},
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        with _CLUSTERER_LOCK:
            _CLUSTERER_CACHE[path] = payload

    @classmethod
    def load(cls, path: str = _CLUSTER_PATH) -> "FinancialPersonalityClusterer":
        """
        Loads a pretrained clusterer artifact for use in the live application.
        Reconstructs the scaler and a KMeans predictor from saved centroids
        (uses in-memory cached payload if available, else reads from disk, thread-safe).
        """
        instance = cls()
        global _CLUSTERER_CACHE
        if path in _CLUSTERER_CACHE:
            payload = _CLUSTERER_CACHE[path]
        else:
            with _CLUSTERER_LOCK:
                if path in _CLUSTERER_CACHE:
                    payload = _CLUSTERER_CACHE[path]
                else:
                    if not os.path.exists(path):
                        return instance  # stays unfitted -> callers fall back to rule-based logic
                    try:
                        with open(path, "r") as f:
                            payload = json.load(f)
                        _CLUSTERER_CACHE[path] = payload
                    except Exception:
                        return instance

        try:
            instance.scaler = StandardScaler()
            instance.scaler.mean_  = np.array(payload["scaler_mean"])
            instance.scaler.scale_ = np.array(payload["scaler_scale"])
            instance.scaler.var_   = instance.scaler.scale_ ** 2
            instance.scaler.n_features_in_ = len(payload["scaler_mean"])

            centroids = np.array(payload["centroids"])
            instance.kmeans = KMeans(n_clusters=len(centroids), random_state=42, n_init=1)
            instance.kmeans.cluster_centers_ = centroids
            instance.kmeans._n_threads = 1
            instance.kmeans.n_features_in_ = centroids.shape[1]

            instance._label_map = {int(k): v for k, v in payload["label_map"].items()}
            instance._is_fitted = True
        except Exception:
            instance._is_fitted = False
        return instance

    @classmethod
    def clear_cache(cls) -> None:
        """Clears the in-memory clusterer artifact cache."""
        global _CLUSTERER_CACHE
        with _CLUSTERER_LOCK:
            _CLUSTERER_CACHE.clear()

    def is_fitted(self) -> bool:
        return self._is_fitted

    def predict_personality(self, twin) -> str:
        """
        Predicts the personality label for a FinancialDigitalTwin instance.

        Args:
            twin: A FinancialDigitalTwin object.

        Returns:
            One of: 'Saver', 'Investor', 'Spender', 'Debt Heavy', 'Balanced Planner'
            Falls back to rule-based if model is not fitted.
        """
        features = self.get_feature_vector(twin)

        if not self._is_fitted:
            return self._rule_based_fallback(features)

        X = np.array([[features[c] for c in FEATURE_COLS]])
        X_scaled = self.scaler.transform(X)
        cluster_idx = int(self.kmeans.predict(X_scaled)[0])
        return self._label_map.get(cluster_idx, "Balanced Planner")

    def get_feature_vector(self, twin) -> Dict[str, float]:
        """
        Extracts and clips the 4 ratio features from a FinancialDigitalTwin.

        Args:
            twin: A FinancialDigitalTwin object.

        Returns:
            Dict with keys: savings_ratio, expense_ratio, investment_ratio,
            debt_to_income_ratio — all in range [0, 1].
        """
        inc = twin.total_income if twin.total_income > 0 else 1.0
        savings  = max(inc - twin.basic_expenses - twin.monthly_emi, 0.0)
        features = {
            "savings_ratio":        np.clip(savings / inc,             0.0, 1.0),
            "expense_ratio":        np.clip(twin.basic_expenses / inc, 0.0, 1.0),
            "investment_ratio":     np.clip(twin.sip_amount / inc,     0.0, 1.0),
            "debt_to_income_ratio": np.clip(twin.monthly_emi / inc,    0.0, 1.0),
        }
        return {k: float(v) for k, v in features.items()}

    def get_cluster_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns per-cluster mean statistics as a formatted DataFrame.
        Requires the model to be fitted.

        Args:
            df: Population DataFrame (same one used for fit_on_population).

        Returns:
            DataFrame with columns: Personality, Count, Savings Ratio,
            Expense Ratio, Investment Ratio, Debt Ratio.
        """
        if not self._is_fitted or df.empty:
            return pd.DataFrame()

        X = df[FEATURE_COLS].fillna(0.0).values
        X_scaled = self.scaler.transform(X)
        df = df.copy()
        df["cluster"] = self.kmeans.predict(X_scaled)
        df["personality"] = df["cluster"].map(self._label_map)

        stats = (
            df.groupby("personality")[FEATURE_COLS]
            .agg(["mean", "count"])
            .round(3)
        )

        # Flatten for display
        rows = []
        for label, grp in df.groupby("personality"):
            rows.append({
                "Personality":      label,
                "Count":            len(grp),
                "Savings Ratio":    f"{grp['savings_ratio'].mean():.2%}",
                "Expense Ratio":    f"{grp['expense_ratio'].mean():.2%}",
                "Investment Ratio": f"{grp['investment_ratio'].mean():.2%}",
                "Debt Ratio":       f"{grp['debt_to_income_ratio'].mean():.2%}",
            })
        return pd.DataFrame(rows).sort_values("Count", ascending=False).reset_index(drop=True)

    @staticmethod
    def load_saved_cluster_benchmarks(path: str = None) -> pd.DataFrame:
        """
        Loads the aggregate-only per-cluster statistics produced once by
        training/train_offline.py (data/models/cluster_benchmarks.json).
        Contains only per-cluster means/counts — never individual training
        rows — so it's safe for the live app to display.
        """
        bench_path = path or os.path.join(_MODEL_DIR, "cluster_benchmarks.json")
        if not os.path.exists(bench_path):
            return pd.DataFrame()
        try:
            with open(bench_path, "r") as f:
                payload = json.load(f)
            rows = payload.get("clusters", [])
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(rows).sort_values("Count", ascending=False).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def get_population_with_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the population DataFrame with cluster index and personality label columns added.
        Used for scatter plot visualization.
        """
        if not self._is_fitted or df.empty:
            return df
        X = df[FEATURE_COLS].fillna(0.0).values
        X_scaled = self.scaler.transform(X)
        out = df.copy()
        out["cluster"]     = self.kmeans.predict(X_scaled)
        out["personality"] = out["cluster"].map(self._label_map)
        return out

    @staticmethod
    def get_cluster_definitions() -> Dict[str, Dict[str, Any]]:
        """
        Returns the full metadata dict for all 5 personality archetypes.
        """
        return PERSONALITY_META

    # ── Private Helpers ────────────────────────────────────────────────────────

    def _assign_labels_from_centroids(self) -> Dict[int, str]:
        """
        Maps each KMeans cluster index to a personality label by inspecting
        centroid values in the *original* (unscaled) feature space.

        Logic:
          1. Transform centroids back to original scale.
          2. For each centroid, find the dominant ratio signal.
          3. Assign label; fall back to 'Balanced Planner' for residual cluster.
        """
        # Inverse-transform centroids to original feature space
        centroids_orig = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        centroids_df = pd.DataFrame(centroids_orig, columns=FEATURE_COLS)
        centroids_df = centroids_df.clip(0.0, 1.0)

        label_map: Dict[int, str] = {}
        used_labels: set = set()

        # Candidate rules: (label, feature, threshold)
        # Each cluster is assigned the first unclamed rule it satisfies
        RULES = [
            ("Debt Heavy",       "debt_to_income_ratio", 0.30),
            ("Spender",          "expense_ratio",         0.60),
            ("Investor",         "investment_ratio",      0.12),
            ("Saver",            "savings_ratio",         0.35),
        ]
        fallback = "Balanced Planner"

        for idx, row in centroids_df.iterrows():
            assigned = None
            for label, feature, threshold in RULES:
                if label not in used_labels and row[feature] >= threshold:
                    assigned = label
                    used_labels.add(label)
                    break
            if assigned is None:
                if fallback not in used_labels:
                    assigned = fallback
                    used_labels.add(fallback)
                else:
                    # Pick whichever candidate label is still free
                    for label, _, _ in RULES:
                        if label not in used_labels:
                            assigned = label
                            used_labels.add(label)
                            break
                    if assigned is None:
                        assigned = fallback  # last resort
            label_map[idx] = assigned

        return label_map

    def _rule_based_fallback(self, features: Dict[str, float]) -> str:
        """
        Pure rule-based personality assignment — used when KMeans is not fitted.
        """
        if features["debt_to_income_ratio"] >= 0.35:
            return "Debt Heavy"
        if features["expense_ratio"] >= 0.60:
            return "Spender"
        if features["investment_ratio"] >= 0.12:
            return "Investor"
        if features["savings_ratio"] >= 0.30:
            return "Saver"
        return "Balanced Planner"


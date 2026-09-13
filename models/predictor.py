"""
Financial Forecasting Predictor  (Module 6)
Uses XGBoost Regressors to predict future savings and net worth
at 6-month and 12-month horizons.

Training Data Strategy:
  The database contains single-snapshot balance sheets per user — not historical
  time-series. To generate sufficient training data, each user snapshot is
  projected forward 24 synthetic monthly steps using realistic growth/decay rates:
    - Income:      +0.5% / month  (salary drift)
    - Expenses:    +0.4% / month  (inflation drift)
    - Investments: +1.0% / month  (equity compounding)
    - Debt:        -0.3% / month  (amortisation)

  Two separate XGBoost models are trained:
    model_savings   → predicts monthly_savings at horizon H
    model_net_worth → predicts net_worth at horizon H

Input Features (5 as specified):
  monthly_income, monthly_expenses, monthly_savings, monthly_investments, monthly_emi

Targets:
  savings_at_horizon, net_worth_at_horizon

Evaluation:
  80/20 train-test split, 5-fold cross-validation, MAE, RMSE, R²

Model Persistence:
  Saved as XGBoost native JSON to data/models/{savings,net_worth}_model.json
"""

import os
import json
import threading
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ──────────────────────────────────────────────────────────────────────
_MODEL_DIR          = os.path.join(os.path.dirname(__file__), "..", "data", "models")
_SAVINGS_MODEL_PATH = os.path.join(_MODEL_DIR, "savings_model.json")
_NW_MODEL_PATH      = os.path.join(_MODEL_DIR, "net_worth_model.json")
_METRICS_PATH       = os.path.join(_MODEL_DIR, "model_metrics.json")

FEATURE_COLS = [
    "monthly_income",
    "monthly_expenses",
    "monthly_savings",
    "monthly_investments",
    "monthly_emi",
]

# ── Synthetic Generation Parameters ───────────────────────────────────────────
_INCOME_GROWTH      = 0.005   # +0.5% / month
_EXPENSE_GROWTH     = 0.004   # +0.4% / month (inflation)
_INVEST_GROWTH      = 0.010   # +1.0% / month (equity compounding)
_DEBT_DECAY         = 0.003   # -0.3% / month (amortisation)
_SYNTHETIC_STEPS    = 24      # months per user
_NOISE_STD          = 0.02    # 2% Gaussian noise on each feature

# ── In-Memory Model Cache & Lock ───────────────────────────────────────────────
_MODEL_CACHE: Dict[str, Tuple[XGBRegressor, XGBRegressor, Dict[str, Any]]] = {}
_MODEL_LOCK = threading.Lock()


def _build_training_data(df_population: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Generates a synthetic time-series training dataset from population snapshots.

    For each user, propagates their snapshot forward `_SYNTHETIC_STEPS` months
    using compound growth rates + small Gaussian noise.  The target variables
    (savings and net_worth) are computed `horizon` months ahead of each step.

    Args:
        df_population : DataFrame from DBManager.get_population_features()
                        (must also contain raw columns: total_income, total_expenses,
                         sip_amount, monthly_emi, loan_amount, bank_savings,
                         fd_amount, mutual_funds, stocks, ppf_investment, nps_investment).
        horizon       : Prediction horizon in months (6 or 12).

    Returns:
        (X, y_savings, y_net_worth) ready for XGBoost training.
    """
    rows = []
    rng  = np.random.default_rng(seed=42)

    required = {"total_income", "total_expenses", "sip_amount", "monthly_emi", "loan_amount"}
    if df_population.empty or not required.issubset(df_population.columns):
        return pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float)

    for _, user in df_population.iterrows():
        inc  = float(user.get("total_income",    0))
        exp  = float(user.get("total_expenses",  0))
        sip  = float(user.get("sip_amount",      0))
        emi  = float(user.get("monthly_emi",     0))
        debt = float(user.get("loan_amount",     0))

        # Rough net worth proxy from available fields
        assets = (
            float(user.get("bank_savings",    0)) +
            float(user.get("fd_amount",       0)) +
            float(user.get("emergency_fund",  0)) +
            float(user.get("mutual_funds",    0)) +
            float(user.get("stocks",          0)) +
            float(user.get("ppf_investment",  0)) +
            float(user.get("nps_investment",  0))
        )
        nw = assets - debt

        for t in range(_SYNTHETIC_STEPS):
            noise = rng.normal(1.0, _NOISE_STD, size=5)

            cur_inc  = inc  * (1 + _INCOME_GROWTH)  ** t * noise[0]
            cur_exp  = exp  * (1 + _EXPENSE_GROWTH)  ** t * noise[1]
            cur_sip  = sip  * (1 + _INVEST_GROWTH)   ** t * noise[2]
            cur_emi  = emi  * (1 - _DEBT_DECAY)       ** t * noise[3]
            cur_debt = debt * (1 - _DEBT_DECAY)       ** t

            cur_sav = max(cur_inc - cur_exp - cur_emi, 0.0)

            # Target: state at t + horizon
            fut_inc  = cur_inc  * (1 + _INCOME_GROWTH)  ** horizon
            fut_exp  = cur_exp  * (1 + _EXPENSE_GROWTH)  ** horizon
            fut_sip  = cur_sip  * (1 + _INVEST_GROWTH)   ** horizon
            fut_emi  = cur_emi  * (1 - _DEBT_DECAY)       ** horizon
            fut_debt = cur_debt * (1 - _DEBT_DECAY)       ** horizon

            fut_sav   = max(fut_inc - fut_exp - fut_emi, 0.0)
            # Net worth grows by accumulated savings + investment returns
            invest_corpus = cur_sip * (((1 + _INVEST_GROWTH) ** horizon - 1) / _INVEST_GROWTH)
            fut_nw = nw + cur_sav * horizon + invest_corpus - (debt - fut_debt)

            rows.append({
                "monthly_income":      cur_inc,
                "monthly_expenses":    cur_exp,
                "monthly_savings":     cur_sav,
                "monthly_investments": cur_sip,
                "monthly_emi":         cur_emi,
                "target_savings":      fut_sav,
                "target_net_worth":    fut_nw,
            })

    df_train = pd.DataFrame(rows).dropna()
    X        = df_train[FEATURE_COLS]
    y_sav    = df_train["target_savings"]
    y_nw     = df_train["target_net_worth"]
    return X, y_sav, y_nw


def _evaluate(model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Computes MAE, RMSE, R² on the held-out test set."""
    y_pred = model.predict(X_test)
    mae    = float(mean_absolute_error(y_test, y_pred))
    rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2     = float(r2_score(y_test, y_pred))
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "R2": round(r2, 4)}


class FinancialPredictor:
    """
    XGBoost-based financial forecasting engine.

    Trains two models (savings, net_worth) on synthetic population data,
    evaluates with train/test split + 5-fold CV, and persists models to disk.
    """

    _XGBOOST_PARAMS = {
        "n_estimators":  300,
        "max_depth":     5,
        "learning_rate": 0.05,
        "subsample":     0.8,
        "colsample_bytree": 0.8,
        "random_state":  42,
        "verbosity":     0,
    }

    def __init__(self):
        self.model_savings:   Optional[XGBRegressor] = None
        self.model_net_worth: Optional[XGBRegressor] = None
        self._metrics: Dict[str, Dict[str, float]]   = {}
        self._horizon: int = 6     # default, overridden on train
        self._is_fitted: bool = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def train_and_save(self, df_population: pd.DataFrame, horizon: int = 6) -> None:
        """
        Builds synthetic training data, trains both XGBoost models with 80/20
        split and 5-fold CV, evaluates metrics, and saves models to disk.

        Args:
            df_population : Population DataFrame from DBManager.get_population_features().
            horizon       : Prediction horizon in months (6 or 12).
        """
        self._horizon = horizon
        X, y_sav, y_nw = _build_training_data(df_population, horizon)

        if X.empty:
            return

        # ── Train / Test Split ─────────────────────────────────────────────
        X_tr, X_te, ys_tr, ys_te, ynw_tr, ynw_te = train_test_split(
            X, y_sav, y_nw, test_size=0.2, random_state=42
        )

        # ── Savings Model ──────────────────────────────────────────────────
        self.model_savings = XGBRegressor(**self._XGBOOST_PARAMS)
        self.model_savings.fit(X_tr, ys_tr)

        # ── Net Worth Model ────────────────────────────────────────────────
        self.model_net_worth = XGBRegressor(**self._XGBOOST_PARAMS)
        self.model_net_worth.fit(X_tr, ynw_tr)

        # ── 5-Fold Cross-Validation (on training set) ──────────────────────
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_sav = cross_val_score(
            XGBRegressor(**self._XGBOOST_PARAMS), X_tr, ys_tr,
            cv=kf, scoring="r2"
        )
        cv_nw = cross_val_score(
            XGBRegressor(**self._XGBOOST_PARAMS), X_tr, ynw_tr,
            cv=kf, scoring="r2"
        )

        # ── Test Set Evaluation ────────────────────────────────────────────
        sav_metrics = _evaluate(self.model_savings,   X_te, ys_te)
        nw_metrics  = _evaluate(self.model_net_worth, X_te, ynw_te)

        sav_metrics["CV_R2_mean"] = round(float(cv_sav.mean()), 4)
        sav_metrics["CV_R2_std"]  = round(float(cv_sav.std()),  4)
        nw_metrics["CV_R2_mean"]  = round(float(cv_nw.mean()),  4)
        nw_metrics["CV_R2_std"]   = round(float(cv_nw.std()),   4)
        sav_metrics["train_size"] = len(X_tr)
        sav_metrics["test_size"]  = len(X_te)
        nw_metrics["horizon_months"] = horizon

        self._metrics = {"savings": sav_metrics, "net_worth": nw_metrics}
        self._is_fitted = True

        # ── Persist to disk ────────────────────────────────────────────────
        self.save_model()

    def load_or_train(self, df_population: pd.DataFrame, horizon: int = 6) -> None:
        """
        Loads pre-trained models from disk if they exist and match the horizon;
        otherwise calls train_and_save(). Intended for OFFLINE/ADMIN retraining
        tooling only (e.g. training/train_offline.py) — the live Streamlit app
        should call load_only() instead, which never trains on live data.

        Args:
            df_population : Population DataFrame (offline/training use only).
            horizon       : Prediction horizon in months.
        """
        self._horizon = horizon
        if (
            os.path.exists(_SAVINGS_MODEL_PATH) and
            os.path.exists(_NW_MODEL_PATH) and
            os.path.exists(_METRICS_PATH)
        ):
            try:
                self.load_model()
                # Verify horizon consistency
                saved_horizon = self._metrics.get("net_worth", {}).get("horizon_months", -1)
                if saved_horizon == horizon:
                    return
            except Exception:
                pass
        self.train_and_save(df_population, horizon)

    def load_only(self, horizon: int = 6) -> bool:
        """
        Loads pre-trained models from disk for use in the LIVE application.
        Never trains on live data — if no saved model matches the requested
        horizon, leaves the predictor unfitted and predict() automatically
        falls back to a simple linear projection from the user's own numbers.

        Returns:
            True if a matching pretrained model was loaded, False otherwise.
        """
        self._horizon = horizon
        if not (
            os.path.exists(_SAVINGS_MODEL_PATH) and
            os.path.exists(_NW_MODEL_PATH) and
            os.path.exists(_METRICS_PATH)
        ):
            return False
        try:
            self.load_model()
            saved_horizon = self._metrics.get("net_worth", {}).get("horizon_months", -1)
            if saved_horizon == horizon:
                return True
            self._is_fitted = False
            return False
        except Exception:
            self._is_fitted = False
            return False

    def predict(self, twin, horizon_months: int = 6) -> Dict[str, float]:
        """
        Predicts future savings and net worth for a FinancialDigitalTwin.

        Args:
            twin          : FinancialDigitalTwin instance.
            horizon_months: 6 or 12.

        Returns:
            Dict with keys: predicted_savings, predicted_net_worth, horizon_months.
        """
        if not self._is_fitted:
            # Fallback: simple linear projection
            sav = max(twin.total_income - twin.basic_expenses - twin.monthly_emi, 0) * horizon_months
            nw  = twin.net_worth + sav
            return {"predicted_savings": sav, "predicted_net_worth": nw, "horizon_months": horizon_months}

        features = self._extract_features(twin)
        X = pd.DataFrame([features])[FEATURE_COLS]
        pred_sav = float(self.model_savings.predict(X)[0])
        pred_nw  = float(self.model_net_worth.predict(X)[0])
        return {
            "predicted_savings":   max(round(pred_sav,  2), 0.0),
            "predicted_net_worth": round(pred_nw, 2),
            "horizon_months":      horizon_months
        }

    def generate_trajectory(self, twin, months: int = 12) -> pd.DataFrame:
        """
        Generates a month-by-month DataFrame of projected savings and net worth.
        Uses simple compound growth from current state — does NOT re-query XGBoost
        each month (too slow); XGBoost endpoint is used for the 6M / 12M anchors.

        Args:
            twin  : FinancialDigitalTwin instance.
            months: Forecast horizon (default 12).

        Returns:
            DataFrame with columns: Month, Projected_Savings, Projected_Net_Worth.
        """
        monthly_surplus = max(twin.total_income - twin.basic_expenses - twin.monthly_emi, 0)
        invest_growth   = _INVEST_GROWTH

        rows = []
        nw = twin.net_worth
        for m in range(1, months + 1):
            sav = monthly_surplus * (1 + _INCOME_GROWTH) ** m
            invest_val = twin.sip_amount * (((1 + invest_growth) ** m - 1) / invest_growth)
            nw_proj = twin.net_worth + monthly_surplus * m + invest_val
            rows.append({
                "Month":               m,
                "Projected_Savings":   round(sav, 2),
                "Projected_Net_Worth": round(nw_proj, 2),
            })
        return pd.DataFrame(rows)

    def get_model_metrics(self) -> Dict[str, Dict[str, float]]:
        """Returns evaluation metrics for both models."""
        return self._metrics

    def save_model(self) -> None:
        """Saves both XGBoost models and metrics to data/models/ directory and updates in-memory cache."""
        os.makedirs(_MODEL_DIR, exist_ok=True)
        if self.model_savings:
            self.model_savings.save_model(_SAVINGS_MODEL_PATH)
        if self.model_net_worth:
            self.model_net_worth.save_model(_NW_MODEL_PATH)
        with open(_METRICS_PATH, "w") as f:
            json.dump(self._metrics, f, indent=2)
        with _MODEL_LOCK:
            if self.model_savings and self.model_net_worth:
                _MODEL_CACHE["default"] = (self.model_savings, self.model_net_worth, self._metrics)

    def load_model(self) -> None:
        """Loads pre-trained models and metrics from memory cache or disk (thread-safe)."""
        global _MODEL_CACHE
        if "default" in _MODEL_CACHE:
            self.model_savings, self.model_net_worth, self._metrics = _MODEL_CACHE["default"]
            self._is_fitted = True
            return

        with _MODEL_LOCK:
            if "default" in _MODEL_CACHE:
                self.model_savings, self.model_net_worth, self._metrics = _MODEL_CACHE["default"]
                self._is_fitted = True
                return

            self.model_savings = XGBRegressor()
            self.model_savings.load_model(_SAVINGS_MODEL_PATH)
            self.model_net_worth = XGBRegressor()
            self.model_net_worth.load_model(_NW_MODEL_PATH)
            with open(_METRICS_PATH) as f:
                self._metrics = json.load(f)
            _MODEL_CACHE["default"] = (self.model_savings, self.model_net_worth, self._metrics)
            self._is_fitted = True

    @classmethod
    def clear_cache(cls) -> None:
        """Clears the in-memory XGBoost model artifact cache and associated explainers."""
        global _MODEL_CACHE
        with _MODEL_LOCK:
            _MODEL_CACHE.clear()
        try:
            from models.explainability import clear_tree_explainer_cache
            clear_tree_explainer_cache()
        except Exception:
            pass

    # ── Private Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_features(twin) -> Dict[str, float]:
        """Extracts the 5 input features from a FinancialDigitalTwin with finite validation."""
        from utils.validators import sanitize_numeric
        tot_inc = sanitize_numeric(twin.total_income, min_val=0.0)
        basic_exp = sanitize_numeric(twin.basic_expenses, min_val=0.0)
        emi = sanitize_numeric(twin.monthly_emi, min_val=0.0)
        sip = sanitize_numeric(twin.sip_amount, min_val=0.0)
        sav = max(tot_inc - basic_exp - emi, 0.0)

        return {
            "monthly_income":      float(tot_inc),
            "monthly_expenses":    float(basic_exp),
            "monthly_savings":     float(sav),
            "monthly_investments": float(sip),
            "monthly_emi":         float(emi),
        }


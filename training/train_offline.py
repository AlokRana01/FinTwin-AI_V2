"""
Phase 1 — Offline Model Training Pipeline.

This is the ONLY script in the project that is allowed to open
`data/raw/indian_salaried_financial_data.csv`. It:

  1. Loads the raw synthetic training dataset.
  2. Derives the same ratio features the live app uses
     (savings_ratio, expense_ratio, investment_ratio, debt_to_income_ratio).
  3. Fits the K-Means personality clusterer and saves it to
     data/models/personality_cluster_model.json.
  4. Trains the XGBoost forecasting models (reusing FinancialPredictor,
     unchanged) and saves them to data/models/*.json.
  5. Computes AGGREGATE, ANONYMOUS cohort benchmark statistics
     (mean spend/savings by occupation, population percentiles) and
     saves them to data/models/cohort_benchmarks.json and
     data/models/population_benchmarks.json.

Nothing produced by this script contains individual training records —
only trained model parameters (centroids/scaler/XGBoost trees) and
aggregated statistics. The live Streamlit app never imports pandas
against the raw CSV; it only loads these artifacts.

Run this once (and again whenever the dataset changes):
    python training/train_offline.py
"""

import os
import sys
import json

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import BASE_DIR
from models.clustering import FinancialPersonalityClusterer

try:
    from models.predictor import FinancialPredictor
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

RAW_CSV_PATH = BASE_DIR / "data" / "raw" / "indian_salaried_financial_data.csv"
MODEL_DIR    = BASE_DIR / "data" / "models"

MIN_COHORT_SIZE = 5  # occupations with fewer rows than this are dropped from benchmarks


def load_raw_dataset() -> pd.DataFrame:
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {RAW_CSV_PATH}. "
            "Run data/generator.py first, or place the dataset there."
        )
    return pd.read_csv(RAW_CSV_PATH)


def build_population_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduces the exact feature engineering used by
    DBManager.get_population_features(), but computed directly from the
    training CSV instead of the live database.
    """
    df = pd.DataFrame()
    df["user_id"]           = df_raw["User_ID"]
    df["occupation"]        = df_raw["Occupation"]
    df["total_income"]      = df_raw["Monthly_Income"].fillna(0) + df_raw["Additional_Income"].fillna(0)
    df["total_expenses"]    = (
        df_raw["Rent"].fillna(0) + df_raw["Groceries"].fillna(0) + df_raw["Utilities"].fillna(0) +
        df_raw["Transport"].fillna(0) + df_raw["Food_Delivery"].fillna(0) +
        df_raw["Entertainment"].fillna(0) + df_raw["Shopping"].fillna(0)
    )
    df["sip_amount"]      = df_raw["SIP_Amount"].fillna(0)
    df["monthly_emi"]     = df_raw["Monthly_EMI"].fillna(0)
    df["loan_amount"]     = df_raw["Loan_Amount"].fillna(0)
    df["bank_savings"]    = df_raw["Bank_Savings"].fillna(0)
    df["fd_amount"]       = df_raw["FD_Amount"].fillna(0)
    df["emergency_fund"]  = df_raw["Emergency_Fund"].fillna(0)
    df["mutual_funds"]    = df_raw["Mutual_Fund_Value"].fillna(0)
    df["stocks"]          = df_raw["Stock_Value"].fillna(0)
    df["ppf_investment"]  = df_raw["PPF_Investment"].fillna(0)
    df["nps_investment"]  = df_raw["NPS_Investment"].fillna(0)

    # Also keep spend-category columns (snake_case) for cohort benchmark averages
    for col, snake in [
        ("Rent", "rent"), ("Groceries", "groceries"), ("Utilities", "utilities"),
        ("Transport", "transport"), ("Food_Delivery", "food_delivery"),
        ("Entertainment", "entertainment"), ("Shopping", "shopping"),
    ]:
        df[snake] = df_raw[col].fillna(0)

    df = df[df["total_income"] > 0].copy()

    df["savings_ratio"]        = (df["total_income"] - df["total_expenses"] - df["monthly_emi"]) / df["total_income"]
    df["expense_ratio"]        = df["total_expenses"] / df["total_income"]
    df["investment_ratio"]     = df["sip_amount"] / df["total_income"]
    df["debt_to_income_ratio"] = df["monthly_emi"] / df["total_income"]

    for col in ["savings_ratio", "expense_ratio", "investment_ratio", "debt_to_income_ratio"]:
        df[col] = df[col].clip(0.0, 1.0)

    return df


def train_clustering_model(df_pop: pd.DataFrame) -> FinancialPersonalityClusterer:
    print("Training K-Means personality clusterer on the offline dataset...")
    clusterer = FinancialPersonalityClusterer()
    clusterer.fit_on_population(df_pop)
    clusterer.save()
    print(f"  Saved -> {MODEL_DIR / 'personality_cluster_model.json'}")

    # Save aggregate (non-identifying) cluster statistics for the
    # "Cluster Visualization" tab, so the live app can display cohort
    # shape without ever loading individual training rows.
    stats_df = clusterer.get_cluster_stats(df_pop)
    stats_records = stats_df.to_dict(orient="records") if not stats_df.empty else []
    with open(MODEL_DIR / "cluster_benchmarks.json", "w") as f:
        json.dump({"clusters": stats_records}, f, indent=2)
    print(f"  Saved -> {MODEL_DIR / 'cluster_benchmarks.json'}")
    return clusterer


def train_forecast_models(df_pop: pd.DataFrame) -> None:
    print("Training XGBoost forecasting models on the offline dataset...")
    predictor_6mo = FinancialPredictor()
    predictor_6mo.train_and_save(df_pop, horizon=6)
    print("  6-month horizon models saved.")
    # Note: model_metrics.json / *_model.json are overwritten with the
    # latest horizon trained; app pages call load_or_train() with horizon=6
    # by default, matching this training run.


def compute_cohort_benchmarks(df_pop: pd.DataFrame) -> None:
    """
    Aggregate-only occupation benchmarks (never individual rows), used as a
    cold-start fallback by DBManager.get_cohort_averages() until enough real
    registered users of a given occupation exist in the live database.
    """
    print("Computing anonymous per-occupation cohort benchmarks...")
    spend_cols = ["rent", "groceries", "utilities", "transport",
                  "food_delivery", "entertainment", "shopping",
                  "sip_amount", "monthly_emi", "emergency_fund"]

    grouped = df_pop.groupby("occupation")[spend_cols].mean()
    counts  = df_pop.groupby("occupation").size()

    benchmarks = {}
    for occupation, row in grouped.iterrows():
        if counts[occupation] < MIN_COHORT_SIZE:
            continue
        benchmarks[occupation] = {col: round(float(row[col]), 2) for col in spend_cols}

    # A single overall fallback for occupations not present in the dataset
    overall = df_pop[spend_cols].mean()
    benchmarks["_overall"] = {col: round(float(overall[col]), 2) for col in spend_cols}

    with open(MODEL_DIR / "cohort_benchmarks.json", "w") as f:
        json.dump(benchmarks, f, indent=2)
    print(f"  Saved -> {MODEL_DIR / 'cohort_benchmarks.json'} ({len(benchmarks) - 1} occupations)")


def compute_population_benchmarks(df_pop: pd.DataFrame) -> None:
    """
    Aggregate percentile statistics for the 4 ratio features, used to show
    "you are healthier than X% of similar earners"-style copy without ever
    exposing another individual's record.
    """
    print("Computing population percentile benchmarks...")
    percentiles = [10, 25, 50, 75, 90]
    out = {}
    for col in ["savings_ratio", "expense_ratio", "investment_ratio", "debt_to_income_ratio"]:
        out[col] = {f"p{p}": round(float(np.percentile(df_pop[col], p)), 4) for p in percentiles}
    out["sample_size"] = int(len(df_pop))

    with open(MODEL_DIR / "population_benchmarks.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved -> {MODEL_DIR / 'population_benchmarks.json'}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("=== Phase 1: Offline Model Training (raw dataset used ONLY here) ===")
    df_raw = load_raw_dataset()
    print(f"Loaded {len(df_raw)} training records from {RAW_CSV_PATH}")

    df_pop = build_population_features(df_raw)
    print(f"Derived population feature table: {df_pop.shape}")

    train_clustering_model(df_pop)
    if _XGBOOST_AVAILABLE:
        train_forecast_models(df_pop)
    else:
        print("Skipping XGBoost forecast training: xgboost is not installed "
              "in this environment (it is listed in requirements.txt — run "
              "`pip install -r requirements.txt` and re-run this script).")
    compute_cohort_benchmarks(df_pop)
    compute_population_benchmarks(df_pop)

    print("\nDone. The live application will now read only the artifacts in")
    print(f"{MODEL_DIR}, and will never query the raw training dataset again.")


if __name__ == "__main__":
    main()

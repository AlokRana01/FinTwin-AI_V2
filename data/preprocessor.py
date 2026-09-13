"""
Data Preprocessing Pipeline
Handles feature engineering, scaling, clustering preparation, and forecasting training pipeline sets.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, Dict, Any

class FinancialFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to calculate key financial ratios and metrics.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Avoid modifying the original dataframe
        X = X.copy()
        
        # Calculate Total Income (handling missing values as 0 during computation)
        total_income = X["Monthly_Income"].fillna(0) + X["Additional_Income"].fillna(0)
        
        # Calculate Total Basic Expenses
        basic_expense_cols = ["Rent", "Groceries", "Utilities", "Transport", "Food_Delivery", "Entertainment", "Shopping"]
        basic_expenses = X[basic_expense_cols].fillna(0).sum(axis=1)
        
        # 1. Savings_Ratio = (Total_Income - Basic_Expenses - Monthly_EMI) / Total_Income
        emi = X["Monthly_EMI"].fillna(0)
        surplus = total_income - basic_expenses - emi
        X["Savings_Ratio"] = np.where(total_income > 0, surplus / total_income, 0.0)
        
        # 2. Expense_Ratio = Basic_Expenses / Total_Income
        X["Expense_Ratio"] = np.where(total_income > 0, basic_expenses / total_income, 0.0)
        
        # 3. Debt_To_Income_Ratio = Monthly_EMI / Total_Income
        X["Debt_To_Income_Ratio"] = np.where(total_income > 0, emi / total_income, 0.0)
        
        # 4. Investment_Ratio = SIP_Amount / Total_Income
        sip = X["SIP_Amount"].fillna(0)
        X["Investment_Ratio"] = np.where(total_income > 0, sip / total_income, 0.0)
        
        # 5. Emergency_Fund_Coverage = Emergency_Fund / (Basic_Expenses + Monthly_EMI)
        total_outflow = basic_expenses + emi
        emergency_fund = X["Emergency_Fund"].fillna(0)
        X["Emergency_Fund_Coverage"] = np.where(total_outflow > 0, emergency_fund / total_outflow, 0.0)
        
        # 6. Net_Worth = Total_Assets - Total_Liabilities
        assets = (
            X["Bank_Savings"].fillna(0) +
            X["FD_Amount"].fillna(0) +
            X["Emergency_Fund"].fillna(0) +
            X["Mutual_Fund_Value"].fillna(0) +
            X["Stock_Value"].fillna(0) +
            X["PPF_Investment"].fillna(0) +
            X["NPS_Investment"].fillna(0)
        )
        liabilities = X["Loan_Amount"].fillna(0)
        X["Net_Worth"] = assets - liabilities
        
        # 7. Insurance_Coverage_Score
        # Life term cover >= 10x annual salary (120x monthly income)
        monthly_income = X["Monthly_Income"].fillna(0)
        life_ins = X["Life_Insurance"].fillna(0)
        life_adequacy = np.where(monthly_income > 0, life_ins / (monthly_income * 120.0), np.where(life_ins > 0, 1.0, 0.0))
        life_adequacy = np.clip(life_adequacy, 0.0, 1.0)
        
        # Health cover >= 5L (500,000)
        health_ins = X["Health_Insurance"].fillna(0)
        health_adequacy = np.clip(health_ins / 500000.0, 0.0, 1.0)
        
        X["Insurance_Coverage_Score"] = 0.5 * life_adequacy + 0.5 * health_adequacy
        
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        new_features = [
            "Savings_Ratio",
            "Expense_Ratio",
            "Debt_To_Income_Ratio",
            "Investment_Ratio",
            "Emergency_Fund_Coverage",
            "Net_Worth",
            "Insurance_Coverage_Score"
        ]
        return np.array(list(input_features) + new_features, dtype=object)


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Custom transformer to clip outliers using IQR or percentile thresholds.
    """
    def __init__(self, method='percentile', lower_percentile=1.0, upper_percentile=99.0, factor=1.5):
        self.method = method
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.factor = factor
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.lower_bounds_ = {}
            self.upper_bounds_ = {}
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    if self.method == 'iqr':
                        q25 = X[col].quantile(0.25)
                        q75 = X[col].quantile(0.75)
                        iqr = q75 - q25
                        self.lower_bounds_[col] = q25 - self.factor * iqr
                        self.upper_bounds_[col] = q75 + self.factor * iqr
                    else:
                        self.lower_bounds_[col] = X[col].quantile(self.lower_percentile / 100.0)
                        self.upper_bounds_[col] = X[col].quantile(self.upper_percentile / 100.0)
        else:
            X_arr = np.asarray(X)
            self.lower_bounds_ = []
            self.upper_bounds_ = []
            for col_idx in range(X_arr.shape[1]):
                col_data = X_arr[:, col_idx]
                if self.method == 'iqr':
                    q25 = np.percentile(col_data, 25)
                    q75 = np.percentile(col_data, 75)
                    iqr = q75 - q25
                    self.lower_bounds_.append(q25 - self.factor * iqr)
                    self.upper_bounds_.append(q75 + self.factor * iqr)
                else:
                    self.lower_bounds_.append(np.percentile(col_data, self.lower_percentile))
                    self.upper_bounds_.append(np.percentile(col_data, self.upper_percentile))
        return self

    def transform(self, X):
        X = X.copy()
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                if self.lower_bounds_ is not None and col in self.lower_bounds_:
                    X[col] = np.clip(X[col], self.lower_bounds_[col], self.upper_bounds_[col])
        else:
            for col_idx in range(X.shape[1]):
                if self.lower_bounds_ is not None and col_idx < len(self.lower_bounds_):
                    X[:, col_idx] = np.clip(X[:, col_idx], self.lower_bounds_[col_idx], self.upper_bounds_[col_idx])
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features


class DataPreprocessor:
    """
    Cleans and transforms raw database queries into structured arrays or dataframes for modeling.
    """

    @staticmethod
    def aggregate_monthly_spending(df_transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates transactional lines into monthly category-wise matrices.
        """
        # Placeholder for transaction aggregator
        return pd.DataFrame()

    @staticmethod
    def engineer_health_features(df_users: pd.DataFrame, df_twins: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculates key financial ratios (debt-to-income, expense-to-income, emergency-fund ratio)
        used for score training and prediction.
        """
        df = df_users
        if df_twins is not None and not df_twins.empty:
            df = pd.merge(df_users, df_twins, on="User_ID", how="left")
            
        engineer = FinancialFeatureEngineer()
        return engineer.transform(df)

    @staticmethod
    def build_preprocessing_pipeline() -> Pipeline:
        """
        Creates and returns a Scikit-Learn Pipeline that implements the complete 
        preprocessing, feature engineering, outlier clipping, and scaling flow.
        """
        original_numeric_cols = [
            "Age", "Monthly_Income", "Bonus", "Additional_Income", "Rent", 
            "Groceries", "Utilities", "Transport", "Food_Delivery", "Entertainment", 
            "Shopping", "Bank_Savings", "FD_Amount", "Emergency_Fund", "SIP_Amount", 
            "Mutual_Fund_Value", "Stock_Value", "PPF_Investment", "NPS_Investment", 
            "Loan_Amount", "Car_Loan", "Home_Loan", "Credit_Card_Debt", "Monthly_EMI", 
            "Health_Insurance", "Life_Insurance", "Goal_Amount"
        ]
        
        engineered_numeric_cols = [
            "Savings_Ratio", "Expense_Ratio", "Debt_To_Income_Ratio", "Investment_Ratio",
            "Emergency_Fund_Coverage", "Net_Worth", "Insurance_Coverage_Score"
        ]
        
        numeric_cols = original_numeric_cols + engineered_numeric_cols
        categorical_cols = ["City", "Occupation", "Goal_Type"]
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('clipper', OutlierClipper(method='percentile', lower_percentile=1.0, upper_percentile=99.0)),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='drop'
        )
        
        pipeline = Pipeline(steps=[
            ('engineer', FinancialFeatureEngineer()),
            ('preprocessor', preprocessor)
        ])
        
        # Configure output format to be Pandas
        pipeline.set_output(transform="pandas")
        
        return pipeline

    @staticmethod
    def prepare_clustering_features(df_features: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Standardizes financial metrics for clustering algorithms.
        """
        pipeline = DataPreprocessor.build_preprocessing_pipeline()
        df_scaled = pipeline.fit_transform(df_features)
        
        # Extract the scaler from the pipeline
        scaler = pipeline.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler']
        
        return df_scaled, scaler

    @staticmethod
    def prepare_forecasting_data(df_monthly: pd.DataFrame, lag_months: int = 3) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Structures time-series monthly expenses into lag features for autoregressive predictions.
        """
        if df_monthly.empty or "Total_Expenses" not in df_monthly.columns:
            return pd.DataFrame(), pd.Series(dtype=float)
            
        df = df_monthly.copy()
        features = {}
        for lag in range(1, lag_months + 1):
            features[f"lag_{lag}"] = df["Total_Expenses"].shift(lag)
            
        df_features = pd.DataFrame(features).dropna()
        target = df["Total_Expenses"].iloc[lag_months:]
        
        return df_features, target

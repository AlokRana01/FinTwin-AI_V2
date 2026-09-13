import pandas as pd
import re
from datetime import datetime
from typing import Tuple

def prepare_csv_export(df: pd.DataFrame, filename_prefix: str) -> Tuple[bytes, str]:
    """
    Cleans, formats, and prepares a dataframe for a professional, Excel-compatible CSV export.
    
    Args:
        df: The pandas DataFrame to export.
        filename_prefix: Base name for the export file.
        
    Returns:
        Tuple containing the CSV bytes (utf-8-sig encoded) and the final filename string.
    """
    if df.empty:
        return "".encode("utf-8-sig"), f"{filename_prefix}_{datetime.now().strftime('%Y-%m-%d')}.csv"
        
    out_df = df.copy()
    
    # 1. Remove JSON/dicts/lists columns
    cols_to_drop = []
    for col in out_df.columns:
        if out_df[col].apply(lambda x: isinstance(x, (dict, list))).any():
            cols_to_drop.append(col)
    out_df.drop(columns=cols_to_drop, inplace=True)
    
    # 2. Remove completely empty columns
    out_df.dropna(axis=1, how="all", inplace=True)
    
    # 3. Remove duplicate columns
    out_df = out_df.loc[:, ~out_df.columns.duplicated()]
    
    # 4. Remove completely duplicate rows
    out_df.drop_duplicates(inplace=True)
    
    # 5. Clean string data (trim whitespace & escape formula injection characters)
    for col in out_df.select_dtypes(include=["object", "string"]).columns:
        out_df[col] = out_df[col].apply(
            lambda x: f"'{str(x).strip()}" if pd.notnull(x) and str(x).strip().startswith(('=', '+', '-', '@'))
            else (str(x).strip() if pd.notnull(x) else x)
        )
        
    # 6. Human-readable column names & currency indicators
    new_cols = {}
    money_keywords = [
        "income", "surplus", "price", "cash", "amount", "worth", "savings", 
        "expense", "cost", "emi", "fund", "tax", "value", "p10", "p25", "p50", 
        "p75", "p90", "burn", "deficit", "balance", "corpus", "sip", "target",
        "return", "base case", "optimistic", "pessimistic"
    ]
    
    for col in out_df.columns:
        # Convert snake_case or unformatted strings to Title Case
        name = str(col).replace("_", " ").title()
        
        # Determine if it represents currency and isn't already labeled
        is_money = any(kw in str(col).lower() for kw in money_keywords)
        # Exception for month/year or pure percentages/rates
        if any(kw in str(col).lower() for kw in ["month", "year", "duration", "rate", "pct", "percent"]):
            is_money = False
            
        if is_money and "(₹)" not in name and "₹" not in name:
            # Check if column is numeric to avoid labeling string columns accidentally
            if pd.api.types.is_numeric_dtype(out_df[col]):
                name = f"{name} (₹)"
                
        new_cols[col] = name
        
    out_df.rename(columns=new_cols, inplace=True)
    
    # 7. Generate Excel-compatible bytes and descriptive filename
    csv_bytes = out_df.to_csv(index=False).encode("utf-8-sig")
    filename = f"{filename_prefix}_{datetime.now().strftime('%Y-%m-%d')}.csv"
    
    return csv_bytes, filename

"""Preprocessing and transformation pipeline.
Drops direct identifiers, handles categorical encoding, and normalizes numeric values.
"""
from typing import List, Tuple
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_DIRECT_IDENTIFIERS = ["customerid", "id", "ssn", "name", "email", "phone"]

def identify_columns(df: pd.DataFrame, target_col: str = "Churn", id_cols: List[str] = None) -> Tuple[List[str], List[str], List[str]]:
    """Separates columns into identifier, categorical feature, and numeric feature groups."""
    if id_cols is None:
        id_cols = [c for c in df.columns if c.lower() in DEFAULT_DIRECT_IDENTIFIERS]
    
    feature_cols = [c for c in df.columns if c not in id_cols and c != target_col]
    cat_cols = df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()
    return id_cols, cat_cols, num_cols

def clean_and_split_features(df: pd.DataFrame, target_col: str = "Churn", drop_identifiers: bool = True):
    """Prepares feature dataframe without direct identifiers."""
    id_cols, cat_cols, num_cols = identify_columns(df, target_col=target_col)
    
    processed = df.copy()
    if drop_identifiers and id_cols:
        processed = processed.drop(columns=id_cols)
        
    return processed, id_cols, cat_cols, num_cols

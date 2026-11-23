"""
Data Preparation, Cleaning, and Quality Gates Pipeline
End-to-End Enterprise ML Architecture - Customer Churn & Retention Engine
Author: Abdul Wasiu Adebayo (BAN6440, Nexford University)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_and_audit_raw_data(data_path: str):
    """
    Loads raw customer churn telemetry and performs initial schema audit.
    Identifies data types, sentinel representations, and baseline nulls.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw data file not found at: {data_path}")
    
    df_raw = pd.read_csv(data_path)
    audit = {
        "raw_record_count": len(df_raw),
        "raw_column_count": len(df_raw.columns),
        "initial_nulls": df_raw.isnull().sum().to_dict(),
        "whitespace_totalcharges_count": (df_raw['TotalCharges'] == ' ').sum() if 'TotalCharges' in df_raw.columns else 0
    }
    return df_raw, audit

def clean_and_validate_data(df_raw: pd.DataFrame):
    """
    Applies data cleansing policies:
    1. Replaces whitespace sentinel values in TotalCharges with numeric 0.0 for zero-tenure accounts.
    2. Drops non-predictive entity primary key (customerID) to prevent identity leakage.
    3. Normalizes binary and multi-class categorical features.
    4. Enforces strict schema typing and records data quality gates.
    """
    df = df_raw.copy()
    
    # Pre-cleansing quality check
    pre_clean_stats = {
        "total_records": len(df),
        "missing_total_charges": int((df['TotalCharges'].astype(str).str.strip() == '').sum()),
        "duplicate_records": int(df.duplicated(subset=['customerID']).sum())
    }
    
    # 1. Sentinel replacement & type casting
    df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', '0.0', regex=True).astype(float)
    
    # Verification: If tenure is 0, TotalCharges must be 0.0
    zero_tenure_mask = (df['tenure'] == 0)
    df.loc[zero_tenure_mask, 'TotalCharges'] = 0.0
    
    # 2. Binary target mapping: 'Yes' -> 1, 'No' -> 0
    df['Churn'] = (df['Churn'].str.strip() == 'Yes').astype(int)
    
    # 3. Drop entity identifier to prevent target leakage or identity bias
    customer_ids = df['customerID'].copy()
    df_clean = df.drop(columns=['customerID'])
    
    # Post-cleansing quality check
    post_clean_stats = {
        "clean_records": len(df_clean),
        "null_values_remaining": int(df_clean.isnull().sum().sum()),
        "churn_rate": float(df_clean['Churn'].mean()),
        "feature_count": df_clean.shape[1] - 1
    }
    
    return df_clean, customer_ids, pre_clean_stats, post_clean_stats

def split_and_transform_features(df_clean: pd.DataFrame, random_state: int = 42):
    """
    Executes leak-free data partitioning and feature transformation:
    - Partition: 70% Train, 15% Validation, 15% Test (Stratified on Churn)
    - Fit StandardScaler and OneHotEncoder STRICTLY on Train split
    - Transform Validation and Test splits using Train parameters
    """
    X = df_clean.drop(columns=['Churn'])
    y = df_clean['Churn'].values
    
    # Split Train (70%) and Temp (30%)
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    
    # Split Temp into Validation (15% overall) and Test (15% overall)
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )
    
    # Feature taxonomy
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = [c for c in X.columns if c not in num_cols]
    
    # Preprocessor pipeline definition
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )
    
    # FIT strictly on training data
    preprocessor.fit(X_train_raw)
    
    # Transform partitions
    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    # Extract feature names
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
    feature_names = num_cols + cat_feature_names
    
    split_meta = {
        "train_samples": X_train.shape[0],
        "val_samples": X_val.shape[0],
        "test_samples": X_test.shape[0],
        "total_transformed_features": len(feature_names),
        "train_churn_rate": float(np.mean(y_train)),
        "val_churn_rate": float(np.mean(y_val)),
        "test_churn_rate": float(np.mean(y_test))
    }
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), (X_train_raw, X_val_raw, X_test_raw), feature_names, preprocessor, split_meta

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, "../data/telco_customer_churn.csv")
    df_raw, audit = load_and_audit_raw_data(data_file)
    print("Raw Data Audit:", audit)
    
    df_clean, c_ids, pre_stats, post_stats = clean_and_validate_data(df_raw)
    print("Pre-Clean Quality:", pre_stats)
    print("Post-Clean Quality:", post_stats)
    
    train_data, val_data, test_data, raw_splits, feature_names, preproc, split_meta = split_and_transform_features(df_clean)
    print("Split Metadata:", split_meta)
    print(f"Engineered Feature Vector Dimension: {len(feature_names)}")

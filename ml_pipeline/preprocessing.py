import os
import pandas as pd
import numpy as np
import joblib

def preprocess_inference(df, pipeline_path="models/xgb_pipeline.pkl"):
    """
    Applies the exact same feature engineering as train_xgboost.py
    and ensures the dataframe matches the exact expected feature schema 
    for the trained pipeline.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load pipeline to get expected features
    pipeline = joblib.load(os.path.join(BASE_DIR, "..", pipeline_path))
    preprocessor = pipeline.named_steps['preprocessor']
    num_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][2]
    expected_features = num_features + cat_features
    
    # Work on a copy
    df = df.copy()
    
    # 2. Feature Engineering Logic (Replicated from training)
    # Clean term
    if 'term' in df.columns:
        df["term_months"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)
        if 'int_rate' in df.columns:
            df["risk_term_pressure"]  = df["int_rate"] * (df["term_months"] == 60).astype(int)
        else:
            df["risk_term_pressure"] = 0
    else:
        df["risk_term_pressure"] = 0
        df["term_months"] = 36

    # Safe access for numeric features
    for col in ['loan_amnt', 'annual_inc', 'int_rate', 'fico_range_high', 'fico_range_low', 'dti', 'revol_util', 'open_acc', 'acc_open_past_24mths', 'tot_hi_cred_lim', 'bc_util', 'percent_bc_gt_75', 'num_tl_op_past_12m', 'inq_last_6mths']:
        if col not in df.columns:
            df[col] = 0.0
            
    df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1e-6)
    df['debt_pressure'] = (df['loan_amnt'] * df['int_rate']) / (df['annual_inc'] + 1e-6)
    df['fico_gap'] = df['fico_range_high'] - df['fico_range_low']
    df['log_loan_amnt'] = np.log1p(df['loan_amnt'])
    df['log_annual_income'] = np.log1p(df['annual_inc'])
    df['dti_squared'] = df['dti'] ** 2
    
    df["activity_density"]    = df["acc_open_past_24mths"] / (df["open_acc"] + 1)
    df["overextension"]       = df["revol_util"] * df["acc_open_past_24mths"]
    df["credit_depth"]        = df["tot_hi_cred_lim"] / (df["open_acc"] + 1)
    df["utilization_pressure"]= df["bc_util"] * df["percent_bc_gt_75"]
    df["credit_concentration"]= df["revol_util"] / (df["bc_util"] + 1e-6)
    
    df["recent_activity_ratio"]      = df["num_tl_op_past_12m"] / (df["open_acc"] + 1)
    df["inquiry_pressure"]           = df["inq_last_6mths"] / (df["open_acc"] + 1)
    df["account_activity_intensity"] = df["acc_open_past_24mths"] / (df["num_tl_op_past_12m"] + 1)

    df["stress_core"]     = df["dti"] * df["int_rate"] * df["revol_util"]
    df["stress_extended"] = (df["loan_amnt"] / (df["annual_inc"] + 1)) * df["int_rate"] * (df["revol_util"] / 100)
    df["risk_burden_index"] = df["dti"] + (df["int_rate"] / 100) + (df["revol_util"] / 100)

    # FICO risk band
    df["fico_risk_band"] = pd.cut(
        df["fico_range_high"],
        bins=[0, 649, 699, 749, 900],
        labels=[0, 1, 2, 3],
        right=True,
    ).astype(float)
    if df['fico_risk_band'].isna().any():
        df['fico_risk_band'] = df['fico_risk_band'].fillna(1.0) # Default mid-tier

    # 3. Macroeconomic Data Merge
    macro_file = os.path.join(BASE_DIR, "..", "data", "processed", "macro_data.csv")
    if os.path.exists(macro_file) and 'issue_d' in df.columns:
        macro_df = pd.read_csv(macro_file)
        df = df.merge(macro_df, on='issue_d', how='left')
    else:
        df['macro_unemployment'] = 0
        df['macro_fed_funds'] = 0
        df['macro_cpi'] = 0
        df['macro_cpi_yoy'] = 0

    # 4. Handle any infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 5. Conform to exact pipeline features
    for col in expected_features:
        if col not in df.columns:
            # If missing, assign sensible defaults
            if col in cat_features:
                df[col] = "missing"
            else:
                df[col] = 0.0
                
    return df[expected_features]

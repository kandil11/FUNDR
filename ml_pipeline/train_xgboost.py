import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from fetch_data import ensure_data_exists
import joblib


def train_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(BASE_DIR, "..", "data", "processed", "enriched_dataset.csv")
    results_dir = os.path.join(BASE_DIR, "..", "results", "summary")
    models_dir = os.path.join(BASE_DIR, "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    ensure_data_exists()

    print("Loading enriched dataset (with behavioral columns)...")
    df = pd.read_csv(input_file, low_memory=False)
    
    print("Computing advanced V2 & V3 engineered features...")
    # Clean term
    if 'term' in df.columns:
        df["term_months"] = df["term"].str.extract(r"(\d+)").astype(float)
        df["risk_term_pressure"]  = df["int_rate"] * (df["term_months"] == 60).astype(int)
    else:
        df["risk_term_pressure"] = 0

    df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1e-6)
    df['debt_pressure'] = (df['loan_amnt'] * df['int_rate']) / (df['annual_inc'] + 1e-6)
    df['fico_gap'] = df['fico_range_high'] - df['fico_range_low']
    df['log_loan_amnt'] = np.log1p(df['loan_amnt'])
    df['log_annual_income'] = np.log1p(df['annual_inc'])
    df['dti_squared'] = df['dti'] ** 2
    
    if 'acc_open_past_24mths' in df.columns and 'open_acc' in df.columns:
        df["activity_density"]    = df["acc_open_past_24mths"] / (df["open_acc"] + 1)
        df["overextension"]       = df["revol_util"] * df["acc_open_past_24mths"]
        
    if 'tot_hi_cred_lim' in df.columns and 'open_acc' in df.columns:
        df["credit_depth"]          = df["tot_hi_cred_lim"] / (df["open_acc"] + 1)
    if 'bc_util' in df.columns and 'percent_bc_gt_75' in df.columns:
        df["utilization_pressure"]  = df["bc_util"] * df["percent_bc_gt_75"]
        df["credit_concentration"]  = df["revol_util"] / (df["bc_util"] + 1e-6)
    
    if 'num_tl_op_past_12m' in df.columns and 'open_acc' in df.columns:
        df["recent_activity_ratio"]      = df["num_tl_op_past_12m"] / (df["open_acc"] + 1)
    if 'inq_last_6mths' in df.columns and 'open_acc' in df.columns:
        df["inquiry_pressure"]           = df["inq_last_6mths"] / (df["open_acc"] + 1)
    if 'acc_open_past_24mths' in df.columns and 'num_tl_op_past_12m' in df.columns:
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

    # Handle any infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Merge Macroeconomic Data
    macro_file = os.path.join(BASE_DIR, "..", "data", "processed", "macro_data.csv")
    if os.path.exists(macro_file):
        macro_df = pd.read_csv(macro_file)
        df = df.merge(macro_df, on='issue_d', how='left')
    else:
        print("Macro data not found. Skipping merge.")
        df['macro_unemployment'] = 0
        df['macro_fed_funds'] = 0
        df['macro_cpi'] = 0
        df['macro_cpi_yoy'] = 0

    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off', 'Default'])]

    target_map = {'Fully Paid': 0, 'Charged Off': 1, 'Default': 1}
    df['is_default'] = df['loan_status'].map(target_map)

    all_cols = df.columns.tolist()
    
    # Categoricals
    cat_features = ['grade', 'home_ownership', 'purpose', 'addr_state'] + [c for c in all_cols if '_Label' in c]
    if 'term' in all_cols: cat_features.append('term')
    if 'verification_status' in all_cols: cat_features.append('verification_status')
    
    # Exclude targets, non-numeric, dates
    exclude_cols = ['loan_status', 'is_default', 'issue_d', 'earliest_cr_line', 'term_months', 'fico_risk_band'] + cat_features
    num_features = [c for c in all_cols if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]

    X = df[num_features + cat_features]
    y = df['is_default']

    print(f"Features: {len(num_features)} numerical, {len(cat_features)} categorical.")

    print("Splitting data...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
    )
    
    # Calculate scale_pos_weight for imbalanced classes
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    print("Building pipelines...")
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])

    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        ))
    ])

    print("Executing Deep Hyperparameter Tuning (n_iter=20)...")
    param_grid = {
        'classifier__n_estimators': [100, 200, 300, 400],
        'classifier__max_depth': [3, 5, 7, 9],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__subsample': [0.7, 0.8, 1.0],
        'classifier__colsample_bytree': [0.7, 0.8, 1.0],
        'classifier__min_child_weight': [1, 3, 5]
    }

    search = RandomizedSearchCV(
        xgb_pipeline,
        param_distributions=param_grid,
        n_iter=20,
        cv=3,
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_
    
    print("Finding best threshold based on F1-score...")
    y_pred_proba = best_pipeline.predict_proba(X_val)[:, 1]
    best_threshold = 0.5
    best_f1 = 0
    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred_temp = (y_pred_proba >= threshold).astype(int)
        f1_temp = f1_score(y_val, y_pred_temp)
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_threshold = threshold
            
    print(f"Best Threshold: {best_threshold:.2f} with F1: {best_f1:.4f}")
    
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "best_threshold.json"), "w") as f:
        json.dump({"threshold": float(best_threshold), "f1": float(best_f1)}, f, indent=2)
    
    print("Saving models...")
    joblib.dump(best_pipeline, os.path.join(models_dir, "xgb_pipeline.pkl"))
    joblib.dump((X_test, y_test), os.path.join(models_dir, "test_data.pkl"))
    
    print("Models saved successfully.")

if __name__ == "__main__":
    train_model()
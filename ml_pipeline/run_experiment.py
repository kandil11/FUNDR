"""
run_experiment.py  –  v2 vs v3 feature-engineering comparison (Updated for Enriched Dataset)
=============================================================
v2 = baseline + engineered features that are computable
v3 = v2 + advanced stress/structural features + all new trend/growth features

Same XGBoost pipeline, same splits, same threshold-optimisation logic.
Outputs:
  results/experiment/metrics_v2.json
  results/experiment/metrics_v3.json
  results/experiment/experiment_comparison_v3.json
  results/experiment/plots/probability_overlap_v2_v3.png
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "..", "data", "processed", "enriched_dataset.csv")
MACRO_PATH  = os.path.join(BASE_DIR, "..", "data", "processed", "macro_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results", "experiment")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
MODELS_DIR  = os.path.join(BASE_DIR, "..", "models")
for d in (RESULTS_DIR, PLOTS_DIR, MODELS_DIR):
    os.makedirs(d, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────
def load_and_prepare():
    print("[1/5] Loading data …")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])].copy()
    df["is_default"] = df["loan_status"].map(
        {"Fully Paid": 0, "Charged Off": 1, "Default": 1}
    )

    # ── engineered features (v2) ──
    if 'term' in df.columns:
        df["term_months"] = df["term"].str.extract(r"(\d+)").astype(float)
        df["risk_term_pressure"]  = df["int_rate"] * (df["term_months"] == 60).astype(int)
    
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
    
    # Handle infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    if os.path.exists(MACRO_PATH):
        macro_df = pd.read_csv(MACRO_PATH)
        df = df.merge(macro_df, on='issue_d', how='left')
    else:
        df['macro_unemployment'] = 0
        df['macro_fed_funds'] = 0
        df['macro_cpi'] = 0
        df['macro_cpi_yoy'] = 0

    # ── Define feature lists ──
    all_cols = df.columns.tolist()
    
    cat_features = ['grade', 'home_ownership', 'purpose', 'addr_state'] + [c for c in all_cols if '_Label' in c]
    if 'term' in all_cols: cat_features.append('term')
    if 'verification_status' in all_cols: cat_features.append('verification_status')
    
    exclude_cols = ['loan_status', 'is_default', 'issue_d', 'earliest_cr_line', 'term_months', 'fico_risk_band'] + cat_features
    
    v3_advanced = [
        'credit_depth', 'utilization_pressure', 'credit_concentration',
        'recent_activity_ratio', 'inquiry_pressure', 'account_activity_intensity',
        'stress_core', 'stress_extended', 'risk_burden_index'
    ]
    trend_cols = [c for c in all_cols if ('_Growth_' in c or '_Consistency_' in c or '_Trend_%' in c)]

    num_v3 = [c for c in all_cols if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    # V2 excludes advanced stress/structural features and trend/growth features
    num_v2 = [c for c in num_v3 if c not in v3_advanced and c not in trend_cols]

    return df, cat_features, num_v2, num_v3


def build_pipeline(num_feats, cat_feats, scale_pos_weight):
    numeric_tx = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_tx = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_tx,     num_feats),
        ("cat", categorical_tx, cat_feats),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric="auc",
        )),
    ])


PARAM_GRID = {
    'classifier__n_estimators': [100, 200, 300, 400],
    'classifier__max_depth': [3, 5, 7, 9],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'classifier__subsample': [0.7, 0.8, 1.0],
    'classifier__colsample_bytree': [0.7, 0.8, 1.0],
    'classifier__min_child_weight': [1, 3, 5]
}


def optimise_threshold(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.90, 0.02):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def evaluate(model, X_test, y_test, threshold):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy":  float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall":    float(recall_score(y_test, preds, zero_division=0)),
        "f1":        float(f1_score(y_test, preds)),
        "auc":       float(roc_auc_score(y_test, probs)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }


def train_version(label, num_feats, cat_feats, X_train, y_train, X_val, y_val, X_test, y_test, scale_pos_weight):
    print(f"  Training {label} ({len(num_feats)} numeric features) …")
    pipe = build_pipeline(num_feats, cat_feats, scale_pos_weight)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=PARAM_GRID,
        n_iter=10,
        cv=3,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    thr, val_f1 = optimise_threshold(model, X_val, y_val)
    metrics = evaluate(model, X_test, y_test, thr)
    metrics["val_f1"] = val_f1
    metrics["version"] = label
    metrics["best_params"] = {k: str(v) for k, v in search.best_params_.items()}

    joblib.dump(model, os.path.join(MODELS_DIR, f"xgb_pipeline_{label}.pkl"))
    with open(os.path.join(RESULTS_DIR, f"metrics_{label}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  {label}  AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}  "
          f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  thr={thr:.2f}")
    return model, metrics


def plot_probability_overlap(model_v2, model_v3, X_test, y_test):
    print("[4/5] Generating overlap distribution plot …")
    p2 = model_v2.predict_proba(X_test)[:, 1]
    p3 = model_v3.predict_proba(X_test)[:, 1]

    plt.figure(figsize=(10, 6))
    sns.kdeplot(p2, label="v2 (Base)", fill=True, alpha=0.3)
    sns.kdeplot(p3, label="v3 (Enriched + Behavioral)", fill=True, alpha=0.3)
    plt.axvline(0.5, color="k", linestyle="--", alpha=0.5)
    plt.title("Distribution of Predicted Probabilities: v2 vs v3")
    plt.xlabel("Predicted Probability of Default")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "probability_overlap_v2_v3.png"))
    plt.close()


def main():
    print("=========================================================")
    print("  FUNDR EXPERIMENT: v2 vs v3 (Enriched Features + Deep Tuning) ")
    print("=========================================================")

    df, cat_features, num_v2, num_v3 = load_and_prepare()

    print("[2/5] Splitting data …")
    X = df[num_v3 + cat_features]
    y = df["is_default"]

    X_tr_full, X_te, y_tr_full, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr_full, y_tr_full, test_size=0.2, stratify=y_tr_full, random_state=42
    )

    pos, neg = sum(y_tr), len(y_tr) - sum(y_tr)
    scale_pos = neg / pos if pos != 0 else 1.0

    print("[3/5] Training models …")
    model_v2, met_v2 = train_version(
        "v2", num_v2, cat_features,
        X_tr, y_tr, X_va, y_va, X_te, y_te, scale_pos
    )
    model_v3, met_v3 = train_version(
        "v3", num_v3, cat_features,
        X_tr, y_tr, X_va, y_va, X_te, y_te, scale_pos
    )

    plot_probability_overlap(model_v2, model_v3, X_te, y_te)

    print("[5/5] Saving final comparison report …")
    comp = {
        "v2": met_v2,
        "v3": met_v3,
        "deltas": {
            "auc": met_v3["auc"] - met_v2["auc"],
            "f1":  met_v3["f1"] - met_v2["f1"]
        }
    }
    with open(os.path.join(RESULTS_DIR, "experiment_comparison_v3.json"), "w") as f:
        json.dump(comp, f, indent=2)

    print("\n--- SUMMARY ---")
    print(f"V2 AUC: {met_v2['auc']:.4f} | F1: {met_v2['f1']:.4f}")
    print(f"V3 AUC: {met_v3['auc']:.4f} | F1: {met_v3['f1']:.4f}")
    print(f"Delta AUC: {comp['deltas']['auc']:+.4f}")
    print(f"Delta F1:  {comp['deltas']['f1']:+.4f}")
    print("\nDone! Check results/experiment/ for full details.")

if __name__ == "__main__":
    main()

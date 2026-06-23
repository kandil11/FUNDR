"""
run_experiment.py  –  v2 vs v3 feature-engineering comparison
=============================================================
v2 = baseline + 11 interaction/affordability/behavioral features (previous iteration)
v3 = v2 + 15 new stress/structure/behavioral-proxy/buffer/risk-flag features

Same XGBoost pipeline, same splits, same threshold-optimisation logic.
Outputs:
  results/experiment/metrics_v2.json
  results/experiment/metrics_v3.json
  results/experiment/experiment_comparison_v3.json
  results/experiment/plots/probability_overlap_v2_v3.png
  results/experiment/plots/shap_v3.png
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
DATA_PATH   = os.path.join(BASE_DIR, "..", "data", "processed", "accepted_loans_final.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results", "experiment")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
MODELS_DIR  = os.path.join(BASE_DIR, "..", "models")
for d in (RESULTS_DIR, PLOTS_DIR, MODELS_DIR):
    os.makedirs(d, exist_ok=True)

# ── columns to load ─────────────────────────────────────────────────────
LOAD_COLS = [
    "loan_status", "fico_range_high", "fico_range_low", "int_rate", "dti",
    "loan_amnt", "annual_inc", "revol_util", "term", "home_ownership",
    "purpose", "verification_status", "inq_last_6mths", "delinq_2yrs",
    "pub_rec", "installment", "acc_open_past_24mths", "mort_acc",
    "tot_cur_bal", "bc_util", "percent_bc_gt_75", "pub_rec_bankruptcies",
    "num_tl_op_past_12m", "tot_hi_cred_lim", "open_acc",
]

# ── feature lists ────────────────────────────────────────────────────────
NUM_BASE = [
    "fico_range_high", "fico_range_low", "int_rate", "dti", "loan_amnt",
    "annual_inc", "revol_util", "inq_last_6mths", "delinq_2yrs", "pub_rec",
    "installment", "acc_open_past_24mths", "mort_acc", "tot_cur_bal",
    "bc_util", "percent_bc_gt_75", "pub_rec_bankruptcies",
    "num_tl_op_past_12m", "tot_hi_cred_lim", "open_acc",
]

V2_ENGINEERED = [
    "dti_x_int_rate", "fico_x_int_rate", "risk_term_pressure",
    "loan_to_income", "debt_pressure", "activity_density", "overextension",
    "fico_gap", "log_loan_amnt", "log_annual_income", "dti_squared",
]

V3_NEW = [
    # stress stack
    "stress_core", "stress_extended", "risk_burden_index",
    # credit structure
    "credit_depth", "utilization_pressure", "credit_concentration",
    # behavioural proxy
    "recent_activity_ratio", "inquiry_pressure", "account_activity_intensity",
    # financial buffer
    "income_buffer", "installment_burden", "debt_service_ratio",
    # nonlinear risk flags
    "high_risk_flag", "ultra_stress_flag", "fico_risk_band",
]

CAT_FEATURES = ["term", "home_ownership", "purpose", "verification_status"]

NUM_V2 = NUM_BASE + V2_ENGINEERED
NUM_V3 = NUM_V2 + V3_NEW


# ── helpers ──────────────────────────────────────────────────────────────
def load_and_prepare():
    """Load CSV, filter target classes, build all engineered columns."""
    print("[1/5] Loading data …")
    df = pd.read_csv(DATA_PATH, usecols=LOAD_COLS, low_memory=False)

    # target
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off", "Default"])].copy()
    df["is_default"] = df["loan_status"].map(
        {"Fully Paid": 0, "Charged Off": 1, "Default": 1}
    )

    # clean term → numeric months (kept as string for OHE, but numeric for flags)
    df["term_months"] = df["term"].str.extract(r"(\d+)").astype(float)

    # ── v2 features ──
    df["dti_x_int_rate"]      = df["dti"] * df["int_rate"]
    df["fico_x_int_rate"]     = df["fico_range_high"] * df["int_rate"]
    df["risk_term_pressure"]  = df["int_rate"] * (df["term_months"] == 60).astype(int)
    df["loan_to_income"]      = df["loan_amnt"] / (df["annual_inc"] + 1)
    df["debt_pressure"]       = (df["loan_amnt"] * df["int_rate"]) / (df["annual_inc"] + 1)
    df["activity_density"]    = df["acc_open_past_24mths"] / (df["open_acc"] + 1)
    df["overextension"]       = df["revol_util"] * df["acc_open_past_24mths"]
    df["fico_gap"]            = df["fico_range_high"] - df["fico_range_low"]
    df["log_loan_amnt"]       = np.log1p(df["loan_amnt"])
    df["log_annual_income"]   = np.log1p(df["annual_inc"])
    df["dti_squared"]         = df["dti"] ** 2

    # ── v3 NEW features ──
    # 1. Stress Stack
    df["stress_core"]     = df["dti"] * df["int_rate"] * df["revol_util"]
    df["stress_extended"] = (df["loan_amnt"] / (df["annual_inc"] + 1)) * df["int_rate"] * (df["revol_util"] / 100)
    df["risk_burden_index"] = df["dti"] + (df["int_rate"] / 100) + (df["revol_util"] / 100)

    # 2. Credit Structure
    df["credit_depth"]          = df["tot_hi_cred_lim"] / (df["open_acc"] + 1)
    df["utilization_pressure"]  = df["bc_util"] * df["percent_bc_gt_75"]
    df["credit_concentration"]  = df["revol_util"] / (df["bc_util"] + 1e-6)

    # 3. Behavioral Proxy
    df["recent_activity_ratio"]      = df["num_tl_op_past_12m"] / (df["open_acc"] + 1)
    df["inquiry_pressure"]           = df["inq_last_6mths"] / (df["open_acc"] + 1)
    df["account_activity_intensity"] = df["acc_open_past_24mths"] / (df["num_tl_op_past_12m"] + 1)

    # 4. Financial Buffer
    df["income_buffer"]       = df["annual_inc"] / (df["loan_amnt"] + 1)
    df["installment_burden"]  = df["installment"] / (df["annual_inc"] + 1)
    df["debt_service_ratio"]  = (df["installment"] * 12) / (df["annual_inc"] + 1)

    # 5. Nonlinear Risk Flags
    df["high_risk_flag"]    = ((df["dti"] > 20) & (df["int_rate"] > 15)).astype(int)
    df["ultra_stress_flag"] = ((df["revol_util"] > 80) & (df["bc_util"] > 80)).astype(int)
    # FICO risk band: very_high >=750, high 700-749, medium 650-699, low <650
    df["fico_risk_band"] = pd.cut(
        df["fico_range_high"],
        bins=[0, 649, 699, 749, 900],
        labels=[0, 1, 2, 3],   # 0=low, 1=medium, 2=high, 3=very_high
        right=True,
    ).astype(float)

    return df


def build_pipeline(num_feats, scale_pos_weight):
    """Return sklearn Pipeline with ColumnTransformer + XGBClassifier."""
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
        ("cat", categorical_tx, CAT_FEATURES),
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
    "classifier__n_estimators":    [100, 200, 300],
    "classifier__max_depth":       [3, 5, 7],
    "classifier__learning_rate":   [0.05, 0.1, 0.2],
    "classifier__subsample":       [0.8, 1.0],
    "classifier__colsample_bytree":[0.8, 1.0],
}


def optimise_threshold(model, X_val, y_val):
    """Sweep thresholds on validation set; return (best_threshold, best_f1)."""
    probs = model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.10, 0.90, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def evaluate(model, X_test, y_test, threshold):
    """Compute all metrics on the test set at the given threshold."""
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


def train_version(label, num_feats, X_train, y_train, X_val, y_val,
                   X_test, y_test, scale_pos_weight):
    """Train, tune, optimise threshold, evaluate, save model + metrics."""
    print(f"  Training {label} ({len(num_feats)} numeric features) …")
    pipe = build_pipeline(num_feats, scale_pos_weight)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=PARAM_GRID,
        n_iter=8,
        cv=3,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    thr, val_f1 = optimise_threshold(model, X_val, y_val)
    metrics = evaluate(model, X_test, y_test, thr)
    metrics["val_f1"] = val_f1
    metrics["version"] = label
    metrics["best_params"] = {k: str(v) for k, v in search.best_params_.items()}

    # persist
    joblib.dump(model, os.path.join(MODELS_DIR, f"xgb_pipeline_{label}.pkl"))
    with open(os.path.join(RESULTS_DIR, f"metrics_{label}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  {label}  AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}  "
          f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  thr={thr:.2f}")
    return model, metrics


def plot_probability_overlap(model_v2, model_v3, X_test, y_test):
    """KDE of predicted probabilities for v2 vs v3."""
    p2 = model_v2.predict_proba(X_test)[:, 1]
    p3 = model_v3.predict_proba(X_test)[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # overall overlap
    ax = axes[0]
    sns.kdeplot(p2, label="v2 (prev. engineered)", fill=True, alpha=0.35, ax=ax)
    sns.kdeplot(p3, label="v3 (+ stress/struct)",  fill=True, alpha=0.35, ax=ax)
    ax.set_title("Overall probability overlap")
    ax.set_xlabel("P(default)")
    ax.legend()

    # per-class
    ax = axes[1]
    mask_pos = y_test == 1
    sns.kdeplot(p3[mask_pos],  label="v3 – defaults",     fill=True, alpha=0.35, ax=ax, color="tomato")
    sns.kdeplot(p3[~mask_pos], label="v3 – non-defaults", fill=True, alpha=0.35, ax=ax, color="steelblue")
    ax.set_title("v3 class separation")
    ax.set_xlabel("P(default)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "probability_overlap_v2_v3.png"), dpi=150)
    plt.close()
    print("  → probability_overlap_v2_v3.png saved")


def plot_shap(model, X_test, label):
    """SHAP summary plot for the given model."""
    preprocessor = model.named_steps["preprocessor"]
    classifier   = model.named_steps["classifier"]
    X_transformed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    explainer   = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_transformed,
        feature_names=feature_names,
        show=False, max_display=25,
    )
    plt.title(f"SHAP summary – {label}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"shap_{label}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → shap_{label}.png saved")


# ── main ─────────────────────────────────────────────────────────────────
def main():
    df = load_and_prepare()
    y  = df["is_default"]

    # use ALL v3 columns in the dataframe (v2 is a subset)
    X = df[NUM_V3 + CAT_FEATURES]

    # split (identical for both versions)
    print("[2/5] Splitting data …")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.20, stratify=y_train_full, random_state=42
    )
    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = neg / pos if pos else 1.0
    print(f"  train={len(y_train)}  val={len(y_val)}  test={len(y_test)}  "
          f"pos_rate={pos/len(y_train):.3f}  scale_pos_weight={spw:.2f}")

    # train v2
    print("[3/5] Training v2 …")
    model_v2, metrics_v2 = train_version(
        "v2", NUM_V2, X_train, y_train, X_val, y_val, X_test, y_test, spw
    )

    # train v3
    print("[4/5] Training v3 …")
    model_v3, metrics_v3 = train_version(
        "v3", NUM_V3, X_train, y_train, X_val, y_val, X_test, y_test, spw
    )

    # comparison JSON
    delta_auc = metrics_v3["auc"] - metrics_v2["auc"]
    delta_f1  = metrics_v3["f1"]  - metrics_v2["f1"]
    conclusion = (
        f"v3 AUC={metrics_v3['auc']:.4f} vs v2 AUC={metrics_v2['auc']:.4f} "
        f"(Δ={delta_auc:+.4f}).  "
        f"v3 F1={metrics_v3['f1']:.4f} vs v2 F1={metrics_v2['f1']:.4f} "
        f"(Δ={delta_f1:+.4f}).  "
    )
    if delta_auc > 0.005:
        conclusion += "Stress/structure features IMPROVED separability."
    elif delta_auc < -0.005:
        conclusion += "New features DEGRADED performance – possible noise."
    else:
        conclusion += "Marginal change; new features did NOT meaningfully improve AUC."

    comparison = {
        "v2": metrics_v2,
        "v3": metrics_v3,
        "delta_auc": delta_auc,
        "delta_f1":  delta_f1,
        "conclusion": conclusion,
    }
    out_path = os.path.join(RESULTS_DIR, "experiment_comparison_v3.json")
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Comparison saved → {out_path}")

    # plots
    print("[5/5] Generating plots …")
    plot_probability_overlap(model_v2, model_v3, X_test, y_test)
    plot_shap(model_v3, X_test, "v3")

    # summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"  v2  AUC={metrics_v2['auc']:.4f}  F1={metrics_v2['f1']:.4f}")
    print(f"  v3  AUC={metrics_v3['auc']:.4f}  F1={metrics_v3['f1']:.4f}")
    print(f"  ΔAUC = {delta_auc:+.4f}    ΔF1 = {delta_f1:+.4f}")
    print(f"\n  {conclusion}")
    print("=" * 60)


if __name__ == "__main__":
    main()

import os
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

def evaluate_model():

    # Load best threshold from JSON (fallback to 0.5)
    results_dir = os.path.join(BASE_DIR, "..", "results", "summary")
    try:
        with open(os.path.join(results_dir, "best_threshold.json"), "r") as f:
            best_info = json.load(f)
            threshold = best_info.get("threshold", 0.5)
    except Exception:
        threshold = 0.5
    model = joblib.load(os.path.join(MODEL_DIR, "xgb_pipeline.pkl"))
    X_test, y_test = joblib.load(os.path.join(MODEL_DIR, "test_data.pkl"))

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\nEvaluation at threshold = {threshold:.4f}")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print("AUC:", auc)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save plots
    plots_dir = os.path.join(BASE_DIR, "..", "results", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(plots_dir, 'roc_curve.png'))
    plt.close()
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.figure()
    plt.plot(recall, precision, label=f'F1 = {f1:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(plots_dir, 'pr_curve.png'))
    plt.close()


if __name__ == "__main__":
    evaluate_model()
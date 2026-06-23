import os
import joblib
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results", "summary")


def find_best_threshold():
    # Load test data (which includes the original test split)
    X_test_full, y_test_full = joblib.load(os.path.join(MODEL_DIR, "test_data.pkl"))
    # Further split this into validation (20%) and final test (80%)
    X_val, X_test_final, y_val, y_test_final = train_test_split(
        X_test_full, y_test_full, test_size=0.8, stratify=y_test_full, random_state=42
    )
    # Load model
    model = joblib.load(os.path.join(MODEL_DIR, "xgb_pipeline.pkl"))
    y_prob = model.predict_proba(X_val)[:, 1]

    best_t = 0.0
    best_f1 = 0.0
    for t in np.arange(0.1, 0.9, 0.02):
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    # Save best threshold as JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    best_info = {"threshold": best_t, "f1": best_f1}
    with open(os.path.join(RESULTS_DIR, "best_threshold.json"), "w") as f:
        json.dump(best_info, f, indent=2)
    print(f"Best threshold: {best_t:.4f}, F1: {best_f1:.4f}")

if __name__ == "__main__":
    find_best_threshold()
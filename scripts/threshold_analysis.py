import os
import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

def find_best_threshold():

    model = joblib.load(os.path.join(MODEL_DIR, "xgb_pipeline.pkl"))
    X_test, y_test = joblib.load(os.path.join(MODEL_DIR, "test_data.pkl"))

    y_prob = model.predict_proba(X_test)[:, 1]

    best_t = 0
    best_f1 = 0

    print("\nThreshold tuning results:\n")

    for t in np.arange(0.1, 0.9, 0.02):
        y_pred = (y_prob >= t).astype(int)

        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        print(f"t={t:.2f} | F1={f1:.4f} | P={prec:.4f} | R={rec:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    print("\nBEST THRESHOLD:")
    print(best_t, "F1:", best_f1)
    joblib.dump(best_t, os.path.join(MODEL_DIR, "best_threshold.pkl"))
    print("Best threshold saved.")


if __name__ == "__main__":
    find_best_threshold()
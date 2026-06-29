import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.pipeline import Pipeline

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
RESULTS_PLOTS_DIR = os.path.join(BASE_DIR, "..", "results", "plots")
os.makedirs(RESULTS_PLOTS_DIR, exist_ok=True)

# Load model and test data
model_path = os.path.join(MODEL_DIR, "xgb_pipeline.pkl")
model: Pipeline = joblib.load(model_path)
X_test, y_test = joblib.load(os.path.join(MODEL_DIR, "test_data.pkl"))

# Extract preprocessor and classifier
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# Get feature names after one‑hot encoding
numeric_features = preprocessor.transformers_[0][2]
categorical_features = preprocessor.transformers_[1][2]
# numeric feature names are unchanged
num_features = list(numeric_features)
# categorical one‑hot feature names
cat_ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = cat_ohe.get_feature_names_out(categorical_features)
feature_names = np.concatenate([num_features, cat_feature_names])

# 1. Feature importance (gain based)
importances = classifier.feature_importances_
# Sort
sorted_idx = np.argsort(importances)[::-1]
top_n = 20
plt.figure(figsize=(10, 6))
plt.barh(range(top_n), importances[sorted_idx[:top_n]][::-1])
plt.yticks(range(top_n), feature_names[sorted_idx[:top_n]][::-1])
plt.xlabel('Gain Importance')
plt.title('Top 20 Feature Importances (Gain)')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PLOTS_DIR, 'feature_importance.png'))
plt.close()

# 2. SHAP values for top features
# Transform test data through preprocessor only
X_test_transformed = preprocessor.transform(X_test)
explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_test_transformed)
# Use mean absolute SHAP value per feature
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_idx = np.argsort(mean_abs_shap)[::-1]
# Plot summary for top 10 features
shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, max_display=10, plot_type='bar', show=False)
plt.title('Top 10 Features by Mean |SHAP|')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PLOTS_DIR, 'shap_top10.png'))
plt.close()

# 3. Predicted probability distribution per class
y_prob = model.predict_proba(X_test)[:, 1]
plt.figure(figsize=(8, 5))
# Plot distributions for positive and negative classes
sns.kdeplot(y_prob[y_test == 0], label='Negative (0)', shade=True)
sns.kdeplot(y_prob[y_test == 1], label='Positive (1)', shade=True)
plt.xlabel('Predicted Probability')
plt.title('Predicted Probability Distribution by True Class')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PLOTS_DIR, 'probability_distribution.png'))
plt.close()

# 4. Quick diagnostics
cumulative_importance = np.cumsum(importances[sorted_idx]) / importances.sum()
# How many features explain 80% of importance?
num_features_80 = np.searchsorted(cumulative_importance, 0.80) + 1
# Identify near‑zero importance features
zero_imp_features = feature_names[importances < 1e-4]

diag = {
    "total_features": len(feature_names),
    "features_explaining_80pct_importance": int(num_features_80),
    "num_zero_importance_features": int(len(zero_imp_features)),
    "zero_importance_features": list(zero_imp_features[:20])  # sample first few
}
# Save diagnostics as JSON
with open(os.path.join(RESULTS_PLOTS_DIR, 'model_diagnostics.json'), 'w') as f:
    json.dump(diag, f, indent=2)

print('Analysis completed. Plots saved to', RESULTS_PLOTS_DIR)

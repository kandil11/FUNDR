import subprocess
import os
import sys

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = BASE_DIR

def run_script(script_name):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    print(f"Running {script_name}...")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

if __name__ == "__main__":
    # 0. Fetch Macroeconomic data
    run_script("fetch_macro_data.py")
    # 1. Train the XGBoost model
    run_script("train_xgboost.py")
    # 2. Find the best threshold on validation set
    run_script("threshold_analysis.py")
    # 3. Evaluate the model on test set with selected threshold
    run_script("evaluate_xgboost.py")
    # 4. Generate Risk Ranking & Segmentation Profile
    run_script("risk_ranking_analysis.py")
    # 5. Profit/Loss Optimization
    run_script("profit_loss_optimization.py")
    # 6. (Optional) Analyze model importance and SHAP (if script exists)
    analyze_path = os.path.join(SCRIPTS_DIR, "analyze_model.py")
    if os.path.exists(analyze_path):
        run_script("analyze_model.py")

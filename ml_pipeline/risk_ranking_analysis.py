import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results", "summary")
PLOTS_DIR = os.path.join(BASE_DIR, "..", "results", "plots")

def run_risk_ranking():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading model and test data...")
    model = joblib.load(os.path.join(MODEL_DIR, "xgb_pipeline.pkl"))
    X_test, y_test = joblib.load(os.path.join(MODEL_DIR, "test_data.pkl"))

    print("Predicting probabilities...")
    y_prob = model.predict_proba(X_test)[:, 1]

    # Create a DataFrame for analysis
    df_eval = pd.DataFrame({
        'prob': y_prob,
        'actual': y_test.values
    })

    # Segment into 10 Deciles
    df_eval['Risk_Decile'] = pd.qcut(df_eval['prob'], 10, labels=np.arange(1, 11))

    # Calculate Default Rate per Decile
    decile_stats = df_eval.groupby('Risk_Decile', observed=False).agg(
        Borrowers=('prob', 'count'),
        Defaults=('actual', 'sum'),
        Min_Prob=('prob', 'min'),
        Max_Prob=('prob', 'max'),
        Mean_Prob=('prob', 'mean')
    )
    decile_stats['Default_Rate'] = decile_stats['Defaults'] / decile_stats['Borrowers']
    decile_stats['Capture_Rate'] = decile_stats['Defaults'] / decile_stats['Defaults'].sum()
    decile_stats['Cumulative_Capture'] = decile_stats['Capture_Rate'][::-1].cumsum()[::-1]

    print("\n--- Risk Decile Segmentation Summary ---")
    # Print clean table
    print(decile_stats[['Borrowers', 'Default_Rate', 'Capture_Rate', 'Cumulative_Capture']].to_string(formatters={
        'Default_Rate': '{:.1%}'.format,
        'Capture_Rate': '{:.1%}'.format,
        'Cumulative_Capture': '{:.1%}'.format
    }))

    # Top-K Default Detection (e.g., Top 20%)
    top_20_defaults = decile_stats.loc[9:10, 'Defaults'].sum()
    total_defaults = decile_stats['Defaults'].sum()
    print(f"\nThe Top 20% Riskiest Borrowers (Deciles 9 & 10) account for {top_20_defaults / total_defaults:.1%} of all defaults.")

    # Plot Default Rate by Risk Decile
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x=decile_stats.index, 
        y=decile_stats['Default_Rate'] * 100, 
        palette='coolwarm'
    )
    plt.title('Actual Default Rate by Predicted Risk Decile')
    plt.xlabel('Risk Decile (1 = Lowest Risk, 10 = Highest Risk)')
    plt.ylabel('Actual Default Rate (%)')

    # Add data labels
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points')

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'risk_decile_default_rates.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"\nSaved decile segmentation plot to {plot_path}")

    # Save decile stats to CSV
    csv_path = os.path.join(RESULTS_DIR, 'risk_decile_stats.csv')
    decile_stats.to_csv(csv_path)
    print(f"Saved decile statistics to {csv_path}")


if __name__ == "__main__":
    run_risk_ranking()

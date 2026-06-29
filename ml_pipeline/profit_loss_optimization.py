import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Running Profit/Loss Optimization...")
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/summary', exist_ok=True)
    
    print("Loading model and test data...")
    model = joblib.load('models/xgb_pipeline.pkl')
    X_test, y_test = joblib.load('models/test_data.pkl')
    
    print("Generating predictions...")
    y_prob = model.predict_proba(X_test)[:, 1]
    
    loan_amnt = X_test['loan_amnt'].values
    int_rate = X_test['int_rate'].values / 100.0
    actual = y_test.values
    
    # Financial Assumptions
    RECOVERY_RATE = 0.20
    
    thresholds = np.linspace(0.01, 0.99, 99)
    profits = []
    
    # Calculate baseline profit (Approve everyone)
    baseline_revenue = np.sum(loan_amnt[actual == 0] * int_rate[actual == 0])
    baseline_loss = np.sum(loan_amnt[actual == 1] * (1 - RECOVERY_RATE))
    baseline_profit = baseline_revenue - baseline_loss
    
    for t in thresholds:
        # Predict 0 (Approve) if prob < t, else Predict 1 (Reject)
        approve = (y_prob < t)
        
        # True Negatives (Approve and Paid)
        tn_idx = approve & (actual == 0)
        revenue = np.sum(loan_amnt[tn_idx] * int_rate[tn_idx])
        
        # False Negatives (Approve and Default)
        fn_idx = approve & (actual == 1)
        loss = np.sum(loan_amnt[fn_idx] * (1 - RECOVERY_RATE))
        
        profit = revenue - loss
        profits.append(profit)
        
    best_idx = np.argmax(profits)
    best_threshold = thresholds[best_idx]
    max_profit = profits[best_idx]
    
    print(f"Baseline Profit (Approve All): ${baseline_profit:,.2f}")
    print(f"Optimal Threshold: {best_threshold:.2f}")
    print(f"Maximized Profit: ${max_profit:,.2f}")
    print(f"Net Gain from Model: ${(max_profit - baseline_profit):,.2f}")
    
    # Save results
    results = {
        'baseline_profit': float(baseline_profit),
        'optimal_threshold': float(best_threshold),
        'max_profit': float(max_profit),
        'profit_curve': {
            'thresholds': thresholds.tolist(),
            'profits': profits
        }
    }
    with open('results/summary/profit_optimization.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    # Plot curve
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, profits, label='Portfolio Profit', color='#2563eb', linewidth=2)
    plt.axvline(x=best_threshold, color='#ef4444', linestyle='--', label=f'Optimal Threshold ({best_threshold:.2f})')
    plt.axhline(y=baseline_profit, color='#10b981', linestyle=':', label='Baseline Profit (Approve All)')
    plt.title('Profit vs. Decision Threshold')
    plt.xlabel('Probability Threshold (Reject if > t)')
    plt.ylabel('Expected Portfolio Profit ($)')
    
    # Format Y-axis as currency in millions
    def currency_formatter(x, pos):
        return f"${x/1e6:.0f}M"
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(currency_formatter))
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/plots/profit_curve.png', dpi=300)
    plt.close()
    print("Saved profit_curve.png and profit_optimization.json")

if __name__ == "__main__":
    main()

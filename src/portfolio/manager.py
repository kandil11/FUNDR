import os
import joblib
import pandas as pd
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SUMMARY_DIR = os.path.join(BASE_DIR, "results", "summary")
MODELS_DIR = os.path.join(BASE_DIR, "models")

class PortfolioManager:
    """Handles generating the investment portfolio and aggregating portfolio strategy."""

    def generate_opportunities(self):
        """Scores all loans in test_data and generates investment_opportunities.csv."""
        print("Loading test data and model...")
        test_path = os.path.join(MODELS_DIR, "test_data.pkl")
        pipeline_path = os.path.join(MODELS_DIR, "xgb_pipeline.pkl")
        
        try:
            X_test, _ = joblib.load(test_path)
            model = joblib.load(pipeline_path)
        except Exception as e:
            print(f"Error loading data or model: {e}")
            return
            
        print("Scoring loans...")
        probs = model.predict_proba(X_test)
        
        portfolio = X_test.copy()
        portfolio['Probability_of_Default'] = probs[:, 1]
        
        LGD = 0.60
        if 'int_rate' in portfolio.columns:
            expected_yield = portfolio['int_rate'] / 100
            portfolio['Expected_Loss'] = portfolio['Probability_of_Default'] * LGD
            portfolio['Risk_Adjusted_Return'] = (expected_yield * (1 - portfolio['Probability_of_Default'])) - portfolio['Expected_Loss']
        else:
            print("Warning: int_rate not found. Cannot calculate financial returns.")
            return

        opportunities = portfolio[portfolio['Risk_Adjusted_Return'] > 0.0].copy()
        opportunities = opportunities.sort_values(by='Risk_Adjusted_Return', ascending=False)
        
        cols_to_keep = ['loan_amnt', 'term', 'grade', 'int_rate', 'Probability_of_Default', 'Expected_Loss', 'Risk_Adjusted_Return', 'annual_inc', 'dti', 'purpose', 'addr_state']
        final_cols = [c for c in cols_to_keep if c in opportunities.columns]
        opportunities_clean = opportunities[final_cols]
        
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        out_path = os.path.join(PROCESSED_DIR, "investment_opportunities.csv")
        opportunities_clean.to_csv(out_path, index=False)
        print(f"Saved {len(opportunities_clean)} profitable loans to {out_path}")
        
        top_50 = opportunities_clean.head(50).to_dict(orient='records')
        os.makedirs(SUMMARY_DIR, exist_ok=True)
        with open(os.path.join(SUMMARY_DIR, "top_investments.json"), "w") as f:
            json.dump(top_50, f, indent=2)
            
        sample_size = min(2000, len(opportunities_clean))
        graph_data_df = opportunities_clean.sample(n=sample_size, random_state=42)
        graph_data = graph_data_df.to_dict(orient='records')
        with open(os.path.join(SUMMARY_DIR, "investment_graph_data.json"), "w") as f:
            json.dump(graph_data, f, indent=2)

    def analyze_strategy(self, ai_client=None):
        """Analyzes demographics of the investment portfolio and generates an AI Thesis."""
        print("Analyzing investment portfolio strategy...")
        data_path = os.path.join(PROCESSED_DIR, "investment_opportunities.csv")
        
        if not os.path.exists(data_path):
            print("Error: investment_opportunities.csv not found. Run generate_opportunities() first.")
            return
            
        df = pd.read_csv(data_path)
        
        top_state = df['addr_state'].value_counts().idxmax() if 'addr_state' in df.columns else "N/A"
        top_state_pct = (df['addr_state'].value_counts().max() / len(df)) * 100 if 'addr_state' in df.columns else 0
        top_purpose = df['purpose'].value_counts().idxmax().replace('_', ' ').title() if 'purpose' in df.columns else "N/A"
        top_purpose_pct = (df['purpose'].value_counts().max() / len(df)) * 100 if 'purpose' in df.columns else 0
        avg_income = df['annual_inc'].mean() if 'annual_inc' in df.columns else 0
        avg_dti = df['dti'].mean() if 'dti' in df.columns else 0

        ai_thesis = "AI Thesis generation failed."
        if ai_client:
            print("Generating AI Investment Thesis via OpenRouter...")
            ai_thesis = ai_client.generate_portfolio_thesis(
                top_state, top_state_pct, top_purpose, top_purpose_pct, avg_income, avg_dti
            )

        strategy = {
            "top_state": top_state,
            "top_state_pct": top_state_pct,
            "top_purpose": top_purpose,
            "top_purpose_pct": top_purpose_pct,
            "avg_income": avg_income,
            "avg_dti": avg_dti,
            "total_opportunities": len(df),
            "ai_thesis": ai_thesis
        }
        
        os.makedirs(SUMMARY_DIR, exist_ok=True)
        out_path = os.path.join(SUMMARY_DIR, "portfolio_strategy.json")
        with open(out_path, "w") as f:
            json.dump(strategy, f, indent=2)
            
        print("Portfolio Strategy Analysis Complete.")

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

class AIClient:
    """Handles communication with the OpenRouter AI API."""
    
    def __init__(self, env_path: str = ".env"):
        load_dotenv(env_path)
        self.api_key = os.getenv("apikey")
        if not self.api_key:
            print("Warning: Missing 'apikey' in .env. AI features disabled.")
            self.client = None
        else:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
            )
            
    def is_available(self) -> bool:
        return self.client is not None

    def generate_loan_thesis(self, prob_bad: float, input_dict: dict, increases_str: str, decreases_str: str) -> str:
        """Generates an Investment Thesis for a single loan applicant."""
        if not self.is_available():
            return "AI Client not available."
            
        prompt = f"""You are the Chief Investment Officer of a quantitative hedge fund. Your job is to evaluate this specific loan applicant for our portfolio.
The XGBoost model just evaluated this loan applicant and assigned a {prob_bad:.1%} probability of Default/Charged Off.

Mathematical Risk Factors from SHAP Analysis:
- Top factors INCREASING risk probability: {increases_str if increases_str else 'None significant'}
- Top factors DECREASING risk probability: {decreases_str if decreases_str else 'None significant'}

Applicant Profile:
{json.dumps(input_dict, indent=2)}

Task:
Write a concise, confident 2-3 sentence 'AI Investment Thesis' for this specific loan.
State whether we should INVEST or REJECT this loan, and explain why using the applicant profile and the SHAP mathematical risk factors provided.
Do not use bullet points or intros. Just write the thesis paragraph.
"""
        try:
            response = self.client.chat.completions.create(
                model="openrouter/free",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenRouter API Error: {e}")
            return "OpenRouter API temporarily unavailable."

    def generate_portfolio_thesis(self, top_state: str, top_state_pct: float, top_purpose: str, top_purpose_pct: float, avg_income: float, avg_dti: float) -> str:
        """Generates an Investment Thesis for the entire portfolio aggregate data."""
        if not self.is_available():
            return "AI Thesis generation failed."

        prompt = f"""
You are the Chief Investment Officer of a quantitative hedge fund. 
Our XGBoost machine learning model has identified the most profitable consumer loans to invest in.
Here is the demographic profile of our "winning" portfolio:
- Top State: {top_state} ({top_state_pct:.1f}% of loans)
- Primary Loan Purpose: {top_purpose} ({top_purpose_pct:.1f}% of loans)
- Target Borrower Average Income: ${avg_income:,.0f}
- Target Borrower Average DTI (Debt-to-Income): {avg_dti:.1f}%

Write a concise, confident 2-3 sentence 'Investment Thesis' explaining to our stakeholders WHY this specific demographic profile represents a strong, risk-adjusted investment strategy. 
Do not use bullet points or intros. Just write the thesis paragraph.
"""
        try:
            response = self.client.chat.completions.create(
                model="openrouter/free",
                messages=[
                    {"role": "system", "content": "You are a Chief Investment Officer. Provide concise financial analysis."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenRouter API Error: {e}")
            return "OpenRouter API temporarily unavailable."

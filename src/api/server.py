from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

from src.core.ai_client import AIClient
from src.models.risk_engine import RiskEngine

app = FastAPI(title="FUNDR Analytics Engine")

# Mount static files for dashboard data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "results", "summary")), name="static")

# Initialize Domain Services
ai_client = AIClient(env_path=os.path.join(BASE_DIR, ".env"))
risk_engine = RiskEngine()

class LoanRequest(BaseModel):
    loan_amnt: float
    term: str
    int_rate: float
    annual_inc: float

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the main Stitch UI dashboard."""
    ui_path = os.path.join(BASE_DIR, "stitch_ui.html")
    try:
        with open(ui_path, "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="stitch_ui.html not found.")

@app.post("/predict")
async def predict(request: Request):
    """Handles real-time loan simulations from the UI."""
    try:
        data = await request.json()
        input_dict = {
            'loan_amnt': float(data.get('loan_amnt', 0)),
            'term': data.get('term', '36 months'),
            'int_rate': float(data.get('int_rate', 0)),
            'annual_inc': float(data.get('annual_inc', 0)),
        }
        
        # 1. Run XGBoost Prediction & SHAP
        try:
            prob_bad, increases_risk, decreases_risk = risk_engine.predict_risk(input_dict)
        except Exception as e:
            return JSONResponse({"error": f"Risk Engine Error: {str(e)}"}, status_code=500)
            
        increases_str = ", ".join([f"{f['feature']}" for f in increases_risk])
        decreases_str = ", ".join([f"{f['feature']}" for f in decreases_risk])
        
        # 2. Generate AI Investment Thesis
        ai_auditor_decision = ai_client.generate_loan_thesis(
            prob_bad=prob_bad, 
            input_dict=input_dict, 
            increases_str=increases_str, 
            decreases_str=decreases_str
        )
        
        # 3. Return combined response
        return JSONResponse({
            "xgboost_risk": float(prob_bad),
            "ai_auditor_decision": ai_auditor_decision,
            "shap_factors": {
                "increases_risk": increases_risk,
                "decreases_risk": decreases_risk
            }
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)

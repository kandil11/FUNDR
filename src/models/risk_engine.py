import os
import joblib
import pandas as pd
import shap
from typing import Tuple, List, Dict, Any

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
from ml_pipeline.preprocessing import preprocess_inference

class RiskEngine:
    """Handles XGBoost model loading, predictions, and SHAP value generation."""
    
    def __init__(self, model_path: str = "models/xgb_pipeline.pkl"):
        self.model_path = os.path.join(BASE_DIR, model_path)
        try:
            self.model = joblib.load(self.model_path)
        except Exception as e:
            print(f"Failed to load model from {self.model_path}: {e}")
            self.model = None

    def is_ready(self) -> bool:
        return self.model is not None

    def predict_risk(self, input_dict: dict) -> Tuple[float, List[dict], List[dict]]:
        """
        Runs the inference pipeline on a single loan dictionary.
        Returns (Probability_of_Default, Increases_Risk_Factors, Decreases_Risk_Factors)
        """
        if not self.is_ready():
            raise RuntimeError("Model is not loaded.")

        # Prepare dummy fields that pipeline expects but UI doesn't provide
        full_input = input_dict.copy()
        if 'dti' not in full_input:
            full_input['dti'] = 15.0
        
        dummy_fields = {
            'fico_range_low': 700,
            'fico_range_high': 704,
            'revol_util': 30.0,
            'installment': 500.0,
            'delinq_2yrs': 0,
            'inq_last_6mths': 0,
            'open_acc': 10,
            'pub_rec': 0,
            'tot_cur_bal': 50000,
            'mths_since_rcnt_il': 12,
            'acc_open_past_24mths': 2,
            'bc_util': 30.0,
            'mo_sin_old_il_acct': 120,
            'mo_sin_old_rev_tl_op': 120,
            'mo_sin_rcnt_rev_tl_op': 12,
            'mo_sin_rcnt_tl': 12,
            'mths_since_recent_bc': 12,
            'mths_since_recent_inq': 6,
            'num_tl_op_past_12m': 1,
            'percent_bc_gt_75': 0.0,
            'tot_hi_cred_lim': 60000,
            'grade': 'B',
            'home_ownership': 'MORTGAGE',
            'purpose': 'debt_consolidation',
            'addr_state': 'CA',
            'verification_status': 'Verified',
            'mort_acc': 1,
            'pub_rec_bankruptcies': 0
        }
        
        for k, v in dummy_fields.items():
            if k not in full_input:
                full_input[k] = v
                
        single_row = pd.DataFrame([full_input])
        processed_df = preprocess_inference(single_row, pipeline_path=self.model_path)
        
        probs = self.model.predict_proba(processed_df)
        prob_bad = probs[0, 1]
        
        # Calculate SHAP Values
        preprocessor = self.model.named_steps['preprocessor']
        classifier = self.model.named_steps['classifier']
        
        X_transformed = preprocessor.transform(processed_df)
        feature_names = preprocessor.get_feature_names_out()
        
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_transformed)[0]
        
        # Clean feature names
        clean_names = [n.replace('num__', '').replace('cat__', '').replace('_', ' ').title() for n in feature_names]
        
        # Sort factors
        factors = list(zip(clean_names, shap_values))
        factors.sort(key=lambda x: x[1], reverse=True)
        
        increases_risk = [{"feature": f[0], "impact": float(f[1])} for f in factors if f[1] > 0.05][:3]
        decreases_risk = [{"feature": f[0], "impact": float(f[1])} for f in reversed(factors) if f[1] < -0.05][:3]
        
        return prob_bad, increases_risk, decreases_risk

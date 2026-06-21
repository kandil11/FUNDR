import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from fetch_data import ensure_data_exists

def train_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(BASE_DIR, "..", "data", "processed", "accepted_loans_final.csv")
    results_dir = os.path.join(BASE_DIR, "..", "results", "summary")
    
    # Ensure data is downloaded from Google Drive if missing
    ensure_data_exists()
    
    print("Loading data for XGBoost training...")
    cols = [
        'loan_status', 'fico_range_high', 'int_rate', 'dti', 'loan_amnt', 'annual_inc', 'revol_util', 
        'term', 'home_ownership', 'purpose', 'verification_status', 'inq_last_6mths', 
        'delinq_2yrs', 'pub_rec', 'installment', 'acc_open_past_24mths', 'mort_acc', 
        'tot_cur_bal', 'bc_util', 'percent_bc_gt_75', 'pub_rec_bankruptcies', 'num_tl_op_past_12m', 'tot_hi_cred_lim'
    ]
    df = pd.read_csv(input_file, usecols=cols, low_memory=False)
    print(f"Data shape loaded: {df.shape}")
    
    # Feature Engineering
    print("Engineering custom financial ratios...")
    # Add a small epsilon to denominator to prevent division by zero
    df['loan_to_income_ratio'] = df['loan_amnt'] / (df['annual_inc'] + 1.0)
    df['payment_to_income'] = df['installment'] / ((df['annual_inc'] / 12.0) + 1.0)
    
    # 1. Define target (is_default)
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off', 'Default'])]
    
    target_map = {
        'Fully Paid': 0,
        'Charged Off': 1,
        'Default': 1
    }
    df['is_default'] = df['loan_status'].map(target_map)
    print(f"Filtered to completed loans. Shape: {df.shape}")
        
    print(f"Using entire dataset of size: {len(df)}")
    
    num_features = [
        'fico_range_high', 'int_rate', 'dti', 'loan_amnt', 'annual_inc', 'revol_util', 
        'inq_last_6mths', 'delinq_2yrs', 'pub_rec', 'installment', 'acc_open_past_24mths', 
        'mort_acc', 'tot_cur_bal', 'bc_util', 'percent_bc_gt_75', 'pub_rec_bankruptcies', 
        'num_tl_op_past_12m', 'tot_hi_cred_lim', 'loan_to_income_ratio', 'payment_to_income'
    ]
    cat_features = ['term', 'home_ownership', 'purpose', 'verification_status']
    
    X = df[num_features + cat_features]
    y = df['is_default']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 2. Build Pipeline Preprocessor
    print("Building Preprocessor Pipeline...")
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ])

    # 3. Hyperparameter Tuning with XGBoost
    print("Initializing Hyperparameter Tuning with RandomizedSearchCV...")
    scale_pos_weight = 2.0
    
    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='auc'
        ))
    ])
    
    # Param grid for the classifier inside the pipeline
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.05, 0.1, 0.2],
        'classifier__subsample': [0.8, 1.0],
        'classifier__colsample_bytree': [0.8, 1.0]
    }
    
    search = RandomizedSearchCV(
        xgb_pipeline, 
        param_distributions=param_grid, 
        n_iter=5, # Keep it relatively low to save time (5 fits * 3 cv = 15 fits)
        cv=3, 
        scoring='roc_auc', 
        verbose=1, 
        random_state=42,
        n_jobs=-1
    )
    
    print("Training XGBoost with RandomizedSearchCV (This may take a few minutes)...")
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_
    print(f"Best Parameters found: {search.best_params_}")
    
    # 4. Train Logistic Regression to extract actual weights for the browser simulator
    print("Training LogisticRegression model for coefficient extraction (no scaling)...")
    
    lr_numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    lr_preprocessor = ColumnTransformer(
        transformers=[
            ('num', lr_numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ])
        
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', lr_preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=3000, random_state=42))
    ])
    
    lr_pipeline.fit(X_train, y_train)
    
    # Extract feature names after encoding
    ohe = lr_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = ohe.get_feature_names_out(cat_features)
    all_feature_names = num_features + list(cat_feature_names)
    
    lr_model = lr_pipeline.named_steps['classifier']
    
    model_weights = {}
    for feat, coef in zip(all_feature_names, lr_model.coef_[0]):
        clean_feat = feat.replace(" ", "_").replace("<", "lt").replace(">", "gt")
        model_weights[clean_feat] = float(coef)
    model_weights['intercept'] = float(lr_model.intercept_[0])
    
    # 5. Predict & Evaluate (using the BEST XGBoost Pipeline)
    y_pred = best_pipeline.predict(X_test)
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    
    print("\nMODEL PERFORMANCE (Optimized):")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    
    # Feature Importances from the best XGBoost model
    xgb_model = best_pipeline.named_steps['classifier']
    importances = xgb_model.feature_importances_
    
    xgb_ohe = best_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    xgb_cat_names = xgb_ohe.get_feature_names_out(cat_features)
    xgb_all_features = num_features + list(xgb_cat_names)
    
    feat_imp = [{"feature": f, "importance": float(imp)} for f, imp in zip(xgb_all_features, importances)]
    feat_imp = sorted(feat_imp, key=lambda x: x['importance'], reverse=True)
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    indices = np.linspace(0, len(fpr) - 1, 15, dtype=int)
    fpr_sampled = [float(fpr[i]) for i in indices]
    tpr_sampled = [float(tpr[i]) for i in indices]
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # 6. Read & Overwrite Dashboard Data
    json_path = os.path.join(results_dir, "dashboard_data.json")
    js_path = os.path.join(results_dir, "dashboard_data.js")
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            dashboard_data = json.load(f)
            
        dashboard_data['xgboost'] = {
            "feature_importance": feat_imp,
            "metrics": {
                "auc": round(float(auc), 4),
                "accuracy": round(float(acc) * 100, 2),
                "precision": round(float(prec) * 100, 2),
                "recall": round(float(rec) * 100, 2),
                "f1_score": round(float(f1) * 100, 2)
            },
            "roc_curve": {
                "fpr": fpr_sampled,
                "tpr": tpr_sampled
            },
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp)
            },
            "model_weights": model_weights
        }
        
        dashboard_data['kpis']['model_auc'] = round(float(auc), 3)
        dashboard_data['kpis']['model_acc'] = round(float(acc) * 100, 1)
        
        with open(json_path, 'w') as f:
            json.dump(dashboard_data, f, indent=2)
            
        with open(js_path, 'w') as f:
            f.write(f"window.DASHBOARD_DATA = {json.dumps(dashboard_data, indent=2)};")
            
        print("\nDashboard files updated successfully with optimal pipeline metrics!")
    else:
        print(f"Error: dashboard_data.json not found in {results_dir}")

if __name__ == "__main__":
    train_model()

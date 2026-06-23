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
import joblib


def train_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(BASE_DIR, "..", "data", "processed", "accepted_loans_final.csv")
    results_dir = os.path.join(BASE_DIR, "..", "results", "summary")
    models_dir = os.path.join(BASE_DIR, "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    ensure_data_exists()

    cols = [
        'loan_status', 'fico_range_high', 'int_rate', 'dti', 'loan_amnt', 'annual_inc', 'revol_util',
        'term', 'home_ownership', 'purpose', 'verification_status', 'inq_last_6mths',
        'delinq_2yrs', 'pub_rec', 'installment', 'acc_open_past_24mths', 'mort_acc',
        'tot_cur_bal', 'bc_util', 'percent_bc_gt_75', 'pub_rec_bankruptcies', 'num_tl_op_past_12m', 'tot_hi_cred_lim'
    ]

    df = pd.read_csv(input_file, usecols=cols, low_memory=False)

    df['loan_to_income_ratio'] = df['loan_amnt'] / (df['annual_inc'] + 1.0)
    df['payment_to_income'] = df['installment'] / ((df['annual_inc'] / 12.0) + 1.0)

    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off', 'Default'])]

    target_map = {'Fully Paid': 0, 'Charged Off': 1, 'Default': 1}
    df['is_default'] = df['loan_status'].map(target_map)

    num_features = [
        'fico_range_high', 'int_rate', 'dti', 'loan_amnt', 'annual_inc', 'revol_util',
        'inq_last_6mths', 'delinq_2yrs', 'pub_rec', 'installment', 'acc_open_past_24mths',
        'mort_acc', 'tot_cur_bal', 'bc_util', 'percent_bc_gt_75', 'pub_rec_bankruptcies',
        'num_tl_op_past_12m', 'tot_hi_cred_lim', 'loan_to_income_ratio', 'payment_to_income'
    ]

    cat_features = ['term', 'home_ownership', 'purpose', 'verification_status']

    X = df[num_features + cat_features]
    y = df['is_default']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])

    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            scale_pos_weight=2.0,
            random_state=42,
            n_jobs=-1,
            eval_metric='auc'
        ))
    ])

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
        n_iter=5,
        cv=3,
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_

    lr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            class_weight='balanced',
            max_iter=3000,
            random_state=42
        ))
    ])

    lr_pipeline.fit(X_train, y_train)

    # ===================== ADDED: SAVE MODELS =====================
    joblib.dump(best_pipeline, os.path.join(models_dir, "xgb_pipeline.pkl"))
    joblib.dump(lr_pipeline, os.path.join(models_dir, "lr_pipeline.pkl"))
    joblib.dump((X_test, y_test), os.path.join(models_dir, "test_data.pkl"))

    print("Models saved successfully.")


if __name__ == "__main__":
    train_model()
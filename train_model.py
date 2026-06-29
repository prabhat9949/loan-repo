import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from utils import LoanPreprocessor


DATA_PATH = Path("loan_data.csv")
MODEL_PATH = Path("loan_model.pkl")
SCALER_PATH = Path("scaler.pkl")
METRICS_PATH = Path("model_metrics.json")


def create_dataset(path: Path) -> None:
    rng = np.random.default_rng(42)
    n_rows = 900
    data = {
        "age": rng.integers(22, 70, size=n_rows),
        "gender": rng.choice(["Male", "Female", "Other"], size=n_rows, p=[0.55, 0.4, 0.05]),
        "marital_status": rng.choice(["Single", "Married", "Divorced"], size=n_rows, p=[0.35, 0.55, 0.1]),
        "dependents": rng.integers(0, 4, size=n_rows),
        "education": rng.choice(["Graduate", "Postgraduate", "High School"], size=n_rows, p=[0.5, 0.3, 0.2]),
        "employment_status": rng.choice(["Employed", "Self Employed", "Unemployed"], size=n_rows, p=[0.7, 0.2, 0.1]),
        "years_employed": rng.integers(0, 20, size=n_rows),
        "self_employed": rng.choice([0, 1], size=n_rows, p=[0.8, 0.2]),
        "residence_type": rng.choice(["Owned", "Rented"], size=n_rows, p=[0.65, 0.35]),
        "housing_status": rng.choice(["Owned", "Rented", "Mortgaged"], size=n_rows, p=[0.55, 0.25, 0.2]),
        "city": rng.choice(["Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad"], size=n_rows),
        "state": rng.choice(["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "Telangana"], size=n_rows),
        "country": rng.choice(["India"], size=n_rows),
        "monthly_income": rng.normal(65000, 22000, size=n_rows).clip(12000, 250000).astype(int),
        "annual_income": rng.normal(780000, 300000, size=n_rows).clip(150000, 3000000).astype(int),
        "existing_emi": rng.normal(8000, 6000, size=n_rows).clip(0, 50000).astype(int),
        "savings": rng.normal(250000, 180000, size=n_rows).clip(5000, 1500000).astype(int),
        "bank_balance": rng.normal(180000, 130000, size=n_rows).clip(10000, 2000000).astype(int),
        "loan_amount": rng.normal(450000, 220000, size=n_rows).clip(50000, 2500000).astype(int),
        "loan_purpose": rng.choice(["Education", "Home", "Business", "Vehicle", "Medical"], size=n_rows),
        "loan_term": rng.choice([12, 24, 36, 48, 60], size=n_rows),
        "interest_rate": rng.uniform(8.5, 16.0, size=n_rows),
        "collateral_available": rng.choice([0, 1], size=n_rows, p=[0.35, 0.65]),
        "coapplicant_income": rng.normal(15000, 12000, size=n_rows).clip(0, 180000).astype(int),
        "credit_score": rng.normal(720, 110, size=n_rows).clip(300, 900).astype(int),
        "past_loan_defaults": rng.choice([0, 1], size=n_rows, p=[0.85, 0.15]),
        "credit_card_holder": rng.choice([0, 1], size=n_rows, p=[0.75, 0.25]),
        "num_open_accounts": rng.integers(1, 12, size=n_rows),
        "current_debt": rng.normal(180000, 95000, size=n_rows).clip(0, 1000000).astype(int),
        "debt_to_income_ratio": rng.uniform(0.1, 0.8, size=n_rows),
        "cibil_score": rng.normal(720, 110, size=n_rows).clip(300, 900).astype(int),
        "vehicle_ownership": rng.choice([0, 1], size=n_rows, p=[0.7, 0.3]),
        "property_ownership": rng.choice([0, 1], size=n_rows, p=[0.6, 0.4]),
        "business_ownership": rng.choice([0, 1], size=n_rows, p=[0.8, 0.2]),
        "existing_loans": rng.integers(0, 5, size=n_rows),
        "pan_verified": rng.choice([0, 1], size=n_rows, p=[0.9, 0.1]),
        "aadhaar_verified": rng.choice([0, 1], size=n_rows, p=[0.95, 0.05]),
        "income_tax_filed": rng.choice([0, 1], size=n_rows, p=[0.82, 0.18]),
        "bank_statement_available": rng.choice([0, 1], size=n_rows, p=[0.85, 0.15]),
        "salary_slips_available": rng.choice([0, 1], size=n_rows, p=[0.8, 0.2]),
    }
    df = pd.DataFrame(data)
    base_score = (
        0.015 * (df["credit_score"] - 300)
        + 0.00003 * df["monthly_income"]
        + 0.04 * df["years_employed"]
        + 0.8 * df["collateral_available"]
        + 1.2 * (df["past_loan_defaults"] == 0)
        + 0.5 * df["pan_verified"]
        + 0.4 * df["aadhaar_verified"]
        + 0.35 * df["bank_statement_available"]
        + 0.4 * (df["education"] == "Postgraduate")
        + 0.2 * (df["gender"] == "Female")
        - 0.00001 * df["loan_amount"]
        - 0.00001 * df["current_debt"]
        - 0.05 * df["debt_to_income_ratio"]
        - 0.06 * df["existing_emi"]
        - 0.03 * (df["self_employed"] == 1)
    )
    noise = rng.normal(0, 7, size=n_rows)
    score = base_score + noise
    positive_count = 220
    positive_indices = np.argsort(score)[-positive_count:]
    df["loan_approved"] = 0
    df.loc[positive_indices, "loan_approved"] = 1
    df.to_csv(path, index=False)


def train_model() -> None:
    if not DATA_PATH.exists():
        print("Creating Loan Dataset...")
        create_dataset(DATA_PATH)

    print("Loading Dataset...")
    df = pd.read_csv(DATA_PATH)
    if df["loan_approved"].nunique() < 2 or df["loan_approved"].value_counts().min() < 2:
        print("Rebuilding Dataset with balanced labels...")
        create_dataset(DATA_PATH)
        df = pd.read_csv(DATA_PATH)

    print("Cleaning Missing Values...")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    print("Encoding Features...")
    feature_columns = [col for col in df.columns if col != "loan_approved"]
    X = df[feature_columns]
    y = df["loan_approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = LoanPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=220, random_state=42, class_weight="balanced")
    model.fit(X_train_processed, y_train)

    predictions = model.predict(X_test_processed)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_processed)[:, 1])
    cm = confusion_matrix(y_test, predictions)

    print(f"Accuracy : {accuracy * 100:.1f}%")
    print("Saving Model...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, SCALER_PATH)

    metrics = {
        "algorithm": "Random Forest Classifier",
        "training_accuracy": float(accuracy_score(y_train, model.predict(X_train_processed))),
        "testing_accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
    }
    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("Model Saved Successfully")


if __name__ == "__main__":
    train_model()

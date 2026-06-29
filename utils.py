import json
import math
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class LoanPreprocessor:
    def __init__(self):
        self.numeric_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.numeric_imputer = SimpleImputer(strategy="median")
        self.categorical_imputer = SimpleImputer(strategy="most_frequent")
        self.scaler = StandardScaler()
        try:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    def _split_columns(self, df: pd.DataFrame) -> None:
        self.numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        self.categorical_columns = [col for col in df.columns if col not in self.numeric_columns]

    def fit(self, df: pd.DataFrame):
        self._split_columns(df)
        numeric_df = df[self.numeric_columns].copy()
        categorical_df = df[self.categorical_columns].copy()

        self.numeric_imputer.fit(numeric_df)
        self.categorical_imputer.fit(categorical_df)

        numeric_imputed = self.numeric_imputer.transform(numeric_df)
        self.scaler.fit(numeric_imputed)

        categorical_imputed = self.categorical_imputer.transform(categorical_df)
        self.encoder.fit(categorical_imputed)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_df = df[self.numeric_columns].copy()
        categorical_df = df[self.categorical_columns].copy()

        numeric_imputed = self.numeric_imputer.transform(numeric_df)
        numeric_scaled = self.scaler.transform(numeric_imputed)

        categorical_imputed = self.categorical_imputer.transform(categorical_df)
        categorical_encoded = self.encoder.transform(categorical_imputed)

        numeric_frame = pd.DataFrame(numeric_scaled, index=df.index, columns=self.numeric_columns)
        categorical_frame = pd.DataFrame(
            categorical_encoded,
            index=df.index,
            columns=self.encoder.get_feature_names_out(self.categorical_columns),
        )
        return pd.concat([numeric_frame, categorical_frame], axis=1)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)


def load_assets(model_path: str = "loan_model.pkl", scaler_path: str = "scaler.pkl") -> Tuple[object, object]:
    model = joblib.load(model_path)
    preprocessor = joblib.load(scaler_path)
    return model, preprocessor


def predict_eligibility(model, preprocessor, input_df: pd.DataFrame) -> Tuple[int, float, List[str]]:
    transformed = preprocessor.transform(input_df)
    probabilities = model.predict_proba(transformed)[0]
    approval_probability = float(probabilities[1])
    approved = int(approval_probability >= 0.55)

    reasons = []
    if input_df.iloc[0]["credit_score"] >= 700:
        reasons.append("Excellent Credit Score")
    if input_df.iloc[0]["monthly_income"] >= 60000:
        reasons.append("Stable Income")
    if input_df.iloc[0]["years_employed"] >= 3:
        reasons.append("Strong Employment History")
    if input_df.iloc[0]["debt_to_income_ratio"] <= 0.4:
        reasons.append("Healthy Debt Ratio")
    if input_df.iloc[0]["savings"] >= 200000:
        reasons.append("Healthy Savings")

    if not reasons:
        reasons = ["Positive Applicant Profile"]

    return approved, approval_probability, reasons


def build_pdf_report(approval: int, probability: float, reasons: List[str], input_df: pd.DataFrame) -> bytes:
    content = []
    content.append("%PDF-1.4")
    content.append("1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj")
    content.append("2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj")
    content.append("3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj")
    content.append("4 0 obj << /Length 0 >> stream")
    text = f"AI Loan Eligibility Report\n\nDecision: {'Approved' if approval else 'Rejected'}\nProbability: {probability:.2%}\n\nReasons:\n"
    text += "\n".join(f"- {reason}" for reason in reasons)
    text += f"\n\nApplicant Income: {input_df.iloc[0]['monthly_income']:.0f}\nLoan Amount: {input_df.iloc[0]['loan_amount']:.0f}"
    content.append(text)
    content.append("endstream endobj")
    content.append("5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj")
    content.append("xref")
    content.append("0 6")
    content.append("0000000000 65535 f ")
    return "\n".join(content).encode("latin-1")


def save_prediction_history(history: List[Dict], history_path: str = "prediction_history.csv") -> None:
    df = pd.DataFrame(history)
    df.to_csv(history_path, index=False)


def load_prediction_history(history_path: str = "prediction_history.csv") -> List[Dict]:
    if not os.path.exists(history_path):
        return []
    df = pd.read_csv(history_path)
    return df.to_dict(orient="records")

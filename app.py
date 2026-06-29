import pandas as pd
import streamlit as st
from pathlib import Path

from utils import load_assets, predict_eligibility


MODEL_PATH = Path("loan_model.pkl")
SCALER_PATH = Path("scaler.pkl")


st.set_page_config(page_title="Loan Eligibility Checker", page_icon="🏦", layout="wide")


def inject_theme_css() -> None:
    st.markdown(
        """
        <style>
            .stApp { background: linear-gradient(135deg, #f8fbff 0%, #eef7ff 100%); }
            .card { background: white; border-radius: 18px; padding: 18px; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08); margin-bottom: 14px; }
            .title { font-size: 2rem; font-weight: 700; color: #0f172a; }
            .subtitle { color: #475569; font-size: 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_input_frame() -> pd.DataFrame:
    with st.form("loan_form"):
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='title'>Loan Eligibility Checker</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Enter the applicant details below and the system will predict whether the loan is eligible and with what confidence.</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=70, value=35, step=1)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0)
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], index=1)
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1, step=1)
            education = st.selectbox("Education Level", ["High School", "Graduate", "Postgraduate"], index=1)
            employment_status = st.selectbox("Employment Status", ["Employed", "Self Employed", "Unemployed"], index=0)
            years_employed = st.number_input("Years of Employment", min_value=0, max_value=40, value=4, step=1)
            monthly_income = st.number_input("Monthly Income", min_value=1000, value=65000, step=1000)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=720, step=1)
            loan_amount = st.number_input("Loan Amount", min_value=1000, value=450000, step=1000)
            debt_to_income_ratio = st.number_input("Debt to Income Ratio", min_value=0.0, max_value=2.0, value=0.35, step=0.01)

        with col2:
            existing_emi = st.number_input("Existing EMIs", min_value=0, value=8000, step=1000)
            savings = st.number_input("Savings", min_value=0, value=250000, step=1000)
            bank_balance = st.number_input("Bank Balance", min_value=0, value=180000, step=1000)
            loan_purpose = st.selectbox("Loan Purpose", ["Education", "Home", "Business", "Vehicle", "Medical"], index=1)
            loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60], index=2)
            collateral_available = st.selectbox("Collateral Available", [0, 1], index=1)
            cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=720, step=1)
            pan_verified = st.selectbox("PAN Verified", [0, 1], index=1)
            aadhaar_verified = st.selectbox("Aadhaar Verified", [0, 1], index=1)
            bank_statement_available = st.selectbox("Bank Statement Available", [0, 1], index=1)
            salary_slips_available = st.selectbox("Salary Slips Available", [0, 1], index=1)

        submitted = st.form_submit_button("Check Eligibility")
        st.markdown("</div>", unsafe_allow_html=True)

        if submitted:
            if age < 18 or age > 70:
                st.error("Age must be between 18 and 70.")
            elif monthly_income <= 0:
                st.error("Monthly income must be greater than zero.")
            elif not 300 <= credit_score <= 900:
                st.error("Credit score must be between 300 and 900.")
            elif loan_amount <= 0:
                st.error("Loan amount must be greater than zero.")
            else:
                input_data = pd.DataFrame([{
                    "age": age,
                    "gender": gender,
                    "marital_status": marital_status,
                    "dependents": dependents,
                    "education": education,
                    "employment_status": employment_status,
                    "years_employed": years_employed,
                    "self_employed": 1 if employment_status == "Self Employed" else 0,
                    "residence_type": "Owned",
                    "housing_status": "Owned",
                    "city": "Mumbai",
                    "state": "Maharashtra",
                    "country": "India",
                    "monthly_income": monthly_income,
                    "annual_income": monthly_income * 12,
                    "existing_emi": existing_emi,
                    "savings": savings,
                    "bank_balance": bank_balance,
                    "loan_amount": loan_amount,
                    "loan_purpose": loan_purpose,
                    "loan_term": loan_term,
                    "interest_rate": 10.5,
                    "collateral_available": collateral_available,
                    "coapplicant_income": 0,
                    "credit_score": credit_score,
                    "past_loan_defaults": 0,
                    "credit_card_holder": 0,
                    "num_open_accounts": 3,
                    "current_debt": existing_emi * 12,
                    "debt_to_income_ratio": debt_to_income_ratio,
                    "cibil_score": cibil_score,
                    "vehicle_ownership": 0,
                    "property_ownership": 0,
                    "business_ownership": 0,
                    "existing_loans": 1 if existing_emi > 0 else 0,
                    "pan_verified": pan_verified,
                    "aadhaar_verified": aadhaar_verified,
                    "income_tax_filed": 1,
                    "bank_statement_available": bank_statement_available,
                    "salary_slips_available": salary_slips_available,
                }])
                return input_data

        return None


def main() -> None:
    inject_theme_css()

    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        st.error("Model files are missing. Please run the training script first.")
        return

    model, preprocessor = load_assets(MODEL_PATH, SCALER_PATH)
    input_data = build_input_frame()

    if input_data is not None:
        approved, probability, reasons = predict_eligibility(model, preprocessor, input_data)
        confidence = "High" if probability >= 0.75 else "Medium" if probability >= 0.6 else "Low"
        result_color = "#16a34a" if approved else "#dc2626"
        title = "Loan Eligible" if approved else "Loan Not Eligible"
        subtitle = "The applicant profile looks strong for approval." if approved else "The applicant profile is risky for approval."

        st.markdown(f"<div class='card' style='border-left: 6px solid {result_color};'>", unsafe_allow_html=True)
        st.markdown(f"<div class='title'>{title}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='subtitle'>{subtitle}</div>", unsafe_allow_html=True)
        st.metric("Approval Probability", f"{probability * 100:.2f}%")
        st.metric("Confidence Level", confidence)
        st.write("**Main Reasons**")
        for reason in reasons:
            st.write(f"• {reason}")
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

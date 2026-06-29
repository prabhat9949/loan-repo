import pandas as pd
import streamlit as st
from pathlib import Path

from utils import load_assets, predict_eligibility

MODEL_PATH = Path("loan_model.pkl")
SCALER_PATH = Path("scaler.pkl")

st.set_page_config(
    page_title="Loan Eligibility Checker",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_theme_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg: #f4f7fb;
                --panel: rgba(255, 255, 255, 0.92);
                --panel-border: rgba(15, 23, 42, 0.08);
                --text: #0f172a;
                --muted: #64748b;
                --primary: #2563eb;
                --primary-dark: #1d4ed8;
                --success: #16a34a;
                --danger: #dc2626;
                --shadow: 0 16px 40px rgba(15, 23, 42, 0.10);
                --radius: 22px;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(37, 99, 235, 0.10), transparent 30%),
                    radial-gradient(circle at top right, rgba(14, 165, 233, 0.10), transparent 28%),
                    linear-gradient(135deg, #f8fbff 0%, #eef5ff 100%);
                color: var(--text);
            }

            section.main > div {
                padding-top: 1.4rem;
                padding-bottom: 2rem;
                max-width: 1280px;
            }

            .hero {
                background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #2563eb 100%);
                color: white;
                border-radius: 28px;
                padding: 28px 28px 22px 28px;
                box-shadow: var(--shadow);
                margin-bottom: 1.2rem;
                border: 1px solid rgba(255,255,255,0.12);
            }

            .hero-title {
                font-size: 2.1rem;
                font-weight: 800;
                line-height: 1.15;
                margin-bottom: 0.35rem;
                letter-spacing: -0.02em;
            }

            .hero-subtitle {
                font-size: 1rem;
                color: rgba(255,255,255,0.88);
                max-width: 760px;
            }

            .card {
                background: var(--panel);
                backdrop-filter: blur(10px);
                border: 1px solid var(--panel-border);
                border-radius: var(--radius);
                padding: 22px;
                box-shadow: var(--shadow);
                margin-bottom: 16px;
            }

            .section-title {
                font-size: 1.15rem;
                font-weight: 700;
                color: var(--text);
                margin-bottom: 0.25rem;
            }

            .section-subtitle {
                color: var(--muted);
                font-size: 0.95rem;
                margin-bottom: 1rem;
            }

            .result-box {
                border-radius: 22px;
                padding: 22px;
                box-shadow: var(--shadow);
                background: white;
                border: 1px solid rgba(15, 23, 42, 0.08);
            }

            .metric-wrap {
                background: #f8fafc;
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 18px;
                padding: 14px 16px;
            }

            .stMetric {
                background: transparent;
            }

            .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div,
            .stTextInput input, .stTextArea textarea {
                border-radius: 14px !important;
            }

            .stFormSubmitButton > button {
                width: 100%;
                background: linear-gradient(135deg, var(--primary), var(--primary-dark));
                color: white;
                border: none;
                border-radius: 14px;
                font-weight: 700;
                padding: 0.8rem 1rem;
                box-shadow: 0 10px 20px rgba(37, 99, 235, 0.24);
            }

            .stFormSubmitButton > button:hover {
                filter: brightness(1.03);
                transform: translateY(-1px);
            }

            .tiny-note {
                color: var(--muted);
                font-size: 0.86rem;
            }

            div[data-testid="stAlert"] {
                border-radius: 16px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Loan Eligibility Checker</div>
            <div class="hero-subtitle">
                Enter applicant details and get a clean, AI-style eligibility prediction with probability, confidence, and reason highlights.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_input_frame() -> pd.DataFrame | None:
    with st.form("loan_form", clear_on_submit=False):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Applicant Information</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Fill in verified financial and profile details to evaluate the loan application.</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2, gap="large")

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

        st.markdown("</div>", unsafe_allow_html=True)

        submit_col1, submit_col2, submit_col3 = st.columns([1, 1.5, 1])
        with submit_col2:
            submitted = st.form_submit_button("Check Eligibility")

        if submitted:
            if age < 18 or age > 70:
                st.error("Age must be between 18 and 70.")
                return None
            if monthly_income <= 0:
                st.error("Monthly income must be greater than zero.")
                return None
            if not 300 <= credit_score <= 900:
                st.error("Credit score must be between 300 and 900.")
                return None
            if loan_amount <= 0:
                st.error("Loan amount must be greater than zero.")
                return None

            input_data = pd.DataFrame(
                [
                    {
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
                    }
                ]
            )
            return input_data

    return None


def render_result(approved: bool, probability: float, reasons: list[str]) -> None:
    confidence = "High" if probability >= 0.75 else "Medium" if probability >= 0.6 else "Low"
    result_color = "#16a34a" if approved else "#dc2626"
    title = "Loan Eligible" if approved else "Loan Not Eligible"
    subtitle = (
        "The applicant profile looks strong for approval."
        if approved
        else "The applicant profile is risky for approval."
    )

    st.markdown(
        f"""
        <div class="result-box" style="border-left: 7px solid {result_color};">
            <div class="section-title" style="font-size: 1.5rem; margin-bottom: 0.3rem;">{title}</div>
            <div class="section-subtitle" style="margin-bottom: 1.2rem;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3, gap="large")
    with m1:
        st.markdown('<div class="metric-wrap">', unsafe_allow_html=True)
        st.metric("Approval Probability", f"{probability * 100:.2f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-wrap">', unsafe_allow_html=True)
        st.metric("Confidence Level", confidence)
        st.markdown("</div>", unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="metric-wrap">', unsafe_allow_html=True)
        st.metric("Decision", "Approved" if approved else "Rejected")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("### Main reasons")
    if reasons:
        for reason in reasons:
            st.write(f"• {reason}")
    else:
        st.info("No reason details were returned by the model.")


def main() -> None:
    inject_theme_css()
    render_header()

    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        st.error("Model files are missing. Please run the training script first.")
        st.stop()

    with st.spinner("Loading model assets..."):
        model, preprocessor = load_assets(MODEL_PATH, SCALER_PATH)

    st.caption("Professional loan eligibility dashboard with model-backed prediction and validation.")
    input_data = build_input_frame()

    if input_data is not None:
        with st.spinner("Evaluating application..."):
            approved, probability, reasons = predict_eligibility(model, preprocessor, input_data)
        render_result(approved, probability, reasons)
    else:
        st.markdown(
            """
            <div class="card">
                <div class="section-title">Ready for prediction</div>
                <div class="section-subtitle">Complete the form and click <b>Check Eligibility</b> to see the result.</div>
                <div class="tiny-note">Tip: use accurate credit and income values for better output quality.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
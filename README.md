# AI Loan Eligibility Prediction System

A production-style Streamlit application that predicts loan eligibility using a trained machine-learning model.

## Features
- Modern banking dashboard UI
- Loan eligibility form with validation
- Trained Random Forest classifier
- Probability-based recommendation
- Analytics and model details pages
- Deployment-ready for Streamlit Cloud and Render

## Run Locally
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Deployment
- Streamlit Cloud: connect the repository and set the app entry to app.py.
- Render: use the provided render.yaml configuration.

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

@st.cache_resource
def load_artifacts():
    model    = joblib.load("churn_model.pkl")
    scaler   = joblib.load("churn_scaler.pkl")
    features = joblib.load("churn_features.pkl")
    return model, scaler, features

model, scaler, feature_names = load_artifacts()

st.title("📉 Customer Churn Prediction")
st.markdown("Predict whether a telecom customer is likely to leave.")
st.divider()

with st.sidebar:
    st.header("ℹ️ About")
    st.info("Dataset: Telecom Customer Churn\nModel: Random Forest\nAccuracy: ~82%")

st.subheader("Enter Customer Details")
col1, col2, col3 = st.columns(3)

with col1:
    gender         = st.selectbox("Gender", ["Male", "Female"])
    senior         = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner        = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents     = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure         = st.slider("Tenure (months)", 0, 72, 12)

with col2:
    phone_service  = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet       = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec     = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup  = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

with col3:
    device_prot    = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support   = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv   = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_mov  = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

col4, col5, col6 = st.columns(3)
with col4:
    contract       = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless      = st.selectbox("Paperless Billing", ["Yes", "No"])
with col5:
    payment        = st.selectbox("Payment Method", [
                        "Electronic check", "Mailed check",
                        "Bank transfer (automatic)", "Credit card (automatic)"])
with col6:
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
    total_charges   = st.number_input("Total Charges ($)", 0.0, 10000.0, 800.0)

st.divider()

if st.button("🔍 Predict Churn", use_container_width=True, type="primary"):

    # Build raw dict matching original CSV (after drop customerID)
    raw = {
        "SeniorCitizen":    1 if senior == "Yes" else 0,
        "tenure":           float(tenure),
        "MonthlyCharges":   float(monthly_charges),
        "TotalCharges":     float(total_charges),
        # get_dummies columns (drop_first=True)
        "gender_Male":                          1 if gender == "Male" else 0,
        "Partner_Yes":                          1 if partner == "Yes" else 0,
        "Dependents_Yes":                       1 if dependents == "Yes" else 0,
        "PhoneService_Yes":                     1 if phone_service == "Yes" else 0,
        "MultipleLines_No phone service":       1 if multiple_lines == "No phone service" else 0,
        "MultipleLines_Yes":                    1 if multiple_lines == "Yes" else 0,
        "InternetService_Fiber optic":          1 if internet == "Fiber optic" else 0,
        "InternetService_No":                   1 if internet == "No" else 0,
        "OnlineSecurity_No internet service":   1 if online_sec == "No internet service" else 0,
        "OnlineSecurity_Yes":                   1 if online_sec == "Yes" else 0,
        "OnlineBackup_No internet service":     1 if online_backup == "No internet service" else 0,
        "OnlineBackup_Yes":                     1 if online_backup == "Yes" else 0,
        "DeviceProtection_No internet service": 1 if device_prot == "No internet service" else 0,
        "DeviceProtection_Yes":                 1 if device_prot == "Yes" else 0,
        "TechSupport_No internet service":      1 if tech_support == "No internet service" else 0,
        "TechSupport_Yes":                      1 if tech_support == "Yes" else 0,
        "StreamingTV_No internet service":      1 if streaming_tv == "No internet service" else 0,
        "StreamingTV_Yes":                      1 if streaming_tv == "Yes" else 0,
        "StreamingMovies_No internet service":  1 if streaming_mov == "No internet service" else 0,
        "StreamingMovies_Yes":                  1 if streaming_mov == "Yes" else 0,
        "Contract_One year":                    1 if contract == "One year" else 0,
        "Contract_Two year":                    1 if contract == "Two year" else 0,
        "PaperlessBilling_Yes":                 1 if paperless == "Yes" else 0,
        "PaymentMethod_Credit card (automatic)":  1 if payment == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check":         1 if payment == "Electronic check" else 0,
        "PaymentMethod_Mailed check":             1 if payment == "Mailed check" else 0,
    }

    # CRITICAL: exact column order from training
    X_input = pd.DataFrame([raw]).reindex(columns=feature_names, fill_value=0).astype(float)
    X_scaled = scaler.transform(X_input)

    pred       = model.predict(X_scaled)[0]
    pred_proba = model.predict_proba(X_scaled)[0]

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if pred == 1:
            st.error("⚠️ HIGH CHURN RISK — This customer is likely to leave!")
        else:
            st.success("✅ LOW CHURN RISK — This customer is likely to stay!")
    with col_r2:
        st.metric("Churn Probability",     f"{pred_proba[1]*100:.1f}%")
        st.metric("Retention Probability", f"{pred_proba[0]*100:.1f}%")

    st.subheader("Probability Breakdown")
    st.bar_chart(pd.DataFrame({
        "Outcome": ["Will Stay", "Will Churn"],
        "Probability": [pred_proba[0], pred_proba[1]]
    }).set_index("Outcome"))

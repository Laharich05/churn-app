import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

# ── TRAIN MODEL FROM CSV (runs once, then cached) ─────────────────────────────
@st.cache_resource
def train_and_load():
    df = pd.read_csv("telecom_churn_data.csv")

    # Drop customerID
    df.drop("customerID", axis=1, inplace=True)

    # Fix TotalCharges (stored as string in raw CSV)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.fillna(df.median(numeric_only=True), inplace=True)
    df.drop_duplicates(inplace=True)

    # Create target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Outlier removal
    for col in df.select_dtypes(include=np.number).columns:
        if col == "Churn":
            continue
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    # Encode
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)

    model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    )
    model.fit(X_train_sc, y_train)

    return model, scaler, feature_names

# Show spinner while training
with st.spinner("⏳ Setting up model... (only happens once, takes ~15 seconds)"):
    model, scaler, feature_names = train_and_load()

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("📉 Customer Churn Prediction")
st.markdown("Predict whether a telecom customer is likely to leave.")
st.divider()

with st.sidebar:
    st.header("ℹ️ About")
    st.info("Dataset: Telecom Customer Churn\nModel: Random Forest\nAccuracy: ~82%")

# ── INPUT FORM ────────────────────────────────────────────────────────────────
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

# ── PREDICT ───────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Churn", use_container_width=True, type="primary"):

    raw = {
        "SeniorCitizen":                        1 if senior == "Yes" else 0,
        "tenure":                               float(tenure),
        "MonthlyCharges":                       float(monthly_charges),
        "TotalCharges":                         float(total_charges),
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
        "PaymentMethod_Credit card (automatic)": 1 if payment == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check":        1 if payment == "Electronic check" else 0,
        "PaymentMethod_Mailed check":            1 if payment == "Mailed check" else 0,
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
        "Outcome":     ["Will Stay", "Will Churn"],
        "Probability": [pred_proba[0], pred_proba[1]]
    }).set_index("Outcome"))

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

st.title("💳 AI Fraud Detection System")

st.markdown("Enter transaction details to check fraud risk.")

# -------------------------
# INPUT SECTION
# -------------------------

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount", value=1000.0)
    oldbalanceOrg = st.number_input("Sender Old Balance", value=5000.0)
    newbalanceOrig = st.number_input("Sender New Balance", value=4000.0)
    oldbalanceDest = st.number_input("Receiver Old Balance", value=2000.0)

with col2:
    newbalanceDest = st.number_input("Receiver New Balance", value=3000.0)
    balanceError = st.number_input("Balance Error", value=0.0)
    destBalanceError = st.number_input("Destination Balance Error", value=0.0)

    highValueTransaction = st.selectbox(
        "High Value Transaction",
        [0, 1]
    )

    accountDrained = st.selectbox(
        "Account Drained",
        [0, 1]
    )

# -------------------------
# PREDICT BUTTON
# -------------------------

if st.button("🔍 Predict Fraud"):

    # Simple demo fraud logic
    if (
        highValueTransaction == 1
        or accountDrained == 1
        or balanceError > 1000
        or destBalanceError > 1000
        or amount > 500000
    ):
        prediction = 1
        fraud_probability = 0.89
        risk_level = "HIGH"
    else:
        prediction = 0
        fraud_probability = 0.07
        risk_level = "LOW"

    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Fraud Prediction",
        "🚨 Fraud" if prediction == 1 else "✅ Not Fraud"
    )

    col2.metric(
        "Fraud Probability",
        f"{fraud_probability * 100:.2f}%"
    )

    col3.metric(
        "Risk Level",
        risk_level
    )

# -------------------------
# POWER BI ANALYTICS
# -------------------------

st.markdown("---")

st.subheader("📊 Advanced Analytics")

st.write(
    "Click below to open the detailed analytics dashboard."
)

powerbi_url = "PASTE_YOUR_POWERBI_EMBED_LINK_HERE"

if st.button("Open Power BI Dashboard"):
    st.markdown(
        f"[Click here to view Power BI Dashboard]({powerbi_url})",
        unsafe_allow_html=True
    )
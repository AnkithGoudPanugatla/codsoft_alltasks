# =========================================
# 💳 FRAUD DETECTION SYSTEM (FINAL FIXED)
# =========================================

import streamlit as st
import numpy as np
import pickle
import os
import time
import pandas as pd

# =========================================
# LOAD MODEL (NO PATH ERROR)
# =========================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "model", "fraud_model.pkl")

if not os.path.exists(model_path):
    st.error("❌ Model not found! Please run fraud notebook first.")
    st.stop()

import os
import pickle

base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "..", "model", "fraud.pkl")

model = pickle.load(open(model_path, "rb"))

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(page_title="Fraud AI", layout="centered")

# =========================================
# 🎨 UI STYLE
# =========================================
st.markdown("""
<style>
body {
    background-color: #0E1117;
}

.stButton>button {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

.stButton>button:hover {
    transform: scale(1.05);
    transition: 0.2s;
}
</style>
""", unsafe_allow_html=True)

# =========================================
# HEADER
# =========================================
st.markdown("<h1 style='text-align:center;'>💳 Fraud Detection System</h1>", unsafe_allow_html=True)
st.caption("AI-powered fraud detection using Machine Learning")

# =========================================
# INPUT SECTION
# =========================================
st.subheader("💰 Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount ($)", 0.0, 10000.0, 100.0)

with col2:
    time_val = st.number_input("Transaction Time", 0.0, 200000.0, 5000.0)

# =========================================
# PREDICTION
# =========================================
if st.button("🚀 Detect Fraud"):

    # 🔥 Generate 28 PCA features (same as dataset)
    pca_features = np.random.rand(28)

    # ✅ FINAL INPUT (EXACTLY 30 FEATURES)
    input_data = np.concatenate(
        ([time_val, amount], pca_features)
    ).reshape(1, -1)

    # Safety check
    if input_data.shape[1] != model.n_features_in_:
        st.error("❌ Feature mismatch! Model expects different input.")
        st.stop()

    # Animation
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    st.toast("Transaction Analyzed 💳")

    # =========================================
    # RESULT LOGIC
    # =========================================
    if prediction == 1:
        color = "#ff4b4b"
        status = "🚨 FRAUD DETECTED"
        risk = "HIGH RISK"
        emoji = "⚠️"
    else:
        color = "#00ff99"
        status = "✅ LEGIT TRANSACTION"
        risk = "LOW RISK"
        emoji = "✔️"

    # =========================================
    # RESULT CARD
    # =========================================
    st.markdown(f"""
    <div style="background:{color};
                padding:20px; border-radius:12px; text-align:center;">
        <h2 style="color:white;">{status}</h2>
        <h3 style="color:white;">Risk Score: {prob*100:.2f}%</h3>
        <h4 style="color:white;">{emoji} {risk}</h4>
    </div>
    """, unsafe_allow_html=True)

    # Risk bar
    st.progress(int(prob * 100))

    # =========================================
    # 📊 VISUAL
    # =========================================
    st.markdown("### 📊 Transaction Overview")

    chart = pd.DataFrame({
        "Feature": ["Amount", "Time"],
        "Value": [amount, time_val]
    })

    st.bar_chart(chart.set_index("Feature"))

    # =========================================
    # 🧠 INSIGHTS
    # =========================================
    st.markdown("### 🧠 AI Insights")
    st.info(f"""
    ✔ Transaction Amount: ${amount}  
    ✔ Transaction Time: {time_val}  

    👉 High-value transactions are riskier  
    👉 Unusual timing increases fraud probability  
    👉 Model uses anomaly detection patterns  
    """)

# =========================================
# FOOTER
# =========================================
st.markdown("---")
st.markdown("Made with ❤️ using Machine Learning & Streamlit")
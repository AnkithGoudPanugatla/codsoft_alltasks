# =========================================
# 📊 SALES PREDICTOR (TOP 0.1% FINAL)
# =========================================

import streamlit as st
import numpy as np
import pickle
import os
import time
import pandas as pd

# =========================================
# PATH FIX
# =========================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "model", "sales_model.pkl")

model = pickle.load(open("model/sales.pkl", "rb"))

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(page_title="Sales AI", layout="centered")

# =========================================
# 🎨 PREMIUM CSS
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
st.markdown("<h1 style='text-align:center;'>📊 Sales Predictor</h1>", unsafe_allow_html=True)
st.caption("Predict product sales using advertising data")

# =========================================
# INPUT SECTION
# =========================================
st.subheader("💰 Advertising Budget")

col1, col2, col3 = st.columns(3)

with col1:
    tv = st.slider("TV Budget", 0, 300, 150)

with col2:
    radio = st.slider("Radio Budget", 0, 50, 25)

with col3:
    newspaper = st.slider("Newspaper Budget", 0, 100, 30)

# =========================================
# PREDICTION
# =========================================
if st.button("🚀 Predict Sales"):

    input_data = np.array([[tv, radio, newspaper]])

    # 🔥 Loading Animation
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    prediction = model.predict(input_data)[0]

    st.toast("Prediction Complete 📊")

    # =========================================
    # 🎯 PERFORMANCE TAG
    # =========================================
    if prediction > 20:
        tag = "🔥 High Sales"
        color = "#00ff99"
        emoji = "💰"
    elif prediction > 10:
        tag = "📈 متوسط Sales"
        color = "#00c6ff"
        emoji = "📊"
    else:
        tag = "⚠️ Low Sales"
        color = "#ff4b4b"
        emoji = "📉"

    st.markdown(f"### {tag}")

    # =========================================
    # 💰 RESULT CARD
    # =========================================
    st.markdown(f"""
    <div style="background:{color};
                padding:20px; border-radius:12px; text-align:center;">
        <h2 style="color:white;">{emoji} Sales: {prediction:.2f} units</h2>
    </div>
    """, unsafe_allow_html=True)

    # =========================================
    # 📊 PROGRESS
    # =========================================
    st.progress(int(min(prediction * 4, 100)))

    # =========================================
    # 📊 BUDGET VISUAL
    # =========================================
    st.markdown("### 📊 Budget Distribution")
    chart_data = pd.DataFrame({
        "Channel": ["TV", "Radio", "Newspaper"],
        "Budget": [tv, radio, newspaper]
    })
    st.bar_chart(chart_data.set_index("Channel"))

    # =========================================
    # 🧠 FEATURE IMPORTANCE
    # =========================================
    st.markdown("### 🧠 Feature Importance")

    importance = model.feature_importances_

    feat_df = pd.DataFrame({
        "Feature": ["TV", "Radio", "Newspaper"],
        "Importance": importance
    })

    st.bar_chart(feat_df.set_index("Feature"))

    # =========================================
    # 📈 INSIGHTS
    # =========================================
    st.markdown("### 🧠 Insights")
    st.info(f"""
    ✔ TV Budget: {tv}  
    ✔ Radio Budget: {radio}  
    ✔ Newspaper Budget: {newspaper}  

    👉 Increasing TV & Radio ads boosts sales significantly  
    👉 Newspaper has lower impact compared to others  
    """)

# =========================================
# FOOTER
# =========================================
st.markdown("---")
st.markdown("Made with ❤️ using Machine Learning & Streamlit")
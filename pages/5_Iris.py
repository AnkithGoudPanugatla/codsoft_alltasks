# =========================================
# 🌸 IRIS CLASSIFIER (FINAL ADVANCED UI)
# =========================================

import streamlit as st
import numpy as np
import pickle
import os
import time
import pandas as pd

# =========================================
# LOAD MODEL
# =========================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "model", "iris_model.pkl")

model = pickle.load(open("model/iris.pkl", "rb"))

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(page_title="Iris AI", layout="centered")

# =========================================
# UI STYLE
# =========================================
st.markdown("""
<style>
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
}
.stButton>button:hover {
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# =========================================
# HEADER
# =========================================
st.markdown("<h1 style='text-align:center;'>🌸 Iris Flower Classifier</h1>", unsafe_allow_html=True)
st.caption("Classify iris species using ML")

# =========================================
# INPUTS
# =========================================
st.subheader("📏 Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.5)
    sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)

with col2:
    petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
    petal_width = st.slider("Petal Width", 0.1, 2.5, 1.2)

# =========================================
# PREDICTION
# =========================================
if st.button("🚀 Classify Flower"):

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Animation
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    pred = model.predict(input_data)[0]
    probs = model.predict_proba(input_data)[0]

    species_map = {
        0: "🌸 Setosa",
        1: "🌼 Versicolor",
        2: "🌺 Virginica"
    }

    result = species_map[pred]

    st.toast("Prediction Complete 🌸")

    # =========================================
    # RESULT CARD
    # =========================================
    st.markdown(f"""
    <div style="background:#00c6ff;
                padding:20px; border-radius:12px; text-align:center;">
        <h2 style="color:white;">{result}</h2>
    </div>
    """, unsafe_allow_html=True)

    # =========================================
    # PROBABILITY CHART
    # =========================================
    st.markdown("### 📊 Prediction Confidence")

    prob_df = pd.DataFrame({
        "Species": ["Setosa", "Versicolor", "Virginica"],
        "Probability": probs
    })

    st.bar_chart(prob_df.set_index("Species"))

    # =========================================
    # INSIGHTS
    # =========================================
    st.markdown("### 🧠 Insights")
    st.info("""
    ✔ Petal length is the strongest indicator  
    ✔ Setosa is easiest to classify  
    ✔ Virginica & Versicolor are similar  
    """)

# =========================================
# FOOTER
# =========================================
st.markdown("---")
st.markdown("Made with ❤️ using Machine Learning & Streamlit")
# =========================================
# 🚢 TITANIC AI PREDICTOR (TOP 0.1% FINAL)
# =========================================

import streamlit as st
import numpy as np
import pickle
import time

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(page_title="Titanic AI", layout="centered")

# =========================================
# 🎨 PREMIUM CSS
# =========================================
st.markdown("""
<style>
body {
    background-color: #0E1117;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #bbbbbb;
    margin-bottom: 25px;
}

.card {
    background: #1c1f26;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.5);
    margin-bottom: 20px;
}

.stButton>button {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
    font-size: 18px;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD MODEL
# =========================================
model = pickle.load(open("model/titanic.pkl", "rb"))

# =========================================
# HEADER
# =========================================
st.markdown('<div class="title">🚢 Titanic AI Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced Machine Learning Survival Prediction</div>', unsafe_allow_html=True)

# =========================================
# PREMIUM IMAGE
# =========================================
st.markdown("""
<div style="display:flex; justify-content:center;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg"
    style="border-radius:15px; box-shadow:0px 0px 20px rgba(255,255,255,0.2); width:85%;">
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================
# INPUT SECTION
# =========================================
st.markdown("### 🧾 Enter Passenger Details")

col1, col2 = st.columns(2)

with col1:
    Pclass = st.selectbox("Passenger Class", [1, 2, 3])
    Sex = st.selectbox("Sex", ["Male", "Female"])
    Age = st.slider("Age", 1, 80, 25)

with col2:
    Fare = st.slider("Fare", 0, 500, 50)
    Embarked = st.selectbox("Embarked", ["C", "Q", "S"])
    SibSp = st.number_input("Siblings/Spouses", 0, 10, 0)
    Parch = st.number_input("Parents/Children", 0, 10, 0)

# =========================================
# FEATURE ENGINEERING
# =========================================
Sex_val = 1 if Sex == "Male" else 0
Embarked_val = {"C": 0, "Q": 1, "S": 2}[Embarked]

FamilySize = SibSp + Parch + 1
IsAlone = 1 if FamilySize == 1 else 0

if Age <= 12:
    AgeGroup = 0
elif Age <= 20:
    AgeGroup = 1
elif Age <= 40:
    AgeGroup = 2
elif Age <= 60:
    AgeGroup = 3
else:
    AgeGroup = 4

if Fare <= 7.91:
    FareBand = 0
elif Fare <= 14.454:
    FareBand = 1
elif Fare <= 31:
    FareBand = 2
else:
    FareBand = 3

Title = 2

st.markdown("<br>", unsafe_allow_html=True)

# =========================================
# PREDICTION BUTTON
# =========================================
if st.button("🚀 Predict Survival"):

    input_data = np.array([[Pclass, Sex_val, Age, SibSp, Parch, Fare,
                            Embarked_val, Title, FamilySize,
                            IsAlone, AgeGroup, FareBand]])

    # =========================================
    # 🔥 SMOOTH LOADING ANIMATION
    # =========================================
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]

    st.toast("Prediction Completed 🚀")

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================================
    # 🎯 PREMIUM RESULT CARD
    # =========================================
    if prediction[0] == 1:
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #00c6ff, #0072ff);
                    padding:20px; border-radius:12px; text-align:center;">
            <h2 style="color:white;">🎉 SURVIVED</h2>
            <h3 style="color:white;">Confidence: {prob*100:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(90deg, #ff416c, #ff4b2b);
                    padding:20px; border-radius:12px; text-align:center;">
            <h2 style="color:white;">❌ NOT SURVIVED</h2>
            <h3 style="color:white;">Confidence: {(1-prob)*100:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)

    # =========================================
    # 📊 PROGRESS + METRIC
    # =========================================
    st.progress(int(prob * 100))
    st.metric("Survival Probability", f"{prob*100:.1f}%")

    # =========================================
    # 🧠 INSIGHTS
    # =========================================
    st.markdown("### 🧠 AI Insights")
    st.info("""
    ✔ Women had higher survival rates  
    ✔ First-class passengers were prioritized  
    ✔ Children had better chances  
    ✔ Fare reflects socio-economic advantage  
    """)
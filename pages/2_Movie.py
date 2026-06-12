# =========================================
# 🎬 MOVIE RATING PREDICTOR (FINAL UI)
# =========================================

import streamlit as st
import numpy as np
import pickle
import time

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(page_title="Movie AI", layout="centered")

# =========================================
# LOAD MODEL
# =========================================
model = pickle.load(open("model/movie.pkl", "rb"))
import os
import pickle

base_path = os.path.dirname(__file__)
vectorizer_path = os.path.join(base_path, "..", "model", "vectorizer.pkl")

vectorizer = pickle.load(open(vectorizer_path, "rb"))

# =========================================
# HEADER
# =========================================
st.markdown("<h1 style='text-align:center;'>🎬 Movie Rating Predictor</h1>", unsafe_allow_html=True)
st.caption("Predict movie ratings using AI + NLP")

# =========================================
# INPUT SECTION
# =========================================
st.subheader("🧾 Enter Movie Details")

col1, col2 = st.columns(2)

with col1:
    genre = st.text_input("Genre", "Action Drama")
    director = st.text_input("Director", "Christopher Nolan")
    actor1 = st.text_input("Actor 1", "Actor A")

with col2:
    actor2 = st.text_input("Actor 2", "Actor B")
    actor3 = st.text_input("Actor 3", "Actor C")
    votes = st.number_input("Votes", 0, 10000, 1000)
    duration = st.number_input("Duration (min)", 0, 300, 120)

# =========================================
# PREDICTION
# =========================================
if st.button("🚀 Predict Rating"):

    combined = f"{genre} {director} {actor1} {actor2} {actor3}"

    text_data = vectorizer.transform([combined]).toarray()
    final_input = np.hstack((text_data, [[votes, duration]]))

    # 🔥 Animation
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    rating = model.predict(final_input)[0]

    st.toast("Prediction Completed 🎬")

    # =========================================
    # RESULT
    # =========================================
    st.markdown("### 🎯 Predicted Rating")

    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #00c6ff, #0072ff);
                padding:20px; border-radius:12px; text-align:center;">
        <h2 style="color:white;">⭐ {rating:.2f} / 10</h2>
    </div>
    """, unsafe_allow_html=True)

    st.progress(int(rating * 10))

    # =========================================
    # INSIGHTS
    # =========================================
    st.markdown("### 🧠 Insights")
    st.info("""
    ✔ Director strongly influences rating  
    ✔ Popular actors increase engagement  
    ✔ Genre combinations affect audience reception  
    ✔ Higher votes = more reliable rating  
    """)
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="AI Prediction Suite", layout="wide")

# ===============================
# LOAD MODELS
# ===============================
@st.cache_resource
def load_models():
    return {
        "titanic": pickle.load(open("model/titanic.pkl", "rb")),
        "movie": pickle.load(open("model/movie.pkl", "rb")),
        "sales": pickle.load(open("model/sales.pkl", "rb")),
        "fraud": pickle.load(open("model/fraud.pkl", "rb")),
        "iris": pickle.load(open("model/iris.pkl", "rb")),
    }

models = load_models()

# ===============================
# STYLE
# ===============================
st.markdown("""
<style>
.big-title {
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:#00c6ff;
}
.card {
    padding:20px;
    border-radius:12px;
    background:rgba(255,255,255,0.05);
}
</style>
""", unsafe_allow_html=True)

# ===============================
# SIDEBAR NAV
# ===============================
st.sidebar.title("🚀 AI Suite")
task = st.sidebar.radio("Select Task", [
    "🏠 Home",
    "🚢 Titanic",
    "🎬 Movie",
    "📊 Sales",
    "💳 Fraud",
    "🌸 Iris"
])

# ===============================
# HOME
# ===============================
if task == "🏠 Home":
    st.markdown('<div class="big-title">AI Prediction Suite</div>', unsafe_allow_html=True)
    st.write("Multi-Model ML Application")

    col1, col2 = st.columns(2)
    col1.info("🚢 Titanic Survival Prediction")
    col1.info("🎬 Movie Rating Prediction")
    col2.info("📊 Sales Forecasting")
    col2.info("💳 Fraud Detection + 🌸 Iris Classification")

# ===============================
# TITANIC
# ===============================
elif task == "🚢 Titanic":
    st.header("🚢 Titanic Survival Prediction")

    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Class", [1,2,3])
        sex = st.selectbox("Sex", ["male","female"])
        age = st.slider("Age", 1, 80, 25)

    with col2:
        fare = st.slider("Fare", 0, 500, 50)
        embarked = st.selectbox("Embarked", ["C","Q","S"])

    if st.button("Predict"):
        sex = 0 if sex=="male" else 1
        embarked = {"C":0,"Q":1,"S":2}[embarked]

        pred = models["titanic"].predict([[pclass, sex, age, fare, embarked]])

        if pred[0] == 1:
            st.success("Survived ✅")
        else:
            st.error("Not Survived ❌")

# ===============================
# MOVIE
# ===============================
elif task == "🎬 Movie":
    st.header("🎬 Movie Rating Prediction")

    genre = st.text_input("Genre", "Action Drama")
    director = st.text_input("Director", "Nolan")
    actor1 = st.text_input("Actor 1", "Actor A")
    actor2 = st.text_input("Actor 2", "Actor B")
    actor3 = st.text_input("Actor 3", "Actor C")

    if st.button("Predict Rating"):
        text = f"{genre} {director} {actor1} {actor2} {actor3}"
        pred = models["movie"].predict([text])[0]

        st.success(f"⭐ Rating: {round(pred,2)} / 10")

# ===============================
# SALES
# ===============================
elif task == "📊 Sales":
    st.header("📊 Sales Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        tv = st.slider("TV Ads", 0, 300, 100)
    with col2:
        radio = st.slider("Radio Ads", 0, 100, 25)
    with col3:
        news = st.slider("Newspaper Ads", 0, 100, 30)

    if st.button("Predict Sales"):
        pred = models["sales"].predict([[tv, radio, news]])[0]
        st.success(f"📈 Sales: {round(pred,2)}")

# ===============================
# FRAUD
# ===============================
elif task == "💳 Fraud":
    st.header("💳 Fraud Detection")

    amount = st.number_input("Transaction Amount", 0.0, 10000.0, 100.0)

    if st.button("Check Fraud"):
        pred = models["fraud"].predict([[amount]])[0]

        if pred == 1:
            st.error("Fraudulent Transaction 🚨")
        else:
            st.success("Legit Transaction ✅")

# ===============================
# IRIS
# ===============================
elif task == "🌸 Iris":
    st.header("🌸 Iris Classification")

    col1, col2 = st.columns(2)

    with col1:
        sl = st.slider("Sepal Length", 4.0, 8.0, 5.1)
        sw = st.slider("Sepal Width", 2.0, 4.5, 3.5)

    with col2:
        pl = st.slider("Petal Length", 1.0, 7.0, 1.4)
        pw = st.slider("Petal Width", 0.1, 2.5, 0.2)

    if st.button("Predict Species"):
        pred = models["iris"].predict([[sl, sw, pl, pw]])[0]

        names = ["Setosa", "Versicolor", "Virginica"]
        st.success(f"🌼 {names[pred]}")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.caption("Made with ❤️ by Ankith | AI Suite")
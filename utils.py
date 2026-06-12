from pathlib import Path
import pickle
import streamlit as st


def project_root() -> Path:
    return Path(__file__).resolve().parent


def load_model(filename: str = "model.pkl"):
    model_path = project_root() / "model" / filename
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Model not found: {model_path}. Please ensure the model exists.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def inject_styles():
    st.markdown(
        """
    <style>
    /* Dark theme base */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    :root{--bg:#071019;--card:#0f1720;--muted:#9aa4b2;--accent1:#00c6ff;--accent2:#6b46ff}
    html,body,header,#root{background:var(--bg)!important;color:#e6eef6;font-family:Inter, Arial, sans-serif}
    .saaS-title{font-size:34px;font-weight:700;margin-bottom:4px;background:linear-gradient(90deg,var(--accent1),var(--accent2));-webkit-background-clip:text;background-clip:text;color:transparent}
    .saaS-sub{color:var(--muted);margin-bottom:18px}
    .card{background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));padding:18px;border-radius:12px;border:1px solid rgba(255,255,255,0.03);transition:transform .18s ease, box-shadow .18s ease}
    .card:hover{transform:translateY(-6px);box-shadow:0 12px 40px rgba(2,6,23,0.7)}
    .card-title{font-size:18px;font-weight:600}
    .card-desc{color:var(--muted);font-size:13px;margin-top:6px}
    .result-card{background:linear-gradient(90deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));padding:16px;border-radius:10px;border:1px solid rgba(255,255,255,0.04)}
    .metric-value{font-size:20px;font-weight:700}
    .muted{color:var(--muted)}
    .footer{color:var(--muted);font-size:13px;padding:24px 0;text-align:center}
    .large-btn>button{height:44px;border-radius:10px}
    </style>
    """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = ""):
    st.markdown(f"<div class='saaS-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='saaS-sub'>{subtitle}</div>", unsafe_allow_html=True)

# AI Prediction Suite

Professional multi-page Streamlit app.

## Structure
- `app.py` — Dashboard (overview only, no navigation buttons)
- `pages/` — Streamlit pages (1_Titanic.py, 2_Movie.py, 3_Sales.py, 4_Fraud.py, 5_Iris.py, _template.py)
- `model/` — Store model artifacts here (e.g. `model.pkl`, `iris_model.pkl`, `movie_model.pkl`)
- `data/` — Optional CSV data used by notebooks/pages
- `utils.py` — Shared utilities, styling, model loader

## Guidelines
- Use Streamlit native multi-page navigation: pages in `pages/` are auto-discovered and shown in the left sidebar.
- Do NOT use `st.switch_page` or page-level navigation buttons. Sidebar handles navigation.
- Place model files in `model/` at project root. Filenames used by pages:
  - Titanic: `model.pkl`
  - Iris: `iris_model.pkl`
  - Movie: `movie_model.pkl`
  - Sales: `sales_model.pkl`
  - Fraud: `fraud_model.pkl`

## Run locally
```bash
cd Titanic_Project
python -m venv venv  # optional
# activate venv
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Pages show friendly errors if model files are missing.
- Use `pages/_template.py` as a template for adding new modules with consistent header and styles.

If you want, I can now wire real models into Movie/Sales/Fraud pages, or create example dummy models for demo purposes.

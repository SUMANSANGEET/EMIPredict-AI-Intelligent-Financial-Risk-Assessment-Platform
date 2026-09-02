import streamlit as st
from utils import inject_base_style, load_artifacts, fetch_applications

st.set_page_config(
    page_title="EMIPredict AI | Intelligent Financial Risk Assessment Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_base_style()

artifacts = load_artifacts()
meta = artifacts.get("metadata", {})

st.markdown(
    """
    <h1 style='margin-bottom:0;'>💳 EMIPredict AI</h1>
    <p style='color:#94a3b8; font-size:1.05rem; margin-top:0.2rem;'>
    Intelligent Financial Risk Assessment Platform &nbsp;·&nbsp;
    EMI Eligibility Classification + Maximum Monthly EMI Regression
    </p>
    """,
    unsafe_allow_html=True,
)

if not artifacts["ready"]:
    st.warning(
        "⚠️ Trained model artifacts weren't found in `models/`. "
        "Run the modelling notebook first (it saves `best_classifier.pkl`, "
        "`best_regressor.pkl`, `label_encoders.pkl`, `scaler.pkl`, `feature_cols.pkl`, "
        "`clf_target_encoder.pkl`, `metadata.json`) and place them in this app's "
        "`models/` folder. The Real-Time Prediction and MLflow pages need them."
    )

st.divider()

# ---- KPI row -----------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
clf_metrics = meta.get("classifier_test_metrics", {})
reg_metrics = meta.get("regressor_test_metrics", {})
apps_df = fetch_applications()

with col1:
    st.markdown(
        f"<div class='metric-card'><h3>Classifier Accuracy</h3>"
        f"<p>{clf_metrics.get('accuracy', 0)*100:.1f}%</p></div>",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"<div class='metric-card'><h3>Regression RMSE</h3>"
        f"<p>₹{reg_metrics.get('rmse', 0):,.0f}</p></div>",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"<div class='metric-card'><h3>Best Classifier</h3>"
        f"<p style='font-size:1.1rem;'>{meta.get('best_classifier', '—')}</p></div>",
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f"<div class='metric-card'><h3>Applications Logged</h3>"
        f"<p>{len(apps_df)}</p></div>",
        unsafe_allow_html=True,
    )

st.write("")
st.subheader("Platform Overview")

c1, c2 = st.columns([1.4, 1])
with c1:
    st.markdown(
        """
Financial institutions and lending platforms need to assess a customer's EMI
repayment capacity and financial risk **before** approving loans or EMI-based
purchases. EMIPredict AI automates this end-to-end using two champion models
trained on 400,000 applicant records across five EMI scenarios
(E-commerce Shopping, Home Appliances, Vehicle, Personal Loan, Education):

- **Classification** — `Eligible` / `High_Risk` / `Not_Eligible`
- **Regression** — maximum affordable monthly EMI (INR)

**What this app demonstrates:**
- Real-time single-applicant scoring against production-style champion models
- Interactive exploratory analytics over the full applicant population
- MLflow experiment tracking / model comparison across 3 algorithms per task
- An admin review workflow (CRUD) for underwriters to action model output
        """
    )
with c2:
    st.markdown("**Tech Stack**")
    for tag in [
        "Python", "scikit-learn", "XGBoost", "MLflow", "Streamlit",
        "Plotly", "SQLite", "Pandas", "Streamlit Cloud",
    ]:
        st.markdown(f"<span class='badge'>{tag}</span>", unsafe_allow_html=True)

    st.write("")
    st.markdown("**Navigate using the sidebar →**")
    st.page_link("pages/1_Real_Time_Prediction.py", label="🔮 Real-Time Prediction")
    st.page_link("pages/2_Data_Explorer.py", label="📊 Data Explorer")
    st.page_link(
    "pages/3_MLflow_Monitoring.py",
    label="📈 MLflow Model Monitoring"
)
    st.page_link(
    "pages/4_Admin_CRUD.py",
    label="🔐 Admin (CRUD)"
)

st.divider()
st.caption(
    "EMIPredict AI · INNOVEXIS · Data Science & Gen AI · Capstone Project · "
    "Built by P Suman Sangeet"
)

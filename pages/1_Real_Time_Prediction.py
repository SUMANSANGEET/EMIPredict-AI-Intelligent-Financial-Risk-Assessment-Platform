import plotly.graph_objects as go
import streamlit as st
from utils import (
    CAT_OPTIONS, RAW_NUM_FEATURES, inject_base_style,
    insert_application, load_artifacts, predict_application,
)

st.set_page_config(page_title="Real-Time Prediction | EMIPredict AI", page_icon="🔮", layout="wide")
inject_base_style()
st.title("🔮 Real-Time EMI Eligibility Prediction")
st.caption("Enter an applicant's profile to instantly score eligibility and safe EMI capacity.")

artifacts = load_artifacts()
if not artifacts["ready"]:
    st.error(
        "Model artifacts missing from `models/`. Run the notebook to generate them, "
        f"then add: {', '.join(artifacts['missing'])}"
    )
    st.stop()

with st.form("prediction_form"):
    st.subheader("Applicant Profile")

    name_col, scenario_col = st.columns(2)
    applicant_name = name_col.text_input("Applicant Name (optional)", "")
    emi_scenario = scenario_col.selectbox("EMI Scenario", CAT_OPTIONS["emi_scenario"])

    st.markdown("**Demographics**")
    d1, d2, d3, d4 = st.columns(4)
    age = d1.number_input("Age", 18, 75, 32)
    gender = d2.selectbox("Gender", CAT_OPTIONS["gender"])
    marital_status = d3.selectbox("Marital Status", CAT_OPTIONS["marital_status"])
    education = d4.selectbox("Education", CAT_OPTIONS["education"])

    st.markdown("**Employment & Housing**")
    e1, e2, e3, e4 = st.columns(4)
    employment_type = e1.selectbox("Employment Type", CAT_OPTIONS["employment_type"])
    company_type = e2.selectbox("Company Type", CAT_OPTIONS["company_type"])
    years_of_employment = e3.number_input("Years of Employment", 0.0, 45.0, 5.0)
    house_type = e4.selectbox("House Type", CAT_OPTIONS["house_type"])

    st.markdown("**Income & Savings**")
    i1, i2, i3, i4 = st.columns(4)
    monthly_salary = i1.number_input("Monthly Salary (₹)", 5000, 2_000_000, 55000, step=1000)
    bank_balance = i2.number_input("Bank Balance (₹)", 0, 10_000_000, 120000, step=1000)
    emergency_fund = i3.number_input("Emergency Fund (₹)", 0, 5_000_000, 60000, step=1000)
    credit_score = i4.slider("Credit Score", 300, 850, 720)

    st.markdown("**Household & Expenses**")
    h1, h2, h3, h4 = st.columns(4)
    family_size = h1.number_input("Family Size", 1, 15, 3)
    dependents = h2.number_input("Dependents", 0, 10, 1)
    monthly_rent = h3.number_input("Monthly Rent (₹)", 0, 500000, 12000, step=500)
    existing_loans = h4.selectbox("Existing Loans", CAT_OPTIONS["existing_loans"])

    x1, x2, x3, x4 = st.columns(4)
    school_fees = x1.number_input("School Fees (₹/mo)", 0, 200000, 0, step=500)
    college_fees = x2.number_input("College Fees (₹/mo)", 0, 200000, 0, step=500)
    travel_expenses = x3.number_input("Travel Expenses (₹/mo)", 0, 100000, 3000, step=500)
    groceries_utilities = x4.number_input("Groceries & Utilities (₹/mo)", 0, 200000, 8000, step=500)

    o1, o2, o3, o4 = st.columns(4)
    other_monthly_expenses = o1.number_input("Other Monthly Expenses (₹)", 0, 200000, 4000, step=500)
    current_emi_amount = o2.number_input("Current EMI Amount (₹)", 0, 500000, 5000, step=500)
    requested_amount = o3.number_input("Requested Amount (₹)", 1000, 10_000_000, 150000, step=1000)
    requested_tenure = o4.number_input("Requested Tenure (months)", 1, 84, 24)

    submitted = st.form_submit_button("🚀 Predict Eligibility", use_container_width=True)

if submitted:
    raw_input = {
        "age": age, "gender": gender, "marital_status": marital_status, "education": education,
        "employment_type": employment_type, "company_type": company_type,
        "years_of_employment": years_of_employment, "house_type": house_type,
        "monthly_salary": monthly_salary, "bank_balance": bank_balance,
        "emergency_fund": emergency_fund, "credit_score": credit_score,
        "family_size": family_size, "dependents": dependents, "monthly_rent": monthly_rent,
        "existing_loans": existing_loans, "school_fees": school_fees, "college_fees": college_fees,
        "travel_expenses": travel_expenses, "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses, "current_emi_amount": current_emi_amount,
        "requested_amount": requested_amount, "requested_tenure": requested_tenure,
        "emi_scenario": emi_scenario,
    }

    with st.spinner("Scoring applicant..."):
        result = predict_application(raw_input, artifacts)

    insert_application({
        "applicant_name": applicant_name or "N/A",
        "emi_scenario": emi_scenario,
        "monthly_salary": monthly_salary,
        "requested_amount": requested_amount,
        "predicted_eligibility": result["eligibility"],
        "predicted_max_emi": result["predicted_max_monthly_emi"],
        "raw_input": raw_input,
    })

    st.divider()
    st.subheader("Prediction Result")

    color_map = {"Eligible": "#2ecc71", "High_Risk": "#f39c12", "Not_Eligible": "#e74c3c"}
    r1, r2 = st.columns([1, 1.3])

    with r1:
        st.markdown(
            f"""
            <div class='metric-card' style='border-left:6px solid {color_map.get(result["eligibility"], "#38bdf8")};'>
            <h3>Eligibility Decision</h3>
            <p style='color:{color_map.get(result["eligibility"], "#f1f5f9")};'>{result["eligibility"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            f"<div class='metric-card'><h3>Max Recommended Monthly EMI</h3>"
            f"<p>₹{result['predicted_max_monthly_emi']:,.0f}</p></div>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.success("✅ Application logged to Admin queue for underwriter review.")

    with r2:
        classes = list(result["class_probabilities"].keys())
        probs = list(result["class_probabilities"].values())
        fig = go.Figure(go.Bar(
            x=probs, y=classes, orientation="h",
            marker_color=[color_map.get(c, "#38bdf8") for c in classes],
            text=[f"{p*100:.1f}%" for p in probs], textposition="outside",
        ))
        fig.update_layout(
            title="Class Probability Breakdown", xaxis_title="Probability",
            xaxis_range=[0, 1], height=280, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

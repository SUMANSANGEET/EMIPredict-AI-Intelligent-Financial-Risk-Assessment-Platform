import streamlit as st
from utils import (
    delete_application, fetch_applications, inject_base_style,
    insert_application, update_application,
)

st.set_page_config(page_title="Admin | EMIPredict AI", page_icon="🔐", layout="wide")
inject_base_style()
st.title("🔐 Admin — Application Review (CRUD)")
st.caption("Underwriters review, override, and manage every scored application.")

# ---- Simple password gate -------------------------------------------------
ADMIN_PASSWORD = "admin123"

if "admin_authed" not in st.session_state:
    st.session_state.admin_authed = False

if not st.session_state.admin_authed:
    pw = st.text_input("Admin password", type="password")
    if st.button("Login"):
        if pw == ADMIN_PASSWORD:
            st.session_state.admin_authed = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.info(
        "Demo credential: `admin123` (override via `.streamlit/secrets.toml` → "
        "`ADMIN_PASSWORD = \"...\"` before deploying)."
    )
    st.stop()

df = fetch_applications()

status_map = {"Eligible": "🟢", "High_Risk": "🟠", "Not_Eligible": "🔴"}
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Applications", len(df))
k2.metric("Pending Review", int((df["status"] == "Pending").sum()) if len(df) else 0)
k3.metric("Approved", int((df["status"] == "Approved").sum()) if len(df) else 0)
k4.metric("Rejected", int((df["status"] == "Rejected").sum()) if len(df) else 0)

st.divider()
tab_view, tab_update, tab_create, tab_delete = st.tabs(
    ["📋 View All", "✏️ Update Status", "➕ Add Manual Entry", "🗑️ Delete"]
)

with tab_view:
    if df.empty:
        st.info("No applications logged yet — score one on the Real-Time Prediction page.")
    else:
        view = df.copy()
        view["predicted_eligibility"] = view["predicted_eligibility"].apply(
            lambda x: f"{status_map.get(x, '')} {x}"
        )
        st.dataframe(
            view[[
                "id", "created_at", "applicant_name", "emi_scenario", "monthly_salary",
                "requested_amount", "predicted_eligibility", "predicted_max_emi",
                "status", "reviewer_notes",
            ]],
            use_container_width=True, height=420,
        )

with tab_update:
    if df.empty:
        st.info("Nothing to update yet.")
    else:
        app_id = st.selectbox("Select application ID", df["id"].tolist())
        row = df[df["id"] == app_id].iloc[0]
        st.write(f"**{row['applicant_name']}** · {row['emi_scenario']} · "
                 f"Predicted: {row['predicted_eligibility']} · ₹{row['predicted_max_emi']:,.0f}")
        new_status = st.selectbox(
            "Underwriter decision", ["Pending", "Approved", "Rejected", "Needs More Info"],
            index=["Pending", "Approved", "Rejected", "Needs More Info"].index(row["status"])
            if row["status"] in ["Pending", "Approved", "Rejected", "Needs More Info"] else 0,
        )
        notes = st.text_area("Reviewer notes", value=row["reviewer_notes"] or "")
        if st.button("Save Update", use_container_width=True):
            update_application(int(app_id), new_status, notes)
            st.success(f"Application #{app_id} updated to '{new_status}'.")
            st.rerun()

with tab_create:
    st.write("Manually log an application (e.g. one scored outside this app).")
    with st.form("manual_entry"):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Applicant Name")
        scenario = c2.selectbox(
            "EMI Scenario",
            ["E-commerce Shopping", "Home Appliances", "Vehicle", "Personal Loan", "Education"],
        )
        eligibility = c3.selectbox("Eligibility", ["Eligible", "High_Risk", "Not_Eligible"])
        c4, c5, c6 = st.columns(3)
        salary = c4.number_input("Monthly Salary (₹)", 0, 2_000_000, 50000, step=1000)
        requested = c5.number_input("Requested Amount (₹)", 0, 10_000_000, 100000, step=1000)
        max_emi = c6.number_input("Predicted Max EMI (₹)", 0, 500000, 5000, step=500)
        notes = st.text_area("Notes")
        if st.form_submit_button("Add Entry", use_container_width=True):
            insert_application({
                "applicant_name": name or "N/A", "emi_scenario": scenario,
                "monthly_salary": salary, "requested_amount": requested,
                "predicted_eligibility": eligibility, "predicted_max_emi": max_emi,
                "status": "Pending", "reviewer_notes": notes, "raw_input": {},
            })
            st.success("Manual entry added.")
            st.rerun()

with tab_delete:
    if df.empty:
        st.info("Nothing to delete yet.")
    else:
        del_id = st.selectbox("Select application ID to delete", df["id"].tolist(), key="del_id")
        st.warning(f"This permanently deletes application #{del_id}.")
        if st.button("Delete Application", type="primary", use_container_width=True):
            delete_application(int(del_id))
            st.success(f"Application #{del_id} deleted.")
            st.rerun()

st.divider()
if st.button("Log out"):
    st.session_state.admin_authed = False
    st.rerun()

"""
EMIPredict AI - shared utilities
Loads the trained artifacts produced by the modelling notebook and exposes
feature engineering + inference helpers used by every Streamlit page.
"""
import json
import os
import sqlite3
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = "models"
DATA_PATH = "emi_prediction_dataset.csv"
ADMIN_DB = "admin_applications.db"
MLFLOW_DB = "mlflow.db"

CAT_FEATURES = [
    "gender", "marital_status", "education", "employment_type",
    "company_type", "house_type", "existing_loans", "emi_scenario",
]

RAW_NUM_FEATURES = [
    "age", "monthly_salary", "years_of_employment", "monthly_rent",
    "family_size", "dependents", "school_fees", "college_fees",
    "travel_expenses", "groceries_utilities", "other_monthly_expenses",
    "current_emi_amount", "credit_score", "bank_balance",
    "emergency_fund", "requested_amount", "requested_tenure",
]

CAT_OPTIONS = {
    "gender": ["Male", "Female"],
    "marital_status": ["Single", "Married", "Divorced", "Widowed"],
    "education": ["High School", "Graduate", "Post Graduate", "Doctorate"],
    "employment_type": ["Salaried", "Self-Employed", "Government", "Business"],
    "company_type": ["Private", "Government", "MNC", "Startup"],
    "house_type": ["Owned", "Rented", "Family Owned"],
    "existing_loans": ["Yes", "No"],
    "emi_scenario": [
        "E-commerce Shopping", "Home Appliances", "Vehicle",
        "Personal Loan", "Education",
    ],
}


# --------------------------------------------------------------------------- #
# Artifact loading
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load every artifact saved by the notebook. Returns a dict; missing
    files are reported so pages can degrade gracefully instead of crashing."""
    paths = {
        "classifier": f"{MODEL_DIR}/best_classifier.pkl",
        "regressor": f"{MODEL_DIR}/best_regressor.pkl",
        "label_encoders": f"{MODEL_DIR}/label_encoders.pkl",
        "scaler": f"{MODEL_DIR}/scaler.pkl",
        "feature_cols": f"{MODEL_DIR}/feature_cols.pkl",
        "clf_target_encoder": f"{MODEL_DIR}/clf_target_encoder.pkl",
    }
    artifacts, missing = {}, []
    for key, path in paths.items():
        if os.path.exists(path):
            artifacts[key] = joblib.load(path)
        else:
            missing.append(path)

    meta_path = f"{MODEL_DIR}/metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            artifacts["metadata"] = json.load(f)
    else:
        missing.append(meta_path)

    artifacts["missing"] = missing
    artifacts["ready"] = len(missing) == 0
    return artifacts


@st.cache_data(show_spinner=False)
def load_dataset(path: str = DATA_PATH):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Feature engineering (mirrors the notebook's engineer_features exactly)
# --------------------------------------------------------------------------- #
def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    total_expenses = (
        d["monthly_rent"] + d["school_fees"] + d["college_fees"] + d["travel_expenses"]
        + d["groceries_utilities"] + d["other_monthly_expenses"] + d["current_emi_amount"]
    )
    d["total_monthly_expenses"] = total_expenses
    d["debt_to_income_ratio"] = d["current_emi_amount"] / d["monthly_salary"].replace(0, np.nan)
    d["expense_to_income_ratio"] = total_expenses / d["monthly_salary"].replace(0, np.nan)
    d["disposable_income"] = d["monthly_salary"] - total_expenses
    d["affordability_ratio"] = d["disposable_income"] / d["monthly_salary"].replace(0, np.nan)
    d["savings_rate"] = d["bank_balance"] / d["monthly_salary"].replace(0, np.nan)
    d["emergency_coverage_months"] = d["emergency_fund"] / total_expenses.replace(0, np.nan)
    d["requested_to_income_ratio"] = d["requested_amount"] / (
        d["monthly_salary"] * d["requested_tenure"]
    ).replace(0, np.nan)
    d["dependents_per_earner"] = d["dependents"] / 1.0
    d["employment_stability_score"] = d["years_of_employment"] * (
        d["employment_type"] == "Government"
    ).map({True: 1.3, False: 1.0})
    d["credit_risk_score"] = (d["credit_score"] - 300) / (850 - 300)
    d = d.replace([np.inf, -np.inf], np.nan)

    fe_cols = [
        "total_monthly_expenses", "debt_to_income_ratio", "expense_to_income_ratio",
        "disposable_income", "affordability_ratio", "savings_rate",
        "emergency_coverage_months", "requested_to_income_ratio",
        "dependents_per_earner", "employment_stability_score", "credit_risk_score",
    ]
    for c in fe_cols:
        d[c] = d[c].fillna(d[c].median() if d[c].notna().any() else 0)
    return d


def predict_application(raw_input: dict, artifacts: dict) -> dict:
    """Run one raw applicant record through feature engineering, encoding,
    scaling and both champion models. Returns eligibility + EMI prediction."""
    df = pd.DataFrame([raw_input])
    df_fe = engineer_features(df)

    encoders = artifacts["label_encoders"]
    for c in CAT_FEATURES:
        mapping = {label: idx for idx, label in enumerate(encoders[c].classes_)}
        df_fe[c + "_enc"] = df_fe[c].astype(str).map(mapping).fillna(-1).astype(int)

    feature_cols = artifacts["feature_cols"]  # pre-encoding/scaling column order
    scaler = artifacts["scaler"]
    X = df_fe[feature_cols]
    X_scaled = scaler.transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=[c for c in X.columns])

    clf = artifacts["classifier"]
    reg = artifacts["regressor"]
    clf_target_encoder = artifacts["clf_target_encoder"]

    proba = clf.predict_proba(X_scaled)[0]
    pred_class_idx = int(np.argmax(proba))
    eligibility = clf_target_encoder.classes_[pred_class_idx]
    class_probs = dict(zip(clf_target_encoder.classes_, proba.round(4)))

    max_emi = float(reg.predict(X_scaled)[0])

    return {
        "eligibility": eligibility,
        "class_probabilities": class_probs,
        "predicted_max_monthly_emi": round(max_emi, 2),
    }


# --------------------------------------------------------------------------- #
# Admin CRUD - SQLite persistence for reviewed applications
# --------------------------------------------------------------------------- #
def get_db_connection():
    conn = sqlite3.connect(ADMIN_DB, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            applicant_name TEXT,
            emi_scenario TEXT,
            monthly_salary REAL,
            requested_amount REAL,
            predicted_eligibility TEXT,
            predicted_max_emi REAL,
            status TEXT DEFAULT 'Pending',
            reviewer_notes TEXT,
            raw_input TEXT
        )
        """
    )
    conn.commit()
    return conn


def insert_application(record: dict):
    conn = get_db_connection()
    conn.execute(
        """INSERT INTO applications
           (created_at, applicant_name, emi_scenario, monthly_salary, requested_amount,
            predicted_eligibility, predicted_max_emi, status, reviewer_notes, raw_input)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().isoformat(timespec="seconds"),
            record.get("applicant_name", "N/A"),
            record.get("emi_scenario"),
            record.get("monthly_salary"),
            record.get("requested_amount"),
            record.get("predicted_eligibility"),
            record.get("predicted_max_emi"),
            record.get("status", "Pending"),
            record.get("reviewer_notes", ""),
            json.dumps(record.get("raw_input", {})),
        ),
    )
    conn.commit()
    conn.close()


def fetch_applications() -> pd.DataFrame:
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM applications ORDER BY id DESC", conn)
    conn.close()
    return df


def update_application(app_id: int, status: str, reviewer_notes: str):
    conn = get_db_connection()
    conn.execute(
        "UPDATE applications SET status = ?, reviewer_notes = ? WHERE id = ?",
        (status, reviewer_notes, app_id),
    )
    conn.commit()
    conn.close()


def delete_application(app_id: int):
    conn = get_db_connection()
    conn.execute("DELETE FROM applications WHERE id = ?", (app_id,))
    conn.commit()
    conn.close()


# --------------------------------------------------------------------------- #
# Shared UI helpers
# --------------------------------------------------------------------------- #
def inject_base_style():
    st.markdown(
        """
        <style>
        .metric-card {
            background: linear-gradient(135deg,#0f172a,#1e293b);
            padding: 1.1rem 1.3rem; border-radius: 14px; color: #f1f5f9;
            border: 1px solid #334155;
        }
        .metric-card h3 { margin:0; font-size:0.85rem; font-weight:500; color:#94a3b8; }
        .metric-card p { margin:0.15rem 0 0 0; font-size:1.6rem; font-weight:700; }
        .badge {
            display:inline-block; padding:0.25rem 0.7rem; border-radius:999px;
            background:#1e293b; color:#38bdf8; font-size:0.78rem; margin:0.15rem;
            border:1px solid #334155;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

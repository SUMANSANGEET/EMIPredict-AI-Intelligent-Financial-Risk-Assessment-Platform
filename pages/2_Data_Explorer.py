import pandas as pd
import plotly.express as px
import streamlit as st
from utils import inject_base_style, load_dataset

st.set_page_config(
    page_title="Data Explorer | EMIPredict AI",
    page_icon="📊",
    layout="wide",
)

inject_base_style()

st.title("📊 Data Explorer")
st.caption("Interactive exploration of the applicant population behind the models.")

# ============================================================
# LOAD DATA
# ============================================================

df = load_dataset()

if df is None:
    st.info(
        "Place `emi_prediction_dataset.csv` in the app's root directory, "
        "or upload a sample below to explore the same schema."
    )

    uploaded = st.file_uploader(
        "Upload a CSV with the same schema",
        type="csv",
    )

    if uploaded:
        df = pd.read_csv(uploaded, low_memory=False)
    else:
        st.stop()


# ============================================================
# NUMERIC CLEANING FUNCTION
# ============================================================

def clean_numeric(series):
    """
    Safely convert a column to numeric.

    Handles:
    - commas: 270,700
    - extra spaces
    - malformed decimal values: 270700.0.0
    - non-numeric values

    Invalid values are converted to NaN.
    """

    s = (
        series.astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
    )

    # Fix repeated decimal suffixes such as:
    # 270700.0.0 -> 270700.0
    # 45000.00.0 -> 45000.00
    s = s.str.replace(
        r"\.0\.0$",
        ".0",
        regex=True,
    )

    return pd.to_numeric(
        s,
        errors="coerce",
    )


# ============================================================
# CONVERT NUMERIC COLUMNS SAFELY
# ============================================================

numeric_columns = [
    "monthly_salary",
    "loan_amount",
    "loan_tenure",
    "age",
    "credit_score",
    "years_of_employment",
    "monthly_rent",
    "family_size",
    "dependents",
    "current_emi_amount",
    "bank_balance",
    "emergency_fund",
    "requested_amount",
    "requested_tenure",
    "max_monthly_emi",
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = clean_numeric(df[col])


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "emi_scenario",
    "emi_eligibility",
    "monthly_salary",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        f"Missing required columns: {', '.join(missing_columns)}"
    )
    st.stop()


# ============================================================
# FILTERS
# ============================================================

with st.sidebar:

    st.header("Filters")

    scenarios = st.multiselect(
        "EMI Scenario",
        sorted(df["emi_scenario"].dropna().unique()),
        default=sorted(df["emi_scenario"].dropna().unique()),
    )

    eligibilities = st.multiselect(
        "Eligibility",
        sorted(df["emi_eligibility"].dropna().unique()),
        default=sorted(df["emi_eligibility"].dropna().unique()),
    )

    # Remove missing salaries before calculating slider limits
    salary_values = df["monthly_salary"].dropna()

    if len(salary_values) == 0:
        st.error("No valid numeric values found in monthly_salary.")
        st.stop()

    salary_min = int(salary_values.min())
    salary_max = int(salary_values.max())

    # Prevent slider error if min == max
    if salary_min == salary_max:
        salary_range = (salary_min, salary_max)
        st.info(f"Monthly salary: ₹{salary_min:,}")
    else:
        salary_range = st.slider(
            "Monthly Salary Range (₹)",
            min_value=salary_min,
            max_value=salary_max,
            value=(salary_min, salary_max),
        )


# ============================================================
# APPLY FILTERS
# ============================================================

fdf = df[
    df["emi_scenario"].isin(scenarios)
    & df["emi_eligibility"].isin(eligibilities)
    & df["monthly_salary"].between(*salary_range)
].copy()


st.markdown(
    f"**{len(fdf):,}** applicants match the current filters "
    f"(of {len(df):,} total)."
)


# ============================================================
# KPI CARDS
# ============================================================

k1, k2, k3, k4 = st.columns(4)

if len(fdf) > 0:

    eligible_pct = (
        (fdf["emi_eligibility"] == "Eligible").mean() * 100
    )

    high_risk_pct = (
        (fdf["emi_eligibility"] == "High_Risk").mean() * 100
    )

    not_eligible_pct = (
        (fdf["emi_eligibility"] == "Not_Eligible").mean() * 100
    )

    if "max_monthly_emi" in fdf.columns:
        avg_emi = fdf["max_monthly_emi"].mean()
    else:
        avg_emi = float("nan")

else:

    eligible_pct = 0
    high_risk_pct = 0
    not_eligible_pct = 0
    avg_emi = float("nan")


k1.metric(
    "Eligible %",
    f"{eligible_pct:.1f}%"
)

k2.metric(
    "High Risk %",
    f"{high_risk_pct:.1f}%"
)

k3.metric(
    "Not Eligible %",
    f"{not_eligible_pct:.1f}%"
)

k4.metric(
    "Avg Max Monthly EMI",
    (
        f"₹{avg_emi:,.0f}"
        if pd.notna(avg_emi)
        else "N/A"
    ),
)


st.divider()


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Eligibility Overview",
        "Financial Correlations",
        "Demographics",
        "Raw Data",
    ]
)


# ============================================================
# COLOR MAP
# ============================================================

color_map = {
    "Eligible": "#2ecc71",
    "High_Risk": "#f39c12",
    "Not_Eligible": "#e74c3c",
}


# ============================================================
# TAB 1 — ELIGIBILITY OVERVIEW
# ============================================================

with tab1:

    if len(fdf) == 0:

        st.warning("No applicants match the current filters.")

    else:

        c1, c2 = st.columns(2)

        # --------------------------------------------------------
        # Eligibility Distribution
        # --------------------------------------------------------

        with c1:

            counts = (
                fdf["emi_eligibility"]
                .value_counts()
                .reset_index()
            )

            counts.columns = [
                "emi_eligibility",
                "count",
            ]

            fig = px.pie(
                counts,
                names="emi_eligibility",
                values="count",
                hole=0.45,
                color="emi_eligibility",
                color_discrete_map=color_map,
                title="Eligibility Distribution",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        # --------------------------------------------------------
        # Eligibility by EMI Scenario
        # --------------------------------------------------------

        with c2:

            cross = (
                pd.crosstab(
                    fdf["emi_scenario"],
                    fdf["emi_eligibility"],
                    normalize="index",
                )
                * 100
            )

            cross = (
                cross
                .reset_index()
                .melt(
                    id_vars="emi_scenario",
                    var_name="emi_eligibility",
                    value_name="pct",
                )
            )

            fig = px.bar(
                cross,
                x="emi_scenario",
                y="pct",
                color="emi_eligibility",
                barmode="stack",
                color_discrete_map=color_map,
                title="Eligibility Rate by EMI Scenario",
            )

            fig.update_xaxes(
                tickangle=25
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


# ============================================================
# TAB 2 — FINANCIAL CORRELATIONS
# ============================================================

with tab2:

    c1, c2 = st.columns(2)

    # --------------------------------------------------------
    # Correlation Heatmap
    # --------------------------------------------------------

    with c1:

        corr_cols = [
            "age",
            "monthly_salary",
            "years_of_employment",
            "monthly_rent",
            "family_size",
            "dependents",
            "current_emi_amount",
            "credit_score",
            "bank_balance",
            "emergency_fund",
            "requested_amount",
            "requested_tenure",
            "max_monthly_emi",
        ]

        # Keep only columns that actually exist
        available_corr_cols = [
            col
            for col in corr_cols
            if col in fdf.columns
        ]

        if len(available_corr_cols) < 2:

            st.warning(
                "Not enough numeric columns are available "
                "to calculate correlations."
            )

        else:

            # Create a separate dataframe
            # so the filtered dataframe is not modified.
            corr_df = fdf[
                available_corr_cols
            ].copy()

            # ------------------------------------------------
            # IMPORTANT:
            # Clean every correlation column.
            # This prevents errors such as:
            # '270700.0.0'
            # ------------------------------------------------

            for col in available_corr_cols:

                corr_df[col] = clean_numeric(
                    corr_df[col]
                )

            # Calculate correlation
            corr = corr_df.corr(
                numeric_only=True
            )

            fig = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Correlation Heatmap",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


    # --------------------------------------------------------
    # Salary vs Maximum EMI
    # --------------------------------------------------------

    with c2:

        if (
            "monthly_salary" in fdf.columns
            and "max_monthly_emi" in fdf.columns
            and len(fdf) > 0
        ):

            sample = fdf.sample(
                min(8000, len(fdf)),
                random_state=42,
            )

            fig = px.scatter(
                sample,
                x="monthly_salary",
                y="max_monthly_emi",
                color="emi_eligibility",
                opacity=0.5,
                color_discrete_map=color_map,
                title="Monthly Salary vs Max Monthly EMI",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:

            st.warning(
                "Salary or maximum EMI data is unavailable."
            )


# ============================================================
# TAB 3 — DEMOGRAPHICS
# ============================================================

with tab3:

    if len(fdf) == 0:

        st.warning(
            "No applicants match the current filters."
        )

    else:

        c1, c2 = st.columns(2)

        # --------------------------------------------------------
        # Employment × Education
        # --------------------------------------------------------

        with c1:

            if (
                "employment_type" in fdf.columns
                and "education" in fdf.columns
            ):

                grp = (
                    fdf
                    .groupby(
                        [
                            "employment_type",
                            "education",
                        ]
                    )["emi_eligibility"]
                    .apply(
                        lambda s: (
                            s == "Eligible"
                        ).mean() * 100
                    )
                    .reset_index(
                        name="eligible_pct"
                    )
                )

                fig = px.density_heatmap(
                    grp,
                    x="employment_type",
                    y="education",
                    z="eligible_pct",
                    histfunc="avg",
                    color_continuous_scale="Greens",
                    title=(
                        "Eligibility Rate: "
                        "Employment × Education"
                    ),
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            else:

                st.warning(
                    "Employment or education columns are unavailable."
                )


        # --------------------------------------------------------
        # Requested Amount by Scenario
        # --------------------------------------------------------

        with c2:

            if (
                "requested_amount" in fdf.columns
                and "emi_scenario" in fdf.columns
            ):

                fig = px.box(
                    fdf,
                    x="emi_scenario",
                    y="requested_amount",
                    color="emi_scenario",
                    title="Requested Amount by Scenario",
                    points=False,
                )

                fig.update_yaxes(
                    type="log",
                    title="Requested Amount (INR, log)",
                )

                fig.update_layout(
                    showlegend=False
                )

                fig.update_xaxes(
                    tickangle=25
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            else:

                st.warning(
                    "Requested amount or EMI scenario "
                    "columns are unavailable."
                )


# ============================================================
# TAB 4 — RAW DATA
# ============================================================

with tab4:

    st.dataframe(
        fdf.head(1000),
        use_container_width=True,
        height=450,
    )

    st.download_button(
        "⬇️ Download filtered data (CSV)",
        fdf.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="emipredict_filtered.csv",
        mime="text/csv",
    )


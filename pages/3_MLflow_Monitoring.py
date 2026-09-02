import os

import pandas as pd
import plotly.express as px
import streamlit as st
from utils import MLFLOW_DB, inject_base_style, load_artifacts

st.set_page_config(page_title="MLflow Monitoring | EMIPredict AI", page_icon="📈", layout="wide")
inject_base_style()
st.title("📈 MLflow Model Monitoring")
st.caption("Experiment tracking across every model trained during development.")

artifacts = load_artifacts()
meta = artifacts.get("metadata", {})

mlflow_available = os.path.exists(MLFLOW_DB)

if not mlflow_available:
    st.info(
        f"No local MLflow store found at `{MLFLOW_DB}`. Copy the `mlflow.db` produced by "
        "the notebook into this app's root directory to see full run history here, or run "
        f"`mlflow ui --backend-store-uri sqlite:///{MLFLOW_DB}` locally to browse it directly. "
        "Showing the champion-model summary from `metadata.json` below in the meantime."
    )
else:
    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()

        exp_names = [e.name for e in experiments if e.name != "Default"]
        chosen = st.multiselect("Experiments", exp_names, default=exp_names)

        all_runs = []
        for exp in experiments:
            if exp.name not in chosen:
                continue
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            if not runs.empty:
                runs["experiment"] = exp.name
                all_runs.append(runs)

        if all_runs:
            runs_df = pd.concat(all_runs, ignore_index=True)
            metric_cols = [c for c in runs_df.columns if c.startswith("metrics.")]
            display_cols = ["experiment", "tags.mlflow.runName", "start_time"] + metric_cols
            display_cols = [c for c in display_cols if c in runs_df.columns]

            st.subheader("Run History")
            st.dataframe(
                runs_df[display_cols].rename(columns=lambda c: c.replace("metrics.", "")),
                use_container_width=True,
            )

            st.subheader("Metric Comparison")
            for exp_name in runs_df["experiment"].unique():
                sub = runs_df[runs_df["experiment"] == exp_name]
                m_cols = [c for c in sub.columns if c.startswith("metrics.")]
                if not m_cols:
                    continue
                melt = sub.melt(
                    id_vars="tags.mlflow.runName", value_vars=m_cols,
                    var_name="metric", value_name="value",
                )
                melt["metric"] = melt["metric"].str.replace("metrics.", "", regex=False)
                fig = px.bar(
                    melt, x="tags.mlflow.runName", y="value", color="metric", barmode="group",
                    title=f"{exp_name} — Metric Comparison",
                    labels={"tags.mlflow.runName": "Run"},
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No runs found for the selected experiments.")
    except Exception as e:
        st.error(f"Could not read the MLflow store: {e}")

st.divider()
st.subheader("🏆 Champion Model Summary")

if meta:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Classification — EMI Eligibility**")
        st.markdown(f"- Best model: `{meta.get('best_classifier', '—')}`")
        cm = meta.get("classifier_test_metrics", {})
        st.markdown(
            f"- Accuracy: **{cm.get('accuracy', 0)*100:.2f}%** "
            f"(target >90%: {'✅ Met' if cm.get('accuracy', 0) > 0.90 else '❌ Not met'})"
        )
        st.markdown(f"- F1 (weighted): **{cm.get('f1', 0):.4f}**")
        st.markdown(f"- ROC-AUC (weighted): **{cm.get('roc_auc', 0):.4f}**")
    with c2:
        st.markdown("**Regression — Maximum Monthly EMI**")
        st.markdown(f"- Best model: `{meta.get('best_regressor', '—')}`")
        rm = meta.get("regressor_test_metrics", {})
        st.markdown(
            f"- RMSE: **₹{rm.get('rmse', 0):,.0f}** "
            f"(target <₹2,000: {'✅ Met' if rm.get('rmse', 1e9) < 2000 else '❌ Not met'})"
        )
        st.markdown(f"- R²: **{rm.get('r2', 0):.4f}**")
else:
    st.warning("`models/metadata.json` not found — run the notebook to generate it.")

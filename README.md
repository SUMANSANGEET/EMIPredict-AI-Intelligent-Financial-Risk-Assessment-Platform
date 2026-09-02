# EMIPredict AI — Streamlit App

Multi-page recruiter-facing app for the EMIPredict AI capstone: Home, Real-Time
Prediction, Data Explorer, MLflow Model Monitoring, and Admin (CRUD).

## 1. File Structure
```
emipredict_app/
├── app.py                                # Home
├── pages/
│   ├── 1_🔮_Real_Time_Prediction.py
│   ├── 2_📊_Data_Explorer.py
│   ├── 3_📈_MLflow_Monitoring.py
│   └── 4_🔐_Admin_CRUD.py
├── utils.py                              # shared loading / feature-engineering / DB logic
├── models/                               # <- put trained artifacts here (see below)
├── .streamlit/
│   ├── config.toml                       # dark theme
│   └── secrets.toml.example              # copy to secrets.toml, set ADMIN_PASSWORD
├── requirements.txt
└── emi_prediction_dataset.csv            # <- add for the Data Explorer page (optional)
```

## 2. Required artifacts (produced by the notebook, not included here)
The notebook saves these under `models/` when run end-to-end — copy them into
this app's `models/` folder before deploying:

| File | Produced in notebook cell |
|---|---|
| `best_classifier.pkl` | Step 6 |
| `best_regressor.pkl` | Step 6 |
| `label_encoders.pkl` | Step 3 (encoding) |
| `scaler.pkl` | Step 3 (encoding) |
| `feature_cols.pkl` | Step 3 (encoding) |
| `clf_target_encoder.pkl` | Step 3 (target encoding) |
| `metadata.json` | Step 6 |

The app runs and shows clear warnings if any file is missing — it won't crash,
but Real-Time Prediction and the champion-model summary need them.

Optional: copy `mlflow.db` (created via `mlflow.set_tracking_uri('sqlite:///mlflow.db')`
in the notebook) into the app root to power the full MLflow Monitoring page with
real run history; otherwise it falls back to the `metadata.json` summary.

Optional: copy the cleaned `emi_prediction_dataset.csv` into the app root for
the Data Explorer page (or let users upload their own file with the same schema).

`.pkl`/`.csv`/`.db` files are usually too large for a normal git push — use
[Git LFS](https://git-lfs.com/) if any exceeds GitHub's 100 MB limit.

## 3. Run locally
```bash
cd emipredict_app
pip install -r requirements.txt
streamlit run app.py
```

## 4. Deploy to Streamlit Community Cloud
1. Push this folder to a public (or private, if you connect it) GitHub repo,
   including the `models/` artifacts (via Git LFS if large).
2. Go to https://share.streamlit.io → **New app**.
3. Select the repo, branch, and set **Main file path** to `app.py`.
4. Under **Advanced settings → Secrets**, paste:
   ```toml
   ADMIN_PASSWORD = "your-chosen-password"
   ```
5. Click **Deploy**. First boot installs `requirements.txt`; the multi-page
   nav (Home / Real-Time Prediction / Data Explorer / MLflow Monitoring /
   Admin) appears automatically from the `pages/` folder.
6. SQLite files (`admin_applications.db`, `mlflow.db`) are written to the app's
   ephemeral filesystem — on Community Cloud that storage resets on redeploy/
   sleep, so for a persistent admin queue point `utils.py`'s `ADMIN_DB` at an
   external database (e.g. Postgres via `st.connection`) before using this in
   production review workflows.

## 5. What each page does
- **Home** — KPI cards (accuracy, RMSE, best models, logged applications),
  project/business summary, tech-stack badges, quick nav.
- **Real-Time Prediction** — full applicant intake form → feature engineering
  → scaling/encoding → classifier + regressor inference → probability chart →
  auto-logs the result to the Admin queue.
- **Data Explorer** — sidebar filters (scenario / eligibility / salary range),
  eligibility distribution, correlation heatmap, income vs. EMI scatter,
  employment × education risk heatmap, requested-amount box plot, filtered
  CSV download.
- **MLflow Monitoring** — live run table + metric comparison charts read
  straight from `mlflow.db` when present, else a static champion-model
  summary from `metadata.json`.
- **Admin (CRUD)** — password-gated: view all scored applications, update
  status/notes, manually add an entry, delete an entry.

## 6. Author
P Suman Sangeet · Data Science & AI Intern, INNOVEXIS · LABMENTIX Bold
Analytics Cohort 2025.

# ⚡ Q-Commerce Vadodara — Research Dashboard

> A Statistical Study on Consumer Usage and Adoption of Q-Commerce Applications in Vadodara  
> **Department of Statistics · The Maharaja Sayajirao University of Baroda · 2024–25**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

---

## 📋 Dashboard Pages

| # | Page | Content |
|---|------|---------|
| 🏠 | **Overview** | Project introduction, Q-Commerce explanation, team, KPIs |
| 🎯 | **Objectives** | Primary + 5 secondary objectives with methods |
| 📐 | **Sampling & Design** | Zonal division, ward selection, Cochran's formula |
| 📋 | **Questionnaire** | 8-page instrument design, Likert scale |
| 👥 | **Demographics** | Interactive charts: age, gender, education, occupation, income |
| 📱 | **Obj 1 — App Usage** | Market share, awareness vs. usage |
| 🔗 | **Obj 2 — Adoption** | Chi-Square, Cramér's V, Shapiro-Wilk, stacked bar adoption rates |
| 📊 | **Obj 3 — Behavior** | Descriptive profiling, Kruskal-Wallis, Spearman correlation |
| 🔍 | **Obj 4 — Drivers** | Cronbach's α, EFA/Factor Analysis, non-user barriers |
| 🤖 | **Obj 5 — Predictive** | Logistic Regression (statsmodels), ROC/AUC, Hosmer-Lemeshow |
| ✨ | **Summary** | All findings + recommendations + limitations |

---

## 🚀 Deployment Guide

### Step 1 — Create GitHub Repository

1. Go to [github.com](https://github.com) → **New repository**
2. Name: `qcommerce-vadodara-dashboard`
3. Set to **Public** → **Create repository**

### Step 2 — Upload Files

**Option A — Drag & Drop (easiest):**
1. Click **"uploading an existing file"** on your repo page
2. Drag the entire `qc_dashboard` folder contents
3. **IMPORTANT:** Maintain this exact folder structure:
```
qc_dashboard/
├── app.py                        ← Main entry point
├── utils.py                      ← Shared utilities
├── requirements.txt
├── README.md
├── data/
│   ├── me.xlsx                   ← Raw survey data
│   └── analysis_ready.xlsx       ← Processed data
├── pages/
│   ├── 1_Objectives.py
│   ├── 2_Sampling.py
│   ├── 3_Questionnaire.py
│   ├── 4_Demographics.py
│   ├── 5_Obj1_Apps.py
│   ├── 6_Obj2_Adoption.py
│   ├── 7_Obj3_Behavior.py
│   ├── 8_Obj4_Drivers.py
│   ├── 9_Obj5_Predictive.py
│   └── 10_Summary.py
└── .streamlit/
    └── config.toml               ← Theme config
```
4. Commit message: `Initial dashboard upload`

**Option B — Git CLI:**
```bash
cd qc_dashboard
git init
git add .
git commit -m "Initial dashboard upload"
git remote add origin https://github.com/YOUR_USERNAME/qcommerce-vadodara-dashboard.git
git branch -M main
git push -u origin main
```

### Step 3 — Deploy on Streamlit Cloud (Free)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Fill in:
   - **Repository:** `YOUR_USERNAME/qcommerce-vadodara-dashboard`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy!"** — live in ~3 minutes

### Step 4 — Share

Your dashboard URL: `https://YOUR_USERNAME-qcommerce-vadodara-dashboard-app.streamlit.app`

---

## 💻 Run Locally

```bash
cd qc_dashboard
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 📊 Statistical Methods Covered

| Objective | Tests Used |
|-----------|-----------|
| Obj 1 | Descriptive analysis, frequency distributions |
| Obj 2 | Chi-Square Test, Cramér's V, Shapiro-Wilk normality |
| Obj 3 | Kruskal-Wallis H Test, Spearman Correlation, Dunn's Post-Hoc |
| Obj 4 | Cronbach's Alpha, Item-Total Correlation, KMO, Bartlett's, EFA (PCA + Varimax), Mann-Whitney U |
| Obj 5 | Binary Logistic Regression (statsmodels), Wald Tests, Hosmer-Lemeshow, ROC/AUC, Nagelkerke R² |

---

## 🛠️ Tech Stack

- **Python 3.11** — Core language
- **Streamlit** — Dashboard framework
- **Plotly** — All interactive charts
- **statsmodels** — Logistic regression & statistical inference
- **SciPy** — Statistical tests
- **scikit-learn** — EFA/PCA, preprocessing
- **Pandas / NumPy** — Data processing

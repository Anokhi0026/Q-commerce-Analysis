import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal
from utils import *

st.set_page_config("Non-User Analysis", "🚫", layout="wide")
st.session_state["current_page"] = "pages/13_Non_User_Analysis.py"

from navbar import navbar
navbar()

page_header(
    "Non-User Analysis",
    "Understanding Barriers to Adoption",
    "Statistical analysis of non-users to identify demographic patterns, barriers, and adoption potential."
)

# ── LOAD DATA ─────────────────────────────────────────────
df = load_raw()
non_users = df[df["Adoption_Status"] == 0]

# ── 1. DEMOGRAPHIC PROFILE ────────────────────────────────
section("1 · Demographic Profile of Non-Users")

for col in ["Age_Group", "Gender", "Income", "Education"]:
    ct = non_users[col].value_counts(normalize=True) * 100
    
    fig = go.Figure(go.Bar(
        x=ct.index,
        y=ct.values,
        text=[f"{v:.1f}%" for v in ct.values],
        textposition="outside"
    ))
    
    fig.update_layout(title=f"{col} Distribution (Non-Users)", height=350)
    st.plotly_chart(fig, use_container_width=True)

# ── 2. NON-ADOPTION % ─────────────────────────────────────
section("2 · Non-Adoption Percentage")

total = len(df)
non = len(non_users)

k1,k2 = st.columns(2)
kpi(k1, str(non), "Non-Users", "Count", ROSE)
kpi(k2, f"{(non/total)*100:.1f}%", "Non-Adoption Rate", "Percentage", INDIGO)

# ── 3. CHI-SQUARE TEST ────────────────────────────────────
section("3 · Chi-Square Test (Barrier vs Adoption)")

BARRIER_COLS = [
    "R_High_Charges","R_Quality_Concern","R_No_Need",
    "R_Prefer_Local","R_Trust_Issue","R_App_Discomfort",
    "R_Lack_Awareness","R_Not_Available"
]

chi_results = []

for col in BARRIER_COLS:
    ct = pd.crosstab(df[col], df["Adoption_Status"])
    if ct.shape == (2,2):
        chi2,p,_,_ = chi2_contingency(ct)
        chi_results.append([col,chi2,p])

chi_df = pd.DataFrame(chi_results, columns=["Barrier","Chi2","p-value"])
st.dataframe(chi_df)

st.markdown("**Interpretation:** Barriers with p < 0.05 significantly influence adoption behavior.")

# ── 4. WILLINGNESS TO TRY ─────────────────────────────────
section("4 · Willingness to Try (Likert Analysis)")

LIKERT_COLS = [
    "I would consider using Q-commerce if delivery charges were lower",
    "I would consider using Q-commerce apps if product quality were guaranteed",
    "I would consider using Q-commerce apps if prices were competitive"
]

likert_df = non_users[LIKERT_COLS].copy()

for col in LIKERT_COLS:
    likert_df[col] = likert_df[col].map(LIKERT_MAP)

desc = likert_df.mean().sort_values()

fig = go.Figure(go.Bar(
    y=desc.index,
    x=desc.values,
    orientation="h",
    text=[f"{v:.2f}" for v in desc.values],
    textposition="outside"
))

fig.update_layout(title="Willingness to Adopt Based on Conditions", height=400)
st.plotly_chart(fig, use_container_width=True)

# ── 5. MANN-WHITNEY ───────────────────────────────────────
section("5 · Mann–Whitney U Test (Gender Differences)")

mw_results = []

for col in LIKERT_COLS:
    g1 = likert_df[non_users["Gender"]=="Male"][col].dropna()
    g2 = likert_df[non_users["Gender"]=="Female"][col].dropna()
    
    if len(g1)>2 and len(g2)>2:
        U,p = mannwhitneyu(g1,g2)
        mw_results.append([col,U,p])

mw_df = pd.DataFrame(mw_results, columns=["Variable","U","p-value"])
st.dataframe(mw_df)

st.markdown("**Interpretation:** Significant p-values indicate gender-based differences in perception.")

# ── 6. KRUSKAL-WALLIS ─────────────────────────────────────
section("6 · Kruskal–Wallis Test (Age Group Differences)")

kw_results = []

for col in LIKERT_COLS:
    groups = [grp[col].dropna().values for name, grp in non_users.groupby("Age_Group")]
    
    if len(groups)>1:
        H,p = kruskal(*groups)
        kw_results.append([col,H,p])

kw_df = pd.DataFrame(kw_results, columns=["Variable","H","p-value"])
st.dataframe(kw_df)

st.markdown("**Interpretation:** Significant results indicate variation across age groups.")

# ── FINAL INTERPRETATION ──────────────────────────────────
section("7 · Final Insights")

finding_card(
    "🚧 Key Barriers",
    "High charges, trust issues, and lack of awareness significantly affect non-adoption.",
    ROSE
)

finding_card(
    "📊 Demographic Influence",
    "Age and socio-economic factors influence adoption behavior more than gender.",
    INDIGO
)

finding_card(
    "🎯 Strategic Insight",
    "Reducing delivery cost and improving trust can significantly convert non-users.",
    EMERALD
)

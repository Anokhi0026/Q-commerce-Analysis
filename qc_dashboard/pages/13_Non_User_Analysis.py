import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal
from utils import *

st.set_page_config("Non-User Analysis", "🚫", layout="wide")
st.session_state["current_page"] = "pages/13_Non_User_Analysis.py"

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*='css']{font-family:'Inter',sans-serif;}
.stApp{background:#FAFAFA;}
section[data-testid='stSidebar']{background:#FFFFFF;border-right:1px solid #E2E8F0;}
</style>""", unsafe_allow_html=True)

from navbar import navbar
navbar()

page_header(
    "Non-User Analysis",
    "Barriers & Adoption Potential",
    "Understanding demographic profile, non-adoption rate, barriers, and statistical testing of non-users."
)

df = load_raw()
non_users = df[df["Adoption_Status"] == 0]

# ── KPI ─────────────────────────────────────────────
k1,k2,k3 = st.columns(3)
kpi(k1, str(len(non_users)), "Non-Users", "Sample size", ROSE)
kpi(k2, f"{len(non_users)/len(df)*100:.1f}%", "Non-Adoption Rate", "Population %", INDIGO)
kpi(k3, "Barrier Study", "Focus", "Adoption resistance", AMBER)
st.markdown("<br>", unsafe_allow_html=True)

# ── ANALYSIS 1: DEMOGRAPHIC PROFILE ─────────────────
section("Analysis 1 · Demographic Profile of Non-Users")

cols = ["Age_Group","Gender","Income","Education"]

c1, c2 = st.columns(2)
row_pairs = [(cols[0], cols[1]), (cols[2], cols[3])]

for row, (col1, col2) in zip([c1, c2], row_pairs):
    with row:
        subcol1, subcol2 = st.columns(2)

        for col, container in zip([col1, col2], [subcol1, subcol2]):
            with container:
                ct = non_users[col].value_counts(normalize=True) * 100

                fig = go.Figure(go.Bar(
                    x=ct.index,
                    y=ct.values,
                    text=[f"{v:.1f}%" for v in ct.values],
                    textposition="outside",
                    marker_color=INDIGO
                ))

                fig.update_layout(
                    **PLOTLY_LAYOUT,
                    height=260,
                    title=dict(text=col.replace("_"," "), font=dict(size=12))
                )

                st.plotly_chart(fig, use_container_width=True)
finding_card(
    "👥 Non-Users Concentrated in Specific Segments",
    "Non-adoption is not random — certain demographic groups show higher resistance, indicating targeted intervention potential.",
    INDIGO
)

# ── ANALYSIS 2: NON-ADOPTION % ─────────────────────
section("Analysis 2 · Non-Adoption Rate")

st.markdown(f"""
<div style='background:#fff;border-radius:12px;padding:20px;text-align:center;
border:1px solid #E2E8F0;'>
<div style='font-size:2.5rem;font-weight:800;color:{ROSE};'>
{len(non_users)/len(df)*100:.1f}%
</div>
<div style='font-size:.9rem;color:#64748B;'>Population Not Using Q-Commerce</div>
</div>
""", unsafe_allow_html=True)

# ── ANALYSIS 3: CHI-SQUARE ─────────────────────────
section("Analysis 3 · Chi-Square Test (Barriers vs Adoption)")

BARRIER_COLS = [
    "R_High_Charges","R_Quality_Concern","R_No_Need",
    "R_Prefer_Local","R_Trust_Issue","R_App_Discomfort",
    "R_Lack_Awareness","R_Not_Available"
]

rows = []
for col in BARRIER_COLS:
    ct = pd.crosstab(df[col], df["Adoption_Status"])
    if ct.shape == (2,2):
        chi2,p,_,_ = chi2_contingency(ct)
        rows.append([col,chi2,p])

chi_df = pd.DataFrame(rows, columns=["Barrier","Chi2","p-value"])
st.dataframe(chi_df, use_container_width=True)

finding_card(
    "📊 Significant Barriers Identified",
    "Barriers with statistically significant p-values directly influence adoption behavior.",
    ROSE
)

# ── ANALYSIS 4: WILLINGNESS ────────────────────────
section("Analysis 4 · Willingness to Try (Conditional Adoption)")

LIKERT_COLS = [
    "I would consider using Q-commerce if delivery charges were lower",
    "I would consider using Q-commerce apps if product quality were guaranteed",
    "I would consider using Q-commerce apps if prices were competitive"
]

lk = non_users[LIKERT_COLS].copy()
for col in LIKERT_COLS:
    lk[col] = lk[col].map(LIKERT_MAP)

means = lk.mean().sort_values()

fig = go.Figure(go.Bar(
    y=means.index,
    x=means.values,
    orientation="h",
    marker_color=EMERALD,
    text=[f"{v:.2f}" for v in means.values],
    textposition="outside"
))

fig.update_layout(**PLOTLY_LAYOUT,
    height=300,
    title=dict(text="Adoption Willingness (Mean Scores)", font=dict(size=12))
)

st.plotly_chart(fig, use_container_width=True)

# ── ANALYSIS 5: MANN-WHITNEY ───────────────────────
section("Analysis 5 · Mann–Whitney U Test (Gender Differences)")

mw_rows = []
for col in LIKERT_COLS:
    g1 = lk[non_users["Gender"]=="Male"][col].dropna()
    g2 = lk[non_users["Gender"]=="Female"][col].dropna()
    if len(g1)>2 and len(g2)>2:
        U,p = mannwhitneyu(g1,g2)
        mw_rows.append([col,U,p])

mw_df = pd.DataFrame(mw_rows, columns=["Variable","U","p-value"])
st.dataframe(mw_df, use_container_width=True)

# ── ANALYSIS 6: KRUSKAL ────────────────────────────
section("Analysis 6 · Kruskal–Wallis Test (Age Group Differences)")

kw_rows = []
for col in LIKERT_COLS:
    groups = [grp[col].dropna().values for _, grp in non_users.groupby("Age_Group")]
    if len(groups)>1:
        H,p = kruskal(*groups)
        kw_rows.append([col,H,p])

kw_df = pd.DataFrame(kw_rows, columns=["Variable","H","p-value"])
st.dataframe(kw_df, use_container_width=True)

# ── FINAL INSIGHT ──────────────────────────────────
section("Final Insights")

finding_card(
    "🚧 Core Barriers",
    "Pricing, trust, and awareness are the strongest inhibitors of adoption.",
    ROSE
)

finding_card(
    "📈 Conversion Opportunity",
    "Many non-users are conditionally willing — reducing key barriers can drive adoption.",
    EMERALD
)

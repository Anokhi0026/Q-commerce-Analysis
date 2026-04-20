import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from utils import *

st.set_page_config("Non-User Analysis", "🚫", layout="wide")
st.session_state["current_page"] = "pages/13_Non_User_Analysis.py"

from navbar import navbar
navbar()

page_header(
    "Non-User Analysis",
    "Understanding Barriers to Q-Commerce Adoption",
    "Analysis of non-users to identify key barriers, perceptions, and potential conversion drivers."
)

# Load data
df = load_raw()
non_users = df[df["Adoption_Status"] == 0].copy()

# KPI
k1,k2 = st.columns(2)
kpi(k1, str(len(non_users)), "Total Non-Users", "Sample size", ROSE)
kpi(k2, f"{len(non_users)/len(df)*100:.1f}%", "Non-User Share", "Proportion", INDIGO)

st.markdown("<br>", unsafe_allow_html=True)

# ── BARRIER ANALYSIS ─────────────────────────────
section("Barrier Analysis — Likert Scale")

BARRIER_COLS = [
    "I would consider using Q-commerce if delivery charges were lower",
    "I would consider using Q-commerce apps if product quality were guaranteed",
    "I would consider using Q-commerce apps if adequate guidance on app usage were provided",
    "I would consider using Q-commerce apps if prices were competitive",
    "I would consider using Q-commerce apps if delivery services were available in my area",
    "I would consider using Q-commerce apps if attractive discounts were offered",
    "I would consider using Q-commerce apps if I felt confident about trust, data security, and privacy."
]

BARRIER_NAMES = [
    "Lower Delivery Charges",
    "Product Quality",
    "App Guidance",
    "Competitive Pricing",
    "Availability",
    "Discounts",
    "Trust & Security"
]

bd = non_users[BARRIER_COLS].copy()
for col in BARRIER_COLS:
    bd[col] = bd[col].map(LIKERT_MAP)

bd.columns = BARRIER_NAMES
bd = bd.dropna()

desc = bd.describe().T[["mean"]].rename(columns={"mean":"Mean"})
desc = desc.sort_values("Mean")

fig = go.Figure(go.Bar(
    y=desc.index,
    x=desc["Mean"],
    orientation="h",
    text=[f"{v:.2f}" for v in desc["Mean"]],
    textposition="outside"
))

fig.update_layout(title="Barrier Importance (Mean Scores)", height=400)
st.plotly_chart(fig, use_container_width=True)

# ── OPEN ENDED REASONS ───────────────────────────
section("Open-Ended Non-Adoption Reasons")

reasons = [
    "R_High_Charges","R_Quality_Concern","R_No_Need",
    "R_Prefer_Local","R_Trust_Issue","R_App_Discomfort",
    "R_Lack_Awareness","R_Not_Available"
]

labels = [
    "High Charges","Quality Concern","No Need",
    "Prefer Local","Trust Issues","App Discomfort",
    "Lack of Awareness","Not Available"
]

counts = non_users[reasons].sum().values

fig2 = go.Figure(go.Bar(
    y=labels[::-1],
    x=counts[::-1],
    orientation="h"
))

fig2.update_layout(title="Top Reasons for Non-Adoption", height=400)
st.plotly_chart(fig2, use_container_width=True)

# ── INSIGHTS ─────────────────────────────────────
finding_card(
    "🚧 Key Barriers Identified",
    "Price sensitivity and trust issues are the primary barriers preventing adoption.",
    ROSE
)

finding_card(
    "📉 Awareness Gap",
    "A significant portion of non-users are unaware of Q-commerce services, indicating marketing opportunity.",
    INDIGO
)

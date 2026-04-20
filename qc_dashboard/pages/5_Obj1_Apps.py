import streamlit as st
import plotly.graph_objects as go
import pandas as pd, numpy as np
from utils import *

st.set_page_config("Obj 1 — App Usage", "📱", layout="wide")
st.markdown("<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');html,body,[class*='css']{font-family:'Inter',sans-serif;}.stApp{background:#FAFAFA;}section[data-testid='stSidebar']{background:#FFFFFF;border-right:1px solid #E2E8F0;}</style>",unsafe_allow_html=True)
sidebar()
page_header("Objective 1", "Identifying the Most Used Q-Commerce Apps in Vadodara",
            "Descriptive analysis of app awareness and primary app usage among 228 Q-Commerce users.")

df    = load_raw()
users = get_users()

k1,k2,k3,k4 = st.columns(4)
kpi(k1,"228","Total Users","Q-Commerce adopters")
kpi(k2,f"{(users['App_Used']=='Blinkit').sum()}","Blinkit Users","62% market share",INDIGO)
kpi(k3,f"{(users['App_Used']=='Swiggy Instamart').sum()}","Swiggy Instamart","2nd most used",ROSE)
kpi(k4,f"{(users['App_Used']=='Zepto').sum()}","Zepto","3rd most used",EMERALD)

st.markdown("<br>",unsafe_allow_html=True)
c1, c2 = st.columns(2, gap="large")

with c1:
    section("Primary App Usage — Market Share")
    app_cnt = users["App_Used"].value_counts()
    total_u = len(users)
    fig = go.Figure(go.Pie(
        labels=app_cnt.index, values=app_cnt.values, hole=0.55,
        marker_colors=[INDIGO, ROSE, EMERALD, AMBER],
        textinfo="label+percent+value",
        textfont=dict(size=12),
        hovertemplate="%{label}: %{value} users (%{percent})<extra></extra>",
        pull=[0.05,0,0,0]
    ))
    fig.update_layout(**{k:v for k,v in PLOTLY_LAYOUT.items() if k not in ["xaxis","yaxis"]},
                       height=380, showlegend=True,
                       title=dict(text="Primary App Used (n=228 users)", font=dict(size=13)),
                       annotations=[dict(text="<b>Blinkit</b><br>Leads", x=0.5, y=0.5,
                                         font=dict(size=13, color="#1E1E2E"), showarrow=False)])
    st.plotly_chart(fig, use_container_width=True)

with c2:
    section("App Awareness vs. Primary Usage")
    apps = ["Blinkit", "Zepto", "Swiggy Instamart", "Other"]
    # Count aware respondents (multi-select columns)
    aware_counts = {}
    for app in ["Blinkit","Zepto","Swiggy Instamart"]:
        cnt = sum((df[col].str.strip() == app).sum() for col in [f"App_usex{i}" for i in range(1,7)] if col in df.columns)
        aware_counts[app] = cnt
    aware_counts["Other"] = 0
    usage_counts = users["App_Used"].value_counts().reindex(apps).fillna(0).astype(int)
    aware_vals   = [aware_counts.get(a, 0) for a in apps]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Aware (heard of)", x=apps, y=aware_vals,
                          marker_color=[c+"55" for c in [INDIGO,EMERALD,ROSE,AMBER]],
                          text=aware_vals, textposition="outside"))
    fig.add_trace(go.Bar(name="Primary User", x=apps, y=usage_counts.values,
                          marker_color=[INDIGO,EMERALD,ROSE,AMBER],
                          text=usage_counts.values, textposition="outside"))
    fig.update_layout(**PLOTLY_LAYOUT, **PLOTLY_LAYOUT, barmode="group", height=380,
                       title=dict(text="Awareness vs. Active Usage by App", font=dict(size=13)), legend=dict(x=0.6, y=0.95))
    fig.update_yaxes(title="Count", gridcolor="#F1F5F9")
    st.plotly_chart(fig, use_container_width=True)

section("Key Findings")
finding_card("🏆 Blinkit Dominates with 62% Market Share",
             "142 out of 228 users (62.3%) identify Blinkit as their primary Q-Commerce app in Vadodara. "
             "This near-oligopolistic position reflects strong first-mover advantage and brand recognition.", INDIGO)
finding_card("📊 Two-Tier Market Structure",
             "Swiggy Instamart (18.9%) and Zepto (16.2%) form a distant second tier. Together they account for "
             "only ~35% of the user base, indicating limited competition at the top.", ROSE)
finding_card("👁️ High Awareness, Lower Active Usage",
             "Awareness of all three major platforms is much higher than active usage, suggesting "
             "that brand recognition has not yet converted to adoption for a significant segment of aware consumers.", AMBER)
finding_card("🔒 Platform Lock-in Observed",
             "Most users stick to a single primary app — Q-Commerce in Vadodara does not show "
             "significant multi-app usage, indicating platform loyalty (or lock-in) once adoption occurs.", EMERALD)

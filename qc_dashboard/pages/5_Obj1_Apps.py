import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from utils import *

st.set_page_config("Obj 1 — App Usage", "📱", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*='css']{font-family:'Inter',sans-serif;}
.stApp{background:#FAFAFA;}
section[data-testid='stSidebar']{background:#FFFFFF;border-right:1px solid #E2E8F0;}
</style>
""", unsafe_allow_html=True)

sidebar()

page_header(
    "Objective 1",
    "Identifying the Most Used Q-Commerce Apps in Vadodara",
    "Descriptive analysis of app awareness and primary app usage among 228 Q-Commerce users."
)

df = load_raw()
users = get_users()

# ── KPI ───────────────────────────────────────
k1,k2,k3,k4 = st.columns(4)
kpi(k1,"228","Total Users","Q-Commerce adopters")
kpi(k2,f"{(users['App_Used']=='Blinkit').sum()}","Blinkit Users","62% market share",INDIGO)
kpi(k3,f"{(users['App_Used']=='Swiggy Instamart').sum()}","Swiggy Instamart","2nd most used",ROSE)
kpi(k4,f"{(users['App_Used']=='Zepto').sum()}","Zepto","3rd most used",EMERALD)

st.markdown("<br>", unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="large")

# ── PIE CHART ─────────────────────────────────
with c1:
    section("Primary App Usage — Market Share")

    app_cnt = users["App_Used"].value_counts()

    fig = go.Figure(go.Pie(
        labels=app_cnt.index,
        values=app_cnt.values,
        hole=0.55,
        marker_colors=[INDIGO, ROSE, EMERALD, AMBER],
        textinfo="label+percent+value",
        hovertemplate="%{label}: %{value} users (%{percent})<extra></extra>",
        pull=[0.05,0,0,0]
    ))

    fig.update_layout(
        **{k:v for k,v in PLOTLY_LAYOUT.items() if k not in ["xaxis","yaxis"]},
        height=380,
        title=dict(text="Primary App Used (n=228 users)", font=dict(size=13))
    )

    st.plotly_chart(fig, width="stretch")


# ── BAR CHART ─────────────────────────────────
with c2:
    section("App Awareness vs. Primary Usage")

    apps = ["Blinkit", "Zepto", "Swiggy Instamart", "Other"]

    # Awareness calculation (SAFE)
    aware_counts = {}
    for app in ["Blinkit","Zepto","Swiggy Instamart"]:
        cnt = sum(
            (df[col].astype(str).str.strip() == app).sum()
            for col in [f"App_usex{i}" for i in range(1,7)]
            if col in df.columns
        )
        aware_counts[app] = cnt

    aware_counts["Other"] = 0

    usage_counts = users["App_Used"].value_counts().reindex(apps).fillna(0).astype(int)
    aware_vals   = [aware_counts.get(a, 0) for a in apps]

    # ✅ FIXED COLORS (NO + "55")
    aware_colors = [
        "rgba(79,70,229,0.3)",   # INDIGO
        "rgba(5,150,105,0.3)",   # EMERALD
        "rgba(225,29,72,0.3)",   # ROSE
        "rgba(217,119,6,0.3)"    # AMBER
    ]

    usage_colors = [INDIGO, EMERALD, ROSE, AMBER]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Aware (heard of)",
        x=apps,
        y=aware_vals,
        marker_color=aware_colors,
        text=aware_vals,
        textposition="outside"
    ))

    fig.add_trace(go.Bar(
        name="Primary User",
        x=apps,
        y=usage_counts.values,
        marker_color=usage_colors,
        text=usage_counts.values,
        textposition="outside"
    ))

    # ✅ FIXED LAYOUT (no axis duplication)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        height=380,
        title=dict(text="Awareness vs. Active Usage by App", font=dict(size=13))
    )

    fig.update_yaxes(title="Count")
    fig.update_layout(legend=dict(x=0.6, y=0.95))

    st.plotly_chart(fig, width="stretch")


# ── FINDINGS ─────────────────────────────────
section("Key Findings")

finding_card(
    "🏆 Blinkit Dominates with 62% Market Share",
    "Blinkit clearly leads the market with a dominant share among users.",
    INDIGO
)

finding_card(
    "📊 Two-Tier Market Structure",
    "Swiggy Instamart and Zepto form a secondary tier behind Blinkit.",
    ROSE
)

finding_card(
    "👁️ High Awareness, Lower Active Usage",
    "Awareness exceeds actual usage — indicating conversion gaps.",
    AMBER
)

finding_card(
    "🔒 Platform Lock-in Observed",
    "Users tend to stick to one primary app, showing loyalty or lock-in.",
    EMERALD
)
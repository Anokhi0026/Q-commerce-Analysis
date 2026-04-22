import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from utils import *

st.set_page_config("Demographics", "👥", layout="wide")
st.session_state["current_page"] = "pages/4_Demographics.py"
st.markdown("<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');html,body,[class*='css']{font-family:'Inter',sans-serif;}.stApp{background:#FAFAFA;}section[data-testid='stSidebar']{background:#FFFFFF;border-right:1px solid #E2E8F0;}</style>", unsafe_allow_html=True)
from navbar import navbar
navbar()
page_header("Demographic Profile", "Sample Characteristics",
            "Visualising the demographic composition of all 341 respondents across key variables.")

df  = load_raw()
dfA = load_analysis()

k1,k2,k3,k4 = st.columns(4)
kpi(k1,"341","Total Sample","Vadodara residents")
kpi(k2,f"{(df['Gender']=='Female').sum()}","Female",f"{(df['Gender']=='Female').sum()/len(df)*100:.0f}% of sample",ROSE)
kpi(k3,f"{(df['Gender']=='Male').sum()}","Male",f"{(df['Gender']=='Male').sum()/len(df)*100:.0f}% of sample",SKY)
kpi(k4,f"{df['Age'].median():.0f}","Median Age","Years",AMBER)

st.markdown("<br>",unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Age","Gender","Education","Occupation","Income"])

with tab1:
    age_cnt = df["Age_Group"].value_counts().reindex(AGE_ORDER).fillna(0)
    fig = go.Figure(go.Bar(x=age_cnt.index, y=age_cnt.values,
                            marker_color=PALETTE[:5], text=age_cnt.values, textposition="outside"))
    fig.update_layout(**PLOTLY_LAYOUT, height=320, title=dict(text="Age Group Distribution (n=341)", font=dict(size=13)))
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False)
    st.plotly_chart(fig, use_container_width=True)
    finding_card("Young Adults Dominate the Sample",
                 "The majority of respondents fall in the 18–25 and 26–33 age brackets, making the sample "
                 "predominantly young. Older age groups (42 and above) are comparatively underrepresented.", INDIGO)

with tab2:
    gender_order = ["Male","Female","Prefer not to say"]
    g_cnt = df["Gender"].value_counts().reindex(gender_order).dropna()
    fig = go.Figure(go.Pie(labels=g_cnt.index, values=g_cnt.values,
                            marker_colors=[SKY,ROSE,"#94A3B8"],
                            hole=0.5, textinfo="label+percent"))
    fig.update_layout(**{k:v for k,v in PLOTLY_LAYOUT.items() if k not in ["xaxis","yaxis"]},
                       height=320, title=dict(text="Gender Distribution", font=dict(size=13)))
    st.plotly_chart(fig, use_container_width=True)
    finding_card("Balanced Gender Representation",
                 "The sample is fairly balanced between male and female respondents, ensuring gender-neutral "
                 "insights. A small proportion preferred not to disclose their gender.", SLATE)

with tab3:
    e_cnt = df["Education"].value_counts().reindex(EDU_ORDER).fillna(0)
    fig = go.Figure(go.Bar(x=e_cnt.index, y=e_cnt.values,
                            marker_color=PALETTE[:5], text=e_cnt.values, textposition="outside"))
    fig.update_layout(**PLOTLY_LAYOUT, height=320, title=dict(text="Education Distribution", font=dict(size=13)))
    fig.update_xaxes(tickangle=-20)
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False)
    st.plotly_chart(fig, use_container_width=True)
    finding_card("Highly Educated Sample",
                 "Undergraduate and postgraduate respondents form the bulk of the sample, reflecting an "
                 "educated, digitally literate population well-suited to evaluate Q-Commerce platforms.", EMERALD)

with tab4:
    o_cnt = df["Occupation"].value_counts().reindex(OCC_ORDER).fillna(0)
    fig = go.Figure(go.Bar(x=o_cnt.index, y=o_cnt.values,
                            marker_color=PALETTE[:5], text=o_cnt.values, textposition="outside"))
    fig.update_layout(**PLOTLY_LAYOUT, height=320, title=dict(text="Occupation Distribution", font=dict(size=13)))
    fig.update_xaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False)
    st.plotly_chart(fig, use_container_width=True)
    finding_card("Students & Working Professionals Are the Core Respondents",
                 "Students and working professionals together make up the majority of the sample — both groups "
                 "typically have busy schedules and high digital engagement, making them a relevant audience "
                 "for Q-Commerce research.", AMBER)

with tab5:
    i_cnt = df["Income"].value_counts().reindex(INCOME_ORDER).fillna(0)
    fig = go.Figure(go.Bar(x=i_cnt.index, y=i_cnt.values,
                            marker_color=PALETTE[:5], text=i_cnt.values, textposition="outside"))
    fig.update_layout(**PLOTLY_LAYOUT, height=320, title=dict(text="Income Distribution", font=dict(size=13)))
    fig.update_xaxes(tickangle=-25)
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False)
    st.plotly_chart(fig, use_container_width=True)
    finding_card("Middle-Income Groups Are Most Represented",
                 "Respondents in the ₹20,000–₹60,000 monthly income range form the largest segment, "
                 "reflecting the urban middle class of Vadodara — the primary target demographic for "
                 "Q-Commerce platforms.", VIOLET)

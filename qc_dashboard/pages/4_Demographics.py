import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from utils import *

st.set_page_config("Demographics", "👥", layout="wide")
st.markdown("<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');html,body,[class*='css']{font-family:'Inter',sans-serif;}.stApp{background:#FAFAFA;}section[data-testid='stSidebar']{background:#FFFFFF;border-right:1px solid #E2E8F0;}</style>", unsafe_allow_html=True)
sidebar()
page_header("Demographic Profile", "Sample Characteristics",
            "Visualising the demographic composition of all 341 respondents and comparing users vs. non-users across key variables.")

df  = load_raw()
dfA = load_analysis()

k1,k2,k3,k4 = st.columns(4)
kpi(k1,"341","Total Sample","Vadodara residents")
kpi(k2,f"{(df['Gender']=='Female').sum()}","Female",f"{(df['Gender']=='Female').sum()/len(df)*100:.0f}% of sample",ROSE)
kpi(k3,f"{(df['Gender']=='Male').sum()}","Male",f"{(df['Gender']=='Male').sum()/len(df)*100:.0f}% of sample",SKY)
kpi(k4,f"{df['Age'].median():.0f}","Median Age","Years",AMBER)

st.markdown("<br>",unsafe_allow_html=True)

def adoption_bar(df_in, col, order, title):
    ct = pd.crosstab(df_in[col], df_in["Adoption_Status"])
    ct.columns = ["Non-User","User"]
    ct = ct.reindex([o for o in order if o in ct.index])
    ct_pct = ct.div(ct.sum(axis=1), axis=0)*100
    fig = go.Figure()
    fig.add_trace(go.Bar(name="User", x=ct_pct.index, y=ct_pct["User"].round(1),
                          marker_color=INDIGO, text=ct_pct["User"].round(1),
                          texttemplate="%{text}%", textposition="inside"))
    fig.add_trace(go.Bar(name="Non-User", x=ct_pct.index, y=ct_pct["Non-User"].round(1),
                          marker_color="#CBD5E1", text=ct_pct["Non-User"].round(1),
                          texttemplate="%{text}%", textposition="inside"))
    fig.update_layout(**PLOTLY_LAYOUT, barmode="stack", height=320,
                       title=dict(text=title, font=dict(size=13)),
                       yaxis=dict(title="% of group", range=[0,105],gridcolor="#F1F5F9"),
                       xaxis=dict(tickangle=-20))
    return fig

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Age","Gender","Education","Occupation","Income"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        age_cnt = df["Age_Group"].value_counts().reindex(AGE_ORDER).fillna(0)
        fig = go.Figure(go.Bar(x=age_cnt.index, y=age_cnt.values,
                                marker_color=PALETTE[:5], text=age_cnt.values, textposition="outside"))
        fig.update_layout(**PLOTLY_LAYOUT, height=320, title=dict(text="Age Group Distribution (n=341)", font=dict(size=13)))
        fig.update_yaxes(title="% of group", range=[0,105])
        fig.update_xaxes(tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.plotly_chart(adoption_bar(df,"Age_Group",AGE_ORDER,"Adoption Rate by Age Group (%)"), use_container_width=True)
    finding_card("Age is the Strongest Demographic Predictor",
                 "Younger age groups (18–25, 26–33) show significantly higher Q-Commerce adoption rates. "
                 "Cramér's V = 0.588 (Strong association). χ² = 117.806, p < 0.001.", INDIGO)

with tab2:
    c1, c2 = st.columns(2)
    gender_order = ["Male","Female","Prefer not to say"]
    with c1:
        g_cnt = df["Gender"].value_counts().reindex(gender_order).dropna()
        fig = go.Figure(go.Pie(labels=g_cnt.index, values=g_cnt.values,
                                marker_colors=[SKY,ROSE,"#94A3B8"],
                                hole=0.5, textinfo="label+percent"))
        fig.update_layout(**{k:v for k,v in PLOTLY_LAYOUT.items() if k not in ["xaxis","yaxis"]},
                           height=320, title=dict(text="Gender Distribution", font=dict(size=13)))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.plotly_chart(adoption_bar(df,"Gender",gender_order,"Adoption Rate by Gender (%)"), use_container_width=True)
    finding_card("Gender Shows No Significant Effect",
                 "Male and female respondents adopt Q-Commerce at statistically equivalent rates. "
                 "χ² = 0.457, p = 0.79 (Not significant). Cramér's V = 0.037 (Negligible).", SLATE)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        e_cnt = df["Education"].value_counts().reindex(EDU_ORDER).fillna(0)
        fig = go.Figure(go.Bar(x=e_cnt.index, y=e_cnt.values,
                                marker_color=PALETTE[:5], text=e_cnt.values, textposition="outside"))
        fig.update_layout(**PLOTLY_LAYOUT, height=320, title=dict(text="Education Distribution", font=dict(size=13)),
                           xaxis=dict(tickangle=-20))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.plotly_chart(adoption_bar(df,"Education",EDU_ORDER,"Adoption Rate by Education (%)"), use_container_width=True)
    finding_card("Higher Education Drives Adoption",
                 "Postgraduate and professionally-educated respondents show the highest adoption rates, "
                 "reflecting digital literacy and app familiarity. Cramér's V = 0.407 (Moderate). χ² = 56.36, p < 0.001.", EMERALD)

with tab4:
    c1, c2 = st.columns(2)
    with c1:
        o_cnt = df["Occupation"].value_counts().reindex(OCC_ORDER).fillna(0)
        fig = go.Figure(go.Bar(x=o_cnt.index, y=o_cnt.values,
                                marker_color=PALETTE[:5], text=o_cnt.values, textposition="outside"))
        fig.update_layout(**PLOTLY_LAYOUT, height=320, title=dict(text="Occupation Distribution", font=dict(size=13)))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.plotly_chart(adoption_bar(df,"Occupation",OCC_ORDER,"Adoption Rate by Occupation (%)"), use_container_width=True)
    finding_card("Students & Working Professionals Adopt Most",
                 "Time-constrained occupations (students, working professionals) show significantly higher adoption. "
                 "Retired and homemaker groups lag behind. Cramér's V = 0.271 (Weak). χ² = 24.958, p < 0.001.", AMBER)

with tab5:
    c1, c2 = st.columns(2)
    with c1:
        i_cnt = df["Income"].value_counts().reindex(INCOME_ORDER).fillna(0)
        fig = go.Figure(go.Bar(x=i_cnt.index, y=i_cnt.values,
                                marker_color=PALETTE[:5], text=i_cnt.values, textposition="outside"))
        fig.update_layout(**PLOTLY_LAYOUT, height=320, title=dict(text="Income Distribution", font=dict(size=13)),
                           xaxis=dict(tickangle=-25))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.plotly_chart(adoption_bar(df,"Income",INCOME_ORDER,"Adoption Rate by Income (%)"), use_container_width=True)
    finding_card("Income Shows Weak But Significant Association",
                 "Higher income groups show slightly higher adoption rates, but the effect is weak. "
                 "Cramér's V = 0.246 (Weak). χ² = 20.702, p < 0.001.", VIOLET)

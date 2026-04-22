import streamlit as st
from utils import *

st.set_page_config("Objectives", "🎯", layout="wide")
st.session_state["current_page"] = "pages/1_Objectives.py"
st.markdown("<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');html,body,[class*='css']{font-family:'Inter',sans-serif;}.stApp{background:#FAFAFA;}section[data-testid='stSidebar']{background:#FFFFFF;border-right:1px solid #E2E8F0;}</style>", unsafe_allow_html=True)
from navbar import navbar
navbar()
page_header("Objectives of the Project", "Objectives",
            "A structured framework of one primary and five secondary objectives guiding this statistical study.")

st.markdown(f"""
<div style='background:linear-gradient(135deg,{INDIGO},#7C3AED);border-radius:16px;
            padding:28px 32px;color:#fff;margin-bottom:24px;'>
  <div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:.12em;opacity:.8;margin-bottom:8px;'>Primary Objective</div>
  <div style='font-size:1.15rem;font-weight:700;line-height:1.5;'>
    To statistically analyze the usage patterns and adoption behavior of Q-Commerce apps
    among consumers of Vadodara.
  </div>
</div>""", unsafe_allow_html=True)

section("Secondary Objectives", "Five targeted objectives derived from the primary research question")

objs = [
    ("1","Identify Most-Used Apps",
     "Identify the most used Q-Commerce applications in Vadodara and assess their relative market share.",
     "Descriptive Analysis · Bar Charts · Market Share",
     INDIGO),
    ("2","Demographic Adoption Patterns",
     "Understand consumer adoption patterns based on demographics — age, gender, income, education, and occupation.",
     "Chi-Square Test · Cramér's V · Stacked Bar Charts",
     ROSE),
    ("3","Usage Behavior Analysis",
     "Examine usage behavior patterns including order frequency, average spend, preferred delivery times, payment methods, and product categories.",
     "Kruskal-Wallis · Spearman Correlation · Descriptive Stats",
     EMERALD),
    ("4","Key Drivers of Adoption",
     "Investigate the key drivers of app adoption — convenience, pricing, product variety, usability, and promotions — and identify barriers for non-users.",
     "Cronbach's Alpha · EFA · Factor Analysis · Mann-Whitney U",
     AMBER),
    ("5","Predictive Models",
     "Develop predictive models for adoption likelihood based on socio-demographic and lifestyle variables.",
     "Binary Logistic Regression · ROC/AUC · statsmodels · Hosmer-Lemeshow",
     VIOLET),
]

for tag, title, desc, methods, color in objs:
    st.markdown(f"""
    <div style='background:#fff;border:1px solid #E2E8F0;border-radius:14px;
                padding:20px 24px;margin-bottom:12px;display:flex;gap:18px;align-items:flex-start;
                box-shadow:0 1px 3px rgba(0,0,0,.04);'>
      <div style='min-width:48px;height:48px;border-radius:12px;background:{color}15;
                  display:flex;align-items:center;justify-content:center;font-size:1.4rem;'>
        {tag}
      </div>
      <div style='font-weight:600; font-size:0.95rem; color:#1E1E2E;'>
       {tag}
      </div>
      <div style='flex:1;'>
        <div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;'>
          <span style='font-size:.7rem;font-weight:700;color:{color};text-transform:uppercase;
                       letter-spacing:.1em;background:{color}15;padding:2px 8px;border-radius:20px;'>
        </div>
        <p style='color:#475569;font-size:.85rem;line-height:1.6;margin:0 0 8px;'>{desc}</p>
        <div style='font-size:.75rem;color:{color};font-weight:500;'> • {methods}</div>
      </div>
    </div>""", unsafe_allow_html=True)

section("Research Framework")
c1, c2 = st.columns(2)
for col, title, items in [
    (c1,  "Data Collection", ["Primary data via Google Forms","341 respondents","Vadodara residents aged 18+","Multi-stage sampling"]),
    (c2, "Analytical Methods", ["Descriptive statistics","Inferential tests (χ², KW)","Factor & cluster analysis","Logistic regression"]),
]:
    with col:
        st.markdown(f"""
        <div style='background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:20px;height:100%;'>
          <div style='font-size:1.3rem;margin-bottom:8px;'>{icon}</div>
          <div style='font-weight:700;font-size:.9rem;color:#1E1E2E;margin-bottom:10px;'>{title}</div>
          {''.join(f'<div style="font-size:.8rem;color:#475569;padding:4px 0;border-bottom:1px solid #F1F5F9;">• {i}</div>' for i in items)}
        </div>""", unsafe_allow_html=True)

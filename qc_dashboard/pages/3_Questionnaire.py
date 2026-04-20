import streamlit as st
from utils import *

st.set_page_config("Questionnaire", "📋", layout="wide")
st.session_state["current_page"] = "pages/3_Questionnaire.py"
st.markdown("<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');html,body,[class*='css']{font-family:'Inter',sans-serif;}.stApp{background:#FAFAFA;}section[data-testid='stSidebar']{background:#FFFFFF;border-right:1px solid #E2E8F0;}</style>", unsafe_allow_html=True)
from navbar import navbar
navbar()
page_header("Questionnaire Design", "Data Collection",
            "An 8-page structured questionnaire administered via Google Forms, covering demographics, user behaviour, non-user barriers, and adoption drivers.")

# ── Overview ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
kpi(c1, "7",      "Total Sections",     "Questionnaire sections")
kpi(c2, "~35",    "Questions",       "Across all sections", EMERALD)
kpi(c3, "5–10",   "Minutes",         "Estimated completion", AMBER)
kpi(c4, "Google", "Platform",        "Forms-based survey", VIOLET)

st.markdown("<br>", unsafe_allow_html=True)

# ── Questionnaire Sections ─────────────────────────────────────────────────────
section("Questionnaire Structure & Flow")

sections = [
    ("1","Demographics","👤",INDIGO,
     ["Age","Gender","Education","Occupation","Monthly Household Income"],
     "Captures respondent profile for demographic analysis and segmentation."),
    ("2","Q-Commerce Awareness","👁️",SKY,
     ["Aware of Q-Commerce?","Ever used a QC app?"],
     "Screens respondents into user vs. non-user branches for subsequent routing."),
    ("3","User Section — App Usage","📱",EMERALD,
     ["Which apps heard of?","Primary app used","Usage duration","Avg. order value","Delivery time preference","Payment method","Items ordered"],
     "Captures detailed behavioural patterns for Objective 3 analysis."),
    ("4","Factors Influencing Adoption","⭐",VIOLET,
     ["10-item Likert scale: Time saving, Delivery speed, Discounts, Product variety,\nPromo offers, Ease of use, Urgent needs, Quality, Schedule barrier, Lifestyle fit"],
     "Measures attitude intensity on a 5-point Strongly Disagree → Strongly Agree scale. Used in Objective 4 EFA."),
    ("5","Satisfaction & Future Usage","😊",AMBER,
     ["Overall satisfaction (1–5)","Likelihood of continuing (1–5)","Likelihood to recommend (1–5)","Biggest reason for using"],
     "Captures loyalty and Net Promoter Score-style metrics."),
    ("6","Non-User Section","🚫",ROSE,
     ["Why don't you use QC apps? (8 reasons)","Do you find QC apps reliable?","Willing to try in future?","Family member uses QC?"],
     "Routes respondents who never used QC apps — maps barriers for Objective 4 non-user analysis."),
    ("7","Non-User Barrier Likert","🔒","#DB2777",
     ["7 conditional adoption statements: Lower charges, Quality guarantee, App guidance,\nCompetitive prices, Delivery availability, Discounts, Trust & security"],
     "Identifies what would trigger adoption among non-users — key for strategic recommendations."),
]

for num, title, icon, color, items, note in sections:
    with st.expander(f"Section {num} — {title}", expanded=(num in ["1","3","4","6"])):
        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.markdown(f"<div style='font-size:.83rem;color:#475569;margin-bottom:10px;'>{note}</div>", unsafe_allow_html=True)
            for item in items:
                st.markdown(f"<div style='padding:4px 0;font-size:.82rem;color:#374151;border-bottom:1px solid #F8FAFC;'>• {item}</div>", unsafe_allow_html=True)
        with col_r:
            st.markdown(f"""
            <div style='background:{color}10;border:1px solid {color}30;border-radius:12px;
                        padding:16px;text-align:center;'>
              <div style='font-size:2rem;'>{icon}</div>
              <div style='font-weight:700;font-size:.85rem;color:{color};margin-top:8px;'>Section {num}</div>
              <div style='font-size:.75rem;color:#64748B;margin-top:4px;'>{title}</div>
            </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
section("Questionnaire Link")
st.markdown(f"""
<div style='background:{INDIGO}08;border:1px solid {INDIGO}30;border-radius:14px;
            padding:20px 24px;display:flex;justify-content:space-between;align-items:center;'>
  <div>
    <div style='font-weight:600;font-size:.9rem;color:#1E1E2E;'>Google Forms Survey</div>
    <div style='font-size:.8rem;color:#64748B;margin-top:4px;'>
      https://forms.gle/KQpowsuTjEddczZm6
    </div>
  </div>
  <div style='font-size:1.5rem;'>📋</div>
</div>""", unsafe_allow_html=True)

section("Likert Scale Used in Sections 4 & 7")
likert_rows = [("1","Strongly Disagree","#EF4444"),("2","Disagree","#F97316"),
               ("3","Neutral","#EAB308"),("4","Agree","#22C55E"),("5","Strongly Agree","#16A34A")]
cols = st.columns(5)
for col, (val, label, color) in zip(cols, likert_rows):
    col.markdown(f"""
    <div style='background:{color}15;border:1px solid {color}40;border-radius:10px;
                padding:12px;text-align:center;'>
      <div style='font-size:1.4rem;font-weight:800;color:{color};'>{val}</div>
      <div style='font-size:.75rem;color:#374151;margin-top:4px;font-weight:500;'>{label}</div>
    </div>""", unsafe_allow_html=True)

"""
utils.py — shared data loading, constants, and style helpers
"""
import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import chi2_contingency, kruskal, spearmanr

# ── Colour palette (consistent across all pages) ───────────────────────────────
INDIGO   = "#4F46E5"
VIOLET   = "#7C3AED"
ROSE     = "#E11D48"
AMBER    = "#D97706"
EMERALD  = "#059669"
SKY      = "#0284C7"
SLATE    = "#475569"
LIGHT    = "#F1F5F9"

PALETTE  = [INDIGO, ROSE, EMERALD, AMBER, SKY, VIOLET, "#DB2777", "#EA580C"]
C_SCALE  = [[0, "#EEF2FF"], [0.5, "#818CF8"], [1, "#3730A3"]]   # indigo gradient

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#374151", size=12),
    margin=dict(t=50, b=30, l=20, r=20),
    xaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
)

# ── Likert / variable constants ────────────────────────────────────────────────
LIKERT_MAP = {"Strongly Disagree":1,"Disagree":2,"Neutral":3,"Agree":4,"Strongly Agree":5}

LIKERT_COLS = [
    "Q-commerce apps save my time",
    "Delivery speed is very convenient",
    "Discounts influence my usage",
    "Product variety meets my needs",
    "Promotional offers attract me",
    "Apps are easy to navigate",
    "Urgent needs motivate me to use these apps",
    "Product quality is reliable",
    "My work/study schedule makes offline shopping difficult",
    "Q-commerce fits my lifestyle",
]
SHORT_NAMES = [
    "Time Saving","Delivery Speed","Discounts","Product Variety",
    "Promo Offers","Ease of Use","Urgent Needs","Quality Reliable",
    "Schedule Barrier","Lifestyle Fit",
]

AGE_ORDER    = ["18-25","26-33","34-41","42-49","50 or above"]
INCOME_ORDER = ["Below ₹20,000","₹20,000 - ₹40,000","₹40,000 - ₹60,000",
                "₹60,000 - ₹1,00,000","Above ₹1,00,000"]
EDU_ORDER    = ["No Formal Education","School Level","Undergraduate",
                "Postgraduate","Professional Degree"]
OCC_ORDER    = ["Student","Working professional","Self-employed","Homemaker","Retired"]

# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data
def load_raw():
    df = pd.read_excel("data/me.xlsx", sheet_name="raw", header=1)
    return df

@st.cache_data
def load_analysis():
    df = pd.read_excel("data/analysis_ready.xlsx")
    return df

@st.cache_data
def get_users():
    df = load_raw()
    return df[df["Adoption_Status"] == 1].copy().reset_index(drop=True)

@st.cache_data
def get_non_users():
    df = load_analysis()
    return df[df["Adoption_Status"] == 0].copy().reset_index(drop=True)

@st.cache_data
def get_likert():
    users = get_users()
    ld = users[LIKERT_COLS].copy()
    for col in LIKERT_COLS:
        ld[col] = ld[col].map(LIKERT_MAP)
    ld.columns = SHORT_NAMES
    return ld.dropna().reset_index(drop=True)

# ── Stats helpers ──────────────────────────────────────────────────────────────
def cramers_v(chi2_val, n, r, c):
    phi2 = chi2_val / n
    phi2corr = max(0, phi2 - ((c-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    ccorr = c - ((c-1)**2)/(n-1)
    denom = min(rcorr-1, ccorr-1)
    if denom <= 0:
        return phi2corr
    return np.sqrt(phi2corr / denom)

def chi2_test(df, row_var, col_var):
    ct = pd.crosstab(df[row_var], df[col_var])
    chi2, p, dof, _ = chi2_contingency(ct)
    n = ct.values.sum()
    r, c = ct.shape
    v = cramers_v(chi2, n, r, c)
    return chi2, p, dof, v, ct

def cronbach_alpha(df_in):
    k = df_in.shape[1]
    iv = df_in.var(axis=0, ddof=1).sum()
    tv = df_in.sum(axis=1).var(ddof=1)
    return (k/(k-1))*(1 - iv/tv)

# ── Sidebar (shared) ───────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='padding:16px 0 8px; text-align:center;'>
          <div style='font-size:1.5rem;'>⚡</div>
          <div style='font-weight:700; font-size:1rem; color:#1E1E2E;'>Q-Commerce Vadodara</div>
          <div style='font-size:0.72rem; color:#64748B; margin-top:2px;'>Research Dashboard</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<div style='font-size:0.7rem;font-weight:600;color:#94A3B8;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;'>Pages</div>", unsafe_allow_html=True)
        st.page_link("app.py",                        label="🏠  Overview")
        st.page_link("pages/1_Objectives.py",          label="🎯  Objectives")
        st.page_link("pages/2_Sampling.py",            label="📐  Sampling & Design")
        st.page_link("pages/3_Questionnaire.py",       label="📋  Questionnaire")
        st.page_link("pages/4_Demographics.py",        label="👥  Demographics")
        st.page_link("pages/5_Obj1_Apps.py",           label="📱  Obj 1 — App Usage")
        st.page_link("pages/6_Obj2_Adoption.py",       label="🔗  Obj 2 — Adoption")
        st.page_link("pages/7_Obj3_Behavior.py",       label="📊  Obj 3 — Behavior")
        st.page_link("pages/8_Obj4_Drivers.py",        label="🔍  Obj 4 — Drivers")
        st.page_link("pages/9_Obj5_Predictive.py",        label="🤖  Obj 5 — Predictive")
        st.page_link("pages/11_Cluster_Analysis.py",    label="🧩  Cluster Analysis")
        st.page_link("pages/12_Correspondence_Analysis.py", label="📍  Correspondence Analysis")
        st.page_link("pages/10_Summary.py",             label="✨  Summary")
        st.markdown("---")
        st.markdown("""<div style='font-size:0.7rem;color:#94A3B8;line-height:1.7;'>
        🎓 MSc Statistics<br>MS University of Baroda<br>📍 Vadodara, Gujarat<br>n = 341
        </div>""", unsafe_allow_html=True)

# ── Section title helper ───────────────────────────────────────────────────────
def section(label, sub=""):
    st.markdown(f"""
    <div style='margin:28px 0 6px;'>
      <div style='font-size:0.68rem;font-weight:600;color:{INDIGO};text-transform:uppercase;
                  letter-spacing:.1em;margin-bottom:4px;'>{label}</div>
      {'<div style="font-size:0.85rem;color:#64748B;">'+sub+'</div>' if sub else ''}
    </div>""", unsafe_allow_html=True)

def page_header(tag, title, sub=""):
    st.markdown(f"""
    <div style='padding:8px 0 20px;'>
      <div style='font-size:0.72rem;font-weight:600;color:{INDIGO};text-transform:uppercase;
                  letter-spacing:.12em;margin-bottom:6px;'>{tag}</div>
      <h1 style='font-size:2rem;font-weight:800;color:#1E1E2E;margin:0 0 8px;
                 line-height:1.2;'>{title}</h1>
      {'<p style="color:#64748B;max-width:680px;line-height:1.7;font-size:0.95rem;margin:0;">'+sub+'</p>' if sub else ''}
    </div>""", unsafe_allow_html=True)

def kpi(col, value, label, sub="", color=INDIGO):
    col.markdown(f"""
    <div style='background:#fff;border:1px solid #E2E8F0;border-radius:14px;
                padding:20px 16px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.04);'>
      <div style='font-size:1.9rem;font-weight:800;color:{color};font-family:monospace;
                  line-height:1;'>{value}</div>
      <div style='font-size:0.72rem;text-transform:uppercase;letter-spacing:.08em;
                  color:#64748B;margin-top:6px;'>{label}</div>
      {'<div style="font-size:0.75rem;color:'+color+';margin-top:3px;">'+sub+'</div>' if sub else ''}
    </div>""", unsafe_allow_html=True)

def finding_card(title, text, color=INDIGO):
    st.markdown(f"""
    <div style='background:#fff;border:1px solid #E2E8F0;border-left:4px solid {color};
                border-radius:0 12px 12px 0;padding:14px 18px;margin:8px 0;
                box-shadow:0 1px 2px rgba(0,0,0,.03);'>
      <div style='font-weight:600;font-size:0.9rem;color:#1E1E2E;margin-bottom:4px;'>{title}</div>
      <div style='font-size:0.83rem;color:#475569;line-height:1.6;'>{text}</div>
    </div>""", unsafe_allow_html=True)

def sig_badge(p):
    if p < 0.001:  return "🟢 p < 0.001"
    if p < 0.01:   return "🟢 p < 0.01"
    if p < 0.05:   return "🟡 p < 0.05"
    return "🔴 Not significant"

"""
utils.py — shared data loading, constants, and style helpers
"""
import os as _os
import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import chi2_contingency, kruskal, spearmanr

# ── Absolute path to data directory — tries multiple locations for Streamlit Cloud compatibility
def _find_data_dir():
    candidates = [
        # Same folder as utils.py  →  works locally
        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data"),
        # CWD/data  →  works when Streamlit sets CWD to the app folder
        _os.path.join(_os.getcwd(), "data"),
        # One level up from utils.py  →  monorepo layouts
        _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data"),
    ]
    for path in candidates:
        if _os.path.isdir(path):
            return path
    # Fall back to the first candidate so errors point to a meaningful path
    return candidates[0]
 
# ── Ocean / Teal Colour Palette ────────────────────────────────────────────────
INDIGO   = "#0D9488"   # Teal-600  (was Indigo-600)   — primary accent
VIOLET   = "#0E7490"   # Cyan-700  (was Violet-600)   — secondary accent
ROSE     = "#0F766E"   # Teal-700  (was Rose-600)     — tertiary accent
AMBER    = "#0891B2"   # Cyan-600  (was Amber-600)    — highlight
EMERALD  = "#22D3EE"   # Cyan-400  (was Emerald-600)  — light accent
SKY      = "#14B8A6"   # Teal-400  (was Sky-600)      — soft teal
SLATE    = "#334E68"   # Ocean slate                  — neutral text/icon
LIGHT    = "#F0FDFA"   # Teal-50   (was Slate-100)    — page background tint
 
PALETTE  = [
    "#0D9488",  # Teal-600
    "#0891B2",  # Cyan-600
    "#14B8A6",  # Teal-400
    "#22D3EE",  # Cyan-400
    "#0E7490",  # Cyan-700
    "#0F766E",  # Teal-700
    "#67E8F9",  # Cyan-300
    "#2DD4BF",  # Teal-300
]
 
# Teal → deep ocean gradient for heatmaps / sequential scales
C_SCALE  = [[0, "#F0FDFA"], [0.5, "#2DD4BF"], [1, "#0F766E"]

def hex_alpha(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex color + alpha (0–1) to an rgba() string.
    Use this instead of appending 2-digit hex alpha (e.g. INDIGO+'99')
    because Plotly does not accept 8-digit hex colors."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# xaxis/yaxis are NOT in PLOTLY_LAYOUT — define them explicitly per chart
# to avoid "got multiple values for keyword argument" TypeError
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#374151", size=12),
    margin=dict(t=50, b=30, l=20, r=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
)

# Standard axis defaults — use with fig.update_xaxes / fig.update_yaxes
AXIS_STYLE = dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False)

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
    df = pd.read_excel(_os.path.join(_DATA_DIR, "me.xlsx"), sheet_name="raw", header=1)
    return df

@st.cache_data
def load_analysis():
    df = pd.read_excel(_os.path.join(_DATA_DIR, "analysis_ready.xlsx"))
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
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    </style>
    """, unsafe_allow_html=True)
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

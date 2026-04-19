import pandas as pd
import numpy as np
import streamlit as st
import os

# ── BASE PATH ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── COLORS ───────────────────────────────────────────────
INDIGO   = "#4F46E5"
VIOLET   = "#7C3AED"
ROSE     = "#E11D48"
AMBER    = "#D97706"
EMERALD  = "#059669"
SKY      = "#0284C7"
SLATE    = "#475569"
LIGHT    = "#F1F5F9"

PALETTE = [INDIGO, ROSE, EMERALD, AMBER, SKY, VIOLET]

# ── DATA LOADERS (FIXED PATHS) ───────────────────────────
@st.cache_data(ttl=3600)
def load_raw():
    file_path = os.path.join(BASE_DIR, "data", "me.xlsx")
    df = pd.read_excel(file_path, sheet_name="raw", header=1)
    df.columns = df.columns.str.strip()
    return df

@st.cache_data(ttl=3600)
def load_analysis():
    file_path = os.path.join(BASE_DIR, "data", "analysis_ready.xlsx")
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    return df

@st.cache_data(ttl=3600)
def get_users():
    df = load_raw()
    return df[df["Adoption_Status"] == 1].copy()

# ── SAFE KPI ─────────────────────────────────────────────
def kpi(col, value, label, sub="", color="#4F46E5"):
    col.markdown(f"""
    <div style='background:white;padding:14px;border-radius:10px;border:1px solid #E5E7EB;'>
        <div style='font-size:1.2rem;font-weight:700;color:{color};'>{value}</div>
        <div style='font-size:0.8rem;color:#374151;'>{label}</div>
        <div style='font-size:0.7rem;color:#9CA3AF;'>{sub}</div>
    </div>
    """, unsafe_allow_html=True)

# ── SIDEBAR (SAFE) ───────────────────────────────────────
def sidebar():
    st.sidebar.title("📊 Q-Commerce Dashboard")
    st.sidebar.markdown("Navigate using pages")

# ── SECTION ──────────────────────────────────────────────
def section(title, subtitle=""):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)

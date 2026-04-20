import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from utils import *

st.set_page_config("Sampling & Design", "📐", layout="wide")
st.session_state["current_page"] = "pages/2_Sampling.py"
st.markdown("<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');html,body,[class*='css']{font-family:'Inter',sans-serif;}.stApp{background:#FAFAFA;}section[data-testid='stSidebar']{background:#FFFFFF;border-right:1px solid #E2E8F0;}</style>", unsafe_allow_html=True)
from navbar import navbar
navbar()
page_header("Sampling Frame & Sample Size", "Sampling Design",
            "Multi-stage stratified sampling across four zones of Vadodara with Cochran's formula for sample size determination.")

# ── Sampling Info ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
kpi(c1, "Primary", "Data Type",    "Collected via questionnaire")
kpi(c2, "18+",     "Target Age",   "Adult Vadodara residents", EMERALD)
kpi(c3, "1.74M",  "Population",   "2011 Census estimate", AMBER)
kpi(c4, "Multi-Stage","Sampling",  "Zone → Ward → Respondent", VIOLET)

st.markdown("<br>", unsafe_allow_html=True)
left, right = st.columns([1, 1.2], gap="large")

with left:
    section("Zonal Division of Vadodara")
    zones = {
        "North":"1, 2, 3, 7, 13","South":"16, 17, 18, 19",
        "East":"4, 5, 6, 14, 15","West":"8, 9, 10, 11, 12"
    }
    colors_z = [INDIGO, ROSE, EMERALD, AMBER]
    for (zone, wards), color in zip(zones.items(), colors_z):
        st.markdown(f"""
        <div style='background:#fff;border:1px solid #E2E8F0;border-left:4px solid {color};
                    border-radius:0 10px 10px 0;padding:12px 16px;margin-bottom:8px;
                    display:flex;justify-content:space-between;align-items:center;'>
          <div>
            <span style='font-weight:700;color:#1E1E2E;font-size:.9rem;'>{zone} Zone</span>
            <div style='font-size:.78rem;color:#64748B;margin-top:2px;'>Wards: {wards}</div>
          </div>
          <div style='font-size:1.2rem;'>{"🧭"}</div>
        </div>""", unsafe_allow_html=True)

    section("Selected Wards for Data Collection")
    selected = [
        ("1","North","Ward 07","92,325","77"),
        ("2","South","Ward 17","86,708","64"),
        ("3","East","Ward 04","83,367","81"),
        ("4","West","Ward 11","98,724","86"),
    ]
    for num, zone, ward, pop, n_sample in selected:
        color = colors_z[int(num)-1]
        st.markdown(f"""
        <div style='background:{color}08;border:1px solid {color}30;border-radius:10px;
                    padding:10px 14px;margin-bottom:6px;display:flex;align-items:center;gap:12px;'>
          <div style='width:28px;height:28px;border-radius:50%;background:{color};color:#fff;
                      font-weight:700;font-size:.8rem;display:flex;align-items:center;
                      justify-content:center;flex-shrink:0;'>{num}</div>
          <div style='flex:1;'>
            <div style='font-weight:600;font-size:.85rem;color:#1E1E2E;'>{zone} — {ward}</div>
            <div style='font-size:.75rem;color:#64748B;'>Pop ≈ {pop} | Sample: <b style="color:{color};">{n_sample}</b></div>
          </div>
        </div>""", unsafe_allow_html=True)

with right:
    section("Sample Size per Ward")
    wards   = ["Ward 07\n(North)","Ward 17\n(South)","Ward 04\n(East)","Ward 11\n(West)"]
    samples = [77, 64, 81, 86]
    fig_bar = go.Figure(go.Bar(
        x=wards, y=samples, marker_color=colors_z,
        text=samples, textposition="outside",
        hovertemplate="%{x}: n=%{y}<extra></extra>"
    ))
    fig_bar.update_layout(**PLOTLY_LAYOUT, height=320,
        title=dict(text="Sample Size by Ward (Total = 308, adjusted to 341)", font=dict(size=13)))
    fig_bar.update_xaxes(title="Selected Ward")
    fig_bar.update_yaxes(title="Sample Size", range=[0,110], gridcolor="#F1F5F9")
    st.plotly_chart(fig_bar, use_container_width=True)

    section("Sample Size Formula — Cochran's Method")
    st.markdown("""
    <div style='background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:20px;'>
      <div style='font-family:monospace;font-size:1rem;text-align:center;padding:16px;
                  background:#F8FAFC;border-radius:8px;margin-bottom:14px;'>
        n = Z²·p·q / e²
      </div>
      <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;'>
    """ + "".join([f"""        <div style='background:#F8FAFC;border-radius:8px;padding:10px 12px;'>
          <div style='font-family:monospace;font-weight:700;color:{INDIGO};font-size:.95rem;'>{sym}</div>
          <div style='font-size:.75rem;color:#64748B;margin-top:2px;'>{desc}</div>
        </div>"""
        for sym, desc in [
            ("Z = 1.96","95% confidence level"),
            ("p = 0.85","Estimated population proportion"),
            ("q = 1 - p = 0.15",""),
            ("e = 0.05","5% margin of error"),
        ]]) + """
      </div>
    </div>""", unsafe_allow_html=True)

    section("Pilot Survey")
    st.markdown("""
    <div style='background:#fff;border:1px solid #E2E8F0;border-radius:12px;padding:16px 20px;'>
      <p style='font-size:.85rem;color:#475569;line-height:1.7;margin:0;'>
        A pilot survey was conducted before the main data collection phase to:
        <br>• Verify questionnaire clarity and flow
        <br>• Estimate non-response rates for adjustment
        <br>• Identify ambiguous or culturally sensitive questions
        <br>• Validate the sampling procedure across all four zones
      </p>
    </div>""", unsafe_allow_html=True)

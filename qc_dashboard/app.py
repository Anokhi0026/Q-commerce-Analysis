import streamlit as st
import plotly.graph_objects as go
from utils import *


st.set_page_config("Q-Commerce Vadodara", "⚡", layout="wide", initial_sidebar_state="expanded")
st.session_state["current_page"] = "app.py"
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#FAFAFA;}
section[data-testid="stSidebar"]{background:#FFFFFF;border-right:1px solid #E2E8F0;}
</style>""", unsafe_allow_html=True)    background: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

/* HEADINGS */
h1 { font-size: 2.2rem !important; }
h2 { font-size: 1.8rem !important; }
h3 { font-size: 1.5rem !important; }

/* TEXT ELEMENTS */
p, div, span {
    font-size: 1rem;   /* 🔥 control normal text */
}

/* KPI / METRIC TEXT (important for your dashboard) */
[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.9rem !important;
}

</style>
""", unsafe_allow_html=True)
 
from navbar import navbar
navbar()
 
# ── Hero ───────────────────────────────────────────────────────────────────────
# Build pill HTML outside the f-string to avoid rendering issues
info_pills = "".join(
    f'<span style="background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.3);'
    f'border-radius:20px;padding:4px 14px;font-size:.8rem;">{t}</span>'
    for t in ['MSc Statistics 2024–26']
)

team_pills = "".join(
    f'<span style="background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.25);'
    f'border-radius:20px;padding:4px 12px;font-size:.78rem;">{t}</span>'
    for t in ['Anokhi Desai','Sanjana Kumari', 'Vedant Ghaisas','Ritika Sharma']
)

st.markdown(f"""
<div style='background:linear-gradient(135deg,#0D9488 0%,#0E7490 60%,#0F766E 100%);
            border-radius:20px;padding:48px 40px;margin-bottom:28px;color:#fff;'>
  <div style='font-size:0.75rem;font-weight:600;text-transform:uppercase;
              letter-spacing:.15em;opacity:.8;margin-bottom:10px;'>
    The Maharaja Sayajirao University of Baroda · Department of Statistics
  </div>
  <h1 style='font-size:2.4rem;font-weight:800;margin:0 0 12px;line-height:1.2;color:#fff;'>
    Consumer Usage & Adoption of<br>Q-Commerce Apps in Vadodara
  </h1>
  <p style='font-size:1rem;opacity:.85;max-width:680px;line-height:1.7;margin:0 0 20px;'>
    A comprehensive statistical study examining quick-commerce adoption behaviour,
    usage patterns, key drivers, and predictive models across 341 consumers in Vadodara, Gujarat.
  </p>
    <div style='width:1px;background:rgba(255,255,255,.3);align-self:stretch;margin:0 4px;'></div>
    <div style='display:flex;flex-direction:column;gap:6px;'>
      <div style='font-size:0.65rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:.12em;opacity:.6;margin-bottom:2px;'>Batch</div>
    <div style='display:flex;gap:10px;flex-wrap:wrap;align-items:center;'>
      {info_pills}
    </div>
    <div style='width:1px;background:rgba(255,255,255,.3);align-self:stretch;margin:0 4px;'></div>
    <div style='display:flex;flex-direction:column;gap:6px;'>
      <div style='font-size:0.65rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:.12em;opacity:.6;margin-bottom:2px;'>Team</div>
      <div style='display:flex;gap:8px;flex-wrap:wrap;'>{team_pills}</div>
    </div>
    <div style='display:flex;flex-direction:column;gap:6px;'>
      <div style='font-size:0.65rem;font-weight:700;text-transform:uppercase;
                  letter-spacing:.12em;opacity:.6;margin-bottom:2px;'>Mentor</div>
      <div style='width:fit-content;'>
        <span style='background:rgba(255,255,255,.28);border:1px solid rgba(255,255,255,.55);
                     border-radius:20px;padding:4px 14px;font-size:.78rem;font-weight:600;
                     letter-spacing:.02em;white-space:nowrap;'>Prof. Vipul Kalamkar</span>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
 
# ── Load data ──────────────────────────────────────────────────────────────────
df = load_raw()
users = get_users()
non_users_df = load_analysis()
nu = non_users_df[non_users_df["Adoption_Status"] == 0]
 
# ── KPI Row ────────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5)
kpi(k1, "341",  "Total Respondents", "Vadodara, 2024–25")
kpi(k2, "228",  "Q-Commerce Users",  "66.9% adoption rate", EMERALD)
kpi(k3, "113",  "Non-Users",         "33.1% non-adoption",  ROSE)
kpi(k4, f"{(df['Aware_QC']=='Yes').sum()}", "QC Aware", "Heard of Q-Commerce", SKY)
kpi(k5, "5",    "Objectives",        "Primary + Secondary", VIOLET)

 
st.markdown("<br>", unsafe_allow_html=True)
 
# ── What is Q-Commerce ─────────────────────────────────────────────────────────
left, right = st.columns([1.2, 1], gap="large")
 
with left:
    section("Introduction", "What is Q-Commerce and why does it matter?")
    st.markdown("""
    <div style='background:#fff;border:1px solid #E2E8F0;border-radius:16px;padding:24px;'>
      <p style='color:#374151;line-height:1.8;font-size:0.92rem;margin:0;'>
        <b>Q-Commerce (Quick Commerce)</b> is an ultra-fast e-commerce model that delivers
        groceries and daily essentials within <b>10–30 minutes</b>, powered by hyperlocal
        <i>dark stores</i> and AI-driven logistics. Unlike traditional e-commerce, Q-Commerce
        prioritises speed over selection, stocking only high-demand, high-turnover SKUs.
      </p>
      <div style='margin-top:18px;display:grid;grid-template-columns:1fr 1fr;gap:12px;'>
    """ + "".join([f"""        <div style='background:#F8FAFC;border-radius:10px;padding:12px 14px;'>
          <div style='font-size:1.2rem;'>{icon}</div>
          <div style='font-weight:600;font-size:0.82rem;color:#1E1E2E;margin:4px 0 2px;'>{title}</div>
          <div style='font-size:0.75rem;color:#64748B;'>{desc}</div>
        </div>"""
        for icon,title,desc in [
            ("🏪","Dark Stores","Micro-fulfilment centres in urban neighbourhoods"),
            ("⚡","10–30 Min Delivery","Speed as the core value proposition"),
            ("🤖","AI-Driven Logistics","Real-time routing & inventory optimisation"),
            ("📱","App-First","Entirely mobile-driven ordering experience"),
        ]]) + """
      </div>
    </div>
    """, unsafe_allow_html=True)
 
    section("Why We Chose This Topic")
    st.markdown("""
    <div style='background:#fff;border:1px solid #E2E8F0;border-radius:16px;padding:20px 24px;'>
    """ + "".join([f"""    <div style='display:flex;align-items:flex-start;gap:12px;padding:8px 0;
                border-bottom:1px solid #F1F5F9;'>
      <span style='font-size:1.1rem;'>{icon}</span>
      <div>
        <div style='font-weight:600;font-size:0.95rem;color:#1E1E2E;'>{title}</div>
        <div style='font-size:0.78rem;color:#64748B;margin-top:2px;'>{desc}</div>
      </div>
    </div>"""
    for icon,title,desc in [
        ("1.","Explosive Market Growth","India's Q-Commerce sector growing at 40–50% CAGR"),
        ("2.","Vadodara — An Underexplored Market","Tier-2 city dynamics differ from metros"),
        ("3.","Statistical Gap","Limited academic work on consumer adoption in Tier-2 cities"),
        ("4.","Policy Relevance","Findings can guide platform expansion strategies"),
    ]]) + """
    </div>""", unsafe_allow_html=True)
 
with right:
    section("Sample Composition")
    # Adoption donut
    fig_donut = go.Figure(go.Pie(
        labels=["Q-Commerce Users","Non-Users"],
        values=[228, 113], hole=0.65,
        marker_colors=[INDIGO, "#E2E8F0"],
        textinfo="label+percent",
        textfont=dict(size=12),
        hovertemplate="%{label}: %{value}<extra></extra>"
    ))
    fig_donut.update_layout(
        **{k:v for k,v in PLOTLY_LAYOUT.items() if k not in ["xaxis","yaxis"]},
        height=350, showlegend=False,
        annotations=[dict(text="<b>66.9%</b><br>Adoption", x=0.5, y=0.5,
                          font=dict(size=15, color="#1E1E2E"), showarrow=False)]
    )
    st.plotly_chart(fig_donut, use_container_width=True)
 
    section("Apps Awareness vs Usage")
    apps    = ["Blinkit", "Zepto", "Swiggy Instamart"]
    aware   = [
        (df["App_usex1"] == "Blinkit").sum() + (df["App_usex2"] == " Blinkit").sum(),
        (df["App_usex2"] == " Zepto").sum() + (df["App_usex1"] == "Zepto").sum(),
        (df["App_usex3"] == " Swiggy Instamart").sum(),
    ]
    usage   = [
        (users["App_Used"] == "Blinkit").sum(),
        (users["App_Used"] == "Zepto").sum(),
        (users["App_Used"] == "Swiggy Instamart").sum(),
    ]
    fig_app = go.Figure()
    fig_app.add_trace(go.Bar(name="Aware", x=apps, y=aware,
                              marker_color=hex_alpha(INDIGO, 0.4), text=aware, textposition="outside"))
    fig_app.add_trace(go.Bar(name="Primary Users", x=apps, y=usage,
                              marker_color=INDIGO, text=usage, textposition="outside"))
    fig_app.update_layout(**PLOTLY_LAYOUT, height=340, barmode="group",
                           title=dict(text="Awareness vs. Active Usage", font=dict(size=13)))
    fig_app.update_xaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False)
    fig_app.update_yaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False)
    st.plotly_chart(fig_app, use_container_width=True)
 
  

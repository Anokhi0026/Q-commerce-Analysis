import streamlit as st
import plotly.graph_objects as go
from utils import *

st.set_page_config("Q-Commerce Vadodara", "⚡", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#FAFAFA;}
section[data-testid="stSidebar"]{background:#FFFFFF;border-right:1px solid #E2E8F0;}
</style>""", unsafe_allow_html=True)

sidebar()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='background:linear-gradient(135deg,#4F46E5 0%,#7C3AED 100%);
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
  <div style='display:flex;gap:12px;flex-wrap:wrap;'>
    {''.join(f'<span style="background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.3);border-radius:20px;padding:4px 14px;font-size:.8rem;">{t}</span>'
    for t in ['n = 341 respondents','Vadodara, Gujarat','MSc Statistics 2024–25','Prof. Vipul Kalamkar'])}
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df = load_raw()
users = get_users()
non_users_df = load_analysis()
nu = non_users_df[non_users_df["Adoption_Status"] == 0]

if "Aware_QC" in df.columns:
    aware = (df["Aware_QC"] == "Yes").sum()
else:
    aware = 0

# ── KPI Row ────────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)
kpi(k1, "341",  "Total Respondents", "Vadodara, 2024–25")
kpi(k2, "228",  "Q-Commerce Users",  "66.9% adoption rate", EMERALD)
kpi(k3, "113",  "Non-Users",         "33.1% non-adoption",  ROSE)
kpi(k4, f"{(df['Aware_QC']=='Yes').sum()}", "QC Aware", "Heard of Q-Commerce", SKY)
kpi(k5, "5",    "Objectives",        "Primary + Secondary", VIOLET)
kpi(k6, "23",   "Statistical Tests", "Across all analyses", AMBER)

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
    """ + "".join([f"""
        <div style='background:#F8FAFC;border-radius:10px;padding:12px 14px;'>
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
    """ + "".join([f"""
    <div style='display:flex;align-items:flex-start;gap:12px;padding:8px 0;
                border-bottom:1px solid #F1F5F9;'>
      <span style='font-size:1.1rem;'>{icon}</span>
      <div>
        <div style='font-weight:600;font-size:0.85rem;color:#1E1E2E;'>{title}</div>
        <div style='font-size:0.78rem;color:#64748B;margin-top:2px;'>{desc}</div>
      </div>
    </div>"""
    for icon,title,desc in [
        ("📈","Explosive Market Growth","India's Q-Commerce sector growing at 40–50% CAGR"),
        ("🏙️","Vadodara — An Underexplored Market","Tier-2 city dynamics differ from metros"),
        ("🔬","Statistical Gap","Limited academic work on consumer adoption in Tier-2 cities"),
        ("🎯","Policy Relevance","Findings can guide platform expansion strategies"),
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
        height=250, showlegend=False,
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
                              marker_color=INDIGO+"99", text=aware, textposition="outside"))
    fig_app.add_trace(go.Bar(name="Primary Users", x=apps, y=usage,
                              marker_color=INDIGO, text=usage, textposition="outside"))
    fig_app.update_layout(**PLOTLY_LAYOUT, height=240, barmode="group",
                           title=dict(text="Awareness vs. Active Usage", font=dict(size=13)))
    st.plotly_chart(fig_app, use_container_width=True)

    section("Team")
    for name, role in [
        ("Anokhi Desai","Team Member"), ("Ritika Sharma","Team Member"),
        ("Sanjana Kumari","Team Member"), ("Vedant Ghaisas","Team Member"),
        ("Prof. Vipul Kalamkar","Mentor"),
    ]:
        color = VIOLET if "Prof" in role else INDIGO
        st.markdown(f"""
        <div style='background:#F8FAFC;border-radius:8px;padding:8px 12px;
                    margin-bottom:4px;display:flex;align-items:center;gap:10px;'>
          <div style='width:28px;height:28px;border-radius:50%;background:{color};
                      display:flex;align-items:center;justify-content:center;
                      color:#fff;font-size:.8rem;font-weight:700;flex-shrink:0;'>
            {name[0]}
          </div>
          <div>
            <div style='font-size:.82rem;font-weight:600;color:#1E1E2E;'>{name}</div>
            <div style='font-size:.72rem;color:#64748B;'>{role}</div>
          </div>
        </div>""", unsafe_allow_html=True)

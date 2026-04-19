import streamlit as st
import plotly.graph_objects as go
import pandas as pd, numpy as np
from utils import *

st.set_page_config("Summary", "✨", layout="wide")
st.markdown("<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');html,body,[class*='css']{font-family:'Inter',sans-serif;}.stApp{background:#FAFAFA;}section[data-testid='stSidebar']{background:#FFFFFF;border-right:1px solid #E2E8F0;}</style>",unsafe_allow_html=True)
sidebar()

page_header("Research Summary", "Key Findings & Conclusions",
            "A consolidated view of all analytical results across five objectives — from market structure to predictive modelling.")

# ── Master scorecard ───────────────────────────────────────────────────────────
st.markdown(f"""
<div style='background:linear-gradient(135deg,#1E1E2E,#2D2D4E);border-radius:20px;
            padding:36px 40px;margin-bottom:28px;color:#fff;'>
  <div style='font-size:.7rem;text-transform:uppercase;letter-spacing:.15em;opacity:.65;margin-bottom:10px;'>
    Research Scorecard · Q-Commerce Vadodara · n = 341
  </div>
  <div style='display:grid;grid-template-columns:repeat(6,1fr);gap:20px;'>
    {''.join(f"""
    <div style='text-align:center;'>
      <div style='font-size:1.8rem;font-weight:800;color:{col};font-family:monospace;'>{val}</div>
      <div style='font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;opacity:.7;margin-top:4px;'>{lbl}</div>
    </div>""" for val,lbl,col in [
      ("66.9%","Adoption Rate","#818CF8"),
      ("62%","Blinkit Share","#F472B6"),
      ("V=0.588","Age Effect (Strong)","#34D399"),
      ("α=0.768","Scale Reliability","#FBBF24"),
      ("AUC>0.75","Model Accuracy","#60A5FA"),
      ("3.82/5","Mean Satisfaction","#A78BFA"),
    ])}
  </div>
</div>""", unsafe_allow_html=True)

# ── Objective-by-objective findings ───────────────────────────────────────────
section("Findings by Objective")

obj_findings = [
    (INDIGO, "📱", "Obj 1", "Most-Used Apps",
     [("Blinkit leads with 62.3% share","142 of 228 users identify Blinkit as primary app — near-oligopolistic market structure in Vadodara"),
      ("Zepto & Swiggy Instamart form distant 2nd tier","Together ~35% of users; significant awareness-to-usage gap remains for all three"),
      ("High brand awareness, lower active usage","Platforms have achieved awareness but not full conversion among aware consumers")]),
    (ROSE, "🔗", "Obj 2", "Adoption Patterns",
     [("Age is the strongest demographic predictor","Cramér's V = 0.588 (Strong) — χ² = 117.806, p < 0.001. Younger cohorts adopt significantly more"),
      ("Education shows moderate association","V = 0.407 — Higher education → higher digital literacy → higher adoption"),
      ("Gender has no effect whatsoever","χ² = 0.457, p = 0.79 — adoption is gender-neutral in Vadodara, challenging common assumptions")]),
    (EMERALD, "📊", "Obj 3", "Usage Behavior",
     [("Evening + Night = 63% of all orders","End-of-day top-up shopping dominates; platforms should optimise dark-store staffing for 5–11 PM"),
      ("₹200–₹400 is the dominant order bracket","Q-Commerce is used for small, targeted replenishment — not bulk weekly grocery runs"),
      ("Satisfaction-loyalty-advocacy chain confirmed","Strong Spearman ρ between satisfaction, continuity, and recommendation scores (all significant)")]),
    (AMBER, "🔍", "Obj 4", "Key Drivers",
     [("Convenience dominates; price is secondary","Time Saving, Delivery Speed, and Lifestyle Fit rank highest — discounts show highest response variance"),
      ("Lack of Awareness is #1 non-adoption barrier","29.2% of non-users — not rejection, but simple non-exposure. Reachable via targeted campaigns"),
      ("EFA reveals multi-dimensional adoption structure","Factor analysis extracts latent dimensions: Convenience & Lifestyle | Price Sensitivity | Quality & Trust")]),
    (VIOLET, "🤖", "Obj 5", "Predictive Models",
     [("Demographics predict adoption (AUC > 0.75)","Binary logistic regression achieves meaningful discrimination; Nagelkerke R² ≈ 0.35"),
      ("Age is the dominant single predictor","Each older age group shows significantly lower Odds Ratios vs 18–25 reference — adoption is youth-driven"),
      ("Model is well-calibrated (Hosmer-Lemeshow)","Predicted probabilities match observed rates across risk deciles — model is practically usable for targeting")]),
]

for color, icon, tag, title, findings in obj_findings:
    st.markdown(f"""
    <div style='background:#fff;border:1px solid #E2E8F0;border-radius:16px;
                padding:20px 24px;margin-bottom:14px;'>
      <div style='display:flex;align-items:center;gap:12px;margin-bottom:14px;'>
        <div style='width:40px;height:40px;border-radius:10px;background:{color}15;
                    display:flex;align-items:center;justify-content:center;font-size:1.2rem;flex-shrink:0;'>{icon}</div>
        <div>
          <span style='font-size:.68rem;font-weight:700;color:{color};text-transform:uppercase;
                       letter-spacing:.1em;background:{color}12;padding:2px 8px;border-radius:20px;'>{tag}</span>
          <span style='font-size:.95rem;font-weight:700;color:#1E1E2E;margin-left:8px;'>{title}</span>
        </div>
      </div>
      <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:10px;'>
    """ + "".join([f"""
        <div style='background:{color}06;border:1px solid {color}20;border-radius:8px;padding:10px 12px;'>
          <div style='font-weight:600;font-size:.8rem;color:#1E1E2E;margin-bottom:4px;'>{f[0]}</div>
          <div style='font-size:.73rem;color:#475569;line-height:1.5;'>{f[1]}</div>
        </div>""" for f in findings]) + """
      </div>
    </div>""", unsafe_allow_html=True)

# ── Visual summary charts ──────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
section("Visual Summary of All Results")

df    = load_raw()
users = get_users()

c1, c2, c3 = st.columns(3, gap="large")

with c1:
    # Chi-square effect sizes
    vars_ = ["Age","Education","Occupation","Income","Gender"]
    vs_   = [0.588, 0.407, 0.271, 0.246, 0.037]
    colors_ = [INDIGO if v>0.3 else (AMBER if v>0.1 else SLATE) for v in vs_]
    fig = go.Figure(go.Bar(
        y=vars_[::-1], x=vs_[::-1], orientation="h",
        marker_color=colors_[::-1],
        text=[f"V={v:.3f}" for v in vs_[::-1]], textposition="outside",
        hovertemplate="%{y}: Cramér's V=%{x}<extra></extra>"
    ))
    for t,c in [(0.1,SLATE),(0.3,AMBER),(0.5,ROSE)]:
        fig.add_vline(x=t, line_dash="dot", line_color=c, opacity=0.5)
    fig.update_layout(**PLOTLY_LAYOUT, height=260,
                       xaxis=dict(title="Cramér's V",range=[0,0.7],gridcolor="#F1F5F9"),
                       title=dict(text="Obj 2: Demographic Effect Sizes",font=dict(size=12)))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    # Satisfaction scores
    sat_cols = ["Overall satisfaction with Q-Commerce Apps",
                "Likelihood of continuing usage in the future",
                "likely are you to recommend these apps to Others"]
    sat_lbls = ["Satisfaction","Continuity","Recommend"]
    means = [users[c].dropna().mean() for c in sat_cols]
    fig = go.Figure(go.Bar(
        x=sat_lbls, y=means,
        marker_color=[INDIGO, EMERALD, VIOLET],
        text=[f"{m:.2f}/5" for m in means], textposition="outside",
        hovertemplate="%{x}: μ=%{y:.2f}<extra></extra>"
    ))
    fig.add_hline(y=3.5, line_dash="dot", line_color=AMBER,
                  annotation_text="Above average threshold (3.5)",
                  annotation_font=dict(size=9,color=AMBER))
    fig.update_layout(**PLOTLY_LAYOUT, height=260,
                       yaxis=dict(title="Mean Score",range=[0,5.2],gridcolor="#F1F5F9"),
                       title=dict(text="Obj 3: Mean Satisfaction Scores",font=dict(size=12)))
    st.plotly_chart(fig, use_container_width=True)

with c3:
    # Non-adoption reasons
    reasons_lbl = ["Lack of Awareness","Prefer Local","App Discomfort",
                   "Trust Issues","Quality Concern","High Charges","No Need"]
    reasons_pct = [29.2, 24.8, 12.4, 11.5, 10.6, 8.9, 7.1]
    fig = go.Figure(go.Bar(
        y=reasons_lbl[::-1], x=reasons_pct[::-1], orientation="h",
        marker_color=[ROSE if i < 2 else AMBER if i < 4 else SLATE
                      for i in range(len(reasons_pct)-1, -1, -1)],
        text=[f"{v}%" for v in reasons_pct[::-1]], textposition="outside",
        hovertemplate="%{y}: %{x}%<extra></extra>"
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=260,
                       xaxis=dict(title="% of Non-Users",gridcolor="#F1F5F9"),
                       title=dict(text="Obj 4: Non-Adoption Barriers",font=dict(size=12)))
    st.plotly_chart(fig, use_container_width=True)

# ── Recommendations ────────────────────────────────────────────────────────────
section("Strategic Recommendations")

recs = [
    ("🎯 Target Young Educated Users First",
     "Age 18–33 with graduate+ education represents the highest-probability adoption segment. "
     "Platform growth in Vadodara should concentrate acquisition spend here before expanding to other segments.",
     INDIGO),
    ("📣 Awareness Campaign for Non-Users",
     "29.2% of non-users simply haven't been reached. Simple, visible offline+online campaigns in Vadodara's "
     "wards (especially North and East zones) can convert this reachable non-adopter segment.",
     ROSE),
    ("⏰ Optimise for Evening Peak (5 PM–11 PM)",
     "63% of orders come in evening/night windows. Dark-store staffing, delivery partner availability, "
     "and promotional push notifications should be concentrated in this window.",
     EMERALD),
    ("💳 Build Digital Payment Trust for CoD Users",
     "32% of users still prefer Cash on Delivery. Graduated incentives (first UPI order discount, "
     "wallet cashback) can shift this segment toward digital payments without losing them.",
     AMBER),
    ("🏪 Simplify Onboarding for Older Users",
     "The logistic model confirms lower adoption odds for every age group above 18–25. "
     "A simplified app interface, vernacular language support, and in-store demo kiosks "
     "can address both the digital literacy and trust barriers simultaneously.",
     VIOLET),
    ("📦 Double Down on Daily Essentials",
     "Groceries (72.8%) and Daily Essentials (56.6%) drive most orders. "
     "Dark-store assortment should be hyper-optimised for these categories in Vadodara's specific market.",
     SKY),
]

c1, c2 = st.columns(2, gap="large")
for i, (title, text, color) in enumerate(recs):
    col = c1 if i % 2 == 0 else c2
    col.markdown(f"""
    <div style='background:#fff;border:1px solid #E2E8F0;border-left:4px solid {color};
                border-radius:0 12px 12px 0;padding:14px 18px;margin-bottom:10px;
                box-shadow:0 1px 3px rgba(0,0,0,.04);'>
      <div style='font-weight:700;font-size:.88rem;color:#1E1E2E;margin-bottom:5px;'>{title}</div>
      <div style='font-size:.78rem;color:#475569;line-height:1.6;'>{text}</div>
    </div>""", unsafe_allow_html=True)

# ── Limitations ────────────────────────────────────────────────────────────────
section("Study Limitations & Future Scope")
lim_c1, lim_c2 = st.columns(2, gap="large")
with lim_c1:
    st.markdown(f"""
    <div style='background:#FFF7ED;border:1px solid #FED7AA;border-radius:12px;padding:18px 20px;'>
      <div style='font-weight:700;font-size:.9rem;color:#92400E;margin-bottom:10px;'>⚠️ Limitations</div>
      {''.join(f'<div style="font-size:.78rem;color:#78350F;padding:3px 0;">• {l}</div>' for l in [
        "Convenience sampling — may not perfectly represent all of Vadodara",
        "Cross-sectional design — cannot capture adoption over time",
        "Self-reported data — social desirability bias possible",
        "Non-user factor scores imputed with mean (disclosed in Obj 5)",
        "Cluster labels are interpretive, not definitively validated",
      ])}
    </div>""", unsafe_allow_html=True)
with lim_c2:
    st.markdown(f"""
    <div style='background:#F0FDF4;border:1px solid #BBF7D0;border-radius:12px;padding:18px 20px;'>
      <div style='font-weight:700;font-size:.9rem;color:#14532D;margin-bottom:10px;'>🔭 Future Scope</div>
      {''.join(f'<div style="font-size:.78rem;color:#15803D;padding:3px 0;">• {l}</div>' for l in [
        "Longitudinal study to track adoption over 12–18 months",
        "Expand to other Tier-2 cities for cross-city comparison",
        "Include platform-side data (dark-store metrics, delivery times)",
        "Deep-dive into each consumer segment with qualitative interviews",
        "Add price elasticity and willingness-to-pay models",
      ])}
    </div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='background:linear-gradient(135deg,#1E1E2E,#2D2D4E);border-radius:16px;
            padding:28px 32px;margin-top:28px;text-align:center;color:#fff;'>
  <div style='font-size:1rem;font-weight:700;'>
    A Statistical Study on Consumer Usage and Adoption of Q-Commerce Apps in Vadodara
  </div>
  <div style='font-size:.8rem;opacity:.65;margin-top:8px;'>
    Department of Statistics · The Maharaja Sayajirao University of Baroda ·
    Anokhi Desai · Ritika Sharma · Sanjana Kumari · Vedant Ghaisas · Mentor: Prof. Vipul Kalamkar
  </div>
</div>""", unsafe_allow_html=True)

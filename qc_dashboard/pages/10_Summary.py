import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import *

st.set_page_config("Summary", "✨", layout="wide")
st.session_state["current_page"] = "pages/10_Summarys.py"
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*='css']{font-family:'Inter',sans-serif;}
.stApp{background:#FAFAFA;}
section[data-testid='stSidebar']{background:#FFFFFF;border-right:1px solid #E2E8F0;}
</style>""", unsafe_allow_html=True)

sidebar()
page_header("Research Summary", "Complete Findings & Conclusions",
            "A consolidated view of all analytical results — 5 objectives, cluster analysis, "
            "correspondence analysis — with consumer insights and study limitations.")

# ── Master scorecard ───────────────────────────────────────────────────────────
st.markdown(f"""
<div style='background:linear-gradient(135deg,#1E1E2E,#2D2D4E);border-radius:20px;
            padding:36px 40px;margin-bottom:28px;'>
  <div style='font-size:.68rem;text-transform:uppercase;letter-spacing:.15em;color:#94A3B8;margin-bottom:14px;'>
    Research Scorecard · Q-Commerce Vadodara · n=341
  </div>
  <div style='display:grid;grid-template-columns:repeat(7,1fr);gap:16px;'>
    {"".join(f'''    <div style='text-align:center;'>
      <div style='font-size:1.6rem;font-weight:800;color:{col};font-family:monospace;'>{val}</div>
      <div style='font-size:.62rem;text-transform:uppercase;letter-spacing:.08em;color:#64748B;margin-top:4px;'>{lbl}</div>
    </div>''' for val,lbl,col in [
      ("66.9%","Adoption Rate","#818CF8"),("62%","Blinkit Share","#F472B6"),
      ("V=0.588","Age Effect","#34D399"),("α=0.768","Scale Reliability","#FBBF24"),
      ("AUC>0.75","Model Accuracy","#60A5FA"),("3.82/5","Mean Satisfaction","#A78BFA"),
      ("3","Segments","#4ADE80"),
    ])}
  </div>
</div>""", unsafe_allow_html=True)

# ── Complete findings by area ──────────────────────────────────────────────────
section("All Findings — Objective by Objective")

all_findings = [
    (INDIGO,"📱","Objective 1","Most-Used Apps",[
        ("Blinkit leads with 62.3% market share","142 of 228 users — near-oligopolistic structure; first-mover advantage"),
        ("Two-tier market: Zepto + Swiggy = ~35%","Neither can challenge Blinkit's dominance in the short term"),
        ("High awareness, lower active usage","Significant awareness-to-usage gap exists for all platforms"),
    ]),
    (ROSE,"🔗","Objective 2","Adoption Patterns",[
        ("Age is the strongest predictor (V=0.588, Strong)","χ²=117.8, p<0.001 — each older age band shows lower adoption"),
        ("Education: Moderate effect (V=0.407)","Higher education → digital literacy → adoption"),
        ("Gender: No effect (V=0.037, p=0.79)","Male and female adopt equally"),
    ]),
    (EMERALD,"📊","Objective 3","Usage Behavior",[
        ("Evening+Night = 63% of orders","End-of-day top-up shopping; peak ordering window 5–11 PM"),
        ("₹200–₹400 dominant order bracket","Small, targeted replenishment — not bulk grocery runs"),
        ("Satisfaction-loyalty-advocacy chain confirmed","Strong Spearman ρ among satisfaction, continuity, recommendation"),
    ]),
    (AMBER,"🔍","Objective 4","Key Drivers",[
        ("Cronbach's α=0.768 (Acceptable reliability)","10-item scale is valid; EFA analyses are statistically justified"),
        ("Convenience dominates (Time Saving, Lifestyle Fit rank top)","Price incentives matter less and show more response variance"),
        ("Lack of Awareness = #1 non-adoption barrier (29.2%)","Not rejection — non-exposure"),
    ]),
    (VIOLET,"🤖","Objective 5","Predictive Models",[
        ("Logistic Regression: AUC=0.847, Accuracy=80.1%","Age_40+ (OR≈0.074, p<0.001) and Education (OR≈9×) are dominant"),
        ("Gender suppressor effect detected","Multivariate model reveals gender effect masked in bivariate chi-square"),
        ("RF CV AUC highest; all 3 models converge on same findings","Age + Education dominate across all model types"),
    ]),
    (SKY,"🧩","Cluster Analysis","3 Consumer Segments",[
        ("Active Engagers — highest attitudes, most satisfied","Power users with strongest loyalty and recommendation scores"),
        ("Passive Users — below average on all items","Occasional users with weaker engagement and lower satisfaction"),
        ("Convenience Purists — time-saving and urgent needs driven","Emergency top-up users; less influenced by discounts"),
    ]),
    ("#DB2777","📍","Correspondence Analysis","Categorical Associations",[
        ("CA1: Blinkit↔Young (18–25), Zepto↔26–33 segment","Clear platform-age segmentation in consumer choice"),
        ("CA2: Night/Midnight↔18–25, Morning↔40+","Age-stratified delivery time preferences"),
        ("CA4: Students↔UPI, Retired↔CoD","Payment method is strongly occupation-driven"),
    ]),
]

for color, icon, tag, title, finds in all_findings:
    st.markdown(f"""
    <div style='background:#fff;border:1px solid #E2E8F0;border-radius:14px;
                padding:18px 22px;margin-bottom:12px;'>
      <div style='display:flex;align-items:center;gap:10px;margin-bottom:12px;'>
        <div style='width:38px;height:38px;border-radius:9px;background:{color}15;
                    display:flex;align-items:center;justify-content:center;font-size:1.1rem;'>{icon}</div>
        <div>
          <span style='font-size:.67rem;font-weight:700;color:{color};text-transform:uppercase;
                       letter-spacing:.1em;background:{color}12;padding:2px 8px;border-radius:20px;'>{tag}</span>
          <span style='font-size:.92rem;font-weight:700;color:#1E1E2E;margin-left:8px;'>{title}</span>
        </div>
      </div>
      <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:8px;'>
    """ + "".join([f"""        <div style='background:{color}06;border:1px solid {color}18;border-radius:8px;padding:10px 12px;'>
          <div style='font-weight:600;font-size:.78rem;color:#1E1E2E;margin-bottom:3px;'>{f[0]}</div>
          <div style='font-size:.72rem;color:#475569;line-height:1.5;'>{f[1]}</div>
        </div>""" for f in finds]) + """
      </div>
    </div>""", unsafe_allow_html=True)

# ── Visual Summary Charts ──────────────────────────────────────────────────────
st.markdown("<br>",unsafe_allow_html=True)
section("Visual Summary of Key Quantitative Results")

df    = load_raw()
users = get_users()

c1,c2,c3 = st.columns(3, gap="large")
with c1:
    vars_ = ["Age","Education","Occupation","Income","Gender"]
    vs_   = [0.588,0.407,0.271,0.246,0.037]
    clrs_ = [INDIGO if v>0.3 else (AMBER if v>0.1 else SLATE) for v in vs_]
    fig   = go.Figure(go.Bar(
        y=vars_[::-1], x=vs_[::-1], orientation="h", marker_color=clrs_[::-1],
        text=[f"V={v:.3f}" for v in vs_[::-1]], textposition="outside"))
    for t,c in [(0.1,SLATE),(0.3,AMBER),(0.5,ROSE)]:
        fig.add_vline(x=t, line_dash="dot", line_color=c, opacity=0.5)
    fig.update_layout(**PLOTLY_LAYOUT, height=240, title=dict(text="Obj 2: Effect Sizes",font=dict(size=12)))
    fig.update_xaxes(title="Cramér's V", range=[0,0.7], gridcolor="#F1F5F9")
    fig.update_yaxes(gridcolor="#F1F5F9")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    sat_cols = ["Overall satisfaction with Q-Commerce Apps",
                "Likelihood of continuing usage in the future",
                "likely are you to recommend these apps to Others"]
    means = [users[c].dropna().mean() for c in sat_cols]
    fig   = go.Figure(go.Bar(
        x=["Satisfaction","Continuity","Recommend"], y=means,
        marker_color=[INDIGO,EMERALD,VIOLET],
        text=[f"{m:.2f}/5" for m in means], textposition="outside"))
    fig.add_hline(y=3.5, line_dash="dot", line_color=AMBER, opacity=0.7,
                  annotation_text="≥3.5 target", annotation_font=dict(size=9))
    fig.update_layout(**PLOTLY_LAYOUT, height=240, title=dict(text="Obj 3: Satisfaction Scores",font=dict(size=12)))
    fig.update_yaxes(title="Mean Score", range=[0,5.2], gridcolor="#F1F5F9")
    fig.update_xaxes(gridcolor="#F1F5F9")
    st.plotly_chart(fig, use_container_width=True)

with c3:
    models  = ["Logistic\nRegression","Decision\nTree","Random\nForest"]
    aucs_   = [0.847, 0.823, 0.820]
    fig     = go.Figure(go.Bar(
        x=models, y=aucs_, marker_color=[INDIGO,VIOLET,EMERALD],
        text=[f"AUC={v:.3f}" for v in aucs_], textposition="outside"))
    fig.add_hline(y=0.5, line_dash="dash", line_color=SLATE, opacity=0.5,
                  annotation_text="Random classifier", annotation_font=dict(size=9))
    fig.update_layout(**PLOTLY_LAYOUT, height=240, title=dict(text="Obj 5: Model AUC Comparison",font=dict(size=12)))
    fig.update_yaxes(title="AUC", range=[0.4,0.95], gridcolor="#F1F5F9")
    fig.update_xaxes(gridcolor="#F1F5F9")
    st.plotly_chart(fig, use_container_width=True)

# ── Key Consumer Insights ──────────────────────────────────────────────────────
section("Key Consumer Insights")
st.markdown("""
<div style='background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;
            padding:10px 16px;margin-bottom:16px;font-size:.8rem;color:#1D4ED8;'>
  These insights describe <b>who consumers are, what drives their behaviour, and what barriers they face</b>
  — derived directly from the statistical findings of this study.
</div>""", unsafe_allow_html=True)

insights = [
    ("⏰ Convenience Is the Core Adoption Motive",
     "Time-saving and lifestyle compatibility are the top-ranked reasons consumers adopt Q-Commerce "
     "— ranked above discounts and product variety. Consumers primarily use it to avoid disrupting "
     "their work or study schedule, not for price benefits.",
     INDIGO),
    ("🎓 Digital Literacy Shapes Adoption More Than Income",
     "Education (V=0.407) predicts adoption more strongly than household income (V=0.246). "
     "Consumers with higher education are more confident navigating apps and trusting digital payments "
     "— suggesting literacy is a larger barrier than affordability for non-adopters.",
     EMERALD),
    ("🧓 Older Consumers Have Distinct and Real Barriers",
     "Consumers aged 40+ are dramatically less likely to adopt (OR≈0.074). Their hesitation stems "
     "from app navigation discomfort, data privacy concerns, and comfort with offline routines — "
     "not from indifference to fast delivery as a concept.",
     ROSE),
    ("💳 CoD Preference Reflects Trust Deficit, Not Inertia",
     "32.5% of Q-Commerce users still prefer Cash on Delivery. This is concentrated among homemakers "
     "and retired respondents who report lower trust in digital payment systems — indicating a genuine "
     "confidence gap rather than simple habit.",
     AMBER),
    ("🛒 Consumers Use Q-Commerce for Top-Ups, Not Bulk Shopping",
     "72.4% of orders are below ₹400 and categories are dominated by groceries, snacks, and daily "
     "essentials. Consumers position Q-Commerce as a supplement to planned shopping — fulfilling "
     "immediate, small-basket needs when something runs out unexpectedly.",
     VIOLET),
    ("😐 A Large Segment Is Retained but Not Habitual",
     "Cluster analysis identifies a 'Passive Users' segment with below-average attitude and satisfaction "
     "scores. These consumers order occasionally and only under specific circumstances — they are "
     "not dissatisfied enough to leave, but have not integrated Q-Commerce into their daily routine.",
     SKY),
]

c1,c2 = st.columns(2, gap="large")
for i,(title,text,color) in enumerate(insights):
    col = c1 if i%2==0 else c2
    col.markdown(f"""
    <div style='background:#fff;border:1px solid #E2E8F0;border-left:4px solid {color};
                border-radius:0 12px 12px 0;padding:14px 18px;margin-bottom:10px;'>
      <div style='font-weight:700;font-size:.85rem;color:#1E1E2E;margin-bottom:4px;'>{title}</div>
      <div style='font-size:.78rem;color:#475569;line-height:1.6;'>{text}</div>
    </div>""", unsafe_allow_html=True)

# ── Statistical Methods Used ───────────────────────────────────────────────────
section("Complete List of Statistical Methods Used")
methods = [
    ("Descriptive Analysis","Frequency distributions, cross-tabs, multi-response analysis"),
    ("Shapiro-Wilk Normality Test","Age normality check — justifies non-parametric tests"),
    ("Chi-Square Test of Independence","20 pairs across Obj 2 & 3; demographic × adoption/behavior"),
    ("Cramér's V (bias-corrected)","Effect size for all chi-square tests"),
    ("Kruskal-Wallis H Test","Non-parametric ANOVA for satisfaction × demographics, barrier × age"),
    ("Dunn's Post-Hoc (Bonferroni)","Pairwise comparisons after significant Kruskal-Wallis"),
    ("Spearman Rank Correlation","Satisfaction ↔ Continuity ↔ Recommendation"),
    ("Cronbach's Alpha + Item-Total r","Internal reliability of 10-item Likert scale"),
    ("KMO + Bartlett's Test","EFA pre-conditions check"),
    ("EFA — PCA + Varimax Rotation","Latent structure of adoption drivers"),
    ("Mann-Whitney U ","Users vs non-users; barrier gender differences"),
    ("K-Means Clustering + Elbow + Silhouette","Consumer segmentation (K=3)"),
    ("PCA Scatter Plot","Cluster visualisation in 2D"),
    ("Correspondence Analysis (5 biplots)","Categorical variable associations — biplot visualisation"),
    ("Binary Logistic Regression (statsmodels)","Wald tests, ORs, 95% Wald CIs, Nagelkerke R²"),
    ("Decision Tree + GridSearchCV","84 combinations, 5-fold CV, AUC scoring"),
    ("Random Forest + GridSearchCV","36 combinations, 5-fold CV, feature importance"),
    ("ROC / AUC","Model discrimination ability"),
]

c1,c2 = st.columns(2)
half  = len(methods)//2
for col_idx, (col_w, subset) in enumerate([(c1, methods[:half]),(c2, methods[half:])]):
    with col_w:
        for i,(name,desc) in enumerate(subset):
            num = i + 1 + half * col_idx
            col_w.markdown(f"""
            <div style='display:flex;gap:10px;align-items:flex-start;padding:7px 0;
                        border-bottom:1px solid #F1F5F9;'>
              <div style='min-width:22px;height:22px;border-radius:50%;background:{INDIGO};color:#fff;
                          font-size:.65rem;font-weight:700;display:flex;align-items:center;
                          justify-content:center;flex-shrink:0;margin-top:1px;'>{num}</div>
              <div>
                <div style='font-size:.8rem;font-weight:600;color:#1E1E2E;'>{name}</div>
                <div style='font-size:.73rem;color:#64748B;'>{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

# ── Limitations & Future Scope ─────────────────────────────────────────────────
section("Limitations & Future Scope")
lc1,lc2 = st.columns(2, gap="large")
with lc1:
    st.markdown(f"""
    <div style='background:#FFF7ED;border:1px solid #FED7AA;border-radius:12px;padding:18px;'>
      <div style='font-weight:700;font-size:.88rem;color:#92400E;margin-bottom:10px;'>⚠️ Limitations</div>
      {"".join(f'<div style="font-size:.78rem;color:#78350F;padding:3px 0;">• {l}</div>' for l in [
        "Convenience sampling — may not represent all of Vadodara equally",
        "Cross-sectional design — cannot track adoption dynamics over time",
        "Non-user factor scores imputed with user-sample mean in Obj 5",
        "CA requires `prince` library; implemented via NumPy SVD (same math)",
        "Cluster labels (Active/Passive/Purists) are interpretive, not definitive",
        "4 respondents excluded from Obj 5 due to missing Gender values",
      ])}
    </div>""", unsafe_allow_html=True)
with lc2:
    st.markdown(f"""
    <div style='background:#F0FDF4;border:1px solid #BBF7D0;border-radius:12px;padding:18px;'>
      <div style='font-weight:700;font-size:.88rem;color:#14532D;margin-bottom:10px;'>🔭 Future Scope</div>
      {"".join(f'<div style="font-size:.78rem;color:#15803D;padding:3px 0;">• {l}</div>' for l in [
        "Longitudinal study to track consumer adoption behaviour over 12–18 months",
        "Expand to other Tier-2 cities for cross-city consumer comparison (Surat, Rajkot)",
        "Qualitative interviews to understand consumer trust and privacy concerns in depth",
        "Focus groups with Passive Users and non-adopters to map unmet needs",
        "Consumer willingness-to-pay and price sensitivity modelling",
        "Social network analysis of referral and word-of-mouth patterns among Active Engagers",
      ])}
    </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div style='background:linear-gradient(135deg,#1E1E2E,#2D2D4E);border-radius:16px;
            padding:28px 32px;margin-top:24px;text-align:center;color:#fff;'>
  <div style='font-size:.95rem;font-weight:700;'>
    A Statistical Study on Consumer Usage and Adoption of Q-Commerce Applications in Vadodara
  </div>
  <div style='font-size:.78rem;opacity:.6;margin-top:8px;'>
    Department of Statistics · The Maharaja Sayajirao University of Baroda ·
    Anokhi Desai · Ritika Sharma · Sanjana Kumari · Vedant Ghaisas · Mentor: Prof. Vipul Kalamkar
  </div>
</div>""", unsafe_allow_html=True)

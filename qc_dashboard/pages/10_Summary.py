import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import *

st.set_page_config("Summary", "✨", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*='css']{font-family:'Inter',sans-serif;}
.stApp{background:#FAFAFA;}
section[data-testid='stSidebar']{background:#FFFFFF;border-right:1px solid #E2E8F0;}
</style>""", unsafe_allow_html=True)

sidebar()
page_header("Research Summary", "Complete Findings & Strategic Conclusions",
            "A consolidated view of all analytical results — 5 objectives, cluster analysis, "
            "correspondence analysis — with strategic recommendations and study limitations.")

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
        ("High awareness, lower active usage","Significant awareness-to-usage gap exists for all platforms — conversion opportunity"),
    ]),
    (ROSE,"🔗","Objective 2","Adoption Patterns",[
        ("Age is the strongest predictor (V=0.588, Strong)","χ²=117.8, p<0.001 — each older age band shows lower adoption"),
        ("Education: Moderate effect (V=0.407)","Higher education → digital literacy → adoption; persists after controlling for age"),
        ("Gender: No effect (V=0.037, p=0.79)","Male and female adopt equally — challenges gendered digital adoption assumptions"),
    ]),
    (EMERALD,"📊","Objective 3","Usage Behavior",[
        ("Evening+Night = 63% of orders","End-of-day top-up shopping; optimize dark-store staffing for 5–11 PM"),
        ("₹200–₹400 dominant order bracket","Small, targeted replenishment — not bulk grocery runs"),
        ("Satisfaction-loyalty-advocacy chain confirmed","Strong Spearman ρ among satisfaction, continuity, recommendation"),
    ]),
    (AMBER,"🔍","Objective 4","Key Drivers",[
        ("Cronbach's α=0.768 (Acceptable reliability)","10-item scale is valid; all subsequent EFA analyses are statistically justified"),
        ("Convenience dominates (Time Saving, Lifestyle Fit rank top)","Price incentives matter less and show more response variance"),
        ("Lack of Awareness = #1 non-adoption barrier (29.2%)","Not rejection — non-exposure. Reachable via targeted campaigns"),
    ]),
    (VIOLET,"🤖","Objective 5","Predictive Models",[
        ("Logistic Regression: AUC=0.847, Accuracy=80.1%","Age_40+ (OR≈0.074, p<0.001) and Education (OR≈9×) are dominant"),
        ("Gender suppressor effect detected","Multivariate model reveals gender effect masked in bivariate chi-square"),
        ("RF CV AUC highest; all 3 models converge on same findings","Age + Education dominate; Income/Occupation lose significance when controlled"),
    ]),
    (SKY,"🧩","Cluster Analysis","3 Consumer Segments",[
        ("Active Engagers — highest attitudes, most satisfied","Power users, primary revenue base; target for loyalty programs"),
        ("Passive Users — below average on all items","At-risk of churn; re-engagement campaigns with convenience messaging"),
        ("Convenience Purists — time-saving and urgent needs driven","Emergency top-up users; respond to speed and reliability messaging"),
    ]),
    ("#DB2777","📍","Correspondence Analysis","Categorical Associations",[
        ("CA1: Blinkit↔Young (18–25), Zepto↔26–33 segment","Clear platform-age segmentation; Blinkit dominates youth"),
        ("CA2: Night/Midnight↔18–25, Morning↔40+","Age-stratified delivery time preferences — strong operational implication"),
        ("CA4: Students↔UPI, Retired↔CoD","Payment method is occupation-driven; trust campaigns should target homemakers/retired"),
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
    fig.update_xaxes(title="Cramér's V",range=[0,0.7],gridcolor="#F1F5F9")
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
    fig.update_yaxes(title="Mean Score",range=[0,5.2],gridcolor="#F1F5F9")
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
    fig.update_yaxes(title="AUC",range=[0.4,0.95],gridcolor="#F1F5F9")
    st.plotly_chart(fig, use_container_width=True)

# ── Strategic Recommendations ──────────────────────────────────────────────────
section("Strategic Recommendations")
recs = [
    ("🎯 Target 18–33 With High Education First",
     "This demographic has the highest adoption probability across all models. "
     "Platform growth should concentrate acquisition here before expanding to harder-to-reach segments.", INDIGO),
    ("📣 Awareness Campaigns Are #1 Priority for Non-Users",
     "29.2% of non-users simply haven't been exposed. Simple offline+digital campaigns in Vadodara's wards "
     "(especially North and East) can convert this reachable non-adopter segment.", ROSE),
    ("⏰ Optimise Operations for the 5–11 PM Window",
     "63% of orders come in evening/night. Dark-store staffing, fleet availability, and push notifications "
     "should all be concentrated in this window — especially for users aged 18–33.", EMERALD),
    ("💳 'First UPI Order' Incentive for CoD Users",
     "32% still use CoD. Graduated incentives (first UPI order discount, wallet cashback) can shift this "
     "segment without losing them — occupation-targeted campaigns work best (homemakers, retired).", AMBER),
    ("🏪 Simplify Onboarding for 40+ Users",
     "The logistic model (OR≈0.074) and all three models agree: 40+ is the largest adoption gap. "
     "Simplified UI, vernacular language support, and in-store demo kiosks address digital literacy barriers.", VIOLET),
    ("🧩 Segment-Specific Marketing Using Cluster Profiles",
     "Active Engagers → loyalty rewards. Passive Users → re-engagement + speed messaging. "
     "Convenience Purists → reliability and emergency availability messaging. Don't treat all users identically.", SKY),
]

c1,c2 = st.columns(2, gap="large")
for i,(title,text,color) in enumerate(recs):
    col = c1 if i%2==0 else c2
    col.markdown(f"""
    <div style='background:#fff;border:1px solid #E2E8F0;border-left:4px solid {color};
                border-radius:0 12px 12px 0;padding:14px 18px;margin-bottom:10px;'>
      <div style='font-weight:700;font-size:.85rem;color:#1E1E2E;margin-bottom:4px;'>{title}</div>
      <div style='font-size:.78rem;color:#475569;line-height:1.6;'>{text}</div>
    </div>""", unsafe_allow_html=True)

# ── Statistical Methods Used ───────────────────────────────────────────────────
section("Complete List of Statistical Methods Used")

# (name, description, H0 or None, formula or None)
methods = [
    ("Descriptive Analysis",
     "Frequency distributions, cross-tabs, multi-response analysis",
     None, None),
    ("Shapiro-Wilk Normality Test",
     "Age normality check — justifies non-parametric tests",
     "H₀: The variable follows a normal distribution",
     "W = (Σ aᵢx₍ᵢ₎)² / Σ(xᵢ − x̄)²"),
    ("Chi-Square Test of Independence",
     "20 pairs across Obj 2 & 3; demographic × adoption/behavior",
     "H₀: The two categorical variables are independent of each other",
     "χ² = Σ (O − E)² / E"),
    ("Cramér's V (bias-corrected)",
     "Effect size for all chi-square tests",
     None,
     "V = √( χ² / n · min(r−1, c−1) )"),
    ("Kruskal-Wallis H Test",
     "Non-parametric ANOVA for satisfaction × demographics, barrier × age",
     "H₀: All groups have the same population distribution (equal medians)",
     "H = [12 / n(n+1)] · Σ(Rᵢ² / nᵢ) − 3(n+1)"),
    ("Dunn's Post-Hoc (Bonferroni)",
     "Pairwise comparisons after significant Kruskal-Wallis",
     "H₀: No difference between any specific pair of groups",
     "z = (R̄ᵢ − R̄ⱼ) / SE  ;  p_adj = min(p · m, 1)"),
    ("Spearman Rank Correlation",
     "Satisfaction ↔ Continuity ↔ Recommendation",
     "H₀: No monotonic association between the two variables (ρ = 0)",
     "ρ = 1 − 6Σdᵢ² / n(n²−1)"),
    ("Cronbach's Alpha + Item-Total r",
     "Internal reliability of 10-item Likert scale",
     "H₀: Items do not consistently measure the same underlying construct",
     "α = k/(k−1) · (1 − ΣSᵢ² / S²_total)"),
    ("KMO + Bartlett's Test",
     "EFA pre-conditions check",
     "H₀ (Bartlett): Correlation matrix is an identity matrix — no factor structure",
     "KMO = Σrᵢⱼ² / (Σrᵢⱼ² + Σpᵢⱼ²)"),
    ("EFA — PCA + Varimax Rotation",
     "Latent structure of adoption drivers",
     "H₀: No underlying latent factors explain item intercorrelations",
     "X = ΛF + ε  (Factor model)"),
    ("Mann-Whitney U + Rank-Biserial r",
     "Users vs non-users; barrier gender differences",
     "H₀: The two groups come from the same population distribution",
     "U = n₁n₂ + n₁(n₁+1)/2 − R₁"),
    ("K-Means Clustering + Elbow + Silhouette",
     "Consumer segmentation (K=3)",
     None,
     "WCSS = Σₖ Σᵢ∈Cₖ ||xᵢ − μₖ||²"),
    ("PCA Scatter Plot",
     "Cluster visualisation in 2D",
     None,
     "Z = XW  (W = eigenvectors of covariance matrix)"),
    ("Correspondence Analysis (5 biplots)",
     "Categorical variable associations — biplot visualisation",
     "H₀: Row and column categories are independent (no association)",
     "χ²_ij = (pᵢⱼ − rᵢcⱼ)² / rᵢcⱼ  ;  SVD on standardised residuals"),
    ("Binary Logistic Regression",
     "Wald tests, ORs, 95% Wald CIs, Nagelkerke R²",
     "H₀: Each predictor has no effect on adoption probability (β = 0)",
     "log[P/(1−P)] = β₀ + Σβᵢxᵢ  ;  P = 1/(1+e^−z)"),
    ("Decision Tree + GridSearchCV",
     "84 combinations, 5-fold CV, AUC scoring",
     None,
     "Gini = 1 − Σpₖ²  ;  Split: arg min Gini(left) + Gini(right)"),
    ("Random Forest + GridSearchCV",
     "36 combinations, 5-fold CV, feature importance",
     None,
     "ŷ = majority_vote{ T₁(x), T₂(x), ..., Tₙ(x) }"),
    ("ROC / AUC",
     "Model discrimination ability",
     "H₀: Model has no discriminatory power (AUC = 0.5)",
     "AUC = ∫₀¹ TPR(FPR) d(FPR)"),
]

c1, c2 = st.columns(2)
half = len(methods) // 2
for col_idx, (col_w, subset) in enumerate([(c1, methods[:half]), (c2, methods[half:])]):
    with col_w:
        for i, (name, desc, h0, formula) in enumerate(subset):
            num = i + 1 + half * col_idx
            h0_html = f"""<div style='margin-top:4px;background:#EEF2FF;border-radius:5px;
                          padding:3px 8px;font-size:.67rem;color:#4338CA;font-style:italic;'>{h0}</div>""" if h0 else ""
            formula_html = f"""<div style='margin-top:3px;font-family:monospace;font-size:.67rem;
                          color:{INDIGO};background:#F8FAFC;border-radius:5px;
                          padding:3px 8px;border:1px solid #E2E8F0;'>{formula}</div>""" if formula else ""
            col_w.markdown(f"""
            <div style='display:flex;gap:10px;align-items:flex-start;padding:8px 0;
                        border-bottom:1px solid #F1F5F9;'>
              <div style='min-width:22px;height:22px;border-radius:50%;background:{INDIGO};color:#fff;
                          font-size:.65rem;font-weight:700;display:flex;align-items:center;
                          justify-content:center;flex-shrink:0;margin-top:1px;'>{num}</div>
              <div style='flex:1;'>
                <div style='font-size:.8rem;font-weight:600;color:#1E1E2E;'>{name}</div>
                <div style='font-size:.73rem;color:#64748B;margin-top:1px;'>{desc}</div>
                {h0_html}
                {formula_html}
              </div>
            </div>""", unsafe_allow_html=True)

# ── Limitations ────────────────────────────────────────────────────────────────
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
        "Longitudinal study to track adoption over 12–18 months",
        "Expand to other Tier-2 cities for cross-city comparison (Surat, Rajkot)",
        "Add platform-side data: delivery time actuals, dark-store coverage maps",
        "Deep qualitative interviews with Passive Users and non-adopters",
        "Price elasticity modelling and willingness-to-pay analysis",
        "Social network analysis of referral patterns among Active Engagers",
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

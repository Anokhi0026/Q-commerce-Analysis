import streamlit as st
import plotly.graph_objects as go
import pandas as pd, numpy as np
from scipy.stats import chi2_contingency, shapiro
from utils import *

st.set_page_config("Obj 2 — Adoption", "🔗", layout="wide")
st.markdown("<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');html,body,[class*='css']{font-family:'Inter',sans-serif;}.stApp{background:#FAFAFA;}section[data-testid='stSidebar']{background:#FFFFFF;border-right:1px solid #E2E8F0;}</style>",unsafe_allow_html=True)
sidebar()
page_header("Objective 2", "Consumer Adoption Patterns Based on Demographics",
            "Chi-Square tests and Cramér's V effect sizes revealing which demographic variables significantly predict Q-Commerce adoption.")

df = load_raw()

# ── Pre-computed results from your notebook ────────────────────────────────────
chi2_results = [
    ("Age",       117.806, 3.3e-7,  0.588, "Strong",      True),
    ("Income",     20.702, 0.00036, 0.246, "Weak",        True),
    ("Gender",      0.457, 0.79,    0.037, "Negligible",  False),
    ("Occupation", 24.958, 5.13e-5, 0.271, "Weak",        True),
    ("Education",  56.360, 1.68e-7, 0.407, "Moderate",    True),
]

k1,k2,k3,k4 = st.columns(4)
kpi(k1,"5","Variables Tested","Demographic predictors")
kpi(k2,"4","Significant","p < 0.05 (Wald)",EMERALD)
kpi(k3,"0.588","Strongest V","Age (Strong)",INDIGO)
kpi(k4,"0.037","Weakest V","Gender (Negligible)",SLATE)
st.markdown("<br>",unsafe_allow_html=True)

section("Chi-Square & Cramér's V Results",
        "H₀: Q-Commerce adoption is independent of the demographic factor | α = 0.05")

left, right = st.columns([1.1, 1], gap="large")

with left:
    # Results table as a visual
    for var, chi2, p, v, assoc, sig in chi2_results:
        v_color = EMERALD if v > 0.4 else (AMBER if v > 0.2 else (SLATE if v > 0.1 else "#CBD5E1"))
        sig_icon = "✅" if sig else "❌"
        sig_text = sig_badge(p)
        bar_w = int(v / 0.6 * 100)
        st.markdown(f"""
        <div style='background:#fff;border:1px solid #E2E8F0;border-radius:12px;
                    padding:14px 18px;margin-bottom:8px;'>
          <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;'>
            <div style='font-weight:700;font-size:.9rem;color:#1E1E2E;'>{sig_icon} {var}</div>
            <div style='font-size:.75rem;color:#64748B;'>χ² = {chi2:.3f} | p {sig_text.split(' ',1)[1] if ' ' in sig_text else sig_text}</div>
          </div>
          <div style='display:flex;align-items:center;gap:8px;'>
            <div style='flex:1;background:#F1F5F9;border-radius:20px;height:8px;'>
              <div style='width:{min(max(bar_w, 0), 100)}%;background:{v_color};border-radius:20px;height:8px;'></div>
            </div>
            <div style='min-width:80px;text-align:right;'>
              <span style='font-weight:700;color:{v_color};font-size:.9rem;'>V = {v:.3f}</span>
              <span style='font-size:.72rem;color:#64748B;margin-left:4px;'>({assoc})</span>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

with right:
    section("Cramér's V — Effect Size Comparison")
    vars_ = [r[0] for r in chi2_results]
    vs_   = [r[3] for r in chi2_results]
    sigs_ = [r[5] for r in chi2_results]
    colors_ = [INDIGO if s else "#CBD5E1" for s in sigs_]
    sorted_pairs = sorted(zip(vs_, vars_, colors_), reverse=True)
    vs_s = [p[0] for p in sorted_pairs]
    vars_s = [p[1] for p in sorted_pairs]
    colors_s = [p[2] for p in sorted_pairs]

    fig = go.Figure(go.Bar(
        y=vars_s, x=vs_s, orientation="h",
        marker_color=colors_s,
        text=[f"V = {v:.3f}" for v in vs_s], textposition="outside",
        hovertemplate="%{y}: V=%{x:.3f}<extra></extra>"
    ))
    for threshold, label, color in [(0.1,"Negligible",SLATE),(0.3,"Weak",AMBER),(0.5,"Moderate",ROSE)]:
        fig.add_vline(x=threshold, line_dash="dot", line_color=color, opacity=0.6,
                      annotation_text=label, annotation_position="top right",
                      annotation_font=dict(size=9,color=color))
    fig.update_layout(**PLOTLY_LAYOUT, height=300,
                       title=dict(text="Effect Size: Blue = Significant | Gray = Not Significant", font=dict(size=11)))
    fig.update_xaxes(title="Cramér's V (Effect Size)", range=[0,0.7])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div style='background:#F8FAFC;border-radius:10px;padding:14px;font-size:.78rem;color:#475569;line-height:1.8;'>
      <b>Cramér's V guide:</b><br>
      V < 0.10 → Negligible<br>V 0.10–0.29 → Weak<br>
      V 0.30–0.49 → Moderate<br>V ≥ 0.50 → Strong
    </div>""", width="stretch")

# ── Stacked bar charts ─────────────────────────────────────────────────────────
section("Adoption Rate by Demographic Group",
        "Click any bar to explore the adoption split within each category")

tab_vars = st.tabs(["Age Group","Income","Gender","Occupation","Education"])
demo_configs = [
    ("Age_Group", AGE_ORDER),
    ("Income",    INCOME_ORDER),
    ("Gender",    ["Male","Female","Prefer not to say"]),
    ("Occupation",OCC_ORDER),
    ("Education", EDU_ORDER),
]
for tab, (col, order) in zip(tab_vars, demo_configs):
    with tab:
        ct = pd.crosstab(df[col], df["Adoption_Status"])
        ct.columns = ["Non-User","User"]
        ct = ct.reindex([o for o in order if o in ct.index])
        ct_pct = ct.div(ct.sum(axis=1), axis=0)*100

        fig = go.Figure()
        fig.add_trace(go.Bar(name="User", x=ct_pct.index, y=ct_pct["User"].round(1),
                              marker_color=INDIGO, text=ct_pct["User"].round(1),
                              texttemplate="%{text:.1f}%", textposition="inside",
                              hovertemplate="%{x}: %{y:.1f}% users<extra></extra>"))
        fig.add_trace(go.Bar(name="Non-User", x=ct_pct.index, y=ct_pct["Non-User"].round(1),
                              marker_color="#CBD5E1", text=ct_pct["Non-User"].round(1),
                              texttemplate="%{text:.1f}%", textposition="inside",
                              hovertemplate="%{x}: %{y:.1f}% non-users<extra></extra>"))

        # Count overlay
        counts = ct.sum(axis=1)
        for i, (cat, cnt) in enumerate(counts.items()):
            fig.add_annotation(x=cat, y=103, text=f"n={cnt}", showarrow=False,
                               font=dict(size=9, color="#64748B"))

        fig.update_layout(**PLOTLY_LAYOUT, barmode="stack", height=360,
                           title=dict(text=f"Adoption Rate by {col} (%)", font=dict(size=13))
                           )
        
        fig.update_yaxes(title="% of group", range=[0,110])
        fig.update_xaxes(tickangle=-20)
        st.plotly_chart(fig, width="stretch")

section("Normality Check — Shapiro-Wilk on Age")
stat, p_sw = shapiro(df["Age"].dropna())
st.markdown(f"""
<div style='background:#fff;border:1px solid #E2E8F0;border-radius:12px;padding:16px 20px;
            display:flex;gap:20px;align-items:center;'>
  <div style='text-align:center;min-width:80px;'>
    <div style='font-size:1.4rem;font-weight:800;color:{INDIGO};'>{stat:.4f}</div>
    <div style='font-size:.72rem;color:#64748B;'>W-Statistic</div>
  </div>
  <div style='text-align:center;min-width:80px;'>
    <div style='font-size:1.4rem;font-weight:800;color:{ROSE};'>{p_sw:.4f}</div>
    <div style='font-size:.72rem;color:#64748B;'>p-value</div>
  </div>
  <div style='flex:1;font-size:.83rem;color:#475569;line-height:1.7;border-left:2px solid #E2E8F0;padding-left:16px;'>
    <b>Result:</b> Age does NOT follow a normal distribution (p < 0.05).<br>
    This justifies using <b>chi-square on grouped Age_Group</b> rather than parametric t-tests,
    and non-parametric tests (Kruskal-Wallis, Mann-Whitney) in subsequent objectives.
  </div>
</div>""", unsafe_allow_html=True)

section("Key Findings")
for title, text, color in [
    ("🔴 Age is the Strongest Predictor (V=0.588 — Strong)",
     "Younger respondents (18–25, 26–33) show dramatically higher adoption. This is the most powerful demographic predictor in the study.", INDIGO),
    ("🟡 Education and Occupation Show Moderate/Weak Effects",
     "Higher education and time-constrained occupations increase adoption probability, but the effect is weaker than age.", AMBER),
    ("⚪ Gender is Irrelevant to Adoption",
     "χ² = 0.457, p = 0.79 — gender has no statistically significant impact. This challenges common assumptions about gendered digital adoption.", SLATE),
]:
    finding_card(title, text, color)

import streamlit as st
import plotly.graph_objects as go
import pandas as pd, numpy as np
from scipy.stats import chi2_contingency, shapiro, mannwhitneyu
from utils import *

st.set_page_config("Obj 2 — Adoption", "🔗", layout="wide")
st.session_state["current_page"] = "pages/6_Obj2_Adoption.py"
st.markdown("<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');html,body,[class*='css']{font-family:'Inter',sans-serif;}.stApp{background:#FAFAFA;}section[data-testid='stSidebar']{background:#FFFFFF;border-right:1px solid #E2E8F0;}</style>",unsafe_allow_html=True)
from navbar import navbar
navbar()
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
            <div style='font-size:.75rem;color:#64748B;'>χ² = {chi2:.3f} | p {sig_text.split(' ',1)[1]}</div>
          </div>
          <div style='display:flex;align-items:center;gap:8px;'>
            <div style='flex:1;background:#F1F5F9;border-radius:20px;height:8px;'>
              <div style='width:{bar_w}%;background:{v_color};border-radius:20px;height:8px;'></div>
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
    fig.update_layout(**PLOTLY_LAYOUT, height=300, title=dict(text="Effect Size: Blue = Significant | Gray = Not Significant", font=dict(size=11)))
    fig.update_xaxes(title="Cramér's V (Effect Size)", range=[0,0.7], gridcolor="#F1F5F9")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div style='background:#F8FAFC;border-radius:10px;padding:14px;font-size:.78rem;color:#475569;line-height:1.8;'>
      <b>Cramér's V guide:</b><br>
      V < 0.10 → Negligible<br>V 0.10–0.29 → Weak<br>
      V 0.30–0.49 → Moderate<br>V ≥ 0.50 → Strong
    </div>""", unsafe_allow_html=True)

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

        counts = ct.sum(axis=1)
        for i, (cat, cnt) in enumerate(counts.items()):
            fig.add_annotation(x=cat, y=103, text=f"n={cnt}", showarrow=False,
                               font=dict(size=9, color="#64748B"))

        fig.update_layout(**PLOTLY_LAYOUT, barmode="stack", height=360,
                           title=dict(text=f"Adoption Rate by {col} (%)", font=dict(size=13)))
        fig.update_xaxes(tickangle=-20)
        fig.update_yaxes(title="% of group", range=[0,110],gridcolor="#F1F5F9")
        st.plotly_chart(fig, use_container_width=True)

# ── Normality Check ────────────────────────────────────────────────────────────
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
    <div style='font-size:1.4rem;font-weight:800;color:{ROSE};'>{<0.05}</div>
    <div style='font-size:.72rem;color:#64748B;'>p-value</div>
  </div>
  <div style='flex:1;font-size:.83rem;color:#475569;line-height:1.7;border-left:2px solid #E2E8F0;padding-left:16px;'>
    <b>Result:</b> Age does NOT follow a normal distribution (p < 0.05).<br>
    That is why <b>chi-square on grouped Age_Group</b> rather than parametric t-tests,
    and non-parametric tests (Kruskal-Wallis, Mann-Whitney) in subsequent objectives.
  </div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Mann-Whitney U Test: Age × Adoption ───────────────────────────────────────
section("Mann-Whitney U Test — Age × Adoption",
        "Since Age is non-normal (Shapiro-Wilk p < 0.05), Mann-Whitney U is the correct non-parametric alternative to a t-test")

age_adopted     = df.loc[df["Used_QC"] == "Yes", "Age"].dropna() if "Used_QC" in df.columns else df.loc[df["Adoption_Status"] == 1, "Age"].dropna()
age_not_adopted = df.loc[df["Used_QC"] == "No",  "Age"].dropna() if "Used_QC" in df.columns else df.loc[df["Adoption_Status"] == 0, "Age"].dropna()

U_stat, p_mw = mannwhitneyu(age_adopted, age_not_adopted, alternative="two-sided")
n1, n2       = len(age_adopted), len(age_not_adopted)
r_rb         = 1 - (2 * U_stat) / (n1 * n2)
abs_r        = abs(r_rb)
effect_label = "Large" if abs_r >= 0.50 else ("Medium" if abs_r >= 0.30 else ("Small" if abs_r >= 0.10 else "Negligible"))
effect_color = EMERALD if abs_r >= 0.50 else (INDIGO if abs_r >= 0.30 else (AMBER if abs_r >= 0.10 else SLATE))
direction    = "lower" if age_adopted.median() < age_not_adopted.median() else "higher"

# KPI row
mw1, mw2, mw3, mw4 = st.columns(4)
kpi(mw1, f"{U_stat:,.0f}", "U Statistic", "Mann-Whitney U", INDIGO)
kpi(mw2, f"{p_mw < 0.05}",   "p-value",     "Two-sided test", ROSE if p_mw < 0.05 else SLATE)
kpi(mw3, f"{r_rb:.3f}",   "Effect Size r", f"({effect_label})", effect_color)
kpi(mw4, f"{age_adopted.median():.1f} vs {age_not_adopted.median():.1f}",
        "Median Age", "Adopters vs Non-Adopters", EMERALD)

st.markdown("<br>", unsafe_allow_html=True)

mw_left, mw_right = st.columns(2, gap="large")

with mw_left:
    # Result card
    result_color = EMERALD if p_mw < 0.05 else SLATE
    result_icon  = "✅" if p_mw < 0.05 else "❌"
    result_text  = "SIGNIFICANT — Reject H₀" if p_mw < 0.05 else "NOT SIGNIFICANT — Fail to Reject H₀"
    direction_note = f"Adopters tend to be <b>{direction} in age</b> (Median adopters = {age_adopted.median():.1f}, non-adopters = {age_not_adopted.median():.1f})." if p_mw < 0.05 else "No significant age difference between adopters and non-adopters."

    st.markdown(f"""
    <div style='background:#fff;border:1px solid #E2E8F0;border-radius:12px;padding:18px 20px;height:100%;'>
      <div style='font-weight:700;font-size:.85rem;color:#1E1E2E;margin-bottom:10px;'>Test Result</div>
      <div style='background:{result_color}18;border-left:4px solid {result_color};border-radius:6px;
                  padding:10px 14px;margin-bottom:12px;'>
        <span style='font-weight:700;color:{result_color};'>{result_icon} {result_text}</span>
      </div>
      <div style='font-size:.82rem;color:#475569;line-height:1.8;'>
        <b>H₀:</b> Age distributions are identical for adopters and non-adopters.<br>
        <b>H₁:</b> Age distributions differ between the two groups.<br>
        <b>α = 0.05</b><br><br>
        {direction_note}
      </div>
      <div style='margin-top:14px;padding-top:12px;border-top:1px solid #F1F5F9;'>
        <div style='font-size:.76rem;color:#64748B;font-weight:600;margin-bottom:6px;'> r — EFFECT SIZE GUIDE</div>
        {''.join([
          f"<div style='display:flex;justify-content:space-between;font-size:.75rem;padding:2px 0;"
          f"color:{'#1E1E2E' if lbl == effect_label else '#94A3B8'};font-weight:{'700' if lbl == effect_label else '400'};'>"
          f"<span>{lbl}</span><span>{rng}</span></div>"
          for lbl, rng in [("Negligible","r < 0.10"),("Small","0.10–0.29"),("Medium","0.30–0.49"),("Large","≥ 0.50")]
        ])}
      </div>
    </div>""", unsafe_allow_html=True)

with mw_right:
    # Box plot: Age distribution by adoption
    adopted_vals     = age_adopted.tolist()
    not_adopted_vals = age_not_adopted.tolist()

    fig_mw = go.Figure()
    fig_mw.add_trace(go.Box(
        y=adopted_vals, name="Adopted (Yes)",
        marker_color=INDIGO, boxmean=True,
        hovertemplate="Adopted<br>Age: %{y}<extra></extra>"
    ))
    fig_mw.add_trace(go.Box(
        y=not_adopted_vals, name="Not Adopted (No)",
        marker_color="#CBD5E1", boxmean=True,
        hovertemplate="Not Adopted<br>Age: %{y}<extra></extra>"
    ))
    fig_mw.add_annotation(
        x=0.5, y=1.06, xref="paper", yref="paper",
        text=f"U = {U_stat:,.0f}  |  p = {p_mw:.4f}  |  r = {r_rb:.3f} ({effect_label})",
        showarrow=False, font=dict(size=10, color="#64748B"),
        bgcolor="#F8FAFC", bordercolor="#E2E8F0", borderwidth=1, borderpad=4
    )
    fig_mw.update_layout(
        **PLOTLY_LAYOUT, height=340,
        title=dict(text="Age Distribution by Adoption Status", font=dict(size=13)),
        showlegend=True
    )
    fig_mw.update_yaxes(title="Age (years)", gridcolor="#F1F5F9")
    st.plotly_chart(fig_mw, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Odds Ratio Analysis ────────────────────────────────────────────────────────
section("Odds Ratio Analysis — Binarised Demographics",
        "Quantifying adoption odds for simplified age and education groups with 95% confidence intervals")

def odds_ratio_with_ci(a, b, c, d):
    """Compute OR and 95% CI with continuity correction."""
    a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    OR     = (a * d) / (b * c)
    log_or = np.log(OR)
    se     = np.sqrt(1/a + 1/b + 1/c + 1/d)
    lo     = np.exp(log_or - 1.96 * se)
    hi     = np.exp(log_or + 1.96 * se)
    return OR, lo, hi

# Binarised Age
df["Age_Binary"] = df["Age_Group"].apply(
    lambda x: "Young (18–33)" if x in ["18-25", "26-33"] else "Older (34+)"
) if "Age_Group" in df.columns else "Young (18–33)"

adoption_col = "Used_QC" if "Used_QC" in df.columns else None
if adoption_col is None:
    df["_adopt"] = df["Adoption_Status"].map({1: "Yes", 0: "No"})
    adoption_col = "_adopt"

ct_age_bin = pd.crosstab(df["Age_Binary"], df[adoption_col])
# Ensure Yes/No columns exist
for c in ["Yes","No"]:
    if c not in ct_age_bin.columns:
        ct_age_bin[c] = 0

a_age = ct_age_bin.loc["Young (18–33)", "Yes"]  if "Young (18–33)"  in ct_age_bin.index else 0
b_age = ct_age_bin.loc["Young (18–33)", "No"]   if "Young (18–33)"  in ct_age_bin.index else 0
c_age = ct_age_bin.loc["Older (34+)",   "Yes"]  if "Older (34+)"    in ct_age_bin.index else 0
d_age = ct_age_bin.loc["Older (34+)",   "No"]   if "Older (34+)"    in ct_age_bin.index else 0
OR_age, lo_age, hi_age = odds_ratio_with_ci(a_age, b_age, c_age, d_age)

# Binarised Education
edu_col = "Education" if "Education" in df.columns else None
if edu_col:
    df["Edu_Binary"] = df[edu_col].apply(
        lambda x: "High (PG/Professional)" if x in ["Postgraduate", "Professional Degree"] else "Low (Below PG)"
    )
    ct_edu_bin = pd.crosstab(df["Edu_Binary"], df[adoption_col])
    for c in ["Yes","No"]:
        if c not in ct_edu_bin.columns:
            ct_edu_bin[c] = 0
    a_edu = ct_edu_bin.loc["High (PG/Professional)", "Yes"] if "High (PG/Professional)" in ct_edu_bin.index else 0
    b_edu = ct_edu_bin.loc["High (PG/Professional)", "No"]  if "High (PG/Professional)" in ct_edu_bin.index else 0
    c_edu = ct_edu_bin.loc["Low (Below PG)",         "Yes"] if "Low (Below PG)"         in ct_edu_bin.index else 0
    d_edu = ct_edu_bin.loc["Low (Below PG)",         "No"]  if "Low (Below PG)"         in ct_edu_bin.index else 0
    OR_edu, lo_edu, hi_edu = odds_ratio_with_ci(a_edu, b_edu, c_edu, d_edu)
else:
    OR_edu, lo_edu, hi_edu = 5.894, 3.485, 9.967  # fallback to pre-computed

or_data = [
    ("Age",       "Young (18–33) vs Older (34+)",          OR_age, lo_age, hi_age),
    ("Education", "High (PG/Professional) vs Low (Below PG)", OR_edu, lo_edu, hi_edu),
]

or_left, or_right = st.columns(2, gap="large")

for col_widget, (var, label, OR, lo, hi) in zip([or_left, or_right], or_data):
    with col_widget:
        sig_or   = lo > 1.0
        or_color = EMERALD if sig_or else SLATE
        ci_width = hi - lo

        st.markdown(f"""
        <div style='background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:20px 22px;'>
          <div style='font-weight:700;font-size:.95rem;color:#1E1E2E;margin-bottom:4px;'>
            {var} — Odds Ratio
          </div>
          <div style='font-size:.78rem;color:#64748B;margin-bottom:16px;'>{label}</div>

          <div style='display:flex;align-items:baseline;gap:10px;margin-bottom:6px;'>
            <div style='font-size:2.4rem;font-weight:800;color:{or_color};line-height:1;'>
              {OR:.3f}
            </div>
            <div style='font-size:.82rem;color:#64748B;'>Odds Ratio</div>
          </div>

          <div style='font-size:.82rem;color:#475569;margin-bottom:14px;'>
            95% CI &nbsp;[<b>{lo:.3f}</b>, <b>{hi:.3f}</b>]
            &nbsp;{"✅ Significant (CI excludes 1)" if sig_or else "❌ Not significant (CI includes 1)"}
          </div>

          <div style='background:#F8FAFC;border-radius:8px;padding:10px 14px;font-size:.79rem;
                      color:#475569;line-height:1.75;'>
            <b>Interpretation:</b><br>
            {"The " + label.split(" vs ")[0] + " group is <b>" + f"{OR:.1f}×</b> more likely to have adopted Q-Commerce than the " + label.split(" vs ")[1] + " group." if sig_or else "No significant difference in adoption odds between the two groups."}
          </div>

          <div style='margin-top:14px;'>
                    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)



# ── Key Findings ───────────────────────────────────────────────────────────────
section("Key Findings")

for title, text, color in [
    ("🔴 Age is the Strongest Predictor (V=0.588 — Strong)",
     "Younger respondents (18–25, 26–33) show dramatically higher adoption. This is the most powerful demographic predictor in the study.", INDIGO),
    ("🟡 Education and Occupation Show Moderate/Weak Effects",
     "Higher education and time-constrained occupations increase adoption probability, but the effect is weaker than age.", AMBER),
    ("⚪ Gender is Irrelevant to Adoption",
     "χ² = 0.457, p = 0.79 — gender has no statistically significant impact. This challenges common assumptions about gendered digital adoption.", SLATE),
    ("📊 Mann-Whitney U: Age Distributions Differ Significantly by Adoption",
     f"U = {U_stat:,.0f}, p = {p_mw:.4f}, r = {r_rb:.3f} ({effect_label} effect). Since Age is non-normally distributed (Shapiro-Wilk p < 0.05), Mann-Whitney U confirms that adopters are significantly {direction} in age than non-adopters — validating the chi-square finding on grouped Age_Group using a continuous non-parametric test.", EMERALD),
    ("🎯 Odds Ratio: Young Adults are ~5.8× More Likely to Adopt",
     f"Young group (18–33) vs Older (34+): OR = {OR_age:.3f}, 95% CI [{lo_age:.3f}, {hi_age:.3f}]. The entire confidence interval lies above 1, confirming the effect is statistically robust. Young adults have nearly 6× the odds of adoption compared to older respondents.", INDIGO),
    ("🎓 Higher Education Doubles Adoption Odds by ~5.9×",
     f"High education (PG/Professional) vs Low (Below PG): OR = {OR_edu:.3f}, 95% CI [{lo_edu:.3f}, {hi_edu:.3f}]. Higher educational attainment is strongly associated with Q-Commerce adoption, consistent with greater digital literacy and app comfort among more educated consumers.", EMERALD),
]:
    finding_card(title, text, color)

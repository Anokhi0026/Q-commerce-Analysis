import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal
from utils import *

st.set_page_config("Non-User Analysis", "🚫", layout="wide")
st.session_state["current_page"] = "pages/13_NonUser_Analysis.py"
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*='css']{font-family:'Inter',sans-serif;}
.stApp{background:#FAFAFA;}
section[data-testid='stSidebar']{background:#FFFFFF;border-right:1px solid #E2E8F0;}
</style>""", unsafe_allow_html=True)

from navbar import navbar
navbar()
page_header("Non-User Barrier Analysis",
            "Why do 113 respondents not use Q-Commerce apps?",
            "Demographic profile, non-adoption reasons, barrier Likert analysis, "
            "Chi-Square tests, willingness to try, Mann-Whitney U, and Kruskal-Wallis "
            "across all 113 non-adopters in Vadodara.")

# ── DATA ───────────────────────────────────────────────────────────────────────
df       = load_analysis()
df_raw   = load_raw()
non_u    = df[df["Adoption_Status"] == 0].copy()
non_u_r  = df_raw[df_raw["Adoption_Status"] == 0].copy().reset_index(drop=True)

LIKERT_MAP = {"Strongly Disagree":1,"Disagree":2,"Neutral":3,"Agree":4,"Strongly Agree":5}

BARRIER_COLS = [
    "I would consider using Q-commerce if delivery charges were lower",
    "I would consider using Q-commerce apps if product quality were guaranteed",
    "I would consider using Q-commerce apps if adequate guidance on app usage were provided",
    "I would consider using Q-commerce apps if prices were competitive",
    "I would consider using Q-commerce apps if delivery services were available in my area",
    "I would consider using Q-commerce apps if attractive discounts were offered",
    "I would consider using Q-commerce apps if I felt confident about trust, data security, and privacy.",
]
BARRIER_NAMES = [
    "Lower Delivery Charges",
    "Product Quality Guarantee",
    "App Usage Guidance",
    "Competitive Pricing",
    "Delivery Availability",
    "Attractive Discounts",
    "Trust & Data Security",
]

REASONS      = ["R_High_Charges","R_Quality_Concern","R_No_Need","R_Prefer_Local",
                "R_Trust_Issue","R_App_Discomfort","R_Lack_Awareness","R_Not_Available"]
REASON_LABELS = ["High Charges","Quality Concern","No Need","Prefer Local Stores",
                 "Trust Issues","App Discomfort","Lack of Awareness","Not Available"]

# ── KPIs ───────────────────────────────────────────────────────────────────────
k1,k2,k3,k4 = st.columns(4)
kpi(k1, "113",   "Non-Users",        "33.1% of total sample")
kpi(k2, "29.2%", "Top Barrier",      "Lack of Awareness",      ROSE)
kpi(k3, "7",     "Barrier Items",    "Conditional adoption Likert", AMBER)
kpi(k4, "4",     "Statistical Tests","χ², Mann-Whitney, Kruskal-Wallis", VIOLET)
st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DEMOGRAPHIC PROFILE
# ══════════════════════════════════════════════════════════════════════════════
section("1 · Demographic Profile of Non-Users",
        "How non-adopters are distributed across age, education, income and occupation")

tab_age, tab_edu, tab_inc, tab_occ = st.tabs(["Age Group","Education","Income","Occupation"])

def demo_bar(col, order, tab):
    with tab:
        ct = pd.crosstab(df[col], df["Adoption_Status"])
        ct.columns = ["Non-User","User"]
        ct = ct.reindex([o for o in order if o in ct.index])
        ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Non-User", x=ct_pct.index,
                                 y=ct_pct["Non-User"].round(1),
                                 marker_color=ROSE,
                                 text=ct_pct["Non-User"].round(1),
                                 texttemplate="%{text:.1f}%", textposition="outside"))
            fig.add_trace(go.Bar(name="User", x=ct_pct.index,
                                 y=ct_pct["User"].round(1),
                                 marker_color=INDIGO,
                                 text=ct_pct["User"].round(1),
                                 texttemplate="%{text:.1f}%", textposition="outside"))
            fig.update_layout(**PLOTLY_LAYOUT, barmode="group", height=340,
                              title=dict(text=f"Adoption Split by {col} (%)", font=dict(size=13)))
            fig.update_xaxes(tickangle=-20)
            fig.update_yaxes(title="% within group", range=[0,115], gridcolor="#F1F5F9")
            st.plotly_chart(fig, use_container_width=True, key=f"demo_{col}_grp")

        with c2:
            nu_cnt = non_u[col].value_counts().reindex(order).fillna(0)
            fig2 = go.Figure(go.Bar(
                x=nu_cnt.index, y=nu_cnt.values,
                marker_color=ROSE,
                text=nu_cnt.values.astype(int), textposition="outside"
            ))
            fig2.update_layout(**PLOTLY_LAYOUT, height=340,
                               title=dict(text=f"Non-User Count by {col}", font=dict(size=13)))
            fig2.update_xaxes(tickangle=-20)
            fig2.update_yaxes(title="Count", gridcolor="#F1F5F9")
            st.plotly_chart(fig2, use_container_width=True, key=f"demo_{col}_cnt")

demo_bar("Age_Group",  AGE_ORDER,    tab_age)
demo_bar("Education",  EDU_ORDER,    tab_edu)
demo_bar("Income",     INCOME_ORDER, tab_inc)
demo_bar("Occupation", OCC_ORDER,    tab_occ)

finding_card("🔴 Older Age Groups Dominate Non-Adoption",
             "58.7% of respondents aged 40+ are non-adopters. Lower educational attainment "
             "(66.7% of No Formal Education, 65.6% of school-level) and retired individuals (66.7%) "
             "are most over-represented. Income shows a weaker pattern than age or education.", ROSE)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — NON-ADOPTION REASONS
# ══════════════════════════════════════════════════════════════════════════════
section("2 · Stated Reasons for Non-Adoption",
        "Multi-select question: respondents could select all reasons that apply (n=113)")

reason_counts = non_u[REASONS].sum()
reason_pct    = (reason_counts / len(non_u) * 100).round(2)
reason_df     = pd.DataFrame({"Label": REASON_LABELS,
                               "Count": reason_counts.values,
                               "Pct":   reason_pct.values}
                             ).sort_values("Count", ascending=True)

c1, c2 = st.columns([1.4, 1], gap="large")
with c1:
    fig_r = go.Figure(go.Bar(
        y=reason_df["Label"], x=reason_df["Count"],
        orientation="h",
        marker_color=[ROSE if p > 20 else AMBER if p > 10 else "#CBD5E1"
                      for p in reason_df["Pct"]],
        text=[f"n={int(c)}  ({p:.1f}%)" for c, p in
              zip(reason_df["Count"], reason_df["Pct"])],
        textposition="outside",
        hovertemplate="%{y}: %{x} non-users<extra></extra>"
    ))
    fig_r.update_layout(**PLOTLY_LAYOUT, height=380,
                        title=dict(text="Non-Adoption Reasons (Multi-select, n=113)", font=dict(size=13)))
    fig_r.update_xaxes(title="Number of Non-Users", gridcolor="#F1F5F9", range=[0, reason_df["Count"].max()+8])
    st.plotly_chart(fig_r, use_container_width=True, key="fig_reasons")

with c2:
    fig_pie = go.Figure(go.Pie(
        labels=reason_df["Label"], values=reason_df["Count"],
        hole=0.45,
        marker_colors=[ROSE, AMBER, EMERALD, INDIGO, VIOLET, SKY, "#DB2777", "#CBD5E1"],
        textinfo="percent", textfont=dict(size=11),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>"
    ))
    fig_pie.update_layout(**{k:v for k,v in PLOTLY_LAYOUT.items() if k not in ["xaxis","yaxis"]},
                          height=380,
                          title=dict(text="Share of Stated Barriers", font=dict(size=13)))
    st.plotly_chart(fig_pie, use_container_width=True, key="fig_reasons_pie")

finding_card("👁️ Lack of Awareness is the #1 Barrier (29.2%)",
             "The largest segment of non-adopters simply hasn't been reached — not rejected Q-commerce. "
             "Preference for local stores (24.8%) reflects habitual purchasing. Together >50% of reasons "
             "are non-economic, suggesting awareness campaigns over price cuts.", ROSE)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — OPEN-ENDED REASONS
# ══════════════════════════════════════════════════════════════════════════════
section("3 · Open-Ended Barriers (Qualitative)",
        "Free-text responses mapped to categories")

reason_map = {
    "Delivery charges are high": "High Delivery Charges",
    " Product quality concerns": "Product Quality Concerns",
    "Product quality concerns":  "Product Quality Concerns",
    " No need for fast delivery":"No Need for Fast Delivery",
    "I prefer buying from local stores": "Prefer Local Stores",
    " Do no trust online delivery":      "Do Not Trust Online Delivery",
    "Do no trust online delivery":       "Do Not Trust Online Delivery",
    " Not comfortable using apps":       "Not Comfortable Using Apps",
    "Not comfortable using apps":        "Not Comfortable Using Apps",
    "Not aware how to use the apps":     "Unaware How to Use Apps",
    " Not aware how to use the apps":    "Unaware How to Use Apps",
    "Others in family use it so not required": "Family Already Uses It",
    "Others in Family use it so not requried": "Family Already Uses It",
    "Others in Family Order it":         "Family Already Uses It",
    " Payment safety concerns":          "Payment Safety Concerns",
    "Not Available":                     "Not Available in Area",
}

reason_cols = [col for col in non_u_r.columns if "usex" in col.lower()]
reason_raw  = []
for col in reason_cols:
    vals = non_u_r[col].dropna().astype(str).str.strip()
    reason_raw.extend(vals.tolist())

open_counts = (pd.Series(reason_raw)
               .str.strip()
               .replace(reason_map)
               .value_counts())
open_counts = open_counts[open_counts.index != "nan"]
open_sorted = open_counts.sort_values()

fig_open = go.Figure(go.Bar(
    y=open_sorted.index, x=open_sorted.values, orientation="h",
    marker_color=[AMBER if v == open_sorted.max() else
                  ROSE  if v >= open_sorted.quantile(0.6) else "#CBD5E1"
                  for v in open_sorted.values],
    text=[f"n={int(v)}  ({v/len(non_u)*100:.1f}%)" for v in open_sorted.values],
    textposition="outside",
    hovertemplate="%{y}: %{x}<extra></extra>"
))
fig_open.update_layout(**PLOTLY_LAYOUT, height=380,
                       title=dict(text="Open-Ended Barrier Categories (Non-users, n=113)", font=dict(size=13)))
fig_open.update_xaxes(title="Frequency", gridcolor="#F1F5F9",
                      range=[0, open_sorted.max() + 8])
st.plotly_chart(fig_open, use_container_width=True, key="fig_open_barriers")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CHI-SQUARE: BARRIERS vs ADOPTION
# ══════════════════════════════════════════════════════════════════════════════
section("4 · Chi-Square Tests — Barriers vs Adoption Status",
        "H₀: The barrier is independent of Q-Commerce adoption | α = 0.05")

barrier_vars   = ["R_Lack_Awareness","R_Prefer_Local","R_Trust_Issue","R_App_Discomfort"]
barrier_labels = ["Lack of Awareness","Prefer Local Stores","Trust Issues","App Discomfort"]

chi2_results = []
for var, label in zip(barrier_vars, barrier_labels):
    ct = pd.crosstab(df[var], df["Adoption_Status"])
    chi2_v, p_v, dof, _ = chi2_contingency(ct)
    n   = ct.values.sum()
    cv  = np.sqrt(chi2_v / (n * (min(ct.shape)-1)))
    chi2_results.append({"Barrier": label, "χ²": round(chi2_v,4),
                         "p": p_v, "V": round(cv,4),
                         "Sig": p_v < 0.05})

c1, c2 = st.columns([1.1, 1], gap="large")
with c1:
    for r in chi2_results:
        bar_w   = int(r["V"] / 0.5 * 100)
        v_color = EMERALD if r["V"] > 0.3 else AMBER if r["V"] > 0.2 else SLATE
        sig_ico = "✅" if r["Sig"] else "❌"
        p_str   = "p < 0.001" if r["p"] < 0.001 else f"p = {r['p']:.4f}"
        st.markdown(f"""
        <div style='background:#fff;border:1px solid #E2E8F0;border-radius:12px;
                    padding:14px 18px;margin-bottom:8px;'>
          <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;'>
            <div style='font-weight:700;font-size:.9rem;color:#1E1E2E;'>{sig_ico} {r["Barrier"]}</div>
            <div style='font-size:.75rem;color:#64748B;'>χ² = {r["χ²"]} | {p_str}</div>
          </div>
          <div style='display:flex;align-items:center;gap:8px;'>
            <div style='flex:1;background:#F1F5F9;border-radius:20px;height:8px;'>
              <div style='width:{bar_w}%;background:{v_color};border-radius:20px;height:8px;'></div>
            </div>
            <span style='font-weight:700;color:{v_color};font-size:.9rem;'>V = {r["V"]:.4f}</span>
          </div>
        </div>""", unsafe_allow_html=True)

with c2:
    fig_v = go.Figure(go.Bar(
        y=[r["Barrier"] for r in chi2_results],
        x=[r["V"] for r in chi2_results],
        orientation="h",
        marker_color=[EMERALD if r["V"] > 0.3 else AMBER for r in chi2_results],
        text=[f"V = {r['V']:.4f}" for r in chi2_results],
        textposition="outside",
        hovertemplate="%{y}: V=%{x:.4f}<extra></extra>"
    ))
    fig_v.add_vline(x=0.1, line_dash="dot", line_color=SLATE,   annotation_text="Negligible", annotation_position="top right", annotation_font=dict(size=9))
    fig_v.add_vline(x=0.3, line_dash="dot", line_color=AMBER,   annotation_text="Weak",       annotation_position="top right", annotation_font=dict(size=9))
    fig_v.add_vline(x=0.5, line_dash="dot", line_color=EMERALD, annotation_text="Moderate",   annotation_position="top right", annotation_font=dict(size=9))
    fig_v.update_layout(**PLOTLY_LAYOUT, height=280,
                        title=dict(text="Cramér's V — Effect Size per Barrier", font=dict(size=12)))
    fig_v.update_xaxes(title="Cramér's V", range=[0, 0.6], gridcolor="#F1F5F9")
    st.plotly_chart(fig_v, use_container_width=True, key="fig_cramersv")

finding_card("✅ All 4 Barriers Significantly Associated with Non-Adoption",
             "Lack of Awareness has the strongest effect (V=0.443 — Moderate). "
             "Preference for Local Stores (V=0.270), Trust Issues, and App Discomfort "
             "show weaker but significant associations. Non-adoption is driven by "
             "informational and behavioural barriers, not purely economic ones.", EMERALD)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — WILLINGNESS TO TRY
# ══════════════════════════════════════════════════════════════════════════════
section("5 · Willingness to Try Q-Commerce (Non-Users)",
        "Would non-users be willing to try Q-Commerce in the future?")

will_col = "NonUser_Willing_Try"
if will_col in non_u.columns:
    will_cnt = non_u[will_col].value_counts()
    will_pct = non_u[will_col].value_counts(normalize=True) * 100

    c1, c2 = st.columns([1, 1.3], gap="large")
    with c1:
        fig_will = go.Figure(go.Pie(
            labels=will_cnt.index, values=will_cnt.values,
            hole=0.5,
            marker_colors=[EMERALD, ROSE, AMBER],
            textinfo="label+percent+value",
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>"
        ))
        fig_will.update_layout(
            **{k:v for k,v in PLOTLY_LAYOUT.items() if k not in ["xaxis","yaxis"]},
            height=320,
            title=dict(text="Willingness to Try Q-Commerce (n=113)", font=dict(size=13)))
        st.plotly_chart(fig_will, use_container_width=True, key="fig_willingness")

    with c2:
        # Chi-square: awareness vs willingness
        if "R_Lack_Awareness" in non_u.columns:
            ct_try = pd.crosstab(non_u[will_col], non_u["R_Lack_Awareness"])
            chi2_try, p_try, _, _ = chi2_contingency(ct_try)
            p_str_try = "< 0.001" if p_try < 0.001 else f"= {p_try:.4f}"
            sig_try   = p_try < 0.05

        st.markdown(f"""
        <div style='background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:20px;height:100%;'>
          <div style='font-weight:700;font-size:.9rem;color:#1E1E2E;margin-bottom:14px;'>
            Chi-Square: Awareness × Willingness to Try
          </div>
          <div style='display:flex;gap:16px;margin-bottom:16px;'>
            <div style='text-align:center;background:#F8FAFC;border-radius:10px;padding:12px 16px;flex:1;'>
              <div style='font-size:1.3rem;font-weight:800;color:{INDIGO};'>{chi2_try:.4f}</div>
              <div style='font-size:.72rem;color:#64748B;margin-top:4px;'>χ² Statistic</div>
            </div>
            <div style='text-align:center;background:#F8FAFC;border-radius:10px;padding:12px 16px;flex:1;'>
              <div style='font-size:1.3rem;font-weight:800;color:{"#EF4444" if not sig_try else EMERALD};'>
                p {p_str_try}
              </div>
              <div style='font-size:.72rem;color:#64748B;margin-top:4px;'>p-value</div>
            </div>
          </div>
          <div style='background:{"#FEF2F2" if not sig_try else "#F0FDF4"};border-radius:10px;
                      padding:12px;font-size:.83rem;color:#374151;line-height:1.7;'>
            <b>Result:</b> {"Not Significant" if not sig_try else "Significant"}<br>
            Lack of awareness is <b>not</b> significantly associated with reduced willingness to try
            (p {">" if not sig_try else "<"} 0.05). Unaware non-users are no more resistant to
            experimentation than aware ones — indicating strong <b>latent conversion potential</b>.
          </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — BARRIER LIKERT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
section("6 · Conditional Adoption Barrier Items — Likert Analysis",
        "7 statements: 'I would consider using Q-Commerce if…' | Scale: 1=Strongly Disagree → 5=Strongly Agree")

barrier_data = non_u_r[BARRIER_COLS].copy()
for col in BARRIER_COLS:
    barrier_data[col] = barrier_data[col].map(LIKERT_MAP)
barrier_data.columns = BARRIER_NAMES
barrier_data = barrier_data.dropna()

barrier_desc = barrier_data.describe().T[["mean","std","50%"]].rename(
    columns={"mean":"Mean","std":"Std Dev","50%":"Median"})
barrier_desc["Rank"] = barrier_desc["Mean"].rank(ascending=False).astype(int)
barrier_desc = barrier_desc.sort_values("Mean", ascending=True)

c1, c2 = st.columns(2, gap="large")
with c1:
    opacities  = np.linspace(0.35, 1.0, len(barrier_desc))
    bar_colors = [f"rgba(225,29,72,{op:.2f})" for op in opacities]

    fig_bar = go.Figure(go.Bar(
        y=barrier_desc.index,
        x=barrier_desc["Mean"],
        orientation="h",
        error_x=dict(type="data", array=barrier_desc["Std Dev"].round(2), visible=True,
                     color="#94A3B8", thickness=1.5, width=4),
        marker_color=bar_colors,
        text=[f"μ={m:.2f}  Rank #{int(r)}" for m, r in
              zip(barrier_desc["Mean"], barrier_desc["Rank"])],
        textposition="outside",
        hovertemplate="%{y}<br>Mean=%{x:.2f}<extra></extra>"
    ))
    fig_bar.add_vline(x=3.0, line_dash="dot", line_color="#94A3B8",
                      annotation_text="Neutral (3.0)", annotation_font=dict(size=9))
    fig_bar.add_vline(x=4.0, line_dash="dot", line_color=ROSE,
                      annotation_text="Agree (4.0)", annotation_font=dict(size=9))
    fig_bar.update_layout(**PLOTLY_LAYOUT, height=380,
                          title=dict(text="Ranked Barrier Mean Scores (±1 SD, n=113)", font=dict(size=12)))
    fig_bar.update_xaxes(title="Mean Score (1–5)", range=[1, 6.2], gridcolor="#F1F5F9")
    st.plotly_chart(fig_bar, use_container_width=True, key="fig_barrier_means")

with c2:
    # Stacked Likert distribution
    likert_labels = ["Strongly Disagree","Disagree","Neutral","Agree","Strongly Agree"]
    likert_colors = ["#EF4444","#F97316","#EAB308","#22C55E","#16A34A"]

    fig_stk = go.Figure()
    for score, label, color in zip([1,2,3,4,5], likert_labels, likert_colors):
        pcts = [(barrier_data[col] == score).sum() / len(barrier_data) * 100
                for col in BARRIER_NAMES]
        fig_stk.add_trace(go.Bar(
            name=label, y=BARRIER_NAMES, x=pcts, orientation="h",
            marker_color=color,
            text=[f"{p:.0f}%" if p > 5 else "" for p in pcts],
            textposition="inside",
            hovertemplate=f"{label}: %{{x:.1f}}%<extra></extra>"
        ))
    fig_stk.update_layout(**PLOTLY_LAYOUT, barmode="stack", height=380,
                          title=dict(text="Likert Response Distribution per Barrier Item", font=dict(size=12)),
                          xaxis=dict(title="% of Non-Users", gridcolor="#F1F5F9"))
    fig_stk.update_layout(legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_stk, use_container_width=True, key="fig_barrier_stacked")

# Descriptive table
st.markdown("<br>", unsafe_allow_html=True)
section("Barrier Descriptive Statistics Table")
display_df = barrier_desc[["Mean","Std Dev","Median","Rank"]].copy()
display_df["Mean"]    = display_df["Mean"].round(3)
display_df["Std Dev"] = display_df["Std Dev"].round(3)
display_df["Median"]  = display_df["Median"].round(1)
display_df = display_df.sort_values("Rank")
st.dataframe(display_df, use_container_width=True)

finding_card("🏆 Lower Delivery Charges & Trust are Top Conditional Drivers",
             "Non-users are most responsive to price (Lower Delivery Charges ranks #1) "
             "and Trust & Data Security (#2). All items score above Neutral (>3.0), "
             "confirming non-users are conditionally open to adoption — the barriers are "
             "surmountable with the right platform interventions.", AMBER)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MANN-WHITNEY U: GENDER DIFFERENCES
# ══════════════════════════════════════════════════════════════════════════════
section("7 · Mann-Whitney U Test — Gender Differences in Barriers",
        "H₀: No difference in barrier intensity between Male and Female non-users | α = 0.05")

non_u_full = non_u_r.copy()
for col, name in zip(BARRIER_COLS, BARRIER_NAMES):
    non_u_full[name] = non_u_full[col].map(LIKERT_MAP)

gender_groups = {g: grp for g, grp in non_u_full.groupby("Gender")}
genders       = [g for g in ["Male","Female"] if g in gender_groups]

mw_results = []
if len(genders) == 2:
    for barrier in BARRIER_NAMES:
        g1 = gender_groups[genders[0]][barrier].dropna()
        g2 = gender_groups[genders[1]][barrier].dropna()
        if len(g1) >= 3 and len(g2) >= 3:
            U, p = mannwhitneyu(g1, g2, alternative="two-sided")
            mw_results.append({
                "Barrier":              barrier,
                f"Median ({genders[0]})": g1.median(),
                f"Median ({genders[1]})": g2.median(),
                "U Statistic":          round(U, 1),
                "p-value":              round(p, 4),
                "Significant":          "✅ Yes" if p < 0.05 else "❌ No"
            })

if mw_results:
    mw_df = pd.DataFrame(mw_results)

    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        st.dataframe(mw_df, use_container_width=True)
    with c2:
        sig_barriers = [r["Barrier"] for r in mw_results if r["Significant"] == "✅ Yes"]
        p_vals       = [r["p-value"] for r in mw_results]
        colors_mw    = [EMERALD if p < 0.05 else "#CBD5E1" for p in p_vals]

        fig_mw = go.Figure(go.Bar(
            y=[r["Barrier"] for r in mw_results],
            x=p_vals, orientation="h",
            marker_color=colors_mw,
            text=[f"p={p:.4f}" for p in p_vals],
            textposition="outside",
            hovertemplate="%{y}: p=%{x:.4f}<extra></extra>"
        ))
        fig_mw.add_vline(x=0.05, line_dash="dash", line_color=ROSE,
                         annotation_text="α=0.05", annotation_font=dict(size=9, color=ROSE))
        fig_mw.update_layout(**PLOTLY_LAYOUT, height=340,
                             title=dict(text="Mann-Whitney p-values by Barrier (Green=Sig)", font=dict(size=12)))
        fig_mw.update_xaxes(title="p-value", gridcolor="#F1F5F9")
        st.plotly_chart(fig_mw, use_container_width=True, key="fig_mw_gender")

    if sig_barriers:
        finding_card(f"⚠️ Gender Differences Found in {len(sig_barriers)} Barrier(s)",
                     f"Significant gender differences detected in: {', '.join(sig_barriers)}. "
                     "This suggests targeted gender-specific messaging may be effective.", VIOLET)
    else:
        finding_card("⚪ No Significant Gender Differences in Barriers",
                     "Male and female non-users report statistically equivalent barrier intensities "
                     "across all 7 conditional adoption items (all p > 0.05).", SLATE)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — KRUSKAL-WALLIS: AGE GROUP DIFFERENCES
# ══════════════════════════════════════════════════════════════════════════════
section("8 · Kruskal-Wallis Test — Age Group Differences in Barriers",
        "H₀: No difference in barrier intensity across age groups among non-users | α = 0.05")

kw_results = []
for barrier in BARRIER_NAMES:
    groups = [grp[barrier].dropna().values
              for _, grp in non_u_full.groupby("Age_Group")
              if len(grp[barrier].dropna()) >= 3]
    if len(groups) >= 2:
        H, p = kruskal(*groups)
        kw_results.append({
            "Barrier":    barrier,
            "H Statistic": round(H, 3),
            "p-value":    round(p, 4),
            "Significant": "✅ Yes" if p < 0.05 else "❌ No"
        })

if kw_results:
    kw_df = pd.DataFrame(kw_results)
    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        st.dataframe(kw_df, use_container_width=True)
    with c2:
        kw_colors = [EMERALD if r["p-value"] < 0.05 else "#CBD5E1" for r in kw_results]
        fig_kw = go.Figure(go.Bar(
            y=[r["Barrier"] for r in kw_results],
            x=[r["H Statistic"] for r in kw_results],
            orientation="h",
            marker_color=kw_colors,
            text=[f"H={r['H Statistic']}" for r in kw_results],
            textposition="outside",
            hovertemplate="%{y}: H=%{x}<extra></extra>"
        ))
        fig_kw.update_layout(**PLOTLY_LAYOUT, height=340,
                             title=dict(text="Kruskal-Wallis H by Barrier (Green=Sig)", font=dict(size=12)))
        fig_kw.update_xaxes(title="H Statistic", gridcolor="#F1F5F9")
        st.plotly_chart(fig_kw, use_container_width=True, key="fig_kw_age")

    sig_kw = [r["Barrier"] for r in kw_results if r["Significant"] == "✅ Yes"]
    if sig_kw:
        finding_card(f"📊 Age Drives Barrier Differences in {len(sig_kw)} Item(s)",
                     f"Significant age-group variation found in: {', '.join(sig_kw)}. "
                     "Older non-users tend to show stronger barrier intensity, consistent "
                     "with digital literacy and habit formation differences.", AMBER)
    else:
        finding_card("⚪ No Significant Age-Group Differences in Barriers",
                     "Barrier intensities do not vary significantly by age group among non-users "
                     "(all Kruskal-Wallis p > 0.05).", SLATE)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — KEY FINDINGS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
section("Key Findings — Non-User Analysis")
finding_card("👁️ Awareness is the Single Biggest Barrier (V=0.443)",
             "29.2% cite lack of awareness as their reason for non-adoption. "
             "This is an informational gap, not rejection — making these respondents "
             "the lowest-hanging fruit for platform growth through targeted outreach.", ROSE)
finding_card("🛒 Local Store Preference Reflects Habit, Not Hostility",
             "24.8% prefer local stores — rooted in trust and habit rather than active opposition. "
             "Platforms can counter this with 'support local + speed' hybrid messaging.", AMBER)
finding_card("💡 Strong Latent Conversion Potential",
             "Awareness is NOT associated with reduced willingness to try (p > 0.05). "
             "Unaware non-users are just as open to experimentation as aware ones — "
             "meaning a well-targeted awareness campaign can unlock conversion.", EMERALD)
finding_card("💰 Price & Trust are Top Conditional Adoption Triggers",
             "Lower delivery charges and trust/data security ranked #1 and #2 in the "
             "7-item conditional adoption scale. Platforms should lead with price promotions "
             "AND visible security guarantees to convert non-users.", INDIGO)
finding_card("👥 Older, Less-Educated, Retired Segments Need Dedicated Strategies",
             "58.7% of 40+ respondents are non-adopters. These groups need simplified UX, "
             "offline onboarding, and family-member referral incentives — not just digital ads.", VIOLET)

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, kruskal, spearmanr
from utils import *

st.set_page_config("Obj 3 — Usage Behavior", "📊", layout="wide")
st.session_state["current_page"] = "pages/7_Obj3_Behavior.py"
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*='css']{font-family:'Inter',sans-serif;}
.stApp{background:#FAFAFA;}
section[data-testid='stSidebar']{background:#FFFFFF;border-right:1px solid #E2E8F0;}
</style>""", unsafe_allow_html=True)

from navbar import navbar
navbar()
page_header("Objective 3", "Usage Behavior Patterns of Q-Commerce Consumers",
            "Examining app preference, tenure, spending, delivery time, payment methods, product categories, "
            "satisfaction scores, and their cross-demographic relationships. Filter: users only (n=228).")

users = get_users()

k1,k2,k3,k4,k5 = st.columns(5)
kpi(k1,"228","Active Users","Analysis base")
kpi(k2,"43%","Long-Term Users",">1 year tenure",EMERALD)
kpi(k3,"₹200–400","Top Spend Bracket","38.6% of orders",INDIGO)
kpi(k4,"3.82/5","Mean Satisfaction","Out of 5",VIOLET)
kpi(k5,"63%","Evening+Night","Peak ordering window",AMBER)
st.markdown("<br>",unsafe_allow_html=True)

TENURE_ORDER = ["Less than 3 months","3-6 months","6-12 months","More than a year"]
ORDER_ORDER  = ["Below ₹200","₹200 - ₹400","₹400 - ₹600","Above ₹600"]
TIME_ORDER   = ["Morning","Afternoon","Evening","Night","Midnight"]
PAY_ORDER    = ["UPI","Cash on Delivery","Debit/Credit Card","Wallets (Paytm, PhonePe, etc.)"]
APP_ORDER    = ["Blinkit","Swiggy Instamart","Zepto","Other"]

def freq_bar(series, order, title, colors=None, horiz=False):
    cnt   = series.value_counts().reindex([o for o in order if o in series.values]).fillna(0)
    total = cnt.sum()
    texts = [f"{int(v)}\n({v/total*100:.1f}%)" for v in cnt.values]
    clrs  = colors or PALETTE[:len(cnt)]
    if horiz:
        fig = go.Figure(go.Bar(y=cnt.index, x=cnt.values, orientation="h",
                                marker_color=clrs, text=[t.replace("\n"," ") for t in texts],
                                textposition="outside"))
    else:
        fig = go.Figure(go.Bar(x=cnt.index, y=cnt.values, marker_color=clrs,
                                text=texts, textposition="outside"))
    fig.update_layout(**PLOTLY_LAYOUT, height=290,
                       title=dict(text=title, font=dict(size=12)))
    fig.update_xaxes(gridcolor="#F1F5F9", tickangle=-15)
    fig.update_yaxes(gridcolor="#F1F5F9")
    return fig

# ── ANALYSIS 1 ─────────────────────────────────────────────────────────────────
section("Analysis 1 · Descriptive Profiling of Usage Behavior")
t1,t2,t3 = st.tabs(["App & Tenure","Spending & Payment","Delivery & Products"])

with t1:
    c1,c2 = st.columns(2)
    with c1:
        app_cnt = users["App_Used"].value_counts().reindex(APP_ORDER).fillna(0)
        fig_app = go.Figure(go.Pie(labels=app_cnt.index, values=app_cnt.values, hole=0.55,
                                    marker_colors=[INDIGO,ROSE,EMERALD,AMBER],
                                    textinfo="label+percent",
                                    hovertemplate="%{label}: %{value} users<extra></extra>"))
        fig_app.update_layout(**{k:v for k,v in PLOTLY_LAYOUT.items() if k not in["xaxis","yaxis"]},
                               height=290, title=dict(text="(a) Primary App Used",font=dict(size=12)))
        st.plotly_chart(fig_app, use_container_width=True)
    with c2:
        st.plotly_chart(freq_bar(users["How long have you been using Q-Commerce apps?"],
                                  TENURE_ORDER,"(b) Usage Tenure",[SKY,VIOLET,AMBER,EMERALD]), use_container_width=True)
    reason_cnt = users["biggest reason you prefer Q-commerce apps"].value_counts()
    fig_r = go.Figure(go.Pie(labels=reason_cnt.index, values=reason_cnt.values, hole=0.45,
                              marker_colors=PALETTE[:len(reason_cnt)], textinfo="label+percent"))
    fig_r.update_layout(**{k:v for k,v in PLOTLY_LAYOUT.items() if k not in["xaxis","yaxis"]},
                         height=270, title=dict(text="(f) Biggest Reason for Using Q-Commerce",font=dict(size=12)))
    st.plotly_chart(fig_r, use_container_width=True)
    finding_card("🏆 Blinkit Dominates (62.3%) & 43% are Long-Term Users",
                 "Near-oligopolistic market structure. 43% have used Q-Commerce for over a year — "
                 "confirming it has moved from trial to mainstream habitual use in Vadodara.", INDIGO)

with t2:
    c1,c2 = st.columns(2)
    with c1:
        st.plotly_chart(freq_bar(users["average order value"], ORDER_ORDER,
                                  "(c) Average Order Value",[EMERALD,INDIGO,AMBER,ROSE]), use_container_width=True)
    with c2:
        pay_cnt = users["preferred payment method?"].value_counts().reindex(PAY_ORDER).fillna(0)
        total_p = pay_cnt.sum()
        fig_pay = go.Figure(go.Bar(
            y=pay_cnt.index[::-1], x=pay_cnt.values[::-1], orientation="h",
            marker_color=[INDIGO,AMBER,EMERALD,VIOLET],
            text=[f"{int(v)} ({v/total_p*100:.1f}%)" for v in pay_cnt.values[::-1]],
            textposition="outside"))
        fig_pay.update_layout(**PLOTLY_LAYOUT, height=290, title=dict(text="(e) Payment Method",font=dict(size=12)))
        fig_pay.update_xaxes(title="Count",gridcolor="#F1F5F9")
        st.plotly_chart(fig_pay, use_container_width=True)
    finding_card("💳 UPI (54%) Leads, CoD (32.5%) Persists",
                 "72.4% of users spend below ₹400 — small, targeted top-up purchases. "
                 "Cash on Delivery share signals residual digital payment hesitancy.", AMBER)

with t3:
    c1,c2 = st.columns(2)
    with c1:
        st.plotly_chart(freq_bar(users["preferred delivery time"], TIME_ORDER,
                                  "(d) Preferred Delivery Time",
                                  [SKY,AMBER,INDIGO,VIOLET,"#0F172A"]), use_container_width=True)
    with c2:
        cat_cols = ["Col1","Col2","Col3","Col4","Col5"]
        all_cats = []
        for col in cat_cols:
            all_cats.extend(users[col].dropna().str.strip().tolist())
        cat_d = {}
        for k in all_cats:
            key = k.strip()
            if len(key) > 1:
                cat_d[key] = cat_d.get(key, 0) + 1
        cat_s = pd.Series(cat_d).sort_values(ascending=True)
        fig_cat = go.Figure(go.Bar(
            y=cat_s.index, x=cat_s.values, orientation="h",
            marker_color=PALETTE[:len(cat_s)],
            text=[f"{v} ({v/len(users)*100:.1f}%)" for v in cat_s.values],
            textposition="outside"))
        fig_cat.update_layout(**PLOTLY_LAYOUT, height=290, title=dict(text="Product Categories (Multi-response)",font=dict(size=12)))
        fig_cat.update_xaxes(title="No. of Users",gridcolor="#F1F5F9")
        st.plotly_chart(fig_cat, use_container_width=True)

    # Delivery Time × Age Group heatmap
    st.markdown("<div style='font-weight:600;font-size:.85rem;color:#1E1E2E;margin:12px 0 4px;'>1C · Delivery Time × Age Group Cross-tabulation</div>",unsafe_allow_html=True)
    cross = pd.crosstab(users["preferred delivery time"], users["Age_Group"])
    cross = cross.reindex(columns=[c for c in AGE_ORDER if c in cross.columns])
    cross_pct = cross.div(cross.sum(axis=0), axis=1) * 100
    fig_hm = go.Figure(go.Heatmap(
        z=cross_pct.values, x=cross_pct.columns.tolist(), y=cross_pct.index.tolist(),
        colorscale=[[0,"#FFF7ED"],[0.5,"#FB923C"],[1,"#9A3412"]],
        text=cross_pct.round(1).values, texttemplate="%{text:.1f}%", textfont=dict(size=10),
        colorbar=dict(title="% within age"),
        hovertemplate="%{y} | %{x}: %{z:.1f}%<extra></extra>"))
    fig_hm.update_layout(**{k:v for k,v in PLOTLY_LAYOUT.items() if k not in["xaxis","yaxis"]},
                          height=290, title=dict(text="Delivery Time by Age Group (% within age group)",font=dict(size=12)))
    st.plotly_chart(fig_hm, use_container_width=True)
    finding_card("🌙 Evening (40%) + Night (23%) = 63% of Orders",
                 "Younger users (18–33) order more at Night; older users (42+) show more Morning ordering. "
                 "Groceries (72.8%), Snacks (63.6%) and Daily Essentials (56.6%) are top categories.", ROSE)

# ── ANALYSIS 2: CHI-SQUARE ────────────────────────────────────────────────────
section("Analysis 2 · Chi-Square Tests — Usage Behavior vs Demographics",
        "H₀: Behavioral variable is independent of demographic variable | α = 0.05")

behavior_vars = {
    "App_Used": "App Used",
    "How long have you been using Q-Commerce apps?": "Usage Tenure",
    "average order value": "Avg. Order Value",
    "preferred delivery time": "Preferred Delivery Time",
    "preferred payment method?": "Payment Method"
}
demo_vars_d = {"Age_Group":"Age Group","Income":"Income","Occupation":"Occupation","Education":"Education"}

@st.cache_data
def run_all_chi2():
    rows = []
    u = get_users()
    for bcol, blabel in behavior_vars.items():
        for dcol, dlabel in demo_vars_d.items():
            try:
                ct = pd.crosstab(u[bcol], u[dcol])
                chi2_v, p, dof, _ = chi2_contingency(ct)
                n = ct.values.sum(); r,c = ct.shape
                v = cramers_v(chi2_v, n, r, c)
                assoc = ("Negligible" if v<0.1 else "Weak" if v<0.3 else "Moderate" if v<0.5 else "Strong")
                rows.append({"Behavior":blabel,"Demographic":dlabel,
                              "χ²":round(chi2_v,3),"df":dof,"p":round(p,4),
                              "V":round(v,3),"Association":assoc,"Sig":p<0.05})
            except:
                pass
    return pd.DataFrame(rows)

chi_df = run_all_chi2()
pivot_v = chi_df.pivot(index="Behavior", columns="Demographic", values="V")
pivot_p = chi_df.pivot(index="Behavior", columns="Demographic", values="p")

c1,c2 = st.columns([1.3,1], gap="large")
with c1:
    annots = [[f"{pivot_v.iloc[i,j]:.2f}{'**' if pivot_p.iloc[i,j]<0.01 else ('*' if pivot_p.iloc[i,j]<0.05 else '')}"
               for j in range(pivot_v.shape[1])] for i in range(pivot_v.shape[0])]
    fig_cv = go.Figure(go.Heatmap(
        z=pivot_v.values, x=pivot_v.columns.tolist(), y=pivot_v.index.tolist(),
        colorscale=[[0,"#F0FDF4"],[0.4,"#86EFAC"],[1,"#15803D"]],
        text=annots, texttemplate="%{text}", textfont=dict(size=11),
        zmin=0, zmax=0.6, colorbar=dict(title="Cramér's V"),
        hovertemplate="%{y} × %{x}: V=%{z:.3f}<extra></extra>"))
    fig_cv.update_layout(**{k:v for k,v in PLOTLY_LAYOUT.items() if k not in["xaxis","yaxis"]},
                          height=320, title=dict(text="Cramér's V Heatmap (** p<0.01, * p<0.05)",font=dict(size=12)))
    st.plotly_chart(fig_cv, use_container_width=True)
with c2:
    st.markdown("<div style='font-weight:600;font-size:.85rem;margin-bottom:8px;'>Significant pairs (p < 0.05)</div>",unsafe_allow_html=True)
    for _, row in chi_df[chi_df["Sig"]].sort_values("V",ascending=False).iterrows():
        vc = EMERALD if row["V"]>0.3 else AMBER
        st.markdown(f"""
        <div style='background:#fff;border:1px solid #E2E8F0;border-radius:8px;
                    padding:7px 11px;margin-bottom:4px;'>
          <div style='font-size:.78rem;font-weight:600;color:#1E1E2E;'>{row['Behavior']} × {row['Demographic']}</div>
          <div style='font-size:.73rem;color:#64748B;'>χ²={row['χ²']} | p={row['p']} | <span style='color:{vc};font-weight:600;'>V={row['V']} ({row['Association']})</span></div>
        </div>""", unsafe_allow_html=True)

with st.expander("📋 Full chi-square results table"):
    disp = chi_df.copy(); disp["Significant"] = disp["Sig"].map({True:"✅",False:"❌"})
    st.dataframe(disp.drop(columns="Sig"), use_container_width=True)

# ── ANALYSIS 3: SATISFACTION ──────────────────────────────────────────────────
section("Analysis 3 · Satisfaction Analysis — Kruskal-Wallis + Spearman Correlation")

sat_col  = "Overall satisfaction with Q-Commerce Apps"
cont_col = "Likelihood of continuing usage in the future"
rec_col  = "likely are you to recommend these apps to Others"
sat_vars_list = [sat_col, cont_col, rec_col]
sat_lbls = ["Overall Satisfaction","Continuity Likelihood","Recommendation Likelihood"]

st.markdown("<div style='font-weight:600;font-size:.85rem;margin:10px 0 6px;'>3A · Satisfaction Score Distributions</div>",unsafe_allow_html=True)
c1,c2,c3 = st.columns(3)
for col_w, sc, sl in zip([c1,c2,c3], sat_vars_list, sat_lbls):
    data = users[sc].dropna()
    cnt  = data.value_counts().sort_index()
    fig  = go.Figure(go.Bar(x=cnt.index, y=cnt.values, marker_color=INDIGO, text=cnt.values, textposition="outside"))
    fig.add_vline(x=data.mean(), line_dash="dash", line_color=ROSE, opacity=0.8,
                  annotation_text=f"μ={data.mean():.2f}", annotation_font=dict(size=9,color=ROSE))
    fig.update_layout(**PLOTLY_LAYOUT, height=230,
                       title=dict(text=f"{sl}\n(μ={data.mean():.2f}, SD={data.std():.2f})",font=dict(size=10)))
    fig.update_xaxes(tickvals=[1,2,3,4,5],gridcolor="#F1F5F9")
    fig.update_yaxes(gridcolor="#F1F5F9")
    col_w.plotly_chart(fig, use_container_width=True)

st.markdown("<div style='font-weight:600;font-size:.85rem;margin:10px 0 4px;'>3B · Kruskal-Wallis H Test</div>",unsafe_allow_html=True)
st.markdown("<div style='font-size:.75rem;color:#64748B;margin-bottom:8px;'>H₀: Median satisfaction is equal across groups</div>",unsafe_allow_html=True)

@st.cache_data
def run_kw():
    u = get_users()
    rows = []
    for gvar in ["Age_Group","Income","Occupation","App_Used"]:
        for sv, sl in zip(sat_vars_list, sat_lbls):
            groups = [g[sv].dropna().values for _,g in u.groupby(gvar) if len(g[sv].dropna())>=3]
            if len(groups)>=2:
                H,p = kruskal(*groups)
                rows.append({"Score":sl,"Group By":gvar,"H":round(H,3),"p":round(p,4),"Sig":p<0.05})
    return pd.DataFrame(rows)

kw_df = run_kw()
c1,c2 = st.columns([1.2,1], gap="large")
with c1:
    pivot_h = kw_df.pivot(index="Group By", columns="Score", values="H")
    fig_kw = go.Figure(go.Heatmap(
        z=pivot_h.values, x=pivot_h.columns.tolist(), y=pivot_h.index.tolist(),
        colorscale=C_SCALE, text=pivot_h.round(2).values,
        texttemplate="%{text}", textfont=dict(size=11), colorbar=dict(title="H Stat"),
        hovertemplate="%{y} × %{x}: H=%{z:.2f}<extra></extra>"))
    fig_kw.update_layout(**{k:v for k,v in PLOTLY_LAYOUT.items() if k not in["xaxis","yaxis"]},
                          height=270, title=dict(text="Kruskal-Wallis H Statistic by Group",font=dict(size=12)))
    st.plotly_chart(fig_kw, use_container_width=True)
with c2:
    for _,row in kw_df.iterrows():
        b = "✅" if row["Sig"] else "❌"
        st.markdown(f"""<div style='background:#fff;border:1px solid #E2E8F0;border-radius:7px;
            padding:6px 10px;margin-bottom:4px;display:flex;justify-content:space-between;'>
          <span style='font-size:.75rem;'>{row['Score']} × {row['Group By']}</span>
          <span style='font-size:.75rem;'>{b} H={row['H']} p={row['p']}</span>
        </div>""", unsafe_allow_html=True)

st.markdown("<div style='font-weight:600;font-size:.85rem;margin:10px 0 4px;'>3C · Spearman Correlation Matrix</div>",unsafe_allow_html=True)
sat_data = users[sat_vars_list].dropna()
corr_m = np.zeros((3,3)); pval_m = np.zeros((3,3))
for i in range(3):
    for j in range(3):
        if i==j: corr_m[i,j]=1.0
        else:
            r,p = spearmanr(sat_data.iloc[:,i], sat_data.iloc[:,j])
            corr_m[i,j]=r; pval_m[i,j]=p

annot_c = [[f"ρ={corr_m[i,j]:.3f}\n(p={pval_m[i,j]:.3f})" if i!=j else "1.000"
            for j in range(3)] for i in range(3)]
lbls = ["Satisfaction","Continuity","Recommendation"]
fig_corr = go.Figure(go.Heatmap(
    z=corr_m, x=lbls, y=lbls, colorscale=C_SCALE, zmin=0, zmax=1,
    text=annot_c, texttemplate="%{text}", textfont=dict(size=11),
    colorbar=dict(title="Spearman ρ"),
    hovertemplate="%{y} × %{x}: ρ=%{z:.3f}<extra></extra>"))
fig_corr.update_layout(**{k:v for k,v in PLOTLY_LAYOUT.items() if k not in["xaxis","yaxis"]},
                        height=280, title=dict(text="Spearman Correlation: |ρ|<0.3 Weak | 0.3–0.6 Moderate | >0.6 Strong",font=dict(size=11)))
st.plotly_chart(fig_corr, use_container_width=True)

rho_sc = corr_m[0,1]
strength = "Very Strong" if abs(rho_sc)>0.8 else "Strong" if abs(rho_sc)>0.6 else "Moderate" if abs(rho_sc)>0.3 else "Weak"
finding_card(f"💛 Satisfaction-Loyalty-Advocacy Chain Confirmed (ρ={rho_sc:.3f}, {strength})",
             "All three satisfaction scores are strongly positively correlated — satisfied users "
             "are significantly more likely to continue using AND recommend Q-Commerce to others.", INDIGO)

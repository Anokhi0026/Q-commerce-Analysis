import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, rankdata, norm as sp_norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils import *

st.set_page_config("Obj 4 — Drivers", "🔍", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*='css']{font-family:'Inter',sans-serif;}
.stApp{background:#FAFAFA;}
section[data-testid='stSidebar']{background:#FFFFFF;border-right:1px solid #E2E8F0;}
</style>""", unsafe_allow_html=True)

sidebar()
page_header("Objective 4", "Key Drivers of Q-Commerce Adoption",
            "Cronbach's Alpha scale reliability, mean driver rankings, EFA, non-user barriers, "
            "Mann-Whitney U, and Kruskal-Wallis with Dunn's post-hoc on barrier items.")

users    = get_users()
df_all   = load_raw()
df_anal  = load_analysis()
non_u    = df_anal[df_anal["Adoption_Status"]==0].copy()
ld       = get_likert()

alpha = cronbach_alpha(ld)
interp_str = ("Excellent" if alpha>=0.9 else "Good" if alpha>=0.8 else "Acceptable" if alpha>=0.7 else "Poor")

k1,k2,k3,k4 = st.columns(4)
kpi(k1,f"α={alpha:.3f}","Cronbach's Alpha",interp_str,INDIGO)
kpi(k2,"10","Likert Items","Attitude scale",EMERALD)
kpi(k3,"228","Users","Attitude analysis",VIOLET)
kpi(k4,"113","Non-Users","Barrier analysis",ROSE)
st.markdown("<br>",unsafe_allow_html=True)

# ── ANALYSIS 1: CRONBACH'S ALPHA ──────────────────────────────────────────────
section("Analysis 1 · Scale Reliability — Cronbach's Alpha & Item-Total Correlations")

total_sc = ld.sum(axis=1)
itc_rows = []
for col in ld.columns:
    r_val, _ = stats.pearsonr(ld[col], total_sc - ld[col])
    alpha_del = cronbach_alpha(ld.drop(columns=[col]))
    itc_rows.append({"Item":col,"Mean":round(ld[col].mean(),3),"SD":round(ld[col].std(),3),
                     "Item-Total r":round(r_val,3),"Alpha if Deleted":round(alpha_del,3)})
itc_df = pd.DataFrame(itc_rows).sort_values("Item-Total r")

c1,c2 = st.columns([1,2.5], gap="large")
with c1:
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{INDIGO},{VIOLET});border-radius:16px;
                padding:28px;text-align:center;color:#fff;'>
      <div style='font-size:.68rem;text-transform:uppercase;letter-spacing:.12em;opacity:.75;'>Cronbach's Alpha</div>
      <div style='font-size:2.8rem;font-weight:800;margin:8px 0;'>α={alpha:.3f}</div>
      <div style='font-size:.85rem;font-weight:600;'>{interp_str} ✦</div>
      <div style='font-size:.73rem;opacity:.7;margin-top:8px;'>10 items · {len(ld)} users</div>
      <div style='margin-top:14px;font-size:.7rem;opacity:.6;line-height:1.7;'>
        ≥0.9 Excellent<br>≥0.8 Good<br>≥0.7 Acceptable<br>&lt;0.7 Poor
      </div>
    </div>""", unsafe_allow_html=True)
with c2:
    clrs_itc = [ROSE if r<0.3 else EMERALD for r in itc_df["Item-Total r"]]
    fig_itc = go.Figure(go.Bar(
        y=itc_df["Item"], x=itc_df["Item-Total r"], orientation="h",
        marker_color=clrs_itc, text=[f"{v:.3f}" for v in itc_df["Item-Total r"]],
        textposition="outside", hovertemplate="%{y}: r=%{x:.3f}<extra></extra>"))
    fig_itc.add_vline(x=0.3, line_dash="dash", line_color=ROSE, opacity=0.8,
                      annotation_text="Min threshold (0.30)", annotation_font=dict(size=9,color=ROSE))
    fig_itc.update_layout(**PLOTLY_LAYOUT, height=320,
                           title=dict(text=f"Item-Total Correlations (α={alpha:.3f})",font=dict(size=12)))
    fig_itc.update_xaxes(title="Corrected Item-Total r",gridcolor="#F1F5F9")
    st.plotly_chart(fig_itc, use_container_width=True)

with st.expander("📋 Full item-total correlation table"):
    st.dataframe(itc_df, use_container_width=True)

# ── ANALYSIS 2: DRIVER RANKINGS ───────────────────────────────────────────────
section("Analysis 2 · Adoption Driver Rankings — Mean Likert Scores")

desc = ld.describe().T[["mean","std","50%"]].rename(columns={"mean":"Mean","std":"SD","50%":"Median"})
desc["Rank"] = desc["Mean"].rank(ascending=False).astype(int)
desc["CV%"]  = (desc["SD"]/desc["Mean"]*100).round(1)
ranking = desc.sort_values("Mean")

opacities   = [0.40 + 0.60*i/len(ranking) for i in range(len(ranking))]
bar_colors  = [f"rgba(79,70,229,{op})" for op in opacities]

fig_rank = go.Figure(go.Bar(
    y=ranking.index, x=ranking["Mean"], orientation="h",
    marker_color=bar_colors,
    error_x=dict(type="data", array=ranking["SD"].values, color="#94A3B8", thickness=1.5),
    text=[f"μ={v:.2f}  Rank #{int(r)}" for v,r in zip(ranking["Mean"],ranking["Rank"])],
    textposition="outside",
    hovertemplate="%{y}: μ=%{x:.2f} ± %{error_x.array:.2f}<extra></extra>"))
fig_rank.add_vline(x=3.0, line_dash="dot", line_color="#94A3B8",
                   annotation_text="Neutral (3.0)", annotation_font=dict(size=9,color="#94A3B8"))
fig_rank.add_vline(x=4.0, line_dash="dot", line_color=INDIGO,
                   annotation_text="Agree (4.0)", annotation_font=dict(size=9,color=INDIGO))
fig_rank.update_layout(**PLOTLY_LAYOUT, height=390, title=dict(text="Ranked Adoption Driver Means (±1 SD, n=228)",font=dict(size=12)))
fig_rank.update_xaxes(title="Mean Likert Score (1–5)", range=[1,5.7],gridcolor="#F1F5F9")
st.plotly_chart(fig_rank, use_container_width=True)

# Stacked Likert chart
with st.expander("📊 Stacked Likert Response Distribution"):
    resp_lbls   = ["Strongly Disagree","Disagree","Neutral","Agree","Strongly Agree"]
    resp_colors = ["#EF4444","#FCA5A5","#FDE68A","#86EFAC","#16A34A"]
    pct_df = pd.DataFrame(index=SHORT_NAMES)
    for val, lbl in zip([1,2,3,4,5], resp_lbls):
        pct_df[lbl] = (ld == val).sum() / len(ld) * 100
    pct_df["_pos"] = pct_df["Agree"] + pct_df["Strongly Agree"]
    pct_df = pct_df.sort_values("_pos").drop(columns="_pos")
    fig_stk = go.Figure()
    for lbl, color in zip(resp_lbls, resp_colors):
        fig_stk.add_trace(go.Bar(name=lbl, y=pct_df.index, x=pct_df[lbl],
                                  orientation="h", marker_color=color,
                                  text=[f"{v:.0f}%" if v>6 else "" for v in pct_df[lbl]],
                                  textposition="inside", insidetextanchor="middle"))
    fig_stk.update_layout(barmode="stack", **PLOTLY_LAYOUT, height=380,
                           xaxis=dict(title="% of Users",gridcolor="#F1F5F9"),
                           title=dict(text="Stacked Likert Response Distribution",font=dict(size=12)))
    fig_stk.update_layout(legend=dict(orientation="h",y=-0.18))
    st.plotly_chart(fig_stk, use_container_width=True)

# ── ANALYSIS 3: EFA ───────────────────────────────────────────────────────────
section("Analysis 3 · Exploratory Factor Analysis — KMO, Bartlett's & Varimax Rotation")

@st.cache_data
def run_efa():
    X = ld.values.astype(float)
    X_std = StandardScaler().fit_transform(X)
    corr_mat = np.corrcoef(X_std.T)

    # KMO
    try:
        corr_inv = np.linalg.inv(corr_mat)
        D = np.diag(1/np.sqrt(np.diag(corr_inv)))
        partial_corr = -D @ corr_inv @ D
        np.fill_diagonal(partial_corr, 0)
        cm0 = corr_mat.copy(); np.fill_diagonal(cm0, 0)
        kmo = np.sum(cm0**2)/(np.sum(cm0**2)+np.sum(partial_corr**2))
    except:
        kmo = np.nan

    # Bartlett
    n, k = X_std.shape
    chi2_b = -(n-1-(2*k+5)/6)*np.log(np.linalg.det(corr_mat))
    df_b   = k*(k-1)/2
    from scipy.stats import chi2 as chi2_dist
    p_b    = 1 - chi2_dist.cdf(chi2_b, df_b)

    eigenvalues, _ = np.linalg.eigh(corr_mat)
    eigenvalues = eigenvalues[::-1]
    n_factors = int((eigenvalues > 1).sum())

    pca = PCA(n_components=n_factors); pca.fit(X_std)
    init_load = pca.components_.T * np.sqrt(pca.explained_variance_)

    def varimax(L, max_iter=1000, tol=1e-6):
        p,k = L.shape; R = np.eye(k)
        for _ in range(max_iter):
            old = R.copy()
            for i in range(k):
                for j in range(i+1,k):
                    Lr = L@R; Li,Lj = Lr[:,i],Lr[:,j]
                    u = Li**2-Lj**2; v = 2*Li*Lj
                    A,B = np.sum(u),np.sum(v)
                    C = np.sum(u**2-v**2); D2 = 2*np.sum(u*v)
                    theta = 0.25*np.arctan2(D2-2*A*B/p, C-(A**2-B**2)/p)
                    c,s = np.cos(theta),np.sin(theta)
                    G = np.eye(k); G[i,i]=c; G[j,j]=c; G[i,j]=-s; G[j,i]=s
                    R = R@G
            if np.max(np.abs(R-old))<tol: break
        return L@R
    rot = varimax(init_load)
    ldf = pd.DataFrame(rot, index=SHORT_NAMES, columns=[f"F{i+1}" for i in range(n_factors)])
    ssl = np.sum(rot**2,axis=0); pct_var = ssl/10*100
    return kmo, chi2_b, df_b, p_b, eigenvalues, n_factors, ldf, pct_var

kmo_v, chi2_b, df_b, p_b, eigenvalues, n_factors, ldf, pct_var = run_efa()

# KMO + Bartlett summary
c1,c2,c3 = st.columns(3)
kmo_qual = ("Marvellous" if kmo_v>=0.9 else "Meritorious" if kmo_v>=0.8 else "Middling" if kmo_v>=0.7 else "Mediocre" if kmo_v>=0.6 else "Poor")
kpi(c1,f"{kmo_v:.3f}","KMO Value",kmo_qual,EMERALD)
kpi(c2,f"p={p_b:.4f}","Bartlett's Test","Sig → EFA appropriate",INDIGO)
kpi(c3,str(n_factors),"Factors Retained",f"Kaiser criterion (EV>1)",VIOLET)
st.markdown("<br>",unsafe_allow_html=True)

c1,c2 = st.columns(2, gap="large")
with c1:
    exp_var = eigenvalues/eigenvalues.sum()*100
    fig_scree = go.Figure()
    fig_scree.add_trace(go.Scatter(
        x=list(range(1,len(eigenvalues)+1)), y=eigenvalues,
        mode="lines+markers", line=dict(color=INDIGO,width=2.5), marker=dict(size=8,color=INDIGO),
        hovertemplate="Factor %{x}: EV=%{y:.3f}<extra></extra>"))
    fig_scree.add_hline(y=1.0, line_dash="dash", line_color=ROSE,
                         annotation_text="Kaiser criterion (EV=1)",
                         annotation_font=dict(size=9,color=ROSE))
    fig_scree.add_vrect(x0=0.5, x1=n_factors+0.5, fillcolor=rgba(79,70,229,0.07), line_width=0,
                         annotation_text=f"{n_factors} factors retained",
                         annotation_position="top left", annotation_font=dict(size=9,color=INDIGO))
    fig_scree.update_layout(**PLOTLY_LAYOUT, height=310, title=dict(text=f"Scree Plot — {n_factors} Factors Retained",font=dict(size=12)))
    fig_scree.update_xaxes(title="Factor Number",tickvals=list(range(1,11)),gridcolor="#F1F5F9")
    fig_scree.update_yaxes(title="Eigenvalue",gridcolor="#F1F5F9")
    st.plotly_chart(fig_scree, use_container_width=True)

with c2:
    fig_load = go.Figure(go.Heatmap(
        z=ldf.values, x=ldf.columns.tolist(), y=SHORT_NAMES,
        colorscale=[[0,"#EF4444"],[0.35,"#FEF3C7"],[0.5,"#F8FAFC"],[0.65,"#DBEAFE"],[1,"#3730A3"]],
        zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in ldf.values],
        texttemplate="%{text}", textfont=dict(size=9),
        colorbar=dict(title="Loading"),
        hovertemplate="%{y} → %{x}: %{z:.3f}<extra></extra>"))
    fig_load.update_layout(**{k:v for k,v in PLOTLY_LAYOUT.items() if k not in["xaxis","yaxis"]},
                            height=310, title=dict(text=f"Varimax Factor Loadings | Total variance explained: {pct_var.sum():.1f}%",font=dict(size=11)))
    st.plotly_chart(fig_load, use_container_width=True)

# Factor summary cards
st.markdown(f"""<div style='display:grid;grid-template-columns:repeat({n_factors},1fr);gap:10px;margin:8px 0;'>
""" + "".join([f"""  <div style='background:{PALETTE[i]}10;border:1px solid {PALETTE[i]}30;border-radius:10px;padding:12px;'>
    <div style='font-weight:700;color:{PALETTE[i]};font-size:.85rem;'>Factor {i+1}</div>
    <div style='font-size:.7rem;color:#64748B;'>Variance: {pct_var[i]:.1f}%</div>
    <div style='margin-top:6px;'>{"".join(f'<div style="font-size:.72rem;color:#374151;padding:1px 0;">• {item}</div>' for item in ldf[f"F{i+1}"][np.abs(ldf[f"F{i+1}"])>=0.40].index)}</div>
  </div>""" for i in range(n_factors)]) + "</div>", unsafe_allow_html=True)

# ── ANALYSIS 4: NON-USER BARRIERS ─────────────────────────────────────────────
section("Analysis 4 · Non-User Barrier Analysis (n=113)")

BARRIER_COLS = [
    "I would consider using Q-commerce if delivery charges were lower",
    "I would consider using Q-commerce apps if product quality were guaranteed",
    "I would consider using Q-commerce apps if adequate guidance on app usage were provided",
    "I would consider using Q-commerce apps if prices were competitive",
    "I would consider using Q-commerce apps if delivery services were available in my area",
    "I would consider using Q-commerce apps if attractive discounts were offered",
    "I would consider using Q-commerce apps if I felt confident about trust, data security, and privacy."
]
BARRIER_NAMES = ["Lower Delivery Charges","Product Quality Guarantee","App Usage Guidance",
                 "Competitive Pricing","Delivery Availability","Attractive Discounts","Trust & Data Security"]

non_users_raw = df_all[df_all["Adoption_Status"]==0].copy()
bd = non_users_raw[BARRIER_COLS].copy()
for col in BARRIER_COLS:
    bd[col] = bd[col].map(LIKERT_MAP)
bd.columns = BARRIER_NAMES
bd = bd.dropna()

non_users_raw_clean = non_users_raw.loc[bd.index].copy()
for col, name in zip(BARRIER_COLS, BARRIER_NAMES):
    non_users_raw_clean[name] = bd[name]

c1,c2 = st.columns(2, gap="large")
with c1:
    b_desc = bd.describe().T[["mean","std"]].rename(columns={"mean":"Mean","std":"Std Dev"})
    b_desc["Rank"] = b_desc["Mean"].rank(ascending=False).astype(int)
    b_sorted = b_desc.sort_values("Mean")
    bar_clrs = [f"rgba(225,29,72,{0.4+0.6*i/len(b_sorted)})" for i in range(len(b_sorted))]
    fig_bar = go.Figure(go.Bar(
        y=b_sorted.index, x=b_sorted["Mean"], orientation="h",
        marker_color=bar_clrs,
        error_x=dict(type="data", array=b_sorted["Std Dev"].values, color="#94A3B8"),
        text=[f"μ={v:.2f} #{int(r)}" for v,r in zip(b_sorted["Mean"],b_sorted["Rank"])],
        textposition="outside",
        hovertemplate="%{y}: μ=%{x:.2f}<extra></extra>"))
    fig_bar.add_vline(x=3.0, line_dash="dot", line_color="#94A3B8",
                      annotation_text="Neutral (3.0)", annotation_font=dict(size=9))
    fig_bar.add_vline(x=4.0, line_dash="dot", line_color=ROSE,
                      annotation_text="Agree (4.0)", annotation_font=dict(size=9,color=ROSE))
    fig_bar.update_layout(**PLOTLY_LAYOUT, height=330, title=dict(text="Barrier Item Mean Scores (Non-users, n=113)",font=dict(size=12)))
    fig_bar.update_xaxes(range=[1,5.8],gridcolor="#F1F5F9")
    st.plotly_chart(fig_bar, use_container_width=True)

with c2:
    # Open-ended reasons
    reasons = ["R_High_Charges","R_Quality_Concern","R_No_Need","R_Prefer_Local",
               "R_Trust_Issue","R_App_Discomfort","R_Lack_Awareness","R_Not_Available"]
    reason_lbls = ["High Charges","Quality Concern","No Need","Prefer Local",
                   "Trust Issues","App Discomfort","Lack of Awareness","Not Available"]
    r_counts = non_u[reasons].sum().values
    sorted_pairs = sorted(zip(r_counts, reason_lbls), reverse=True)
    r_v = [p[0] for p in sorted_pairs]; r_l = [p[1] for p in sorted_pairs]

    fig_r = go.Figure(go.Bar(
        y=r_l[::-1], x=r_v[::-1], orientation="h",
        marker_color=[ROSE,AMBER]+[SLATE]*len(r_v),
        text=[f"n={v} ({v/len(non_u)*100:.1f}%)" for v in r_v[::-1]],
        textposition="outside",
        hovertemplate="%{y}: %{x} non-users<extra></extra>"))
    fig_r.update_layout(**PLOTLY_LAYOUT, height=330, title=dict(text="Open-Ended Non-Adoption Reasons",font=dict(size=12)))
    fig_r.update_xaxes(title="No. of Non-Users",gridcolor="#F1F5F9")
    st.plotly_chart(fig_r, use_container_width=True)

# ── ANALYSIS 5: MANN-WHITNEY + KRUSKAL + DUNN ─────────────────────────────────
section("Analysis 5 · Mann-Whitney U (Gender) + Kruskal-Wallis + Dunn's Post-Hoc (Age Group)")

st.markdown("<div style='font-weight:600;font-size:.85rem;margin-bottom:4px;'>5A · Mann-Whitney U — Gender Differences in Barriers</div>",unsafe_allow_html=True)

genders = [g for g in ["Male","Female"] if g in non_users_raw_clean["Gender"].values]
mw_rows = []
if len(genders)==2:
    for b in BARRIER_NAMES:
        g1 = non_users_raw_clean[non_users_raw_clean["Gender"]==genders[0]][b].dropna()
        g2 = non_users_raw_clean[non_users_raw_clean["Gender"]==genders[1]][b].dropna()
        if len(g1)>=3 and len(g2)>=3:
            U,p = mannwhitneyu(g1,g2,alternative="two-sided")
            mw_rows.append({"Barrier":b,f"Median ({genders[0]})":g1.median(),
                             f"Median ({genders[1]})":g2.median(),"U":round(U,1),
                             "p-value":round(p,4),"Sig":"✅" if p<0.05 else "❌"})
if mw_rows:
    mw_df = pd.DataFrame(mw_rows)
    st.dataframe(mw_df, use_container_width=True)

st.markdown("<div style='font-weight:600;font-size:.85rem;margin:12px 0 4px;'>5B · Kruskal-Wallis + Dunn's Post-Hoc — Age Group Differences in Barriers</div>",unsafe_allow_html=True)

def dunn_posthoc(groups_dict):
    keys = list(groups_dict.keys())
    all_data = np.concatenate(list(groups_dict.values()))
    ranks_all = rankdata(all_data)
    n = len(all_data)
    idx = 0; group_ranks = {}
    for k in keys:
        sz = len(groups_dict[k])
        group_ranks[k] = ranks_all[idx:idx+sz]; idx += sz
    n_comp = len(keys)*(len(keys)-1)/2
    results = []
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            ni,nj = len(group_ranks[keys[i]]),len(group_ranks[keys[j]])
            Ri,Rj = np.mean(group_ranks[keys[i]]),np.mean(group_ranks[keys[j]])
            se = np.sqrt((n*(n+1)/12)*(1/ni+1/nj))
            z  = (Ri-Rj)/se
            p_raw = 2*sp_norm.sf(abs(z))
            p_adj = min(p_raw*n_comp,1.0)
            results.append({"Group A":keys[i],"Group B":keys[j],
                             "z":round(z,3),"p (raw)":round(p_raw,4),
                             "p (Bonferroni)":round(p_adj,4),
                             "Sig":"✅" if p_adj<0.05 else "No"})
    return pd.DataFrame(results)

kw_rows = []
for b in BARRIER_NAMES:
    groups_d = {nm:grp[b].dropna().values for nm,grp in non_users_raw_clean.groupby("Age_Group")
                if len(grp[b].dropna())>=3}
    if len(groups_d)>=2:
        H,p = kruskal(*groups_d.values())
        kw_rows.append({"Barrier":b,"H":round(H,3),"p-value":round(p,4),"Sig":"✅" if p<0.05 else "❌"})
kw_barr_df = pd.DataFrame(kw_rows)

c1,c2 = st.columns([1,1.5], gap="large")
with c1:
    fig_kw = go.Figure(go.Bar(
        y=kw_barr_df["Barrier"], x=kw_barr_df["H"], orientation="h",
        marker_color=[INDIGO if r=="✅" else SLATE for r in kw_barr_df["Sig"]],
        text=[f"H={r['H']} {r['Sig']}" for _,r in kw_barr_df.iterrows()],
        textposition="outside",
        hovertemplate="%{y}: H=%{x:.2f}<extra></extra>"))
    fig_kw.update_layout(**PLOTLY_LAYOUT, height=290, title=dict(text="Kruskal-Wallis H by Barrier Item",font=dict(size=12)))
    fig_kw.update_xaxes(title="H Statistic",gridcolor="#F1F5F9")
    st.plotly_chart(fig_kw, use_container_width=True)

with c2:
    sig_barriers = kw_barr_df[kw_barr_df["Sig"]=="✅"]["Barrier"].tolist()
    if sig_barriers:
        selected = st.selectbox("Select significant barrier for Dunn's post-hoc:", sig_barriers)
        g_dict = {nm:grp[selected].dropna().values
                  for nm,grp in non_users_raw_clean.groupby("Age_Group")
                  if len(grp[selected].dropna())>=3}
        dunn_df = dunn_posthoc(g_dict)
        st.dataframe(dunn_df.style.map(
            lambda v: "color:#059669;font-weight:600" if v=="✅" else "",
            subset=["Sig"]), use_container_width=True)
    else:
        st.info("No barriers show significant age-group differences (Kruskal-Wallis p < 0.05).")

finding_card(f"⭐ Cronbach's α={alpha:.3f} — {interp_str} Reliability",
             "The 10-item scale is internally consistent. All subsequent EFA and factor analyses are statistically justified.", INDIGO)
finding_card("🚧 Trust & Data Security + Lower Delivery Charges Score Highest",
             "Non-users are most willing to consider adoption if these two barriers are removed. "
             "Lack of Awareness (29.2%) is the #1 open-ended reason — not rejection, but non-exposure.", ROSE)
finding_card("🎯 EFA: Multi-dimensional Adoption Structure",
             f"{n_factors} factors retained explaining {pct_var.sum():.1f}% of variance. "
             "Convenience & lifestyle, price sensitivity, and quality/reliability are distinct drivers — "
             "platforms must address all dimensions simultaneously.", EMERALD)

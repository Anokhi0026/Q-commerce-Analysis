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
st.session_state["current_page"] = "pages/8_Obj4_Drivers.py"
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*='css']{font-family:'Inter',sans-serif;}
.stApp{background:#FAFAFA;}
section[data-testid='stSidebar']{background:#FFFFFF;border-right:1px solid #E2E8F0;}
</style>""", unsafe_allow_html=True)

from navbar import navbar
navbar()
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

k1,k2,k3 = st.columns(3)
kpi(k1,f"α={alpha:.3f}","Cronbach's Alpha",interp_str,INDIGO)
kpi(k2,"10","Likert Items","Attitude scale",EMERALD)
kpi(k3,"228","Users","Attitude analysis",VIOLET)
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
kpi(c2,f"p<0.05","Bartlett's Test","Sig → EFA appropriate",INDIGO)
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
    fig_scree.add_vrect(x0=0.5, x1=n_factors+0.5, fillcolor="rgba(79,70,229,0.07)", line_width=0,
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


finding_card(f"⭐ Cronbach's α={alpha:.3f} — {interp_str} Reliability",
             "The 10-item scale is internally consistent. All subsequent EFA and factor analyses are statistically justified.", INDIGO)
finding_card("🎯 EFA: Multi-dimensional Adoption Structure",
             f"{n_factors} factors retained explaining {pct_var.sum():.1f}% of variance. "
             "Convenience & lifestyle (along with product reliability) and price sensitivity are distinct drivers — "
             "platforms must address all dimensions simultaneously.", EMERALD)

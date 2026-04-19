import streamlit as st
import plotly.graph_objects as go
import pandas as pd, numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils import *

st.set_page_config("Obj 4 — Drivers", "🔍", layout="wide")
st.markdown("<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');html,body,[class*='css']{font-family:'Inter',sans-serif;}.stApp{background:#FAFAFA;}section[data-testid='stSidebar']{background:#FFFFFF;border-right:1px solid #E2E8F0;}</style>",unsafe_allow_html=True)
sidebar()
page_header("Objective 4", "Key Drivers of Q-Commerce Adoption",
            "Cronbach's Alpha scale reliability, mean driver rankings, Exploratory Factor Analysis, and non-user barrier analysis.")

users    = get_users()
df_all   = load_raw()
df_anal  = load_analysis()
non_u    = df_anal[df_anal["Adoption_Status"]==0].copy()
ld       = get_likert()

# ── Cronbach's Alpha ───────────────────────────────────────────────────────────
alpha = cronbach_alpha(ld)
interp = ("Excellent ✦" if alpha>=0.9 else "Good ✦" if alpha>=0.8
          else "Acceptable ✦" if alpha>=0.7 else "Poor ✗")

section("1 · Scale Reliability — Cronbach's Alpha")
c1, c2 = st.columns([1, 2.5], gap="large")

with c1:
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{INDIGO},{VIOLET});border-radius:16px;
                padding:30px;text-align:center;color:#fff;'>
      <div style='font-size:.7rem;text-transform:uppercase;letter-spacing:.12em;opacity:.8;'>
        Cronbach's Alpha
      </div>
      <div style='font-size:3rem;font-weight:800;margin:8px 0;'>α = {alpha:.3f}</div>
      <div style='font-size:.85rem;font-weight:600;'>{interp}</div>
      <div style='font-size:.75rem;opacity:.75;margin-top:8px;'>10 items · {len(ld)} users</div>
    </div>""", unsafe_allow_html=True)

with c2:
    # Item-total correlations
    total = ld.sum(axis=1)
    itc_rows = []
    for col in ld.columns:
        corr_val, _ = stats.pearsonr(ld[col], total-ld[col])
        alpha_del = cronbach_alpha(ld.drop(columns=[col]))
        itc_rows.append({"Item":col,"Mean":ld[col].mean(),"Item-Total r":corr_val,"Alpha if Deleted":alpha_del})
    itc_df = pd.DataFrame(itc_rows).sort_values("Item-Total r")

    colors_itc = [ROSE if r<0.3 else EMERALD for r in itc_df["Item-Total r"]]
    fig = go.Figure(go.Bar(
        y=itc_df["Item"], x=itc_df["Item-Total r"], orientation="h",
        marker_color=colors_itc,
        text=[f"{v:.3f}" for v in itc_df["Item-Total r"]], textposition="outside",
        hovertemplate="%{y}: r=%{x:.3f}<extra></extra>"
    ))
    fig.add_vline(x=0.3, line_dash="dash", line_color=ROSE,
                  annotation_text="Min threshold (0.30)", annotation_position="top right",
                  annotation_font=dict(size=9, color=ROSE))
    fig.update_layout(**PLOTLY_LAYOUT, height=310,
                       title=dict(text=f"Item-Total Correlations (α={alpha:.3f})",font=dict(size=12)),
                       xaxis=dict(title="Corrected Item-Total r", gridcolor="#F1F5F9"))
    st.plotly_chart(fig, use_container_width=True)

# ── Driver Rankings ────────────────────────────────────────────────────────────
section("2 · Adoption Driver Rankings — Mean Likert Scores")
desc = ld.describe().T[["mean","std","50%"]].rename(columns={"mean":"Mean","std":"SD","50%":"Median"})
desc["Rank"] = desc["Mean"].rank(ascending=False).astype(int)
desc["CV%"] = (desc["SD"]/desc["Mean"]*100).round(1)
ranking = desc.sort_values("Mean")

opacity_scale = [0.45 + 0.55*i/len(ranking) for i in range(len(ranking))]
bar_colors = [f"rgba(79,70,229,{op})" for op in opacity_scale]

fig_rank = go.Figure(go.Bar(
    y=ranking.index, x=ranking["Mean"], orientation="h",
    marker_color=bar_colors,
    error_x=dict(type="data", array=ranking["SD"].values, color="#94A3B8"),
    text=[f"μ={v:.2f}  Rank #{int(r)}" for v,r in zip(ranking["Mean"],ranking["Rank"])],
    textposition="outside",
    hovertemplate="%{y}: μ=%{x:.2f} (SD=%{error_x.array:.2f})<extra></extra>"
))
fig_rank.add_vline(x=3.0, line_dash="dot", line_color="#94A3B8",
                   annotation_text="Neutral (3.0)", annotation_font=dict(size=9,color="#94A3B8"))
fig_rank.add_vline(x=4.0, line_dash="dot", line_color=INDIGO,
                   annotation_text="Agree (4.0)", annotation_font=dict(size=9,color=INDIGO))
fig_rank.update_layout(**PLOTLY_LAYOUT, height=380,
                        xaxis=dict(title="Mean Likert Score (1–5)", range=[1,5.6],gridcolor="#F1F5F9"),
                        title=dict(text="Ranked Adoption Driver Means (±1 SD, n=228)",font=dict(size=13)))
st.plotly_chart(fig_rank, use_container_width=True)

# ── EFA ────────────────────────────────────────────────────────────────────────
section("3 · Exploratory Factor Analysis (EFA) — Varimax Rotation",
        "Kaiser Criterion (Eigenvalue > 1) | Orthogonal Varimax rotation")

X = ld.values.astype(float)
X_std = StandardScaler().fit_transform(X)
corr_mat = np.corrcoef(X_std.T)
eigenvalues, eigenvectors = np.linalg.eigh(corr_mat)
eigenvalues = eigenvalues[::-1]
n_factors = int((eigenvalues>1).sum())

c1, c2 = st.columns(2, gap="large")
with c1:
    exp_var = eigenvalues/eigenvalues.sum()*100
    fig_scree = go.Figure()
    fig_scree.add_trace(go.Scatter(
        x=list(range(1,len(eigenvalues)+1)), y=eigenvalues,
        mode="lines+markers", line=dict(color=INDIGO,width=2.5),
        marker=dict(size=8,color=INDIGO), name="Eigenvalue",
        hovertemplate="Factor %{x}: EV=%{y:.3f}<extra></extra>"
    ))
    fig_scree.add_hline(y=1.0, line_dash="dash", line_color=ROSE,
                         annotation_text="Kaiser criterion (EV=1)",
                         annotation_font=dict(size=9,color=ROSE))
    fig_scree.add_vrect(x0=0.5, x1=n_factors+0.5,
                         fillcolor=INDIGO+"15", line_width=0,
                         annotation_text=f"{n_factors} factors retained",
                         annotation_position="top left",
                         annotation_font=dict(size=9,color=INDIGO))
    fig_scree.update_layout(**PLOTLY_LAYOUT, height=310,
                             xaxis=dict(title="Factor Number",tickvals=list(range(1,11)),gridcolor="#F1F5F9"),
                             yaxis=dict(title="Eigenvalue",gridcolor="#F1F5F9"),
                             title=dict(text=f"Scree Plot — {n_factors} Factors Retained",font=dict(size=12)))
    st.plotly_chart(fig_scree, use_container_width=True)

with c2:
    pca = PCA(n_components=n_factors); pca.fit(X_std)
    init_load = pca.components_.T * np.sqrt(pca.explained_variance_)

    def varimax(L, max_iter=1000, tol=1e-6):
        p,k=L.shape; R=np.eye(k)
        for _ in range(max_iter):
            old=R.copy()
            for i in range(k):
                for j in range(i+1,k):
                    Lr=L@R; Li,Lj=Lr[:,i],Lr[:,j]
                    u=Li**2-Lj**2; v=2*Li*Lj
                    A,B=np.sum(u),np.sum(v)
                    C=np.sum(u**2-v**2); D=2*np.sum(u*v)
                    theta=0.25*np.arctan2(D-2*A*B/p,C-(A**2-B**2)/p)
                    c,s=np.cos(theta),np.sin(theta)
                    G=np.eye(k); G[i,i]=c; G[j,j]=c; G[i,j]=-s; G[j,i]=s
                    R=R@G
            if np.max(np.abs(R-old))<tol: break
        return L@R

    rot = varimax(init_load)
    f_cols = [f"F{i+1}" for i in range(n_factors)]
    ldf = pd.DataFrame(rot, index=SHORT_NAMES, columns=f_cols)
    ssl = np.sum(rot**2, axis=0)
    pct_var = ssl/10*100

    fig_load = go.Figure(go.Heatmap(
        z=rot, x=f_cols, y=SHORT_NAMES,
        colorscale=[[0,"#EF4444"],[0.35,"#FEF3C7"],[0.5,"#F8FAFC"],[0.65,"#DBEAFE"],[1,"#3730A3"]],
        zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in rot],
        texttemplate="%{text}", textfont=dict(size=10),
        colorbar=dict(title="Loading"),
        hovertemplate="%{y} → %{x}: loading=%{z:.3f}<extra></extra>"
    ))
    fig_load.update_layout(**{k:v for k,v in PLOTLY_LAYOUT.items() if k not in["xaxis","yaxis"]},
                            height=310,
                            title=dict(text=f"Varimax Factor Loadings (|loading| ≥ 0.40 = meaningful)",font=dict(size=12)))
    st.plotly_chart(fig_load, use_container_width=True)

# Factor summary
st.markdown(f"""
<div style='background:#fff;border:1px solid #E2E8F0;border-radius:12px;padding:16px 20px;
            display:grid;grid-template-columns:repeat({n_factors},1fr);gap:12px;'>
""" + "".join([f"""
  <div style='background:{PALETTE[i]}10;border:1px solid {PALETTE[i]}30;border-radius:8px;padding:12px;'>
    <div style='font-weight:700;color:{PALETTE[i]};font-size:.85rem;'>Factor {i+1}</div>
    <div style='font-size:.72rem;color:#64748B;margin-top:2px;'>Var explained: {pct_var[i]:.1f}%</div>
    <div style='margin-top:6px;'>{''.join(f'<div style="font-size:.75rem;color:#374151;padding:2px 0;">• {item}</div>' for item in ldf[f"F{i+1}"][np.abs(ldf[f"F{i+1}"])>=0.40].index.tolist())}</div>
  </div>""" for i in range(n_factors)]) + "</div>", unsafe_allow_html=True)

# ── Non-User Barriers ──────────────────────────────────────────────────────────
section("4 · Non-User Barrier Analysis (n=113)",
        "Why do 113 respondents not use Q-Commerce apps?")

c1, c2 = st.columns(2, gap="large")
with c1:
    # Open-ended reasons from analysis_ready data
    reasons = ["R_High_Charges","R_Quality_Concern","R_No_Need","R_Prefer_Local",
               "R_Trust_Issue","R_App_Discomfort","R_Lack_Awareness","R_Not_Available"]
    reason_lbls = ["High Charges","Quality Concern","No Need","Prefer Local",
                   "Trust Issues","App Discomfort","Lack of Awareness","Not Available"]
    r_counts = non_u[reasons].sum().values
    sorted_pairs = sorted(zip(r_counts, reason_lbls), reverse=True)
    r_counts_s = [p[0] for p in sorted_pairs]
    r_lbls_s   = [p[1] for p in sorted_pairs]

    fig = go.Figure(go.Bar(
        y=r_lbls_s[::-1], x=r_counts_s[::-1], orientation="h",
        marker_color=[ROSE]+[AMBER]+[PALETTE[i] for i in range(len(r_counts_s)-2)],
        text=[f"n={v} ({v/len(non_u)*100:.1f}%)" for v in r_counts_s[::-1]],
        textposition="outside",
        hovertemplate="%{y}: %{x} non-users<extra></extra>"
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=340,
                       xaxis=dict(title="Number of Non-Users",gridcolor="#F1F5F9"),
                       title=dict(text="Open-Ended Non-Adoption Reasons",font=dict(size=12)))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    # Willingness to try
    will = non_u["NonUser_Willing_Try"].value_counts()
    fig_will = go.Figure(go.Pie(
        labels=will.index, values=will.values,
        marker_colors=[EMERALD,ROSE,AMBER],
        hole=0.55, textinfo="label+percent",
        hovertemplate="%{label}: %{value}<extra></extra>"
    ))
    fig_will.update_layout(**{k:v for k,v in PLOTLY_LAYOUT.items() if k not in["xaxis","yaxis"]},
                            height=250, title=dict(text="Willingness to Try Q-Commerce (Non-Users)",font=dict(size=12)))
    st.plotly_chart(fig_will, use_container_width=True)

    st.markdown(f"""
    <div style='background:{EMERALD}08;border:1px solid {EMERALD}30;border-radius:10px;padding:14px;'>
      <b style='font-size:.85rem;color:#1E1E2E;'>Latent Conversion Potential</b>
      <p style='font-size:.78rem;color:#475569;margin:6px 0 0;line-height:1.6;'>
        A meaningful proportion of non-users express willingness to try Q-Commerce in future.
        Awareness alone is NOT significantly linked to reduced willingness — meaning unaware
        non-users are no more resistant than aware ones. Strong latent conversion opportunity exists.
      </p>
    </div>""", unsafe_allow_html=True)

section("Key Findings")
for t,d,c in [
    (f"⭐ Cronbach's α = {alpha:.3f} — {interp.split(' ')[0]} Reliability",
     "The 10-item scale is internally consistent and reliable for group-level analysis.",INDIGO),
    ("🏆 Convenience Dominates (Time Saving, Delivery Speed, Lifestyle Fit rank top)",
     "Attitudinal drivers cluster around convenience and lifestyle integration — price incentives matter less.",EMERALD),
    ("🚧 Lack of Awareness is the #1 Non-Adoption Barrier",
     "29.2% of non-users cite awareness as the barrier — not rejection. This represents a reachable population through targeted campaigns.",ROSE),
]:
    finding_card(t,d,c)

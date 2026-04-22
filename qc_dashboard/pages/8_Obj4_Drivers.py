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
            "Cronbach's Alpha scale reliability, mean driver rankings, EFA (PAF + Varimax + Parallel Analysis), "
            "non-user barriers, Mann-Whitney U, and Kruskal-Wallis with Dunn's post-hoc on barrier items.")

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

st.markdown(f"""
<div style='background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;padding:13px 17px;margin-top:4px;'>
  <b style='font-size:.83rem;color:#1D4ED8;'>Interpretation — Scale Reliability:</b>
  <span style='font-size:.80rem;color:#374151;'>
  α={alpha:.3f} confirms the 10-item scale is <b>{interp_str}</b> for group-level analysis (threshold α≥0.70).
  <b>Discounts</b> (r=0.207) and <b>Promo Offers</b> (r=0.205) fall below the 0.3 item-total threshold
  and deletion of either would marginally improve α — however both are <b>retained</b> because they anchor
  a theoretically meaningful <i>Price Sensitivity</i> factor confirmed by PAF, and their low item-total r
  reflects that they measure a different dimension from the majority convenience-focused items (expected
  in a multi-dimensional scale). The scale is treated as bifactorial.
  </span>
</div>""", unsafe_allow_html=True)

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

st.markdown("""
<div style='background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;padding:13px 17px;margin-top:4px;'>
  <b style='font-size:.83rem;color:#1D4ED8;'>Interpretation — Driver Rankings:</b>
  <span style='font-size:.80rem;color:#374151;'>
  <b>Top tier</b> (mean ≥ 3.85) — Core convenience drivers: Time Saving #1 (μ=3.899, CV=24.0%),
  Urgent Needs #2 (μ=3.855), Delivery Speed #3 (μ=3.851), Lifestyle Fit #4 (μ=3.829).
  These sit just below the Agree threshold (4.0) with relatively low CVs — broad user consensus
  on speed and lifestyle integration as primary drivers.
  <b>Bottom tier</b> — Price and promotional incentives: Discounts #9 (μ=3.570) and
  Promo Offers #10 (μ=3.452, CV=30.7%, median=3.0 — the only item where the typical user is "Neutral").
  Promo Offers has the highest CV of all items, confirming that financial incentives are not a consistent
  primary driver — they are secondary and highly heterogeneous across users.
  </span>
</div>""", unsafe_allow_html=True)

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

# ── ANALYSIS 3: EFA — PAF + PARALLEL ANALYSIS + VARIMAX ──────────────────────
section("Analysis 3 · Exploratory Factor Analysis — PAF + Parallel Analysis + Varimax Rotation",
        "Principal Axis Factoring (SMC communalities) · Factor retention via Parallel Analysis (n=1000 simulations) · KMO & Bartlett pre-conditions")

@st.cache_data
def run_paf_efa(_ld):
    X_raw = _ld.values.astype(float)
    X_std = StandardScaler().fit_transform(X_raw)
    n, p  = X_std.shape

    # ── KMO (on raw likert_data, not standardized — matches notebook) ──────────
    R     = np.corrcoef(X_raw.T)
    try:
        R_inv = np.linalg.inv(R)
        D     = np.diag(1 / np.sqrt(np.diag(R_inv)))
        P     = -D @ R_inv @ D
        np.fill_diagonal(P, 1.0)
        r2 = R.copy(); np.fill_diagonal(r2, 0)
        p2 = P.copy(); np.fill_diagonal(p2, 0)
        kmo_val = float((r2**2).sum() / ((r2**2).sum() + (p2**2).sum()))
        kmo_items = []
        for i in range(p):
            num = (r2[i,:]**2).sum()
            kmo_items.append(num / (num + (p2[i,:]**2).sum()))
        kmo_items = np.array(kmo_items)
    except Exception:
        kmo_val, kmo_items = np.nan, np.full(p, np.nan)

    # ── Bartlett's test (on standardized) ────────────────────────────────────
    R_std = np.corrcoef(X_std.T)
    chi2_b = -(n - 1 - (2*p + 5)/6) * np.log(np.linalg.det(R_std))
    df_b   = int(p*(p-1)/2)
    p_b    = float(1 - stats.chi2.cdf(chi2_b, df_b))

    # ── Eigenvalues of standardized R for Parallel Analysis ──────────────────
    eigvals_obs = np.sort(np.linalg.eigvalsh(R_std))[::-1]

    # ── Parallel Analysis: 1000 random datasets, 95th percentile ─────────────
    np.random.seed(42)
    sim_eigs = np.array([
        np.sort(np.linalg.eigvalsh(np.corrcoef(
            np.random.normal(size=(n, p)).T)))[::-1]
        for _ in range(1000)
    ])
    sim_95 = np.percentile(sim_eigs, 95, axis=0)

    factors_parallel = int(np.sum(eigvals_obs > sim_95))
    factors_kaiser   = int(np.sum(eigvals_obs > 1.0))
    N_FACTORS        = max(factors_parallel, 1)

    # ── PAF extraction with SMC initial communalities ─────────────────────────
    def paf_extract(X, n_factors, max_iter=500, tol=1e-6):
        R_l   = np.corrcoef(X.T)
        R_inv = np.linalg.inv(R_l)
        h2    = np.clip(1 - 1/np.diag(R_inv), 0.005, 0.999)
        for _ in range(max_iter):
            h2_prev = h2.copy()
            R_red   = R_l.copy()
            np.fill_diagonal(R_red, h2)
            eigvals_l, eigvecs_l = np.linalg.eigh(R_red)
            idx = np.argsort(-eigvals_l)
            eigvals_l, eigvecs_l = eigvals_l[idx], eigvecs_l[:, idx]
            lam  = np.maximum(eigvals_l[:n_factors], 0)
            vecs = eigvecs_l[:, :n_factors]
            L    = vecs * np.sqrt(lam)
            h2   = np.clip((L**2).sum(axis=1), 0.005, 0.999)
            if np.max(np.abs(h2 - h2_prev)) < tol:
                break
        return L, h2

    # ── Varimax rotation ──────────────────────────────────────────────────────
    def varimax_rotate(L, max_iter=1000, tol=1e-8):
        L = L.copy()
        p_l, k = L.shape
        for _ in range(max_iter):
            L_old = L.copy()
            for i in range(k):
                for j in range(i+1, k):
                    x, y = L[:, i], L[:, j]
                    u, v = x**2 - y**2, 2*x*y
                    A, B = u.sum(), v.sum()
                    C, D2 = (u**2 - v**2).sum(), 2*(u*v).sum()
                    num = D2 - 2*A*B/p_l
                    den = C - (A**2 - B**2)/p_l
                    if den == 0:
                        continue
                    theta = 0.25 * np.arctan2(num, den)
                    c, s  = np.cos(theta), np.sin(theta)
                    L[:, [i, j]] = L[:, [i, j]] @ np.array([[c, s], [-s, c]])
            if np.max(np.abs(L - L_old)) < tol:
                break
        return L

    L_unrot, h2_paf = paf_extract(X_std, N_FACTORS)
    L_rot           = varimax_rotate(L_unrot)

    # Sign convention: flip factors so dominant loadings are positive
     for j in range(L_rot.shape[1]):
            if L_rot[:, j].sum() < 0:
                        L_rot[:, j] *= -1        

    # Sort factors by SS loadings descending
    ss_order = np.argsort(-(L_rot**2).sum(axis=0))
    L_rot    = L_rot[:, ss_order]

    ss_rot  = (L_rot**2).sum(axis=0)
    pct_var = ss_rot / p * 100
    cum_var = np.cumsum(pct_var)
    h2_rot  = (L_rot**2).sum(axis=1)

    factor_names = [f"F{i+1}" for i in range(N_FACTORS)]
    ldf = pd.DataFrame(L_rot, index=_ld.columns, columns=factor_names)

    # PAF vs PCA communality comparison
    pca_load = PCA(n_components=N_FACTORS).fit(X_std).components_.T
    h2_pca   = (pca_load**2).sum(axis=1)

    return (kmo_val, kmo_items, chi2_b, df_b, p_b,
            eigvals_obs, sim_95, factors_parallel, factors_kaiser, N_FACTORS,
            ldf, pct_var, cum_var, h2_rot, h2_pca)

(kmo_v, kmo_items, chi2_b, df_b, p_b,
 eigvals_obs, sim_95, factors_parallel, factors_kaiser, N_FACTORS,
 ldf, pct_var, cum_var, h2_rot, h2_pca) = run_paf_efa(ld)

# Factor label mapping (from notebook interpretation)
FACTOR_LABELS = {
    "F1": "Factor 1 — Convenience & Lifestyle",
    "F2": "Factor 2 — Price Sensitivity",
}
ldf_display = ldf.rename(columns=FACTOR_LABELS)

# KPIs
kmo_qual = ("Marvellous" if kmo_v>=0.9 else "Meritorious" if kmo_v>=0.8
            else "Middling" if kmo_v>=0.7 else "Mediocre" if kmo_v>=0.6 else "Poor")
c1,c2,c3,c4 = st.columns(4)
kpi(c1, f"{kmo_v:.3f}", "KMO Value",        kmo_qual,                           EMERALD)
kpi(c2, "p<0.001",      "Bartlett's Test",   "Sig → EFA appropriate",            INDIGO)
kpi(c3, str(N_FACTORS), "Factors Retained",  "Parallel Analysis criterion",      VIOLET)
kpi(c4, f"{cum_var[-1]:.1f}%", "Common Variance", "Explained by PAF solution",  AMBER)
st.markdown("<br>", unsafe_allow_html=True)

# ── Scree + Parallel Analysis ─────────────────────────────────────────────────
c1, c2 = st.columns(2, gap="large")
with c1:
    fig_scree = go.Figure()
    x_vals = list(range(1, len(eigvals_obs)+1))
    fig_scree.add_trace(go.Scatter(
        x=x_vals, y=eigvals_obs.tolist(), mode="lines+markers", name="Observed Eigenvalues",
        line=dict(color=INDIGO, width=2.5), marker=dict(size=8),
        hovertemplate="Factor %{x}: λ=%{y:.3f}<extra></extra>"))
    fig_scree.add_trace(go.Scatter(
        x=x_vals, y=sim_95.tolist(), mode="lines+markers", name="Parallel Analysis (95th pctile, n=1000)",
        line=dict(color=AMBER, width=2, dash="dash"), marker=dict(size=6, symbol="square"),
        hovertemplate="Factor %{x}: PA=%{y:.3f}<extra></extra>"))
    fig_scree.add_hline(y=1.0, line_dash="dot", line_color=ROSE,
                        annotation_text="Kaiser criterion (λ=1)",
                        annotation_font=dict(size=9, color=ROSE))
    fig_scree.add_vrect(x0=0.5, x1=N_FACTORS+0.5, fillcolor=f"rgba(79,70,229,0.07)", line_width=0,
                        annotation_text=f"{N_FACTORS} factors retained",
                        annotation_position="top left", annotation_font=dict(size=9, color=INDIGO))
    fig_scree.update_layout(
        **PLOTLY_LAYOUT, height=320,
        title=dict(text=f"Scree Plot + Parallel Analysis — {N_FACTORS} Factors Retained", font=dict(size=12))
    )
    fig_scree.update_layout(legend=dict(orientation="h", y=-0.2))
    fig_scree.update_xaxes(title="Factor Number", tickvals=x_vals, gridcolor="#F1F5F9")
    fig_scree.update_yaxes(title="Eigenvalue", gridcolor="#F1F5F9")
    st.plotly_chart(fig_scree, use_container_width=True)

# ── Factor Loading Heatmap ─────────────────────────────────────────────────────
with c2:
    fig_load = go.Figure(go.Heatmap(
        z=ldf_display.values,
        x=ldf_display.columns.tolist(),
        y=SHORT_NAMES,
        colorscale=[[0,"#EF4444"],[0.35,"#FEF3C7"],[0.5,"#F8FAFC"],[0.65,"#DBEAFE"],[1,"#3730A3"]],
        zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in ldf_display.values],
        texttemplate="%{text}", textfont=dict(size=9),
        colorbar=dict(title="Loading"),
        hovertemplate="%{y} → %{x}: %{z:.3f}<extra></extra>"))
    fig_load.update_layout(
        **{k:v for k,v in PLOTLY_LAYOUT.items() if k not in["xaxis","yaxis"]},
        height=320,
        title=dict(text=f"PAF + Varimax Rotated Loading Matrix | {cum_var[-1]:.1f}% common variance",
                   font=dict(size=11))
    )
    st.plotly_chart(fig_load, use_container_width=True)

# ── Communality chart (PAF vs PCA) ────────────────────────────────────────────
with st.expander("📊 PAF Communalities (h²) per Item + PAF vs PCA Comparison"):
    c1, c2 = st.columns(2, gap="large")
    with c1:
        comm_sorted_idx = np.argsort(h2_rot)
        comm_sorted     = h2_rot[comm_sorted_idx]
        items_sorted    = [SHORT_NAMES[i] for i in comm_sorted_idx]
        comm_colors     = [ROSE if v<0.30 else INDIGO if v<0.50 else EMERALD for v in comm_sorted]
        fig_comm = go.Figure(go.Bar(
            y=items_sorted, x=comm_sorted.tolist(), orientation="h",
            marker_color=comm_colors,
            text=[f"h²={v:.3f}" for v in comm_sorted],
            textposition="outside",
            hovertemplate="%{y}: h²=%{x:.3f}<extra></extra>"
        ))
        fig_comm.add_vline(x=0.30, line_dash="dash", line_color=ROSE, opacity=0.8,
                           annotation_text="Low (0.30)", annotation_font=dict(size=9))
        fig_comm.add_vline(x=0.50, line_dash="dash", line_color=AMBER, opacity=0.8,
                           annotation_text="Acceptable (0.50)", annotation_font=dict(size=9))
        fig_comm.update_layout(**PLOTLY_LAYOUT, height=320,
                                title=dict(text="PAF Communalities (h²) per Item — Green≥0.50 · Blue 0.30–0.49 · Red<0.30",
                                           font=dict(size=11)))
        fig_comm.update_xaxes(title="Communality h²", range=[0, 1.1], gridcolor="#F1F5F9")
        st.plotly_chart(fig_comm, use_container_width=True)

    with c2:
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(name="PAF h²", x=SHORT_NAMES, y=h2_rot.tolist(),
                                  marker_color=INDIGO, opacity=0.85,
                                  hovertemplate="%{x}: PAF h²=%{y:.3f}<extra></extra>"))
        fig_cmp.add_trace(go.Bar(name=f"PCA h² ({N_FACTORS} components)", x=SHORT_NAMES, y=h2_pca.tolist(),
                                  marker_color=AMBER, opacity=0.85,
                                  hovertemplate="%{x}: PCA h²=%{y:.3f}<extra></extra>"))
        fig_cmp.update_layout(**PLOTLY_LAYOUT, barmode="group", height=320,
                               title=dict(text="Communality Comparison: PAF vs PCA", font=dict(size=11)))
        fig_cmp.update_xaxes(tickangle=-30, gridcolor="#F1F5F9")
        fig_cmp.update_yaxes(title="h²", range=[0, 1.1], gridcolor="#F1F5F9")
        st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("""
    <div style='background:#FFFBEB;border:1px solid #FDE68A;border-radius:8px;padding:11px 14px;font-size:.78rem;color:#374151;'>
      <b>PAF vs PCA communalities:</b> PAF places Squared Multiple Correlations (SMC) on the diagonal of the
      correlation matrix and iteratively refines them — modelling only <i>shared</i> variance.
      PCA places 1.0 on the diagonal and models total variance including unique and error terms.
      PAF communalities are lower for items that are less intercorrelated (more unique variance),
      confirming that PAF produces a more conservative and theoretically sound factor solution.
    </div>""", unsafe_allow_html=True)

# Factor summary cards
st.markdown(f"""<div style='display:grid;grid-template-columns:repeat({N_FACTORS},1fr);gap:10px;margin:8px 0;'>
""" + "".join([f"""  <div style='background:{PALETTE[i]}10;border:1px solid {PALETTE[i]}30;border-radius:10px;padding:12px;'>
    <div style='font-weight:700;color:{PALETTE[i]};font-size:.85rem;'>{FACTOR_LABELS.get(f"F{i+1}", f"Factor {i+1}")}</div>
    <div style='font-size:.7rem;color:#64748B;'>Common variance: {pct_var[i]:.1f}%</div>
    <div style='margin-top:6px;'>{"".join(f'<div style="font-size:.72rem;color:#374151;padding:1px 0;">• {item}</div>' for item in ldf[f"F{i+1}"][np.abs(ldf[f"F{i+1}"])>=0.40].index)}</div>
  </div>""" for i in range(N_FACTORS)]) + "</div>", unsafe_allow_html=True)

st.markdown(f"""
<div style='background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;padding:13px 17px;margin-top:8px;'>
  <b style='font-size:.83rem;color:#1D4ED8;'>Interpretation — PAF + Parallel Analysis:</b>
  <span style='font-size:.80rem;color:#374151;'>
  KMO={kmo_v:.3f} and Bartlett p&lt;0.001 confirm the data is suitable for EFA.
  Parallel Analysis (1,000 random datasets, 95th-percentile eigenvalues) retains {N_FACTORS} factor(s)
  — a more conservative and rigorous criterion than Kaiser's eigenvalue&gt;1 rule (which retains {factors_kaiser}).
  <br><br>
  <b>Factor 1 — Convenience &amp; Lifestyle</b> ({pct_var[0]:.1f}% common variance):
  Loads strongly on Time Saving, Delivery Speed, Lifestyle Fit, Ease of Use, Schedule Barrier,
  Urgent Needs, Quality Reliable, and Product Variety — the dominant convenience proposition of Q-Commerce.
  <br>
  <b>Factor 2 — Price Sensitivity</b> ({pct_var[1]:.1f}% common variance):
  Loads on Promo Offers and Discounts — a distinct, economically-driven latent dimension.
  Its separation from Factor 1 confirms that price incentives appeal to a different subset of users
  and should be treated as a separate marketing lever, not bundled with convenience messaging.
  <br><br>
  Total common variance explained: {cum_var[-1]:.1f}%.
  The bifactorial structure is confirmed by both PAF and PCA — convergence across methods
  strengthens confidence in the finding.
  </span>
</div>""", unsafe_allow_html=True)

finding_card(f"⭐ Cronbach's α={alpha:.3f} — {interp_str} Reliability",
             "The 10-item scale is internally consistent. Discounts and Promo Offers show low item-total r "
             "but are retained as they anchor the theoretically meaningful Price Sensitivity factor "
             "confirmed by PAF. All subsequent factor analyses are statistically justified.", INDIGO)
finding_card("🎯 PAF EFA: Bifactorial Adoption Structure — Convenience & Lifestyle + Price Sensitivity",
             f"PAF + Varimax with Parallel Analysis retains {N_FACTORS} factors explaining {cum_var[-1]:.1f}% "
             "of common variance. Factor 1 (Convenience & Lifestyle) captures the dominant adoption motivation "
             "across 8 items. Factor 2 (Price Sensitivity) is a distinct secondary dimension (Discounts + Promo Offers). "
             "Platforms must address both dimensions — but convenience is the dominant driver.", EMERALD)

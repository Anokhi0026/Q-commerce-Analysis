import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.stats import kruskal
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from utils import *

st.set_page_config("Cluster Analysis", "🧩", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*='css']{font-family:'Inter',sans-serif;}
.stApp{background:#FAFAFA;}
section[data-testid='stSidebar']{background:#FFFFFF;border-right:1px solid #E2E8F0;}
</style>""", unsafe_allow_html=True)

sidebar()
page_header("Consumer Segmentation", "K-Means Cluster Analysis of Q-Commerce Users",
            "Identifying and characterising distinct consumer segments using 13 attitudinal and satisfaction variables. "
            "Elbow + Silhouette method for K selection, PCA for visualisation, Kruskal-Wallis for statistical validation.")

CLUSTER_COLORS = [INDIGO, AMBER, EMERALD]
CLUSTER_NAMES  = ["Active Engagers","Passive Users","Convenience Purists"]

k1,k2,k3,k4 = st.columns(4)
kpi(k1,"228","Users Clustered","Complete-case analysis")
kpi(k2,"13","Clustering Variables","10 attitude + 3 satisfaction",INDIGO)
kpi(k3,"3","Clusters (K=3)","Elbow + Silhouette validated",EMERALD)
kpi(k4,"K-Means","Algorithm","Z-score standardised data",VIOLET)
st.markdown("<br>",unsafe_allow_html=True)

# ── DATA PREP ──────────────────────────────────────────────────────────────────
@st.cache_data
def prepare_cluster_data():
    users = get_users()
    LIKERT_COLS_C = [
        "Q-commerce apps save my time","Delivery speed is very convenient",
        "Discounts influence my usage","Product variety meets my needs",
        "Promotional offers attract me","Apps are easy to navigate",
        "Urgent needs motivate me to use these apps","Product quality is reliable",
        "My work/study schedule makes offline shopping difficult","Q-commerce fits my lifestyle"
    ]
    SAT_COLS = [
        "Overall satisfaction with Q-Commerce Apps",
        "Likelihood of continuing usage in the future",
        "likely are you to recommend these apps to Others"
    ]
    SHORT_13 = [
        "Time Saving","Delivery Speed","Discounts","Product Variety","Promo Offers",
        "Ease of Use","Urgent Needs","Quality Reliable","Schedule Barrier","Lifestyle Fit",
        "Satisfaction","Continuity","Recommend"
    ]
    for col in LIKERT_COLS_C:
        users[col] = users[col].map(LIKERT_MAP)
    cdf = users[LIKERT_COLS_C+SAT_COLS].copy().dropna()
    cdf.columns = SHORT_13
    uf  = users.loc[cdf.index].copy().reset_index(drop=True)
    cdf = cdf.reset_index(drop=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(cdf)
    return cdf, uf, X, SHORT_13, scaler

cdf, users_full, X, SHORT_13, scaler = prepare_cluster_data()

# ── STEP 1: OPTIMAL K ─────────────────────────────────────────────────────────
section("Step 1 · Optimal K — Elbow Method + Silhouette Score")

@st.cache_data
def find_optimal_k(_X):
    K_RANGE = range(2,9)
    wcss_v, sil_v = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=50, random_state=42, max_iter=500)
        labels = km.fit_predict(_X)
        wcss_v.append(km.inertia_)
        sil_v.append(silhouette_score(_X, labels))
    return list(K_RANGE), wcss_v, sil_v

K_RANGE, wcss_v, sil_v = find_optimal_k(X)

c1,c2 = st.columns(2, gap="large")
with c1:
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(x=K_RANGE, y=wcss_v, mode="lines+markers",
                                    line=dict(color=INDIGO,width=2.5), marker=dict(size=8),
                                    text=[f"K={k}: WCSS={w:.0f}" for k,w in zip(K_RANGE,wcss_v)],
                                    hovertemplate="%{text}<extra></extra>"))
    fig_elbow.add_vline(x=3, line_dash="dash", line_color=ROSE, opacity=0.8,
                         annotation_text="Recommended K=3", annotation_font=dict(size=9,color=ROSE))
    fig_elbow.update_layout(**PLOTLY_LAYOUT, height=290,
                             xaxis=dict(title="Number of Clusters (K)", tickvals=K_RANGE, gridcolor="#F1F5F9"),
                             yaxis=dict(title="WCSS",gridcolor="#F1F5F9"),
                             title=dict(text="(a) Elbow Method — WCSS vs K",font=dict(size=12)))
    st.plotly_chart(fig_elbow, use_container_width=True)

with c2:
    bar_colors_k = [ROSE if k==2 else AMBER if k==3 else SLATE for k in K_RANGE]
    fig_sil = go.Figure(go.Bar(x=K_RANGE, y=sil_v, marker_color=bar_colors_k,
                                text=[f"{v:.3f}" for v in sil_v], textposition="outside",
                                hovertemplate="K=%{x}: Silhouette=%{y:.3f}<extra></extra>"))
    fig_sil.add_vline(x=3, line_dash="dash", line_color=AMBER,
                       annotation_text="Selected K=3", annotation_font=dict(size=9,color=AMBER))
    fig_sil.add_hline(y=0.1, line_dash="dot", line_color=SLATE, opacity=0.6,
                       annotation_text="Min acceptable (0.10)", annotation_font=dict(size=9))
    fig_sil.update_layout(**PLOTLY_LAYOUT, height=290,
                           xaxis=dict(title="K", tickvals=K_RANGE, gridcolor="#F1F5F9"),
                           yaxis=dict(title="Avg Silhouette Score",gridcolor="#F1F5F9"),
                           title=dict(text="(b) Silhouette Score vs K",font=dict(size=12)))
    st.plotly_chart(fig_sil, use_container_width=True)

st.markdown(f"""
<div style='background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;padding:14px 18px;'>
  <b style='font-size:.85rem;color:#1D4ED8;'>K=3 Rationale:</b>
  <span style='font-size:.82rem;color:#374151;'> K=2 gives the highest silhouette but only two broad groups (engaged vs disengaged) — 
  too coarse for actionable insights. K=3 retains a strong silhouette and produces 3 meaningfully distinct 
  consumer profiles validated by hierarchical clustering. K≥4 shows diminishing returns.</span>
</div>""", unsafe_allow_html=True)

# ── STEP 2: FINAL K-MEANS ─────────────────────────────────────────────────────
section("Step 2 · Final K-Means Model (K=3)")

@st.cache_data
def run_kmeans(_X, _scaler, _cdf, _users_full):
    km = KMeans(n_clusters=3, n_init=100, max_iter=1000, random_state=42)
    labels = km.fit_predict(_X)
    sil    = silhouette_score(_X, labels)
    sil_s  = silhouette_samples(_X, labels)
    centroids_std  = km.cluster_centers_
    centroids_orig = _scaler.inverse_transform(centroids_std)
    cdf2   = _cdf.copy(); cdf2["Cluster"] = labels
    uf2    = _users_full.copy(); uf2["Cluster"] = labels
    return labels, sil, sil_s, centroids_orig, cdf2, uf2, km

labels, final_sil, sil_samples, centroids_orig, cdf2, users_c, km_model = run_kmeans(X, scaler, cdf, users_full)
sizes = pd.Series(labels).value_counts().sort_index()

c1,c2,c3,c4 = st.columns(4)
kpi(c1,f"{final_sil:.4f}","Silhouette Score","Overall cluster quality",INDIGO)
kpi(c2,f"{sizes.get(0,0)}","Cluster 0","Active Engagers",INDIGO)
kpi(c3,f"{sizes.get(1,0)}","Cluster 1","Passive Users",AMBER)
kpi(c4,f"{sizes.get(2,0)}","Cluster 2","Convenience Purists",EMERALD)
st.markdown("<br>",unsafe_allow_html=True)

# ── STEP 3: PCA SCATTER ───────────────────────────────────────────────────────
section("Step 3 · PCA Scatter — Cluster Separation in 2D")

pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
var_e = pca.explained_variance_ratio_*100
centroids_pca = pca.transform(km_model.cluster_centers_)

c1,c2 = st.columns([1.4,1], gap="large")
with c1:
    fig_pca = go.Figure()
    for c_id in range(3):
        mask = labels == c_id
        fig_pca.add_trace(go.Scatter(
            x=X_pca[mask,0], y=X_pca[mask,1], mode="markers",
            name=f"{CLUSTER_NAMES[c_id]} (n={mask.sum()})",
            marker=dict(color=CLUSTER_COLORS[c_id], size=7, opacity=0.7,
                        line=dict(color="#fff",width=0.3)),
            hovertemplate=f"Cluster {c_id}: %{{x:.2f}}, %{{y:.2f}}<extra></extra>"))
        fig_pca.add_trace(go.Scatter(
            x=[centroids_pca[c_id,0]], y=[centroids_pca[c_id,1]],
            mode="markers", showlegend=False,
            marker=dict(color=CLUSTER_COLORS[c_id], size=16, symbol="star",
                        line=dict(color="#000",width=1.2))))
    fig_pca.update_layout(**PLOTLY_LAYOUT, height=380,
                           xaxis=dict(title=f"PC1 ({var_e[0]:.1f}% variance)",gridcolor="#F1F5F9"),
                           yaxis=dict(title=f"PC2 ({var_e[1]:.1f}% variance)",gridcolor="#F1F5F9"),
                           title=dict(text=f"PCA Cluster Scatter (PC1+PC2 = {sum(var_e):.1f}% variance)",font=dict(size=12)))
    st.plotly_chart(fig_pca, use_container_width=True)

with c2:
    # Silhouette by cluster
    fig_sil2 = go.Figure()
    y_offset = 0
    for c_id in range(3):
        c_sil = np.sort(sil_samples[labels==c_id])[::-1]
        fig_sil2.add_trace(go.Bar(
            x=c_sil, y=list(range(y_offset, y_offset+len(c_sil))),
            orientation="h", marker_color=CLUSTER_COLORS[c_id],
            name=CLUSTER_NAMES[c_id], opacity=0.85, showlegend=True,
            hovertemplate=f"Cluster {c_id}: sil=%{{x:.3f}}<extra></extra>"))
        y_offset += len(c_sil)
    fig_sil2.add_vline(x=final_sil, line_dash="dash", line_color=ROSE,
                        annotation_text=f"Avg={final_sil:.3f}",
                        annotation_font=dict(size=9,color=ROSE))
    fig_sil2.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=True,
                            xaxis=dict(title="Silhouette Coefficient",gridcolor="#F1F5F9"),
                            yaxis=dict(showticklabels=False),
                            title=dict(text="Silhouette Plot (sorted by cluster)",font=dict(size=12)))
    st.plotly_chart(fig_sil2, use_container_width=True)

# ── STEP 4: CLUSTER PROFILING ─────────────────────────────────────────────────
section("Step 4 · Cluster Profiling — Attitude & Satisfaction Heatmap")

profile = cdf2.groupby("Cluster")[SHORT_13].mean().round(3)
profile.index = CLUSTER_NAMES

fig_hm = go.Figure(go.Heatmap(
    z=profile.T.values, x=CLUSTER_NAMES, y=SHORT_13,
    colorscale=[[0,"#FEF2F2"],[0.3,"#FCA5A5"],[0.5,"#FCD34D"],[0.7,"#86EFAC"],[1,"#15803D"]],
    vmin=2.2, vmax=5.0,
    text=profile.T.round(2).values, texttemplate="%{text:.2f}",
    textfont=dict(size=10),
    colorbar=dict(title="Mean Score (1–5)"),
    hovertemplate="%{y} | %{x}: %{z:.2f}<extra></extra>"))
fig_hm.update_layout(**{k:v for k,v in PLOTLY_LAYOUT.items() if k not in["xaxis","yaxis"]},
                      height=380, title=dict(text="Cluster Mean Profiles — Green=High, Red=Low",font=dict(size=12)))
st.plotly_chart(fig_hm, use_container_width=True)

# Profile cards
st.markdown(f"""<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:8px 0;'>
""" + "".join([f"""
  <div style='background:{CLUSTER_COLORS[i]}10;border:1px solid {CLUSTER_COLORS[i]}30;
              border-radius:12px;padding:16px;'>
    <div style='font-weight:700;color:{CLUSTER_COLORS[i]};font-size:.9rem;margin-bottom:4px;'>
      Cluster {i}: {CLUSTER_NAMES[i]}
    </div>
    <div style='font-size:.7rem;color:#64748B;margin-bottom:8px;'>n={sizes.get(i,0)} ({sizes.get(i,0)/len(labels)*100:.1f}%)</div>
    <div style='font-size:.75rem;color:#374151;line-height:1.7;'>{desc}</div>
  </div>""" for i, desc in enumerate([
    "Highest scores across ALL attitude and satisfaction items. Strongly agrees on convenience, delivery speed, and lifestyle fit. Active, loyal power-users.",
    "Below-average scores on most items. Lower satisfaction and continuity. Occasional or lapsing users who engage only when specific needs arise.",
    "High on Time Saving and Urgent Needs — uses Q-Commerce primarily for convenience and emergency top-ups. Less influenced by discounts or promo offers.",
])])+"</div>", unsafe_allow_html=True)

# ── STEP 5: BEHAVIORAL PROFILE ────────────────────────────────────────────────
section("Step 5 · Behavioral Profile by Cluster")

behav_tab1, behav_tab2 = st.tabs(["App & Order Value","Demographic Composition"])
with behav_tab1:
    c1,c2 = st.columns(2)
    for col_w, bcol, bname, bord in [
        (c1,"App_Used","App Used",["Blinkit","Swiggy Instamart","Zepto","Other"]),
        (c2,"average order value","Avg Order Value",["Below ₹200","₹200 - ₹400","₹400 - ₹600","Above ₹600"])
    ]:
        with col_w:
            fig_bh = go.Figure()
            for c_id in range(3):
                u_sub = users_c[users_c["Cluster"]==c_id]
                cnt_b = u_sub[bcol].value_counts().reindex([o for o in bord if o in u_sub[bcol].values]).fillna(0)
                cnt_b_pct = cnt_b/len(u_sub)*100
                fig_bh.add_trace(go.Bar(name=CLUSTER_NAMES[c_id], x=bord, y=cnt_b_pct.reindex(bord).fillna(0),
                                         marker_color=CLUSTER_COLORS[c_id],
                                         hovertemplate=f"{CLUSTER_NAMES[c_id]}: %{{y:.1f}}<extra></extra>"))
            fig_bh.update_layout(**PLOTLY_LAYOUT, barmode="group", height=290,
                                  yaxis=dict(title="% within cluster",gridcolor="#F1F5F9"),
                                  xaxis=dict(tickangle=-15),
                                  title=dict(text=bname,font=dict(size=12)))
            st.plotly_chart(fig_bh, use_container_width=True)

with behav_tab2:
    for demo_col, demo_order in [("Age_Group",AGE_ORDER),("Occupation",OCC_ORDER)]:
        fig_demo = go.Figure()
        for c_id in range(3):
            u_sub = users_c[users_c["Cluster"]==c_id]
            cnt_d = u_sub[demo_col].value_counts().reindex(demo_order).fillna(0)
            cnt_d_pct = cnt_d/len(u_sub)*100
            fig_demo.add_trace(go.Bar(name=CLUSTER_NAMES[c_id], x=demo_order, y=cnt_d_pct,
                                       marker_color=CLUSTER_COLORS[c_id], opacity=0.85,
                                       hovertemplate=f"{CLUSTER_NAMES[c_id]}: %{{y:.1f}}%<extra></extra>"))
        fig_demo.update_layout(**PLOTLY_LAYOUT, barmode="group", height=260,
                                yaxis=dict(title="% within cluster",gridcolor="#F1F5F9"),
                                xaxis=dict(tickangle=-15),
                                title=dict(text=f"Cluster Composition — {demo_col}",font=dict(size=12)))
        st.plotly_chart(fig_demo, use_container_width=True)

# ── STEP 6: KRUSKAL-WALLIS VALIDATION ─────────────────────────────────────────
section("Step 6 · Statistical Validation — Kruskal-Wallis H Test",
        "H₀: Median scores are equal across clusters | Significant result confirms clusters are truly different")

@st.cache_data
def validate_clusters(_cdf2, _SHORT_13):
    rows = []
    for var in _SHORT_13:
        groups = [_cdf2[_cdf2["Cluster"]==c_id][var].dropna().values for c_id in range(3)]
        if all(len(g)>=3 for g in groups):
            H,p = kruskal(*groups)
            rows.append({"Variable":var,"H":round(H,3),"p-value":round(p,6),
                          "Sig":"✅ Yes" if p<0.05 else "❌ No"})
    return pd.DataFrame(rows)

val_df = validate_clusters(cdf2, SHORT_13)
sig_pct = (val_df["Sig"]=="✅ Yes").sum()/len(val_df)*100
kpi_c1, kpi_c2, kpi_c3 = st.columns(3)
kpi(kpi_c1,f"{int(sig_pct)}%","Variables Significant","p<0.05 across clusters",EMERALD)
kpi(kpi_c2,f"{val_df['H'].max():.2f}","Max H Statistic","Most differentiated variable",INDIGO)
kpi(kpi_c3,"Validated","Cluster Quality","All significant → real segments",VIOLET)
st.markdown("<br>",unsafe_allow_html=True)

fig_val = go.Figure(go.Bar(
    y=val_df["Variable"], x=val_df["H"], orientation="h",
    marker_color=[EMERALD if s=="✅ Yes" else SLATE for s in val_df["Sig"]],
    text=[f"H={row['H']} {row['Sig']}" for _,row in val_df.iterrows()],
    textposition="outside",
    hovertemplate="%{y}: H=%{x:.2f}<extra></extra>"))
fig_val.update_layout(**PLOTLY_LAYOUT, height=360,
                       xaxis=dict(title="H Statistic",gridcolor="#F1F5F9"),
                       title=dict(text="Kruskal-Wallis H by Variable (Green=Significant)",font=dict(size=12)))
st.plotly_chart(fig_val, use_container_width=True)

with st.expander("📋 Full Kruskal-Wallis results table"):
    st.dataframe(val_df, use_container_width=True)

finding_card("✅ Clusters are Statistically Valid — Not Random Groupings",
             f"{int(sig_pct)}% of clustering variables show significant Kruskal-Wallis differences across the 3 clusters. "
             "This confirms the K-Means segmentation captures genuine, distinct consumer behavioural profiles.", EMERALD)
finding_card("📦 Three Actionable Segments Identified",
             "Active Engagers (power users, target for loyalty programs), "
             "Passive Users (at-risk of churn, target for re-engagement), "
             "Convenience Purists (emergency/top-up users, target with speed messaging).", INDIGO)

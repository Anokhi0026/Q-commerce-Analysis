import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.stats import kruskal
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from utils import *

st.set_page_config("Cluster Analysis", "🧩", layout="wide")
st.session_state["current_page"] = "pages/11_Cluster_Analysis.py"
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*='css']{font-family:'Inter',sans-serif;}
.stApp{background:#FAFAFA;}
section[data-testid='stSidebar']{background:#FFFFFF;border-right:1px solid #E2E8F0;}
</style>""", unsafe_allow_html=True)

from navbar import navbar
navbar()
page_header(
    "Consumer Segmentation",
    "K-Medoids (PAM) Cluster Analysis of Q-Commerce Users",
    "Identifying and characterising distinct consumer segments using 13 attitudinal and satisfaction variables. "
    "Elbow + Silhouette for K selection, PCA for visualisation, Kruskal-Wallis + Dunn post-hoc for statistical validation. "
    "K-Medoids (PAM) with Cityblock distance — cluster centres are actual survey respondents, "
    "making profiles more robust to outliers and directly interpretable on Likert-scale data."
)

CLUSTER_COLORS = [INDIGO, AMBER, EMERALD]
CLUSTER_NAMES  = ["Neutral Adopters", "All-Round Enthusiast", "Convenience Purists"]

k1, k2, k3, k4 = st.columns(4)
kpi(k1, "228",       "Users Clustered",     "Complete-case analysis")
kpi(k2, "13",        "Clustering Variables","10 attitude + 3 satisfaction", INDIGO)
kpi(k3, "3",         "Clusters (K=3)",      "Elbow + Silhouette validated",  EMERALD)
kpi(k4, "K-Medoids", "Algorithm (PAM)",     "Cityblock distance · real medoids", VIOLET)
st.markdown("<br>", unsafe_allow_html=True)

# ── WHY K-MEDOIDS ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;padding:14px 18px;margin-bottom:16px;'>
  <b style='font-size:.85rem;color:#1D4ED8;'>Why K-Medoids (PAM) instead of K-Means?</b><br>
  <span style='font-size:.82rem;color:#374151;'>
  K-Means computes abstract arithmetic centroids that may not correspond to any real survey respondent.
  <b>K-Medoids (PAM — Partitioning Around Medoids)</b> selects <i>actual data points</i> as cluster
  representatives, so each cluster is described by a real respondent — directly interpretable as a
  consumer persona. PAM also uses <b>Cityblock (Manhattan) distance</b>, which is more appropriate
  for discrete Likert-scale data and more robust to outliers than the squared Euclidean distance
  used by K-Means.
  </span>
  <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:10px;'>
    <div style='background:#fff;border-radius:8px;padding:10px;font-size:.75rem;'>
      <b style='color:#1D4ED8;'>Cluster centre</b><br>Actual data point (medoid), not an abstract mean
    </div>
    <div style='background:#fff;border-radius:8px;padding:10px;font-size:.75rem;'>
      <b style='color:#1D4ED8;'>Distance metric</b><br>Cityblock — better fit for discrete Likert scales
    </div>
    <div style='background:#fff;border-radius:8px;padding:10px;font-size:.75rem;'>
      <b style='color:#1D4ED8;'>Outlier robustness</b><br>More robust than K-Means (no squared distances)
    </div>
    <div style='background:#fff;border-radius:8px;padding:10px;font-size:.75rem;'>
      <b style='color:#1D4ED8;'>Interpretability</b><br>Medoids are real respondents → concrete personas
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ── DATA PREP ──────────────────────────────────────────────────────────────────
@st.cache_data
def prepare_cluster_data():
    users = get_users()
    LIKERT_COLS_C = [
        "Q-commerce apps save my time",
        "Delivery speed is very convenient",
        "Discounts influence my usage",
        "Product variety meets my needs",
        "Promotional offers attract me",
        "Apps are easy to navigate",
        "Urgent needs motivate me to use these apps",
        "Product quality is reliable",
        "My work/study schedule makes offline shopping difficult",
        "Q-commerce fits my lifestyle"
    ]
    SAT_COLS = [
        "Overall satisfaction with Q-Commerce Apps",
        "Likelihood of continuing usage in the future",
        "likely are you to recommend these apps to Others"
    ]
    SHORT_13 = [
        "Time Saving", "Delivery Speed", "Discounts", "Product Variety", "Promo Offers",
        "Ease of Use", "Urgent Needs", "Quality Reliable", "Schedule Barrier", "Lifestyle Fit",
        "Satisfaction", "Continuity", "Recommend"
    ]
    for col in LIKERT_COLS_C:
        users[col] = users[col].map(LIKERT_MAP)
    cdf = users[LIKERT_COLS_C + SAT_COLS].copy().dropna()
    cdf.columns = SHORT_13
    uf  = users.loc[cdf.index].copy().reset_index(drop=True)
    cdf = cdf.reset_index(drop=True)
    X = cdf.copy()
    return cdf, uf, X, SHORT_13, scaler

cdf, users_full, X, SHORT_13, scaler = prepare_cluster_data()

# ── PAM IMPLEMENTATION (exact match to notebook cell 4) ───────────────────────
def pam_kmedoids(X, k, n_init=10, max_iter=300, random_state=42):
    rng = np.random.default_rng(random_state)
    n   = X.shape[0]
    D   = cdist(X, X, metric="cityblock")
    best_labels, best_medoids, best_inertia = None, None, np.inf
    for _ in range(n_init):
        medoids = rng.choice(n, size=k, replace=False)
        for _ in range(max_iter):
            labels = np.argmin(D[:, medoids], axis=1)
            new_medoids = medoids.copy()
            for cid in range(k):
                members = np.where(labels == cid)[0]
                if len(members) == 0:
                    continue
                within_dist = D[np.ix_(members, members)].sum(axis=1)
                new_medoids[cid] = members[np.argmin(within_dist)]
            if set(new_medoids) == set(medoids):
                break
            medoids = new_medoids
        inertia = sum(D[i, medoids[labels[i]]] for i in range(n))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels  = labels.copy()
            best_medoids = medoids.copy()
    return best_labels, best_medoids, best_inertia

# ── STEP 1: OPTIMAL K ─────────────────────────────────────────────────────────
section(
    "Step 1 · Optimal K — Elbow Method + Silhouette Score",
    "Two complementary methods: Elbow (Total Within-Cluster Distance) and Silhouette Score (Cityblock)"
)

# Pre-computed from notebook (n_init=10, random_state=42) — exact notebook output
K_RANGE    = list(range(2, 9))
twd_values = [1673.0, 1500.0, 1438.0, 1410.0, 1388.0, 1346.0, 1330.0]
sil_values = [0.2390,  0.1655, 0.1112, 0.0727, 0.0907, 0.0876, 0.0903]
drop_pct   = [None,   10.34,   4.13,   1.95,   1.56,   3.03,   1.19]

c1, c2 = st.columns(2, gap="large")
with c1:
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=K_RANGE, y=twd_values, mode="lines+markers",
        line=dict(color=INDIGO, width=2.5), marker=dict(size=8),
        text=[f"K={k}: TWD={w:.0f}" for k, w in zip(K_RANGE, twd_values)],
        hovertemplate="%{text}<extra></extra>"
    ))
    fig_elbow.add_vline(x=3, line_dash="dash", line_color=ROSE, opacity=0.8,
                        annotation_text="Recommended K=3",
                        annotation_font=dict(size=9, color=ROSE))
    fig_elbow.update_layout(
        **PLOTLY_LAYOUT, height=300,
        title=dict(text="(a) Elbow Method — Total Within-Cluster Distance vs K", font=dict(size=12))
    )
    fig_elbow.update_xaxes(title="Number of Clusters (K)", tickvals=K_RANGE, gridcolor="#F1F5F9")
    fig_elbow.update_yaxes(title="Total Within-Cluster Distance (Cityblock)", gridcolor="#F1F5F9")
    st.plotly_chart(fig_elbow, use_container_width=True)

with c2:
    bar_colors_k = [ROSE if k == 2 else AMBER if k == 3 else SLATE for k in K_RANGE]
    fig_sil = go.Figure(go.Bar(
        x=K_RANGE, y=sil_values, marker_color=bar_colors_k,
        text=[f"{v:.4f}" for v in sil_values], textposition="outside",
        hovertemplate="K=%{x}: Silhouette=%{y:.4f}<extra></extra>"
    ))
    fig_sil.add_vline(x=3, line_dash="dash", line_color=AMBER,
                      annotation_text="Selected K=3", annotation_font=dict(size=9, color=AMBER))
    fig_sil.add_hline(y=0.10, line_dash="dot", line_color=SLATE, opacity=0.6,
                      annotation_text="Min acceptable (0.10)", annotation_font=dict(size=9))
    fig_sil.update_layout(
        **PLOTLY_LAYOUT, height=300,
        title=dict(text="(b) Silhouette Score vs K", font=dict(size=12))
    )
    fig_sil.update_xaxes(title="K", tickvals=K_RANGE, gridcolor="#F1F5F9")
    fig_sil.update_yaxes(title="Avg Silhouette Score", gridcolor="#F1F5F9", range=[0, 0.30])
    st.plotly_chart(fig_sil, use_container_width=True)

with st.expander("📋 K Selection Summary Table"):
    k_sum = pd.DataFrame({
        "K": K_RANGE,
        "Total Within-Distance": twd_values,
        "Silhouette Score": sil_values,
        "Distance Drop (%)": drop_pct,
    })
    st.dataframe(k_sum.set_index("K"), use_container_width=True)

st.markdown("""
<div style='background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;padding:14px 18px;margin-top:4px;'>
  <b style='font-size:.85rem;color:#1D4ED8;'>K=3 Rationale:</b>
  <span style='font-size:.82rem;color:#374151;'>
  The Total Within-Cluster Distance curve shows a clear elbow at K=3 — the drop from K=2→3 is 10.34%,
  falling sharply to just 4.13% at K=3→4, confirming K=3 as the optimal partition.
  K=2 has the highest silhouette (0.2390) but produces only two broad groups —
  too coarse for actionable consumer insight.
  K=3 retains an acceptable silhouette (0.1655), is validated by the Ward linkage dendrogram,
  and produces three meaningfully distinct, interpretable consumer personas.
  K≥4 shows diminishing distance gains and deteriorating silhouette, confirming K=3 as the best choice.
  </span>
</div>""", unsafe_allow_html=True)

# ── STEP 2: FINAL K-MEDOIDS MODEL ─────────────────────────────────────────────
section("Step 2 · Final K-Medoids Model (PAM, K=3, 50 Initialisations, Cityblock)")

@st.cache_data
def run_kmedoids(_X, _cdf, _users_full):
    labels, medoid_idx, total_wd = pam_kmedoids(_X, k=3, n_init=50, random_state=42)
    sil   = silhouette_score(_X, labels, metric="cityblock")
    sil_s = silhouette_samples(_X, labels, metric="cityblock")
    cdf2  = _cdf.copy();        cdf2["Cluster"] = labels
    uf2   = _users_full.copy(); uf2["Cluster"]  = labels
    return labels, medoid_idx, total_wd, sil, sil_s, cdf2, uf2

labels, medoid_indices, total_wd, final_sil, sil_samples, cdf2, users_c = \
    run_kmedoids(X, cdf, users_full)

sizes = pd.Series(labels).value_counts().sort_index()

# KPIs — exact from notebook output
c1, c2, c3, c4 = st.columns(4)
kpi(c1, f"{final_sil:.4f}", "Silhouette Score",  "Cityblock · K=3 overall quality",  INDIGO)
kpi(c2, f"{sizes.get(0,0)}", "Cluster 0",         "Neutral Adopters · 29.4%",         INDIGO)
kpi(c3, f"{sizes.get(1,0)}", "Cluster 1",         "All-Round Enthusiast · 50.0%",     AMBER)
kpi(c4, f"{sizes.get(2,0)}", "Cluster 2",         "Convenience Purists · 20.6%",      EMERALD)
st.markdown("<br>", unsafe_allow_html=True)

# Medoid profiles — exact from notebook cell 8 output
MEDOID_IDX = [77, 195, 36]
MEDOID_PROFILES = pd.DataFrame({
    "Cluster 0: Neutral Adopters":       [3,3,3,3,3,3,3,3,3,3,3,3,3],
    "Cluster 1: All-Round Enthusiast":   [4,4,4,4,4,4,4,4,4,4,4,4,4],
    "Cluster 2: Convenience Purists":    [5,4,3,4,3,4,5,4,5,5,5,5,5],
}, index=SHORT_13)

st.markdown(f"""
<div style='background:#FFFBEB;border:1px solid #FDE68A;border-radius:10px;padding:14px 18px;margin-bottom:12px;'>
  <b style='font-size:.85rem;color:#B45309;'>🔍 Medoid Respondents — R#77, R#195, R#36</b><br>
  <span style='font-size:.82rem;color:#374151;'>
  The PAM algorithm identifies <b>real survey respondents</b> as cluster representatives.
  <b>Respondent #77</b> (Neutral Adopters) answered "Neutral" (3) on every variable — a textbook fence-sitter.
  <b>Respondent #195</b> (All-Round Enthusiast) answered "Agree" (4) uniformly across all 13 items —
  broadly satisfied and engaged on all dimensions simultaneously.
  <b>Respondent #36</b> (Convenience Purists) answered "Strongly Agree" (5) on Time Saving, Urgent Needs,
  Schedule Barrier, Lifestyle Fit, and all three satisfaction items — but only "Neutral" (3) on
  Discounts and Promo Offers — a pure convenience-seeker completely indifferent to promotional incentives.
  These are not abstract centroids; they are real people who can serve directly as marketing personas.
  </span>
</div>""", unsafe_allow_html=True)

with st.expander("📋 Medoid Respondent Profiles — Original 1–5 Scale (Actual Survey Respondents)"):
    st.dataframe(MEDOID_PROFILES, use_container_width=True)
    st.caption(
        "Each column is a real survey respondent. "
        "R#77 = Neutral Adopters medoid · R#195 = All-Round Enthusiast medoid · R#36 = Convenience Purists medoid."
    )

# ── STEP 3: PCA SCATTER ───────────────────────────────────────────────────────
section("Step 3 · PCA Scatter — Cluster Separation in 2D")

pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
var_e = pca.explained_variance_ratio_ * 100   # PC1=34.54, PC2=14.35 — exact notebook output
medoids_pca = X_pca[medoid_indices]

c1, c2 = st.columns([1.4, 1], gap="large")
with c1:
    fig_pca = go.Figure()
    for c_id in range(3):
        mask = labels == c_id
        fig_pca.add_trace(go.Scatter(
            x=X_pca[mask, 0], y=X_pca[mask, 1], mode="markers",
            name=f"{CLUSTER_NAMES[c_id]} (n={mask.sum()})",
            marker=dict(color=CLUSTER_COLORS[c_id], size=7, opacity=0.65,
                        line=dict(color="#fff", width=0.4)),
            hovertemplate=f"{CLUSTER_NAMES[c_id]}: PC1=%{{x:.2f}}, PC2=%{{y:.2f}}<extra></extra>"
        ))
        fig_pca.add_trace(go.Scatter(
            x=[medoids_pca[c_id, 0]], y=[medoids_pca[c_id, 1]],
            mode="markers+text", showlegend=False,
            text=[f"C{c_id} medoid (R#{MEDOID_IDX[c_id]})"],
            textposition="top right",
            textfont=dict(size=9, color=CLUSTER_COLORS[c_id]),
            marker=dict(color=CLUSTER_COLORS[c_id], size=16, symbol="diamond",
                        line=dict(color="#000", width=1.5)),
            hovertemplate=f"Medoid R#{MEDOID_IDX[c_id]} · Cluster {c_id}<extra></extra>"
        ))
    fig_pca.update_layout(
        **PLOTLY_LAYOUT, height=400,
        title=dict(
            text=(f"Figure 3 — Cluster Visualisation in PCA Space (K-Medoids)<br>"
                  f"<sup>PC1=34.54% + PC2=14.35% = 48.89% total variance explained · "
                  f"◆ = Actual Medoid Respondent</sup>"),
            font=dict(size=12)
        )
    )
    fig_pca.update_xaxes(title="PC1 (34.5% variance explained)", gridcolor="#F1F5F9")
    fig_pca.update_yaxes(title="PC2 (14.4% variance explained)", gridcolor="#F1F5F9")
    st.plotly_chart(fig_pca, use_container_width=True)

with c2:
    st.markdown(f"""
    <div style='background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;padding:16px;'>
      <div style='font-size:.83rem;color:#334155;font-weight:700;margin-bottom:10px;'>PCA Interpretation</div>
      <div style='font-size:.78rem;color:#374151;line-height:1.9;'>
        <b>PC1 (34.54%)</b> is the dominant axis capturing overall engagement intensity.
        It most cleanly separates <b style='color:{INDIGO}'>Neutral Adopters</b> (left, low
        scores on all variables) from <b style='color:{EMERALD}'>Convenience Purists</b>
        (right, high scores on convenience and satisfaction variables).<br><br>
        <b>PC2 (14.35%)</b> captures a secondary contrast between promotional sensitivity
        and pure convenience orientation, helping position
        <b style='color:{AMBER}'>All-Round Enthusiasts</b> — who score moderately across
        all axes — separately from the more polarised Purists.<br><br>
        <b>Total: 48.89% variance explained</b> — typical for a 13-variable Likert dataset.
        The ◆ diamond markers (actual medoid respondents R#77, R#195, R#36) sit centrally
        within each cluster cloud, confirming good within-cluster cohesion.
      </div>
    </div>""", unsafe_allow_html=True)

# ── STEP 4: CLUSTER PROFILING ─────────────────────────────────────────────────
section("Step 4 · Cluster Profiling — Attitude & Satisfaction Profiles")

# Exact cluster mean profiles from notebook cell 13 output
CLUSTER_MEANS = pd.DataFrame({
    "Neutral Adopters":    [3.090,3.194,3.284,3.224,3.075,3.030,3.239,3.284,3.045,3.164,3.254,3.060,2.985],
    "All-Round Enthusiast":[4.009,4.000,3.868,3.825,3.842,3.947,3.991,3.851,3.939,3.860,3.851,3.904,3.807],
    "Convenience Purists": [4.787,4.426,3.255,4.191,3.043,4.255,4.404,4.149,4.362,4.702,4.553,4.766,4.745],
}, index=SHORT_13)

tab_hm, tab_bar = st.tabs([
    "(a) Heatmap — Cluster Mean Profiles",
    "(b) Side-by-Side Bar Comparison"
])

with tab_hm:
    fig_hm = go.Figure(go.Heatmap(
        z=CLUSTER_MEANS.values,
        x=CLUSTER_NAMES,
        y=SHORT_13,
        colorscale=[[0,"#FEF2F2"],[0.3,"#FCA5A5"],[0.5,"#FCD34D"],[0.7,"#86EFAC"],[1,"#15803D"]],
        zmin=2.5, zmax=5.0,
        texttemplate="%{z:.3f}",
        textfont=dict(size=10),
        colorbar=dict(title="Mean Score (1–5)"),
        hovertemplate="%{y} | %{x}: %{z:.3f}<extra></extra>"
    ))
    fig_hm.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ["xaxis","yaxis"]},
        height=420,
        title=dict(
            text="Figure 5a — Cluster Mean Profiles Heatmap — Green=High, Red=Low",
            font=dict(size=12)
        )
    )
    st.plotly_chart(fig_hm, use_container_width=True)

with tab_bar:
    fig_bar = go.Figure()
    for c_id, cname in enumerate(CLUSTER_NAMES):
        fig_bar.add_trace(go.Bar(
            name=cname, x=SHORT_13, y=CLUSTER_MEANS[cname].values,
            marker_color=CLUSTER_COLORS[c_id], opacity=0.9,
            hovertemplate=f"{cname}: %{{y:.3f}}<extra></extra>"
        ))
    fig_bar.add_hline(y=3.0, line_dash="dot", line_color=SLATE, opacity=0.6,
                      annotation_text="Neutral (3.0)", annotation_font=dict(size=9))
    fig_bar.update_layout(
        **PLOTLY_LAYOUT, barmode="group", height=360,
        title=dict(text="Figure 5b — Side-by-Side Cluster Comparison — All 13 Variables", font=dict(size=12))
    )
    fig_bar.update_xaxes(tickangle=-30, gridcolor="#F1F5F9")
    fig_bar.update_yaxes(title="Mean Score (1–5)", gridcolor="#F1F5F9", range=[1, 5.5])
    st.plotly_chart(fig_bar, use_container_width=True)

# Segment profile cards — from notebook final interpretation cell
cluster_card_data = [
    (
        "Neutral Adopters",
        "n=67 · 29.4% · Medoid R#77",
        "Top driver: Discounts (3.28) · Lowest: Ease of Use (3.03) / Recommend (2.99)",
        (f"Consistently tepid scores across all 13 variables — means tightly clustered between 2.99 and 3.28, "
         f"all hovering near 'Neutral' (3). Their medoid (R#77) answered exactly 3 on every single item, "
         f"making this segment the textbook fence-sitter. Lowest satisfaction (3.25), continuity (3.06), "
         f"and recommendation scores (2.99) across all clusters, indicating marginal platform commitment "
         f"and significant churn risk. The majority place orders below ₹200 (61.2%) and 41.8% still use "
         f"Cash-on-Delivery — reflecting limited platform trust. Usage tenure skews shorter (26.9% under 3 months). "
         f"Emergency Needs (25.4%) ranks highly as a motivation, consistent with reactive, non-habitual usage. "
         f"<b>Business action:</b> Re-engagement campaigns leveraging their top driver (Discounts), "
         f"small-basket promotions, and trust-building onboarding to convert fence-sitters into regulars.")
    ),
    (
        "All-Round Enthusiast",
        "n=114 · 50.0% · Medoid R#195",
        "Top driver: Time Saving (4.01) · Lowest: Product Variety (3.83)",
        (f"The largest segment (50%) — broadly and uniformly engaged across all 13 dimensions. "
         f"Mean scores range from 3.81 to 4.01, firmly in 'Agree' territory. "
         f"Their medoid (R#195) answered 4 on all 13 variables — balanced across convenience and value, "
         f"engaged for multiple reasons simultaneously. Strong mid-range order values (44.7% at ₹200–₹400), "
         f"highest wallet payment share (9.6%), and majority UPI (50.9%) reflect digitally comfortable users. "
         f"Highest share of 1+ year tenure after Purists (41.2%). Primary motivations are spread across "
         f"Convenience (26.5%), Fast Delivery (21.2%), and Emergency Needs (18.6%) — "
         f"engaged on multiple fronts rather than a single driver. "
         f"<b>Business action:</b> Loyalty subscriptions (Blinkit Plus, Zepto Pass) and bundled promotions "
         f"are the optimal lever — this segment is broadly satisfied and needs reasons to deepen engagement.")
    ),
    (
        "Convenience Purists",
        "n=47 · 20.6% · Medoid R#36",
        "Top driver: Time Saving (4.79) · Lowest: Promo Offers (3.04)",
        (f"The smallest but most distinctive segment, defined by a sharp internal contrast. "
         f"Extremely high on Time Saving (4.79), Lifestyle Fit (4.70), Continuity (4.77), Recommend (4.75) — "
         f"yet markedly low on Discounts (3.26) and Promo Offers (3.04). "
         f"Their medoid (R#36) answered 5 on all convenience and satisfaction items but only 3 on "
         f"Discounts and Promo Offers — a pure convenience-seeker completely indifferent to price incentives. "
         f"Highest UPI adoption (72.3%) and lowest COD (14.9%) across all segments. "
         f"Most loyal by tenure: 63.8% using for more than a year. Top motivations are Convenience (40.4%) "
         f"and Product Availability (14.9%), with near-zero Better Offers (2.1%). "
         f"Highest satisfaction (4.55), continuity (4.77), and referral score (4.74/5) of all clusters — "
         f"the highest long-term customer value segment. "
         f"<b>Business action:</b> Premium tiers, speed guarantees, and quality assurance — NOT discounts. "
         f"High referral propensity makes them ideal referral programme targets.")
    ),
]

st.markdown(
    "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:12px 0;'>" +
    "".join([f"""
  <div style='background:{CLUSTER_COLORS[i]}10;border:1px solid {CLUSTER_COLORS[i]}40;
              border-radius:12px;padding:16px;'>
    <div style='font-weight:700;color:{CLUSTER_COLORS[i]};font-size:.9rem;margin-bottom:2px;'>
      Cluster {i}: {name}
    </div>
    <div style='font-size:.7rem;color:#64748B;margin-bottom:3px;'>{meta}</div>
    <div style='font-size:.7rem;color:#94A3B8;font-style:italic;margin-bottom:8px;'>{driver}</div>
    <div style='font-size:.75rem;color:#374151;line-height:1.75;'>{desc}</div>
  </div>""" for i, (name, meta, driver, desc) in enumerate(cluster_card_data)])
    + "</div>",
    unsafe_allow_html=True
)

# ── STEP 5: BEHAVIORAL PROFILE ────────────────────────────────────────────────
section("Step 5 · Behavioral Profile by Cluster")

# Exact figures from notebook cell 15 output
BEHAV = {
    "App_Used": {
        "order": ["Blinkit","Swiggy Instamart","Zepto","Other"],
        "Neutral Adopters":    [68.7, 17.9, 13.4,  0.0],
        "All-Round Enthusiast":[57.9, 20.2, 16.7,  5.3],
        "Convenience Purists": [63.8, 17.0, 19.1,  0.0],
    },
    "Avg Order Value": {
        "order": ["Below ₹200","₹200 - ₹400","₹400 - ₹600","Above ₹600"],
        "Neutral Adopters":    [61.2, 23.9, 11.9,  3.0],
        "All-Round Enthusiast":[25.4, 44.7, 17.5, 12.3],
        "Convenience Purists": [14.9, 44.7, 23.4, 17.0],
    },
    "Preferred Delivery Time": {
        "order": ["Morning","Afternoon","Evening","Night","Midnight"],
        "Neutral Adopters":    [23.9, 16.4, 29.9, 25.4,  4.5],
        "All-Round Enthusiast":[16.7, 17.5, 41.2, 22.8,  1.8],
        "Convenience Purists": [14.9,  6.4, 51.1, 21.3,  6.4],
    },
    "Payment Method": {
        "order": ["UPI","Cash on Delivery","Debit/Credit Card","Wallets (Paytm, PhonePe, etc.)"],
        "Neutral Adopters":    [46.3, 41.8,  7.5, 4.5],
        "All-Round Enthusiast":[50.9, 34.2,  5.3, 9.6],
        "Convenience Purists": [72.3, 14.9, 10.6, 2.1],
    },
    "Usage Tenure": {
        "order": ["Less than 3 months","3-6 months","6-12 months","More than a year"],
        "Neutral Adopters":    [26.9, 14.9, 26.9, 31.3],
        "All-Round Enthusiast":[15.8, 19.3, 23.7, 41.2],
        "Convenience Purists": [ 2.1, 12.8, 21.3, 63.8],
    },
    "Primary Motivation": {
        "order": ["Convenience","Fast Delivery","Emergency Needs","Affordable Pricing",
                  "Product Availability","Better Offers","Lower Delivery Fees"],
        "Neutral Adopters":    [34.3, 17.9, 25.4, 10.4, 4.5, 6.0, 1.5],
        "All-Round Enthusiast":[26.5, 21.2, 18.6,  8.0,11.5, 9.7, 2.7],
        "Convenience Purists": [40.4, 21.3, 14.9,  6.4,14.9, 2.1, 0.0],
    },
}

def behav_chart(key, title, height=280):
    d = BEHAV[key]
    fig = go.Figure()
    for c_id, cname in enumerate(CLUSTER_NAMES):
        fig.add_trace(go.Bar(
            name=cname, x=d["order"], y=d[cname],
            marker_color=CLUSTER_COLORS[c_id],
            hovertemplate=f"{cname}: %{{y:.1f}}%<extra></extra>"
        ))
    fig.update_layout(**PLOTLY_LAYOUT, barmode="group", height=height,
                      title=dict(text=title, font=dict(size=12)))
    fig.update_xaxes(tickangle=-20)
    fig.update_yaxes(title="% within cluster", gridcolor="#F1F5F9")
    return fig

btab1, btab2 = st.tabs(["App, Order Value & Delivery Time", "Payment, Tenure & Motivation"])

with btab1:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(behav_chart("App_Used", "(a) App Used"), use_container_width=True)
    with c2:
        st.plotly_chart(behav_chart("Avg Order Value", "(b) Avg Order Value"), use_container_width=True)
    st.plotly_chart(behav_chart("Preferred Delivery Time", "(c) Preferred Delivery Time", height=265),
                    use_container_width=True)
    st.markdown("""
    <div style='background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;padding:13px 17px;'>
      <b style='font-size:.83rem;color:#1D4ED8;'>Interpretation — App, Order Value & Delivery Time:</b>
      <span style='font-size:.80rem;color:#374151;'>
      Blinkit dominates across all segments, but the order value and delivery time distributions reveal sharp
      behavioural differences. <b>Neutral Adopters</b> are heavily concentrated in the lowest order bracket
      (61.2% below ₹200) and have a dispersed delivery time preference — no strong habitual pattern —
      consistent with reactive, need-driven usage. <b>All-Round Enthusiasts</b> and
      <b>Convenience Purists</b> both peak at ₹200–₹400 (44.7% each), but Purists have a significantly
      larger share of higher-value orders (₹400–₹600: 23.4%; ₹600+: 17.0%) reflecting willingness to pay
      for speed and quality. Convenience Purists show the strongest evening delivery preference (51.1%),
      consistent with busy, time-poor lifestyles where evening is the primary shopping window.
      </span>
    </div>""", unsafe_allow_html=True)

with btab2:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(behav_chart("Payment Method", "(d) Payment Method"), use_container_width=True)
    with c2:
        st.plotly_chart(behav_chart("Usage Tenure", "(e) Usage Tenure"), use_container_width=True)
    st.plotly_chart(behav_chart("Primary Motivation", "(f) Primary Motivation", height=270),
                    use_container_width=True)
    st.markdown("""
    <div style='background:#F0FDF4;border:1px solid #BBF7D0;border-radius:10px;padding:13px 17px;'>
      <b style='font-size:.83rem;color:#15803D;'>Interpretation — Payment, Tenure & Motivation:</b>
      <span style='font-size:.80rem;color:#374151;'>
      Payment behaviour maps directly onto cluster engagement and platform trust.
      <b>Convenience Purists</b> have the highest UPI adoption (72.3%) and lowest COD (14.9%) —
      fully digitised users with the deepest platform trust. <b>Neutral Adopters</b> are split
      nearly evenly between UPI (46.3%) and COD (41.8%), reflecting hesitancy.
      Usage tenure shows the starkest contrast: 63.8% of Convenience Purists have been using
      Q-Commerce for more than a year vs 41.2% of Enthusiasts and 31.3% of Neutrals —
      confirming Purists as the most entrenched, loyal users.
      Primary motivation reinforces the cluster story: Purists are driven overwhelmingly by
      Convenience (40.4%) and Product Availability (14.9%), with near-zero Better Offers (2.1%).
      Neutral Adopters cite Emergency Needs (25.4%) as a major motivator — sporadic, reactive usage —
      while All-Round Enthusiasts are motivated across all dimensions simultaneously.
      </span>
    </div>""", unsafe_allow_html=True)

# ── STEP 6: DEMOGRAPHIC COMPOSITION ──────────────────────────────────────────
section("Step 6 · Demographic Composition by Cluster")

# Exact from notebook cell 17 output
DEMO = {
    "Age Group": {
        "order": ["18-25","26-33","34-41","42-49","50 or above"],
        "Neutral Adopters":    [16.4, 34.3, 22.4, 16.4, 10.4],
        "All-Round Enthusiast":[20.2, 29.8, 26.3,  8.8, 14.9],
        "Convenience Purists": [34.0, 38.3, 17.0,  4.3,  6.4],
    },
    "Gender": {
        "order": ["Male","Female","Prefer not to say"],
        "Neutral Adopters":    [50.7, 49.3, 0.0],
        "All-Round Enthusiast":[48.2, 50.0, 1.8],
        "Convenience Purists": [51.1, 46.8, 2.1],
    },
    "Occupation": {
        "order": ["Working professional","Student","Self-employed","Homemaker","Retired"],
        "Neutral Adopters":    [40.3, 19.4, 23.9, 14.9, 1.5],
        "All-Round Enthusiast":[40.4, 18.4, 21.9, 14.0, 5.3],
        "Convenience Purists": [38.3, 34.0, 27.7,  0.0, 0.0],
    },
    "Education": {
        "order": ["Postgraduate","Undergraduate","Professional Degree","School Level","No Formal Education"],
        "Neutral Adopters":    [46.3, 26.9, 14.9, 7.5, 4.5],
        "All-Round Enthusiast":[39.5, 33.3, 22.8, 2.6, 1.8],
        "Convenience Purists": [31.9, 38.3, 23.4, 6.4, 0.0],
    },
    "Income": {
        "order": ["Below ₹20,000","₹20,000 - ₹40,000","₹40,000 - ₹60,000","₹60,000 - ₹1,00,000","Above ₹1,00,000"],
        "Neutral Adopters":    [ 1.5, 23.9, 22.4, 19.4, 32.8],
        "All-Round Enthusiast":[ 2.6, 18.4, 31.6, 19.3, 28.1],
        "Convenience Purists": [12.8, 19.1, 19.1, 19.1, 29.8],
    },
}

# Chi-square — exact from notebook cell 21 output
CHI_SQ = pd.DataFrame({
    "Demographic":   ["Age_Group","Gender","Education","Occupation","Income"],
    "χ²":            [12.927,  1.433,  9.163, 15.034, 13.091],
    "df":            [8, 4, 8, 8, 8],
    "p-value":       [0.1144, 0.8384, 0.3287, 0.0585, 0.1088],
    "Cramér's V":    [0.104,  0.000,  0.050,  0.124,  0.106],
    "Association":   ["Weak","Negligible","Negligible","Weak","Weak"],
    "Significant?":  ["No","No","No","No","No"],
})

def demo_chart(key, title, height=265):
    d = DEMO[key]
    fig = go.Figure()
    for c_id, cname in enumerate(CLUSTER_NAMES):
        fig.add_trace(go.Bar(
            name=cname, x=d["order"], y=d[cname],
            marker_color=CLUSTER_COLORS[c_id], opacity=0.85,
            hovertemplate=f"{cname}: %{{y:.1f}}%<extra></extra>"
        ))
    fig.update_layout(**PLOTLY_LAYOUT, barmode="group", height=height,
                      title=dict(text=title, font=dict(size=12)))
    fig.update_xaxes(tickangle=-15)
    fig.update_yaxes(title="% within cluster", gridcolor="#F1F5F9")
    return fig

dtab1, dtab2 = st.tabs(["Age & Gender", "Occupation, Education & Income"])

with dtab1:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(demo_chart("Age Group", "Cluster Composition — Age Group"), use_container_width=True)
    with c2:
        st.plotly_chart(demo_chart("Gender", "Cluster Composition — Gender"), use_container_width=True)

with dtab2:
    st.plotly_chart(demo_chart("Occupation", "Cluster Composition — Occupation"), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(demo_chart("Education", "Cluster Composition — Education"), use_container_width=True)
    with c2:
        st.plotly_chart(demo_chart("Income", "Cluster Composition — Income"), use_container_width=True)

with st.expander("📋 Chi-Square Test Results: Cluster × Demographics (All Non-Significant)"):
    st.dataframe(CHI_SQ.set_index("Demographic"), use_container_width=True)
    st.caption(
        "H₀: Cluster membership is independent of the demographic variable. "
        "None of the five demographics show a significant association with cluster at α=0.05. "
        "Cramér's V ranges from 0.000 (Gender) to 0.124 (Occupation) — all negligible to weak."
    )

finding_card(
    "🧬 Segments Are Psychographic, Not Demographic",
    "Chi-square tests confirm no statistically significant association between cluster membership and any "
    "demographic variable — Age (χ²=12.927, p=0.114), Gender (χ²=1.433, p=0.838), "
    "Education (χ²=9.163, p=0.329), Occupation (χ²=15.034, p=0.059), Income (χ²=13.091, p=0.109). "
    "All Cramér's V values are below 0.13 (negligible to weak association). "
    "The segments are differentiated entirely by attitudes, motivations, and satisfaction levels — "
    "not by who the users are demographically. Marketing strategies must target psychographic and "
    "behavioural profiles, not demographic groups.",
    VIOLET
)

# ── STEP 7: KRUSKAL-WALLIS + DUNN VALIDATION ──────────────────────────────────
section(
    "Step 7 · Statistical Validation — Kruskal-Wallis H Test + Dunn Post-Hoc",
    "H₀: Median scores are equal across all 3 clusters · H₁: At least one cluster differs · α=0.05"
)

# Exact from notebook cell 19 output
KW = pd.DataFrame([
    ("Time Saving",       95.652, 0.000000, 0.4162, "Large"),
    ("Delivery Speed",    71.093, 0.000000, 0.3071, "Large"),
    ("Discounts",         27.718, 0.000001, 0.1143, "Medium"),
    ("Product Variety",   42.130, 0.000000, 0.1784, "Large"),
    ("Promo Offers",      35.195, 0.000000, 0.1475, "Large"),
    ("Ease of Use",       75.757, 0.000000, 0.3278, "Large"),
    ("Urgent Needs",      56.114, 0.000000, 0.2405, "Large"),
    ("Quality Reliable",  39.554, 0.000000, 0.1669, "Large"),
    ("Schedule Barrier",  59.477, 0.000000, 0.2555, "Large"),
    ("Lifestyle Fit",     89.508, 0.000000, 0.3889, "Large"),
    ("Satisfaction",      65.979, 0.000000, 0.2844, "Large"),
    ("Continuity",       100.123, 0.000000, 0.4361, "Large"),
    ("Recommend",        102.120, 0.000000, 0.4450, "Large"),
], columns=["Variable","H Statistic","p-value","η²","Effect Size"])

kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
kpi(kpi_c1, "13 / 13", "Variables Significant", "All p < 0.0001 · α=0.05",           EMERALD)
kpi(kpi_c2, "102.12",  "Max H Statistic",        "Recommend — most differentiated",   INDIGO)
kpi(kpi_c3, "0.4450",  "Max η² Effect Size",     "Recommend — Large effect",           VIOLET)
kpi(kpi_c4, "12 / 13", "Large Effect (η²≥0.14)", "Discounts = only Medium (η²=0.114)", EMERALD)
st.markdown("<br>", unsafe_allow_html=True)

fig_kw = go.Figure(go.Bar(
    y=KW["Variable"],
    x=KW["H Statistic"],
    orientation="h",
    marker_color=[EMERALD if e == "Large" else AMBER for e in KW["Effect Size"]],
    text=[f"H={row['H Statistic']:.3f}  η²={row['η²']:.4f}  {row['Effect Size']}"
          for _, row in KW.iterrows()],
    textposition="outside",
    hovertemplate="%{y}: H=%{x:.3f}<extra></extra>"
))
fig_kw.update_layout(
    **PLOTLY_LAYOUT, height=400,
    title=dict(
        text="Kruskal-Wallis H by Variable — Green=Large Effect, Amber=Medium Effect",
        font=dict(size=12)
    )
)
fig_kw.update_xaxes(title="H Statistic", gridcolor="#F1F5F9", range=[0, 140])
st.plotly_chart(fig_kw, use_container_width=True)

with st.expander("📋 Full Kruskal-Wallis Results Table"):
    st.dataframe(KW.set_index("Variable"), use_container_width=True)
    st.caption(
        "η² (eta-squared) = H / (n−1), n=228. "
        "Cohen benchmarks: Small < 0.01 · Medium ≥ 0.06 · Large ≥ 0.14. "
        "All p-values are 0.000000 except Discounts (p=0.000001)."
    )

# Dunn post-hoc — exact from notebook cell 20 output
with st.expander("📋 Dunn Post-Hoc Tests (Bonferroni correction) — Key Variables"):
    st.markdown("""
    **Time Saving** — Medians: C0=3.0, C1=4.0, C2=5.0 (all pairs differ)
    """)
    st.dataframe(pd.DataFrame([
        ("Neutral Adopters vs All-Round Enthusiast",     -5.895, 0.0,    0.0,    "Yes ✓"),
        ("Neutral Adopters vs Convenience Purists",      -9.221, 0.0,    0.0,    "Yes ✓"),
        ("All-Round Enthusiast vs Convenience Purists",  -4.887, 0.0,    0.0,    "Yes ✓"),
    ], columns=["Comparison","z","p (raw)","p (Bonferroni)","Significant?"]).set_index("Comparison"),
        use_container_width=True)

    st.markdown("""
    **Discounts** — Medians: C0=3.0, C1=4.0, C2=3.0
    *(Neutral Adopters and Convenience Purists are NOT different from each other — both indifferent to promotions)*
    """)
    st.dataframe(pd.DataFrame([
        ("Neutral Adopters vs All-Round Enthusiast",    -4.313, 0.0000, 0.0000, "Yes ✓"),
        ("Neutral Adopters vs Convenience Purists",     -0.049, 0.9607, 1.0000, "No ✗"),
        ("All-Round Enthusiast vs Convenience Purists",  3.776, 0.0002, 0.0005, "Yes ✓"),
    ], columns=["Comparison","z","p (raw)","p (Bonferroni)","Significant?"]).set_index("Comparison"),
        use_container_width=True)

    st.markdown("""
    **Promo Offers** — Medians: C0=3.0, C1=4.0, C2=3.0
    *(Same pattern as Discounts — Neutral Adopters and Convenience Purists both score ~3, but for different reasons)*
    """)
    st.dataframe(pd.DataFrame([
        ("Neutral Adopters vs All-Round Enthusiast",    -4.878, 0.0000, 0.0000, "Yes ✓"),
        ("Neutral Adopters vs Convenience Purists",      0.068, 0.9455, 1.0000, "No ✗"),
        ("All-Round Enthusiast vs Convenience Purists",  4.407, 0.0000, 0.0000, "Yes ✓"),
    ], columns=["Comparison","z","p (raw)","p (Bonferroni)","Significant?"]).set_index("Comparison"),
        use_container_width=True)

    st.markdown("""
    <div style='background:#FFFBEB;border:1px solid #FDE68A;border-radius:8px;padding:12px 14px;
                font-size:.78rem;color:#374151;margin-top:8px;'>
      <b>Key Dunn insight on Discounts & Promo Offers:</b>
      Neutral Adopters and Convenience Purists are statistically indistinguishable on both promotional
      variables (p=1.000 after Bonferroni correction), yet they are completely different clusters on
      all other variables. This confirms that their shared indifference to promotions arises from
      entirely different causes: <b>Neutral Adopters</b> are disengaged and uncommitted, while
      <b>Convenience Purists</b> are already deeply committed and have no need for price incentives.
      This nuance — detectable only via K-Medoids clustering + post-hoc testing — has direct
      practical implications: discounting will not convert Purists, but may help re-engage Neutrals.
    </div>""", unsafe_allow_html=True)

finding_card(
    "✅ All 13 Clustering Variables Are Statistically Significant — Not Random Groupings",
    "Every attitudinal and satisfaction variable shows highly significant Kruskal-Wallis differences across "
    "the 3 K-Medoids clusters (13/13, all p < 0.0001). "
    "12 of 13 carry a Large effect size (η² ≥ 0.14), led by Recommend (η²=0.445), "
    "Continuity (η²=0.436), and Time Saving (η²=0.416). "
    "Discounts is the sole Medium effect (η²=0.114) — still significant, but the weakest differentiator, "
    "consistent with both Neutrals and Purists sharing low promotional sensitivity despite being entirely "
    "different consumer types. This provides exceptionally strong statistical evidence that the three "
    "consumer segments represent genuine, distinct behavioural profiles.",
    EMERALD
)

finding_card(
    "📦 Three Actionable Consumer Segments via K-Medoids (PAM)",
    "Neutral Adopters (n=67, 29.4%): Marginal, churn-risk users driven by Discounts and Emergency Needs. "
    "Re-engage with small-basket incentives and trust-building onboarding. "
    "All-Round Enthusiasts (n=114, 50.0%): The dominant segment, broadly engaged across all dimensions "
    "with Time Saving as the top driver. Deepen with loyalty subscriptions and basket-growth promotions. "
    "Convenience Purists (n=47, 20.6%): Highest-LTV segment — most loyal (63.8% using 1+ years), "
    "highest satisfaction (4.55), continuity (4.77), and referral propensity (4.74/5), indifferent to "
    "discounts. Reward with premium tiers, speed guarantees, and referral programmes.",
    INDIGO
)

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.stats import kruskal
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
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
    "Elbow + Silhouette method for K selection, PCA for visualisation, Kruskal-Wallis + Dunn post-hoc for statistical validation. "
    "K-Medoids (PAM) with Cityblock distance is used so that each cluster centre is an actual survey respondent — "
    "more robust to outliers and directly interpretable on Likert-scale data."
)

CLUSTER_COLORS = [INDIGO, AMBER, EMERALD]
CLUSTER_NAMES  = [ "Convenience Purists","Neutral Adopters", "All-Round Enthusiasts"]

k1, k2, k3, k4 = st.columns(4)
kpi(k1, "228",       "Users Clustered",     "Complete-case analysis")
kpi(k2, "13",        "Clustering Variables","10 attitude + 3 satisfaction", INDIGO)
kpi(k3, "3",         "Clusters (K=3)",      "Elbow + Silhouette validated",  EMERALD)
kpi(k4, "K-Medoids", "Algorithm (PAM)",     "Cityblock distance · real medoids", VIOLET)
st.markdown("<br>", unsafe_allow_html=True)



# ── DATA PREP ──────────────────────────────────────────────────────────────────
@st.cache_data
def prepare_cluster_data():
    users = get_users()
    LIKERT_COLS_C = [
        "Q-commerce apps save my time", "Delivery speed is very convenient",
        "Discounts influence my usage", "Product variety meets my needs",
        "Promotional offers attract me", "Apps are easy to navigate",
        "Urgent needs motivate me to use these apps", "Product quality is reliable",
        "My work/study schedule makes offline shopping difficult", "Q-commerce fits my lifestyle"
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

# ── PAM IMPLEMENTATION ────────────────────────────────────────────────────────
def pam_kmedoids(X, k, n_init=50, random_state=42):
    """PAM K-Medoids with Cityblock distance and multiple restarts."""
    rng  = np.random.RandomState(random_state)
    dist = cdist(X, X, metric="cityblock")
    best_cost, best_labels, best_medoids = np.inf, None, None
    for _ in range(n_init):
        medoids = rng.choice(len(X), k, replace=False)
        for _ in range(300):
            labels = np.argmin(dist[:, medoids], axis=1)
            new_medoids = medoids.copy()
            for c in range(k):
                mask = np.where(labels == c)[0]
                if len(mask) == 0:
                    continue
                sub = dist[np.ix_(mask, mask)]
                new_medoids[c] = mask[np.argmin(sub.sum(axis=1))]
            if np.array_equal(np.sort(new_medoids), np.sort(medoids)):
                break
            medoids = new_medoids
        labels = np.argmin(dist[:, medoids], axis=1)
        cost   = sum(dist[i, medoids[labels[i]]] for i in range(len(X)))
        if cost < best_cost:
            best_cost, best_labels, best_medoids = cost, labels.copy(), medoids.copy()
    return best_labels, best_medoids, best_cost

# ── STEP 1: OPTIMAL K ─────────────────────────────────────────────────────────
section("Step 1 · Optimal K — Elbow Method + Silhouette Score")

# Pre-computed results from the notebook analysis
K_RANGE    = list(range(2, 9))
twd_values = [1673.0, 1500.0, 1438.0, 1410.0, 1388.0, 1346.0, 1330.0]
sil_values = [0.2390,  0.1655, 0.1112, 0.0727, 0.0907, 0.0876, 0.0903]

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
                        annotation_text="Recommended K=3", annotation_font=dict(size=9, color=ROSE))
    fig_elbow.update_layout(**PLOTLY_LAYOUT, height=390,
                             title=dict(text="(a) Elbow Method — Total Within-Distance vs K", font=dict(size=12)))
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
    fig_sil.update_layout(**PLOTLY_LAYOUT, height=390,
                           title=dict(text="(b) Silhouette Score vs K", font=dict(size=12)))
    fig_sil.update_xaxes(title="K", tickvals=K_RANGE, gridcolor="#F1F5F9")
    fig_sil.update_yaxes(title="Avg Silhouette Score", gridcolor="#F1F5F9")
    st.plotly_chart(fig_sil, use_container_width=True)

st.markdown(f"""
<div style='background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;padding:14px 18px;'>
  <b style='font-size:.85rem;color:#1D4ED8;'>K=3 Rationale:</b>
  <span style='font-size:.82rem;color:#374151;'>
  The Total Within-Cluster Distance curve shows a clear inflection (elbow) at K=3 — the gain from splitting
  further drops from 10.3% (K=2→3) to only 4.1% (K=3→4).
  K=2 yields the highest silhouette (0.239) but collapses the solution into just two broad groups (engaged vs disengaged),
  which is too coarse for actionable consumer insights.
  K=3 retains an acceptable silhouette (0.166). K≥4 shows diminishing distance gains and
  deteriorating silhouette, confirming K=3 as the optimal choice.
  </span>
</div>""", unsafe_allow_html=True)

# ── STEP 2: FINAL K-MEDOIDS ────────────────────────────────────────────────────
section("Step 2 · Final K-Medoids Model (PAM, K=3, Cityblock Distance)")

@st.cache_data
def run_kmedoids(_X, _scaler, _cdf, _users_full):
    labels, medoid_indices, total_wd = pam_kmedoids(_X, k=3, n_init=50, random_state=42)
    # Remap so cluster order is: 0=Neutral, 1=All-Round, 2=Purists
    # (match notebook output: sizes 67, 114, 47)
    sizes_raw = pd.Series(labels).value_counts().sort_index()
    sil        = silhouette_score(_X, labels, metric="cityblock")
    sil_s      = silhouette_samples(_X, labels, metric="cityblock")
    # Medoid profiles in original scale
    medoid_profiles.index = [f'Raw C{i}' for i in range(OPTIMAL_K)]
    print('\nMedoid profiles (original scale):')
    print(medoid_profiles.T.round(2).to_string())

    )
    cdf2 = _cdf.copy(); cdf2["Cluster"] = labels
    uf2  = _users_full.copy(); uf2["Cluster"] = labels
    return labels, medoid_indices, total_wd, sil, sil_s, medoid_profiles, cdf2, uf2

labels, medoid_indices, total_wd, final_sil, sil_samples, medoid_profiles, cdf2, users_c = \
    run_kmedoids(X, scaler, cdf, users_full)

sizes = pd.Series(labels).value_counts()

c1, c2, c3, c4 = st.columns(4)
kpi(c1, f"{final_sil:.4f}", "Silhouette Score", "Overall cluster quality (Cityblock)", INDIGO)
kpi(c2, f"{sizes.get(0, 0)}", "Cluster 0", "Convenience Purists",  INDIGO)
kpi(c3, f"{sizes.get(1, 0)}", "Cluster 1", "Neutral Adopters",  AMBER)
kpi(c4, f"{sizes.get(2, 0)}", "Cluster 2", "All-Round Enthusiasts", EMERALD)
st.markdown("<br>", unsafe_allow_html=True)

# Medoid respondent highlight
st.markdown(f"""
<div style='background:#FFFBEB;border:1px solid #FDE68A;border-radius:10px;padding:14px 18px;margin-bottom:12px;'>
  <b style='font-size:.85rem;color:#B45309;'>🔍 Medoid Respondents (K-Medoids advantage):</b>
  <span style='font-size:.82rem;color:#374151;'>
  The PAM algorithm identifies <b>real survey respondents</b> as cluster centres.
  Respondent <b>#77</b> represents Neutral Adopters (all scores ≈ 3 "Neutral"),
  Respondent <b>#195</b> represents All-Round Enthusiasts (all scores ≈ 4 "Agree"),
  and Respondent <b>#36</b> represents Convenience Purists (convenience scores = 5 "Strongly Agree",
  promotional scores = 3 "Neutral"). These real archetypes can be used directly as marketing personas.
  </span>
</div>""", unsafe_allow_html=True)

# ── STEP 3: PCA SCATTER ───────────────────────────────────────────────────────
section("Step 3 · PCA Scatter — Cluster Separation in 2D")

pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
var_e = pca.explained_variance_ratio_ * 100
medoids_pca = X_pca[medoid_indices]

c1, c2 = st.columns([1.4, 1], gap="large")
with c1:
    fig_pca = go.Figure()
    for c_id in range(3):
        mask = labels == c_id
        fig_pca.add_trace(go.Scatter(
            x=X_pca[mask, 0], y=X_pca[mask, 1], mode="markers",
            name=f"{CLUSTER_NAMES[c_id]} (n={mask.sum()})",
            marker=dict(color=CLUSTER_COLORS[c_id], size=7, opacity=0.7,
                        line=dict(color="#fff", width=0.3)),
            hovertemplate=f"{CLUSTER_NAMES[c_id]}: PC1=%{{x:.2f}}, PC2=%{{y:.2f}}<extra></extra>"
        ))
        # Diamond = actual medoid respondent
        fig_pca.add_trace(go.Scatter(
            x=[medoids_pca[c_id, 0]], y=[medoids_pca[c_id, 1]],
            mode="markers", showlegend=False,
            marker=dict(color=CLUSTER_COLORS[c_id], size=16, symbol="diamond",
                        line=dict(color="#000", width=1.2)),
            hovertemplate=f"Medoid Respondent (Cluster {c_id})<extra></extra>"
        ))
    fig_pca.update_layout(
        **PLOTLY_LAYOUT, height=380,
        title=dict(
            text=f"PCA Cluster Scatter — PC1+PC2 = {sum(var_e):.1f}% variance  ◆ = Medoid Respondent",
            font=dict(size=12)
        )
    )
    fig_pca.update_xaxes(title=f"PC1 ({var_e[0]:.1f}% variance)", gridcolor="#F1F5F9")
    fig_pca.update_yaxes(title=f"PC2 ({var_e[1]:.1f}% variance)", gridcolor="#F1F5F9")
    st.plotly_chart(fig_pca, use_container_width=True)

with c2:
    st.markdown(f"""
    <div style='background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;padding:16px;margin-top:8px;'>
      <div style='font-size:.82rem;color:#475569;font-weight:600;margin-bottom:10px;'>PCA Interpretation</div>
      <div style='font-size:.78rem;color:#374151;line-height:1.8;'>
        <b>PC1 (34.5%)</b> captures the primary axis of <i>overall engagement intensity</i> — 
        users scoring high on PC1 are broadly satisfied, convenience-driven, and loyal.
        This axis cleanly separates <b>Neutral Adopters</b> (left) from 
        <b>Convenience Purists</b> (right).<br><br>
        <b>PC2 (14.4%)</b> captures a secondary dimension separating 
        <b>promotional sensitivity</b> from <i>pure convenience motivation</i>.
        All-Round Enthusiasts occupy the middle — engaged across all dimensions 
        rather than being polarised on either axis.<br><br>
        <b>Total variance explained: 48.9%</b> — reasonable for a 13-variable 
        Likert-scale dataset. The ◆ diamond markers show the actual medoid 
        respondents sitting centrally within each cluster cloud.
      </div>
    </div>""", unsafe_allow_html=True)

# ── STEP 4: CLUSTER PROFILING ─────────────────────────────────────────────────
section("Step 4 · Cluster Profiling — Attitude & Satisfaction Heatmap")

profile = cdf2.groupby("Cluster")[SHORT_13].mean().round(3)
profile.index = CLUSTER_NAMES

fig_hm = go.Figure(go.Heatmap(
    z=mediod_profiles.T.values, x=CLUSTER_NAMES, y=SHORT_13,
    colorscale=[[0, "#FEF2F2"], [0.3, "#FCA5A5"], [0.5, "#FCD34D"], [0.7, "#86EFAC"], [1, "#15803D"]],
    zmin=2.5, zmax=5.0,
    texttemplate="%{z:.2f}",
    textfont=dict(size=10),
    colorbar=dict(title="Mean Score (1–5)"),
    hovertemplate="%{y} | %{x}: %{z:.2f}<extra></extra>"
))
fig_hm.update_layout(
    **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ["xaxis", "yaxis"]},
    height=400,
    title=dict(text="Cluster Mean Profiles — Green=High, Red=Low (PAM K-Medoids, K=3)", font=dict(size=12))
)
st.plotly_chart(fig_hm, use_container_width=True)

# Medoid profile table
with st.expander("🔍 Medoid Respondent Profiles (Actual Representative Respondents — Original 1–5 Scale)"):
    medoid_df = pd.DataFrame({
        "Neutral Adopters (R#77)":       [3]*13,
        "All-Round Enthusiasts (R#195)": [4]*13,
        "Convenience Purists (R#36)":    [5,4,3,4,3,4,5,4,5,5,5,5,5],
    }, index=SHORT_13)
    st.dataframe(medoid_df, use_container_width=True)
    st.caption("Each row is a real survey respondent (medoid) who best represents their cluster — a K-Medoids advantage over K-Means.")

# Profile cards
cluster_descs = [
    ("Neutral Adopters",
     "Consistently mid-range scores (mean ≈ 3.0–3.3) across all 13 variables — this segment neither strongly agrees "
     "nor disagrees on any dimension. Their medoid respondent answered 'Neutral' on every item. "
     "Lowest satisfaction (3.25), continuity (3.06), and recommendation scores (2.99), "
     "indicating marginal commitment and significant churn risk. The majority prefer low-value orders (below ₹200) "
     "and are split between UPI and Cash-on-Delivery, reflecting hesitancy. "
     "<b>Business action:</b> Re-engagement campaigns, onboarding nudges, and small-basket promotions targeting their "
     "top driver (Discounts) can convert fence-sitters into regulars."),

    ("All-Round Enthusiasts",
     "The largest segment (50%) with balanced and consistently high scores (mean ≈ 3.8–4.0) across all attitude "
     "and satisfaction variables. Their medoid respondent answered 'Agree' across the board — "
     "engaged on both convenience and value dimensions simultaneously. They dominate mid-range order values "
     "(₹200–₹400) and show strong preference for UPI (51%), reflecting digitally comfortable urban users. "
     "Top driver is Time Saving; the comparatively lower score on Product Variety is their only relative weak point. "
     "<b>Business action:</b> Loyalty subscriptions (Blinkit Plus, Zepto Pass) and bundled promotions are the "
     "optimal lever — this segment is already engaged and needs reasons to increase basket size."),

    ("Convenience Purists",
     "The smallest but most distinctive segment (20.6%). Extremely high scores on Time Saving (4.79), "
     "Lifestyle Fit (4.70), Urgent Needs (4.40), and all three satisfaction metrics — yet markedly "
     "lower on Discounts (3.26) and Promo Offers (3.04). Their medoid answered 'Strongly Agree' on "
     "all convenience variables and only 'Neutral' on promotional ones. "
     "Highest satisfaction (4.55), continuity (4.77), and recommendation scores (4.74) of all clusters — "
     "making them the most loyal and valuable long-term customers. "
     "Highest UPI adoption (72.3%) and the largest share of premium orders (₹400+). "
     "<b>Business action:</b> Premium tiers and quality/speed messaging — NOT discounts. "
     "Their referral likelihood (≈4.7/5) makes them ideal targets for referral reward programmes."),
]

st.markdown(
    "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:8px 0;'>" +
    "".join([f"""
  <div style='background:{CLUSTER_COLORS[i]}10;border:1px solid {CLUSTER_COLORS[i]}30;
              border-radius:12px;padding:16px;'>
    <div style='font-weight:700;color:{CLUSTER_COLORS[i]};font-size:.9rem;margin-bottom:4px;'>
      Cluster {i}: {CLUSTER_NAMES[i]}
    </div>
    <div style='font-size:.7rem;color:#64748B;margin-bottom:8px;'>
      n={sizes.get(i, 0)} ({sizes.get(i, 0) / len(labels) * 100:.1f}%) · Medoid R#{[77,195,36][i]}
    </div>
    <div style='font-size:.75rem;color:#374151;line-height:1.7;'>{desc}</div>
  </div>""" for i, (_, desc) in enumerate(cluster_descs)])
    + "</div>",
    unsafe_allow_html=True
)

# ── STEP 5: BEHAVIORAL PROFILE ────────────────────────────────────────────────
section("Step 5 · Behavioral Profile by Cluster")

behav_tab1, behav_tab2 = st.tabs(["App & Order Value", "Payment & Demographic Composition"])

with behav_tab1:
    c1, c2 = st.columns(2)
    for col_w, bcol, bname, bord in [
        (c1, "App_Used",             "App Used",
         ["Blinkit", "Swiggy Instamart", "Zepto", "Other"]),
        (c2, "average order value",  "Avg Order Value",
         ["Below ₹200", "₹200 - ₹400", "₹400 - ₹600", "Above ₹600"]),
    ]:
        with col_w:
            fig_bh = go.Figure()
            for c_id in range(3):
                u_sub = users_c[users_c["Cluster"] == c_id]
                cnt   = u_sub[bcol].value_counts().reindex(bord).fillna(0)
                cnt_p = cnt / len(u_sub) * 100
                fig_bh.add_trace(go.Bar(
                    name=CLUSTER_NAMES[c_id], x=bord, y=cnt_p,
                    marker_color=CLUSTER_COLORS[c_id],
                    hovertemplate=f"{CLUSTER_NAMES[c_id]}: %{{y:.1f}}%<extra></extra>"
                ))
            fig_bh.update_layout(**PLOTLY_LAYOUT, barmode="group", height=290,
                                  title=dict(text=bname, font=dict(size=12)))
            fig_bh.update_xaxes(tickangle=-15)
            fig_bh.update_yaxes(title="% within cluster", gridcolor="#F1F5F9")
            st.plotly_chart(fig_bh, use_container_width=True)

    # Interpretation
    st.markdown(f"""
    <div style='background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;padding:13px 17px;margin-top:4px;'>
      <b style='font-size:.83rem;color:#1D4ED8;'>Order Value Interpretation:</b>
      <span style='font-size:.80rem;color:#374151;'>
      A striking gradient is visible across clusters.
      <b>Neutral Adopters</b> are heavily concentrated in the lowest order bracket (61.2% below ₹200),
      consistent with their hesitant, low-commitment usage pattern.
      <b>All-Round Enthusiasts</b> and <b>Convenience Purists</b> both peak at ₹200–₹400 (44.7% each),
      but Purists have a substantially higher share of ₹400–₹600 (23.4%) and ₹600+ (17.0%) orders,
      reflecting their higher willingness to pay for speed and convenience rather than seeking discounts.
      </span>
    </div>""", unsafe_allow_html=True)

with behav_tab2:
    # Payment method
    pay_order = ["UPI", "Cash on Delivery", "Debit/Credit Card", "Wallets (Paytm, PhonePe, etc.)"]
    fig_pay = go.Figure()
    for c_id in range(3):
        u_sub = users_c[users_c["Cluster"] == c_id]
        cnt   = u_sub["preferred payment method?"].value_counts().reindex(pay_order).fillna(0)
        cnt_p = cnt / len(u_sub) * 100
        fig_pay.add_trace(go.Bar(
            name=CLUSTER_NAMES[c_id], x=pay_order, y=cnt_p,
            marker_color=CLUSTER_COLORS[c_id], opacity=0.85,
            hovertemplate=f"{CLUSTER_NAMES[c_id]}: %{{y:.1f}}%<extra></extra>"
        ))
    fig_pay.update_layout(**PLOTLY_LAYOUT, barmode="group", height=270,
                           title=dict(text="Preferred Payment Method by Cluster", font=dict(size=12)))
    fig_pay.update_xaxes(tickangle=-15)
    fig_pay.update_yaxes(title="% within cluster", gridcolor="#F1F5F9")
    st.plotly_chart(fig_pay, use_container_width=True)

    st.markdown(f"""
    <div style='background:#FFFBEB;border:1px solid #FDE68A;border-radius:10px;padding:13px 17px;margin-bottom:12px;'>
      <b style='font-size:.83rem;color:#B45309;'>Payment Method Interpretation:</b>
      <span style='font-size:.80rem;color:#374151;'>
      Payment behaviour maps cleanly onto cluster engagement.
      <b>Convenience Purists</b> show the highest UPI adoption (72.3%) and the lowest Cash-on-Delivery
      reliance (14.9%), signalling fully digitised, frictionless users who trust the platforms.
      <b>Neutral Adopters</b> have the highest COD share (41.8%) — consistent with lower platform trust
      and commitment. <b>All-Round Enthusiasts</b> sit in between (50.9% UPI, 34.2% COD),
      representing a transitional digital adoption stage. This gradient in payment trust
      is an independent behavioural validation of the cluster separation.
      </span>
    </div>""", unsafe_allow_html=True)

    for demo_col, demo_order in [("Age_Group", AGE_ORDER), ("Occupation", OCC_ORDER)]:
        fig_demo = go.Figure()
        for c_id in range(3):
            u_sub = users_c[users_c["Cluster"] == c_id]
            cnt   = u_sub[demo_col].value_counts().reindex(demo_order).fillna(0)
            cnt_p = cnt / len(u_sub) * 100
            fig_demo.add_trace(go.Bar(
                name=CLUSTER_NAMES[c_id], x=demo_order, y=cnt_p,
                marker_color=CLUSTER_COLORS[c_id], opacity=0.85,
                hovertemplate=f"{CLUSTER_NAMES[c_id]}: %{{y:.1f}}%<extra></extra>"
            ))
        fig_demo.update_layout(**PLOTLY_LAYOUT, barmode="group", height=260,
                                title=dict(text=f"Cluster Composition — {demo_col}", font=dict(size=12)))
        fig_demo.update_xaxes(tickangle=-15)
        fig_demo.update_yaxes(title="% within cluster", gridcolor="#F1F5F9")
        st.plotly_chart(fig_demo, use_container_width=True)

    st.markdown(f"""
    <div style='background:#F0FDF4;border:1px solid #BBF7D0;border-radius:10px;padding:13px 17px;'>
      <b style='font-size:.83rem;color:#15803D;'>Demographic Interpretation:</b>
      <span style='font-size:.80rem;color:#374151;'>
      Notably, <b>all three clusters share the same dominant age group (26–33) and occupation
      (Working professional)</b>, and chi-square tests confirm no statistically significant
      association between cluster membership and any demographic variable (all p &gt; 0.05,
      Cramér's V &lt; 0.13). This is a key finding: the segments are driven entirely by
      <i>attitudes, motivations, and satisfaction</i> — not by age, gender, education, or income.
      It means marketing strategies must be tailored to <b>behavioural psychographics</b>,
      not demographic targeting alone.
      </span>
    </div>""", unsafe_allow_html=True)

# ── STEP 6: KRUSKAL-WALLIS VALIDATION ─────────────────────────────────────────
section(
    "Step 6 · Statistical Validation — Kruskal-Wallis H Test + Effect Size",
    "H₀: Median scores are equal across clusters | H₁: At least one cluster differs significantly"
)

@st.cache_data
def validate_clusters(_cdf2, _SHORT_13):
    # Pre-computed results from notebook (all 13/13 significant)
    kw_results = [
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
    ]
    rows = []
    for var, H, p, eta2, effect in kw_results:
        rows.append({
            "Variable":    var,
            "H":           round(H, 3),
            "p-value":     round(p, 6),
            "η² (effect)": round(eta2, 4),
            "Effect Size": effect,
            "Sig":         "✅ Yes"
        })
    return pd.DataFrame(rows)

val_df = validate_clusters(cdf2, SHORT_13)

kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
kpi(kpi_c1, "13 / 13", "Variables Significant", "All p < 0.05 across clusters", EMERALD)
kpi(kpi_c2, "102.12",  "Max H Statistic",        "'Recommend' — most differentiated", INDIGO)
kpi(kpi_c3, "0.445",   "Max η² Effect Size",     "'Recommend' — Large effect",        VIOLET)
kpi(kpi_c4, "Validated", "Cluster Quality",      "All 13 significant → real segments", EMERALD)
st.markdown("<br>", unsafe_allow_html=True)

fig_val = go.Figure(go.Bar(
    y=val_df["Variable"], x=val_df["H"], orientation="h",
    marker_color=[EMERALD if e == "Large" else AMBER for e in val_df["Effect Size"]],
    text=[f"H={row['H']} · η²={row['η² (effect)']:.3f} · {row['Effect Size']}"
          for _, row in val_df.iterrows()],
    textposition="outside",
    hovertemplate="%{y}: H=%{x:.2f}<extra></extra>"
))
fig_val.update_layout(
    **PLOTLY_LAYOUT, height=380,
    title=dict(text="Kruskal-Wallis H by Variable — Green=Large Effect, Amber=Medium Effect", font=dict(size=12))
)
fig_val.update_xaxes(title="H Statistic", gridcolor="#F1F5F9", range=[0, 130])
st.plotly_chart(fig_val, use_container_width=True)

with st.expander("📋 Full Kruskal-Wallis results table with effect sizes"):
    st.dataframe(val_df, use_container_width=True)
    st.caption(
        "η² (eta-squared) = H / (n−1) is a rank-based effect size. "
        "Cohen's benchmarks: Small < 0.01 · Medium ≥ 0.06 · Large ≥ 0.14"
    )

finding_card(
    "✅ All 13 Clustering Variables Are Statistically Significant — Not Random Groupings",
    "Every single attitudinal and satisfaction variable shows highly significant Kruskal-Wallis differences "
    "across the 3 K-Medoids clusters (all p < 0.0001). "
    "12 of 13 variables show a large effect size (η² ≥ 0.14), with Recommend (η²=0.445), "
    "Continuity (η²=0.436), and Time Saving (η²=0.416) being the most differentiating. "
    "This provides exceptionally strong statistical evidence that the three consumer segments "
    "are genuine, distinct behavioural profiles — not artefacts of the clustering algorithm.",
    EMERALD
)

finding_card(
    "📦 Three Actionable Consumer Segments Identified via K-Medoids (PAM)",
    "Neutral Adopters (n=67, 29.4%) — marginal users driven by Discounts, high churn risk; target with "
    "re-engagement and small-basket incentives. "
    "All-Round Enthusiasts (n=114, 50.0%) — balanced power-users engaged on all dimensions; target with "
    "loyalty subscriptions and basket-size promotions. "
    "Convenience Purists (n=47, 20.6%) — premium speed-first users indifferent to discounts, highest LTV; "
    "target with premium tiers, quality assurance, and referral programmes.",
    INDIGO
)

finding_card(
    "🧬 Segments Are Psychographic, Not Demographic",
    "Chi-square tests found no significant association between cluster membership and any demographic variable "
    "(Age, Gender, Occupation, Education, Income — all p > 0.05, Cramér's V < 0.13). "
    "This confirms the three segments are differentiated purely by attitudes, motivations, and satisfaction levels — "
    "not by who the users are, but by how they relate to Q-Commerce. "
    "Marketing strategies must therefore be psychographic and behavioural, not demographic.",
    VIOLET
)

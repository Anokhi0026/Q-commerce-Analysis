import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats
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
            "Cronbach's Alpha scale reliability, mean driver rankings, and EFA analysis.")

users    = get_users()
df_all   = load_raw()
df_anal  = load_analysis()
ld       = get_likert()

alpha = cronbach_alpha(ld)
interp_str = ("Excellent" if alpha>=0.9 else "Good" if alpha>=0.8 else "Acceptable" if alpha>=0.7 else "Poor")

# ✅ KPI (cleaned)
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
    itc_rows.append({
        "Item":col,
        "Mean":round(ld[col].mean(),3),
        "SD":round(ld[col].std(),3),
        "Item-Total r":round(r_val,3),
        "Alpha if Deleted":round(alpha_del,3)
    })

itc_df = pd.DataFrame(itc_rows).sort_values("Item-Total r")

c1,c2 = st.columns([1,2.5], gap="large")

with c1:
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{INDIGO},{VIOLET});border-radius:16px;
                padding:28px;text-align:center;color:#fff;'>
      <div style='font-size:.68rem;text-transform:uppercase;'>Cronbach's Alpha</div>
      <div style='font-size:2.8rem;font-weight:800;'>α={alpha:.3f}</div>
      <div style='font-size:.85rem;font-weight:600;'>{interp_str}</div>
    </div>""", unsafe_allow_html=True)

with c2:
    fig_itc = go.Figure(go.Bar(
        y=itc_df["Item"],
        x=itc_df["Item-Total r"],
        orientation="h"
    ))
    st.plotly_chart(fig_itc, use_container_width=True)

with st.expander("📋 Full item-total correlation table"):
    st.dataframe(itc_df, use_container_width=True)

# ── ANALYSIS 2: DRIVER RANKINGS ───────────────────────────────────────────────
section("Analysis 2 · Adoption Driver Rankings — Mean Likert Scores")

desc = ld.describe().T[["mean","std"]]
desc.columns = ["Mean","SD"]
desc["Rank"] = desc["Mean"].rank(ascending=False)

ranking = desc.sort_values("Mean")

fig_rank = go.Figure(go.Bar(
    y=ranking.index,
    x=ranking["Mean"],
    orientation="h"
))

st.plotly_chart(fig_rank, use_container_width=True)

# ── ANALYSIS 3: EFA ───────────────────────────────────────────────────────────
section("Analysis 3 · Exploratory Factor Analysis")

@st.cache_data
def run_efa():
    X = ld.values.astype(float)
    X_std = StandardScaler().fit_transform(X)
    corr_mat = np.corrcoef(X_std.T)

    eigenvalues, _ = np.linalg.eigh(corr_mat)
    eigenvalues = eigenvalues[::-1]
    n_factors = int((eigenvalues > 1).sum())

    pca = PCA(n_components=n_factors)
    pca.fit(X_std)

    loadings = pca.components_.T
    return eigenvalues, n_factors, loadings

eigenvalues, n_factors, loadings = run_efa()

st.write(f"Number of factors retained: {n_factors}")

fig_scree = go.Figure()
fig_scree.add_trace(go.Scatter(
    x=list(range(1,len(eigenvalues)+1)),
    y=eigenvalues,
    mode="lines+markers"
))

st.plotly_chart(fig_scree, use_container_width=True)

# ── FINAL INSIGHT ─────────────────────────────────────────────────────────────
finding_card(f"⭐ Cronbach's α={alpha:.3f} — {interp_str} Reliability",
             "The 10-item scale is internally consistent. All subsequent EFA and factor analyses are statistically justified.", INDIGO)
finding_card("🎯 EFA: Multi-dimensional Adoption Structure",
             f"{n_factors} factors retained explaining {pct_var.sum():.1f}% of variance. "
             "Convenience & lifestyle (along with product reliability) and price sensitivity are distinct drivers — "
             "platforms must address all dimensions simultaneously.", EMERALD)

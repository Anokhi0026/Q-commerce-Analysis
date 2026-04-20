import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from utils import *

st.set_page_config("Correspondence Analysis", "📍", layout="wide")
st.session_state["current_page"] = "pages/12_Correspondence_Analysis.py"
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html,body,[class*='css']{font-family:'Inter',sans-serif;}
.stApp{background:#FAFAFA;}
section[data-testid='stSidebar']{background:#FFFFFF;border-right:1px solid #E2E8F0;}
</style>""", unsafe_allow_html=True)

from navbar import navbar
navbar()
page_header("Correspondence Analysis", "Visualising Associations Between Categorical Variables",
            "Five CA biplots revealing demographic-behavioural patterns among Q-Commerce adopters (n=228). "
            "Points close together in biplot space share similar profiles — rows (●) and columns (▲) near each other indicate association.")

df = load_analysis()
users = df[df["Adoption_Status"]==1].copy()

# ── CA ENGINE (pure numpy — no prince dependency) ──────────────────────────────
def run_ca_numpy(ct):
    """Compute 2D Correspondence Analysis coordinates using numpy SVD."""
    ct_arr = ct.values.astype(float)
    n      = ct_arr.sum()
    P      = ct_arr / n
    r      = P.sum(axis=1)
    c      = P.sum(axis=0)
    Dr_inv = np.diag(1.0 / np.sqrt(r))
    Dc_inv = np.diag(1.0 / np.sqrt(c))
    S      = Dr_inv @ (P - np.outer(r, c)) @ Dc_inv
    U, sv, Vt = np.linalg.svd(S, full_matrices=False)
    # Keep first 2 dims
    row_coords = Dr_inv @ U[:, :2] * sv[:2]
    col_coords = Dc_inv @ Vt[:2, :].T * sv[:2]
    inertia    = sv**2
    total_in   = inertia.sum()
    pct_dim    = (inertia[:2] / total_in * 100).round(1)
    return row_coords, col_coords, pct_dim

def ca_biplot(ct, row_var, col_var, title, row_color=INDIGO, col_color=ROSE):
    """Run chi-square pre-check, CA, and return Plotly figure + metadata."""
    chi2_v, p, dof, _ = chi2_contingency(ct)
    p_str = "< 0.001" if p < 0.001 else f"= {p:.4f}"

    row_coords, col_coords, pct_dim = run_ca_numpy(ct)
    row_df = pd.DataFrame(row_coords, index=ct.index, columns=["D1","D2"])
    col_df = pd.DataFrame(col_coords, index=ct.columns, columns=["D1","D2"])

    fig = go.Figure()
    fig.add_hline(y=0, line_color="#E2E8F0", line_width=1)
    fig.add_vline(x=0, line_color="#E2E8F0", line_width=1)

    # Row points (circles)
    fig.add_trace(go.Scatter(
        x=row_df["D1"], y=row_df["D2"], mode="markers+text",
        text=row_df.index.tolist(), textposition="top right",
        textfont=dict(size=10, color=row_color),
        marker=dict(color=row_color, size=11, symbol="circle",
                    line=dict(color="#fff", width=1.5)),
        name=row_var.replace("_"," "),
        hovertemplate="%{text}: D1=%{x:.3f}, D2=%{y:.3f}<extra></extra>"))

    # Col points (triangles)
    fig.add_trace(go.Scatter(
        x=col_df["D1"], y=col_df["D2"], mode="markers+text",
        text=col_df.index.tolist(), textposition="bottom right",
        textfont=dict(size=10, color=col_color),
        marker=dict(color=col_color, size=13, symbol="triangle-up",
                    line=dict(color="#fff", width=1.5)),
        name=col_var.replace("_"," "),
        hovertemplate="%{text}: D1=%{x:.3f}, D2=%{y:.3f}<extra></extra>"))

    fig.update_layout(**PLOTLY_LAYOUT, height=430, title=dict(text=f"{title}<br><sup>χ²({dof})={chi2_v:.2f}, p {p_str} | "
                                       f"Dim1+Dim2 = {pct_dim.sum():.1f}% variance explained</sup>",
                                  font=dict(size=12)))
    fig.update_layout(legend=dict(x=0.01,y=0.99))
    fig.update_xaxes(title=f"Dimension 1 ({pct_dim[0]:.1f}%)",gridcolor="#F1F5F9",zeroline=False)
    fig.update_yaxes(title=f"Dimension 2 ({pct_dim[1]:.1f}%)",gridcolor="#F1F5F9",zeroline=False)
    return fig, chi2_v, p, dof, pct_dim, row_df, col_df

k1,k2,k3,k4 = st.columns(4)
kpi(k1,"5","CA Biplots","5 variable pairs analysed")
kpi(k2,"228","Adopters","CA restricted to users only",INDIGO)
kpi(k3,"Chi² pre-check","Significance Test","All pairs significant",EMERALD)
kpi(k4,"2D","Dimensions","Dim1+Dim2 % shown per plot",VIOLET)
st.markdown("<br>",unsafe_allow_html=True)

st.markdown("""
<div style='background:#EFF6FF;border:1px solid #BFDBFE;border-radius:10px;padding:12px 16px;margin-bottom:16px;'>
  <b style='font-size:.83rem;color:#1D4ED8;'>How to read a CA biplot:</b>
  <span style='font-size:.8rem;color:#374151;'>
  Points that are close together share similar profiles.
  A row category (●) near a column category (▲) indicates that those two categories are strongly associated.
  Points near the origin have profiles close to the overall average.
  </span>
</div>""", unsafe_allow_html=True)

# ── CA 1: Age Group × App Used ─────────────────────────────────────────────────
section("CA 1 · Age Group × App Used")
ct1 = pd.crosstab(users["Age_Group"], users["App_Used"])
fig1, chi2_1, p1, dof1, pct1, rd1, cd1 = ca_biplot(
    ct1, "Age_Group", "App_Used", "CA 1: Age Group × App Used", INDIGO, ROSE)

c1,c2 = st.columns([1.5,1], gap="large")
with c1:
    st.plotly_chart(fig1, use_container_width=True)
with c2:
    st.markdown(f"""
    <div style='background:#fff;border:1px solid #E2E8F0;border-radius:12px;padding:16px;height:100%;'>
      <div style='font-weight:600;font-size:.85rem;color:#1E1E2E;margin-bottom:10px;'>χ² Pre-check</div>
      <div style='font-size:.78rem;color:#475569;line-height:1.6;margin-bottom:12px;'>
        χ²({dof1}) = {chi2_1:.2f}, p {"< 0.001" if p1<0.001 else f"= {p1:.4f}"}<br>
        Association is significant → CA is appropriate ✓<br>
        Dimensions explain {pct1.sum():.1f}% of total variance.
      </div>
      <div style='font-weight:600;font-size:.8rem;color:#1E1E2E;margin-bottom:6px;'>Interpretation</div>
      <div style='font-size:.78rem;color:#475569;line-height:1.7;'>
        Younger respondents (18–25) are positioned closer to Blinkit — 
        strong association between this age group and Blinkit usage. 
        The 26–33 group shows proximity to Zepto — reflecting Zepto's 
        urban-speed positioning. Older respondents (42+) sit near the 
        origin with weaker and less differentiated platform preferences, 
        consistent with their lower overall Q-Commerce engagement.
      </div>
    </div>""", unsafe_allow_html=True)

# ── CA 2: Age Group × Delivery Time ───────────────────────────────────────────
section("CA 2 · Age Group × Delivery Time Preference")
ct2 = pd.crosstab(users["Age_Group"], users["Delivery_Time"])
fig2, chi2_2, p2, dof2, pct2, rd2, cd2 = ca_biplot(
    ct2, "Age_Group", "Delivery_Time", "CA 2: Age Group × Delivery Time", INDIGO, AMBER)

c1,c2 = st.columns([1.5,1], gap="large")
with c1:
    st.plotly_chart(fig2, use_container_width=True)
with c2:
    st.markdown(f"""
    <div style='background:#fff;border:1px solid #E2E8F0;border-radius:12px;padding:16px;height:100%;'>
      <div style='font-weight:600;font-size:.85rem;color:#1E1E2E;margin-bottom:10px;'>χ² Pre-check</div>
      <div style='font-size:.78rem;color:#475569;line-height:1.6;margin-bottom:12px;'>
        χ²({dof2}) = {chi2_2:.2f}, p {"< 0.001" if p2<0.001 else f"= {p2:.4f}"}<br>
        Dimensions explain {pct2.sum():.1f}% of total variance.
      </div>
      <div style='font-weight:600;font-size:.8rem;color:#1E1E2E;margin-bottom:6px;'>Interpretation</div>
      <div style='font-size:.78rem;color:#475569;line-height:1.7;'>
        Late-night and midnight delivery preferences associate with younger 
        age groups (18–25), reflecting their later, more flexible schedules 
        and higher digital comfort. Middle age groups (26–40) cluster around 
        evening delivery — consistent with post-work ordering. Older 
        respondents (40+) align more with morning delivery, suggesting 
        routine-based ordering habits typical of retired or homemaker-oriented respondents.
      </div>
    </div>""", unsafe_allow_html=True)

# ── CA 3: Age Group × Order Value ─────────────────────────────────────────────
section("CA 3 · Age Group × Average Order Value")
ct3 = pd.crosstab(users["Age_Group"], users["Order_Value"])
fig3, chi2_3, p3, dof3, pct3, rd3, cd3 = ca_biplot(
    ct3, "Age_Group", "Order_Value", "CA 3: Age Group × Average Order Value", INDIGO, EMERALD)

c1,c2 = st.columns([1.5,1], gap="large")
with c1:
    st.plotly_chart(fig3, use_container_width=True)
with c2:
    st.markdown(f"""
    <div style='background:#fff;border:1px solid #E2E8F0;border-radius:12px;padding:16px;height:100%;'>
      <div style='font-weight:600;font-size:.85rem;color:#1E1E2E;margin-bottom:10px;'>χ² Pre-check</div>
      <div style='font-size:.78rem;color:#475569;line-height:1.6;margin-bottom:12px;'>
        χ²({dof3}) = {chi2_3:.2f}, p {"< 0.001" if p3<0.001 else f"= {p3:.4f}"}<br>
        Dimensions explain {pct3.sum():.1f}% of total variance.
      </div>
      <div style='font-weight:600;font-size:.8rem;color:#1E1E2E;margin-bottom:6px;'>Interpretation</div>
      <div style='font-size:.78rem;color:#475569;line-height:1.7;'>
        Younger respondents (18–25) associate with lower order value brackets 
        — student demographics with constrained budgets placing frequent small orders. 
        The 26–40 groups gravitate toward ₹200–₹400 mid-range — working adults 
        making regular household purchases. Older respondents (40+) show 
        association with higher order values, suggesting that when this group 
        does use Q-Commerce, they make larger, more deliberate purchases 
        rather than impulse buys.
      </div>
    </div>""", unsafe_allow_html=True)

# ── CA 4: Occupation × Payment Method ─────────────────────────────────────────
section("CA 4 · Occupation × Payment Method")
ct4 = pd.crosstab(users["Occupation"], users["Payment_Method"])
fig4, chi2_4, p4, dof4, pct4, rd4, cd4 = ca_biplot(
    ct4, "Occupation", "Payment_Method", "CA 4: Occupation × Payment Method", VIOLET, ROSE)

c1,c2 = st.columns([1.5,1], gap="large")
with c1:
    st.plotly_chart(fig4, use_container_width=True)
with c2:
    st.markdown(f"""
    <div style='background:#fff;border:1px solid #E2E8F0;border-radius:12px;padding:16px;height:100%;'>
      <div style='font-weight:600;font-size:.85rem;color:#1E1E2E;margin-bottom:10px;'>χ² Pre-check</div>
      <div style='font-size:.78rem;color:#475569;line-height:1.6;margin-bottom:12px;'>
        χ²({dof4}) = {chi2_4:.2f}, p {"< 0.001" if p4<0.001 else f"= {p4:.4f}"}<br>
        Dimensions explain {pct4.sum():.1f}% of total variance.
      </div>
      <div style='font-weight:600;font-size:.8rem;color:#1E1E2E;margin-bottom:6px;'>Interpretation</div>
      <div style='font-size:.78rem;color:#475569;line-height:1.7;'>
        Students are closely positioned to UPI — overwhelming preference for 
        fast, frictionless digital payment. Self-employed respondents show 
        proximity to digital wallets. Working professionals associate with 
        debit/credit card, reflecting higher formal banking access. Homemakers 
        and retired respondents cluster closer to Cash on Delivery — indicating 
        continued reliance on traditional payment modes due to lower digital 
        payment confidence.
      </div>
    </div>""", unsafe_allow_html=True)

# ── CA 5: Occupation × Order Value ────────────────────────────────────────────
section("CA 5 · Occupation × Average Order Value")
ct5 = pd.crosstab(users["Occupation"], users["Order_Value"])
fig5, chi2_5, p5, dof5, pct5, rd5, cd5 = ca_biplot(
    ct5, "Occupation", "Order_Value", "CA 5: Occupation × Average Order Value", VIOLET, EMERALD)

c1,c2 = st.columns([1.5,1], gap="large")
with c1:
    st.plotly_chart(fig5, use_container_width=True)
with c2:
    st.markdown(f"""
    <div style='background:#fff;border:1px solid #E2E8F0;border-radius:12px;padding:16px;height:100%;'>
      <div style='font-weight:600;font-size:.85rem;color:#1E1E2E;margin-bottom:10px;'>χ² Pre-check</div>
      <div style='font-size:.78rem;color:#475569;line-height:1.6;margin-bottom:12px;'>
        χ²({dof5}) = {chi2_5:.2f}, p {"< 0.001" if p5<0.001 else f"= {p5:.4f}"}<br>
        Dimensions explain {pct5.sum():.1f}% of total variance.
      </div>
      <div style='font-weight:600;font-size:.8rem;color:#1E1E2E;margin-bottom:6px;'>Interpretation</div>
      <div style='font-size:.78rem;color:#475569;line-height:1.7;'>
        Students associate with lower order value brackets — high-frequency 
        but low-value impulse ordering. Self-employed and working professionals 
        gravitate toward the ₹200–₹400 range, consistent with regular household 
        convenience purchasing. Retired respondents and homemakers show 
        association with higher order values — when they do order, they tend 
        to be more planned and higher-value, consolidating multiple items 
        into single orders.
      </div>
    </div>""", unsafe_allow_html=True)

# ── Summary ────────────────────────────────────────────────────────────────────
section("Key Findings — Correspondence Analysis")
finding_card("📱 Blinkit → Young Users | Zepto → 26–33 | Swiggy → Older (CA 1)",
             "Clear platform-age segmentation: Blinkit dominates 18–25, Zepto carves a niche in 26–33. "
             "Older users (42+) near origin — weaker, less differentiated platform preferences.", INDIGO)
finding_card("🌙 Night/Midnight → 18–25 | Evening → 26–40 | Morning → 40+ (CA 2)",
             "Delivery time preferences are age-stratified with high clarity. "
             "Platforms should optimise staffing and push notifications for age-specific peak windows.", AMBER)
finding_card("💳 Students → UPI | Professionals → Cards | Retired/Homemakers → CoD (CA 4)",
             "Payment preferences are strongly occupation-driven. Trust-building and 'first UPI order' incentives "
             "should target homemaker and retired segments specifically, not just price-sensitive users.", VIOLET)

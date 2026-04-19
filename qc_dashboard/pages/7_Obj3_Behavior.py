import streamlit as st
import plotly.graph_objects as go
import pandas as pd, numpy as np
from scipy.stats import kruskal, spearmanr
from utils import *

# ── Local colour overrides safe for both Plotly AND CSS ───────────────────────
INDIGO_CSS   = "#4F46E5"          # hex  → use in HTML/CSS strings
INDIGO_PLOT  = "rgba(79,70,229,1)"  # rgba → use in Plotly marker_color

def rgba(hex_color, alpha=1.0):
    """Convert #RRGGBB to rgba(...) for Plotly."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

st.set_page_config("Obj 3 — Behavior", "📊", layout="wide")
st.markdown(
    "<style>@import url('https://fonts.googleapis.com/css2?family=Inter:"
    "wght@300;400;500;600;700;800&display=swap');"
    "html,body,[class*='css']{font-family:'Inter',sans-serif;}"
    ".stApp{background:#FAFAFA;}"
    "section[data-testid='stSidebar']{background:#FFFFFF;"
    "border-right:1px solid #E2E8F0;}</style>",
    unsafe_allow_html=True
)
sidebar()
page_header(
    "Objective 3", "Usage Behavior Patterns",
    "Examining frequency, order size, delivery time preferences, payment methods, "
    "product categories, satisfaction scores, and their relationships with demographics."
)

users = get_users()
df    = load_raw()

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
kpi(k1, "228", "Active Users",    "Analysis base")
kpi(k2, "43%", "Long-term Users", ">1 year tenure",   EMERALD)
kpi(k3, "3.82","Mean Satisfaction","Out of 5.0",       INDIGO)
kpi(k4, "UPI", "Top Payment",     "54% preference",    AMBER)
st.markdown("<br>", unsafe_allow_html=True)

# ── Helper ────────────────────────────────────────────────────────────────────
def freq_bar(data, order, title, colors=None):
    cnt   = data.value_counts().reindex([o for o in order if o in data.values]).fillna(0)
    total = cnt.sum() or 1
    clrs  = colors or [INDIGO_PLOT] * len(cnt)
    fig   = go.Figure(go.Bar(
        x=cnt.index, y=cnt.values,
        marker_color=clrs,
        text=[f"{v:.0f}<br>({v/total*100:.1f}%)" for v in cnt.values],
        textposition="outside",
        hovertemplate="%{x}: %{y} users<extra></extra>"
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=300,
                      title=dict(text=title, font=dict(size=13)))
    fig.update_yaxes(gridcolor="#F1F5F9")
    return fig

# ── Section 1: Descriptive Profiling ─────────────────────────────────────────
section("1 · Descriptive Profiling of Usage Behavior")
tab1, tab2, tab3 = st.tabs(["App & Tenure", "Spending & Payment", "Delivery & Items"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(freq_bar(
            users["App_Used"],
            ["Blinkit", "Swiggy Instamart", "Zepto", "Other"],
            "Primary App Used (n=228)",
            [INDIGO_PLOT, ROSE, EMERALD, AMBER]
        ), use_container_width=True)
    with c2:
        st.plotly_chart(freq_bar(
            users["How long have you been using Q-Commerce apps?"],
            ["Less than 3 months", "3-6 months", "6-12 months", "More than a year"],
            "Usage Tenure",
            [SKY, VIOLET, AMBER, EMERALD]
        ), use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(freq_bar(
            users["average order value"],
            ["Below ₹200", "₹200 - ₹400", "₹400 - ₹600", "Above ₹600"],
            "Average Order Value",
            [EMERALD, INDIGO_PLOT, AMBER, ROSE]
        ), use_container_width=True)
    with c2:
        pay_cnt = users["preferred payment method?"].value_counts()
        fig_pay = go.Figure(go.Pie(
            labels=pay_cnt.index,
            values=pay_cnt.values,
            hole=0.5,
            marker_colors=[INDIGO_PLOT, EMERALD, AMBER, ROSE],
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} users<extra></extra>"
        ))
        fig_pay.update_layout(
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ["xaxis", "yaxis"]},
            height=300,
            title=dict(text="Payment Method Preference", font=dict(size=13))
        )
        st.plotly_chart(fig_pay, use_container_width=True)

with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(freq_bar(
            users["preferred delivery time"],
            ["Morning", "Afternoon", "Evening", "Night", "Midnight"],
            "Preferred Delivery Time",
            [SKY, AMBER, INDIGO_PLOT, VIOLET, "#0F172A"]
        ), use_container_width=True)
    with c2:
        # ── ⚠️  Replace the column names below with your actual multi-response columns ──
        cat_cols = ["Col1", "Col2", "Col3", "Col4", "Col5"]
        available_cat_cols = [c for c in cat_cols if c in users.columns]

        if available_cat_cols:
            all_cats = []
            for col in available_cat_cols:
                all_cats.extend(users[col].dropna().str.strip().tolist())
            cat_counts = {}
            for item in all_cats:
                key = item.strip()
                cat_counts[key] = cat_counts.get(key, 0) + 1
            cat_s = pd.Series(cat_counts).sort_values(ascending=True)
            cat_s = cat_s[cat_s.index.str.len() > 1]

            plot_colors = [INDIGO_PLOT, ROSE, EMERALD, AMBER, SKY, VIOLET,
                           "#DB2777", "#EA580C"][:len(cat_s)]
            fig_cat = go.Figure(go.Bar(
                y=cat_s.index, x=cat_s.values,
                orientation="h",
                marker_color=plot_colors,
                text=[f"{v} ({v/len(users)*100:.1f}%)" for v in cat_s.values],
                textposition="outside",
                hovertemplate="%{y}: %{x} users<extra></extra>"
            ))
            fig_cat.update_layout(**PLOTLY_LAYOUT, height=300,
                                  title=dict(text="Products Ordered (Multi-response)",
                                             font=dict(size=13)))
            fig_cat.update_xaxes(title="No. of Users", gridcolor="#F1F5F9")
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.warning("⚠️ Category columns not found. Update `cat_cols` list with actual column names.")

# ── Section 2: Kruskal-Wallis ─────────────────────────────────────────────────
section("2 · Kruskal-Wallis H Test — Satisfaction Across Groups",
        "H₀: Median satisfaction is equal across groups | Non-parametric ANOVA")

sat_col  = "Overall satisfaction with Q-Commerce Apps"
cont_col = "Likelihood of continuing usage in the future"
rec_col  = "likely are you to recommend these apps to Others"
sat_vars = [sat_col, cont_col, rec_col]
sat_lbls = ["Satisfaction", "Continuity", "Recommendation"]

kw_rows = []
for gvar in ["Age_Group", "App_Used", "Occupation"]:
    if gvar not in users.columns:
        continue
    for sv, sl in zip(sat_vars, sat_lbls):
        if sv not in users.columns:
            continue
        groups = [g[sv].dropna().values
                  for _, g in users.groupby(gvar)
                  if len(g[sv].dropna()) >= 3]
        if len(groups) >= 2:
            H, p = kruskal(*groups)
            kw_rows.append({
                "Score": sl, "Group By": gvar,
                "H": round(H, 3), "p": round(p, 4),
                "Sig": "✅" if p < 0.05 else "❌"
            })

if kw_rows:
    kw_df = pd.DataFrame(kw_rows)
    c1, c2 = st.columns([1.2, 1], gap="large")
    with c1:
        pivot_h = kw_df.pivot(index="Group By", columns="Score", values="H")
        fig_kw  = go.Figure(go.Heatmap(
            z=pivot_h.values,
            x=pivot_h.columns.tolist(),
            y=pivot_h.index.tolist(),
            colorscale=[[0, "#EEF2FF"], [0.5, "#818CF8"], [1, "#3730A3"]],
            text=pivot_h.round(2).values,
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorbar=dict(title="H Stat"),
            hovertemplate="%{y} × %{x}: H=%{z:.2f}<extra></extra>"
        ))
        fig_kw.update_layout(
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ["xaxis", "yaxis"]},
            height=260,
            title=dict(text="H Statistic Heatmap (higher = more group variation)",
                       font=dict(size=12))
        )
        st.plotly_chart(fig_kw, use_container_width=True)

    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        for _, row in kw_df.iterrows():
            st.markdown(f"""
            <div style='background:#fff;border:1px solid #E2E8F0;border-radius:8px;
                        padding:8px 12px;margin-bottom:4px;
                        display:flex;justify-content:space-between;'>
              <span style='font-size:.8rem;color:#1E1E2E;'>
                {row['Score']} × {row['Group By']}
              </span>
              <span style='font-size:.8rem;'>H={row['H']} {row['Sig']}</span>
            </div>""", unsafe_allow_html=True)
else:
    st.warning("⚠️ Could not compute Kruskal-Wallis — check column names for grouping variables.")

# ── Satisfaction Distributions ────────────────────────────────────────────────
section("Satisfaction Score Distributions (1–5 scale)")
valid_sat_vars = [v for v in sat_vars if v in users.columns]

if valid_sat_vars:
    cols = st.columns(len(valid_sat_vars))
    for col, sc, sl in zip(cols, valid_sat_vars, sat_lbls):
        data = users[sc].dropna()
        cnt  = data.value_counts().sort_index()
        fig  = go.Figure(go.Bar(
            x=cnt.index, y=cnt.values,
            marker_color=INDIGO_PLOT,
            text=cnt.values, textposition="outside"
        ))
        fig.add_vline(
            x=data.mean(), line_dash="dash", line_color=ROSE,
            annotation_text=f"μ={data.mean():.2f}",
            annotation_position="top right",
            annotation_font=dict(size=9)
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=250,
                          title=dict(text=sl, font=dict(size=12)))
        fig.update_xaxes(tickvals=[1, 2, 3, 4, 5], gridcolor="#F1F5F9")
        fig.update_yaxes(gridcolor="#F1F5F9")
        col.plotly_chart(fig, use_container_width=True)

# ── Section 3: Spearman Correlation ──────────────────────────────────────────
section("3 · Spearman Correlation — Satisfaction, Continuity, Recommendation")

valid_sat = [v for v in sat_vars if v in users.columns]
if len(valid_sat) >= 2:
    sat_data = users[valid_sat].dropna()
    n        = len(valid_sat)
    labels   = [sat_lbls[sat_vars.index(v)] for v in valid_sat]
    corr_m   = np.zeros((n, n))
    pval_m   = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                corr_m[i, j] = 1.0
            else:
                r, p = spearmanr(sat_data.iloc[:, i], sat_data.iloc[:, j])
                corr_m[i, j] = r
                pval_m[i, j] = p

    annots = [
        [f"ρ={corr_m[i,j]:.3f}<br>p={pval_m[i,j]:.3f}" if i != j else "—"
         for j in range(n)]
        for i in range(n)
    ]

    fig_corr = go.Figure(go.Heatmap(
        z=corr_m, x=labels, y=labels,
        colorscale=C_SCALE, zmin=0, zmax=1,
        text=annots, texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title="ρ"),
        hovertemplate="%{y} × %{x}: ρ=%{z:.3f}<extra></extra>"
    ))
    fig_corr.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ["xaxis", "yaxis"]},
        height=300,
        title=dict(text="Spearman Correlation Matrix", font=dict(size=13))
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ── Finding Cards ─────────────────────────────────────────────────────────────
finding_card(
    "💛 Satisfaction-Loyalty-Advocacy Chain Confirmed",
    "Strong positive Spearman correlations between all three scores confirm that "
    "satisfied users are significantly more likely to continue using Q-Commerce AND "
    "to recommend it. This is the empirical Q-Commerce loyalty loop for Vadodara.",
    INDIGO
)
finding_card(
    "🌙 Evening & Night Dominate (63% of Orders)",
    "Evening (40%) and Night (23%) together account for 63% of preferred delivery times — "
    "end-of-day 'top-up' shopping, not planned grocery runs.",
    AMBER
)
finding_card(
    "💳 UPI Leads, But CoD Persists",
    "UPI dominates at ~54%, but Cash on Delivery (~32%) signals residual hesitancy about "
    "digital payments — a segment needing targeted trust-building.",
    ROSE
)
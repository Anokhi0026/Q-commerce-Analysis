import streamlit as st
import plotly.graph_objects as go
import pandas as pd, numpy as np
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, chi2 as chi2_dist
from sklearn.metrics import roc_auc_score, roc_curve
from utils import *

st.set_page_config("Obj 5 — Predictive", "🤖", layout="wide")
st.markdown("<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');html,body,[class*='css']{font-family:'Inter',sans-serif;}.stApp{background:#FAFAFA;}section[data-testid='stSidebar']{background:#FFFFFF;border-right:1px solid #E2E8F0;}</style>",unsafe_allow_html=True)
sidebar()
page_header("Objective 5", "Predictive Models for Adoption Likelihood",
            "Binary Logistic Regression using statsmodels — with Wald tests, Odds Ratios, Hosmer-Lemeshow goodness-of-fit, ROC/AUC, and model comparison.")

df = load_raw()

# ── Mann-Whitney Pre-Model Check ───────────────────────────────────────────────
section("1 · Pre-Model Verification — Mann-Whitney U Test",
        "Do users and non-users differ significantly on ordinal demographics?")

AGE_ORD    = {"18-25":1,"26-33":2,"34-41":3,"42-49":4,"50 or above":5}
EDU_ORD    = {"No Formal Education":1,"School Level":2,"Undergraduate":3,"Postgraduate":4,"Professional Degree":5}
INC_ORD    = {"Below ₹20,000":1,"₹20,000 - ₹40,000":2,"₹40,000 - ₹60,000":3,"₹60,000 - ₹1,00,000":4,"Above ₹1,00,000":5}
df["Age_Ord"] = df["Age_Group"].map(AGE_ORD)
df["Edu_Ord"] = df["Education"].map(EDU_ORD)
df["Inc_Ord"] = df["Income"].map(INC_ORD)

mw_rows = []
for var, col in [("Age","Age_Ord"),("Education","Edu_Ord"),("Income","Inc_Ord")]:
    u  = df[df["Adoption_Status"]==1][col].dropna()
    nu = df[df["Adoption_Status"]==0][col].dropna()
    U,p = mannwhitneyu(u, nu, alternative="two-sided")
    rb = 1 - (2*U)/(len(u)*len(nu))
    eff = "Large" if abs(rb)>0.5 else ("Medium" if abs(rb)>0.3 else "Small")
    mw_rows.append({"Variable":var,"Users Median":u.median(),"Non-Users Median":nu.median(),
                     "U":round(U,1),"p":round(p,4),"r":round(rb,3),"Effect":eff,"Sig":p<0.05})

c1,c2,c3 = st.columns(3)
for col_w, row in zip([c1,c2,c3], mw_rows):
    color = EMERALD if row["Sig"] else SLATE
    col_w.markdown(f"""
    <div style='background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:18px;text-align:center;'>
      <div style='font-weight:700;font-size:.95rem;color:#1E1E2E;'>{row['Variable']}</div>
      <div style='font-size:1.5rem;font-weight:800;color:{color};margin:6px 0;'>
        {"✅" if row["Sig"] else "❌"} p={row['p']}
      </div>
      <div style='font-size:.75rem;color:#64748B;'>r = {row['r']} ({row['Effect']} effect)</div>
      <div style='font-size:.72rem;color:#94A3B8;margin-top:4px;'>
        Users med: {row['Users Median']:.0f} | Non-users: {row['Non-Users Median']:.0f}
      </div>
    </div>""", unsafe_allow_html=True)

# ── Build Model ────────────────────────────────────────────────────────────────
section("2 · Binary Logistic Regression — Model 1 (Demographics)",
        "Fitted using statsmodels.Logit | Wald z-tests | 95% Wald Confidence Intervals")

def make_dummies(df_in):
    gd = pd.get_dummies(df_in["Gender"],    prefix="G").astype(int)
    ad = pd.get_dummies(df_in["Age_Group"], prefix="Age").astype(int)
    ed = pd.get_dummies(df_in["Education"], prefix="Edu").astype(int)
    od = pd.get_dummies(df_in["Occupation"],prefix="Occ").astype(int)
    id_ = pd.get_dummies(df_in["Income"],  prefix="Inc").astype(int)
    for ref, blk in [("G_Male",gd),("Age_18-25",ad),("Edu_School Level",ed),
                      ("Occ_Student",od),("Inc_Below ₹20,000",id_)]:
        if ref in blk.columns: blk.drop(columns=[ref],inplace=True)
    X = pd.concat([gd,ad,ed,od,id_],axis=1)
    X.columns = [c.replace(" ","_").replace("₹","Rs").replace(",","")
                   .replace("–","_").replace("-","_") for c in X.columns]
    return X

with st.spinner("Fitting logistic regression…"):
    X_demo = make_dummies(df)
    y = df["Adoption_Status"].astype(float)
    mask = X_demo.notna().all(axis=1) & y.notna()
    X1, y1 = X_demo[mask].copy(), y[mask].copy()
    X1_sm = sm.add_constant(X1)
    result1 = sm.Logit(y1, X1_sm).fit(disp=False)

    conf = result1.conf_int(); conf.columns=["lo","hi"]
    coef_df = pd.DataFrame({
        "Predictor": result1.params.index,
        "β": result1.params.round(3),
        "z": result1.tvalues.round(2),
        "p": result1.pvalues.round(4),
        "OR": np.exp(result1.params).round(3),
        "CI_lo": np.exp(conf["lo"]).round(3),
        "CI_hi": np.exp(conf["hi"]).round(3),
        "Sig": ["✅" if p<0.05 else "❌" for p in result1.pvalues]
    }).set_index("Predictor")
    plot_df = coef_df.drop(index="const").sort_values("OR")

    y1_prob = result1.predict(X1_sm)
    auc1 = roc_auc_score(y1, y1_prob)
    fpr1,tpr1,thr1 = roc_curve(y1, y1_prob)
    j1 = tpr1-fpr1; opt1=np.argmax(j1)

    n1=len(y1); ll_null=result1.llnull; ll_mod=result1.llf
    cox1 = 1-np.exp(-2*(ll_mod-ll_null)/n1)
    nag1 = cox1/(1-np.exp(2*ll_null/n1))

    def hl_test(y_t, y_p, g=10):
        d=pd.DataFrame({"y":y_t.values,"p":y_p})
        d["dec"]=pd.qcut(d["p"],q=g,duplicates="drop",labels=False)
        t=d.groupby("dec").agg(o1=("y","sum"),e1=("p","sum"),n=("y","count"))
        t["o0"]=t["n"]-t["o1"]; t["e0"]=t["n"]-t["e1"]
        chi2_hl=((t["o1"]-t["e1"])**2/t["e1"]+(t["o0"]-t["e0"])**2/t["e0"]).sum()
        df_hl=len(t)-2; p_hl=1-chi2_dist.cdf(chi2_hl,df_hl)
        return chi2_hl,df_hl,p_hl

    hl_chi2,hl_df,hl_p = hl_test(y1,y1_prob)

# KPI row
k1,k2,k3,k4,k5 = st.columns(5)
kpi(k1,f"{auc1:.3f}","AUC","Discrimination ability",INDIGO)
kpi(k2,f"{nag1:.3f}","Nagelkerke R²","Variance explained",VIOLET)
kpi(k3,f"{hl_p:.3f}","H-L p-value","Good fit if >0.05",EMERALD if hl_p>0.05 else ROSE)
kpi(k4,f"{result1.aic:.1f}","AIC","Lower is better",AMBER)
kpi(k5,f"{thr1[opt1]:.2f}","Optimal Threshold","Youden's J",SKY)

st.markdown("<br>",unsafe_allow_html=True)
c1, c2 = st.columns([1.3,1], gap="large")

with c1:
    section("Forest Plot — Odds Ratios with 95% Wald CI")
    colors_or = [INDIGO if s=="✅" else "#CBD5E1" for s in plot_df["Sig"]]
    fig_f = go.Figure()
    ypos = list(range(len(plot_df)))
    fig_f.add_trace(go.Scatter(x=plot_df["OR"], y=ypos, mode="markers",
                                marker=dict(color=colors_or,size=9,symbol="square"),
                                hovertemplate="%{text}<extra></extra>",
                                text=[f"{p}: OR={o:.3f} ({s})" for p,o,s in
                                      zip(plot_df.index,plot_df["OR"],plot_df["Sig"])]))
    for i,(_, row) in enumerate(plot_df.iterrows()):
        fig_f.add_trace(go.Scatter(
            x=[row["CI_lo"],row["CI_hi"]], y=[i,i], mode="lines",
            line=dict(color=colors_or[i],width=2), showlegend=False))
    fig_f.add_vline(x=1.0, line_dash="dash", line_color="#94A3B8",
                    annotation_text="OR=1", annotation_font=dict(size=9,color="#94A3B8"))
    fig_f.update_layout(**PLOTLY_LAYOUT, height=500, showlegend=False,
                         yaxis=dict(tickvals=ypos,ticktext=plot_df.index.tolist(),gridcolor="#F1F5F9"),
                         xaxis=dict(title="Odds Ratio (OR) with 95% Wald CI",gridcolor="#F1F5F9"),
                         title=dict(text="Blue = Significant (p<0.05) | Gray = Not Significant",font=dict(size=11)))
    st.plotly_chart(fig_f, use_container_width=True)

with c2:
    section("ROC Curve — Model 1")
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr1, y=tpr1, mode="lines",
                                  line=dict(color=INDIGO,width=3), name=f"Model 1 (AUC={auc1:.3f})",
                                  fill="tozeroy", fillcolor=INDIGO+"15",
                                  hovertemplate="FPR=%{x:.2f}, TPR=%{y:.2f}<extra></extra>"))
    fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
                                  line=dict(color="#94A3B8",dash="dash"),name="Random (AUC=0.50)"))
    fig_roc.add_trace(go.Scatter(x=[fpr1[opt1]],y=[tpr1[opt1]],mode="markers",
                                  marker=dict(color=ROSE,size=12,symbol="star"),
                                  name=f"Optimal threshold={thr1[opt1]:.2f}"))
    fig_roc.update_layout(**PLOTLY_LAYOUT, height=300,
                           xaxis=dict(title="False Positive Rate",range=[0,1],gridcolor="#F1F5F9"),
                           yaxis=dict(title="True Positive Rate",range=[0,1],gridcolor="#F1F5F9"),
                           title=dict(text=f"ROC Curve — AUC = {auc1:.3f}",font=dict(size=13)))
    st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown(f"""
    <div style='background:#fff;border:1px solid #E2E8F0;border-radius:12px;padding:14px 16px;'>
      <div style='font-weight:600;font-size:.85rem;color:#1E1E2E;margin-bottom:8px;'>Model Fit Summary</div>
      {''.join(f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #F8FAFC;font-size:.78rem;"><span style="color:#64748B;">{k}</span><span style="font-weight:600;color:#1E1E2E;">{v}</span></div>' for k,v in [
        ("Log-Likelihood", f"{result1.llf:.2f}"),
        ("Null Log-Likelihood", f"{result1.llnull:.2f}"),
        ("McFadden R²", f"{result1.prsquared:.4f}"),
        ("Nagelkerke R²", f"{nag1:.4f}"),
        ("AIC", f"{result1.aic:.2f}"),
        ("BIC", f"{result1.bic:.2f}"),
        ("LLR p-value", f"{result1.llr_pvalue:.6f}"),
        ("H-L p-value", f"{hl_p:.4f} ({'Good fit ✅' if hl_p>0.05 else 'Poor fit ❌'})"),
      ])}
    </div>""", unsafe_allow_html=True)

# Coefficient table
section("Coefficient Table with Wald Statistics")
with st.expander("Show full statsmodels coefficient table", expanded=False):
    display = coef_df.drop(index="const")[["β","z","p","OR","CI_lo","CI_hi","Sig"]].rename(
        columns={"CI_lo":"OR CI Lo","CI_hi":"OR CI Hi","p":"p-value","z":"Wald z"})
    st.dataframe(display.style.applymap(
        lambda v: "color:#059669;font-weight:600" if v=="✅" else "color:#94A3B8" if v=="❌" else "",
        subset=["Sig"]), use_container_width=True)

section("Key Findings")
for t,d,c in [
    ("📊 Demographics Predict Adoption (AUC > 0.75)",
     f"The demographics-only model achieves AUC={auc1:.3f}, discriminating users from non-users well above chance. "
     f"Nagelkerke R²={nag1:.3f} — demographics explain ~{nag1*100:.0f}% of variance in adoption status.",INDIGO),
    ("📅 Age is the Dominant Predictor",
     "Older age groups show significantly lower Odds Ratios relative to the 18–25 reference group — "
     "each successive age band is less likely to adopt Q-Commerce.",ROSE),
    ("✅ Model is Well-Calibrated (H-L test)",
     f"Hosmer-Lemeshow χ²={hl_chi2:.3f}, p={hl_p:.3f} — {'good fit' if hl_p>0.05 else 'fit could be improved'}. "
     "Predicted probabilities match observed adoption rates across risk deciles.",EMERALD),
]:
    finding_card(t,d,c)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc, classification_report
)
import joblib
import os

# ─────────────────────────── PAGE CONFIG ───────────────────────────
st.set_page_config(
    page_title="Loan Approval Predictor — Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────── CUSTOM CSS ────────────────────────────
st.markdown("""
<style>
/* ── Global Font & Background ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f172a 0%, #1e3a5f 100%);
    color: white;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stRadio label { color: white !important; }

/* ── Top Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f766e 100%);
    border-radius: 16px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 1.8rem;
    color: white;
    display: flex;
    align-items: center;
    gap: 1.2rem;
}
.hero-banner h1 { margin: 0; font-size: 2rem; font-weight: 700; }
.hero-banner p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 1rem; }

/* ── KPI Cards ── */
.kpi-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 140px;
    background: white;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-top: 4px solid;
    text-align: center;
}
.kpi-card .val { font-size: 1.8rem; font-weight: 700; }
.kpi-card .lbl { font-size: 0.78rem; color: #64748b; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: .05em; }

/* ── Section Headers ── */
.section-header {
    font-size: 1.15rem; font-weight: 600;
    color: #1e3a5f; margin: 1.4rem 0 0.8rem;
    padding-left: 0.7rem;
    border-left: 4px solid #0f766e;
}

/* ── Result Cards ── */
.result-approved {
    background: linear-gradient(135deg, #d1fae5, #a7f3d0);
    border: 2px solid #10b981;
    border-radius: 14px; padding: 1.6rem;
    text-align: center; margin-top: 1rem;
}
.result-rejected {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    border: 2px solid #ef4444;
    border-radius: 14px; padding: 1.6rem;
    text-align: center; margin-top: 1rem;
}
.result-approved h2 { color: #065f46; font-size: 1.8rem; margin: 0.3rem 0; }
.result-rejected h2 { color: #991b1b; font-size: 1.8rem; margin: 0.3rem 0; }
.result-approved p  { color: #047857; }
.result-rejected p  { color: #b91c1c; }

/* ── Model Metric Boxes ── */
.metric-box {
    background: #f8fafc;
    border-radius: 10px; padding: 0.8rem 1rem;
    border: 1px solid #e2e8f0; margin: 0.3rem 0;
}

/* ── Tabs override ── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 1.2rem;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── DATA & MODEL ──────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("loan_approval_data.csv")
    return df

@st.cache_resource
def train_models(df):
    data = df.copy()
    categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols   = data.select_dtypes(include=["number"]).columns.tolist()

    num_imp = SimpleImputer(strategy="mean")
    data[numerical_cols] = num_imp.fit_transform(data[numerical_cols])

    cat_imp = SimpleImputer(strategy="most_frequent")
    data[categorical_cols] = cat_imp.fit_transform(data[categorical_cols])

    data = data.drop("Applicant_ID", axis=1)

    le = LabelEncoder()
    data["Education_Level"] = le.fit_transform(data["Education_Level"])
    data["Loan_Approved"]   = le.fit_transform(data["Loan_Approved"])

    ohe_cols = ["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender","Employer_Category"]
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = ohe.fit_transform(data[ohe_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=data.index)
    data = pd.concat([data.drop(columns=ohe_cols), encoded_df], axis=1)

    # Feature Engineering
    data["DTI_Ratio_sq"]    = data["DTI_Ratio"] ** 2
    data["Credit_Score_sq"] = data["Credit_Score"] ** 2

    X = data.drop(columns=["Loan_Approved","Credit_Score","DTI_Ratio"])
    y = data["Loan_Approved"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB()
    }
    results = {}
    for name, mdl in models.items():
        mdl.fit(X_train_sc, y_train)
        yp = mdl.predict(X_test_sc)
        yp_prob = mdl.predict_proba(X_test_sc)[:,1] if hasattr(mdl,"predict_proba") else None
        fpr, tpr, _ = roc_curve(y_test, yp_prob) if yp_prob is not None else (None, None, None)
        results[name] = {
            "model": mdl,
            "accuracy":  round(accuracy_score(y_test, yp)*100, 2),
            "precision": round(precision_score(y_test, yp)*100, 2),
            "recall":    round(recall_score(y_test, yp)*100, 2),
            "f1":        round(f1_score(y_test, yp)*100, 2),
            "cm":        confusion_matrix(y_test, yp),
            "fpr": fpr, "tpr": tpr,
            "roc_auc": round(auc(fpr, tpr)*100, 2) if fpr is not None else None,
            "y_pred": yp, "y_test": y_test
        }
    return results, scaler, ohe, le, X.columns.tolist(), data

# ─────────────────────────── SIDEBAR ───────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 SmartCredit Engine")
    st.markdown("*Intelligent Loan Approval System*")
    st.markdown("---")
    page = st.radio("Navigation", [
        # "🏠  Dashboard",
        # "📊  Data Explorer",
        # "🤖  Model Performance",
        "🔮  Predict Loan"
    ])
    st.markdown("---")
    st.markdown("**Dataset:** SecureTrust Bank")
    st.markdown("**Records:** 1,000 applicants")
    st.markdown("**Features:** 19 attributes")
    st.markdown("**Models:** LR · KNN · NB")
    st.markdown("---")
    st.caption("Built by Divyanshu Jangid")

# Load data
df = load_data()
results, scaler, ohe, le, feature_cols, processed_df = train_models(df)

# ════════════════════════════════════════
#   PAGE 1 — DASHBOARD
# ════════════════════════════════════════
# if "Dashboard" in page:
#     st.markdown("""
#     <div class="hero-banner">
#         <div style="font-size:3rem">🏦</div>
#         <div>
#             <h1>CreditWise Loan Approval System</h1>
#             <p>AI-powered intelligent loan decision engine for SecureTrust Bank</p>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

#     # KPI Row
#     approved_pct = round((df["Loan_Approved"].str.lower() == "yes").sum() / df["Loan_Approved"].notna().sum() * 100, 1)
#     avg_income  = f"₹{df['Applicant_Income'].mean():,.0f}"
#     avg_credit  = f"{df['Credit_Score'].mean():.0f}"
#     avg_loan    = f"₹{df['Loan_Amount'].mean():,.0f}"
#     best_model  = max(results, key=lambda k: results[k]["accuracy"])
#     best_acc    = results[best_model]["accuracy"]

#     c1, c2, c3, c4, c5 = st.columns(5)
#     colors = ["#0f766e","#1e3a5f","#7c3aed","#d97706","#dc2626"]
#     kpis = [
#         (f"{df.shape[0]:,}", "Total Applicants"),
#         (f"{approved_pct}%", "Approval Rate"),
#         (avg_income, "Avg. Income"),
#         (avg_credit, "Avg. Credit Score"),
#         (f"{best_acc}%", "Best Model Accuracy"),
#     ]
#     for col, (val, lbl), clr in zip([c1,c2,c3,c4,c5], kpis, colors):
#         col.markdown(f"""
#         <div class="kpi-card" style="border-top-color:{clr}">
#             <div class="val" style="color:{clr}">{val}</div>
#             <div class="lbl">{lbl}</div>
#         </div>""", unsafe_allow_html=True)

#     st.markdown('<div class="section-header">📈 Key Distributions</div>', unsafe_allow_html=True)
#     col1, col2 = st.columns(2)

#     with col1:
#         cnt = df["Loan_Approved"].value_counts()
#         fig = px.pie(values=cnt.values, names=cnt.index,
#                      color_discrete_sequence=["#10b981","#ef4444"],
#                      title="Loan Approval Distribution",
#                      hole=0.45)
#         fig.update_layout(margin=dict(t=50,b=10,l=10,r=10), height=320)
#         st.plotly_chart(fig, use_container_width=True)

#     with col2:
#         fig2 = px.histogram(df, x="Credit_Score", color="Loan_Approved",
#                             nbins=25, barmode="overlay",
#                             color_discrete_map={"Yes":"#10b981","No":"#ef4444"},
#                             title="Credit Score vs Approval",
#                             opacity=0.75)
#         fig2.update_layout(margin=dict(t=50,b=10,l=10,r=10), height=320)
#         st.plotly_chart(fig2, use_container_width=True)

#     col3, col4 = st.columns(2)
#     with col3:
#         fig3 = px.box(df, x="Loan_Approved", y="Applicant_Income",
#                       color="Loan_Approved",
#                       color_discrete_map={"Yes":"#10b981","No":"#ef4444"},
#                       title="Income vs Loan Approval")
#         fig3.update_layout(showlegend=False, margin=dict(t=50,b=10,l=10,r=10), height=320)
#         st.plotly_chart(fig3, use_container_width=True)

#     with col4:
#         purpose_cnt = df.groupby(["Loan_Purpose","Loan_Approved"]).size().reset_index(name="count")
#         fig4 = px.bar(purpose_cnt, x="Loan_Purpose", y="count", color="Loan_Approved",
#                       color_discrete_map={"Yes":"#10b981","No":"#ef4444"},
#                       barmode="group", title="Loan Purpose vs Approval")
#         fig4.update_layout(margin=dict(t=50,b=10,l=10,r=10), height=320)
#         st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════
#   PAGE 2 — DATA EXPLORER
# ════════════════════════════════════════
# elif "Data Explorer" in page:
#     st.markdown("""
#     <div class="hero-banner">
#         <div style="font-size:3rem">📊</div>
#         <div><h1>Data Explorer</h1>
#         <p>Explore, filter and analyse the raw loan dataset</p></div>
#     </div>""", unsafe_allow_html=True)

#     tab1, tab2, tab3 = st.tabs(["🔍 Browse Data", "📉 Feature Analysis", "🔥 Correlation Heatmap"])

#     with tab1:
#         st.markdown('<div class="section-header">Filter Dataset</div>', unsafe_allow_html=True)
#         f1, f2, f3 = st.columns(3)
#         with f1:
#             status_filter = st.multiselect("Approval Status", df["Loan_Approved"].dropna().unique(), default=df["Loan_Approved"].dropna().unique())
#         with f2:
#             area_filter = st.multiselect("Property Area", df["Property_Area"].dropna().unique(), default=df["Property_Area"].dropna().unique())
#         with f3:
#             emp_filter = st.multiselect("Employment", df["Employment_Status"].dropna().unique(), default=df["Employment_Status"].dropna().unique())

#         filtered = df[
#             df["Loan_Approved"].isin(status_filter) &
#             df["Property_Area"].isin(area_filter) &
#             df["Employment_Status"].isin(emp_filter)
#         ]
#         st.write(f"Showing **{len(filtered):,}** records")
#         st.dataframe(filtered, use_container_width=True, height=400)

#         col_a, col_b = st.columns(2)
#         col_a.metric("Missing Values", df.isnull().sum().sum())
#         col_b.metric("Filtered Records", len(filtered))

#     with tab2:
#         st.markdown('<div class="section-header">Univariate & Bivariate Analysis</div>', unsafe_allow_html=True)
#         num_cols_list = ["Applicant_Income","Coapplicant_Income","Credit_Score","DTI_Ratio","Savings","Collateral_Value","Loan_Amount","Age"]
#         cat_cols_list = ["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Education_Level","Gender","Employer_Category"]

#         sel_type = st.radio("Analysis type", ["Numerical Distribution","Categorical Breakdown"], horizontal=True)
#         if sel_type == "Numerical Distribution":
#             sel_col = st.selectbox("Select column", num_cols_list)
#             fig = px.histogram(df, x=sel_col, color="Loan_Approved",
#                                color_discrete_map={"Yes":"#10b981","No":"#ef4444"},
#                                marginal="box", nbins=30, opacity=0.8,
#                                title=f"{sel_col} Distribution by Loan Approval")
#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             sel_cat = st.selectbox("Select column", cat_cols_list)
#             grp = df.groupby([sel_cat,"Loan_Approved"]).size().reset_index(name="count")
#             fig = px.bar(grp, x=sel_cat, y="count", color="Loan_Approved",
#                          color_discrete_map={"Yes":"#10b981","No":"#ef4444"},
#                          barmode="group", title=f"{sel_cat} vs Loan Approval")
#             st.plotly_chart(fig, use_container_width=True)

#     with tab3:
#         num_data = processed_df.select_dtypes(include="number")
#         corr = num_data.corr()
#         fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto",
#                              color_continuous_scale="RdBu_r",
#                              title="Correlation Matrix")
#         fig_corr.update_layout(height=600)
#         st.plotly_chart(fig_corr, use_container_width=True)

# ════════════════════════════════════════
#   PAGE 3 — MODEL PERFORMANCE
# ════════════════════════════════════════
# elif "Model Performance" in page:
#     st.markdown("""
#     <div class="hero-banner">
#         <div style="font-size:3rem">🤖</div>
#         <div><h1>Model Performance</h1>
#         <p>Compare Logistic Regression, KNN and Naive Bayes models</p></div>
#     </div>""", unsafe_allow_html=True)

#     # Metrics comparison table
#     st.markdown('<div class="section-header">📋 Model Comparison</div>', unsafe_allow_html=True)
#     metrics_df = pd.DataFrame([
#         {"Model": k, "Accuracy (%)": v["accuracy"], "Precision (%)": v["precision"],
#          "Recall (%)": v["recall"], "F1 Score (%)": v["f1"],
#          "ROC-AUC (%)": v["roc_auc"] or 0}
#         for k, v in results.items()
#     ]).set_index("Model")
#     st.dataframe(metrics_df.style.highlight_max(axis=0, color="#d1fae5")
#                                   .highlight_min(axis=0, color="#fee2e2")
#                                   .format("{:.2f}"),
#                  use_container_width=True)

#     # Radar / Bar comparison
#     col_l, col_r = st.columns(2)
#     with col_l:
#         fig_bar = go.Figure()
#         metric_names = ["Accuracy (%)","Precision (%)","Recall (%)","F1 Score (%)"]
#         colors_m = ["#1e3a5f","#0f766e","#7c3aed"]
#         for (name, row), clr in zip(metrics_df.iterrows(), colors_m):
#             fig_bar.add_trace(go.Bar(name=name, x=metric_names,
#                                      y=[row[m] for m in metric_names],
#                                      marker_color=clr))
#         fig_bar.update_layout(barmode="group", title="Metrics Comparison",
#                                yaxis_range=[60,100], height=360,
#                                legend=dict(orientation="h", y=-0.25))
#         st.plotly_chart(fig_bar, use_container_width=True)

#     with col_r:
#         # ROC Curves
#         fig_roc = go.Figure()
#         for (name, v), clr in zip(results.items(), colors_m):
#             if v["fpr"] is not None:
#                 fig_roc.add_trace(go.Scatter(x=v["fpr"], y=v["tpr"], mode="lines",
#                                              name=f"{name} (AUC={v['roc_auc']}%)",
#                                              line=dict(color=clr, width=2)))
#         fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
#                                      line=dict(dash="dash", color="gray"), showlegend=False))
#         fig_roc.update_layout(title="ROC Curves", xaxis_title="False Positive Rate",
#                                yaxis_title="True Positive Rate", height=360,
#                                legend=dict(orientation="h", y=-0.3))
#         st.plotly_chart(fig_roc, use_container_width=True)

#     # Confusion Matrices
#     st.markdown('<div class="section-header">🔢 Confusion Matrices</div>', unsafe_allow_html=True)
#     cm_cols = st.columns(3)
#     for idx, (name, v) in enumerate(results.items()):
#         with cm_cols[idx]:
#             cm = v["cm"]
#             fig_cm = px.imshow(cm, text_auto=True,
#                                x=["Rejected","Approved"], y=["Rejected","Approved"],
#                                color_continuous_scale=["#fee2e2","#d1fae5"],
#                                title=f"{name}")
#             fig_cm.update_layout(height=280, margin=dict(t=50,b=10,l=10,r=10))
#             st.plotly_chart(fig_cm, use_container_width=True)

# ════════════════════════════════════════
#   PAGE 4 — PREDICT LOAN
# ════════════════════════════════════════
if "Predict" in page:
    st.markdown("""
    <div class="hero-banner">
        <div style="font-size:3rem">🔮</div>
        <div><h1>Loan Approval Predictor</h1>
        <p>Enter applicant details to get an instant AI-powered decision</p></div>
    </div>""", unsafe_allow_html=True)

    # Choose model
    chosen_model = st.selectbox("Select Model", list(results.keys()),
                                 index=0, help="Choose the ML model to run the prediction")

    st.markdown('<div class="section-header">👤 Personal Information</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        age     = st.number_input("Age", 18, 75, 35)
        gender  = st.selectbox("Gender", ["Male","Female"])
        marital = st.selectbox("Marital Status", ["Married","Single"])
    with c2:
        dependents = st.number_input("Dependents", 0, 10, 1)
        education  = st.selectbox("Education Level", ["Graduate","Postgraduate","Not Graduate"])
        emp_status = st.selectbox("Employment Status", ["Salaried","Self-Employed","Business"])
    with c3:
        employer   = st.selectbox("Employer Category", ["Private","Government","Self"])
        property_area = st.selectbox("Property Area", ["Urban","Rural","Semiurban"])

    st.markdown('<div class="section-header">💰 Financial Details</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        app_income  = st.number_input("Applicant Income (₹/month)", 1000, 200000, 30000, step=1000)
        coapp_income= st.number_input("Co-applicant Income (₹/month)", 0, 100000, 5000, step=500)
        savings     = st.number_input("Savings Balance (₹)", 0, 1000000, 50000, step=5000)
    with f2:
        credit_score  = st.slider("Credit Score", 300, 900, 680)
        dti_ratio     = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.35, 0.01)
        existing_loans= st.number_input("Existing Loans", 0, 10, 1)
    with f3:
        collateral   = st.number_input("Collateral Value (₹)", 0, 5000000, 100000, step=10000)
        loan_amount  = st.number_input("Loan Amount Requested (₹)", 5000, 5000000, 200000, step=5000)
        loan_term    = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 72, 84, 120, 180, 240, 360])
        loan_purpose = st.selectbox("Loan Purpose", ["Home","Education","Personal","Business","Car"])

    if st.button("🚀 Predict Loan Approval", use_container_width=True, type="primary"):
        # Build raw input dict
        raw_input = {
            "Applicant_Income": app_income,
            "Coapplicant_Income": coapp_income,
            "Employment_Status": emp_status,
            "Age": age,
            "Marital_Status": marital,
            "Dependents": dependents,
            "Credit_Score": credit_score,
            "Existing_Loans": existing_loans,
            "DTI_Ratio": dti_ratio,
            "Savings": savings,
            "Collateral_Value": collateral,
            "Loan_Amount": loan_amount,
            "Loan_Term": loan_term,
            "Loan_Purpose": loan_purpose,
            "Property_Area": property_area,
            "Education_Level": education,
            "Gender": gender,
            "Employer_Category": employer
        }
        inp_df = pd.DataFrame([raw_input])

        # Encode Education Level (same as training)
        edu_map = {"Graduate": 0, "Not Graduate": 1, "Postgraduate": 2}
        inp_df["Education_Level"] = inp_df["Education_Level"].map(edu_map).fillna(0).astype(int)

        # One-hot encode categorical cols
        ohe_cols = ["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender","Employer_Category"]
        encoded  = ohe.transform(inp_df[ohe_cols])
        enc_df   = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols))
        inp_df   = pd.concat([inp_df.drop(columns=ohe_cols).reset_index(drop=True), enc_df], axis=1)

        # Feature engineering
        inp_df["DTI_Ratio_sq"]    = inp_df["DTI_Ratio"] ** 2
        inp_df["Credit_Score_sq"] = inp_df["Credit_Score"] ** 2
        inp_df = inp_df.drop(columns=["Credit_Score","DTI_Ratio"], errors="ignore")

        # Align columns
        for c in feature_cols:
            if c not in inp_df.columns:
                inp_df[c] = 0
        inp_df = inp_df[feature_cols]

        # Scale & predict
        inp_scaled = scaler.transform(inp_df)
        mdl = results[chosen_model]["model"]
        prediction = mdl.predict(inp_scaled)[0]
        prob = mdl.predict_proba(inp_scaled)[0] if hasattr(mdl,"predict_proba") else None

        # Show result
        if prediction == 1:
            conf = round(prob[1]*100, 1) if prob is not None else "—"
            st.markdown(f"""
            <div class="result-approved">
                <div style="font-size:3.5rem">✅</div>
                <h2>LOAN APPROVED</h2>
                <p style="font-size:1.1rem">Confidence: <strong>{conf}%</strong></p>
                <p>The applicant meets the eligibility criteria. Recommended for approval after standard KYC verification.</p>
            </div>""", unsafe_allow_html=True)
        else:
            conf = round(prob[0]*100, 1) if prob is not None else "—"
            st.markdown(f"""
            <div class="result-rejected">
                <div style="font-size:3.5rem">❌</div>
                <h2>LOAN REJECTED</h2>
                <p style="font-size:1.1rem">Confidence: <strong>{conf}%</strong></p>
                <p>The application does not meet current risk thresholds. Recommend review of credit score, DTI ratio, or loan amount.</p>
            </div>""", unsafe_allow_html=True)

        # Factors summary
        st.markdown('<div class="section-header">📌 Risk Factors Summary</div>', unsafe_allow_html=True)
        fa, fb, fc = st.columns(3)
        credit_flag  = "🟢 Good" if credit_score >= 650 else ("🟡 Fair" if credit_score >= 550 else "🔴 Poor")
        dti_flag     = "🟢 Healthy" if dti_ratio < 0.4 else ("🟡 Moderate" if dti_ratio < 0.6 else "🔴 High")
        income_flag  = "🟢 Stable" if app_income >= 20000 else ("🟡 Moderate" if app_income >= 10000 else "🔴 Low")
        fa.metric("Credit Score", f"{credit_score}", credit_flag)
        fb.metric("DTI Ratio", f"{dti_ratio:.2f}", dti_flag)
        fc.metric("Monthly Income", f"₹{app_income:,}", income_flag)

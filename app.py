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

# PAGE CONFIG
st.set_page_config(
    page_title="Loan Approval Predictor — Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

#CUSTOM CSS 
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

#DATA & MODEL
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

# SIDEBAR
with st.sidebar:
    st.markdown("## 🏦 SmartCredit Engine")
    st.markdown("*Intelligent Loan Approval System*")
    st.markdown("---")
    page = st.radio("Navigation", [
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



#   PAGE 4 — PREDICT LOAN

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

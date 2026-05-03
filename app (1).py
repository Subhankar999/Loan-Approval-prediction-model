import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global dark background ── */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .stApp {
        background-color: #0d1117 !important;
    }
    [data-testid="stHeader"] { background-color: #0d1117 !important; }
    [data-testid="stSidebar"] { background-color: #161b22 !important; }

    /* ── Main content block ── */
    .block-container {
        background: #161b22 !important;
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.6);
        border: 1px solid #30363d;
    }

    /* ── Typography ── */
    h1, h2, h3, h4, p, label, span, div {
        color: #e6edf3 !important;
    }
    .stMarkdown p { color: #c9d1d9 !important; }

    /* ── Inputs & selects ── */
    .stSelectbox > div > div,
    .stSlider > div,
    .stNumberInput > div > div > input,
    [data-testid="stTextInput"] input {
        background-color: #21262d !important;
        color: #e6edf3 !important;
        border-color: #30363d !important;
        border-radius: 8px !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background-color: #21262d !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
    }
    [data-testid="stExpander"] summary { color: #e6edf3 !important; }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: #21262d !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        padding: 0.75rem !important;
    }
    [data-testid="stMetricValue"] { color: #58a6ff !important; }
    [data-testid="stMetricLabel"] { color: #8b949e !important; }

    /* ── Dataframe / table ── */
    [data-testid="stDataFrame"] iframe { background: #21262d !important; }
    .stDataFrame { background: #21262d !important; border-radius: 8px !important; }

    /* ── Info / alert boxes ── */
    [data-testid="stInfo"] {
        background: #1f2d3d !important;
        color: #58a6ff !important;
        border-left-color: #58a6ff !important;
    }

    /* ── Divider ── */
    hr { border-color: #30363d !important; }

    /* ── Caption / footer ── */
    .stCaption, [data-testid="stCaptionContainer"] { color: #6e7681 !important; }

    /* ── Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #58a6ff, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        font-weight: bold !important;
        font-size: 1rem !important;
        width: 100% !important;
        transition: all 0.3s !important;
    }
    .stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

    /* ── Result banners ── */
    .result-approved {
        background: linear-gradient(135deg, #0d4429, #1a7f4e);
        border: 1px solid #238636;
        color: #3fb950;
        padding: 1.5rem; border-radius: 12px;
        text-align: center; font-size: 1.5rem; font-weight: bold; margin: 1rem 0;
    }
    .result-rejected {
        background: linear-gradient(135deg, #4a0f0f, #7f1d1d);
        border: 1px solid #da3633;
        color: #f85149;
        padding: 1.5rem; border-radius: 12px;
        text-align: center; font-size: 1.5rem; font-weight: bold; margin: 1rem 0;
    }

    /* ── Progress bar ── */
    [data-testid="stProgressBar"] > div { background-color: #58a6ff !important; }
    [data-testid="stProgressBar"] { background-color: #21262d !important; }
</style>
""", unsafe_allow_html=True)

# ─── Load & Train Model ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training model on dataset…")
def load_and_train():
    df = pd.read_csv("Loan_Default.csv")

    # Fill numerical nulls with mean
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        df[col] = df[col].fillna(df[col].mean())

    # Fill categorical nulls with forward fill then mode
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].ffill()
        df[col] = df[col].fillna(df[col].mode()[0])

    categorical_cols = ["loan_limit", "approv_in_adv", "occupancy_type",
                        "business_or_commercial", "interest_only",
                        "lump_sum_payment", "Neg_ammortization", "Credit_Worthiness"]
    numeric_cols = ["Credit_Score", "LTV", "income", "dtir1"]

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[numeric_cols])
    scaled_df = pd.DataFrame(scaled, columns=numeric_cols)

    X = pd.concat([encoded_df, scaled_df], axis=1)
    y = df["Status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, encoder, scaler, categorical_cols, numeric_cols, acc, cm, report, encoded_df.columns.tolist()


model, encoder, scaler, cat_cols, num_cols, accuracy, cm, report, encoded_col_names = load_and_train()

# ─── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🏦 Loan Approval Predictor")
st.markdown("**AI-powered loan decision system** using Random Forest on real loan default data.")
st.divider()

# ─── Layout: Form | Metrics ──────────────────────────────────────────────────────
col_form, col_info = st.columns([3, 2], gap="large")

with col_info:
    st.markdown("### 📊 Model Performance")
    m1, m2 = st.columns(2)
    m1.metric("Accuracy", f"{accuracy*100:.1f}%")
    m2.metric("Training Samples", "~148K")

    st.markdown("#### Class Report")
    r_df = pd.DataFrame(report).T.loc[["0", "1"], ["precision", "recall", "f1-score"]]
    r_df.index = ["Not Approved", "Approved"]
    r_df = r_df.round(3)
    st.dataframe(r_df, use_container_width=True)

    st.markdown("#### Confusion Matrix")
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: Not Approved", "Actual: Approved"],
        columns=["Pred: Not Approved", "Pred: Approved"]
    )
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("#### 📁 Dataset Info")
    st.info("**Loan_Default.csv**\n\n• 34 features\n• Binary target: Status (0 / 1)\n• Random Forest (50 estimators)")

with col_form:
    st.markdown("### 🧾 Enter Applicant Details")

    with st.expander("📋 Categorical Features", expanded=True):
        c1, c2 = st.columns(2)
        loan_limit = c1.selectbox("Loan Limit", ["cf", "ncf"], help="cf = Conforming, ncf = Non-conforming")
        approv_in_adv = c2.selectbox("Approval in Advance", ["nopre", "pre"], help="pre = Pre-approved")
        occupancy_type = c1.selectbox("Occupancy Type", ["pr", "sr", "ir"], help="pr = Primary, sr = Secondary, ir = Investment")
        business_or_commercial = c2.selectbox("Business/Commercial", ["nob/c", "b/c"], help="b/c = Business or commercial loan")
        interest_only = c1.selectbox("Interest Only", ["not_int", "int_only"])
        lump_sum_payment = c2.selectbox("Lump Sum Payment", ["not_lpsm", "lpsm"])
        neg_ammortization = c1.selectbox("Negative Amortization", ["not_neg", "neg_amm"])
        credit_worthiness = c2.selectbox("Credit Worthiness", ["l1", "l2"], help="l1 = Level 1 (higher), l2 = Level 2")

    with st.expander("💰 Numerical Features", expanded=True):
        n1, n2 = st.columns(2)
        credit_score = n1.slider("Credit Score", 300, 850, 700, step=1)
        ltv = n2.slider("LTV (Loan-to-Value %)", 0.0, 200.0, 80.0, step=0.5)
        income = n1.number_input("Monthly Income ($)", min_value=0.0, max_value=500000.0, value=5000.0, step=100.0)
        dtir1 = n2.slider("Debt-to-Income Ratio (%)", 0.0, 100.0, 40.0, step=0.5)

    predict_btn = st.button("🔍 Predict Loan Approval", use_container_width=True)

    if predict_btn:
        # Build input dataframes
        user_cat = {
            "loan_limit": loan_limit,
            "approv_in_adv": approv_in_adv,
            "occupancy_type": occupancy_type,
            "business_or_commercial": business_or_commercial,
            "interest_only": interest_only,
            "lump_sum_payment": lump_sum_payment,
            "Neg_ammortization": neg_ammortization,
            "Credit_Worthiness": credit_worthiness,
        }
        user_num = {
            "Credit_Score": credit_score,
            "LTV": ltv,
            "income": income,
            "dtir1": dtir1,
        }

        cat_df = pd.DataFrame([user_cat], columns=cat_cols)
        num_df = pd.DataFrame([user_num], columns=num_cols)

        encoded_input = encoder.transform(cat_df)
        encoded_input_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(cat_cols))
        # Align columns
        for c in encoded_col_names:
            if c not in encoded_input_df.columns:
                encoded_input_df[c] = 0
        encoded_input_df = encoded_input_df[encoded_col_names]

        scaled_input = scaler.transform(num_df)
        scaled_input_df = pd.DataFrame(scaled_input, columns=num_cols)

        final_input = pd.concat([encoded_input_df, scaled_input_df], axis=1)

        prediction = model.predict(final_input)[0]
        proba = model.predict_proba(final_input)[0]

        st.divider()
        if prediction == 1:
            st.markdown(f'<div class="result-approved">✅ LOAN APPROVED<br><small>Confidence: {proba[1]*100:.1f}%</small></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-rejected">❌ LOAN NOT APPROVED<br><small>Confidence: {proba[0]*100:.1f}%</small></div>', unsafe_allow_html=True)

        # Probability bar
        st.markdown("#### 📈 Approval Probability")
        prob_col1, prob_col2 = st.columns(2)
        prob_col1.metric("Approved", f"{proba[1]*100:.1f}%")
        prob_col2.metric("Not Approved", f"{proba[0]*100:.1f}%")
        st.progress(float(proba[1]))

        # Summary
        st.markdown("#### 📝 Input Summary")
        summary = pd.DataFrame({
            "Feature": ["Credit Score", "LTV", "Monthly Income", "Debt-to-Income Ratio",
                        "Loan Limit", "Pre-approved", "Occupancy", "Business Loan",
                        "Interest Only", "Lump Sum", "Neg Amortization", "Credit Worthiness"],
            "Value": [credit_score, f"{ltv}%", f"${income:,.0f}", f"{dtir1}%",
                      loan_limit, approv_in_adv, occupancy_type, business_or_commercial,
                      interest_only, lump_sum_payment, neg_ammortization, credit_worthiness]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

# ─── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit · Random Forest Classifier · Loan Default Dataset")

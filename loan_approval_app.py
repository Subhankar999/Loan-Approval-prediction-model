import streamlit as st
import pandas as pd
import numpy as np
import gdown
import os
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
GDRIVE_FILE_ID = "1UvhABfIVrZEfFvPFTEhh2p4j9bG3RKRE"
CSV_PATH = "Loan_Default.csv"
MODEL_CACHE = "model_cache.pkl"

CATEGORIAL_COLS = [
    "loan_limit", "approv_in_adv", "occupancy_type",
    "business_or_commercial", "interest_only",
    "lump_sum_payment", "Neg_ammortization", "Credit_Worthiness"
]
NUMERIC_COLS = ["Credit_Score", "LTV", "income", "dtir1"]

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    """Download dataset from Google Drive and return a DataFrame."""
    if not os.path.exists(CSV_PATH):
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, CSV_PATH, quiet=True)

    df = pd.read_csv(CSV_PATH)

    # Fill numeric nulls with column mean
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        df[col].fillna(df[col].mean(), inplace=True)

    # Fill object nulls with forward fill
    for col in df.select_dtypes(include=["object"]).columns:
        df[col].fillna(method="ffill", inplace=True)

    return df


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model(df):
    """Encode, scale and train RandomForestClassifier. Returns model + transformers."""
    if os.path.exists(MODEL_CACHE):
        with open(MODEL_CACHE, "rb") as f:
            return pickle.load(f)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df[CATEGORIAL_COLS])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(CATEGORIAL_COLS))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[NUMERIC_COLS])
    scaled_df = pd.DataFrame(scaled, columns=NUMERIC_COLS)

    X = pd.concat([encoded_df, scaled_df], axis=1)
    y = df["Status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)

    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    artifacts = {
        "model": model,
        "encoder": encoder,
        "scaler": scaler,
        "encoded_df_columns": encoded_df.columns.tolist(),
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_test, y_pred),
        "cm": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }

    with open(MODEL_CACHE, "wb") as f:
        pickle.dump(artifacts, f)

    return artifacts


# ─────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────
def predict(artifacts, cat_inputs: dict, num_inputs: dict) -> int:
    encoder = artifacts["encoder"]
    scaler  = artifacts["scaler"]
    model   = artifacts["model"]
    enc_cols = artifacts["encoded_df_columns"]

    cat_df  = pd.DataFrame([cat_inputs], columns=CATEGORIAL_COLS)
    num_df  = pd.DataFrame([num_inputs], columns=NUMERIC_COLS)

    enc_arr = encoder.transform(cat_df)
    enc_df  = pd.DataFrame(enc_arr, columns=encoder.get_feature_names_out(CATEGORIAL_COLS))
    enc_df  = enc_df.reindex(columns=enc_cols, fill_value=0)

    sc_arr  = scaler.transform(num_df)
    sc_df   = pd.DataFrame(sc_arr, columns=NUMERIC_COLS)

    final   = pd.concat([enc_df, sc_df], axis=1)
    return model.predict(final)[0]


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
st.title("🏦 Loan Approval Prediction System")
st.markdown("Powered by a **Random Forest** model trained on the Loan Default dataset from Google Drive.")

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("📥 Downloading dataset from Google Drive…"):
    try:
        df = load_data()
        data_ok = True
    except Exception as e:
        st.error(f"❌ Could not load data: {e}")
        st.info("Make sure `gdown` is installed: `pip install gdown`")
        data_ok = False

if not data_ok:
    st.stop()

# ── Train model ────────────────────────────────────────────────────────────────
with st.spinner("⚙️ Training model (cached after first run)…"):
    artifacts = train_model(df)

st.success(f"✅ Model ready — Accuracy: **{artifacts['accuracy']*100:.2f}%**")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Metrics", "🗂️ Dataset"])

# ══════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Enter Applicant Details")
    st.markdown("Fill in the fields below and click **Predict** to get the loan decision.")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Categorical Features")
        loan_limit = st.selectbox(
            "Loan Limit",
            ["cf", "ncf"],
            help="cf = Conforming, ncf = Non-conforming"
        )
        approv_in_adv = st.selectbox(
            "Pre-approval in Advance",
            ["pre", "nopre"],
            help="Whether loan was pre-approved"
        )
        occupancy_type = st.selectbox(
            "Occupancy Type",
            ["pr", "sr", "ir"],
            help="pr = Primary, sr = Secondary, ir = Investment"
        )
        business_or_commercial = st.selectbox(
            "Business or Commercial",
            ["b/c", "nob/c"],
            help="Loan purpose"
        )
        interest_only = st.selectbox(
            "Interest Only",
            ["int_only", "not_int"],
            help="Interest-only loan?"
        )
        lump_sum_payment = st.selectbox(
            "Lump Sum Payment",
            ["lpsm", "not_lpsm"],
            help="Lump sum allowed?"
        )
        neg_ammortization = st.selectbox(
            "Negative Amortization",
            ["not_neg", "neg_amm"],
            help="Negative amortization flag"
        )
        credit_worthiness = st.selectbox(
            "Credit Worthiness",
            ["l1", "l2"],
            help="l1 = High, l2 = Low"
        )

    with col_r:
        st.markdown("#### Numerical Features")
        credit_score = st.slider(
            "Credit Score", min_value=300, max_value=850, value=700, step=1
        )
        ltv = st.slider(
            "Loan-to-Value Ratio (LTV)", min_value=0.0, max_value=200.0, value=85.0, step=0.1
        )
        income = st.number_input(
            "Monthly Income (USD)", min_value=0.0, value=5000.0, step=100.0
        )
        dtir1 = st.slider(
            "Debt-to-Income Ratio (DTIR %)", min_value=0.0, max_value=100.0, value=40.0, step=0.1
        )

        st.markdown("---")
        predict_btn = st.button("🔮 Predict Loan Approval", use_container_width=True, type="primary")

        if predict_btn:
            cat_inputs = {
                "loan_limit": loan_limit,
                "approv_in_adv": approv_in_adv,
                "occupancy_type": occupancy_type,
                "business_or_commercial": business_or_commercial,
                "interest_only": interest_only,
                "lump_sum_payment": lump_sum_payment,
                "Neg_ammortization": neg_ammortization,
                "Credit_Worthiness": credit_worthiness,
            }
            num_inputs = {
                "Credit_Score": credit_score,
                "LTV": ltv,
                "income": income,
                "dtir1": dtir1,
            }

            result = predict(artifacts, cat_inputs, num_inputs)

            st.markdown("---")
            if result == 1:
                st.success("✅ **Loan Approved!**  The applicant meets the criteria.")
                st.balloons()
            else:
                st.error("❌ **Loan Not Approved.**  The applicant does not meet the criteria.")

# ══════════════════════════════════════════════
# TAB 2 — MODEL METRICS
# ══════════════════════════════════════════════
with tab2:
    st.subheader("Model Performance")

    m1, m2, m3 = st.columns(3)
    report = artifacts["report"]
    m1.metric("Accuracy", f"{artifacts['accuracy']*100:.2f}%")
    m2.metric("Precision (avg)", f"{report['weighted avg']['precision']*100:.2f}%")
    m3.metric("Recall (avg)", f"{report['weighted avg']['recall']*100:.2f}%")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Confusion Matrix**")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        sns.heatmap(artifacts["cm"], annot=True, fmt="d", cmap="Blues", ax=ax1)
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")
        ax1.set_title("Confusion Matrix")
        st.pyplot(fig1)
        plt.close(fig1)

    with col_b:
        st.markdown("**Actual vs Predicted (first 60 test samples)**")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(artifacts["y_test"].values[:60], label="Actual", linewidth=1.5)
        ax2.plot(artifacts["y_pred"][:60], label="Predicted", linestyle="--", linewidth=1.5)
        ax2.legend()
        ax2.set_title("Actual vs Predicted")
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("Status")
        st.pyplot(fig2)
        plt.close(fig2)

    st.markdown("**Full Classification Report**")
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — DATASET
# ══════════════════════════════════════════════
with tab3:
    st.subheader("Dataset Overview")
    st.caption(f"Source: Google Drive — File ID `{GDRIVE_FILE_ID}`")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Target: Approved (%)", f"{(df['Status']==1).mean()*100:.1f}%")

    st.dataframe(df.head(100), use_container_width=True)

    st.markdown("**Feature Distributions**")
    num_feat = st.selectbox("Select a numerical feature to plot", NUMERIC_COLS)
    fig3, ax3 = plt.subplots(figsize=(7, 3))
    df[num_feat].hist(ax=ax3, bins=40, edgecolor="white", color="#4C9BE8")
    ax3.set_title(f"Distribution of {num_feat}")
    ax3.set_xlabel(num_feat)
    ax3.set_ylabel("Count")
    st.pyplot(fig3)
    plt.close(fig3)

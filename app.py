"""Streamlit Web Application — Bank Loan Approval and Customer Risk Analysis.

Provides an interactive user interface for loan officers and applicants to:
1. Input applicant demographic and financial details.
2. Obtain instant loan approval decisions with model prediction probability.
3. Review an explainable customer risk tier (Low, Medium, High) with driving factors.
4. Explore model benchmark metrics and feature importance insights.
"""

import os
import sys
from typing import Any, Dict

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.preprocessing import FEATURE_COLUMNS, prepare_applicant_features
from src.train_risk import evaluate_risk_tier

# ==========================================
# Application Configuration & Styling
# ==========================================
st.set_page_config(
    page_title="LoanGuard | Loan Approval & Risk Analyzer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "models/loan_approval_model.pkl"
RISK_PATH = "models/risk_scoring.pkl"
COMPARISON_CSV = "models/model_comparison.csv"


@st.cache_resource
def load_model_artifacts() -> Dict[str, Any]:
    """Loads and caches serialized classification and risk scoring artifacts.

    Returns:
        Dictionary containing loaded models and metadata.
    """
    artifacts = {}
    if os.path.exists(MODEL_PATH):
        artifacts["approval"] = joblib.load(MODEL_PATH)
    else:
        artifacts["approval"] = None

    if os.path.exists(RISK_PATH):
        artifacts["risk"] = joblib.load(RISK_PATH)
    else:
        artifacts["risk"] = None

    if os.path.exists(COMPARISON_CSV):
        artifacts["comparison_df"] = pd.read_csv(COMPARISON_CSV, index_col=0)
    else:
        artifacts["comparison_df"] = None

    return artifacts


# ==========================================
# Main App Layout
# ==========================================
def main() -> None:
    """Entry point for the Streamlit web application."""
    artifacts = load_model_artifacts()

    # Header section
    st.title("🏦 Bank Loan Approval & Customer Risk Analysis")
    st.markdown(
        """
        Evaluate loan applications with machine learning classification and assess 
        applicant risk profile with multidimensional credit and collateral scoring.
        """
    )
    st.divider()

    # Sidebar: Model and Dataset Information
    with st.sidebar:
        st.header("📌 System Overview")
        if artifacts["approval"]:
            all_models = artifacts["approval"].get("all_models", {})
            model_options = list(all_models.keys()) if all_models else [artifacts["approval"].get("model_name", "Random Forest")]
            default_idx = model_options.index(artifacts["approval"].get("model_name", "Random Forest")) if artifacts["approval"].get("model_name", "Random Forest") in model_options else 0
            
            selected_model_name = st.selectbox(
                "Active Classification Algorithm",
                options=model_options,
                index=default_idx,
                help="Choose which trained algorithm evaluates the loan application."
            )
            
            # Display metrics for selected model
            if artifacts["comparison_df"] is not None and selected_model_name in artifacts["comparison_df"].index:
                row = artifacts["comparison_df"].loc[selected_model_name]
                st.metric("Test Accuracy", f"{row['Accuracy']:.2%}")
                st.metric("Test F1-Score", f"{row['F1-Score']:.4f}")
                st.metric("Test ROC-AUC", f"{row['ROC-AUC']:.4f}")
            else:
                acc = artifacts["approval"]["champion_metrics"].get("Accuracy", 0)
                st.metric("Test Accuracy", f"{acc:.2%}")
        else:
            selected_model_name = "Random Forest"
            st.warning("Model artifact not found. Please run training pipeline.")

        st.markdown("---")
        st.subheader("💡 Evaluation Principles")
        st.info(
            """
            - **Decision Model**: Supervised binary classifier evaluated on 4,269 historical applications.
            - **Risk Profiling**: Composite assessment of credit history (CIBIL), debt-to-income (LTI), and asset coverage.
            """
        )

    # Layout: Two columns (Input form on left, Results/Insights on right)
    col_form, col_results = st.columns([1.1, 1], gap="large")

    with col_form:
        st.subheader("📋 Applicant Information Form")
        st.caption("Enter applicant details to predict approval and risk tier.")

        with st.form("loan_application_form"):
            # Section: Personal & Profile
            st.markdown("##### 👤 Personal & Employment Profile")
            p1, p2, p3 = st.columns(3)
            with p1:
                no_of_dependents = st.number_input(
                    "Dependents", min_value=0, max_value=10, value=2, step=1,
                    help="Number of dependent family members."
                )
            with p2:
                education = st.selectbox(
                    "Education Level", ["Graduate", "Not Graduate"], index=0
                )
            with p3:
                self_employed = st.selectbox(
                    "Self Employed?", ["No", "Yes"], index=0
                )

            # Section: Credit & Loan Request
            st.markdown("##### 💳 Credit & Loan Request")
            c1, c2, c3 = st.columns(3)
            with c1:
                cibil_score = st.slider(
                    "CIBIL Credit Score",
                    min_value=300,
                    max_value=900,
                    value=720,
                    step=5,
                    help="Credit bureau score (300 = Poor, 900 = Exceptional)."
                )
            with c2:
                loan_term = st.number_input(
                    "Loan Term (Years)",
                    min_value=1,
                    max_value=30,
                    value=10,
                    step=1,
                    help="Requested repayment duration in years."
                )
            with c3:
                income_annum = st.number_input(
                    "Annual Income (INR)",
                    min_value=100000,
                    max_value=20000000,
                    value=4500000,
                    step=100000,
                    format="%d",
                    help="Gross annual earnings in Indian Rupees."
                )

            loan_amount = st.number_input(
                "Loan Amount Requested (INR)",
                min_value=100000,
                max_value=50000000,
                value=12000000,
                step=500000,
                format="%d",
                help="Total loan principal requested."
            )

            # Section: Asset Portfolio
            st.markdown("##### 🏛️ Collateral & Asset Portfolio (INR)")
            a1, a2 = st.columns(2)
            with a1:
                residential_assets_value = st.number_input(
                    "Residential Assets Value", min_value=0, max_value=50000000, value=5000000, step=250000
                )
                commercial_assets_value = st.number_input(
                    "Commercial Assets Value", min_value=0, max_value=50000000, value=3000000, step=250000
                )
            with a2:
                luxury_assets_value = st.number_input(
                    "Luxury Assets Value", min_value=0, max_value=50000000, value=8000000, step=250000
                )
                bank_asset_value = st.number_input(
                    "Bank Deposit / Liquid Assets", min_value=0, max_value=50000000, value=3500000, step=250000
                )

            submit_button = st.form_submit_button("⚡ Evaluate Application", use_container_width=True, type="primary")

    with col_results:
        st.subheader("📊 Underwriting Decision & Risk Output")

        if submit_button:
            # Package inputs into dictionary
            applicant_data = {
                "no_of_dependents": no_of_dependents,
                "education": education,
                "self_employed": self_employed,
                "income_annum": income_annum,
                "loan_amount": loan_amount,
                "loan_term": loan_term,
                "cibil_score": cibil_score,
                "residential_assets_value": residential_assets_value,
                "commercial_assets_value": commercial_assets_value,
                "luxury_assets_value": luxury_assets_value,
                "bank_asset_value": bank_asset_value,
            }

            # 1. Prepare features using shared preprocessing pipeline
            features_df = prepare_applicant_features(applicant_data)

            # Extract derived ratios for risk scoring
            loan_to_income = features_df["loan_to_income"].iloc[0]
            asset_to_loan = features_df["asset_to_loan"].iloc[0]
            total_assets = features_df["total_assets"].iloc[0]

            # 2. Loan Approval Classification
            approval_artifact = artifacts["approval"]
            if approval_artifact:
                if "all_models" in approval_artifact and selected_model_name in approval_artifact["all_models"]:
                    model = approval_artifact["all_models"][selected_model_name]
                else:
                    model = approval_artifact["model"]
                pred = model.predict(features_df)[0]
                proba = model.predict_proba(features_df)[0]
                is_approved = (pred == 1)
                approval_confidence = proba[1] if is_approved else proba[0]
            else:
                # Fallback if artifact is absent
                is_approved = cibil_score >= 550
                approval_confidence = 0.90

            # 3. Customer Risk Scoring
            risk_result = evaluate_risk_tier(
                cibil_score=cibil_score,
                loan_to_income=loan_to_income,
                asset_to_loan=asset_to_loan,
            )

            # --- DISPLAY RESULTS ---
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.markdown("#### Loan Decision")
                if is_approved:
                    st.success("### ✅ APPROVED")
                else:
                    st.error("### ❌ REJECTED")

                st.write(f"**Confidence:** `{approval_confidence:.1%}`")
                st.progress(float(approval_confidence))

            with res_col2:
                st.markdown("#### Customer Risk Profile")
                tier = risk_result["risk_tier"]
                score = risk_result["risk_score"]

                if tier == "Low Risk":
                    st.success(f"### 🟢 {tier}")
                elif tier == "Medium Risk":
                    st.warning(f"### 🟡 {tier}")
                else:
                    st.error(f"### 🔴 {tier}")

                st.write(f"**Risk Score:** `{score} / 100` (Lower is safer)")

            st.divider()

            # Key Financial Ratios & Risk Factors
            st.markdown("##### 🔍 Financial Ratios & Driving Factors")
            r1, r2, r3 = st.columns(3)
            r1.metric("Debt-to-Income", f"{loan_to_income:.2f}x")
            r2.metric("Asset Coverage", f"{asset_to_loan:.2f}x")
            r3.metric("Total Assets", f"₹{total_assets / 1e5:.1f} Lakhs")

            st.markdown("**Key Risk Drivers Identified:**")
            for factor in risk_result["key_factors"]:
                st.markdown(f"- {factor}")

        else:
            st.info("👈 Fill out the applicant form and click **'Evaluate Application'** to view prediction results.")

    # --- INSIGHTS & BENCHMARKS SECTION ---
    st.divider()
    st.subheader("📈 Model Diagnostics & Analytics")
    tab1, tab2, tab3 = st.tabs(["Algorithm Benchmark", "Feature Importances", "Visual Reports"])

    with tab1:
        if artifacts["comparison_df"] is not None:
            st.write("Holdout test set evaluation (854 records):")
            st.dataframe(artifacts["comparison_df"], use_container_width=True)
            st.caption("All models trained with identical random seeds and stratified splits.")
        else:
            st.write("Benchmark table not found.")

    with tab2:
        if artifacts["approval"] and "feature_importances" in artifacts["approval"]:
            fi = artifacts["approval"]["feature_importances"]
            fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance Score"]).sort_values(
                by="Importance Score", ascending=False
            )
            st.bar_chart(data=fi_df, x="Feature", y="Importance Score", use_container_width=True)
        else:
            st.write("Feature importance data unavailable.")

    with tab3:
        v1, v2 = st.columns(2)
        with v1:
            if os.path.exists("report/figures/confusion_matrices.png"):
                st.image("report/figures/confusion_matrices.png", caption="Model Confusion Matrices")
        with v2:
            if os.path.exists("report/figures/roc_curves.png"):
                st.image("report/figures/roc_curves.png", caption="Receiver Operating Characteristic (ROC)")


if __name__ == "__main__":
    main()
